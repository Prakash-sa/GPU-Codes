// Warp primitives and cooperative groups: one small, tested kernel per idea.
// Companion to notes/13-warp-primitives.md.
//
// Every host function is `extern "C"` and takes device pointers, like the LeetGPU solutions.
// Kernels use one thread per element and blocks of 256 threads (8 full warps) unless noted.
//
// Rule followed everywhere: threads past the end of the data do NOT return early. They stay
// in the warp with a neutral value (0, -inf, -1), because every warp collective below uses
// the full mask 0xffffffff, which promises that all 32 lanes call it.
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <climits>
#include <math.h>

namespace cg = cooperative_groups;

#define FULL 0xffffffffu
#define THREADS 256

static int blocks_for(int n) { return n > 0 ? (n + THREADS - 1) / THREADS : 1; }

// ---------------------------------------------------------------- 1. __shfl_down_sync
// Sum of each warp's 32 elements. The tree halves the distance each step, so after
// 5 steps lane 0 holds the total; the other lanes hold partial sums.

__device__ __forceinline__ float warp_sum_down(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(FULL, v, offset);          // lane i reads lane i + offset
    return v;                                            // valid in lane 0 only
}

__global__ void warp_sums_kernel(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float s = warp_sum_down(i < n ? x[i] : 0.0f);
    if ((threadIdx.x & 31) == 0 && i < n) out[i / 32] = s;
}

extern "C" void warp_sums(const float* x, float* out, int n) {
    warp_sums_kernel<<<blocks_for(n), THREADS>>>(x, out, n);
    cudaDeviceSynchronize();
}

// ---------------------------------------------------------------- 2. __shfl_xor_sync
// Butterfly all-reduce: lanes swap with the partner whose index differs in one bit, so
// every lane ends with the warp maximum (no broadcast step needed afterwards).

__device__ __forceinline__ float warp_max_all(float v) {
    for (int mask = 16; mask > 0; mask >>= 1)
        v = fmaxf(v, __shfl_xor_sync(FULL, v, mask));    // lane i reads lane i ^ mask
    return v;                                            // valid in every lane
}

__global__ void warp_max_kernel(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float m = warp_max_all(i < n ? x[i] : -INFINITY);
    if (i < n) out[i] = m;
}

extern "C" void warp_max(const float* x, float* out, int n) {
    warp_max_kernel<<<blocks_for(n), THREADS>>>(x, out, n);
    cudaDeviceSynchronize();
}

// ---------------------------------------------------------------- 3. __shfl_sync + width
// Broadcast: every lane reads lane 0 of its segment. `width` (a power of two <= 32) splits
// the warp into independent segments of that many lanes.

__global__ void segment_broadcast_kernel(const float* x, float* out, int n, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = __shfl_sync(FULL, i < n ? x[i] : 0.0f, 0, width);   // lane 0 of my segment
    if (i < n) out[i] = v;
}

extern "C" void segment_broadcast(const float* x, float* out, int n, int width) {
    segment_broadcast_kernel<<<blocks_for(n), THREADS>>>(x, out, n, width);
    cudaDeviceSynchronize();
}

// ---------------------------------------------------------------- 4. __shfl_up_sync
// Inclusive prefix sum (Hillis-Steele) inside segments of `width` lanes. log2(width) steps;
// a lane adds the value `offset` lanes below it only if that lane exists in its segment.

__device__ __forceinline__ int segment_scan(int v, int width) {
    int lane = threadIdx.x & (width - 1);
    for (int offset = 1; offset < width; offset <<= 1) {
        int other = __shfl_up_sync(FULL, v, offset, width);   // lane i reads lane i - offset
        if (lane >= offset) v += other;                       // lanes < offset keep their value
    }
    return v;
}

__global__ void warp_scan_kernel(const int* x, int* out, int n, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int s = segment_scan(i < n ? x[i] : 0, width);
    if (i < n) out[i] = s;
}

extern "C" void warp_scan(const int* x, int* out, int n, int width) {
    warp_scan_kernel<<<blocks_for(n), THREADS>>>(x, out, n, width);
    cudaDeviceSynchronize();
}

// ---------------------------------------------------------------- 5. warp scans -> block scan
// Scan of the whole array with one block of 1024 threads: scan inside each warp, scan the
// 32 warp totals with one more warp scan, add each warp's offset, carry into the next chunk.

__global__ void block_scan_kernel(const int* x, int* out, int n) {
    __shared__ int warp_total[32];
    __shared__ int carry;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (threadIdx.x == 0) carry = 0;
    __syncthreads();
    for (int base = 0; base < n; base += blockDim.x) {
        int i = base + threadIdx.x;
        int v = segment_scan(i < n ? x[i] : 0, 32);       // inclusive, within the warp
        if (lane == 31) warp_total[warp] = v;
        __syncthreads();                                  // totals written before warp 0 reads
        if (warp == 0) {
            int t = lane < (int)(blockDim.x >> 5) ? warp_total[lane] : 0;
            warp_total[lane] = segment_scan(t, 32) - t;   // exclusive: offset of each warp
        }
        __syncthreads();                                  // offsets ready
        v += warp_total[warp] + carry;
        if (i < n) out[i] = v;
        __syncthreads();                                  // everyone has read carry
        if (threadIdx.x == blockDim.x - 1) carry = v;     // last thread holds the chunk total
        __syncthreads();                                  // carry updated before next chunk
    }
}

extern "C" void block_scan(const int* x, int* out, int n) {
    block_scan_kernel<<<1, 1024>>>(x, out, n);
    cudaDeviceSynchronize();
}

// ---------------------------------------------------------------- 6. votes
// One vote per warp: did any lane see a negative, are all lanes positive, and which lanes
// are positive (a 32-bit ballot). Tail lanes vote neutral (not negative, positive).

__global__ void warp_votes_kernel(const float* x, int* any_neg, int* all_pos, unsigned* ballot, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = i < n ? x[i] : 1.0f;
    int a = __any_sync(FULL, v < 0.0f);
    int b = __all_sync(FULL, v > 0.0f);
    unsigned m = __ballot_sync(FULL, i < n && v > 0.0f);   // bit k = lane k's predicate
    if ((threadIdx.x & 31) == 0 && i < n) {
        any_neg[i / 32] = a;
        all_pos[i / 32] = b;
        ballot[i / 32] = m;
    }
}

extern "C" void warp_votes(const float* x, int* any_neg, int* all_pos, unsigned* ballot, int n) {
    warp_votes_kernel<<<blocks_for(n), THREADS>>>(x, any_neg, all_pos, ballot, n);
    cudaDeviceSynchronize();
}

// ---------------------------------------------------------------- 7. ballot + popc: compaction
// Keep the positive elements. One atomicAdd per warp instead of one per element: the ballot
// says how many lanes keep a value, lane 0 reserves that many slots, a shuffle broadcasts the
// base, and each lane's slot is the number of keeping lanes below it.
// Order is kept inside a warp; warps land in whatever order their atomics run.

__global__ void compact_positive_kernel(const float* x, float* out, int* count, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    bool keep = i < n && x[i] > 0.0f;
    unsigned m = __ballot_sync(FULL, keep);
    int base = 0;
    if (lane == 0 && m) base = atomicAdd(count, __popc(m));
    base = __shfl_sync(FULL, base, 0);
    unsigned below = m & ((1u << lane) - 1u);            // keeping lanes with a lower index
    if (keep) out[base + __popc(below)] = x[i];
}

extern "C" void compact_positive(const float* x, float* out, int* count, int n) {
    cudaMemset(count, 0, sizeof(int));
    compact_positive_kernel<<<blocks_for(n), THREADS>>>(x, out, count, n);
    cudaDeviceSynchronize();
}

// ---------------------------------------------------------------- 8. __match_any_sync: histogram
// Lanes that hit the same bin find each other, and only the lowest of them does the atomic,
// adding the group size. Skewed data (many lanes per bin) gets far fewer atomics.

__global__ void histogram_match_kernel(const int* bins, int* hist, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int b = i < n ? bins[i] : -1;                        // -1: tail lanes form their own group
    unsigned peers = __match_any_sync(FULL, b);          // lanes holding the same bin
    int leader = __ffs(peers) - 1;                       // lowest lane of the group
    if (b >= 0 && lane == leader) atomicAdd(&hist[b], __popc(peers));
}

extern "C" void histogram_match(const int* bins, int* hist, int n, int nbins) {
    cudaMemset(hist, 0, nbins * sizeof(int));
    histogram_match_kernel<<<blocks_for(n), THREADS>>>(bins, hist, n);
    cudaDeviceSynchronize();
}

// ---------------------------------------------------------------- 9. __reduce_*_sync (sm_80+)
// Integer warp reductions in one instruction on Ampere and later; shuffles elsewhere.

__device__ __forceinline__ int warp_add_int(int v) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    return __reduce_add_sync(FULL, v);
#else
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL, v, o);
    return v;
#endif
}

__device__ __forceinline__ int warp_max_int(int v) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    return __reduce_max_sync(FULL, v);
#else
    for (int o = 16; o > 0; o >>= 1) v = max(v, __shfl_xor_sync(FULL, v, o));
    return v;
#endif
}

__global__ void int_sum_max_kernel(const int* x, int* sum, int* mx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int s = warp_add_int(i < n ? x[i] : 0);
    int m = warp_max_int(i < n ? x[i] : INT_MIN);
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(sum, s);                               // integer atomics: order does not matter
        atomicMax(mx, m);
    }
}

extern "C" void int_sum_max(const int* x, int* sum, int* mx, int n) {
    int zero = 0, lowest = INT_MIN;
    cudaMemcpy(sum, &zero, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(mx, &lowest, sizeof(int), cudaMemcpyHostToDevice);
    int_sum_max_kernel<<<blocks_for(n), THREADS>>>(x, sum, mx, n);
    cudaDeviceSynchronize();
}

// ---------------------------------------------------------------- 10. __syncthreads_count
// A barrier that also counts: how many threads of the block passed a true predicate.

__global__ void count_positive_kernel(const float* x, int* per_block, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int c = __syncthreads_count(i < n && x[i] > 0.0f);   // every thread of the block calls it
    if (threadIdx.x == 0) per_block[blockIdx.x] = c;
}

extern "C" void count_positive(const float* x, int* per_block, int n) {
    count_positive_kernel<<<blocks_for(n), THREADS>>>(x, per_block, n);
    cudaDeviceSynchronize();
}

// ---------------------------------------------------------------- 11. cooperative groups: block sum
// The same reduction as warp_sum_down + a shared-memory step, written with group objects:
// no masks, no lane arithmetic, and cg::reduce picks the best instruction for the type.

__global__ void cg_block_sum_kernel(const float* x, float* per_block, int n) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    __shared__ float warp_sum[THREADS / 32];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float s = cg::reduce(warp, i < n ? x[i] : 0.0f, cg::plus<float>());   // every lane gets it
    if (warp.thread_rank() == 0) warp_sum[warp.meta_group_rank()] = s;
    block.sync();                                        // same as __syncthreads()
    if (warp.meta_group_rank() == 0) {
        float t = warp.thread_rank() < warp.meta_group_size() ? warp_sum[warp.thread_rank()] : 0.0f;
        t = cg::reduce(warp, t, cg::plus<float>());
        if (warp.thread_rank() == 0) per_block[blockIdx.x] = t;
    }
}

extern "C" void cg_block_sum(const float* x, float* per_block, int n) {
    cg_block_sum_kernel<<<blocks_for(n), THREADS>>>(x, per_block, n);
    cudaDeviceSynchronize();
}

// ---------------------------------------------------------------- 12. cooperative groups: tile scans
// Tiles of 16 threads: inclusive and exclusive prefix sums inside each tile.

__global__ void cg_tile_scan_kernel(const int* x, int* inc, int* exc, int n) {
    auto tile = cg::tiled_partition<16>(cg::this_thread_block());
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int v = i < n ? x[i] : 0;
    int a = cg::inclusive_scan(tile, v);                 // default operator: plus
    int b = cg::exclusive_scan(tile, v);                 // rank 0 gets 0
    if (i < n) {
        inc[i] = a;
        exc[i] = b;
    }
}

extern "C" void cg_tile_scan(const int* x, int* inc, int* exc, int n) {
    cg_tile_scan_kernel<<<blocks_for(n), THREADS>>>(x, inc, exc, n);
    cudaDeviceSynchronize();
}

// ---------------------------------------------------------------- 13. grid.sync(): one-kernel normalize
// y = x / sum(x) in a single kernel. Phase 1: each warp writes a partial sum. grid.sync()
// waits for every block of the grid. Phase 2: every thread adds the partials (in the same
// order, so the same bits each run) and scales its elements.
//
// grid.sync() only works with cudaLaunchCooperativeKernel and a grid small enough that every
// block is resident at once, so the grid is sized from the occupancy calculator, and a
// grid-stride loop covers any n.

#define COOP_THREADS 128

__global__ void normalize_kernel(const float* x, float* y, float* partial, int n) {
    cg::grid_group grid = cg::this_grid();
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    float s = 0.0f;
    for (long long i = grid.thread_rank(); i < n; i += grid.size()) s += x[i];
    s = cg::reduce(warp, s, cg::plus<float>());
    int warps = (int)(grid.size() / 32);
    if (warp.thread_rank() == 0) partial[grid.thread_rank() / 32] = s;

    grid.sync();                                         // all partials written, grid-wide

    float total = 0.0f;
    for (int w = 0; w < warps; ++w) total += partial[w];
    for (long long i = grid.thread_rank(); i < n; i += grid.size()) y[i] = x[i] / total;
}

extern "C" void normalize_by_sum(const float* x, float* y, int n) {
    int dev = 0, coop = 0, sms = 0, per_sm = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, dev);
    if (!coop) return;                                   // device cannot do grid.sync()
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, normalize_kernel, COOP_THREADS, 0);
    int blocks = sms * per_sm;                           // all of them can be resident at once
    float* partial;
    cudaMalloc(&partial, blocks * (COOP_THREADS / 32) * sizeof(float));
    void* args[] = {(void*)&x, (void*)&y, (void*)&partial, (void*)&n};
    cudaLaunchCooperativeKernel(normalize_kernel, dim3(blocks), dim3(COOP_THREADS), args, 0, 0);
    cudaDeviceSynchronize();
    cudaFree(partial);
}
