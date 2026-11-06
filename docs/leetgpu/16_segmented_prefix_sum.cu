// LeetGPU 70 - Segmented exclusive prefix sum (perf N = 50M, segments of 256).
//
//     flags[i] = 1 starts a new segment;  out[i] = sum of values[s .. i-1] in i's segment
//
// Trick 1 - exclusive without subtraction. Let w[i] = 0 if flags[i] else values[i-1].
// An *inclusive* segmented scan of w is the exclusive scan of values.
//
// Trick 2 - a segmented scan is an ordinary scan with a different operator on (value, flag):
//     (v1, f1) + (v2, f2) = (f2 ? v2 : v1 + v2,  f1 | f2)
// It is associative, so the usual warp-shuffle scan works unchanged.
//
// Three phases ("reduce-then-scan"), 1024 elements per block:
//   1. each block scans its elements and writes its aggregate (value, any-flag);
//   2. one block scans the aggregates -> a carry into each block;
//   3. each block rescans and applies the carry to elements before its first flag.
// Everything is accumulated in double: a segment can span the whole array, and a float
// running sum over 50K elements already drifts past the 1e-3 tolerance.
#include <cuda_runtime.h>

#define THREADS 256
#define ITEMS 4
#define PER_BLOCK (THREADS * ITEMS)

struct Seg { double v; int f; };

__device__ __forceinline__ Seg combine(Seg a, Seg b) { return {b.f ? b.v : a.v + b.v, a.f | b.f}; }

// Inclusive scan across the warp.
__device__ __forceinline__ Seg warp_scan(Seg x) {
    int lane = threadIdx.x & 31;
    for (int o = 1; o < 32; o <<= 1) {
        Seg prev = {__shfl_up_sync(0xffffffffu, x.v, o), __shfl_up_sync(0xffffffffu, x.f, o)};
        if (lane >= o) x = combine(prev, x);
    }
    return x;
}

// Exclusive scan across the block; also returns the block total. blockDim.x: multiple of 32.
__device__ Seg block_exclusive_scan(Seg x, Seg* total) {
    __shared__ Seg warp_tot[32];                         // mem: shared, one per warp
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarps = blockDim.x >> 5;
    Seg incl = warp_scan(x);
    if (lane == 31) warp_tot[warp] = incl;
    __syncthreads();
    if (warp == 0) {
        Seg t = lane < nwarps ? warp_tot[lane] : Seg{0.0, 0};
        t = warp_scan(t);
        if (lane < nwarps) warp_tot[lane] = t;           // now inclusive over warps
    }
    __syncthreads();
    Seg warp_prefix = warp > 0 ? warp_tot[warp - 1] : Seg{0.0, 0};
    Seg ex = {__shfl_up_sync(0xffffffffu, incl.v, 1), __shfl_up_sync(0xffffffffu, incl.f, 1)};
    ex = lane == 0 ? warp_prefix : combine(warp_prefix, ex);
    *total = warp_tot[nwarps - 1];
    __syncthreads();                                     // warp_tot is reused by the next call
    return ex;
}

// Phase 1 (carry == nullptr): write each block's aggregate.
// Phase 3 (carry != nullptr): write out[i] = carry + local inclusive scan of w.
__global__ void block_scan_kernel(const float* __restrict__ values, const int* __restrict__ flags,
                                  float* __restrict__ out, const double* __restrict__ carry,
                                  double* __restrict__ agg_v, int* __restrict__ agg_f, int N) {
    const int base = blockIdx.x * PER_BLOCK + threadIdx.x * ITEMS;
    Seg item[ITEMS];
    Seg run = {0.0, 0};
    for (int k = 0; k < ITEMS; ++k) {
        int i = base + k;
        Seg w = {0.0, 0};
        if (i < N) {
            w.f = flags[i] != 0;                         // mem: DRAM -> registers
            w.v = (w.f || i == 0) ? 0.0 : (double)values[i - 1];
        }
        run = combine(run, w);
        item[k] = run;                                   // inclusive within this thread
    }
    Seg total;
    Seg prefix = block_exclusive_scan(run, &total);
    if (carry == nullptr) {
        if (threadIdx.x == 0) { agg_v[blockIdx.x] = total.v; agg_f[blockIdx.x] = total.f; }
        return;
    }
    Seg c = {carry[blockIdx.x], 0};
    for (int k = 0; k < ITEMS; ++k) {
        int i = base + k;
        if (i < N) out[i] = (float)combine(combine(c, prefix), item[k]).v;   // mem: write once
    }
}

// Phase 2: one block turns the block aggregates into exclusive carries.
__global__ void carry_kernel(const double* __restrict__ agg_v, const int* __restrict__ agg_f,
                             double* __restrict__ carry, int nb) {
    int per = (nb + blockDim.x - 1) / blockDim.x;
    int lo = threadIdx.x * per, hi = min(lo + per, nb);
    Seg run = {0.0, 0};
    for (int b = lo; b < hi; ++b) run = combine(run, Seg{agg_v[b], agg_f[b]});
    Seg total;
    Seg prefix = block_exclusive_scan(run, &total);
    for (int b = lo; b < hi; ++b) {
        carry[b] = prefix.v;
        prefix = combine(prefix, Seg{agg_v[b], agg_f[b]});
    }
}

// values, flags, output are device pointers
extern "C" void solve(const float* values, const int* flags, float* output, int N) {
    int nb = (N + PER_BLOCK - 1) / PER_BLOCK;
    double *agg_v, *carry;
    int* agg_f;
    cudaMalloc(&agg_v, nb * sizeof(double));
    cudaMalloc(&agg_f, nb * sizeof(int));
    cudaMalloc(&carry, nb * sizeof(double));
    block_scan_kernel<<<nb, THREADS>>>(values, flags, output, nullptr, agg_v, agg_f, N);
    carry_kernel<<<1, 1024>>>(agg_v, agg_f, carry, nb);
    block_scan_kernel<<<nb, THREADS>>>(values, flags, output, carry, agg_v, agg_f, N);
    cudaDeviceSynchronize();
    cudaFree(agg_v);
    cudaFree(agg_f);
    cudaFree(carry);
}
