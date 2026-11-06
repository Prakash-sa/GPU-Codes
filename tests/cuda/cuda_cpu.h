// CPU emulation of the CUDA features the solutions use, so the .cu files can be tested on a
// machine with no NVIDIA GPU. Not a general CUDA emulator: just enough, and strict about it.
//
//   * Blocks run one at a time; the threads of a block run as real OS threads, so races,
//     missing __syncthreads and wrong shared-memory indexing show up as wrong results.
//   * __syncthreads is a real barrier. A thread that returns early drops out of it, which is
//     how Volta and later GPUs treat exited threads.
//   * Warp shuffles exchange values between the 32 threads of a warp through a barrier, so
//     code that relies on implicit warp lock-step (no __syncwarp) gets wrong answers here.
//   * __shared__ becomes `static`: correct because only one block runs at a time. A
//     cooperative launch (grid.sync) runs all blocks at once, so its kernels must not use
//     __shared__ here.
//   * Warp votes, match and __reduce_*_sync see the lanes of the warp that have not exited;
//     the mask argument is not checked, so call them from converged code.
//   * Device memory is host memory; cudaMemcpy is memcpy.
//
// run_tests.py rewrites `kernel<<<grid, block>>>(args)` into cuda_cpu::launch(...) and
// compiles each .cu file as C++20 with this header in front.
#pragma once

#include <algorithm>
#include <atomic>
#include <barrier>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __noinline__
#define __restrict__ __restrict
#define __launch_bounds__(...)
#define __align__(n) __attribute__((aligned(n)))
#define __shared__ static

struct dim3 {
    unsigned x, y, z;
    constexpr dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) : x(x_), y(y_), z(z_) {}
};
struct uint3 { unsigned x, y, z; };

inline thread_local uint3 threadIdx, blockIdx;
inline thread_local dim3 blockDim, gridDim;
constexpr int warpSize = 32;

// ------------------------------------------------------------------ vector types, half

struct alignas(8) float2 { float x, y; };
struct alignas(16) float4 { float x, y, z, w; };
inline float2 make_float2(float x, float y) { return {x, y}; }
inline float4 make_float4(float x, float y, float z, float w) { return {x, y, z, w}; }

typedef _Float16 __half;
typedef __half half;
inline float __half2float(half h) { return static_cast<float>(h); }
inline half __float2half(float f) { return static_cast<half>(f); }
inline half __float2half_rn(float f) { return static_cast<half>(f); }

// ------------------------------------------------------------------ math intrinsics

inline float rsqrtf(float x) { return 1.0f / std::sqrt(x); }
inline double rsqrt(double x) { return 1.0 / std::sqrt(x); }
inline float __expf(float x) { return std::exp(x); }
inline float __logf(float x) { return std::log(x); }
inline float __fdividef(float a, float b) { return a / b; }
inline float __saturatef(float x) { return std::fmin(std::fmax(x, 0.0f), 1.0f); }
// Round-to-nearest single operations that the compiler may not fuse into an FMA.
inline float __fadd_rn(float a, float b) { volatile float r = a + b; return r; }
inline float __fsub_rn(float a, float b) { volatile float r = a - b; return r; }
inline float __fmul_rn(float a, float b) { volatile float r = a * b; return r; }
template <class A, class B> inline auto max(A a, B b) { return a > b ? a : b; }
template <class A, class B> inline auto min(A a, B b) { return a < b ? a : b; }
template <class T> inline T __ldg(const T* p) { return *p; }

// ------------------------------------------------------------------ block and warp sync

namespace cuda_cpu {

struct Block {
    std::barrier<> bar;
    std::vector<std::unique_ptr<std::barrier<>>> warp_bar;
    std::vector<uint64_t> slot;              // one shuffle slot per thread
    std::vector<int> pred;                   // one vote slot per thread
    std::vector<char> alive;                 // cleared when a thread returns
    std::barrier<>* grid_bar = nullptr;      // set for cooperative launches
    explicit Block(int n) : bar(n), slot(n), pred(n), alive(n, 1) {
        for (int w = 0; w * 32 < n; ++w)
            warp_bar.push_back(std::make_unique<std::barrier<>>(std::min(32, n - w * 32)));
    }
};

inline thread_local Block* blk = nullptr;
inline thread_local int tid = 0;             // linear thread index in the block

template <class T> inline T shuffle(T v, int src_lane) {
    static_assert(sizeof(T) <= 8, "shuffle of types up to 8 bytes");
    int base = tid & ~31;
    uint64_t bits = 0;
    std::memcpy(&bits, &v, sizeof(T));
    blk->slot[tid] = bits;
    blk->warp_bar[tid / 32]->arrive_and_wait();
    T out;
    uint64_t got = blk->slot[base + src_lane];
    std::memcpy(&out, &got, sizeof(T));
    blk->warp_bar[tid / 32]->arrive_and_wait();     // nobody overwrites a slot until all read
    return out;
}

// Every live lane of the calling warp publishes v; returns all 32 values and the live mask.
template <class T> inline unsigned gather(T v, T (&all)[32]) {
    int base = tid & ~31, n = std::min(32, int(blk->slot.size()) - base);
    uint64_t bits = 0;
    std::memcpy(&bits, &v, sizeof(T));
    blk->slot[tid] = bits;
    blk->warp_bar[tid / 32]->arrive_and_wait();
    unsigned live = 0;
    for (int l = 0; l < n; ++l) {
        if (!blk->alive[base + l]) continue;
        live |= 1u << l;
        std::memcpy(&all[l], &blk->slot[base + l], sizeof(T));
    }
    blk->warp_bar[tid / 32]->arrive_and_wait();
    return live;
}

// Barrier over the block that also combines one int per thread (__syncthreads_count etc.).
template <class Op> inline int block_vote(int p, Op op, int init) {
    blk->pred[tid] = p;
    blk->bar.arrive_and_wait();
    int r = init;
    for (size_t t = 0; t < blk->pred.size(); ++t)
        if (blk->alive[t]) r = op(r, blk->pred[t]);
    blk->bar.arrive_and_wait();
    return r;
}

inline void run_thread(Block& b, int t, dim3 grid, dim3 block, uint3 bid, auto&& body) {
    blk = &b;
    tid = t;
    blockDim = block;
    gridDim = grid;
    blockIdx = bid;
    threadIdx = {unsigned(t % block.x), unsigned(t / block.x % block.y),
                 unsigned(t / (block.x * block.y))};
    body();
    b.alive[t] = 0;
    b.bar.arrive_and_drop();                                // exited: stop counting
    b.warp_bar[t / 32]->arrive_and_drop();
    if (b.grid_bar) b.grid_bar->arrive_and_drop();
}

// All blocks run at the same time, as cudaLaunchCooperativeKernel guarantees, so a
// grid-wide barrier works.
template <class K, class... A>
void launch_cooperative(dim3 grid, dim3 block, K kernel, A... args) {
    const int n = int(block.x * block.y * block.z), nb = int(grid.x * grid.y * grid.z);
    std::barrier<> gbar(n * nb);
    std::vector<std::unique_ptr<Block>> blocks;
    std::vector<std::thread> threads;
    for (int k = 0; k < nb; ++k) {
        blocks.push_back(std::make_unique<Block>(n));
        blocks.back()->grid_bar = &gbar;
    }
    for (int k = 0; k < nb; ++k)
        for (int t = 0; t < n; ++t)
            threads.emplace_back([&, k, t] {
                uint3 bid{unsigned(k % grid.x), unsigned(k / grid.x % grid.y), unsigned(k / (grid.x * grid.y))};
                run_thread(*blocks[k], t, grid, block, bid, [&] { kernel(args...); });
            });
    for (auto& th : threads) th.join();
}

template <class K, class... A>
void launch(dim3 grid, dim3 block, K kernel, A... args) {
    const int n = int(block.x * block.y * block.z);
    for (unsigned bz = 0; bz < grid.z; ++bz)
        for (unsigned by = 0; by < grid.y; ++by)
            for (unsigned bx = 0; bx < grid.x; ++bx) {
                Block b(n);
                std::vector<std::thread> threads;
                threads.reserve(n);
                for (int t = 0; t < n; ++t)
                    threads.emplace_back([&, t] {
                        run_thread(b, t, grid, block, uint3{bx, by, bz}, [&] { kernel(args...); });
                    });
                for (auto& th : threads) th.join();
            }
}

}  // namespace cuda_cpu

inline void __syncthreads() { cuda_cpu::blk->bar.arrive_and_wait(); }
inline void __syncwarp(unsigned = 0xffffffffu) { cuda_cpu::blk->warp_bar[cuda_cpu::tid / 32]->arrive_and_wait(); }

template <class T> inline T __shfl_sync(unsigned, T v, int src, int width = 32) {
    int lane = cuda_cpu::tid & 31;
    return cuda_cpu::shuffle(v, (lane & ~(width - 1)) + (src & (width - 1)));
}
template <class T> inline T __shfl_down_sync(unsigned, T v, unsigned d, int width = 32) {
    int lane = cuda_cpu::tid & 31;
    int src = (lane % width) + int(d) < width ? lane + int(d) : lane;
    return cuda_cpu::shuffle(v, src);
}
template <class T> inline T __shfl_up_sync(unsigned, T v, unsigned d, int width = 32) {
    int lane = cuda_cpu::tid & 31;
    int src = (lane % width) >= int(d) ? lane - int(d) : lane;
    return cuda_cpu::shuffle(v, src);
}
template <class T> inline T __shfl_xor_sync(unsigned, T v, int mask, int width = 32) {
    int lane = cuda_cpu::tid & 31;
    int src = lane ^ mask;
    return cuda_cpu::shuffle(v, ((src & ~(width - 1)) == (lane & ~(width - 1))) ? src : lane);
}

// ------------------------------------------------------------------ votes, match, reduce

inline unsigned __activemask() {                       // here: the warp's live lanes
    int all[32];
    return cuda_cpu::gather(0, all);
}
inline unsigned __ballot_sync(unsigned, int p) {
    int all[32];
    unsigned live = cuda_cpu::gather(p ? 1 : 0, all), m = 0;
    for (int l = 0; l < 32; ++l)
        if ((live >> l & 1) && all[l]) m |= 1u << l;
    return m;
}
inline int __any_sync(unsigned mask, int p) { return __ballot_sync(mask, p) != 0; }
inline int __all_sync(unsigned, int p) {
    int all[32];
    unsigned live = cuda_cpu::gather(p ? 1 : 0, all);
    for (int l = 0; l < 32; ++l)
        if ((live >> l & 1) && !all[l]) return 0;
    return 1;
}
template <class T> inline unsigned __match_any_sync(unsigned, T v) {
    T all[32];
    unsigned live = cuda_cpu::gather(v, all), m = 0;
    for (int l = 0; l < 32; ++l)
        if ((live >> l & 1) && all[l] == v) m |= 1u << l;
    return m;
}
template <class T> inline unsigned __match_all_sync(unsigned, T v, int* pred) {
    T all[32];
    unsigned live = cuda_cpu::gather(v, all);
    for (int l = 0; l < 32; ++l)
        if ((live >> l & 1) && all[l] != v) { *pred = 0; return 0; }
    *pred = 1;
    return live;
}
template <class T, class Op> inline T warp_combine(T v, Op op) {
    T all[32];
    unsigned live = cuda_cpu::gather(v, all);
    T r{};
    bool first = true;
    for (int l = 0; l < 32; ++l)
        if (live >> l & 1) { r = first ? all[l] : op(r, all[l]); first = false; }
    return r;
}
inline unsigned __reduce_add_sync(unsigned, unsigned v) { return warp_combine(v, [](unsigned a, unsigned b) { return a + b; }); }
inline int __reduce_add_sync(unsigned, int v) { return warp_combine(v, [](int a, int b) { return a + b; }); }
inline int __reduce_min_sync(unsigned, int v) { return warp_combine(v, [](int a, int b) { return a < b ? a : b; }); }
inline int __reduce_max_sync(unsigned, int v) { return warp_combine(v, [](int a, int b) { return a > b ? a : b; }); }
inline unsigned __reduce_and_sync(unsigned, unsigned v) { return warp_combine(v, [](unsigned a, unsigned b) { return a & b; }); }
inline unsigned __reduce_or_sync(unsigned, unsigned v) { return warp_combine(v, [](unsigned a, unsigned b) { return a | b; }); }

inline int __syncthreads_count(int p) { return cuda_cpu::block_vote(p != 0, [](int a, int b) { return a + b; }, 0); }
inline int __syncthreads_and(int p) { return cuda_cpu::block_vote(p != 0, [](int a, int b) { return a & b; }, 1); }
inline int __syncthreads_or(int p) { return cuda_cpu::block_vote(p != 0, [](int a, int b) { return a | b; }, 0); }
inline void __threadfence_block() { std::atomic_thread_fence(std::memory_order_seq_cst); }
inline void __threadfence() { std::atomic_thread_fence(std::memory_order_seq_cst); }
inline void __threadfence_system() { std::atomic_thread_fence(std::memory_order_seq_cst); }

inline int __popc(unsigned x) { return __builtin_popcount(x); }
inline int __ffs(unsigned x) { return __builtin_ffs(int(x)); }
inline int __clz(unsigned x) { return x ? __builtin_clz(x) : 32; }
inline unsigned __brev(unsigned x) {
    unsigned r = 0;
    for (int i = 0; i < 32; ++i) r |= ((x >> i) & 1u) << (31 - i);
    return r;
}

// ------------------------------------------------------------------ cooperative groups (subset)
// thread_block, thread_block_tile<N> from tiled_partition, grid_group with a cooperative
// launch, cg::reduce and cg::inclusive_scan / exclusive_scan with plus / less / greater.

namespace cooperative_groups {

struct thread_block {
    void sync() const { __syncthreads(); }
    unsigned thread_rank() const { return unsigned(cuda_cpu::tid); }
    unsigned size() const { return blockDim.x * blockDim.y * blockDim.z; }
    unsigned num_threads() const { return size(); }
    dim3 group_index() const { return dim3(blockIdx.x, blockIdx.y, blockIdx.z); }
    dim3 thread_index() const { return dim3(threadIdx.x, threadIdx.y, threadIdx.z); }
    dim3 dim_threads() const { return blockDim; }
};
inline thread_block this_thread_block() { return {}; }

template <unsigned N> struct thread_block_tile {
    static_assert(N && N <= 32 && (N & (N - 1)) == 0, "tile size: power of two, at most 32");
    void sync() const { __syncwarp(); }
    unsigned thread_rank() const { return unsigned(cuda_cpu::tid) % N; }
    static constexpr unsigned size() { return N; }
    static constexpr unsigned num_threads() { return N; }
    unsigned meta_group_rank() const { return unsigned(cuda_cpu::tid) / N; }
    unsigned meta_group_size() const { return (this_thread_block().size() + N - 1) / N; }
    template <class T> T shfl(T v, int src) const { return __shfl_sync(~0u, v, src, N); }
    template <class T> T shfl_down(T v, unsigned d) const { return __shfl_down_sync(~0u, v, d, N); }
    template <class T> T shfl_up(T v, unsigned d) const { return __shfl_up_sync(~0u, v, d, N); }
    template <class T> T shfl_xor(T v, int m) const { return __shfl_xor_sync(~0u, v, m, N); }
    unsigned tile_bits(unsigned warp_bits) const {
        unsigned sh = (unsigned(cuda_cpu::tid) & 31) / N * N;
        return N == 32 ? warp_bits : (warp_bits >> sh) & ((1u << N) - 1);
    }
    unsigned ballot(int p) const { return tile_bits(__ballot_sync(~0u, p)); }
    int any(int p) const { return ballot(p) != 0; }
    int all(int p) const { return ballot(p) == (N == 32 ? ~0u : (1u << N) - 1); }
    template <class T> unsigned match_any(T v) const { return tile_bits(__match_any_sync(~0u, v)); }
};
template <unsigned N, class P> inline thread_block_tile<N> tiled_partition(const P&) { return {}; }

struct grid_group {
    void sync() const { cuda_cpu::blk->grid_bar->arrive_and_wait(); }
    bool is_valid() const { return cuda_cpu::blk->grid_bar != nullptr; }
    unsigned long long thread_rank() const {
        unsigned long long b = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
        return b * this_thread_block().size() + unsigned(cuda_cpu::tid);
    }
    unsigned long long size() const {
        return 1ull * gridDim.x * gridDim.y * gridDim.z * this_thread_block().size();
    }
    unsigned long long num_threads() const { return size(); }
};
inline grid_group this_grid() { return {}; }

template <class T = void> struct plus { T operator()(T a, T b) const { return a + b; } };
template <class T = void> struct less { T operator()(T a, T b) const { return a < b ? a : b; } };
template <class T = void> struct greater { T operator()(T a, T b) const { return a > b ? a : b; } };

template <unsigned N, class T, class Op> T reduce(const thread_block_tile<N>& g, T v, Op op) {
    for (unsigned o = N / 2; o > 0; o >>= 1) v = op(v, g.shfl_xor(v, int(o)));
    return v;
}
template <unsigned N, class T, class Op> T inclusive_scan(const thread_block_tile<N>& g, T v, Op op) {
    for (unsigned o = 1; o < N; o <<= 1) {
        T u = g.shfl_up(v, o);
        if (g.thread_rank() >= o) v = op(v, u);
    }
    return v;
}
template <unsigned N, class T> T inclusive_scan(const thread_block_tile<N>& g, T v) {
    return inclusive_scan(g, v, plus<T>());
}
template <unsigned N, class T, class Op> T exclusive_scan(const thread_block_tile<N>& g, T v, Op op) {
    T inc = inclusive_scan(g, v, op);
    T prev = g.shfl_up(inc, 1);
    return g.thread_rank() == 0 ? T{} : prev;
}
template <unsigned N, class T> T exclusive_scan(const thread_block_tile<N>& g, T v) {
    return exclusive_scan(g, v, plus<T>());
}

}  // namespace cooperative_groups

// ------------------------------------------------------------------ atomics

template <class T> inline T atomic_rmw(T* p, auto op) {
    std::atomic_ref<T> a(*p);
    T old = a.load();
    while (!a.compare_exchange_weak(old, op(old))) {}
    return old;
}
inline float atomicAdd(float* p, float v) { return atomic_rmw(p, [v](float o) { return o + v; }); }
inline double atomicAdd(double* p, double v) { return atomic_rmw(p, [v](double o) { return o + v; }); }
inline int atomicAdd(int* p, int v) { return std::atomic_ref<int>(*p).fetch_add(v); }
inline unsigned atomicAdd(unsigned* p, unsigned v) { return std::atomic_ref<unsigned>(*p).fetch_add(v); }
inline unsigned long long atomicAdd(unsigned long long* p, unsigned long long v) {
    return std::atomic_ref<unsigned long long>(*p).fetch_add(v);
}
inline int atomicMax(int* p, int v) { return atomic_rmw(p, [v](int o) { return o > v ? o : v; }); }
inline int atomicMin(int* p, int v) { return atomic_rmw(p, [v](int o) { return o < v ? o : v; }); }
inline unsigned long long atomicMin(unsigned long long* p, unsigned long long v) {
    return atomic_rmw(p, [v](unsigned long long o) { return o < v ? o : v; });
}
inline int atomicCAS(int* p, int cmp, int v) {
    std::atomic_ref<int>(*p).compare_exchange_strong(cmp, v);
    return cmp;
}
inline int atomicExch(int* p, int v) { return std::atomic_ref<int>(*p).exchange(v); }

// ------------------------------------------------------------------ runtime API

typedef int cudaError_t;
typedef void* cudaStream_t;
constexpr cudaError_t cudaSuccess = 0;
enum cudaMemcpyKind { cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost,
                      cudaMemcpyDeviceToDevice, cudaMemcpyDefault };

inline cudaError_t cudaMalloc(void** p, size_t n) {
    *p = std::aligned_alloc(256, (n + 255) / 256 * 256 + 256);
    return cudaSuccess;
}
template <class T> inline cudaError_t cudaMalloc(T** p, size_t n) { return cudaMalloc((void**)p, n); }
inline cudaError_t cudaFree(void* p) { std::free(p); return cudaSuccess; }
inline cudaError_t cudaMemcpy(void* d, const void* s, size_t n, cudaMemcpyKind) {
    std::memmove(d, s, n);
    return cudaSuccess;
}
inline cudaError_t cudaMemset(void* p, int v, size_t n) { std::memset(p, v, n); return cudaSuccess; }
inline cudaError_t cudaMemsetAsync(void* p, int v, size_t n, cudaStream_t = nullptr) { return cudaMemset(p, v, n); }
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline cudaError_t cudaGetDevice(int* d) { *d = 0; return cudaSuccess; }
enum cudaDeviceAttr { cudaDevAttrMultiProcessorCount, cudaDevAttrCooperativeLaunch };
inline cudaError_t cudaDeviceGetAttribute(int* v, cudaDeviceAttr a, int) {
    *v = a == cudaDevAttrMultiProcessorCount ? 4 : 1;
    return cudaSuccess;
}
template <class K> inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* n, K, int, size_t) {
    *n = 2;
    return cudaSuccess;
}

// cudaLaunchCooperativeKernel(kernel, grid, block, args): args[i] points at parameter i.
template <class... P, size_t... I>
inline void coop_unpack(void (*k)(P...), dim3 g, dim3 b, void** args, std::index_sequence<I...>) {
    cuda_cpu::launch_cooperative(g, b, k, *static_cast<std::remove_reference_t<P>*>(args[I])...);
}
template <class... P>
inline cudaError_t cudaLaunchCooperativeKernel(void (*k)(P...), dim3 g, dim3 b, void** args,
                                               size_t = 0, cudaStream_t = nullptr) {
    coop_unpack(k, g, b, args, std::index_sequence_for<P...>{});
    return cudaSuccess;
}
