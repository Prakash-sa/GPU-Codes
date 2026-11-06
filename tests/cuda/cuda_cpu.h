// CPU emulation of the CUDA features the solutions use, so the .cu files can be tested on a
// machine with no NVIDIA GPU. Not a general CUDA emulator: just enough, and strict about it.
//
//   * Blocks run one at a time; the threads of a block run as real OS threads, so races,
//     missing __syncthreads and wrong shared-memory indexing show up as wrong results.
//   * __syncthreads is a real barrier. A thread that returns early drops out of it, which is
//     how Volta and later GPUs treat exited threads.
//   * Warp shuffles exchange values between the 32 threads of a warp through a barrier, so
//     code that relies on implicit warp lock-step (no __syncwarp) gets wrong answers here.
//   * __shared__ becomes `static`: correct because only one block runs at a time.
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
#include <vector>

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __noinline__
#define __restrict__ __restrict
#define __launch_bounds__(...)
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
    explicit Block(int n) : bar(n), slot(n) {
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
                        blk = &b;
                        tid = t;
                        blockDim = block;
                        gridDim = grid;
                        blockIdx = {bx, by, bz};
                        threadIdx = {unsigned(t % block.x), unsigned(t / block.x % block.y),
                                     unsigned(t / (block.x * block.y))};
                        kernel(args...);
                        b.bar.arrive_and_drop();                  // exited: stop counting
                        b.warp_bar[t / 32]->arrive_and_drop();
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
enum cudaDeviceAttr { cudaDevAttrMultiProcessorCount };
inline cudaError_t cudaDeviceGetAttribute(int* v, cudaDeviceAttr, int) { *v = 4; return cudaSuccess; }
