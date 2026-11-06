// LeetGPU 5 - Softmax over a whole 1-D array (float32, N up to 500K; perf N = 500K).
//
// One global max and one global sum are needed, so blocks must combine results. Two kernels:
//   1. Each block streams its part of x once, keeping a running (max m, sum s) per
//      thread with the online-softmax update, merges them across the block and writes
//      one partial (m_b, s_b).
//   2. Every block re-merges all partials (only a few hundred), giving M and S, then
//      writes exp(x - M) / S for its part.
// Merge rule:  (m1, s1) + (m2, s2) = (m, s1 * exp(m1 - m) + s2 * exp(m2 - m)),  m = max(m1, m2).
// Traffic: 2 reads + 1 write of x; 2 launches.
#include <cuda_runtime.h>
#include <math.h>

#define THREADS 256

struct MS { float m, s; };

__device__ __forceinline__ MS merge(MS a, MS b) {
    float m = fmaxf(a.m, b.m);
    if (m == -INFINITY) return {m, 0.0f};                // both empty
    return {m, a.s * __expf(a.m - m) + b.s * __expf(b.m - m)};
}

// Merge (m, s) across the block; every thread gets the result.
__device__ MS block_merge(MS v) {
    __shared__ MS part[32];
    __shared__ MS total;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int o = 16; o > 0; o >>= 1) {
        MS other = {__shfl_xor_sync(0xffffffffu, v.m, o), __shfl_xor_sync(0xffffffffu, v.s, o)};
        v = merge(v, other);
    }
    if (lane == 0) part[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = lane < (blockDim.x >> 5) ? part[lane] : MS{-INFINITY, 0.0f};
        for (int o = 16; o > 0; o >>= 1) {
            MS other = {__shfl_xor_sync(0xffffffffu, v.m, o), __shfl_xor_sync(0xffffffffu, v.s, o)};
            v = merge(v, other);
        }
        if (lane == 0) total = v;
    }
    __syncthreads();
    return total;
}

__global__ void partial_kernel(const float* __restrict__ x, MS* __restrict__ partial, int N) {
    MS v = {-INFINITY, 0.0f};
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x)
        v = merge(v, MS{x[i], 1.0f});                    // mem: DRAM -> registers, read 1
    v = block_merge(v);
    if (threadIdx.x == 0) partial[blockIdx.x] = v;
}

__global__ void normalize_kernel(const float* __restrict__ x, const MS* __restrict__ partial,
                                 int P, float* __restrict__ y, int N) {
    MS v = {-INFINITY, 0.0f};
    for (int i = threadIdx.x; i < P; i += blockDim.x) v = merge(v, partial[i]);   // mem: L2
    v = block_merge(v);
    float inv = 1.0f / v.s;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x)
        y[i] = __expf(x[i] - v.m) * inv;                 // mem: read 2 of x, write y
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int blocks = (N + THREADS * 4 - 1) / (THREADS * 4);
    blocks = blocks < 1 ? 1 : (blocks > 1024 ? 1024 : blocks);
    MS* partial;
    cudaMalloc(&partial, blocks * sizeof(MS));
    partial_kernel<<<blocks, THREADS>>>(input, partial, N);
    normalize_kernel<<<blocks, THREADS>>>(input, partial, blocks, output, N);
    cudaDeviceSynchronize();
    cudaFree(partial);
}
