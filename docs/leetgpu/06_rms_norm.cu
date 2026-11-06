// LeetGPU 50 - RMS Normalization of a 1-D array with scalar gamma/beta (perf N = 100K).
//
//     y = gamma * x / sqrt(mean(x^2) + eps) + beta
//
// Same shape of problem as 05_softmax.cu: one global statistic, then an elementwise pass.
//   1. Each block writes the partial sum of x^2 over its part of the array.
//   2. Each block re-adds the partials (a few hundred, in double), computes rstd once and
//      writes its part.
#include <cuda_runtime.h>

#define THREADS 256

template <class T> __device__ __forceinline__ T warp_sum(T v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Sum over the block; every thread gets the result. blockDim.x: a multiple of 32.
template <class T> __device__ T block_sum_all(T v) {
    __shared__ T part[32];
    __shared__ T total;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) part[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = lane < (blockDim.x >> 5) ? part[lane] : T(0);
        v = warp_sum(v);
        if (lane == 0) total = v;
    }
    __syncthreads();
    return total;
}

__global__ void sumsq_kernel(const float* __restrict__ x, float* __restrict__ partial, int N) {
    float s = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x)
        s = fmaf(x[i], x[i], s);                         // mem: DRAM -> registers
    s = block_sum_all(s);
    if (threadIdx.x == 0) partial[blockIdx.x] = s;
}

__global__ void scale_kernel(const float* __restrict__ x, const float* __restrict__ partial, int P,
                             float gamma, float beta, float eps, float* __restrict__ y, int N) {
    double s = 0.0;
    for (int i = threadIdx.x; i < P; i += blockDim.x) s += partial[i];          // mem: L2
    s = block_sum_all(s);
    float scale = gamma * (float)(1.0 / sqrt(s / N + eps));
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x)
        y[i] = fmaf(x[i], scale, beta);                  // mem: read x again, write y
}

// input, output are device pointers
extern "C" void solve(const float* input, float gamma, float beta, float* output, int N, float eps) {
    int blocks = (N + THREADS * 4 - 1) / (THREADS * 4);
    blocks = blocks < 1 ? 1 : (blocks > 1024 ? 1024 : blocks);
    float* partial;
    cudaMalloc(&partial, blocks * sizeof(float));
    sumsq_kernel<<<blocks, THREADS>>>(input, partial, N);
    scale_kernel<<<blocks, THREADS>>>(input, partial, blocks, gamma, beta, eps, output, N);
    cudaDeviceSynchronize();
    cudaFree(partial);
}
