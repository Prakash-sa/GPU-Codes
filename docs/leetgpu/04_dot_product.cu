// LeetGPU 17 - Dot Product: result[0] = sum(A * B) (float32, perf N = 4M).
//
// The reduction pattern from 03_reduction.cu with the multiply fused into the load loop:
// A * B is never written to memory. Partials are combined in double, in a fixed order,
// so the result is deterministic.
#include <cuda_runtime.h>

#define THREADS 256

template <class T> __device__ __forceinline__ T warp_sum(T v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Sum over the block; the result is valid in thread 0. blockDim.x: a multiple of 32.
template <class T> __device__ T block_sum(T v) {
    __shared__ T part[32];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) part[warp] = v;
    __syncthreads();
    v = threadIdx.x < (blockDim.x >> 5) ? part[threadIdx.x] : T(0);
    if (warp == 0) v = warp_sum(v);
    return v;
}

__global__ void partial_dot_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                   float* __restrict__ partial, int N) {
    float s = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x)
        s = fmaf(A[i], B[i], s);                         // mem: DRAM -> registers, 2 loads
    s = block_sum(s);
    if (threadIdx.x == 0) partial[blockIdx.x] = s;
}

__global__ void final_sum_kernel(const float* __restrict__ partial, int P, float* __restrict__ out) {
    double s = 0.0;
    for (int i = threadIdx.x; i < P; i += blockDim.x) s += partial[i];
    s = block_sum(s);
    if (threadIdx.x == 0) out[0] = (float)s;
}

// A, B, result are device pointers
extern "C" void solve(const float* A, const float* B, float* result, int N) {
    int sms = 0, dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    int blocks = (N + THREADS * 8 - 1) / (THREADS * 8);
    blocks = blocks < 1 ? 1 : (blocks > 4 * sms ? 4 * sms : blocks);
    float* partial;
    cudaMalloc(&partial, blocks * sizeof(float));
    partial_dot_kernel<<<blocks, THREADS>>>(A, B, partial, N);
    final_sum_kernel<<<1, 1024>>>(partial, blocks, result);
    cudaDeviceSynchronize();
    cudaFree(partial);
}
