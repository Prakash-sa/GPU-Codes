// LeetGPU 4 - Reduction: output[0] = sum(input) (float32, N up to 100M; perf N = 4M).
//
// Two deterministic passes, no atomics:
//   1. A fixed grid (a few blocks per SM) streams the array with a grid-stride loop. Each
//      thread sums its elements in a register, the block combines the threads with warp
//      shuffles and one shared-memory step, and the block writes one partial.
//   2. One block adds the partials in double.
// Same inputs -> same bits every run, which floating-point atomics cannot promise.
#include <cuda_runtime.h>

#define THREADS 256

template <class T> __device__ __forceinline__ T warp_sum(T v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Sum over the block; the result is valid in thread 0. blockDim.x: a multiple of 32.
template <class T> __device__ T block_sum(T v) {
    __shared__ T part[32];                               // mem: shared memory, one per warp
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_sum(v);                                     // mem: registers, via shuffles
    if (lane == 0) part[warp] = v;
    __syncthreads();
    v = threadIdx.x < (blockDim.x >> 5) ? part[threadIdx.x] : T(0);
    if (warp == 0) v = warp_sum(v);
    return v;
}

__global__ void partial_sums_kernel(const float* __restrict__ x, float* __restrict__ partial, int N) {
    float s = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x)
        s += x[i];                                       // mem: DRAM -> registers, coalesced
    s = block_sum(s);
    if (threadIdx.x == 0) partial[blockIdx.x] = s;
}

__global__ void final_sum_kernel(const float* __restrict__ partial, int P, float* __restrict__ out) {
    double s = 0.0;
    for (int i = threadIdx.x; i < P; i += blockDim.x) s += partial[i];
    s = block_sum(s);
    if (threadIdx.x == 0) out[0] = (float)s;
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int sms = 0, dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    int blocks = (N + THREADS * 8 - 1) / (THREADS * 8);      // >= 8 elements per thread
    blocks = blocks < 1 ? 1 : (blocks > 4 * sms ? 4 * sms : blocks);
    float* partial;
    cudaMalloc(&partial, blocks * sizeof(float));
    partial_sums_kernel<<<blocks, THREADS>>>(input, partial, N);
    final_sum_kernel<<<1, 1024>>>(partial, blocks, output);
    cudaDeviceSynchronize();
    cudaFree(partial);
}
