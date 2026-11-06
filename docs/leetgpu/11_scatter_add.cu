// LeetGPU 125 - Scatter Add: output[index[i], :] += src[i, :]
// (src N x D float32, index int32; perf N = 1M, D = 128, M = 50K).
//
// This is message aggregation in a graph neural network: edges -> nodes.
//
// One thread per element of src, one atomicAdd per element. Consecutive threads take
// consecutive columns of the same source row, so a warp's 32 atomics go to 32 adjacent
// addresses of one output row: coalesced, and resolved in L2. With M = 50K targets,
// collisions between rows are spread thin. The summation order varies between runs, so
// the result is not bit-reproducible (it still meets 1e-4); for that, sort by target and
// reduce each run of rows in a fixed order (see the Triton version's solve_sorted).
#include <cuda_runtime.h>

#define THREADS 256

__global__ void scatter_add_kernel(const float* __restrict__ src, const int* __restrict__ index,
                                   float* __restrict__ out, int N, int D) {
    size_t total = (size_t)N * D;
    for (size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x; e < total;
         e += (size_t)gridDim.x * blockDim.x) {
        int i = (int)(e / D), d = (int)(e % D);
        int target = index[i];                           // mem: L1 hit for the rest of the row
        atomicAdd(&out[(size_t)target * D + d], src[e]); // mem: DRAM read; atomic in L2
    }
}

// src, index, output are device pointers
extern "C" void solve(const float* src, const int* index, float* output, int N, int D, int M) {
    cudaMemset(output, 0, (size_t)M * D * sizeof(float));   // output may hold stale data
    size_t total = (size_t)N * D;
    size_t blocks = (total + THREADS - 1) / THREADS;
    if (blocks > 65535 * 8) blocks = 65535 * 8;
    if (blocks < 1) blocks = 1;
    scatter_add_kernel<<<(unsigned)blocks, THREADS>>>(src, index, output, N, D);
    cudaDeviceSynchronize();
}
