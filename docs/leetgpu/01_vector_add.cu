// LeetGPU 1 - Vector Addition: C = A + B (float32, N up to 100M; perf N = 25M).
//
// Bandwidth bound: 12 bytes moved per FLOP. Each thread handles four consecutive
// elements with one 16-byte float4 load per input and one float4 store (fewer, wider
// memory instructions); the last N % 4 elements take a scalar path. A grid-stride loop
// lets a fixed-size grid cover any N.
//
// Comments that start with `mem:` say where the data lives. See memory-hierarchy.md.
#include <cuda_runtime.h>

__global__ void vector_add_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                  float* __restrict__ C, int N) {
    const int n4 = N / 4;
    const float4* A4 = reinterpret_cast<const float4*>(A);   // cudaMalloc: 256-byte aligned
    const float4* B4 = reinterpret_cast<const float4*>(B);
    float4* C4 = reinterpret_cast<float4*>(C);
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        // mem: DRAM -> registers, 16 bytes per thread, 512 contiguous bytes per warp
        float4 a = A4[i], b = B4[i];
        // mem: registers -> DRAM
        C4[i] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }
    for (int i = 4 * n4 + blockIdx.x * blockDim.x + threadIdx.x; i < N; i += stride)
        C[i] = A[i] + B[i];
}

// A, B, C are device pointers
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    const int threads = 256;
    int blocks = (N / 4 + threads - 1) / threads;
    blocks = blocks < 1 ? 1 : (blocks > 65535 ? 65535 : blocks);
    vector_add_kernel<<<blocks, threads>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
