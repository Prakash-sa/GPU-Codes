// LeetGPU 30 - Batched Matrix Multiplication: C[b] = A[b] @ B[b], float32, row-major
// (A is BATCH x M x K, B is BATCH x K x N, C is BATCH x M x N; perf BATCH = 32,
// M = N = K = 256; atol = rtol = 1e-5).
//
// Exact FP32 on the CUDA cores (the 1e-5 tolerance rules out TF32). Every product in the
// batch runs in one launch: blockIdx.z is the batch index.
//
// Tile choice is the interesting part. The 128 x 128 tile of 21_matrix_multiplication.cu
// gives 32 x 2 x 2 = 128 blocks here: under one block per SM on an H100 (132 SMs), with
// each block alone on its SM. A 64 x 64 tile (4 x 4 per thread) gives 32 x 4 x 4 = 512
// blocks, several per SM, which hides latency better on a problem this small, at the
// cost of lower reuse per shared-memory read.
//
// Comments that start with `mem:` say where the data lives. See memory-hierarchy.md.
#include <cuda_runtime.h>

#define BM 64
#define BN 64
#define BK 16
#define TM 4
#define TN 4
#define THREADS 256

__global__ void __launch_bounds__(THREADS)
bmm_fp32_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                int M, int N, int K) {
    __shared__ float As[BK][BM + 1];                     // mem: shared, A tile k-major (+1: no bank conflicts on the transpose)
    __shared__ float Bs[BK][BN];
    const size_t batch = blockIdx.z;
    A += batch * M * K;                                  // size_t: no overflow at 128 x 1024^2
    B += batch * K * N;
    C += batch * M * N;
    const int tid = threadIdx.x;
    const int tx = tid % (BN / TN), ty = tid / (BN / TN);
    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
    float acc[TM][TN] = {};                              // mem: registers

    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int idx = tid; idx < BM * BK; idx += THREADS) {
            int m = idx / BK, k = idx % BK;
            int r = row0 + m, kk = k0 + k;
            As[k][m] = (r < M && kk < K) ? A[(size_t)r * K + kk] : 0.0f;    // mem: DRAM/L2
        }
        for (int idx = tid; idx < BK * BN; idx += THREADS) {
            int k = idx / BN, n = idx % BN;
            int kk = k0 + k, c = col0 + n;
            Bs[k][n] = (kk < K && c < N) ? B[(size_t)kk * N + c] : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = As[k][ty * TM + i];          // mem: shared -> registers
#pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = Bs[k][tx * TN + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        int r = row0 + ty * TM + i;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            int c = col0 + tx * TN + j;
            if (r < M && c < N) C[(size_t)r * N + c] = acc[i][j];           // mem: registers -> DRAM
        }
    }
}

// A, B, C are device pointers
extern "C" void solve(const float* A, const float* B, float* C, int BATCH, int M, int N, int K) {
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, BATCH);
    bmm_fp32_kernel<<<grid, THREADS>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
