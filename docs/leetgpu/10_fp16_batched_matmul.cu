// LeetGPU 57 - FP16 Batched MatMul: C[b] = A[b] @ B[b]  (perf BATCH = 32, M = N = K = 256).
//
// The tiled GEMM from 09_gemm.cu with the batch index on gridDim.z: one launch covers all
// batches, so 32 x 16 = 512 blocks run together instead of 32 launches of 16 blocks that
// would each leave most SMs idle. Batch offsets use size_t.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define BM 64
#define BN 64
#define BK 16
#define TM 4
#define TN 4
#define THREADS 256

__global__ void __launch_bounds__(THREADS)
bmm_kernel(const half* __restrict__ A, const half* __restrict__ B, half* __restrict__ C,
           int M, int N, int K) {
    __shared__ float As[BK][BM + 1];
    __shared__ float Bs[BK][BN];
    const size_t batch = blockIdx.z;
    A += batch * M * K;
    B += batch * K * N;
    C += batch * M * N;
    const int tid = threadIdx.x;
    const int tx = tid % (BN / TN), ty = tid / (BN / TN);
    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
    float acc[TM][TN] = {};

    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int idx = tid; idx < BM * BK; idx += THREADS) {
            int m = idx / BK, k = idx % BK;
            int r = row0 + m, kk = k0 + k;
            As[k][m] = (r < M && kk < K) ? __half2float(A[(size_t)r * K + kk]) : 0.0f;
        }
        for (int idx = tid; idx < BK * BN; idx += THREADS) {
            int k = idx / BN, n = idx % BN;
            int kk = k0 + k, c = col0 + n;
            Bs[k][n] = (kk < K && c < N) ? __half2float(B[(size_t)kk * N + c]) : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = As[k][ty * TM + i];
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
            if (r < M && c < N) C[(size_t)r * N + c] = __float2half(acc[i][j]);
        }
    }
}

// A, B, C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int BATCH, int M, int N, int K) {
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, BATCH);
    bmm_kernel<<<grid, THREADS>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
