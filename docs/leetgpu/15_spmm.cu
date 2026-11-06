// LeetGPU 75 - Sparse (M x N, ~65% zeros) x Dense (N x K) matmul, float32
// (perf M = 4096, N = 2048, K = 512; atol = rtol = 1e-3).
//
// The "sparse" A arrives as a dense row-major array. At ~35% density a sparse format
// does not pay for itself: building CSR costs a full pass over A plus a prefix sum, and a
// CSR SpMM gathers rows of B by column index (irregular access). A dense tiled GEMM reads
// A exactly once per column tile and keeps everything coalesced. The rule of thumb is that
// CSR wins below roughly 1-5% density, not at 35%.
//
// So this is the register-tiled FP32 GEMM (64 x 64 tile per block, 4 x 4 per thread, K
// steps of 16). FP32 FMA keeps full precision for the 1e-3 tolerance on 2048-long dot
// products; TF32 tensor cores alone would not (see the Triton version's tf32x3).
//
// One sparsity trick is free: skip a whole K-step when the A tile is all zeros. At 35%
// density that almost never happens, so it is left out.
#include <cuda_runtime.h>

#define BM 64
#define BN 64
#define BK 16
#define TM 4
#define TN 4
#define THREADS 256

__global__ void __launch_bounds__(THREADS)
sgemm_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
             int M, int N, int K) {
    // A: M x N, B: N x K, C: M x K (LeetGPU's names: the inner dimension is N)
    __shared__ float As[BK][BM + 1];                     // mem: shared, A tile k-major
    __shared__ float Bs[BK][BN];
    const int tid = threadIdx.x;
    const int tx = tid % (BN / TN), ty = tid / (BN / TN);
    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
    float acc[TM][TN] = {};

    for (int k0 = 0; k0 < N; k0 += BK) {
        for (int idx = tid; idx < BM * BK; idx += THREADS) {
            int m = idx / BK, k = idx % BK;
            int r = row0 + m, kk = k0 + k;
            As[k][m] = (r < M && kk < N) ? A[(size_t)r * N + kk] : 0.0f;
        }
        for (int idx = tid; idx < BK * BN; idx += THREADS) {
            int k = idx / BN, n = idx % BN;
            int kk = k0 + k, c = col0 + n;
            Bs[k][n] = (kk < N && c < K) ? B[(size_t)kk * K + c] : 0.0f;
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
            if (r < M && c < K) C[(size_t)r * K + c] = acc[i][j];
        }
    }
}

// A, B, C are device pointers
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K, int nnz) {
    (void)nnz;                                           // not needed by the dense kernel
    dim3 grid((K + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_kernel<<<grid, THREADS>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
