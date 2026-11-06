// LeetGPU 22 - GEMM: C = alpha * (A @ B) + beta * C   (FP16 in/out, perf M = N = K = 1024).
//
// Classic shared-memory + register tiling on the CUDA cores, FP32 accumulation:
//   * a block of 256 threads owns a 64 x 64 tile of C and loops over K in steps of 16;
//   * each step, the block stages a 64 x 16 tile of A and a 16 x 64 tile of B in shared
//     memory (converted to float once), then each thread computes a 4 x 4 sub-tile of C
//     from registers: 8 shared loads feed 16 FMAs;
//   * the alpha/beta epilogue is fused: C is read once and written once.
// Each element of A and B is loaded from DRAM M/64 or N/64 times instead of N or M times.
//
// The next step up is the tensor cores (WMMA / mma.sync, or cuBLAS / CUTLASS), which is
// what the Triton version gets from tl.dot. This version is the portable baseline, and
// the shape of every tiled GEMM.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define BM 64
#define BN 64
#define BK 16
#define TM 4
#define TN 4
#define THREADS 256                                      // (BM / TM) * (BN / TN)

__global__ void __launch_bounds__(THREADS)
gemm_kernel(const half* __restrict__ A, const half* __restrict__ B, half* __restrict__ C,
            int M, int N, int K, float alpha, float beta) {
    __shared__ float As[BK][BM + 1];                     // mem: shared, A tile stored k-major
    __shared__ float Bs[BK][BN];                         // mem: shared, B tile
    const int tid = threadIdx.x;
    const int tx = tid % (BN / TN), ty = tid / (BN / TN);
    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
    float acc[TM][TN] = {};                              // mem: registers

    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int idx = tid; idx < BM * BK; idx += THREADS) {     // 4 elements per thread
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
            for (int i = 0; i < TM; ++i) a[i] = As[k][ty * TM + i];   // mem: shared -> registers
#pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = Bs[k][tx * TN + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        __syncthreads();                                 // before the next tile overwrites As/Bs
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        int r = row0 + ty * TM + i;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            int c = col0 + tx * TN + j;
            if (r < M && c < N) {
                size_t o = (size_t)r * N + c;
                C[o] = __float2half(alpha * acc[i][j] + beta * __half2float(C[o]));   // mem: read + write C once
            }
        }
    }
}

// A, B, C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_kernel<<<grid, THREADS>>>(A, B, C, M, N, K, alpha, beta);
    cudaDeviceSynchronize();
}
