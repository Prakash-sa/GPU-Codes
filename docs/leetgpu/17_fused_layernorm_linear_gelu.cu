// LeetGPU 131 - Fused LayerNorm -> Linear -> exact GELU (float32; perf M = K = N = 4096).
//
//     xn = (x - mu) / sqrt(var + eps) * gamma + beta;  h = xn @ W + bias;  y = GELU_erf(h)
//
// Unfused: LayerNorm writes xn (M x K) to memory, the GEMM reads it back, and the
// bias + GELU pass reads and writes h (M x N) again. Fused, in two kernels:
//   1. row_stats_kernel: one warp per row computes mean and rstd (centred two-pass
//      variance: the inputs carry a large common offset, ~ -1000, where E[x^2] - E[x]^2
//      cancels catastrophically).
//   2. ln_gemm_gelu_kernel: the register-tiled GEMM from 15_spmm.cu, except that each
//      x tile is normalized while it is staged into shared memory (prologue fusion), and
//      bias + GELU are applied before the single store (epilogue fusion). xn and h never
//      exist in DRAM.
// Normalize first, then multiply: rewriting h = rstd * (x @ W') - ... runs the GEMM on
// the raw, offset x and cancels to garbage at this tolerance.
#include <cuda_runtime.h>
#include <math.h>

#define BM 64
#define BN 64
#define BK 16
#define TM 4
#define TN 4
#define THREADS 256
#define STAT_WARPS 8

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__global__ void row_stats_kernel(const float* __restrict__ x, float* __restrict__ mean,
                                 float* __restrict__ rstd, int M, int K, float eps) {
    int row = blockIdx.x * STAT_WARPS + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= M) return;                                // the whole warp exits together
    const float* xr = x + (size_t)row * K;
    float s = 0.0f;
    for (int k = lane; k < K; k += 32) s += xr[k];
    float mu = warp_sum(s) / K;
    float q = 0.0f;
    for (int k = lane; k < K; k += 32) { float d = xr[k] - mu; q = fmaf(d, d, q); }   // mem: L1/L2 re-read
    float var = warp_sum(q) / K;                         // all 32 lanes join the shuffles
    if (lane == 0) { mean[row] = mu; rstd[row] = rsqrtf(var + eps); }
}

__global__ void __launch_bounds__(THREADS)
ln_gemm_gelu_kernel(const float* __restrict__ x, const float* __restrict__ mean,
                    const float* __restrict__ rstd, const float* __restrict__ gamma,
                    const float* __restrict__ beta, const float* __restrict__ W,
                    const float* __restrict__ bias, float* __restrict__ y, int M, int K, int N) {
    __shared__ float As[BK][BM + 1];                     // mem: shared, normalized x tile
    __shared__ float Bs[BK][BN];                         // mem: shared, W tile
    const int tid = threadIdx.x;
    const int tx = tid % (BN / TN), ty = tid / (BN / TN);
    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
    float acc[TM][TN] = {};

    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int idx = tid; idx < BM * BK; idx += THREADS) {
            int m = idx / BK, k = idx % BK;
            int r = row0 + m, kk = k0 + k;
            float v = 0.0f;
            if (r < M && kk < K)                         // prologue fusion: LayerNorm on load
                v = (x[(size_t)r * K + kk] - mean[r]) * rstd[r] * gamma[kk] + beta[kk];
            As[k][m] = v;
        }
        for (int idx = tid; idx < BK * BN; idx += THREADS) {
            int k = idx / BN, n = idx % BN;
            int kk = k0 + k, c = col0 + n;
            Bs[k][n] = (kk < K && c < N) ? W[(size_t)kk * N + c] : 0.0f;
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
            if (r < M && c < N) {
                float h = acc[i][j] + bias[c];           // epilogue fusion: bias + exact GELU
                y[(size_t)r * N + c] = 0.5f * h * (1.0f + erff(h * 0.70710678118654752f));
            }
        }
    }
}

// x, gamma, beta, W, bias, y are device pointers
extern "C" void solve(const float* x, const float* gamma, const float* beta, const float* W,
                      const float* bias, float* y, int M, int K, int N, float eps) {
    float *mean, *rstd;
    cudaMalloc(&mean, M * sizeof(float));
    cudaMalloc(&rstd, M * sizeof(float));
    row_stats_kernel<<<(M + STAT_WARPS - 1) / STAT_WARPS, 32 * STAT_WARPS>>>(x, mean, rstd, M, K, eps);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    ln_gemm_gelu_kernel<<<grid, THREADS>>>(x, mean, rstd, gamma, beta, W, bias, y, M, K, N);
    cudaDeviceSynchronize();
    cudaFree(mean);
    cudaFree(rstd);
}
