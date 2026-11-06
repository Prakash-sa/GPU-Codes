// LeetGPU 144 - Layer Norm backward, given the saved mean and rstd
// (float32; perf N = 16384, C = 4096; tol 1e-3).
//
//     xhat = (x - mean) * rstd,  g = dy * gamma
//     dx     = rstd * (g - mean(g) - xhat * mean(g * xhat))    <- row reductions
//     dgamma = sum over rows of dy * xhat                        <- column reductions
//     dbeta  = sum over rows of dy
//
// Same structure as 19_rms_norm_backward.cu, with one more row statistic (mean(g)) and
// two column accumulators per block (dgamma and dbeta partials) in a 2 x P x C buffer,
// each finished by a fixed-order column sum. Deterministic, no atomics.
#include <cuda_runtime.h>

#define THREADS 256

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Sum over the block; every thread gets the result. Safe to call repeatedly.
__device__ float block_sum_all(float v) {
    __shared__ float part[32];
    __shared__ float total;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) part[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = lane < (blockDim.x >> 5) ? part[lane] : 0.0f;
        v = warp_sum(v);
        if (lane == 0) total = v;
    }
    __syncthreads();
    float r = total;
    __syncthreads();
    return r;
}

__global__ void ln_bwd_rows_kernel(const float* __restrict__ x, const float* __restrict__ gamma,
                                   const float* __restrict__ mean, const float* __restrict__ rstd,
                                   const float* __restrict__ dy, float* __restrict__ dx,
                                   float* __restrict__ dg_part, float* __restrict__ db_part, int N, int C) {
    float* my_dg = dg_part + (size_t)blockIdx.x * C;     // mem: partial rows, L2
    float* my_db = db_part + (size_t)blockIdx.x * C;
    for (int c = threadIdx.x; c < C; c += blockDim.x) { my_dg[c] = 0.0f; my_db[c] = 0.0f; }
    for (int row = blockIdx.x; row < N; row += gridDim.x) {
        const float* xr = x + (size_t)row * C;
        const float* dyr = dy + (size_t)row * C;
        const float mu = mean[row], rs = rstd[row];
        float sg = 0.0f, sgx = 0.0f;
        for (int c = threadIdx.x; c < C; c += blockDim.x) {
            float g = dyr[c] * gamma[c], xhat = (xr[c] - mu) * rs;   // mem: DRAM, first read
            sg += g;
            sgx = fmaf(g, xhat, sgx);
        }
        float mg = block_sum_all(sg) / C;
        float mgx = block_sum_all(sgx) / C;
        for (int c = threadIdx.x; c < C; c += blockDim.x) {
            float g = dyr[c] * gamma[c], xhat = (xr[c] - mu) * rs;   // mem: L1/L2 re-read
            dx[(size_t)row * C + c] = rs * (g - mg - xhat * mgx);
            my_dg[c] += dyr[c] * xhat;
            my_db[c] += dyr[c];
        }
    }
}

__global__ void column_sum_kernel(const float* __restrict__ part, float* __restrict__ out, int P, int C) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float s = 0.0f;
    for (int p = 0; p < P; ++p) s += part[(size_t)p * C + c];
    out[c] = s;
}

// x, gamma, mean, rstd, dy, dx, dgamma, dbeta are device pointers
extern "C" void solve(const float* x, const float* gamma, const float* mean, const float* rstd,
                      const float* dy, float* dx, float* dgamma, float* dbeta, int N, int C) {
    int sms = 0, dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    int P = N < 2 * sms ? N : 2 * sms;
    float* part;
    cudaMalloc(&part, 2 * (size_t)P * C * sizeof(float));
    ln_bwd_rows_kernel<<<P, THREADS>>>(x, gamma, mean, rstd, dy, dx, part, part + (size_t)P * C, N, C);
    int cb = (C + THREADS - 1) / THREADS;
    column_sum_kernel<<<cb, THREADS>>>(part, dgamma, P, C);
    column_sum_kernel<<<cb, THREADS>>>(part + (size_t)P * C, dbeta, P, C);
    cudaDeviceSynchronize();
    cudaFree(part);
}
