// LeetGPU 142 - RMS Norm backward (float32; perf N = 16384, C = 4096; tol 1e-3).
//
// Forward: y = x * rstd * w, rstd = 1 / sqrt(mean(x^2) + eps).  Given dy:
//     xhat = x * rstd,  g = dy * w
//     dx   = rstd * (g - xhat * mean(g * xhat))        <- a reduction along each ROW
//     dw   = sum over rows of dy * xhat                <- a reduction along each COLUMN
//
// The two reductions go in different directions:
//   * Kernel 1 runs P blocks (a few per SM). Block b takes rows b, b + P, ...; for each
//     row it reduces across the block (shuffles + shared memory) and writes dx. Thread t
//     owns columns t, t + 256, ... and adds dy * xhat for them into row b of a P x C
//     partial-dw buffer. No other thread touches those addresses, so no atomics.
//   * Kernel 2 sums the P partial rows per column, in a fixed order.
// Deterministic. Extra traffic: P x C floats, small next to x, dy and dx.
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
    __syncthreads();                                     // part/total are reused next call
    return r;
}

__global__ void rms_bwd_rows_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                    const float* __restrict__ dy, float* __restrict__ dx,
                                    float* __restrict__ dw_part, int N, int C, float eps) {
    float* my_dw = dw_part + (size_t)blockIdx.x * C;     // mem: this block's partial row (L2)
    for (int c = threadIdx.x; c < C; c += blockDim.x) my_dw[c] = 0.0f;
    for (int row = blockIdx.x; row < N; row += gridDim.x) {
        const float* xr = x + (size_t)row * C;
        const float* dyr = dy + (size_t)row * C;
        float ss = 0.0f;
        for (int c = threadIdx.x; c < C; c += blockDim.x) ss = fmaf(xr[c], xr[c], ss);
        float rstd = rsqrtf(block_sum_all(ss) / C + eps);
        float gx = 0.0f;
        for (int c = threadIdx.x; c < C; c += blockDim.x) gx += dyr[c] * w[c] * xr[c];   // mem: L1/L2
        float mgx = block_sum_all(gx) * rstd / C;        // mean(g * xhat)
        for (int c = threadIdx.x; c < C; c += blockDim.x) {
            float xhat = xr[c] * rstd, g = dyr[c] * w[c];
            dx[(size_t)row * C + c] = rstd * (g - xhat * mgx);   // mem: write dx once
            my_dw[c] += dyr[c] * xhat;                   // only this thread owns column c
        }
    }
}

__global__ void column_sum_kernel(const float* __restrict__ part, float* __restrict__ out, int P, int C) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float s = 0.0f;
    for (int p = 0; p < P; ++p) s += part[(size_t)p * C + c];   // mem: coalesced across threads
    out[c] = s;
}

// x, weight, dy, dx, dweight are device pointers
extern "C" void solve(const float* x, const float* weight, const float* dy, float* dx, float* dweight,
                      int N, int C, float eps) {
    int sms = 0, dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    int P = N < 2 * sms ? N : 2 * sms;
    float* part;
    cudaMalloc(&part, (size_t)P * C * sizeof(float));
    rms_bwd_rows_kernel<<<P, THREADS>>>(x, weight, dy, dx, part, N, C, eps);
    column_sum_kernel<<<(C + THREADS - 1) / THREADS, THREADS>>>(part, dweight, P, C);
    cudaDeviceSynchronize();
    cudaFree(part);
}
