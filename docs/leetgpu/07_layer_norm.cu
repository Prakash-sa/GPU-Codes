// LeetGPU 113 - Layer Normalization over rows of an N x C matrix (perf N = 65536, C = 512).
//
//     y = (x - mean) / sqrt(var + eps) * weight + bias      (biased variance, per row)
//
// One warp per row: rows are short (C = 512 is 16 floats per lane), so a warp holds a
// whole row in registers when C <= 32 * MAX_PER_LANE and needs no shared memory or
// __syncthreads at all: the reductions are warp shuffles. x is read once, y written once.
// Longer rows fall back to re-reading x from L1/L2.
//
// The variance is computed from centred values (sum((x - mean)^2)), not E[x^2] - E[x]^2,
// which cancels badly when |mean| >> std (the test adds an offset of 1000).
#include <cuda_runtime.h>

#define WARPS_PER_BLOCK 8
#define MAX_PER_LANE 16                                  // registers: rows up to C = 512

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__global__ void layer_norm_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                  const float* __restrict__ b, float* __restrict__ y,
                                  int N, int C, float eps) {
    int row = blockIdx.x * WARPS_PER_BLOCK + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    if (row >= N) return;                                // whole warp exits together
    const float* xr = x + (size_t)row * C;
    float* yr = y + (size_t)row * C;

    if (C <= 32 * MAX_PER_LANE) {
        float v[MAX_PER_LANE];                           // mem: the row, in registers
        float s = 0.0f;
#pragma unroll
        for (int k = 0; k < MAX_PER_LANE; ++k) {
            int c = lane + 32 * k;
            v[k] = c < C ? xr[c] : 0.0f;                 // mem: DRAM, coalesced per warp
            s += v[k];
        }
        float mean = warp_sum(s) / C;
        float q = 0.0f;
#pragma unroll
        for (int k = 0; k < MAX_PER_LANE; ++k) {
            float d = lane + 32 * k < C ? v[k] - mean : 0.0f;
            q = fmaf(d, d, q);
        }
        float rstd = rsqrtf(warp_sum(q) / C + eps);
#pragma unroll
        for (int k = 0; k < MAX_PER_LANE; ++k) {
            int c = lane + 32 * k;
            if (c < C) yr[c] = (v[k] - mean) * rstd * w[c] + b[c];
        }
    } else {                                             // long rows: three passes, L1/L2 hits
        float s = 0.0f;
        for (int c = lane; c < C; c += 32) s += xr[c];
        float mean = warp_sum(s) / C;
        float q = 0.0f;
        for (int c = lane; c < C; c += 32) { float d = xr[c] - mean; q = fmaf(d, d, q); }
        float rstd = rsqrtf(warp_sum(q) / C + eps);
        for (int c = lane; c < C; c += 32) yr[c] = (xr[c] - mean) * rstd * w[c] + b[c];
    }
}

// input, weight, bias, output are device pointers
extern "C" void solve(const float* input, const float* weight, const float* bias, float* output,
                      int N, int C, float eps) {
    int blocks = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    layer_norm_kernel<<<blocks, 32 * WARPS_PER_BLOCK>>>(input, weight, bias, output, N, C, eps);
    cudaDeviceSynchronize();
}
