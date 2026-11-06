// LeetGPU 83 - Fused Residual Add + RMS Norm over rows (perf N = C = 4096).
//
//     z = x + residual;   out = z / sqrt(mean(z^2) + eps) * weight
//
// Unfused PyTorch runs ~5 kernels and writes z and the intermediates to memory. Fused:
// one block per row reads x and residual, keeps z in registers, reduces sum(z^2) across
// the block, and writes out once: 12 bytes per element instead of ~40.
//
// C can be up to 65536. With 256 threads and PER_THREAD = 16 registers each, rows up to
// C = 4096 stay in registers; longer rows recompute z from x and residual in the second
// pass (the re-read of a row of <= 512 KB mostly hits L2). Offsets use size_t: N * C can
// exceed 2^31.
#include <cuda_runtime.h>

#define THREADS 256
#define PER_THREAD 16

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Sum over the block; every thread gets the result. blockDim.x: a multiple of 32.
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
    return total;
}

__global__ void residual_rms_norm_kernel(const float* __restrict__ x, const float* __restrict__ res,
                                         const float* __restrict__ w, float* __restrict__ out,
                                         int C, float eps) {
    size_t base = (size_t)blockIdx.x * C;
    const float* xr = x + base;
    const float* rr = res + base;
    float* orow = out + base;
    float s = 0.0f;
    if (C <= THREADS * PER_THREAD) {
        float z[PER_THREAD];                             // mem: z lives in registers
#pragma unroll
        for (int k = 0; k < PER_THREAD; ++k) {
            int c = threadIdx.x + k * THREADS;
            z[k] = c < C ? xr[c] + rr[c] : 0.0f;         // mem: DRAM, read x and residual once
            s = fmaf(z[k], z[k], s);
        }
        float rstd = rsqrtf(block_sum_all(s) / C + eps);
#pragma unroll
        for (int k = 0; k < PER_THREAD; ++k) {
            int c = threadIdx.x + k * THREADS;
            if (c < C) orow[c] = z[k] * rstd * w[c];     // mem: write out once
        }
    } else {
        for (int c = threadIdx.x; c < C; c += THREADS) { float z = xr[c] + rr[c]; s = fmaf(z, z, s); }
        float rstd = rsqrtf(block_sum_all(s) / C + eps);
        for (int c = threadIdx.x; c < C; c += THREADS)   // mem: re-read x, residual from L2
            orow[c] = (xr[c] + rr[c]) * rstd * w[c];
    }
}

// x, residual, weight, out are device pointers
extern "C" void solve(const float* x, const float* residual, const float* weight, float* out,
                      int N, int C, float eps) {
    residual_rms_norm_kernel<<<N, THREADS>>>(x, residual, weight, out, C, eps);
    cudaDeviceSynchronize();
}
