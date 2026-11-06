// LeetGPU 101 - N-body force (perf N = 8192):
//
//     F_i = sum_j m_j * (p_j - p_i) / (|p_j - p_i|^2 + eps^2)^(3/2),   eps = 1e-3
//
// The j = i term is zero because p_j - p_i = 0, so no test is needed for it.
//
// The classic GPU Gems 3 (ch. 31) layout: one thread per body i; the block streams all
// bodies through shared memory in tiles of THREADS, and every thread reuses each tile
// THREADS times from shared memory. The force accumulates in registers.
// (r^2)^(-3/2) = rsqrt(r^2)^3: one rsqrtf and two multiplies, no pow, no divide.
// About 20 FLOPs per pair, 67M pairs at N = 8192: compute bound.
#include <cuda_runtime.h>

#define THREADS 256
#define EPS2 1e-6f

__global__ void nbody_kernel(const float* __restrict__ pos, const float* __restrict__ mass,
                             float* __restrict__ force, int N) {
    __shared__ float4 tile[THREADS];                     // mem: shared, (x, y, z, m) per body
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float px = 0.0f, py = 0.0f, pz = 0.0f;
    if (i < N) { px = pos[3 * i]; py = pos[3 * i + 1]; pz = pos[3 * i + 2]; }
    float fx = 0.0f, fy = 0.0f, fz = 0.0f;               // mem: registers
    for (int t0 = 0; t0 < N; t0 += THREADS) {
        int j = t0 + threadIdx.x;
        tile[threadIdx.x] = j < N ? make_float4(pos[3 * j], pos[3 * j + 1], pos[3 * j + 2], mass[j])
                                  : make_float4(0.0f, 0.0f, 0.0f, 0.0f);   // m = 0: no force
        __syncthreads();
#pragma unroll 8
        for (int k = 0; k < THREADS; ++k) {
            float4 b = tile[k];                          // mem: shared, broadcast to the warp
            float dx = b.x - px, dy = b.y - py, dz = b.z - pz;
            float r2 = fmaf(dx, dx, fmaf(dy, dy, fmaf(dz, dz, EPS2)));
            float inv = rsqrtf(r2);
            float s = b.w * inv * inv * inv;
            fx = fmaf(dx, s, fx); fy = fmaf(dy, s, fy); fz = fmaf(dz, s, fz);
        }
        __syncthreads();
    }
    if (i < N) { force[3 * i] = fx; force[3 * i + 1] = fy; force[3 * i + 2] = fz; }
}

// positions, masses, forces are device pointers
extern "C" void solve(const float* positions, const float* masses, float* forces, int N) {
    nbody_kernel<<<(N + THREADS - 1) / THREADS, THREADS>>>(positions, masses, forces, N);
    cudaDeviceSynchronize();
}
