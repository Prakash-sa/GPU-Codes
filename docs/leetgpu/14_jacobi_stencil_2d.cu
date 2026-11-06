// LeetGPU 69 - 2-D Jacobi stencil, one sweep (float32, perf 8192 x 8192):
//
//     out[r, c] = 0.25 * (in[r-1, c] + in[r+1, c] + in[r, c-1] + in[r, c+1])   (interior)
//     out[r, c] = in[r, c]                                                       (boundary)
//
// Explicit shared-memory tiling, the step the Triton version leaves to the caches: a
// block of 32 x 8 threads loads a (TILE_R + 2) x (TILE_C + 2) tile including the one-cell
// halo into shared memory once, then each thread computes TILE_R / 8 = 4 outputs from
// shared memory. DRAM traffic: one read per cell plus the halo, one write.
#include <cuda_runtime.h>

#define TILE_C 32
#define TILE_R 32
#define BLOCK_R 8

__global__ void jacobi_kernel(const float* __restrict__ in, float* __restrict__ out, int rows, int cols) {
    __shared__ float s[TILE_R + 2][TILE_C + 2];          // mem: shared, tile + halo
    const int r0 = blockIdx.y * TILE_R - 1, c0 = blockIdx.x * TILE_C - 1;   // halo origin
    const int tid = threadIdx.y * TILE_C + threadIdx.x;
    for (int idx = tid; idx < (TILE_R + 2) * (TILE_C + 2); idx += TILE_C * BLOCK_R) {
        int lr = idx / (TILE_C + 2), lc = idx % (TILE_C + 2);
        int r = r0 + lr, c = c0 + lc;
        s[lr][lc] = (r >= 0 && r < rows && c >= 0 && c < cols) ? in[(size_t)r * cols + c] : 0.0f;
    }
    __syncthreads();
    const int c = blockIdx.x * TILE_C + threadIdx.x;
    for (int k = threadIdx.y; k < TILE_R; k += BLOCK_R) {
        int r = blockIdx.y * TILE_R + k;
        if (r >= rows || c >= cols) continue;
        int lr = k + 1, lc = threadIdx.x + 1;
        bool interior = r > 0 && r < rows - 1 && c > 0 && c < cols - 1;
        out[(size_t)r * cols + c] = interior                                 // mem: write once
            ? 0.25f * (s[lr - 1][lc] + s[lr + 1][lc] + s[lr][lc - 1] + s[lr][lc + 1])
            : s[lr][lc];
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 block(TILE_C, BLOCK_R);
    dim3 grid((cols + TILE_C - 1) / TILE_C, (rows + TILE_R - 1) / TILE_R);
    jacobi_kernel<<<grid, block>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
