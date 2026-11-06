// LeetGPU 38 - Nearest Neighbor: for each of N 3-D points, the index j != i of the
// closest point (squared distance; perf N = 10K). Checked exactly: ties go to the
// lowest index, as in torch.argmin.
//
// One thread per query point. The block walks over all points in tiles of THREADS:
// each thread loads one point of the tile into shared memory, and then every thread
// compares its query against the whole tile from shared memory. Each point is read from
// DRAM once per block instead of once per query. O(N^2) arithmetic, O(N) traffic:
// compute bound.
//
// Exact ties need exact arithmetic: dx*dx + dy*dy + dz*dz is evaluated with __fmul_rn /
// __fadd_rn so the compiler cannot fuse it into FMAs, which round differently from the
// reference and can flip a near-tie.
#include <cuda_runtime.h>
#include <float.h>

#define THREADS 256

__global__ void nearest_kernel(const float* __restrict__ pts, int* __restrict__ out, int N) {
    __shared__ float sx[THREADS], sy[THREADS], sz[THREADS];   // mem: shared, one tile (SoA)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float px = 0.0f, py = 0.0f, pz = 0.0f;
    if (i < N) { px = pts[3 * i]; py = pts[3 * i + 1]; pz = pts[3 * i + 2]; }
    float best = FLT_MAX;
    int best_j = -1;
    for (int t0 = 0; t0 < N; t0 += THREADS) {
        int j = t0 + threadIdx.x;
        if (j < N) { sx[threadIdx.x] = pts[3 * j]; sy[threadIdx.x] = pts[3 * j + 1]; sz[threadIdx.x] = pts[3 * j + 2]; }
        __syncthreads();
        int n = min(THREADS, N - t0);
        for (int k = 0; k < n; ++k) {                    // mem: shared -> registers (broadcast)
            float dx = __fsub_rn(px, sx[k]), dy = __fsub_rn(py, sy[k]), dz = __fsub_rn(pz, sz[k]);
            float d = __fadd_rn(__fadd_rn(__fmul_rn(dx, dx), __fmul_rn(dy, dy)), __fmul_rn(dz, dz));
            int jj = t0 + k;
            if (jj != i && (d < best || best_j < 0)) { best = d; best_j = jj; }   // strict <: lowest j wins
        }
        __syncthreads();                                 // before the next tile overwrites shared
    }
    if (i < N) out[i] = best_j < 0 ? 0 : best_j;         // N = 1: no j != i; argmin gives 0
}

// points, indices are device pointers
extern "C" void solve(const float* points, int* indices, int N) {
    nearest_kernel<<<(N + THREADS - 1) / THREADS, THREADS>>>(points, indices, N);
    cudaDeviceSynchronize();
}
