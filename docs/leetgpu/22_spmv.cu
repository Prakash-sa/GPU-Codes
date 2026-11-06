// LeetGPU 18 - Sparse Matrix-Vector Multiplication: y = A @ x, float32
// (A is M x N row-major with ~60-70% zeros, *stored dense*; perf M = 1,000, N = 10,000;
// atol = rtol = 1e-3).
//
// A arrives dense, so every element must be read to know whether it is zero: the 40 MB of
// A *is* the cost, at 0.5 FLOP per byte. Skipping zeros saves FMAs (free here) and no
// bytes, and converting to CSR adds a second pass. So this is a bandwidth-bound dense GEMV:
//   * long rows (N > 2048, the perf case): one 256-thread block per row, float4 loads
//     (when N % 4 == 0 every row starts 16-byte aligned), a warp-shuffle + shared-memory
//     block sum at the end;
//   * short rows: one warp per row, 8 rows per block, so short rows don't leave 7/8 of
//     a block idle;
//   * x is read through the read-only path (__ldg) and stays in L1/L2 (40 KB);
//   * fixed reduction order: the same bits every run, no atomics.
//
// Comments that start with `mem:` say where the data lives. See memory-hierarchy.md.
#include <cuda_runtime.h>

#define FULL 0xffffffffu
#define BLOCK_THREADS 256
#define ROWS_PER_BLOCK 8

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL, v, o);
    return v;
}

// One block per row.
__global__ void __launch_bounds__(BLOCK_THREADS)
gemv_block_per_row(const float* __restrict__ A, const float* __restrict__ x, float* __restrict__ y, int N) {
    __shared__ float part[BLOCK_THREADS / 32];
    const float* a = A + (size_t)blockIdx.x * N;
    float s = 0.0f;
    if ((N & 3) == 0) {
        const float4* a4 = reinterpret_cast<const float4*>(a);
        const float4* x4 = reinterpret_cast<const float4*>(x);
        for (int i = threadIdx.x; i < N / 4; i += BLOCK_THREADS) {
            float4 av = a4[i];                           // mem: DRAM, 16 B per thread, coalesced
            float4 xv = __ldg(x4 + i);                   // mem: L1/L2 (x is 40 KB, reused by all rows)
            s = fmaf(av.x, xv.x, s);
            s = fmaf(av.y, xv.y, s);
            s = fmaf(av.z, xv.z, s);
            s = fmaf(av.w, xv.w, s);
        }
    } else {
        for (int i = threadIdx.x; i < N; i += BLOCK_THREADS) s = fmaf(a[i], __ldg(x + i), s);
    }
    s = warp_sum(s);                                     // mem: registers, via shuffles
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x < 32) {
        s = threadIdx.x < BLOCK_THREADS / 32 ? part[threadIdx.x] : 0.0f;
        s = warp_sum(s);
        if (threadIdx.x == 0) y[blockIdx.x] = s;         // mem: registers -> DRAM
    }
}

// One warp per row, for short rows.
__global__ void __launch_bounds__(32 * ROWS_PER_BLOCK)
gemv_warp_per_row(const float* __restrict__ A, const float* __restrict__ x, float* __restrict__ y, int M, int N) {
    int row = blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    if (row >= M) return;                                // the whole warp leaves together
    const float* a = A + (size_t)row * N;
    float s = 0.0f;
    for (int i = lane; i < N; i += 32) s = fmaf(a[i], __ldg(x + i), s);
    s = warp_sum(s);
    if (lane == 0) y[row] = s;
}

// A, x, y are device pointers
extern "C" void solve(const float* A, const float* x, float* y, int M, int N, int nnz) {
    (void)nnz;                                           // the dense kernel does not need it
    if (N > 2048)
        gemv_block_per_row<<<M, BLOCK_THREADS>>>(A, x, y, N);
    else
        gemv_warp_per_row<<<(M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, 32 * ROWS_PER_BLOCK>>>(A, x, y, M, N);
    cudaDeviceSynchronize();
}
