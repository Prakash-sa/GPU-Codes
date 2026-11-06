// LeetGPU 2 - Matrix Multiplication: C = A @ B, float32, row-major
// (A is M x N, B is N x K, C is M x K: the inner dimension is N; perf M = 8192, N = 6144,
// K = 4096; atol = rtol = 1e-4).
//
// The classic register-blocked SGEMM on the CUDA cores (exact FP32, so the 1e-4 tolerance
// is never in question):
//   * a block of 256 threads owns a 128 x 128 tile of C and walks N in steps of BK = 8;
//   * each thread owns an 8 x 8 sub-tile: 64 accumulators in registers, so every value it
//     reads from shared memory feeds 8 FMAs (arithmetic intensity 2 * 8 * 8 / 16 floats);
//   * the 8 rows and 8 columns of a thread are split 4 + 4 (rows ty*4.. and 64 + ty*4..),
//     so a warp's float4 reads from shared memory hit distinct banks or broadcast;
//   * global loads are float4 when the row length allows it (N % 4 == 0, K % 4 == 0),
//     scalar and bounds-checked otherwise, so any M, N, K from 1 to 8192 works;
//   * accuracy: one FMA chain 6,144 long drifts past the 1e-4 tolerance at the largest
//     graded size (measured on the page). Each run of CHUNK = 64 products goes into a
//     fresh partial sum `part`, which is then added to `acc`: two short chains instead
//     of one long one, at the cost of 64 more registers.
// What it leaves out (the next steps on the page): double-buffered shared memory,
// warp tiling, and tensor cores (WMMA/mma.sync with 3xTF32, or cuBLAS/CUTLASS).
//
// Comments that start with `mem:` say where the data lives. See memory-hierarchy.md.
#include <cuda_runtime.h>

#define BM 128
#define BN 128
#define BK 8
#define THREADS 256
#define CHUNK_TILES 8                                    // 8 x BK = 64 products per partial sum

__global__ void __launch_bounds__(THREADS)
sgemm_128x128_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                     int M, int N, int K) {
    __shared__ __align__(16) float As[BK][BM];          // mem: shared, A tile stored k-major
    __shared__ __align__(16) float Bs[BK][BN];          // mem: shared, B tile
    const int tid = threadIdx.x;
    const int tx = tid % 16, ty = tid / 16;              // 16 x 16 threads, 8 x 8 outputs each
    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
    const bool vecA = (N % 4) == 0, vecB = (K % 4) == 0;

    // Load assignment: A tile 128 x 8 -> thread t loads row t / 2, k = (t % 2) * 4 .. + 3.
    //                  B tile 8 x 128 -> thread t loads k = t / 32, cols (t % 32) * 4 .. + 3.
    const int a_r = tid / 2, a_k = (tid % 2) * 4;
    const int b_k = tid / 32, b_c = (tid % 32) * 4;

    float acc[8][8] = {};                                // mem: registers, 64 per thread
    float part[8][8] = {};                               // the current chunk's partial sums
    int tiles = 0;

    for (int k0 = 0; k0 < N; k0 += BK) {
        {   // A: one row segment of 4 floats
            int r = row0 + a_r, kk = k0 + a_k;
            float v[4];
            if (vecA && r < M && kk + 3 < N) {
                float4 t = *reinterpret_cast<const float4*>(A + (size_t)r * N + kk);   // mem: DRAM, 16 B
                v[0] = t.x; v[1] = t.y; v[2] = t.z; v[3] = t.w;
            } else {
#pragma unroll
                for (int q = 0; q < 4; ++q) v[q] = (r < M && kk + q < N) ? A[(size_t)r * N + kk + q] : 0.0f;
            }
#pragma unroll
            for (int q = 0; q < 4; ++q) As[a_k + q][a_r] = v[q];   // transpose into k-major
        }
        {   // B: one row segment of 4 floats
            int kk = k0 + b_k, c = col0 + b_c;
            if (vecB && kk < N && c + 3 < K) {
                *reinterpret_cast<float4*>(&Bs[b_k][b_c]) =
                    *reinterpret_cast<const float4*>(B + (size_t)kk * K + c);       // mem: DRAM, 16 B
            } else {
#pragma unroll
                for (int q = 0; q < 4; ++q) Bs[b_k][b_c + q] = (kk < N && c + q < K) ? B[(size_t)kk * K + c + q] : 0.0f;
            }
        }
        __syncthreads();                                 // tiles complete before anyone reads

#pragma unroll
        for (int k = 0; k < BK; ++k) {
            // mem: shared -> registers, four float4 reads feed 64 FMAs
            float4 a0 = *reinterpret_cast<const float4*>(&As[k][ty * 4]);
            float4 a1 = *reinterpret_cast<const float4*>(&As[k][64 + ty * 4]);
            float4 b0 = *reinterpret_cast<const float4*>(&Bs[k][tx * 4]);
            float4 b1 = *reinterpret_cast<const float4*>(&Bs[k][64 + tx * 4]);
            float a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            float b[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int j = 0; j < 8; ++j) part[i][j] = fmaf(a[i], b[j], part[i][j]);
        }
        __syncthreads();                                 // done reading before the next load
        if (++tiles == CHUNK_TILES || k0 + BK >= N) {    // close the chunk
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    acc[i][j] += part[i][j];
                    part[i][j] = 0.0f;
                }
            tiles = 0;
        }
    }

    // mem: registers -> DRAM. Row i of the thread is ty*4+i (i < 4) or 64+ty*4+(i-4).
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        int r = row0 + (i < 4 ? ty * 4 + i : 64 + ty * 4 + (i - 4));
        if (r >= M) continue;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            int c = col0 + h * 64 + tx * 4;
            float* dst = C + (size_t)r * K + c;
            if (vecB && c + 3 < K) {
                *reinterpret_cast<float4*>(dst) =
                    make_float4(acc[i][h * 4], acc[i][h * 4 + 1], acc[i][h * 4 + 2], acc[i][h * 4 + 3]);
            } else {
#pragma unroll
                for (int q = 0; q < 4; ++q)
                    if (c + q < K) dst[q] = acc[i][h * 4 + q];
            }
        }
    }
}

// A, B, C are device pointers
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 grid((K + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_128x128_kernel<<<grid, THREADS>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
