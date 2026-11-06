// LeetGPU 6 - Softmax attention: out = softmax(Q K^T / sqrt(d)) V
// (float32; Q: M x d, K, V: N x d; d <= 128; perf M = 512, N = 256, d = 128; tol 1e-4).
//
// FlashAttention's idea at its simplest: one block per query row streams K and V in
// tiles of TJ keys and never writes the M x N score matrix. The softmax is computed
// *online*: keep a running max m and sum l, and when a tile raises the max, rescale
// what was accumulated by exp(m_old - m_new).
//
// Per tile:
//   1. scores: warp w computes keys w, w + 4, ...; its 32 lanes split the d-long dot
//      product and combine it with shuffles. Scores go to shared memory.
//   2. every thread reads the TJ scores, updates (m, l) identically, and accumulates
//      p_j * V[j, c] for the output column c it owns (thread c of 128: d <= 128).
// K and V rows are read once per query row; reads of V[j, :] by the block are coalesced.
//
// A production kernel processes a block of query rows per CTA and uses tensor cores for
// both products (FlashAttention 2/3); this is the readable version of the same algorithm.
#include <cuda_runtime.h>
#include <math.h>

#define THREADS 128                                      // one thread per output column (d <= 128)
#define TJ 64                                            // keys per tile

__global__ void attention_kernel(const float* __restrict__ Q, const float* __restrict__ K,
                                 const float* __restrict__ V, float* __restrict__ out,
                                 int N, int d, float scale) {
    __shared__ float q[THREADS];                         // mem: shared, this query row
    __shared__ float s[TJ];                              // mem: shared, one tile of scores
    const int row = blockIdx.x, c = threadIdx.x, lane = c & 31, warp = c >> 5;
    if (c < d) q[c] = Q[(size_t)row * d + c];
    __syncthreads();

    float m = -INFINITY, l = 0.0f, acc = 0.0f;           // mem: registers
    for (int j0 = 0; j0 < N; j0 += TJ) {
        int nj = min(TJ, N - j0);
        for (int jj = warp; jj < nj; jj += THREADS / 32) {
            const float* kr = K + (size_t)(j0 + jj) * d;
            float dot = 0.0f;
            for (int k = lane; k < d; k += 32) dot = fmaf(q[k], kr[k], dot);   // mem: K row, coalesced
            for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
            if (lane == 0) s[jj] = dot * scale;
        }
        __syncthreads();
        float m_new = m;
        for (int jj = 0; jj < nj; ++jj) m_new = fmaxf(m_new, s[jj]);
        float corr = __expf(m - m_new);                  // 0 on the first tile (m = -inf)
        l *= corr;
        acc *= corr;
        for (int jj = 0; jj < nj; ++jj) {
            float p = __expf(s[jj] - m_new);
            l += p;
            if (c < d) acc = fmaf(p, V[(size_t)(j0 + jj) * d + c], acc);       // mem: V row, coalesced
        }
        m = m_new;
        __syncthreads();                                 // before the next tile overwrites s
    }
    if (c < d) out[(size_t)row * d + c] = acc / l;
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    attention_kernel<<<M, THREADS>>>(Q, K, V, output, N, d, 1.0f / sqrtf((float)d));
    cudaDeviceSynchronize();
}
