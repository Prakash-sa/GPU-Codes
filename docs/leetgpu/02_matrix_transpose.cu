// LeetGPU 3 - Matrix Transpose: output[c, r] = input[r, c] (float32, perf 7000 x 6000).
//
// A naive transpose either reads or writes with a stride of `rows`/`cols` floats, so one
// side is uncoalesced. Here a 32 x 32 tile is read row-by-row (coalesced), parked in
// shared memory, and written row-by-row to the output (coalesced). The tile is 32 x 33:
// the extra column shifts each row by one bank, so reading a tile *column* (32 floats,
// stride 33) hits 32 different banks instead of one: no bank conflicts.
//
// Block = 32 x 8 threads; each thread moves 4 elements of the tile.
#include <cuda_runtime.h>

#define TILE 32
#define ROWS_PER_PASS 8

__global__ void transpose_kernel(const float* __restrict__ in, float* __restrict__ out,
                                 int rows, int cols) {
    __shared__ float tile[TILE][TILE + 1];               // mem: shared memory, padded
    int c = blockIdx.x * TILE + threadIdx.x;
    int r0 = blockIdx.y * TILE + threadIdx.y;
    for (int k = 0; k < TILE; k += ROWS_PER_PASS) {
        int r = r0 + k;
        if (r < rows && c < cols)
            tile[threadIdx.y + k][threadIdx.x] = in[(size_t)r * cols + c];   // mem: DRAM, coalesced
    }
    __syncthreads();
    // Output tile (blockIdx.x, blockIdx.y) swapped: output row = input column.
    int oc = blockIdx.y * TILE + threadIdx.x;
    int or0 = blockIdx.x * TILE + threadIdx.y;
    for (int k = 0; k < TILE; k += ROWS_PER_PASS) {
        int orow = or0 + k;
        if (orow < cols && oc < rows)
            out[(size_t)orow * rows + oc] = tile[threadIdx.x][threadIdx.y + k];   // mem: DRAM, coalesced
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 block(TILE, ROWS_PER_PASS);
    dim3 grid((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);
    transpose_kernel<<<grid, block>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
