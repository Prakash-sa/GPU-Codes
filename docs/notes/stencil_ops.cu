// Discrete gradient, divergence and Laplacian as CUDA stencils: the CUDA C++ companion
// to stencil_ops.py, with the same conventions.
//
// Fields carry one layer of ghost cells: a 2-D field of nx x ny points is stored as a
// (ny+2) x (nx+2) array, row-major, x fastest; a 3-D field as (nz+2) x (ny+2) x (nx+2).
// The boundary condition fills the ghost layer beforehand, so the kernels have no
// branches for boundaries. Second-order central differences on a collocated grid.
//
//   grad2d: one thread per point, a 32 x 8 block; neighbours come through L1/L2.
//   div2d:  same, on a vector field stored as two arrays (SoA, coalesced).
//   lap3d:  2.5-D blocking. A 32 x 8 block owns a 32 x 8 column of the x-y plane and
//           marches up z. The centre plane (with its x-y halo) sits in shared memory for
//           the in-plane neighbours; the planes below and above sit in registers. Each
//           plane is read from DRAM once.
#include <cuda_runtime.h>

#define BX 32
#define BY 8

__global__ void grad2d_kernel(const float* __restrict__ u, float* __restrict__ gx, float* __restrict__ gy,
                              int nx, int ny, float inv2dx, float inv2dy) {
    int i = 1 + blockIdx.x * BX + threadIdx.x;           // interior index, ghost layer at 0
    int j = 1 + blockIdx.y * BY + threadIdx.y;
    if (i > nx || j > ny) return;
    int w = nx + 2;
    size_t o = (size_t)j * w + i;
    // mem: four neighbours; E/W share the centre row's cache lines, N/S were loaded by
    // the neighbouring rows of the block: ~1 DRAM read per point
    gx[o] = (u[o + 1] - u[o - 1]) * inv2dx;
    gy[o] = (u[o + w] - u[o - w]) * inv2dy;
}

__global__ void div2d_kernel(const float* __restrict__ vx, const float* __restrict__ vy,
                             float* __restrict__ out, int nx, int ny, float inv2dx, float inv2dy) {
    int i = 1 + blockIdx.x * BX + threadIdx.x;
    int j = 1 + blockIdx.y * BY + threadIdx.y;
    if (i > nx || j > ny) return;
    int w = nx + 2;
    size_t o = (size_t)j * w + i;
    out[o] = (vx[o + 1] - vx[o - 1]) * inv2dx + (vy[o + w] - vy[o - w]) * inv2dy;
}

__global__ void lap3d_kernel(const float* __restrict__ u, float* __restrict__ out,
                             int nx, int ny, int nz, float cx, float cy, float cz) {
    __shared__ float plane[BY + 2][BX + 2];              // mem: shared, centre plane + halo
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int i = 1 + blockIdx.x * BX + tx, j = 1 + blockIdx.y * BY + ty;
    const bool inside = i <= nx && j <= ny;
    const int w = nx + 2;
    const size_t ps = (size_t)w * (ny + 2);              // plane stride
    const size_t xy = (size_t)j * w + i;

    float below = inside ? u[xy] : 0.0f;                 // mem: registers, k = 0 (ghost plane)
    float centre = inside ? u[ps + xy] : 0.0f;           // k = 1
    for (int k = 1; k <= nz; ++k) {
        float above = inside ? u[(size_t)(k + 1) * ps + xy] : 0.0f;   // mem: the only new plane
        // Stage plane k with its x-y halo: interior from registers, halo from L2.
        if (inside) plane[ty + 1][tx + 1] = centre;
        const size_t pk = (size_t)k * ps;
        if (tx == 0 && j <= ny) plane[ty + 1][0] = u[pk + (size_t)j * w + i - 1];
        if ((tx == BX - 1 || i == nx) && j <= ny && i <= nx) plane[ty + 1][tx + 2] = u[pk + (size_t)j * w + i + 1];
        if (ty == 0 && i <= nx) plane[0][tx + 1] = u[pk + (size_t)(j - 1) * w + i];
        if ((ty == BY - 1 || j == ny) && i <= nx && j <= ny) plane[ty + 2][tx + 1] = u[pk + (size_t)(j + 1) * w + i];
        __syncthreads();
        if (inside) {
            float c = plane[ty + 1][tx + 1];
            float lx = plane[ty + 1][tx] + plane[ty + 1][tx + 2] - 2.0f * c;
            float ly = plane[ty][tx + 1] + plane[ty + 2][tx + 1] - 2.0f * c;
            float lz = above + below - 2.0f * c;
            out[pk + xy] = lx * cx + ly * cy + lz * cz;  // mem: write once
        }
        __syncthreads();                                 // before plane k+1 overwrites shared
        below = centre;                                  // shift the register queue
        centre = above;
    }
}

// All pointers are device pointers to ghosted arrays (see the header comment).
extern "C" void grad2d(const float* u, float* gx, float* gy, int nx, int ny, float dx, float dy) {
    dim3 grid((nx + BX - 1) / BX, (ny + BY - 1) / BY);
    grad2d_kernel<<<grid, dim3(BX, BY)>>>(u, gx, gy, nx, ny, 0.5f / dx, 0.5f / dy);
    cudaDeviceSynchronize();
}

extern "C" void div2d(const float* vx, const float* vy, float* out, int nx, int ny, float dx, float dy) {
    dim3 grid((nx + BX - 1) / BX, (ny + BY - 1) / BY);
    div2d_kernel<<<grid, dim3(BX, BY)>>>(vx, vy, out, nx, ny, 0.5f / dx, 0.5f / dy);
    cudaDeviceSynchronize();
}

extern "C" void lap3d(const float* u, float* out, int nx, int ny, int nz, float dx, float dy, float dz) {
    dim3 grid((nx + BX - 1) / BX, (ny + BY - 1) / BY);
    lap3d_kernel<<<grid, dim3(BX, BY)>>>(u, out, nx, ny, nz, 1.0f / (dx * dx), 1.0f / (dy * dy), 1.0f / (dz * dz));
    cudaDeviceSynchronize();
}
