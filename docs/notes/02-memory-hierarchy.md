# 2. Memory hierarchy

## The levels

| Memory | Scope | Lifetime | Approx. latency | Notes |
|---|---|---|---|---|
| Registers | thread | thread | ~1 cycle | 64K × 32-bit per SM; overflow spills to "local" memory |
| Shared memory | block | block | ~20–30 cycles | Shares on-chip SRAM with L1 (up to ~228 KB/SM on H100) |
| L1 cache | SM | — | ~30 cycles | Caches global/local loads |
| L2 cache | whole GPU | — | ~200 cycles | 4 MB (T4) to 128 MB (RTX PRO 6000) |
| Global (HBM/GDDR) | grid + host | allocation | ~400–800 cycles | Where `cudaMalloc` lives; bandwidth is the key spec |
| Constant | grid | program | cached | 64 KB; fast when all threads in a warp read the same address |
| Local | thread | thread | same as global | Register spills and dynamically indexed arrays |

## Global memory and coalescing

A warp's loads are combined into as few **32-byte sectors** as possible. If thread
`k` reads element `base + k`, the warp touches contiguous memory and gets full
bandwidth.

```cpp
// coalesced: consecutive threads -> consecutive addresses
float v = a[blockIdx.x * blockDim.x + threadIdx.x];

// strided: each thread touches a different sector -> up to 32x the traffic
float w = a[(blockIdx.x * blockDim.x + threadIdx.x) * stride];
```

- Prefer **Structure of Arrays (SoA)** over Array of Structures (AoS). For LBM, store
  `f[q][z][y][x]` with `x` innermost, not `f[z][y][x][q]`.
- Align allocations (`cudaMalloc` gives ≥ 256-byte alignment). Pad row pitch with
  `cudaMallocPitch` / `cudaMalloc3D` for 2D/3D arrays.
- Use vector loads (`float4`, `double2`) when the data is aligned, to issue fewer
  instructions.

## Shared memory

Shared memory is a programmer-managed cache, used for data reuse (tiling) and for
reorganizing access patterns (for example, a transpose).

```cpp
__global__ void transpose(const float* in, float* out, int n) {
    __shared__ float tile[32][33];               // +1 column avoids bank conflicts
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    tile[threadIdx.y][threadIdx.x] = in[y * n + x];          // coalesced read
    __syncthreads();
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    out[y * n + x] = tile[threadIdx.x][threadIdx.y];         // coalesced write
}
```

- 32 **banks**, each 4 bytes wide. When several threads in a warp hit different
  addresses in the same bank, the accesses serialize (a **bank conflict**). All
  threads reading the *same* address is a broadcast, which is free.
- Dynamic size: `extern __shared__ float buf[];` plus the third launch parameter
  `<<<g, b, bytes>>>`. To use more than 48 KB you must opt in with
  `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)`.
- `__syncthreads()` must be reached by **every** thread of the block. Never put it in a
  divergent branch.

## Registers and spills

- Registers per thread × threads per block limits how many blocks fit on an SM.
- Check usage with `nvcc -Xptxas -v` (it prints registers and spill stores/loads).
- Cap registers with `__launch_bounds__(maxThreads, minBlocksPerSM)` or
  `-maxrregcount`. This trades spills for occupancy, so measure the result.

## Read-only and constant data

- `const __restrict__` pointers let the compiler use the read-only/texture path
  (`__ldg`).
- `__constant__` is for small parameter tables (e.g. LBM lattice weights and
  velocities) that every thread reads at the same index.

## Host ↔ device

| Mechanism | API | When to use |
|---|---|---|
| Pageable copy | `cudaMemcpy` | Simple, but synchronous and slower |
| Pinned (page-locked) host memory | `cudaMallocHost` / `cudaHostRegister` | Required for truly async copies; higher PCIe bandwidth |
| Unified memory | `cudaMallocManaged` | Prototyping, oversubscription; add `cudaMemPrefetchAsync` for performance |
| Zero-copy | mapped pinned memory | Data read once, small, or on integrated GPUs |

!!! warning "PCIe is the bottleneck"
    PCIe 4.0 x16 moves ~25 GB/s in practice, versus 1–3 TB/s for on-card memory.
    Keep data **device-resident** across time steps and copy back only what you need
    (diagnostics, checkpoints).

## Memory capacity planning

Work out the footprint before you allocate:

- LBM D3Q27, double, two buffers (A/B):
  \(N_x N_y N_z \times 27 \times 8 \times 2\) bytes. A 256³ grid is ≈ 7.2 GB.
- Statevector of \(W\) qubits, complex double: \(2^W \times 16\) bytes.
  30 qubits = 16 GiB, 32 qubits = 64 GiB.

Leave headroom for library workspaces (cuBLAS, cuFFT, NCCL buffers) and the CUDA
context (~300–500 MB).
