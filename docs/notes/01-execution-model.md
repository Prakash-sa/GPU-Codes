# 1. Execution model

## Host and device

- **Host** = CPU + system RAM. **Device** = GPU + its own memory (HBM/GDDR).
- The host launches **kernels** (functions marked `__global__`) on the device. Launches
  are **asynchronous**: control returns to the CPU immediately.
- Data must be on the device before the kernel reads it, either through explicit
  `cudaMemcpy` or through unified memory.

```cpp
__global__ void saxpy(int n, float a, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // global thread index
    if (i < n) y[i] = a * x[i] + y[i];               // guard: grid may overshoot n
}

int threads = 256;
int blocks  = (n + threads - 1) / threads;           // ceil-divide
saxpy<<<blocks, threads>>>(n, 2.0f, d_x, d_y);
cudaDeviceSynchronize();                              // wait + surface errors
```

## The thread hierarchy

| Level | Built-in | Runs on | Can cooperate via |
|---|---|---|---|
| Thread | `threadIdx` | one CUDA core lane | registers |
| Warp (32 threads) | `threadIdx.x / 32` | one SM scheduler, in lockstep | warp shuffles, `__syncwarp()` |
| Block (≤ 1024 threads) | `blockIdx`, `blockDim` | one SM, for the block's whole life | shared memory, `__syncthreads()` |
| Cluster (sm_90+) | `cluster.block_rank()` | a group of SMs (GPC) | distributed shared memory |
| Grid | `gridDim` | the whole GPU | global memory, atomics, separate launches |

- Blocks are **independent**. The hardware may run them in any order, so never assume
  block A finishes before block B starts within one launch.
- To sync the whole grid, end the kernel and launch another one (or use cooperative
  groups' `grid.sync()` with a cooperative launch).

## SMs, warps and SIMT

- A GPU is a set of **Streaming Multiprocessors (SMs)**. Each SM has its own register
  file, shared memory/L1, warp schedulers and functional units.
- The SM issues instructions **per warp**. All 32 threads execute the same instruction
  (SIMT, Single Instruction Multiple Threads).
- **Latency hiding**: when a warp stalls on memory (hundreds of cycles), the scheduler
  switches to another ready warp with zero cost. GPUs therefore need *many* resident
  warps. That is what occupancy measures.

## Warp divergence

```cpp
if (threadIdx.x % 2 == 0) a(); else b();   // both paths serialize inside each warp
if (threadIdx.x / 32 == 0) a(); else b();  // warp-uniform: no divergence
```

- When threads in one warp take different branches, the warp runs **both** paths with
  lanes masked off. Make branches uniform per warp where you can.
- Since Volta (sm_70), threads have independent scheduling. Don't rely on implicit
  warp-synchronous behavior; use `__syncwarp()` and the `*_sync` intrinsics.

## Choosing launch configuration

- Block size: a **multiple of 32**, typically 128–256. Use 2D/3D blocks for stencils
  (e.g. `dim3(32, 8)`), keeping `x` the fastest-varying, contiguous dimension.
- Enough blocks to fill every SM several times (≥ 2–4 × SM count).
- `cudaOccupancyMaxPotentialBlockSize` gives a reasonable starting point.
- **Grid-stride loops** decouple grid size from problem size:

```cpp
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    y[i] = a * x[i] + y[i];
```

## Compute capability

The `sm_XY` number sets which features and instructions are available. Compile for
the cards you target, and include PTX so newer cards can JIT-compile it:

```bash
nvcc -O3 -gencode arch=compute_80,code=sm_80 \
         -gencode arch=compute_90,code=sm_90 \
         -gencode arch=compute_90,code=compute_90 kernel.cu
```

Examples: T4 = sm_75, A100 = sm_80, L40S = sm_89, H100 = sm_90, RTX PRO 6000
Blackwell = sm_120 (needs CUDA ≥ 12.8).

## Error checking

Kernel launches do not return errors directly. Always check:

```cpp
#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) {        \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e));  \
    exit(1); } } while (0)

kernel<<<g, b>>>(...);
CUDA_CHECK(cudaGetLastError());        // launch-configuration errors
CUDA_CHECK(cudaDeviceSynchronize());   // errors raised during execution (debug builds)
```
