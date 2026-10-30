# Cheat sheet

## Indexing

```cpp
int i  = blockIdx.x * blockDim.x + threadIdx.x;                 // 1D
int x  = blockIdx.x * blockDim.x + threadIdx.x;                 // 2D/3D
int y  = blockIdx.y * blockDim.y + threadIdx.y;
int z  = blockIdx.z * blockDim.z + threadIdx.z;
size_t idx = ((size_t)z * NY + y) * NX + x;                     // x fastest
int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
int blocks = (n + threads - 1) / threads;
```

## Hardware limits (sm_70 to sm_120)

| Limit | Value |
|---|---|
| Warp size | 32 |
| Max threads / block | 1024 |
| Max block dims | 1024 × 1024 × 64 |
| Max grid dims | (2³¹−1) × 65535 × 65535 |
| Registers / SM | 65,536 (32-bit) |
| Max registers / thread | 255 |
| Shared memory / block | 48 KB default; opt-in up to ~99 KB (sm_89/sm_120), 163 KB (sm_80), 227 KB (sm_90) |
| Constant memory | 64 KB |

Query them at runtime with `cudaGetDeviceProperties` or `deviceQuery`.

## Core API

| Task | API |
|---|---|
| Allocate / free device | `cudaMalloc`, `cudaFree`, `cudaMallocAsync` |
| Pinned host | `cudaMallocHost`, `cudaHostRegister` |
| Unified | `cudaMallocManaged`, `cudaMemPrefetchAsync` |
| Copy | `cudaMemcpy[Async]`, `cudaMemcpy2D/3D`, `cudaMemset[Async]` |
| Streams | `cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)` |
| Events | `cudaEventRecord`, `cudaEventElapsedTime`, `cudaStreamWaitEvent` |
| Sync | `cudaDeviceSynchronize`, `cudaStreamSynchronize` |
| Errors | `cudaGetLastError`, `cudaGetErrorString` |
| Occupancy | `cudaOccupancyMaxPotentialBlockSize`, `cudaOccupancyMaxActiveBlocksPerMultiprocessor` |

## Device intrinsics

| Purpose | Intrinsic |
|---|---|
| Block barrier | `__syncthreads()` |
| Warp barrier | `__syncwarp(mask)` |
| Warp shuffle | `__shfl_sync`, `__shfl_down_sync`, `__shfl_xor_sync` |
| Warp vote | `__ballot_sync`, `__any_sync`, `__all_sync` |
| Atomics | `atomicAdd`, `atomicCAS`, `atomicMax`, `atomicExch` |
| Read-only load | `__ldg(ptr)` |
| Memory fence | `__threadfence()`, `__threadfence_block()` |

## nvcc flags

| Flag | Effect |
|---|---|
| `-O3` | host optimization (device code is optimized by default) |
| `-arch=sm_90` / `-gencode ...` | target architecture(s) |
| `-lineinfo` | source lines for profilers (no slowdown) |
| `-G` | device debug (slow; disables optimization) |
| `-Xptxas -v` | print registers, shared memory, spills |
| `--use_fast_math` | fast approximate math, flush denormals |
| `--fmad=false` | disable FMA contraction (helps CPU parity) |
| `-maxrregcount=N` | cap registers per thread |
| `--default-stream per-thread` | per-thread default stream |

## Formulas

| Quantity | Formula |
|---|---|
| Effective bandwidth | (bytes read + bytes written) / time |
| Arithmetic intensity | FLOPs / DRAM bytes |
| Roofline | min(peak FLOPS, I × peak BW) |
| LBM bytes per cell-update | Q × sizeof(real) × 2 (read + write) |
| MLUPS upper bound | BW / bytes per cell-update / 1e6 |
| Statevector memory | 2^W × 16 B (complex128), 2^W × 8 B (complex64) |
| Strong-scaling efficiency | T₁ / (p · T_p) |

## Environment variables

| Variable | Use |
|---|---|
| `CUDA_VISIBLE_DEVICES=0,2` | restrict/reorder visible GPUs |
| `CUDA_LAUNCH_BLOCKING=1` | make launches synchronous (debugging) |
| `NCCL_DEBUG=INFO` | show NCCL transport and topology |
| `NCCL_P2P_DISABLE=1` | force off P2P (diagnosing topology issues) |
| `CUPY_ACCELERATORS=cub,cutensor` | enable faster CuPy backends |

## Useful commands

```bash
nvidia-smi                      # status
nvidia-smi topo -m              # interconnect topology
nvidia-smi dmon -s pucvmet      # live per-second metrics
nvidia-smi -q -d CLOCK,POWER    # clocks and power details
nvcc --version                  # toolkit version
nsys profile -t cuda,nvtx ./app
ncu --set full -k <kernel> ./app
compute-sanitizer ./app
```
