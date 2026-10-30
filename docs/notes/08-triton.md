# 8. Programming with Triton

[Triton](https://triton-lang.org) is a Python DSL and compiler for writing GPU kernels.
You write the kernel for a **block of data**, not for one thread. The compiler handles
the details CUDA makes you manage yourself: coalescing, vectorized loads, shared
memory staging, and tensor-core instructions.

## Mental model: CUDA vs Triton

| | CUDA | Triton |
|---|---|---|
| Unit you program | one **thread** | one **program** (≈ a thread block) working on a tile |
| Index | `blockIdx.x * blockDim.x + threadIdx.x` (a scalar) | `pid * BLOCK + tl.arange(0, BLOCK)` (a vector) |
| Bounds check | `if (i < n)` | `mask = offs < n`, passed to `tl.load` / `tl.store` |
| Shared memory | explicit `__shared__` + `__syncthreads()` | implicit, chosen by the compiler |
| Coalescing / vectorization | your job | the compiler's job (given contiguous offsets) |
| Tensor cores | WMMA / `mma.sync` / CUTLASS | `tl.dot(a, b)` |
| Tuning knobs | block size, registers | `BLOCK` constexprs, `num_warps`, `num_stages` |

What you still control: **which tile each program handles**, the memory access
pattern, the fusion of operations, and the block sizes.

## Setup

```bash
pip install triton          # Linux + NVIDIA (sm_75+) or AMD ROCm; bundled with PyTorch
```

Triton kernels take **PyTorch tensors** (anything with `.data_ptr()` and a dtype). For
CuPy arrays, wrap them without a copy: `torch.from_dlpack(cupy_array)`.

## First kernel: vector add

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)               # which block am I?
    offs = pid * BLOCK + tl.arange(0, BLOCK)  # vector of BLOCK indices
    mask = offs < n                           # guard the tail
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = out.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)   # number of programs
    add_kernel[grid](x, y, out, n, BLOCK=1024)
    return out

x = torch.rand(10_000_000, device="cuda")
y = torch.rand_like(x)
torch.testing.assert_close(add(x, y), x + y)
```

Things to notice:

- Tensor arguments arrive as **pointers**; `x_ptr + offs` is a block of pointers.
- `BLOCK: tl.constexpr` is a compile-time constant. Each distinct value compiles a new
  kernel. `tl.arange` sizes must be **powers of two**.
- `grid` can be a tuple or a function of the meta-parameters. The function form is
  what makes autotuning work.

## Core language

| Need | Triton |
|---|---|
| Program id / count | `tl.program_id(axis)`, `tl.num_programs(axis)` |
| Index vectors | `tl.arange(0, N)`; 2D via broadcasting: `rows[:, None]`, `cols[None, :]` |
| Load / store | `tl.load(ptrs, mask=m, other=0.0)`, `tl.store(ptrs, val, mask=m)` |
| Allocate a tile | `tl.zeros((BM, BN), dtype=tl.float32)`, `tl.full(shape, value, dtype)` |
| Elementwise math | `+ - * /`, `tl.exp`, `tl.log`, `tl.sqrt`, `tl.abs`, `tl.maximum`, `tl.where(c, a, b)` |
| Reductions | `tl.sum(x, axis)`, `tl.max(x, axis)`, `tl.min(x, axis)`, `tl.argmax(x, axis)` |
| Matrix multiply | `tl.dot(a, b)` (tiles ≥ 16 in each dimension) |
| Casts | `x.to(tl.float32)`, `pid.to(tl.int64)` |
| Atomics | `tl.atomic_add(ptrs, val, mask=m)`, `tl.atomic_max`, `tl.atomic_cas` |
| Compile-time loops | `for i in tl.static_range(4):` (unrolled) |
| Helpers | `triton.cdiv(a, b)`, `triton.next_power_of_2(n)` |

`other=` sets the value loaded where `mask` is false. Pick it so it doesn't affect the
result: `0` for sums, `-inf` for max.

## Fused row-wise softmax

Softmax in plain PyTorch takes several kernels (max, subtract, exp, sum, divide), each
of which reads and writes the whole matrix. In Triton one program handles one row, and
the row is read once and written once:

```python
@triton.jit
def softmax_kernel(out_ptr, in_ptr, in_stride, out_stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(in_ptr + row * in_stride + cols, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)          # numerical stability
    num = tl.exp(x)
    y = num / tl.sum(num, axis=0)
    tl.store(out_ptr + row * out_stride + cols, y, mask=mask)

def softmax(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(n_cols)        # whole row in one tile
    num_warps = 4 if BLOCK <= 2048 else 8 if BLOCK <= 4096 else 16
    softmax_kernel[(n_rows,)](y, x, x.stride(0), y.stride(0), n_cols,
                              BLOCK=BLOCK, num_warps=num_warps)
    return y
```

This is the main reason to use Triton: **fusion turns several memory-bound passes
into one**.

## 2D stencil (Jacobi / 5-point Laplacian)

This is the pattern behind most structured-grid solvers. Each program updates a
`BY × BX` tile and reads its neighbors straight from global memory, where the L1 and
L2 caches absorb the overlap between neighboring loads.

```python
@triton.jit
def jacobi_kernel(u_ptr, out_ptr, nx, ny, BX: tl.constexpr, BY: tl.constexpr):
    x = tl.program_id(0) * BX + tl.arange(0, BX)
    y = tl.program_id(1) * BY + tl.arange(0, BY)
    xx = x[None, :]                     # (1, BX)
    yy = y[:, None]                     # (BY, 1)
    interior = (xx > 0) & (xx < nx - 1) & (yy > 0) & (yy < ny - 1)
    idx = yy * nx + xx                  # row-major, x contiguous
    n = tl.load(u_ptr + idx - nx, mask=interior)
    s = tl.load(u_ptr + idx + nx, mask=interior)
    w = tl.load(u_ptr + idx - 1, mask=interior)
    e = tl.load(u_ptr + idx + 1, mask=interior)
    tl.store(out_ptr + idx, 0.25 * (n + s + w + e), mask=interior)

def jacobi(u: torch.Tensor, steps: int) -> torch.Tensor:
    ny, nx = u.shape
    a, b = u.clone(), u.clone()         # b keeps the boundary values
    grid = (triton.cdiv(nx, 64), triton.cdiv(ny, 16))
    for _ in range(steps):
        jacobi_kernel[grid](a, b, nx, ny, BX=64, BY=16)
        a, b = b, a                     # ping-pong buffers
    return a
```

- Map the **contiguous dimension** (`x`) to the tile's last axis so loads coalesce.
- For 3D grids (e.g. LBM), use a 2D tile in `(y, x)` and loop over `z`, or launch a 3D
  grid with `tl.program_id(2)`. Store populations as SoA, `f[q, z, y, x]`.
- Count bytes: here it is 5 loads + 1 store per cell before caching. Benchmark against
  peak bandwidth (see below).

## Matrix multiply with autotuning

```python
@triton.autotune(
    configs=[
        triton.Config({"BM": 128, "BN": 128, "BK": 32}, num_warps=8, num_stages=3),
        triton.Config({"BM": 64,  "BN": 128, "BK": 32}, num_warps=4, num_stages=4),
        triton.Config({"BM": 64,  "BN": 64,  "BK": 64}, num_warps=4, num_stages=4),
    ],
    key=["M", "N", "K"],          # re-tune when these change
)
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a_ptrs = a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
    b_ptrs = b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn

    acc = tl.zeros((BM, BN), dtype=tl.float32)    # accumulate in fp32
    for k in range(0, tl.cdiv(K, BK)):
        k_left = K - k * BK
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] < k_left), other=0.0)
        b = tl.load(b_ptrs, mask=(rk[:, None] < k_left) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)                       # tensor cores
        a_ptrs += BK * stride_ak
        b_ptrs += BK * stride_bk

    c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty),
             mask=(rm[:, None] < M) & (rn[None, :] < N))

def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BM"]), triton.cdiv(N, meta["BN"]))
    matmul_kernel[grid](a, b, c, M, N, K,
                        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                        c.stride(0), c.stride(1))
    return c
```

- Pass **strides**, not shapes alone, so the kernel works on transposed and sliced
  views.
- `num_warps` = warps per program (4–8 typical). `num_stages` = software-pipelining
  depth for the K loop (more stages hide more latency but use more shared memory).
- The first call with a new `key` benchmarks every config. Later calls reuse the
  winner. Compiled kernels are cached in `~/.triton/cache` (override with
  `TRITON_CACHE_DIR`).
- The production version of this kernel (the official tutorial) also reorders
  program ids in "grouped" order for better L2 reuse.

## Benchmarking

```python
from triton.testing import do_bench

ms = do_bench(lambda: add(x, y))                        # median time, handles warm-up
gbps = 3 * x.numel() * x.element_size() / (ms * 1e-3) / 1e9
print(f"{ms:.3f} ms, {gbps:.0f} GB/s")                  # compare with peak bandwidth

ms = do_bench(lambda: matmul(a, b))
tflops = 2 * M * N * K / (ms * 1e-3) / 1e12
```

For a streaming kernel, 70–85% of the card's peak bandwidth is a good result (see
the [spec sheets](../gpu-spec-sheets.md) and [performance notes](03-performance.md)).
`triton.testing.perf_report` sweeps sizes and plots the results.

## Debugging

| Tool | How |
|---|---|
| CPU interpreter | `TRITON_INTERPRET=1 python script.py`: runs the kernel in NumPy, so `print` and `pdb` work |
| Device printing | `tl.device_print("x", x)`; `tl.static_print(BLOCK)` at compile time |
| Assertions | `tl.device_assert(x >= 0)` (enabled when `TRITON_DEBUG=1`), `tl.static_assert(BLOCK % 16 == 0)` |
| Memory errors | `compute-sanitizer python script.py` |
| Generated code | `k = add_kernel[grid](...)`, then `k.asm["ttir"]`, `k.asm["ttgir"]`, `k.asm["ptx"]` |
| Profiling | Nsight Systems / Nsight Compute work as for CUDA ([profiling notes](06-profiling.md)) |

## Pitfalls

!!! warning "32-bit index overflow"
    `pid * BLOCK + tl.arange(...)` is **int32**. For tensors with more than
    $2^{31}$ elements (e.g. a 512³ × 27 LBM lattice), the offsets wrap around and
    the kernel reads garbage. Promote early: `pid = tl.program_id(0).to(tl.int64)`.

- **Recompilation:** every new `constexpr` value compiles a new kernel. Integer
  arguments are also specialized (equal to 1, divisible by 16), so varying sizes can
  trigger a few compiles. Keep block sizes to a small fixed set.
- **Power-of-two shapes:** `tl.arange` and tile shapes must be powers of two. Handle
  the remainder with masks.
- **Non-contiguous inputs:** pass strides, or call `.contiguous()` before launching.
- **dtype:** `tl.load` returns the pointer's element type. Accumulate reductions and
  dot products in `tl.float32` (or `float64`) and cast at the store.
- **FP64:** elementwise and reduction kernels work in `float64`. `tl.dot` support for
  fp64 depends on the Triton version and the GPU, so check before relying on it. On
  cards with low FP64 throughput (T4, L40S, RTX PRO 6000) FP64 kernels can become
  compute bound.
- **Launch overhead:** each launch costs several µs of Python plus driver time. For
  many small launches per step, capture them in a CUDA Graph
  (`torch.cuda.CUDAGraph`).

## When to use Triton

| Situation | Choose |
|---|---|
| Fusing elementwise/reduction chains on tensors | **Triton** |
| Custom tiled kernels (attention, stencils, matmul variants) in Python | **Triton** |
| Standard ops (GEMM, FFT, sort) | cuBLAS / cuFFT / CUB via CuPy or PyTorch |
| Warp-level tricks, explicit shared-memory layouts, cooperative groups | CUDA C++ |
| Porting existing NumPy code quickly | CuPy |

## Further reading

- [Official tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html):
  vector add, fused softmax, matmul, layer norm, fused attention.
- [`triton.language` API reference](https://triton-lang.org/main/python-api/triton.language.html)
