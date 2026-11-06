# 21. Matrix multiplication (FP32)

| | |
|---|---|
| **Question** | *[Matrix Multiplication](https://leetgpu.com/challenges/matrix-multiplication)* (easy) on LeetGPU · [official spec and reference](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/easy/2_matrix_multiplication) |
| **Solution** | [21_matrix_multiplication.py](https://github.com/Prakash-sa/GPU-Codes/blob/main/docs/leetgpu/21_matrix_multiplication.py) (Triton) · [21_matrix_multiplication.cu](https://github.com/Prakash-sa/GPU-Codes/blob/main/docs/leetgpu/21_matrix_multiplication.cu) (CUDA C++) |
| **Task** | $C = A \times B$, float32, row-major. A is **M × N**, B is **N × K** (the inner dimension is *N*); 1 ≤ M, N, K ≤ 8192; perf 8192 × 6144 × 4096; tolerance 1e-4 |
| **Bound by** | compute (≈ 945 FLOP per byte) |

## Idea

This is the problem behind every other matmul page ([09](09-gemm.md), [10](10-fp16-batched-matmul.md),
[15](15-spmm.md), [17](17-fused-layernorm-linear-gelu.md)), in its plainest form: FP32
in, FP32 out, no epilogue. The algorithm is the standard tiled GEMM:

![One program computing one C tile](../assets/img/triton-matmul.gif)

1. Each program (block) owns one tile of C and walks the inner dimension in steps,
   multiplying an A tile by a B tile into an accumulator that never leaves registers.
2. **Grouped program order** makes programs that run at the same time share A row
   panels and B column panels, so those panels are read from L2, not DRAM.

    ![Row-major vs grouped program order](../assets/img/leetgpu/grouped-order.png)

3. **Autotuning** picks the tile shape, warps and pipeline depth for each GPU.

Watch the names: LeetGPU calls the inner dimension **N**, not K. A is `M × N`, B is
`N × K`, C is `M × K`. Both solutions keep that naming, so `solve()` matches the
signature exactly.

## Precision and summation order decide the result

The data is FP32. Tensor cores don't take FP32 inputs, so the options are:

- **TF32**: a 10-bit mantissa.
- **`tf32x3`**: each operand split into a high and a low TF32 part, and three
  tensor-core products.
- **Exact FP32 FMA** on the CUDA cores.

Then there is a second, less obvious choice: **the order in which the 6,144 products
of each output are added.**

I simulated it on the CPU over 262,144 outputs of LeetGPU's largest graded test (8192 ×
6144 × 4096, inputs in [−1, 1], tolerance 1e-4), on two random seeds. The table shows
the worst error as a fraction of the allowed error, so below 1 passes:

| Method | Worst error / allowed |
|---|---|
| TF32 | **212**: fails |
| FP32, one FMA chain per output (`acc = fma(a, b, acc)` over all 6,144 products) | **1.12–1.28: fails** |
| FP32, a fresh partial sum every 8 products, added to the accumulator | 0.55–0.69 |
| FP32, a fresh partial every 32 products (the Triton version) | 0.51–0.60 |
| FP32, a fresh partial every 64 products (the CUDA version) | 0.46–0.57 |
| `tf32x3`, a fresh partial per 32-wide tile | about the same as FP32 with partials of 32 |
| The exact answer, rounded once to FP32 | 0.42–0.48 |

The last row is the floor: PyTorch's own FP32 result is that far from the truth, so no
kernel can score lower. Three conclusions:

- **Plain TF32 is out.** That rules out Triton's default `tl.dot` for FP32 inputs.
- **Exact FP32 arithmetic isn't enough on its own.** The textbook kernel keeps one
  accumulator per output and chains every product into it. Its rounding error grows with
  the chain length, and at 6,144 products it crosses the tolerance on a few outputs out of
  millions. That is enough to fail the test. Splitting the sum into a short partial chain
  plus a short chain of partials brings it close to the floor.
- **`tf32x3` with partials is as accurate as FP32 with partials.** So the Triton version
  uses it on sm_80+ (about 2.5× the FP32 CUDA-core rate on an H100). It uses exact
  `"ieee"` FP32 on a T4, which has no TF32 hardware.

In code, the fix is one line in Triton: `acc += tl.dot(a_t, b_t)` instead of
`acc = tl.dot(a_t, b_t, acc)`. In the CUDA version it is a second 8 × 8 register array,
`part`, emptied into `acc` every 8 K tiles. Both solutions' tests include an inner
dimension of 6,144.

These numbers come from a CPU simulation of each summation order against a CPU FP32
reference; LeetGPU's reference is cuBLAS. The safety margin is real but not large, so
this is the first thing to recheck on a GPU.

## Code

=== "Triton"

    ```python
    --8<-- "docs/leetgpu/21_matrix_multiplication.py"
    ```

=== "CUDA C++"

    ```cuda
    --8<-- "docs/leetgpu/21_matrix_multiplication.cu"
    ```

    The classic register-blocked SGEMM. A 256-thread block owns a 128 × 128 tile of C,
    and each thread owns an 8 × 8 sub-tile: 64 accumulators in registers. Each step reads
    two float4s of A and two of B from shared memory and does 64 FMAs with them. That is
    4 FMAs per float read, twice the 4 × 4 kernel in [15](15-spmm.md). A thread's rows
    and columns are split 4 + 4 (`ty*4…` and `64 + ty*4…`), so a warp's float4 reads
    from shared memory either broadcast or hit distinct banks. Global loads and stores
    use float4 when the row length is a multiple of 4, and bounds-checked scalars
    otherwise, so every size from 1 × 1 × 1 up works. The second array of 64 partial
    sums, needed for accuracy (see above), raises the count from 111 to 185 registers,
    with no spills on sm_90. That allows 1 block of 256 threads per SM instead of 2:
    occupancy given up for correctness.

## Performance

FLOPs = $2 \cdot 8192 \cdot 6144 \cdot 4096$ = 412 GFLOP; bytes (A, B read once, C written
once) = 436 MB.

| | T4 | H100 SXM |
|---|---|---|
| Memory floor (436 MB) | 1.4 ms | 0.13 ms |
| Exact FP32 (`"ieee"`, CUDA cores) | 412 G / 8.1 TFLOP/s ≈ **51 ms** | 412 G / 67 TFLOP/s ≈ **6.2 ms** |
| `tf32x3` (3 TF32 products) | not available | 3 × 412 G / 495 TFLOP/s ≈ **2.5 ms** |
| Tiles at 128 × 128 | 2,048 | 2,048: ~16 waves at 1 block per SM (CUDA version) |

The memory floor is 40× below the compute floor, so this is compute bound everywhere.
These are floors, not measurements. The CUDA kernel here has none of the double
buffering or warp tiling of a tuned SGEMM, so expect it well below cuBLAS. The page
[Benchmark, profile, scale](benchmarking.md) shows how to measure where it sits.

## Where the data lives

Levels as in [Where the data lives](memory-hierarchy.md). DRAM is GDDR6 on the T4 and HBM3 on the H100.

| Level | In this kernel |
|---|---|
| Registers | the accumulator and the current partial sum: 128 × 128 FP32 each per program (Triton), 8 × 8 each per thread (CUDA). For `tf32x3`, also the hi/lo halves of each operand |
| Shared memory | the A and B tiles. Triton at 128 × 128 × 32 in FP32: 32 KB per stage, 96 KB at 3 stages (fits an H100, not a T4's 64 KB, so the autotuner drops it there). CUDA: two 4 KB tiles |
| L2 | A row panels and B column panels shared across programs, thanks to grouped order. B is 100 MB, so it does not fit even the H100's 50 MB L2: the grouping is what keeps re-reads cheap |
| DRAM | A (201 MB) and B (101 MB) read, C (134 MB) written. With good L2 reuse, close to once each |

## T4 vs H100

| | T4 (sm_75) | H100 (sm_90) |
|---|---|---|
| `_dot_precision()` picks | `"ieee"` (no TF32 tensor cores) | `"tf32x3"` |
| Tiles | 64 × 64 × 32 or 128 × 64 (≤ 64 KB shared) | 128 × 128 × 32, 3 stages |
| CUDA version | the same exact FP32 kernel | the same; a faster CUDA route is 3xTF32 with `mma.sync`/WMMA, or CUTLASS |

## Benchmark and profile

```bash
python docs/leetgpu/21_matrix_multiplication.py         # tests, then vs torch.matmul
PROFILE=1 python docs/leetgpu/21_matrix_multiplication.py
python tests/cuda/run_tests.py --gpu 21_matrix          # the CUDA version on a GPU
```

On a GPU, also compare `torch.backends.cuda.matmul.allow_tf32 = True` against `False`.
PyTorch's FP32 matmul is exact FP32 by default, and turning TF32 on shows the speed you
would buy and the accuracy you would lose.

## Scaling up: the SGEMM ladder

The CUDA version is step 4 of the usual sequence. Each step is a measurable win:

1. One thread per output, reading A and B from DRAM.
2. Coalesced loads (threads along the row of C).
3. Shared-memory tiles (each value loaded once per block).
4. **Register blocking** (this kernel): each thread computes an 8 × 8 tile.
5. Vectorized and transposed loads, then **double buffering** (`cp.async` on sm_80+),
   then warp tiling.
6. Tensor cores: 3xTF32 or BF16 with an FP32 accumulator, through `mma.sync`,
   WGMMA on Hopper, or CUTLASS.

## Watch and read

- [freeCodeCamp CUDA course](https://www.youtube.com/watch?v=86FAWCzIe_4): "optimized matrix multiplication" chapter (tiling from first principles)
- [GPU MODE Lecture 23: Tensor Cores](https://www.youtube.com/watch?v=hQ9GPnV0-50): TF32 and precision trade-offs
- [Triton tutorial 03: Matrix multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html): grouped ordering explained
- Notes: [Autotuned matmul in Triton](../notes/08-triton.md)
