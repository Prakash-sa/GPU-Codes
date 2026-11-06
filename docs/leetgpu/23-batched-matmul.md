# 23. Batched matrix multiplication (FP32)

| | |
|---|---|
| **Question** | *[Batched Matrix Multiplication](https://leetgpu.com/challenges/batched-matrix-multiplication)* (medium) on LeetGPU · [official spec and reference](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/30_batched_matrix_multiplication) |
| **Solution** | [23_batched_matmul.py](https://github.com/Prakash-sa/GPU-Codes/blob/main/docs/leetgpu/23_batched_matmul.py) (Triton) · [23_batched_matmul.cu](https://github.com/Prakash-sa/GPU-Codes/blob/main/docs/leetgpu/23_batched_matmul.cu) (CUDA C++) |
| **Task** | `C[b] = A[b] @ B[b]`, **float32**; A is BATCH × M × K, B is BATCH × K × N; BATCH ≤ 128, M, N, K ≤ 1024; perf 32 × 256³; tolerance **1e-5** |
| **Bound by** | compute on the CUDA cores, and latency (each product is small) |

## Idea

The FP32 sibling of [10. FP16 batched matmul](10-fp16-batched-matmul.md). The batching
idea is the same: **put the batch on a grid axis**, so one launch runs every product's
tiles together. A Python loop of 32 launches would leave the GPU mostly idle. What is
different is precision, and it changes which hardware the kernel can use.

### 1e-5 is FP32's own noise level

Simulated on the CPU against an FP32 reference:

| Method | 32 × 256³, inputs in [−1, 1] | 1024³ (the size limit) |
|---|---|---|
| TF32 tensor cores (`tl.dot` default) | **fails**: 531× the allowed error | fails: 1025× |
| `tf32x3` | passes: 0.57 of the allowed | fails: 2.0× |
| FP32, summed in a different order than PyTorch | passes: about 0.5 of the allowed | fails: 1.5–1.9× |

At 1024³, even exact FP32 arithmetic misses the tolerance, just from adding the
products in a different order. The tolerance leaves no room for any approximation, so
both solutions use **exact FP32 FMA** (`input_precision="ieee"` in Triton). The graded
functional tests are small (up to 16 × 16 × 16), and those pass with every FP32 method. The
kernels keep one FMA chain per output: with K ≤ 256 in the perf case the chain is short,
unlike the 6,144-long chains in [21](21-matrix-multiplication.md#precision-and-summation-order-decide-the-result),
where splitting the sum was necessary.

### Tile size vs program count

The 256³ products are small, so the tile size has to balance two things:

| Tile | Programs at 32 × 256³ | Per SM on an H100 (132 SMs) |
|---|---|---|
| 128 × 128 | 32 × 4 = 128 | < 1: a quarter of the SMs idle at the tail |
| 64 × 64 | 32 × 16 = 512 | 3.9 |
| 32 × 32 | 32 × 64 = 2,048 | 15.5, but each tile reuses data half as much |

64 × 64 is the default in both versions. The Triton autotuner also tries 32 × 64 and
128 × 64.

## Code

=== "Triton"

    ```python
    --8<-- "docs/leetgpu/23_batched_matmul.py"
    ```

=== "CUDA C++"

    ```cuda
    --8<-- "docs/leetgpu/23_batched_matmul.cu"
    ```

    The 64 × 64 register-tiled FP32 GEMM, with the batch index on `blockIdx.z` and
    `size_t` batch offsets. The A tile is stored transposed with one float of padding
    (`BM + 1`), so the transposing stores don't hit the same bank. Compare with
    [21](21-matrix-multiplication.md), whose 128 × 128 tile is the right choice for one
    big product but would give only 128 blocks here.

## Performance

FLOPs = $2 \cdot 32 \cdot 256^3$ = 1.07 GFLOP; bytes = $3 \cdot 32 \cdot 256^2 \cdot 4$ = 25 MB
(twice the FP16 version's).

| | T4 | H100 SXM |
|---|---|---|
| Memory floor (25 MB) | 79 µs | 7.5 µs |
| FP32 compute floor | 1.07 G / 8.1 TFLOP/s ≈ **132 µs** | 1.07 G / 67 TFLOP/s ≈ **16 µs** |
| FP16 version ([10](10-fp16-batched-matmul.md)), for scale | ~39 µs | ~4 µs |

The precision requirement costs about 4× on an H100: FP32 CUDA cores instead of FP16
tensor cores. That price is set by the tolerance, not by the code.

## Where the data lives

Levels as in [Where the data lives](memory-hierarchy.md). DRAM is GDDR6 on the T4 and HBM3 on the H100.

| Level | In this kernel |
|---|---|
| Registers | the FP32 accumulator: 64 × 64 per program (Triton), 4 × 4 per thread (CUDA) |
| Shared memory | A and B tiles: 64 × 32 + 32 × 64 floats = 16 KB per stage in Triton; 8 KB in the CUDA version |
| L2 | one batch's A and B are 256 KB each. All 32 batches (A, B, C) total 25 MB: inside the H100's 50 MB L2, larger than the T4's 4 MB |
| DRAM | A and B read and C written once: 25 MB |

## T4 vs H100

| | T4 | H100 |
|---|---|---|
| Programs at 64 × 64 | 512: 12.8 per SM | 512: 3.9 per SM |
| Bound by | FP32 compute (132 µs vs 79 µs of memory) | FP32 compute (16 µs vs 7.5 µs) |
| If the tolerance allowed it | — | `tf32x3` would roughly halve the time |

## Benchmark and profile

```bash
python docs/leetgpu/23_batched_matmul.py        # vs torch.bmm
PROFILE=1 python docs/leetgpu/23_batched_matmul.py
python tests/cuda/run_tests.py --gpu 23_batched
```

Compare with [10](10-fp16-batched-matmul.md) at the same size. The difference is the
price of FP32. Then try `BLOCK_M = BLOCK_N = 128` to see what too few programs costs.

## Scaling up

- **Variable shapes per batch** (MoE experts): grouped GEMM, one problem descriptor per
  program.
- **Strided batches** (attention heads inside a `[B, H, S, D]` tensor): pass batch
  strides instead of assuming packed `M × K` blocks, as `torch.bmm` does through
  `as_strided`.
- **Many tiny matrices** (3 × 3, 4 × 4): one thread per matrix, everything in registers;
  tiling is pure overhead there.
- **Large batches at 1024³**: compute bound with plenty of programs, so the
  [21](21-matrix-multiplication.md) 128 × 128 tile becomes the better choice again.

## Watch and read

- [Triton tutorial 03: Matrix multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
- [GPU MODE Lecture 23: Tensor Cores](https://www.youtube.com/watch?v=hQ9GPnV0-50): why FP32 inputs can't use them directly
