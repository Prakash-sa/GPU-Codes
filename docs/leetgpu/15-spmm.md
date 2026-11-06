# 15. Sparse × dense matrix multiplication

| | |
|---|---|
| **Question** | *Sparse Matrix-Dense Matrix Multiplication* (medium) on [LeetGPU](https://leetgpu.com/challenges) · [official spec and reference](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/75_sparse_matrix_dense_matrix_multiplication) |
| **Solution** | [15_spmm.py](https://github.com/Prakash-sa/GPU-Codes/blob/main/docs/leetgpu/15_spmm.py) (Triton) · [15_spmm.cu](https://github.com/Prakash-sa/GPU-Codes/blob/main/docs/leetgpu/15_spmm.cu) (CUDA C++) |
| **Task** | C = A·B, A is M × N with **60–70% zeros, stored dense**, B dense N × K, float32; perf 4096 × 2048 × 512; tolerance 1e-3 |
| **Bound by** | compute |

## Idea: question the premise

The word "sparse" invites a CSR kernel, but measure before deciding:

![Dense tiles vs CSR](../assets/img/leetgpu/spmm-layouts.png)

| | Dense tiled GEMM (`solve`) | CSR SpMM (`solve_csr`) |
|---|---|---|
| Work | $2MNK$ = 8.6 GFLOP, on **tensor cores** | $2 \cdot nnz \cdot K$ ≈ 3 GFLOP, on CUDA cores |
| Memory pattern | regular tiles, one pass over A | gather a B row per non-zero (irregular) |
| Setup | none | build CSR: a pass over A plus a scan |

At ~35% density the dense path wins on tensor-core GPUs. CSR wins below roughly 1–5%
density. `solve()` is therefore a tiled dense GEMM, and `solve_csr()` is included to
measure the crossover yourself.

**Precision.** A and B are FP32, and the tolerance is 1e-3 on sums of 2048 products of
values up to ±10. Plain TF32 (10-bit mantissa) can miss that. `input_precision="tf32x3"`
splits each operand into a high and a low TF32 part and does 3 tensor-core products,
which is near-FP32 accuracy at well above the FP32 CUDA-core rate.

## Code

=== "Triton"

    ```python
    --8<-- "docs/leetgpu/15_spmm.py"
    ```

=== "CUDA C++"

    ```cuda
    --8<-- "docs/leetgpu/15_spmm.cu"
    ```

    A register-tiled FP32 GEMM on the CUDA cores. It is exact FP32 (no TF32 precision
    question), so it passes the 1e-3 tolerance directly; the Triton version reaches the
    same accuracy on tensor cores with `tf32x3`.

## Performance

| | T4 (FP32 FMA, "ieee") | H100 SXM ("tf32x3") |
|---|---|---|
| Floor | 8.6 GFLOP / 8.1 TFLOP/s ≈ 1.06 ms | 3 × 8.6 / 495 ≈ 52 µs |
| CSR alternative | ~3 GFLOP on CUDA cores, but irregular: likely slower | CUDA-core bound, ≥ 45 µs plus conversion |

## Where the data lives

Levels as in [Where the data lives](memory-hierarchy.md). DRAM is GDDR6 on the T4 and HBM3 on the H100.

| Level | In this kernel |
|---|---|
| Registers | the fp32 accumulator; for `"tf32x3"`, the hi/lo halves of each operand, split before three tensor-core products |
| Shared memory | **fp32** tiles are twice the size of the fp16 ones in [09](09-gemm.md): 128×32 + 32×128 floats = 32 KB per stage, 96 KB at 3 stages. That fits an H100, not a T4 (64 KB), and the autotuner drops it there |
| L2 | A and B panel reuse across programs, as in 09. B is 2048 × 512 × 4 = 4 MB: it fits the H100's L2 but is the T4's entire L2, so expect misses there |
| DRAM | read A and B, write C: ~46 MB at 4096 × 2048 × 512. The zeros in A still cost bandwidth, because A is stored dense |

`solve_csr` reads `row_ptr`, `col_idx` and `vals` as scalars and gathers rows of B. The
gathers are random, so they depend on B staying in L2.

## T4 vs H100

This is the page where **the code must differ**:

| | T4 (sm_75) | H100 (sm_90) |
|---|---|---|
| TF32 tensor cores | **none**: `tf32x3` would do 3× the FP32 work | yes |
| `_dot_precision()` picks | `"ieee"` (exact FP32 FMA) | `"tf32x3"` |
| Tiles | ≤ 64 KB of shared memory: 64×64×32, 2 stages | 128×128×32, 3 stages |

`_dot_precision()` reads `torch.cuda.get_device_capability()` at call time, so the same
file runs correctly on both GPUs.

## Benchmark and profile

```bash
python docs/leetgpu/15_spmm.py      # dense vs CSR vs torch.matmul
```

Try densities of 35%, 10%, 2% and 0.5% (change the `> 0.65` mask threshold) and plot
time against density to find the crossover. That plot is the real deliverable.

## Scaling up

- **Structured sparsity**: on Ampere/Hopper, 2:4 sparsity runs on sparse tensor cores
  (2× peak), but the matrix must have exactly 2 non-zeros per 4.
- **Block sparsity** (zero tiles, not zero elements) lets a dense kernel *skip* tiles.
  That is how block-sparse attention works. With random 65% sparsity no 64×32 tile is
  ever all-zero.
- **Genuinely sparse** (graphs, ≪ 1%): CSR/ELL, or a sorted-COO SpMM, with row
  splitting for load balance across power-law degree distributions.

## Watch and read

- [GPU MODE Lecture 11: Sparsity](https://www.youtube.com/watch?v=mGDnOLcfE8g)
- [GPU MODE Lecture 23: Tensor Cores](https://www.youtube.com/watch?v=hQ9GPnV0-50): TF32 and precision trade-offs
