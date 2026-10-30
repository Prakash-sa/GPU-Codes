# 04. Dot product

| | |
|---|---|
| **Question** | *Dot Product* (medium) on [LeetGPU](https://leetgpu.com/challenges) · [official spec and reference](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/17_dot_product) |
| **Solution** | [04_dot_product.py](04_dot_product.py) |
| **Task** | `result[0] = sum(A * B)`, float32; perf N = 4,194,304; tolerance 1e-5 |
| **Bound by** | memory bandwidth (2 FLOPs per 8 bytes) |

## Idea

This is [03. Reduction](03-reduction.md) with the multiply **fused into the load loop**.
The eager PyTorch version `(A * B).sum()` writes a 16 MB temporary and reads it back.
`torch.dot` avoids that, and so does this kernel: `acc += a * b` in registers (one FMA).
Partials are combined in float64 for accuracy and determinism.

## Code

```python
--8<-- "docs/leetgpu/04_dot_product.py"
```

## Performance

Bytes = $2 \cdot 4N$ = 33.6 MB.

| | T4 | H100 SXM |
|---|---|---|
| Roofline floor | 105 µs | 10 µs |
| `(A*B).sum()` eager | ~2.5× the bytes (temporary write + re-read) | same |

## Accuracy and precision

- Per-lane FP32 FMA accumulation over N / (P·BLOCK) terms, then a tree, then FP64 across
  programs. This easily meets 1e-5 for inputs in [−1, 1].
- The FP16 variant on LeetGPU (*FP16 Dot Product*) should still **accumulate in FP32**.
  An FP16 accumulator loses everything past ~2048 terms (11-bit mantissa).

## T4 vs H100

Same as 03: `P = 4 × SMs` adapts automatically. On H100, try BLOCK = 8192 with
`num_warps=16`. With two input streams there are more loads in flight per iteration
already, so the gain is usually small.

## Benchmark and profile

```bash
python docs/leetgpu/04_dot_product.py
PROFILE=1 python docs/leetgpu/04_dot_product.py
```

Compare with both `torch.dot` (one cuBLAS kernel) and `(a*b).sum()` (two kernels plus a
temporary). The profiler table makes the extra kernel visible.

## Scaling up

- **Batched dot products** (`[B, N]` rows): one program per row (or per block of rows),
  `tl.sum` along axis 1. No second pass is needed if a row fits one program.
- When a dot product feeds a matrix product, use `tl.dot` (tensor cores) instead. See 09.
- At very large N with mixed signs, consider Kahan or pairwise compensation per lane.

## Watch and read

- [Mark Harris, *Optimizing Parallel Reduction in CUDA*](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
- [GPU MODE Lecture 14: Practitioner's Guide to Triton](https://www.youtube.com/watch?v=DdTsX6DQk24)
