# 17. Fused LayerNorm → Linear → GELU

| | |
|---|---|
| **Question** | *Fused LayerNorm Linear GELU* (medium) on [LeetGPU](https://leetgpu.com/challenges) · [official spec and reference](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/131_fused_layernorm_linear_gelu) |
| **Solution** | [17_fused_layernorm_linear_gelu.py](https://github.com/Prakash-sa/GPU-Codes/blob/main/docs/leetgpu/17_fused_layernorm_linear_gelu.py) |
| **Task** | $y = \text{GELU}_{erf}(\text{LN}(x)\,W + b)$; float32; M, K, N ≤ 8192; perf 4096³; tolerance 1e-3; inputs carry a large offset (≈ −1000) |
| **Bound by** | compute (GEMM) |

## Idea

This is the first half of a transformer MLP block, and the standard example of
**fusion around a GEMM**:

![Prologue and epilogue fusion](../assets/img/leetgpu/ln-gemm-fusion.png)

1. `row_stats_kernel`: mean and rstd per row (2 × M floats), using a **centred two-pass
   variance**. With a −1000 offset, $E[x^2]-E[x]^2$ subtracts two numbers near 10^6 and
   keeps ~1 significant digit.
2. `ln_gemm_gelu_kernel`: a tiled GEMM that
    - **normalizes the x tile as it is loaded** (prologue): `(x - mu) * rstd * gamma + beta`,
      then forces the padding to exactly 0;
    - adds the bias and applies the exact erf-GELU **before the single store** (epilogue).

    The intermediates `xn` (M × K) and `h` (M × N) never exist in DRAM. That saves
    ~256 MB of traffic at 4096³.

!!! warning "Don't apply the tempting algebraic fusion"
    $h = \text{rstd}\cdot(x W') - \text{rstd}\,\mu\,(\mathbf{1}^T W') + (\beta W + b)$ lets you
    run the GEMM on the *raw* x. But raw x ≈ −1000, so the GEMM output is huge and the
    subtraction cancels catastrophically, especially in TF32. Normalize first, then
    multiply. Getting the physics/numerics right beats saving a few FLOPs.

## Code

```python
--8<-- "docs/leetgpu/17_fused_layernorm_linear_gelu.py"
```

## Performance

$2MNK$ = 137 GFLOP. The normalization costs $O(MK \cdot N/\text{BLOCK\_N})$ elementwise
ops, which is small next to the GEMM.

| | T4 ("ieee" FP32 FMA) | H100 SXM ("tf32x3") |
|---|---|---|
| Floor | 137 / 8.1 ≈ **17 ms** | 3 × 137 / 495 ≈ 0.83 ms |
| Plain TF32 (if the tolerance allowed it) | n/a | ≈ 0.28 ms |

## T4 vs H100

| | T4 | H100 |
|---|---|---|
| Dot precision (auto, `_dot_precision`) | `"ieee"`: T4 has no TF32 | `"tf32x3"` |
| Configs | 64×64×32, 2 stages (64 KB of shared memory) | 128×128×32, 3 stages, 8 warps |
| Faster option | store W in FP16 and run an FP16 dot (65 TFLOP/s), **if** a tolerance study allows it | BF16 operands with FP32 accumulation: ~2× over TF32. Again only with a measured tolerance study |

## Benchmark and profile

```bash
TRITON_PRINT_AUTOTUNING=1 python docs/leetgpu/17_fused_layernorm_linear_gelu.py
PROFILE=1 python docs/leetgpu/17_fused_layernorm_linear_gelu.py
```

The torch reference shows `layer_norm`, a GEMM (`sgemm`/`cutlass`), an add and a `gelu`
kernel. The fused version shows a tiny stats kernel and one GEMM kernel. Compare GEMM
time alone (`torch.mm(x, W)`) against the fused kernel. The fused kernel's overhead over
a bare GEMM is the cost of the prologue/epilogue.

## Scaling up

- At realistic sizes this is GEMM bound. The levers are the operand precision
  (BF16/FP8 on H100) and the GEMM quality (persistent kernel, TMA, larger tiles).
- **Normalize once**: if N/BLOCK_N is large, the prologue repeats work for every column
  of tiles. An alternative writes `xn` in BF16 (½ the bytes) and runs a plain GEMM.
  Measure both.
- The second half of the MLP (`GELU(h) @ W2`) can't fuse the same way (a reduction over
  N). It is a separate GEMM with a residual-add epilogue.
- Tensor-parallel (multi-GPU): shard W by columns (N). Each GPU computes its slice of
  `y` with no communication until the next row-parallel GEMM.

## Watch and read

- GPU MODE Lecture 18: Fused Kernels: [slides and code](https://github.com/gpu-mode/lectures/tree/main/lecture_018). The video is on the [GPU MODE channel](https://www.youtube.com/@GPUMODE).
- [GPU MODE Lecture 23: Tensor Cores](https://www.youtube.com/watch?v=hQ9GPnV0-50)
- [Triton tutorials 03 (matmul)](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) and [05 (layer norm)](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)
