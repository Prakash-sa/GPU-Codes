# 09. GEMM (FP16)

| | |
|---|---|
| **Question** | *General Matrix Multiplication (GEMM)* (medium) on [LeetGPU](https://leetgpu.com/challenges) · [official spec and reference](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/22_gemm) |
| **Solution** | [09_gemm.py](09_gemm.py) |
| **Task** | $C = \alpha AB + \beta C$, FP16 in and out; 16 ≤ M, N, K ≤ 4096; perf 1024³; tolerance 0.05 |
| **Bound by** | tensor-core compute (at 1024³ on H100: latency, since the problem is small) |

## Idea

Each program owns a `BLOCK_M × BLOCK_N` tile of C and loops over K. It loads an A tile
and a B tile, then `acc = tl.dot(a, b, acc)`. That one call is the tensor-core MMA,
with FP16 inputs and an **FP32 accumulator**.

![One program computing one C tile](../assets/img/triton-matmul.gif)

Three optimizations on top of that:

1. **Grouped program order.** Programs that run at the same time should share A rows
   and B columns, so those tiles stay in L2.

    ![Row-major vs grouped program order](../assets/img/leetgpu/grouped-order.png)

2. **Fused epilogue.** `alpha * acc + beta * C_old` is computed before the single store.
   C is read once and written once, with no second kernel.
3. **Autotuning** over tile shape, `num_warps` and `num_stages` (software-pipelining
   depth: how many K tiles are loaded ahead while the current one is multiplied).

## Code

```python
--8<-- "docs/leetgpu/09_gemm.py"
```

## Performance

FLOPs = $2MNK$ = 2.15 GFLOP; bytes ≈ 8 MB, so intensity is ≈ 270 FLOP/byte: compute bound.

| | T4 | H100 SXM |
|---|---|---|
| FP16 tensor peak | 65 TFLOP/s | 989 TFLOP/s |
| Floor at 1024³ | 33 µs | 2.2 µs |
| Realistic | 60–75% of peak | 1024³ is too small to fill 132 SMs (64 tiles of 128²); latency bound. At 4096³ expect 60–75% of peak |

`torch.addmm` (cuBLAS) is the baseline. Matching it within ~10–20% is a good result.

## T4 vs H100

| | T4 (sm_75) | H100 (sm_90) |
|---|---|---|
| Configs | ≤ 64 KB shared memory: 128×128×32 with 2 stages = 32 KB OK; 128×256×64×3 stages fails (`OutOfResources`, which the autotuner skips) | add 128×256×64, 3–4 stages, 8 warps; Triton emits **wgmma** automatically |
| `num_stages` | 2 (no `cp.async` on Turing, so deep pipelining doesn't help) | 3–4 (async copies / TMA) |
| Small problems | 1024³ = 64 tiles of 128² vs 40 SMs: OK | prefer 64×64 tiles (256 programs) or **split-K** for small M·N |
| Beyond | — | persistent kernel + TMA descriptors (`tl.make_tensor_descriptor`), FP8 |

## Benchmark and profile

```bash
TRITON_PRINT_AUTOTUNING=1 python docs/leetgpu/09_gemm.py   # which config won per shape
ncu --kernel-name regex:gemm --launch-skip 10 --launch-count 1 --set full python docs/leetgpu/09_gemm.py
```

In ncu, check *Compute (SM) Throughput* and the **tensor pipe utilization**. Also check
*Launch Statistics → Waves Per SM*: a fractional last wave is the tail effect.

## Scaling up

- Large K with small M·N: **split-K** (several programs per C tile, then reduce) to get
  enough parallelism.
- **Precision ladder**: FP16 → BF16 (same speed, more range, H100 only) → FP8 on H100
  (2× throughput, needs scaling).
- **Epilogue fusion** is the main way to beat cuBLAS in practice: bias, activation,
  residual or quantization in the same kernel (see 17).
- Multi-GPU: tensor parallelism shards N (column-parallel) or K (row-parallel, then
  all-reduce).

## Watch and read

- [GPU MODE Lecture 23: Tensor Cores](https://www.youtube.com/watch?v=hQ9GPnV0-50)
- [freeCodeCamp CUDA course](https://www.youtube.com/watch?v=86FAWCzIe_4): "optimized matrix multiplication" chapter (tiling from first principles)
- [Triton tutorial 03: Matrix multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html): grouped ordering explained
- Notes: [Autotuned matmul in Triton](../notes/08-triton.md)
