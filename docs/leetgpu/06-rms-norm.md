# 06. RMS normalization (whole 1-D array)

| | |
|---|---|
| **Question** | *RMS Normalization* (medium) on [LeetGPU](https://leetgpu.com/challenges) · [official spec and reference](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/50_rms_normalization) |
| **Solution** | [06_rms_norm.py](https://github.com/Prakash-sa/GPU-Codes/blob/main/docs/leetgpu/06_rms_norm.py) |
| **Task** | $y = \gamma \cdot x / \sqrt{\text{mean}(x^2) + \epsilon} + \beta$ with **scalar** γ, β; N ≤ 100K; tolerance 1e-5 |
| **Bound by** | launch overhead (1.2 MB of traffic) |

## Idea

This has the same shape as [05. Softmax](05-softmax.md): one global statistic, then an
elementwise pass.

1. `sumsq_partial_kernel` writes $\sum x^2$ per chunk.
2. `rms_apply_kernel`: every program re-adds the ≤ 25 partials, computes
   `scale = gamma * rsqrt(total / N + eps)` **once**, and writes `x * scale + beta`. That
   is one multiply-add per element, and no divide.

## Code

```python
--8<-- "docs/leetgpu/06_rms_norm.py"
```

## Performance

| | T4 | H100 SXM |
|---|---|---|
| Roofline floor (1.2 MB) | 3.8 µs | 0.4 µs |
| Two launches | ~6–10 µs | ~6–10 µs: **launch bound on both** |

Profiling will show two very short kernels with a gap between them. Bandwidth tuning
cannot help at this size. What helps:

- **one launch**: N ≤ 100K floats = 400 KB, and a single program with a loop (or a
  cooperative pattern) handles it. A one-program version, looping twice over the array,
  is often faster at N = 100K. Measure it.
- **CUDA Graphs** if it runs in a loop.

## Where the data lives

Levels as in [Where the data lives](memory-hierarchy.md). DRAM is GDDR6 on the T4 and HBM3 on the H100.

| Level | In this kernel |
|---|---|
| Registers | each program's block of x; `gamma`, `beta` and `eps` are scalar kernel arguments (constant cache, then registers) |
| Shared memory | the cross-warp step of the sum of squares |
| L2 | the P partial sums; the re-read of x in kernel 2 (N ≤ 100K is 400 KB, well inside L2) |
| DRAM | read x, write y: 800 KB. At this size the two launches cost more than the data |

## T4 vs H100

| | T4 | H100 |
|---|---|---|
| Programs (BLOCK=4096) | 25: fewer than 40 SMs | 25 ≪ 132 SMs |
| Suggestion | BLOCK=2048 → 49 programs | BLOCK=1024 → 98 programs, or the single-program variant |

## Benchmark and profile

```bash
python docs/leetgpu/06_rms_norm.py
PROFILE=1 python docs/leetgpu/06_rms_norm.py   # note CPU launch time vs CUDA time per kernel
```

In the profiler table, compare *CPU total* per launch with *CUDA total*. When the CPU
time is larger, you are launch bound.

## Scaling up

- The model-relevant version is **row-wise** RMSNorm over `[tokens, hidden]` with a
  per-channel weight. That is page [08](08-fused-residual-rms-norm.md), and in practice
  it is always fused with the residual add.
- At large N the two-pass structure is bandwidth bound and fine, at 12 bytes per
  element.

## Watch and read

- [GPU MODE Lecture 28: Liger Kernel](https://www.youtube.com/watch?v=gWble4FreV4): RMSNorm and other fused Triton kernels in production
- [Triton tutorial 05: Layer normalization](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)
