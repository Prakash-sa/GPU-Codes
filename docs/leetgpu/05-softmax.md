# 05. Softmax (whole 1-D array)

| | |
|---|---|
| **Question** | *Softmax* (medium) on [LeetGPU](https://leetgpu.com/challenges) · [official spec and reference](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/5_softmax) |
| **Solution** | [05_softmax.py](05_softmax.py) |
| **Task** | $\sigma(x)_i = e^{x_i - \max x} / \sum_j e^{x_j - \max x}$ over the whole array, float32, N ≤ 500K; tolerance 1e-5 |
| **Bound by** | launch overhead on H100, bandwidth on T4 (only 6 MB moved) |

## Idea

This is *not* the row-wise softmax from the Triton tutorial: the whole array is one
softmax, too long for one program. The max and the sum have to be combined across
programs. The **online-softmax merge rule** makes that a single cheap step:

$$
M = \max_b m_b, \qquad S = \sum_b s_b\, e^{m_b - M}, \qquad
\text{where } m_b = \max(x_b),\ s_b = \textstyle\sum e^{x_b - m_b}
$$

1. `softmax_partial_kernel`: each program reads its chunk once and writes $(m_b, s_b)$.
2. `softmax_normalize_kernel`: **every** program loads all partials (only N / 4096 ≈ 123
   of them), merges them, and writes $e^{x - M}/S$ for its chunk. Merging redundantly in
   every program avoids a third launch.

The max trick (subtract $M$ before `exp`) prevents overflow. The test includes an
all-80.0 input, which overflows a naive `exp`.

![Softmax partials merge](../assets/img/leetgpu/softmax-partials.png)

## Code

```python
--8<-- "docs/leetgpu/05_softmax.py"
```

## Performance

Traffic is 2 reads + 1 write = 6 MB. That is 19 µs on T4 but only 1.8 µs on H100, where
the two ~3–5 µs launches cost more than the data movement.

| | T4 | H100 SXM |
|---|---|---|
| Roofline floor | 19 µs | 1.8 µs |
| What dominates | DRAM | kernel launches |

## T4 vs H100

| | T4 | H100 |
|---|---|---|
| BLOCK | 4096 (123 programs, 3 per SM) | 4096 → 123 programs < 132 SMs. Try BLOCK = 2048 (245 programs) so every SM gets work |
| Launch overhead | minor | dominant: capture both kernels in a **CUDA Graph** when called repeatedly |

**Single-launch alternative:** one kernel where every program writes its partial, bumps
an atomic counter, and spin-waits until all partials are present, then normalizes.
This only works if every program is resident at the same time (grid ≤ SMs × programs
per SM). Otherwise it deadlocks. It is a fine interview talking point, but risky to
ship.

## Benchmark and profile

```bash
python docs/leetgpu/05_softmax.py
PROFILE=1 python docs/leetgpu/05_softmax.py     # look at the GPU-idle gap between the 2 kernels
```

`torch.softmax(x, 0)` on a 1-D tensor is also a multi-kernel reduction. Compare the
kernel counts in the trace.

## Scaling up

- **Row-wise softmax** (`[rows, N]`, the transformer case): one program per row when the
  row fits in registers (the [Triton tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)).
  For long rows, use the online one-pass loop ($m, s$ updated per tile).
- **Fuse the consumer:** softmax followed by a matmul is attention. See 18, which never
  materializes the softmax at all.
- Very large N: the partials array grows (N / BLOCK). Merge it hierarchically, or have
  one program reduce it and the others read the result.

## Watch and read

- [GPU MODE Lecture 12: Flash Attention](https://www.youtube.com/watch?v=zEuwuCTEf_0): online softmax from first principles
- [Milakov & Gimelshein, *Online normalizer calculation for softmax*](https://arxiv.org/abs/1805.02867)
- [Triton tutorial 02: Fused softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
