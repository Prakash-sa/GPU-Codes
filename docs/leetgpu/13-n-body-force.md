# 13. N-body gravitational force

| | |
|---|---|
| **Question** | *N-body Gravitational Force* (medium) on [LeetGPU](https://leetgpu.com/challenges) · [official spec and reference](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/101_n_body_force) |
| **Solution** | [13_n_body_force.py](https://github.com/Prakash-sa/GPU-Codes/blob/main/docs/leetgpu/13_n_body_force.py) |
| **Task** | $F_i = \sum_j m_j (p_j - p_i) / (\lVert p_j - p_i\rVert^2 + \epsilon^2)^{3/2}$, ε = 1e-3; N ≤ 65,536; perf N = 8192; tolerance 1e-2 |
| **Bound by** | FP32 compute |

## Idea

This uses the same tiling as [12. Nearest neighbor](12-nearest-neighbor.md): a program
holds `BLOCK_I` bodies, streams every body in `BLOCK_J` tiles, and accumulates
`(Fx, Fy, Fz)` in registers. Each loaded j-tile is reused `BLOCK_I` times.

![All-pairs tiling](../assets/img/leetgpu/pairwise-tiling.gif)

The maths is rewritten for the GPU:

$$
(r^2+\epsilon^2)^{-3/2} = \text{rsqrt}(r^2+\epsilon^2)^3
$$

One `rsqrt` (special-function unit) plus two multiplies replaces `pow` and a divide.

- **No `i == j` mask is needed.** The self term has $r = 0$, so it contributes 0, and ε
  keeps the denominator finite.
- **Padded bodies get mass 0** (`other=0.0`), so they contribute nothing either.

## Code

```python
--8<-- "docs/leetgpu/13_n_body_force.py"
```

## Performance

$N^2$ = 67M interactions × ~20 FLOPs ≈ 1.3 GFLOP.

| | T4 | H100 SXM |
|---|---|---|
| FP32 floor | 0.17 ms | 20 µs |
| PyTorch reference (chunked, materializes [N, 1024, 3]) | many kernels, GBs of temporaries | same |

## T4 vs H100

| | T4 | H100 |
|---|---|---|
| Tiles | 64 × 64, 4 warps (128 programs ≈ 3 per SM) | 64 × 64 gives 128 programs < 132 SMs. Use BLOCK_I = 32 (256 programs) or split j across programs and reduce |
| FP64 | **avoid** (1/32 rate) | FP64 at half the FP32 rate is viable if the science needs it |

A split-j variant (2–4 programs per i-block, each over a range of j, partial forces
summed at the end) is the standard fix when N is small relative to the SM count.

## Benchmark and profile

```bash
python docs/leetgpu/13_n_body_force.py
ncu --kernel-name regex:nbody --launch-count 1 --section SpeedOfLight --section InstructionStats python docs/leetgpu/13_n_body_force.py
```

Check that FMA and MUFU (rsqrt) dominate the instruction mix, and that DRAM is near
idle.

## Scaling up

- **Cutoff + neighbour lists** (molecular dynamics): forces beyond r_c are dropped, so
  use the cell lists from page 12 for $O(N)$ work.
- **Long-range forces at large N**: Barnes–Hut (tree, $O(N \log N)$) or particle-mesh
  Ewald (FFT).
- **Newton's third law** ($F_{ij} = -F_{ji}$) halves the pair count, but needs atomics or
  a symmetric reduction. On GPUs the plain "compute both" version is often faster.
- **Mixed precision**: FP32 forces with FP64 position integration is the usual MD
  compromise.
- Multi-GPU: replicate positions (small), split i-blocks; one all-gather per step.

## Watch and read

- [GPU Gems 3, Chapter 31: Fast N-Body Simulation with CUDA](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-31-fast-n-body-simulation-cuda)
- [GPU MODE Lecture 14: Practitioner's Guide to Triton](https://www.youtube.com/watch?v=DdTsX6DQk24)
