# 12. Nearest neighbor (3-D points)

| | |
|---|---|
| **Question** | *Nearest Neighbor* (medium) on [LeetGPU](https://leetgpu.com/challenges) · [official spec and reference](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/38_nearest_neighbor) |
| **Solution** | [12_nearest_neighbor.py](https://github.com/Prakash-sa/GPU-Codes/blob/main/docs/leetgpu/12_nearest_neighbor.py) |
| **Task** | for each point, the index j ≠ i of the closest point; N ≤ 100K; perf N = 10K; **exact match** (atol = 0) |
| **Bound by** | FP32 compute ($O(N^2)$ distance evaluations) |

## Idea

The PyTorch reference builds the full N × N distance matrix: 400 MB at N = 10K and
40 GB at N = 100K (out of memory). The kernel never builds it:

- a program owns `BLOCK_I` query points (x, y, z in registers);
- it sweeps all points in tiles of `BLOCK_J`, computes a `BLOCK_I × BLOCK_J` block of
  squared distances in registers, and masks `j == i` and padding with `+inf`;
- it keeps a running `(best_d, best_j)` per query.

![All-pairs tiling](../assets/img/leetgpu/pairwise-tiling.gif)

**Exact ties matter** because the checker uses `atol = 0`. `torch.argmin` returns the
*lowest* index on ties. Two details reproduce that:

1. within a tile, `tl.min(d, axis=1, return_indices=True)` picks the leftmost minimum;
2. across tiles, the update uses a **strict** `<`, so an earlier tile wins ties.

The test includes 300 identical points, which are all ties.

## Code

```python
--8<-- "docs/leetgpu/12_nearest_neighbor.py"
```

## Performance

~$N^2$ = 10^8 pairs × ~9 FLOPs ≈ 0.9 GFLOP; memory traffic is tiny (120 KB of points,
re-read from L1/L2).

| | T4 | H100 SXM |
|---|---|---|
| FP32 floor | 0.11 ms | 13 µs |
| vs PyTorch reference | avoids 400 MB of traffic and 5+ kernels: large speedup | same |

## T4 vs H100

| | T4 | H100 |
|---|---|---|
| Programs at BLOCK_I = 64 | 157 for 40 SMs: fine | 157 for 132 SMs: ~1.2 waves, tail effect. Use BLOCK_I = 32 (313 programs) |
| Registers | 64 × 128 tile = 32 values/thread at 8 warps: fine | same |

**Rounding caveat.** The compiler may fuse `dx*dx + dy*dy` into an FMA, which changes the
last bit of some distances relative to PyTorch. That can only flip the answer for
near-exact ties, which are vanishingly rare with random coordinates. If LeetGPU reports
a single mismatch, compute `d` in exactly the reference's order.

## Benchmark and profile

```bash
python docs/leetgpu/12_nearest_neighbor.py
ncu --kernel-name regex:nearest --launch-count 1 --section SpeedOfLight --section ComputeWorkloadAnalysis python docs/leetgpu/12_nearest_neighbor.py
```

Expect SM throughput high and DRAM low, the opposite of every kernel before this one.

## Scaling up

This is the bridge to **molecular neighbour lists**:

- N = 1M atoms: $O(N^2)$ = 10^12 pairs, too slow. Use a **cell list / uniform grid**.
  Bin atoms into cells of size ≥ cutoff (sort by cell id), then compare each atom only
  with the 27 neighbouring cells: $O(N)$.
- k-nearest instead of 1-nearest: keep a small sorted top-k per query in registers.
- Periodic boxes: apply the minimum-image convention to dx, dy, dz before squaring.
- **SoA layout** (separate x, y, z arrays) makes the j-tile loads fully coalesced. AoS
  with stride 3 wastes 2/3 of each sector. That is OK here thanks to caching, but it
  matters at scale.

## Watch and read

- [GPU MODE Lecture 4: Compute and Memory Basics](https://www.youtube.com/watch?v=lTmYrKwjSOU)
- [GPU Gems 3, Chapter 31: Fast N-Body Simulation with CUDA](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-31-fast-n-body-simulation-cuda): the same all-pairs tiling
