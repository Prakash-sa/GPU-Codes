# 14. 2-D Jacobi stencil

| | |
|---|---|
| **Question** | *2D Jacobi Stencil* (medium) on [LeetGPU](https://leetgpu.com/challenges) · [official spec and reference](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/69_jacobi_stencil_2d) |
| **Solution** | [14_jacobi_stencil_2d.py](https://github.com/Prakash-sa/GPU-Codes/blob/main/docs/leetgpu/14_jacobi_stencil_2d.py) (Triton) · [14_jacobi_stencil_2d.cu](https://github.com/Prakash-sa/GPU-Codes/blob/main/docs/leetgpu/14_jacobi_stencil_2d.cu) (CUDA C++) |
| **Task** | interior: average of 4 neighbours; boundary: copy; up to 16,384²; perf 8192²; tolerance 1e-5 |
| **Bound by** | memory bandwidth |

## Idea

![5-point stencil and a program tile](../assets/img/leetgpu/jacobi.png)

One program per `BLOCK_R × BLOCK_C` tile issues four shifted loads (up, down, left,
right) with an `interior` mask. The shifted loads overlap almost entirely: the "left"
tile is the "right" tile moved by one column. So after the first load they hit L1/L2,
and DRAM sees ~one read and one write per cell. Boundary cells take the `center` load
and are copied.

The tiles are wide along columns (16 × 128), so every row segment is a 512-byte
coalesced access.

## Code

=== "Triton"

    ```python
    --8<-- "docs/leetgpu/14_jacobi_stencil_2d.py"
    ```

=== "CUDA C++"

    ```cuda
    --8<-- "docs/leetgpu/14_jacobi_stencil_2d.cu"
    ```

    Explicit shared-memory tiling, the step the Triton version leaves to the caches: a 32
    × 8 block loads a 34 × 34 tile (32 × 32 plus a one-cell halo) once and computes four
    outputs per thread from shared memory.

## Performance

Bytes ≈ $2 \cdot 4 \cdot 8192^2$ = 537 MB.

| | T4 | H100 SXM |
|---|---|---|
| Floor | 1.7 ms | 0.16 ms |
| PyTorch slicing reference | clone + 4 slice reads + adds: 3–5× the traffic | same |

## Where the data lives

Levels as in [Where the data lives](memory-hierarchy.md). DRAM is GDDR6 on the T4 and HBM3 on the H100.

| Level | In this kernel |
|---|---|
| Registers | the four neighbour tiles (16 × 128 each) and the result |
| L1 | `left` and `right` are the centre row shifted by one: the same cache lines, so L1 hits. `up` and `down` are this tile's neighbouring rows, mostly loaded already |
| L2 | the halo rows above and below the tile were loaded by the neighbouring programs: L2 hits |
| DRAM | ideally each element read once and written once: 537 MB at 8192² |

Four loads per point look like 4× the traffic, but L1 and L2 absorb the repeats. Nsight
Compute's DRAM bytes should come out near 2 × 4 × rows × cols. A CUDA stencil would load
the tile and halo into `__shared__` explicitly. Triton has no shared-memory API and
relies on the caches.

## T4 vs H100

| | T4 | H100 |
|---|---|---|
| Tile | 16 × 128, 4 warps | 32 × 256, 8 warps (more bytes in flight) |
| Halo reuse | 4 MB of L2: neighbouring tiles' rows may be evicted, so explicit shared-memory tiling helps more | 50 MB of L2 holds many rows: implicit reuse works well |

## Benchmark and profile

```bash
python docs/leetgpu/14_jacobi_stencil_2d.py
ncu --kernel-name regex:jacobi --launch-count 1 --section MemoryWorkloadAnalysis python docs/leetgpu/14_jacobi_stencil_2d.py
```

Compare DRAM bytes against 537 MB. If they are ≫ 537 MB, the halo reloads are
missing in L2. Also check the L1 hit rate (should be high for the shifted loads).

## Scaling up

This is the pattern behind CFD and LBM solvers:

- **Many iterations**: a time loop of one-step kernels is bandwidth bound at 8 B/cell
  per step. **Temporal blocking** (several steps per tile with a wider halo, kept in
  shared memory) cuts DRAM passes. Ping-pong buffers avoid copies.
- **3-D stencils** (7- or 27-point, D3Q27 LBM): use 2.5-D blocking. March along z, keep
  three planes in registers or shared memory.
- **Convergence checks** need a residual reduction. Do it every k steps, and on the
  device, to avoid `.item()` syncs.
- **Multi-GPU**: domain decomposition with halo exchange, overlapping the exchange with
  the interior update. See [Multi-GPU: halos](../notes/05-multi-gpu.md).

## Watch and read

- [GPU MODE Lecture 4: Compute and Memory Basics](https://www.youtube.com/watch?v=lTmYrKwjSOU)
- [GPU MODE Lecture 8: CUDA Performance Checklist](https://www.youtube.com/watch?v=SGhfUhlowB4)
- Notes: [2-D Jacobi stencil in Triton](../notes/08-triton.md), [halo exchange](../notes/05-multi-gpu.md)
