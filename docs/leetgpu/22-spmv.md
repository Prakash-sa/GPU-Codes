# 22. Sparse matrix-vector multiplication

| | |
|---|---|
| **Question** | *[Sparse Matrix-Vector Multiplication](https://leetgpu.com/challenges/sparse-matrix-vector-multiplication)* (medium) on LeetGPU · [official spec and reference](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/18_sparse_matrix_vector_multiplication) |
| **Solution** | [22_spmv.py](https://github.com/Prakash-sa/GPU-Codes/blob/main/docs/leetgpu/22_spmv.py) (Triton) · [22_spmv.cu](https://github.com/Prakash-sa/GPU-Codes/blob/main/docs/leetgpu/22_spmv.cu) (CUDA C++) |
| **Task** | $y = A x$, float32. A is M × N with 60–70% zeros, **stored dense** and row-major; `nnz` is given; M, N ≤ 10,000; perf M = 1,000, N = 10,000 (3.5M non-zeros); tolerance 1e-3 |
| **Bound by** | memory bandwidth (0.5 FLOP per byte) |

## Idea: count the bytes before choosing a format

The name says "sparse", but the input says dense: A arrives as a full `M × N` array. To
know which entries are zero, a kernel has to read every one of them. Reading A is
therefore unavoidable, and at 0.5 FLOP per byte it is also the entire cost:

| | Dense GEMV (`solve`) | Convert to CSR, then SpMV (`solve_csr`) |
|---|---|---|
| Read A | 40 MB, once | 40 MB, once (to find the non-zeros) |
| Extra traffic | none | write the non-zero indices and values, then read them back (8 bytes per non-zero ≈ 28 MB each way), plus a scan for the row pointers |
| x | 40 KB, from L2/L1 for every row | gathered by column index: scattered, but still cached |
| Launches | 1 | several (nonzero, bincount, cumsum, gather, SpMV) |

Skipping a zero saves one FMA, and FMAs are free in a memory-bound kernel. It saves no
bytes, because the zero has already been read. So the fast answer is a **dense GEMV that
streams A once**. CSR pays off when the matrix *arrives* in CSR and is reused across
many products (iterative solvers), or when it is far sparser than 35% dense.

The same reasoning applied to the matrix-matrix version is on [15. SpMM](15-spmm.md). There
the dense path wins for a different reason: tensor cores.

## Code

=== "Triton"

    ```python
    --8<-- "docs/leetgpu/22_spmv.py"
    ```

=== "CUDA C++"

    ```cuda
    --8<-- "docs/leetgpu/22_spmv.cu"
    ```

    Two shapes of the same kernel, chosen by row length. For the perf case (N = 10,000)
    each row gets a 256-thread block: float4 loads of A (every row starts 16-byte aligned
    when N % 4 == 0), x through `__ldg`, then a [warp-shuffle](../notes/13-warp-primitives.md)
    plus shared-memory block sum. Short rows (N ≤ 2048) get one warp each, 8 rows per
    block, so a 20-element row doesn't leave 236 threads idle. The reduction order is
    fixed, so results are bit-identical run to run.

## Performance

Bytes = $4(MN + N + M)$ = 40.0 MB at the perf size; FLOPs = $2MN$ = 20 MFLOP.

| | T4 | H100 SXM |
|---|---|---|
| Roofline floor (40 MB) | 125 µs | 12 µs |
| Programs / blocks | 1,000 (25 per SM) | 1,000 (7.6 per SM) |
| Launch overhead | negligible vs 125 µs | a visible fraction of 12 µs |
| CSR path | ≥ 2× the bytes, plus 5+ launches | same |

## Where the data lives

Levels as in [Where the data lives](memory-hierarchy.md). DRAM is GDDR6 on the T4 and HBM3 on the H100.

| Level | In this kernel |
|---|---|
| Registers | the running partial sums (a BLOCK_N vector per program in Triton; one float per thread in CUDA) |
| Shared memory | only the final cross-warp sum |
| L1 / L2 | **x**: 40 KB, read by all 1,000 rows. It stays in L2 (and largely L1), so it costs DRAM bandwidth once. The Triton version marks A loads `evict_first` and x loads `evict_last` to protect it |
| DRAM | A, streamed once: 40 MB. This is the kernel's runtime |

## T4 vs H100

The code is the same; only the bytes in flight differ. An H100 needs about 10× the
bytes in flight of a T4 to reach its bandwidth. If `ncu` shows DRAM throughput well
under 80% on an H100, unroll the loop (two float4s per thread per iteration), or give
each row two blocks and add the halves. With only 1,000 rows, the block count is near
the bottom of what keeps 132 SMs busy.

## Benchmark and profile

```bash
python docs/leetgpu/22_spmv.py           # dense GEMV vs CSR (incl. conversion) vs A @ x
PROFILE=1 python docs/leetgpu/22_spmv.py
python tests/cuda/run_tests.py --gpu 22_spmv
```

The benchmark prints GB/s for the dense kernel; compare it with the GPU's peak. The CSR
line includes the conversion, which is the honest comparison for a matrix used once.
Time the CSR kernel alone too: it shows what you would get if A arrived in CSR.

## Scaling up

- **Genuinely sparse matrices** (FEM, graphs, ≪ 1% dense) arrive in CSR. Use one warp
  per row for short rows (CSR-vector), and split long rows across warps. Power-law rows
  (a few rows with millions of entries) need merge-based or row-splitting schemes for
  load balance.
- **Many vectors at once** (`A @ X` with X of shape N × k): that is SpMM, where A's bytes
  are amortized over k columns. See [15](15-spmm.md).
- **Repeated products** with the same A (Krylov solvers, PageRank): convert to CSR, or a
  blocked format such as BSR, once, and amortize the conversion.
- In a CFD solver, the "matrix" is usually a stencil, and the fastest SpMV never stores
  it at all. See [stencils](../notes/12-stencils-and-discretization.md).

## Watch and read

- [GPU MODE Lecture 11: Sparsity](https://www.youtube.com/watch?v=mGDnOLcfE8g)
- [Mark Harris, *Optimizing Parallel Reduction in CUDA*](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf): the per-row sum is this reduction
- Notes: [Warp primitives](../notes/13-warp-primitives.md), [Roofline](../notes/03-performance.md)
