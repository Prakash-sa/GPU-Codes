# 11. Scatter add

| | |
|---|---|
| **Question** | *Scatter Add* (medium) on [LeetGPU](https://leetgpu.com/challenges) · [official spec and reference](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/125_scatter_add) |
| **Solution** | [11_scatter_add.py](11_scatter_add.py) |
| **Task** | `output[index[i], :] += src[i, :]`; src N × D float32, index int32; perf N = 1M, D = 128, M = 50K; tolerance 1e-4 |
| **Bound by** | bandwidth (reading src), plus atomic contention when many rows hit the same target |

## Idea

This is **message aggregation in a graph neural network**: every edge (source row) adds
its message to a node (output row). Several sources target the same node, so writes
**collide**.

![Scatter add collisions](../assets/img/leetgpu/scatter-add.png)

Two solutions:

| | `solve()`: atomics | `solve_sorted()`: sort + segment sum |
|---|---|---|
| How | each program loads `BLOCK_N` rows × `BLOCK_D` cols and issues `tl.atomic_add` per element | sort rows by target (once), build `row_ptr`; one program per output row sums its run in order |
| Passes over src | 1 | 1 (plus the sort) |
| Deterministic | **no**: add order varies between runs | **yes**: bitwise reproducible |
| Best when | the index changes every call | the graph is static and reused (sort once, amortize) |

Details in the atomic kernel:

- rows are contiguous along D, so 32 lanes add to 32 consecutive floats of the same
  target row. The atomics coalesce into a few L2 transactions;
- `sem="relaxed"`: only atomicity is needed, not ordering, so it is cheaper than the
  default `acq_rel`;
- `output.zero_()` first. LeetGPU's output buffer may hold stale data, and the test
  pre-fills it with 123.0 to check this.

## Code

```python
--8<-- "docs/leetgpu/11_scatter_add.py"
```

## Performance

Bytes ≈ read src (512 MB) + index (4 MB) + output read-modify-write in L2 (~26 MB of
targets) ≈ 540 MB.

| | T4 | H100 SXM |
|---|---|---|
| Floor | 1.7 ms | 0.16 ms |
| Contention | M = 50K targets, ~20 adds per target: low | same |

The output (50K × 128 × 4 = 25.6 MB) fits in H100's 50 MB L2, so atomics resolve in L2.
On T4 (4 MB of L2) they spill to DRAM, and atomics cost more there.

## T4 vs H100

| | T4 | H100 |
|---|---|---|
| Output in L2? | no (4 MB): atomic traffic goes to DRAM | yes (50 MB) |
| Tuning | BLOCK_N × BLOCK_D = 4096 with 4 warps | try 8192 with 8 warps |
| Hot targets (power-law graphs) | pre-aggregate inside the program: sum rows with equal targets in registers before one atomic | same |

## Benchmark and profile

```bash
python docs/leetgpu/11_scatter_add.py     # atomics vs sorted vs torch index_add_
ncu --kernel-name regex:scatter_add --launch-count 1 --section MemoryWorkloadAnalysis python docs/leetgpu/11_scatter_add.py
```

In ncu, look at **L2 atomic throughput** and DRAM bytes. Then rerun with a skewed index
(`index = torch.randint(0, 100, ...)`, so 10K adds per target) to see contention.

## Scaling up

- **Molecular and GNN models**: the scatter is usually fused with the message
  computation (gather node features, compute the edge MLP or distance-based weight,
  scatter-add) so per-edge messages never hit DRAM.
- **Determinism for science**: reproducible training/MD runs → use the sorted variant
  (or CSR from the neighbour list) and make the sort part of graph construction.
- **Low precision**: BF16 atomics on H100 are fast but accumulate badly. Accumulate in
  FP32.
- Multi-GPU: partition nodes; edges crossing partitions need a halo-style exchange of
  messages.

## Watch and read

- [GPU MODE Lecture 11: Sparsity](https://www.youtube.com/watch?v=mGDnOLcfE8g): irregular access and sparse formats
- [GPU MODE Lecture 8: CUDA Performance Checklist](https://www.youtube.com/watch?v=SGhfUhlowB4): atomics section
- Notes: [Atomics and reductions](../notes/09-optimization-strategies.md#atomics-and-reductions)
