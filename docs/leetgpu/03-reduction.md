# 03. Reduction (sum)

| | |
|---|---|
| **Question** | *Reduction* (medium) on [LeetGPU](https://leetgpu.com/challenges) · [official spec and reference](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/4_reduction) |
| **Solution** | [03_reduction.py](https://github.com/Prakash-sa/GPU-Codes/blob/main/docs/leetgpu/03_reduction.py) |
| **Task** | `output[0] = sum(input)`, float32, N ≤ 100M; perf N = 4,194,304; tolerance 1e-5 vs a float64 reference |
| **Bound by** | bandwidth at large N; launch overhead at the perf size on H100 |

## Idea

A reduction across the whole array needs communication between programs. There are
two ways to do it:

1. **Atomics:** every program does `tl.atomic_add(out, partial)`. That is one kernel,
   but the order of float additions changes run to run, so the result is **not
   reproducible**, and the output must be zeroed first (another launch).
2. **Two passes (used here):** a **persistent** grid of P = 4 × SMs programs
   grid-strides through the array. Each program keeps a `BLOCK`-wide vector
   accumulator, so each lane adds only N / (P·BLOCK) numbers, which is good for
   accuracy. Each program writes one partial. A second, single-program kernel adds the P
   partials in **float64**. The same input always gives the same bits.

![Two-pass reduction](../assets/img/leetgpu/two-pass-reduction.gif)

## Code

```python
--8<-- "docs/leetgpu/03_reduction.py"
```

## Accuracy

Summing 4M values one at a time in FP32 has error growing like $O(N\varepsilon)$. Here the
error grows like ~(N / (P·BLOCK)) per lane plus a log-depth tree (`tl.sum`), and the
cross-program step runs in FP64. Inputs can be ±1000 with heavy cancellation, and
LeetGPU's reference is a float64 sum, so the precision matters.

## Performance

| | T4 | H100 SXM |
|---|---|---|
| Roofline floor (16.8 MB) | 52 µs | 5 µs |
| Launch overhead (2 launches) | small vs 52 µs | **comparable to the kernel itself** |

On H100 at this size, the second launch is a visible fraction of the time. Options are
CUDA Graphs, or a single kernel where the *last program to finish* does the final sum
(an atomic counter: "threadfence reduction"). The second keeps determinism and saves
one launch.

## Where the data lives

Levels as in [Where the data lives](memory-hierarchy.md). DRAM is GDDR6 on the T4 and HBM3 on the H100.

| Level | In this kernel |
|---|---|
| Registers | `acc`: a 4,096-wide fp32 vector, 16 per thread, carried through the whole grid-stride loop. Each element of x is added into a register and never written back |
| Shared memory | only the final `tl.sum`: warp shuffles, then one value per warp through shared memory |
| L2 | the P float64 partials written by kernel 1 are read by kernel 2 straight from L2 |
| DRAM | read x once: 16 MB. Everything else is a few KB |

The persistent loop is what keeps the running sum in registers. One program per block
would instead write one partial per 4,096 elements, so 1,024 partials, and need a
second reduction pass over them.

The 16 MB input fits in the H100's 50 MB L2. A timing loop that doesn't flush L2 then
measures L2 bandwidth, not DRAM. `do_bench` flushes it.

## T4 vs H100

| | T4 | H100 |
|---|---|---|
| P (programs) | 4 × 40 = 160 | 4 × 132 = 528 |
| BLOCK / num_warps | 4096 / 8 | 4096–8192 / 8–16: more loads in flight per SM |
| FP64 final sum | P adds only, negligible even at 1/32 FP64 rate | negligible |

`_num_sms()` reads the SM count at run time, so the same code adapts to both GPUs.

## Benchmark and profile

```bash
python docs/leetgpu/03_reduction.py
PROFILE=1 python docs/leetgpu/03_reduction.py   # 2 kernels: partial_sum (~all time) + final_sum (tiny)
```

In `ncu`, check that DRAM throughput is > 80% in `partial_sum_kernel`. If not, raise
BLOCK or P (not enough bytes in flight).

## Scaling up

- 100M elements: `start` and `offs` stay below 2^31, but above 2^31 elements switch
  the offsets to int64.
- **Multiple reductions** (per row, per segment): the same pattern with a 2-D grid. See
  07 (rows) and 16 (segments).
- **Multi-GPU:** reduce locally, then `torch.distributed.all_reduce` on the scalar. Use
  FP64, or Kahan summation, for the cross-GPU step if values cancel.

## Watch and read

- [GPU MODE Lecture 8: CUDA Performance Checklist](https://www.youtube.com/watch?v=SGhfUhlowB4)
- [Mark Harris, *Optimizing Parallel Reduction in CUDA*](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf): the seven classic optimization steps
- GPU MODE Lecture 9: Reductions: [slides and code](https://github.com/gpu-mode/lectures/tree/main/lecture_009). The video is on the [GPU MODE channel](https://www.youtube.com/@GPUMODE).
- Notes: [Atomics and reductions](../notes/09-optimization-strategies.md#atomics-and-reductions)
