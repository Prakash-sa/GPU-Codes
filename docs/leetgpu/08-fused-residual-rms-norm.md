# 08. Fused residual add + RMS norm

| | |
|---|---|
| **Question** | *Fused Residual Add and RMS Norm* (medium) on [LeetGPU](https://leetgpu.com/challenges) · [official spec and reference](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/83_fused_residual_add_rms_norm) |
| **Solution** | [08_fused_residual_rms_norm.py](https://github.com/Prakash-sa/GPU-Codes/blob/main/docs/leetgpu/08_fused_residual_rms_norm.py) (Triton) · [08_fused_residual_rms_norm.cu](https://github.com/Prakash-sa/GPU-Codes/blob/main/docs/leetgpu/08_fused_residual_rms_norm.cu) (CUDA C++) |
| **Task** | $z = x + r$, $\text{out} = z / \sqrt{\text{mean}(z^2)+\epsilon} \cdot w$ per row; N, C ≤ 65,536; perf 4096 × 4096; tolerance 1e-5 |
| **Bound by** | memory bandwidth |

## Idea

This is the most "interview-shaped" problem in the set: a chain of cheap elementwise ops
and one reduction, which PyTorch eager runs as ~5 kernels. Each kernel reads and writes
full tensors. **Fusion** cuts traffic to the minimum: read `x` and `residual` once, write
`out` once.

![Bytes per element, fused vs unfused](../assets/img/leetgpu/fusion-traffic.png)

Two code paths, chosen at compile time with a `constexpr`:

- **ONE_PASS** (C ≤ 8192): the whole row is in registers. Load, reduce, scale, store.
- **two-pass** (larger C, up to 65,536): loop over the row accumulating $\sum z^2$, then
  loop again to write. The second read of a ≤ 512 KB row mostly hits L2.

Row offsets are int64. N × C can be 65,536², which is 2^32 and overflows int32.

## Code

=== "Triton"

    ```python
    --8<-- "docs/leetgpu/08_fused_residual_rms_norm.py"
    ```

=== "CUDA C++"

    ```cuda
    --8<-- "docs/leetgpu/08_fused_residual_rms_norm.cu"
    ```

    One block per row. For C ≤ 4096, z = x + residual stays in registers (16 per thread)
    and is written once after the block-wide sum of z². Longer rows recompute z from x and
    residual in the second pass (L2 hits).

## Performance

Fused bytes = $3 \cdot 4 \cdot 4096^2$ = 201 MB.

| | T4 | H100 SXM |
|---|---|---|
| Roofline floor, fused | 0.63 ms | 0.06 ms |
| Eager PyTorch (≈ 40 B/elem + 5 launches) | ~3× slower | ~3× slower |

The speedup comes from bytes, not FLOPs. Say it that way in an interview: "eager moves
~40 bytes per element, fused moves 12, so ~3× is the expected ceiling."

## Where the data lives

Levels as in [Where the data lives](memory-hierarchy.md). DRAM is GDDR6 on the T4 and HBM3 on the H100.

| Level | In this kernel |
|---|---|
| Registers | ONE_PASS (C ≤ 8192): the whole row of `z = x + r` stays in registers between the sum of squares and the output. The intermediate `z` is never written to DRAM |
| Spills → DRAM | the reason for the 8,192 cut-off. A longer row in registers would spill to local memory, which is DRAM. The two-pass path keeps only a 4,096-wide chunk in registers |
| L2 | two-pass: the second read of the row (≤ 512 KB for x and r) is meant to hit L2. That holds while the rows in flight fit: likely on an H100 (50 MB), not guaranteed on a T4 (4 MB) |
| DRAM | read x and r, write out: 201 MB at 4096 × 4096. The unfused version also writes and re-reads `z` |

This is fusion seen through the hierarchy: the intermediate moves from DRAM to registers.

## T4 vs H100

| | T4 | H100 |
|---|---|---|
| `num_warps` (C = 4096) | 8 | 8–16 |
| `MAX_ONE_PASS` | 8192 fp32 = 32 KB of registers per program. On T4 that limits residency; consider 4096 | 8192 is fine |
| Real models | store `z` too (the next residual needs it): +4 B/elem | same |

## Benchmark and profile

```bash
python docs/leetgpu/08_fused_residual_rms_norm.py
PROFILE=1 python docs/leetgpu/08_fused_residual_rms_norm.py
```

The PyTorch reference in the profile shows `add`, `pow`, `mean`, `add/sqrt`, `div`,
`mul`, all with their own launches. The Triton version shows one kernel. That
before/after table is the evidence to present.

## Scaling up

- Write the residual sum back (`z`) in the same kernel. Real transformer blocks reuse it.
- **BF16 storage, FP32 math**: halves the bytes. Load with `.to(tl.float32)` and store
  back as bf16. That is the next 2× (H100 only: T4 has no BF16, so use FP16 there).
- **Backward** of this op is page 19.
- With `torch.compile`, Inductor produces a similar fused kernel automatically. Compare
  against it, not only against eager.

## Watch and read

- [GPU MODE Lecture 28: Liger Kernel](https://www.youtube.com/watch?v=gWble4FreV4): exactly this kind of fused norm in production training
- GPU MODE Lecture 18: Fused Kernels: [slides and code](https://github.com/gpu-mode/lectures/tree/main/lecture_018). The video is on the [GPU MODE channel](https://www.youtube.com/@GPUMODE).
- Notes: [Fusion](../notes/09-optimization-strategies.md#fusion)
