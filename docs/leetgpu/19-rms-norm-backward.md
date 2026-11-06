# 19. RMS norm backward

| | |
|---|---|
| **Question** | *RMS Norm Backward* (medium) on [LeetGPU](https://leetgpu.com/challenges) · [official spec and reference](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/142_rms_norm_backward) |
| **Solution** | [19_rms_norm_backward.py](https://github.com/Prakash-sa/GPU-Codes/blob/main/docs/leetgpu/19_rms_norm_backward.py) (Triton) · [19_rms_norm_backward.cu](https://github.com/Prakash-sa/GPU-Codes/blob/main/docs/leetgpu/19_rms_norm_backward.cu) (CUDA C++) |
| **Task** | given x, w, dy, compute dx and dw; N ≤ 65,536, C ≤ 16,384; perf 16384 × 4096; tolerance 1e-3 |
| **Bound by** | memory bandwidth |

## Idea

With $\hat x = x \cdot r$, $r = 1/\sqrt{\text{mean}(x^2)+\epsilon}$, and $g = dy \odot w$:

$$
dx = r\,\big(g - \hat x \cdot \text{mean}(g \odot \hat x)\big) \qquad
dw = \sum_{\text{rows}} dy \odot \hat x
$$

`dx` needs a reduction **along each row**. `dw` needs a reduction **down each column**,
across all rows. That is the whole difficulty:

![Row reductions for dx, column partials for dw](../assets/img/leetgpu/norm-backward.png)

- **Kernel 1 (persistent):** P = 2 × SMs programs. Program `p` processes rows
  `p, p+P, p+2P, …`. Each row is loaded once and gives `dx` from registers. Meanwhile
  `dy * xhat` is accumulated into a **C-wide register vector** across the program's
  rows. At the end each program stores one partial-dw row: a `[P, C]` buffer.
- **Kernel 2:** sum the P partial rows per column.

There are no atomics, so dw is bitwise reproducible. The extra traffic, P × C floats
(~2 MB), is negligible next to 800 MB.

## Code

=== "Triton"

    ```python
    --8<-- "docs/leetgpu/19_rms_norm_backward.py"
    ```

=== "CUDA C++"

    ```cuda
    --8<-- "docs/leetgpu/19_rms_norm_backward.cu"
    ```

    A persistent grid (2 × SM-count blocks) takes rows round-robin. Row reductions use
    shuffles plus shared memory; for the column reduction each thread owns its columns of
    the block's partial-dw row, so there are no atomics. A second kernel sums the partial
    rows in a fixed order.

## Performance

Bytes = read x, dy + write dx = $3 \cdot 4 \cdot 16384 \cdot 4096$ = 805 MB.

| | T4 | H100 SXM |
|---|---|---|
| Floor | 2.5 ms | 0.24 ms |
| PyTorch autograd | recomputes rstd, several elementwise kernels + a column `sum`: ~2–3× the bytes | same |

## Where the data lives

Levels as in [Where the data lives](memory-hierarchy.md). DRAM is GDDR6 on the T4 and HBM3 on the H100.

| Level | In this kernel |
|---|---|
| Registers | `w` (loaded once per program, reused for all its rows), the row of x and dy, and the `dw_acc` column accumulator. That is four C-wide vectors, 32 floats per thread at C = 4096 with 16 warps. At C = 16,384 it is 4× more, which is where spills start |
| Shared memory | the two row sums per row (`sum(x²)` and `sum(g · x̂)`) across warps |
| L2 | the `[P, C]` partials: 4.3 MB on an H100 (P = 264), 1.3 MB on a T4 (P = 80). Both fit in L2, so the column-sum kernel reads them from L2 |
| DRAM | read x and dy, write dx: 805 MB at 16384 × 4096 |

dw is a column sum over all rows. The persistent grid turns it into a register
accumulation per program plus one small L2-resident pass. Atomics into a C-wide array
would hit the same L2 lines from every program.

## T4 vs H100

| | T4 | H100 |
|---|---|---|
| P | 80 | 264 |
| `BLOCK_C` at C = 16,384 | 16K floats per tensor per program: **register spills likely**. Use a chunked two-pass row loop | same risk; check `n_spills` |
| `num_warps` | 16 for C ≥ 4096 | 16 |

Check with `k = rmsnorm_bwd_kernel[(P,)](...)` and then `print(k.n_regs, k.n_spills)`. If
it spills, chunk the row: pass 1 accumulates `sum(x²)` and `sum(g·x)` over chunks, and
pass 2 writes dx per chunk. The dw accumulator no longer fits in registers, so each
program adds into its own row of the `[P, C]` buffer chunk by chunk (no atomics needed,
because the row belongs to that program).

## Benchmark and profile

```bash
python docs/leetgpu/19_rms_norm_backward.py      # vs torch.autograd.grad of F.rms_norm
PROFILE=1 python docs/leetgpu/19_rms_norm_backward.py
```

## Scaling up

- **Fuse with the forward's saved state**: save `rstd` (N floats) in the forward pass to
  skip one reduction here (LayerNorm backward on page 20 gets mean/rstd as inputs).
- **Fuse with the residual-add backward** (the gradient flows to both branches).
- **BF16 activations / FP32 accumulators** halve the dominant traffic.
- For data-parallel training, dw is then all-reduced across GPUs. Keeping it FP32 and
  deterministic makes runs reproducible.

## Watch and read

- [GPU MODE Lecture 28: Liger Kernel](https://www.youtube.com/watch?v=gWble4FreV4): fused norm forward and backward kernels in Triton
- [Triton tutorial 05: Layer normalization](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html): its backward uses partial-dw buffers with locks, a different trade-off
