# 07. Layer normalization

| | |
|---|---|
| **Question** | *Layer Normalization* (medium) on [LeetGPU](https://leetgpu.com/challenges) · [official spec and reference](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/113_layer_normalization) |
| **Solution** | [07_layer_norm.py](07_layer_norm.py) |
| **Task** | per row of `N × C`: $(x-\mu)/\sqrt{\sigma^2+\epsilon}\cdot w + b$, biased variance; N ≤ 65,536, C ≤ 4096; perf 65536 × 512; tolerance 1e-4 |
| **Bound by** | memory bandwidth |

## Idea

Rows are short (512 floats = 2 KB). One program per row would launch 65,536 tiny
programs, each with very little work. Instead each program normalizes **BLOCK_M rows**
from a `[BLOCK_M, BLOCK_C]` register tile (~4K elements):

- read x once; the mean and variance live in registers; write y once. That is
  8 bytes/element, the minimum;
- **centred variance**: compute `d = x - mean`, then `mean(d²)`. The one-pass formula
  $E[x^2]-E[x]^2$ cancels catastrophically when $|\mu| \gg \sigma$. The test adds a
  +1000 offset to catch this;
- zero the padded columns *before* squaring (`tl.where(mask, v - mean, 0)`), or they
  add $\mu^2$ to the variance.

## Code

```python
--8<-- "docs/leetgpu/07_layer_norm.py"
```

## Performance

Bytes = $2 \cdot 4 \cdot 65536 \cdot 512$ = 268 MB.

| | T4 | H100 SXM |
|---|---|---|
| Roofline floor | 0.84 ms | 0.08 ms |
| `torch.nn.functional.layer_norm` | already a fused CUDA kernel. Expect parity to modest wins, not 5× | same |

## T4 vs H100

| | T4 | H100 |
|---|---|---|
| Elements per program | 4096 (BLOCK_M = 8 rows of 512) | try 8192 (16 rows) with `num_warps=8`: more loads in flight per SM |
| C = 4096 | BLOCK_M = 1, 4096 floats per program, fine | fine; consider `num_warps=16` |

Register pressure: a tile of 4096 fp32 values × (x, d, output) with 8 warps is ~48
registers per thread for data. Comfortable on both.

## Benchmark and profile

```bash
python docs/leetgpu/07_layer_norm.py
ncu --kernel-name regex:layernorm --launch-count 1 --section SpeedOfLight --section Occupancy \
    python docs/leetgpu/07_layer_norm.py
```

Expect DRAM throughput > 80%. If it is lower, check *Occupancy* (registers) and the
grid size (65536 / 8 = 8192 programs is plenty).

## Scaling up

- C up to 65K (some LLM hidden sizes): a row no longer fits in registers. Loop over
  column chunks with a **two-pass** or **Welford** online mean/variance (see 08 and 17
  for the chunked pattern).
- `N × C > 2^31` needs int64 row offsets. The kernel already casts `rows` to int64.
- In a model, fuse what is around it: residual add before, the next GEMM after (17).

## Watch and read

- [GPU MODE Lecture 8: CUDA Performance Checklist](https://www.youtube.com/watch?v=SGhfUhlowB4)
- [GPU MODE Lecture 28: Liger Kernel](https://www.youtube.com/watch?v=gWble4FreV4)
- [Triton tutorial 05: Layer normalization](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)
