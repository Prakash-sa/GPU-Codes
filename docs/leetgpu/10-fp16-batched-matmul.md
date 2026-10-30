# 10. FP16 batched matmul

| | |
|---|---|
| **Question** | *FP16 Batched Matrix Multiplication* (medium) on [LeetGPU](https://leetgpu.com/challenges) · [official spec and reference](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/57_fp16_batched_matmul) |
| **Solution** | [10_fp16_batched_matmul.py](https://github.com/Prakash-sa/GPU-Codes/blob/main/docs/leetgpu/10_fp16_batched_matmul.py) |
| **Task** | `C[b] = A[b] @ B[b]`, FP16; BATCH ≤ 128, M, N, K ≤ 1024; perf 32 × 256³; tolerance 0.05 |
| **Bound by** | latency / occupancy (each matmul is small) |

## Idea

One 256³ matmul is 64 tiles of 32² or 16 tiles of 64²: not enough programs for 132 SMs.
A Python loop of 32 launches makes it worse (32 × launch overhead, each under-filled).
Instead, **put the batch on grid axis 1**: one launch, `tiles × 32` programs running
together. Each program offsets its base pointers by `batch × M × K` and so on, in
**int64** (128 × 1024 × 1024 fits int32, but larger batches won't).

The inner loop is the same as [09. GEMM](09-gemm.md): FP16 tiles, `tl.dot`, FP32
accumulation, FP16 store.

## Code

```python
--8<-- "docs/leetgpu/10_fp16_batched_matmul.py"
```

## Performance

FLOPs = $2 \cdot 32 \cdot 256^3$ = 1.07 GFLOP; bytes = 12.6 MB.

| | T4 | H100 SXM |
|---|---|---|
| Floor | ~39 µs (memory: 12.6 MB at 320 GB/s; compute alone is 16 µs) | ~4 µs (memory: 12.6 MB at 3.35 TB/s) |
| Programs at 64×64 tiles | 512: 12.8 per SM ✓ | 512: 3.9 per SM, still OK |

## T4 vs H100

| | T4 | H100 |
|---|---|---|
| Good tile | 64×64×32, 4 warps, 2 stages | 64×64 or 128×64, 3–4 stages; small tiles keep more programs |
| Tiny matrices (M, N ≤ 32) | `BLOCK_M=16` config | same, and tensor-core efficiency drops |

Autotuning keys on (M, N, K), not BATCH. The best tile is a property of each matmul,
and the batch only multiplies the programs.

## Benchmark and profile

```bash
python docs/leetgpu/10_fp16_batched_matmul.py      # vs torch.bmm
PROFILE=1 python docs/leetgpu/10_fp16_batched_matmul.py
```

Also time the naive `for b in range(B): c[b] = a[b] @ b[b]`. The profile shows 32
launches with gaps, the clearest picture of launch overhead in this set.

## Scaling up

- **Variable-size batches** (MoE experts, graphs of different sizes): *grouped GEMM*. Pass
  an array of pointers and shapes, and let each program look up its problem (LeetGPU
  *Grouped GEMM*).
- Attention's $QK^T$ and $PV$ are batched matmuls over (batch × heads). Flash attention
  (18) fuses them.
- Many tiny matrices (3×3 rotations, small molecular blocks): tensor cores need
  dimensions ≥ 16. Use one program per *several* matrices with plain FMA math instead.

## Watch and read

- [GPU MODE Lecture 23: Tensor Cores](https://www.youtube.com/watch?v=hQ9GPnV0-50)
- [Triton tutorial 03: Matrix multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
