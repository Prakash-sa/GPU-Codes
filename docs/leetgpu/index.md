# LeetGPU solutions in Triton

Optimized Triton solutions for 20 [LeetGPU](https://leetgpu.com/challenges) challenges, in
the order that builds up the most useful GPU skills: bandwidth basics, then reductions,
fusion, matmul, irregular access, scientific kernels, and finally attention and backward
passes.

Each challenge has two files:

- **`NN_name.py`**: the kernels and a `solve()` with LeetGPU's exact signature. Paste the
  file's top half into LeetGPU unchanged. Its `__main__` block tests awkward sizes and
  benchmarks against PyTorch.
- **`NN-name.md`**: the explanation. It covers the idea (with a figure where it helps), the
  code, what bounds the kernel, where its data lives (registers, shared memory, L2, DRAM),
  changes for T4 vs H100, how to benchmark and profile it,
  what changes at larger scale, and videos and reading.

Start with **[Benchmark, profile, scale](benchmarking.md)**: it is the workflow every page
uses.

**[Where the data lives](memory-hierarchy.md)** explains registers, shared memory/L1, L2
and DRAM (HBM). It shows how Triton code maps onto them. Every page has a section
saying where its own data sits.

## The problems

| # | Challenge | Pattern you learn | Bound by | Figure |
|---|---|---|---|---|
| [01](01-vector-add.md) | Vector Addition | masks, launch grid, coalescing | bandwidth | |
| [02](02-matrix-transpose.md) | Matrix Transpose | 2-D tiles, `tl.trans` through shared memory | bandwidth | GIF |
| [03](03-reduction.md) | Reduction | persistent grid, deterministic two-pass | bandwidth | GIF |
| [04](04-dot-product.md) | Dot Product | fused multiply + reduction | bandwidth | |
| [05](05-softmax.md) | Softmax (1-D) | cross-program max/sum merge | launch / bandwidth | PNG |
| [06](06-rms-norm.md) | RMS Normalization (1-D) | global statistic then elementwise | launch | |
| [07](07-layer-norm.md) | Layer Normalization | multi-row tiles, centred variance | bandwidth | |
| [08](08-fused-residual-rms-norm.md) | Fused Residual Add + RMS Norm | fusion, int64 offsets, long rows | bandwidth | PNG |
| [09](09-gemm.md) | GEMM (FP16) | `tl.dot`, grouped order, autotune, epilogue | compute | PNG + GIF |
| [10](10-fp16-batched-matmul.md) | FP16 Batched MatMul | batching in the grid | compute / latency | |
| [11](11-scatter-add.md) | Scatter Add | atomics vs sort-based determinism | bandwidth / atomics | PNG |
| [12](12-nearest-neighbor.md) | Nearest Neighbor | all-pairs tiling, exact argmin ties | compute | GIF |
| [13](13-n-body-force.md) | N-body Force | all-pairs tiling, `rsqrt` math | compute | GIF |
| [14](14-jacobi-stencil-2d.md) | 2D Jacobi Stencil | stencil, boundaries, cache reuse | bandwidth | PNG |
| [15](15-spmm.md) | Sparse x Dense MatMul | dense vs CSR, `tf32x3` precision | compute | PNG |
| [16](16-segmented-prefix-sum.md) | Segmented Prefix Sum | `tl.associative_scan`, 3-phase scan | bandwidth | PNG |
| [17](17-fused-layernorm-linear-gelu.md) | Fused LayerNorm Linear GELU | prologue/epilogue fusion around a GEMM | compute | PNG |
| [18](18-softmax-attention.md) | Softmax Attention | online softmax (FlashAttention) | latency / compute | GIF |
| [19](19-rms-norm-backward.md) | RMS Norm Backward | row + column reductions, no atomics | bandwidth | PNG |
| [20](20-layer-norm-backward.md) | Layer Norm Backward | same, three outputs | bandwidth | PNG |

## Running the solutions

```bash
pip install torch triton
python docs/leetgpu/09_gemm.py                      # tests + benchmark on a CUDA GPU
PROFILE=1 python docs/leetgpu/09_gemm.py            # + torch.profiler table and trace
TRITON_INTERPRET=1 python docs/leetgpu/09_gemm.py   # no GPU: kernels run on the CPU
```

!!! warning "On a Mac (or any machine without an NVIDIA GPU)"
    Triton has no macOS wheels and `ncu`/`nsys` need an NVIDIA GPU, so `pip install triton`
    fails and nothing above runs natively. Two options:

    - **Check correctness locally** with the CPU interpreter inside a Linux container:

        ```bash
        docker build -t triton-interp - <<'EOF'
        FROM python:3.11-slim
        RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
         && pip install --no-cache-dir triton numpy
        EOF
        docker run --rm --platform linux/amd64 -e TRITON_INTERPRET=1 -v "$PWD":/w -w /w \
            triton-interp python docs/leetgpu/01_vector_add.py
        ```

        This prints the correctness checks and skips the benchmark.
    - **Benchmark and profile** on a real GPU: a Colab T4 runtime, LeetGPU Pro, or a cloud
      VM. On Colab, `!ncu` is usually blocked; use `PROFILE=1` (torch.profiler) there.

!!! info "What has been verified"
    Every solution passes its correctness tests, which include LeetGPU's tolerances and
    awkward sizes (1, non-powers of two, ragged tiles, ties and long segments). They were
    run under Triton's CPU interpreter (`TRITON_INTERPRET=1`). The interpreter checks
    indexing, masking and numerics, but not GPU-specific behaviour such as
    `num_warps`, pipelining, tensor-core precision or speed. Run each file on a real T4
    or H100 (LeetGPU Pro, Colab, a cloud VM) before quoting any timing. The benchmark
    numbers on each page are *roofline floors* computed from bytes and FLOPs, not
    measurements.

!!! note "Links to the questions"
    LeetGPU is a single-page app, so deep links can't be checked from outside. Each page
    links the challenge list (search the exact title shown) and the challenge's
    **official spec and reference implementation** in
    [AlphaGPU/leetgpu-challenges](https://github.com/AlphaGPU/leetgpu-challenges), which
    is where the signatures, tolerances and performance sizes come from.

## Videos used across the pages

All links were checked to exist. Watch with the code open and pause to write the kernel
yourself.

| Video | Covers | Used in |
|---|---|---|
| [CUDA Programming Course: High-Performance Computing with GPUs](https://www.youtube.com/watch?v=86FAWCzIe_4) (freeCodeCamp) | threads, memory, matmul, Triton chapter | 01, 02, 09 |
| [GPU MODE Lecture 1: How to profile CUDA kernels in PyTorch](https://www.youtube.com/watch?v=LuhJEEJQgUM) | profiler, ncu from Python | all (benchmarking) |
| [GPU MODE Lecture 3: Getting Started With CUDA for Python Programmers](https://www.youtube.com/watch?v=4sgKnKbR-WE) | first kernels from Python | 01 |
| [GPU MODE Lecture 4: Compute and Memory Basics](https://www.youtube.com/watch?v=lTmYrKwjSOU) | memory hierarchy, coalescing | 02, 14 |
| [GPU MODE Lecture 8: CUDA Performance Checklist](https://www.youtube.com/watch?v=SGhfUhlowB4) | coalescing, occupancy, tiling, minimizing traffic | 02, 03, 07, 14 |
| [GPU MODE Lecture 11: Sparsity](https://www.youtube.com/watch?v=mGDnOLcfE8g) | sparse formats, when sparsity pays | 15 |
| [GPU MODE Lecture 12: Flash Attention](https://www.youtube.com/watch?v=zEuwuCTEf_0) | online softmax, tiling | 05, 18 |
| [GPU MODE Lecture 14: Practitioner's Guide to Triton](https://www.youtube.com/watch?v=DdTsX6DQk24) | Triton model, masks, debugging | all |
| [GPU MODE Lecture 16: On Hands Profiling](https://www.youtube.com/watch?v=SKV6kDk1s94) | nsys / ncu in practice | all (benchmarking) |
| [GPU MODE Lecture 20: Scan Algorithm](https://www.youtube.com/watch?v=ZKrWyEqqPVY) | parallel prefix sum | 16 |
| [GPU MODE Lecture 23: Tensor Cores](https://www.youtube.com/watch?v=hQ9GPnV0-50) | MMA, precisions, tile shapes | 09, 10, 15, 17 |
| [GPU MODE Lecture 28: Liger Kernel](https://www.youtube.com/watch?v=gWble4FreV4) | fused Triton kernels for training (norms, backward) | 07, 08, 19, 20 |
| [FlashAttention, Tri Dao (Stanford MedAI #54)](https://www.youtube.com/watch?v=FThvfkXWqtE) | IO-aware attention | 18 |

## Regenerating the figures

```bash
.venv/bin/python scripts/make_leetgpu_figures.py    # writes docs/assets/img/leetgpu/
```
