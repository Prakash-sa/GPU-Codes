# GPU programming notes

Working notes on programming NVIDIA GPUs, mostly CUDA C++ with some Python. Each page
is short, with the mental model first and then the practical rules.

| # | Topic | Key question |
|---|-------|--------------|
| 1 | [Execution model](01-execution-model.md) | How does my kernel actually run? |
| 2 | [Memory hierarchy](02-memory-hierarchy.md) | Where does my data live, and how fast is it? |
| 3 | [Performance optimization](03-performance.md) | Am I bound by memory or by compute, and what do I fix first? |
| 4 | [Streams and concurrency](04-streams.md) | How do I overlap copies, kernels and the host? |
| 5 | [Multi-GPU and NCCL](05-multi-gpu.md) | How do I scale past one card? |
| 6 | [Profiling and debugging](06-profiling.md) | How do I find out what is really slow or broken? |
| 7 | [GPU from Python](07-python.md) | CuPy, Numba, Triton or PyTorch: which one? |
| 8 | [Programming with Triton](08-triton.md) | How do I write fused, tiled kernels in Python? |
| 9 | [Optimization strategies](09-optimization-strategies.md) | What is my bottleneck, and which fix applies? |
| — | [Cheat sheet](cheatsheet.md) | Quick lookup |

!!! tip "The one rule"
    Most scientific kernels (stencils, LBM, SpMV, statevector updates) are
    **memory-bandwidth bound**. Before optimizing arithmetic, count bytes moved per
    step and compare against the card's peak bandwidth
    (see [spec sheets](../gpu-spec-sheets.md)).
