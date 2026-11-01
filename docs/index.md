# GPU Codes

Notes on GPU programming and a hardware reference sheet for the NVIDIA GPUs I
have run workloads on.

## Start here

- **[GPU spec sheets](gpu-spec-sheets.md)**: T4, L40S, RTX PRO 6000 Blackwell,
  A100 and H100 side by side. The interactive version is
  [on Claude](https://claude.ai/artifact/TMvw41QWkQS1HHFvvfZArN).
- **[GPU programming notes](notes/index.md)**: CUDA from the execution model up to
  multi-GPU and profiling.
- **[Cheat sheet](notes/cheatsheet.md)**: one page of APIs, limits and formulas.
- **[Interview Q&A](interview-qa.md)**: GPU, ML-systems and multi-GPU interview questions with short answers.
- **[Before GPU Engineering](before-gpu/index.html)**: a 56-chapter, first-principles notebook on how GPU hardware works, from transistors to profiling. By Ranjan Yadav (MIT), mirrored here with a chapter navbar.
- **[LeetGPU solutions in Triton](leetgpu/index.md)**: 20 optimized, tested kernels with explanations, T4 vs H100 notes, profiling steps and videos.

## Reading order

1. [Execution model](notes/01-execution-model.md)
2. [Memory hierarchy](notes/02-memory-hierarchy.md)
3. [Performance optimization](notes/03-performance.md)
4. [Streams and concurrency](notes/04-streams.md)
5. [Multi-GPU and NCCL](notes/05-multi-gpu.md)
6. [Profiling and debugging](notes/06-profiling.md)
7. [GPU from Python](notes/07-python.md)
8. [Programming with Triton](notes/08-triton.md)
9. [Optimization strategies](notes/09-optimization-strategies.md)
10. [How GPUs talk to each other](notes/10-multi-gpu-communication.md)
