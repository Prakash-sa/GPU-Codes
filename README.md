# GPU Codes

Notes on GPU programming (CUDA, memory, performance, multi-GPU, profiling, Python
GPU stacks) and a GPU hardware reference sheet.

**Read online:** <https://prakash-sa.github.io/GPU-Codes/>

## Contents

| Page | What it covers |
|------|----------------|
| [GPU spec sheets](docs/gpu-spec-sheets.md) | T4, L40S, RTX PRO 6000 Blackwell, A100, H100 side by side ([interactive version](https://claude.ai/artifact/TMvw41QWkQS1HHFvvfZArN)) |
| [1. Execution model](docs/notes/01-execution-model.md) | Threads, warps, blocks, grids, SMs, divergence |
| [2. Memory hierarchy](docs/notes/02-memory-hierarchy.md) | Registers, shared, L1/L2, global, constant, unified memory |
| [3. Performance optimization](docs/notes/03-performance.md) | Roofline, coalescing, occupancy, bank conflicts, tiling |
| [4. Streams and concurrency](docs/notes/04-streams.md) | Streams, events, async copies, pinned memory, CUDA Graphs |
| [5. Multi-GPU and NCCL](docs/notes/05-multi-gpu.md) | P2P, NVLink vs PCIe, NCCL collectives, CUDA-aware MPI |
| [6. Profiling and debugging](docs/notes/06-profiling.md) | Nsight Systems, Nsight Compute, NVTX, compute-sanitizer |
| [7. GPU from Python](docs/notes/07-python.md) | CuPy, Numba, Triton, PyTorch |
| [8. Programming with Triton](docs/notes/08-triton.md) | Triton kernels: vector add, fused softmax, stencils, autotuned matmul, debugging |
| [Cheat sheet](docs/notes/cheatsheet.md) | Quick reference of APIs, limits and formulas |

## Building locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdocs serve          # http://127.0.0.1:8000
```

## Publishing to GitHub Pages

`.github/workflows/pages.yml` builds the site with `mkdocs build --strict` and
deploys it on every push to `main`. You only need to turn it on once: in the
repository on GitHub, go to **Settings → Pages → Build and deployment → Source**
and choose **GitHub Actions**.
