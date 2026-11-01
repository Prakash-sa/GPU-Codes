# Before GPU Engineering

*A first-principles climb from the absolute bottom of hardware to the top of GPU architecture.* A 56-chapter notebook by Ranjan Yadav: transistors, memory buses and warps first, then CUDA's execution model, the memory wall, and profiling. Read it before (or next to) the [GPU programming notes](../index.md); the chapters build on each other, so start at Chapter 1 and read in order.

<div id="bge-continue"></div>

!!! note "Source and license"
    From [Before GPU Engineering](https://github.com/BEFORE-GPU-ENGINEERING/BEFORE-GPU-ENGINEERING) by Ranjan Yadav, used under the [MIT license](LICENSE.txt). The upstream repository is still adding chapters (Phase 5 is in progress); `scripts/import_before_gpu.py` pulls new ones in.

## Phase 1

### Part 1: The Architecture

- [Chapter 1: Why CPUs Can't Have 10,000 Cores](architecture/chapter-1.md)
- [Chapter 2: Why a GPU multiplies matrices faster](architecture/chapter-2.md)
- [Chapter 3: What "parallelizable" actually means](architecture/chapter-3.md)

### Part 2: Feeding the Beast

- [Chapter 4: Why GPUs care so much about memory bandwidth](feeding-the-beast/chapter-4.md)
- [Chapter 5: Compute-bound vs. memory-bound](feeding-the-beast/chapter-5.md)
- [Chapter 6: The GPU memory hierarchy](feeding-the-beast/chapter-6.md)

### Part 3: Inside the Silicon

- [Chapter 7: What problem do warps solve?](inside-the-silicon/chapter-7.md)
- [Chapter 8: Why branching is bad on GPUs](inside-the-silicon/chapter-8.md)
- [Chapter 9: What is SIMT? (and how it differs from SIMD)](inside-the-silicon/chapter-9.md)
- [Chapter 10: Why occupancy matters](inside-the-silicon/chapter-10.md)

### Part 4: Putting it Together

- [Chapter 11: Why AI runs on GPUs but databases don't](putting-it-together/chapter-11.md)
- [Chapter 12: What is latency hiding](putting-it-together/chapter-12.md)
- [Chapter 13: What if a GPU had CPU-style caches](putting-it-together/chapter-13.md)
- [Chapter 14: The memory wall & HBM](putting-it-together/chapter-14.md)
- [Chapter 15: Designing an LLM training processor](putting-it-together/chapter-15.md)
## Phase 2

### The Translation Barrier

- [Chapter 16: Why C++ won't auto-compile to GPU](the-translation-barrier/chapter-16.md)
- [Chapter 17: A kernel is not a function](the-translation-barrier/chapter-17.md)
- [Chapter 18: Why thread indexing exists](the-translation-barrier/chapter-18.md)

### The Hardware Hierarchy

- [Chapter 19: Why threads group into blocks](the-hardware-hierarchy/chapter-19.md)
- [Chapter 20: Why shared memory is fast](the-hardware-hierarchy/chapter-20.md)
- [Chapter 21: Why RAM isn't visible to the GPU](the-hardware-hierarchy/chapter-21.md)
- [Chapter 22: Kernel launch overhead](the-hardware-hierarchy/chapter-22.md)

### Synchronization & Hazards

- [Chapter 23: What syncthreads protects](synchronization-hazards/chapter-23.md)
- [Chapter 24: Why race conditions happen](synchronization-hazards/chapter-24.md)
- [Chapter 25: What makes CUDA code "correct"](synchronization-hazards/chapter-25.md)

### Scale & Performance

- [Chapter 26: Why the GPU context switch is free but CPU's is slow](scale-and-performance/chapter-26.md)
- [Chapter 27: The step-by-step anatomy of a kernel launch](scale-and-performance/chapter-27.md)
- [Chapter 28: Why NVIDIA made CUDA asynchronous](scale-and-performance/chapter-28.md)
- [Chapter 29: Coalesced vs strided memory access](scale-and-performance/chapter-29.md)
- [Chapter 30: Why GPU programs actually slow down](scale-and-performance/chapter-30.md)
- [Chapter 31: Why matrix add scales but a linked list dies](scale-and-performance/chapter-31.md)
- [Chapter 32: When NOT to use a GPU — and the reduction tree](scale-and-performance/chapter-32.md)
## Phase 3

### The Memory Wall

- [Chapter 33: Your ALUs are idle. Memory is the real bottleneck.](the-memory-wall/chapter-33.md)
- [Chapter 34: What is arithmetic intensity? The currency of GPU optimization.](the-memory-wall/chapter-34.md)
- [Chapter 35: Two hard ceilings. One tells you to fix your math. The other says: fix your memory.](the-memory-wall/chapter-35.md)

### The Register Physics

- [Chapter 36: Why not build a GPU entirely from registers?](the-register-physics/chapter-36.md)
- [Chapter 37: Where should data live? The register-occupancy tradeoff.](the-register-physics/chapter-37.md)
- [Chapter 38: What is occupancy, really? It's a ratio, not a count — and 100% is not the goal.](the-register-physics/chapter-38.md)
- [Chapter 39: Why higher occupancy can make performance worse.](the-register-physics/chapter-39.md)

### Register Spilling

- [Chapter 40: What is register spilling? The cliff between 1 ms and 50 ms.](register-spilling/chapter-40.md)
- [Chapter 41: Splitting the giant](register-spilling/chapter-41.md)
- [Chapter 42: When spilling is acceptable: and when it destroys performance.](register-spilling/chapter-42.md)

### Global Memory & Coalescing

- [Chapter 43: Why global memory latency is 350 to 1,000 cycles: the physics of distance.](global-memory-and-coalescing/chapter-43.md)
- [Chapter 44: What is memory coalescing? 32 threads, 1 transaction vs 32 transactions.](global-memory-and-coalescing/chapter-44.md)
- [Chapter 45: DRAM row-buffer thrashing: why scattered access is structurally broken.](global-memory-and-coalescing/chapter-45.md)

### Shared Memory & Tiling

- [Chapter 46: What is predication? Why GPUs refuse to branch.](shared-memory-and-tiling/chapter-46.md)
- [Chapter 47: Compare-and-swap: The universal atomic primitive.](shared-memory-and-tiling/chapter-47.md)
- [Chapter 48: Register tiling: The tile inside the tile.](shared-memory-and-tiling/chapter-48.md)

### GEMM & The Synthesis

- [Chapter 49: Wave quantization: the hidden cost of the last wave](gemm-and-the-synthesis/chapter-49.md)
- [Chapter 50: The traffic equation: throughput, latency, and concurrency](gemm-and-the-synthesis/chapter-50.md)
- [Chapter 51: The second lever: hiding latency within a single thread](gemm-and-the-synthesis/chapter-51.md)
- [Chapter 52: Measure, don't guess: the top-down profiling discipline](gemm-and-the-synthesis/chapter-52.md)
## Phase 4

### The Measurement Mandate

- [Chapter 53: The 91 missing TFLOPS](the-measurement-mandate/chapter-53.md)
- [Chapter 54: Observe, don't assume](the-measurement-mandate/chapter-54.md)
- [Chapter 55: Nsight Systems vs Compute](the-measurement-mandate/chapter-55.md)
- [Chapter 56: What kernel time measures](the-measurement-mandate/chapter-56.md)
