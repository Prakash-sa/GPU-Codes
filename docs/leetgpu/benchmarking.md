# Benchmark, profile and scale up

The workflow every solution page follows. It is the same loop you would run in an
interview on unfamiliar PyTorch code:
**correctness, then baseline, then profile, then classify, then change one thing, then re-measure.**

## 1. Benchmark correctly

GPU work is **asynchronous**: a kernel launch returns to Python before the GPU finishes.
Timing with `time.time()` around a launch measures the launch, not the kernel.

```python
# Wrong: measures ~5 us of launch overhead, whatever the kernel does
t0 = time.time(); solve(x, y, n); print(time.time() - t0)

# Right: CUDA events, recorded on the GPU's own timeline
start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
for _ in range(10):                 # warmup: JIT compile, autotune, caches, clocks
    solve(x, y, n)
times = []
for _ in range(100):
    start.record(); solve(x, y, n); end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end))         # ms
print(f"median {statistics.median(times)*1e3:.1f} us, p10..p90 "
      f"{np.percentile(times,10)*1e3:.1f}..{np.percentile(times,90)*1e3:.1f} us")
```

`triton.testing.do_bench(fn)` does this for you: warmup, L2 flush between runs, median.
The harness's `bench()` uses it and also prints GB/s, TFLOP/s and the speedup over the
PyTorch reference:

```bash
python docs/leetgpu/07_layer_norm.py
#   layernorm 65536x512    <t> us   <bw> GB/s  | torch  <t_ref> us  (<speedup>x)
```

Rules:

- **Warm up.** The first call JIT-compiles the kernel, and with `@triton.autotune` it
  benchmarks every config. That can take seconds. Never include it in a timing.
- **Report medians and spread**, not a single run.
- **Test several shapes**, including the awkward ones. A kernel tuned for one size can
  be slow, or wrong, at another.
- **Compare against the right baseline**: PyTorch eager, `torch.compile`, and the
  vendor library (cuBLAS through `torch.matmul`).
- **Separate the steady state from one-off costs**: compilation, autotuning,
  allocations (`torch.empty` inside `solve()` costs ~µs; preallocate in production),
  host-to-device copies.

## 2. Know the speed limit before you start

Every page computes a **roofline floor** for its LeetGPU size:

$$
t_{\min} = \max\left(\frac{\text{bytes moved}}{\text{bandwidth}},\ \frac{\text{FLOPs}}{\text{peak FLOP/s}}\right)
$$

| | T4 (sm_75) | H100 SXM (sm_90) |
|---|---|---|
| SMs | 40 | 132 |
| DRAM bandwidth | 320 GB/s | 3,350 GB/s |
| L2 cache | 4 MB | 50 MB |
| Shared memory per SM | 64 KB | 228 KB |
| Registers per SM | 64 K × 32-bit | 64 K × 32-bit |
| FP32 (CUDA cores) | 8.1 TFLOP/s | 67 TFLOP/s |
| TF32 tensor | **none** | 495 TFLOP/s |
| FP16/BF16 tensor | 65 TFLOP/s (FP16 only, no BF16) | 989 TFLOP/s |
| FP64 | 0.25 TFLOP/s (1/32 rate) | 34 TFLOP/s |
| Async copy / TMA | no / no | `cp.async` / TMA, wgmma |
| Ridge point (FP32) | ~25 FLOP/byte | ~20 FLOP/byte |

Arithmetic intensity (FLOPs per byte) below the ridge point means **bandwidth bound**:
optimize bytes. Above it means **compute bound**: optimize the math (tensor cores,
precision). Small problems are **launch or latency bound** whatever their intensity: a
launch costs ~3–5 µs, and a 1 MB problem moves in 0.3 µs on an H100.

Achieved bandwidth = bytes / time. **70–90% of peak is excellent** for a streaming kernel.
If you are there, stop tuning the kernel and look at fusion or the algorithm instead.

## 3. Profile

### torch.profiler: which kernels, how many launches, how long

```bash
PROFILE=1 python docs/leetgpu/08_fused_residual_rms_norm.py
```

This prints a table sorted by CUDA time and writes `trace_<name>.json`. Open it in
<https://ui.perfetto.dev> or `chrome://tracing`.

- **Count the kernels per call.** The PyTorch reference for 08 launches ~5. The fused
  Triton version launches 1.
- **Look for gaps between kernels** in the trace. Gaps mean launch overhead, host work
  or a synchronization (`.item()`, `print(tensor)`, `.cpu()`).
- Use `record_shapes=True` and `profile_memory=True` when hunting allocations.

### Nsight Systems: the timeline of a whole program

```bash
nsys profile -t cuda,nvtx -o run python docs/leetgpu/16_segmented_prefix_sum.py
nsys stats run.nsys-rep            # per-kernel time table
```

### Nsight Compute: why one kernel is slow

Triton kernels show up under their Python function name:

```bash
ncu --kernel-name regex:layernorm --launch-skip 5 --launch-count 1 --set full \
    -o ln python docs/leetgpu/07_layer_norm.py
ncu-ui ln.ncu-rep
```

| Section | Question it answers |
|---|---|
| GPU Speed Of Light | memory % vs compute % of peak, which is the bound |
| Memory Workload Analysis | DRAM bytes vs your estimate; L2 hit rate; sectors/request (4 = coalesced fp32) |
| Occupancy | achieved vs theoretical warps; the limiter (registers, shared memory, block size) |
| Warp State Statistics | why warps stall (long scoreboard = waiting on DRAM) |
| Launch Statistics | registers/thread, shared memory/block, grid size vs SM count (waves) |

On LeetGPU Pro the Perfetto trace and the DRAM/SM throughput and occupancy numbers are
the same quantities.

### Triton-specific inspection

```python
k = layernorm_kernel[grid](...)          # a launch returns the compiled kernel
print(k.n_regs, k.n_spills)              # registers per thread; spills > 0 is bad
print(k.metadata.shared)                 # shared memory bytes per program
print(k.asm["ptx"][:2000])               # also "ttir", "ttgir", "sass" (if cuobjdump)
```

```bash
TRITON_PRINT_AUTOTUNING=1 python docs/leetgpu/09_gemm.py   # which config won, per shape
TRITON_INTERPRET=1 python ...                              # debug indexing on CPU, print with tl.device_print
```

## 4. Classify, then pick the fix

| Profile says | Typical fix in these solutions |
|---|---|
| Many small kernels, gaps | fuse (05, 06, 08, 17), CUDA Graphs, fewer launches |
| DRAM % high, SM % low | fewer bytes: fusion, lower precision storage, no intermediates |
| DRAM % low, SM % low | more parallelism / bytes in flight: bigger blocks, more programs, persistent grid |
| SM % high, tensor pipe idle | move the math to `tl.dot`; FP16/BF16/TF32 where tolerance allows |
| Sectors/request ≫ 4 | fix the access pattern (02 transpose, SoA layouts) |
| Register spills | smaller BLOCK, more `num_warps`, split the kernel, chunk long rows (08) |
| Results differ between runs | atomics or reduction order: use two-pass deterministic reductions (03, 11, 19) |

## 5. What changes at larger scale

These apply to every page. The per-page "Scaling up" sections add the specifics.

- **Index overflow.** `pid * BLOCK + arange` is int32. Past 2^31 elements (8 GB of
  fp32) offsets wrap silently. Cast row indices to int64 (`.to(tl.int64)`) before
  multiplying by a stride, as 08, 11 and 17 do.
- **Grid limits.** Axis 0 allows up to 2^31−1 programs, but axes 1 and 2 only 65,535.
  Put the large dimension on axis 0, or use a persistent grid-stride loop.
- **Persistent kernels.** Launch ~1–4 programs per SM and loop over work (03, 19). That
  removes tail effects, keeps per-program state in registers, and makes per-SM
  partial results cheap.
- **Autotune keys.** Autotuning reruns for every new value of `key`. With many distinct
  shapes, bucket them (round to powers of two) or fix the config per shape class.
- **Memory capacity.** T4 has 16 GB and H100 80 GB. Don't allocate big intermediates in
  `solve()`. Reuse buffers, fuse, or stream in chunks.
- **Precision.** At larger sizes, sums get longer and rounding error grows. Keep
  accumulators in FP32 (or FP64 for scans and long reductions, 03 and 16), and validate
  against a float64 reference at the *largest* size you will run.
- **Many GPUs.** Shard the independent dimension (rows, batch, i-blocks of N-body)
  across GPUs. Reductions then need a cross-GPU step (NCCL all-reduce), and stencils
  need halo exchange. See [Multi-GPU and NCCL](../notes/05-multi-gpu.md).
- **Launch overhead at scale.** Per-step pipelines of many small kernels should be
  captured with CUDA Graphs (`torch.cuda.graphs`). See
  [Streams](../notes/04-streams.md#cuda-graphs).

## Reading

- [PyTorch profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- Notes in this site: [Profiling and debugging](../notes/06-profiling.md),
  [Optimization strategies](../notes/09-optimization-strategies.md),
  [Programming with Triton](../notes/08-triton.md)
- Videos: [GPU MODE Lecture 1: profiling CUDA kernels in PyTorch](https://www.youtube.com/watch?v=LuhJEEJQgUM),
  [GPU MODE Lecture 16: On Hands Profiling](https://www.youtube.com/watch?v=SKV6kDk1s94)
