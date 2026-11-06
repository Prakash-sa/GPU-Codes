# GPU interview questions and answers

Questions that come up in GPU, ML-systems and HPC interviews, grouped by topic. Each
answer is the short version you would say out loud, with a link to the page that goes
deeper. Click a question to open its answer. Try answering before you look.

**How to answer any "make it faster" question:** correctness first, then a baseline,
then profile, then decide whether it is bound by memory, compute or latency. Change one
thing, then re-measure. Say the bytes and FLOPs out loud.

## Execution model

??? question "What is a warp, and why does it matter?"
    A warp is 32 threads that issue the same instruction together (SIMT). AMD's equivalent,
    the wavefront, is 64 threads on CDNA. Threads in a warp share an instruction stream,
    so:

    - divergent branches run both paths, one after the other;
    - memory accesses are coalesced per warp;
    - reductions inside a warp can use shuffles instead of shared memory.

    See [Execution model](notes/01-execution-model.md).

??? question "How does a GPU hide memory latency?"
    By having many warps resident on each SM. When one warp waits on a ~500-cycle DRAM load,
    the scheduler issues another ready warp the next cycle, at no switching cost, because
    every warp's registers stay on chip. Latency is hidden when there is enough parallelism:
    enough resident warps (occupancy), or enough independent loads per thread (ILP).

??? question "What is occupancy? Is higher always better?"
    Occupancy is resident warps divided by the maximum warps per SM. It is limited by
    registers per thread, shared memory per block, and block size. Higher is not always
    better: a kernel with high ILP or heavy register/shared-memory tiling (a GEMM) can run
    at peak with low occupancy. Occupancy is a means to hide latency, not a goal. Check
    whether warps are stalled on memory first.

??? question "What is warp divergence, and how do you avoid it?"
    Threads in one warp take different branches, so the warp executes each path with
    the other threads masked off. To avoid it:

    - make the branch condition uniform per warp (branch on `warp_id`, not `thread_id`);
    - sort or bin the work;
    - replace branches with arithmetic or `select`.

    Divergence *between* warps costs nothing.

??? question "What is the difference between `__global__`, `__device__` and `__host__`?"
    | | Runs on | Called from | Returns |
    |---|---|---|---|
    | `__global__` | GPU | CPU via `<<<grid, block>>>` (or GPU with dynamic parallelism) | `void` only |
    | `__device__` | GPU | GPU code only | anything |
    | `__host__` | CPU | CPU code only (the default) | anything |

    - `__host__ __device__` compiles the function for both sides; `__CUDA_ARCH__` tells
      the two copies apart.
    - CPU code can't call a `__device__` function; the only way onto the GPU is a launch,
      and launches are asynchronous.
    - On a *variable*, `__device__` means "lives in global memory", not "runs on the GPU";
      `__shared__` and `__constant__` pick other memory spaces.

    [More detail](notes/01-execution-model.md#function-qualifiers-__global__-__device__-__host__)

??? question "Grid, block, thread: how do you choose the launch configuration?"
    - **Block size:** a multiple of 32, usually 128–512. Enough warps per SM to hide
      latency, without exceeding the register or shared-memory budget.
    - **Grid:** cover the problem, and at least a few waves of blocks over all SMs.
      Alternatively, launch a persistent grid of ~1–4 blocks per SM that loops over the
      work.
    - In Triton you choose `BLOCK` (elements per program) and `num_warps` instead.

??? question "What happens when you launch a kernel from PyTorch?"
    1. The launch is queued on the current CUDA stream, and the call returns to Python
       immediately: execution is asynchronous.
    2. The GPU runs kernels in stream order.
    3. Python only waits at a synchronization point: `.item()`, `.cpu()`, `print(tensor)`,
       `torch.cuda.synchronize()`, or some allocations.

    This is why naive `time.time()` timing is wrong.

## Memory hierarchy

??? question "Walk me through the GPU memory hierarchy."
    | Level | Scope | Size | Speed |
    |---|---|---|---|
    | Registers | per thread | 256 KB per SM | fastest |
    | Shared memory / L1 | per SM, the same SRAM | H100: 256 KB, up to 228 KB as shared | fast |
    | L2 | whole GPU | 50 MB on H100 | |
    | DRAM (HBM3 on H100, GDDR6 on T4) | whole GPU | 80 GB on H100 | 3.35 TB/s on H100 |

    Each level down is bigger and several times slower. Good kernels load from DRAM once
    and reuse data from the upper levels. See [Where the data lives](leetgpu/memory-hierarchy.md).

??? question "What is memory coalescing?"
    Consecutive threads in a warp access consecutive addresses. The hardware then serves
    the warp with the fewest 32-byte sectors (4 sectors for 32 floats). A strided or random
    pattern touches many sectors per request, and wastes bandwidth up to 8–32×. The fixes:

    - a structure-of-arrays layout instead of array-of-structures;
    - tiling through shared memory (transpose);
    - choosing which dimension maps to `threadIdx.x`.

??? question "What are shared-memory bank conflicts?"
    Shared memory has 32 banks, 4 bytes wide. If several threads of a warp access
    *different* addresses in the same bank, the accesses serialize. The classic case is
    column access in a 32×32 tile. Fix it by padding the tile to 33 columns, or by an XOR
    swizzle. Threads reading the *same* address get a broadcast, which is not a conflict.

??? question "What is a register spill, and why is it bad?"
    The compiler runs out of registers and stores values in *local memory*, which is
    per-thread but physically in DRAM (cached in L1/L2). Every spill turns a free register
    access into memory traffic. Check `n_spills` in Triton, or `-Xptxas -v` / Nsight
    Compute in CUDA. Fix it with smaller tiles, fewer live values, or splitting the kernel.

??? question "HBM vs DRAM vs SRAM: what's the difference?"
    - **DRAM** is the GPU's device memory. **HBM** is a kind of DRAM: stacked dies next to
      the GPU with a very wide bus (H100: HBM3, 3.35 TB/s). Consumer and inference cards
      such as the T4 use GDDR6 (320 GB/s).
    - **SRAM** is the on-chip memory: registers, shared memory/L1 and L2.
    - "Keep it in SRAM" (as in FlashAttention) means: don't write the intermediate out to
      HBM.

## Performance analysis

??? question "How do you know if a kernel is memory bound or compute bound?"
    Compute the **arithmetic intensity**, FLOPs per byte moved, and compare it with the
    ridge point:

    $$
    \text{ridge point} = \frac{\text{peak FLOP/s}}{\text{peak bandwidth}}
    $$

    The ridge point is ~20–25 FLOP/byte for FP32 CUDA cores on a T4 or H100, and ~300 for
    H100 FP16 tensor cores. Below the ridge, the kernel is bandwidth bound: minimize
    bytes. Above it, it is compute bound: use tensor cores or lower precision. Then
    confirm with Nsight Compute's Speed of Light section (memory % vs compute %). See the
    [roofline section](notes/03-performance.md).

??? question "Vector add on an H100 takes X µs. Is that good?"
    Compute the floor first:

    - bytes = 3 × 4 × N;
    - floor time = bytes / 3.35 TB/s;
    - achieved bandwidth = bytes / measured time.

    70–90 % of peak is excellent. If it is far below, check:

    - whether the problem is tiny, and so dominated by launch overhead (~3–5 µs);
    - coalescing;
    - whether you timed the compile or the first call.

??? question "How do you benchmark a GPU kernel correctly?"
    - Warm up first, to exclude JIT compilation and autotuning.
    - Time with CUDA events or `triton.testing.do_bench`, never host time around an async
      launch.
    - Flush L2 between runs, so you measure DRAM rather than cache.
    - Report the median and spread over many runs.
    - Test several shapes, including awkward ones.
    - Compare against the right baseline: PyTorch eager, `torch.compile`, cuBLAS.

    See [Benchmark, profile, scale](leetgpu/benchmarking.md).

??? question "Nsight Systems vs Nsight Compute: when do you use which?"
    - **Nsight Systems** (`nsys`): the timeline of the whole program. Use it to find gaps,
      synchronizations, launch overhead, copy/compute overlap and communication. Start
      here.
    - **Nsight Compute** (`ncu`): deep metrics for one kernel. It shows memory vs compute
      throughput, DRAM bytes, cache hit rates, occupancy and stall reasons.
    - On AMD: rocprof and the ROCm Systems / Compute Profilers. In PyTorch,
      `torch.profiler` gives a quick first view.

??? question "A profile shows many tiny kernels with gaps between them. What do you do?"
    The run is launch-bound or host-bound. In order:

    1. Remove host synchronizations (`.item()`, prints).
    2. Fuse elementwise chains: `torch.compile`, or a Triton kernel.
    3. Capture the step with CUDA Graphs.
    4. Batch the small operations.

    Look at the trace to see which kernels sit next to each other and could be one.

## Kernels and algorithms

??? question "How do you write a fast reduction?"
    1. Each thread accumulates many elements in registers (grid-stride or persistent loop).
    2. Reduce within the warp with shuffles.
    3. Combine warps through shared memory.
    4. Combine blocks with a second tiny kernel, which is deterministic.

       Atomics also work, but floating-point atomics make the result order-dependent, so
       not bitwise reproducible.

    Accumulate in FP32, or FP64 for long sums. See [03. Reduction](leetgpu/03-reduction.md).

??? question "How do you parallelize a prefix sum (scan)?"
    Within a block, use a work-efficient scan (Blelloch, or Kogge-Stone with shuffles;
    Triton's `tl.associative_scan`). Across blocks, there are two options:

    - **Three phases:** a reduce pass, a scan of the block sums, then add each block's
      carry.
    - **Single pass with decoupled look-back** (CUB). It reads the input once, but is
      harder to write.

    A segmented scan uses an associative operator on (value, flag) pairs. See
    [16](leetgpu/16-segmented-prefix-sum.md).

??? question "Explain online softmax and why FlashAttention is fast."
    Softmax needs the max and the sum over the whole row. Online softmax keeps a running
    max `m` and sum `l`. When a new block raises the max, it rescales the old sum by
    `exp(m_old − m_new)`, so a single pass suffices.

    FlashAttention applies this to attention: it tiles over K/V and keeps the score tile,
    `m`, `l` and the output accumulator in SRAM. The N × N score matrix is never written to
    HBM. The FLOPs are the same, but memory traffic drops from O(N²) to O(N·d) per head,
    and the kernel becomes compute bound. See [18](leetgpu/18-softmax-attention.md).

??? question "How is a GEMM tiled on a GPU?"
    - Each block computes a BM × BN tile of C, looping over K in BK steps.
    - Each step loads an A tile and a B tile into shared memory. They are double- or
      triple-buffered, so the next tile loads while tensor cores (MMA) work on the current
      one.
    - The accumulator stays in registers.
    - Grouped (swizzled) block order improves L2 reuse.
    - The epilogue (bias, activation, scaling) is fused before the single store.

    See [09. GEMM](leetgpu/09-gemm.md).

??? question "How do you discretize gradient, divergence and the Laplacian? How accurate is it?"
    Replace each derivative with a central difference along its axis:
    $\partial_x u \approx (u_{i+1} - u_{i-1})/2h$ and
    $\partial_{xx} u \approx (u_{i+1} - 2u_i + u_{i-1})/h^2$. The gradient applies the first
    to each axis, the divergence sums it over the components, and the Laplacian sums the
    second over the axes (5-point in 2-D, 7-point in 3-D). Both are second order: halving
    $h$ cuts the error about 4×. Verify that with a manufactured solution in FP64.
    Fourth-order versions need radius-2 stencils and two ghost layers.
    ([stencils](notes/12-stencils-and-discretization.md#from-derivatives-to-stencils))

??? question "Why is a 3-D stencil memory bound, and how do you make it fast?"
    A 7-point Laplacian does ~10–14 FLOPs per 8 bytes (FP32, one read, one write): about
    1.5 FLOP/byte, against ~20 needed to be compute bound on an H100. So the goal is
    one DRAM read per point:

    - x is the fastest index, and threads run along x (coalescing);
    - **2.5-D blocking**: a 2-D tile per block, marching along z with planes k−1, k, k+1
      in registers (and the centre plane in shared memory in CUDA);
    - fuse operators and the time update so intermediates never hit DRAM.

    Check DRAM bytes per point in Nsight Compute against the ideal 8.
    ([stencils](notes/12-stencils-and-discretization.md#making-stencils-fast))

??? question "What is the checkerboard problem, and why use a staggered grid?"
    On a collocated grid, the divergence of a central-difference gradient is a Laplacian
    with spacing 2h. It never compares a point with its direct neighbours, so a +1/−1
    checkerboard has zero gradient and zero div(grad). In pressure–velocity solvers this
    allows odd-even pressure oscillations. A staggered (MAC) grid puts pressure at cell
    centres and velocities on faces, so divergence and gradient use compact h-spaced
    stencils. Collocated codes use Rhie–Chow interpolation instead.
    ([stencils](notes/12-stencils-and-discretization.md#collocated-vs-staggered-grids))

??? question "Why is a naive matrix transpose slow, and how do you fix it?"
    Reading rows is coalesced, but writing columns is strided: each write touches a
    different sector. The fix is to stage a tile in shared memory, read it row-wise and
    write it row-wise transposed. Pad the tile to avoid bank conflicts. See
    [02](leetgpu/02-matrix-transpose.md).

??? question "What is kernel fusion, and when does it help?"
    Fusion combines operations into one kernel, so the intermediates stay in registers
    instead of round-tripping through DRAM. It helps memory-bound chains (elementwise,
    norm + activation, residual + norm, GEMM epilogues). It barely matters inside a
    compute-bound GEMM. The cost is register pressure and code complexity. Examples:
    [08](leetgpu/08-fused-residual-rms-norm.md), [17](leetgpu/17-fused-layernorm-linear-gelu.md).

??? question "Scatter-add with atomics vs sorting: trade-offs?"
    - **Atomics:** simple, and run in L2's atomic units. But they are non-deterministic
      for floats, and slow under contention (many updates to the same index).
    - **Sort by index, then segmented reduction:** deterministic and contention-free, but
      the sort costs extra memory passes.

    Choose based on reproducibility needs and the index distribution. See
    [11](leetgpu/11-scatter-add.md).

## Triton

??? question "How does Triton's programming model differ from CUDA's?"
    In Triton you write one **program** that works on blocks (tensors) of data, not one
    thread. The compiler handles:

    - the mapping onto warps and threads;
    - coalescing;
    - shared-memory staging;
    - tensor-core use (`tl.dot`);
    - software pipelining (`num_stages`).

    You control the block sizes, masks and `num_warps`. It is much faster to write and
    usually within ~10–20 % of hand-tuned CUDA for fusable, tile-based work. It gives less
    control over warp-level tricks and shared-memory layout.

??? question "What do `num_warps` and `num_stages` do?"
    - `num_warps` is the number of warps (× 32 threads) that execute one program. More
      warps split a block's work and hide latency, but leave fewer registers per thread.
    - `num_stages` is the software-pipelining depth for loads feeding `tl.dot` in a loop.
      It sets how many shared-memory buffers are in flight at once (async copies on sm_80+).
      Deeper pipelines hide more latency but use more shared memory.

    `triton.autotune` searches over both.

??? question "When would you choose CUDA over Triton?"
    Choose CUDA (or CUTLASS/CuTe) when you need:

    - warp-level control (shuffles, warp specialization);
    - explicit shared-memory layouts;
    - Hopper features Triton doesn't expose well yet;
    - irregular control flow;
    - the last 10–20 % of a hot GEMM or attention kernel.

    Choose Triton for fused elementwise, norm and reduction kernels and custom
    attention variants, where development speed matters.

## PyTorch and ML systems

??? question "What does `torch.compile` do?"
    It traces the Python code into an FX graph (TorchDynamo), then compiles it with
    TorchInductor. Inductor fuses pointwise and reduction operations into generated Triton
    kernels and can use CUDA Graphs (`mode="reduce-overhead"`). It mostly helps
    memory-bound, launch-heavy models. Watch for graph breaks and recompilation on new
    shapes.

??? question "What are CUDA Graphs, and when do they help?"
    A CUDA Graph records a sequence of kernel launches once and replays the whole sequence
    with one launch, so the per-kernel CPU launch overhead disappears. They help small-batch
    inference and steps made of many small kernels. Requirements:

    - static shapes and memory addresses;
    - no host synchronization inside the graph.

??? question "FP16 vs BF16 vs TF32 vs FP8: what do you pick?"
    | Format | Range | Precision | Typical use |
    |---|---|---|---|
    | FP16 | small: needs loss scaling in training | 10-bit mantissa | inference |
    | BF16 | FP32's range | 7-bit mantissa | the training default on Ampere+ |
    | TF32 | FP32's range | 10-bit mantissa | FP32 matmuls on tensor cores, automatic on Ampere+ |
    | FP8 (E4M3 / E5M2) | small | very low | Hopper+, with per-tensor or per-block scaling |

    Always accumulate in FP32. T4 has no BF16 or TF32 tensor cores.

??? question "Why do results differ between runs on GPU?"
    The floating-point summation order changes: atomics, split-K, nondeterministic
    reductions in cuDNN/cuBLAS, a different number of GPUs. For reproducibility:

    - use deterministic algorithms (`torch.use_deterministic_algorithms(True)`);
    - use two-pass reductions instead of atomics;
    - fix seeds and configurations.

??? question "How do you reduce GPU memory use in training?"
    - Mixed precision.
    - Activation checkpointing: recompute instead of storing.
    - Shard state with FSDP/ZeRO: parameters, gradients, optimizer state.
    - Smaller micro-batches with gradient accumulation.
    - Fused kernels that don't save intermediates.
    - 8-bit optimizers.
    - Check for fragmentation with `torch.cuda.memory_summary()`.

## Multi-GPU and communication

??? question "Explain ring all-reduce. What does it cost?"
    The p GPUs form a ring.

    1. **Reduce-scatter:** p − 1 steps, each GPU sending one 1/p chunk per step. Every GPU
       ends up owning one fully reduced chunk.
    2. **All-gather:** another p − 1 steps, passing the reduced chunks around.

    Each GPU sends $2\frac{p-1}{p}S$ bytes, almost independent of p, so the ring is
    bandwidth-optimal. Its latency grows linearly with p, which is why NCCL uses trees for
    small messages and many nodes. See [10](notes/10-multi-gpu-communication.md).

??? question "NVLink vs NVSwitch vs PCIe?"
    - **PCIe** is the general bus to the CPU: Gen5 x16 is ~64 GB/s each way, and it is
      shared with host traffic.
    - **NVLink** is a direct GPU-to-GPU link: H100 has 18 links, 900 GB/s total.
    - **NVSwitch** connects all GPUs' NVLinks through switch chips, so every GPU reaches
      every other at full bandwidth in one hop. H100 NVSwitch can also reduce in the
      switch (NVLS).

    Without a switch, NVLink bandwidth is split between peers.

??? question "NCCL vs MPI: when do you use which?"
    - **NCCL** (RCCL on AMD): topology-aware GPU collectives that run as kernels on a CUDA
      stream. Best for big all-reduce, all-gather and reduce-scatter in training.
    - **MPI**: general host-driven messaging and job launch. A **GPU-aware** MPI passes
      device pointers and uses GPUDirect. Best for point-to-point halos, irregular
      patterns and existing HPC codes.

    Common pattern: MPI launches the ranks and broadcasts the NCCL id; NCCL moves the bulk
    data.

??? question "What is GPUDirect RDMA?"
    The network card reads and writes GPU memory directly over PCIe, without staging
    through host memory. It needs:

    - driver support (`nvidia-peermem` or dma-buf);
    - ideally, the NIC and GPU under the same PCIe switch.

    On AMD the equivalent is ROCm RDMA (PeerDirect). Without it, every inter-node message
    costs an extra device-host copy on each side.

??? question "How do you map tensor, pipeline and data parallelism onto hardware?"
    | Parallelism | Communication | Put it on |
    |---|---|---|
    | Tensor | all-reduce in every layer, on the critical path | one NVLink/NVSwitch domain, usually TP ≤ 8 |
    | Pipeline | small activation sends between stages | can cross nodes |
    | Data / FSDP | once per step or per layer, overlappable with backward | across nodes |
    | MoE all-to-all | every MoE layer | NVLink or a high-bisection network |

??? question "Training scales well to 8 GPUs but poorly to 16. Why?"
    Most likely 16 GPUs means 2 nodes: you crossed from NVLink (~450 GB/s each way) to the
    network (~50 GB/s per NIC). Checks:

    - nccl-tests across the nodes;
    - `NCCL_DEBUG=INFO`: is it using IB or sockets, and is GPUDirect RDMA on?
    - the Nsight Systems timeline: is communication overlapped?

    Fixes: keep TP inside a node, overlap the DDP/FSDP communication, larger buckets, BF16
    gradients, more gradient accumulation.

??? question "A distributed job hangs. How do you debug it?"
    The usual cause is ranks calling different collectives, or in a different order (a
    data-dependent branch, or an exception on one rank). To debug:

    - set a timeout and `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`, so the job fails instead of
      hanging;
    - `TORCH_DISTRIBUTED_DEBUG=DETAIL` reports mismatched collectives;
    - `NCCL_DEBUG=INFO`;
    - check the rendezvous address/port and network interface (`NCCL_SOCKET_IFNAME`);
    - py-spy each rank to see where it is stuck.

??? question "Where should data *not* go in a multi-GPU program?"
    - Through host memory between GPUs on the same node: enable P2P, and never copy
      device → host → device.
    - Across CPU sockets (`SYS` in `nvidia-smi topo -m`) for chatty traffic.
    - Across nodes, for anything per-layer and on the critical path (tensor parallel).
    - To the CPU every step: `.item()` in the training loop.

    Keep data on the GPU, fuse, and match each communication pattern to the fastest link
    that can carry it. See [where data should move](notes/10-multi-gpu-communication.md#where-data-should-move-and-where-not).

??? question "What is NUMA, and why does it matter on a GPU server?"
    On a multi-socket server, each CPU socket has its own memory and its own PCIe root
    complex. Memory on another socket is reached over UPI or Infinity Fabric: slower,
    with higher latency and shared bandwidth. Each GPU and NIC hangs off one socket, its
    NUMA node.

    If a rank's pinned buffers, launch thread or DataLoader workers sit on the other node,
    every host-to-device copy and every CPU step crosses the socket link. A NIC on the
    wrong socket can also disable GPUDirect RDMA. GPU-to-GPU traffic over NVLink is not
    affected. See [NUMA](notes/10-multi-gpu-communication.md#numa-the-cpu-side-of-the-topology).

??? question "Host-to-device bandwidth is fine on GPUs 0–3 but low and noisy on 4–7. Why?"
    Most likely NUMA. The process (or its pinned buffers) lives on socket 0, but GPUs 4–7
    are attached to socket 1, so those copies cross the socket link. Pages are placed on
    first touch, by whichever CPU first wrote them. To check and fix:

    1. `nvidia-smi topo -m` shows each GPU's NUMA affinity.
    2. Run one process per GPU, bound to that GPU's node *before* it allocates
       (`numactl --cpunodebind=N --membind=N`, `srun --gpu-bind=closest`, or
       `mpirun --bind-to numa`).
    3. Re-measure.

??? question "How do you bind distributed ranks to the right CPUs?"
    One rank per GPU, each bound to the cores and memory of its GPU's NUMA node, using
    the NIC on the same socket. Options:

    - Slurm: `--gpus-per-task=1 --gpu-bind=closest --cpu-bind=cores`.
    - Open MPI: `--map-by ppr:4:numa --bind-to numa`.
    - torchrun: a wrapper that runs each rank under `numactl` for the GPU's node.

    DataLoader workers inherit the binding. Check with `numactl --show` or `taskset -p`
    inside a rank.

## NVIDIA vs AMD

??? question "What changes when you port a CUDA kernel to AMD?"
    - **The API:** HIP is nearly 1:1 with CUDA, and `hipify` converts most source.
    - **Wavefront 64** instead of warp 32 on CDNA. Code that hard-codes 32 breaks: warp
      shuffles, ballots (64-bit masks), reductions and `tid % 32` tricks.
    - **Names:** shared memory is LDS (64 KB per CU on MI300X); tensor cores are Matrix
      Cores (MFMA).
    - **Libraries:** cuBLAS → rocBLAS/hipBLASLt, NCCL → RCCL, Nsight → rocprof.

    Triton and PyTorch hide most of these differences: PyTorch on ROCm keeps the
    `torch.cuda` API.

??? question "How does an 8 × MI300X node differ from an 8 × H100 node for communication?"
    | | 8 × H100 | 8 × MI300X |
    |---|---|---|
    | Topology | NVSwitch: each GPU reaches any other at full 900 GB/s, one hop | Infinity Fabric mesh, no switch: 7 links per GPU (896 GB/s total) |
    | One pair of GPUs | full bandwidth | ~1/7 of the total |
    | In-switch reductions | yes (NVLS) | no |
    | Memory per GPU | 80 GB | 192 GB: fewer GPUs needed per model |

    Collectives using all peers use every MI300X link; point-to-point between two GPUs
    does not.

## Scenario questions

??? question "This PyTorch training step is slow. Walk me through what you'd do."
    1. Get a reproducible baseline: time per step, after warmup.
    2. Run `torch.profiler` or Nsight Systems.
        - Is the GPU idle (dataloader, host syncs, launch overhead)?
        - Is the time in a few big kernels (GEMMs, attention)?
        - Or in many small ones (elementwise, norms)?
    3. **GPU idle:** more dataloader workers, pinned memory, remove `.item()`, CUDA Graphs.
    4. **Many small kernels:** `torch.compile`, or fused Triton kernels.
    5. **Big GEMMs:** BF16/TF32, aligned shapes (multiples of 8/16), FlashAttention.
    6. **Multi-GPU:** check communication overlap and topology.

    Re-measure after each change.

??? question "You wrote a Triton kernel that's 3× slower than PyTorch. Why might that be?"
    Common causes:

    - timing included compilation or autotuning;
    - tiny grid (too few programs for the SMs);
    - `BLOCK` too large, causing register spills;
    - uncoalesced pointer arithmetic (a wrong stride on the fastest axis);
    - fp32 `tl.dot` without TF32;
    - missing `num_stages` pipelining;
    - many launches where PyTorch uses one fused cuBLAS call.

    Check `n_regs`, `n_spills`, the autotune choice, and Nsight Compute's DRAM throughput.

??? question "How would you implement a 2-D stencil across 4 GPUs?"
    1. Domain-decompose: give each GPU a slab, plus halo rows.
    2. Each step:
        - compute the boundary rows first;
        - start the halo exchange (NCCL send/recv, or P2P copies, on a separate stream);
        - compute the interior meanwhile;
        - wait, then swap buffers.
    3. Communication grows with the surface and compute with the volume, so bigger slabs
       scale better. On PCIe-only nodes, overlap is essential. See
       [14](leetgpu/14-jacobi-stencil-2d.md) and [Multi-GPU](notes/05-multi-gpu.md).

??? question "What would you optimize first at 10× the problem size?"
    1. Re-check what the kernel is bound by: caches stop holding the working set
       (the L2 hits disappear).
    2. Int32 index overflow past 2³¹ elements.
    3. Memory capacity: stream in chunks, or shard.
    4. Precision of longer reductions: use FP32/FP64 accumulators.
    5. Autotune keys for new shapes.
    6. Then multi-GPU: sharding plus communication overlap.

    See "Scaling up" on each [LeetGPU page](leetgpu/index.md).

## Quick numbers to know

| | T4 | A100 | H100 SXM | MI300X |
|---|---|---|---|---|
| Memory | 16 GB GDDR6 | 40/80 GB HBM2 / HBM2e | 80 GB HBM3 | 192 GB HBM3 |
| Bandwidth | 320 GB/s | 1.6–2.0 TB/s | 3.35 TB/s | 5.3 TB/s |
| FP16/BF16 dense tensor | 65 TFLOP/s (FP16) | 312 TFLOP/s | 989 TFLOP/s | 1,307 TFLOP/s |
| GPU-to-GPU | PCIe Gen3 | NVLink 3: 600 GB/s | NVLink 4: 900 GB/s | Infinity Fabric: 896 GB/s |
| SMs / CUs | 40 | 108 | 132 | 304 |

Launch overhead ~3–5 µs. DRAM latency ~400–800 cycles. Warp 32, wavefront 64. More in the
[spec sheets](gpu-spec-sheets.md) and the [cheat sheet](notes/cheatsheet.md).
