# 11. Inside AWS GPU instances

What is wired to what inside six AWS GPU instances, from a single L40S to eight B200s: the
GPUs, NVLink and NVSwitch, the PCIe switches, the two CPU sockets, the EFA network cards
and the local NVMe. Each diagram is followed by what it means for where your data should
move. The general rules behind this are on
[How GPUs talk to each other](10-multi-gpu-communication.md).

!!! note "Where the numbers come from"
    Counts, CPUs, memory, NICs and NVMe are from AWS's
    [accelerated-computing instance specifications](https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html)
    and the [P5](https://aws.amazon.com/ec2/instance-types/p5/) and
    [P6](https://aws.amazon.com/ec2/instance-types/p6/) pages. NVSwitch counts and link speeds
    are NVIDIA's HGX board specs. AWS does not publish its exact PCIe switch wiring, so the
    8-GPU diagrams use the HGX reference shape: one PCIe switch per GPU pair, with that
    pair's NICs and NVMe drives on it. Check the real layout with `nvidia-smi topo -m`
    ([below](#check-it-on-the-instance)).

## At a glance

| Instance | GPUs | GPU memory | GPU ↔ GPU | CPU ↔ GPU (host root) | Network per GPU | CPU | On-demand $/h (approx.) |
|---|---|---|---|---|---|---|---|
| g6e.2xlarge | 1× L40S | 48 GB GDDR6 (44 GiB usable) | none | PCIe Gen4 x16, ~32 GB/s | up to 20 Gb/s ENA, no EFA | EPYC 7R13, 8 vCPU | 2.24 |
| p5.4xlarge | 1× H100 SXM | 80 GB HBM3 | none (P2P N/A) | PCIe Gen4, ~32 GB/s | 100 Gb/s EFA, no GPUDirect RDMA | EPYC 7R13, 16 vCPU | 6.88 |
| p4d.24xlarge | 8× A100 | 8 × 40 GB HBM2 | NVLink 3 + 6 NVSwitch, 600 GB/s | PCIe Gen3, ~16 GB/s | 50 Gb/s (4 × 100 Gb shared) | 2× Xeon 8275CL, 96 vCPU | 21.96 |
| p5.48xlarge | 8× H100 SXM | 8 × 80 GB HBM3 | NVLink 4 + 4 NVSwitch, 900 GB/s | PCIe Gen4, ~32 GB/s | 400 Gb/s (32 × 100 Gb) | 2× EPYC 7R13, 192 vCPU | 55.04 |
| p5en.48xlarge | 8× H200 | 8 × 141 GB HBM3e | NVLink 4 + 4 NVSwitch, 900 GB/s | PCIe Gen5, ~64 GB/s | 400 Gb/s (16 × 200 Gb) | 2× Sapphire Rapids, 192 vCPU | 63.30 |
| p6-b200.48xlarge | 8× B200 | 8 × 179 GiB HBM3e | NVLink 5 + 2 NVSwitch, 1.8 TB/s | PCIe Gen5, ~64 GB/s | 400 Gb/s (8 × 400 Gb) | 2× Emerald Rapids, 192 vCPU | 113.93 |

GPU ↔ GPU figures are per GPU, both directions added (half each way). PCIe figures are each
way for one x16 link. Prices are on-demand Linux and change; check the
[EC2 pricing page](https://aws.amazon.com/ec2/pricing/on-demand/).

Three things set the shape of each box:

- **NVLink or not.** Only the 8-GPU P instances have it. The single-GPU sizes have no peer,
  and the L40S has no NVLink at all.
- **The host PCIe root.** The GPUs are Gen4 (A100, L40S) or Gen5 (H100, H200, B200), but
  the CPU decides the host link: Cascade Lake is Gen3, EPYC Milan is Gen4, Sapphire and
  Emerald Rapids are Gen5.
- **NICs per GPU.** All three current 8-GPU instances give each GPU 400 Gb/s of EFA, split
  over 4, 2 or 1 cards; p4d gives each GPU pair one 100 Gb card.

## g6e.2xlarge: 1× L40S

![g6e.2xlarge: L40S on PCIe Gen4 under an EPYC Milan slice](../assets/img/instances/g6e-2xlarge.png)

- Everything goes over one PCIe Gen4 x16 link (~32 GB/s each way at best). The GPU's own memory is 864 GB/s, so a kernel that re-reads host data
  every step runs at PCIe speed, not GDDR6 speed.
- No EFA, so this is a single-node instance. Larger g6e sizes add GPUs, still with no
  NVLink: GPU-to-GPU traffic goes over PCIe, through the host if the GPUs sit under
  different roots.

## p5.4xlarge: 1× H100

![p5.4xlarge: one H100 SXM with unused NVLink ports, PCIe to an EPYC slice](../assets/img/instances/p5-4xlarge.png)

- It is the same H100 SXM (80 GB HBM3, 3.35 TB/s) as in p5.48xlarge, so single-GPU kernel
  timings carry over directly.
- The GPU's NVLink ports have no peer in the instance: AWS lists GPU peer-to-peer as N/A.
- It has a 100 Gb EFA card, but **no GPUDirect RDMA**. Multi-node traffic is copied
  GPU → host memory → NIC, so NCCL across several p5.4xlarge instances is limited by those
  PCIe copies. For multi-GPU work, use one 8-GPU instance instead of several of these.

## p4d.24xlarge: 8× A100 40 GB

![p4d.24xlarge: HGX A100 with 6 NVSwitch chips, 4 PCIe switches and Cascade Lake CPUs](../assets/img/instances/p4d-24xlarge.png)

- Every A100 has 12 NVLink 3 links, two to each of the 6 NVSwitch chips. Any two GPUs talk
  at the full 600 GB/s (300 each way), so the topology is flat: no "near" or "far" GPU.
- The host side is the weak link. Cascade Lake roots are PCIe Gen3 (~16 GB/s each way),
  half what the A100's Gen4 port can do. Keep the data on the GPUs and stage input with
  pinned memory and overlap ([streams](04-streams.md)).
- There are 4 × 100 Gb EFA cards, one per PCIe switch, so two GPUs share each card
  (50 Gb/s per GPU). Across nodes, p4d is network-bound much sooner than p5.

## p5.48xlarge: 8× H100 80 GB

![p5.48xlarge: HGX H100 with 4 NVSwitch chips, 4 EFA cards per GPU and EPYC Milan CPUs](../assets/img/instances/p5-48xlarge.png)

### How the eight H100s connect

- Each H100 has 18 NVLink 4 links, split over the 4 NVSwitch chips in groups of
  5 + 4 + 4 + 5. Every GPU reaches every other GPU in one switch hop at 900 GB/s
  (450 each way). `nvidia-smi topo -m` shows `NV18` for every GPU pair.
- Third-generation NVSwitch can add up data inside the switch (NVLink SHARP). NCCL uses it
  for all-reduce with the NVLS algorithm, which roughly halves the bytes each GPU has to
  send: each byte goes up once instead of about twice in a ring.
- GPU-to-GPU never touches PCIe or the CPUs. A tensor-parallel group of 8 lives entirely
  on NVLink.

### Host and network

- The H100s are PCIe Gen5, but the EPYC Milan roots are Gen4, so host ↔ GPU copies top
  out near 32 GB/s each way. That is ~14× slower than NVLink's 450 GB/s each way:
  never route GPU-to-GPU data through host memory.
- There are 32 × 100 Gb EFA cards, 4 per GPU, on the same PCIe switch as their GPU.
  GPUDirect RDMA sends straight from HBM to the NIC without the CPU. Bind each rank to its
  socket's cores and memory (GPUs 0–3 on socket 0 and 4–7 on socket 1 in this layout;
  confirm with `nvidia-smi topo -m`; [NUMA](10-multi-gpu-communication.md#numa-the-cpu-side-of-the-topology)).

## p5en.48xlarge: 8× H200

![p5en.48xlarge: HGX H200 with 4 NVSwitch chips, Gen5 host links and Sapphire Rapids CPUs](../assets/img/instances/p5en-48xlarge.png)

### How the eight H200s connect

- **The GPU-to-GPU fabric is the same as on p5**: 18 NVLink 4 links per GPU over
  4 NVSwitch chips, 900 GB/s per GPU, `NV18` between every pair. NCCL collectives inside
  the node run at the same speed as on p5.
- The H200 is the same Hopper chip with more and faster memory: 141 GB of HBM3e at
  4.8 TB/s, compared with 80 GB at 3.35 TB/s. Memory-bound kernels speed up about 1.4×,
  and bigger models or batches fit per GPU, which often means less tensor parallelism and
  less NVLink traffic.

### What changes from p5

- The CPUs are Sapphire Rapids, so the host root is PCIe Gen5 end to end (~64 GB/s each
  way), about twice p5's host-copy bandwidth. AWS calls this out as Gen5 PCIe between
  CPU and GPU.
- There are 16 × 200 Gb third-generation EFA cards, 2 per GPU, for the same 400 Gb/s per
  GPU as p5 with fewer cards.

## p6-b200.48xlarge: 8× B200

![p6-b200.48xlarge: HGX B200 with 2 NVSwitch chips, NVLink 5, Emerald Rapids CPUs](../assets/img/instances/p6-b200-48xlarge.png)

- Each B200 has 18 NVLink 5 links at twice the speed: 1.8 TB/s per GPU (900 each way),
  through 2 larger NVSwitch chips. Still one hop between any pair.
- Each GPU has 179 GiB of HBM3e at about 8 TB/s.
- Gen5 to the Emerald Rapids CPUs; 8 × 400 Gb fourth-generation EFA cards, one per GPU.
- The ratio that matters: NVLink is now 900 GB/s each way, while the network stays at
  400 Gb/s (50 GB/s) each way per GPU, an 18× gap. Keep tensor and expert parallelism inside the node and
  send only data-parallel gradients (or pipeline activations) across nodes.

## Where data should move on these instances

| Path | g6e.2xlarge | p5.4xlarge | p4d | p5 / p5en | p6-b200 |
|---|---|---|---|---|---|
| GPU ↔ GPU, same instance | — | — | NVLink, 300 GB/s each way | NVLink, 450 GB/s | NVLink, 900 GB/s |
| Host ↔ GPU | Gen4, ~32 GB/s | Gen4, ~32 GB/s | Gen3, ~16 GB/s | Gen4 (p5) / Gen5 (p5en) | Gen5, ~64 GB/s |
| GPU ↔ another instance | not for multi-node | EFA via host memory, 12.5 GB/s | EFA + GPUDirect, 6.25 GB/s per GPU | EFA + GPUDirect, 50 GB/s per GPU | EFA + GPUDirect, 50 GB/s per GPU |

Rules that follow:

1. Inside an 8-GPU P instance, put the chattiest parallelism (tensor, expert, sequence) on
   NVLink. It is 9× (p5) to 18× (p6) faster per GPU than EFA, and ~50× on p4d.
2. Never bounce GPU-to-GPU data through the host on an NVLink instance. Check that
   `NCCL_P2P_DISABLE` is unset and that `NCCL_DEBUG=INFO` shows P2P or NVLS channels
   inside the node, not SHM.
3. Across instances, use EFA with the AWS OFI NCCL plugin, one rank per GPU, each bound to
   the NUMA node of its GPU and NICs.
4. On the single-GPU sizes, the host link is your only link: load the working set once and
   keep it resident.

## Running them well: performance, efficiency, utilization

The goal is the lowest **cost per finished result** (time to solution × $/h), not the
highest GPU-Util number. Work through the levels in order: pick the right box, set it up
so nothing outside the GPU is the bottleneck, keep the GPU busy with useful work, and
stop paying when it isn't.

### 1. Pick the instance by what limits the job

Cost per unit of each resource, from the prices in the table above:

| Instance | $/GPU-hour | $/h per 100 GB of GPU memory | $/h per TB/s of GPU memory bandwidth |
|---|---|---|---|
| g6e.2xlarge | 2.24 | 4.67 | 2.59 |
| p5.4xlarge | 6.88 | 8.60 | 2.05 |
| p4d.24xlarge | 2.75 | 6.86 | 1.77 |
| p5.48xlarge | 6.88 | 8.60 | 2.05 |
| p5en.48xlarge | 7.91 | 5.61 | **1.65** |
| p6-b200.48xlarge | 14.24 | 7.96 | 1.78 |

- **Memory-bandwidth-bound jobs** (stencils, LBM, SpMV, statevector, LLM decode) run at a
  speed roughly proportional to HBM bandwidth, so the last column is the cost per unit of
  work. The cheapest GPU-hour (g6e) is the most expensive way to move bytes; p5en is the
  cheapest. Measure your own kernel, but start from this column, not the first.
- **Capacity-bound jobs** (the model, lattice or statevector must fit): the middle column.
  An H200 holds 1.76× an H100's data, which can turn a 2-GPU job into a 1-GPU job and
  remove the communication entirely.
- **Double precision:** the L40S does ~1.4 TFLOPS FP64, against 34 (67 with tensor
  cores) on the H100. Don't run FP64 solvers on g6e
  ([spec sheets](../gpu-spec-sheets.md)).
- **Tensor-core training and inference:** H100/H200 add FP8 (Transformer Engine) and B200
  adds FP4. These matter only if your framework uses them; otherwise compare BF16
  throughput.
- **One GPU or eight:** never build a multi-GPU job from several p5.4xlarge instances
  (no NVLink, no GPUDirect RDMA). If you need more than one GPU's memory or tight
  coupling, use one 8-GPU instance.
- **Many small independent jobs:** run one per GPU on an 8-GPU instance
  (`CUDA_VISIBLE_DEVICES=i`, each pinned to its GPU's NUMA node), or split GPUs with MIG
  (A100, H100 and H200 support up to 7 instances each). Only do this if you can keep all
  eight busy; otherwise single-GPU instances waste less.

### 2. Set up the instance so the host and network keep up

- **Start from a matching software stack.** The AWS Deep Learning AMIs and containers ship
  the NVIDIA driver, CUDA, NCCL, the EFA installer and the `aws-ofi-nccl` plugin as a
  tested set. Mismatched NCCL and plugin versions are the most common cause of slow
  multi-node runs.
- **Attach every EFA interface.** On p4d, p5, p5en and p6, launch with one EFA interface
  per network card (4, 32, 16 and 8). Fewer interfaces silently means less bandwidth. For
  multi-node on-demand or Spot, put instances in a **cluster placement group**; instances
  in a Capacity Block are placed close together by AWS and don't take a placement group.
- **Let the plugin pick the network settings.** Recent `aws-ofi-nccl` sets its own EFA
  defaults; keep it current instead of copying old lists of `FI_*` and `NCCL_*` variables.
  Check with `NCCL_DEBUG=INFO` that NCCL loads the OFI plugin and finds every EFA device.
- **Bind to NUMA.** One rank per GPU, with its CPU threads, pinned buffers and DataLoader
  workers on that GPU's socket ([`numa_bind.sh`](10-multi-gpu-communication.md#binding-each-rank-to-its-gpus-numa-node)).
- **Use the local NVMe.** The 8 instance-store drives are fast but wiped when the
  instance stops. Stripe them into one RAID 0 volume for datasets, scratch and
  checkpoints, then copy what you need to keep to S3 (or FSx for Lustre for shared data).
- **Check the hardware before a long run.** Run `nvidia-smi -pm 1` (persistence mode),
  `dcgmi diag -r 1` (quick health check) and one `nccl-tests` all-reduce. A single slow
  GPU or missing EFA device slows every rank of a synchronous job.

### 3. Keep the GPUs busy with useful work

- **Measure real utilization.** `nvidia-smi`'s GPU-Util only says *some* kernel was
  running. Use DCGM's profiling counters, for example
  `dcgmi dmon -e 1002,1003,1004,1005` (SM active, SM occupancy, tensor pipe active,
  DRAM active), or Nsight Systems for the timeline ([profiling](06-profiling.md)).
- **Feed the GPU.** Use pinned host memory with `non_blocking=True` copies, enough
  DataLoader workers, and prefetching, so the next batch is already on the GPU. On p4d
  (PCIe Gen3) and p5 (Gen4) this matters more than on p5en and p6 (Gen5).
- **Use the fast math units.** BF16 or TF32 matmuls
  (`torch.set_float32_matmul_precision("high")`), FP8 where your stack supports it on
  Hopper and Blackwell, fused kernels (FlashAttention, fused optimizers), and
  `torch.compile`.
- **Cut launch overhead.** Small kernels leave a fast GPU idle between launches. Use CUDA
  Graphs, fusion and larger batches ([optimization strategies](09-optimization-strategies.md)).
- **Spend the memory.** On H200 and B200, use the extra memory for larger batches or less
  sharding before reaching for activation checkpointing, which trades memory for
  recomputation.
- **Overlap communication with compute.** DDP gradient buckets and FSDP prefetch hide
  collectives behind the backward pass; check in the timeline that NCCL kernels run
  alongside compute, not after it.

### 4. Map the parallelism onto the links

- **Inside one 8-GPU instance (NVLink):** tensor, sequence and expert parallelism, and
  FSDP sharding. These send data every layer.
- **Across instances (EFA):** data parallelism and pipeline stages, which send once per
  step or once per micro-batch. With FSDP, use hybrid sharding (shard inside the
  instance, replicate across instances) so the per-layer gathers stay on NVLink.
- **Domain decomposition (CFD, stencils):** give each GPU a slab, exchange halos with
  NCCL send/recv or CUDA-aware MPI over NVLink, and keep the halo small relative to the
  interior so compute hides the exchange.
- **Check the result.** `nccl-tests` bus bandwidth inside one instance should be close
  to the NVLink figure, and across instances close to the EFA figure. A large shortfall
  means a setup problem, not a property of the hardware.

### 5. Don't pay for idle hardware

- **Stop or terminate instances as soon as the job ends.** An 8-GPU instance idling for
  an hour costs as much as a full hour of work. Auto-stop on low GPU utilization, or run
  jobs through a scheduler (AWS Batch, ParallelCluster with Slurm, EKS, SageMaker) that
  releases nodes when the queue is empty.
- **Match the purchase option to the job.** Spot for jobs that checkpoint often and can
  restart; [EC2 Capacity Blocks for ML](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-capacity-blocks.html)
  to reserve p4d, p5 (both sizes), p5en or p6-b200 for days or weeks (no cancellations);
  Savings Plans for steady long-term use.
- **Checkpoint for restarts, not just for safety.** Write to local NVMe asynchronously,
  then copy to S3, so a checkpoint costs seconds of GPU time and a lost node costs only
  the steps since the last one.
- **Track cost per result.** Log time per step (or per solve) and the instance price.
  Re-run the comparison when you change instance, precision or parallel layout.

### Per-instance checklist

| Instance | Best use | Watch out for |
|---|---|---|
| g6e.2xlarge | FP32/BF16 inference, development, small single-GPU jobs | Weak FP64; PCIe-only; no EFA, so no multi-node |
| p5.4xlarge | Single-GPU H100 work: profiling, kernel tuning, jobs that fit in 80 GB | No NVLink or GPUDirect RDMA: don't chain several for one job |
| p4d.24xlarge | 8-GPU NVLink jobs on a budget; FP64 (A100: 9.7 / 19.5 tensor TFLOPS) | Gen3 host link; 50 Gb/s EFA per GPU limits multi-node scaling |
| p5.48xlarge | Large training, multi-GPU CFD, NVLS all-reduce | Gen4 host link: keep data resident; attach all 32 EFA interfaces |
| p5en.48xlarge | Memory-bound or memory-hungry jobs (141 GB per GPU, 4.8 TB/s) | Same NVLink as p5; benefits come from memory and the Gen5 host link |
| p6-b200.48xlarge | Largest models, FP8/FP4 inference and training | Highest $/GPU-hour: needs high utilization to pay off; keep TP inside the instance |

## Eight 1-GPU instances instead of one 8-GPU instance

You can run an 8-GPU job on eight p5.4xlarge instances instead of one p5.48xlarge. The
code barely changes; the hardware between the GPUs does.

| | 1 × p5.48xlarge | 8 × p5.4xlarge |
|---|---|---|
| GPUs | 8 × H100 80 GB | 8 × H100 80 GB |
| Price (on-demand, above) | $55.04/h | 8 × $6.88 = **$55.04/h** |
| GPU ↔ GPU | NVLink + NVSwitch, 450 GB/s each way | EFA, 100 Gb/s = 12.5 GB/s each way |
| GPUDirect RDMA | yes | no: GPU → host memory → NIC |
| vCPUs / RAM / NVMe | 192 / 2 TiB / 8 × 3.8 TB | 128 / 8 × 256 GiB / 8 × 3.8 TB |
| 1 GB ring all-reduce (lower bound) | ~4 ms (less with NVLS) | ~140 ms |

At the same price, the 8-GPU instance has a GPU-to-GPU link ~36× faster. The all-reduce
times are lower bounds from link speed (each GPU sends 2 × 7/8 × 1 GB = 1.75 GB), not
measurements; the host staging on p5.4xlarge only adds to them.

### How to run it

1. **Launch** eight p5.4xlarge instances in one Availability Zone, in a cluster placement
   group (or a Capacity Block), from the Deep Learning AMI. Attach each instance's network
   card as an **EFA** interface, and use a security group that allows all traffic to and
   from itself (EFA requires this).
2. **Share code and data**: FSx for Lustre or EFS mounted on every node, or sync from S3
   at start-up. Use one directory for checkpoints that every node can read.
3. **Start one process per instance.** The training script is the same as single-node
   DDP or FSDP; only the launcher changes:

    ```bash
    # on each node i = 0..7 (HEAD = private IP of node 0)
    torchrun --nnodes=8 --nproc_per_node=1 --node_rank=$i \
             --rdzv_backend=c10d --rdzv_endpoint=$HEAD:29500 train.py

    # or with Slurm (e.g. AWS ParallelCluster): srun starts torchrun once per node
    HEAD=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
    srun --nodes=8 --ntasks-per-node=1 --gpus-per-node=1 \
         torchrun --nnodes=8 --nproc_per_node=1 \
                  --rdzv_backend=c10d --rdzv_endpoint=$HEAD:29500 train.py
    ```

    `LOCAL_RANK` is 0 everywhere and `RANK` runs 0–7. Shard the data with
    `DistributedSampler` and write checkpoints from rank 0 only.
4. **Check that NCCL uses EFA.** With `NCCL_DEBUG=INFO`, the log should show the OFI
   plugin with the `efa` provider. `NET/Socket` means it fell back to TCP. Then run
   `nccl-tests` `all_reduce_perf` across the 8 nodes. Bus bandwidth can't exceed the
   ~12 GB/s line rate, compared with hundreds of GB/s inside one p5.48xlarge.

### Make the job fit the slower link

- **Data parallelism only.** Never split one layer (tensor parallelism) or one expert
  layer across these instances; that traffic needs NVLink.
- **Communicate less often.** Use gradient accumulation (all-reduce every *k*
  micro-batches), larger DDP buckets, and BF16 gradient compression (DDP's
  `bf16_compress_hook`). Make sure communication overlaps the backward pass.
- **FSDP**: full sharding across nodes gathers parameters over EFA every layer. Shard only
  if the model doesn't fit in 80 GB, and expect communication to dominate.
- **CFD and stencils**: give each GPU a large subdomain, so the halo (surface) is small
  next to the interior (volume), and compute the interior while halos are in flight.
- **Rule of thumb**: if a step's compute time is well above its communication time
  (bytes ÷ 12.5 GB/s), eight small instances scale fine. If not, they don't.

### When eight small instances are the better choice

- **Independent jobs**: parameter sweeps, ensembles, separate simulations or inference
  replicas. There is no communication at all, so the interconnect doesn't matter, and
  each instance can start, finish and be stopped on its own schedule.
- **Capacity**: single-GPU instances can be easier to get than a whole 8-GPU box.
- **Fault isolation and elasticity**: one failure takes down one GPU, not eight, and you
  can scale between 1 and 8 as the queue changes (with `torchrun --nnodes=1:8` for
  elastic training).

For anything that talks every step, at the same price, take the single 8-GPU instance.
The same reasoning rules out 8 × g6e.2xlarge for coupled jobs: no EFA, only up to
20 Gb/s of ordinary networking. For 8 L40S in one box, g6e.48xlarge keeps the traffic on
PCIe inside a single host.

## Check it on the instance

```bash
nvidia-smi topo -m                 # NV12 (p4d) or NV18 (p5, p5en, p6) between every GPU pair;
                                   # PIX/PXB = same PCIe switch, SYS = across sockets
nvidia-smi nvlink --status -i 0    # per-link speed for GPU 0
lscpu | grep -i numa               # NUMA nodes and which CPUs belong to each
fi_info -p efa -t FI_EP_RDM        # EFA devices visible to libfabric / NCCL
NCCL_DEBUG=INFO torchrun --nproc_per_node=8 docs/notes/multi_gpu_comm.py   # transport chosen + busbw
```

The probe script is explained on
[How GPUs talk to each other](10-multi-gpu-communication.md#try-it-verified-probe). On
p5.48xlarge, an 8-GPU all-reduce of large buffers should approach NVLink-limited bus
bandwidth (several hundred GB/s). If it reports PCIe-like numbers (tens of GB/s), P2P is
off or the GPUs are not where you think they are.

Regenerate the diagrams with `python scripts/make_instance_figures.py`.
