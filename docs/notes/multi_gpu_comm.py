"""Multi-GPU communication probe: correctness, a halo-style ring exchange, and
all-reduce bandwidth, using torch.distributed.

    torchrun --nproc_per_node=8 docs/notes/multi_gpu_comm.py   # NVIDIA: NCCL. AMD (ROCm build): RCCL
    torchrun --nproc_per_node=2 docs/notes/multi_gpu_comm.py   # no GPU: Gloo on CPU, checks the logic only
    NCCL_DEBUG=INFO torchrun ...                               # prints the transport NCCL/RCCL picked
    NCCL_P2P_DISABLE=1 torchrun ...                            # force the host-memory path, to compare

Across nodes, run torchrun on each node with --nnodes, --node_rank and --rdzv_endpoint.
"""
import os
import time

import torch
import torch.distributed as dist


def sync(on_gpu):
    if on_gpu:
        torch.cuda.synchronize()


def main():
    on_gpu = torch.cuda.is_available()           # also True on ROCm builds of PyTorch
    dist.init_process_group("nccl" if on_gpu else "gloo")   # on ROCm, "nccl" is RCCL
    rank, world = dist.get_rank(), dist.get_world_size()
    if on_gpu:
        dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
        torch.cuda.set_device(dev)               # one process per GPU
    else:
        dev = torch.device("cpu")

    # 1. All-reduce correctness: every rank contributes its rank id.
    x = torch.full((1024,), float(rank), device=dev)
    dist.all_reduce(x)                           # in place, sum by default
    assert torch.all(x == world * (world - 1) / 2), x[:4]

    # 2. Point-to-point ring (the shape of a 1-D halo exchange): send right, receive left.
    send = torch.full((1024,), float(rank), device=dev)
    recv = torch.empty_like(send)
    ops = [dist.P2POp(dist.isend, send, (rank + 1) % world),
           dist.P2POp(dist.irecv, recv, (rank - 1) % world)]
    for req in dist.batch_isend_irecv(ops):     # batched: no deadlock from send/recv order
        req.wait()
    assert torch.all(recv == (rank - 1) % world), recv[:4]
    if rank == 0:
        print(f"world={world} backend={dist.get_backend()} device={dev}: all_reduce and ring exchange ok")

    # 3. All-reduce bandwidth. busbw is comparable across GPU counts (same as nccl-tests).
    sizes = [1 << k for k in range(20, 31, 2)] if on_gpu else [1 << k for k in (12, 16, 20)]
    for nbytes in sizes:
        buf = torch.ones(nbytes // 4, device=dev)
        for _ in range(5):                       # warmup: builds communicators and rings
            dist.all_reduce(buf)
        sync(on_gpu)
        dist.barrier()
        iters = 20
        t0 = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(buf)                 # queued on the GPU stream: returns at once
        sync(on_gpu)                             # so wait before stopping the clock
        t = (time.perf_counter() - t0) / iters
        algbw = nbytes / t / 1e9
        busbw = algbw * 2 * (world - 1) / world  # bytes each GPU really sends, ring all-reduce
        if rank == 0:
            size = f"{nbytes >> 10} KiB" if nbytes < 1 << 20 else f"{nbytes >> 20} MiB"
            print(f"  {size:>9}  {t * 1e6:9.1f} us  algbw {algbw:7.2f} GB/s  busbw {busbw:7.2f} GB/s")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
