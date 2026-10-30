"""LeetGPU 4 - Reduction: output[0] = sum(input) (float32, N up to 100M; perf N = 4M).

Two deterministic passes, no atomics:
  1. A persistent grid (a few programs per SM) streams the array. Each program keeps a
     BLOCK-wide vector accumulator (so each lane sums only N / (P * BLOCK) values) and
     writes one partial.
  2. One program adds the P partials in float64.
Same inputs -> same bits every run, which floating-point atomics cannot promise.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def partial_sum_kernel(x, partials, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(pid * BLOCK, N, nprog * BLOCK):      # grid-stride loop
        offs = start + tl.arange(0, BLOCK)
        acc += tl.load(x + offs, mask=offs < N, other=0.0)
    tl.store(partials + pid, tl.sum(acc.to(tl.float64), axis=0))


@triton.jit
def final_sum_kernel(partials, out, P, BLOCK_P: tl.constexpr):
    offs = tl.arange(0, BLOCK_P)
    p = tl.load(partials + offs, mask=offs < P, other=0.0)
    tl.store(out, tl.sum(p, axis=0).to(tl.float32))


def _num_sms(t):
    return torch.cuda.get_device_properties(t.device).multi_processor_count if t.is_cuda else 4


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK = 4096
    P = max(1, min(triton.cdiv(N, BLOCK), 4 * _num_sms(input)))
    partials = torch.empty(P, device=input.device, dtype=torch.float64)
    partial_sum_kernel[(P,)](input, partials, N, BLOCK=BLOCK, num_warps=8)
    final_sum_kernel[(1,)](partials, output, P, BLOCK_P=triton.next_power_of_2(P))


if __name__ == "__main__":
    from harness import DEVICE, ON_GPU, bench, check

    sizes = [1, 7, 4095, 4097, 300_001] + ([4_194_304, 100_000_000] if ON_GPU else [])
    for n in sizes:
        x = torch.empty(n, device=DEVICE).uniform_(-1000, 1000)
        out = torch.empty(1, device=DEVICE)
        solve(x, out, n)
        check(f"N={n}", out, x.double().sum().float().reshape(1), 1e-5, 1e-5)
    if ON_GPU:
        x = torch.empty(4_194_304, device=DEVICE).uniform_(0, 1000)
        bench("reduction N=4M", lambda: solve(x, out, x.numel()), lambda: x.sum(),
              bytes_moved=4 * x.numel())
