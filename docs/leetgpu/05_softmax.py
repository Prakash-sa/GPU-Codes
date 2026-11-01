"""LeetGPU 5 - Softmax over a whole 1-D array (float32, N up to 500K; perf N = 500K).

The array is too long for one program, so the max and the sum must be combined across
programs. Two kernels:
  1. Each program reads its chunk once and writes a partial (m_b, s_b):
         m_b = max(x_chunk),  s_b = sum(exp(x_chunk - m_b))
  2. Each program loads all partials (only ~N / BLOCK of them), merges them with the
     online-softmax rule  S = sum_b s_b * exp(m_b - M),  then writes exp(x - M) / S.
Traffic: 2 reads + 1 write of x (the minimum for a two-pass softmax); 2 launches.

Comments that start with `mem:` say where the data lives: registers, shared memory,
the L1/L2 caches, or DRAM (GDDR6 on a T4, HBM3 on an H100). See memory-hierarchy.md.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_partial_kernel(x, part_max, part_sum, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    # mem: DRAM -> registers, 4,096 values per program
    v = tl.load(x + offs, mask=offs < N, other=-float("inf"))
    # mem: reductions = shuffles in registers, then shared memory across warps
    m = tl.max(v, axis=0)
    s = tl.sum(tl.exp(v - m), axis=0)              # masked lanes: exp(-inf) = 0
    # mem: two floats per program -> L2, where kernel 2 finds them
    tl.store(part_max + pid, m)
    tl.store(part_sum + pid, s)


@triton.jit
def softmax_normalize_kernel(x, y, part_max, part_sum, N, P,
                             BLOCK: tl.constexpr, BLOCK_P: tl.constexpr):
    # Every program merges the (small) partial arrays itself: no third launch needed.
    p = tl.arange(0, BLOCK_P)
    # mem: every program reads all P partials: L1/L2 hits
    pm = tl.load(part_max + p, mask=p < P, other=-float("inf"))
    ps = tl.load(part_sum + p, mask=p < P, other=0.0)
    M = tl.max(pm, axis=0)
    S = tl.sum(ps * tl.exp(pm - M), axis=0)

    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    # mem: second read of x. N <= 500K is <= 2 MB, which fits L2, so this is mostly an L2 hit
    v = tl.load(x + offs, mask=mask, other=-float("inf"))
    # mem: registers -> DRAM
    tl.store(y + offs, tl.exp(v - M) / S, mask=mask)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK = 4096
    P = triton.cdiv(N, BLOCK)
    part_max = torch.empty(P, device=input.device, dtype=torch.float32)
    part_sum = torch.empty(P, device=input.device, dtype=torch.float32)
    softmax_partial_kernel[(P,)](input, part_max, part_sum, N, BLOCK=BLOCK, num_warps=8)
    softmax_normalize_kernel[(P,)](input, output, part_max, part_sum, N, P,
                                   BLOCK=BLOCK, BLOCK_P=triton.next_power_of_2(P),
                                   num_warps=8)


if __name__ == "__main__":
    from harness import DEVICE, ON_GPU, bench, check

    def ref(x):
        e = torch.exp(x - x.max())
        return e / e.sum()

    sizes = [1, 2, 4095, 4097, 40_001] + ([500_000] if ON_GPU else [])
    for n in sizes:
        x = torch.empty(n, device=DEVICE).uniform_(-10, 10)
        y = torch.empty(n, device=DEVICE)
        solve(x, y, n)
        check(f"N={n}", y, ref(x), 1e-5, 1e-5)
    x = torch.full((5000,), 80.0, device=DEVICE)    # would overflow without the max trick
    y = torch.empty_like(x)
    solve(x, y, x.numel())
    check("large equal values (overflow)", y, ref(x), 1e-5, 1e-5)
    if ON_GPU:
        x = torch.empty(500_000, device=DEVICE).uniform_(-10, 10)
        y = torch.empty_like(x)
        bench("softmax N=500K", lambda: solve(x, y, x.numel()),
              lambda: torch.softmax(x, 0), bytes_moved=3 * 4 * x.numel())
