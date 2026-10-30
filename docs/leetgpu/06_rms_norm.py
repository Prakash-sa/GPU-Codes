"""LeetGPU 50 - RMS Normalization of a 1-D array with scalar gamma/beta (perf N = 100K).

    y = gamma * x / sqrt(mean(x^2) + eps) + beta

Same shape of problem as 05_softmax.py: one global statistic, then an elementwise pass.
  1. Each program writes the partial sum of x^2 for its chunk.
  2. Each program re-adds the partials (cheap: ~N / BLOCK values), computes rstd once,
     and writes its chunk.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def sumsq_partial_kernel(x, partials, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(x + offs, mask=offs < N, other=0.0)
    tl.store(partials + pid, tl.sum(v * v, axis=0))


@triton.jit
def rms_apply_kernel(x, y, partials, gamma, beta, N, P, eps,
                     BLOCK: tl.constexpr, BLOCK_P: tl.constexpr):
    p = tl.arange(0, BLOCK_P)
    total = tl.sum(tl.load(partials + p, mask=p < P, other=0.0), axis=0)
    scale = gamma * tl.rsqrt(total / N + eps)      # fold gamma into one multiplier

    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    v = tl.load(x + offs, mask=mask)
    tl.store(y + offs, v * scale + beta, mask=mask)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, gamma: float, beta: float, output: torch.Tensor, N: int, eps: float):
    BLOCK = 4096
    P = triton.cdiv(N, BLOCK)
    partials = torch.empty(P, device=input.device, dtype=torch.float32)
    sumsq_partial_kernel[(P,)](input, partials, N, BLOCK=BLOCK, num_warps=8)
    rms_apply_kernel[(P,)](input, output, partials, gamma, beta, N, P, eps,
                           BLOCK=BLOCK, BLOCK_P=triton.next_power_of_2(P), num_warps=8)


if __name__ == "__main__":
    from harness import DEVICE, ON_GPU, bench, check

    def ref(x, g, b, eps):
        return g * x / torch.sqrt(torch.mean(x ** 2) + eps) + b

    sizes = [1, 3, 4095, 4097, 33_333] + ([100_000] if ON_GPU else [])
    for n in sizes:
        x = torch.empty(n, device=DEVICE).uniform_(-100, 100)
        y = torch.empty(n, device=DEVICE)
        solve(x, 1.5, -2.0, y, n, 1e-5)
        check(f"N={n}", y, ref(x, 1.5, -2.0, 1e-5), 1e-5, 1e-5)
    if ON_GPU:
        bench("rms_norm N=100K", lambda: solve(x, 1.5, 0.0, y, n, 1e-5),
              lambda: ref(x, 1.5, 0.0, 1e-5), bytes_moved=3 * 4 * n)
