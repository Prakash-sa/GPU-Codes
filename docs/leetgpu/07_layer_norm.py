"""LeetGPU 113 - Layer Normalization over rows of an N x C matrix (perf N = 65536, C = 512).

    y = (x - mean) / sqrt(var + eps) * weight + bias      (biased variance, per row)

Rows are short (C <= 4096), so a program normalizes BLOCK_M rows at once from a
[BLOCK_M, BLOCK_C] register tile: x is read once, y written once, and the mean and
variance never touch memory. The variance is computed from centred values
(sum((x - mean)^2)), not E[x^2] - E[x]^2, which cancels badly when |mean| >> std.

Comments that start with `mem:` say where the data lives: registers, shared memory,
the L1/L2 caches, or DRAM (GDDR6 on a T4, HBM3 on an H100). See memory-hierarchy.md.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_kernel(x, w, b, y, N, C, eps,
                     BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_C)
    cmask = cols < C
    mask = (rows[:, None] < N) & cmask[None, :]
    offs = rows[:, None].to(tl.int64) * C + cols[None, :]

    # mem: DRAM -> registers, BLOCK_M x BLOCK_C; each element is read once
    v = tl.load(x + offs, mask=mask, other=0.0)
    # mem: row sums = shuffles in registers (+ shared memory if a row spans warps)
    mean = tl.sum(v, axis=1) / C
    d = tl.where(mask, v - mean[:, None], 0.0)       # zero the padding before squaring
    # mem: d = x - mean is still in registers, so the centred variance costs no extra DRAM read
    var = tl.sum(d * d, axis=1) / C
    rstd = tl.rsqrt(var + eps)

    # mem: w and b (2 KB) are read by every program: L1/L2 hits
    wv = tl.load(w + cols, mask=cmask)
    bv = tl.load(b + cols, mask=cmask)
    # mem: registers -> DRAM
    tl.store(y + offs, d * rstd[:, None] * wv[None, :] + bv[None, :], mask=mask)


# input, weight, bias, output are tensors on the GPU
def solve(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
          output: torch.Tensor, N: int, C: int, eps: float):
    BLOCK_C = triton.next_power_of_2(C)
    BLOCK_M = max(1, min(16, 4096 // BLOCK_C))     # ~4K elements per program
    num_warps = 4 if BLOCK_M * BLOCK_C <= 2048 else 8
    grid = (triton.cdiv(N, BLOCK_M),)
    layernorm_kernel[grid](input, weight, bias, output, N, C, eps,
                           BLOCK_M=BLOCK_M, BLOCK_C=BLOCK_C, num_warps=num_warps)


if __name__ == "__main__":
    from harness import DEVICE, ON_GPU, bench, check

    def ref(x, w, b, eps):
        return torch.nn.functional.layer_norm(x, (x.shape[1],), w, b, eps)

    shapes = [(1, 1), (3, 7), (17, 100), (33, 512), (5, 4096)] + ([(65536, 512)] if ON_GPU else [])
    for n, c in shapes:
        x = torch.empty(n, c, device=DEVICE).uniform_(-100, 100) + 1000.0   # large offset
        w = torch.empty(c, device=DEVICE).uniform_(0.1, 10)
        b = torch.empty(c, device=DEVICE).uniform_(-10, 10)
        y = torch.empty_like(x)
        solve(x, w, b, y, n, c, 1e-5)
        check(f"{n}x{c}", y, ref(x.double(), w.double(), b.double(), 1e-5), 1e-4, 1e-4)
    bench("layernorm 65536x512", lambda: solve(x, w, b, y, n, c, 1e-5),
          lambda: ref(x, w, b, 1e-5), bytes_moved=2 * 4 * n * c)
