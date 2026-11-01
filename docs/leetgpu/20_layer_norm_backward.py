"""LeetGPU 144 - Layer Norm backward, given the saved mean and rstd
(float32; perf N = 16384, C = 4096; tol 1e-3).

    xhat = (x - mean) * rstd,  g = dy * gamma
    dx     = rstd * (g - mean(g) - xhat * mean(g * xhat))    <- row reductions
    dgamma = sum over rows of dy * xhat                        <- column reductions
    dbeta  = sum over rows of dy

Same structure as 19_rms_norm_backward.py, with one more row statistic (mean(g)) and
two column accumulators per program (dgamma and dbeta partials) in a [2, P, C] buffer,
finished by one column-sum launch for each. Deterministic, no atomics.

Comments that start with `mem:` say where the data lives: registers, shared memory,
the L1/L2 caches, or DRAM (GDDR6 on a T4, HBM3 on an H100). See memory-hierarchy.md.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_bwd_kernel(x, gamma, mean, rstd, dy, dx, part, N, C, BLOCK_C: tl.constexpr):
    pid = tl.program_id(0)
    P = tl.num_programs(0)
    cols = tl.arange(0, BLOCK_C)
    cmask = cols < C
    # mem: gamma: DRAM -> registers once per program, reused for all of its rows
    gv = tl.load(gamma + cols, mask=cmask, other=0.0)
    # mem: two column accumulators (dgamma, dbeta) in registers across the rows
    dg_acc = tl.zeros([BLOCK_C], dtype=tl.float32)
    db_acc = tl.zeros([BLOCK_C], dtype=tl.float32)
    for row in range(pid, N, P):
        off = row * C                               # N*C <= 2^30: int32 is safe here
        # mem: the saved mean/rstd: one scalar each per row, cheaper than recomputing
        mu = tl.load(mean + row)
        rs = tl.load(rstd + row)
        # mem: one row of x and dy: DRAM -> registers, read exactly once
        xv = tl.load(x + off + cols, mask=cmask, other=0.0)
        dyv = tl.load(dy + off + cols, mask=cmask, other=0.0)
        xhat = tl.where(cmask, (xv - mu) * rs, 0.0)
        g = dyv * gv
        # mem: row sums = shuffles + shared memory across warps
        mean_g = tl.sum(g, axis=0) / C
        proj = tl.sum(g * xhat, axis=0) / C
        # mem: registers -> DRAM
        tl.store(dx + off + cols, rs * (g - mean_g - xhat * proj), mask=cmask)
        dg_acc += dyv * xhat
        db_acc += dyv
    base = pid.to(tl.int64) * C + cols
    # mem: one row of each partial -> L2 ([2, P, C] <= 8.6 MB, read back from L2)
    tl.store(part + base, dg_acc, mask=cmask)                 # part[0, pid, :]
    tl.store(part + P * C + base, db_acc, mask=cmask)         # part[1, pid, :]


@triton.jit
def column_sum_kernel(part, out, P, C, BLOCK_P: tl.constexpr, BLOCK_C: tl.constexpr):
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    cmask = cols < C
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)
    for p0 in range(0, P, BLOCK_P):
        p = p0 + tl.arange(0, BLOCK_P)
        # mem: partials from L2; the column sums accumulate in registers
        tile = tl.load(part + p[:, None] * C + cols[None, :],
                       mask=(p[:, None] < P) & cmask[None, :], other=0.0)
        acc += tl.sum(tile, axis=0)
    tl.store(out + cols, acc, mask=cmask)


def _num_sms(t):
    return torch.cuda.get_device_properties(t.device).multi_processor_count if t.is_cuda else 4


# x, gamma, mean, rstd, dy, dx, dgamma, dbeta are tensors on the GPU
def solve(x: torch.Tensor, gamma: torch.Tensor, mean: torch.Tensor, rstd: torch.Tensor,
          dy: torch.Tensor, dx: torch.Tensor, dgamma: torch.Tensor, dbeta: torch.Tensor,
          N: int, C: int):
    BLOCK_C = triton.next_power_of_2(C)
    P = max(1, min(N, 2 * _num_sms(x)))
    part = torch.empty(2, P, C, device=x.device, dtype=torch.float32)
    layernorm_bwd_kernel[(P,)](x, gamma, mean, rstd, dy, dx, part, N, C, BLOCK_C=BLOCK_C,
                               num_warps=min(16, max(4, BLOCK_C // 256)))
    grid = (triton.cdiv(C, 128),)
    column_sum_kernel[grid](part[0], dgamma, P, C, BLOCK_P=32, BLOCK_C=128)
    column_sum_kernel[grid](part[1], dbeta, P, C, BLOCK_P=32, BLOCK_C=128)


if __name__ == "__main__":
    from harness import DEVICE, ON_GPU, bench, check

    def ref(x, gamma, mean, rstd, dy):
        xhat = (x - mean[:, None]) * rstd[:, None]
        g = dy * gamma
        dx = rstd[:, None] * (g - g.mean(1, keepdim=True) - xhat * (g * xhat).mean(1, keepdim=True))
        return dx, (dy * xhat).sum(0), dy.sum(0)

    shapes = [(1, 1), (3, 5), (17, 100), (300, 129), (9, 2000)] + ([(16384, 4096)] if ON_GPU else [])
    for N, C in shapes:
        x = torch.empty(N, C, device=DEVICE).uniform_(-250, 250)
        gamma = torch.empty(C, device=DEVICE).uniform_(-2, 2)
        dy = torch.empty(N, C, device=DEVICE).uniform_(-10, 10)
        mean = x.mean(1)
        rstd = torch.rsqrt(x.var(1, unbiased=False) + 1e-5)
        dx = torch.empty_like(x)
        dg = torch.empty(C, device=DEVICE)
        db = torch.empty(C, device=DEVICE)
        solve(x, gamma, mean, rstd, dy, dx, dg, db, N, C)
        want = ref(*(t.double() for t in (x, gamma, mean, rstd, dy)))
        for name, got, w in zip(("dx", "dgamma", "dbeta"), (dx, dg, db), want):
            check(f"{name} {N}x{C}", got, w, 1e-3, 1e-3)
    if ON_GPU:
        bench("layernorm bwd 16384x4096", lambda: solve(x, gamma, mean, rstd, dy, dx, dg, db, N, C),
              bytes_moved=3 * 4 * N * C)
