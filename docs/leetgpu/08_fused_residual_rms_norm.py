"""LeetGPU 83 - Fused Residual Add + RMS Norm over rows (perf N = C = 4096).

    z = x + residual;   out = z / sqrt(mean(z^2) + eps) * weight

Unfused PyTorch runs ~5 kernels (add, pow, mean, sqrt/div, mul) and writes z and the
intermediates to memory. Fused: read x and residual once, write out once (12 bytes per
element instead of ~40).

C can be up to 65536, too long for one register tile, so there are two code paths:
  ONE_PASS  (C <= 8192): the whole row lives in registers.
  two-pass  (larger C):  loop over the row for sum(z^2), then loop again to write.
                         The second read mostly hits L2 (a row is <= 512 KB).
Row offsets use int64: N * C can reach 2^32, which overflows int32.

Comments that start with `mem:` say where the data lives: registers, shared memory,
the L1/L2 caches, or DRAM (GDDR6 on a T4, HBM3 on an H100). See memory-hierarchy.md.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def residual_rmsnorm_kernel(x, res, w, out, C, eps,
                            BLOCK_C: tl.constexpr, ONE_PASS: tl.constexpr):
    row = tl.program_id(0).to(tl.int64) * C
    if ONE_PASS:
        cols = tl.arange(0, BLOCK_C)
        mask = cols < C
        # mem: DRAM -> registers. The whole row of z = x + r stays in registers;
        # z is never written to DRAM (that is the fusion)
        z = tl.load(x + row + cols, mask=mask, other=0.0) + tl.load(res + row + cols, mask=mask, other=0.0)
        rstd = tl.rsqrt(tl.sum(z * z, axis=0) / C + eps)
        # mem: w is shared by every row: L1/L2 hit
        wv = tl.load(w + cols, mask=mask)
        # mem: registers -> DRAM
        tl.store(out + row + cols, z * rstd * wv, mask=mask)
    else:
        acc = tl.zeros([BLOCK_C], dtype=tl.float32)
        for start in range(0, C, BLOCK_C):
            cols = start + tl.arange(0, BLOCK_C)
            mask = cols < C
            # mem: pass 1, one 4,096-wide chunk in registers at a time. A whole long row
            # would spill, and spills live in DRAM
            z = tl.load(x + row + cols, mask=mask, other=0.0) + tl.load(res + row + cols, mask=mask, other=0.0)
            acc += z * z
        rstd = tl.rsqrt(tl.sum(acc, axis=0) / C + eps)
        for start in range(0, C, BLOCK_C):
            cols = start + tl.arange(0, BLOCK_C)
            mask = cols < C
            # mem: pass 2 re-reads the row (<= 512 KB for x and r): mostly from L2
            z = tl.load(x + row + cols, mask=mask, other=0.0) + tl.load(res + row + cols, mask=mask, other=0.0)
            wv = tl.load(w + cols, mask=mask)
            tl.store(out + row + cols, z * rstd * wv, mask=mask)


# x, residual, weight, out are tensors on the GPU
def solve(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, out: torch.Tensor,
          N: int, C: int, eps: float):
    MAX_ONE_PASS = 8192
    one_pass = C <= MAX_ONE_PASS
    BLOCK_C = triton.next_power_of_2(C) if one_pass else 4096
    num_warps = min(16, max(4, BLOCK_C // 512))
    residual_rmsnorm_kernel[(N,)](x, residual, weight, out, C, eps,
                                  BLOCK_C=BLOCK_C, ONE_PASS=one_pass, num_warps=num_warps)


if __name__ == "__main__":
    from harness import DEVICE, ON_GPU, bench, check

    def ref(x, r, w, eps):
        z = x + r
        return z / torch.sqrt(torch.mean(z ** 2, dim=-1, keepdim=True) + eps) * w

    shapes = [(1, 1), (3, 5), (7, 1000), (2, 8192), (2, 8193), (3, 20_000)]
    shapes += [(4096, 4096)] if ON_GPU else []
    for n, c in shapes:
        x = torch.empty(n, c, device=DEVICE).uniform_(-10, 10)
        r = torch.empty(n, c, device=DEVICE).uniform_(-10, 10)
        w = torch.empty(c, device=DEVICE).uniform_(0.5, 1.5)
        out = torch.empty_like(x)
        solve(x, r, w, out, n, c, 1e-5)
        check(f"{n}x{c}", out, ref(x, r, w, 1e-5), 1e-5, 1e-5)
    bench("residual+rmsnorm 4096x4096", lambda: solve(x, r, w, out, n, c, 1e-5),
          lambda: ref(x, r, w, 1e-5), bytes_moved=3 * 4 * n * c)
