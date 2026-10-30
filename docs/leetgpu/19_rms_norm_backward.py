"""LeetGPU 142 - RMS Norm backward (float32; perf N = 16384, C = 4096; tol 1e-3).

Forward: y = x * rstd * w, rstd = 1 / sqrt(mean(x^2) + eps).  Given dy:
    xhat = x * rstd,  g = dy * w
    dx   = rstd * (g - xhat * mean(g * xhat))        <- a reduction along each ROW
    dw   = sum over rows of dy * xhat                <- a reduction along each COLUMN

The two reductions go in different directions. That is the whole difficulty.
  * Kernel 1, persistent: P programs (a few per SM) each take rows pid, pid+P, ...,
    compute dx for each row from registers, and accumulate dy * xhat for their rows in
    a C-wide register vector. At the end each program writes one partial-dw row: P x C.
  * Kernel 2 sums the P partial rows per column.
No atomics, so the result is deterministic. Extra traffic: P x C floats, ~2 MB for
P = 132, against 768 MB of x/dy/dx.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_bwd_kernel(x, w, dy, dx, dw_part, N, C, eps, BLOCK_C: tl.constexpr):
    pid = tl.program_id(0)
    P = tl.num_programs(0)
    cols = tl.arange(0, BLOCK_C)
    cmask = cols < C
    wv = tl.load(w + cols, mask=cmask, other=0.0)
    dw_acc = tl.zeros([BLOCK_C], dtype=tl.float32)
    for row in range(pid, N, P):
        off = row * C                               # N*C <= 2^30: int32 is safe here
        xv = tl.load(x + off + cols, mask=cmask, other=0.0)
        dyv = tl.load(dy + off + cols, mask=cmask, other=0.0)
        rstd = tl.rsqrt(tl.sum(xv * xv, axis=0) / C + eps)
        xhat = xv * rstd
        g = dyv * wv
        proj = tl.sum(g * xhat, axis=0) / C
        tl.store(dx + off + cols, rstd * (g - xhat * proj), mask=cmask)
        dw_acc += dyv * xhat
    tl.store(dw_part + pid.to(tl.int64) * C + cols, dw_acc, mask=cmask)


@triton.jit
def column_sum_kernel(part, out, P, C, BLOCK_P: tl.constexpr, BLOCK_C: tl.constexpr):
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    cmask = cols < C
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)
    for p0 in range(0, P, BLOCK_P):
        p = p0 + tl.arange(0, BLOCK_P)
        tile = tl.load(part + p[:, None] * C + cols[None, :],
                       mask=(p[:, None] < P) & cmask[None, :], other=0.0)
        acc += tl.sum(tile, axis=0)
    tl.store(out + cols, acc, mask=cmask)


def _num_sms(t):
    return torch.cuda.get_device_properties(t.device).multi_processor_count if t.is_cuda else 4


# x, weight, dy, dx, dweight are tensors on the GPU
def solve(x: torch.Tensor, weight: torch.Tensor, dy: torch.Tensor, dx: torch.Tensor,
          dweight: torch.Tensor, N: int, C: int, eps: float):
    BLOCK_C = triton.next_power_of_2(C)
    P = max(1, min(N, 2 * _num_sms(x)))
    dw_part = torch.empty(P, C, device=x.device, dtype=torch.float32)
    rmsnorm_bwd_kernel[(P,)](x, weight, dy, dx, dw_part, N, C, eps, BLOCK_C=BLOCK_C,
                             num_warps=min(16, max(4, BLOCK_C // 256)))
    column_sum_kernel[(triton.cdiv(C, 128),)](dw_part, dweight, P, C, BLOCK_P=32, BLOCK_C=128)


if __name__ == "__main__":
    from harness import DEVICE, ON_GPU, bench, check

    def ref(x, w, dy, eps):
        rstd = torch.rsqrt(torch.mean(x * x, 1, keepdim=True) + eps)
        xhat = x * rstd
        g = dy * w
        return rstd * (g - xhat * torch.mean(g * xhat, 1, keepdim=True)), (dy * xhat).sum(0)

    shapes = [(1, 1), (3, 5), (17, 100), (300, 129), (9, 2000)] + ([(16384, 4096)] if ON_GPU else [])
    for N, C in shapes:
        x = torch.empty(N, C, device=DEVICE).uniform_(-250, 250)
        w = torch.empty(C, device=DEVICE).uniform_(-2, 2)
        dy = torch.empty(N, C, device=DEVICE).uniform_(-10, 10)
        dx = torch.empty_like(x)
        dw = torch.empty(C, device=DEVICE)
        solve(x, w, dy, dx, dw, N, C, 1e-5)
        want_dx, want_dw = ref(x.double(), w.double(), dy.double(), 1e-5)
        check(f"dx {N}x{C}", dx, want_dx, 1e-3, 1e-3)
        check(f"dw {N}x{C}", dw, want_dw, 1e-3, 1e-3)
    if ON_GPU:
        xr = x.clone().requires_grad_()
        wr = w.clone().requires_grad_()
        y = torch.nn.functional.rms_norm(xr, (C,), wr, 1e-5)
        bench("rmsnorm bwd 16384x4096", lambda: solve(x, w, dy, dx, dw, N, C, 1e-5),
              lambda: torch.autograd.grad(y, (xr, wr), dy, retain_graph=True),
              bytes_moved=3 * 4 * N * C)
