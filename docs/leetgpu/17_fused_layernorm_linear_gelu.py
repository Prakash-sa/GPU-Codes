"""LeetGPU 131 - Fused LayerNorm -> Linear -> exact GELU (float32; perf M = K = N = 4096).

    xn = (x - mu) / sqrt(var + eps) * gamma + beta;  h = xn @ W + bias;  y = GELU_erf(h)

Unfused: LayerNorm writes xn (M x K) to memory, the GEMM reads it back, and the
bias + GELU pass reads and writes h (M x N) once more. Fused, in two kernels:
  1. row_stats_kernel: mean and rstd per row (2 x M floats). Centred two-pass variance:
     the inputs carry a large common offset (~ -1000), where E[x^2] - E[x]^2 cancels
     catastrophically.
  2. ln_gemm_gelu_kernel: a tiled GEMM that normalizes each x tile *as it is loaded*
     (prologue fusion) and applies bias + GELU before the single store (epilogue fusion).
     xn and h never exist in DRAM.

Do not apply the "algebraic" fusion h = rstd * (x @ W') - ... here: it runs the GEMM on
the raw, offset x and cancels to garbage at this tolerance. Normalize first, then multiply.

Precision: FP32 operands on TF32 tensor cores lose ~3 decimal digits; "tf32x3" recovers
near-FP32 accuracy for the 1e-3 tolerance at roughly 3x the TF32 cost. On T4 (no TF32)
_dot_precision() picks "ieee" instead.
"""
import os

import torch
import triton
import triton.language as tl

_CONFIGS = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=4),
]
if os.environ.get("TRITON_INTERPRET") == "1":
    _CONFIGS = _CONFIGS[-1:]


@triton.jit
def row_stats_kernel(x, mean, rstd, K, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64) * K
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, K, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        acc += tl.load(x + row + cols, mask=cols < K, other=0.0)
    mu = tl.sum(acc, axis=0) / K
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, K, BLOCK):                 # second pass hits L2 (row <= 32 KB)
        cols = start + tl.arange(0, BLOCK)
        d = tl.where(cols < K, tl.load(x + row + cols, mask=cols < K, other=0.0) - mu, 0.0)
        acc += d * d
    tl.store(mean + tl.program_id(0), mu)
    tl.store(rstd + tl.program_id(0), tl.rsqrt(tl.sum(acc, axis=0) / K + eps))


@triton.autotune(configs=_CONFIGS, key=["M", "N", "K"])
@triton.jit
def ln_gemm_gelu_kernel(x, mean, rstd, gamma, beta, w, bias, y, M, N, K,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                        GROUP_M: tl.constexpr, PREC: tl.constexpr):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    group = GROUP_M * num_n
    first_m = (pid // group) * GROUP_M
    group_rows = min(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group) % group_rows
    pid_n = (pid % group) // group_rows

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    mmask = rm < M
    mu = tl.load(mean + rm, mask=mmask, other=0.0)
    rs = tl.load(rstd + rm, mask=mmask, other=0.0)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + rk
        kmask = k < K
        amask = mmask[:, None] & kmask[None, :]
        xt = tl.load(x + rm[:, None].to(tl.int64) * K + k[None, :], mask=amask, other=0.0)
        g = tl.load(gamma + k, mask=kmask, other=0.0)
        bt = tl.load(beta + k, mask=kmask, other=0.0)
        a = (xt - mu[:, None]) * rs[:, None] * g[None, :] + bt[None, :]
        a = tl.where(amask, a, 0.0)                  # padding must be 0, not (0 - mu) * ...
        wt = tl.load(w + k[:, None].to(tl.int64) * N + rn[None, :],
                     mask=kmask[:, None] & (rn[None, :] < N), other=0.0)
        acc = tl.dot(a, wt, acc, input_precision=PREC)

    h = acc + tl.load(bias + rn, mask=rn < N, other=0.0)[None, :]
    out = 0.5 * h * (1.0 + tl.erf(h * 0.7071067811865476))
    tl.store(y + rm[:, None].to(tl.int64) * N + rn[None, :], out,
             mask=mmask[:, None] & (rn[None, :] < N))


def _dot_precision(t):
    # TF32 tensor cores exist from Ampere (sm_80) on; T4 (sm_75) must use plain FP32 FMA.
    if t.is_cuda and torch.cuda.get_device_capability(t.device)[0] >= 8:
        return "tf32x3"
    return "ieee"


# x, gamma, beta, W, bias, y are tensors on the GPU
def solve(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, W: torch.Tensor,
          bias: torch.Tensor, y: torch.Tensor, M: int, K: int, N: int, eps: float):
    mean = torch.empty(M, device=x.device, dtype=torch.float32)
    rstd = torch.empty(M, device=x.device, dtype=torch.float32)
    row_stats_kernel[(M,)](x, mean, rstd, K, eps, BLOCK=min(triton.next_power_of_2(K), 4096))
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    ln_gemm_gelu_kernel[grid](x, mean, rstd, gamma, beta, W, bias, y, M, N, K, PREC=_dot_precision(x))


if __name__ == "__main__":
    from harness import DEVICE, ON_GPU, bench, check

    def ref(x, g, b, W, bias, eps):
        mu = x.mean(1, keepdim=True)
        var = ((x - mu) ** 2).mean(1, keepdim=True)
        h = ((x - mu) / torch.sqrt(var + eps) * g + b) @ W + bias
        return 0.5 * h * (1 + torch.erf(h / 2 ** 0.5))

    shapes = [(1, 1, 1), (3, 5, 7), (40, 70, 33), (65, 130, 100)] + ([(4096, 4096, 4096)] if ON_GPU else [])
    for M, K, N in shapes:
        x = torch.empty(M, K, device=DEVICE).uniform_(-1, 1) - 1000.0   # large common offset
        g = torch.empty(K, device=DEVICE).uniform_(0.5, 1.5)
        b = torch.empty(K, device=DEVICE).uniform_(-0.5, 0.5)
        W = torch.empty(K, N, device=DEVICE).uniform_(-1, 1) / K ** 0.5
        bias = torch.empty(N, device=DEVICE).uniform_(-0.5, 0.5)
        y = torch.empty(M, N, device=DEVICE)
        solve(x, g, b, W, bias, y, M, K, N, 1e-5)
        want = ref(*(t.double() for t in (x, g, b, W, bias)), 1e-5)
        check(f"M={M} K={K} N={N}", y, want, 1e-3, 1e-3)
    if ON_GPU:
        ln = torch.nn.functional.layer_norm
        torch_ref = lambda: torch.nn.functional.gelu(ln(x, (K,), g, b, 1e-5) @ W + bias)
        bench("ln+linear+gelu 4096^3", lambda: solve(x, g, b, W, bias, y, M, K, N, 1e-5),
              torch_ref, flops=2 * M * N * K)
