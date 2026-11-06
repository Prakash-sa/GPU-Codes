"""LeetGPU 2 - Matrix Multiplication: C = A @ B, float32, row-major
(A is M x N, B is N x K, C is M x K: the inner dimension is N; perf M = 8192, N = 6144,
K = 4096; atol = rtol = 1e-4).

2MNK = 412 GFLOP on 4 * (MN + NK + MK) = 436 MB, so ~945 FLOP per byte: compute bound
on every GPU. The plan is the standard tiled GEMM:
  * each program owns a BLOCK_M x BLOCK_N tile of C and walks the inner dimension in
    BLOCK_K steps with `tl.dot` (FP32 accumulator in registers);
  * grouped ("swizzled") program order, so programs running at the same time share A row
    panels and B column panels in L2;
  * `triton.autotune` picks the tile shape, num_warps and num_stages (pipelining depth).

Precision decides the speed. The inputs are FP32 and the tolerance is 1e-4:
  * plain TF32 tensor cores (10-bit mantissa) miss it by ~200x at the largest graded size;
  * "tf32x3" splits each operand into a high and a low TF32 part and does three tensor-core
    products: its error is the same size as FP32's own rounding (measured on the page);
  * "ieee" is exact FP32 on the CUDA cores: the only option on a T4 (no TF32 hardware).

Summation order matters as much. `acc = tl.dot(a, b, acc)` chains all 6,144 products
of an output into one accumulator, and at the largest graded size that chain alone
drifts past 1e-4 (measured on the page). `acc += tl.dot(a, b)` starts each BLOCK_K tile
from zero and adds the tile's partial sum: two short chains instead of one long one.

Comments that start with `mem:` say where the data lives: registers, shared memory,
the L1/L2 caches, or DRAM (GDDR6 on a T4, HBM3 on an H100). See memory-hierarchy.md.
"""
import os

import torch
import triton
import triton.language as tl

_CONFIGS = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=4),
]
if os.environ.get("TRITON_INTERPRET") == "1":
    _CONFIGS = [triton.Config({"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 16, "GROUP_M": 2})]


@triton.autotune(configs=_CONFIGS, key=["M", "N", "K"])
@triton.jit
def matmul_kernel(a, b, c, M, N, K,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                  GROUP_M: tl.constexpr, PREC: tl.constexpr):
    # Tile names follow C: rows m (of M), columns k (of K); the inner dimension is n (of N).
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    num_k = tl.cdiv(K, BLOCK_N)
    # Grouped order: GROUP_M row-tiles are walked column by column, so consecutive
    # programs reuse the same B column panel and a small set of A row panels from L2.
    group = pid // (GROUP_M * num_k)
    first_m = group * GROUP_M
    size_m = min(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % (GROUP_M * num_k)) % size_m
    pid_k = (pid % (GROUP_M * num_k)) // size_m

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)
    rn = tl.arange(0, BLOCK_K)
    a_ptrs = a + rm[:, None].to(tl.int64) * N + rn[None, :]
    b_ptrs = b + rn[:, None].to(tl.int64) * K + rk[None, :]
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)          # mem: registers
    for n0 in range(0, N, BLOCK_K):
        # mem: DRAM/L2 -> shared memory, num_stages tiles in flight (32 KB per stage at
        # 128x128x32 in FP32)
        a_t = tl.load(a_ptrs, mask=(rm[:, None] < M) & (rn[None, :] + n0 < N), other=0.0)
        b_t = tl.load(b_ptrs, mask=(rn[:, None] + n0 < N) & (rk[None, :] < K), other=0.0)
        # mem: tensor cores ("tf32x3": 3 MMAs on hi/lo parts) or FP32 FMA ("ieee").
        # A fresh partial per tile, then one add: keeps the FP32 error under 1e-4
        acc += tl.dot(a_t, b_t, input_precision=PREC)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * K
    # mem: registers -> DRAM, each C element written once
    tl.store(c + rm[:, None].to(tl.int64) * K + rk[None, :], acc, mask=(rm[:, None] < M) & (rk[None, :] < K))


def _dot_precision(t):
    # TF32 tensor cores exist from Ampere (sm_80) on; T4 (sm_75) must use plain FP32 FMA.
    if t.is_cuda and torch.cuda.get_device_capability(t.device)[0] >= 8:
        return "tf32x3"
    return "ieee"


# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int):
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(K, meta["BLOCK_N"]),)
    matmul_kernel[grid](a, b, c, M, N, K, PREC=_dot_precision(a))


if __name__ == "__main__":
    from harness import DEVICE, ON_GPU, bench, check

    shapes = [(1, 1, 1), (2, 3, 1), (1, 5, 3), (5, 3, 1), (8, 16, 32), (33, 70, 17), (64, 48, 80),
              (4, 6144, 16)]                                 # the graded inner length
    shapes += [(1024, 1024, 1024), (8192, 6144, 4096)] if ON_GPU else []
    for M, N, K in shapes:
        a = torch.empty(M, N, device=DEVICE).uniform_(-1, 1)
        b = torch.empty(N, K, device=DEVICE).uniform_(-1, 1)
        c = torch.empty(M, K, device=DEVICE)
        solve(a, b, c, M, N, K)
        check(f"{M}x{N}x{K}", c, a @ b, 1e-4, 1e-4)
    if ON_GPU:
        bench("matmul 8192x6144x4096", lambda: solve(a, b, c, M, N, K), lambda: torch.matmul(a, b),
              flops=2 * M * N * K)
