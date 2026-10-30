"""LeetGPU 22 - GEMM: C = alpha * (A @ B) + beta * C   (FP16 in/out, perf M = N = K = 1024).

Compute bound (2MNK FLOPs on ~6 bytes per output element), so the goal is to keep the
tensor cores busy:
  * each program owns a BLOCK_M x BLOCK_N tile of C and loops over K with `tl.dot`
    (FP16 inputs, FP32 accumulator);
  * grouped ("swizzled") program order so neighbouring programs reuse the same A rows
    and B columns from L2;
  * the alpha/beta epilogue is fused: C is read once and written once;
  * `triton.autotune` picks the tile shape, num_warps and num_stages (software
    pipelining depth) per (M, N, K).
"""
import os

import torch
import triton
import triton.language as tl

_CONFIGS = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=5),
]
if os.environ.get("TRITON_INTERPRET") == "1":    # the CPU interpreter cannot benchmark configs
    _CONFIGS = _CONFIGS[-1:]


@triton.autotune(configs=_CONFIGS, key=["M", "N", "K"])
@triton.jit
def gemm_kernel(a, b, c, M, N, K, alpha, beta,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                GROUP_M: tl.constexpr):
    # Grouped ordering: walk GROUP_M tile-rows down a column of tiles before moving right.
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
    a_ptrs = a + rm[:, None] * K + rk[None, :]
    b_ptrs = b + rk[:, None] * N + rn[None, :]

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a_t = tl.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] + k < K), other=0.0)
        b_t = tl.load(b_ptrs, mask=(rk[:, None] + k < K) & (rn[None, :] < N), other=0.0)
        acc = tl.dot(a_t, b_t, acc)                  # FP16 x FP16 -> FP32 on tensor cores
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N

    cmask = (rm[:, None] < M) & (rn[None, :] < N)
    c_ptrs = c + rm[:, None] * N + rn[None, :]
    c_old = tl.load(c_ptrs, mask=cmask, other=0.0).to(tl.float32)
    tl.store(c_ptrs, (alpha * acc + beta * c_old).to(tl.float16), mask=cmask)


# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int,
          alpha: float, beta: float):
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    gemm_kernel[grid](a, b, c, M, N, K, alpha, beta)


if __name__ == "__main__":
    from harness import DEVICE, ON_GPU, bench, check

    def ref(a, b, c, alpha, beta):
        return (alpha * (a.float() @ b.float()) + beta * c.float()).half()

    shapes = [(16, 16, 16), (17, 33, 50), (64, 48, 100), (100, 70, 129)]
    shapes += [(1024, 1024, 1024), (4096, 4096, 4096)] if ON_GPU else []
    for M, N, K in shapes:
        dt = torch.float16 if ON_GPU else torch.float32   # the interpreter's dot wants fp32
        a = torch.empty(M, K, device=DEVICE).uniform_(-1, 1).to(dt)
        b = torch.empty(K, N, device=DEVICE).uniform_(-1, 1).to(dt)
        c = torch.empty(M, N, device=DEVICE).uniform_(-1, 1).to(dt)
        want = ref(a, b, c, 0.5, 1.5)
        solve(a, b, c, M, N, K, 0.5, 1.5)
        check(f"{M}x{N}x{K}", c, want, 0.05, 0.05)
    if ON_GPU:
        M = N = K = 1024
        a, b, c = (torch.randn(M, K, device=DEVICE, dtype=torch.float16) for _ in range(3))
        bench("gemm 1024^3", lambda: solve(a, b, c, M, N, K, 1.0, 1.0),
              lambda: torch.addmm(c, a, b), flops=2 * M * N * K)
