"""LeetGPU 57 - FP16 Batched MatMul: C[b] = A[b] @ B[b]  (perf BATCH = 32, M = N = K = 256).

Each 256^3 matmul is only 64 tiles of 32x32, too small to fill an H100 on its own. One
launch covers all batches: grid axis 1 is the batch index, so 32 x tiles programs run
together instead of 32 separate under-filled launches. Batch offsets are int64.

Comments that start with `mem:` say where the data lives: registers, shared memory,
the L1/L2 caches, or DRAM (GDDR6 on a T4, HBM3 on an H100). See memory-hierarchy.md.
"""
import os

import torch
import triton
import triton.language as tl

_CONFIGS = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 64}, num_warps=2, num_stages=4),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=2, num_stages=3),
]
if os.environ.get("TRITON_INTERPRET") == "1":
    _CONFIGS = _CONFIGS[-1:]


@triton.autotune(configs=_CONFIGS, key=["M", "N", "K"])
@triton.jit
def bmm_kernel(a, b, c, M, N, K,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    batch = tl.program_id(1).to(tl.int64)
    num_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_n
    pid_n = pid % num_n

    a += batch * M * K
    b += batch * K * N
    c += batch * M * N
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    a_ptrs = a + rm[:, None] * K + rk[None, :]
    b_ptrs = b + rk[:, None] * N + rn[None, :]

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        # mem: DRAM -> shared memory, num_stages tiles; one batch's A and B (128 KB each) stay in L2
        a_t = tl.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] + k < K), other=0.0)
        b_t = tl.load(b_ptrs, mask=(rk[:, None] + k < K) & (rn[None, :] < N), other=0.0)
        # mem: tensor cores; the fp32 accumulator is in registers
        acc = tl.dot(a_t, b_t, acc)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N

    mask = (rm[:, None] < M) & (rn[None, :] < N)
    # mem: registers -> DRAM, written once
    tl.store(c + rm[:, None] * N + rn[None, :], acc.to(c.dtype.element_ty), mask=mask)


# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, BATCH: int, M: int, N: int, K: int):
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]), BATCH)
    bmm_kernel[grid](a, b, c, M, N, K)


if __name__ == "__main__":
    from harness import DEVICE, ON_GPU, bench, check

    shapes = [(1, 1, 1, 1), (3, 17, 33, 9), (2, 64, 40, 100)] + ([(32, 256, 256, 256)] if ON_GPU else [])
    for B, M, N, K in shapes:
        dt = torch.float16 if ON_GPU else torch.float32
        a = torch.empty(B, M, K, device=DEVICE).uniform_(-5, 5).to(dt)
        b = torch.empty(B, K, N, device=DEVICE).uniform_(-5, 5).to(dt)
        c = torch.empty(B, M, N, device=DEVICE, dtype=dt)
        solve(a, b, c, B, M, N, K)
        check(f"{B}x{M}x{N}x{K}", c, torch.bmm(a.float(), b.float()).to(dt), 0.05, 0.05)
    bench("bmm 32x256^3", lambda: solve(a, b, c, B, M, N, K), lambda: torch.bmm(a, b),
          flops=2 * B * M * N * K)
