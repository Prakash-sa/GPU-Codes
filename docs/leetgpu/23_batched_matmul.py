"""LeetGPU 30 - Batched Matrix Multiplication: C[b] = A[b] @ B[b], float32, row-major
(A is BATCH x M x K, B is BATCH x K x N, C is BATCH x M x N; 1 <= BATCH <= 128,
1 <= M, N, K <= 1024; perf BATCH = 32, M = N = K = 256; atol = rtol = 1e-5).

The FP32 sibling of 10 (FP16 batched matmul), with two differences that matter:

  * Precision. 1e-5 is at FP32's own rounding noise: plain TF32 misses it by ~500x, and
    even exact FP32 summed in a different order than PyTorch misses it at 1024^3. So the
    dot uses input_precision="ieee" (FP32 FMA on the CUDA cores) and keeps the error
    budget for summation order.
  * Size. One 256^3 product is 33.5 MFLOP, far too little for a GPU. All 32 products go
    in one launch: grid axis 1 is the batch, so 32 x (tiles per product) programs run
    together. Tile size trades per-program efficiency for program count: at 64 x 64 there
    are 32 x 16 = 512 programs, enough to fill 132 SMs several times.

Batch offsets are computed in int64 (128 x 1024 x 1024 floats is past 2^31).

Comments that start with `mem:` say where the data lives: registers, shared memory,
the L1/L2 caches, or DRAM (GDDR6 on a T4, HBM3 on an H100). See memory-hierarchy.md.
"""
import os

import torch
import triton
import triton.language as tl

_CONFIGS = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 16}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=2, num_stages=3),
]
if os.environ.get("TRITON_INTERPRET") == "1":
    _CONFIGS = [triton.Config({"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 16})]


@triton.autotune(configs=_CONFIGS, key=["M", "N", "K"])
@triton.jit
def bmm_fp32_kernel(a, b, c, M, N, K,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    batch = tl.program_id(1).to(tl.int64)
    num_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_n
    pid_n = pid % num_n

    a += batch * M * K                                   # int64: no overflow at 128 x 1024^2
    b += batch * K * N
    c += batch * M * N
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    a_ptrs = a + rm[:, None] * K + rk[None, :]
    b_ptrs = b + rk[:, None] * N + rn[None, :]

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)          # mem: registers
    for k in range(0, K, BLOCK_K):
        # mem: DRAM/L2 -> shared memory, pipelined num_stages deep
        a_t = tl.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] + k < K), other=0.0)
        b_t = tl.load(b_ptrs, mask=(rk[:, None] + k < K) & (rn[None, :] < N), other=0.0)
        # mem: CUDA cores, exact FP32 FMA (no TF32: the tolerance is 1e-5)
        acc = tl.dot(a_t, b_t, acc, input_precision="ieee")
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N
    # mem: registers -> DRAM
    tl.store(c + rm[:, None] * N + rn[None, :], acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, BATCH: int, M: int, N: int, K: int):
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]), BATCH)
    bmm_fp32_kernel[grid](a, b, c, M, N, K)


if __name__ == "__main__":
    from harness import DEVICE, ON_GPU, bench, check

    shapes = [(2, 2, 2, 3), (1, 3, 3, 3), (4, 2, 2, 2), (8, 4, 3, 2), (16, 16, 16, 16), (2, 8, 6, 4),
              (3, 33, 17, 40)]
    shapes += [(32, 256, 256, 256)] if ON_GPU else []
    for B, M, N, K in shapes:
        a = torch.empty(B, M, K, device=DEVICE).uniform_(-1, 1)
        b = torch.empty(B, K, N, device=DEVICE).uniform_(-1, 1)
        c = torch.empty(B, M, N, device=DEVICE)
        solve(a, b, c, B, M, N, K)
        check(f"{B}x{M}x{N}x{K}", c, torch.bmm(a, b), 1e-5, 1e-5)
    if ON_GPU:
        bench("bmm fp32 32x256^3", lambda: solve(a, b, c, B, M, N, K), lambda: torch.bmm(a, b),
              flops=2 * B * M * N * K)
