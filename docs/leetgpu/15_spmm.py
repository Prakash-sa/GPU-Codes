"""LeetGPU 75 - Sparse (M x N, ~65% zeros) x Dense (N x K) matmul, float32
(perf M = 4096, N = 2048, K = 512; atol = rtol = 1e-3).

The "sparse" A arrives as a *dense* row-major array. At ~35% density a sparse format
does not pay for itself here:
  * building CSR costs a full pass over A plus a prefix sum (no free lunch);
  * CSR SpMM gathers rows of B by column index: irregular, no tensor cores;
  * dense GEMM runs on tensor cores and reads A exactly once.
The rule of thumb is that CSR beats dense below roughly 1-5% density, not at 35%.

So solve() is a tiled dense GEMM. A and B are FP32, and plain TF32 tensor cores (10-bit
mantissa) would miss a 1e-3 tolerance on 2048-long dot products. `input_precision=
"tf32x3"` splits each operand into big + small TF32 parts and does three tensor-core
products, which gets close to FP32 accuracy at well above the FP32 CUDA-core rate.
T4 (sm_75) has no TF32 tensor cores, so _dot_precision() falls back to "ieee" (FP32 FMA).

solve_csr() is the sparse version, for comparison and for genuinely sparse inputs.
It builds CSR with torch.nonzero, so it is for study, not for LeetGPU.

Comments that start with `mem:` say where the data lives: registers, shared memory,
the L1/L2 caches, or DRAM (GDDR6 on a T4, HBM3 on an H100). See memory-hierarchy.md.
"""
import os

import torch
import triton
import triton.language as tl

_CONFIGS = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=4),
]
if os.environ.get("TRITON_INTERPRET") == "1":
    _CONFIGS = _CONFIGS[-1:]


@triton.autotune(configs=_CONFIGS, key=["M", "N", "K"])
@triton.jit
def dense_mm_kernel(a, b, c, M, N, K,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, PREC: tl.constexpr):
    # C[M, K] = A[M, N] @ B[N, K]; tile names follow C's dims (rows m, cols k), inner n.
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)
    rn = tl.arange(0, BLOCK_K)
    a_ptrs = a + rm[:, None].to(tl.int64) * N + rn[None, :]
    b_ptrs = b + rn[:, None].to(tl.int64) * K + rk[None, :]
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for n0 in range(0, N, BLOCK_K):
        # mem: DRAM -> shared memory, num_stages fp32 tiles: 32 KB per stage at 128x128x32.
        # Fits the H100's 228 KB; the T4's 64 KB only takes the smaller configs
        a_t = tl.load(a_ptrs, mask=(rm[:, None] < M) & (rn[None, :] + n0 < N), other=0.0)
        b_t = tl.load(b_ptrs, mask=(rn[:, None] + n0 < N) & (rk[None, :] < K), other=0.0)
        # mem: tensor cores (tf32x3 splits hi/lo in registers: 3 MMAs) or FP32 FMA
        # on T4; acc stays in registers
        acc = tl.dot(a_t, b_t, acc, input_precision=PREC)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * K
    mask = (rm[:, None] < M) & (rk[None, :] < K)
    # mem: registers -> DRAM
    tl.store(c + rm[:, None].to(tl.int64) * K + rk[None, :], acc, mask=mask)


def _dot_precision(t):
    # TF32 tensor cores exist from Ampere (sm_80) on; T4 (sm_75) must use plain FP32 FMA.
    if t.is_cuda and torch.cuda.get_device_capability(t.device)[0] >= 8:
        return "tf32x3"
    return "ieee"


# A, B, C are tensors on the GPU
def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, M: int, N: int, K: int, nnz: int):
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(K, meta["BLOCK_N"]))
    dense_mm_kernel[grid](A, B, C, M, N, K, PREC=_dot_precision(A))


@triton.jit
def csr_spmm_kernel(row_ptr, col_idx, vals, b, c, K, BLOCK_K: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    cmask = cols < K
    start = tl.load(row_ptr + row)
    end = tl.load(row_ptr + row + 1)
    acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    for p in range(start, end):                    # only the non-zeros of this row
        # mem: CSR metadata: scalar loads via L2
        col = tl.load(col_idx + p).to(tl.int64)
        # mem: gather a row of B. The rows are random, so this relies on B (4 MB) staying in L2
        acc += tl.load(vals + p) * tl.load(b + col * K + cols, mask=cmask, other=0.0)
    # mem: registers -> DRAM
    tl.store(c + row.to(tl.int64) * K + cols, acc, mask=cmask)


def solve_csr(A, B, C, M, N, K, nnz):
    A2 = A.view(M, N)
    nz = A2.nonzero()
    row_ptr = torch.zeros(M + 1, device=A.device, dtype=torch.int64)
    row_ptr[1:] = torch.cumsum(torch.bincount(nz[:, 0], minlength=M), 0)
    col_idx = nz[:, 1].to(torch.int32).contiguous()
    vals = A2[nz[:, 0], nz[:, 1]].contiguous()
    BLOCK_K = min(triton.next_power_of_2(K), 256)
    csr_spmm_kernel[(M, triton.cdiv(K, BLOCK_K))](row_ptr, col_idx, vals, B, C, K, BLOCK_K=BLOCK_K)


if __name__ == "__main__":
    from harness import DEVICE, ON_GPU, bench, check

    shapes = [(1, 1, 1), (5, 7, 3), (33, 70, 17), (100, 130, 65)]
    shapes += [(4096, 2048, 512)] if ON_GPU else []
    for M, N, K in shapes:
        A = torch.empty(M, N, device=DEVICE).uniform_(-10, 10)
        A *= torch.rand(M, N, device=DEVICE) > 0.65
        B = torch.empty(N, K, device=DEVICE).uniform_(-10, 10)
        want = (A.double() @ B.double()).float()
        for fn in (solve, solve_csr):
            C = torch.empty(M, K, device=DEVICE)
            fn(A, B, C, M, N, K, int((A != 0).sum()))
            check(f"{fn.__name__} {M}x{N}x{K}", C, want, 1e-3, 1e-3)
    if ON_GPU:
        nnz = int((A != 0).sum())
        bench("spmm dense tf32x3", lambda: solve(A, B, C, M, N, K, nnz),
              lambda: torch.matmul(A, B), flops=2 * M * N * K)
        bench("spmm CSR (incl. conversion)", lambda: solve_csr(A, B, C, M, N, K, nnz))
