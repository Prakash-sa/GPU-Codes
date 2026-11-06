"""LeetGPU 18 - Sparse Matrix-Vector Multiplication: y = A @ x, float32
(A is M x N row-major with ~60-70% zeros, *stored dense*; perf M = 1,000, N = 10,000,
nnz = 3.5M; atol = rtol = 1e-3).

Question the word "sparse". A arrives as a dense array, so finding its non-zeros means
reading every element of A anyway, and reading A is the whole cost:
  * bytes: 4 * M * N = 40 MB of A, plus 40 KB of x and 4 KB of y;
  * FLOPs: 2 * M * N = 20 MFLOP, i.e. 0.5 FLOP per byte: deep in the memory-bound zone.
Skipping the zeros saves multiply-adds, which cost nothing here, and no bytes. Building
CSR first reads A once and *then* reads the CSR arrays (8 bytes per non-zero: value +
column), so it can only be slower for a single product. The kernel is therefore a dense
GEMV that streams A once at full bandwidth.

  * one program per row: 1,000 programs at the perf size, each streaming a 40 KB row in
    coalesced BLOCK_N chunks and reducing with `tl.sum`;
  * x (40 KB) is re-read by every row, from L2 (and L1), never from DRAM after the first;
  * accumulation in FP32 per lane, then one tree reduction: deterministic.

solve_csr() is the sparse path, kept to measure the difference. It builds CSR with
torch.nonzero, so it is for study, not for LeetGPU.

Comments that start with `mem:` say where the data lives: registers, shared memory,
the L1/L2 caches, or DRAM (GDDR6 on a T4, HBM3 on an H100). See memory-hierarchy.md.
"""
import os

import torch
import triton
import triton.language as tl

_INTERP = os.environ.get("TRITON_INTERPRET") == "1"


@triton.jit
def gemv_row_kernel(A, x, y, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    a_row = A + row * N
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)          # mem: registers, one partial per lane
    for n0 in range(0, N, BLOCK_N):
        cols = n0 + tl.arange(0, BLOCK_N)
        m = cols < N
        # mem: DRAM -> registers, streamed once, coalesced (the 40 MB that set the time)
        a = tl.load(a_row + cols, mask=m, other=0.0, eviction_policy="evict_first")
        # mem: L2/L1 -> registers; x is 40 KB and every row re-reads it from cache
        xv = tl.load(x + cols, mask=m, other=0.0, eviction_policy="evict_last")
        acc += a * xv
    # mem: registers -> DRAM, one float per row
    tl.store(y + row, tl.sum(acc, axis=0))


# A, x, y are tensors on the GPU
def solve(A: torch.Tensor, x: torch.Tensor, y: torch.Tensor, M: int, N: int, nnz: int):
    BLOCK_N = min(triton.next_power_of_2(N), 64 if _INTERP else 1024)
    gemv_row_kernel[(M,)](A, x, y, N, BLOCK_N=BLOCK_N, num_warps=4)


@triton.jit
def csr_spmv_kernel(row_ptr, col_idx, vals, x, y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    start = tl.load(row_ptr + row)
    end = tl.load(row_ptr + row + 1)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for p0 in range(start, end, BLOCK):
        p = p0 + tl.arange(0, BLOCK)
        m = p < end
        # mem: 8 bytes per non-zero (value + column), then a *gather* of x by column
        v = tl.load(vals + p, mask=m, other=0.0)
        c = tl.load(col_idx + p, mask=m, other=0)
        acc += v * tl.load(x + c, mask=m, other=0.0)
    tl.store(y + row, tl.sum(acc, axis=0))


def solve_csr(A, x, y, M, N, nnz):
    A2 = A.view(M, N)
    nz = A2.nonzero()                                    # a full read of A, plus a scan
    row_ptr = torch.zeros(M + 1, device=A.device, dtype=torch.int64)
    row_ptr[1:] = torch.cumsum(torch.bincount(nz[:, 0], minlength=M), 0)
    col_idx = nz[:, 1].to(torch.int32).contiguous()
    vals = A2[nz[:, 0], nz[:, 1]].contiguous()
    csr_spmv_kernel[(M,)](row_ptr, col_idx, vals, x, y, BLOCK=64 if _INTERP else 512, num_warps=4)


if __name__ == "__main__":
    from harness import DEVICE, ON_GPU, bench, check

    shapes = [(1, 1), (2, 2), (3, 4), (10, 8), (20, 20), (7, 130), (33, 1000)]
    shapes += [(1000, 10000), (10000, 10000)] if ON_GPU else []
    for M, N in shapes:
        A = torch.empty(M, N, device=DEVICE).uniform_(-10, 10)
        A *= torch.rand(M, N, device=DEVICE) > 0.65
        x = torch.empty(N, device=DEVICE).uniform_(-5, 5)
        nnz = int((A != 0).sum())
        for fn in (solve, solve_csr):
            y = torch.empty(M, device=DEVICE)
            fn(A, x, y, M, N, nnz)
            check(f"{fn.__name__} {M}x{N}", y, A @ x, 1e-3, 1e-3)
    if ON_GPU:
        M, N = 1000, 10000
        A = torch.empty(M, N, device=DEVICE).uniform_(-10, 10)
        A *= torch.rand(M, N, device=DEVICE) > 0.65
        x = torch.empty(N, device=DEVICE).uniform_(-5, 5)
        y = torch.empty(M, device=DEVICE)
        nnz = int((A != 0).sum())
        bench("spmv dense GEMV", lambda: solve(A, x, y, M, N, nnz), lambda: A @ x,
              bytes_moved=4 * (M * N + N + M))
        bench("spmv CSR (incl. conversion)", lambda: solve_csr(A, x, y, M, N, nnz))
