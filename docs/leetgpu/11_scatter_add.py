"""LeetGPU 125 - Scatter Add: output[index[i], :] += src[i, :]
(src N x D float32, index int32; perf N = 1M, D = 128, M = 50K).

This is message aggregation in a graph neural network: edges -> nodes.

solve() - atomics. Each program takes BLOCK_N source rows x BLOCK_D columns and issues
    one `tl.atomic_add` per element. Rows are contiguous along D, so the atomics are
    coalesced, and with M = 50K targets collisions are spread thin. Summation order
    varies between runs, so the result is not bit-reproducible (it still meets 1e-4).

solve_sorted() - deterministic. Sort rows by target once (torch.sort, outside the
    kernel), then one program per output row adds its run of sources in a fixed order:
    no atomics, bit-reproducible. Worth it when the same graph is reused every step.
    Uses torch for the sort, so it is for study, not for LeetGPU.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_kernel(src, index, out, N, D, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    cols = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    rmask = rows < N
    mask = rmask[:, None] & (cols[None, :] < D)
    tgt = tl.load(index + rows, mask=rmask, other=0).to(tl.int64)
    vals = tl.load(src + rows[:, None].to(tl.int64) * D + cols[None, :], mask=mask, other=0.0)
    tl.atomic_add(out + tgt[:, None] * D + cols[None, :], vals, mask=mask, sem="relaxed")


# src, index, output are tensors on the GPU
def solve(src: torch.Tensor, index: torch.Tensor, output: torch.Tensor, N: int, D: int, M: int):
    output.zero_()
    BLOCK_D = min(triton.next_power_of_2(D), 128)
    BLOCK_N = max(1, 4096 // BLOCK_D)
    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(D, BLOCK_D))
    scatter_add_kernel[grid](src, index, output, N, D, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D, num_warps=4)


@triton.jit
def segment_sum_kernel(src, perm, row_ptr, out, D, BLOCK_D: tl.constexpr):
    m = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    cmask = cols < D
    start = tl.load(row_ptr + m)
    end = tl.load(row_ptr + m + 1)
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for p in range(start, end):                    # fixed order -> reproducible
        r = tl.load(perm + p).to(tl.int64)
        acc += tl.load(src + r * D + cols, mask=cmask, other=0.0)
    tl.store(out + m.to(tl.int64) * D + cols, acc, mask=cmask)


def solve_sorted(src, index, output, N, D, M):
    idx = index.long()
    order = torch.sort(idx, stable=True).indices.to(torch.int32)
    row_ptr = torch.zeros(M + 1, device=src.device, dtype=torch.int64)
    row_ptr[1:] = torch.cumsum(torch.bincount(idx, minlength=M), 0)
    BLOCK_D = min(triton.next_power_of_2(D), 128)
    segment_sum_kernel[(M, triton.cdiv(D, BLOCK_D))](src, order, row_ptr, output, D, BLOCK_D=BLOCK_D)


if __name__ == "__main__":
    from harness import DEVICE, ON_GPU, bench, check

    def ref(src, index, M):
        return torch.zeros(M, src.shape[1], device=src.device).index_add_(0, index.long(), src)

    cases = [(1, 1, 1), (4, 3, 3), (2, 5, 1), (1000, 37, 50), (3000, 200, 7)]
    cases += [(1_000_000, 128, 50_000)] if ON_GPU else []
    for N, D, M in cases:
        src = torch.empty(N, D, device=DEVICE).uniform_(-10, 10)
        index = torch.randint(0, M, (N,), device=DEVICE, dtype=torch.int32)
        want = ref(src, index, M)
        for fn in (solve, solve_sorted):
            out = torch.full((M, D), 123.0, device=DEVICE)   # stale data must be cleared
            fn(src, index, out, N, D, M)
            check(f"{fn.__name__} N={N} D={D} M={M}", out, want, 1e-4, 1e-4)
    if ON_GPU:
        bench("scatter_add atomics", lambda: solve(src, index, out, N, D, M),
              lambda: ref(src, index, M), bytes_moved=4 * (N * D + N + M * D))
        bench("scatter_add sorted (+sort)", lambda: solve_sorted(src, index, out, N, D, M))
