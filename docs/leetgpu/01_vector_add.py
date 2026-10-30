"""LeetGPU 1 - Vector Addition: C = A + B (float32, N up to 100M; perf N = 25M).

Bandwidth bound: 12 bytes moved per FLOP. One program per BLOCK_SIZE elements,
contiguous offsets so every load/store is fully coalesced.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(a, b, c, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements                      # the last block is usually partial
    x = tl.load(a + offs, mask=mask)
    y = tl.load(b + offs, mask=mask)
    tl.store(c + offs, x + y, mask=mask)


# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    vector_add_kernel[grid](a, b, c, N, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)


if __name__ == "__main__":
    from harness import DEVICE, ON_GPU, bench, check

    sizes = [1, 31, 33, 1023, 1025, 100_003] + ([25_000_000] if ON_GPU else [])
    for n in sizes:
        a = torch.empty(n, device=DEVICE).uniform_(-1000, 1000)
        b = torch.empty(n, device=DEVICE).uniform_(-1000, 1000)
        c = torch.empty(n, device=DEVICE)
        solve(a, b, c, n)
        check(f"N={n}", c, a + b, 1e-5, 1e-5)
    bench("vector_add N=25M", lambda: solve(a, b, c, n), lambda: torch.add(a, b, out=c),
          bytes_moved=3 * 4 * n)
