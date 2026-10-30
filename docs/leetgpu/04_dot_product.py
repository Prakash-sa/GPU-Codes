"""LeetGPU 17 - Dot Product: result[0] = sum(A * B) (float32, perf N = 4M).

The reduction pattern from 03_reduction.py with the multiply fused into the load loop:
A * B is never written to memory. Partials are combined in float64.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def partial_dot_kernel(a, b, partials, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(pid * BLOCK, N, nprog * BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < N
        acc += tl.load(a + offs, mask=mask, other=0.0) * tl.load(b + offs, mask=mask, other=0.0)
    tl.store(partials + pid, tl.sum(acc.to(tl.float64), axis=0))


@triton.jit
def final_sum_kernel(partials, out, P, BLOCK_P: tl.constexpr):
    offs = tl.arange(0, BLOCK_P)
    p = tl.load(partials + offs, mask=offs < P, other=0.0)
    tl.store(out, tl.sum(p, axis=0).to(tl.float32))


def _num_sms(t):
    return torch.cuda.get_device_properties(t.device).multi_processor_count if t.is_cuda else 4


# a, b, result are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, result: torch.Tensor, n: int):
    BLOCK = 4096
    P = max(1, min(triton.cdiv(n, BLOCK), 4 * _num_sms(a)))
    partials = torch.empty(P, device=a.device, dtype=torch.float64)
    partial_dot_kernel[(P,)](a, b, partials, n, BLOCK=BLOCK, num_warps=8)
    final_sum_kernel[(1,)](partials, result, P, BLOCK_P=triton.next_power_of_2(P))


if __name__ == "__main__":
    from harness import DEVICE, ON_GPU, bench, check

    sizes = [1, 31, 4097, 250_007] + ([4_194_304] if ON_GPU else [])
    for n in sizes:
        a = torch.empty(n, device=DEVICE).uniform_(-1, 1)
        b = torch.empty(n, device=DEVICE).uniform_(-1, 1)
        out = torch.zeros(1, device=DEVICE)
        solve(a, b, out, n)
        check(f"N={n}", out, (a.double() @ b.double()).float().reshape(1), 1e-5, 1e-5)
    bench("dot N=4M", lambda: solve(a, b, out, n), lambda: torch.dot(a, b),
          bytes_moved=2 * 4 * n)
