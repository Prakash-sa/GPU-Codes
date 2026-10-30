"""LeetGPU 70 - Segmented exclusive prefix sum (perf N = 50M, segments of 256).

    flags[i] = 1 starts a new segment;  out[i] = sum of values[s .. i-1] in i's segment

Key trick 1 - exclusive without subtraction. Let w[i] = 0 if flags[i] else values[i-1].
An *inclusive* segmented scan of w is exactly the exclusive scan of values (no
out[i] = incl[i] - values[i], which loses precision).

Key trick 2 - segmented scan is an ordinary scan with a different operator:
    (v1, f1) (+) (v2, f2) = (v2 if f2 else v1 + v2,  f1 | f2)
It is associative, so `tl.associative_scan` runs it in parallel within a block.

Across blocks (classic 3-phase "reduce-then-scan"):
  1. each block scans locally and writes its aggregate (tail sum, any-flag);
  2. one program scans the ~12K aggregates -> carry into each block (float64);
  3. each block rescans and adds the carry to elements before its first flag.

Precision: everything is scanned in float64. A segment can span the whole array
(N up to 100M), and a float32 running sum over a single 50K segment already drifts
by ~0.02, which fails the 1e-3 tolerance near zero crossings.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def seg_combine(v1, f1, v2, f2):
    return tl.where(f2 != 0, v2, v1 + v2), f1 | f2


@triton.jit
def _load_shifted(values, flags, offs, N):
    mask = offs < N
    f = tl.load(flags + offs, mask=mask, other=0)
    prev = tl.load(values + offs - 1, mask=mask & (offs > 0), other=0.0)
    return tl.where(f != 0, 0.0, prev).to(tl.float64), f


@triton.jit
def block_aggregate_kernel(values, flags, agg_v, agg_f, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    w, f = _load_shifted(values, flags, offs, N)
    sv, sf = tl.associative_scan((w, f), 0, seg_combine)
    last = tl.arange(0, BLOCK) == BLOCK - 1         # padded lanes are identity (0, 0)
    tl.store(agg_v + pid, tl.sum(tl.where(last, sv, 0.0), axis=0))   # float64
    tl.store(agg_f + pid, tl.max(tl.where(last, sf, 0), axis=0))


@triton.jit
def carry_scan_kernel(agg_v, agg_f, carry, NB, BLOCK: tl.constexpr):
    # carry[b] = segmented inclusive scan of the aggregates up to block b-1.
    tl.store(carry, 0.0)
    run = tl.zeros([1], dtype=tl.float64)
    idx = tl.arange(0, BLOCK)
    for start in range(0, NB, BLOCK):
        b = start + idx
        m = b < NB
        v = tl.load(agg_v + b, mask=m, other=0.0)
        f = tl.load(agg_f + b, mask=m, other=0)
        sv, sf = tl.associative_scan((v, f), 0, seg_combine)
        inc = tl.where(sf != 0, sv, run + sv)
        tl.store(carry + b + 1, inc, mask=m)
        run = tl.sum(tl.where(idx == BLOCK - 1, inc, 0.0), axis=0, keep_dims=True)


@triton.jit
def block_scan_kernel(values, flags, carry, out, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    w, f = _load_shifted(values, flags, offs, N)
    sv, sf = tl.associative_scan((w, f), 0, seg_combine)
    c = tl.load(carry + pid)
    res = tl.where(sf != 0, sv, c + sv)
    tl.store(out + offs, res.to(tl.float32), mask=offs < N)


# values, flags, output are tensors on the GPU
def solve(values: torch.Tensor, flags: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK = 4096
    NB = triton.cdiv(N, BLOCK)
    dev = values.device
    agg_v = torch.empty(NB, device=dev, dtype=torch.float64)
    agg_f = torch.empty(NB, device=dev, dtype=torch.int32)
    carry = torch.empty(NB + 1, device=dev, dtype=torch.float64)
    block_aggregate_kernel[(NB,)](values, flags, agg_v, agg_f, N, BLOCK=BLOCK, num_warps=8)
    carry_scan_kernel[(1,)](agg_v, agg_f, carry, NB, BLOCK=1024, num_warps=4)
    block_scan_kernel[(NB,)](values, flags, carry, output, N, BLOCK=BLOCK, num_warps=8)


if __name__ == "__main__":
    from harness import DEVICE, ON_GPU, bench, check

    def ref(values, flags):
        v = values.double()
        excl = torch.zeros_like(v)
        excl[1:] = torch.cumsum(v[:-1], 0)
        seg = torch.cumsum(flags.long(), 0) - 1
        return (excl - excl[flags.bool()][seg]).float()

    def make(n, p_flag):
        v = torch.empty(n, device=DEVICE).uniform_(-100, 100)
        f = (torch.rand(n, device=DEVICE) < p_flag).int()
        f[0] = 1
        return v, f

    cases = [(1, 0.5), (2, 0.0), (5, 1.0), (4095, 0.01), (4097, 0.0), (9000, 0.3), (50_000, 0.0)]
    cases += [(50_000_000, 1 / 256), (20_000_000, 0.0)] if ON_GPU else []
    for n, p in cases:
        v, f = make(n, p)
        out = torch.empty(n, device=DEVICE)
        solve(v, f, out, n)
        check(f"N={n} p(flag)={p:.3g}", out, ref(v, f), 1e-3, 1e-3)
    if ON_GPU:
        v, f = make(50_000_000, 1 / 256)
        out = torch.empty_like(v)
        bench("segmented scan N=50M", lambda: solve(v, f, out, v.numel()),
              lambda: ref(v, f), bytes_moved=12 * v.numel())
