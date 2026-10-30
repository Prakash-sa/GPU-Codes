"""Shared correctness and benchmark helpers for the LeetGPU solutions.

Every solution file is self-contained: its kernels and `solve()` can be pasted into
LeetGPU unchanged. Only the `__main__` test block imports this module.

    python docs/leetgpu/05_softmax.py                     # on a CUDA machine
    PROFILE=1 python docs/leetgpu/05_softmax.py           # + torch.profiler table and trace
    TRITON_INTERPRET=1 python docs/leetgpu/05_softmax.py  # no GPU: runs kernels on the CPU

The interpreter is slow, so the tests use small, awkward sizes there and only run the
LeetGPU performance size (and the benchmark) on a real GPU.
"""
import os

import torch

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
PROFILE = os.environ.get("PROFILE") == "1"
DEVICE = "cuda" if torch.cuda.is_available() and not INTERPRET else "cpu"
ON_GPU = DEVICE == "cuda"


def check(name, got, want, atol, rtol):
    """Compare like LeetGPU does (torch.allclose) and print the worst error."""
    if got.dtype in (torch.int32, torch.int64):
        ok = torch.equal(got.long(), want.long())
        err = (got.long() != want.long()).sum().item()
        msg = f"{err} mismatches"
    else:
        ok = torch.allclose(got.float(), want.float(), atol=atol, rtol=rtol)
        err = (got.double() - want.double()).abs().max().item() if got.numel() else 0.0
        msg = f"max|err| = {err:.2e}"
    print(f"  {'ok  ' if ok else 'FAIL'} {name:<34} {msg}")
    if not ok:
        raise AssertionError(f"{name}: {msg} (atol={atol}, rtol={rtol})")


def bench(name, fn, ref_fn=None, bytes_moved=None, flops=None):
    """Median time with triton.testing.do_bench; prints GB/s, TFLOP/s and speedup vs ref."""
    if not ON_GPU:
        print(f"  (no GPU: skipping benchmark for {name})")
        return
    from triton.testing import do_bench

    ms = do_bench(fn, warmup=25, rep=100)
    line = f"  {name:<34} {ms * 1e3:9.1f} us"
    if bytes_moved:
        line += f"  {bytes_moved / ms / 1e6:8.1f} GB/s"
    if flops:
        line += f"  {flops / ms / 1e9:7.2f} TFLOP/s"
    if ref_fn is not None:
        ref_ms = do_bench(ref_fn, warmup=25, rep=100)
        line += f"  | torch {ref_ms * 1e3:9.1f} us  ({ref_ms / ms:.2f}x)"
    print(line)
    if PROFILE:
        profile(name, fn)


def profile(name, fn, iters=10):
    """Kernel-level breakdown: which kernels ran, how many launches, CUDA time each.

    Also writes a Chrome/Perfetto trace (open in chrome://tracing or ui.perfetto.dev)
    to see gaps between kernels, i.e. launch or host overhead.
    """
    from torch.profiler import ProfilerActivity, profile as torch_profile

    fn()                                           # compile/autotune outside the trace
    torch.cuda.synchronize()
    with torch_profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=8))
    trace = f"trace_{name.split()[0]}.json"
    prof.export_chrome_trace(trace)
    print(f"  trace written to {trace}")


def sm_count():
    """Number of SMs, used to size persistent grids. A placeholder value on the CPU."""
    return torch.cuda.get_device_properties(0).multi_processor_count if ON_GPU else 4
