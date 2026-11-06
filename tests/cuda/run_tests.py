"""Test the CUDA C++ solutions, with or without a GPU.

Without a GPU (default): each .cu file is rewritten (kernel<<<grid, block>>>(args) ->
cuda_cpu::launch(...)), compiled as C++20 against cuda_cpu.h, loaded with ctypes, and its
solve() is compared with a PyTorch reference on the same cases as the Triton tests.
With --gpu: the same files are built with nvcc and run on CUDA tensors.

    docker run --rm -v "$PWD":/w -w /w cuda-cpu-test python tests/cuda/run_tests.py [name-filter]
    python tests/cuda/run_tests.py --gpu [name-filter]          # on a machine with nvcc + a GPU

The image is python:3.11-slim plus g++ and CPU PyTorch (see the page "Running the CUDA
versions"). tests/cuda/nvcc_check.sh compiles the same files with the real nvcc.
"""
import ctypes
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BUILD = Path(tempfile.gettempdir()) / "cuda_cpu_build"
FAILED = []
GPU = "--gpu" in sys.argv


# ---------------------------------------------------------------- build

def _match(src, i, open_ch, close_ch):
    depth = 0
    for k in range(i, len(src)):
        depth += src[k] == open_ch
        depth -= src[k] == close_ch
        if depth == 0:
            return k
    raise ValueError("unbalanced")


def _split_top(s):
    parts, depth, cur = [], 0, ""
    for ch in s:
        depth += ch in "([{"
        depth -= ch in ")]}"
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    return parts + [cur]


def to_cpp(src):
    """CUDA source -> C++ for the CPU emulator."""
    src = re.sub(r"#include\s*<(cuda(_runtime|_fp16)?\.h|cooperative_groups(/\w+)?\.h)>\s*\n", "", src)
    out, i = [], 0
    while (k := src.find("<<<", i)) >= 0:
        j, depth = k, 0                                   # walk back over name<template args>
        while j > 0:
            ch = src[j - 1]
            if ch == ">":
                depth += 1
            elif ch == "<":
                depth -= 1
            elif depth == 0 and not (ch.isalnum() or ch in "_:"):
                break
            j -= 1
        e = src.index(">>>", k)
        grid, block = (p.strip() for p in _split_top(src[k + 3:e])[:2])
        p = src.index("(", e)
        q = _match(src, p, "(", ")")
        out.append(src[i:j])
        out.append(f"cuda_cpu::launch(dim3({grid}), dim3({block}), {src[j:k].strip()}, {src[p + 1:q]})")
        i = q + 1
    out.append(src[i:])
    return '#include "cuda_cpu.h"\n' + "".join(out)


def build(cu):
    BUILD.mkdir(exist_ok=True)
    lib = BUILD / (cu.stem + ".so")
    if GPU:
        subprocess.run(["nvcc", "-O3", "-arch=native", "-shared", "-Xcompiler", "-fPIC",
                        "-o", str(lib), str(cu)], check=True)
        return ctypes.CDLL(str(lib))
    cpp = BUILD / (cu.stem + ".cpp")
    cpp.write_text(to_cpp(cu.read_text()))
    subprocess.run(["g++", "-std=c++20", "-O2", "-ffp-contract=off", "-fPIC", "-shared", "-pthread",
                    "-Wno-unknown-pragmas", f"-I{HERE}", "-o", str(lib), str(cpp)], check=True)
    return ctypes.CDLL(str(lib))


def fn(lib, name, *types):
    f = getattr(lib, name)
    f.argtypes = list(types)
    f.restype = None
    return f


P, I, F = ctypes.c_void_p, ctypes.c_int, ctypes.c_float


def ptr(t):
    return ctypes.c_void_p(t.data_ptr())


def check(name, got, want, atol, rtol):
    got, want = got.double(), want.double()
    err = (got - want).abs()
    ok = bool((err <= atol + rtol * want.abs()).all()) and got.shape == want.shape
    worst = err.max().item() if err.numel() else 0.0
    print(f"  {'ok  ' if ok else 'FAIL'} {name:40s} max abs err {worst:.2e}")
    if not ok:
        FAILED.append(name)


# ---------------------------------------------------------------- problems

def t01(lib):
    solve = fn(lib, "solve", P, P, P, I)
    for n in [1, 31, 33, 1023, 1025, 100_003]:
        a, b = torch.empty(n).uniform_(-1000, 1000), torch.empty(n).uniform_(-1000, 1000)
        c = torch.empty(n)
        solve(ptr(a), ptr(b), ptr(c), n)
        check(f"N={n}", c, a + b, 1e-5, 1e-5)


def t02(lib):
    solve = fn(lib, "solve", P, P, I, I)
    for r, c in [(1, 1), (1, 70), (70, 1), (63, 65), (129, 31), (200, 333)]:
        x = torch.empty(r, c).uniform_(-10, 10)
        y = torch.empty(c, r)
        solve(ptr(x), ptr(y), r, c)
        check(f"{r}x{c}", y, x.t(), 0, 0)


def t03(lib):
    solve = fn(lib, "solve", P, P, I)
    for n in [1, 7, 4095, 4097, 300_001]:
        x = torch.empty(n).uniform_(-1000, 1000)
        out = torch.empty(1)
        solve(ptr(x), ptr(out), n)
        check(f"N={n}", out, x.double().sum().reshape(1), 1e-5, 1e-5)


def t04(lib):
    solve = fn(lib, "solve", P, P, P, I)
    for n in [1, 31, 4097, 250_007]:
        a, b = torch.empty(n).uniform_(-1, 1), torch.empty(n).uniform_(-1, 1)
        out = torch.zeros(1)
        solve(ptr(a), ptr(b), ptr(out), n)
        check(f"N={n}", out, (a.double() @ b.double()).reshape(1), 1e-5, 1e-5)


def t05(lib):
    solve = fn(lib, "solve", P, P, I)
    ref = lambda x: torch.softmax(x.double(), 0)
    for n in [1, 2, 4095, 4097, 40_001]:
        x = torch.empty(n).uniform_(-10, 10)
        y = torch.empty(n)
        solve(ptr(x), ptr(y), n)
        check(f"N={n}", y, ref(x), 1e-5, 1e-5)
    x = torch.full((5000,), 80.0)
    y = torch.empty_like(x)
    solve(ptr(x), ptr(y), 5000)
    check("large equal values (overflow)", y, ref(x), 1e-5, 1e-5)


def t06(lib):
    solve = fn(lib, "solve", P, F, F, P, I, F)
    for n in [1, 3, 4095, 4097, 33_333]:
        x = torch.empty(n).uniform_(-100, 100)
        y = torch.empty(n)
        solve(ptr(x), 1.5, -2.0, ptr(y), n, 1e-5)
        xd = x.double()
        check(f"N={n}", y, 1.5 * xd / torch.sqrt((xd ** 2).mean() + 1e-5) - 2.0, 1e-5, 1e-5)


def t07(lib):
    solve = fn(lib, "solve", P, P, P, P, I, I, F)
    for n, c in [(1, 1), (3, 7), (17, 100), (33, 512), (9, 1024), (5, 4096)]:
        x = torch.empty(n, c).uniform_(-100, 100) + 1000.0
        w, b = torch.empty(c).uniform_(0.1, 10), torch.empty(c).uniform_(-10, 10)
        y = torch.empty_like(x)
        solve(ptr(x), ptr(w), ptr(b), ptr(y), n, c, 1e-5)
        want = torch.nn.functional.layer_norm(x.double(), (c,), w.double(), b.double(), 1e-5)
        check(f"{n}x{c}", y, want, 1e-4, 1e-4)


def t08(lib):
    solve = fn(lib, "solve", P, P, P, P, I, I, F)
    for n, c in [(1, 1), (3, 5), (7, 1000), (2, 4096), (2, 4097), (3, 20_000)]:
        x, r = torch.empty(n, c).uniform_(-10, 10), torch.empty(n, c).uniform_(-10, 10)
        w = torch.empty(c).uniform_(0.5, 1.5)
        out = torch.empty_like(x)
        solve(ptr(x), ptr(r), ptr(w), ptr(out), n, c, 1e-5)
        z = (x + r).double()
        check(f"{n}x{c}", out, z / torch.sqrt((z ** 2).mean(-1, keepdim=True) + 1e-5) * w, 1e-5, 1e-5)


def t09(lib):
    solve = fn(lib, "solve", P, P, P, I, I, I, F, F)
    for M, N, K in [(16, 16, 16), (17, 33, 50), (64, 48, 100), (100, 70, 129)]:
        a = torch.empty(M, K).uniform_(-1, 1).half()
        b = torch.empty(K, N).uniform_(-1, 1).half()
        c = torch.empty(M, N).uniform_(-1, 1).half()
        want = (0.5 * (a.float() @ b.float()) + 1.5 * c.float()).half()
        solve(ptr(a), ptr(b), ptr(c), M, N, K, 0.5, 1.5)
        check(f"{M}x{N}x{K}", c, want, 0.05, 0.05)


def t10(lib):
    solve = fn(lib, "solve", P, P, P, I, I, I, I)
    for B, M, N, K in [(1, 1, 1, 1), (3, 17, 33, 9), (2, 64, 40, 100), (2, 70, 65, 33)]:
        a = torch.empty(B, M, K).uniform_(-5, 5).half()
        b = torch.empty(B, K, N).uniform_(-5, 5).half()
        c = torch.empty(B, M, N).half()
        solve(ptr(a), ptr(b), ptr(c), B, M, N, K)
        check(f"{B}x{M}x{N}x{K}", c, torch.bmm(a.float(), b.float()).half(), 0.05, 0.05)


def t11(lib):
    solve = fn(lib, "solve", P, P, P, I, I, I)
    for N, D, M in [(1, 1, 1), (4, 3, 3), (2, 5, 1), (1000, 37, 50), (3000, 64, 7)]:
        src = torch.empty(N, D).uniform_(-10, 10)
        index = torch.randint(0, M, (N,), dtype=torch.int32)
        out = torch.full((M, D), 123.0)                   # stale data must be cleared
        solve(ptr(src), ptr(index), ptr(out), N, D, M)
        want = torch.zeros(M, D, dtype=torch.float64).index_add_(0, index.long(), src.double())
        check(f"N={N} D={D} M={M}", out, want, 1e-4, 1e-4)


def t12(lib):
    solve = fn(lib, "solve", P, P, I)

    def ref(p, n):
        q = p.view(n, 3)
        d = ((q[:, None, :] - q[None, :, :]) ** 2).sum(2)
        d.fill_diagonal_(float("inf"))
        return d.argmin(1).int()

    for n in [1, 2, 3, 65, 129, 1000]:
        pts = torch.empty(n * 3).uniform_(-1000, 1000)
        idx = torch.full((n,), -1, dtype=torch.int32)
        solve(ptr(pts), ptr(idx), n)
        check(f"N={n}", idx, ref(pts, n), 0, 0)
    pts = torch.tensor([0, 0, 0, 1, 0, 0, 5, 5, 5], dtype=torch.float32)
    idx = torch.empty(3, dtype=torch.int32)
    solve(ptr(pts), ptr(idx), 3)
    check("LeetGPU example", idx, torch.tensor([1, 0, 1]), 0, 0)
    pts = torch.zeros(300 * 3)
    idx = torch.empty(300, dtype=torch.int32)
    solve(ptr(pts), ptr(idx), 300)
    check("all ties", idx, ref(pts, 300), 0, 0)


def t13(lib):
    solve = fn(lib, "solve", P, P, P, I)
    for n in [1, 2, 63, 65, 500]:
        p = torch.empty(n, 3).uniform_(-10, 10)
        m = torch.empty(n).uniform_(0.1, 10)
        f = torch.zeros(n, 3)
        solve(ptr(p), ptr(m), ptr(f), n)
        pd, md = p.double(), m.double()
        r = pd[None, :, :] - pd[:, None, :]
        want = (md[None, :, None] * r / (((r * r).sum(2) + 1e-6) ** 1.5)[:, :, None]).sum(1)
        check(f"N={n}", f, want, 1e-2, 1e-2)


def t14(lib):
    solve = fn(lib, "solve", P, P, I, I)
    for r, c in [(1, 1), (1, 9), (9, 1), (2, 2), (3, 3), (17, 130), (100, 257)]:
        x = torch.empty(r, c).uniform_(-100, 100)
        y = torch.empty_like(x)
        solve(ptr(x), ptr(y), r, c)
        want = x.clone()
        want[1:-1, 1:-1] = 0.25 * (x[:-2, 1:-1] + x[2:, 1:-1] + x[1:-1, :-2] + x[1:-1, 2:])
        check(f"{r}x{c}", y, want, 1e-5, 1e-5)


def t15(lib):
    solve = fn(lib, "solve", P, P, P, I, I, I, I)
    for M, N, K in [(1, 1, 1), (5, 7, 3), (33, 70, 17), (100, 130, 65)]:
        A = torch.empty(M, N).uniform_(-10, 10) * (torch.rand(M, N) > 0.65)
        B = torch.empty(N, K).uniform_(-10, 10)
        C = torch.empty(M, K)
        solve(ptr(A), ptr(B), ptr(C), M, N, K, int((A != 0).sum()))
        check(f"{M}x{N}x{K}", C, A.double() @ B.double(), 1e-3, 1e-3)


def t16(lib):
    solve = fn(lib, "solve", P, P, P, I)

    def ref(v, f):
        vd = v.double()
        excl = torch.zeros_like(vd)
        excl[1:] = torch.cumsum(vd[:-1], 0)
        seg = torch.cumsum(f.long(), 0) - 1
        return excl - excl[f.bool()][seg]

    for n, p in [(1, 0.5), (2, 0.0), (5, 1.0), (4095, 0.01), (4097, 0.0), (9000, 0.3),
                 (50_000, 0.0), (300_000, 1 / 256)]:
        v = torch.empty(n).uniform_(-100, 100)
        f = (torch.rand(n) < p).int()
        f[0] = 1
        out = torch.empty(n)
        solve(ptr(v), ptr(f), ptr(out), n)
        check(f"N={n} p(flag)={p:.3g}", out, ref(v, f), 1e-3, 1e-3)


def t17(lib):
    solve = fn(lib, "solve", P, P, P, P, P, P, I, I, I, F)
    for M, K, N in [(1, 1, 1), (3, 5, 7), (40, 70, 33), (65, 130, 100)]:
        x = torch.empty(M, K).uniform_(-1, 1) - 1000.0
        g, b = torch.empty(K).uniform_(0.5, 1.5), torch.empty(K).uniform_(-0.5, 0.5)
        W = torch.empty(K, N).uniform_(-1, 1) / K ** 0.5
        bias = torch.empty(N).uniform_(-0.5, 0.5)
        y = torch.empty(M, N)
        solve(ptr(x), ptr(g), ptr(b), ptr(W), ptr(bias), ptr(y), M, K, N, 1e-5)
        xd, gd, bd, Wd, biasd = (t.double() for t in (x, g, b, W, bias))
        mu = xd.mean(1, keepdim=True)
        var = ((xd - mu) ** 2).mean(1, keepdim=True)
        h = ((xd - mu) / torch.sqrt(var + 1e-5) * gd + bd) @ Wd + biasd
        check(f"M={M} K={K} N={N}", y, 0.5 * h * (1 + torch.erf(h / math.sqrt(2))), 1e-3, 1e-3)


def t18(lib):
    solve = fn(lib, "solve", P, P, P, P, I, I, I)
    for M, N, d in [(1, 1, 1), (2, 3, 4), (17, 100, 33), (70, 129, 128), (5, 7, 100)]:
        Q, K, V = (torch.empty(n, d).uniform_(-1, 1) for n in (M, N, N))
        out = torch.empty(M, d)
        solve(ptr(Q), ptr(K), ptr(V), ptr(out), M, N, d)
        Qd, Kd, Vd = Q.double(), K.double(), V.double()
        check(f"M={M} N={N} d={d}", out, torch.softmax(Qd @ Kd.T / d ** 0.5, 1) @ Vd, 1e-4, 1e-4)


def t19(lib):
    solve = fn(lib, "solve", P, P, P, P, P, I, I, F)
    for N, C in [(1, 1), (3, 5), (17, 100), (300, 129), (9, 2000)]:
        x = torch.empty(N, C).uniform_(-250, 250)
        w = torch.empty(C).uniform_(-2, 2)
        dy = torch.empty(N, C).uniform_(-10, 10)
        dx, dw = torch.empty_like(x), torch.empty(C)
        solve(ptr(x), ptr(w), ptr(dy), ptr(dx), ptr(dw), N, C, 1e-5)
        xd, wd, dyd = x.double(), w.double(), dy.double()
        rstd = torch.rsqrt((xd * xd).mean(1, keepdim=True) + 1e-5)
        xhat, g = xd * rstd, dyd * wd
        check(f"dx {N}x{C}", dx, rstd * (g - xhat * (g * xhat).mean(1, keepdim=True)), 1e-3, 1e-3)
        check(f"dw {N}x{C}", dw, (dyd * xhat).sum(0), 1e-3, 1e-3)


def t20(lib):
    solve = fn(lib, "solve", P, P, P, P, P, P, P, P, I, I)
    for N, C in [(1, 1), (3, 5), (17, 100), (300, 129), (9, 2000)]:
        x = torch.empty(N, C).uniform_(-250, 250)
        gamma = torch.empty(C).uniform_(-2, 2)
        dy = torch.empty(N, C).uniform_(-10, 10)
        mean = x.mean(1)
        rstd = torch.rsqrt(x.var(1, unbiased=False) + 1e-5)
        dx, dg, db = torch.empty_like(x), torch.empty(C), torch.empty(C)
        solve(ptr(x), ptr(gamma), ptr(mean), ptr(rstd), ptr(dy), ptr(dx), ptr(dg), ptr(db), N, C)
        xd, gd, md, rd, dyd = (t.double() for t in (x, gamma, mean, rstd, dy))
        xhat = (xd - md[:, None]) * rd[:, None]
        g = dyd * gd
        want_dx = rd[:, None] * (g - g.mean(1, keepdim=True) - xhat * (g * xhat).mean(1, keepdim=True))
        check(f"dx {N}x{C}", dx, want_dx, 1e-3, 1e-3)
        check(f"dgamma {N}x{C}", dg, (dyd * xhat).sum(0), 1e-3, 1e-3)
        check(f"dbeta {N}x{C}", db, dyd.sum(0), 1e-3, 1e-3)


def stencil(lib):
    grad = fn(lib, "grad2d", P, P, P, I, I, F, F)
    div = fn(lib, "div2d", P, P, P, I, I, F, F)
    lap = fn(lib, "lap3d", P, P, I, I, I, F, F, F)
    dx, dy, dz = 0.1, 0.25, 0.3
    for ny, nx in [(1, 1), (3, 5), (17, 70), (64, 129)]:
        u = torch.randn(ny + 2, nx + 2)
        gx, gy = torch.zeros_like(u), torch.zeros_like(u)
        grad(ptr(u), ptr(gx), ptr(gy), nx, ny, dx, dy)
        check(f"grad2d {ny}x{nx} d/dx", gx[1:-1, 1:-1], (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx), 1e-5, 1e-5)
        check(f"grad2d {ny}x{nx} d/dy", gy[1:-1, 1:-1], (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dy), 1e-5, 1e-5)
        vx, vy, out = torch.randn_like(u), torch.randn_like(u), torch.zeros_like(u)
        div(ptr(vx), ptr(vy), ptr(out), nx, ny, dx, dy)
        want = (vx[1:-1, 2:] - vx[1:-1, :-2]) / (2 * dx) + (vy[2:, 1:-1] - vy[:-2, 1:-1]) / (2 * dy)
        check(f"div2d  {ny}x{nx}", out[1:-1, 1:-1], want, 1e-5, 1e-5)
    for nz, ny, nx in [(1, 1, 1), (4, 3, 5), (9, 17, 70), (5, 9, 33)]:
        u = torch.randn(nz + 2, ny + 2, nx + 2)
        out = torch.zeros_like(u)
        lap(ptr(u), ptr(out), nx, ny, nz, dx, dy, dz)
        c = u[1:-1, 1:-1, 1:-1]
        want = ((u[1:-1, 1:-1, 2:] + u[1:-1, 1:-1, :-2] - 2 * c) / dx ** 2
                + (u[1:-1, 2:, 1:-1] + u[1:-1, :-2, 1:-1] - 2 * c) / dy ** 2
                + (u[2:, 1:-1, 1:-1] + u[:-2, 1:-1, 1:-1] - 2 * c) / dz ** 2)
        check(f"lap3d  {nz}x{ny}x{nx}", out[1:-1, 1:-1, 1:-1], want, 1e-4, 1e-4)


def warp(lib):
    """notes/warp_primitives.cu: shuffles, votes, match, reduce, cooperative groups."""
    U = ctypes.c_uint
    ws = fn(lib, "warp_sums", P, P, I)
    wm = fn(lib, "warp_max", P, P, I)
    sb = fn(lib, "segment_broadcast", P, P, I, I)
    sc = fn(lib, "warp_scan", P, P, I, I)
    bs = fn(lib, "block_scan", P, P, I)
    vo = fn(lib, "warp_votes", P, P, P, P, I)
    cp = fn(lib, "compact_positive", P, P, P, I)
    hm = fn(lib, "histogram_match", P, P, I, I)
    im = fn(lib, "int_sum_max", P, P, P, I)
    cpos = fn(lib, "count_positive", P, P, I)
    cbs = fn(lib, "cg_block_sum", P, P, I)
    cts = fn(lib, "cg_tile_scan", P, P, P, I)
    nz = fn(lib, "normalize_by_sum", P, P, I)

    def pad(t, m, value):                               # pad to a multiple of m, view as rows
        k = (-t.numel()) % m
        return torch.cat([t, torch.full((k,), value, dtype=t.dtype)]).view(-1, m)

    def seg_scan(t, w):
        return torch.cat([c.cumsum(0) for c in t.split(w)]).to(t.dtype)

    for n in [1, 31, 32, 33, 1000, 4099]:
        x = torch.randn(n)
        out = torch.zeros((n + 31) // 32)
        ws(ptr(x), ptr(out), n)
        check(f"shfl_down warp sums n={n}", out, pad(x, 32, 0.0).sum(1), 1e-5, 1e-5)

        out = torch.zeros(n)
        wm(ptr(x), ptr(out), n)
        check(f"shfl_xor warp max (all lanes) n={n}", out,
              pad(x, 32, -math.inf).amax(1).repeat_interleave(32)[:n], 0, 0)

        for w in [4, 8, 32]:
            sb(ptr(x), ptr(out), n, w)
            check(f"shfl width={w} broadcast n={n}", out, x[(torch.arange(n) // w) * w], 0, 0)

        xi = torch.randint(-5, 6, (n,), dtype=torch.int32)
        oi = torch.zeros(n, dtype=torch.int32)
        for w in [8, 32]:
            sc(ptr(xi), ptr(oi), n, w)
            check(f"shfl_up scan width={w} n={n}", oi, seg_scan(xi, w), 0, 0)

        bs(ptr(xi), ptr(oi), n)
        check(f"block scan n={n}", oi, xi.cumsum(0).to(torch.int32), 0, 0)

        nw = (n + 31) // 32
        xv = torch.randn(n).abs() + 0.1
        xv[torch.rand(n) < 0.02] *= -1                  # a few negatives
        xv[: min(n, 32)] = xv[: min(n, 32)].abs()       # first warp: all positive
        an, al = torch.zeros(nw, dtype=torch.int32), torch.zeros(nw, dtype=torch.int32)
        bal = torch.zeros(nw, dtype=torch.int32)
        vo(ptr(xv), ptr(an), ptr(al), ptr(bal), n)
        rows = pad(xv, 32, 1.0)
        check(f"any_sync n={n}", an, (rows < 0).any(1).int(), 0, 0)
        check(f"all_sync n={n}", al, (rows > 0).all(1).int(), 0, 0)
        bits = (pad(xv > 0, 32, False).long() << torch.arange(32)).sum(1)
        bits = torch.where(bits >= 2 ** 31, bits - 2 ** 32, bits).int()
        check(f"ballot_sync n={n}", bal, bits, 0, 0)

        out, cnt = torch.zeros(n), torch.zeros(1, dtype=torch.int32)
        cp(ptr(x), ptr(out), ptr(cnt), n)
        k = int(cnt.item())
        want = x[x > 0]
        check(f"ballot+popc compaction count n={n}", cnt, torch.tensor([want.numel()]), 0, 0)
        check(f"ballot+popc compaction values n={n}", out[:k].sort().values, want.sort().values, 0, 0)

        nb = 16
        b = (torch.randn(n).abs() * 3).long().clamp(max=nb - 1).int()   # skewed: low bins common
        h = torch.zeros(nb, dtype=torch.int32)
        hm(ptr(b), ptr(h), n, nb)
        check(f"match_any histogram n={n}", h, torch.bincount(b, minlength=nb).int(), 0, 0)

        su, mx = torch.zeros(1, dtype=torch.int32), torch.zeros(1, dtype=torch.int32)
        xr = torch.randint(-1000, 1000, (n,), dtype=torch.int32)
        im(ptr(xr), ptr(su), ptr(mx), n)
        check(f"reduce_add_sync n={n}", su, xr.sum().reshape(1).int(), 0, 0)
        check(f"reduce_max_sync n={n}", mx, xr.max().reshape(1), 0, 0)

        nbk = (n + 255) // 256
        pb = torch.zeros(nbk, dtype=torch.int32)
        cpos(ptr(x), ptr(pb), n)
        check(f"syncthreads_count n={n}", pb, pad(x > 0, 256, False).sum(1).int(), 0, 0)

        pf = torch.zeros(nbk)
        cbs(ptr(x), ptr(pf), n)
        check(f"cg::reduce block sums n={n}", pf, pad(x, 256, 0.0).sum(1), 1e-4, 1e-5)

        inc, exc = torch.zeros(n, dtype=torch.int32), torch.zeros(n, dtype=torch.int32)
        cts(ptr(xi), ptr(inc), ptr(exc), n)
        check(f"cg::inclusive_scan tile16 n={n}", inc, seg_scan(xi, 16), 0, 0)
        check(f"cg::exclusive_scan tile16 n={n}", exc, seg_scan(xi, 16) - xi, 0, 0)

    for n in [1, 1000, 100_003]:
        x = torch.rand(n) + 0.5
        y = torch.zeros(n)
        nz(ptr(x), ptr(y), n)
        check(f"grid.sync normalize n={n}", y, x / x.double().sum(), 1e-6, 1e-4)


TESTS = {f"{n:02d}": globals()[f"t{n:02d}"] for n in range(1, 21)}
EXTRA = {"stencil_ops": stencil, "warp_primitives": warp}


def main():
    torch.manual_seed(0)
    if GPU:
        torch.set_default_device("cuda")                 # every test tensor lives on the GPU
    args = [a for a in sys.argv[1:] if a != "--gpu"]
    pattern = args[0] if args else ""
    files = sorted((ROOT / "docs" / "leetgpu").glob("[0-9][0-9]_*.cu")) + [ROOT / "docs" / "notes" / f"{k}.cu" for k in EXTRA]
    for cu in files:
        if pattern not in cu.name:
            continue
        print(cu.relative_to(ROOT))
        test = EXTRA.get(cu.stem) or TESTS[cu.name[:2]]
        test(build(cu))
    print(f"\n{len(FAILED)} failed" + (": " + ", ".join(FAILED) if FAILED else ""))
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
