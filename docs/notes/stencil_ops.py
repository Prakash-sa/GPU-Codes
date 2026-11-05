"""Discrete gradient, divergence and Laplacian as GPU stencils (Triton), with tests.

    python docs/notes/stencil_ops.py                    # CUDA GPU: tests + bandwidth
    TRITON_INTERPRET=1 python docs/notes/stencil_ops.py # no GPU: kernels run on the CPU

Fields carry one layer of ghost cells: a 2-D field of nx x ny points is stored as a
(ny+2, nx+2) array, row-major, x fastest. The boundary condition only fills the ghost
layer (fill_ghosts_*); the operators then treat every interior point the same way, with
no branches. All operators are second-order central differences on a collocated grid:

    grad u   = ((u[j,i+1] - u[j,i-1]) / 2dx,  (u[j+1,i] - u[j-1,i]) / 2dy)
    div v    = (vx[j,i+1] - vx[j,i-1]) / 2dx + (vy[j+1,i] - vy[j-1,i]) / 2dy
    lap u    = sum over axes of (u[+1] - 2u + u[-1]) / h^2       (5-point 2-D, 7-point 3-D)

Comments that start with `mem:` say where the data lives (see ../leetgpu/memory-hierarchy.md).
"""
import math
import time

import torch
import triton
import triton.language as tl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- 2-D gradient

@triton.jit
def grad2d_kernel(u, gx, gy, nx, ny, inv2dx, inv2dy, BY: tl.constexpr, BX: tl.constexpr):
    # interior point (j, i), 1-based inside the ghosted array of width nx + 2
    j = 1 + tl.program_id(0) * BY + tl.arange(0, BY)[:, None]
    i = 1 + tl.program_id(1) * BX + tl.arange(0, BX)[None, :]
    m = (j <= ny) & (i <= nx)
    w = nx + 2
    p = u + j * w + i
    # mem: four shifted tiles -> registers; they overlap, so after the first touch they
    # hit L1/L2 and DRAM sees ~1 read of u per point
    e = tl.load(p + 1, mask=m)
    west = tl.load(p - 1, mask=m)
    n = tl.load(p + w, mask=m)
    s = tl.load(p - w, mask=m)
    # mem: registers -> DRAM, two writes per point
    tl.store(gx + j * w + i, (e - west) * inv2dx, mask=m)
    tl.store(gy + j * w + i, (n - s) * inv2dy, mask=m)


def grad2d(u, dx, dy, BY=16, BX=64):
    """Gradient of a ghosted (ny+2, nx+2) field; returns (gx, gy) with the same shape."""
    ny, nx = u.shape[0] - 2, u.shape[1] - 2
    gx, gy = torch.zeros_like(u), torch.zeros_like(u)
    grid = (triton.cdiv(ny, BY), triton.cdiv(nx, BX))
    grad2d_kernel[grid](u, gx, gy, nx, ny, 0.5 / dx, 0.5 / dy, BY=BY, BX=BX)
    return gx, gy


# ---------------------------------------------------------------- 2-D divergence

@triton.jit
def div2d_kernel(vx, vy, out, nx, ny, inv2dx, inv2dy, BY: tl.constexpr, BX: tl.constexpr):
    j = 1 + tl.program_id(0) * BY + tl.arange(0, BY)[:, None]
    i = 1 + tl.program_id(1) * BX + tl.arange(0, BX)[None, :]
    m = (j <= ny) & (i <= nx)
    w = nx + 2
    o = j * w + i
    # mem: vector field stored as two separate arrays (SoA), so each load is a
    # contiguous row segment: coalesced
    dvx = tl.load(vx + o + 1, mask=m) - tl.load(vx + o - 1, mask=m)
    dvy = tl.load(vy + o + w, mask=m) - tl.load(vy + o - w, mask=m)
    tl.store(out + o, dvx * inv2dx + dvy * inv2dy, mask=m)


def div2d(vx, vy, dx, dy, BY=16, BX=64):
    ny, nx = vx.shape[0] - 2, vx.shape[1] - 2
    out = torch.zeros_like(vx)
    grid = (triton.cdiv(ny, BY), triton.cdiv(nx, BX))
    div2d_kernel[grid](vx, vy, out, nx, ny, 0.5 / dx, 0.5 / dy, BY=BY, BX=BX)
    return out


# ---------------------------------------------------------------- 3-D Laplacian, 2.5-D blocking

@triton.jit
def lap3d_kernel(u, out, nx, ny, nz, cx, cy, cz, BY: tl.constexpr, BX: tl.constexpr):
    """7-point Laplacian. Each program owns a BY x BX column of the x-y plane and marches
    up z, keeping the planes k-1, k and k+1 in registers: each plane is read from memory
    once per program instead of three times."""
    j = 1 + tl.program_id(0) * BY + tl.arange(0, BY)[:, None]
    i = 1 + tl.program_id(1) * BX + tl.arange(0, BX)[None, :]
    m = (j <= ny) & (i <= nx)
    w = nx + 2
    plane = w * (ny + 2)
    xy = j * w + i
    # mem: the register queue. below and centre carry over from the previous step
    below = tl.load(u + xy, mask=m)                 # k = 0: the ghost plane
    centre = tl.load(u + plane + xy, mask=m)        # k = 1
    for k in range(1, nz + 1):
        above = tl.load(u + (k + 1) * plane + xy, mask=m)      # mem: the only new plane
        c = u + k * plane + xy
        # mem: in-plane neighbours overlap the centre tile: L1/L2 hits
        lx = tl.load(c + 1, mask=m) + tl.load(c - 1, mask=m) - 2.0 * centre
        ly = tl.load(c + w, mask=m) + tl.load(c - w, mask=m) - 2.0 * centre
        lz = above + below - 2.0 * centre
        tl.store(out + k * plane + xy, lx * cx + ly * cy + lz * cz, mask=m)
        below = centre
        centre = above


def lap3d(u, dx, dy, dz, BY=8, BX=64):
    """Laplacian of a ghosted (nz+2, ny+2, nx+2) field."""
    nz, ny, nx = (s - 2 for s in u.shape)
    out = torch.zeros_like(u)
    grid = (triton.cdiv(ny, BY), triton.cdiv(nx, BX))
    lap3d_kernel[grid](u, out, nx, ny, nz, 1 / dx**2, 1 / dy**2, 1 / dz**2, BY=BY, BX=BX)
    return out


# ---------------------------------------------------------------- boundary conditions

def fill_ghosts_periodic(u):
    """Copy the opposite edge into each ghost layer, axis by axis (also fills corners)."""
    for d in range(u.dim()):
        n = u.shape[d] - 2
        u.narrow(d, 0, 1).copy_(u.narrow(d, n, 1))
        u.narrow(d, n + 1, 1).copy_(u.narrow(d, 1, 1))
    return u


def fill_ghosts_dirichlet(u, value=0.0):
    """u = value on the wall halfway between ghost and first interior point:
    ghost = 2 * value - interior (second-order accurate at the wall)."""
    for d in range(u.dim()):
        n = u.shape[d] - 2
        u.narrow(d, 0, 1).copy_(2 * value - u.narrow(d, 1, 1))
        u.narrow(d, n + 1, 1).copy_(2 * value - u.narrow(d, n, 1))
    return u


def fill_ghosts_neumann(u):
    """du/dn = 0 at the wall: ghost = mirror of the first interior point."""
    for d in range(u.dim()):
        n = u.shape[d] - 2
        u.narrow(d, 0, 1).copy_(u.narrow(d, 1, 1))
        u.narrow(d, n + 1, 1).copy_(u.narrow(d, n, 1))
    return u


# ---------------------------------------------------------------- tests

def interior(a):
    return a[(slice(1, -1),) * a.dim()]


def check(name, got, want, tol):
    err = (interior(got) - interior(want)).abs().max().item()
    scale = max(1.0, interior(want).abs().max().item())
    assert err <= tol * scale, f"{name}: max error {err:.3e}"
    print(f"  ok  {name:44s} max err {err:.2e}")


def ref_grad2d(u, dx, dy):
    gx, gy = torch.zeros_like(u), torch.zeros_like(u)
    gx[1:-1, 1:-1] = (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx)
    gy[1:-1, 1:-1] = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dy)
    return gx, gy


def ref_lap3d(u, dx, dy, dz):
    out = torch.zeros_like(u)
    c = u[1:-1, 1:-1, 1:-1]
    out[1:-1, 1:-1, 1:-1] = ((u[1:-1, 1:-1, 2:] + u[1:-1, 1:-1, :-2] - 2 * c) / dx**2
                             + (u[1:-1, 2:, 1:-1] + u[1:-1, :-2, 1:-1] - 2 * c) / dy**2
                             + (u[2:, 1:-1, 1:-1] + u[:-2, 1:-1, 1:-1] - 2 * c) / dz**2)
    return out


def periodic_grid(n, dim):
    """Cell centres x = (i + 0.5) h on [0, 1), with a ghost layer, for each axis."""
    h = 1.0 / n
    x = (torch.arange(-1, n + 1, device=DEVICE, dtype=torch.float64) + 0.5) * h
    return h, torch.meshgrid(*([x] * dim), indexing="ij")     # axis order z, y, x


def order_of_accuracy():
    """Manufactured solution: errors must fall ~4x per halving of h (second order)."""
    k = 2 * math.pi
    print("convergence (float64, periodic):")
    rows = []
    for n in (16, 32, 64):
        h, (y, x) = periodic_grid(n, 2)
        u = torch.sin(k * x) * torch.cos(k * y)
        gx, gy = grad2d(u, h, h)
        e_grad = (interior(gx) - interior(k * torch.cos(k * x) * torch.cos(k * y))).abs().max().item()
        div = div2d(fill_ghosts_periodic(gx), fill_ghosts_periodic(gy), h, h)
        e_div = (interior(div) - interior(-2 * k * k * u)).abs().max().item()
        h3, (z3, y3, x3) = periodic_grid(n, 3)
        u3 = torch.sin(k * x3) * torch.sin(k * y3) * torch.sin(k * z3)
        e_lap = (interior(lap3d(u3, h3, h3, h3)) - interior(-3 * k * k * u3)).abs().max().item()
        rows.append((n, e_grad, e_div, e_lap))
    for (n0, *e0), (n1, *e1) in zip(rows, rows[1:]):
        orders = [math.log2(a / b) for a, b in zip(e0, e1)]
        print(f"  n {n0:3d} -> {n1:3d}: order grad {orders[0]:.2f}  div(grad) {orders[1]:.2f}"
              f"  lap3d {orders[2]:.2f}")
        assert all(1.8 < o < 2.2 for o in orders), orders


def wide_stencil_pitfall():
    """div(grad u) with central differences is NOT the compact 5-point Laplacian: it is a
    Laplacian with spacing 2h, which cannot see a checkerboard mode."""
    n, h = 16, 1.0 / 16
    jj, ii = torch.meshgrid(torch.arange(n + 2, device=DEVICE), torch.arange(n + 2, device=DEVICE),
                            indexing="ij")
    checker = ((ii + jj) % 2 * 2 - 1).to(torch.float64)       # +1 / -1 alternating
    gx, gy = grad2d(checker, h, h)
    wide = div2d(fill_ghosts_periodic(gx), fill_ghosts_periodic(gy), h, h)
    compact = ref_lap3d(checker[None].expand(3, -1, -1).contiguous(), h, h, 1.0)[1]
    print("checkerboard mode:")
    print(f"  div(grad u) max |value| = {interior(wide).abs().max().item():.1f}  (blind to it)")
    print(f"  5-point lap max |value| = {interior(compact).abs().max().item():.1f}  (sees it: -8/h^2)")
    assert interior(wide).abs().max().item() == 0.0


def bench(name, fn, bytes_moved, iters=20):
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    t = (time.perf_counter() - t0) / iters
    print(f"  {name:28s} {t * 1e3:7.3f} ms  {bytes_moved / t / 1e9:7.1f} GB/s effective")


if __name__ == "__main__":
    torch.manual_seed(0)
    print(f"device: {DEVICE}")
    print("correctness against PyTorch (float32, random fields):")
    for ny, nx in [(1, 1), (3, 5), (17, 70), (64, 129)]:
        u = torch.randn(ny + 2, nx + 2, device=DEVICE)
        dx, dy = 0.1, 0.25
        for got, want, ax in zip(grad2d(u, dx, dy), ref_grad2d(u, dx, dy), "xy"):
            check(f"grad2d {ny}x{nx} d/d{ax}", got, want, 1e-5)
        vx, vy = torch.randn_like(u), torch.randn_like(u)
        want = torch.zeros_like(u)
        want[1:-1, 1:-1] = ((vx[1:-1, 2:] - vx[1:-1, :-2]) / (2 * dx)
                            + (vy[2:, 1:-1] - vy[:-2, 1:-1]) / (2 * dy))
        check(f"div2d  {ny}x{nx}", div2d(vx, vy, dx, dy), want, 1e-5)
    for nz, ny, nx in [(1, 1, 1), (4, 3, 5), (9, 17, 70)]:
        u = torch.randn(nz + 2, ny + 2, nx + 2, device=DEVICE)
        check(f"lap3d  {nz}x{ny}x{nx}", lap3d(u, 0.1, 0.2, 0.3), ref_lap3d(u, 0.1, 0.2, 0.3), 1e-5)

    print("boundary conditions (ghost fill):")
    a = torch.arange(1.0, 10.0, device=DEVICE).reshape(3, 3)
    g = torch.zeros(5, 5, device=DEVICE)
    g[1:-1, 1:-1] = a
    assert torch.equal(fill_ghosts_periodic(g.clone())[1:-1, 0], a[:, -1])
    assert torch.equal(fill_ghosts_neumann(g.clone())[1:-1, 0], a[:, 0])
    assert torch.equal(fill_ghosts_dirichlet(g.clone(), 1.0)[1:-1, 0], 2.0 - a[:, 0])
    print("  ok  periodic, Neumann and Dirichlet ghost layers")

    order_of_accuracy()
    wide_stencil_pitfall()

    if DEVICE == "cuda":
        print("bandwidth (float32):")
        n = 8192
        u = torch.randn(n + 2, n + 2, device=DEVICE)
        bench("grad2d 8192^2", lambda: grad2d(u, 1.0, 1.0), 3 * 4 * n * n)     # 1 read, 2 writes
        n3 = 512
        u3 = torch.randn(n3 + 2, n3 + 2, n3 + 2, device=DEVICE)
        bench("lap3d 512^3", lambda: lap3d(u3, 1.0, 1.0, 1.0), 2 * 4 * n3**3)  # 1 read, 1 write
    print("all checks passed")
