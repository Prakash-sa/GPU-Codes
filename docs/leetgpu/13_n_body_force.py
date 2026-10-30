"""LeetGPU 101 - N-body force (perf N = 8192):

    F_i = sum_j m_j * (p_j - p_i) / (|p_j - p_i|^2 + eps^2)^(3/2),   eps = 1e-3

The j = i term is zero because p_j - p_i = 0, so no mask is needed for it.

Same tiling as 12_nearest_neighbor.py: a program owns BLOCK_I bodies, streams all
bodies in tiles of BLOCK_J, and accumulates (Fx, Fy, Fz) in registers. Each j-tile load
is reused BLOCK_I times. The math is reshaped for the GPU:
    (r^2)^(-3/2) = rsqrt(r^2)^3    -> one rsqrt plus two multiplies, no pow, no divide.
About 20 FLOPs per pair, 67M pairs at N = 8192: compute bound.
"""
import torch
import triton
import triton.language as tl

EPS = 1e-3


@triton.jit
def nbody_kernel(pos, mass, force, N, eps2, BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr):
    i = tl.program_id(0) * BLOCK_I + tl.arange(0, BLOCK_I)
    mi = i < N
    xi = tl.load(pos + 3 * i + 0, mask=mi, other=0.0)
    yi = tl.load(pos + 3 * i + 1, mask=mi, other=0.0)
    zi = tl.load(pos + 3 * i + 2, mask=mi, other=0.0)
    fx = tl.zeros([BLOCK_I], dtype=tl.float32)
    fy = tl.zeros([BLOCK_I], dtype=tl.float32)
    fz = tl.zeros([BLOCK_I], dtype=tl.float32)

    for start in range(0, N, BLOCK_J):
        j = start + tl.arange(0, BLOCK_J)
        mj = j < N
        rx = tl.load(pos + 3 * j + 0, mask=mj, other=0.0)[None, :] - xi[:, None]
        ry = tl.load(pos + 3 * j + 1, mask=mj, other=0.0)[None, :] - yi[:, None]
        rz = tl.load(pos + 3 * j + 2, mask=mj, other=0.0)[None, :] - zi[:, None]
        mj_val = tl.load(mass + j, mask=mj, other=0.0)          # padded bodies weigh 0
        inv = tl.rsqrt(rx * rx + ry * ry + rz * rz + eps2)
        s = mj_val[None, :] * (inv * inv * inv)
        fx += tl.sum(s * rx, axis=1)
        fy += tl.sum(s * ry, axis=1)
        fz += tl.sum(s * rz, axis=1)

    tl.store(force + 3 * i + 0, fx, mask=mi)
    tl.store(force + 3 * i + 1, fy, mask=mi)
    tl.store(force + 3 * i + 2, fz, mask=mi)


# positions, masses, forces are tensors on the GPU
def solve(positions: torch.Tensor, masses: torch.Tensor, forces: torch.Tensor, N: int):
    BLOCK_I, BLOCK_J = 64, 64
    nbody_kernel[(triton.cdiv(N, BLOCK_I),)](positions, masses, forces, N, EPS * EPS,
                                             BLOCK_I=BLOCK_I, BLOCK_J=BLOCK_J, num_warps=4)


if __name__ == "__main__":
    from harness import DEVICE, ON_GPU, bench, check

    def ref(p, m):
        r = p[None, :, :] - p[:, None, :]                       # r[i, j] = p_j - p_i
        denom = ((r * r).sum(2) + EPS * EPS) ** 1.5
        return (m[None, :, None] * r / denom[:, :, None]).sum(1)

    sizes = [1, 2, 63, 65, 500] + ([8192] if ON_GPU else [])
    for n in sizes:
        p = torch.empty(n, 3, device=DEVICE).uniform_(-10, 10)
        m = torch.empty(n, device=DEVICE).uniform_(0.1, 10)
        f = torch.zeros(n, 3, device=DEVICE)
        solve(p, m, f, n)
        check(f"N={n}", f, ref(p.double(), m.double()), 1e-2, 1e-2)
    if ON_GPU:
        bench("nbody N=8192", lambda: solve(p, m, f, n), flops=20 * n * n)
