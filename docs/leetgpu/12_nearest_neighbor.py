"""LeetGPU 38 - Nearest Neighbor: for each of N 3-D points, the index j != i of the
closest point (squared distance; perf N = 10K). Checked with atol = 0: indices must
match exactly, including ties (lowest index wins, as in torch.argmin).

PyTorch builds the full N x N distance matrix (400 MB at N = 10K). This kernel never
materializes it: a program owns BLOCK_I query points and sweeps the other points in
tiles of BLOCK_J, keeping a running (best distance, best index) per query in registers.
Total work O(N^2) arithmetic, O(N) memory traffic: compute bound, not memory bound.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def nearest_kernel(pts, out, N, BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr):
    i = tl.program_id(0) * BLOCK_I + tl.arange(0, BLOCK_I)
    mi = i < N
    xi = tl.load(pts + 3 * i + 0, mask=mi, other=0.0)
    yi = tl.load(pts + 3 * i + 1, mask=mi, other=0.0)
    zi = tl.load(pts + 3 * i + 2, mask=mi, other=0.0)

    best_d = tl.full([BLOCK_I], float("inf"), tl.float32)
    best_j = tl.zeros([BLOCK_I], dtype=tl.int32)
    for start in range(0, N, BLOCK_J):              # tiles in increasing j order
        j = start + tl.arange(0, BLOCK_J)
        mj = j < N
        dx = xi[:, None] - tl.load(pts + 3 * j + 0, mask=mj, other=0.0)[None, :]
        dy = yi[:, None] - tl.load(pts + 3 * j + 1, mask=mj, other=0.0)[None, :]
        dz = zi[:, None] - tl.load(pts + 3 * j + 2, mask=mj, other=0.0)[None, :]
        d = dx * dx + dy * dy + dz * dz
        d = tl.where((i[:, None] == j[None, :]) | (~mj[None, :]), float("inf"), d)
        tile_d, tile_j = tl.min(d, axis=1, return_indices=True)   # leftmost on ties
        better = tile_d < best_d                    # strict: an earlier tile wins ties
        best_d = tl.where(better, tile_d, best_d)
        best_j = tl.where(better, start + tile_j, best_j)
    tl.store(out + i, best_j, mask=mi)


# points and indices are tensors on the GPU
def solve(points: torch.Tensor, indices: torch.Tensor, N: int):
    BLOCK_I, BLOCK_J = 64, 128
    nearest_kernel[(triton.cdiv(N, BLOCK_I),)](points, indices, N,
                                               BLOCK_I=BLOCK_I, BLOCK_J=BLOCK_J, num_warps=8)


if __name__ == "__main__":
    from harness import DEVICE, ON_GPU, bench, check

    def ref(points, N):
        p = points.view(N, 3)
        d = ((p[:, None, :] - p[None, :, :]) ** 2).sum(2)
        d.fill_diagonal_(float("inf"))
        return d.argmin(1).int()

    sizes = [1, 2, 3, 65, 129, 1000] + ([10_000] if ON_GPU else [])
    for n in sizes:
        pts = torch.empty(n * 3, device=DEVICE).uniform_(-1000, 1000)
        idx = torch.full((n,), -1, device=DEVICE, dtype=torch.int32)
        solve(pts, idx, n)
        check(f"N={n}", idx, ref(pts, n), 0, 0)
    pts = torch.tensor([0, 0, 0, 1, 0, 0, 5, 5, 5], device=DEVICE, dtype=torch.float32)
    idx = torch.empty(3, device=DEVICE, dtype=torch.int32)
    solve(pts, idx, 3)
    check("LeetGPU example", idx, torch.tensor([1, 0, 1], dtype=torch.int32, device=DEVICE), 0, 0)
    pts = torch.zeros(300 * 3, device=DEVICE)                  # all points coincide: pure ties
    idx = torch.empty(300, device=DEVICE, dtype=torch.int32)
    solve(pts, idx, 300)
    check("all ties", idx, ref(pts, 300), 0, 0)
    if ON_GPU:
        n = 10_000
        pts = torch.empty(n * 3, device=DEVICE).uniform_(-1000, 1000)
        idx = torch.empty(n, device=DEVICE, dtype=torch.int32)
        bench("nearest N=10K", lambda: solve(pts, idx, n), lambda: ref(pts, n), flops=8 * n * n)
