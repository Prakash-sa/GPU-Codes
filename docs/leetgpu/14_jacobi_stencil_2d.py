"""LeetGPU 69 - 2-D Jacobi stencil, one sweep (float32, perf 8192 x 8192):

    out[r, c] = 0.25 * (in[r-1, c] + in[r+1, c] + in[r, c-1] + in[r, c+1])   (interior)
    out[r, c] = in[r, c]                                                       (boundary)

One program per BLOCK_R x BLOCK_C tile. The four shifted loads overlap almost entirely,
so after the first one they hit L1/L2: DRAM traffic stays close to the minimum of one
read and one write per cell (8 bytes), and the kernel is bandwidth bound. Tiles are
wide along columns so every row segment is a coalesced 512-byte access.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def jacobi_kernel(src, dst, rows, cols, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)[:, None]
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]
    inb = (r < rows) & (c < cols)
    interior = (r > 0) & (r < rows - 1) & (c > 0) & (c < cols - 1)
    p = src + r * cols + c

    center = tl.load(p, mask=inb & ~interior, other=0.0)        # only the boundary needs it
    up = tl.load(p - cols, mask=interior, other=0.0)
    down = tl.load(p + cols, mask=interior, other=0.0)
    left = tl.load(p - 1, mask=interior, other=0.0)
    right = tl.load(p + 1, mask=interior, other=0.0)
    val = tl.where(interior, 0.25 * (up + down + left + right), center)
    tl.store(dst + r * cols + c, val, mask=inb)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    BLOCK_R, BLOCK_C = 16, 128
    grid = (triton.cdiv(rows, BLOCK_R), triton.cdiv(cols, BLOCK_C))
    jacobi_kernel[grid](input, output, rows, cols, BLOCK_R=BLOCK_R, BLOCK_C=BLOCK_C, num_warps=4)


if __name__ == "__main__":
    from harness import DEVICE, ON_GPU, bench, check

    def ref(x):
        y = x.clone()
        y[1:-1, 1:-1] = 0.25 * (x[:-2, 1:-1] + x[2:, 1:-1] + x[1:-1, :-2] + x[1:-1, 2:])
        return y

    shapes = [(1, 1), (1, 9), (9, 1), (2, 2), (3, 3), (17, 130), (100, 257)]
    shapes += [(8192, 8192)] if ON_GPU else []
    for rows, cols in shapes:
        x = torch.empty(rows, cols, device=DEVICE).uniform_(-100, 100)
        y = torch.empty_like(x)
        solve(x, y, rows, cols)
        check(f"{rows}x{cols}", y, ref(x), 1e-5, 1e-5)
    bench("jacobi 8192^2", lambda: solve(x, y, rows, cols), lambda: ref(x),
          bytes_moved=2 * 4 * rows * cols)
