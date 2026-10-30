"""LeetGPU 3 - Matrix Transpose: output[c, r] = input[r, c] (float32, perf 7000 x 6000).

The starter launches one program per *element* (grid = rows x cols): 42M programs, each
moving 8 bytes. This version moves a BLOCK x BLOCK tile per program. The tile is read
row-by-row from `input` (coalesced) and written row-by-row to `output` (coalesced);
`tl.trans` makes the compiler re-shuffle the tile through shared memory in between.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def matrix_transpose_kernel(input, output, rows, cols,
                            stride_ir, stride_ic, stride_or, stride_oc,
                            BLOCK: tl.constexpr):
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    r = pid_r * BLOCK + tl.arange(0, BLOCK)
    c = pid_c * BLOCK + tl.arange(0, BLOCK)
    mask = (r[:, None] < rows) & (c[None, :] < cols)

    # [BLOCK_r, BLOCK_c] tile, contiguous along c -> coalesced read
    tile = tl.load(input + r[:, None] * stride_ir + c[None, :] * stride_ic, mask=mask)

    # [BLOCK_c, BLOCK_r] tile, contiguous along r -> coalesced write
    out_ptrs = output + c[:, None] * stride_or + r[None, :] * stride_oc
    tl.store(out_ptrs, tl.trans(tile), mask=tl.trans(mask))


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    stride_ir, stride_ic = cols, 1
    stride_or, stride_oc = rows, 1
    BLOCK = 64
    grid = (triton.cdiv(rows, BLOCK), triton.cdiv(cols, BLOCK))
    matrix_transpose_kernel[grid](
        input, output, rows, cols, stride_ir, stride_ic, stride_or, stride_oc,
        BLOCK=BLOCK, num_warps=8,
    )


if __name__ == "__main__":
    from harness import DEVICE, ON_GPU, bench, check

    shapes = [(1, 1), (1, 70), (70, 1), (63, 65), (129, 31), (200, 333)]
    shapes += [(7000, 6000)] if ON_GPU else []
    for rows, cols in shapes:
        x = torch.empty(rows, cols, device=DEVICE).uniform_(-10, 10)
        y = torch.empty(cols, rows, device=DEVICE)
        solve(x, y, rows, cols)
        check(f"{rows}x{cols}", y, x.t(), 1e-5, 1e-5)
    bench("transpose 7000x6000", lambda: solve(x, y, rows, cols),
          lambda: y.copy_(x.t()), bytes_moved=2 * 4 * rows * cols)
