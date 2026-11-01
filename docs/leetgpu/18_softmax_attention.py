"""LeetGPU 6 - Softmax attention: out = softmax(Q K^T / sqrt(d)) V
(float32; Q: M x d, K, V: N x d; d <= 128; perf M = 512, N = 256, d = 128; tol 1e-4).

FlashAttention-style: a program owns BLOCK_M query rows and streams K and V in
BLOCK_N tiles. The M x N score matrix is never written to memory. The softmax is
computed *online*: keep a running row max m and row sum l, and when a new tile raises
the max, rescale what was accumulated so far by exp(m_old - m_new).

Small-problem note: with M = 512 and BLOCK_M = 64 there are only 8 programs for 132 SMs
on an H100. Autotuning includes BLOCK_M = 16 (32 programs); for very small M you would
also split N across programs and merge the partial (m, l, acc) at the end
(flash-decoding).

Both dots use "tf32x3" on Ampere/Hopper for FP32-level accuracy and "ieee" on T4, which
has no TF32 tensor cores (see _dot_precision).

Comments that start with `mem:` say where the data lives: registers, shared memory,
the L1/L2 caches, or DRAM (GDDR6 on a T4, HBM3 on an H100). See memory-hierarchy.md.
"""
import math
import os

import torch
import triton
import triton.language as tl

_CONFIGS = [
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=3),
]
if os.environ.get("TRITON_INTERPRET") == "1":
    _CONFIGS = _CONFIGS[:1]


@triton.autotune(configs=_CONFIGS, key=["M", "N", "d"])
@triton.jit
def attention_kernel(q, k, v, out, M, N, d, qk_scale,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, PREC: tl.constexpr):
    rm = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    rd = tl.arange(0, BLOCK_D)
    dmask = rd < d
    # mem: Q tile: DRAM -> registers, loaded once and reused for every K/V block
    qt = tl.load(q + rm[:, None] * d + rd[None, :], mask=(rm[:, None] < M) & dmask[None, :], other=0.0)

    # mem: m_i, l_i and acc are the softmax state: registers for the whole loop
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    for start in range(0, N, BLOCK_N):
        rn = start + tl.arange(0, BLOCK_N)
        nmask = rn < N
        kv_mask = nmask[:, None] & dmask[None, :]
        # mem: K and V tiles -> shared memory for tl.dot. Every program reads all of K and V
        # (128 KB each), so after the first program they come from L2
        kt = tl.load(k + rn[:, None] * d + rd[None, :], mask=kv_mask, other=0.0)
        vt = tl.load(v + rn[:, None] * d + rd[None, :], mask=kv_mask, other=0.0)

        # mem: s and p (BLOCK_M x BLOCK_N scores) exist only here, in registers.
        # The full M x N score matrix is never written to DRAM
        s = tl.dot(qt, tl.trans(kt), input_precision=PREC) * qk_scale   # log2 domain
        s = tl.where(nmask[None, :], s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        p = tl.exp2(s - m_new[:, None])
        alpha = tl.exp2(m_i - m_new)                 # rescale the old partial results
        l_i = l_i * alpha + tl.sum(p, axis=1)
        # mem: p goes registers -> dot operand layout -> tensor cores
        acc = acc * alpha[:, None] + tl.dot(p, vt, input_precision=PREC)
        m_i = m_new

    # mem: registers -> DRAM, written once
    tl.store(out + rm[:, None] * d + rd[None, :], acc / l_i[:, None],
             mask=(rm[:, None] < M) & dmask[None, :])


def _dot_precision(t):
    # TF32 tensor cores exist from Ampere (sm_80) on; T4 (sm_75) must use plain FP32 FMA.
    if t.is_cuda and torch.cuda.get_device_capability(t.device)[0] >= 8:
        return "tf32x3"
    return "ieee"


# Q, K, V, output are tensors on the GPU
def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor, M: int, N: int, d: int):
    BLOCK_D = max(16, triton.next_power_of_2(d))     # tl.dot needs every dim >= 16
    qk_scale = 1.0 / math.sqrt(d) * 1.4426950408889634   # fold log2(e): exp(x) = exp2(x*log2e)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    attention_kernel[grid](Q, K, V, output, M, N, d, qk_scale, BLOCK_D=BLOCK_D, PREC=_dot_precision(Q))


if __name__ == "__main__":
    from harness import DEVICE, ON_GPU, bench, check

    def ref(Q, K, V):
        return torch.softmax(Q @ K.T / Q.shape[1] ** 0.5, dim=1) @ V

    shapes = [(1, 1, 1), (2, 3, 4), (17, 100, 33), (70, 129, 128), (5, 7, 100)]
    shapes += [(512, 256, 128), (4096, 4096, 64)] if ON_GPU else []
    for M, N, d in shapes:
        Q = torch.empty(M, d, device=DEVICE).uniform_(-1, 1)
        K = torch.empty(N, d, device=DEVICE).uniform_(-1, 1)
        V = torch.empty(N, d, device=DEVICE).uniform_(-1, 1)
        out = torch.empty(M, d, device=DEVICE)
        solve(Q, K, V, out, M, N, d)
        check(f"M={M} N={N} d={d}", out, ref(Q.double(), K.double(), V.double()), 1e-4, 1e-4)
    if ON_GPU:
        M, N, d = 512, 256, 128
        Q, K, V = (torch.empty(n, d, device=DEVICE).uniform_(-0.1, 0.1) for n in (M, N, N))
        out = torch.empty(M, d, device=DEVICE)
        sdpa = torch.nn.functional.scaled_dot_product_attention
        bench("attention 512x256x128", lambda: solve(Q, K, V, out, M, N, d),
              lambda: sdpa(Q[None], K[None], V[None]), flops=4 * M * N * d)
