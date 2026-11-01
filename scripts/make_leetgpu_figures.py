"""Generate the figures and animations for the LeetGPU solution pages.

Run from the repo root:

    .venv/bin/python scripts/make_leetgpu_figures.py

Writes PNG and GIF files into docs/assets/img/leetgpu/. Reuses the palette and helpers
from make_figures.py so both sets look the same.
"""
import numpy as np

import make_figures as mf
from make_figures import (AMBER, AMBER_L, BLUE, BLUE_L, GRAY, GRAY_L, INK, MONO, PURPLE,
                          PURPLE_L, RED, RED_L, TEAL, TEAL_L, arrow, box, canvas, plt, reset)

OUT = mf.OUT / "leetgpu"
OUT.mkdir(parents=True, exist_ok=True)
mf.OUT = OUT                      # save() / save_anim() write here from now on
save, save_anim = mf.save, mf.save_anim


def cells(ax, x0, y0, vals, w=1.0, h=1.0, fcs=None, fs=10, ec="white"):
    for i, v in enumerate(vals):
        fc = fcs[i] if fcs else GRAY_L
        box(ax, x0 + i * w, y0, w, h, fc, ec, lw=1.5)
        ax.text(x0 + i * w + w / 2, y0 + h / 2, v, ha="center", va="center", fontsize=fs)


# ---------------------------------------------------------------- 02 transpose

def transpose():
    n, s = 8, 0.62
    fig, ax = canvas(12, 4.8, (0, 24), (1.6, 11.2))
    frames = list(range(n * 2)) + [n * 2 - 1] * 3

    def mat(x0, top, hl, color, label):
        for r in range(n):
            for c in range(n):
                fc = color if (r, c) in hl else GRAY_L
                box(ax, x0 + c * s, top - (r + 1) * s, s, s, fc, "white", lw=1)
        ax.text(x0 + n * s / 2, top + 0.35, label, ha="center", fontsize=11)

    def update(f):
        reset(ax, (0, 24), (1.6, 11.2))
        ax.text(0.3, 10.6, "Naive: one warp reads a row, but writes a column", fontsize=12, weight="bold", color=RED)
        ax.text(12.3, 10.6, "Tiled: read a tile row-wise, write it row-wise", fontsize=12, weight="bold", color=TEAL)
        r = f % n
        mat(0.4, 9.4, {(r, c) for c in range(n)}, BLUE, "input (read: coalesced)")
        mat(6.0, 9.4, {(c, r) for c in range(n)}, RED, "output (write: 8 sectors)")
        t = (f // 4) % 4
        tr, tc = divmod(t, 2)
        tile = {(tr * 4 + i, tc * 4 + j) for i in range(4) for j in range(4)}
        mat(12.4, 9.4, tile, BLUE, "input tile")
        mat(18.0, 9.4, {(c, rr) for rr, c in tile}, TEAL, "output tile")
        arrow(ax, (14.9, 4.0), (20.5, 4.0), AMBER)
        ax.text(17.7, 3.35, "tl.trans: shuffle through shared memory (SRAM)", ha="center", fontsize=9, color=AMBER)
        ax.text(0.4, 2.6, "Each output column write touches a different 32-byte sector:\n"
                "~8x the memory transactions of a coalesced write.", fontsize=10, color=RED)
        ax.text(12.4, 2.6, "Both global load and store are contiguous rows.", fontsize=10, color=TEAL)

    save_anim(fig, update, frames, "transpose.gif", fps=3)


# ---------------------------------------------------------------- 03/04 reduction

def reduction():
    P, nblk = 4, 16
    fig, ax = canvas(12, 4.6, (0, 24), (0, 9.2))
    colors = [TEAL, BLUE, AMBER, PURPLE]
    lights = [TEAL_L, BLUE_L, AMBER_L, PURPLE_L]
    frames = list(range(nblk // P + 3)) + [nblk // P + 2] * 3

    def update(f):
        reset(ax, (0, 24), (0, 9.2))
        ax.text(0.3, 8.6, "Pass 1: P persistent programs grid-stride over the array", fontsize=12, weight="bold")
        step = min(f, nblk // P)
        for b in range(nblk):
            p = b % P
            done = b // P < step
            box(ax, 0.5 + b * 1.4, 6.4, 1.3, 1.0, colors[p] if done else lights[p], "white")
            ax.text(1.15 + b * 1.4, 6.9, f"p{p}", ha="center", va="center", fontsize=9,
                    color="white" if done else INK)
        ax.text(0.5, 5.8, "block b is handled by program b mod P; each lane keeps a running vector sum",
                fontsize=10, color=GRAY)
        for p in range(P):
            x = 3 + p * 4.5
            box(ax, x, 3.4, 2.4, 1.0, colors[p] if step == nblk // P else GRAY_L, "white")
            ax.text(x + 1.2, 3.9, f"partial[{p}]", ha="center", va="center", fontsize=10,
                    color="white" if step == nblk // P else INK)
        if f >= nblk // P + 1:
            ax.text(13.2, 0.75, "Pass 2: one program adds the P partials\nin float64 (fixed order -> same bits every run)",
                    fontsize=10, weight="bold")
            for p in range(P):
                arrow(ax, (4.2 + p * 4.5, 3.3), (10.5, 1.4), GRAY, lw=1.2)
            box(ax, 9.3, 0.4, 3.0, 1.0, TEAL if f >= nblk // P + 2 else GRAY_L, "white")
            ax.text(10.8, 0.9, "output[0]", ha="center", va="center", fontsize=11,
                    color="white" if f >= nblk // P + 2 else INK)

    save_anim(fig, update, frames, "two-pass-reduction.gif", fps=1.5)


# ---------------------------------------------------------------- 05 softmax

def softmax_partials():
    fig, ax = canvas(12, 4.2, (0, 24), (0, 8.4))
    ax.text(0.3, 7.8, "1-D softmax across programs: partials, then merge", fontsize=12, weight="bold")
    xs = [0.5, 6.3, 12.1, 17.9]
    ms = ["2.1", "4.7", "3.3", "4.9"]
    for i, x in enumerate(xs):
        box(ax, x, 5.6, 5.4, 1.2, BLUE_L, "white")
        ax.text(x + 2.7, 6.2, f"chunk {i}  (BLOCK elements)", ha="center", va="center", fontsize=10)
        box(ax, x + 0.9, 3.8, 3.6, 1.0, AMBER_L, "white")
        ax.text(x + 2.7, 4.3, f"m{i}={ms[i]}, s{i}", ha="center", va="center", fontsize=10, family=MONO)
        arrow(ax, (x + 2.7, 5.5), (x + 2.7, 4.9), GRAY, lw=1.2)
        arrow(ax, (x + 2.7, 3.7), (12.0, 2.4), GRAY, lw=1.0)
    box(ax, 2.5, 1.0, 19.0, 1.3, TEAL_L, "white")
    ax.text(12.0, 1.65, "M = max(m_b)     S = sum_b s_b * exp(m_b - M)     y = exp(x - M) / S",
            ha="center", va="center", fontsize=11, family=MONO)
    ax.text(0.5, 0.2, "kernel 1 writes (m_b, s_b); kernel 2 merges them in every program and writes y",
            fontsize=10, color=GRAY)
    save(fig, "softmax-partials.png")


# ---------------------------------------------------------------- 08/17 fusion traffic

def fusion_traffic():
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.4))
    cases = [
        ("Residual add + RMSNorm\n(bytes per element)", ["PyTorch eager", "Fused Triton"], [40, 12]),
        ("LayerNorm -> Linear -> GELU\n(extra DRAM traffic, M=K=N=4096)", ["Unfused", "Fused"], [256, 0.03]),
    ]
    for ax, (title, labels, vals) in zip(axes, cases):
        bars = ax.barh(labels, vals, color=[RED, TEAL])
        ax.invert_yaxis()
        unit = "B" if "element" in title else "MB"
        for b, v in zip(bars, vals):
            ax.text(b.get_width() + max(vals) * 0.01, b.get_y() + b.get_height() / 2,
                    f"{v:g} {unit}", va="center", fontsize=10)
        ax.set_title(title, fontsize=11, weight="bold", loc="left")
        ax.set_xlim(0, max(vals) * 1.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    save(fig, "fusion-traffic.png")


# ---------------------------------------------------------------- 09 grouped ordering

def grouped_order():
    n, s = 6, 0.9
    fig, ax = canvas(12, 4.4, (0, 24), (0, 8.8))
    for k, (title, order) in enumerate([("Row-major program order", "row"), ("Grouped order (GROUP_M = 3)", "group")]):
        x0 = 1.0 + k * 12
        ax.text(x0, 8.2, title, fontsize=12, weight="bold")
        for pid in range(n * n):
            if order == "row":
                r, c = divmod(pid, n)
            else:
                g, rem = divmod(pid, 3 * n)
                r, c = g * 3 + rem % 3, rem // 3
            first = pid < 9
            box(ax, x0 + c * s, 7.2 - (r + 1) * s, s, s, TEAL if first else GRAY_L, "white")
            ax.text(x0 + c * s + s / 2, 7.2 - (r + 1) * s + s / 2, str(pid), ha="center", va="center",
                    fontsize=9, color="white" if first else INK)
        need_a, need_b = (2, 6) if order == "row" else (3, 3)
        ax.text(x0 + 6.0, 5.2, f"first 9 programs need\n{need_a} rows of A + {need_b} cols of B\n"
                f"= {need_a + need_b} tiles in L2", fontsize=10, color=TEAL if order == "group" else RED)
    save(fig, "grouped-order.png")


# ---------------------------------------------------------------- 11 scatter add

def scatter_add():
    fig, ax = canvas(12, 4.6, (0, 24), (0, 9.2))
    ax.text(0.3, 8.6, "Scatter add: many source rows land on the same output row", fontsize=12, weight="bold")
    idx = [0, 2, 0, 3, 2, 0, 1, 3]
    cols = [TEAL, BLUE, AMBER, PURPLE]
    for i, t in enumerate(idx):
        box(ax, 1.0, 7.4 - i * 0.9, 4.0, 0.75, GRAY_L, "white")
        ax.text(3.0, 7.77 - i * 0.9, f"src[{i}]   index={t}", ha="center", va="center", fontsize=10, family=MONO)
        arrow(ax, (5.1, 7.77 - i * 0.9), (12.9, 6.6 - t * 1.6 + 0.35), cols[t], lw=1.3)
    for m in range(4):
        hits = idx.count(m)
        box(ax, 13.0, 6.6 - m * 1.6, 4.4, 0.75, [TEAL_L, BLUE_L, AMBER_L, PURPLE_L][m], "white")
        ax.text(15.2, 6.97 - m * 1.6, f"out[{m}]  ({hits} adds)", ha="center", va="center", fontsize=10, family=MONO)
    ax.text(18.0, 6.0, "atomics: order varies\nrun to run (not bitwise\nreproducible)", fontsize=10, color=RED)
    ax.text(18.0, 3.0, "sort by index first:\none program per output\nrow, fixed order", fontsize=10, color=TEAL)
    save(fig, "scatter-add.png")


# ---------------------------------------------------------------- 12/13 pairwise tiling

def pairwise_tiling():
    nb, s = 6, 1.1
    fig, ax = canvas(12, 5.6, (0, 24), (0, 11.2))
    frames = list(range(nb + 1)) + [nb] * 3

    def update(f):
        reset(ax, (0, 24), (0, 11.2))
        ax.text(0.3, 10.6, "All-pairs kernel: one program owns an i-block, streams j-tiles", fontsize=12, weight="bold")
        x0, top = 3.0, 9.4
        for r in range(nb):
            for c in range(nb):
                fc = GRAY_L
                if r == 2 and c < f:
                    fc = "#9fcfc4"
                if r == 2 and c == f:
                    fc = TEAL
                box(ax, x0 + c * s, top - (r + 1) * s, s, s, fc, "white")
        ax.text(x0 + nb * s / 2, top + 0.25, "j tiles (BLOCK_J points)", ha="center", fontsize=10)
        ax.text(x0 - 0.3, top - 2.5 * s, "program 2\n(BLOCK_I\npoints)", ha="right", va="center", fontsize=10)
        tx = 12.0
        lines = ["load x_i, y_i, z_i once (registers)", "best_d = inf  /  F = 0"]
        if f < nb:
            lines += [f"tile j={f}: load BLOCK_J points", "  d[i, j] = |p_i - p_j|^2  (BI x BJ)",
                      "  nearest: tl.min(d, 1, return_indices)", "  n-body : F += sum_j m_j r rsqrt(d)^3"]
        else:
            lines += ["all tiles done:", "  store best index / force", "", "N x N matrix never hits DRAM"]
        for i, line in enumerate(lines):
            ax.text(tx, 9.0 - i * 0.8, line, family=MONO, fontsize=10)
        ax.text(0.5, 1.2, "Each j-tile load is reused by BLOCK_I points -> O(N^2) math, O(N) memory: compute bound",
                fontsize=10, color=GRAY)

    save_anim(fig, update, frames, "pairwise-tiling.gif", fps=1.5)


# ---------------------------------------------------------------- 14 jacobi

def jacobi():
    n, s = 10, 0.62
    fig, ax = canvas(12, 4.6, (0, 24), (0, 9.2))
    ax.text(0.3, 8.6, "5-point Jacobi: interior averages 4 neighbours, boundary is copied", fontsize=12, weight="bold")
    x0, top = 1.0, 7.8
    for r in range(n):
        for c in range(n):
            boundary = r in (0, n - 1) or c in (0, n - 1)
            fc = AMBER_L if boundary else GRAY_L
            if 2 <= r < 5 and 2 <= c < 7:
                fc = BLUE_L
            box(ax, x0 + c * s, top - (r + 1) * s, s, s, fc, "white")
    pr, pc = 3, 4
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        box(ax, x0 + (pc + dc) * s, top - (pr + dr + 1) * s, s, s, TEAL_L, TEAL, lw=1.5)
    box(ax, x0 + pc * s, top - (pr + 1) * s, s, s, TEAL, TEAL)
    tx = 9.0
    for i, (fc, text) in enumerate([(AMBER_L, "boundary: out = in"), (BLUE_L, "one program's BLOCK_R x BLOCK_C tile"),
                                    (TEAL, "cell being computed"), (TEAL_L, "its 4 neighbours (loads overlap -> L1/L2 hits)")]):
        box(ax, tx, 6.6 - i * 1.1, 0.7, 0.7, fc, "white")
        ax.text(tx + 1.0, 6.95 - i * 1.1, text, va="center", fontsize=10)
    ax.text(tx, 1.4, "DRAM traffic ~ 1 read + 1 write per cell (8 B): bandwidth bound.\n"
            "Wide tiles along columns keep each row segment coalesced.", fontsize=10, color=GRAY)
    save(fig, "jacobi.png")


# ---------------------------------------------------------------- 15 spmm

def spmm():
    fig, ax = canvas(12, 4.6, (0, 24), (0, 9.2))
    ax.text(0.3, 8.6, "Same A, two layouts: dense tiles vs CSR", fontsize=12, weight="bold")
    rng = np.random.default_rng(3)
    A = (rng.random((6, 8)) > 0.65) * rng.integers(1, 9, (6, 8))
    s = 0.7
    for r in range(6):
        for c in range(8):
            nz = A[r, c] != 0
            box(ax, 0.6 + c * s, 7.6 - (r + 1) * s, s, s, BLUE if nz else GRAY_L, "white")
            if nz:
                ax.text(0.6 + c * s + s / 2, 7.6 - (r + 1) * s + s / 2, str(A[r, c]), ha="center", va="center",
                        fontsize=9, color="white")
    ax.text(0.6, 2.2, "~35% non-zero: tensor cores on dense tiles\nusually win (regular loads, one pass over A)",
            fontsize=10, color=TEAL)
    nz = np.argwhere(A)
    row_ptr = np.concatenate([[0], np.cumsum((A != 0).sum(1))])
    y = 6.6
    for name, arr, fc in [("row_ptr", row_ptr, AMBER_L), ("col_idx", nz[:, 1], PURPLE_L), ("values", A[A != 0], BLUE_L)]:
        ax.text(7.6, y + 0.3, name, fontsize=10, family=MONO, ha="right")
        vals = [str(v) for v in arr][:20]
        cells(ax, 7.9, y, vals, w=0.75, h=0.6, fcs=[fc] * len(vals), fs=9)
        y -= 1.3
    ax.text(7.9, 2.0, "CSR: only non-zeros stored, but B rows are gathered by col_idx\n"
            "(irregular, CUDA cores). Pays off below ~1-5% density.", fontsize=10, color=GRAY)
    save(fig, "spmm-layouts.png")


# ---------------------------------------------------------------- 16 segmented scan

def segmented_scan():
    fig, ax = canvas(12, 4.8, (0, 24), (0, 9.6))
    ax.text(0.3, 9.0, "Segmented exclusive scan via the shifted-input trick", fontsize=12, weight="bold")
    v = [3, 1, 4, 1, 5, 9, 2, 6]
    f = [1, 0, 0, 1, 0, 0, 1, 0]
    w = [0 if f[i] else v[i - 1] for i in range(8)]
    out, acc = [], 0
    for i in range(8):
        acc = w[i] if f[i] else acc + w[i]
        out.append(acc)
    seg = np.cumsum(f) - 1
    segc = [TEAL_L, BLUE_L, AMBER_L]
    rows = [("values", v), ("flags", f), ("w = flag ? 0 : v[i-1]", w), ("inclusive seg-scan(w)", out)]
    for k, (name, arr) in enumerate(rows):
        y = 7.4 - k * 1.25
        ax.text(6.3, y + 0.4, name, fontsize=10, family=MONO, ha="right")
        cells(ax, 6.6, y, [str(a) for a in arr], w=1.1, h=0.8, fcs=[segc[s] for s in seg], fs=10)
    ax.text(16.2, 7.6, "(v1,f1) (+) (v2,f2) =\n  (f2 ? v2 : v1+v2, f1|f2)", fontsize=10, family=MONO)
    ax.text(16.2, 5.7, "associative -> tl.associative_scan", fontsize=10, color=TEAL)
    ax.text(16.2, 3.4, "across blocks:\n 1. block aggregates\n 2. scan aggregates (fp64)\n 3. rescan + add carry",
            fontsize=10)
    ax.text(0.5, 1.4, "Inclusive scan of the shifted w equals the exclusive scan of v, with no subtraction "
            "(subtracting loses precision).", fontsize=10, color=GRAY)
    save(fig, "segmented-scan.png")


# ---------------------------------------------------------------- 17 ln + gemm fusion

def ln_gemm_fusion():
    fig, ax = canvas(12, 4.4, (0, 24), (0, 8.8))
    ax.text(0.3, 8.2, "Prologue + epilogue fusion around one GEMM", fontsize=12, weight="bold")
    steps = [(0.5, BLUE_L, "load x tile\n[BM, BK]"), (4.6, AMBER_L, "(x - mu) * rstd\n* gamma + beta"),
             (8.7, TEAL, "tl.dot(a, W tile)\nacc += ..."), (12.8, AMBER_L, "+ bias\nGELU(erf)"),
             (16.9, BLUE_L, "store y tile\n[BM, BN]")]
    for x, fc, t in steps:
        box(ax, x, 4.6, 3.6, 1.8, fc, "white")
        ax.text(x + 1.8, 5.5, t, ha="center", va="center", fontsize=10, color="white" if fc == TEAL else INK)
    for x in (4.15, 8.25, 12.35, 16.45):
        arrow(ax, (x, 5.5), (x + 0.4, 5.5), GRAY, lw=1.5)
    ax.text(4.6, 3.9, "prologue (in registers)", fontsize=10, color=AMBER)
    ax.text(12.8, 3.9, "epilogue (in registers)", fontsize=10, color=AMBER)
    ax.annotate("", xy=(4.6, 7.0), xytext=(12.3, 7.0),
                arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.5, connectionstyle="arc3,rad=0.25"))
    ax.text(8.5, 7.5, "loop over K", ha="center", fontsize=10, color=TEAL)
    ax.text(0.5, 2.2, "mu and rstd per row come from a tiny stats kernel (2 x M floats).\n"
            "xn (M x K) and h (M x N) never exist in DRAM.", fontsize=10, color=GRAY)
    save(fig, "ln-gemm-fusion.png")


# ---------------------------------------------------------------- 18 flash attention

def flash_attention():
    nkv = 5
    fig, ax = canvas(12, 4.2, (0, 24), (2.0, 10.4))
    frames = list(range(nkv + 1)) + [nkv] * 3
    rng = np.random.default_rng(1)
    maxes = np.maximum.accumulate(rng.uniform(1, 5, nkv))

    def update(f):
        reset(ax, (0, 24), (2.0, 10.4))
        ax.text(0.3, 9.8, "Online softmax: stream K/V tiles, rescale as the max grows", fontsize=12, weight="bold")
        box(ax, 0.6, 5.6, 2.2, 2.4, BLUE, "white")
        ax.text(1.7, 6.8, "Q block\n[BM, d]", ha="center", va="center", color="white", fontsize=10)
        for t in range(nkv):
            fc = AMBER if t == f else ("#e8cf9b" if t < f else GRAY_L)
            box(ax, 4.0 + t * 1.5, 7.0, 1.3, 1.6, fc, "white")
            box(ax, 4.0 + t * 1.5, 4.8, 1.3, 1.6, PURPLE if t == f else ("#c9a9d6" if t < f else GRAY_L), "white")
        ax.text(3.7, 7.8, "K", ha="right", va="center", fontsize=11)
        ax.text(3.7, 5.6, "V", ha="right", va="center", fontsize=11)
        tx = 12.4
        if f < nkv:
            m_old = maxes[f - 1] if f else float("-inf")
            lines = [f"tile {f}:", "  s = Q @ K_t^T * scale", f"  m_new = max(m, rowmax(s)) = {maxes[f]:.2f}",
                     f"  alpha = exp(m_old - m_new) = {np.exp(m_old - maxes[f]) if f else 0:.2f}",
                     "  p = exp(s - m_new)", "  l   = l * alpha + rowsum(p)", "  acc = acc * alpha + p @ V_t"]
        else:
            lines = ["done:", "  out = acc / l", "", "scores (M x N) never written", "to DRAM: memory O(M d)"]
        for i, line in enumerate(lines):
            ax.text(tx, 8.6 - i * 0.75, line, family=MONO, fontsize=10)
        ax.text(0.6, 2.6, "Registers per program: running max m, running sum l, accumulator acc [BM, d].",
                fontsize=10, color=GRAY)

    save_anim(fig, update, frames, "flash-attention.gif", fps=1.2)


# ---------------------------------------------------------------- 19/20 norm backward

def norm_backward():
    n, c, s = 6, 10, 0.6
    fig, ax = canvas(12, 4.6, (0, 24), (0, 9.2))
    ax.text(0.3, 8.6, "Norm backward: dx reduces along rows, dw along columns", fontsize=12, weight="bold")
    x0, top = 1.0, 7.6
    pcols = [TEAL_L, BLUE_L, AMBER_L]
    for r in range(n):
        for k in range(c):
            box(ax, x0 + k * s, top - (r + 1) * s, s, s, pcols[r % 3], "white")
        ax.text(x0 + c * s + 0.2, top - (r + 0.5) * s, f"p{r % 3}: row sums -> dx", va="center", fontsize=9)
    ax.text(x0, top - n * s - 0.6, "program p takes rows p, p+P, ... (P = 3 here)", fontsize=10, color=GRAY)
    px = 12.5
    ax.text(px, 7.3, "partial dw  [P, C]  (registers -> one store each)", fontsize=10)
    for p in range(3):
        for k in range(c):
            box(ax, px + k * s, 6.3 - p * s, s, s, pcols[p], "white")
    for k in range(c):
        box(ax, px + k * s, 3.6, s, s, TEAL, "white")
    arrow(ax, (px + c * s / 2, 5.0), (px + c * s / 2, 4.3), GRAY)
    ax.text(px, 3.0, "kernel 2: column sums -> dw  (deterministic)", fontsize=10, color=TEAL)
    save(fig, "norm-backward.png")


# ---------------------------------------------------------------- memory hierarchy

def memory_hierarchy():
    fig, ax = canvas(14, 7.0, (0, 28), (0, 14))
    ax.text(0.3, 13.3, "Where the data lives", fontsize=14, weight="bold", color=INK)
    for x, label in ((9.6, "T4 (sm_75)"), (14.2, "H100 SXM (sm_90)"), (19.2, "Triton code that lands here")):
        ax.text(x, 12.3, label, fontsize=11, weight="bold", color=INK)
    rows = [
        ("Registers", "per thread", BLUE_L, BLUE, 3.4,
         "256 KB / SM", "256 KB / SM",
         "block values from tl.load,\naccumulators, loop-carried state"),
        ("Shared memory / L1", "SRAM, per SM", TEAL_L, TEAL, 5.0,
         "96 KB / SM\n(≤ 64 KB shared)", "256 KB / SM\n(≤ 228 KB shared)",
         "tl.dot operand tiles (num_stages),\ntl.trans, cross-warp tl.sum / scan"),
        ("L2 cache", "SRAM, whole GPU", AMBER_L, AMBER, 6.6,
         "4 MB", "50 MB",
         "every load and store passes here;\ntl.atomic_add; re-reads of hot data"),
        ("DRAM", "GDDR6 on T4, HBM3 on H100", PURPLE_L, PURPLE, 8.2,
         "16 GB\n320 GB/s", "80 GB\n3,350 GB/s",
         "torch tensors: inputs, outputs,\npartials, register spills"),
    ]
    y = 10.0
    for name, sub, fc, ec, w, t4, h100, triton_txt in rows:
        x0 = 5.0 - w / 2
        box(ax, x0, y, w, 2.0, fc, ec, lw=2)
        ax.text(5.0, y + 1.25, name, ha="center", fontsize=11, weight="bold", color=INK)
        ax.text(5.0, y + 0.55, sub, ha="center", fontsize=8.5, color=GRAY)
        ax.text(9.6, y + 1.0, t4, va="center", fontsize=10, color=INK)
        ax.text(14.2, y + 1.0, h100, va="center", fontsize=10, color=INK)
        ax.text(19.2, y + 1.0, triton_txt, va="center", fontsize=9.5, family=MONO, color=INK)
        y -= 2.7
    arrow(ax, (0.35, 11.8), (0.35, 2.0), GRAY)
    ax.text(0.9, 0.5, "each level down: bigger, ~several x slower;\nDRAM is ~100x a register's latency",
            fontsize=9, color=GRAY)
    save(fig, "memory-hierarchy.png")


if __name__ == "__main__":
    transpose()
    reduction()
    softmax_partials()
    fusion_traffic()
    grouped_order()
    scatter_add()
    pairwise_tiling()
    jacobi()
    spmm()
    segmented_scan()
    ln_gemm_fusion()
    flash_attention()
    norm_backward()
    memory_hierarchy()
