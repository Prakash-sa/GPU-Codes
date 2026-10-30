"""Generate the figures and animations used in the docs.

Run from the repo root:

    .venv/bin/pip install -r requirements-figures.txt
    .venv/bin/python scripts/make_figures.py

Writes PNG and GIF files into docs/assets/img/.
"""
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patches  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "assets" / "img"
OUT.mkdir(parents=True, exist_ok=True)

TEAL, TEAL_L = "#0f6b5c", "#cfe7e1"
BLUE, BLUE_L = "#2f6db3", "#d6e4f5"
AMBER, AMBER_L = "#c98a1e", "#f6e3c0"
PURPLE, PURPLE_L = "#7d3c98", "#e8daef"
RED, RED_L = "#c0392b", "#f5d0cb"
GRAY, GRAY_L = "#8a9490", "#eceeed"
INK = "#1b211f"
GPU_COLORS = [TEAL, BLUE, AMBER, PURPLE]
GPU_COLORS_L = [TEAL_L, BLUE_L, AMBER_L, PURPLE_L]
MONO = "DejaVu Sans Mono"

plt.rcParams.update({
    "font.size": 11,
    "axes.edgecolor": GRAY,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


# ---------------------------------------------------------------- helpers

def save(fig, name):
    fig.savefig(OUT / name, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def save_anim(fig, update, frames, name, fps):
    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(OUT / name, writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)
    print("wrote", name)


def canvas(w, h, xlim, ylim):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(0.01, 0.01, 0.99, 0.99)
    reset(ax, xlim, ylim)
    return fig, ax


def reset(ax, xlim, ylim):
    ax.clear()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")


def box(ax, x, y, w, h, fc, ec=INK, lw=1.0, **kw):
    r = patches.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=lw, **kw)
    ax.add_patch(r)
    return r


def arrow(ax, xy_from, xy_to, color=INK, lw=2.0):
    ax.annotate("", xy=xy_to, xytext=xy_from,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=18))


# ---------------------------------------------------------------- 01 execution model

def thread_hierarchy():
    fig, ax = canvas(12, 4.8, (0, 24), (0, 9.6))
    ax.text(0.3, 8.9, "Grid  (gridDim = 3 × 2 blocks)", fontsize=13, weight="bold")
    bw, gap, x0, y0 = 2.2, 0.35, 0.3, 2.0
    hl_pos = None
    for by in range(2):
        for bx in range(3):
            x = x0 + bx * (bw + gap)
            y = y0 + (1 - by) * (bw + gap)
            hl = (bx, by) == (2, 0)
            if hl:
                hl_pos = (x, y)
            box(ax, x, y, bw, bw, TEAL_L if hl else GRAY_L, TEAL if hl else GRAY, lw=2.5 if hl else 1)
            ax.text(x + bw / 2, y + bw / 2, f"Block\n({bx},{by})", ha="center", va="center",
                    fontsize=10, weight="bold" if hl else "normal")
    ax.text(0.3, 1.2, "Blocks run independently,\nin any order, on any SM.", fontsize=10, color=GRAY)

    zx, zy, cw, ch, rowgap, nwarps = 9.4, 1.6, 0.43, 0.8, 0.55, 4
    zw = 32 * cw
    zh = nwarps * ch + (nwarps - 1) * rowgap
    ax.text(zx, 8.9, "Block (2,0)  ·  blockDim = 128 threads = 4 warps", fontsize=13, weight="bold")
    for w in range(nwarps):
        y = zy + (nwarps - 1 - w) * (ch + rowgap)
        ax.text(zx, y + ch + 0.1, f"warp {w}:  threadIdx.x {32 * w}–{32 * w + 31}", fontsize=9, color=GRAY)
        for lane in range(32):
            tid = 32 * w + lane
            fc = TEAL if tid == 37 else (TEAL_L if w % 2 == 0 else BLUE_L)
            box(ax, zx + lane * cw, y, cw, ch, fc, "white", lw=0.8)
    box(ax, zx - 0.15, zy - 0.15, zw + 0.3, zh + 0.6, "none", TEAL, lw=1.5, linestyle="--")
    hx, hy = hl_pos
    ax.plot([hx + bw, zx - 0.15], [hy + bw, zy + zh + 0.45], "--", color=TEAL, lw=1)
    ax.plot([hx + bw, zx - 0.15], [hy, zy - 0.15], "--", color=TEAL, lw=1)
    ax.text(zx, 0.55, "thread 37 →  i = blockIdx.x * blockDim.x + threadIdx.x = 2 * 128 + 37 = 293",
            fontsize=10.5, family=MONO, color=TEAL)
    save(fig, "thread-hierarchy.png")


def simulate_warps(nw, T, comp=2, lat=6):
    """Greedy SM scheduler: a warp runs `comp` cycles, then waits `lat` cycles on memory."""
    ready_at, left, cur = [0] * nw, [comp] * nw, None
    grid = np.zeros((nw, T), int)  # 0 ready-but-idle, 1 running, 2 waiting on memory
    issue = np.zeros(T, int)
    for t in range(T):
        ready = [w for w in range(nw) if ready_at[w] <= t]
        if cur not in ready:
            cur = min(ready, key=lambda w: (ready_at[w], w)) if ready else None
        for w in range(nw):
            grid[w, t] = 1 if w == cur else (2 if ready_at[w] > t else 0)
        if cur is not None:
            issue[t] = 1
            left[cur] -= 1
            if left[cur] == 0:
                left[cur], ready_at[cur], cur = comp, t + 1 + lat, None
    return grid, issue


def latency_hiding():
    T, cw, chh, x0 = 32, 0.5, 0.6, 3.2
    one = simulate_warps(1, T)
    four = simulate_warps(4, T)
    fig, ax = canvas(12, 6, (0, 20), (0, 10))
    colors = {0: GRAY_L, 1: TEAL, 2: AMBER_L}

    def panel(state, title, top, f):
        grid, issue = state
        busy = issue[:f].sum()
        util = f"   SM busy {100 * busy / f:.0f}%" if f else ""
        ax.text(0.2, top, title + util, fontsize=12, weight="bold")
        rows = [(f"warp {w}", grid[w]) for w in range(grid.shape[0])]
        for r, (label, row) in enumerate(rows):
            y = top - 1.0 - r * 0.75
            ax.text(x0 - 0.2, y + chh / 2, label, ha="right", va="center", fontsize=10)
            for t in range(f):
                box(ax, x0 + t * cw, y, cw, chh, colors[row[t]], "white", lw=0.6)
        y = top - 1.0 - len(rows) * 0.75
        ax.text(x0 - 0.2, y + chh / 2, "SM issues", ha="right", va="center", fontsize=10, weight="bold")
        for t in range(f):
            box(ax, x0 + t * cw, y, cw, chh, TEAL if issue[t] else RED_L, "white", lw=0.6)

    def update(f):
        reset(ax, (0, 20), (0, 10))
        panel(one, "1 warp per SM: the SM sits idle while the warp waits on memory", 9.6, f)
        panel(four, "4 warps per SM: another warp runs while the others wait", 6.6, f)
        for i, (c, label) in enumerate([(TEAL, "executing"), (AMBER_L, "waiting on memory"),
                                         (GRAY_L, "ready, not scheduled"), (RED_L, "SM idle")]):
            box(ax, x0 + i * 4.0, 0.3, 0.5, 0.45, c, "white")
            ax.text(x0 + i * 4.0 + 0.7, 0.52, label, va="center", fontsize=9.5)

    frames = list(range(T + 1)) + [T] * 10
    save_anim(fig, update, frames, "latency-hiding.gif", fps=6)


def warp_divergence():
    lines = ["x = in[i];", "if (lane % 2 == 0)", "    y = a(x);", "else", "    y = b(x);", "out[i] = y;"]
    everyone = [True] * 32
    steps = [(0, everyone), (1, everyone), (2, [k % 2 == 0 for k in range(32)]),
             (4, [k % 2 == 1 for k in range(32)]), (5, everyone)]
    fig, ax = canvas(12, 4.6, (0, 24), (0, 9.2))
    lx, cw = 9.6, 0.42

    def y_of(k):
        return 7.0 - 0.9 * k

    def update(f):
        reset(ax, (0, 24), (0, 9.2))
        ax.text(0.3, 8.5, "Kernel code (one warp)", fontsize=12, weight="bold")
        ax.text(lx, 8.5, "Lanes 0 … 31 of the warp at each line", fontsize=12, weight="bold")
        n = min(f // 3 + 1, len(steps))
        cur_line = steps[n - 1][0]
        for k, text in enumerate(lines):
            if k == cur_line and f < len(steps) * 3:
                box(ax, 0.15, y_of(k) - 0.05, 8.8, 0.7, TEAL_L, "none")
            ax.text(0.3, y_of(k) + 0.3, text, family=MONO, fontsize=12, va="center")
        for line, mask in steps[:n]:
            for lane in range(32):
                box(ax, lx + lane * cw, y_of(line), cw, 0.6, TEAL if mask[lane] else GRAY_L, "white", lw=0.6)
            active = sum(mask)
            ax.text(lx + 32 * cw + 0.2, y_of(line) + 0.3, f"{active}/32", va="center", fontsize=10)
        if f >= len(steps) * 3:
            ax.text(0.3, 0.9, "Both branches run one after the other, each with half the lanes "
                    "masked off,\nso the if/else costs the time of a + b at 50% lane efficiency.",
                    fontsize=11, color=RED)

    frames = list(range(len(steps) * 3)) + [len(steps) * 3] * 6
    save_anim(fig, update, frames, "warp-divergence.gif", fps=2)


# ---------------------------------------------------------------- 02 memory

def memory_hierarchy():
    levels = [
        ("Registers", "~256 KB per SM", "~1 cycle"),
        ("Shared memory / L1", "up to ~256 KB per SM", "~30 cycles"),
        ("L2 cache", "4–128 MB per GPU", "~200 cycles"),
        ("Global memory (HBM / GDDR)", "16–96 GB per GPU", "~500 cycles · 0.3–3.4 TB/s"),
        ("Host RAM (over PCIe)", "100s of GB", "microseconds · 12–50 GB/s"),
    ]
    fills = [TEAL, "#3f8f7a", "#7fb3a4", AMBER_L, RED_L]
    fig, ax = canvas(12, 5.2, (0, 24), (0, 10.4))
    cx, h, top = 7.8, 1.7, 9.6
    for i, ((name, size, lat), fc) in enumerate(zip(levels, fills)):
        w_top, w_bot = 2.0 + i * 2.0, 2.0 + (i + 1) * 2.0
        y1, y0 = top - i * h, top - (i + 1) * h
        poly = patches.Polygon([(cx - w_top / 2, y1), (cx + w_top / 2, y1), (cx + w_bot / 2, y0),
                                (cx - w_bot / 2, y0)], closed=True, facecolor=fc, edgecolor="white", lw=2)
        ax.add_patch(poly)
        ax.text(cx, (y0 + y1) / 2, name, ha="center", va="center", fontsize=10.5,
                color="white" if i < 2 else INK, weight="bold")
        ax.text(14.3, (y0 + y1) / 2 + 0.25, size, fontsize=10.5, va="center")
        ax.text(14.3, (y0 + y1) / 2 - 0.35, lat, fontsize=10, va="center", color=GRAY)
    ax.text(14.3, 10.0, "capacity · latency · bandwidth", fontsize=11, weight="bold")
    arrow(ax, (0.6, 1.4), (0.6, 9.4), TEAL)
    ax.text(1.0, 5.4, "faster,\nsmaller,\ncloser to\nthe cores", fontsize=10, color=TEAL, va="center")
    save(fig, "memory-hierarchy.png")


def coalescing():
    fig, ax = canvas(12, 5.6, (0, 24), (0, 11.2))
    sp = 0.66
    x0 = 2.4

    def tx(k):
        return x0 + k * sp

    panels = [("Coalesced:  a[tid]", 10.4, lambda k: (k // 8, k % 8)),
              ("Strided:  a[tid * 8]", 5.0, lambda k: (k, 0))]

    def update(f):
        reset(ax, (0, 24), (0, 11.2))
        for title, top, mapping in panels:
            ax.text(0.2, top, title, fontsize=12, weight="bold", family=MONO)
            ty, my = top - 1.0, top - 3.4
            touched = {mapping(k)[0] for k in range(f)}
            for s in range(32):
                box(ax, tx(s) - sp / 2, my, sp, 0.7, AMBER if s in touched else GRAY_L, "white", lw=1)
            ax.text(0.2, my + 0.35, "memory", fontsize=9, va="center", color=GRAY)
            ax.text(x0 - sp / 2, my - 0.45, "one box = one 32-byte sector (8 floats)",
                    fontsize=9, color=GRAY)
            for k in range(32):
                ax.add_patch(patches.Circle((tx(k), ty), 0.22, color=TEAL if k < f else GRAY_L))
                if k < f:
                    s, j = mapping(k)
                    mx = tx(s) - sp / 2 + (j + 0.5) * sp / 8
                    ax.plot([tx(k), mx], [ty - 0.25, my + 0.7], color=TEAL, lw=0.7)
            ax.text(0.2, ty, "warp", fontsize=9, va="center", color=GRAY)
            n = len(touched)
            if f:
                used = 4 * f
                ax.text(tx(16), top, f"{n} sectors = {32 * n} B fetched for {used} B used  →  "
                        f"{100 * used / (32 * n):.0f}% efficient", fontsize=10.5, ha="left",
                        color=TEAL if used == 32 * n else RED)

    frames = list(range(33)) + [32] * 12
    save_anim(fig, update, frames, "coalescing.gif", fps=8)


def bank_conflicts():
    cases = [
        ("float s[64];  s[tid]", lambda k: k % 32, "1 address per bank → no conflict"),
        ("float s[64];  s[2 * tid]", lambda k: (2 * k) % 32, "2 addresses per bank → 2-way conflict (2× slower)"),
        ("float tile[32][32];  tile[tid][0]", lambda k: (32 * k) % 32, "all 32 in bank 0 → 32-way conflict"),
        ("float tile[32][33];  tile[tid][0]", lambda k: (33 * k) % 32, "padding by 1 column → no conflict"),
    ]
    fig, ax = canvas(12, 10, (0, 24), (0, 20))
    sp = 0.66
    for p, (title, bank, verdict) in enumerate(cases):
        top = 19.6 - p * 5.0
        counts = np.bincount([bank(k) for k in range(32)], minlength=32)
        worst = counts.max()
        color = TEAL if worst == 1 else (AMBER if worst == 2 else RED)
        ax.text(0.2, top, title, family=MONO, fontsize=11.5, weight="bold")
        ax.text(23.8, top, verdict, fontsize=10.5, ha="right", color=color)
        ty, by = top - 0.9, top - 3.3
        for k in range(32):
            x = 2.4 + k * sp
            ax.add_patch(patches.Circle((x, ty), 0.2, color=TEAL))
            b = bank(k)
            ax.plot([x, 2.4 + b * sp], [ty - 0.22, by + 0.65], color=color, lw=0.7)
        for b in range(32):
            c = counts[b]
            fc = GRAY_L if c == 0 else (TEAL_L if c == 1 else (AMBER_L if c == 2 else RED_L))
            box(ax, 2.4 + b * sp - sp / 2 + 0.03, by, sp - 0.06, 0.65, fc, "white")
        ax.text(0.2, by + 0.3, "banks", fontsize=9, color=GRAY, va="center")
        ax.text(0.2, ty, "threads", fontsize=9, color=GRAY, va="center")
    save(fig, "bank-conflicts.png")


# ---------------------------------------------------------------- 03 performance

GPUS = {  # name: (bandwidth TB/s, FP64 TFLOPS, FP32 TFLOPS)
    "T4": (0.32, 0.25, 8.1),
    "L40S": (0.864, 1.4, 91.6),
    "RTX PRO 6000": (1.597, 1.9, 120),
    "A100 40 GB": (1.555, 9.7, 19.5),
    "H100 SXM": (3.35, 34, 67),
}
GPU_LINE = {"T4": GRAY, "L40S": BLUE, "RTX PRO 6000": PURPLE, "A100 40 GB": AMBER, "H100 SXM": TEAL}


def roofline():
    I = np.logspace(-2, 3, 400)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, (idx, title) in zip(axes, [(1, "FP64 roofline"), (2, "FP32 roofline (non-tensor)")]):
        ax.axvspan(0.05, 1.0, color=TEAL_L, alpha=0.6)
        ax.axvspan(30, 300, color=AMBER_L, alpha=0.6)
        ax.text(0.22, 250, "stencils\nLBM, SpMV\nstatevector", ha="center", fontsize=9, color=TEAL)
        ax.text(95, 250, "dense\nGEMM", ha="center", fontsize=9, color=AMBER)
        for name, spec in GPUS.items():
            bw, peak = spec[0], spec[idx]
            ax.loglog(I, np.minimum(peak, I * bw), color=GPU_LINE[name], lw=2, label=name)
            ax.plot(peak / bw, peak, "o", color=GPU_LINE[name], ms=5)
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_xlabel("arithmetic intensity (FLOP / byte)")
        ax.set_xlim(0.01, 1000)
        ax.set_ylim(0.002, 1000)
        ax.grid(True, which="major", color=GRAY_L)
    axes[0].set_ylabel("attainable TFLOPS")
    axes[1].legend(loc="lower right", fontsize=9, frameon=False)
    fig.text(0.5, -0.02, "Sloped part = memory bound (performance = intensity × bandwidth). "
             "Flat part = compute bound. Dots mark each GPU's ridge point.",
             ha="center", fontsize=10, color=GRAY)
    fig.tight_layout()
    save(fig, "roofline.png")


# ---------------------------------------------------------------- 04 streams

def streams_overlap():
    serial = [(0, 0, 4, 0, "H2D all"), (1, 4, 8, 0, "kernel all"), (2, 8, 12, 0, "D2H all")]
    piped = []
    for c in range(4):
        piped += [(0, c, c + 1, c, f"H2D {c}"), (1, c + 1, c + 2, c, f"K {c}"), (2, c + 2, c + 3, c, f"D2H {c}")]
    lanes = ["copy engine H2D", "compute (SMs)", "copy engine D2H"]
    u, x0 = 1.3, 4.6
    fig, ax = canvas(12, 6.2, (0, 23), (0, 12))

    def panel(ops, title, top, t, total, note):
        ax.text(0.2, top, title, fontsize=12, weight="bold")
        for i, name in enumerate(lanes):
            y = top - 1.3 - i * 0.95
            ax.text(x0 - 0.2, y + 0.35, name, ha="right", va="center", fontsize=9.5)
            box(ax, x0, y, 12 * u, 0.7, GRAY_L, "none")
        for lane, s, e, c, label in ops:
            if t > s:
                y = top - 1.3 - lane * 0.95
                w = (min(e, t) - s) * u
                box(ax, x0 + s * u, y, w, 0.7, GPU_COLORS[c], "white", lw=1)
                if t >= e:
                    ax.text(x0 + (s + e) / 2 * u, y + 0.35, label, ha="center", va="center",
                            fontsize=8.5, color="white")
        if t >= total:
            ax.text(x0 + total * u + 0.2, top - 1.3 - 2 * 0.95 + 0.35, note, va="center",
                    fontsize=10.5, weight="bold", color=TEAL)

    def update(t):
        reset(ax, (0, 23), (0, 12))
        panel(serial, "One stream: copy, then compute, then copy back", 11.4, t, 12, "12 units")
        panel(piped, "4 chunks on several streams: copies overlap with compute", 6.4, t, 6,
              "6 units → 2× faster")
        ax.plot([x0 + t * u] * 2, [0.9, 11.0], color=RED, lw=1.2)
        ax.text(x0 + t * u, 0.5, f"t = {t:.1f}", ha="center", fontsize=9, color=RED)

    frames = list(np.arange(0, 12.01, 0.25)) + [12.0] * 10
    save_anim(fig, update, frames, "streams-overlap.gif", fps=8)


# ---------------------------------------------------------------- 05 multi-GPU

def halo_exchange():
    fig, ax = canvas(12, 5.6, (0, 24), (0, 11.2))
    ax.text(0.5, 10.6, "Global domain split across 4 GPUs", fontsize=12, weight="bold")
    n, cs, ox, top = 12, 0.62, 0.8, 9.8
    for j in range(n):
        for i in range(n):
            g = (j >= 6) * 2 + (i >= 6)
            need = (i == 6 and j <= 6) or (j == 6 and i <= 6)
            fc = GPU_COLORS[g] if need else GPU_COLORS_L[g]
            box(ax, ox + i * cs, top - (j + 1) * cs, cs, cs, fc, "white", lw=0.6)
    for g, (gi, gj) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)]):
        ax.text(ox + (gi * 6 + 3) * cs, top - (gj * 6 + 3) * cs, f"GPU{g}", ha="center", va="center",
                fontsize=13, weight="bold", color=GPU_COLORS[g])
    box(ax, ox, top - 6 * cs, 6 * cs, 6 * cs, "none", INK, lw=2.2)
    ax.text(ox, top - n * cs - 0.6, "Dark cells: neighbor data GPU0 needs every step",
            fontsize=9.5, color=GRAY)

    arrow(ax, (ox + n * cs + 0.4, 6.0), (11.8, 6.0), INK)
    ax.text((ox + n * cs + 11.8) / 2, 6.35, "halo\nexchange", ha="center", fontsize=10)

    lx, lt, lc = 12.6, 9.8, 0.95
    ax.text(lx, 10.6, "GPU0 local array: 6×6 interior + 1-cell halo", fontsize=12, weight="bold")
    for j in range(8):
        for i in range(8):
            if 1 <= i <= 6 and 1 <= j <= 6:
                fc = TEAL_L
            elif i == 7 and 1 <= j <= 6:
                fc = BLUE
            elif j == 7 and 1 <= i <= 6:
                fc = AMBER
            elif i == 7 and j == 7:
                fc = PURPLE
            else:
                fc = GRAY_L
            box(ax, lx + i * lc, lt - (j + 1) * lc, lc, lc, fc, "white", lw=0.8)
    ax.text(lx + 4 * lc, lt - 4 * lc, "interior\n(owned)", ha="center", va="center", fontsize=10, color=TEAL)
    nx = lx + 8 * lc + 0.3
    ax.text(nx, lt - 3.5 * lc, "← from GPU1", fontsize=10, color=BLUE, va="center")
    ax.text(nx, lt - 7.5 * lc, "← corner from GPU3\n   (diagonal stencils)", fontsize=10, color=PURPLE, va="center")
    ax.text(lx + 3.5 * lc, lt - 8 * lc - 0.45, "↑ from GPU2", fontsize=10, color=AMBER, ha="center")
    ax.text(lx, lt + 0.15, "grey = physical boundary (no neighbor)", fontsize=9, color=GRAY)
    save(fig, "halo-exchange.png")


def ring_allreduce():
    N = 4
    state = [[{r} for _ in range(N)] for r in range(N)]
    frames = [("Start: every GPU holds its own full vector, split into 4 chunks", [], [row[:] for row in state])]
    for s in range(N - 1):
        sends = [(r, (r + 1) % N, (r - s) % N) for r in range(N)]
        snap = [[set(c) for c in row] for row in state]
        for src, dst, c in sends:
            state[dst][c] = state[dst][c] | snap[src][c]
        frames.append((f"Reduce-scatter step {s + 1}/{N - 1}: send one chunk to the next GPU, which adds it",
                       sends, [[set(c) for c in row] for row in state]))
    for s in range(N - 1):
        sends = [(r, (r + 1) % N, (r + 1 - s) % N) for r in range(N)]
        snap = [[set(c) for c in row] for row in state]
        for src, dst, c in sends:
            state[dst][c] = set(snap[src][c])
        frames.append((f"All-gather step {s + 1}/{N - 1}: pass the finished chunks around the ring",
                       sends, [[set(c) for c in row] for row in state]))

    centers = [(6, 9.3), (18, 9.3), (18, 3.3), (6, 3.3)]
    fig, ax = canvas(12, 6.6, (0, 24), (0, 13.2))

    def edge(src, dst):
        (x1, y1), (x2, y2) = centers[src], centers[dst]
        if y1 == y2:
            d = 1 if x2 > x1 else -1
            return (x1 + d * 3.0, y1), (x2 - d * 3.0, y2)
        d = 1 if y2 > y1 else -1
        return (x1, y1 + d * 1.9), (x2, y2 - d * 1.9)

    def update(f):
        title, sends, st = frames[f]
        reset(ax, (0, 24), (0, 13.2))
        ax.text(12, 12.7, title, ha="center", fontsize=12, weight="bold")
        sent = {(dst, c) for _, dst, c in sends}
        for r, (cx, cy) in enumerate(centers):
            box(ax, cx - 2.9, cy - 1.8, 5.8, 3.6, "white", GPU_COLORS[r], lw=2.5)
            ax.text(cx - 2.6, cy + 1.25, f"GPU{r}", fontsize=11, weight="bold", color=GPU_COLORS[r])
            for c in range(N):
                x, y, w, h = cx - 2.5 + c * 1.27, cy - 1.3, 1.1, 2.1
                contrib = sorted(st[r][c])
                for k, g in enumerate(contrib):
                    box(ax, x, y + k * h / len(contrib), w, h / len(contrib), GPU_COLORS[g], "none")
                full = len(contrib) == N
                box(ax, x, y, w, h, "none", RED if (r, c) in sent else (INK if full else GRAY),
                    lw=2.5 if (r, c) in sent else (1.8 if full else 0.8))
                ax.text(x + w / 2, y - 0.3, f"c{c}" + (" Σ" if full else ""), ha="center", fontsize=8.5)
        for src, dst, c in sends:
            a, b = edge(src, dst)
            arrow(ax, a, b, RED, lw=1.8)
            ax.text((a[0] + b[0]) / 2 + 0.3, (a[1] + b[1]) / 2 + 0.3, f"c{c}", color=RED, fontsize=10)
        ax.text(12, 0.35, "Stripe colors = which GPUs' data a chunk contains.  Each GPU sends 2(N−1)/N of "
                "the vector in total, whatever N is.", ha="center", fontsize=9.5, color=GRAY)

    seq = [i for i in range(len(frames)) for _ in range(3)] + [len(frames) - 1] * 3
    save_anim(fig, update, seq, "ring-allreduce.gif", fps=1.5)


# ---------------------------------------------------------------- 08 Triton

def triton_program_grid():
    M, N, B, s = 200, 300, 64, 1 / 22
    gm, gn = -(-M // B), -(-N // B)
    ox, top = 0.8, 12.6
    fig, ax = canvas(12, 7, (0, 24), (0, 14))

    def update(f):
        reset(ax, (0, 24), (0, 14))
        ax.text(0.8, 13.4, f"{M}×{N} matrix, BLOCK = {B}×{B}  →  grid = ({gm}, {gn}) programs",
                fontsize=12, weight="bold")
        box(ax, ox, top - M * s, N * s, M * s, GRAY_L, INK, lw=1.5)
        pid = min(f, gm * gn - 1)
        for p in range(pid + 1):
            m, n = divmod(p, gn)
            x, y = ox + n * B * s, top - (m + 1) * B * s
            vw, vh = min(B, N - n * B) * s, min(B, M - m * B) * s
            if p == pid:
                box(ax, x, y, B * s, B * s, "none", RED, lw=0, hatch="///")
                box(ax, x, top - m * B * s - vh, vw, vh, TEAL, TEAL, lw=2.5)
                ax.text(x + vw / 2, top - m * B * s - vh / 2, f"({m},{n})", ha="center", va="center",
                        fontsize=10, weight="bold", color="white")
            else:
                box(ax, x, top - m * B * s - vh, vw, vh, "#b9d6cd", "white", lw=1)
        for m in range(gm + 1):
            ax.plot([ox, ox + gn * B * s], [top - m * B * s] * 2, ":", color=GRAY, lw=0.8)
        for n in range(gn + 1):
            ax.plot([ox + n * B * s] * 2, [top, top - gm * B * s], ":", color=GRAY, lw=0.8)
        m, n = divmod(pid, gn)
        rows, cols = min(B, M - m * B), min(B, N - n * B)
        tx = 18.0
        info = [f"pid_m = {m}, pid_n = {n}",
                f"rows = {m * B} + arange({B})",
                f"cols = {n * B} + arange({B})",
                "mask = (rows < M) &",
                "       (cols < N)",
                f"valid: {rows} × {cols}"]
        for k, line in enumerate(info):
            ax.text(tx, 11.8 - k * 0.8, line, family=MONO, fontsize=10.5)
        box(ax, tx, 5.9, 0.6, 0.5, "none", RED, lw=0, hatch="///")
        ax.text(tx + 0.8, 6.15, "masked off (out of bounds)", fontsize=9.5, va="center")
        box(ax, tx, 5.1, 0.6, 0.5, TEAL, TEAL, lw=1.5)
        ax.text(tx + 0.8, 5.35, "this program's tile", fontsize=9.5, va="center")
        box(ax, tx, 4.3, 0.6, 0.5, "#b9d6cd", "white")
        ax.text(tx + 0.8, 4.55, "already launched programs", fontsize=9.5, va="center")

    frames = list(range(gm * gn)) + [gm * gn - 1] * 3
    save_anim(fig, update, frames, "triton-program-grid.gif", fps=2)


def triton_matmul():
    ts, K, pm, pn = 1.35, 4, 1, 2
    A = (4.6, 5.6)          # x, top
    Bm = (12.0, 12.4)
    C = (12.0, 5.6)
    fig, ax = canvas(12, 6.6, (0, 24), (0, 13.2))

    def grid(x0, top, rows, cols, hl=None, done=(), fill=TEAL_L, alpha=1.0):
        for r in range(rows):
            for c in range(cols):
                fc = GRAY_L
                if (r, c) in done:
                    fc = "#c9d3d0"
                if hl == (r, c):
                    fc = fill
                box(ax, x0 + c * ts, top - (r + 1) * ts, ts, ts, fc, "white", lw=1.5, alpha=alpha if hl == (r, c) else 1)

    def update(f):
        k = min(f // 2, K)
        reset(ax, (0, 24), (0, 13.2))
        ax.text(0.4, 12.6, "One program computes one C tile, looping over K", fontsize=12, weight="bold")
        done_a = {(pm, j) for j in range(k)}
        done_b = {(j, pn) for j in range(k)}
        grid(*A, 3, K, hl=(pm, k) if k < K else None, done=done_a, fill=BLUE)
        grid(*Bm, K, 3, hl=(k, pn) if k < K else None, done=done_b, fill=AMBER)
        grid(*C, 3, 3)
        acc = k / K
        box(ax, C[0] + pn * ts, C[1] - (pm + 1) * ts, ts, ts, TEAL, TEAL, lw=2, alpha=0.25 + 0.75 * acc)
        ax.text(A[0] + K * ts / 2, A[1] + 0.3, "A  (M × K)", ha="center", fontsize=11)
        ax.text(Bm[0] - 0.3, Bm[1] - K * ts / 2, "B\n(K × N)", ha="right", va="center", fontsize=11)
        ax.text(C[0] + 1.5 * ts, C[1] - 3 * ts - 0.5, "C  (M × N)", ha="center", fontsize=11)
        tx = 17.2
        lines = [f"program (pid_m={pm}, pid_n={pn})", "acc = zeros((BM, BN))"]
        if k < K:
            lines += [f"k = {k}:", f"  a = load A[{pm}, {k}]", f"  b = load B[{k}, {pn}]", "  acc += tl.dot(a, b)"]
        else:
            lines += ["loop done:", "  store(C[1, 2], acc)"]
        for i, line in enumerate(lines):
            ax.text(tx, 11.0 - i * 0.75, line, family=MONO, fontsize=10)

    frames = list(range(2 * K + 1)) + [2 * K] * 3
    save_anim(fig, update, frames, "triton-matmul.gif", fps=1.5)


# ---------------------------------------------------------------- spec sheets

def spec_bars():
    names = list(GPUS)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    for ax, (idx, title, unit) in zip(axes, [(0, "Memory bandwidth", "TB/s"), (1, "FP64 throughput", "TFLOPS")]):
        vals = [GPUS[n][idx] for n in names]
        bars = ax.barh(names, vals, color=[GPU_LINE[n] for n in names])
        ax.invert_yaxis()
        for b, v in zip(bars, vals):
            ax.text(b.get_width() + max(vals) * 0.01, b.get_y() + b.get_height() / 2,
                    f"{v:g}", va="center", fontsize=10)
        ax.set_title(title, fontsize=12, weight="bold", loc="left")
        ax.set_xlabel(unit)
        ax.set_xlim(0, max(vals) * 1.15)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    save(fig, "spec-bars.png")


if __name__ == "__main__":
    thread_hierarchy()
    latency_hiding()
    warp_divergence()
    memory_hierarchy()
    coalescing()
    bank_conflicts()
    roofline()
    streams_overlap()
    halo_exchange()
    ring_allreduce()
    triton_program_grid()
    triton_matmul()
    spec_bars()
