"""Generate the figures for the warp primitives and cooperative groups page.

Run from the repo root:

    .venv/bin/python scripts/make_warp_figures.py

Writes PNG files into docs/assets/img/. Reuses the palette and helpers from
make_figures.py so all figures look the same. Eight lanes stand in for a warp's 32.
"""
from make_figures import (AMBER, AMBER_L, BLUE, BLUE_L, GRAY, GRAY_L, INK, MONO, PURPLE,
                          PURPLE_L, RED, TEAL, TEAL_L, box, canvas, save)

L = 8                       # lanes drawn
W, H = 1.0, 0.75            # lane box size


def lanes(ax, x0, y, values, fc, ec, active=None, fs=10):
    for k, v in enumerate(values):
        on = active is None or active[k]
        box(ax, x0 + k * W, y, W * 0.9, H, fc if on else GRAY_L, ec if on else GRAY, lw=1.2)
        ax.text(x0 + k * W + W * 0.45, y + H / 2, str(v), ha="center", va="center",
                fontsize=fs, family=MONO, color=INK if on else GRAY)


def lane_numbers(ax, x0, y):
    for k in range(L):
        ax.text(x0 + k * W + W * 0.45, y, str(k), ha="center", va="center", fontsize=8, color=GRAY)


def move(ax, x0, src, dst, y_top, y_bot, color):
    ax.annotate("", xy=(x0 + dst * W + W * 0.45, y_bot + H), xytext=(x0 + src * W + W * 0.45, y_top),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4, mutation_scale=12))


# ---------------------------------------------------------------- the four shuffles

def shuffles():
    fig, ax = canvas(16, 7.6, (0, 40), (0, 19))
    ax.text(0.3, 18.3, "Warp shuffles: every lane reads one register from another lane (8 of 32 lanes shown)",
            fontsize=14, weight="bold")
    vals = list("abcdefgh")
    panels = [
        (0.5, 9.6, "__shfl_down_sync(mask, v, 2)", "lane i reads lane i + 2", BLUE, BLUE_L,
         lambda i: i + 2 if i + 2 < L else i, "lanes 6, 7: no source, keep their own value"),
        (20.5, 9.6, "__shfl_up_sync(mask, v, 2)", "lane i reads lane i - 2", TEAL, TEAL_L,
         lambda i: i - 2 if i >= 2 else i, "lanes 0, 1: no source, keep their own value"),
        (0.5, 0.6, "__shfl_xor_sync(mask, v, 1)", "lane i reads lane i ^ 1 (swap pairs)", PURPLE, PURPLE_L,
         lambda i: i ^ 1, "every lane gets a value: the butterfly behind all-reduce"),
        (20.5, 0.6, "__shfl_sync(mask, v, 3)", "every lane reads lane 3 (broadcast)", AMBER, AMBER_L,
         lambda i: 3, "srcLane can differ per lane: any permutation or gather"),
    ]
    for x0, y0, title, sub, c, cl, src, note in panels:
        ax.text(x0, y0 + 7.6, title, fontsize=12, weight="bold", family=MONO, color=c)
        ax.text(x0, y0 + 6.9, sub, fontsize=10.5)
        top, bot = y0 + 5.0, y0 + 1.6
        lane_numbers(ax, x0, top + H + 0.35)
        lanes(ax, x0, top, vals, GRAY_L, INK)
        out = [vals[src(i)] for i in range(L)]
        kept = [src(i) == i and title.startswith(("__shfl_down", "__shfl_up")) for i in range(L)]
        lanes(ax, x0, bot, out, cl, c)
        for i in range(L):
            if not kept[i]:
                move(ax, x0, src(i), i, top, bot, c)
            else:
                box(ax, x0 + i * W, bot, W * 0.9, H, "none", RED, lw=2.0, ls="--")
        ax.text(x0 + L * W + 0.4, top + H / 2, "before", va="center", fontsize=9.5, color=GRAY)
        ax.text(x0 + L * W + 0.4, bot + H / 2, "after", va="center", fontsize=9.5, color=GRAY)
        ax.text(x0, y0 + 0.8, note, fontsize=9.5, color=RED if "keep" in note else INK)
    save(fig, "warp-shuffles.png")


# ---------------------------------------------------------------- reduce: tree vs butterfly

def reductions():
    fig, ax = canvas(16, 7.0, (0, 40), (0, 17.5))
    ax.text(0.3, 16.8, "Warp reduction in log2(32) = 5 steps (3 shown for 8 lanes)", fontsize=14, weight="bold")
    x = [3, 1, 4, 1, 5, 9, 2, 6]

    def run(x0, title, sub, c, cl, step):
        ax.text(x0, 15.4, title, fontsize=12, weight="bold", family=MONO, color=c)
        ax.text(x0, 14.7, sub, fontsize=10.5)
        v = list(x)
        y = 12.4
        lane_numbers(ax, x0, y + H + 0.35)
        lanes(ax, x0, y, v, GRAY_L, INK)
        for o in (4, 2, 1):
            nv, srcs = step(v, o)
            yb = y - 3.6
            lanes(ax, x0, yb, nv, cl, c, fs=10)
            for i, s in srcs:
                move(ax, x0, s, i, y, yb, c)
            ax.text(x0 + L * W + 0.4, yb + H / 2, f"offset {o}", va="center", fontsize=9.5, color=GRAY)
            v, y = nv, yb
        return v

    def tree(v, o):                                     # += __shfl_down_sync(v, o)
        nv = [v[i] + (v[i + o] if i + o < L else v[i]) for i in range(L)]
        return nv, [(i, i + o) for i in range(L) if i + o < L]

    def fly(v, o):                                      # += __shfl_xor_sync(v, o)
        return [v[i] + v[i ^ o] for i in range(L)], [(i, i ^ o) for i in range(L)]

    run(0.5, "v += __shfl_down_sync(FULL, v, o)", "tree: the total ends in lane 0", BLUE, BLUE_L, tree)
    box(ax, 0.5, 1.6, W * 0.9, H, "none", RED, lw=2.4)
    ax.text(0.5, 0.6, "lane 0 = 31 (the sum); other lanes hold partial sums, ignore them",
            fontsize=10, color=RED)
    run(20.5, "v += __shfl_xor_sync(FULL, v, o)", "butterfly: every lane ends with the total",
        PURPLE, PURPLE_L, fly)
    ax.text(20.5, 0.6, "all lanes = 31: no broadcast needed before using the result", fontsize=10, color=PURPLE)
    save(fig, "warp-reduce.png")


# ---------------------------------------------------------------- scan with shfl_up

def scan():
    fig, ax = canvas(9.6, 7.4, (0, 24), (-0.6, 17.9))
    ax.text(0.3, 16.8, "Inclusive scan with __shfl_up_sync (Hillis-Steele)", fontsize=14, weight="bold")
    ax.text(0.3, 16.0, "other = __shfl_up_sync(FULL, v, offset);  if (lane >= offset) v += other;",
            fontsize=10.5, family=MONO)
    x0 = 0.5
    v = [1, 2, 3, 4, 5, 6, 7, 8]
    y = 12.6
    lane_numbers(ax, x0, y + H + 0.35)
    lanes(ax, x0, y, v, GRAY_L, INK)
    for o in (1, 2, 4):
        nv = [v[i] + (v[i - o] if i >= o else 0) for i in range(L)]
        yb = y - 3.7
        lanes(ax, x0, yb, nv, TEAL_L, TEAL)
        for i in range(o, L):
            move(ax, x0, i - o, i, y, yb, TEAL)
        for i in range(o):
            box(ax, x0 + i * W, yb, W * 0.9, H, "none", RED, lw=2.0, ls="--")
        ax.text(x0 + L * W + 0.4, yb + H / 2, f"offset {o}: lanes < {o} keep their value",
                va="center", fontsize=9.5, color=GRAY)
        v, y = nv, yb
    ax.text(0.3, -0.2, "Result: running totals 1, 3, 6, 10, ... in log2(n) steps, all in registers.",
            fontsize=10.5)
    save(fig, "warp-scan.png")


# ---------------------------------------------------------------- ballot + popc compaction

def compaction():
    fig, ax = canvas(16, 6.4, (0, 40), (0, 16))
    ax.text(0.3, 15.3, "Stream compaction with __ballot_sync + __popc: one atomic per warp",
            fontsize=14, weight="bold")
    x = [-2, 5, 7, -1, 3, -4, 8, 6]
    keep = [v > 0 for v in x]
    x0 = 9.5
    rows = [
        (12.6, "x (one per lane)", [str(v) for v in x], GRAY_L, INK, None),
        (10.2, "keep = x > 0", ["1" if k else "0" for k in keep], BLUE_L, BLUE, None),
    ]
    lane_numbers(ax, x0, 12.6 + H + 0.35)
    for y, label, vals, fc, ec, act in rows:
        ax.text(0.3, y + H / 2, label, va="center", fontsize=10.5, family=MONO)
        lanes(ax, x0, y, vals, fc, ec, act)
    bits = "".join("1" if k else "0" for k in reversed(keep))
    ax.text(0.3, 8.4, "m = __ballot_sync(FULL, keep)", fontsize=10.5, family=MONO)
    ax.text(x0, 8.4, f"0b{bits}  (bit k = lane k; same value in every lane)", fontsize=10.5,
            family=MONO, color=BLUE)
    ranks = [sum(keep[:i]) for i in range(L)]
    ax.text(0.3, 6.6 + H / 2, "slot = __popc(m & lanes below)", va="center", fontsize=10.5, family=MONO)
    lanes(ax, x0, 6.6, [str(r) if k else "·" for r, k in zip(ranks, keep)], AMBER_L, AMBER, keep)
    ax.text(19.5, 5.2, "lane 0:  base = atomicAdd(count, __popc(m))", fontsize=10.5, family=MONO)
    ax.text(19.5, 4.5, "         one atomic for 5 values", fontsize=10.5, family=MONO, color=AMBER)
    ax.text(19.5, 3.6, "all:     base = __shfl_sync(FULL, base, 0)", fontsize=10.5, family=MONO)
    ax.text(0.3, 2.0 + H / 2, "out[base + slot] = x", va="center", fontsize=10.5, family=MONO)
    kept = [v for v in x if v > 0]
    for k, v in enumerate(kept):
        box(ax, x0 + k * W, 2.0, W * 0.9, H, TEAL_L, TEAL, lw=1.2)
        ax.text(x0 + k * W + W * 0.45, 2.0 + H / 2, str(v), ha="center", va="center", family=MONO)
    for i, k in enumerate(keep):
        if k:
            ax.annotate("", xy=(x0 + ranks[i] * W + W * 0.45, 2.0 + H), xytext=(x0 + i * W + W * 0.45, 6.6),
                        arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.2, mutation_scale=11))
    ax.text(x0 + 5 * W + 0.4, 2.0 + H / 2, "packed, in lane order", va="center", fontsize=10, color=TEAL)
    save(fig, "warp-compaction.png")


# ---------------------------------------------------------------- scopes of sync

def scopes():
    fig, ax = canvas(16, 6.0, (0, 40), (0, 15))
    ax.text(0.3, 14.3, "What each primitive coordinates: lanes, warp, block, grid", fontsize=14, weight="bold")
    # grid
    box(ax, 0.4, 0.6, 39.2, 12.6, "#f7f8f7", GRAY, lw=1.4)
    ax.text(0.8, 12.5, "grid  —  grid.sync()  (cooperative launch only)  or  end the kernel", fontsize=11,
            weight="bold", color=RED)
    for b in range(2):
        bx = 1.0 + b * 19.4
        box(ax, bx, 1.1, 18.6, 10.7, BLUE_L, BLUE, lw=1.4)
        ax.text(bx + 0.3, 11.1, f"block {b}  —  __syncthreads() / block.sync()", fontsize=10.5,
                weight="bold", color=BLUE)
        for w in range(2):
            wx = bx + 0.4 + w * 9.1
            box(ax, wx, 1.6, 8.7, 8.7, TEAL_L, TEAL, lw=1.2)
            ax.text(wx + 0.25, 9.6, f"warp {w}: __syncwarp(), shuffles, votes", fontsize=9, color=TEAL,
                    weight="bold")
            ax.text(wx + 0.25, 9.0, "= thread_block_tile<32>", fontsize=9, color=TEAL, family=MONO)
            for t in range(2):
                tx = wx + 0.3 + t * 4.2
                box(ax, tx, 4.4, 3.9, 3.9, PURPLE_L, PURPLE, lw=1.0)
                ax.text(tx + 1.95, 7.7, "tile<16>", ha="center", fontsize=9, color=PURPLE, family=MONO)
                for k in range(16):
                    box(ax, tx + 0.25 + (k % 4) * 0.85, 4.65 + (k // 4) * 0.72, 0.7, 0.55, "white", PURPLE, lw=0.6)
            ax.text(wx + 0.25, 3.3, "lanes inside: shfl, ballot,", fontsize=8.5)
            ax.text(wx + 0.25, 2.6, "match, reduce, scan", fontsize=8.5)
            ax.text(wx + 0.25, 1.9, "(registers, no shared memory)", fontsize=8.5, color=GRAY)
    save(fig, "sync-scopes.png")


if __name__ == "__main__":
    shuffles()
    reductions()
    scan()
    compaction()
    scopes()
