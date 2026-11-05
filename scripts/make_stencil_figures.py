"""Generate the figures for the stencils and discretization page.

Run from the repo root:

    .venv/bin/python scripts/make_stencil_figures.py

Writes PNG files into docs/assets/img/. Reuses the palette and helpers from
make_figures.py so all figures look the same.
"""
from make_figures import (AMBER, AMBER_L, BLUE, BLUE_L, GRAY, GRAY_L, INK, PURPLE, PURPLE_L,
                          RED, TEAL, TEAL_L, box, canvas, patches, save)
from make_comm_figures import label_box, link


def node(ax, x, y, text, fc, ec, r=0.42, fs=10):
    ax.add_patch(patches.Circle((x, y), r, fc=fc, ec=ec, lw=1.6, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, weight="bold", zorder=4)


# ---------------------------------------------------------------- stencil shapes

def stencil_shapes():
    fig, ax = canvas(16, 5.6, (0, 32), (0, 11.2))
    ax.text(0.3, 10.6, "Stencils: which neighbours each output point reads, with their weights",
            fontsize=14, weight="bold")

    def panel(x0, title, formula):
        ax.text(x0 + 3.6, 9.4, title, ha="center", fontsize=11.5, weight="bold")
        ax.text(x0 + 3.6, 0.7, formula, ha="center", fontsize=9.5, color=INK)

    # 1: d/dx, central
    panel(0.3, "d/dx (central)", "(u[i+1] - u[i-1]) / 2h")
    link(ax, (1.7, 5.0), (6.1, 5.0), GRAY, 1.2)
    for k, w in zip((-1, 0, 1), ("-1", "0", "+1")):
        node(ax, 3.9 + 2.2 * k, 5.0, w, BLUE_L if k else GRAY_L, BLUE if k else GRAY)
    ax.text(3.9, 3.6, "x 1/2h", ha="center", fontsize=10, color=BLUE)

    # 2: 2-D Laplacian, 5-point
    panel(8.3, "2-D Laplacian, 5-point", "(E + W + N + S - 4C) / h^2")
    cx, cy, s = 11.9, 5.2, 2.0
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        link(ax, (cx, cy), (cx + dx * s, cy + dy * s), GRAY, 1.2)
        node(ax, cx + dx * s, cy + dy * s, "1", TEAL_L, TEAL)
    node(ax, cx, cy, "-4", AMBER_L, AMBER)

    # 3: 2-D gradient as two stencils sharing reads
    panel(16.3, "2-D gradient", "(E - W) / 2dx ,  (N - S) / 2dy")
    cx = 19.9
    for (dx, dy), w, c in (((1, 0), "+1", BLUE), ((-1, 0), "-1", BLUE), ((0, 1), "+1", PURPLE),
                           ((0, -1), "-1", PURPLE)):
        link(ax, (cx, cy), (cx + dx * s, cy + dy * s), c, 1.6)
        node(ax, cx + dx * s, cy + dy * s, w, BLUE_L if c == BLUE else PURPLE_L, c)
    node(ax, cx, cy, "0", GRAY_L, GRAY)
    ax.text(cx + 2.6, cy - 2.6, "x: blue\ny: purple", fontsize=9, color=INK)

    # 4: 3-D Laplacian, 7-point (oblique z axis)
    panel(24.3, "3-D Laplacian, 7-point", "sum of 6 neighbours - 6C, / h^2")
    cx = 27.9
    zx, zy = 1.25, 1.0
    for (dx, dy), c in (((s, 0), TEAL), ((-s, 0), TEAL), ((0, s), TEAL), ((0, -s), TEAL),
                        ((zx, zy), PURPLE), ((-zx, -zy), PURPLE)):
        link(ax, (cx, cy), (cx + dx, cy + dy), c, 1.2, ls="-" if c == TEAL else "--")
        node(ax, cx + dx, cy + dy, "1", TEAL_L if c == TEAL else PURPLE_L, c)
    node(ax, cx, cy, "-6", AMBER_L, AMBER)
    ax.text(cx + zx + 0.6, cy + zy + 0.5, "z ±1", fontsize=9, color=PURPLE)
    save(fig, "stencil-shapes.png")


# ---------------------------------------------------------------- grids

def grids():
    fig, ax = canvas(16, 7.4, (0, 32), (0, 14.8))
    ax.text(0.3, 14.2, "Where the unknowns live: collocated grid with ghost cells vs staggered (MAC) grid",
            fontsize=13, weight="bold")
    n, h = 5, 2.0

    # collocated, with ghost ring
    x0, y0 = 1.0, 1.6
    ax.text(x0, 12.6, "Collocated: u, v, p at cell centres", fontsize=11.5, weight="bold")
    for j in range(n + 2):
        for i in range(n + 2):
            ghost = i in (0, n + 1) or j in (0, n + 1)
            box(ax, x0 + i * h * 0.95, y0 + j * h * 0.75, h * 0.95, h * 0.75,
                GRAY_L if ghost else "white", GRAY, lw=0.8)
            node(ax, x0 + (i + 0.5) * h * 0.95, y0 + (j + 0.5) * h * 0.75, "", RED if ghost else TEAL_L,
                 RED if ghost else TEAL, r=0.18)
    ax.text(x0 + 0.1, 0.8, "grey ring = ghost cells, filled by the boundary condition before each step",
            fontsize=9.5, color=INK)

    # staggered MAC
    x0 = 17.4
    ax.text(x0, 12.6, "Staggered (MAC): p centres, u on x-faces, v on y-faces", fontsize=11.5, weight="bold")
    W, H = 2.6, 2.0
    for j in range(n):
        for i in range(n):
            bx, by = x0 + i * W, y0 + j * H
            box(ax, bx, by, W, H, "white", GRAY, lw=0.8)
            node(ax, bx + W / 2, by + H / 2, "", AMBER_L, AMBER, r=0.2)
    for j in range(n):
        for i in range(n + 1):
            ax.plot([x0 + i * W], [y0 + (j + 0.5) * H], ">", color=BLUE, ms=8, zorder=4)
    for j in range(n + 1):
        for i in range(n):
            ax.plot([x0 + (i + 0.5) * W], [y0 + j * H], "^", color=PURPLE, ms=8, zorder=4)
    ax.text(x0 + 0.1, 0.8, "orange = p,  blue > = u (x-velocity),  purple ^ = v (y-velocity)",
            fontsize=9.5, color=INK)
    save(fig, "stencil-grids.png")


# ---------------------------------------------------------------- 2.5-D blocking

def blocking():
    fig, ax = canvas(16, 6.4, (0, 32), (0, 12.8))
    ax.text(0.3, 12.2, "3-D stencil, 2.5-D blocking: march along z, keep three planes in registers",
            fontsize=13, weight="bold")
    # stack of planes
    for k in range(7):
        y = 1.2 + k * 1.45
        state = {2: ("k-1 (below)", BLUE_L, BLUE), 3: ("k (centre)", AMBER_L, AMBER),
                 4: ("k+1 (above): new load", TEAL_L, TEAL)}.get(k)
        fc, ec = (state[1], state[2]) if state else ("white", GRAY)
        pts = [(2.0, y), (11.0, y), (13.0, y + 1.0), (4.0, y + 1.0)]
        ax.add_patch(patches.Polygon(pts, closed=True, fc=fc, ec=ec, lw=1.4))
        ax.text(13.6, y + 0.5, state[0] if state else f"plane {k}", fontsize=9.5,
                color=ec if state else GRAY, va="center", weight="bold" if state else "normal")
    ax.annotate("", xy=(1.2, 10.8), xytext=(1.2, 1.4),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=2))
    ax.text(0.5, 6.0, "z", fontsize=12, color=RED, weight="bold")

    steps = [
        (TEAL, "1. load plane k+1 for this program's x-y tile (the only new DRAM read)"),
        (AMBER, "2. out[k] = in-plane neighbours of k (L1/L2 hits) + above + below - 6 x centre"),
        (BLUE, "3. shift the queue: below <- centre, centre <- above; move to k+1"),
        (GRAY, "Each plane is read from DRAM once per program instead of 3 times."),
        (GRAY, "Tiles stay 2-D, so a program needs registers for 3 tiles, not a 3-D block."),
        (PURPLE, "Same idea on CUDA: keep the centre plane in shared memory for the x-y neighbours."),
    ]
    for i, (c, t) in enumerate(steps):
        ax.plot([18.4], [10.0 - i * 1.5], "s", color=c, ms=8)
        ax.text(18.9, 10.0 - i * 1.5, t, fontsize=9.8, va="center", wrap=True)
    save(fig, "stencil-blocking.png")


if __name__ == "__main__":
    stencil_shapes()
    grids()
    blocking()
