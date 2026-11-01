"""Generate the figures for the multi-GPU communication page.

Run from the repo root:

    .venv/bin/python scripts/make_comm_figures.py

Writes PNG files into docs/assets/img/. Reuses the palette and helpers from
make_figures.py so all figures look the same.
"""
import numpy as np

from make_figures import (AMBER, AMBER_L, BLUE, BLUE_L, GRAY, GRAY_L, INK, PURPLE, PURPLE_L,
                          RED, TEAL, TEAL_L, box, canvas, patches, save)


def label_box(ax, x, y, w, h, fc, ec, title, sub=None, fs=10):
    box(ax, x, y, w, h, fc, ec, lw=1.6)
    if sub:
        ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=fs, weight="bold")
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=fs - 2, color=INK)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fs, weight="bold")


def link(ax, a, b, color, lw=2.0, ls="-"):
    ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, ls=ls, solid_capstyle="round", zorder=0)


# ---------------------------------------------------------------- software/hardware stack

def comm_stack():
    fig, ax = canvas(15, 9.2, (0, 30), (0, 18.4))
    ax.text(0.3, 17.7, "The multi-GPU communication stack", fontsize=14, weight="bold")
    ax.text(10.6, 16.6, "NVIDIA", fontsize=12, weight="bold", color=TEAL, ha="center")
    ax.text(22.9, 16.6, "AMD", fontsize=12, weight="bold", color=RED, ha="center")
    rows = [
        ("Framework", TEAL_L, TEAL,
         "PyTorch DDP / FSDP, DeepSpeed,\nMegatron-LM, JAX",
         "the same frameworks,\nROCm builds"),
        ("Communication API", TEAL_L, TEAL,
         "torch.distributed (backend \"nccl\"),\nmpi4py, NVSHMEM",
         "torch.distributed (\"nccl\" = RCCL),\nmpi4py, rocSHMEM"),
        ("Collective library", BLUE_L, BLUE,
         "NCCL  |  CUDA-aware MPI\n(Open MPI + UCX, MVAPICH2-GDR)",
         "RCCL  |  GPU-aware MPI\n(Open MPI + UCX, Cray MPICH)"),
        ("Transport", AMBER_L, AMBER,
         "P2P (NVLink / PCIe), shared memory,\nIB verbs, sockets, libfabric (EFA)",
         "P2P (xGMI / PCIe), shared memory,\nIB verbs / RoCE, libfabric"),
        ("Runtime and driver", GRAY_L, GRAY,
         "CUDA: cudaMemcpyPeer, CUDA IPC,\nGPUDirect P2P / RDMA",
         "HIP: hipMemcpyPeer, HIP IPC,\nROCm RDMA (PeerDirect)"),
        ("Links inside a node", PURPLE_L, PURPLE,
         "NVLink 3 / 4 / 5 + NVSwitch,\nPCIe Gen3-5, NVLink-C2C",
         "Infinity Fabric (xGMI) mesh,\nPCIe Gen4-5"),
        ("Network between nodes", PURPLE_L, PURPLE,
         "InfiniBand (NDR / XDR), RoCE,\nAWS EFA; ConnectX NICs",
         "InfiniBand or RoCE Ethernet,\nHPE Slingshot (Frontier)"),
    ]
    y, h = 14.3, 1.85
    for name, fc, ec, nv, amd in rows:
        label_box(ax, 0.3, y, 5.0, h, fc, ec, name, fs=10)
        box(ax, 5.8, y, 9.6, h, "white", ec, lw=1.4)
        ax.text(10.6, y + h / 2, nv, ha="center", va="center", fontsize=9.5)
        box(ax, 18.1, y, 9.6, h, "white", ec, lw=1.4)
        ax.text(22.9, y + h / 2, amd, ha="center", va="center", fontsize=9.5)
        y -= h + 0.3
    ax.text(15.95, 10.4, "software", rotation=90, ha="center", va="center", fontsize=10, color=GRAY)
    ax.text(15.95, 3.4, "hardware", rotation=90, ha="center", va="center", fontsize=10, color=GRAY)
    ax.plot([0.3, 27.7], [5.55, 5.55], color=GRAY, lw=1.2, ls="--")
    save(fig, "comm-stack.png")


# ---------------------------------------------------------------- node topologies

def node_topologies():
    fig, ax = canvas(16, 6.6, (0, 32), (0, 13.2))

    # A: PCIe-only node with 4 T4s
    ax.text(0.3, 12.6, "PCIe only: 4x T4", fontsize=12, weight="bold")
    label_box(ax, 3.0, 9.6, 4.0, 1.4, GRAY_L, GRAY, "CPU + host memory", fs=9)
    for k, x in enumerate((0.6, 5.6)):
        label_box(ax, x + 0.5, 6.6, 3.2, 1.1, AMBER_L, AMBER, "PCIe switch", fs=9)
        link(ax, (x + 2.1, 7.7), (5.0, 9.6), AMBER, 2.5)
        for j, gx in enumerate((x, x + 2.3)):
            label_box(ax, gx, 3.6, 1.9, 1.5, TEAL_L, TEAL, f"GPU{2 * k + j}", fs=9)
            link(ax, (gx + 0.95, 5.1), (x + 2.1, 6.6), AMBER, 2.0)
    ax.text(0.3, 2.6, "PCIe Gen3 x16: ~16 GB/s each way,\nshared by all traffic. GPU0-GPU2 crosses\n"
            "the CPU. P2P is often off in VMs:\ncopies then bounce through host memory.",
            fontsize=8.5, va="top")

    # B: HGX H100 with NVSwitch
    ox = 11.0
    ax.text(ox, 12.6, "NVSwitch: 8x H100 SXM", fontsize=12, weight="bold")
    for s in range(4):
        label_box(ax, ox + 0.4 + s * 2.35, 7.4, 2.0, 1.0, PURPLE_L, PURPLE, "NVSwitch", fs=8)
    for g in range(8):
        gx = ox + 0.1 + (g % 4) * 2.35
        gy = 9.9 if g < 4 else 4.6
        label_box(ax, gx, gy, 1.9, 1.4, TEAL_L, TEAL, f"GPU{g}", fs=9)
        for s in range(4):
            sx = ox + 1.4 + s * 2.35
            link(ax, (gx + 0.95, gy if g < 4 else gy + 1.4), (sx, 8.4 if g < 4 else 7.4), PURPLE, 0.8)
    ax.text(ox, 3.6, "NVLink 4: 900 GB/s per GPU (450 each way)\nto any other GPU, one hop. "
            "Plus one\n400 Gb/s NIC per GPU for other nodes.", fontsize=8.5, va="top")

    # C: MI300X Infinity Fabric mesh
    cx, cy, r = 26.5, 7.3, 3.4
    ax.text(21.6, 12.6, "Infinity Fabric mesh: 8x MI300X", fontsize=12, weight="bold")
    pts = [(cx + r * np.cos(np.pi / 2 - 2 * np.pi * g / 8), cy + r * np.sin(np.pi / 2 - 2 * np.pi * g / 8))
           for g in range(8)]
    for a in range(8):
        for b in range(a + 1, 8):
            link(ax, pts[a], pts[b], RED, 0.9)
    for g, (px, py) in enumerate(pts):
        label_box(ax, px - 0.85, py - 0.6, 1.7, 1.2, "#f7dcd8", RED, f"GPU{g}", fs=8.5)
    ax.text(21.6, 3.0, "7 xGMI links per GPU, one to each peer:\n896 GB/s in total, but only "
            "~1/7 of it\nbetween any one pair. No switch.", fontsize=8.5, va="top")
    save(fig, "node-topologies.png")


# ---------------------------------------------------------------- data paths

def data_paths():
    fig, ax = canvas(15, 8.4, (0, 30), (0, 16.8))
    ax.text(0.3, 16.1, "Where a GPU-to-GPU transfer actually goes", fontsize=14, weight="bold")
    for x0, name in ((0.3, "Node A"), (19.6, "Node B")):
        box(ax, x0, 3.7, 10.0 if x0 < 1 else 10.1, 11.4, "none", GRAY, lw=1.2, ls="--")
        ax.text(x0 + 0.3, 14.6, name, fontsize=10, color=GRAY)
    # node A
    label_box(ax, 2.3, 12.0, 6.0, 1.5, GRAY_L, GRAY, "CPU + host memory", fs=10)
    label_box(ax, 2.3, 8.6, 6.0, 1.2, AMBER_L, AMBER, "PCIe switch", fs=10)
    label_box(ax, 0.8, 4.6, 2.6, 1.8, TEAL_L, TEAL, "GPU0", fs=10)
    label_box(ax, 4.0, 4.6, 2.6, 1.8, TEAL_L, TEAL, "GPU1", fs=10)
    label_box(ax, 7.2, 4.6, 2.6, 1.8, PURPLE_L, PURPLE, "NIC", fs=10)
    for gx in (2.1, 5.3, 8.5):
        link(ax, (gx, 6.4), (gx if gx != 8.5 else 7.6, 8.6), AMBER, 2.5)
    link(ax, (5.3, 9.8), (5.3, 12.0), AMBER, 2.5)
    # node B
    label_box(ax, 21.8, 12.0, 6.0, 1.5, GRAY_L, GRAY, "CPU + host memory", fs=10)
    label_box(ax, 21.8, 8.6, 6.0, 1.2, AMBER_L, AMBER, "PCIe switch", fs=10)
    label_box(ax, 20.3, 4.6, 2.6, 1.8, PURPLE_L, PURPLE, "NIC", fs=10)
    label_box(ax, 26.4, 4.6, 2.6, 1.8, TEAL_L, TEAL, "GPU2", fs=10)
    for gx in (21.6, 27.7):
        link(ax, (gx, 6.4), (gx if gx != 21.6 else 22.4, 8.6), AMBER, 2.5)
    link(ax, (24.8, 9.8), (24.8, 12.0), AMBER, 2.5)

    def tag(x, y, n, c):
        ax.add_patch(patches.Circle((x, y), 0.42, color=c, zorder=5))
        ax.text(x, y, str(n), ha="center", va="center", color="white", fontsize=10, weight="bold", zorder=6)

    # 1 NVLink: direct bridge under the GPUs
    ax.plot([2.1, 2.1, 5.3, 5.3], [4.6, 4.0, 4.0, 4.6], color=TEAL, lw=4, zorder=1)
    tag(3.7, 4.0, 1, TEAL)
    # 2 PCIe P2P via the switch
    ax.plot([1.3, 1.3, 4.5, 4.5], [6.4, 8.25, 8.25, 6.4], color=BLUE, lw=2.2, ls="-", zorder=1)
    tag(1.3, 7.5, 2, BLUE)
    # 3 bounce through host memory
    ax.plot([0.9, 0.9, 3.0], [6.4, 12.75, 12.75], color=RED, lw=2.2, ls="--", zorder=1)
    ax.plot([6.3, 6.3], [12.0, 6.4], color=RED, lw=2.2, ls="--", zorder=0)
    tag(0.9, 10.6, 3, RED)
    # 4 GPUDirect RDMA: GPU0 -> switch -> NIC -> network -> NIC -> GPU2
    ax.plot([10.0, 20.1], [5.5, 5.5], color=PURPLE, lw=4, zorder=1)
    ax.text(15.05, 6.0, "InfiniBand / RoCE / EFA\n400 Gb/s = 50 GB/s per NIC", ha="center", fontsize=9, color=PURPLE)
    tag(15.05, 5.0, 4, PURPLE)
    legend = [
        (1, TEAL, "NVLink, direct: 450 GB/s each way on H100. Use it for anything chatty (tensor parallel)."),
        (2, BLUE, "PCIe P2P through a switch: <= 16 / 32 / 64 GB/s (Gen3 / 4 / 5). Fine for halos and DDP."),
        (3, RED, "No P2P (VM, crosses CPU sockets): GPU -> host memory -> GPU. Two PCIe trips. Avoid."),
        (4, PURPLE, "Other node, GPUDirect RDMA: the NIC reads GPU memory directly. Without it, also staged via host."),
    ]
    for k, (n, c, t) in enumerate(legend):
        ax.text(0.5, 3.0 - k * 0.8, f"({n})", color=c, fontsize=10, weight="bold", va="center")
        ax.text(1.6, 3.0 - k * 0.8, t, fontsize=9.5, va="center")
    save(fig, "data-paths.png")


if __name__ == "__main__":
    comm_stack()
    node_topologies()
    data_paths()
