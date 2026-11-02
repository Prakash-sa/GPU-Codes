"""Generate the inside-the-instance diagrams for the AWS GPU instances page.

Run from the repo root:

    .venv/bin/python scripts/make_instance_figures.py

Writes PNG files into docs/assets/img/instances/. Counts and speeds come from the AWS
instance-type specifications and NVIDIA's HGX board specs. AWS does not publish the exact
PCIe switch wiring, so the 8-GPU diagrams use the HGX reference shape: one PCIe switch per
GPU pair, with that pair's NICs and NVMe drives on the same switch.
"""
from make_figures import (AMBER, AMBER_L, BLUE, BLUE_L, GRAY, GRAY_L, INK, OUT, PURPLE, PURPLE_L,
                          RED, TEAL, TEAL_L, box, canvas, save)
from make_comm_figures import label_box, link

(OUT / "instances").mkdir(exist_ok=True)


def notes(ax, lines, y0, dy=0.8):
    for i, (color, text) in enumerate(lines):
        ax.plot([0.5], [y0 - i * dy], "s", color=color, ms=7)
        ax.text(0.9, y0 - i * dy, text, fontsize=10, va="center")


# ---------------------------------------------------------------- 8-GPU HGX instances

def hgx(name, title, cpu, sockets_link, dram, pcie_up, pcie_gpu, gpu, mem, hbm,
        nvswitch, n_switch, nvlink, nics, nvme, extra):
    fig, ax = canvas(16, 11.4, (0, 32), (-1.6, 21.2))
    ax.text(0.3, 20.6, title, fontsize=14, weight="bold")

    cpus = []
    for k, x0 in enumerate((0.3, 16.1)):
        box(ax, x0, 6.1, 15.6, 13.2, "none", GRAY, lw=1.2, ls="--")
        ax.text(x0 + 0.3, 18.8, f"socket {k}  ·  NUMA node {k}", fontsize=10, color=GRAY, weight="bold")
        ddr_x, cpu_x = (x0 + 0.4, x0 + 5.3) if k == 0 else (x0 + 10.6, x0 + 3.1)
        label_box(ax, ddr_x, 15.4, 4.6, 2.2, BLUE_L, BLUE, "DDR memory", dram, fs=10)
        label_box(ax, cpu_x, 15.4, 7.0, 2.2, GRAY_L, GRAY, f"CPU {k}", cpu, fs=10)
        a, b = sorted((ddr_x, cpu_x))
        link(ax, (a + (4.6 if k == 0 else 7.0), 16.5), (b, 16.5), BLUE, 4)
        cpus.append(cpu_x)
    link(ax, (cpus[0] + 7.0, 16.5), (cpus[1], 16.5), RED, 4)
    ax.text(16.0, 17.1, sockets_link, ha="center", fontsize=9.5, color=RED, weight="bold")

    gx = [0.6 + i * 3.9 for i in range(8)]
    for j in range(4):
        sx = gx[2 * j]
        label_box(ax, sx, 10.6, 7.3, 1.3, AMBER_L, AMBER, f"PCIe switch {j}", fs=9.5)
        cpu_x = cpus[j // 2]
        link(ax, (sx + 3.65, 11.9), (sx + 3.65, 14.6), AMBER, 3)        # up between NIC and NVMe
        link(ax, (sx + 3.65, 14.6), (cpu_x + (2.2 if j % 2 == 0 else 4.8), 15.4), AMBER, 3)
        label_box(ax, sx + 0.05, 12.7, 3.25, 1.6, PURPLE_L, PURPLE, "EFA NICs", nics, fs=8.5)
        label_box(ax, sx + 4.0, 12.7, 3.25, 1.6, GRAY_L, GRAY, "NVMe", nvme, fs=8.5)
        link(ax, (sx + 1.7, 12.7), (sx + 1.7, 11.9), AMBER, 1.6)
        link(ax, (sx + 5.6, 12.7), (sx + 5.6, 11.9), AMBER, 1.6)

    for i, x in enumerate(gx):
        label_box(ax, x, 6.6, 3.4, 2.6, TEAL_L, TEAL, f"GPU{i}", f"{gpu}\n{mem}\n{hbm}", fs=9.5)
        link(ax, (x + 1.7, 9.2), (gx[2 * (i // 2)] + 3.65, 10.6), AMBER, 1.8)

    w = (31.3 - 0.6 - (n_switch - 1) * 0.5) / n_switch
    sw = [0.6 + s * (w + 0.5) for s in range(n_switch)]
    for s, x in enumerate(sw):
        label_box(ax, x, 3.3, w, 1.4, PURPLE_L, PURPLE, f"NVSwitch {s}", fs=9.5)
    for x in gx:
        for s in sw:
            link(ax, (x + 1.7, 6.6), (s + w / 2, 4.7), PURPLE, 0.9)

    notes(ax, [(PURPLE, f"{nvlink}; {nvswitch}."),
               (AMBER, f"PCIe: GPU -> switch {pcie_gpu}; switch -> CPU {pcie_up}."),
               *extra], 2.5, dy=0.75)
    save(fig, f"instances/{name}.png")


# ---------------------------------------------------------------- 1-GPU instances

def single(name, title, cpu, dram, pcie, gpu, gpu_sub, nic, nvme, nvlink_note, extra):
    fig, ax = canvas(16, 7.8, (0, 32), (0, 15.6))
    ax.text(0.3, 15.0, title, fontsize=14, weight="bold")
    box(ax, 0.3, 3.3, 31.4, 11.0, "none", GRAY, lw=1.2, ls="--")
    ax.text(0.6, 13.8, "your slice of a larger physical host (other instances share the rest)",
            fontsize=10, color=GRAY, weight="bold")

    label_box(ax, 1.0, 10.0, 6.0, 2.4, BLUE_L, BLUE, "DDR memory", dram, fs=10)
    label_box(ax, 9.0, 10.0, 8.0, 2.4, GRAY_L, GRAY, "vCPUs", cpu, fs=10)
    link(ax, (7.0, 11.2), (9.0, 11.2), BLUE, 4)
    label_box(ax, 9.0, 4.2, 8.0, 3.4, TEAL_L, TEAL, gpu, gpu_sub, fs=11)
    link(ax, (13.0, 7.6), (13.0, 10.0), AMBER, 4)
    ax.text(13.4, 8.8, pcie, fontsize=9.5, color=AMBER, weight="bold", va="center")
    label_box(ax, 21.0, 10.0, 7.0, 2.4, PURPLE_L, PURPLE, "network", nic, fs=10)
    label_box(ax, 21.0, 6.6, 7.0, 2.0, GRAY_L, GRAY, "local NVMe", nvme, fs=10)
    link(ax, (17.0, 11.2), (21.0, 11.2), AMBER, 2.4)
    link(ax, (17.0, 10.4), (21.0, 7.6), AMBER, 2.0)
    if nvlink_note:
        for dy in (-0.9, -0.3, 0.3, 0.9):
            link(ax, (17.0, 5.9 + dy), (19.6, 5.9 + dy * 1.6), PURPLE, 1.4, ls=":")
        ax.text(20.0, 5.0, nvlink_note, fontsize=9.5, color=PURPLE, va="center")
    notes(ax, extra, 2.4)
    save(fig, f"instances/{name}.png")


def main():
    single("g6e-2xlarge", "g6e.2xlarge: 1x L40S over PCIe, no NVLink",
           "8 (4 cores)\nAMD EPYC 7R13 (Milan)", "64 GiB",
           "PCIe Gen4 x16\n~32 GB/s each way",
           "L40S", "48 GB GDDR6 (44 GiB usable)\n864 GB/s", "ENA, up to 20 Gb/s\n(no EFA)", "1 x 450 GB",
           None,
           [(AMBER, "Every byte in or out of the GPU crosses PCIe Gen4: keep data resident and copy rarely."),
            (TEAL, "L40S has no NVLink at all; even the 4- and 8-GPU g6e sizes talk over PCIe only.")])
    single("p5-4xlarge", "p5.4xlarge: 1x H100 SXM, NVLink present but no peer",
           "16 (8 cores)\nAMD EPYC 7R13 (Milan)", "256 GiB",
           "PCIe, Gen4 at the CPU\n~32 GB/s each way",
           "H100 SXM", "80 GB HBM3\n3.35 TB/s", "1x EFA, 100 Gb/s\nno GPUDirect RDMA", "1 x 3.8 TB",
           "18 NVLink ports, unused:\nAWS lists GPU peer-to-peer as N/A",
           [(PURPLE, "Network traffic is staged through host memory: GPU -> PCIe -> DDR -> EFA (no GPUDirect RDMA)."),
            (TEAL, "Same GPU as p5.48xlarge, so single-GPU kernels behave the same; multi-GPU needs another instance.")])

    hgx("p4d-24xlarge", "p4d.24xlarge: 8x A100 40 GB on NVSwitch (HGX A100)",
        "Xeon Platinum 8275CL\n24 cores (Cascade Lake)", "UPI between sockets", "576 GiB", "Gen3 x16 (~16 GB/s each way)",
        "Gen4 x16 (~32 GB/s each way)", "A100", "40 GB HBM2", "1.6 TB/s",
        "6 NVSwitch chips (2nd gen)", 6,
        "NVLink 3: 12 links per GPU, 600 GB/s per GPU (300 each way), any pair at full speed",
        "1x 100 Gb", "2 x 1 TB",
        [(PURPLE, "GPU <-> GPU: NVLink through the NVSwitches, 300 GB/s each way vs ~32 GB/s for PCIe Gen4."),
         (AMBER, "CPU <-> GPU: PCIe Gen3 at the Cascade Lake root, the slowest host link of these instances."),
         (RED, "Network: 4x 100 Gb EFA = 50 Gb/s per GPU, one NIC per GPU pair; GPUDirect RDMA supported.")])
    hgx("p5-48xlarge", "p5.48xlarge: 8x H100 80 GB on NVSwitch (HGX H100)",
        "AMD EPYC 7R13\n48 cores (Milan)", "Infinity Fabric (xGMI) between sockets", "1 TiB", "Gen4 x16 (~32 GB/s each way)",
        "Gen5 x16 (~64 GB/s each way)", "H100 SXM", "80 GB HBM3", "3.35 TB/s",
        "4 NVSwitch chips (3rd gen)", 4,
        "NVLink 4: 18 links per GPU, 900 GB/s per GPU (450 each way), any pair at full speed",
        "8x 100 Gb", "2 x 3.8 TB",
        [(PURPLE, "GPU <-> GPU: each H100 spreads its 18 NVLinks over the 4 NVSwitches; all-reduce can use NVLS."),
         (AMBER, "CPU <-> GPU: the GPUs are Gen5 but Milan's root is Gen4, so host copies top out near 32 GB/s each way."),
         (RED, "Network: 32x 100 Gb EFA = 400 Gb/s per GPU (4 NICs each); GPUDirect RDMA supported.")])
    hgx("p5en-48xlarge", "p5en.48xlarge: 8x H200 141 GB on NVSwitch (HGX H200)",
        "Xeon (Sapphire Rapids)\n48 cores", "UPI between sockets", "1 TiB", "Gen5 x16 (~64 GB/s each way)",
        "Gen5 x16 (~64 GB/s each way)", "H200", "141 GB HBM3e", "4.8 TB/s",
        "4 NVSwitch chips (3rd gen)", 4,
        "NVLink 4: 18 links per GPU, 900 GB/s per GPU (450 each way), same fabric as H100",
        "4x 200 Gb", "2 x 3.8 TB",
        [(PURPLE, "GPU <-> GPU: identical NVLink/NVSwitch fabric to p5; the H200 differs in memory (141 GB, 4.8 TB/s)."),
         (AMBER, "CPU <-> GPU: Gen5 end to end, about twice p5's host-copy bandwidth (AWS: 'Gen5 PCIe between CPU and GPU')."),
         (RED, "Network: 16x 200 Gb EFA (3rd gen) = 400 Gb/s per GPU (2 NICs each); GPUDirect RDMA supported.")])
    hgx("p6-b200-48xlarge", "p6-b200.48xlarge: 8x B200 on NVSwitch (HGX B200)",
        "Xeon (Emerald Rapids)\n48 cores", "UPI between sockets", "1 TiB", "Gen5 x16 (~64 GB/s each way)",
        "Gen5 x16 (~64 GB/s each way)", "B200", "179 GiB HBM3e", "~8 TB/s",
        "2 NVSwitch chips (NVLink 5)", 2,
        "NVLink 5: 18 links per GPU, 1.8 TB/s per GPU (900 each way), any pair at full speed",
        "2x 400 Gb", "2 x 3.8 TB",
        [(PURPLE, "GPU <-> GPU: twice H100/H200's NVLink bandwidth through 2 larger NVSwitch chips."),
         (RED, "Network: 8x 400 Gb EFA (4th gen) = 400 Gb/s per GPU (1 NIC each); GPUDirect RDMA supported.")])


if __name__ == "__main__":
    main()
