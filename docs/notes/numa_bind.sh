#!/usr/bin/env bash
# Run one rank on the CPU cores and memory of the NUMA node its GPU is attached to.
#
#   torchrun --nproc_per_node=8 --no-python docs/notes/numa_bind.sh python train.py
#   NUMA_NODE=1 docs/notes/numa_bind.sh python train.py      # force a node
#
# Finds the GPU's node from its PCI address (NVIDIA; on AMD set NUMA_NODE from
# `rocm-smi --showtoponuma`). Without a known node (-1: single socket, many VMs) or
# without a working numactl, it runs the command unbound.
set -euo pipefail

gpu=${LOCAL_RANK:-0}
node=${NUMA_NODE:-}
if [[ -z "$node" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    bus=$(nvidia-smi -i "$gpu" --query-gpu=pci.bus_id --format=csv,noheader)   # 00000000:18:00.0
    bdf=$(echo "${bus#0000}" | tr '[:upper:]' '[:lower:]')                     # 0000:18:00.0
    node=$(cat "/sys/bus/pci/devices/$bdf/numa_node" 2>/dev/null || echo -1)
fi

if [[ -n "$node" && "$node" -ge 0 ]] && numactl --hardware >/dev/null 2>&1; then
    # --preferred falls back to other nodes when this one is full; --membind would fail instead.
    exec numactl --cpunodebind="$node" --preferred="$node" "$@"
fi
echo "numa_bind: GPU $gpu: no usable NUMA node (node=${node:-unset}); running unbound" >&2
exec "$@"
