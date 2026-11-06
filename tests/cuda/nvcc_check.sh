#!/usr/bin/env bash
# Compile every CUDA solution with the real nvcc for T4 (sm_75), A100 (sm_80) and H100
# (sm_90). Needs no GPU, so it runs in NVIDIA's devel image on any machine:
#
#   docker run --rm -v "$PWD":/w -w /w nvidia/cuda:12.6.3-devel-ubuntu24.04 bash tests/cuda/nvcc_check.sh
#
# Prints ptxas register/spill counts for sm_90 and fails on any error or warning.
set -uo pipefail

arch=(-gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80
      -gencode arch=compute_90,code=sm_90)
status=0
for f in docs/leetgpu/[0-9][0-9]_*.cu docs/notes/stencil_ops.cu docs/notes/warp_primitives.cu; do
    log=$(nvcc -std=c++17 -O3 "${arch[@]}" -Werror all-warnings -Xcompiler -Wall,-Werror \
               -Xptxas -v,-warn-spills -c "$f" -o /tmp/check.o 2>&1)
    rc=$?
    regs=$(echo "$log" | grep -A3 "for 'sm_90'" | grep -o 'Used [0-9]* registers' | grep -o '[0-9]*' | sort -n | uniq | tr '\n' ',' | sed 's/,$//')
    spills=$(echo "$log" | grep -o '[0-9]* bytes spill stores' | sort -u | tr '\n' ' ')
    if [[ $rc -eq 0 ]]; then
        printf 'ok    %-44s sm_90 registers: %-10s spills: %s\n' "$f" "${regs:-?}" "${spills:-none}"
    else
        printf 'FAIL  %s\n%s\n' "$f" "$log"
        status=1
    fi
done
exit $status
