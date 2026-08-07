#!/usr/bin/env bash
# ncu_probe.sh — profile the two kernels that are the whole remaining gap to the roofline.
# ncu needs root on this box. Note (HARDWARE.md): ncu's "Memory Throughput %" is L2 on Thor, so
# read dram__bytes.sum.per_second, not the SpeedOfLight memory row.
set -e
cd "$(dirname "$0")/.."
nvcc -O3 -std=c++17 -lineinfo -gencode arch=compute_110a,code=sm_110a -I include \
  tools/ncu_target.cu kernels/mla_attn.cu kernels/tc_moe_gemm.cu kernels/fp8_block_gemm.cu \
  kernels/tc_fp8_gemm.cu kernels/dscratch.cu -o build/ncu_target
echo "built build/ncu_target"
