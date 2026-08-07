#!/usr/bin/env bash
# arch_probe.sh — establish what sm_110a ACTUALLY exposes, at compile time and at runtime.
#
# CRITICAL FLAG NOTE (LOOP_LOG Finding 29): you MUST target the ARCH-SPECIFIC virtual arch.
#   -arch=sm_110   -> compute_110  -> tcgen05 BLOCKED   <-- the prior project's conclusion came from here
#   -gencode arch=compute_110a,code=sm_110a  -> compute_110a -> tcgen05 OK        (but building an EXECUTABLE this way also
#                                                        emits compute_110 PTX, which fails)
#   -gencode arch=compute_110a,code=sm_110a             <-- correct for executables: SASS only
set -e
cd "$(dirname "$0")/.."
echo "=== compile-time capability (compute_110a) ==="
for P in P_TCGEN05 P_E4M3_M16N8K32 P_E2M1_M16N8K32 P_E2M1_M16N8K64 P_MXF4_M16N8K64 P_CPASYNC_BULK; do
  printf "  %-22s " "$P"
  if nvcc -gencode arch=compute_110a,code=sm_110a -D$P -cubin -o /dev/null tools/arch_probe.cu 2>/tmp/ap.txt
  then echo "COMPILES"
  else echo "BLOCKED: $(grep -oE "Instruction '[^']*' not supported|Feature '[^']*' not supported" /tmp/ap.txt | head -1)"; fi
done
echo "=== runtime (does the silicon execute it?) ==="
nvcc -gencode arch=compute_110a,code=sm_110a -o build/arch_probe_runtime tools/arch_probe_runtime.cu
./build/arch_probe_runtime

echo "=== tcgen05 MMA kinds (the family Thor actually uses for FP4) ==="
for P in K_F16 K_F8F6F4 K_MXF8F6F4 K_MXF4NVF4; do
  printf "  %-14s " "$P"
  nvcc -gencode arch=compute_110a,code=sm_110a -D$P -cubin -o /tmp/t5.cubin tools/tcgen05_probe.cu 2>/dev/null \
    && echo "COMPILES  SASS: $(cuobjdump -sass /tmp/t5.cubin 2>/dev/null | grep -oE '[A-Z]*MMA[A-Z0-9.]*' | head -1)" \
    || echo "BLOCKED"
done
