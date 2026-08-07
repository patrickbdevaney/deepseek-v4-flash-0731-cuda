#!/usr/bin/env bash
# cap_probe.sh — exhaustive capability sweep for a CUDA target. See tools/cap_probe.cu.
# usage: scripts/cap_probe.sh [target]   (default sm_110a)
cd "$(dirname "$0")/.."
T=${1:-sm_110a}
FEATS="C_MMA_F16_M16N8K16 C_MMA_BF16_M16N8K16 C_MMA_TF32_M16N8K8 C_MMA_E4M3_M16N8K32 C_MMA_E5M2_M16N8K32 C_MMA_S8_M16N8K32 C_MMA_S4_M16N8K64 C_MMA_E2M1_M16N8K32 C_MMA_MXF4_BLOCKSCALE \
C_WGMMA_F16 C_WGMMA_FENCE \
C_T5_ALLOC C_T5_MMA_F16 C_T5_MMA_F8F6F4 C_T5_MMA_MXF8F6F4 C_T5_MMA_MXF4NVF4 C_T5_MMA_CTA2 C_T5_LD C_T5_ST C_T5_CP C_T5_SHIFT \
C_CPASYNC_CG C_CPASYNC_BULK C_TMA_TILE_2D C_TMA_TILE_3D C_TMA_IM2COL C_TMA_MULTICAST C_TMA_STORE C_TMA_REDUCE C_TMA_PREFETCH \
C_CLUSTER_BAR C_DSMEM_MAPA C_DSMEM_ST C_CLUSTERID \
C_MBARRIER C_MBARRIER_TRYWAIT C_SETMAXNREG C_ELECT C_GRIDDEPCTRL \
C_CVT_E2M1_TO_F16X2 C_CVT_F32_TO_E2M1X2 C_CVT_E8M0 C_CVT_E3M2 C_CVT_E2M3 C_CVT_FP8X4 \
C_LDMATRIX_X4 C_LDMATRIX_M16N16_B8 C_STMATRIX \
C_L2_EVICT_POLICY C_LD_L1_NOALLOC C_PREFETCH_L2 C_DISCARD C_CPASYNC_L2HINT \
C_DP4A C_REDUX C_LOP3 C_PRMT"
echo "target: $T"
printf "%-26s %-10s %s\n" FEATURE STATUS "SASS / reason"
for F in $FEATS; do
  err=$(nvcc -arch=$T -D$F -cubin -o /tmp/cap.cubin tools/cap_probe.cu 2>&1)
  if [ $? -eq 0 ]; then
    sass=$(cuobjdump -sass /tmp/cap.cubin 2>/dev/null | grep -oE '\b(UTC[A-Z]+[A-Z0-9.]*|[A-Z]*MMA[A-Z0-9.]*|UBLKCP[A-Z0-9.]*|LDSM[A-Z0-9.]*|STSM[A-Z0-9.]*|CCTL[A-Z0-9.]*|LDGSTS[A-Z0-9.]*|PRMT|LOP3|IDP|REDUX|ELECT)\b' | grep -v '^$' | head -1)
    printf "%-26s %-10s %s\n" "$F" "OK" "${sass:-—}"
  else
    reason=$(echo "$err" | grep -oE "Instruction '[^']*' not supported|Feature '[^']*' not supported|Modifier '[^']*' not supported|Illegal modifier[^;]*" | head -1)
    if [ -z "$reason" ]; then printf "%-26s %-10s %s\n" "$F" "probe-err" "$(echo "$err"|grep -oE 'error *: .*'|head -1|cut -c1-60)"
    else printf "%-26s %-10s %s\n" "$F" "BLOCKED" "$reason"; fi
  fi
done
