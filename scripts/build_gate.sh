#!/usr/bin/env bash
# Build the Gate-K unit test (host CUDA). Run: ./build/gate_units ref/goldens
set -e
cd "$(dirname "$0")/.."
nvcc -O2 -std=c++17 -gencode arch=compute_110a,code=sm_110a -I include \
  tests/gate_units.cu kernels/fp8_block_gemm.cu kernels/hc_sinkhorn.cu kernels/mla_attn.cu kernels/moe.cu kernels/hc.cu kernels/compressor.cu kernels/indexer.cu kernels/tc_moe_gemm.cu kernels/tc_fp8_gemm.cu kernels/dscratch.cu kernels/dprof.cu \
  -o build/gate_units
echo "built build/gate_units"
# CPU-only gates (no GPU): chat encoder byte-exactness + OpenAI API shaping.
g++ -O1 -std=c++17 -I include tests/gate_encoding.cpp -o build/gate_encoding && echo "built build/gate_encoding"
nvcc -O3 -std=c++17 -gencode arch=compute_110a,code=sm_110a -I include tests/gate_bf16w.cu kernels/compressor.cu kernels/dscratch.cu kernels/mla_attn.cu kernels/indexer.cu kernels/fp8_block_gemm.cu kernels/tc_fp8_gemm.cu -o build/gate_bf16w && echo "built build/gate_bf16w"
g++ -O1 -std=c++17 -I include tests/gate_api.cpp      -o build/gate_api      && echo "built build/gate_api"
# Gate FORKJOIN_GRAPH — the side-stream fork/join (Finding 55) must survive CUDA-graph capture, 43
# times on one event pair, or the base-AR graph that is worth 1.26x silently stops capturing. Two
# seconds here instead of a 15-minute checkpoint load to find out.
nvcc -O2 -std=c++17 -gencode arch=compute_110a,code=sm_110a -I include \
  tests/gate_forkjoin_graph.cu kernels/dscratch.cu -o build/gate_forkjoin_graph
echo "built build/gate_forkjoin_graph"
# Gate OGGEMV — the M=1 fp8 ogroup GEMV (Finding 35). Not covered by gate_ogroup, which tests the
# f32 bs=8 tensor-core path; this is a different kernel carrying ~22% of the decode step.
nvcc -O2 -std=c++17 -gencode arch=compute_110a,code=sm_110a -I include \
  tests/gate_ogroup_gemv.cu kernels/mla_attn.cu kernels/dscratch.cu -o build/gate_ogroup_gemv
echo "built build/gate_ogroup_gemv"
# Gate TC_FP8_SMEM — the smem-staged FP8 tensor-core GEMM (Finding 41). gate_units checks
# tc_fp8_gemm at ONE (M,N,K); this sweeps every shape and every M the verify path issues, plus the
# N%64 and M%16 tails that the tile mapping gets wrong independently of each other.
nvcc -O2 -std=c++17 -gencode arch=compute_110a,code=sm_110a -I include \
  tests/gate_tc_fp8_smem.cu kernels/fp8_block_gemm.cu kernels/tc_fp8_gemm.cu kernels/dscratch.cu -o build/gate_tc_fp8_smem
echo "built build/gate_tc_fp8_smem"
# Gate PREFILL_LEN — prefix-invariance of the prefill attention chain across prompt LENGTHS, plus a
# drain of the CUDA last-error slot after every stage (Finding 53). Nothing else in the suite varies
# s, which is how an undersized dynamic-shared-memory request survived at T<3 and how a ratio-128
# layer issued five gridDim-0 launches on every prompt this project has ever run. Worth running under
# `compute-sanitizer --tool memcheck` too: that is what found the shared over-read.
nvcc -O2 -std=c++17 -gencode arch=compute_110a,code=sm_110a -I include -lineinfo \
  tests/gate_prefill_len.cu kernels/compressed_decode.cu kernels/compressed_attn.cu kernels/compressor.cu \
  kernels/indexer.cu kernels/mla_attn.cu kernels/mla_forward.cu kernels/mla_decode.cu kernels/hc.cu \
  kernels/hc_sinkhorn.cu kernels/fp8_block_gemm.cu kernels/tc_fp8_gemm.cu kernels/dscratch.cu kernels/dprof.cu \
  -o build/gate_prefill_len
echo "built build/gate_prefill_len"
# Gate MOE_SCAN — the parallel MoE grouping scans (k_moe_prefix_par / k_build_tiles_par) must be
# BIT-IDENTICAL to the <<<1,1>>> kernels they replaced. Equality, not cosine: these are integer
# scans. Sweeps nr below/at/above the 256-thread block width plus empty, exactly-16 and multi-tile
# experts, because per LEVERS.md trap 9 a harness that cannot express the regime confirms itself.
nvcc -O2 -std=c++17 -gencode arch=compute_110a,code=sm_110a -I include \
  tests/gate_moe_scan.cu kernels/moe.cu kernels/tc_moe_gemm.cu kernels/dscratch.cu kernels/dprof.cu \
  kernels/fp8_block_gemm.cu kernels/tc_fp8_gemm.cu kernels/mla_attn.cu -o build/gate_moe_scan
echo "built build/gate_moe_scan"
# Gate TC_FP8_KC — the K-chunked staging (Finding 74) must be BIT-IDENTICAL to KC=1. gate_tc_fp8_smem
# next door is a cosine gate and would pass a reduction-order change; Finding 68 is the reason that
# distinction gets its own binary.
nvcc -O2 -std=c++17 -gencode arch=compute_110a,code=sm_110a -I include \
  tests/gate_tc_fp8_kc.cu kernels/fp8_block_gemm.cu kernels/tc_fp8_gemm.cu kernels/dscratch.cu \
  -o build/gate_tc_fp8_kc
echo "built build/gate_tc_fp8_kc"
# Gate OG_WS1 — the ogroup GEMV scale-hoist (Finding 76) must be BIT-IDENTICAL to the shipped default. Same
# reasoning as gate_tc_fp8_kc: the claim is value equality, so the instrument is memcmp, and
# gate_ogroup_gemv next door is a cosine gate on a DIFFERENT (M=1) kernel.
nvcc -O2 -std=c++17 -gencode arch=compute_110a,code=sm_110a -I include \
  tests/gate_og_ws1.cu kernels/mla_attn.cu kernels/dscratch.cu \
  -o build/gate_og_ws1
echo "built build/gate_og_ws1"

# ---------------------------------------------------------------------------------------------
# THE OTHER EIGHT (Finding 76). Before this block, build_gate.sh built 10 of the 18 gate binaries
# and the rest went stale SILENTLY: F74 found gate_compressed_decode and gate_indexer_decode a day
# out of date, and gate_fp4_gemv turned out to be two engine parameters behind — it had not linked,
# and therefore had not tested anything, since F65/F72 added `rows_hint`/`align8`, while a binary
# from 2026-08-06 sat in build/ looking like a passing gate. A gate that is not in the build script
# is a gate that measures the past. Sources are listed here rather than in each file's header
# comment so there is exactly one place that can drift.
#
# `nvcc ... && echo built` DOES NOT FAIL THE SCRIPT even under `set -e`: the failure is a non-final
# command of an && list, so bash exempts it, the "built" line simply never prints, and the run ends
# 0 with one binary silently missing. That is how gate_grouped_moe went missing on the first pass of
# this very block. `bg` below builds, reports, and records — and the script exits non-zero at the end
# if anything failed, because "a MISSING binary is a FAILURE, not a pass".
GA="-O2 -std=c++17 -gencode arch=compute_110a,code=sm_110a -I include"
GATE_FAILED=""
bg(){ local t="$1"; shift
      if nvcc $GA "tests/$t.cu" "$@" -o "build/$t"; then echo "built build/$t"
      else echo "FAILED TO BUILD build/$t" >&2; GATE_FAILED="$GATE_FAILED $t"; fi; }
trap '[ -n "$GATE_FAILED" ] && { echo "GATE BUILD FAILURES:$GATE_FAILED" >&2; exit 1; }; exit 0' EXIT
CDEC="kernels/compressed_decode.cu kernels/compressed_attn.cu kernels/compressor.cu kernels/indexer.cu \
  kernels/mla_attn.cu kernels/fp8_block_gemm.cu kernels/tc_fp8_gemm.cu kernels/dscratch.cu kernels/dprof.cu"
FULL="$CDEC kernels/mla_forward.cu kernels/mla_decode.cu kernels/hc.cu kernels/hc_sinkhorn.cu \
  kernels/moe.cu kernels/tc_moe_gemm.cu"
for t in gate_compressed_decode gate_indexer_decode; do bg $t $CDEC; done
for t in gate_scratch_init gate_compressed_graph gate_indexer_graph; do bg $t $FULL; done
bg gate_compressor_emit kernels/compressor.cu kernels/mla_attn.cu kernels/indexer.cu \
  kernels/dscratch.cu kernels/fp8_block_gemm.cu kernels/tc_fp8_gemm.cu
# moe.cu, for `fp4_gemm` — the oracle this gate diffs the grouped GEMV against. Its own header build
# line lists only tc_moe_gemm.cu and has not linked since the oracle moved.
bg gate_grouped_moe kernels/tc_moe_gemm.cu kernels/moe.cu kernels/mla_attn.cu kernels/fp8_block_gemm.cu \
  kernels/tc_fp8_gemm.cu kernels/dscratch.cu kernels/dprof.cu
# -lcublasLt: fp4_gemm.cu carries a cublasLt reference path next to the hand-written kernel.
bg gate_fp4_gemv kernels/fp4_gemm.cu kernels/moe.cu kernels/tc_moe_gemm.cu \
  kernels/dscratch.cu kernels/dprof.cu kernels/mla_attn.cu kernels/fp8_block_gemm.cu kernels/tc_fp8_gemm.cu \
  -lcublasLt -lcublas
