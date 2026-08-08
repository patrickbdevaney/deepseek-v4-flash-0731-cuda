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
