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
# Gate OGGEMV — the M=1 fp8 ogroup GEMV (Finding 35). Not covered by gate_ogroup, which tests the
# f32 bs=8 tensor-core path; this is a different kernel carrying ~22% of the decode step.
nvcc -O2 -std=c++17 -gencode arch=compute_110a,code=sm_110a -I include \
  tests/gate_ogroup_gemv.cu kernels/mla_attn.cu kernels/dscratch.cu -o build/gate_ogroup_gemv
echo "built build/gate_ogroup_gemv"
# Gate TC_FP8_SMEM — the smem-staged FP8 tensor-core GEMM (Finding 41). gate_units checks
# tc_fp8_gemm at ONE (M,N,K); this sweeps every shape and every M the verify path issues, plus the
# N%64 and M%16 tails that the tile mapping gets wrong independently of each other.
nvcc -O2 -std=c++17 -gencode arch=compute_110a,code=sm_110a -I include \
  tests/gate_tc_fp8_smem.cu kernels/fp8_block_gemm.cu kernels/tc_fp8_gemm.cu -o build/gate_tc_fp8_smem
echo "built build/gate_tc_fp8_smem"
