#!/usr/bin/env bash
# Build the Gate-K unit test (host CUDA). Run: ./build/gate_units ref/goldens
set -e
cd "$(dirname "$0")/.."
nvcc -O2 -std=c++17 -gencode arch=compute_110a,code=sm_110a -I include \
  tests/gate_units.cu kernels/fp8_block_gemm.cu kernels/hc_sinkhorn.cu kernels/mla_attn.cu kernels/moe.cu kernels/hc.cu kernels/compressor.cu kernels/indexer.cu kernels/tc_moe_gemm.cu kernels/tc_fp8_gemm.cu kernels/dscratch.cu \
  -o build/gate_units
echo "built build/gate_units"
# CPU-only gates (no GPU): chat encoder byte-exactness + OpenAI API shaping.
g++ -O1 -std=c++17 -I include tests/gate_encoding.cpp -o build/gate_encoding && echo "built build/gate_encoding"
nvcc -O3 -std=c++17 -gencode arch=compute_110a,code=sm_110a -I include tests/gate_bf16w.cu kernels/compressor.cu kernels/dscratch.cu kernels/mla_attn.cu kernels/indexer.cu kernels/fp8_block_gemm.cu kernels/tc_fp8_gemm.cu -o build/gate_bf16w && echo "built build/gate_bf16w"
g++ -O1 -std=c++17 -I include tests/gate_api.cpp      -o build/gate_api      && echo "built build/gate_api"
