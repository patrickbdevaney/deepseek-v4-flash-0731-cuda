# RESEARCH_PROMPT.md — maximising physical decode throughput for DeepSeek-V4-Flash-0731-REAP on Jetson Thor (`sm_110a`)

**Purpose.** Find every credible lever — academic, industry, vendor, open-source — that could raise
single-stream decode throughput on the *specific* engine described below, and rank them by expected
gain against *this* hardware's measured limits. The engine already exists, runs correct, and has
been through six gated optimisation rounds; generic advice ("use tensor cores", "try CUDA graphs")
is worthless here. What is wanted is the frontier.

---

## 0. Non-negotiable framing for whoever answers this

1. **Every claim must carry a source and, where possible, a number.** Prefer papers with released
   kernels, vendor docs, and merged PRs over blog posts. Where a number is vendor-reported and
   unreproduced, say so.
2. **Anything that assumes datacenter Blackwell (SM100/GB200) must be flagged.** This is `sm_110a`.
   `tcgen05.*` / UMMA / 5th-gen tensor cores are **empirically absent** (`ptxas`: "Instruction
   'tcgen05.fence' not supported on .target 'sm_110'"). DeepGEMM-class kernels are a **rewrite**,
   not a port. Say explicitly whether a technique survives that.
3. **Batch size is 1.** This is a single-user edge box. Techniques whose benefit appears only at
   batch ≥ 8 (continuous batching, chunked prefill throughput, EP/DP parallelism) are out of scope
   unless they *also* help batch-1 latency.
4. **The binding constraint is memory bandwidth and, at small M, memory-transaction efficiency —
   not FLOPs.** See §2.
5. **Distinguish "reduces bytes moved", "improves bytes/second achieved", and "reduces the number of
   forward passes".** Those are three different levers and they compose multiplicatively.

---

## 1. The hardware, measured (not spec-sheet)

| | value | how established |
|---|---|---|
| SoC | NVIDIA Jetson AGX Thor, Blackwell-derived, compute capability **11.0** (`sm_110a`) | `nvidia-smi`, `nvcc` |
| SMs | 20 | device query |
| Memory | **122 GiB unified LPDDR5X**, shared with the OS (no discrete VRAM) | `free` |
| Peak BW | 273 GB/s (spec) | vendor |
| **Achievable streaming BW** | **240 GB/s measured** (88–89% of spec), 212 under contention | grid-stride `float4` read microbenchmark |
| Toolchain | CUDA 13.0, `nvcc` V13.0.48, driver 580.00, L4T R38.4, aarch64 | measured |

### Empirically established `sm_110a` facts — verify these are still true, and correct us if not

- `tcgen05.*` → **NOT supported** (compile error).
- `cp.async.cg.shared.global` → **OK**.
- `__nv_cvt_fp4x2_to_halfraw2` (HW FP4×2 → half2 unpack) → **OK, and already in use**.
- **FP4 tensor-core COMPUTE** (i.e. `mma` with E2M1 operands) → reported **BLOCKED** on this box on
  all paths tried (ptxas, cuBLASLt reports 0 algos, CUTLASS, SASS inspection), as of CUDA 13.0.
  **THIS IS A PRIORITY RESEARCH QUESTION — see §5.Q1.** The user believes there are hardware FP4
  pathways on Thor. Establish the ground truth: what FP4/FP6/FP8 `mma` and `cvt` shapes does
  `sm_110a`/`sm_120a` actually expose, in which CUDA versions, and via which PTX instructions?
- **`ncu`'s "Memory Throughput %" is L2 throughput on this chip, not DRAM utilisation** — there are
  no memory-controller counters (`dram__cycles_active` absent, MC channel cycles = NaN). A kernel
  measured at 89% of peak reports 30%. Any tooling advice must account for this.

---

## 2. The model and the engine, measured

**Checkpoint:** `0xSero/DeepSeek-V4-Flash-0731-REAP` (K160). 100.400 GiB, 45,821 tensors, 48 shards.

**Architecture** (all read from `config.json`, not assumed):
- 43 backbone layers + **3 chained DSpark MTP blocks** (`mtp.0/1/2`, embedded, REAP-pruned to 160
  experts, `dspark_block_size = 5`, markov head rank 256).
- **MLA**: `hidden 4096`, 64 heads, `head_dim 512`, **1 KV head**, `q_lora_rank`/`o_lora_rank` 1024,
  `o_groups` 8, `qk_rope_head_dim` 64 (interleaved-pair RoPE, *not* rotate-half), learnable attn sink.
- **DSA**: lightning indexer, `index_n_heads` 64, `index_head_dim` 128, **`index_topk` 512**;
  Hadamard-rotated FP4 index cache; present only on `compress_ratio == 4` layers.
- **KV compressor**: learned gated pooling. `compress_ratios` = 2 pure-sliding (0), 21 × ratio-4
  (with indexer), 20 × ratio-128 (strided). `sliding_window` 128.
- **Hyper-connections**: `hc_mult 4`, **20 Sinkhorn iterations** per token per layer (×2: attn+ffn).
- **MoE**: 160 routed experts, **top-6**, 1 always-on shared expert, `moe_intermediate_size` 2048,
  `sqrtsoftplus` scoring, `noaux_tc` top-k, `routed_scaling_factor` 1.5, **first 3 layers
  hash-routed** via `tid2eid`.
- **Quantisation as shipped**: routed experts **OCP MXFP4** (E2M1 data + **E8M0 scale, block 32**
  along K); dense/attention **FP8 e4m3 with E8M0 scales, block 128×128**; norms/embed/`lm_head`/
  compressor/indexer BF16; HC params F32.

**Measured per-token decode traffic (`B_tok`) = 12.26 GB**, of which:
MLA 4599 MB (37.5%) · routed experts top-6 3449 (28%) · `lm_head` 1059 · shared expert 1082 ·
KV compressor 526 · DSA indexer 275 · HC 135 · router 75.
Active parameters ≈ 13.26 B; blended **6.76 bits/active-param** (only the routed experts are 4-bit).

**Current state of the engine** (pure CUDA, no Python on the hot path, every kernel gated bit-exact
or cosine-1.0 against a PyTorch oracle):

| | value |
|---|---|
| Base AR decode | **9.51 tok/s** = 105.2 ms/tok = **116.6 GB/s = 48.6% of achievable** |
| Roofline wall | **21.42 tok/s** @ 240 GB/s |
| DSpark speculative decode | **1.00× of base** (parity) — acceptance **3.12 of 5** tokens/verify |
| M=5 verify cost | `c_v` ≈ 2.6× an M=1 decode (byte model says 2.12×) |
| Memory at runtime | 108.1 GiB of 122.8 |

**Optimisations already applied and gated** (do not re-propose these):
1. MoE grouped GEMM warps-per-block 1 → 4 (occupancy 50% → 100%): 7.80 → 8.63 tok/s.
2. HC Sinkhorn rewritten warp-parallel in registers (it ran **one scalar thread** with a
   runtime-indexed local-memory array): 8.63 → 9.26 tok/s.
3. `lm_head`/markov read **natively as BF16** instead of dequantised to F32 (halved those reads,
   freed 2.1 GiB): 9.26 → 9.51 tok/s.
4. DSpark draft: device-side greedy AR loop replacing a host loop with per-position sync + D2H +
   CPU argmax: draft 75.7 → 54.7 ms, speculation 0.93× → 1.00×.
5. Already in use: hardware FP4×2 unpack in the MoE dequant; CUDA-graph capture of the full
   43-layer step (measured at parity — the engine is GPU-bound, not launch-bound).

**Negative results already established** (do not re-propose without new evidence):
- Wave-quantisation fixes to small-N GEMVs: worth <2% of `B_tok`; clamping hurt the large shapes.
- M=K GEMV as a general replacement for the m16 FP8 tile: wins at M=2–3, **loses at M≥8** (goes
  ALU-bound doing M scalar dots per weight read).
- FP4 tensor-core compute: blocked (but see §5.Q1 — re-verify).

**The current #1 known bottleneck.** At M ≥ 2 the FP8 m16 tile costs ~3× the M=1 GEMV **on cold
weights** for identical bytes, because of B-operand coalescing:
```
M=1 GEMV : lane L reads B[n*K + kb*128 + L*4]  -> 32 lanes x 4B = 128 CONTIGUOUS bytes/warp
m16 tile : lane L reads B[(n0 + L/4)*K + ...]  -> 8 rows STRIDED BY K, 32B from each
```
Hot, L2 hides it (0.71×); cold — which is always, since every layer streams from a 100 GiB working
set — it is 3.12×. The intended fix is a **weight repack into mma-order layout** (the FP4 grouped
MoE path already does exactly this, which is why *its* B reads are contiguous).

---

## 3. The comparison point we must beat or explain

The same REAP checkpoint has been run on **NVIDIA DGX Spark (GB10, `sm_121`)** under vLLM
(`validation/runtime-smoke.json` in the checkpoint: vLLM 0.25.2, `flashinfer_b12x` MoE backend,
CUDA graphs on, **MTP disabled**, torch fallback router because 160 is not in the fused kernel's
supported expert-count set). Published/third-party figures for that class of box report roughly
**12–14 tok/s without speculation and ~24 tok/s with MTP2**.

**Research question:** GB10 and Thor are the same memory-bandwidth class (~273 GB/s). What does the
vLLM/FlashInfer/SGLang stack do on GB10 that this engine does not, and how much of that is
`sm_121`-only? Specifically:
- Which FlashInfer / CUTLASS / DeepGEMM kernels does vLLM actually dispatch for DeepSeek-V4 MLA +
  DSA + MXFP4 MoE on GB10, and what are their reported achieved bandwidths?
- Is `sm_121` (GB10) materially different from `sm_110a` (Thor) for FP4/FP8 `mma`, `tcgen05`,
  TMA/`cp.async.bulk`, or cluster/DSMEM features? **A precise per-feature comparison table is
  wanted.**
- What is the *actual* vLLM MTP acceptance rate and speedup on DeepSeek-V4-Flash, and how does it
  implement the verify step (tree vs linear, expert dedup, KV handling)?

---

## 4. What to search — breadth axes

Search academic (arXiv, MLSys, ASPLOS, ISCA, MICRO, SOSP/OSDI, NeurIPS/ICML systems tracks),
industry (NVIDIA developer blogs and GTC sessions, vLLM/SGLang/TensorRT-LLM/llama.cpp/MLC/ExLlama
repos, issues and merged PRs), and vendor documentation (PTX ISA, CUDA C++ Programming Guide,
CUTLASS, Nsight docs, Jetson/L4T release notes). Prioritise **2025–2026** work, but include
foundational kernels where still SOTA.

1. **Low-precision GEMM/GEMV kernels at batch 1.** MXFP4/NVFP4/FP8 weight-only and W4A8 dequant
   GEMV; Marlin / Machete / EXL3 / Trellis / LiquidGEMM / FlashInfer's fused MoE; register-level
   dequant, funnel shifts, `cp.async` pipelining, L2 residency and cache hints
   (`.L2::no_allocate`, `evict_first`/`evict_last`), TMA where available. **What is the best known
   achieved-bandwidth fraction for a weight-only 4-bit GEMV on a 20-SM, 273 GB/s part?**
2. **Weight layout / repacking for mma.** Canonical mma-order (swizzle) layouts, `ldmatrix`, bank-
   conflict-free shared staging, and what the measured benefit is on *cold* weights specifically.
   Include CUTLASS layout conventions and any work that quantifies coalescing loss at small M.
3. **MoE decode at batch 1.** Grouped/masked GEMM, gather fusion, persistent kernels, expert
   weight prefetch, top-k routing on device, SonicMoE / DeepEP-descendant ideas that survive
   single-node, and anything on **160-expert / top-6** specifically.
4. **MLA-specific decode kernels.** FlashMLA, FlashInfer MLA, absorbed vs non-absorbed projection
   forms (does folding `wq_b`/`wo_a` into the attention change the byte count?), weight-absorption
   tricks for low-rank KV, and DeepSeek's own open-sourced kernels.
5. **DSA / sparse-attention decode.** Lightning-indexer implementations, top-k selection kernels,
   gather-friendly KV layouts. *(Note: measured at 0.05 ms/step here — likely NOT a lever, but
   confirm nobody has shown it should be fused with something adjacent.)*
6. **Speculative decoding, current frontier.** MTP/EAGLE-3/Medusa/HASS/Falcon; **verify-step cost
   reduction** (expert-union dedup, shared KV, tree vs linear); dynamic draft length; typical
   acceptance vs exact rejection sampling; self-speculation without a separate draft model; and
   specifically **DeepSeek DSpark / DeepSpec** (arXiv 2606.19348, github deepseek-ai/DeepSpec) —
   what acceptance and speedup are reported, under what verify implementation?
7. **Fusion and scheduling.** Megakernels / persistent-kernel decode (e.g. "one kernel per token"),
   fused RoPE+norm+quant+KV-write, CUDA graph conditional nodes, PDL, stream-ordered allocators,
   and how much each is worth at batch 1 when already graph-captured.
8. **Reducing bytes rather than moving them faster.** KV cache quantisation (already tiny here),
   **activation-aware weight-only 4-bit for the FP8 dense/attention tensors** (MLA is 37.5% of
   `B_tok` and is currently 8-bit — what is the accuracy cost of taking `wq_b`/`wo_a`/`wo_b` to 4
   bits, and does anyone do this for MLA?), sparsity/activation sparsity at decode, and
   `lm_head` tricks (vocabulary pruning, hierarchical softmax, sampling without full logits).
9. **Unified-memory / Jetson-specific.** Zero-copy vs explicit copy, `cudaMemAdvise`/prefetch,
   page-migration behaviour on Thor, LPDDR5X access patterns, EMC/clock governors, `jetson_clocks`
   and power modes, and whether any of it changes achieved bandwidth.
10. **Anything that beats 21.4 tok/s.** The roofline says 21.42 tok/s at 240 GB/s with `B_tok`
    = 12.26 GB. **The only ways past that wall are: move fewer bytes (§4.8), or emit more than one
    token per weight-read (speculation, §4.6).** Enumerate every credible instance of both.

---

## 5. Specific questions that must be answered

- **Q1 (highest priority).** What FP4/FP6 hardware paths actually exist on `sm_110a` (Thor) and
  `sm_120a`/`sm_121` (GB10/RTX 50)? Give the exact PTX/`mma` shapes, the `cvt` intrinsics, the
  minimum CUDA version, and any CUTLASS support. If `mma` with E2M1 operands is unavailable but a
  fast `cvt` path is, quantify the best achievable dequant-then-`mma.f16` throughput. **Cite the
  PTX ISA section numbers.**
- **Q2.** Is there a published, measured number for weight-only 4-bit GEMV achieving >70% of
  achievable bandwidth on any Blackwell-class part at M=1? What kernel, and what is the layout?
- **Q3.** For an m16-class `mma` tile at M ∈ [2,8], what is the standard fix for B-operand
  coalescing loss on cold weights, and what does it measure? (We intend a repack; is there a better
  answer — e.g. loading B through shared memory with `cp.async` in a K-major staging buffer?)
- **Q4.** For DeepSeek-V4/V3-class MLA, does anyone fold the up-projection into the attention
  ("weight absorption") in a way that reduces bytes read per decode step, and by how much?
- **Q5.** What is the state of the art for reducing speculative **verify** cost specifically, given
  a fixed acceptance rate? Rank by measured effect.
- **Q6.** Given 20 SMs and 273 GB/s, is a **persistent megakernel** decode (whole layer or whole
  step in one kernel launch, weights streamed via `cp.async`) a known win at batch 1, and is there a
  released implementation to learn from?
- **Q7.** What accuracy cost is documented for quantising MLA projections (`wq_b`, `wo_a`, `wo_b`)
  from FP8 to 4-bit in DeepSeek-class models, and are there published recipes?

---

## 6. Deliverable format

For each lever:
1. **Name + one-line mechanism.**
2. **Which of the three axes it moves** (fewer bytes / higher achieved BW / fewer forward passes).
3. **Expected gain on THIS engine**, expressed against the numbers in §2 — not a generic percentage.
4. **`sm_110a` viability**: works / needs rewrite / arch-blocked, with the specific reason.
5. **Implementation cost** and the risk of breaking bit-exactness.
6. **Sources**, with the strongest one first, and an explicit note where a number is unreproduced.

Finish with a **ranked table** ordered by `expected_gain / implementation_cost`, and an explicit
statement of **what the maximum physically achievable decode rate is** for this model on this
hardware — base AR and with DSpark speculation — with the arithmetic shown.
