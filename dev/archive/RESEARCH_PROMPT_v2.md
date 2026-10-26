# RESEARCH_PROMPT_v2.md — the questions left after the levers ran out

`RESEARCH_PROMPT.md` (v1) was written before the optimisation loop had a correct model of its own
bottleneck. It asked "how do we move more bytes". **That question is now answered and it was the
wrong one.** This is the replacement, written 2026-08-07 at 15.3 tok/s speculative / 12.6 tok/s base
AR, after Findings 41-48 retired every structural lever the project could name.

---

## 0. Rules for whoever answers this — unchanged from v1, plus two

1. Every claim carries a source and, where possible, a number. Papers with released kernels, vendor
   docs and merged PRs over blog posts. Vendor-reported-and-unreproduced must be labelled as such.
2. **This is `sm_110a`, not SM100.** Blackwell-datacenter techniques must be explicitly marked as
   surviving or not. But note the scar: **this project has recorded two negative capability claims
   that were both wrong** (Findings 29, 30). Probe before believing an absence — and say what probe
   would settle it.
3. **Batch size is 1.** Single-user edge box. Anything whose benefit appears only at batch >= 8 is
   out of scope unless speculative decoding turns it into an effective batch (it turns B=1 into
   B=5, and that is already exploited).
4. **NEW — answer the diagnosis, not the symptom.** The engine is *not* bandwidth-bound. See §1.
   A technique that moves fewer bytes is worth nothing here unless it also reduces memory *latency
   exposure*. Say which.
5. **NEW — every answer must name the measurement that would falsify it.** This project retires
   levers with numbers, including its own. An answer that cannot be cheaply tested is a citation,
   not a lever.

---

## 1. The measured state — this is the whole context

**Hardware.** Jetson AGX Thor, `sm_110a`, 20 SMs, 122 GiB unified LPDDR5X, CUDA 13.0, 228 KB
smem/SM. Measured achievable read bandwidth, `tools/alloc_probe`, 8-16 GiB working set:

| allocator | stream | strided (the engine's pattern) |
|---|---|---|
| `cudaMalloc` device | 234 | 236 |
| `cudaHostAlloc` mapped | 183 | 204 |
| `cudaMallocManaged` + PreferredLocation + prefetch — **what the engine uses** | 196 | **234** |

Splitting the same bytes across 1/4/16/64/256 separate allocations: 192/183/181/183/184 GB/s.
**Allocation fragmentation is ruled out (5%).**

**Model.** `0xSero/DeepSeek-V4-Flash-0731-REAP`. 43 backbone layers + 3 chained DSpark MTP blocks.
hidden 4096, 64 heads, head_dim 512, `Q_LORA`/`O_LORA` 1024, `O_GROUPS` 8, 160 routed experts top-6
+ 1 shared, `moe_intermediate` 2048, hyper-connections hc=4 with 20 Sinkhorn iterations, first 3
layers hash-routed, compress_ratios 2 pure-sliding / 21 ratio-4 (compressor + DSA indexer) / 20
ratio-128. MLA and dense weights are **FP8 e4m3 with E8M0 128x128 block scales**; routed experts are
**OCP MXFP4** (E2M1 + E8M0, block 32); norms/embed/lm_head/compressor/indexer BF16. `B_tok` = 12.26
GB/token, engine-measured.

**Performance.** Base AR 92.5 ms/token; **79.3 ms with a full-step CUDA graph = 12.6 tok/s**.
Speculative decode (BLK=5, embedded MTP heads) **15.3 tok/s, 1.42x over base**. Byte floor for base
AR at 234 GB/s is 52.6 ms = 19.0 tok/s; cycle floor for speculation is 106.6 ms = 28.1 tok/s at the
measured acceptance.

**The diagnosis (`ncu`, `tools/ncu_target.cu`, real shapes, one launch each):**

| | MoE grouped FP4 GEMV, M=5 | ogroup FP8 GEMV, M=5 |
|---|---|---|
| Memory throughput | 25.4% | 14.3% |
| Compute (SM) throughput | 37.2% | 37.6% |
| Mem pipes busy | 8.8% | 9.4% |
| Executed IPC | 1.35 of 4 | 1.60 of 4 |
| Achieved occupancy | 64.5% | 33.2% (128 registers) |
| **`long_scoreboard` stall** | **84% of warp cycles** | **66%** |

**Nothing is above 40% of peak. The dominant stall in both is waiting on global loads.** The engine
achieves 155 GB/s of weight traffic against a 234 GB/s strided ceiling — a 1.5x gap that is
**memory-latency exposure at low arithmetic intensity**, not bandwidth, not occupancy alone, not
launch overhead, not fragmentation.

---

## 2. What has already been tried and measured — do not propose these

**Adopted:** smem-staged FP8 mma tile (fixes 32-byte-sector over-fetch, 1.6-2.9x/shape) · M=K GEMVs
for lm_head / ogroup / gemm_fp32 (B read once, not M times) · multi-row ogroup (activation reuse) ·
weights moved to managed device-preferred memory · full-step CUDA graph on base AR (1.17x) ·
templated small-M kernels · persistent MoE pointer tables · e8m0 native scales on the draft.

**Retired with a measurement:** m16 B-operand repack · clock/EMC locking · MAXN power mode · expert
prefetch/caching/sticky routing · draft/verify pipelining · 2:4 sparsity · MLA weight absorption ·
FlashMLA port · MoE g-loop software pipelining · wave-quantisation clamp · BF16-native compressor ·
shared-A GEMV · small-M GEMV as default at M>=2 · **speculation block size > 5** (accept sequence
identical at BLK=5 and 8) · **draft refinement** (acceptance 3.00 -> 2.08; the MTP heads are trained
with a noise-token placeholder, so real tokens are off-distribution) · **verify-path graph capture**
(measured 1.05x by capturing one fixed position and replaying — 7.6 ms) · **persistent megakernel**
(a fused kernel takes the max register count of its stages; ogroup's 128 would put every stage at
33% occupancy, and ncu says the stages want *more* warps) · allocation fragmentation.

**Standing scar:** `gemm_bench`-style microbenchmarks that relaunch a kernel on rotating weights
**overstate end-to-end value by 2-4x**, because consecutive launches overlap while the engine
serialises every kernel behind a data dependency, exposing its tail wave.

---

## 3. The questions

### Q1 — the central one. Closing a memory-*latency* gap at arithmetic intensity ~0.25 flop/byte on an integrated LPDDR5X part.

A warp-per-output-row GEMV streaming FP4/FP8 weights achieves 155-183 GB/s where a plain
`float4` grid-stride read of the same allocator achieves 234. `ncu` attributes 66-84% of warp cycles
to `long_scoreboard`. Occupancy is 33-75%; raising it and raising per-warp ILP each bought <1%
end-to-end.

- What closes a gap of **this** shape on Ampere/Ada/Hopper/Blackwell-consumer-class parts?
  Specifically: `cp.async` / `cuda::memcpy_async` multi-stage pipelining into shared memory for a
  *streaming, no-reuse* operand (the textbook case is reuse; here there is none — does async copy
  still help by decoupling issue from arrival?); `cp.async.bulk` / TMA if it exists on `sm_110a`;
  L2 residency control (`cudaAccessPolicyWindow`, `cudaLimitPersistingL2CacheSize`); `__ldcs` vs
  `__ldg` vs `discard`/`no_allocate` hints on a stream-once operand; prefetch distance tuning.
- Is there a published **memory-level-parallelism ceiling** for Orin/Thor-class integrated memory
  (max outstanding requests per SM / per TPC / device-wide)? If the part caps outstanding misses
  well below what 48 warps x N loads would generate, that is the answer and it is a wall.
- **Falsification:** what single micro-benchmark distinguishes "not enough requests in flight" from
  "the memory system caps concurrency"? We can run it in minutes.

### Q2 — `sm_110a` capability surface, probed not assumed.

Two recorded negative claims on this project were later shown wrong. Settle these with the PTX/ISA
evidence and the exact probe:
- `tcgen05.*` (recorded as absent: `ptxas` rejects `tcgen05.fence` on `.target sm_110`) — is that
  the whole family, or does a subset assemble with `sm_110a` (the `a` suffix matters)?
- `cp.async.bulk` / `cp.async.bulk.tensor` (TMA), `tensormap` — present on Thor?
- Thread-block clusters + distributed shared memory (DSMEM) — believed present; what is the actual
  max cluster size and DSMEM bandwidth, and is `mapa`/`cluster.sync` usable at 20 SMs?
- `mma.sync` shapes with block-scaled FP4 (`.kind::mxf4`), and `cvt` families for E2M1/E8M0.
- Is there a **CUTLASS/CuTe arch clause for `1100`**? (v1 flagged a missing `1100` case in
  `arch/reg_reconfig.h` worth 1.74x on FMHA upstream — status now?)

### Q3 — speculative decoding when the *expert union*, not the weights, is the cost.

Our verify at K=5 reads the union of ~25 of 160 experts per layer instead of 6, so the MoE — 52% of
the step — gets almost no weight-sharing benefit from batching. Measured `c_v` is 1.82 against a
byte-model floor of 1.84: **the verify's K-scaling is already optimal given the union.**
- 2025-2026 work on speculative decoding for **sparse MoE** specifically: expert-set-aware drafting,
  routing-consistency objectives, verify-time expert pruning with a correctness argument (not just
  an accuracy-drop tradeoff), union-minimising draft selection.
- Anything that makes the *draft* propose tokens whose expert sets **overlap** the ones the target
  would pick, so the union stays small.

### Q4 — raising acceptance on a **frozen, REAP-pruned** MTP head at inference time.

Acceptance is pinned at 3.00 tokens/verify, flat in block size, and refinement makes it worse. The
heads were trained for the unpruned model; REAP removed experts from the backbone under them.
- Inference-time-only techniques that lift MTP/EAGLE-style acceptance without touching weights:
  tree/multi-candidate drafts sized to a *union-aware* budget, dynamic block length, typical/
  relaxed acceptance criteria that stay output-equivalent under greedy decoding, draft-head
  ensembling across the 3 chained MTP stages, using the markov head's rank-256 bias differently.
- What is the smallest *training* intervention that is known to repair a draft head after backbone
  pruning — LoRA on non-expert parameters only? (Our budget: 0.55 GiB of the 6.53 GiB head is
  non-expert and trainable on-device; full-expert fine-tune needs ~144 GiB and is cloud-only.)

### Q5 — DeepSeek-V4 / DSA / hyper-connections-specific decode work.

- Published decode kernels or optimisations for **DeepSeek Sparse Attention (DSA)** — the
  lightning-indexer + top-k gather at batch 1.
- **Hyper-connections** (hc=4, 20 Sinkhorn iterations per layer, 2x per layer): our `hc_pre` costs
  4.8 ms/token for essentially no bytes. Is there a closed form, a fixed-point shortcut, or a
  published fused kernel?
- The **KV compressor** (ratio 4/128 gated pooling): any work on making the conditional emit cheap?
- Anyone running DeepSeek-V4-class or REAP-pruned MoE on Jetson/edge with published numbers.

### Q6 — what are we not asking?

The loop has retired 20+ levers. Name the technique class this project has *not* considered at all.
Candidates we are aware of but have not evaluated: weight layout/swizzle for the specific access
pattern, output-stationary vs weight-stationary restructuring at B=1, fusing across *layers* rather
than within, asynchronous multi-stream overlap of independent layer sub-chains, exploiting the 3
DSpark MTP blocks as a *cascade* rather than a chain.

---

## 4. Output contract

For each lever: **(a)** what it is and the source; **(b)** the mechanism, tied to the §1 diagnosis —
does it reduce latency exposure, and how; **(c)** does it survive `sm_110a` and batch 1; **(d)**
expected gain with the arithmetic; **(e)** **the cheapest measurement that would falsify it**;
**(f)** implementation size in this codebase.

Rank by `expected_ms_saved / (implementation_days x risk)`. Anything that cannot be falsified in
under one hour of on-box measurement goes at the bottom regardless of promise.
