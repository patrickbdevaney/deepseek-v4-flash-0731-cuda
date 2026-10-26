# RESEARCH_LOG.md — every query run, every result, so no cycle searches the same thing twice

`LEVERS.md` dedups **implementation**. This dedups **search**. A cycle that re-runs a query already in
§2 has spent its budget learning nothing, and web-search budget is per-session and finite.

## 1. The protocol (the executor follows this; it is not advice)

**Cadence.** Research runs when the loop needs candidates, not every cycle: if `LEVERS.md` §4/§5 hold
**≥2 open levers with an expected value ≥1 %**, skip research and go implement. Otherwise run one
research phase. *Every adoption so far — F64, F65, F70, F71, F72 — came from profiling and reading the
engine, not from literature.* Research is the tiebreaker when the profile stops suggesting things, not
the driver.

**Fan-out.** One research phase issues **6–8 queries, at most one per axis** below, and follows **at
most 2** into a lever. Breadth comes from the axes; depth comes from the next cycle picking the lever
up. Do not spend a whole phase on one axis — that is how a loop convinces itself.

**Axes** (rotate; record which were used):

| # | axis | what it covers |
|---|---|---|
| A | this hardware | `sm_110a`, Thor, Orin, Jetson, unified LPDDR5X, iGPU occupancy |
| B | MoE inference | expert caching/prefetch, grouped GEMV, routing cost, union reduction |
| C | speculative decoding | acceptance, MTP/Medusa/EAGLE, tree vs linear, verify width |
| D | bandwidth without quantising | layout, `cp.async`/TMA, swizzle, L2 policy, split-K alternatives |
| E | attention | MLA, DSA/sparse KV, compressed KV, indexer-style top-k |
| F | the number we are stuck on | paste the actual shape: e.g. "fp8 GEMM N=1024 K=4096 batch 5 warp starvation" |

**Constraints that kill most results before you implement them** — check against these *first*, and
record the rejection rather than trying:

- **no additional quantisation** of the checkpoint (rules out most "speed up MoE" papers);
- **no retraining / no draft-model training** (rules out most acceptance work);
- output must stay **lossless vs base AR** — the engine gates this now, and Finding 68 is the worked
  example of a "+28 %" that was a quality collapse;
- it must be implementable against **this** engine, not a rewrite.

**Write-back is mandatory.** A research phase that finds nothing still appends a row to §2 saying so.
An unrecorded search will be run again.

## 2. Queries run

| date | axis | query (short) | outcome | became |
|---|---|---|---|---|
| 2026-08-07 | — | *(session budget exhausted at 200/200 before a research phase could run; F64–F73 came from profiling instead)* | n/a | — |
| 2026-08-07 | C | draft-head fine-tune (operator-supplied, not a search) | **viable and lossless by construction** — see §5 | LEVERS.md **S5** |
| 2026-08-07 | C | n-gram / prompt-lookup speculation (operator-supplied) | viable, free, additive to MTP | LEVERS.md **S6** |
| 2026-08-07 | B | "get the GEMM right for the MoE" (operator-supplied) | **already exists and is 1.9x SLOWER** — closed with a measurement | §3 |
| 2026-08-07 | A | K160 not a fused expert count -> Torch router fallback | **does not apply to this engine** — vLLM artifact | §3 |
| 2026-08-08 | C | **FastMTP full text** (WebFetch arXiv 2509.18362v1) — recipe + is there a data-scaling study? | recipe extracted in full; **NO data-scaling study exists**, but the on-policy ablation is decisive — see §6 | S5 plan |
| 2026-08-08 | — | *(cycle 12: research SKIPPED per §1 — B8 and S6 were both open at ≥1 %, meeting the ≥2-open-levers bar. F76 came from `ncu` on this engine, like every adoption before it. The research branch has still never executed.)* | n/a | — |
| 2026-08-08 | A-F | **operator deep-research dossier** (web-chat, ~24 sources: FastMTP, HyperDFlash, SuffixDecoding, Sequoia, EVICT/EcoSpec/AcceptMoE, REAP, Thor MLPerf) | **4 promoted, 6 already done here, 3 inapplicable** — full triage in §3a | S5 S6 S7 S4 |
| 2026-08-08 | — | *(cycle 13: research SKIPPED per §1 — B8 (~5 %, the tile double-buffer) and S6 were both open at ≥1 %, meeting the ≥2-open-levers bar. F78 came from `ptxas -v` and `gemm_bench` on this engine. The research branch has STILL never executed, and after F78 the kernel queue is down to B8' + S6 — the next cycle is the likeliest one to take it.)* | n/a | — |
| 2026-08-08 | — | *(cycle 14: research SKIPPED per §1 — B8' (~1–2 %) and S6 were both open at ≥1 % at ORIENT time, meeting the ≥2-open-levers bar. F79 came from `ptxas -v` and `gemm_bench` on this engine, like every adoption and every kill before it. **B8' died this cycle, so only S6 is left at ≥1 % and the research branch is DUE next cycle** — it has still never executed.)* | n/a | — |
| **2026-08-08** | **A** | **Jetson Thor sm_110a LLM decode inference optimization unified memory bandwidth 2026** | **nothing usable.** Vendor material plus general "quantise and speculate" advice, both already done here. One number worth recording as a *contradiction to check*: two sources quote Thor's LPDDR5X at **273 GB/s** peak where `bw_probe` measures **233–240 GB/s achievable** — consistent (85–88 % of peak is a normal STREAM efficiency), so it does **not** reopen the roofline. Ghidorah (arXiv 2505.23219) runs the *drafter* on the host CPU cores; inapplicable, because our draft is a 5-layer GPU model whose weights the 14 Neoverse cores cannot stream, and the draft is serially dependent on the previous verify so there is nothing to overlap it with. | — |
| **2026-08-08** | **B** | **MoE inference batch-1 decode expert weight prefetch overlap without quantization 2026** | **nothing usable — the whole class assumes offload.** ST-MoE (2606.15453), SpecPrefetch (2607.24787), Speculating-Experts (2603.19289), "Predictive Prefetching and Expert Replication" (2605.11537) all hide a **CPU/disk → GPU expert transfer** behind compute. **We have no such transfer**: all 100.4 GiB is resident in one unified LPDDR5X pool and `footprint_probe` measures 230–246 GB/s at footprints from 0.5 to 64 GiB, so there is no staging step to overlap and prefetching only moves bytes that are already where they need to be. Their shared *observation* — expert selections correlate between consecutive decoded tokens — is the same one S4/EcoSpec rests on, and our measured K=5 union of **17.53 of 30** already bounds it. | — |
| **2026-08-08** | **C** | **training-free speculative decoding acceptance improvement 2026 draft head alignment no finetuning** | **one promoted, one rejected on a standing constraint.** REJECTED: the 2026 "relaxed / loosely speculative" family (FLy, arXiv 2607.08690) buys acceptance by accepting *semantically equivalent* tokens the target would have rejected — that is exactly the lossless-vs-base-AR constraint, and Finding 68 is this project's worked example of what a quality collapse does to the headline number. PROMOTED: **UniSpec** (ACL 2026 long 285) is training-free, keeps output identical to AR, and its middle component is a **confidence estimate for n-gram drafts** — i.e. it tells a prompt-lookup drafter *when to trust itself*. Folded into **S6** as its gating rule rather than opened as a separate lever (its other two components are device calibration, which F60 already did, and tree expansion, which §3 retired here on a measurement). **MOOT SAME CYCLE:** S6's oracle ceiling measured **+0.0 %** later in this cycle (F80), so there is nothing for a confidence gate to gate. | **S6** (refined, then RETIRED F80) |
| **2026-08-08** | **D** | **cp.async TMA Blackwell small-M GEMV weight streaming decode kernel warp specialization** | **PROMOTED — and it is new evidence against the specific measurement that killed F78.** The consensus design is a *decoupled producer* (one warp issuing `cp.async.bulk.tensor` into a **≥4-stage** smem ring) rather than a deeper register buffer, and the sources state the failure mode explicitly: **"with only 2 stages, a small delay in TMA completion or barrier arrival can stall tensor-core issue almost immediately."** F78 tested exactly a 2-stage *register* double-buffer and measured **+0.28 %**. That is a different mechanism, not a re-argument, so `B8-cpasync` moves from "idea-not-open" to **open at ~1–3 %**. | **B8-cpasync** |
| **2026-08-08** | **E** | **MLA decode kernel batch 1 low-rank absorb sparse KV indexer optimization 2026** | **nothing usable, and the axis is now closed for this checkpoint.** FlashMLA / TyphoonMLA / GQLA (2605.15250) all optimise **classic MLA**, which per §3a **this model is not**: it is GQA + Q-LoRA rank 1024 + grouped O-LoRA. The absorb-vs-naive choice they turn on does not exist in our kernels. Independently the axis is not where the time is: `cattn:indexer` is **3.6 ms = 3 %** after F71 took 57 % off it, and the ogroup marks are attributed and closed (F76/F79). | — |
| **2026-08-08** | **F** | **latency-bound GEMV small SM count GPU persistent kernel wave quantization decode tail effect** | **nothing usable; two named ideas both hit existing kills.** Stream-K (Colfax) is the strongest fit for our 32-tiles-on-**20**-SMs tail, but it is a K-split reduction and **B2/F68 retired that on numerics** — it is not bit-exact, so it needs a tolerance gate the engine does not have. Persistent/megakernel decode is bounded by our own `VERIFYGRAPH` measurement at **1.05x** (F46). Wave-quantisation schemes on the M=1 GEMV were tried twice and were worse or in the noise (§3). One line worth keeping as a *mechanism* for F76's unexplained latency bound: **"since L1 throughput scales with active SMs, reallocating SMs increases GEMV latency"** — consistent with `o:wo_a` stalling 8.1 of 13.0 cycles on an L1TEX scoreboard, and it predicts that spreading that kernel wider would make it worse, which is what the `OGMK_BLOCKS_PER_SM` sweep found. | — |

| **2026-08-08** | **—** | *(cycle 16: research NOT run, and the reason is recorded rather than glossed. §1's bar is "≥2 open levers at ≥1 % → skip"; at ORIENT time there was exactly **1** (`B8-cpasync`), so on the letter of the protocol a phase was due. It was skipped deliberately because that one lever's own falsification order made it resolvable for **two recompiles and one bench, no checkpoint load**, and resolving the open item dominates searching for another. It resolved NEGATIVE — **F81**, bench +15 to +53 %. **The cost of the choice is that the queue is now 0 with nothing promoted to replace it.** So for the next cycle the branch is not merely due, it is the only non-training move: either a research phase produces a lever that survives §1's constraints, or `open_nontraining_levers` reads 0 a second time and the queue-empty arm of the pivot criterion fires alongside the adoption arm that has been firing since cycle 14.)* | n/a | — |
| **2026-08-08** | **D** | *(the axis-D promotion of cycle 15, closed by measurement in cycle 16.* **The literature's central claim for this design was tested and is FALSE on this kernel.** The promoted quote — *"with only 2 stages, a small delay in TMA completion or barrier arrival can stall tensor-core issue almost immediately"* — predicts depth pays. Measured: **NS=2 beats NS=4 on 4 of 6 shapes**, and the whole ring is **15–53 % slower** than the register-staged path it replaces (2–26 % even at cp-size 16, which this engine's 4-byte-aligned weights cannot supply anyway). The ring's stage is one K-block, so it *halves bytes-per-barrier* — and bytes-per-barrier is precisely what F74's adopted win *raised*. **Lesson for this log: a source's stated failure mode is a hypothesis about ITS hardware and tile shape, not evidence about ours** — the sm_110a tile here is 8 KB over 256 threads on 20 SMs, far from the H100/B200 shapes the guidance is written for. sm_110a *does* assemble the whole `cp.async.bulk.tensor` / `mbarrier` family, so the ISA was never the issue.*) | **retired** | ~~B8-cpasync~~ |

| **2026-08-08** | **B** | **MoE decode single-batch fuse expert FFN kernels reduce launch barrier overhead grouped GEMV 2026** | **nothing usable as a kernel lever, but ONE line changed this cycle's direction.** The fused-MoE / grouped-GEMM family (SonicMoE 2512.14080, FlashMoE 2506.04667, the NVFP4 grouped-GEMM worklog) is the *token-dimension* tile-efficiency class already retired in §3 — it is 15-20 % "widening with batch size", degenerate at M=1, and our own measurement has the grouped GEMM **1.9x SLOWER** than the GEMV here. The usable line is TaxBreak's (2603.12465) decomposition: **"MoE models dispatch 8-11x more GPU kernels per output token than dense models, and MoE inference remains persistently host-bound during decode."** Our verify answer to that is already measured and negative (`VERIFYGRAPH` = **1.05x**, F46) — but F46 measured the VERIFY. **Nobody had ever asked it of the DRAFT half**, and reading the draft path for that reason is what found F82. | **F82** (the read, not the paper) |
| **2026-08-08** | **C** | **lossless speculative decoding adaptive draft length dynamic block size acceptance no training 2026** | **nothing usable; every branch lands on something already shipped or already retired here.** FlexDraft (2605.20022) "adjusts verification length based on draft confidence" — that IS our adaptive verify width, same idea and the same EVICT citation, shipped since F49/F59 and re-fitted at F63 (+9-11 % where it engages). AdaEAGLE's MLP draft-length predictor is the same knob with a learned gate, i.e. training. OPT-Tree / SpecBranch / CAS-Spec are all **tree** speculation, retired here on a measurement (§3: correct, does not beat linear, depth-dominated). SPEQ/SubSpec (quantised-substitute self-drafters, training-free and lossless) attack **draft COST**, and our draft is ~14 % of the cycle by a stale F47 measurement — worth checking before believing, which is F82. | — |
| **2026-08-08** | **D** | **CUDA L2 persisting access policy window streaming weights LLM decode bandwidth improvement** | **rejected on this engine's own byte count, not on argument.** The mechanism is real and available (`persistingL2CacheMaxSize` = **24 MB** on this box, `accessPolicyMaxWindowSize` = 128 MB, L2 = 32 MB — `research/MOE_DECODE.md`), but it only pays when a hot buffer is re-read *within* the L2's residency. Base AR streams **12.26 GB** per token and a K=5 verify **18.89 GB**; the gap between two reads of any weight is the whole model. 24 MB of persistable window against a 12-19 GB re-read distance is a 0.2 % cache. There is no reuse to protect and nothing to evict competitively. Recorded so no cycle re-proposes it: it is dead by arithmetic, at 500x. | — |
| **2026-08-08** | **F** | **MXFP4 grouped GEMV MoE expert weights bandwidth bound one row per expert decode percent of peak** | **nothing usable.** Reference points only, and we are ahead of them: llama.cpp's MXFP4 GEMV is quoted at **~10 % of bandwidth peak** (DP4A, no tensor cores) where our MoE GEMV runs at ~72 % of a *measured* 233-240 GB/s achievable. Kimi-K3's CDNA4 "native scaled upcast inside the K loop" is what our MXFP4-straight-to-MMA path already does (we never dequantise to FP8). The recurring recommendation in this class — MXFP4 the dense/always-on linears and `lm_head` too — is **additional quantisation** and blocked by the standing constraint (it is priced in `research/MOE_DECODE.md` at 507 MB/token = 4.1 % of `B_tok` if the constraint were ever relaxed; that is a B5 decision, not a lever). | — |
| **2026-08-08** | **A** | **Jetson Thor Blackwell prefill batched forward pass throughput compute bound tuning LLM 2026** | **nothing usable; vendor and serving-stack material only.** Everything returned is vLLM/TensorRT serving configuration (`max_num_batched_tokens`, chunked prefill, continuous batching, `gpu_memory_utilization`) which presupposes a serving stack this engine is not, plus the standing NVFP4 recommendation which is blocked quantisation. No source anywhere prices a **batch-1-tuned kernel set run at prefill row counts**, which is exactly B9's question and exactly why B9's first step is a `DSV4_DPROF` profile of a PS=1023 prefill rather than a search. | — |
| **2026-08-08** | **F** | **batch-1 LLM decode memory floor CUDA graphs not bandwidth limited study (arXiv 2605.30571)** | **nothing usable, and it is the closest paper to our regime, which makes the null informative.** *"Memory-Bound but Not Bandwidth-Limited"*, 44 cells, 4 GPUs: achieved fraction of the analytic memory floor **falls** as peak bandwidth rises (L4 81 %, H100 27 %), and its headline intervention is **CUDA Graphs at 1.259x on H100 / 1.028x on L4**, attributed to "previously hidden launch-side overhead". Both halves are already answered here with our own numbers: we are at **71 %** of the measured floor for base AR (nearer the L4 end, as its own trend predicts for a 233 GB/s part), and the base-AR full-step graph is **adopted** (F44, 92.5 -> 79.3 ms/tok = 1.17x) while the verify graph measured **1.05x** (F46). The paper's value here is as a *second* independent statement that launch-side cost is where batch-1 decode leaks — which, together with TaxBreak, is why this cycle spent itself reading the one region of our cycle where that had never been checked. | — |
| **2026-08-08** | **E** | **GQA decode kernel KV cache layout bandwidth batch 1 sliding window attention 2026** | **nothing usable; axis E stays closed.** SoA KV layouts and GQA-aware decode tiling (the MI450 Gluon guide, GQLA 2605.15250) target the KV *read* at long context — ours is 43 positions, the window KV is already fp8-simulated on the NoPE dims, and `cattn:indexer` is **3.6 ms = 3 %** after F71. UltraQuant-style 4-bit KV is additional quantisation. This is the second consecutive phase in which axis E returned nothing applicable to this checkpoint. | — |

**Phase verdict (cycle 17).** Seven queries, axes A/B/C/D/E/F (F twice). **Zero promotions.** Every hit was
already implemented here, already retired here with a number, or blocked by a standing constraint —
and three of them (L2 persistence, the MXFP4 dense path, tree/adaptive speculation) are now recorded
with the specific arithmetic that kills them so they cannot be re-proposed as new. Per §5 this is the
**first** of the two consecutive empty research phases that define "done". What the phase *did*
produce is not a citation but a direction: two independent 2026 sources (TaxBreak, arXiv 2605.30571)
name **host/launch-side overhead** as the leak in batch-1 MoE decode, this engine has measured that
claim on its verify half (F46, 1.05x) and never on its draft half, and the draft half's only
attribution in the whole project dates to `evidence/f47.log` — before every verify-side adoption.
That read produced **F82**, which is a profile of this engine, like every adoption before it.

## 3. Ideas seen and rejected before implementation

| idea | source | why rejected here |
|---|---|---|
| FP4/INT4 for MLA + dense weights | general LLM inference practice | moves the byte floor ~1.2x but violates *no additional quantisation*. Needs an explicit decision to relax the constraint — see `LEVERS.md` B5. |
| tree / multi-branch speculation | Medusa, EAGLE-style | measured on this engine already: correct but does not beat linear, depth-dominated. `LEVERS.md` §3. |
| draft refinement (feed proposals back) | standard MTP practice | acceptance **3.00 → 2.08** here: the shipped MTP heads are trained with a noise-token placeholder, so real proposals are off-distribution (F45). |
| grouped **GEMM** (mma) for the MoE instead of the GEMV | "get the GEMM right for the MoE" | **it is already implemented and it loses.** `tc_fp4_grouped_gemm_e8m0` exists and `g_moe_gemv` selects the GEMV over it. ncu on the measured grouping: **GEMV 416 µs vs mma GEMM 783 µs (1.9x slower)**, despite the GEMM reaching 93.9 % occupancy against the GEMV's 53.7 %. At 1.71 rows/expert there is no arithmetic intensity for a tensor core to exploit and the m16 tile wastes ~90 % of its rows; the problem is bandwidth and launch geometry, which is what F64/F65/F70/F72 actually fixed. |
| pad experts 160 → 192 for a "fused router path" | vLLM model-card note that K160 is outside the supported set (16,32,…,512) | **a vLLM kernel artifact, not a hardware or model one, and this engine is not vLLM.** Our MoE grouping is our own code (`k_moe_count`/`k_moe_prefix`/`k_moe_scatter`/`tc_build_tiles`) and has no fixed expert-count assumption anywhere. Measured: routed MoE runs at ~72 % of the achievable roofline and the grouping itself is 2.12 ms of a 132 ms verify, now at the launch-latency floor (F73). Padding to 192 would cost ~20 % more resident memory to fix a path we do not take. |

## 3a. Triage of the 2026-08-08 deep-research dossier

The dossier is good and its architecture reading matches our implementation exactly (GQA + Q-LoRA
rank 1024 + grouped O-LoRA — *not* classic MLA; mHC with `hc_mult=4` and 20 Sinkhorn iterations;
DSpark MTP at layers 40/41/42). Its **Phase 2 conclusion — re-align the draft head — agrees with our
own independent ranking of S5.** But most of **Phase 1 targets vLLM, not this engine**, and our marks
say so with numbers. Scored against `evidence/moescan.log` (K=5 verify = 134.77 ms):

| dossier recommendation | claimed | **what it is worth HERE** |
|---|---|---|
| ① fused K160 `sqrt(softplus)` top-6 router — *"cheapest measurable win", +2-4 tok/s* | removes a pure-Torch fallback | **already implemented.** `router_kernel`/`compute_scores_warp` is a fused CUDA sqrtsoftplus top-6-of-160 with the exact `log1p` stabilisation recommended. `moe:router` = **1.83 ms = 1.4 %** of the verify. Making it *free* is +1.4 %, not +2-4 tok/s. There is no Torch anywhere in this engine. |
| ② autotune the E=160 MXFP4 grouped-GEMV; never dequant to FP8 | +1-3 tok/s | **done, empirically.** F64 row-amortisation, F65/F70 RB, F69 BN, F72 `uint2` loads → MoE at **~72 % of achievable**. We never dequant: MXFP4 feeds the MMA directly. The *grouped GEMM* alternative is measured **1.9× slower** (416 vs 783 µs). |
| ③ pin EMC top-DVFS, MAXN, page placement, `cudaMemAdvise` | +1-2 tok/s | **done (F60):** +3.0 % steady, **+20.7 % on the cold base-AR window**. EMC verified pinned at 4266 MHz. Page placement specifically **does not cost bandwidth here** — `footprint_probe` measures 230-246 GB/s from 0.5 GiB to 64 GiB. |
| ④ persistent/megakernel decode loop | +3-6 tok/s, "~78 % of bandwidth" | **bounded by our own measurement at ~5 %.** The `VERIFYGRAPH` experiment captured the whole 43-layer verify into one graph: **1.05×**, 2788 nodes (F46). The megakernel figure is for a *dense* Llama. Our glue is ~9 % of the verify and the graph already removes the launch half of it. |
| ⑤ retrieval drafting (Prompt-Lookup, **SuffixDecoding**) | 2-5× on agentic | ~~promoted → S6~~ **MEASURED AND RETIRED (F80): the oracle `max(MTP, suffix)` is +0.0 %** over 21 verifies — 0 wins, 17 losses, and a suffix match existed in only 8 of 21. The dossier's number is for *agentic* traces where long spans are copied verbatim; here the drafting anchor is always the **correction** token, so the drafter is queried at the least repetitive point in the sequence (trap 27). **This is the second dossier item whose headline did not survive contact with our own marks**, and the failure mode is the same one as ①: a magnitude inferred from someone else's workload rather than from a profile of this engine. |
| ⑥ **re-align the DSpark MTP head** | ×1.54 → ×2.0-2.8 | **promoted → S5, already our top lever.** The dossier supplies the recipe and the magnitude — see S5. |
| ⑦ cost-aware tree + expert-union truncation (EVICT) | up to 2.35× | **half already shipped.** Our adaptive verify width *is* an EVICT-style cost-effective-prefix truncation and cites the same arXiv 2605.00342; measured +9-11 % where it engages (F63). The **expert-overlap-aware draft *selection*** (EcoSpec) is genuinely new → S4. |
| ⑧ re-prune to K128, or 3-bit experts | raises the roofline | **blocked by standing constraints** — no re-prune, no additional quantisation. Also note ① removes the only reason K128 was attractive. |
| ⑨ HyperDFlash mHC-aligned drafter | τ 2.93 → 3.69, 2.25× → 2.80× | **promoted → S7.** Architecture-specific to *this* model and the only published work on conditioning a drafter on pre-collapse mHC states. |
| ⑩ FP8 → NVFP4 KV | long-context | our window KV is already `act_quant_fp8sim` on the NoPE dims. NVFP4 KV is additional quantisation → blocked, and the dossier itself reports a Spark-class corruption erratum. |

**The one number the dossier gets materially wrong for us is its headline Phase-1 item**, and the
error is instructive: it inferred a router bottleneck from a *model card's note about vLLM's kernel*
rather than from a profile of the engine in question. Our `moe:router` mark answers it in one line.

**Where it is right and we were not:** it supplies concrete re-alignment recipes and recovery curves
that turn S5 from "probably the biggest lever" into a costed plan, and it names the k=7 regression on
this exact model as independent confirmation of the MoE verify-union economics our own `c_v` table
(1.000/1.435/1.744/1.977/2.194 at K=1..5) measures from the other side.

## 4. The ceiling, from MEASURED bytes (not from a parameter-count estimate)

Any external estimate of this model's ceiling has to guess active parameters. We do not have to — the
bytes are measured:

| quantity | value | source |
|---|---|---|
| `B_tok` (all weights, one token) | **12.26 GB** | `ROOFLINE.md` |
| non-routed remainder `B_fixed` | 8.81 GB | derived from the measured K=1 union of exactly 6.00 |
| achievable bandwidth | **233 GB/s** (235.6 pinned) | `bw_probe`, and `footprint_probe` confirms it holds at 64 GiB |
| K=5 verify bytes | 8.81 + 17.53×13.37 MB×43 = **18.89 GB** | measured union (F64) |

- **base AR floor = 52.6 ms/tok = 19.0 tok/s.** Measured 13.50 → **71 % of roofline**, not 40-50 %.
- **spec cycle floor ≈ 94 ms** (81.1 verify + ~13 draft) → at the current acceptance 2.90, **30.8 tok/s**.
  Measured 20.44 → **66 %**.
- acceptance is the multiplier on the same cycle: **3.2 → 34.0, 3.6 → 38.3, 4.0 → 42.5 tok/s**.

So the headroom splits cleanly: ~30 % left in kernel/latency work on a cycle that is already
well-tuned, and a **1.4x sitting in acceptance alone**. That is why S5 outranks everything else open.

## 6. S5 feasibility ON THIS BOX — measured, not estimated

The question "can we fine-tune the draft head on Thor" splits into four, and three now have numbers.

**The structural fact that makes it possible at all: you do not backprop through the frozen backbone.**
The draft head consumes the backbone's hidden state, and `dspark_tap_pool` already emits exactly that
at layers 40/41/42 in the prefill path. So training = (a) capture states with the engine we have,
(b) train a small head on them.

### (a) RAM — not the constraint ~~✗ WRONG AS WRITTEN — corrected 2026-08-08~~

~~FastMTP's head is **210.8M params, <3 % of a 7,833.4M backbone**. AdamW working set: 0.42 GB bf16
weights + 0.84 GB fp32 master + 1.68 GB moments ≈ **3 GB**, plus activations. Against 122 GiB unified
this is nothing; the 12.26 GB backbone can stay resident beside it. **RAM was never the problem.**~~

**210.8 M is FastMTP's MiMo-7B head. It is not ours, and it was never checked against the
checkpoint** — a paper's number was copied into a feasibility conclusion about a different model.
This is the same failure the "no invented model constants" rule exists to prevent; the rule was
written for constants pulled from the air, and it turns out a constant pulled from a *paper* is
just as wrong. Read from the safetensors headers of shards 46–48:

| quantity | value |
|---|---|
| stored elements in `mtp.*` | **6.935 B** (2977 tensors) |
| MoE experts, stored | 6.559 B as `I8` |
| MoE experts, **real** | **12.08 B** — `w1.weight [2048,2048] I8` vs logical `[2048,4096]`, MXFP4 packed 2/byte |
| **non-expert** (attn, norms, Markov head, confidence head, projections) | **476 M** = 158.8 M × 3 blocks |

bf16 master + AdamW fp32 moments at 10 B/param: the **whole head is ~116 GiB and infeasible**;
**non-expert only is 4.44 GiB and comfortable**. The conclusion "RAM was never the problem" survives
only because the recipe freezes the experts — which is independently what NVIDIA's reference DSpark
trainer does. Had we planned a full fine-tune on the strength of the 3 GB figure, it would have
failed at allocation after the capture had already been paid for. See `S5_RECIPE.md` §1.

### (b) Capture rate — MEASURED, and slower than hoped

`evidence/prefill_sweep.log`, `evidence/prefill_long.log`, one checkpoint load each:

| prefill positions | time | tok/s |
|---|---|---|
| 5 | 178.5 ms | 28.0 |
| 255 | 4848.3 ms | **52.6** |
| 511 | 10150.2 ms | 50.3 |
| 1023 | 21452.9 ms | **47.7** |
| 2047 | 46747.1 ms | 43.8 |

**Prefill is ~48 tok/s at realistic sample lengths and declines gently.** That is only **3.4x the base
AR decode rate** (14.20 tok/s in the same run) — far below the order of magnitude a batched forward
ought to give. At PS=255 the 12.26 GB of weights amortise over 255 positions, so this path *should* be
compute-bound and much faster. It is not, because every kernel in it was tuned for M=1 decode. See
`LEVERS.md` **B9**.

Consequence: teacher-forced capture is only **2.2x cheaper than on-policy generation** (47.7 vs 21.68
tok/s), not the 10-50x that would have made the choice obvious.

### (c) On-policy or not — the ablation settles it, and it inverts the obvious answer

FastMTP has **no data-scaling study**: no minimum-sample experiment, no saturation curve. But it does
ablate the thing that matters more, at K=3:

| variant | speedup |
|---|---|
| vanilla MTP (the head as shipped) | 1.21x |
| **fixed-data FT** (original dataset responses, teacher-forced) | **2.54x** |
| self-data FT (on-policy self-distilled) | 2.73x |

**On-policy is worth ~7 %. The win is fine-tuning at all.** So the expensive 3.7-day generation run
buys the last 7 %, and ordinary corpus text run teacher-forced keeps ~93 % of it. Generation is
optional polish, not the price of entry. (Our spec decode is gated LOSSLESS, so if we ever do generate,
its output is bit-identical to base AR and is legitimate on-policy data at 21.68 tok/s.)

### (d) Sizing, at the measured 47.7 tok/s and 8 KB/token (hidden 4096, bf16)

| corpus | tokens | teacher-forced capture | cached states (bf16 / fp8) |
|---|---|---|---|
| 10K samples x ~700 tok | 7M | **1.7 days** | 56 GB / 28 GB |
| **20K samples** | 14M | **3.4 days** | 112 GB / 56 GB |
| 50K samples | 35M | 8.5 days | 280 GB / 140 GB |
| FastMTP full (389.4K) | 273M | 66 days | 2.2 TB / 1.1 TB |

**20K in ~3.4 days is the target**, cached so the head can be trained for multiple epochs and swept for
hyperparameters without re-capture. Storage stops being a constraint the moment the corpus is this
size — the 4.7 TB figure that looked disqualifying was for the full 389.4K corpus.

### The recipe, extracted in full (arXiv 2509.18362v1)

Position-shared single head, 210.8M params. **3 epochs**, AdamW (β=0.9, 0.95), peak LR **5e-5** cosine
with 0.05 warmup, batch 64, **<1 day on a single H20**. Loss: weighted CE with exponential decay
`α_k = β^(k-1)/Σβ^(j-1)`, **β=0.6**, K=3. Corpus mix 42 % general / 18 % math / 13 % code / 27 %
Chinese; generation at T=0.6, top-k 20, top-p 0.95, max 4096. Per-position acceptance **70/11/2 % →
80/56/36 %** at k=1/2/3.

### The two open questions, in priority order

1. ~~**Is there an autograd stack on aarch64 `sm_110a`?**~~ **ANSWERED — yes, and it is already on this
   box.** `evidence/autograd_probe.log`, run in `ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor`:

   ```
   torch 2.10.0 | cuda build 13.0      device: NVIDIA Thor    capability: (11, 0)
   arch_list: ['sm_110', 'sm_121']
     torch.float32   backward OK  grad_finite=True  grad_norm=0.0266
     torch.bfloat16  backward OK  grad_finite=True  grad_norm=0.0269
     AdamW(0.9,0.95) lr=5e-5 step OK
   free/total GiB: [116.4, 122.8]
   ```

   Not a forward-only smoke test: a 4096x4096 two-layer block at batch 64 produces **finite gradients
   in both fp32 and bf16**, and AdamW steps with the FastMTP recipe's exact hyperparameters. **116.4
   GiB free inside the container** against a ~3 GB head working set. The probe is `tools/autograd_probe.py`
   so this is re-runnable rather than a claim. No hand-written backward kernels are needed; S5's
   training half is an ordinary PyTorch job, and only the *capture* half needs our CUDA engine.
2. **Where does acceptance saturate with corpus size?** FastMTP does not say. Medusa- and EAGLE-class
   heads are commonly trained on ShareGPT-scale data (order 60K), which would put 20K within a factor
   of three of the norm — *this needs verifying, it is recollection, not a checked source.*

## 5. What "done" looks like

The loop stops finding levers when: `LEVERS.md` §4/§5 have no open item with expected value ≥1 %, and
two consecutive research phases produce nothing that survives §1's constraints. At that point the
engine is at its structural limit for this checkpoint and the honest move is to say so in
`FLYWHEEL_STATE.json` and stop, not to keep cycling.


## §7 — CORRECTION: fine-tuning a shipped MTP head is worth **+51 %**, not +5.6 %

I told the operator that Nebius's DeepSeek-MTP result was **+5.6 %** and used it to argue S5 was
probably not worth six days. **That was a misreading and it was wrong by an order of magnitude.**

+5.6 % is LK loss's gain **over a KL-trained baseline**. The gain of fine-tuning over the **shipped**
module is on the model card for `nebius/MTP-DeepSeek-V3-0324`:

| | acceptance |
|---|---|
| shipped DeepSeek-V3 MTP | **3.20** (T=0) / 3.09 (T=1) |
| **fine-tuned, hybrid LK** | **4.83** / 4.68 |
| KL-fine-tuned | 4.79 / 4.43 |

**+51 %.** Recipe, all published: Infinity-Instruct-0625 with **DeepSeek-V3-generated responses**,
hybrid LK loss with adaptive lambda, **eta = 3** (which I had recorded as unpublished), **1 epoch
from pretrained MTP weights**, draft length 6.

### The mechanism — and why it may NOT transfer to us

The literature is explicit about *why* that head had 51 % to recover: DeepSeek-V3's MTP module was
**trained for first-token prediction and reused autoregressively** for later positions, so acceptance
degraded with draft depth. Fine-tuning repairs a pathology.

**Ours is not that head.** DSpark is semi-autoregressive and *block*-trained: a parallel backbone
emits every position of the block at once, a Markov head injects intra-block dependency, a confidence
head predicts per-position acceptance — and the DSpark paper trained it for **10 epochs** on
Open-PerfectBlend. So the pathology Nebius repaired is one our architecture was explicitly designed
to avoid, and +51 % is an **upper** bound for us, not an expectation.

### What our own numbers say about the gap

- tau = **2.89** on the canonical 6-token trivia prompt
- tau = **3.57-4.00** on a 1023-token real-code prompt (measured across four runs)
- the DSpark paper reports **6.11** accepted length on math

So a large part of our apparent gap is **prompt and domain**, not head quality — the canonical gate
prompt is 6 tokens of trivia and is the worst case for a drafter. **Any S5 decision must be made
against a realistic multi-prompt evaluation set, not against 2.89.** Building that set is a
prerequisite, and it is cheap.

## §8 — DFlash: KV injection is the architectural lever we do not have

LMSYS, 2026-06. EAGLE-class drafters pass target features **only at the draft model's input**, so the
signal **fades in deeper drafters**. DFlash instead **injects target features into the KV cache of
every draft layer**, which is what lets acceptance scale with draft depth. Block size 8-16 (vs our
5); 4.86x on Qwen3-8B greedy; a 5-layer DFlash at 16-token blocks beats EAGLE-3 at 8-token blocks
**with lower latency**.

Ablations separate the two mechanisms: diffusion-only gives lower acceptance (3.5-3.8) but still
2.9x on GSM8K through faster drafting; **injection-only gives higher acceptance (4.8) and 2.4x**.

**Open question for the workflow: can KV injection be retrofitted to an already-trained MTP head, or
does it require training from scratch?** If it retrofits, it is a bigger lever than the fine-tune.
