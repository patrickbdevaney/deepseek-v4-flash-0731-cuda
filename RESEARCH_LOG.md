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
| 2026-08-08 | A-F | **operator deep-research dossier** (web-chat, ~24 sources: FastMTP, HyperDFlash, SuffixDecoding, Sequoia, EVICT/EcoSpec/AcceptMoE, REAP, Thor MLPerf) | **4 promoted, 6 already done here, 3 inapplicable** — full triage in §3a | S5 S6 S7 S4 |

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
| ⑤ retrieval drafting (Prompt-Lookup, **SuffixDecoding**) | 2-5× on agentic | **promoted → S6**, upgraded from plain n-gram to a suffix automaton on the dossier's evidence. |
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

## 5. What "done" looks like

The loop stops finding levers when: `LEVERS.md` §4/§5 have no open item with expected value ≥1 %, and
two consecutive research phases produce nothing that survives §1's constraints. At that point the
engine is at its structural limit for this checkpoint and the honest move is to say so in
`FLYWHEEL_STATE.json` and stop, not to keep cycling.
