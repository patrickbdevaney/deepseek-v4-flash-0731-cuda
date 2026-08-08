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

## 3. Ideas seen and rejected before implementation

| idea | source | why rejected here |
|---|---|---|
| FP4/INT4 for MLA + dense weights | general LLM inference practice | moves the byte floor ~1.2x but violates *no additional quantisation*. Needs an explicit decision to relax the constraint — see `LEVERS.md` B5. |
| tree / multi-branch speculation | Medusa, EAGLE-style | measured on this engine already: correct but does not beat linear, depth-dominated. `LEVERS.md` §3. |
| draft refinement (feed proposals back) | standard MTP practice | acceptance **3.00 → 2.08** here: the shipped MTP heads are trained with a noise-token placeholder, so real proposals are off-distribution (F45). |
| grouped **GEMM** (mma) for the MoE instead of the GEMV | "get the GEMM right for the MoE" | **it is already implemented and it loses.** `tc_fp4_grouped_gemm_e8m0` exists and `g_moe_gemv` selects the GEMV over it. ncu on the measured grouping: **GEMV 416 µs vs mma GEMM 783 µs (1.9x slower)**, despite the GEMM reaching 93.9 % occupancy against the GEMV's 53.7 %. At 1.71 rows/expert there is no arithmetic intensity for a tensor core to exploit and the m16 tile wastes ~90 % of its rows; the problem is bandwidth and launch geometry, which is what F64/F65/F70/F72 actually fixed. |
| pad experts 160 → 192 for a "fused router path" | vLLM model-card note that K160 is outside the supported set (16,32,…,512) | **a vLLM kernel artifact, not a hardware or model one, and this engine is not vLLM.** Our MoE grouping is our own code (`k_moe_count`/`k_moe_prefix`/`k_moe_scatter`/`tc_build_tiles`) and has no fixed expert-count assumption anywhere. Measured: routed MoE runs at ~72 % of the achievable roofline and the grouping itself is 2.12 ms of a 132 ms verify, now at the launch-latency floor (F73). Padding to 192 would cost ~20 % more resident memory to fix a path we do not take. |

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
