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
| 2026-08-07 | — | *(session budget exhausted at 200/200 before a research phase could run; F64–F72 came from profiling instead)* | n/a | — |

## 3. Ideas seen and rejected before implementation

| idea | source | why rejected here |
|---|---|---|
| FP4/INT4 for MLA + dense weights | general LLM inference practice | moves the byte floor ~1.2x but violates *no additional quantisation*. Needs an explicit decision to relax the constraint — see `LEVERS.md` B5. |
| tree / multi-branch speculation | Medusa, EAGLE-style | measured on this engine already: correct but does not beat linear, depth-dominated. `LEVERS.md` §3. |
| draft refinement (feed proposals back) | standard MTP practice | acceptance **3.00 → 2.08** here: the shipped MTP heads are trained with a noise-token placeholder, so real proposals are off-distribution (F45). |

## 4. What "done" looks like

The loop stops finding levers when: `LEVERS.md` §4/§5 have no open item with expected value ≥1 %, and
two consecutive research phases produce nothing that survives §1's constraints. At that point the
engine is at its structural limit for this checkpoint and the honest move is to say so in
`FLYWHEEL_STATE.json` and stop, not to keep cycling.
