# DECODE_FLYWHEEL.md — the self-propelling loop for maximal decode

A mechanical protocol for surfacing and cashing every remaining lever on the
**(model, hardware, speculation)** tuple, run until further speedup is *demonstrably* impossible
rather than merely not-yet-found. Written 2026-08-06 at 10.3 tok/s.

This document is the loop's state. It is meant to be executed, not read once.

---

## 0. The ceiling arithmetic, stated first  (REWRITTEN — the old numbers were against the wrong roofline)

| quantity | value | source |
|---|---|---|
| `B_tok` (bytes read per token) | **12.26 GB** | `ROOFLINE.md`, engine-measured |
| achievable BW, **strided, out of the allocator the weights live in** | **~233 GB/s** | `tools/alloc_probe`, 16 GiB (Finding 44) |
| — the same, before Finding 44 moved them off mapped-host | ~216 GB/s | same probe |
| — `cudaMalloc`, which `bw_probe`'s 240 measured and the model never used | 234 GB/s | same probe |
| base AR floor | **52.6 ms = 19.0 tok/s** | `B_tok`/BW |
| base AR now | 92.8 ms (10.8) / **81.3 ms (12.3) with the CUDA graph** | `~/managed.log` |
| verify K=5 floor (dense 9.01 GB + union~25 x 12.58 MB x 43) | **96.7 ms** | measured expert size |
| verify K=5 now | 168.8 ms | `~/managed.log` |
| draft floor / now | 9.9 / 27.3 ms | specprof |
| **cycle floor / now** | **106.6 / 201.8 ms** | |
| acceptance | **3.00 tokens/verify, flat in block size** | Finding 43 |

**`c_v` measured 1.82 against a byte-model floor of 1.84: the verify's K-scaling is already optimal.**
The entire remaining gap is a *uniform* 1.89x present equally at K=1 and K=5. There is no
speculation-specific work left — **every ms of base-AR efficiency now converts proportionally into
the spec number**, and the two priorities have merged.

At the floor with acceptance 3.00: **28.1 tok/s**. What the target band would require:

| target | tokens/verify needed | verdict |
|---|---|---|
| 28 | 3.00 (measured) | reachable by kernel work alone |
| 35 | 3.70 | needs a better draft — S3 fine-tune |
| 50 | 5.30 | **exceeds BLK=5's maximum of 5; not reachable at any kernel speed** |

Bigger blocks do not buy acceptance (Finding 43: identical accept sequence at BLK=5 and BLK=8), and
draft refinement makes it worse (Finding 45: 3.00 -> 2.08). So beyond ~28 tok/s the levers are a
REAP-repair fine-tune of the MTP heads (a training task) or quantising MLA (forbidden by the
project's non-negotiables). **Both are user decisions, not defaults.**

### Where the remaining 95 ms of cycle is

Composition of the 1.89x, from the K=5 dprof and the shape bench:

| | measured | byte floor | the difference is |
|---|---|---|---|
| `moe:w1w3`+`w2` | 76.5 ms | 58.0 | kernel: 177 GB/s vs 233 |
| `cattn:ogroup` | 24.3 | 11.8 | ~half glue (rope, act_quant), half kernel |
| `cattn:q_proj` | 17.8 | 7.0 | **12.4 ms is glue** — 7 small kernels x 41 layers |
| `moe:shared` | 11.2 | 4.6 | act_quant/swiglu/accum chain |
| `cattn:indexer` | 10.7 | ~1 | almost entirely small-kernel latency (21 layers) |
| `cattn:compress` | 8.5 | ~1 | ~8 launches per emit per layer |
| `lm_head` | 6.9 | 4.5 | |

**Glue, not GEMM bandwidth, is the majority of it** — which the CUDA-graph re-gate confirmed
independently (1.14x on base AR, ~11.5 ms out of ~600 launches). The two general attacks are
graph-capturing the verify path (needs device-pos variants, as the decode path already has) and
fusing the per-layer glue chains. Byte reduction on the MoE is not the lever it was ranked as.

## 1. The loop

Four phases. Each has an entry condition, a stopping rule, and an artifact. Run A until its
stopping rule fires, then B, then C, then D, then back to A with the refilled queue.

### Phase A — EXPLOIT (cash known levers; needs no new information)

1. Take the top entry of the **LEVER QUEUE** (§3), ranked by `expected_ms_saved / runs_needed`.
2. Gate it *before* measuring it — unit gate on the exact kernel and the exact M that decode uses.
   (Finding 35: the most expensive kernel in attention had no gate, and a comment claimed one.)
3. Measure with **one change per run**, canonical prompt `0,671,6102,294,8760,344`, via
   `scripts/run_model.sh`. Record the band, not the point — run-to-run noise here is ±1 ms.
4. Adopt or reject **on the number**, and write the number into `OPTIMIZATION_LOG.md` either way.
   A rejected lever with a measurement is permanently retired; a rejected lever without one comes
   back and costs another run.
5. Requeue: if the measured gain deviates from the predicted gain by >2×, the *model* that ranked
   it is wrong — fix the model (§2) before taking the next lever.

**Stopping rule:** three consecutive levers each measuring < 0.5% end-to-end. Go to B.

### Phase B — EXHAUST (regenerate the queue from measurement, not intuition)

1. `DSV4_KSWEEP=1 DSV4_DPROF=1` full profile. Verify children ≤ parent (the report now says so
   itself; it did not, and produced a profile with `moe:w1w3` at 216% of `MoE`).
2. For every phase: compute **achieved GB/s = phase_bytes / phase_ms**, and the **ceiling implied
   by that phase's working-set size** from the on-box curve (1 MB → 54, 32 MB → 208, ≥512 MB → 240).
3. Rank residuals by `phase_ms × (1 − achieved/ceiling)`. **Never by efficiency deficit alone** —
   Opt #2 was queued at #2 on a 19% efficiency for a shape carrying 1.6% of the bytes, and was
   reverted as a regression.
4. Refill the LEVER QUEUE from the top residuals. Go to A.

**Stopping rule for base AR:** no phase below 80% of its working-set-implied ceiling. At that point
base AR is finished at ~17 tok/s and only C and D remain.

### Phase C — EXPLORE (structural changes; known-unknowns, no literature needed)

Entry: A and B are both dry. These are larger, riskier, and each needs its own gate. §4 lists them.
Each exploration ends in either a measured adopt/reject or a written statement of why it is
impossible on `sm_110a` — **never in silence**, because a silently-dropped option is
indistinguishable from an option that was never seen.

### Phase D — RESEARCH (question-driven, and it has now been run once)

**Entry condition — REVISED.** v1 said "enter when the queue is empty". That is wrong, and Finding
49 is why: the queue was empty *because the loop's model of its own bottleneck had stopped
generating levers*, not because the levers were gone. The literature had moved past our framing
while we were still optimising inside it. The real trigger is:

> **Enter Phase D when three consecutive levers measure under 0.5% AND the last two Phase-B
> re-rankings produced the same top entry.** The second clause is the important one: a queue that
> keeps re-deriving the same answer is a queue whose *model* is exhausted, which is exactly when
> outside information is worth more than another measurement.

**Protocol.**

1. **Write the prompt against the measured state, not the topic.** `RESEARCH_PROMPT_v2.md` is the
   template. It must carry: the current profile, the ncu diagnosis, the measured roofline for *this*
   allocator, the full retired-with-a-measurement list, and the standing instrument caveats. A
   prompt that describes the model and hardware but not the *residual* gets a survey back.
2. **Every question names its falsifying measurement.** "What technique closes a gap of this shape"
   is answerable; "how do we go faster" is not.
3. **Ask what we are not asking.** §Q6 of the prompt. This is the question that found Finding 49.
4. **Run it against primary sources.** The arXiv API (`export.arxiv.org/api/query`) with
   `search_query=abs:"..."+AND+abs:"..."` sorted by `submittedDate` beats a web search: it is
   complete, dated, and not SEO-shaped. Fetch the abstracts of the top hits directly — the abstract
   usually states losslessness and the headline number, which is enough to rank.
5. **Convert to levers with the §4 output contract**, then re-enter Phase A. **Research output that
   is not implemented and measured in the same cycle is a citation, not a lever.**
6. **Record the negative space too.** If the literature has nothing for a residual, that is a
   finding — it means the residual is either novel or physical.

**Cycle artifact.** Each Phase D produces a dated `RESEARCH_PROMPT_vN.md` grounded in that cycle's
measurements, a candidate list with falsification tests, and additions to the retired list. The
prompts are kept, not overwritten: the diff between vN and vN+1 is the record of how the project's
understanding of its own bottleneck changed, and v1 -> v2 already shows one full inversion
(bandwidth -> latency).

**Run 1 (2026-08-07) — what it returned.** See Finding 49. The residual "speculation gains nothing
from batching because the MoE expert union grows with K" turned out to be an actively-worked 2026
problem with a name (*expert scattering*) and a family of solutions, several training-free and
lossless: EVICT (2605.00342), EcoSpec (2607.12696), MoE-Spec (2602.16052), EdgeXpert (2608.05303),
AcceptMoE (2608.02989), SP-MoE (2510.10302). The loop had spent seven rounds treating the union as
a constant of nature. It is a decision variable.

---

## 1b. The continuous loop, as a standing procedure

```
  A EXPLOIT ──► B EXHAUST ──► A ... ──► C EXPLORE ──► A ...
      ▲                                                │
      │            three levers < 0.5%  AND            │
      └──────── D RESEARCH ◄── same top entry twice ◄───┘
```

- **A** cashes queued levers. Stop after three consecutive < 0.5%.
- **B** regenerates the queue from a fresh profile. Never from intuition. Rank by
  `phase_ms x (1 - achieved/ceiling)`, then apply the correction rules in §2.
- **C** takes the structural swings. Each ends in a measured adopt/reject **or** a written statement
  of why it is impossible — never silence.
- **D** brings in outside information, on the trigger above, with the protocol above.
- **Every phase writes to `LOOP_LOG.md` whether it succeeded or not.** The retired-with-a-measurement
  list is the most valuable artifact the loop produces: it is what stops the next cycle from paying
  twice for the same negative result.

**Two standing instrument caveats, both paid for:**

1. **Microbenchmarks overstate end-to-end value by 2-4x** (Finding 47). `gemm_bench` relaunches a
   kernel on rotating weights so consecutive launches overlap; the engine serialises every kernel
   behind a data dependency, exposing its tail wave. Valid for *ranking two kernels*, invalid for
   *predicting end-to-end gain*. Always confirm in situ before writing a number in the log.
2. **A gate that allocates its own inputs cannot test alignment** (Finding 41). `cudaMalloc` returns
   256-byte-aligned memory; every weight in this engine is a 4-byte-aligned pointer into a mapped
   file. Gates must reproduce the real alignment, and now do (`B+0` and `B+4`).

## 2. The ranking model (and how it has been wrong)

Rank by **`bytes × (1 − efficiency)`**, then correct with these three rules, each of which was paid
for with a wrong prediction:

1. **Byte reduction only pays on a bandwidth-bound kernel** (Finding 32). On a compute/issue-bound
   kernel it is neutral at best, and negative when the narrower format needs conversion. Establish
   which, *then* narrow. Cost of learning this: the BF16 compressor, ranked #1 at +5%, measured −3%.
2. **Do not extrapolate a per-layer cost from an unrepresentative subset** (Finding 35/36). The
   `attn:*` sub-marks covered 2 of 43 layers — the two pure-sliding ones — and overstated `ogroup`
   by 5×. Instrument the path that actually runs.
3. **Predicted ≠ measured by >2× means the model is broken, not the lever.** Stop and fix the model.

---

## 3. LEVER QUEUE (live — reordered after Findings 43-45)

Base AR and speculation are now the SAME queue: `c_v` is at its byte floor, so anything that speeds
the M=1 step speeds the verify by the same factor.

| # | lever | expected | state |
|---|---|---|---|
| G1 | **Make the CUDA graph the default for base AR** | 1.14x, measured | gated PASS, tokens identical; currently behind `GRAPH=1` |
| G2 | **Graph-capture the verify path** | ~12% of 174 ms | needs device-pos `cblock_verify_step_dp`, as the decode path already has |
| F1 | **Fuse the attention glue chain** (act_quant/rmsnorm/rope) | ~12 ms of `cattn:q_proj` at K=5 | 7 small kernels x 41 layers; the single clearest glue target |
| F2 | Fuse the compressor emit chain | ~7 ms | ~8 launches per emit per layer, 21 layers |
| F3 | Fuse the indexer chain | ~9 ms | `cattn:indexer` is 10.7 ms for ~1 ms of bytes |
| M1 | MoE GEMV 177 -> 210+ GB/s | ~10 ms | bench hits 233 hot; in situ 177 — cold/TLB or issue rate, not yet separated |
| S4 | Reduced draft vocabulary + `d2t` | ~5 ms | draft `lm_head` is 1.06 GB; needs a principled token subset, not an invented one |
| R1 | Re-fit `adaptK` across a multi-prompt within-run A/B | validates the shipped 1.5 | blocked on `tools/encode_prompt.cpp` (unbuilt) + a prompt index in `DSV4_BLKSWEEP` |

### Retired with a measurement (do not re-queue)

m16 B-operand repack (0.98x) · clock/EMC locking · MAXN power mode · expert prefetch/caching/sticky
routing · speculation trees · draft/verify pipelining · 2:4 sparsity · MLA weight absorption ·
FlashMLA port · MoE g-loop pipelining · wave-quantisation clamp (Opt #2) · BF16-native compressor
(Finding 32) · shared-A fp8 GEMV (Finding 41: slower at every M) · **block size > 5 (Finding 43:
identical accept sequence at 5 and 8)** · **draft refinement (Finding 45: acceptance 3.00 -> 2.08 —
the MTP heads are trained with the noise token as placeholder, so real tokens are off-distribution)**
· **A6, the CUTLASS `reg_reconfig.h` `1100` patch (Finding 50: retired on code evidence, not a
number — no engine TU includes `cutlass_moe.h`, the only callers of `cutlass_nvfp4_gemm` are that
file's own self-tests, so `build/cutlass_moe.o` is linked and never launched; the patch would change
an object that never runs)**.

**Moved out of Phase A rather than retired: A1, intra-expert activation sparsity** (Finding 50). It
is not lossless, `research/MOE_DECODE.md:98` already priced MXFP4's 32-element blocks as blocking
sub-block skips (only `w3` of the three expert matrices is skippable — `w2`'s neuron axis is its
K axis, and `w1` must be read in full to compute the gate that selects neurons), and §2 rule 1 says
byte reduction does not pay on a phase Finding 47 measured latency-bound. It belongs in §4 as a
structural exploration with a user decision attached, not at the head of the exploit queue.

## 4. Phase-C explorations (structural)

- `tcgen05.mma.kind::mxf4nvf4.block_scale` end-to-end — only alloc/dealloc and TMA are runtime-verified
  so far; a complete MMA has never been run.
- CUDA-graph capture of the whole decode step (Gate G9 re-gate).
- Persistent-kernel / megakernel decode: one launch per token instead of ~600.
- Fusing the entire attention glue chain (`act_quant`/`rmsnorm`/`rope`/KV-write) into layer-scope kernels.
- Cluster/DSMEM (Thor has them; GB10 does not) for cross-block reduction in HC and the router.
- **Intra-expert activation sparsity (A1)**, if and only if the user accepts a lossy change: gate on
  w1's own output, skip w3 rows for inactive neurons (w2 is unskippable at MXFP4 block-32
  granularity), and *first* answer the prior question — whether removing bytes moves a phase that
  ncu says is latency-bound. Needs an accuracy harness this repo does not have.

---

## 5. Invariants (inherited, non-negotiable, and each one has a scar)

- No REAP pruning work; no additional quantisation; no invented model constants.
- **Correctness gates before speed gates.** One change per measurement. Report bands, not points.
- **If a gate fails, stop and report before building on top of it.**
- Bench **COLD** — warm L2 cost this project two wrong diagnoses.
- `ncu`'s "Memory Throughput %" is **L2** throughput on Thor.
- Never benchmark with other GPU work resident; never compile a heavy TU during a full-model load.
- All full-model launches through `scripts/run_model.sh` (flock + headroom check + detached).
- **Never write up a config that has not been run** (Finding 33).
