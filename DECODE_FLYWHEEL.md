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

### Phase D — RESEARCH (only when A, B and C are dry)

Entry condition, strictly: the LEVER QUEUE is empty, the profile shows no phase below 80%, and §4
is exhausted. Research is expensive and, run speculatively, produces surveys instead of answers.

**Research is question-driven, not survey-driven.** Every query must be generated from a *specific
measured residual*, in this form:

> "`<phase>` reads `<bytes>` and achieves `<X>` GB/s against a `<Y>` GB/s working-set ceiling on a
> `<shape>` GEMV. What technique closes a gap of this shape at batch 1?"

Then: implement → gate → measure → back to A. Research output that is not implemented and measured
within the same loop iteration is not an answer; it is a citation.

**Standing caution, learned twice on this project (Findings 29 and 30):** negative capability claims
("Thor has no FP4", "`tcgen05` unsupported") are the least trustworthy class of claim and both of
this project's were wrong. Probe every instruction family — `mma.sync`, `wgmma`, `tcgen05` — before
recording an absence.

---

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
| A1 | Intra-expert activation sparsity | +11-17% | training-free, runtime-only |
| A6 | CUTLASS `reg_reconfig.h` missing `1100` clause | 1.74x on FMHA upstream | two lines, local patch |

### Retired with a measurement (do not re-queue)

m16 B-operand repack (0.98x) · clock/EMC locking · MAXN power mode · expert prefetch/caching/sticky
routing · speculation trees · draft/verify pipelining · 2:4 sparsity · MLA weight absorption ·
FlashMLA port · MoE g-loop pipelining · wave-quantisation clamp (Opt #2) · BF16-native compressor
(Finding 32) · shared-A fp8 GEMV (Finding 41: slower at every M) · **block size > 5 (Finding 43:
identical accept sequence at 5 and 8)** · **draft refinement (Finding 45: acceptance 3.00 -> 2.08 —
the MTP heads are trained with the noise token as placeholder, so real tokens are off-distribution)**.

## 4. Phase-C explorations (structural)

- `tcgen05.mma.kind::mxf4nvf4.block_scale` end-to-end — only alloc/dealloc and TMA are runtime-verified
  so far; a complete MMA has never been run.
- CUDA-graph capture of the whole decode step (Gate G9 re-gate).
- Persistent-kernel / megakernel decode: one launch per token instead of ~600.
- Fusing the entire attention glue chain (`act_quant`/`rmsnorm`/`rope`/KV-write) into layer-scope kernels.
- Cluster/DSMEM (Thor has them; GB10 does not) for cross-block reduction in HC and the router.

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
