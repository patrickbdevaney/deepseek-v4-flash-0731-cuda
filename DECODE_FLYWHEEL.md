# DECODE_FLYWHEEL.md — the self-propelling loop for maximal decode

A mechanical protocol for surfacing and cashing every remaining lever on the
**(model, hardware, speculation)** tuple, run until further speedup is *demonstrably* impossible
rather than merely not-yet-found. Written 2026-08-06 at 10.3 tok/s.

This document is the loop's state. It is meant to be executed, not read once.

---

## 0. The ceiling arithmetic, stated first

Everything below is ranked against these numbers, so they come before the plan.

| quantity | value | source |
|---|---|---|
| `B_tok` (bytes read per token) | **12.26 GB** | `ROOFLINE.md`, engine-measured (not checkpoint-stored) |
| achievable BW, large working set | **240 GB/s** | `tools/bw_probe.cu` on-box |
| achievable BW at our kernel working sets (1.7–50 MB) | **~208 GB/s** | on-box working-set sweep |
| **base AR ceiling** | **19.6 tok/s** (240) / **~17.0** (208) | `B_tok` / BW |
| current base AR | **10.3 tok/s** (126 GB/s, 53%) | `~/opt11.log` |

**Base AR has at most 1.65× left in it, and not one token more.** `B_tok` is fixed: the user's
non-negotiables forbid further quantisation, and every byte lever that does not change the
checkpoint has been enumerated (§3).

So a 35–50 tok/s target decomposes, and only one decomposition is physically available:

```
   35 tok/s  =  17.5 base AR  x  2.00 speculation
   50 tok/s  =  19.0 base AR  x  2.63 speculation
```

The comparison points confirm the shape rather than contradicting it — on this same box,
**Mistral Small 4: 17.5 base → 33.7 with Eagle (1.93×)**; **Qwen 122B-A12B: DFlash 1.70×**.
Neither exceeded ~19 tok/s on base AR either. The 50 tok/s vLLM figure was a different machine.

**Therefore the loop's priority is inverted from where it has been for seven rounds:**

| lever class | current | ceiling | multiplier available |
|---|---|---|---|
| base AR efficiency | 126 GB/s | 208–240 | **1.65×** |
| **speculation** | **1.00×** | **2.0–2.6×** | **2.0–2.6×** |

Speculation is at **1.00×**, i.e. *zero percent harvested*, and it is the larger of the two.
Kernel work continues, but it is no longer the headline.

---

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

## 3. LEVER QUEUE (live — reorder as measurements land)

### Speculation — the 2.0–2.6× (now the headline)

| # | lever | expected | state |
|---|---|---|---|
| **S1** | **Resolve the half2 / acceptance either-or (Finding 31)** | 1.00× → ~1.6× | half2 perturbs target logits below the argmax margin; greedy output is unchanged but acceptance is an *exact* match, so draft/target agreement breaks. `MOE_MMA=1` restores 3.12/5 acceptance at −0.15 tok/s. **Blocking everything else in speculation.** |
| **S2** | Cut verify cost `c_v` 2.6 → ~1.9 | 1.6× → ~2.0× | corrected `E_frac`: the real expert union at K=5 is ~25, not 30 |
| **S3** | REAP-repair fine-tune of the 3 MTP blocks + markov head | acceptance 3.12 → 4.0–4.5 | the principled fix for S1 as well — retrains the head against what the target *actually* computes |
| **S4** | Reduced draft vocabulary + `d2t` | draft `lm_head` is 1059 MB | lossless by construction: verification catches draft errors |
| **S5** | AcceptMoE commitment-weighted expert set at verify | 1.29× measured at batch 1 | −0.27 pp accuracy |

### Base AR — the 1.65×

| # | lever | expected | state |
|---|---|---|---|
| A1 | Intra-expert activation sparsity | **+11–17%** | training-free, runtime-only, no artifact change; derate for block-32 granularity |
| A2 | `tcgen05` MXFP4 with output columns in M | removes dequant entirely | native block-scaled `mma` on the format the weights already ship in |
| A3 | Kernel fusion to raise per-kernel working sets | small kernels run at 23–59% | Thor has 228 KB smem/SM, 2.28× GB10's |
| A4 | half2 rewrite of `gemm_fp32`, then re-try `COMP_BF16=1` | ~620 MB/step | precondition now done (Opt #10) |
| A5 | HC params F32 → FP16 | 68 MB, +0.6% | trivial |
| A6 | CUTLASS `reg_reconfig.h` missing `1100` clause | 1.74× on FMHA upstream | two lines, local patch |

### Retired with a measurement (do not re-queue)

m16 B-operand repack (0.98×) · clock/EMC locking (106.8 vs 105.2) · MAXN power mode (EMC 4266 in
all modes) · expert prefetch/caching/sticky routing · speculation trees · draft/verify pipelining ·
2:4 sparsity · MLA weight absorption · FlashMLA port · MoE g-loop pipelining (101.7 → 102.9) ·
wave-quantisation clamp (Opt #2, 214.6 → 186.3) · BF16-native compressor (100.2 → 103.5).

---

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
