# Compression playbook: getting a frontier MoE resident on Thor with acceptable decode

Companion to `MODEL_SURVEY_APPENDIX.md` §4c, which establishes *that* residency is the whole game.
This is *how*, assuming rentable cloud GPUs and a budget. Same provenance convention: **measured**
here, *searched* with a citation, (analysis) derived.

## 0. VERDICT: the prune/quantise pipeline is SHELVED (decided 2026-08-18)

**Do the kernel work instead.** The pipeline is dominated on every axis, and the reason is a
logical error worth stating plainly because an earlier version of this document made it.

### Byte reduction only pays in proportion to how bandwidth-bound you already are

**measured**: MLA GEMVs run at **115-195 GB/s against 240 GB/s achievable**, and
`dspark-decode-gap-research` concludes the decode gap is **"efficiency not algo"**.

**We are therefore not bandwidth-bound.** In that regime, handing the kernel 18% fewer bytes to
move buys close to nothing — the bottleneck is occupancy, launch overhead or serialisation, not the
bytes. **The "~10% from 3.5-bit" figure earlier in this document is optimistic; the honest estimate
is 0-10%, and which end cannot be known until the kernel is fixed.**

That reframes the question. It is not "kernel or pipeline". **The pipeline cannot even be
evaluated until the kernel work is done**, because a bandwidth optimisation is unmeasurable on a
machine that is not bandwidth-bound.

### Risk survey

| change | decode gain | quality risk | operational risk | invalidates |
|---|---|---|---|---|
| **Kernel (MLA GEMV)** | up to **~2x** | **none** — no model change | bit-exactness gates already catch regressions | nothing |
| **Draft head fine-tune** | ~30% (tau 2.689 -> ~3.5) | **none by construction** — speculative verify is exact | reversible, keep the old head | `PERF.md` |
| **3.5-bit requant** | **0-10%** | sub-4-bit degradation; **CoT loops**; unvalidated at 3.5 on this model | **new GEMV kernels, new gates, collides with the bit-exactness invariant** (`wiki/dense-mla-gemv.md`); compounding error unless the REAP is also redone from the FP8 parent | **the entire accuracy battery** |
| **Deeper REAP (50%)** | **0%** — top-k unchanged | **CoT degeneration (observed first-hand by the operator on other REAPs)**, world knowledge, tool-call formatting, long-context coherence | router recalibration required | **the entire accuracy battery** |

Additional risks specific to the prune/quant path, enumerated rather than assumed away:

* **tool-call and structured-output formatting degrades before prose does** — an agentic model can
  lose its function-calling reliability while still reading fine
* **long-context coherence** can break if experts specialise positionally
* **router quantisation flips routing**, producing erratic expert selection — a direct mechanism
  for incoherence (routers are 0.070 GiB and must stay high-precision)
* **compounding error** means a clean job requires redoing the REAP from the FP8 parent rather than
  requantising our checkpoint — which drags the high-risk pruning step back in

### Verdict

**Not worth it.** Highest engineering cost, highest quality risk, forces a full eval re-run, and
delivers the *smallest* gain — a gain that may be zero until the kernel work that would let us
measure it is done.

1. **Kernel (MLA GEMV efficiency)** — largest prize, no model risk, invalidates nothing.
2. **Draft head fine-tune** — accuracy-neutral by construction, reversible.
3. Stop.

### Re-entry condition

Revisit compression only if **all three** hold:

1. kernel work plateaus well short of the roofline, **and**
2. decode has become genuinely bandwidth-bound, **and**
3. the eval shows 37.5% REAP cost little, making deeper compression defensible.

Until then it is the wrong tool aimed at a bottleneck we do not have.

---

## 0a. The adopted path, in detail (decided 2026-08-18)

Everything below this section is analysis. **This section is the plan.**

### Do not prune further. Stay at K160 (37.5%).

A 50% REAP would free ~17 GiB — **headroom we do not need.** Thor has 117 GiB, the model is
100.4 GiB, so **~16.6 GiB is already free**, and FP8 KV *frees* memory rather than consuming it.
The only thing wanting space is a larger draft head, and the MTP heads are 6.529 GiB today, so they
could be doubled inside the existing headroom.

Pruning deeper therefore trades capability for space we already have. Worse, it risks the two
things we cannot cheaply measure: **world knowledge** (Cerebras' 50% validation is on *coding*
benchmarks — 97.6% non-agentic, 96.7% agentic SWE-Bench on Qwen3-480B-Coder — and says nothing
about GPQA/MMLU-Pro-style breadth, which is exactly what expert count buys) and **chain-of-thought
coherence** (below). Every lever in the adopted path leaves expert count untouched.

### Ranked levers, by measured size of the prize

| # | lever | prize | risk | invalidates |
|---|---|---|---|---|
| 1 | **MLA GEMV efficiency** | **~2x** — decode runs at ~25% of the bandwidth roofline, and MLA GEMVs at 115-195 GB/s against 240 achievable while MLA is **41% of B_tok** | pure kernel work | nothing |
| 2 | **FP8 KV** | attacks the same 41% of bytes; frees GB | low, reversible | `PERF.md` only |
| 3 | **Draft head fine-tune** | tau 2.689 -> higher, a pure multiplier | low | `PERF.md` only — speculative verify is *exact*, so this is **accuracy-neutral by construction** |
| 4 | **Bit-width change** | ~10% overall (see below) | needs a new `sm_110a` kernel | **the whole accuracy battery** |

**The kernel efficiency gap is roughly ten times the bit-width lever**, needs no model change, and
does not invalidate a running evaluation. It goes first.

### If bit-width is attempted anyway: 3.5-bit, not NVFP4

At A13B active:

| format | bits | active bytes | roofline | HW unpack |
|---|---:|---:|---:|:---:|
| NVFP4 | 4.5 | 7.31 GB | 32.8 tok/s | yes (FP4x2) |
| MXFP4 (today) | 4.25 | 6.91 GB | 34.7 | yes |
| 3.5-bit trellis | 3.5 | 5.69 GB | **42.2** | **no** |

**NVFP4 is 4.5 bits — worse than the MXFP4 we already run on both size and speed.** Its only case
is kernel quality for the MLA GEMV problem, which lever 1 addresses directly and more cheaply.
3.5-bit is ~28% faster than NVFP4 at the roofline and 22% smaller.

The risk in 3.5-bit is **not FLOPs** — decode GEMV needs ~480 GFLOP/s at 35 tok/s against 100+
TFLOPS, so dequant could cost 100x more ALU without becoming compute-bound. The risk is
**instruction-level serialisation**, since trellis decode is sequential within a block by
construction. QTIP claims to have solved exactly that (it beats QuIP#/AQLM on *speed*), and
`0xSero/deepseek-v4-flash-0731-spark` is existence proof that 3-bit trellis works on this model.
**Prototype the kernel before committing.**

### CoT degeneration: the failure mode accuracy cannot see

A variant that is 95% as accurate but spirals 20% of the time is **useless as an agent** while
looking fine in a results table. Compression damage to reasoning coherence must be tested for
directly.

* **Truncation rate and `mean tok` are leading indicators, and this repo already records both.**
  Damage shows there *before* accuracy moves: traces lengthen and terminate less often. LCB at
  **59.3% truncated** is this model's canary — a compression step that hurt coherence would move it
  immediately.
* **KL divergence has a structural blind spot here.** §5 recommends a KL sweep for ranking
  configurations, and it is right for that — but **KL is teacher-forced and cannot detect
  degeneration loops**, which are an autoregressive-feedback phenomenon that only appears in
  free-running generation. KL will rank two configs equivalent while one of them loops.
* **Fix, and it is cheap:** alongside the KL sweep, run a few dozen **free-running** generations and
  measure the completion-length distribution and n-gram repetition rate. Minutes, not a battery.
* **Protect the routers.** Routers are tiny but decisive — quantise them hard and routing flips,
  giving erratic expert selection, a direct mechanism for incoherence. The inventory shows router
  at 0.070 GiB and norms at 0.001 GiB, already high precision. **Any new pipeline must preserve
  that**; a naive "quantise everything to N bits" script gets this wrong for free.

### Comparison protocol for any compressed variant

Diff against baseline on **all** of: accuracy, **truncation rate**, **mean tokens-to-answer**, KL /
top-1 agreement (teacher-forced), and free-running repetition rate. A variant that holds accuracy
while `mean tok` rises 20% has damaged the model in the way that matters for agentic work.

### Sequencing

The eval currently running is the **baseline** for every comparison above, and it owns the GPU —
concurrent kernel work would contend for the same 240 GB/s it is measuring, corrupting both. Wait
for `ALL FORCING COMPLETE`. Desktop-side work (the layer-wise pipeline, the KL and free-running
harnesses, the 304 GB FP8 parent download) can proceed in parallel; none of it touches Thor.

## 1. The target, stated exactly

Budget **~107 GB for weights** (117 GiB available **measured**, ~10 GiB reserved for KV,
activations and runtime).

| model | params | bits/param needed | vs NVFP4 |
|---|---:|---:|---:|
| GLM-5.3 | 744B | **1.15** | 3.5x below |
| DeepSeek V4 Pro 0813 | 1.6T | **0.535** | 7.5x below |

**Decode is already solved for both; only footprint blocks residency** (analysis). Active params
are 40B and 49B, so at 2-2.3 bits that is 11.5 and 13.1 GB/token -> **20.9 and 18.3 tok/s at the
240 GB/s roofline**, ~13-26 tok/s after the ~25-50% bandwidth efficiency this repo measures and
DFlash's tau ~ 2.5. Neither model has a decode problem.

**This determines where to spend.** Expert pruning cuts *total* params while leaving top-k routing
and therefore *active* params untouched — it attacks the binding constraint and costs nothing in
decode. Quantization cuts both. Reducing top-k cuts only active, so it is a decode lever held in
reserve for overshoot.

## 2. Which corners are reachable

**GLM-5.3 (744B-A40B)**

| prune | remaining | bits needed | verdict |
|---:|---:|---:|---|
| 34% (shipping today) | 504B | 1.70 | below any published method's floor |
| **50% (REAP-validated)** | **372B** | **2.30** | **QTIP / AQLM territory — the one defensible corner** |
| 65% | 260B | 3.30 | comfortable bits, unvalidated prune rate |
| 71% | 214B | 4.00 (NVFP4) | keeps the hardware unpack path, unvalidated prune rate |

**DeepSeek V4 Pro (1.6T-A49B)**

| prune | remaining | bits needed | verdict |
|---:|---:|---:|---|
| 50% | 800B | 1.07 | not viable |
| **75%** | **400B** | **2.14** | arithmetic closes, but **1.5x beyond validated** |

Only GLM-5.3 at REAP-50% + ~2.3 bits has *both* components at published limits rather than past
them. DeepSeek V4 Pro is a research bet, not an engineering plan.

## 3. Method inventory

### Structured pruning — the footprint lever
1. **REAP** one-shot, no retraining, *searched* as validated to 50% (97.6% coding / 96.7% agentic
   SWE-Bench retention on Qwen3-480B-Coder).
2. **Router-gate KL distillation.** GLM-5.2's REAP freezes the network and trains only the 75 router
   gate matrices — **~0.016% of parameters** — to KL-match the unpruned teacher. Cheapest recovery
   available and already proven at scale.
3. **Non-uniform per-layer rates.** Prune harder where router entropy indicates redundancy. The
   `0xSero/GLM-5.2-REAP-NU176-526B` checkpoint name suggests this already ships.
4. **Expert merging rather than deletion** — ConMoE (prototype reassignment), MAESTRO. Preserves
   more capacity per parameter removed than dropping does.
5. **Workload-matched calibration.** Cerebras published an agentic-reasoning calibration mix
   specifically; generic-text calibration on an agentic model leaves quality unclaimed.

### Quantization — cuts footprint and bytes/token together
1. **NVFP4 / MXFP4** — the only formats with hardware unpack on sm_110a (`dspark-decode-gap-research`
   records FP4x2 unpack yes, tcgen05 no). The fast path, and the reason 4-bit corners are worth
   preferring where the prune rate allows.
2. **QTIP** (trellis + incoherence processing) — *searched* as current SOTA, outperforming QuIP#,
   AQLM and GPTVQ, with **minimal degradation down to 2-3 bits** on zero-shot tasks. Designed for
   inference *speed* as well as ratio, which is the direct answer to the ALU-cost objection against
   codebook VQ.
3. **AQLM / QuIP# / VPTQ** — the vector-quantization family, Pareto-optimal in the sub-3-bit range.
4. **Sensitivity-mixed precision.** Attention, routers and shared experts at 4-bit; routed experts
   at 2-bit. Unusually favourable for MoE because routed experts are the overwhelming bulk of the
   parameters and individually the least critical.
5. **QAT under teacher supervision** — see §4; this is what the budget actually buys.

### Low-rank correction
Quantize aggressively, then add a low-rank residual (LQ-LoRA style). Very cheap in parameters and
recovers a disproportionate share of the loss.

### Architectural surgery
* **top-k reduction (6 -> 4)** — cuts active params, so a *decode* lever, useful if a config
  overshoots on footprint and undershoots on speed.
* Layer/depth pruning; attention-head pruning.
* Retrofitting linear attention (GDN / KDA) to shrink KV.

### Enlarging the budget itself
**FP8 KV instead of FP32 directly enlarges the weight budget.** Non-trivial here given MLA is 41%
of B_tok on the incumbent.

## 4. What cloud GPUs actually unlock

Every published REAP result is **one-shot or router-gate-only**. Nobody in the released artifacts
has done serious distillation recovery after aggressive pruning. **That gap is what a budget
closes**, and it is the only mechanism by which 65-75% pruning becomes viable when one-shot tops
out at 50%.

**Keep it affordable by pre-generating top-k teacher logits offline** rather than co-hosting a
744B-1.6T teacher beside the student. That decouples teacher inference — embarrassingly parallel,
rentable in bursts — from the training loop, and the logit generation is the line item to cost out
first.

Order of magnitude for a 372B-A40B student (analysis): distillation FLOPs ~ 6 x 40B x tokens, so
~10B tokens is ~2.4e21 FLOPs ~ 1,700 H100-hours, i.e. low thousands of dollars; 100B tokens is tens
of thousands. Rough, but it says recovery training is affordable and teacher-logit generation
dominates the bill.

**Ordering is not arbitrary: prune -> recover -> quantize (QAT) -> recover.** Quantization noise
corrupts the activation statistics that pruning saliency estimation depends on.

## 5. Evaluating without the wall-time trap

A full battery on Thor is days (`EVAL_BUDGET_PROTOCOL.md` — 85.2 h of base decode, 64% of it spent
on traces that hit the cap). **Do not use it as the search metric.**

1. **Sweep configurations on cloud GPUs using KL divergence to the unpruned teacher** over held-out
   agentic traces. Cheap, dense, no answer extraction, no truncation problem, and it correlates
   with the thing being protected.
2. **Run the full Thor battery on the two or three finalists only** — which is exactly what
   `tools/eval_suite.py`, `eval_extend.py` and `eval_force.py` exist for.
3. **Calibrate on your own agentic traces.** The Plan-B trace-generation work is not only draft-head
   feedstock; it is the calibration corpus that makes prune and quant minimally lossy *on this
   workload* rather than on generic text.

## 5b. The four trillion-scale candidates, at maximum compression effort

Surveyed 2026-08-18 for Qwen3.8 2.4T-A96B, DeepSeek V4 Pro 0813, Kimi K3 and GLM-5.3, assuming
cloud rental for the compression work.

### Two independent constraints, and different levers fix each

**Footprint** decides residency; **active params** decide speed. Expert pruning leaves top-k
unchanged, so it does **not** reduce active params.

| lever | cuts footprint | cuts active |
|---|:---:|:---:|
| inter-expert pruning (REAP, ConMoE, attribution-guided) | yes | **no** |
| intra-expert slimming (SlimMoE, FlexMoE nested, MoE-SVD / D2-MoE / TD-MoE) | yes | **yes** |
| quantisation | yes | yes |
| top-k reduction | **no** | yes |

**The PTQ floor is ~2 bits** (*searched*). Sub-2-bit PTQ exists — TWLA at 1.58-bit weights, PT2-LLM
ternary, PTQ1.61 — but the literature reports **significant degradation below 2 bits, "a
performance gap of almost half"**. BitNet-style 1.58-bit requires QAT from scratch and does not
scale past ~3B. Treat 2 bits as the floor for "loses virtually nothing".

### The arithmetic against Thor's ~107 GB

| model | total @2-bit | prune needed | active @2-bit | roofline |
|---|---:|---:|---:|---:|
| **GLM-5.3** 744B-A40B | 186 GB | **42%** (inside validated) | 10.0 GB | **24 tok/s** |
| **DSV4 Pro** 1.6T-A49B | 400 GB | **73%** | 12.25 GB | **19.6 tok/s** |
| **Qwen3.8** 2.4T-A96B | 600 GB | **82%** | 24 GB | **10 tok/s** |
| **Kimi K3** 2.78T-A104B | 695 GB | **85%** | 26 GB | **9.2 tok/s** |

**This splits them cleanly.** GLM-5.3 and DSV4 Pro are *footprint-limited only* — active params are
already acceptable, so expert pruning plus 2-bit quantisation suffices with no architectural
surgery. **Qwen3.8 2.4T and Kimi K3 fail the ACTIVE test even with infinite pruning**: ~10 tok/s
roofline is 2.5-5 tok/s at real efficiency, and speculation is fanout-damped. Fixing them requires
cutting active params — top-k reduction (K3's top-16 -> ~top-6) or intra-expert slimming — a fourth
compounding lossy transform that changes the computation per token, not just the pool.

### The maximal stack, in the order the literature requires

1. **Progressive, multi-stage, never one-shot.** SlimMoE is explicit that removing a large fraction
   at once "can result in substantial performance degradation that may hinder distillation
   effectiveness". At 73-85% this is the difference between working and not.
2. **Inter-expert pruning** — REAP router-weighted, attribution-guided coverage-maximised, or
   ConMoE/LightMoE *merging* rather than deletion (merging preserves more capacity per parameter).
3. **Intra-expert slimming** where active must fall — SlimMoE, FlexMoE nested pruning, or tensor
   decomposition (MoE-SVD, D2-MoE, TD-MoE).
4. **Router calibration — mandatory, not optional.** "Is Retraining-Free Enough?" answers no.
   GLM-5.2's REAP already does the cheap version at 0.016% of parameters.
5. **Distillation recovery** — SlimMoE reaches high ratios with **<10% of the original training
   data**, which is what makes this rentable rather than a pretraining budget.
6. **Quantisation last** — QTIP/AQLM at 2-2.5 bit with sensitivity-mixed precision.
7. **Low-rank residual + QAT** to recover quantisation loss.

Anchoring precedent: **SlimQwen took Qwen3-Next-80B-A3B to 23B-A2B** — ~4x total *and* active —
with competitive downstream performance. That is the shape of the operation being proposed.

### Ranking

1. **GLM-5.3 — the only candidate needing no unvalidated step.** 42% pruning sits inside REAP's
   validated 50%; 2-bit sits inside QTIP's demonstrated range; A40B gives a 24 tok/s roofline.
   Rough cloud cost for full distillation recovery on a ~372B student: **$25-35k**, dominated by
   teacher-logit generation. An engineering project.
2. **DeepSeek V4 Pro — plausible, one unvalidated axis.** 73% pruning is ~1.5x beyond validated but
   active params are fine and everything else is known technique. Progressive pruning plus
   SlimMoE-style recovery is precisely the untested regime. A research bet with a defined failure
   mode.
3. **Qwen3.8 2.4T and Kimi K3 — research projects.** Both need 82-85% pruning *and* active-param
   surgery: four compounding lossy transforms.

**Counterpoint held for K3:** 896 experts at top-16 is the most redundant pool of the four, and
pruning to 134 experts still leaves a sane top-16-of-134 configuration, so per-expert removal may
be more tolerable there than on a 256-expert model. Its native MXFP4 training is genuinely
ambiguous — possibly more robust to 4-bit, possibly already at the edge with no headroom. Unknown.

### Sources for this section

[SlimMoE](https://arxiv.org/pdf/2506.18349) · [SlimQwen](https://arxiv.org/pdf/2605.08738) ·
[FlexMoE](https://arxiv.org/pdf/2606.27866) · [ConMoE](https://arxiv.org/pdf/2605.29350) ·
[LightMoE](https://arxiv.org/pdf/2603.12645) ·
[router calibration necessity](https://arxiv.org/pdf/2603.02217) ·
[attribution-guided pruning](https://arxiv.org/pdf/2606.18304) ·
[TWLA 1.58-bit PTQ](https://arxiv.org/pdf/2606.13054) · [PT2-LLM](https://arxiv.org/html/2510.03267) ·
[PTQ1.61](https://arxiv.org/html/2502.13179)

## 5c. The casual-budget path (the $25-35k figure is optional)

That budget buys one thing only: **pushing past validated pruning rates**. Everything at validated
rates is nearly free, and the highest-value moves cost nothing.

### Tier 0 — $0, on hardware already owned

**CORRECTED 2026-08-18. An earlier version of this section claimed requantising the incumbent was a
free ~28% / 22 GiB win. It is not: the checkpoint is already at the MXFP4 floor.** From
`MODEL_INVENTORY.md` (**measured**): routed experts 85.664 GiB in MXFP4, MTP/DSpark heads 6.529
GiB, MLA attention 4.284 GiB, everything else ~3.9 GiB, total 100.400 GiB. MXFP4 is 4 bits plus one
E8M0 scale per 32-element block = **4.25 bits/param, the format's floor**. The ~4.5-bit *average*
is above 4.25 only because attention, embeddings, lm_head and router are deliberately held at
FP8/BF16, which is correct practice rather than fat.

Going below 4.25 bits therefore means **abandoning MXFP4 for a format with no hardware unpack on
`sm_110a`**. Corrected figures: experts 85.7 -> ~70.5 GiB (**~15 GiB freed**), active bytes 6.9 ->
5.7 GB (**~18% decode**), conditional on a software-unpacked format not converting a
bandwidth-bound decode into a compute-bound one. **A real engineering bet gated on the kernel
prototype, not a free win.**

Revised ranking of the genuinely free levers:

1. **FP8 KV — now the best free lever.** MLA is 41% of B_tok (`dsv4-0731-cuda-server`) and the
   battery runs FP32 KV. Cutting KV reads 4x plausibly buys **~15% decode** *and* frees GB, with no
   weight surgery, and it is trivially reversible.
2. **Draft head / tau.** `tokens_per_verify` is **2.689** (**measured**). tau is a pure multiplier,
   the head is a small model trainable on the 3090, and **the MTP heads already exist in the
   checkpoint** (6.529 GiB) to fine-tune rather than build. See `S5_RECIPE.md`, `HEAD_REGISTRY.md`.
3. **Requantisation — only after** the low-bit `sm_110a` kernel prototype clears the format.

The first two leave the weights untouched, so neither invalidates a running evaluation.

### Requantisation: three things established before attempting it

* **Compounding error is unavoidable if MXFP4 is the only source.** Requantising already-quantised
  weights stacks e1 (MXFP4) on e2 (the new step); calibration methods faithfully reproduce whatever
  they are handed, error included. `MODEL_INVENTORY.md` marks the expert format as native MXFP4
  "proof, not assumption", and the DeepSeek V4 line appears MXFP4-native like Kimi K3 — so there
  may be **no higher-precision parent to work from**. Confirm what
  `deepseek-ai/DeepSeek-V4-Flash-0731` actually ships before planning around a BF16 source.
* **The draft head is coupled to the quantisation, but only through tau.** Speculative decoding is
  *exact* — verification guarantees the target's output distribution regardless of draft quality —
  so a mismatched head costs **speed, never correctness**. MTP heads consume the target's hidden
  states, which shift on requantisation, so acceptance drops. **Sequencing: requantise first, then
  train the head against the final quantisation**, or the head work is done twice.
* **"Within rounding-error noise" needs a testable definition, and the battery cannot supply it.**
  GPQA's CI is +-6 points at n=197, so a change costing under ~3 points is invisible *at this
  sample size* — which is not the same as lossless. Published 4-bit -> 3-bit results degrade
  measurably without recovery training. Use **KL divergence and top-1 agreement against the MXFP4
  model on held-out tokens** (§5), which resolves far below the battery's resolution and costs
  almost nothing.

### Tier 0b — CORRECTION: doing our own REAP is affordable, and the source is FP8

Established 2026-08-18 by inspecting the upstream cards (*searched*):

* **`deepseek-ai/DeepSeek-V4-Flash-0731` is 304B total with tensor types `BF16, F32, F8_E4M3, I8`
  — FP8-native.** A higher-precision source than MXFP4 therefore **exists**, so the compounding-error
  objection in the section above is avoidable by working from the parent rather than from our
  MXFP4 checkpoint.
* **`0xSero/deepseek-v4-flash-0731-spark` is EXL3/Trellis at ~3.0 bits** for routed experts with
  FP8 tensors preserved, REAP'd to **216 of 256 experts (15.6% pruned)**, 99.48 GiB, packaged for a
  128 GB DGX Spark. Not usable here — EXL3 is ExLlamaV3's runtime, not a pure-CUDA `sm_110a` path —
  but two things in it are valuable:

  1. **A natural experiment against our own checkpoint**, same parent and essentially the same
     footprint at the opposite Pareto point:

     | | experts kept | bits | size |
     |---|---|---:|---:|
     | ours (K160) | 160/256 (**37.5% pruned**) | 4.25 | 100.4 GiB |
     | spark | 216/256 (**15.6% pruned**) | 3.0 | 99.5 GiB |

     That is a direct read on **which axis is cheaper — prune harder or quantise harder** — the
     central open question of this document. It is also existence proof that **3-bit trellis works
     on this architecture**, which is exactly what the `sm_110a` kernel gate is about.
  2. **The calibration corpus is public**: `0xSero/deepseek-v4-flash-reap-observations-v2`, 21,289
     rows including 2,253 agentic-tool and 4,096 tool-calling rows. That is the workload-matched
     calibration set §3 argues for, already built for this model. **No need to generate our own.**

**REAP is not the expensive part.** It is one-shot, **forward-only**, and layer-sequential: no
backprop, no optimiser state, no teacher hosting. Cerebras' claim is that it needs no retraining up
to 50%. Run layer-by-layer with cached activations (standard GPTQ practice) and it fits the desktop
(analysis):

* 304 GB FP8 parent on the 1 TB SSD, **streamed once** — ~100 s of I/O at ~3 GB/s
* one layer (~5 GB) in 24 GB VRAM at a time
* subsample calibration to a few hundred sequences, not all 21k rows -> activations ~14 GB, fits
  in 64 GB RAM
* compute is forward-only, order **hours on the 3090**

**Cost: a 304 GB download and engineering time. $0 cloud.** What is genuinely unaffordable is
*distillation recovery*, which is only required to exceed the validated 50% — not needed here.

**Consequence.** One clean pass from FP8: prune *and* quantise from the parent directly into
whatever format the kernels want, with agentic-matched calibration, **never touching MXFP4 in the
middle**. No compounding error, our choice of block size and format, full provenance. Strictly
better than requantising the existing checkpoint, and it makes the REAP-fidelity question the eval
is answering apply to a pipeline we control end to end rather than to a third-party artifact.

**The kernel gate still comes first** — decide the target format by prototyping it on `sm_110a`
before committing a multi-hour pruning run to it.

### Tier 1 — $0-300

**Wait for the community REAP.** `0xSero` published this repo's own DSV4-Flash-0731-REAP as well as
GLM-5.2-REAP-504B and GLM-5.2-REAP-NU176-526B. GLM-5.3 shares 5.2's base so the recipe transfers; a
5.3 REAP will plausibly appear within weeks of the ~2026-08-28 weight drop, making the expensive
step free.

**The catch:** community REAP ships at 34% (504B), and 504B at 2-bit is 126 GB — still over Thor's
~107 GB. Reaching ~50% requires an incremental prune of an already-pruned model. REAP is one-shot
with forward passes only, so that is feasible **locally, layer-by-layer on the 3090** — slow but
free. **Unmeasured risk: two stacked prunes may be worse than one clean 50% prune.** Nobody has
published that comparison.

Renting an 8xH100 node for a single clean pruning pass is a few hours at ~$20-30/hr — **$100-250**.

### Tier 2 — $500-1000

Light distillation recovery, ~1B tokens rather than 50B. Student side ~167 H100-hours (~$350-500);
teacher logits ~$200-400. Enough to make validated-rate pruning solid; **not** enough for 73-85%.

### Out of reach on a casual budget

DeepSeek V4 Pro, Qwen3.8 2.4T and Kimi K3 all require 73-85% pruning with full recovery training.
There is no cheap version — that budget exists precisely to buy past what one-shot methods do.

### Recommended order

**Tier 0, in order, and stop.** Requantisation and the draft head are free, compound, carry no
capability risk, and validate the kernels and the KL harness against a model with existing ground
truth. If they reach ~40 tok/s on DSV4-Flash-REAP the GLM question becomes much less urgent; if
they do not, the bandwidth-efficiency gap is located before any money is spent. Reassess when a
GLM-5.3 REAP appears, which costs only patience.

## 6. Verdict and sequencing

1. **GLM-5.3 at REAP-50% + QTIP ~2.3-bit is the one defensible target.** Actionable when weights
   land ~2026-08-28. Both components sit at published limits rather than beyond them.
2. **DeepSeek V4 Pro needs 75% pruning** — beyond anything validated, closing only if distillation
   recovery substantially outperforms published one-shot results. Attempt after the cheaper bet
   pays, not before.
3. **The biggest unpriced risk is kernels.** QTIP and AQLM have no sm_110a implementations. Trellis
   decode is ALU-heavy and Thor's compute is modest, so there is a real chance a low-bit format
   converts a bandwidth-bound decode into a compute-bound one. **Prototype the kernel before
   committing to the format** — it can invalidate the plan cheaply, which is exactly what a first
   gate should do.
4. **None of this is decidable until the current eval reports.** If REAP loss at 37.5% is small,
   extrapolating to 50% with real recovery training is well-founded. If it is already material, the
   aggressive-compression strategy is refuted before a dollar is spent.

## Sources

Quantization: [QTIP](https://proceedings.neurips.cc/paper_files/paper/2024/file/6de2e84b8da47bb2eb5e2ac96c63d2b0-Paper-Conference.pdf) ·
[QTIP speed writeup](https://www.together.ai/blog/even-better-even-faster-quantized-llms-with-qtip) ·
[AQLM](https://arxiv.org/pdf/2401.06118) · [QuIP](https://arxiv.org/pdf/2307.13304) ·
[VPTQ](https://arxiv.org/pdf/2409.17066) ·
[quantization survey list](https://github.com/pprp/awesome-llm-quantization).
Pruning: [REAP repo](https://github.com/CerebrasResearch/reap) ·
[REAP blog](https://www.cerebras.ai/blog/reap) ·
[reap-cuda](https://github.com/egesabanci/reap-cuda) ·
[ConMoE](https://arxiv.org/pdf/2605.29350) · [MAESTRO](https://arxiv.org/pdf/2607.08601) ·
[SlimQwen prune+distill](https://arxiv.org/pdf/2605.08738).
GLM-5.3 timing: [the-decoder](https://the-decoder.com/zhipu-ai-releases-glm-5-3-claims-its-the-strongest-open-weights-coding-model/).
