# Compression playbook: getting a frontier MoE resident on Thor with acceptable decode

Companion to `MODEL_SURVEY_APPENDIX.md` §4c, which establishes *that* residency is the whole game.
This is *how*, assuming rentable cloud GPUs and a budget. Same provenance convention: **measured**
here, *searched* with a citation, (analysis) derived.

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
