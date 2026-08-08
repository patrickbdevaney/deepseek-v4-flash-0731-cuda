# S5 research prompt — the optimal draft-head fine-tuning recipe

**Purpose.** This is the standing prompt for the S5 literature sweep. It is written to be run by
someone (or something) with web access and no prior context on this project, and it is deliberately
over-specified: the failure mode of a vague research prompt is a pile of survey links that restate
what speculative decoding is, when what we need are *numbers we can copy into a training script*.

---

## 0. The system you are researching for — read this before searching

A from-scratch pure-CUDA inference server for `0xSero/DeepSeek-V4-Flash-0731-REAP` on a single
**Jetson AGX Thor** (`sm_110a`, 20 SMs, 122.8 GiB unified LPDDR5X, CUDA 13.0).

| fact | value | why it constrains the answer |
|---|---|---|
| backbone | 43 MoE layers, hidden 4096, 64 heads x head_dim 512 | |
| draft head | **3 chained DSpark MTP blocks, embedded in the checkpoint** (`mtp.0/1/2`, 6.6 GB, shards 46-48) | NOT an EAGLE head, NOT a separate draft model. Recipes assuming a bolt-on head need translating. |
| draft signal | trained on **token cross-entropy**, FastMTP-style | so EAGLE-1's feature-regression saturation curve may not transfer |
| block size | 5 (`dspark_block_size`) | drafts up to 5 tokens/round |
| **current acceptance** | **2.89 of 5** | the number to move |
| current speed | 22.15 tok/s spec / 13.78 tok/s base AR = 1.60x | |
| quantisation | MLA/dense FP8 e4m3; routed experts OCP MXFP4; norms/embed/lm_head/heads BF16 | **hard constraint: no additional quantisation, checkpoint as shipped** |
| free VRAM with model resident | **13.6 GiB** (109.2 of 122.8 used) | training must fit here, or capture-then-train offline |
| capture throughput | prefill **48 tok/s** measured | this, not GPU FLOPs, sets corpus wall time |
| correctness gate | first 8 tokens must match base AR exactly (LOSSLESS) | any recipe that trades output quality for acceptance is disqualified |
| wall-time budget | **a few days**, probed before committing | |

**The objective, stated precisely.** Maximise expected accepted tokens per verify round, subject to
the verified output distribution being unchanged. Speculative decoding is lossless *by construction*
at the verify step, so "quality" here is not the draft's quality — it is (a) whether the recipe
perturbs anything the verify path depends on, and (b) the acceptance/wall-time/corpus Pareto. Say
explicitly when a cited technique gives up losslessness (e.g. relaxed/typical acceptance) and treat
that as a separate, opt-in axis.

---

## 1. Primary question

**What is the highest-acceptance training recipe for a chained multi-token-prediction draft head,
given ~20K captured sequences, a single 122.8 GiB unified-memory device, and a few days of wall
time — and what is the acceptance the recipe should be expected to deliver?**

Return **numbers**, not narrative: learning rates, schedules, optimiser betas, batch/sequence
shapes, epoch counts, loss weights, freeze/unfreeze policy, and the acceptance each produced on
which benchmark at which temperature.

---

## 2. Sub-questions — each needs a numeric answer or an explicit "not published"

### 2.1 Loss design
- Token CE vs feature/hidden-state regression vs combined. What weighting, and what did each
  ablation cost in acceptance?
- For **chained** heads (block k predicts token t+k), is the loss applied per-step, summed, or
  discounted by depth? What discount factor?
- Does anyone report the marginal acceptance of head 2 and head 3 separately? Our chain is depth 3
  and we know from prior work here that DFlash-style linear chains are **depth-dominated**.

### 2.2 The training/inference mismatch (highest-value sub-question)
- **HASS**, and any successor, on harmonised training: the draft is trained on ground-truth
  prefixes but runs on its own outputs. Quantify the acceptance gain from fixing this.
- On-policy / self-distillation / scheduled sampling for draft heads: what fraction of the gain
  needs *online* generation vs what can be had from a static capture with noised prefixes?
- We measured ~7% for on-policy in an earlier internal estimate (2.73x vs 2.54x). Is that
  consistent with the literature, or is our estimate low?

### 2.3 Corpus size and saturation
- Published acceptance-vs-training-tokens curves. Where does each architecture saturate?
- EAGLE-1 reports "low sensitivity to training data"; EAGLE-3 reports an *increasing* curve with no
  saturation at ~8x that. **Which regime does a CE-trained MTP head fall in?** This decides whether
  we capture 5K, 20K, or 100K sequences and is the single biggest wall-time lever.
- What sequence length matters? Is 2K enough or does acceptance need long-context training data?

### 2.4 Data mix
- Does the training corpus need to match the serving distribution? By how much does a mismatched
  corpus cost acceptance?
- ShareGPT / UltraChat / self-generated: which, and in what proportion? Does including the target
  model's *own* generations (rather than human text) matter, and by how much?

### 2.5 Optimisation hyperparameters actually published
- LR, warmup, schedule, optimiser + betas, grad clip, weight decay, epochs, effective batch size in
  tokens, precision (bf16 vs fp32 master weights), for **each** of the recipes below.
- Full fine-tune vs LoRA on the head: any acceptance difference reported?
- Is the backbone frozen? Are the head's own norms/embeddings trained or tied?

### 2.6 Evaluation protocol
- Standard metric definitions: acceptance length tau, speedup ratio, and how temperature is handled.
- **Spec-Bench** and MT-Bench conventions, so our number is comparable to published ones.
- What is the honest way to report acceptance at T=0 vs T>0?

### 2.7 The practical ceiling
- For a top-6-of-160 MoE target at ~180B total / ~12 GB active, what acceptance do published MTP
  heads actually reach at block size 5? Give the distribution of reported values, not one number.
- Diminishing returns: at what acceptance does the verify cost of a longer block overtake the gain?

---

## 3. Sources to cover — named, so nothing is missed

**Must read (primary):**
- **FastMTP** (arXiv 2509.18362) — closest to our architecture: reuses the model's own MTP module.
- **EAGLE-1** (2401.15077), **EAGLE-2** (2406.16858), **EAGLE-3** (2503.01840) — 3 is the
  training-scaling paper and the one whose curve matters.
- **HASS** — harmonised self-distillation, the train/inference mismatch fix.
- **Medusa** (2401.10774) — multi-head, typical acceptance; we have measured typical acceptance
  helping elsewhere (+11%).
- **DeepSeek-V3 technical report** — the MTP objective as the model was originally trained.
- **Nebius** — the operator specifically asked for their published work on speculative decoding /
  draft model training. Find their engineering blog and any papers; extract concrete recipes.

**Should read (secondary):** Hydra, Clover / Clover-2, GLIDE+CAPE, Falcon, Ouroboros, Kangaroo,
Recurrent Drafter (ReDrafter), Sequoia (tree topology), SpecForge / vLLM / SGLang draft-training
tooling, Spec-Bench, and any 2025-2026 survey that tabulates acceptance across methods.

**Also search for, by description rather than name** (the useful result may be recent and unnamed):
- draft-head training recipes released *with* hyperparameters by inference vendors
- anything on training draft heads for **MoE** targets specifically
- anything on draft training under **unified/shared memory** or on Jetson-class hardware
- reports of acceptance **regressions** from over-training or corpus mismatch (negative results are
  as useful here as positive ones)

---

## 4. What a good answer looks like

1. A **recipe table**: hyperparameter -> value -> source -> the acceptance it produced.
2. An **acceptance-vs-corpus-size curve** (or the best available proxy), with the regime our
   CE-trained chained MTP head belongs to, argued from evidence rather than asserted.
3. A **ranked list of levers** by expected acceptance gain per unit of wall time, since wall time is
   our binding constraint and GPU memory is not.
4. Explicit **disqualifications**: techniques that break losslessness, need a second GPU, need more
   than 13.6 GiB resident, or need training data we cannot produce at 48 tok/s prefill.
5. **Confidence markers.** Distinguish "published number", "inferred from a figure", and
   "extrapolated". A wrong number copied confidently into a 3-day training run is the expensive
   failure mode.

## 5. Standing rules for this project that the answer must respect

- No invented constants. Every model number comes from `config.json` / `REAP_MANIFEST.json`.
- No additional quantisation.
- Correctness gates before speed gates; report bands, not points.
- If a recommendation cannot be falsified by a measurement we can run on this box, say so.
