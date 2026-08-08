# S5 — the draft-head fine-tuning recipe, from the research

Written 2026-08-08 from the sweep driven by `RESEARCH_PROMPT_S5.md`. Every hyperparameter below is
either **published** (cited), **measured on this box today** (cited to a command), or **derived**
(the arithmetic is shown). Nothing is asserted.

---

## 0. The finding that reframes everything: the paper is about *our model*

`arXiv 2607.05147` — *DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive
Generation* — is not an analogue. It is our architecture, and its production section deploys on
**V4-Flash**. Its parameters match `config.json` key for key:

| DSpark paper | our `config.json` | |
|---|---|---|
| max block size **γ=5** | `dspark_block_size: 5` | ✓ |
| Markov head low-rank **r=256** | `dspark_markov_rank: 256` | ✓ |
| "three MoE layers with mHC" | 3 `mtp.*` blocks, `hc_mult: 4` | ✓ |
| "sliding window attention of 128" | `sliding_window: 128` | ✓ |

NVIDIA ships a trainer for it (**NeMo AutoModel**, `examples/speculative/dspark/`), and its defaults
agree with the paper's published loss weights. **We are not inventing a recipe — we are reproducing
a documented one on a checkpoint that already contains the head it trains.**

---

## 1. Feasibility, measured today — and a correction

`RESEARCH_LOG.md` §6 recorded the head as **210.8 M parameters ≈ 3 GB of AdamW state**, and
concluded RAM was not a constraint. **That number is FastMTP's MiMo-7B head, not ours.** It was
carried over from the paper and never checked against the checkpoint. The real numbers, read from
the safetensors headers of shards 46–48:

| quantity | value | how |
|---|---|---|
| stored elements in `mtp.*` | **6.935 B** | 2977 tensors, shards 46–48 |
| of which MoE experts | 6.559 B stored as `I8` | |
| **real** expert params | **12.08 B** | `w1.weight [2048,2048] I8` against logical `[2048,4096]` → MXFP4 packed 2/byte |
| **non-expert params** (attn, norms, Markov head, confidence head, projections) | **476 M** | 158.8 M in `mtp.0` × 3 blocks |

**Consequence — this is the decision the recipe turns on:**

| training scope | bf16 master + AdamW fp32 (m,v) = 10 B/param | verdict |
|---|---|---|
| whole head (12.5 B) | **~116 GiB** | infeasible — exceeds the box even with the engine unloaded |
| **non-expert only (476 M)** | **4.44 GiB** | **comfortable** |

So: **freeze the MXFP4 experts, train the 476 M non-expert parameters.** Three independent reasons
this is the right call rather than a concession:

1. **It is what the reference trainer does.** NeMo AutoModel freezes the target's `embed_tokens` and
   `lm_head` and trains "the draft backbone, feature projection, Markov head, and confidence head."
2. **It is where acceptance lives.** The experts hold knowledge; the attention path, Markov head and
   confidence head hold *alignment with the target*, and alignment is what acceptance measures. The
   DSpark paper's own ablation — "a 2-layer DSpark outperforms the 5-layer DFlash baseline" — says
   capacity is not the binding constraint on this architecture.
3. **It keeps the no-additional-quantisation rule intact by construction.** We never write an MXFP4
   tensor, so nothing is ever requantised. A full fine-tune would force a bf16→MXFP4 round trip.

476 M trainable is not small: it is **2.3× FastMTP's entire head**, which lifted MT-Bench acceptance
1.78 → 2.55.

---

## 2. The recipe

### 2.1 Loss — three terms, position-decayed

Published in the DSpark paper and confirmed as the NeMo AutoModel defaults:

```
L = Σ_k  w_k · [ α_ce·CE(p_k, y_k)  +  α_tv·TV(p_k, q_k)  +  α_conf·BCE(c_k, accepted_k) ]

α_ce   = 0.1     (ce_loss_alpha)
α_tv   = 0.9     (l1_loss_alpha)      <- total-variation to the TARGET distribution, dominant term
α_conf = 1.0     (confidence_head_alpha)
w_k    = exp(-(k-1)/γ),  γ = 5        (loss_decay_gamma = block_size)
```

**Why the TV term being dominant matters, and why it is already the right choice.** Nebius's *LK
losses* work is the strongest recent result on draft-head objectives, and its thesis is that **KL is
the wrong loss** — it converges to solutions that do not maximise acceptance under capacity
constraints. Their Gaussian-mixture demonstration: KL reaches 50.2% acceptance, TV reaches **60.2%**
at identical capacity. DSpark's α_tv=0.9 already embodies that finding.

**Optional upgrade, and I rate it worth taking (medium confidence).** Nebius's hybrid schedules
between the two:

```
L_LK^λ = λ·KL(p‖q) + (1-λ)·TV(p,q),   λ = exp(-η·sg[α]),   α = Σ_x min(p(x), q(x))
```

λ→1 early (KL's well-behaved gradients while acceptance is poor), λ→0 late (direct acceptance
optimisation). On **DeepSeek-V3's MTP module fine-tuned from pretrained weights — the closest
published setting to ours — this gave +5.6% acceptance over the KL baseline.** η is not published in
what I retrieved; treat it as the one hyperparameter to sweep (3 values, cheap).

### 2.2 Optimisation

No LR is published for the DSpark trainer. FastMTP is the closest published head-only fine-tune with
a frozen backbone, so its settings are the defensible starting point:

| knob | value | source | confidence |
|---|---|---|---|
| learning rate | **5e-5**, cosine | FastMTP | published |
| warmup ratio | **0.05** | FastMTP | published |
| optimiser | **AdamW**, β=(0.9, 0.95) | FastMTP | published |
| global batch | **64 sequences** | FastMTP | published |
| **epochs** | **1** | Nebius, for MTP *fine-tuned from pretrained* | published, and the important one |
| precision | bf16 params, fp32 optimiser states | derived from the 4.44 GiB budget | derived |
| frozen | all MXFP4 experts, `embed_tokens`, `lm_head` | NeMo AutoModel | published |

**Epochs is where the sources genuinely disagree, and the disagreement is informative.** DSpark
trains **10 epochs** and FastMTP **3** — both *from scratch or near it*. Nebius fine-tuned
DeepSeek-MTP **from pretrained weights for 1 epoch**. Our head is pretrained and shipped inside the
checkpoint. **Start at 1 epoch.** This is also the cheapest possible first answer to the saturation
question, and it costs one run to falsify.

### 2.3 Data — the one point every source agrees on

DSpark, FastMTP, Nebius and NeMo AutoModel independently arrive at the same instruction:
**take prompts from an instruction corpus, throw the human responses away, and regenerate the
responses with the target model.** NeMo states the reason outright — "regenerate before training to
avoid a train/inference distribution mismatch."

| source | corpus | size | mix |
|---|---|---|---|
| DSpark (ours) | Open-PerfectBlend | 1.3 M | chat 17.6 / math 39.4 / **code 38.9** / IF 4.1 |
| Nebius | Infinity-Instruct-0625 | 660 K | — |
| FastMTP | mixed instruction-tuning | 389.4 K | general 42 / math 18 / code 13 / zh 27 |

Sampling parameters for regeneration (FastMTP, published): **T=0.6, top-k 20, top-p 0.95, max 4096**.

Adopt DSpark's mix — it is the one measured on this architecture, and it is **78% math+code**, which
is also where speculative decoding earns the most (its own Table 1: math accepted length 6.11).

### 2.4 The mismatch fix, ranked

`HASS` (ICLR 2025) trains later draft steps on **imperfect draft features rather than clean target
features**, for +8–20% over EAGLE-2. This is the same disease the regenerate-responses step treats,
at a different layer: regeneration fixes the *data* distribution, HASS fixes the *feature*
distribution within the block. Our chain is depth 3, so steps 2 and 3 both run on their own
predecessor's output.

**Ranked by acceptance per unit of wall time:**

1. **Regenerate responses with the target** — mandatory, unanimous, and it is just capture cost.
2. **Position-decayed 3-term loss with α_tv=0.9** — free, it is the published default.
3. **HASS-style context alignment on steps 2–3** — a training-loop change, no extra capture.
4. **LK hybrid λ schedule** — +5.6% on the closest published setting; one η sweep.
5. **Full on-policy (Draft-OPD)** — *deprioritised.* Its claimed gains are large but the retrieved
   summary reports figures ("~2.4× higher acceptance rate") that are not plausible as acceptance
   *lengths* and I could not verify them against the paper body. It also costs 2–3× training compute
   per step. **Low confidence — do not plan around it.** Our own earlier internal estimate put
   on-policy at ~7%, which is consistent with items 1–4 capturing most of the available gain.

---

## 3. Wall time and storage — why prefill comes first

Capture is **teacher-forced prefill**: one pass per sequence, collecting the backbone's last hidden
state per position plus target tokens. So capture time is set by **prefill throughput, measured at
48 tok/s** (F75), and storage by the hidden width, 4096 × bf16 = **8 KB/token**.

Free disk: **128 GB**.

| samples | tokens @1024 | capture @48 tok/s | hidden states bf16 | verdict |
|---|---|---|---|---|
| 5 K | 5.1 M | **29.6 h** | 41 GB | fits |
| 10 K | 10.2 M | **59 h (2.5 d)** | 82 GB | fits, near the disk limit |
| 20 K | 20.5 M | **118 h (4.9 d)** | **164 GB** | **exceeds disk** |
| 389 K (FastMTP) | 398 M | **96 days** | 3.2 TB | impossible |

**Two independent walls, and prefill is under both.** At 48 tok/s we cannot reach any published
corpus size. Doubling prefill halves the time column outright. This is the quantitative form of the
sequencing you called: **B9 (prefill) is now first-order, ahead of the fine-tune, because it is a
multiplier on the fine-tune's cost.**

Prefill is also the *least-explored* part of the engine — nobody has ever run `DSV4_DPROF` on a
prefill (F75 measured it and stopped). At 48 tok/s for a batched forward that should be
compute-bound, against 13.78 tok/s for a memory-bound single token, prefill is only **3.5× decode
when it moves the same bytes for 1000× the work**. That ratio is the anomaly worth attacking.

**Storage mitigations, in order of preference:**
1. **Store hidden states in fp8** (the checkpoint's own MLA format) → halves the disk column; 20 K
   becomes 82 GB and fits. Costs a numerics check against bf16 capture on a small sample.
2. **Shorter sequences.** 1024 is my assumption; the real mean depends on the corpus.
3. **Stream capture→train** in shards, deleting each shard after its epoch. Works only at 1 epoch —
   which is exactly what §2.2 prescribes.

---

## 4. Batching — your correction, with the numbers

I over-sold batching earlier. Measured: the engine sits at **109.2 of 122.8 GiB**, leaving
**13.6 GiB**. Compressed KV is 43 layers × 512 × 2 × bf16 = **86 KiB/token**:

| context | KV per sequence | sequences that fit in 13.6 GiB |
|---|---|---|
| 4 K | 0.34 GiB | ~40 |
| 32 K | 2.69 GiB | **5** |
| 128 K | 10.75 GiB | **1** |

So batching is real at short context and **collapses at long context** — it is not the universal
lever I implied. Correctly stated: it is a *short-context serving* lever, gated on Phase 6, and it
does not belong ahead of the draft head.

---

## 5. Expected outcome

FastMTP's frozen-backbone head-only fine-tune moved acceptance **1.78 → 2.55 on MT-Bench (+43%)**
and **2.02 → 3.16 on math (+56%)**, with a head 2.3× *smaller* than the 476 M we would train.

Our acceptance is **2.89 of a maximum 5**. Applying the measured relationship on this engine
(22.15 spec / 13.78 base / 2.89 accepted = 0.552 tok per accepted-token-slot):

| acceptance | spec tok/s | vs today |
|---|---|---|
| 2.89 (today) | 22.15 | — |
| 3.5 | 26.6 | +20% |
| 4.0 | 30.4 | **+37%** |

**Confidence: medium on reaching 3.5, low on 4.0.** We start from a *pretrained, already-aligned*
head, so the headroom FastMTP exploited (a vanilla MTP head at 1.78) is partly spent — this is the
single biggest uncertainty in the plan, and it is why §2.2 says one epoch and measure.

---

## 6. Order of work

1. **B9 — prefill.** `DSV4_DPROF` on a PS=1023 prefill. Nobody has profiled it. Multiplier on everything below.
2. **Capture harness + fp8 hidden-state storage**, validated on 100 sequences end to end.
3. **Regenerate** Open-PerfectBlend responses with the target at T=0.6/top-k 20/top-p 0.95.
4. **Capture 5 K first** (~30 h at today's prefill, less after B9), train 1 epoch, measure acceptance.
5. **Decide 10 K/20 K from the measured 5 K→acceptance slope**, not from the literature.
6. Re-open the conditionally-retired kernel levers (block >5, NPASS>1, tree drafting, adaptive width) once acceptance justifies a longer block.

**Gate: the LOSSLESS check (first 8 tokens match base AR) must pass on the fine-tuned head.** The
verify step makes speculation lossless by construction, so a failure here means we perturbed
something the verify path depends on — not that the draft got worse.

---

## Sources

- [DSpark: Confidence-Scheduled Speculative Decoding](https://arxiv.org/html/2607.05147v1) — our architecture, production numbers on V4-Flash
- [Train a DSpark Drafter — NVIDIA NeMo AutoModel](https://docs.nvidia.com/nemo/automodel/recipes-e2e-examples/dspark-speculative-decoding) — reference trainer and loss defaults
- [LK losses — Nebius](https://nebius.com/blog/posts/lk-losses) — acceptance-maximising objectives; DeepSeek-MTP +5.6%
- [FastMTP (2509.18362)](https://arxiv.org/abs/2509.18362) — the complete published head-only recipe
- [HASS (2408.15766)](https://arxiv.org/abs/2408.15766) — train/inference context alignment
- [EAGLE-3 (2503.01840)](https://arxiv.org/html/2503.01840v1) — data-scaling regime
- [Draft-OPD (2605.29343)](https://arxiv.org/pdf/2605.29343) — on-policy distillation (low confidence, unverified figures)
- [SpecForge (2603.18567)](https://arxiv.org/pdf/2603.18567) — training framework
- [LK-Speculators collection](https://huggingface.co/collections/nebius/lk-speculators) — released draft models and datasets
