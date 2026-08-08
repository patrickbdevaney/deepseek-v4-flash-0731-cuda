# S5 — fine-tuning the DSpark MTP draft head

The ML side of the project. Speculative decoding's speedup is `acceptance × efficiency`; seventeen
cycles of kernel work took `efficiency` to ~96 % of its realistic ceiling, so **everything left is
`acceptance`**, and acceptance is a training problem.

Sources and full citations: `S5_RECIPE.md`. Search dedup ledger: `RESEARCH_LOG.md`.

---

## 1. The architecture we are training

Not an EAGLE head, not a bolt-on draft model: **three chained DSpark MTP blocks embedded in the
checkpoint itself** (`mtp.0/1/2`, shards 46–48, 6.6 GB, tapping backbone layers 40/41/42).

The decisive research finding is that **the paper describing this architecture describes *our
model*.** `arXiv 2607.05147` — *DSpark: Confidence-Scheduled Speculative Decoding with
Semi-Autoregressive Generation* — matches `config.json` key for key, and its production section
deploys on V4-Flash:

| DSpark paper | our `config.json` |
|---|---|
| max block size **γ=5** | `dspark_block_size: 5` |
| Markov head low-rank **r=256** | `dspark_markov_rank: 256` |
| "three MoE layers with mHC" | 3 `mtp.*` blocks, `hc_mult: 4` |
| "sliding window attention of 128" | `sliding_window: 128` |

NVIDIA ships a trainer for it (**NeMo AutoModel**, `examples/speculative/dspark/`) whose loss
defaults match the paper. **We are reproducing a documented recipe on a checkpoint that already
contains the head it trains.**

The head is *semi-autoregressive*: a parallel backbone produces every position of a block in one
pass, a lightweight serial **Markov head** injects intra-block token dependency, and a **confidence
head** predicts per-position acceptance probability for scheduled verification.

---

## 2. Feasibility — and a correction worth reading

`RESEARCH_LOG.md` originally recorded the head as **210.8 M parameters ≈ 3 GB of AdamW state** and
concluded RAM was not a constraint. **That number is FastMTP's MiMo-7B head. It was copied out of a
paper and never checked against the checkpoint.**

Read from the safetensors headers of shards 46–48:

| quantity | value |
|---|---|
| stored elements in `mtp.*` | **6.935 B** (2977 tensors) |
| MoE experts, stored | 6.559 B as `I8` |
| MoE experts, **real** | **12.08 B** — `w1.weight [2048,2048] I8` against a logical `[2048,4096]`, i.e. MXFP4 packed 2/byte |
| **non-expert** (attn, norms, Markov head, confidence head, feature projection) | **476 M** |

At 10 B/param for bf16 master + AdamW fp32 moments:

| training scope | memory | verdict |
|---|---|---|
| whole head (12.5 B) | **~116 GiB** | infeasible |
| **non-expert only (476 M)** | **4.44 GiB** | comfortable |

**So: freeze the MXFP4 experts, train the 476 M non-expert parameters.** Three independent reasons
this is right rather than a concession:

1. It is what NVIDIA's reference trainer does — it freezes the target's `embed_tokens` and
   `lm_head` and trains "the draft backbone, feature projection, Markov head, and confidence head".
2. **It is where acceptance lives.** The experts hold knowledge; the attention path, Markov head and
   confidence head hold *alignment with the target*, and acceptance measures alignment. The DSpark
   paper's own ablation — "a 2-layer DSpark outperforms the 5-layer DFlash baseline" — says capacity
   is not the binding constraint here.
3. **It keeps the no-additional-quantisation rule true by construction.** We never write an MXFP4
   tensor, so nothing is requantised. A full fine-tune would force a bf16→MXFP4 round trip.

476 M trainable is **2.3× FastMTP's entire head**, which lifted MT-Bench acceptance 1.78 → 2.55.

> The general lesson: the "no invented model constants" rule was written for numbers pulled from the
> air. A number pulled from a *paper about a different model* is exactly as wrong, and this one
> would have failed at allocation **after** a multi-day capture had been paid for.

---

## 3. The recipe

### 3.1 Loss — three terms, position-decayed

Published in the DSpark paper, confirmed as the NeMo AutoModel defaults:

```
L = Σ_k  w_k · [ α_ce·CE(p_k, y_k) + α_tv·TV(p_k, q_k) + α_conf·BCE(c_k, accepted_k) ]

α_ce   = 0.1     α_tv = 0.9     α_conf = 1.0
w_k    = exp(-(k-1)/γ),  γ = 5
```

**Why the TV term dominates, and why that is already correct.** Nebius's *LK losses* work is the
strongest recent result on draft-head objectives, and its thesis is that **KL is the wrong loss** —
it converges to solutions that do not maximise acceptance under capacity constraints, which is
always the practical case. Their Gaussian-mixture demonstration: KL reaches 50.2 % acceptance, TV
reaches **60.2 %** at identical capacity. DSpark's `α_tv = 0.9` already embodies that finding.

**Optional upgrade (medium confidence).** Nebius's hybrid schedules between them:

```
L_LK^λ = λ·KL(p‖q) + (1-λ)·TV(p,q),   λ = exp(-η·sg[α]),   α = Σ_x min(p(x), q(x))
```

λ→1 early (KL's well-behaved gradients while acceptance is poor), λ→0 late (direct acceptance
optimisation). On **DeepSeek-V3's MTP module fine-tuned from pretrained weights — the closest
published setting to ours — this gave +5.6 % acceptance over the KL baseline.** η is unpublished;
treat it as the one knob to sweep.

### 3.2 Optimisation

No LR is published for the DSpark trainer. FastMTP is the closest published head-only fine-tune with
a frozen backbone, so its settings are the defensible starting point:

| knob | value | source |
|---|---|---|
| learning rate | **5e-5**, cosine | FastMTP |
| warmup ratio | **0.05** | FastMTP |
| optimiser | **AdamW**, β=(0.9, 0.95) | FastMTP |
| global batch | **64 sequences** | FastMTP |
| **epochs** | **1** | Nebius, for MTP fine-tuned *from pretrained* |
| precision | bf16 params, fp32 optimiser states | derived from the 4.44 GiB budget |
| frozen | all MXFP4 experts, `embed_tokens`, `lm_head` | NeMo AutoModel |

**Epochs is where the sources disagree, and the disagreement is informative.** DSpark trains **10**
epochs and FastMTP **3** — both from scratch or near it. Nebius fine-tuned DeepSeek-MTP **from
pretrained weights for 1 epoch**. Our head ships pretrained. **Start at 1.** It is also the cheapest
possible first answer to the corpus-saturation question.

### 3.3 Data — the one point every source agrees on

DSpark, FastMTP, Nebius and NeMo AutoModel independently arrive at the same instruction: **take
prompts from an instruction corpus, throw the human responses away, and regenerate the responses
with the target model.** NeMo states the reason outright — "regenerate before training to avoid a
train/inference distribution mismatch."

| source | corpus | size | mix |
|---|---|---|---|
| **DSpark (ours)** | Open-PerfectBlend | 1.3 M | chat 17.6 / math 39.4 / **code 38.9** / IF 4.1 |
| Nebius | Infinity-Instruct-0625 | 660 K | — |
| FastMTP | mixed instruction-tuning | 389.4 K | general 42 / math 18 / code 13 / zh 27 |

Regeneration sampling (FastMTP, published): **T=0.6, top-k 20, top-p 0.95, max 4096**.

Adopt DSpark's mix — it is the one measured on this architecture, and it is 78 % math+code, which is
also where speculative decoding earns most (its Table 1: math accepted length 6.11).

### 3.4 The train/inference mismatch, ranked by value per unit of wall time

1. **Regenerate responses with the target** — mandatory, unanimous, and it is just capture cost.
2. **Position-decayed 3-term loss with α_tv=0.9** — free, it is the published default.
3. **HASS-style context alignment on steps 2–3** — HASS (ICLR 2025) trains later draft steps on
   *imperfect draft features* rather than clean target features, for +8–20 % over EAGLE-2. Our chain
   is depth 3, so steps 2 and 3 both run on their predecessor's output. A training-loop change, no
   extra capture.
4. **LK hybrid λ schedule** — +5.6 % on the closest published setting; one η sweep.
5. **Full on-policy (Draft-OPD)** — **deprioritised, low confidence.** Claimed gains are large but
   the retrieved figures ("~2.4× higher acceptance rate") are not plausible as acceptance *lengths*
   and could not be verified against the paper body. Costs 2–3× training compute per step.

---

## 4. Capture — the engineering that makes it possible

**Capture is teacher-forced prefill**: one forward pass per sequence, collecting the backbone's
layer taps plus target tokens. So capture throughput **is** prefill throughput, which is why
[`prefill-optimisation.md`](prefill-optimisation.md) came first.

### 4.1 What to store, and the trap in it

`src/decode.cu` allocates `mh_pre` as `(SMAX-1) × 3 × d × 4` — the DSpark blocks tap **three**
layers (40/41/42), not one. Two candidate capture points:

| capture point | bytes/token (bf16) | usable? |
|---|---|---|
| `main_x` (after `main_proj`) | 8 KB | **no** — `main_proj` is the "feature projection", which the reference trainer lists as **trainable**. Capturing after it bakes in weights we are about to change. |
| **`mh_pre` (the 3 raw taps)** | **24 KB** | yes |

### 4.2 Storage is not a constraint, because the recipe is one epoch

At 24 KB/token a monolithic 20 K × 1024-token corpus is **492 GB**, against ~122 GB free. But at one
epoch each sample is used exactly once, so: **capture a shard → train on it → delete → next shard.**
Peak usage is one shard.

| shard | bf16 | fp8 |
|---|---|---|
| 1 K samples | 25 GB | 12 GB |
| 2 K samples | 49 GB | 25 GB |
| 5 K samples | 123 GB | 61 GB |

**Corpus size is therefore purely a time decision.** 5 K, 20 K and 100 K all cost the same disk.

The target distribution `p` for the TV term is **not** stored — 129,280 floats/token is 258 KB/token
and impossible. It is recomputed during training from the stored taps through the frozen `lm_head`
(1.06 GB bf16, cheap to keep resident). Storing the taps is the minimal sufficient statistic.

### 4.3 Wall time

At the measured prefill rate (**62.4 tok/s** after B9, from 47.9):

| samples | tokens @1024 | capture |
|---|---|---|
| 5 K | 5.1 M | **23 h** |
| 10 K | 10.2 M | **1.9 d** |
| 20 K | 20.5 M | **3.8 d** |

**These use PS=1022, the slowest realistic point** — F75's sweep shows throughput *falls* with
sequence length (255 → 52.6 tok/s, 1023 → 47.7, 2047 → 43.8) because the sparse attention path is
superlinear in `s`. If regenerated responses average ~512 tokens, real capture is meaningfully
faster.

---

## 5. Corpus saturation — what the literature actually says

The question "is 5–10 K enough?" is **architecture-dependent**, and the two regimes are well
separated:

- **EAGLE-1** reports "low sensitivity to training data" — it saturates early (~68 K), because
  feature-regression drafting hits an expressivity ceiling.
- **EAGLE-3** *removes* the feature-prediction constraint and reports a **scaling law**: increasing
  training data gives proportional speedup improvement, with **no saturation** at ~8× EAGLE-1's
  corpus. It fuses low/middle/upper-layer features rather than only the final one.

**Our head is in the EAGLE-3 regime**: it trains on token CE + distribution matching (not feature
regression) and it taps **three** layers. So more data should keep paying, and 20 K is likely still
on the steep part of the curve.

**But do not plan from that.** The measurement is cheap: capture 5 K, train 1 epoch, measure
acceptance; then decide 10 K/20 K from *our* slope, not the literature's.

---

## 6. Expected outcome, stated with its uncertainty

FastMTP's frozen-backbone head-only fine-tune moved acceptance **1.78 → 2.55 on MT-Bench (+43 %)**
and **2.02 → 3.16 on math (+56 %)**, with a head 2.3× *smaller* than the 476 M we would train.

Ours is at **2.89 of a maximum 5**. Applying the relationship measured on this engine (22.15 spec /
13.83 base / 2.89 accepted = 0.552 tok per accepted-token-slot):

| acceptance | spec tok/s | vs today |
|---|---|---|
| 2.89 (today) | 22.15 | — |
| 3.5 | 26.6 | +20 % |
| 4.0 | 30.4 | **+37 %** |

**Confidence: medium on 3.5, low on 4.0.** We start from a *pretrained, already-aligned* head, so
some of the headroom FastMTP exploited (a vanilla MTP head at 1.78) is already spent. This is the
single biggest uncertainty in the plan and is why the recipe says one epoch and measure.

An unverified observation worth testing: acceptance on a 1023-token real-code prompt measured
**3.57–4.00** against 2.89 on the canonical 6-token prompt. Different prompt and unusually
predictable content, so **not** like-for-like — but if long realistic prompts genuinely accept
better, the baseline this project quotes is pessimistic.

---

## 7. Safety

The base head is backed up at `~/model-backups/dspark-mtp-base-0731REAP` — shards 46–48 plus index
and config, **sha256-verified byte-for-byte against source**. All 2977 `mtp.*` tensors live in those
three shards and those shards contain nothing else, so restore is a file copy.

**Gate: the LOSSLESS check (first 8 tokens match base AR) must pass on the fine-tuned head.** The
verify step makes speculation lossless by construction, so a failure there means we perturbed
something the verify path depends on — not that the draft got worse.
