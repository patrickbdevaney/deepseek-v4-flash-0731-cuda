---
license: other
base_model: 0xSero/DeepSeek-V4-Flash-0731-REAP
tags:
  - speculative-decoding
  - draft-head
  - dspark
  - jetson
library_name: safetensors
---

# DSpark MTP draft head — `s3recap-p25-b0.1`

**Weights and provenance, published as-measured.** This head is a working result from an ongoing
optimisation programme, not a polished release: it is the current best of eleven measured
candidates, and it is uploaded so the weights survive the single machine they were produced on.
Every number below is a measurement from the log included in this repo, not an estimate.

Licence follows the base checkpoint,
[`0xSero/DeepSeek-V4-Flash-0731-REAP`](https://huggingface.co/0xSero/DeepSeek-V4-Flash-0731-REAP);
this repository adds only the fine-tuned draft-head tensors.

## What this is

A fine-tuned DSpark MTP draft head for
[`0xSero/DeepSeek-V4-Flash-0731-REAP`](https://huggingface.co/0xSero/DeepSeek-V4-Flash-0731-REAP)
(K160, native MXFP4), trained to raise speculative-decode acceptance on a Jetson AGX Thor
(`sm_110a`) running a from-scratch pure-CUDA inference server.

| | measured |
|---|---|
| suite mean `tau` | **3.8413 / 5** (77 % of the draft-width ceiling) |
| suite mean tok/s | **28.3825** |
| base AR decode | 14.61 tok/s — speculation is **1.97×** |
| vs the stock shipped head | **22.66 → 28.38 tok/s, +25.3 %** |
| draft block width | **5** (the checkpoint's own `dspark_block_size`) |

Every emitted token is bit-identical to base autoregressive decode, checked on every run. The
speed-up is lossless by construction: speculation changes how tokens are *produced*, never which
tokens they are.

## Files

| file | what it is |
|---|---|
| `mtp_trained.safetensors` | **the source of truth.** 72 `mtp.*` tensors. The loadable checkpoint shards regenerate from this deterministically. |
| `head_card.json` | provenance: base model and revision, engine git rev, per-prompt measurements, recorded sha256 |
| `train_metrics.json` | the full training history, all three chunks |
| `eval.log` | the promotion measurement, unedited |
| `config.json` | the base checkpoint's config, for the architectural constants |
| `BEST_SETUP.md` | **the recipe** — every hyperparameter with the reason it has that value |

## Per-category acceptance — read this before assuming the mean

The suite mean is what promotion is decided on. It is **not** what predicts your workload, because
acceptance varies 2.7x across task shapes. All three columns measured at block 5 on the same
protocol; run-0 is the stock head that ships in the base checkpoint.

| category | run-0 (stock) | `s3` | **this head** |
|---|---:|---:|---:|
| long_context | **5.00** | 4.00 | 4.53 |
| agentic_format | **4.41** | 3.74 | 4.16 |
| code_edit | 4.08 | 3.85 | **4.18** |
| multi_turn | 4.06 | **5.21** | 4.83 |
| short_factual | 3.11 | 3.51 | **3.87** |
| reasoning | 1.82 | **3.70** | 3.57 |
| explanation | 1.70 | 2.94 | **3.00** |
| code_gen | 1.86 | 2.56 | **2.59** |
| **suite mean** | 3.2550 | 3.6888 | **3.8413** |

**This head is distribution-matched, not uniformly better.** It roughly doubles acceptance on the
constructive categories the stock head could barely speculate on -- reasoning 1.82 -> 3.57,
explanation 1.70 -> 3.00, code_gen 1.86 -> 2.59 -- and gives back ground on the two the stock head
was strongest at, long_context 5.00 -> 4.53 and agentic_format 4.41 -> 4.16.

**It does not pass this project's own release rule**, which floors the reconstructive categories at
run-0 minus 0.2 and the constructive ones at the incumbent: it is short by 0.27 on long_context,
0.05 on agentic_format and 0.13 on reasoning. It is published here as a measured result and as an
off-machine backup, not as a head certified for every workload.

**So pick by workload, not by the mean.** A pipeline dominated by long-context reconstruction may
do better with the stock head; a mixed or reasoning-heavy one does substantially better with this
one. The table above is the number to predict from.


## `arms/` — the fifteen heads that did not win

The root of this repo is the head you want. `arms/` is **how it was found**: every other candidate
ever measured, with its weights, its `head_card.json`, its full training history and its unedited
promotion log.

These are published deliberately. A refusal with its weights is a reproducible data point; a
refusal recorded only as a number in a table is a claim — and this project has **twice** had to
re-adjudicate its own refusals after discovering the ruler was wrong, which was only possible
because nothing had been thrown away.

**Block 6 era** (`tau` out of 6 — *not* comparable with the block-5 numbers below):

| arm | `tau` | what it tested |
|---|---:|---|
| `s1` | 3.5762 | first fine-tune; promoted at the time, later shown to be inside the noise band |
| `s2` | 3.6275 | 8-way balanced corpus |
| `s2-abl-ce0.5_tv0.5`, `-ce0.9_tv0.1`, `-ce1.0_tv0.0` | 3.64–3.67 | the CE/TV loss-weight ablation |
| `s3` | 3.8438 | balanced 1536-prompt corpus; served for 8 days |

**Block 5 era** (`tau` out of 5, all against a same-width incumbent):

| arm | `tau` | what it tested |
|---|---:|---|
| `s3recap` | 3.6250 | control — re-capture of the identical corpus after a race was fixed in the engine. **Clean negative: the race had cost nothing**, which retired the worry hanging over every earlier head |
| `s3recap-ce1.0` / `-ce0.5` | 3.7325 / 3.6950 | loss reweighting again at the new width; both short |
| `s3recap-deficit` | 3.7975 | deficit weighting alone (β=0) — real, +2.9 %, still short of the bar |
| **root** (`s3recap-p25-b0.1`) | **3.8413** | **+ the β anchor. The winner.** |
| `s3recap-p25-b0.5` | 3.6738 | β=0.5 over-constrains and lands *below* the incumbent — which is what makes β=0.1 a bracketed optimum rather than an endpoint |
| `s3recap-hass1` / `-hass1-p25` | 3.6225 / 3.7950 | HASS, alone and composed with the winner. **Monotone in the wrong direction; retired** |
| `s3recap-conf1.0` / `-conf0.1` | 3.8025 / 3.7925 | the confidence-head loss term. Refused — and the measurement showed the confidence head that **ships in the base checkpoint** already predicts acceptance at **AUC 0.88** without any fine-tuning |

Two of these are worth more than their rank suggests. `s3recap` is the control that made every
earlier head trustworthy. `s3recap-p25-b0.5` is the arm that proves the winner sits at an optimum
rather than at the edge of a sweep.

`shipped-dspark-0731reap` is absent because it has no separate source: it is the base checkpoint's
own head and lives in
[`0xSero/DeepSeek-V4-Flash-0731-REAP`](https://huggingface.co/0xSero/DeepSeek-V4-Flash-0731-REAP).

## `tau` is not comparable across block widths

`tau` counts tokens committed per target forward, and its **ceiling is the draft width**. A head
scoring 3.84/6 and one scoring 3.84/5 are not the same head measured twice. The same weights that
read 3.8438 at width 6 read **3.6888** at width 5. Any comparison must fix the width first.

## Two hyperparameters that are measurements, not defaults

- **`a_conf = 0`** — the confidence-head loss term is off *deliberately*. Training it requires
  free-running acceptance labels, which requires HASS, which costs −0.046 `tau`; the head gains only
  +0.018 AUC from the fine-tuning. The confidence head that ships in the base checkpoint already
  predicts acceptance at **AUC 0.88** without any of our training.
- **`hass_from = 0`** — HASS was measured twice and is monotone in the wrong direction. Retired.

Both are explained in `BEST_SETUP.md`.
