# EVAL_NEXT_STEPS.md — what to do when the battery lands

Companion to `WHY_THESE_EVALS.md`, which says why these benchmarks. This says what is still
missing, what the numbers cannot yet support, and what to run next. Written **before** the
extension pass so its central prediction is falsifiable rather than fitted after the fact.

---

## 1. A pre-registered prediction for GPQA after the 24k extension

The current row reads **72.6%**, flagged NOT QUOTABLE at 25.9% truncation. That number is not a
measurement of the model — a truncated trace has no extractable answer and is scored wrong, so
72.6% is a measurement of an 8000-token budget. The decomposition:

| | n | accuracy |
|---|---:|---:|
| terminated | 146 | **93.8%** [88.7, 96.7] |
| truncated | 51 | 11.8% (6/51) |
| pooled | 197 | 72.6% |

**Do truncated items differ in kind, or only in budget?** If long reasoning predicted wrongness,
accuracy among *terminated* traces would fall with completion length. It does not:

| completion tokens | n | accuracy |
|---|---:|---:|
| 74 – 376 | 24 | 91.7% |
| 387 – 608 | 24 | 100.0% |
| 616 – 1215 | 24 | 100.0% |
| 1235 – 2304 | 24 | 87.5% |
| 2335 – 4576 | 24 | 95.8% |
| 4583 – 7938 | 26 | 88.5% |

Sextiles are noisy at this n. Pooled into wider buckets the shape is a **mild monotonic decline**
rather than a flat line — 95.7% under 600 tokens, 95.2% to 2000, 92.0% to 4000, 90.6% to 8000.
So the items that wanted more than 8000 tokens are *somewhat* harder, which is why the prediction
below sits under the 93.8% terminated rate rather than at it. What the data does **not** show is a
collapse: a five-point drift across a hundredfold range of thinking length will not turn 93.8%
into anything near 11.8%.

**Prediction, stated in advance: 87–93%, central ~90%.**

| scenario | resulting score |
|---|---:|
| extension changes nothing (floor) | 72.6% |
| truncated resolve at 60% | 85.1% |
| truncated resolve at 75% | 89.0% |
| truncated resolve at the longest-sextile rate, 88.5% | 92.5% |
| truncated resolve at the terminated rate, 93.8% | 93.8% |
| all 51 resolve correct (hard ceiling) | 95.4% |

Two reasons the realized number should sit below the naive 93.8% extrapolation: the truncated
items are self-selected for wanting more than 8000 tokens, which is outside the range the flat
curve was measured over; and some of them will **still** truncate at 24k and stay wrong.

**What would falsify the flat-curve reading:** a landing below 85%. That would mean traces which
fail to converge inside 8000 tokens are qualitatively different — reasoning that wanders rather
than reasoning that is merely long — which is itself a REAP-relevant finding about routing
stability under long chains, and should be reported as one rather than buried.

**What this does to the comparison.** Third-party numbers for unpruned DeepSeek-V4-Flash-0731 sit
at **88.1%** (DataLearner/ZenMux) and **91%** (Artificial Analysis). If the extension lands in the
predicted band, the REAP at `effort=low` on a 24k budget is **inside the unpruned model's reported
range** — measured on a different harness at unknown, probably higher, effort. That is a materially
different conclusion from the one 72.6% invites, and it is the reason not to publish or act on the
current row.

---

## 2. The matched-harness baseline — the only way to price the prune

Comparing 72.6% (or 90%) against 88.1%/91% compares *REAP at low effort, tight budget, our
extraction code* against *unpruned at their effort, their budget, their scoring*. Every one of
those knobs moves GPQA by more than the effect being measured. The delta is uninterpretable.

The fix does not need the 304B unpruned model on Thor. It needs **one run of unpruned 0731 through
this repo's harness**, over the API, with everything below held identical.

### Must be byte-identical

| knob | how it is held |
|---|---|
| prompts | `prompt_sha256` per item; `trace_export.py` already proves rebuild-exactness. The API run must hash-match the same set |
| extraction | `eval_suite.extract(kind, ...)` unchanged |
| correctness | `eval_suite.correct(kind, ...)` unchanged |
| truncation rule | truncated ⇒ scored wrong, both sides |
| budget | same 8000, then the same extension-by-continuation to 24k. **Not** a fresh run at 24k — that is a different estimator (see `eval_extend.py`) |
| temperature / top_p | as run: greedy unless the run record says otherwise |
| CI | same nested bootstrap, same seed |

### Cannot be held identical, and why it is survivable

KV precision, kernels, and hardware differ. **None of them touch correctness scoring** — they set
speed, not answers. So the capability delta is comparable even though the throughput numbers are
not. State this explicitly in the writeup rather than letting a reader assume a full-stack match.

### The weak link: does `effort=low` mean the same thing on both sides?

This is the one assumption that can quietly invalidate the whole comparison. Our `low` is this
repo's setting; the API's reasoning-effort control is DeepSeek's. **Verify before trusting any
delta** — the cheapest check is to compare the *completion-token distributions* on the same
prompts. If the API at `low` produces a visibly different thinking-length distribution, effort is
not matched, and the delta is contaminated by exactly the variable the comparison exists to
exclude. If they cannot be matched, say so and report the comparison as bounded, not clean.

### Scope note

The user has ruled API inference out for now. This section is the protocol for when that changes;
nothing here is queued.

---

## 3. Projecting high reasoning effort

We cannot run `effort=high` on this box — FP32 KV at longer thinking lengths exceeds the memory
headroom, which is *why* the battery runs at `low`. So a high-effort number has to be reasoned to,
and labelled as reasoning rather than measurement.

**The honest method, and its limit.** Most of what higher effort buys on a benchmark like GPQA is
more thinking tokens before committing. The extension pass is a partial natural experiment for
exactly that: it removes the budget constraint without changing effort. So the **extension delta
is a lower bound on the effort delta** — it captures "more room to think" but not whatever else
the effort setting changes (search breadth, self-checking, verification passes).

That gives a defensible three-line claim and nothing more:

1. `low` @ 8000 — measured.
2. `low` @ 24000 — measured after the extension. The gap to (1) is the pure budget effect.
3. `high` @ 24000 — **not measurable here.** Bound it below by (2), and note that published
   effort ladders for the base model, where they exist, indicate the direction but not the
   magnitude for a pruned checkpoint.

**I will produce the numbered speculative estimates once the battery and extension land**, per
request — with each one carrying its basis (budget effect / published ladder / analogy) and an
explicit marker that it is not a measurement. What I will not do is present a single point estimate
for high effort as if it were a result; the whole value of this battery is that its numbers are
defensible, and one fabricated row would compromise the rest.

---

## 4. Does this suite actually test what matters?

Assessed against the two things the checkpoint is being trusted with, and the two things a REAP
specifically puts at risk.

### Well covered

| what | instrument | why it is adequate |
|---|---|---|
| world-knowledge breadth | MMLU-Pro (14 domains), GPQA-Diamond | breadth plus depth-in-science |
| math / long chained reasoning | AIME24+25 (reps=2), MATH-500 | single continuous CoT, where routing drift would show |
| code generation | HumanEval, LiveCodeBench, SciCode | HumanEval is a floor test and will not discriminate; LCB and SciCode will |
| single-turn tool use | BFCL `exec_*` (86.2%), `live_*` (78.7%) | clean reads, near-zero truncation; found the over-declining asymmetry |
| **compounding error** | **SciCode multistep** | the best instrument already in the battery — later steps consume the model's **own earlier code**, so main-problem accuracy requires every step right. The subproblem-vs-main-problem gap *is* the compounding signature |

### The gap: long-horizon multi-hop agentic tool use

`live_*` is single-turn. Nothing currently in the battery runs a stateful multi-turn tool
trajectory, which is the single risk you named first and the closest thing to the production
workload. **BFCL `multi_turn_*` is already downloaded and unused:**

| category | n | turns/item | tests |
|---|---:|---:|---|
| `multi_turn_base` | 200 | 3.7 | multi-hop tool use with state |
| `multi_turn_miss_func` | 200 | 4.7 | recognising no available function fits — the `live_relevance` weakness, in a trajectory |
| `multi_turn_miss_param` | 200 | 4.7 | asking for a missing parameter instead of hallucinating one |
| `multi_turn_composite` | 200 | 5.7 | the above, composed |
| `multi_turn_long_context` | 200 | 3.7 | **the named REAP risk, directly** |

Cost, from the **measured** model in `PERF.md` (`ms/token = 41.1 + 0.01379 × kv_mid`), not assumed:

| category | est. wall clock | note |
|---|---:|---|
| `multi_turn_base` | ~10 h | |
| `multi_turn_miss_func` | ~11 h | |
| `multi_turn_miss_param` | ~11 h | |
| `multi_turn_composite` | ~16 h | |
| `multi_turn_long_context` | ~52 h full, **~13 h at n=50** | the depth term dominates |

**Recommendation: `base` + `miss_func` (~21 h) as the minimum viable addition, then
`long_context` subsampled to 50 (~13 h).** That buys the missing capability axis for about 1.5
days. The real cost is not runtime — it is the harness: multi-turn scoring is *state-based*
(compare final backend state), requiring the 8 stateful APIs in `multi_turn_func_doc/`
(GorillaFileSystem, TradingBot, TravelBooking, VehicleControl, and four more). BFCL ships those
under Apache-2.0 and they should be vendored, not reimplemented.

### Not tractable, and now we know exactly why

SWE-bench Verified is the field's gold standard for agentic coding and it is **out of reach on this
box** — not for the reason usually given. At SWE-bench's typical 50k-token working context the
measured cost model gives `41.1 + 0.01379 × 50000` = **731 ms/token, i.e. 1.4 tok/s**. One instance
generating 5k tokens takes an hour. Even a 50-instance subset is ~50 h of decode before any harness
overhead, and the full 500 is out of the question.

This is the depth term from `PERF.md` deciding a scientific question, not just a kernel one:
**long-horizon agentic evaluation is disproportionately expensive on this hardware, and closing the
depth term is what would make it affordable.** Task #24 is therefore not only a throughput lever —
it gates which benchmarks this project can ever run.

### A REAP-specific analysis worth doing on data we will already have

Pruning removes experts. If that cost tail knowledge, the damage should be **concentrated**, not
uniform — a few subjects collapsing rather than everything sagging. So the REAP signature is
**dispersion across subjects, not the mean**. MMLU-Pro ships 14 categories and GPQA carries a
`subject` field, both already recorded per record. When MMLU-Pro lands, compare per-subject
accuracy dispersion against the published unpruned per-subject profile where one exists. High
dispersion concentrated in narrow domains is prune damage; uniform depression is effort or budget.
This costs no extra runtime and is the sharpest REAP-specific test available from the current plan.

---

## 5. Queue

1. Battery completes: scicode, mmlu_pro, humaneval, math500, lcb, aime24×2, aime25×2.
2. `eval_extend_all.sh` — extension by continuation to 24k for every row over the 5% gate.
3. Check §1's prediction against the realized GPQA number. Report the miss if it misses.
4. Per-subject dispersion analysis on MMLU-Pro and GPQA (§4).
5. Speculative high-effort projections, labelled as such (§3).
6. Build BFCL `multi_turn` — vendor the backends, `base` + `miss_func` first (§4).
7. Matched-harness API baseline, if and when API inference is back in scope (§2).
