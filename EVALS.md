# Capability of the REAP K160 checkpoint, as served by this engine

**Status: RUNNING at seqmax 8192.** Results below are regenerated from `evidence/evals/summary.json`; the "scored"
column always shows how many items of the benchmark were actually completed. Tasks still in flight
are marked. Nothing in this file is a projection.

## What is being measured, and what it is not

Three different things get confused in a sentence like "this model scores X", so they are separated
here up front:

| | what it is | measured here? |
|---|---|---|
| `deepseek-ai/DeepSeek-V4-Flash-0731` | the unpruned model, 304B total | **no** — its published numbers are its own |
| `0xSero/DeepSeek-V4-Flash-0731-REAP` | K160, 37.5 % of routed experts removed, MXFP4 | this is the checkpoint |
| **this repo's engine** | a from-scratch CUDA server for it on one Jetson AGX Thor | **this is what runs the items** |

So a number here is a joint statement about *the pruned checkpoint* and *this implementation of it*.
It cannot separate the two, and it does not try to. What it can do is answer the question the REAP
checkpoint's own card leaves open:

> "Structural and smoke validation do not establish benchmark parity with the unpruned model."
> ... "Coding preservation is plausible ... but must be measured rather than assumed."
> — `README.md` of the checkpoint

## The comparison problem, stated honestly

**The official DeepSeek-V4-Flash-0731 model card publishes no benchmark that can be run here.**
Every number on it is an agent rollout:

| official card benchmark | score | why it cannot be run on this box |
|---|---|---|
| Terminal Bench 2.1 | 82.7 | container harness + multi-turn tool loop; needs >>4096 ctx |
| NL2Repo | 54.2 | repo-scale generation |
| CyberGym | 76.7 | sandboxed exploitation environment |
| DeepSWE | 54.4 | SWE-bench-lineage: repo checkout, patch, test harness |
| Toolathlon-Verified | 70.3 | live tool endpoints |
| Agents' Last Exam | 25.2 | agent harness |
| AutomationBench Public | 25.1 | agent harness |
| DSBench-FullStack / Hard | 68.7 / 59.6 | agent harness |

and they were produced with "the minimal mode of DeepSeek Harness (to be released) as the agent
framework, using the `max` reasoning effort level with `temperature = 1.0, top_p = 0.95`" — a harness
that is not released, so those numbers are not reproducible by anyone outside DeepSeek at present.

That leaves **third-party aggregation** as the only source of head-to-head numbers on the academic
benchmarks this harness can actually score. Those are reported below and labelled as such. They are
**not** official, the aggregator does not attribute who ran them, and they should be read as a
reference point rather than a ground truth.

### Third-party reference numbers for the *unpruned* model

Source: [DataLearner](https://www.datalearner.com/en/ai-models/pretrained-models/deepseek-v4-flash),
which states its sourcing hierarchy as "official releases (GitHub, Hugging Face, papers), then
benchmark leaderboards, then third-party evaluators" but does not attribute these particular rows.
Reproduced with the per-effort breakdown, because effort level moves these by up to 36 points and a
comparison that ignores it is meaningless:

| benchmark | standard | **high** | max |
|---|---|---|---|
| GPQA Diamond | 71.20 | **87.40** | 88.10 |
| MMLU-Pro | 83.00 | **86.40** | 86.20 |
| LiveCodeBench | 55.20 | **88.40** | 91.60 |

**This harness runs at `reasoning_effort: high`, so the `high` column is the like-for-like one.**

Note also that the aggregator lists the model as 284B-total/13B-active while the official card says
304B total. The discrepancy is theirs; it is recorded rather than resolved.

## The context constraint, and what was done about it

DeepSeek recommends **384K output tokens** for the `high` and `max` reasoning levels. This server
originally ran at **4096 total** (prompt + completion), and that was not a tuning choice -- two
attempts to serve at a larger context on 2026-08-12 took the **whole machine** down mid-load
(reboots at 20:02 and 20:11, `last -x reboot`, no oom-kill line in `dmesg` either time).

That ceiling was corrupting the measurement, not merely limiting it. GPQA-Diamond's longest prompt
is **2543 tokens** by the checkpoint's own tokenizer; at seqmax 4096 with any usable completion
budget, `server.cpp`'s `prompt + max_tokens + 8 > seqmax` check rejects it outright. The first run
of this battery lost **all 198 GPQA items** to that, and reported the task as "done".

**The ceiling turned out to be an engine artefact, not an architectural one.** MLA + DSA make this
model's real KV cache **99.4 KiB/token** -- only **3.3 %** of everything that scaled with `seqmax`.
The rest was scratch sized by the wrong quantity, dominated by an arena reserved as
`(512 + 2 x seqmax) MiB` when what it must cover is the widest *batch*, never the context. Measured:
a 4173-token prefill touches **215 MiB** of arena, identical to what a 39-token prefill touches. The
old formula reserved 8704 MiB for it.

Prefill now runs through the same chunked path as a cache extension, the arena is sized by the
batch, and the change is **bit-identical** -- `gate_engine` passes 8/8 before and after with every
generated token, every margin and every `head|d|` unchanged. See
**`wiki/context-ceiling-is-not-the-kv-cache.md`**.

| | before | now |
|---|---:|---:|
| seqmax | 4096 | **8192** |
| arena | 8704 MiB | **640 MiB** (215 used) |
| memory at ready | 120.1 / 122.8 GiB | 119.6 / 122.8 GiB |
| GPQA items that fit | **0** | **198** |
| decode | 20.4 tok/s | 20.5 tok/s |

The recovered memory was spent on **reducing truncation** rather than on a larger context number,
because a truncated item is scored *wrong*: AIME 3400 -> 5000 max tokens, GPQA 3000 -> 4000,
MMLU-Pro 2600 -> 4500, MATH-500 3000 -> 4000, GSM8K/HumanEval 1600 -> 2000.

**Truncation is still real and is counted honestly.** It is detected by token count, not by
`finish_reason` -- the server reports `"stop"` even when a generation ended because it exhausted
`max_tokens`, verified on an MMLU-Pro item that emitted exactly 3500 of 3500 tokens mid-sentence and
was labelled `stop`. Trusting that field would have printed `trunc 0` while items were being cut off.
Truncated items are scored as attempted and **wrong**, never dropped, so **every number here remains
a floor**.

## Method

- **Exactly scorable only.** Integer, letter, normalised expression, or a unit test that passes.
  No LLM judge, no partial credit. A defensible small number beats an indefensible big one.
- **Through the shipping server.** Same binary, tokenizer, chat template, sampler and speculative
  decode path a user gets — `POST /v1/chat/completions`. Not a special eval path.
- **Sampling:** `temperature = 1.0`, `top_p = 0.95`, `reasoning_effort = high` — the card's
  recommended agentic settings, held fixed across every task.
- **Datasets pinned by commit sha** in `evidence/evals/datasets.json`; nothing is downloaded at eval
  time, so a re-run scores the same rows.
- **Subsets are deterministic random draws**, never prefixes (MMLU-Pro is ordered by category, so a
  prefix would score one subject and call it the benchmark). Item ids derive from the source row
  index, so `--n 5` and `--n 250` agree on which item is which.
- **95 % CI is the Wilson score interval** — several tasks are 30 items, where the normal
  approximation is simply wrong near the ends.
- **Every generation is retained** in `evidence/evals/<task>.jsonl`, including the full reasoning
  trace, so any number here can be traced to the text that produced it.

### Is this suite actually defensible? — the honest audit

The benchmarks themselves are the genuine article: GPQA-Diamond, MMLU-Pro, AIME, MATH-500, HumanEval
and GSM8K are what essentially every frontier model card reports, and they are exactly scorable. But
"standard benchmark" and "defensible number" are different claims, and four things separate them.

**1. Statistical power. This is where a naive run of this suite fails.** Wilson half-width at each
planned n:

| benchmark | n | assumed acc | 95 % CI | ± | verdict |
|---|---:|---:|---|---:|---|
| GPQA-Diamond | 198 | 85 % | [79.2, 89.2] | 5.0 | usable |
| MMLU-Pro | 200 | 80 % | [73.9, 85.1] | 5.5 | usable |
| HumanEval | 164 | 85 % | [78.5, 89.5] | 5.5 | usable |
| MATH-500 | 120 | 90 % | [83.2, 94.3] | 5.4 | usable |
| GSM8K | 120 | 92 % | [85.7, 95.8] | 4.9 | usable |
| **AIME 24/25 @ reps=1** | **30** | **70 %** | **[52.1, 83.3]** | **15.6** | **NOT PUBLISHABLE** |

Thirty problems sampled once at `temperature = 1.0` is not a measurement — the interval spans
52-83 %, which cannot distinguish a good model from a mediocre one. This is precisely why published
AIME numbers are avg@16 to avg@64. **AIME therefore runs at `--reps 4` here** (avg@4, ± 8.1), which
is the least it can be run at and still be quoted, and the report states the rep count. Reps are
additive — rep 0 keeps the bare item id — so k can be raised later without regenerating anything.

**2. The gold keys are cross-validated, not trusted.** GPQA-Diamond is read from a mirror because
the canonical `Idavidrein/gpqa` is gated (only its README is retrievable here). A mirror can have a
different option order or a shifted answer key, and GPQA scores are sensitive to both. So the key
was checked against a **second, independent mirror** (`hendrydong/gpqa_diamond`) that stores the
answer as free text rather than as a letter: map each letter back to its option text and compare.

> **196 / 198 = 99.0 % agreement between two independently produced mirrors.** The two
> non-matches are fuzzy-matcher failures on stem collisions, not conflicting keys.

**3. Prompt protocol differs from the published harnesses, and the sign of that is unknown.** These
run zero-shot CoT with an explicit answer-format instruction. MMLU-Pro's published numbers are
conventionally 5-shot CoT; the agentic numbers on the official card come from an unreleased harness.
A prompt-format difference is worth several points in either direction on multiple-choice tasks and
**cannot be calibrated away here**. It is a real limit on head-to-head precision, not a footnote.

**4. Contamination is unmeasured.** AIME 2024, GSM8K and HumanEval all predate the checkpoint and
may be in its training data. That inflates rather than deflates, i.e. it pushes the opposite way
from the context ceiling. Neither effect is quantified, and no claim here is strong enough to
require that they cancel.

**What this suite can therefore support:** "the REAP K160 checkpoint, served by this engine at 4096
tokens of context, scores X ± CI on benchmark B under the stated protocol." **What it cannot
support:** "DeepSeek-V4-Flash-0731 scores X", or a precise claim about how many points the pruning
cost, since the protocol gap and the context ceiling are confounded with the pruning.

### Harness self-gates (run before any model was scored)

A scoring bug is indistinguishable from a capability result, so the scorers were gated first:

| gate | result |
|---|---|
| **C** — every HumanEval canonical solution passes the code scorer | **164/164** |
| **M1** — every MATH-500 gold matches itself through the normaliser | **500/500** |
| **M2** — 10 realistic LaTeX renderings of a gold score correct | 10/10 |
| **X** — extraction from realistic completions (boxed, "Answer: X", A-J) | 8/8 |
| **ID** — item ids stable across `--n` | 4/4 tasks |
| **G** — GPQA gold key vs a second independent mirror | **196/198 = 99.0 %** |

Extraction fallbacks are deliberately one-directional: each can only ever make a score **higher**,
so a low number cannot be dismissed as strict parsing.

## Results

<!-- RESULTS -->
| benchmark | effort | scored | **acc %** | 95 % CI | trunc | err | mean out tok | tok/s | unpruned 0731 @ same effort |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| GPQA-Diamond *(partial)* **— NOT QUOTABLE, 26% truncated at max_tokens=8000; this measures the budget, not the model. Extend with tools/eval_extend.py** | **low** | 197/198 | **72.6** | [66.0, 78.3] | 51 | 0 | 3717 | 15.0 | 71.20 |
| MMLU-Pro *(partial)* **— NOT QUOTABLE, 12% truncated at max_tokens=8000; this measures the budget, not the model. Extend with tools/eval_extend.py** | **low** | 150/150 | **73.3** | [65.7, 79.8] | 18 | 0 | 1948 | 17.3 | 83.00 |
| MATH-500 | **low** | 100/100 | **95.0** | [88.8, 97.8] | 1 | 0 | 940 | 21.6 | — *(none published)* |
| HumanEval | **low** | 164/164 | **95.1** | [90.7, 97.5] | 6 | 0 | 1438 | 20.6 | — *(none published)* |
| BFCL v3 — 4 `exec_*` categories, AST † | **low** | 240/240 | **86.2** | [81.3, 90.0] | 3 | 0 | 216 | 22.8 | — *(none published)* |
| BFCL v3 — `live_*` categories, AST † | **low** | 508/508 | **78.7** | [75.0, 82.1] | 5 | 0 | 247 | 18.5 | — *(none published)* |
| LiveCodeBench — `test6` window † *(partial)* **— NOT QUOTABLE, 59% truncated at max_tokens=8000; this measures the budget, not the model. Extend with tools/eval_extend.py** | **low** | 162/175 | **46.9** | [39.4, 54.6] | 96 | 0 | 5578 | 12.3 | — *(none published)* |
| SciCode — subproblem pass rate *(partial)* **— NOT QUOTABLE, 19% truncated at max_tokens=8000; this measures the budget, not the model. Extend with tools/eval_extend.py** | **low** | 291/291 | **30.2** | [24.1, 36.9] | 54 | 0 | 3534 | 14.7 | — *(none published)* |

1812 items scored, 3,463,627 completion tokens generated. Sampling held at `temperature = 1.0`, `top_p = 0.95`; the reference column is the aggregator number at the SAME reasoning effort as the row.

Interval method per row: BFCL v3 — 4 `exec_*` categories, AST † — wilson, BFCL v3 — `live_*` categories, AST † — wilson, GPQA-Diamond — wilson, HumanEval — wilson, LiveCodeBench — `test6` window † — wilson, MATH-500 — wilson, MMLU-Pro — wilson, SciCode — subproblem pass rate — cluster-bootstrap over 65 problems (291 sub-items). Single-sample tasks use a Wilson score interval; avg@k tasks use a nested bootstrap over PROBLEMS, because k samples of one problem are not k problems and pooling them into a Wilson interval understates the width by up to 2x — most severely when the extra samples bought the least.

Per-task provenance (dataset and pinned snapshot):

| benchmark | dataset | snapshot | max_tokens |
|---|---|---|---:|
| GPQA-Diamond | fingertap/GPQA-Diamond (test, 198) | `68be75644976` | 8000 |
| MMLU-Pro | TIGER-Lab/MMLU-Pro (test, 12032) | `b189ec765aa7` | 8000 |
| MATH-500 | HuggingFaceH4/MATH-500 (500) | `6e4ed1a2a79a` | 8000 |
| HumanEval | openai/openai_humaneval (164) | `7dce6050a7d6` | 8000 |
| BFCL v3 — 4 `exec_*` categories, AST † | gorilla-llm/BFCL v3 (exec subsets, AST-scored, 240) | `f087fb14f26d` | 2000 |
| BFCL v3 — `live_*` categories, AST † | gorilla-llm/BFCL v3 (live_* subsets, AST-scored, 508) | `61fc0608cfd8` | 2000 |
| LiveCodeBench — `test6` window † | livecodebench/code_generation_lite test6 (2025-01..2025-04, 175) | `0fe84c3912ea` | 8000 |
| SciCode — subproblem pass rate | SciCode1/SciCode (test, 65 problems, 291 subproblems) | `4510f6a6aa27` | 8000 |

† This row is **not** the full benchmark — see the protocol block below.
<!-- /RESULTS -->

### A partial run is not just a noisier run — it can be a biased one

The MMLU-Pro run that the UTF-8 server bug cut short at 68 of 200 items is the worked example, and
it is why the table carries a **NOT QUOTABLE** flag rather than only "(partial)".

MMLU-Pro's rows are ordered by category. The harness draws a *random* subset precisely so that a
prefix cannot be mistaken for the benchmark — but an interrupted run turns that random subset back
into a prefix, and the 68 items that scored covered **5 of 14 categories**:

| covered | count | missing entirely |
|---|---:|---|
| law | 26 | math, physics, engineering, computer science, |
| business | 17 | economics, health, history, philosophy, other |
| biology | 15 | |
| psychology | 9 | |
| chemistry | 1 | |

67 of 68 items are law / business / biology / psychology. **There is no math and no physics in it.**
The 54.4 % is a true statement about those 68 items — it re-derives from the stored generations —
but it is not an estimate of MMLU-Pro, and the direction of its bias is not something a confidence
interval can express. `tools/eval_suite.py` now computes stratum coverage for every task and
`tools/eval_publish.py` marks any task that has not touched every stratum as NOT QUOTABLE, so this
cannot be published as a capability number by inattention.

## Protocol

Published in full rather than by reference to a harness, because comparability is set by
the protocol and not by the harness name.

<!-- PROTOCOL -->
| benchmark | split / subset | scenario | answer extraction | scoring | execution | budget | samples |
|---|---|---|---|---|---|---:|---:|
| GPQA-Diamond | test, all 198 | 0-shot generative CoT | final letter A–D | exact letter match | none | 8000 | 1 |
| MMLU-Pro | test, seeded random subset of 12 032 | 0-shot generative CoT | final letter A–J | exact letter match | none | 8000 | 1 |
| MATH-500 | test, seeded random subset of 500 | 0-shot generative CoT | last brace-balanced \boxed{} | string → numeric → sympy equivalence | none | 8000 | 1 |
| HumanEval | all 164 | 0-shot | first ```python block | pass@1 — the benchmark's own check(entry_point) | sandboxed subprocess, 20 s, 2 GiB address space | 8000 | 1 |
| BFCL v3 — 4 `exec_*` categories, AST † | 4 of BFCL v3's `exec_*` categories, 240 items | prompt mode — calls emitted as text, not via a tool-call API | Python call expressions, one per line | BFCL's own AST metric: same function, argument names and values | none — AST comparison, not execution | 2000 | 1 |
| BFCL v3 — `live_*` categories, AST † | 6 `live_*` categories, seeded cap of 150 per category | prompt mode — calls emitted as text | Python call expressions, or the literal NO_CALL | BFCL possible-answer AST match; relevance categories scored on whether a call was emitted at all | none — AST comparison, not execution | 2000 | 1 |
| LiveCodeBench — `test6` window † | `code_generation_lite` `test6`, window 2025-01 → 2025-04 | 0-shot, complete program | first ```python block | all tests pass (public + private) | sandboxed subprocess, 8 s per test, 2 GiB address space | 8000 | 1 |
| SciCode — subproblem pass rate | test split, 65 research problems → 291 subproblems, 16 science subfields | 0-shot per subproblem; the model's OWN earlier steps are in scope, no gold solutions exist | first ```python block | the benchmark's own numeric tests, all must pass | sandboxed subprocess, 90 s, 4 GiB address space | 8000 | 1 |

**Decoding.** `low` reasoning effort at temperature 1.0, top_p 0.95. Sampling is stochastic rather than greedy, which is why interval width and avg@k are reported rather than a bare point estimate. Per-benchmark token budget and sample count are in the table above; each budget was set from an uncensored length calibration, not chosen.

**Intervals.** Single-sample tasks use a Wilson score interval. avg@k tasks use a nested bootstrap over problems: k samples of one problem are not k independent problems, and pooling them into a Wilson interval understates the width by up to 2x here — most severely when the extra samples bought the least. This is the clustering correction of Miller, *Adding Error Bars to Evals* (arXiv:2411.00640), which reports cluster-adjusted standard errors up to 3x the naive ones.

**Deviations from the canonical protocol.** Each is stated with the direction it moves the score.

- **MMLU-Pro**
  - a seeded random subset, not the full split — unbiased in expectation, but wider than the published interval on the full set
- **MATH-500**
  - a seeded random subset, not the full 500
- **BFCL v3 — 4 `exec_*` categories, AST †**
  - runs 4 `exec_*` categories (240 items), **not** the BFCL v3 aggregate, which also spans multi-turn, relevance/irrelevance detection and non-Python languages — **this row is not the BFCL leaderboard quantity**
  - scored by AST match rather than live execution, so a call that is structurally right but fails against a real API counts correct (biases **up**)
- **BFCL v3 — `live_*` categories, AST †**
  - a seeded cap of 150 items per category, because the `live_*` sets run from 1052 rows (`live_multiple`) to 16 (`live_parallel`) and taking all of them would let two categories set the headline
  - category-balanced rather than BFCL's own weighting, so this is **not** the BFCL leaderboard aggregate
- **LiveCodeBench — `test6` window †**
  - at most 25 tests per problem, so a solution failing test 30 of 200 counts correct (biases **up**)
  - single sample per problem, not the official multi-sample pass@1 average
  - the `0731` checkpoint postdates every problem in the window, so contamination is reduced but **not** eliminated (biases **up**)
- **SciCode — subproblem pass rate**
  - subproblem pass rate is the headline; main-problem accuracy (every subproblem of a problem correct) is the harder published metric and is derived from the same records rather than reported in its place
  - later steps run against the model's OWN earlier code, since no gold solutions ship — a wrong early step propagates, which is the intended behaviour and biases **down** relative to a gold-context setting

**What this protocol does and does not license.** Every number here is re-derivable from the stored generations by `tools/eval_verify.py` with no GPU and no model, against a sha-pinned dataset. That makes the numbers *reproducible*. It does not make them *leaderboard-rankable*: the reference column is third-party aggregation whose harness, token budget and sampling count are not ours and are not stated, so a gap of a few points between a row and its reference is not interpretable in either direction. The comparison this evidence genuinely supports is a paired one — REAP against unpruned, same harness, same decoding, only the weights different — and that baseline has not been run here.
<!-- /PROTOCOL -->

## Evidence, and how to check these numbers without trusting this repo

Every result is landed the moment its benchmark finishes -- verified, committed and pushed as its
own commit -- rather than assembled at the end. A battery interrupted after four of seven tasks
leaves four complete, independently re-derived results in `main` and no half-written table.

**Nothing is published that has not been re-derived from the raw text.** `tools/eval_verify.py`
ignores the process that produced the number and rebuilds it:

1. re-reads every stored generation from `evidence/evals/<task>.jsonl`;
2. re-extracts the answer from that text, ignoring the `got` field on the record;
3. re-derives the gold **from the pinned dataset by item id**, not from the `gold` field on the
   record, so a corrupted or hand-edited gold cannot survive;
4. re-scores and requires the recomputed accuracy to equal what the table publishes.

It also fails on duplicate ids, ids that are not in the benchmark, records with neither generated
text nor a recorded engine error, and `correct` flags that disagree with a fresh scoring of the same
text. `scripts/eval_land.sh` refuses to commit if any of that fails.

Run it on a clone. **No GPU, no model, no network:**

```bash
python3 tools/eval_verify.py                 # re-derive every published number
python3 tools/eval_provenance.py             # re-check the published facts about each dataset
```

### What is in the repo for each benchmark

| file | what it proves |
|---|---|
| `evidence/evals/<task>.jsonl` | one record per item: the **full** generation and reasoning trace, the extracted answer, the gold, `finish_reason`, truncation, token usage and the engine's own timings. Every number traces to the text that produced it. |
| `evidence/evals/<task>.meta.json` | dataset, **pinned snapshot sha**, sampling parameters, `max_tokens`, reps, start time |
| `evidence/evals/datasets.json` | the commit sha of every benchmark repo, so the exact rows can be re-fetched |
| `evidence/evals/provenance.json` | each published structural fact about each dataset, checked against the bytes on disk, with its source cited |
| `evidence/evals/preflight.log` | the go/no-go that permitted the battery to start |
| `evidence/evals/verification.json` | the independent re-derivation of the published table |
| `evidence/evals/run.log` | the live run, item by item, with per-item throughput |

A benchmark that aborted appears as **absent**, never as complete: `eval_land.sh` will not land a
task with no records, which is why a failed GPQA run leaves a visible hole rather than a zero.

## What was excluded, and why

Exclusions are part of the report. Nothing below was skipped because it was expected to score badly.

| benchmark | why not run |
|---|---|
| Terminal Bench 2.1, DeepSWE, SWE-bench | agent rollouts: container per instance, repo checkout, multi-turn tool loop, and contexts far past 4096. Not a harness gap — a machine gap. |
| NL2Repo, CyberGym, Toolathlon, DSBench, AutomationBench | same; several also need live external endpoints. |
| LiveCodeBench | tractable in principle and it is one of the three reference numbers, but the release ships problems as versioned `.jsonl.gz` with hidden tests and a scoring harness of its own; running a hand-rolled approximation would produce a number not comparable to the 88.40 it would be printed next to. Deliberately left undone rather than approximated. |
| IFEval | the official score depends on ~25 instruction-verifier classes in `instruction_following_eval`, which is not installed here. A reimplementation would be a different benchmark wearing the same name. |
| MMLU (original) | superseded by MMLU-Pro, which is what the reference column has. |
| Any judge-scored benchmark (Arena-Hard, MT-Bench, AlpacaEval) | needs a grader model; rule 1. |

## Reproducing

```bash
python3 tools/eval_fetch.py                      # pin datasets, writes their commit shas
python3 tools/eval_provenance.py                 # assert the published facts about each dataset
nohup setsid bash scripts/memguard.sh &          # kill the server before the kernel kills the box
SEQMAX=8192 EXT_CHUNK=64 bash scripts/serve.sh & # see the context section for why 8192 and not more
python3 tools/eval_preflight.py                  # MUST print GO before anything below is run
bash scripts/run_evals.sh                        # resumable; skips ids already scored
python3 tools/eval_suite.py --report && python3 tools/eval_publish.py
```

`eval_preflight.py` is not a formality. The first attempt at this battery failed in three ways that
a run log reported as **success** -- GPQA-Diamond abandoned on item 0 and printed "done" with zero
items scored; AIME 2025 problem 19 carrying its gold as `336^\circ` so every model answering `336`
was marked wrong; and AIME quoted from a single sample per problem, an interval of +-15.6 points.
None of the three raised an error and all three would have produced a clean table to publish. The
preflight now checks SERVER / PROVENANCE / SCORERS / CONTEXT / POWER / MEMORY / LIVE, with CONTEXT
verifying **every item of every task** against exact token counts from the checkpoint's own
tokenizer plus the chat-template overhead measured off the running server.

Hardware: Jetson AGX Thor, `sm_110a`, 20 SMs, 122.8 GiB unified LPDDR5X, clocks pinned with
`jetson_clocks`. Serving throughput during these runs is reported per task in the `tok/s` column and
is the engine's own counter, not a wall-clock estimate.
