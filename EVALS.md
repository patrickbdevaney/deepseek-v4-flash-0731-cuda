# Capability of the REAP K160 checkpoint, as served by this engine

**Status: RUNNING.** Results below are regenerated from `evidence/evals/summary.json`; the "scored"
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

## The binding constraint: 4096 tokens of context

This is the single most important caveat and it is not a small one.

DeepSeek recommends **384K output tokens** for the `high` and `max` reasoning levels. This server
runs at **4096 total** (prompt + completion). The reason is arithmetic, not preference:

| | GiB |
|---|---|
| box unified memory | 122.8 |
| checkpoint resident | 100.4 |
| KV + activations + arena at seqmax=4096 | ~19.7 |
| **reported at ready** | **120.1 / 122.8** |

The scratch arena alone is sized `(512 + 2 x seqmax) MiB`, so seqmax=8192 asks for 16.9 GiB where
4096 asks for 8.5 — an extra 8.4 GiB against 2.7 GiB of headroom.

**And that ceiling is an engine artefact, not an architectural one.** MLA + DSA make this model's
real KV cache **99.4 KiB/token**, which is **3.3 %** of everything that scales with `seqmax`; 128K
of context would be 12.4 GiB of KV, and there is ~22 GiB free. The other 96.7 % is scratch sized to
the wrong quantity — the arena heuristic above (68 %) and `xin`, an attention-input history
allocated at `seqmax` when `kernels/compressor.cu:500` proves no group emit reads further back than
`2 x ratio` <= 128 positions (22 %). See
**`wiki/context-ceiling-is-not-the-kv-cache.md`**: sizing the arena from the high-water mark it
already tracks would reach seqmax ~12,800 in the same memory, and adding an `xin` ring buffer
~40,000, both bit-exact. **The cheapest way to raise the scores in this file is to raise the
context, and that is an allocation change.** Two attempts to serve at a larger
context on 2026-08-12 did not fail gracefully: they took the **whole machine** down mid-load
(reboots at 20:02 and 20:11, `last -x reboot`, no oom-kill line in `dmesg` either time).
`scripts/memguard.sh` now watches MemAvailable and kills the server before the kernel has to decide.

**Consequence for every number below:** a reasoning model that would emit tens of thousands of
tokens of deliberation is capped at a few thousand. Items that hit the cap are counted as attempted
and **wrong**, never dropped. The `trunc` column reports exactly how many. **Every score here is
therefore a floor, and the gap to the reference column is an upper bound on the damage from
pruning** — some unknown part of it is the context ceiling instead.

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
| benchmark | scored | **acc %** | 95 % CI (Wilson) | trunc | mean out tok | tok/s | unpruned 0731 @ high |
|---|---:|---:|---|---:|---:|---:|---:|
| GSM8K | 2/2 | **100.0** | [34.2, 100.0] | 0 | 188 | 20.6 | — *(none published)* |
| AIME 2024 *(partial)* | 1/30 | **100.0** | [20.7, 100.0] | 0 | 634 | 24.0 | — *(none published)* |
| MATH-500 *(partial)* | 1/2 | **100.0** | [20.7, 100.0] | 0 | 93 | 17.2 | — *(none published)* |
| HumanEval *(partial)* | 1/164 | **100.0** | [20.7, 100.0] | 0 | 591 | 19.0 | — *(none published)* |

5 items scored, 1,694 completion tokens generated. Sampling held at `temperature = 1.0`, `top_p = 0.95`, `reasoning_effort = high` throughout.

Per-task provenance (dataset and pinned snapshot):

| benchmark | dataset | snapshot | max_tokens |
|---|---|---|---:|
| GSM8K | openai/gsm8k (main, test, 1319) | `740312add88f` | 1600 |
| AIME 2024 | Maxwell-Jia/AIME_2024 (30) | `8d88b2876a82` | 3400 |
| MATH-500 | HuggingFaceH4/MATH-500 (500) | `6e4ed1a2a79a` | 3000 |
| HumanEval | openai/openai_humaneval (164) | `7dce6050a7d6` | 1600 |
<!-- /RESULTS -->

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
bash scripts/memguard.sh &                       # kill the server before the kernel kills the box
SEQMAX=4096 bash scripts/serve.sh &               # anything larger has taken this machine down
bash scripts/run_evals.sh                        # resumable; skips ids already scored
python3 tools/eval_suite.py --report             # regenerate summary.json
```

Hardware: Jetson AGX Thor, `sm_110a`, 20 SMs, 122.8 GiB unified LPDDR5X, clocks pinned with
`jetson_clocks`. Serving throughput during these runs is reported per task in the `tok/s` column and
is the engine's own counter, not a wall-clock estimate.
