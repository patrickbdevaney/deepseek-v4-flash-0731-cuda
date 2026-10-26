# Implementation plan: an eval run that cannot fail silently

Answers `EVAL_RESEARCH_PROMPT.md`. The standard is not "we were careful" — it is **name the failure,
then show the check that catches it**, with the check runnable before the battery starts.

## 1. What the research established, and what changed because of it

Primary sources: `openai/simple-evals` (`gpqa_eval.py`, `common.py`), `EleutherAI/lm-evaluation-harness`
(`minerva_math`), the originating papers, and each benchmark's own repo.

| finding | source | what this harness did | now |
|---|---|---|---|
| GPQA prompt is a fixed template ending `'Answer: $LETTER'` | simple-evals `common.py` | own wording | **verbatim template** |
| extraction is `(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?` | same | own looser regex | **reference pattern first**, looser ones as additive fallbacks |
| failed extraction scores **0.0** | `gpqa_eval.py` | scored wrong | unchanged ✓ |
| GPQA options are shuffled per repeat; `n_repeats=4` | `gpqa_eval.py` | single pass, pre-shuffled mirror | **disclosed**, see §4 |
| MATH equivalence is **symbolic** (sympy), not string | lm-eval `minerva_math` | string+float only | **tiered: string → numeric → sympy** |
| GPQA(simple-evals) and GPQA(lm-eval) are **non-comparable** | simple-evals README | treated as one number | **disclosed as a bound on the comparison** |
| AIME convention is avg@k, k=16–64 | benchmark leaderboards | single pass | **reps=4** (the least quotable) |

The MATH change matters most: a string-only comparator is **stricter** than the published protocol, so
it marks correct answers wrong. `\frac{1}{2}` vs `0.5` and `2\sqrt{2}` vs `\sqrt{8}` now both pass.

## 2. The failure that invalidated the last run, and the fix

**Truncation was not a nuisance, it was the result.** Measured on the aborted battery:

| | truncated | acc if truncated | acc if completed |
|---|---:|---:|---:|
| GPQA-Diamond | 48 % | 23.1 % | **95.2 %** |
| MMLU-Pro | 37 % | 8.0 % | **81.4 %** |

Truncated items score **at chance** (25 % / 10 % random) — they are absent answers, not wrong ones.
The published number was a coin-flip average whose weight was set by `max_tokens`.

Fixed by raising the context ceiling, which required two engine changes, both gated:

- **arena sized by batch, not context** (shipped earlier): seqmax 4096 → 8192, bit-identical.
- **xin ring buffer** (this change): seqmax 8192 → **32768**, bit-identical.

Budgets are now 8000–16000 tokens instead of 2000–4500.

### The ring, and the bug the gate caught

`xin` is the compressor's attention-input history, `[seqmax, DIM]` fp32 × 41 layers = **656 KiB per
token of context**, 6.6× the entire MLA+DSA KV cache. After separating `x_cur` (this block's input,
which the Q/KV build and indexer actually read) from `x_full` (history), the *only* reader of the
history is `compressor_emit_group`, which never looks back further than `2*ratio`. So it can be a ring.

**The first sizing was wrong and `gate_engine` caught it.** A verify step writes its whole K-row batch
*before* emitting any group, so the live window is the batch **plus** the lookback:

    R >= K + 2*ratio          not   R = 2*ratio

At `2*ratio` (8 rows) a 64-row batch lapped the ring eight times before the first group was read.
The engine still produced fluent text and passed the single-prompt probe; the gate caught it on the
cold-vs-cached comparison, where cold had moved to a different argmax (6932/15 vs baseline 9544)
while the cached path — which never batches — stayed correct. That asymmetry is the signature of a
history clobbered by its own batch. The geometry now lives in one place (`include/compressor.h`) so
the allocator and the indexer cannot disagree, and an over-wide batch aborts instead of corrupting.

**Validation: `gate_engine` 8/8 with the ring on, and ring-on vs ring-off is bit-identical** — every
token, every margin, every `head|d|`.

## 3. Failure modes, and the check that catches each

| # | failure | caught by |
|---|---|---|
| 1 | dataset is a lookalike (wrong split, shifted key) | `eval_provenance.py` — 20 published facts asserted against the bytes; GPQA gold cross-checked against a second independent mirror (196/198) |
| 2 | scorer is broken | 164/164 HumanEval canonical solutions; 500/500 MATH-500 gold-vs-gold; extraction cases |
| 3 | an item cannot physically fit the context | `eval_preflight.py` CONTEXT — **every item of every task**, exact tokenizer counts + measured template overhead |
| 4 | server throws mid-run | non-streaming handler now catches, reports and logs; `dump_lossy` for invalid UTF-8 from byte-level BPE |
| 5 | one bad item destroys a benchmark | item scored incorrect + recorded, run continues; hard stop past 10 % errors |
| 6 | partial run published as complete | `eval_land.sh` refuses to land a task with no records |
| 7 | **partial run is biased, not just noisy** | stratum coverage computed per task; **NOT QUOTABLE** unless every stratum is touched |
| 8 | sample too small to mean anything | POWER check — Wilson half-width per task, AIME forced to reps=4 |
| 9 | records from different configs mixed | configs archived on change (`archive_budget4k/`); meta.json records `max_tokens` |
| 10 | box OOMs / reboots | `memguard.sh`, floor tuned to 1500 MB against a 2.4–3.4 GiB steady state |
| 11 | published number is not reproducible | `eval_verify.py` re-derives from raw text, re-deriving gold from the pinned dataset |
| 12 | truncation silently depresses the score | detected by token count (`finish_reason` lies), scored wrong not dropped, reported per task |

## 4. Limitations that are disclosed, not fixed

Honesty is part of the deliverable; these cannot be engineered away here.

1. **GPQA at `n_repeats=1`, not 4.** simple-evals defaults to 4. At 198 items reps=1 gives ±5.6; reps=4
   would give ±2.8 and cost 4× of an already ~10-hour task. Disclosed, not hidden.
2. **Option order is the mirror's, not per-repeat shuffled.** GPQA scores are order-sensitive.
3. **Protocol is not byte-identical to any one published harness**, and simple-evals' own README says
   GPQA(simple-evals) and GPQA(lm-eval) are non-comparable. Any head-to-head carries that bound.
4. **The reference numbers are third-party and unattributed** — the official card publishes only agent
   rollouts, run with an unreleased harness.
5. **Contamination is unmeasured** on AIME 2024, GSM8K, HumanEval.
6. **Subsets are subsets**: MMLU-Pro, MATH-500 and GSM8K are deterministic random draws, reported as
   n-scored-of-n-in-benchmark with Wilson intervals.

## 5. Execution order

1. `eval_provenance.py` → 20/20
2. `gate_engine` with the ring → 8/8, and ring-on == ring-off bit-identical
3. server at `SEQMAX=32768 EXT_CHUNK=64`, `memguard.sh` running
4. `eval_preflight.py` → **GO** (29 checks; battery refuses to start otherwise)
5. `run_evals.sh`, with `eval_watch.sh` landing each benchmark as it completes — verify, publish,
   commit, push
6. `eval_verify.py` on every landed task; `EVALS.md` regenerated from `summary.json`, never by hand
