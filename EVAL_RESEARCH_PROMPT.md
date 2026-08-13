# Research brief: what makes each benchmark's result *the* result, and what makes this run unable to fail

This is the prompt that drives the validation work in `EVAL_PLAN.md`. It is written down rather than
held in someone's head because the failure mode it exists to prevent is specific and has already
happened four times on this project: **a run that produces a clean table of numbers that are wrong,
and reports success while doing it.**

Every failure so far shared one shape. None raised an error.

| what happened | how it presented |
|---|---|
| GPQA-Diamond aborted on item 0, an uncaught `type_error.316` on invalid UTF-8 from byte-level BPE | run log said `gpqa_diamond done`; 0 of 198 scored |
| MMLU-Pro cut off at 68/200 by the same bug | a publishable-looking `54.4 %` — over 5 of 14 categories, no math, no physics |
| AIME 2025 problem 19 gold stored as `336^\circ` | every correct `336` marked wrong; 3.3 points |
| AIME scored from one sample at temperature 1.0 | ±15.6 point interval quoted as a number |
| 48 % of GPQA items truncated at `max_tokens 4000` | truncated items score **at chance** (23.1 % vs 25 % random); the published number becomes a coin-flip average whose weight is set by the token budget |

So the brief is not "find the benchmarks". It is: **for each benchmark, establish what the canonical
protocol actually is, then prove this harness and this engine implement it, then prove the run can
survive to completion.**

---

## Part A — Per-benchmark canonical protocol

For each of **GPQA-Diamond, MMLU-Pro, AIME 2024/2025, MATH-500, HumanEval, GSM8K**, establish from
primary sources (the originating paper, the official repo, and the harness most published numbers
are produced by — `lm-evaluation-harness`, `simple-evals`, or the benchmark's own):

1. **Dataset identity.** Canonical repo and split. Exact item count. For GPQA, which subset
   (Diamond = 198) and whether options are pre-shuffled or shuffled by the harness. Are we reading
   the same rows as the published number?
2. **Prompt protocol.** Zero-shot vs few-shot; if few-shot, how many and from where. CoT or direct.
   The exact answer-format instruction. **Does the reference use a system prompt?** A protocol
   mismatch is worth several points and its sign is not knowable without checking.
3. **Answer extraction.** The exact rule — regex, `\boxed{}`, last-letter, a parser. What the
   reference does when extraction *fails*: score wrong, retry, or drop? (Dropping inflates; we
   score wrong.)
4. **Equivalence checking.** For MATH-500 especially: is it string equality after normalisation, or
   symbolic (sympy) equivalence? A normaliser that is stricter than the reference understates.
5. **Sampling and repeats.** Temperature, top_p, and **how many samples per item**. For AIME the
   convention is avg@k with k=16–64; quoting a single pass is not the same measurement.
6. **Token budget.** *The one this project is failing on.* What `max_tokens` do published
   evaluations of **reasoning** models give? How do they handle a generation that does not
   terminate? Is a truncated item scored wrong, retried, or excluded?
7. **The reference number itself.** Who produced it, at what reasoning effort, with which harness.
   Is it reproducible by a third party at all?

## Part B — Reasoning-model-specific hazards

8. **Budget sensitivity.** Published practice for models that emit long CoT: is there a standard
   budget, or is it model-specific? What is reported when a model is budget-limited — and is it
   legitimate to publish a score at a budget below the model card's recommendation, if disclosed?
9. **Degenerate trajectories.** Reasoning models loop. Does the reference protocol detect and
   handle repetition, or does it simply let the budget expire? (Observed here: the token
   distribution is bimodal — ~1100 tokens or blows past 4000.)
10. **Effort levels.** This model exposes `low`/`high`/`max`. Reference numbers exist per level.
    Which level is the honest comparison for a given budget, and can a lower-effort result be
    published against the lower-effort reference?
11. **Non-determinism.** At temperature 1.0, how many samples does a stable benchmark number need?
    Relate to the Wilson interval already computed per task.

## Part C — Engine and harness fitness (codebase audit, not literature)

12. **Context.** Can the engine give every item a budget large enough that truncation is rare rather
    than dominant? Quantify the achievable `seqmax` and the resulting truncation rate.
13. **Survivability.** Enumerate every way a 30-hour run can die: server exception, OOM/reboot,
    client timeout, disk, a single pathological item. For each: does it stop the run, corrupt the
    result, or get absorbed? Anything that can silently truncate a benchmark is a defect.
14. **Resumption correctness.** After any interruption, does resuming produce the same result as an
    uninterrupted run? Specifically: are records from a *different configuration* (different
    `max_tokens`, different binary) prevented from mixing with new ones?
15. **Stratum integrity.** For subsampled benchmarks, is the scored set representative — and is a
    partial run detectable as biased rather than merely small?
16. **Scorer correctness.** Proven against ground truth the scorer cannot influence (canonical
    solutions, gold-vs-gold identity, cross-mirror agreement).
17. **Reproducibility from artefacts alone.** Can a third party with no GPU re-derive every published
    number from committed files?

## Part D — The deliverable

A plan in which **every named failure mode above is either eliminated by construction or detected
before the battery starts**, with:

- the pre-flight check that would have caught each of the five historical failures,
- an explicit statement of what is *still* a limitation and must be disclosed rather than fixed,
- and the smallest set of engine changes that makes the run survivable, each one gated.

The standard is not "we tried hard". It is: **name the failure, then show the check that catches it.**
