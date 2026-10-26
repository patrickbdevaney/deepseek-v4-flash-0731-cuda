# Why these evaluations exist, and what would falsify the claim

## The north star

`DeepSeek-V4-Flash-0731-REAP` is a pruned checkpoint. The entire question this evaluation
programme exists to answer is:

> **Is the REAP a reliable, minimally lossy stand-in for the unpruned model — close enough in
> practice, with negligible capability difference — that it can be trusted as a local SOTA option on
> Jetson Thor, GB10 and DGX Spark?**

Not "does it produce fluent text." Not "does it beat a 7B." The claim under test is *fidelity*: that
pruning removed parameters without removing capabilities, and that what remains is trustworthy for
production agentic coding, world-knowledge reasoning, and general near-frontier work.

That claim is currently **untested**, and it is the only reason any of this is being run.

## The value proposition being defended

A local deployment on one Thor will never match a frontier provider's GPU cluster on tokens per
second or on parallel request cascades. Competing on speed is a lost argument. What a local
deployment *can* offer is **correctness and intellectual sophistication per completion**:

> One slower, high-quality, correct completion or agentic turn — rather than fast, intellectually
> dim, narrow responses from a smaller model that gets stuck in reasoning loops or writes coarse,
> naive implementations.

This is why decode throughput is not the headline metric here and capability is. A local model
earns its place by being *right* and by being *deep*, on the first try, without supervision. The
comparison class is Opus 5, Fable 5, GPT-5.6 Sol, DeepSeek V4 Pro 0831, GLM 5.2 and Kimi K3 —
not on latency, but on whether the answer is one you can build on.

## Why both legs of the suite are load-bearing

An early draft of this analysis treated the world-knowledge benchmarks as a *construct-validity gap*
— as though only the agentic benchmarks measured the real target. **That was wrong**, and the
correction is the organising principle of the whole suite:

> A really good coder that knows nothing about business, marketing or biology — that cannot
> generalise beyond writing good code and tool calls — is not very useful.

Domain breadth is not adjacent to agentic software engineering; it is a **precondition** for it. An
agent writing a bioinformatics pipeline, a billing system or a simulation must reason correctly
about the domain before any of its code or tool calls matter. So the suite deliberately spans:

| leg | what it establishes | benchmarks |
|---|---|---|
| **World knowledge & reasoning** | the model is smart enough to work in the domain at all | GPQA-Diamond, MMLU-Pro |
| **Novel synthesis** | it can derive, not just recall | AIME, MATH-500 |
| **Code** | it can express a solution correctly | HumanEval, LiveCodeBench |
| **Tool use** | it can drive external systems, and knows when not to | BFCL `exec_*`, BFCL `live_*` |
| **All of it at once** | knowledge → reasoning → working code, in one item | **SciCode** |

SciCode is the keystone. It is the only benchmark here where a single item requires recalling a
scientific fact, reasoning about it, and turning it into numerically correct code. A checkpoint
pruned into knowing less physics fails there in a way it cannot fail on HumanEval. If the REAP has
lost domain breadth, SciCode is where it shows up first.

`live_irrelevance` is load-bearing for the opposite reason: it scores **declining to act**. A model
that fires a plausible-looking tool call at every prompt scores well on every other tool benchmark
and is dangerous driving a terminal.

## What would falsify the claim

Stated in advance, so the result cannot be reinterpreted after the fact:

- **A large gap to the unpruned reference on knowledge tasks** (GPQA, MMLU-Pro) means pruning cost
  domain coverage. Given the intervals here, "large" means **more than ~7 points** — smaller gaps
  are not resolvable by this design and must not be read as either pass or fail.
- **SciCode disproportionately below the coding benchmarks** would be the specific signature of
  REAP damage: code generation intact, the domain knowledge feeding it degraded.
- **High truncation with low terminating accuracy** would indicate reasoning-loop pathology — the
  exact failure mode the value proposition claims to avoid. This is measured directly: GPQA's
  terminating traces score 91.5 % against 10.5 % for truncated ones.
- **Weak `live_irrelevance`** would mean the model cannot decline, disqualifying it for autonomous
  tool use regardless of its other scores.

## Honest critique of this methodology

Written adversarially, against my own work, so a reader does not have to reconstruct it.

### What is genuinely sound

- **Every number is re-derivable.** `tools/eval_verify.py` re-reads the stored generations,
  re-extracts answers from raw text, re-derives golds from sha-pinned datasets rather than from the
  records, and re-scores. No GPU, no model, no trust in the process that produced it.
- **Clustering is handled.** avg@k uses a nested bootstrap over problems. Pooling k samples of one
  problem into a Wilson interval understates width by up to 2.07× here, and understates it *most*
  when the extra samples bought the least. This matches Miller (arXiv:2411.00640), which reports
  cluster-adjusted errors up to 3× naive.
- **Budgets come from uncensored measurement.** The original 8000-token cap came from reading a
  *censored* statistic — the maximum of the terminating traces, which is bounded by the budget by
  construction and always looks reassuring. It produced 31.7 % truncation. Budgets are now set from
  probes run at a deliberately generous ceiling.
- **Truncated rows cannot be published as capability.** Past 5 % truncation a row is marked NOT
  QUOTABLE, because a truncated item is scored wrong and the number becomes a measurement of
  `max_tokens`.
- **Deviations are disclosed with their direction.** A limitation whose sign is unknown has not
  really been disclosed.
- **Untrusted code is sandboxed.** Model-written code runs under an address-space cap in its own
  process group. Verified: a 40 GB allocation dies in 0.1 s instead of taking the box into swap.

### What is weak, and cannot be fixed by more care

1. **No paired baseline.** This is the largest gap. The claim is *REAP ≈ unpruned*, and the only
   design that tests it directly is the same harness, same decoding, same prompts, differing only in
   weights. The reference column is instead third-party aggregation whose harness, token budget and
   sampling count are neither ours nor stated. **A few points of difference is therefore
   uninterpretable in either direction.** This is deferred deliberately, not overlooked — running
   the unpruned model is out of scope for the local-only phase.
2. **Underpowered for small differences.** n=198 gives ±7 points; MMLU-Pro at 150 gives ±8; AIME at
   30 gives ±16. The design can detect gross degradation and cannot resolve the "negligible
   difference" case that the north star actually asserts. It can refute the claim loudly; it can
   only *fail to refute* it quietly.
3. **Contamination is not controlled.** Only LiveCodeBench attempts date windowing, and the `0731`
   checkpoint postdates its entire window. GPQA (2023), MMLU-Pro (2024) and AIME 24/25 all predate
   the checkpoint. Absolute numbers are upper bounds. This applies equally to the published figures
   being compared against, so it does not break the comparison — but none of these are clean
   capability measurements.
4. **Subsets are not the published quantities.** BFCL here is category-balanced across a capped
   subset, not BFCL's own aggregate weighting; LiveCodeBench caps tests per problem and samples once.
   Both are labelled in the results table, and neither is the leaderboard number.
5. **No long-horizon agentic benchmark.** Toolathlon needs 32 live applications; τ-bench needs a
   user-simulator LLM; SWE-bench Verified needs ~50M tokens at this throughput (≈55 days). All were
   assessed and ruled out on feasibility, not on relevance. Agent scores also swing with the
   scaffold rather than the model, so a home-built scaffold would measure our harness as much as the
   checkpoint. **The tool-use leg is therefore measured single-turn only** — this is the most
   substantive remaining hole in coverage.
6. **Low reasoning effort.** The current battery runs the checkpoint's default effort. It is a real
   operating point, and clearly labelled, but it is not the model's ceiling.

### Is this the right set of tests for the stated use profile?

**For the knowledge, reasoning, synthesis and code legs: yes.** GPQA and MMLU-Pro are the standard
knowledge measures; AIME and MATH-500 cover derivation; LiveCodeBench is contamination-windowed and
unsaturated; SciCode uniquely tests the knowledge→code path that the use profile depends on.
HumanEval is saturated industry-wide and is retained as a sanity check, not as signal.

**For the agentic leg: partially.** BFCL `exec_*` plus `live_*` measures single-turn tool selection,
argument correctness and restraint — necessary, and genuinely informative about whether the model
can be trusted with tools. It does not measure multi-step orchestration, recovery from a failed tool
call, or long-horizon state tracking, which is what "agentic engineering" means in practice. Closing
that requires either a stateful BFCL multi-turn implementation (the dataset ships schemas but not
the backend classes the official scorer executes against) or a containerised agent benchmark. Both
are real work and neither fits the current wall-clock budget.

**Verdict.** This suite can establish that the REAP is broadly intact across knowledge, reasoning,
math, code and single-turn tool use, with reproducible evidence and honest intervals. It cannot yet
establish the sharper claim — *negligible* difference from unpruned — because that needs the paired
baseline and more items per benchmark. It should be read as a strong screening result, not a
final equivalence proof.

## Runtime, and why it is a methodological constraint

The engine sustains ~10.6 tok/s at long context, giving a hard ceiling near **0.92M completion
tokens per day**. `tools/eval_budget.py` estimates the remaining cost of every queued task from
measured per-item tokens where records exist and from a stated assumption otherwise, printing which,
so a plan resting on a guess is never mistaken for one resting on data.

That constraint is why AIME runs at reps=2 rather than 4, why BFCL `live_*` is capped per category,
and why SWE-bench is absent. These are budget decisions, and they are recorded as such rather than
presented as methodological preferences.

Current plan: **5.60M tokens ≈ 6.1 days**, inside a 7-day ceiling with 0.9 days spare.

## Reproducing any of this

```
python3 tools/eval_verify.py            # re-derive every number, no GPU, no model
python3 tools/eval_budget.py --days 7   # what the remaining plan costs
python3 tools/eval_publish.py           # regenerate the results and protocol blocks
```

Full protocol — split, scenario, extraction, scoring, execution policy, budget, sample count and
every deviation — is in `EVALS.md`.
