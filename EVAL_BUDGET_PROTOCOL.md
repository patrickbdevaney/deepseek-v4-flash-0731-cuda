# Choosing max_tokens: what this programme cost, and what to do next time

Measured on the DSV4-Flash-0731 low-effort battery, FP32 KV, seqmax 32768. Numbers from
`evidence/evals/*.low.jsonl`; reproducible with no GPU.

## What the staged protocol actually cost

Run at 8000 -> extend truncated traces to 24000 -> budget-force the residue.

**Staging is nearly free in decode.** Total decode is identical between "run at 24k" and "run at 8k
then extend": both are the sum over items of min(L, 24000), because the extension continues from
the stored prefix rather than redrawing. The only surcharge is re-prefilling those prefixes:

| task | truncated | re-prefill |
|---|---:|---:|
| lcb | 96 | 4.6 h |
| scicode | 54 | 2.7 h |
| gpqa_diamond | 51 | 2.4 h |
| mmlu_pro + humaneval + math500 | 25 | 1.1 h |
| **total** | | **10.7 h** |

Ceiling, not a point estimate: the prefix cache (hit ratio 0.234 over the battery) may absorb part.

**The real inefficiency is elsewhere.** Of 85.2 h of base decode, **54.8 h (64%) went into traces
that hit the cap and were therefore scored wrong**. lcb alone is 89.7% of its own decode. No choice
of budget fixes this, because the mass that overruns 8k largely does not terminate at 24k either.

## Why you cannot probe the right budget cheaply

Completion-token quantiles among items that TERMINATED — i.e. everything an 8000-token run can see:

| task | q50 | q90 | q95 | max | unseen |
|---|---:|---:|---:|---:|---:|
| gpqa_diamond | 1262 | 5836 | 6900 | 7938 | 25.9% |
| lcb | 1385 | 5147 | 6902 | 7888 | 59.3% |
| scicode | 1710 | 5831 | 7045 | 7612 | 18.6% |

The survivor distribution is pressed flat against the cap. **The budget you need is not identifiable
from a run at a budget that is too small**, at any sample size. A probe must be uncensored, which
means running at the ceiling.

`tools/eval_preflight.py` does not provide this signal and should not be mistaken for it: its
`LIVE <task>: longest item round-trips` check is n=1 and selects the longest *prompt*, not the
longest generation. It is a plumbing check.

## A pilot works, and its power matches the stakes

2000 bootstrap pilots per size, classifying "does this row clear the 5% truncation gate?":

| task | true | n=20 | n=30 | n=50 | n=80 |
|---|---:|---:|---:|---:|---:|
| math500 | 1.0% | 98.0% | 97.3% | 98.8% | 99.8% |
| humaneval | 3.7% | 84.2% | 70.1% | 72.8% | 83.2% |
| mmlu_pro | 12.0% | 70.2% | 88.7% | 96.0% | 96.8% |
| scicode | 18.6% | 90.8% | 98.5% | 99.5% | 100.0% |
| gpqa_diamond | 25.9% | 98.0% | 99.8% | 100.0% | 100.0% |
| lcb | 59.3% | 100.0% | 100.0% | 100.0% | 100.0% |

n=30 settles every task that is far from the gate. It fails on humaneval — which sits at 3.7%
against a 5% gate, where being wrong costs nothing. humaneval's non-monotonicity in n is threshold
discreteness (at n=30 the gate means <=1 truncation), not sampling noise.

## Protocol for the next programme

1. **Pilot ~40 items per task at the ceiling budget**, uncensored. Random order, not the first 40.
2. Read empirical quantiles. Pick the smallest B with P(L > B) <= gate.
3. If P(L > ceiling) > gate the row can never clear on budget alone: **plan to force from the
   start and skip the extension for that task.**
4. Run the remaining items at B. **Retrospectively censor the pilot traces to B** — truncate the
   stored text and re-score — so the row is homogeneous. Free, and it keeps the pilot items as real
   data rather than overhead.
5. No extension pass. Forcing only where step 3 flagged it.

## What the staged protocol bought that a single 24k run would not

Worth choosing deliberately rather than inheriting:

* an accuracy-vs-thinking-budget sensitivity curve, from three nested comparable views
* complete publishable 8k numbers days before the programme ended
* no exposure to an unknown-depth OOM at the memory ceiling, since depth grew in stages
* unbiasedness by construction: prefix continuation recomposes the target distribution exactly,
  whereas re-running the truncated subset would put fresh mass below the old cap

## Cost of the extension on THIS run, by task

Ceilings; assumes every truncated trace uses all 16000 extra tokens. Predictions are the
right-censored log-normal fit and are parametric — the distributions look bimodal, which
log-normal does not model, so treat them as optimistic.

| task | cost | predicted @24k | outcome |
|---|---:|---:|---|
| mmlu_pro | 5.5 h | 2.4% | clears the gate |
| scicode | 19.0 h | 3.5% | clears the gate |
| gpqa_diamond | 17.5 h | 9.1% | still forced |
| lcb | 39.4 h | 37.8% | still forced |

57 of those 81 hours go to rows that need forcing regardless. That is a defensible spend **only
because unattended wall time on this box is cheap** — forcing after 24k of thinking yields a better
number than forcing after 8k. If wall time ever becomes the binding constraint, skip the extension
for any task step 3 would have flagged, and force it directly.
