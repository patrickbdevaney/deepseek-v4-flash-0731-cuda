# North star — why this project exists

**Local frontier-adjacent intelligence, on hardware you own, fast enough to do long-horizon agentic
work unattended.**

Everything in `LOOP_LOG.md` is a measurement in service of that one sentence. This document is the
argument for why the sentence is worth 104 findings, and — as importantly — where the argument is
still standing on an assumption rather than a number.

---

## 1. The scarce thing is the combination, not either half

Neither half of this is rare on its own:

* **Frontier-adjacent capability** is widely available — through an API, over a network, metered.
* **Fast local decode** is widely available — on models small enough that the capability is not
  frontier-adjacent.

What is genuinely rare is **both at once, on one box, with no network in the loop.** That is the
whole position, and it is why the project is not simply "make a big model faster" or "run a small
model locally."

**For agentic work specifically, this matters more than the raw token rate.** Agentic loops are
latency-sensitive in a way chat is not: every tool call round-trips, and a twenty-turn loop is
twenty round trips. A local model at ~29 tok/s can beat a faster remote model once you count
network latency, rate limits, and the fact that you can leave it running unattended for hours
without a bill or a quota. The unit that matters is **task completed**, not tokens per second.

---

## 2. REAP is not a compromise against the full model — it is the difference between running and not

`REAP_MANIFEST.json`: `original_experts_per_layer 256`, `kept_experts_per_layer 160`, from
`deepseek-ai/DeepSeek-V4-Flash-0731` at revision `9e165c30`. **37.5 % of the routed experts removed.**

The honest comparison is **not** "REAP vs the full model." It is **"REAP on Thor vs nothing on
Thor."** The engine sits at **109.2 of 122.8 GiB** with the model resident; the unpruned checkpoint
does not fit at all. You are not degrading a model you could otherwise run — you are running one you
otherwise could not.

Expert pruning is the right lever for this trade because it removes parameters that are *sparsely
activated by construction*: at top-6-of-160 any given token touches a small fraction of them, so the
capability cost of removing the least-used is far below the memory saved. That is REAP's premise, and
it is why 122.8 GiB of unified LPDDR5X can hold a model of this class.

---

## 3. The speed target is the realistic one, and the workload is the one this system is best at

Measured, clean, `LOSSLESS GATE` passing (F96, `evidence/baseline_blk6_suite.log`):

| prompt category | tok/s |
|---|---|
| **long_context** | **30.77** |
| **agentic_format** (tool/JSON) | **29.98** |
| **multi_turn** | **28.97** |
| code_edit | 26.81 |
| short_factual | 22.78 |
| reasoning | 14.58 |
| explanation | 13.38 |
| code_gen | 13.97 |
| **suite mean** | **22.66** |

**The four fastest categories are precisely the shapes agentic coding produces**: long context, tool
call formats, multi-turn dialogue, and editing existing code. The three slowest are open-ended
free generation — writing prose from nothing — which is the *least* common operation in an agentic
loop and the one where a drafter has least to work with.

That is not a coincidence and it is not luck. Speculative decoding pays exactly where continuation is
**constrained**, and agentic work is constrained almost by definition: the model is completing a tool
schema, continuing a file, or following a plan it already wrote.

**The ceiling is close and we know where it is.** Base AR runs at **13.76 tok/s against a
byte-weighted realistic floor of 14.33** — **97 %** of what the memory system permits
(`tools/byte_floor.py`). 12.26 GB of weights per token at a measured 233 GB/s is the physics; the
remaining headroom is not in the kernels. Aspirational numbers like 50 or 100 tok/s on this box are
not reachable losslessly at this model size, and saying so is more useful than chasing them.

---

## 4. The invariant that makes "fast" and "frontier" compatible rather than competing

**Every speed gain in this project is lossless: the emitted tokens are identical to base
autoregressive decode.** Speculative decoding is lossless by construction at the verify step, and the
`LOSSLESS GATE` checks it on every run:

```
[spec] LOSSLESS GATE: first 8 tokens match base AR -> PASS
```

This is the load-bearing constraint of the whole project. **The moment something trades output
quality for speed — relaxed acceptance, additional quantisation, top-k reduction — the goal quietly
becomes a different and worse one**, and the "frontier-adjacent" half of the claim evaporates while
the tok/s number goes up.

Finding 68 is why the gate exists: split-K produced a **+28 % speedup that passed every other gate**
and was decoding a different sequence. A cosine check cannot catch that. Only comparing emitted
tokens against base AR can.

The measured price of this constraint is known and is deliberately paid: dynamic top-k alone is worth
**+28 %** and is lossy, so it is not taken. **Constraint 3 costs roughly 2x, and it is the reason the
project's output is worth anything.**

---

## 5. The two roles this unlocks

**As executor.** A frontier-adjacent model doing the work directly — reading a repo, editing files,
running tools — locally, unattended, in a loop, at ~29 tok/s on the shapes that work produces. Slower
per token than a hosted frontier model, and without cloud-scale research cascades, but with no
metering and no network.

**As orchestrator.** The more interesting role, and the one the capability profile suits best: a
long-horizon *conductor* that holds the plan, decomposes tasks, and dispatches them to a smaller
faster model — co-resident on the same 122.8 GiB pool, or on another device. The conductor role is
where world knowledge and reasoning matter most and where token rate matters least, because it emits
few tokens and thinks between them.

---

## 6. The one number in this story that is inherited rather than measured

**The ~50 Artificial-Analysis-class capability figure is a claim about the base checkpoint. This
project has not measured it on the REAP variant.**

Every performance number here is measured on this box. **Capability is not.** It is asserted from the
base model's reputation plus REAP's premise, and those are reasonable priors — but they are priors.

This project has been burned **twice** by inherited numbers:

* the draft head recorded as **210.8 M parameters**, which was FastMTP's MiMo-7B head copied from a
  paper about a different model. The real figure is **12.5 B**, and a full fine-tune planned on the
  wrong number would have failed at allocation *after* the capture was paid for.
* the **19.0 tok/s AR roofline**, quoted for the project's whole life, which turned out to be a
  normalisation constant assuming zero fixed cost and full-bandwidth kernels. The realistic floor is
  **14.33**.

**Before the "frontier + fast" framing is used publicly, close this gap with a real eval on the
pruned checkpoint.** GPQA-style knowledge, a reasoning benchmark, and an agentic benchmark on the
artifact we actually run. It is the last inherited number in the story, and the two precedents say
inherited numbers in this project have a poor record.

---

## 7. The sharpest falsifiable form of the goal

> **The highest-capability model that fits in 122 GiB, running losslessly at a token rate that makes
> unattended agentic loops practical.**

That is testable, it is nearly met, and it does not depend on winning a tok/s comparison against
hardware in a different class.

**Status against it:** the speed half is substantially achieved and near its measured ceiling — the
kernel queue has been declared exhausted twice, on independent evidence (F83, F98). The capability
half is asserted. Remaining work is refinement (acceptance via the draft-head fine-tune) plus the one
measurement that would let the claim be made honestly.
