# PHASE2_PLAN.md — the programme after the kernel ladder

Written 2026-08-20, while decode ladder item 1.11 was closing. Successor to `S5_RECIPE.md` (which
remains correct on the training mechanics) and to `DECODE_ZENITH_FINDINGS.md` Phase 2. Every number
below is cited to the finding that measured it. Nothing is asserted.

Scope: what to do when the kernel ladder stops — the draft-head programme, the trace corpus it
needs, the runtime levers that compose with it, production serving quality, and the eval battery
that turns all of it into quotable numbers.

---

## 1. The reframe: `tau` is not a property of the workload, it is a property of the *generation pattern*

The frozen 8-category suite, untrained head, block 6 (ceiling `tau` = 6), from F117's run-0 column:

| category | `tau` | % of block-6 ceiling | pattern |
|---|---:|---:|---|
| `long_context` | **5.54** | 92 % | reconstructive |
| `agentic_format` | **5.15** | 86 % | reconstructive |
| `multi_turn` | 4.46 | 74 % | mixed |
| `code_edit` | 4.43 | 74 % | reconstructive |
| `short_factual` | 3.27 | 55 % | mixed |
| `reasoning` | **1.85** | 31 % | constructive |
| `code_gen` | **1.84** | 31 % | constructive |
| `explanation` | **1.75** | 29 % | constructive |
| **suite mean** | **3.5362** | 59 % | |

**The spread is 3.2x, and it is not random across task labels — it sorts cleanly by whether the
tokens being generated are reconstructed from material already in the context or constructed new.**

- **Reconstructive** — echoing a file, emitting a diff whose surrounding lines are in context,
  filling a rigid tool-call schema, quoting a retrieved passage: **4.4 – 5.5** (74 – 92 % of ceiling).
- **Constructive** — writing a novel function body, explaining a plan in prose, chain-of-thought:
  **1.75 – 1.85** (29 – 31 % of ceiling).

This is the harness-agnostic answer, and it is the central hypothesis of this plan:

> **H1.** Realized `tau` on any agentic coding harness is determined by the reconstructive /
> constructive token ratio of its trace, not by which harness produced it. Two harnesses with the
> same pattern mix will measure the same `tau` within the 3.5 % run-to-run spread.

H1 is falsifiable and §4 says how to falsify it. If it holds, it is worth more than any single
optimisation: it means the corpus can be specified, balanced and audited in units that transfer, and
it means a harness's decode rate is *predictable from its trace* before it is ever run.

---

## 2. The arithmetic — what `tau` buys, honestly

Measured conversion on this engine, from the three back-to-back head measurements in
`HEAD_REGISTRY.md`: shipped 3.5362 -> 22.1425 tok/s (6.26 per unit `tau`), `s3` 3.8438 -> 24.2512
(6.31), suite 24.73 at 3.8438 (6.43). Slightly superlinear, because a higher `tau` also wastes fewer
verifies. F122's ceiling arithmetic anchors the top.

| `tau` | tok/s (band) | what it would require |
|---:|---:|---|
| 3.84 | **24.7** | shipping today (`s3`) |
| 4.3 | 27 – 29 | constructive categories 1.8 -> ~2.6 (FastMTP's measured +43 %) |
| 4.8 | 30 – 33 | constructive 1.8 -> ~3.3, **and no loss on reconstructive** |
| 5.5 | 35 – 37 | **every category performing like `long_context` does today** |
| 6.0 | 38 – **40.6** | perfect acceptance at block 6 — unreachable by construction |

**Read the 5.5 row carefully. It is the honest statement of what "close to 40 tok/s" costs.** It is
not a 15 % improvement on the head; it is bringing prose and novel-code generation up to the level
the head currently achieves only when it is copying. Nothing in the published literature does that.

**Therefore the plan is explicitly two-track:**

- **Track 1 — lift the constructive floor** with a better head. Realistic ceiling **`tau` 4.3 – 4.8,
  i.e. 28 – 32 tok/s.** Medium confidence. This is the S5 programme, corrected (§5).
- **Track 2 — exploit the reconstructive ceiling** with runtime levers that only fire on copy-heavy
  spans, where `tau` is already 5.5 and the *block*, not the head, is the binding constraint (§6).

Track 2 is where the last 5 tok/s lives, and on a real agentic coding harness — which is
reconstruction-heavy by nature — it may be the larger of the two.

---

## 3. Two blockers that gate everything, and must run first

### 3.1 Ladder item 1.10 — the prefill race — gates every *quotable* number

1.9 established that the engine races in `compressed_attn_forward` above ~192 prefill positions.
1.10 names the kernel. It is currently ranked as a correctness item worth 0 % on throughput, and
**that ranking understates it**: every accuracy number this project will publish is only
reproducible if the forward is deterministic. Speculative decoding is exact and the kernel work is
bit-exact, so **accuracy is invariant under everything else on the ladder** — which means the eval
battery could otherwise run at any time. The race is the single reason it cannot.

**1.10 is the gate on the entire eval programme (§8). Promote it accordingly.**

### 3.2 B9 — prefill — gates production quality *and* corpus capture

Measured (B9): **52.6 / 50.3 / 47.7 / 43.8 tok/s at PS = 255 / 511 / 1023 / 2047** — and it
*declines* with prompt size. Today's ladder sweeps reproduce it: 12,282 prompt tokens took
**227.6 s**, i.e. 54 tok/s and **3.8 minutes to first token**.

For an agentic coding harness with 12 k of repo context, a 3.8-minute TTFT is disqualifying
regardless of how good decode gets. Prefix caching hides it on turn 2+; turn 1 is unprotected.

The anomaly is quantified and unexplained: a batched forward amortising 12.26 GB of weights over 255
positions should be compute-bound, and it is running at only **3.4x the M=1 decode rate**. Every
kernel in the path was tuned for M=1 — `RB` is fitted to a 1.71-rows-per-expert histogram prefill
does not have, and the grouped GEMV is chosen over the mma GEMM on a decision that inverts at
prefill row counts (arithmetic intensity is ~150x higher at PS=255; the retired `m16` repack is
alive for prefill at **+16.7 %**, F85).

**First measurement, and nobody has ever taken it: `DSV4_DPROF` on a PS=1023 prefill.** One
checkpoint load.

---

## 4. Trace capture — harness-agnostic by construction

### 4.1 Harvest at the verify forward, not by teacher-forced prefill

`DECODE_ZENITH_FINDINGS` 2.3. Harvest `(h_40/41/42, p_target)` from **live verify forwards**: the
verifier has already computed both, so marginal compute is **zero**, and the data is on-policy and
distribution-matched *by construction*. It replaces S5's teacher-forced prefill capture, which cost
29.6 h per 5 K sequences and was bounded by the prefill rate B9 has not yet fixed.

**The constraint inverts.** The old ceiling was 240 agentic prompts and days of capture. The eval
battery alone generated **3,463,648 completion tokens**; at the measured 33 KB/token that is 114 GB
for one battery run against 128 GB free. So the new problem is **selection and storage, not
collection** — and the plan must specify a sampling policy rather than a capture schedule.

Storage mitigations, in order: fp8 hidden states (halves it, costs one numerics check against bf16
on a small sample); stream capture -> train in shards, deleting each shard after its epoch (valid
only at 1 epoch, which is what the recipe prescribes anyway).

**Guard the feedback loop.** Harvesting from a server running the head we are about to retrain is
self-training. Required: version every shard with the head sha that produced it, and hold the frozen
8-category suite completely out of the harvest.

### 4.2 The pattern label — computed from the token stream, never from the harness

This is what makes the corpus harness-agnostic, and it is the methodological core of the plan.

For each generated token `t_i`, compute mechanically at capture time:

```
recon(i) = 1 if the 8-gram ending at t_i appears anywhere in (prompt + committed output before i)
           0 otherwise
```

Then label each 64-token span by its mean `recon`, bucketed:

| bucket | mean `recon` | expected `tau` (from §1) |
|---|---|---|
| `R-high` | >= 0.75 | 5.0 – 5.5 |
| `R-mid` | 0.35 – 0.75 | 4.0 – 4.6 |
| `C-low` | < 0.35 | 1.7 – 2.0 |

Three things follow, and each is worth the plumbing on its own:

1. **It reports `tau` by pattern rather than by task label** — the instrument H1 needs.
2. **It lets the training corpus be balanced by pattern instead of by category**, which §5 argues is
   the fix for the forgetting problem.
3. **It is computable at runtime from the committed prefix alone**, so the same signal can drive
   adaptive block width (§6.2). The offline label and the online signal are the same function — the
   one design decision that keeps the corpus and the runtime honest with each other.

`DSV4_SUFFIXPROBE=1` already computes the matching machinery read-only and is unaffected by the
GATE and LOSSLESS gates. Extend it rather than writing a second matcher.

### 4.3 Falsifying H1

Capture the same 40 tasks through **two structurally different harnesses** (e.g. a
plan-then-edit loop and a single-shot whole-file rewriter). Compute the pattern mix and the measured
`tau` for each trace. H1 predicts the residual of `tau` on pattern mix is inside 3.5 % and carries no
harness term. **If a harness term survives, H1 is false and the corpus must be harness-stratified** —
that would be a more expensive programme, and it is better to learn it from 40 tasks than from a
finished head.

---

## 5. The training recipe — deltas from `S5_RECIPE.md`

`S5_RECIPE.md` §2 is correct and stays: freeze the MXFP4 experts and train the 476 M non-expert
parameters (4.44 GiB of optimiser state); the 3-term position-decayed loss at `a_ce` 0.1 / `a_tv` 0.9
/ `a_conf` 1.0 with `w_k = exp(-(k-1)/gamma)`; AdamW at 5e-5 cosine, warmup 0.05, batch 64,
**1 epoch**; regenerate all responses with the target at T=0.6 / top-k 20 / top-p 0.95. The LK-loss
hybrid stays rejected (the correction in §2.1 priced it at 21.6 tok/s against 22.15).

Three sessions have run since. What they taught changes the recipe in exactly one place.

### 5.1 The problem to solve: F117 — training degrades the strong categories regardless of data

| category | run 0 | `s1` (reasoning-only) | `s2` (8-way balanced) |
|---|---:|---:|---:|
| `long_context` | 5.54 | 4.53 (**-1.01**) | 4.35 (**-1.19**) |
| `agentic_format` | 5.15 | 4.70 (-0.45) | 4.70 (-0.45) |
| `code_edit` | 4.43 | 3.89 (-0.54) | 4.14 (-0.29) |
| `explanation` | 1.75 | 2.28 (+0.53) | 2.32 (+0.57) |
| `reasoning` | 1.85 | 2.56 (+0.71) | 2.53 (+0.68) |
| `code_gen` | 1.84 | 2.35 (+0.51) | 2.24 (+0.40) |

**Balancing the corpus repaired `short_factual` and halved `code_edit`, and did nothing at all for
`long_context` and `agentic_format`.** F117's own conclusion: *something in the training degrades the
strong cases regardless of what data it sees.* F119 then falsified the CE/TV hypothesis proposed to
explain it — all four ablation arms landed inside the run-to-run spread.

**This is the binding constraint on Track 1, and it is unsolved.** Roughly half of what training buys
on weak categories is given back on strong ones. It is also precisely the wrong trade for an agentic
coding harness, whose hot path is `code_edit` / `long_context` / agentic-diff — the categories that
lose.

### 5.2 The one recipe change: anchor the strong patterns, weight the weak ones

Two additions, both cheap, both aimed straight at F117:

**(a) A self-distillation anchor against the *pre-training* head.** The current loss has no term that
penalises drift on inputs the head already handles well. Add one:

```
L_total = SUM_s  w(s) * L_dspark(s)  +  beta * KL( q_new(s) || q_frozen(s) )   for s in R-high
```

`q_frozen` is the incumbent head, evaluated once and cached — no extra forward at train time if the
capture stores it. **`beta` is the one hyperparameter to sweep: {0, 0.1, 0.5}, three runs.**
`beta = 0` reproduces the current recipe exactly, so the sweep contains its own control.

**(b) Deficit weighting by pattern, not uniform balance.** Weight each sample by how far its pattern
bucket sits below the ceiling, normalised to mean 1:

```
w(s) = (1 - tau_bucket(s) / 6) / mean_buckets(1 - tau/6)
```

Using §1's measured values: `long_context` **0.19**, `agentic_format` 0.35, `multi_turn` 0.63,
`code_edit` 0.64, `short_factual` 1.11, `reasoning` 1.68, `code_gen` 1.69, `explanation` **1.72** —
a **9x** range. The head spends its gradient where there is 4 points of `tau` to win and nearly none
where there is 0.5.

Note this is the *opposite* of what `make_corpus.py` does today. Balanced sampling gives
`long_context` the same weight as `explanation` despite one being at 92 % of ceiling and the other at
29 %. Balance was the right correction to s1's single-domain corpus; it is not the endpoint.

### 5.3 Then, and only then: HASS

`S5_RECIPE.md` §2.4 ranks HASS-style context alignment third and it has never been run. It trains
later draft steps on imperfect *draft* features instead of clean target features (+8 – 20 % over
EAGLE-2). Our chain is depth 3, so steps 2 and 3 both run on their own predecessor's output — the
mismatch is real and structural. It is a training-loop change with no extra capture cost.

**Sequence it after (a) and (b)** because it acts on the same positions and the interaction is
unknown; running it first would confound the `beta` sweep.

### 5.4 Fix `promote_head.py` first — ladder item 2.4

Its docstring criterion 4 says suite **`tau`** must beat the incumbent; the code compares
**`suite_tok_s`**, against an incumbent row recorded on whatever engine revision was current when
*that* head was measured. Ladder 2.2 measured the drift directly: across 8 days and five decode-kernel
rewrites, suite `tau` reproduced to **four decimal places for both heads** while suite tok/s moved
-2.3 % and -5.0 % and base AR moved **-17.4 %**.

`s2` and the three ablation heads were refused on exactly that comparison, one of them at
**25.10 tok/s against an incumbent 24.52**. Fix: require the incumbent to be **re-measured in the same
session** as the candidate, and use `tau` as the cross-session anchor proving the re-measurement was
faithful. Then re-adjudicate the four refused rows. **This must land before any Phase-2 head is
measured, or the programme will grade itself with a broken ruler.**

---

## 6. Track 2 — the runtime levers that exploit the reconstructive ceiling

### 6.1 S6 / prompt-lookup: retired on the wrong workload, and this corpus is its reopening condition

F80 priced suffix-automaton drafting at an **oracle ceiling of +0.0 %** — 0 wins, 17 losses, MTP 61
tokens vs suffix-only 23. The mechanism is deep and is now **trap 27: the anchor is selected against
you.** The token a retrieval drafter is queried from is always the *correction* — the one token the
MTP just mispredicted — because wherever the sequence is predictable the MTP accepts straight
through it. So the drafter is handed the least repetitive point in the sequence by construction;
`mlen = 0` in 13 of 21 verifies.

**But F80's own entry scopes its refutation**, and the scope is exactly the gap this plan fills: it
was measured on a period-8 degenerate repeating decode, and *"reopening needs a long-repeated-context
prompt on which `mlen` routinely reaches the block size — that is the agentic regime SuffixDecoding
actually reports, and this measurement does not refute it."*

**So: turn on `DSV4_SUFFIXPROBE=1` during the §4 trace capture.** It is read-only, it does not
perturb the gates, it counts acceptance as an integer (immune to the 1.5 % timing floor), and it
therefore prices S6 on the agentic workload **at zero marginal cost, as a side effect of work already
being done.** If `mlen` reaches block size on `R-high` spans, S6 reopens with a measurement instead of
an argument. If it does not, trap 27 generalises and S6 dies properly.

This is the highest-value-per-unit-effort item in the whole plan and it costs one environment
variable.

### 6.2 Adaptive block width driven by the pattern signal

F93/F94 measured the block sweep and it is already done: **`tau` rises 5 -> 6 on all four realistic
prompts and FALLS 6 -> 8 on three of four** (`long_context` 5.54 -> 5.43, `code_edit` 4.43 -> 4.04,
agentic 5.15 -> 5.10). So block 6 is the measured optimum **for a head whose mean is 3.5** — and the
loop log's own reading is that *"a better drafter earns wider verifies."*

The static sweep asks the wrong question. `long_context` at 5.54/6 is pressing the ceiling while
`explanation` at 1.75/6 is wasting four verify slots per forward. **Gate the block width on the
`recon` signal from §4.2**, which is computable from the committed prefix at zero cost: wide (8 – 10)
on `R-high` spans, narrow (4 – 5) on `C-low`. Re-run the block sweep *per pattern bucket*, not
pooled — the pooled sweep averages a saturating category against a starving one and reports the
midpoint.

**Caveat, and it is a real one (F129).** The suite is verify-dominated: the same NVFP4 change moved
M=1 by -1.3 % and M=K by **-14 %**. Widening the block moves more time into the M>=2 path. The
retired `m16` B-repack needs **5.05 rows per expert** and block 6 presents **1.67** — but the same
fitted crossover says it pays around **block 40**. Any serious widening re-prices that lever, and it
must be re-measured, not assumed dead.

### 6.3 The confidence head at verify time — ladder item 2.3

It exists in the checkpoint, it is trained by the `a_conf = 1.0` term, and it is **unused**. EVICT-style
`argmax E[A(T_k)]/C(k)` is the principled version of 6.2 and the only remaining ladder item that
touches the multiplier. It should be built on the same signal path as 6.2 so the two are one mechanism
with two inputs, not two competing gates.

### 6.4 adaptK — small, measured, free

F121: threshold 1.5 -> 2.0 is **+1.3 %**, measured three times independently across s1/s2/s3. Take it
during the next registry re-baseline. Note F118's caution: adaptK must be fitted on decode **rate**,
not on `tau` — the two anti-correlate across this range, and fitting `tau` picked the worst threshold
of five on `s1`.

---

## 7. Production and serving quality

Ordered by what a real agentic coding harness actually feels.

| # | item | why | state |
|---|---|---|---|
| 1 | **Prefill / TTFT (B9)** | 3.8 min to first token at 12 k context. Nothing else matters if turn 1 is unusable. | `DSV4_DPROF` on PS=1023 never run |
| 2 | **Prefix-cache behaviour under agentic turn structure** | Measured hit rates exist per task, but an agentic loop re-sends a growing prefix every turn — the one access pattern the battery did not exercise | untested |
| 3 | **Short-context batching** | ~40 sequences at 4 k, **5** at 32 k, **1** at 128 k (S5 §4). Real for parallel subagents, useless for a serial hot path — and it does **not** reduce per-request latency | Phase 6 |
| 4 | **Long-context KV capacity** | `1b.2` is banked and default-OFF: 2048 -> 720 B/row, **2.844x**, bit-exact. It costs +3.23 ms flat to save 0.50 ms/1000 ctx, break-even ctx 6,471 — so it is *already net positive* above 6.5 k and it triples the context that fits | shipped, flag-gated |
| 5 | **Cancellation / streaming under the agentic loop** | A harness that abandons a generation must free the KV promptly or the next turn starts degraded | unaudited |

Item 4 is worth restating: `1b.2` was correctly ruled default-OFF for a decode ladder optimising term
A, but **for a long-context agentic serving profile the same measurement says turn it on.** The
default should be re-decided against the production context distribution, not against the ladder's.

---

## 8. Evals — quotable numbers for every row

**Scheduling insight that saves the most time here: accuracy is invariant under everything on the
kernel ladder.** Speculative decoding is exact by construction and every kernel change was gated
bit-exact, so accuracy numbers do **not** need re-running after each optimisation — only the
throughput columns in `PERF.md` do. The battery is therefore not blocked on the head programme.

It is blocked on exactly one thing: **§3.1, the prefill race.** A non-deterministic forward makes a
published accuracy number irreproducible. Fix 1.10, then the battery can run in parallel with the
head work.

Known state and what each row needs:

| eval | state | action |
|---|---|---|
| `gpqa_diamond` | **72.6 %, NOT QUOTABLE** — 25.9 % truncation at 8 k | 24 k extension. **Pre-registered prediction: 87 – 93 %, central ~90 %** (`EVAL_NEXT_STEPS.md` §1) — do not revise it after the fact |
| `bfcl_mt_base`, `bfcl_mt_miss_func` | **n = 200, 0 completion tokens** — the harness produced nothing | Broken, not slow. Diagnose before re-running |
| `lcb`, `scicode` | 12.26 / 14.69 tok/s, longest completions | Check truncation rate the same way GPQA was decomposed; a truncated trace scores as wrong |
| `aime24`, `aime25` | captured at `low` and `low24k` budgets | Reconcile which budget is quotable |
| `humaneval`, `math500`, `mmlu_pro`, `bfcl`, `bfcl_live` | terminated cleanly | Quotable once the race is fixed |

**Budget protocol, already learned the hard way: pilot at the ceiling.** The needed `max_tokens` is
not identifiable from a run that was too small — a truncated trace cannot tell you how much more it
wanted. Staging up costs only a re-prefill. Every extension pass should start at the largest budget
under consideration.

**Run detached.** `setsid` or systemd, never as a child of an interactive session.

---

## 9. Order of work, with the gate that stops each step

Each step states what would falsify it. Nothing proceeds on a step whose gate did not pass.

| # | step | cost | gate |
|---|---|---|---|
| 0 | **Ladder 1.10** — name the racing kernel | ~3 min/run, 1 load | Race reproduced then eliminated; `DSV4_STEPHASH` clean above 192 positions |
| 1 | **Ladder 2.4** — fix `promote_head.py`, re-adjudicate the 4 refused heads | offline | The re-adjudication reproduces `s3` as incumbent, or names a better one with a same-session re-measurement |
| 2 | **B9 prefill profile** — `DSV4_DPROF` at PS=1023 | 1 load | Names the sub-op holding prefill at 3.4x decode. If it is diffuse, say so and stop |
| 3 | **Eval battery restart** (parallel from here) | days, detached | GPQA lands in **87 – 93 %**. Outside that band, the prediction was wrong and it gets written up as wrong |
| 4 | **Pattern labeller + `SUFFIXPROBE` capture** on 40 tasks x 2 harnesses | ~1 day | **H1**: no harness term survives in the residual. Also prices S6 for free |
| 5 | **Harvest corpus** from live verify forwards, fp8, versioned, suite held out | disk-bound | Shard `tau` by bucket reproduces §1's table within spread — proof the labeller and the engine agree |
| 6 | **`beta` sweep** {0, 0.1, 0.5} + deficit weighting, 1 epoch | 3 runs | A `beta > 0` arm holds `long_context` within -0.2 of run-0 **while** lifting the constructive mean. If none does, F117 is deeper than the loss and Track 1 caps at ~4.0 |
| 7 | **HASS on steps 2 – 3** | 1 run | Beats step 6's winner outside the 3.5 % spread |
| 8 | **Per-bucket block sweep + confidence-head gating** | 2 loads | A pattern-gated width beats static block 6 on the suite. Re-price `m16` if width moves materially |
| 9 | **S6 decision** | free (step 4 output) | Reopen only if `mlen` reaches block size on `R-high` spans |
| 10 | **`PERF.md` re-run, registry re-baseline, `1b.2` default re-decided** | 1 battery | Both coefficients and both `tau` columns recorded together |

---

## 10. Expected outcome

Stated as bands, before the work, so it can be graded rather than narrated.

| outcome | tok/s | confidence |
|---|---:|---|
| today | **24.7** | measured |
| Track 1 alone (head, forgetting unfixed) | 26 – 28 | high — this is roughly what `s1` -> `s3` already did twice |
| Track 1 with F117 fixed (step 6 gate passes) | **28 – 32** | medium |
| Track 1 + Track 2 (pattern-gated width, S6 if it reopens) | **32 – 36** | low-medium, and **entirely dependent on step 4's H1 result** |
| block-6 perfect acceptance | 40.6 | unreachable |

**The honest headline: 30 – 33 tok/s is the defensible target, 36 is the optimistic one, and 40 is a
ceiling that assumes perfection.** On a real agentic coding harness the realized number should land
*above* the suite mean, because the suite is balanced 8 ways while an agentic trace is
reconstruction-heavy — and reconstruction is where this engine already runs at 92 % of ceiling.

That last sentence is the most useful prediction in this document and step 4 tests it directly.
