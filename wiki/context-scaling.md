# Context scaling — the term nobody measured

Every decode number in this repo before 2026-08-20 was taken at **context ~9**. Not 9 thousand —
nine tokens. That is why the wiki said the M=1 kernel path was finished, and why it wasn't.

## The claim this page retires

`README.md` and [`roofline-why-the-needle-wont-move.md`](roofline-why-the-needle-wont-move.md) both
concluded that the kernel path was exhausted and everything left was acceptance. Two independent
methods agreed: a lever queue that had gone dry, and a byte-weighted floor from measured per-kernel
rates. Both were right about the workload they measured. Neither measured a workload with context
in it.

Fitted over 2,156 real generation legs from the evaluation battery, and then over a controlled
sweep to 12,410 tokens:

```
ms per target forward = 130.98 + 7.362 x (context / 1000)      R^2 0.971
```

At the contexts agentic work actually runs at, the second term is most of the cost. At ctx 12,288 it
is **77 ms of a 204 ms step**. Every historical measurement sat at the left edge of that line, where
it contributes 0.07 ms and is invisible.

## Two mechanisms made it invisible, and both were structural

**1. The server's profiler was writing to nowhere.** `src/engine.cu` has called `dprof_init` since
the marks were written and **never once called `dprof_report`**. Every mark the production server
ever recorded was discarded. The only profiles that existed came from `src/decode.cu`'s K-sweep.

**2. That K-sweep runs at the length of `argv[2]`.** `decode.cu` takes its prompt from the command
line. `DSV4_PROMPTS_FILE` only appends prompts 1..N, while both the base-AR and K-sweep paths run
prompt **0**. So a two-token `argv[2]` means every profile in `evidence/` was taken at context 1..9,
regardless of what the prompt file said. A run labelled "ctx 3000" was `seqmax` — an allocation
sized from the longest prompt in the file — not a context. Two such runs at different `seqmax`
values produce byte-identical token streams, which is how the mistake is detectable after the fact
and was not detectable before.

The two compound: the instrument that could have shown context scaling was silently disabled, and
the instrument that was running could not vary context.

## The attribution

`tools/dprof_ctx.py`, one server load, seqmax 16384, 145 steady-state verify steps at four depths
772–12,406. The `b|VB` column holds realised verify width fixed, because width sets bytes-per-forward
and correlates with context; it is the number to read.

| mark | ms/1000 ctx (`b\|VB`) | ms at ctx 12,288 | share of step slope |
|---|---|---|---|
| **STEP (verify + draft)** | 6.969 ± 0.112 | 203.9 | 100 % |
| **`draft:main_kv`** | **3.867 ± 0.001** | **47.87** | **55 %** |
| `ATTENTION` | 3.173 ± 0.109 | 87.4 | 46 % |
| &nbsp;&nbsp;`i:topk` | 0.872 ± 0.021 | 13.47 | 12.5 % |
| &nbsp;&nbsp;`i:score` | 0.644 ± 0.018 | 6.58 | 9 % |
| &nbsp;&nbsp;`cattn:sparse` | 0.709 ± 0.050 | 21.17 | 10 % |
| `MoE` | **−0.079 ± 0.029** | flat | ~0 |
| every dense GEMM | within 1 SE of 0 | flat | ~0 |

**The instrument reproduces the independent wall-clock fit to 0.9 %** (6.969 vs 7.029
width-controlled), and the per-mark slopes sum exactly to the step slope, so nothing is
unaccounted for. Enabling `DSV4_DPROF` costs nothing measurable — the wall-clock legs reproduce the
uninstrumented run point for point.

## What this changes

**Term B is not a bandwidth problem.** At ctx 6592 its byte floor is 0.509 ms/forward against ~43 ms
measured — still ~85× off roofline. It is **serial and redundant work**: one recompute and one
selection sort. `MoE` and every dense GEMM are flat in context, so the bytes story that governs the
constant term says nothing about this one.

**The largest single row had never been timed at all.** `dspark_main_kv` runs **3× per step over the
full `ctxlen`** — an fp8 GEMM + rmsnorm + rope + quant over every position in context, recomputed on
every token. Its R² against context is **1.000**: it is arithmetic, not noise. Every dprof mark in
the repo lived in the verify stack, so the draft side was outside the reported window entirely.

**Two levers were retired on numbers that were wrong by 100×.** `i:topk` was recorded at 0.10 ms and
`i:score` at 0.02 ms. At ctx 12,288 they are 13.47 ms and 6.58 ms — 135× and 330×. Both are O(context)
by construction and both were measured at context 9.

## The rule this produces

Adding to [`measurement-and-traps.md`](measurement-and-traps.md):

> **A decode measurement without a stated context is not a measurement.** Report the context every
> number was taken at, and vary it before concluding anything about scaling. A profiler that has
> never been read at two different contexts cannot tell you which of its marks are constant.

And its corollary, which is why this went nine findings deep:

> **Check that the instrument is switched on.** `dprof_init` without `dprof_report` looks identical
> to a profile with nothing in it.

## Status — 1.0 is done, and the term it removed came off the measured slope

**Adopted 2026-08-20.** `draft:main_kv` is no longer recomputed; only the rows committed since the
last step are built. Mechanism, gates and the full paired table are in
[`kernel-optimisations.md` §2.5](kernel-optimisations.md). The part that belongs on *this* page is
that the cost model above predicted the win and then survived it:

```
before   fwd = 132.24 + 7.220 x (ctx/1000)      R^2 0.976   SE(b) 0.165
after    fwd = 129.96 + 4.006 x (ctx/1000)      R^2 0.888   SE(b) 0.210
```

**The context term is down 44 %.** Measured directly as a paired saving across 30 exactly-paired
legs — legs whose two arms produced bit-identical `tau` and verify width, so the difference is the
kernel and nothing else — it is `3.604 ± 0.076 ms per 1000 context`, R² 0.988, against the
`3.867 ± 0.001` this page attributed to `draft:main_kv`. **93 % of the predicted term, and the 2 SE
band does not cover the prediction**; the missing 0.26 ms/1000 is written down in §2.5 rather than
rounded into agreement. Predicting a slope from a profiler and then recovering it from wall clock
is the strongest form the attribution above could take, and it very nearly held exactly.

User-visible: **7.67 → 9.54 tok/s at ctx 12,282 (+24.4 %)**, +21.7 % at 9,213, +15.8 % at 6,132,
tapering to +4.5 % at 1,536 and nothing at context ~9 — which is precisely why nine findings' worth
of profiling never saw it.

## Status — 1.2 is done too, and it found a term the profile could not see

**Adopted 2026-08-20.** The DSA indexer's top-512 selection is a single-CTA radix select instead of
512 sequential argmax rounds; mechanism, gates and the full paired table are in
[`kernel-optimisations.md` §2.6](kernel-optimisations.md). What belongs on *this* page is the second
prediction-then-recovery, and the way it came up short in one direction and long in another:

```
before   fwd = 130.60 + 3.488 x (ctx/1000)      R^2 0.892   SE(b) 0.179
after    fwd = 129.11 + 2.514 x (ctx/1000)      R^2 0.858   SE(b) 0.151
paired   saving = 3.122 + 0.793 x (ctx/1000)    R^2 0.951   SE(b) 0.031   (35 exactly-paired legs)
```

**The context term is down another 28 %, and 65 % cumulatively from the 7.220 the ladder opened
on.** The slope this page attributed to `i:topk` was `0.872 ± 0.021`; the paired measurement
recovered `0.793 ± 0.031`, whose 2 SE band **[0.730, 0.856] does not cover it**. 9 % short, written
down rather than rounded away — the same shape of small miss 1.0 had, in the same direction.

**And it came in 3.12 ms/forward long in the intercept, which is the finding.** `i:topk` marks
`k_topk_verify` and nothing else. `k_topk_decode`, the identical kernel on the *draft* side, has
never had a dprof mark, so its cost has never appeared in any table on this page — exactly the blind
spot that hid `draft:main_kv` until 0.4 went looking for it. **A win that lands in the intercept
means the profile is still incomplete**, and this is the second time that has been true.

User-visible: **9.88 → 10.69 tok/s at ctx 12,410 (+8.2 %)**, +6.7 % at 9,341, +5.8 % at 6,260,
tapering to +1.9 % at 889.

## Status — 1.3 did not move this term, and that is the entry

**Adopted 2026-08-20, measured null.** The `lim <= topk` early-out inside the radix select skips a
whole discovery level below ctx 2048. It is bit-exact (34/34 legs byte-identical, 66.9 M gated index
slots, 0 FAIL) and it is worth **+2.1 to +4.1 µs per call** standalone — and the context term did
not notice:

```
before (DSV4_TOPK_EARLY=0)  fwd = 127.92 + 3.008 x (ctx/1000)   SE(b) 0.241
after                       fwd = 127.84 + 3.036 x (ctx/1000)   SE(b) 0.240
```

`b` is unchanged inside a tenth of an SE, and it was pre-registered as unable to move: 21 ratio-4
layers × ~4 µs is 0.085 ms of a ~130 ms forward. The one instrument that could see it, the `i:topk`
mark, did: **0.42 → 0.28 ms at ctx 768, 0.52 → 0.34 at 1536, 0.72 → 0.72 at 6144** where it cannot
fire. The change works; the term it optimises was already spent by 1.2 one iteration earlier. See
[`negative-results.md` §4c](negative-results.md) for why it was built anyway, which is the part
worth reading.

## Status — 1.5 is done, and it is the first kernel win since 1.2

**Adopted 2026-08-20. `index_score` as a register-tiled GEMM: `b` falls by
0.572 +/- 0.018 ms per 1000 context, paired, over 16 legs.** This is the item this page named as
"linear, and the next kernel", and it delivered **89 % of the 0.644 +/- 0.018 that 0.4 predicted**
— inside one standard error of the prediction.

```
paired, per leg, control arm first (drift favours the control):
  delta_fwd = +0.358 +/- 0.134  -0.5724 +/- 0.0178 x (ctx/1000)      R^2 0.987, n=16
```

Every one of the 16 legs is faster, and `tau`, mean verify width and the **emitted text hash** are
identical in every one — at ctx up to 12,410, not at ctx 9. tok/s at ctx 3,197 / 6,260 / 12,410:
**12.14 -> 12.27, 11.87 -> 12.15, 10.46 -> 10.93** (+0.99 %, +2.24 %, **+4.57 %**). The gain rising
with context *is* the signature: this is a Term-B change and it behaves like one at both ends —
at ctx 85 on `build/decode` it is a measured **null** (21.03-21.24 against 21.02-21.26).

The mark confirms it directly. dprof medians, both arms, same protocol:

| mark | before, ctx 3072/6144/12,288 | after |
|---|---|---|
| `i:score` | 1.93 / 3.53 / **6.58** | 0.99 / 1.12 / **1.36** |
| `cattn:indexer` (parent) | 4.79 / 6.41 / **9.51** | 3.86 / 4.00 / **4.27** |
| `i:topk` (untouched control) | 0.72 / 0.73 / 0.76 | 0.72 / 0.72 / 0.76 |
| `STEP` | 130.82 / 134.55 / **143.62** | 130.29 / 132.45 / **138.61** |

Three things to read off that table. **`i:score` = 6.58 ms at ctx 12,288 reproduces 0.4's 6.58
exactly**, four iterations and three kernel changes later — the attribution this whole page rests on
is reproducible. **The saving lands entirely inside the parent** (-5.24 of `cattn:indexer` against
-5.22 of `i:score`) and `i:topk` does not move, so no work was relocated into an unmarked region.
And **the slope saving reads 0.463 ms/1000 in `i:score`'s own mark, and 0.484 +/- 0.009 in `STEP`,
against 0.572 +/- 0.018 clean** — dprof's own per-mark syncs compress the difference by 15-20 %,
which is why the ratchet is the clean pair and dprof is only the attribution. (`i:score` itself goes
from **0.503 +/- 0.006 to 0.040 +/- 0.001 ms per 1000**: the mark is 92 % dead, not merely smaller.)

**The tracked `b` is carried forward by subtraction, not by re-fit, and that is deliberate.** 1.5's
sweep spans ctx 3,197-12,410, and over that narrow range against a ~140 ms intercept a fit
determines `b` badly — the two arms fit `139.57 + 1.231 (SE 0.254)` and `140.00 + 0.652 (SE 0.236)`,
R^2 0.701 and 0.433. Same direction, useless precision. The paired number has R^2 0.987 because the
common-mode drift cancels leg by leg. So: **`b` 2.514 -> 1.942 ms/1000; `b x 6592` 16.57 ->
12.80 ms** (the stop condition wants <= 5.0). Cumulatively `b` is down **73 %** from the 7.220 this
ladder opened on. Rule 7 applies to your own arms too.

## Status — 1.7 moved Term A, and left behind a 6 % Term-B tail with a named mechanism

**Adopted 2026-08-20. `sparse_attn` stages the gathered row in shared memory: paired saving
4.227 ± 0.121 ms per forward over 16 legs, of which `b` is only 6 %.** This is the item this page
had ranked last of the four and it is the one that moved the *other* term — which is what the entry
predicted ("the lever here is the 11 ms FLOOR, not the slope").

The 16 paired legs split at the knee the cost model predicts. `topk = WINDOW(128) +
min(INDEX_TOPK(512), ctx/ratio)` saturates at ctx 2048, so above it the ratio-4 layers do a
context-independent amount of work:

```
ctx 1,528  (topk=510, below the knee, n=2)    -3.127 +/- 0.014 ms
ctx >=3,069 (topk=640, saturated,   n=14)     -4.384 +/- 0.065 ms
above the knee:  delta = -3.996 +/- 0.080  -0.0552 +/- 0.0102 x (ctx/1000)   R^2 0.710
```

**94 % of the win is flat.** The surviving `-0.0552 +/- 0.0102 ms per 1000` is small, five sigma
from zero, and — unusually for this page — was *predicted from the layer map before it was fitted*,
which is the only thing that distinguishes it from the artefact rule 7 exists to catch. **20 of the
43 layers are ratio-128, and their `topk = wmax + ctx/128` has no `index_topk` cap at all**, so
their `sparse_attn` work really is linear in context:

```
20 layers x (1000/128 = 7.81 rows per 1000 ctx) x 0.309 us saved per row   =  0.048 ms per 1000
measured                                                                    =  0.055 +/- 0.010
```

Inside one SE. That is the second time on this ladder a per-kernel band has predicted an in-situ
context slope (1.5 was the first), and it is worth noticing *why* it works here: the per-row saving
is a property of the kernel, and the rows-per-context is a property of the config, so the product is
falsifiable without running the engine at all.

**Tracked `b` 1.942 -> 1.887 ms/1000; `b x 6592` 12.80 -> 12.44 ms** (stop wants <= 5.0), carried by
subtraction per rule 7 — 1.7's own arms span ctx 1,528–12,282 against a ~135 ms intercept and fit
`b` to SE 0.174 and 0.180, i.e. uselessly. Cumulatively `b` is down **74 %** from the 7.220 this
ladder opened on.

**And the part this page does not own: `a` 129.11 -> 125.11 ms**, 1.571× -> 1.522× the 82.18 ms byte
floor. Term A had been untouched through six ladder items and is much the larger distance from its
floor; this is the first thing to move it. [`kernel-optimisations.md` §2.9](kernel-optimisations.md).

## Status — 1b.2 moved this term by a QUARTER, and paid for it in Term A (2026-08-20)

The section above ends with "there is no longer a named, linear, kernel-shaped context item
waiting." 1b.2 was not one — it is a **storage** change, ranked for capacity, and
`KV_PRECISION_FINDINGS.md` §4 explicitly told it not to expect a throughput result. It produced the
largest single move in `b` since 1.0 anyway, and the shape of the result is the finding.

**Packing the main KV cache** (2048 → 720 B per row, FP8 E4M3 + 7 x UE8M0 + FP32 RoPE, bit-exact,
`DSV4_KV_PACK=1`) measured over 16 legs, both arms in one session on one build, every leg
byte-identical with `tau` and mean verify width equal on every one:

```
ctx      before -> after   delta   tok/s before -> after
 1,656   130.27  132.45    +2.18    14.27 -> 14.04
 3,197   132.77  134.74    +1.97    12.44 -> 12.26
 6,248   137.07  137.18    +0.10    12.23 -> 12.22
 6,260   134.16  134.17    +0.01    12.13 -> 12.12
12,410   142.27  139.31    -2.97    12.09 -> 12.34
```

Regressed on context, the delta is **+3.233 ± 0.203 ms per forward, −0.4996 ± 0.0290 per 1000
context, R² 0.990** (2 SE bands `[+2.83, +3.64]` and `[−0.558, −0.442]`; both exclude zero). Read
off the independent per-arm fits with the width term controlled it is `b` **0.924 → 0.381** and `a`
**82.83 → 85.74**, which is the same answer by a different route.

**So `b` falls 1.887 → 1.387 (−26 %, `b × 6592` 12.44 → 9.15) and `a` rises 125.11 → 128.34
(+2.6 %). Break-even is ctx 6,471.** Above it packing is a throughput win — +2.1 % tok/s at 12,410 —
and below it a loss.

**Why a byte reduction moved the CONTEXT term and not the flat one, which is the whole mechanism.**
The flat cost is instructions: `sparse_attn` unpacks each gathered row, and 1.7 established that
kernel is issue-bound, so more instructions cost time regardless of context
([`negative-results.md` §4i](negative-results.md) has the microbenchmark — the packed reader is
0.78-0.87x of the FP32 reader at every shape). The context-dependent saving is bytes: the gathered
working set is `topk` rows and `topk` grows with context until it saturates, so what packing shrinks
is precisely the part of the traffic that scales. That is
`KV_PRECISION_FINDINGS.md` §4's L2-residency hypothesis — *"packing the whole context-dependent
working set to 16.2 MB brings it inside Thor's 24 MB persistable L2 window. Hypothesis, not
measurement"* — now measured, and it lands on the term the hypothesis named.

**The Term-A half is a tuning cost, not a floor, and 1b.2 already bought 1.77 ms of it back.** The
first staging loop read a `uint4` per thread (16 codes), which minimises the load count and starves
the block: 28 of 128 threads working. Four codes per thread took the paired flat cost from
**+4.998 ± 0.374 to +3.233 ± 0.203** while leaving the slope alone (**−0.4653 ± 0.0534 →
−0.4996 ± 0.0290**, overlapping) — a clean separation, because the fix touched issue and not bytes.
Whatever is left of the +3.23 is the same currency, and it is what would move the break-even below
the operating context.

**Shipped default OFF.** The trade is real in both directions and where it lands depends on the
operating context, which is an operator decision and is recorded here rather than made silently. The
one place it is unambiguous is capacity: `EVALS.md` records `seqmax` as *the* binding constraint on
which eval items can run at all, and packing is 2.844x on every KV allocation with zero accuracy
exposure.

## What is left of the term

Slopes below are 0.4's, fit over ctx 3k–12k. The `now` column is 1.3's re-attribution, 220 verify
samples over ctx 369–6255, width held fixed
(`evidence/decode_loop/dprof_ctx_1p3_on.txt`).

| mark | 0.4, ms/1000 ctx | now | status |
|---|---|---|---|
| ~~`draft:main_kv`~~ | ~~3.867~~ | **−0.000** | **done, §2.5** |
| ~~`i:topk`~~ | ~~0.872 ± 0.021~~ | **0.084 ± 0.001** | **done, §2.6 + 1.3** |
| `cattn:sparse` | 0.709 ± 0.050 | 1.694 ± 0.065 ⚠ | ladder 1.7 — the lever is its ~20 ms *floor*, not its slope |
| ~~`i:score`~~ | ~~0.644 ± 0.018~~ | **0.040 ± 0.001** | **done, [`kernel-optimisations.md` §2.7](kernel-optimisations.md)** |

⚠ **The two `cattn:sparse` numbers are one concave curve read over two windows, not a disagreement,
and the larger one must not be used to re-rank.** Per-point medians run 9.35 → 15.72 → 19.89 ms at
ctx 768/1536/6144 and then 19.22 → 19.90 → 21.17 at 3072/6144/12,288: **8.29 ms per 1000 across the
first leg, 0.208 across the last.** That is a top-`k` gather saturating once context exceeds
`k × ratio`, which is what this page predicted of it. `i:score` over the identical samples is
0.88 → 1.25 → 3.53 ms, i.e. 0.482 then 0.495 per 1000 — genuinely linear, so its `b` means what it
says. Full argument in [`measurement-and-traps.md` §16](measurement-and-traps.md).

The four were 6.09 of the 6.97 ms/1000 step slope that 0.4 attributed. **Three of the four are now
spent** — `draft:main_kv`, `i:topk` and, as of 1.5, `i:score` — and what remains named is
`cattn:sparse`, whose lever is a ~20 ms *floor* and not a slope at all. Against the tracked
`b = 1.942`, that is **essentially nothing identified in the surviving term**: every mark 0.4 put a
number on has been taken, and the slope did not go to zero with them.

**That is the honest state and it is the next question on this page.** The remaining context cost is
in regions no mark brackets, and the intercept surprise above says at least some of it is on the
draft side where no mark has ever been placed. A fifth attribution pass would be a *new instrument*,
which this ladder deliberately ranks below any available kernel change — but there is no longer a
named, linear, kernel-shaped context item waiting, and that is a change in the situation, not a
gap in the notes.

Term A is now much the larger problem: at ctx 12,410 it is 129.11 ms of a 154.66 ms forward — 83 %.

## 1.11 — a Term-A change that showed up only at long context, and why that is left open (2026-08-20)

Ladder 1.11 defers the ATTN_SPLIT join past `i:qidx` and `i:iw`
([`kernel-optimisations.md` §2.10](kernel-optimisations.md)). It pre-registered a **flat** saving,
and the argument for that is not hand-waving: every byte the deferred window moves is
context-independent. The two `compressor_emit_group` calls read one group of `ratio` tokens whatever
the context is; `i:qidx` is `K × QD` out of `Q_LORA` and `i:iw` is `K × nH` out of `DIM`. Nothing in
the window is sized by the cache. The mark-level confirmation was taken **at context 5**, where
there is no context term to hide in, and it is −1.44 ms on a step carrying one emit.

**The engine disagreed.** Drift-free over 18 paired legs and four checkpoint loads:

| ctx | n | mean ms/forward | 2 SE band | legs faster | sd |
|---|---|---|---|---|---|
| 3,197 | 6 | −0.282 | [−0.759, +0.195] | 3/6 | 0.585 |
| 6,260 | 6 | −0.252 | [−0.858, +0.353] | 3/6 | 0.741 |
| **12,410** | 6 | **−1.092** | **[−1.238, −0.946]** | **6/6** | **0.179** |

Fitted on the same per-leg deltas: flat **+0.150 ± 0.564**, context **−0.0949 ± 0.0685 per 1000**,
R² 0.324.

**Two readings, and neither is established.**

1. *The saving really is in `b`.* Then there is a mechanism nobody has named. The two obvious
   candidates both fail on arithmetic: the emits are context-independent by construction, and the
   emit *rate* moves only ~7 % across these three points (realised verify width 2.56 → 2.74, and
   P(a block of width `w` straddles a ratio-4 boundary) ≈ `w/4`).
2. *The saving is flat and the two short groups are noise.* Their sd is 0.585 and 0.741 against
   **0.179** at 12,410 — a 3–4× difference in scatter between groups, which is enough on its own to
   tilt a three-point least-squares line. This is the exact configuration
   [`measurement-and-traps.md` §16](measurement-and-traps.md) warns about: a slope fitted through
   groups with unequal scatter is not a cost per 1000 context.

**So `b` was NOT moved on this page and `a` was not moved either.** The ratchet is stated where it
resolves — `a + b×ctx` at ctx 12,410 falls 142.24 → 141.15 ms — and the attribution is ladder item
1.13, which needs ≥ 6 context points in **both** arm orders. Writing 1.11's saving into either term
on three points would be inventing a number, and this page exists because somebody did that once.

## 1.12 — a Term-A change that is genuinely flat, and the first clean "no, it is not the context term" (2026-08-20)

1.11 ended this page with an unresolved term: a real drift-free saving whose own mechanism said FLAT
and whose three context points would not say so. **1.12 is the same class of change and it does say
so**, which is worth recording because it shows the design *can* resolve the question when the
effect is big enough.

`gemm_fp32`'s (4,4) warp tile (`kernel-optimisations.md` §2.11) speeds up the compressor emit.
Nothing in an emit scales with context: it reads `x_full` for one group of `ratio` tokens and two
weight matrices whose size is fixed by the config. **Pre-registered as Term A**, in
`scripts/f32mkn_ab_run.sh`, before the run:

    flat    -2.348 +/- 3.302 ms/forward
    context +0.0533 +/- 0.4038 ms per 1000 context     R^2 0.004

The context coefficient is **zero to within a twentieth of its own uncertainty**, and R² 0.004 says
the per-leg deltas carry no context signal at all. Drift-free paired mean **−1.963 ± 1.504
ms/forward**, 14/18 legs, over 18 legs and four checkpoint loads. So `a` **125.11 → 123.15 ms** and
`b` **stays 1.887** — `b × 6592` stays 12.44 ms against a stop condition of 5.0.

**Why this one resolved and 1.11 did not.** Same 18 legs, same three targets, same two arm orders.
The difference is size: 1.11's effect (~0.5 ms) is a third of the ±1.5 ms floor a two-load 18-leg
design has (`measurement-and-traps.md` §34 measures that floor on identical binaries), so its
per-group split was fitting noise. 1.12's is 4× that floor. **The lesson is not "1.11's harness was
wrong"; it is that a term attribution needs roughly the same power as the effect itself, and neither
comes free.**

**And the mark-level instrument agreed without using the fit at all.**
`0.628 × 2.49 ms` (verify TOTAL excess on the 62.8 % of steps carrying a ratio-4 emit) `+ 0.0239 ×
26.42 ms` (the extra excess on the 2.4 % carrying a ratio-128 emit) `= −2.20 ms/forward`, which
falls inside `[−3.467, −0.459]`. Both of those excesses are differences taken **within one arm**, so
they are immune to the between-load offset that dominates the paired band — which is why a
schedule-conditional mark is the right instrument for a Term-A change of this size and a three-point
context fit is not.
