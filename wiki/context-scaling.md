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

## What is left of the term

Slopes below are 0.4's, fit over ctx 3k–12k. The `now` column is 1.3's re-attribution, 220 verify
samples over ctx 369–6255, width held fixed
(`evidence/decode_loop/dprof_ctx_1p3_on.txt`).

| mark | 0.4, ms/1000 ctx | now | status |
|---|---|---|---|
| ~~`draft:main_kv`~~ | ~~3.867~~ | **−0.000** | **done, §2.5** |
| ~~`i:topk`~~ | ~~0.872 ± 0.021~~ | **0.084 ± 0.001** | **done, §2.6 + 1.3** |
| `cattn:sparse` | 0.709 ± 0.050 | 1.694 ± 0.065 ⚠ | ladder 1.7 — the lever is its ~20 ms *floor*, not its slope |
| `i:score` | 0.644 ± 0.018 | **0.629 ± 0.018** | ladder 1.5 — **linear, and the next kernel** |

⚠ **The two `cattn:sparse` numbers are one concave curve read over two windows, not a disagreement,
and the larger one must not be used to re-rank.** Per-point medians run 9.35 → 15.72 → 19.89 ms at
ctx 768/1536/6144 and then 19.22 → 19.90 → 21.17 at 3072/6144/12,288: **8.29 ms per 1000 across the
first leg, 0.208 across the last.** That is a top-`k` gather saturating once context exceeds
`k × ratio`, which is what this page predicted of it. `i:score` over the identical samples is
0.88 → 1.25 → 3.53 ms, i.e. 0.482 then 0.495 per 1000 — genuinely linear, so its `b` means what it
says. Full argument in [`measurement-and-traps.md` §16](measurement-and-traps.md).

The four were 6.09 of the 6.97 ms/1000 step slope that 0.4 attributed. Both large ones are now
spent, and what remains is **1.35 ms/1000 of identified work inside a measured 2.514** — so about
half of the surviving slope is *still* unattributed, and the two named items are each worth roughly
a tenth of what 1.0 was worth. **The easy half of this term is gone; the honest next question is
what the other 1.16 ms/1000 is**, and the intercept surprise above says at least some of it is on
the draft side where no mark has ever been placed.

Term A is now much the larger problem: at ctx 12,410 it is 129.11 ms of a 154.66 ms forward — 83 %.
