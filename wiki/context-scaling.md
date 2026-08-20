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

## Status

`draft:main_kv` caching is item 1.0 on `DECODE_LADDER.md`. Correctness is already established —
`tests/gate_mainkv_incr.cu` shows incremental and from-scratch main-KV byte-identical across 22
split points, with two negative controls that fail it, and the in-situ gate has 384+ checks and
~2.02 M retained rows byte-identical across all three invalidation paths. The throughput A/B is what
remains.
