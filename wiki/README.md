# Wiki

Engineering and ML documentation for `deepseek-v4-flash-0731-cuda`. Primary source for everything
here is `LOOP_LOG.md` (88 findings, chronological); these pages organise it by topic.

| page | what it holds |
|---|---|
| [`kernel-optimisations.md`](kernel-optimisations.md) | every **adopted** AR/spec-decode optimisation, by mechanism: mechanism → measured gain → the gate that proved it |
| [`negative-results.md`](negative-results.md) | levers built and **retired**, with the number that killed each. Longer than the wins list and more useful. |
| [`prefill-optimisation.md`](prefill-optimisation.md) | B9 — why prefill ran decode-shaped kernels, and the four fixes (+30.3 %) |
| [`draft-head-finetuning.md`](draft-head-finetuning.md) | S5 — the ML: architecture, loss, data, hyperparameters, feasibility arithmetic, corpus saturation |
| [`measurement-and-traps.md`](measurement-and-traps.md) | how a number becomes trustworthy here, and the ways one has failed to |
| [`hardware-sm110a.md`](hardware-sm110a.md) | Thor: measured bandwidth **and compute** peaks, ISA facts, operating rules |
| [`context-scaling.md`](context-scaling.md) | **the term nobody measured** — why every profile was taken at context 9, the attribution, and the two adoptions (1.0, 1.2) that took `b` down 65 % |

## The state in one table

**Every row needs a context.** The first block is context ~9, where this project did all of its
historical measurement; the second is where agentic work actually runs.

| at context ~9 | measured | ceiling | |
|---|---|---|---|
| speculative decode | **22.15 tok/s** | 23.2–25.9 | 96 % |
| base AR decode | **13.83 tok/s** | 14.33–15.98 | 97 % |
| acceptance | 2.89 / 5 | — | **the remaining 1.4× lives here** |
| prefill (PS=1022) | **62.4 tok/s** | ~94 with tensor cores | +30.3 % this session |

| at real context | ladder open | after 1.0 | after 1.2 | |
|---|---|---|---|---|
| spec decode @ ctx ~12.3k | 7.67 tok/s | 9.54 tok/s | **10.69 tok/s** | +24.4 % then **+8.2 %** |
| spec decode @ ctx ~9.3k | 9.69 tok/s | 11.79 tok/s | **12.60 tok/s** | +21.7 % then +6.7 % |
| spec decode @ ctx ~6.2k | 9.59 tok/s | 11.10 tok/s | **11.77 tok/s** | +15.8 % then +5.8 % |
| context term `b` | 7.220 ms/1000 | 4.006 ms/1000 | **2.514 ms/1000** | **−65 % cumulative** |

Each arrow is a separate paired A/B on its own before-arm; the columns are not one continuous
sweep, and 1.2's before-arm measured `b = 3.488` where 1.0's after-arm reported 4.006 — one
run-to-run spread on a fitted coefficient, which is why the *paired* saving and not the fit
difference is the ratchet in both cases.

**The context term is no longer where the money is.** Of the 6.97 ms/1000 that 0.4 attributed, the
two large items — `draft:main_kv` (3.867) and `i:topk` (0.872) — are both spent. What is named and
left is `cattn:sparse` (0.709) and `i:score` (0.644), **1.35 of a measured 2.514**, so about half
the surviving slope is still unattributed. Term A is now 129.11 ms of a 154.66 ms forward at ctx
12,410 — **83 %** — and it is 1.57× its byte floor. See [`context-scaling.md`](context-scaling.md).

> ### ⚠️ RETRACTED 2026-08-20 — "the M=1 kernel path is finished"
>
> That claim, and the table above it, were true only of the workload they measured: **context ~9**.
> Every decode profile in this repo before 2026-08-20 was taken at the length of `decode.cu`'s
> `argv[2]`, and the production server's profiler was writing to nowhere (`dprof_init` was called,
> `dprof_report` never was). Decode has a context-linear term of **7.362 ± 0.370 ms per 1000
> tokens** which is 77 ms of a 204 ms step at ctx 12,288, and 55 % of it is a single draft-side
> recompute that had never been timed. Two levers were retired on numbers wrong by 135× and 330×.
>
> See **[`context-scaling.md`](context-scaling.md)**. The acceptance argument still stands on its
> own terms; it is no longer the *only* thing left.
>
> **Update 2026-08-20:** that recompute is now cached
> ([`kernel-optimisations.md` §2.5](kernel-optimisations.md)) — **+24.4 % at ctx 12,282**, gated by
> memcmp on the whole main-KV buffer across 704 calls and 2.59 M retained rows. Proving it also
> established that **the engine does not reproduce itself run-to-run at context**, which makes the
> ladder's "byte-identical token ids" invariant untestable as written at long context; see
> [`measurement-and-traps.md` §12](measurement-and-traps.md).
>
> **Update 2026-08-20, same day:** the second of the two retired-on-a-135×-wrong-number levers is
> also spent — the indexer's top-512 is a single-CTA radix select rather than 512 sequential argmax
> rounds ([`kernel-optimisations.md` §2.6](kernel-optimisations.md)), **+8.2 % at ctx 12,410**,
> gated by memcmp on the whole index array across 11,008 calls and 183 M slots. It also arrived
> **3.12 ms/forward ahead of what the profile predicted**, in a kernel on the draft side that has
> never had a dprof mark — so the profile is *still* incomplete, for the second time.

## If you read one thing

[`negative-results.md`](negative-results.md). The wins are mostly three transformations applied
repeatedly; the retired levers are where the actual knowledge is, and §1 — a **fake +28 % speedup
that passed every gate** — is why the LOSSLESS gate exists.
