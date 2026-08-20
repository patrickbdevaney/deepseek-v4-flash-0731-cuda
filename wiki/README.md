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
| [`context-scaling.md`](context-scaling.md) | **the term nobody measured** — why every profile was taken at context 9, the attribution, and the rule it produces |

## The state in one table

**Every row needs a context.** The first block is context ~9, where this project did all of its
historical measurement; the second is where agentic work actually runs.

| at context ~9 | measured | ceiling | |
|---|---|---|---|
| speculative decode | **22.15 tok/s** | 23.2–25.9 | 96 % |
| base AR decode | **13.83 tok/s** | 14.33–15.98 | 97 % |
| acceptance | 2.89 / 5 | — | **the remaining 1.4× lives here** |
| prefill (PS=1022) | **62.4 tok/s** | ~94 with tensor cores | +30.3 % this session |

| at real context | before 1.0 | after 1.0 | |
|---|---|---|---|
| spec decode @ ctx 12,282 | 7.67 tok/s | **9.54 tok/s** | **+24.4 %** |
| spec decode @ ctx 9,213 | 9.69 tok/s | **11.79 tok/s** | +21.7 % |
| spec decode @ ctx 6,132 | 9.59 tok/s | **11.10 tok/s** | +15.8 % |
| context term | 7.220 ms/1000 | **4.006 ms/1000** | −44 %, and 2.2 ms/1000 of it is still identified work |

The context term is now the smaller half of the problem but not a solved one: `i:topk` (0.872
ms/1000), `cattn:sparse` (0.709) and `i:score` (0.644) remain, and together they are worth about
what 1.0 was worth — see [`context-scaling.md`](context-scaling.md).

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

## If you read one thing

[`negative-results.md`](negative-results.md). The wins are mostly three transformations applied
repeatedly; the retired levers are where the actual knowledge is, and §1 — a **fake +28 % speedup
that passed every gate** — is why the LOSSLESS gate exists.
