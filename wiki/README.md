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

## The state in one table

| | measured | ceiling | |
|---|---|---|---|
| speculative decode | **22.15 tok/s** | 23.2–25.9 | 96 % |
| base AR decode | **13.83 tok/s** | 14.33–15.98 | 97 % |
| acceptance | 2.89 / 5 | — | **the remaining 1.4× lives here** |
| prefill (PS=1022) | **62.4 tok/s** | ~94 with tensor cores | +30.3 % this session |

The M=1 kernel path is finished — two independent methods agree (an exhausted lever queue, and a
byte-weighted floor computed from measured per-kernel rates). Everything left is **acceptance**,
which is a training problem: see [`draft-head-finetuning.md`](draft-head-finetuning.md).

## If you read one thing

[`negative-results.md`](negative-results.md). The wins are mostly three transformations applied
repeatedly; the retired levers are where the actual knowledge is, and §1 — a **fake +28 % speedup
that passed every gate** — is why the LOSSLESS gate exists.
