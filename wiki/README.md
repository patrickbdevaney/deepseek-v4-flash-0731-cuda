# Wiki

Engineering and ML documentation for `deepseek-v4-flash-0731-cuda`. Primary source for everything
here is `LOOP_LOG.md` (88 findings, chronological); these pages organise it by topic.

| page | what it holds |
|---|---|
| [`kernel-optimisations.md`](kernel-optimisations.md) | every **adopted** AR/spec-decode optimisation, by mechanism: mechanism → measured gain → the gate that proved it |
| [`negative-results.md`](negative-results.md) | levers built and **retired**, with the number that killed each. Longer than the wins list and more useful. |
| [`prefill-optimisation.md`](prefill-optimisation.md) | B9 — why prefill ran decode-shaped kernels, and the four fixes (+30.3 %) — **and §7: one of the four had gone 0.80x and 1.7 took it back** |
| [`draft-head-finetuning.md`](draft-head-finetuning.md) | S5 — the ML: architecture, loss, data, hyperparameters, feasibility arithmetic, corpus saturation, **and §8: how a promoted head finally got served (2.2)** |
| [`measurement-and-traps.md`](measurement-and-traps.md) | how a number becomes trustworthy here, and the ways one has failed to |
| [`hardware-sm110a.md`](hardware-sm110a.md) | Thor: measured bandwidth **and compute** peaks, ISA facts, operating rules |
| [`context-scaling.md`](context-scaling.md) | **the term nobody measured** — why every profile was taken at context 9, the attribution, the adoptions (1.0, 1.2, 1.5) that took `b` down 74 %, **1.7, the item that moved Term A instead**, and **1b.2, which moves `b` another 26 % and charges Term A for it** |
| [`context-ceiling-is-not-the-kv-cache.md`](context-ceiling-is-not-the-kv-cache.md) | why `seqmax` is an engine artefact, **and the second ceiling at context 49,140 that is not memory at all** (1.4) |
| [`hardware-sm110a.md` §5](hardware-sm110a.md) | operating rules, **including what the clocks actually do under load (3.1) — the governed box already runs at the pinned frequencies 97.7 % of the time** |
| [`oom-and-memory-safety.md`](oom-and-memory-safety.md) | why the OOM killer never fires here (unified memory is invisible to the cgroup), and the two guards that replace it |

## The state in one table

**Every row needs a context.** The first block is context ~9, where this project did all of its
historical measurement; the second is where agentic work actually runs.

| at context ~9 | measured | ceiling | |
|---|---|---|---|
| speculative decode | **22.15 tok/s** | 23.2–25.9 | 96 % |
| base AR decode | **13.83 tok/s** *(pinned; 11.3 governed — see below)* | 14.33–15.98 | 97 % |
| acceptance | 2.89 / 5 | — | **the remaining 1.4× lives here** |
| prefill (PS=1022) | **62.4 tok/s** | ~94 with tensor cores | +30.3 % this session |
| `build/decode` prefill reproducibility | **byte-identical to 160 positions; racing at 192+** | byte-identical at every length | ladder 1.9; server path unaffected |

| the frozen 8-prompt suite | shipped head | `s3` (live since 2.2) | |
|---|---|---|---|
| suite mean `tau` | 3.5362 | **3.8438** | **+8.70 %**, and both reproduced their archived value to 4 d.p. |
| suite mean tok/s | 22.1425 | **24.2512** | **+9.52 %** (drift-controlled +10.30 %) |
| worst prompt `tau` | 1.75 | **2.49** | no suite prompt below 2 any more; spread −32.8 % |
| base AR (drift control) | 11.41 tok/s | 11.33 tok/s | −0.70 % between the two loads |

> **Those two base-AR rows differ by 21 % and neither is wrong — they were taken in different clock
> states, and until 3.1 nothing recorded which.** `WARM decode` is a 7-step average taken
> immediately after load, inside the ~2 s DVFS ramp, so it times the governor: **88.0 / 88.0 / 88.7 /
> 88.4 ms/tok governed against 72.7 / 72.9 pinned**, six loads, reproducing to a tenth of a
> millisecond in both states. That is the entirety of ladder item 2.5's "unexplained 17.4 % base-AR
> fall" — there is no engine regression. Spec throughput never moved because the suite runs *after*
> the ramp. [`measurement-and-traps.md` §24](measurement-and-traps.md).

Both arms measured back to back on `93699e6`, same binary, same protocol, LOSSLESS and first-token
gates PASS on both. **This row is a weights change, not a kernel one** — `s3` was promoted on
2026-08-12 and no server had ever loaded it, because `promote_head.py` only archives and `serve.sh`
hardcoded the base checkpoint. [`kernel-optimisations.md` §2.8](kernel-optimisations.md),
[`draft-head-finetuning.md` §8](draft-head-finetuning.md).

| at real context | ladder open | after 1.0 | after 1.2 | after 1.3 | after 1.5 | after 1.7 | after 1.8 | |
|---|---|---|---|---|---|---|---|---|
| spec decode @ ctx ~12.3k | 7.67 tok/s | 9.54 tok/s | 10.69 tok/s | *unchanged* | **+4.57 % paired** (10.46 → 10.93 in-session) | **+3.25 % paired** (11.71 → 12.09 in-session) | *no kernel change* | +24.4 %, +8.2 %, nothing, +4.6 %, **+3.3 %**, nothing |
| spec decode @ ctx ~9.3k | 9.69 tok/s | 11.79 tok/s | **12.60 tok/s** | *unchanged* | not swept | not swept | not swept | +21.7 % then +6.7 % |
| spec decode @ ctx ~6.2k | 9.59 tok/s | 11.10 tok/s | 11.77 tok/s | *unchanged* | **+2.24 % paired** (11.87 → 12.15 in-session) | **+3.32 % paired** (11.74 → 12.13 in-session) | *no kernel change* | +15.8 %, +5.8 %, nothing, +2.2 %, **+3.3 %**, nothing |
| spec decode @ ctx ~1.7k | not swept | not swept | not swept | not swept | not swept | **+2.44 % paired** (13.93 → 14.27 in-session) | not swept | the pre-knee leg |
| **forward term `a`** | 136.44 ms | *untouched* | 129.11 ms | *untouched* | *untouched* | **125.11 ms** (−3.996 ± 0.080 paired) | **125.11 ms**, −4.4 ms of *phantom* headroom | 1.571× → **1.522×** its 82.18 ms floor |
| context term `b` | 7.220 ms/1000 | 4.006 ms/1000 | 2.514 ms/1000 | 3.008 → 3.036, its own arms | 1.942 ms/1000 (−0.572 ± 0.018 paired) | **1.887 ms/1000** (−0.0552 ± 0.0102 paired) | *untouched* (the swing is flat in context) | **−74 % cumulative** |

| the one switchable row — `DSV4_KV_PACK` (1b.2, default **OFF**) | default | packed | |
|---|---|---|---|
| KV cache row | 2048 B | **720 B** | **2.844×**, bit-exact; 16/16 legs byte-identical, `tau` equal on every leg |
| forward term `a` | 125.11 ms | 128.34 ms | **+3.233 ± 0.203 paired** — `sparse_attn` unpacks each gathered row and that kernel is issue-bound |
| context term `b` | 1.887 ms/1000 | **1.387 ms/1000** | **−0.4996 ± 0.0290 paired**, R² 0.990 — the L2-residency effect `KV_PRECISION_FINDINGS` §4 predicted |
| net | — | — | **break-even at ctx 6,471**; +2.1 % tok/s at 12,410, −1.4 % at 3,197 |

> **1b.2 is the first row here that is a TRADE rather than a win, and it is off by default for that
> reason.** It is also the answer to "does packing the KV cache make decode faster": below ctx 6.5k,
> no — a 2.84× byte reduction made the reader kernel *slower at every shape*
> (0.78–0.90×), because bytes are not what it is short of.
> [`negative-results.md` §4i](negative-results.md), [`context-scaling.md`](context-scaling.md).

**1.7 is the first item on this ladder to move Term A, and it is the term with the headroom.** `b ×
6592` is now 12.44 ms against a stop condition of 5.0; `a` is 125.11 against a byte floor of 82.18
and a stop condition of 1.25× = 102.7. Six items moved `b` by 74 %; one item has moved `a` by 3.1 %.
[`kernel-optimisations.md` §2.9](kernel-optimisations.md).

**1.8 moved neither term, deleted 4.4 ms of phantom headroom from Term A, and priced something
nothing had ever timed.** The `cattn:q_proj` bimodality (1.71 ms on some steps, 5.47 on others at
identical shape, width and context) is the compressor emits running on `g_side` while the mark is
timed on the main stream. A schedule term computed from the dprof tag alone —
`g = #{ j in [ctx,ctx+VB) : (j+1)%4 == 0 }` — separates the two populations **153/153 on the log
0.4 had already written and 174/174 on a fresh arm, with no overlap**, and under `NO_ATTN_SPLIT=1`
the swing collapses to **1.00–1.02×** while the identical time reappears in `cattn:compress`
(2.50 → 8.34 ms at fixed VB=2). So `a` stays 125.11 ms. What the item bought: the compressed-KV
emit costs **7.02 ms on the 64.9 % of forwards that carry one = 4.56 ms/forward amortised, 3.3 % of
the forward, at 52 % of its 880 MB byte roofline**, and the ATTN_SPLIT overlap already in the engine
is worth a paired **0.81 ms/forward, 2 SE band [0.72, 0.90], 9/9 legs faster and byte-identical**.
[`measurement-and-traps.md` §29](measurement-and-traps.md),
[`negative-results.md` §4h](negative-results.md).

**Read 1.5's column as a paired percentage, not as a new absolute.** Its sweep spans ctx
3,197–12,410 in one session per arm; the 10.93 and 12.15 are that session's after-arm and are not
commensurable with 1.2's 10.69/11.77 from a different load (§19 — the between-load spread is 5.7 %).
The trustworthy quantity is the paired delta, which is why it is the one in bold, and `b` is carried
forward by **subtracting** the paired saving rather than by re-fitting a narrow range.

**1.4 is not in that table at all, and that is the correct place for it.** It is a correctness item,
not a throughput one: it removes a silent garbage-return above context 49,140 from the two top-k
*instruments* (the `DSV4_TOPK_RADIX=0` A/B arm and the `DSV4_TOPK_GATE=1` in-situ reference), which
requested `~4T` bytes of dynamic shared memory against a 49,152 B default. The shipped path has
requested **zero** dynamic shared memory since 1.2, so no measured number in this table moves and
none was claimed — and that claim is not an inference: `kernels/indexer.cu`,
`kernels/compressed_decode.cu` and `kernels/attention.cu` compile to **byte-identical SASS** before
and after, so the change is host-side launch configuration and nothing else. What the item is worth
is that the arms which certify every *future* top-k change now work above the context they were only
ever run below. Closed 2026-08-20 with the engine's own decode step driven at contexts 49,207 and
200,003 (`scripts/gate_topk_smem_ctx.sh`), and with the defect reproduced in the same binary under
`DSV4_TOPK_SMEM_OPTIN=0`: it exits 0, `cudaDeviceSynchronize` says success, and the output is
4096/4096 nonzero with no NaN and the wrong hash. See
[`measurement-and-traps.md` §17](measurement-and-traps.md) — a failed launch that
`cudaStreamSynchronize` reports as success, §18 — the device opt-in maximum that is not the
kernel's maximum — §19 — **the variance is between checkpoint loads, not within them: 0.6 % against
5.7 %, so a number from a previous iteration is not a valid before-arm** — and
[`context-ceiling-is-not-the-kv-cache.md`](context-ceiling-is-not-the-kv-cache.md).

**The ratchet for 1.4 is a null with a same-arm control.** `tau` 2.87 and the 66 generated token ids
are byte-identical across 1.3's recorded leg, a freshly rebuilt **pre-1.4** binary and the 1.4
binary; spec throughput is 19.65–19.76 tok/s for 1.4 and 19.65–19.72 for the pre-1.4 control, both
measured today, against the 20.89 recorded three hours earlier that **neither** binary reproduces.

**1.3 is in that table as a blank on purpose.** It shipped, it is bit-exact across 66.9 M gated
index slots, and it is worth 0.07 % of a forward — a correct optimisation of a term 1.2 had already
spent, picked off a ranking that went stale one commit earlier. It is the reason the ladder now has
a rule 6 ("re-attribute before you pick"), and the reason a wins table has to be able to show a
zero. [`negative-results.md` §4c](negative-results.md).

Each arrow is a separate paired A/B on its own before-arm; the columns are not one continuous
sweep, and 1.2's before-arm measured `b = 3.488` where 1.0's after-arm reported 4.006 — one
run-to-run spread on a fitted coefficient, which is why the *paired* saving and not the fit
difference is the ratchet in both cases.

**The context term is no longer where the money is, and as of 1.5 it is no longer where the named
work is either.** Of the 6.97 ms/1000 that 0.4 attributed, **three of the four items are now
spent**: `draft:main_kv` (3.867 → −0.000), `i:topk` (0.872 → 0.084 ± 0.001) and, with 1.5,
`i:score` (0.644 → **0.040 ± 0.001**). What is named and left is `cattn:sparse`, whose lever is a
~20 ms *floor* rather than a slope — so against the tracked `b = 1.942` there is essentially no
identified linear work remaining, and the slope did not go to zero when the marks did. Term A is
129.11 ms of a 154.66 ms forward at ctx 12,410 — **83 %** — and it is 1.57× its byte floor. See
[`context-scaling.md`](context-scaling.md).

`cattn:sparse` re-fits at **1.694 ± 0.065** over ctx 0.4k–6k against 0.4's **0.709 ± 0.050** over
3k–12k. **That is one concave curve read over two windows, not a regression and not drift**, and
ranking on the bigger number would promote the wrong item — see
[`measurement-and-traps.md` §16](measurement-and-traps.md), which is the second new trap this week
where a clean fit with a tight SE meant something other than it looked like.

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
> **Bounded 2026-08-20 by ladder 1.9, and "at long context" turned out to be the wrong axis.** The
> mechanism is `build/decode`'s **prefill**, inside the compressed layers, and it has a measured
> threshold: the whole 43-layer prefill is byte-identical run-to-run at **160 prefill positions and
> below** (430 layer hashes, zero differences) and nondeterministic at **192 and above**, to 3,071.
> Four repeats of the identical point inside ONE process disagree with each other, so it is a race,
> not per-process state; the arena is exonerated at those lengths too.
> **`dsv4-server` is not affected by this mechanism** — it prefills in `EXT_CHUNK` = 64-row chunks
> through a different function — which is why 1.5's 16-leg server A/B was byte-identical at ctx
> 12,410 and is not a contradiction. [`measurement-and-traps.md` §25–§27](measurement-and-traps.md),
> [`negative-results.md` §4f](negative-results.md).
>
> **Update 2026-08-20, same day:** the second of the two retired-on-a-135×-wrong-number levers is
> also spent — the indexer's top-512 is a single-CTA radix select rather than 512 sequential argmax
> rounds ([`kernel-optimisations.md` §2.6](kernel-optimisations.md)), **+8.2 % at ctx 12,410**,
> gated by memcmp on the whole index array across 11,008 calls and 183 M slots. It also arrived
> **3.12 ms/forward ahead of what the profile predicted**, in a kernel on the draft side that has
> never had a dprof mark — so the profile is *still* incomplete, for the second time.

> **Update 2026-08-20 — 3.1, and it is a null: the clock lever was already applied.** The last
> throughput item before hand-back was `jetson_clocks`, carried from HARDWARE.md at "+3.0–6.4 %".
> Sampling both rails every 2 s — which no run in this repo had ever done — shows a **governed** box
> spends **97.7 % of its compute window at 1386 MHz and 4266 MHz**, the pinned frequencies; the
> 315/2750 idle state is the ~90 s checkpoint load. Pinning buys the ramp, not the ceiling:
> **+1.99 ± 0.15 % (8/8 legs, `tau` 3.8438 in all six arms)**, below the 3.5 % spread and **not
> counted as a win**. MAXN's 1575 MHz is worth **+1.68 ± 5.83 %** — nothing, which is the right
> answer for a bandwidth-bound engine asked about its core clock. Adopted anyway, in the launchers,
> for what it does to the *measurement* (the 21 % base-AR artefact above).
> [`negative-results.md` §4e](negative-results.md).
>
> **Its first attempt voided itself, and that is the more useful half.** The pre-registered drift
> control — arm A repeated last — came back **+8.19 %**, larger than the effect. Cause: the agent
> supervising the arms was still working during them, and a ~GB process of its own exited mid-arm.
> Re-run on a quiet box the drift control falls to +1.78 % and the engine reproduces `tau` on all 9
> prompts instead of 8. **A measurement arm is single-tenant, and the agent is a tenant.**
> [`measurement-and-traps.md` §23](measurement-and-traps.md).

> **Update 2026-08-20, third of the three:** the *last* of the linear items 0.4 named is spent too.
> `index_score` is a register-tiled GEMM rather than a warp-per-(query,row) kernel
> ([`kernel-optimisations.md` §2.7](kernel-optimisations.md)) — **+4.57 % paired at ctx 12,410**,
> **−0.572 ± 0.018 ms per 1000 context**, 16 of 16 legs faster with `tau`, verify width and the
> emitted text hash identical in every one. It delivered **89 % of what 0.4 predicted for the mark**,
> which is the strongest thing this ladder has said about its own cost model. The mechanism worth
> remembering is not the tiling: it is that the 2× version was capped by claiming bit-exactness
> against the *shipped* kernel, and the 6.8× version got there by claiming it against the *scalar
> reference* instead ([`negative-results.md` §4d](negative-results.md)).
>
> **And the gates that were supposed to be watching this subsystem had not compiled in weeks** —
> four in-situ gates, including both CUDA-graph capture gates, silently absent since the `xin`-ring
> commit ([`measurement-and-traps.md` §20](measurement-and-traps.md)). A gate has three outcomes and
> only two of them are ever printed.

## If you read one thing

[`negative-results.md`](negative-results.md). The wins are mostly three transformations applied
repeatedly; the retired levers are where the actual knowledge is, and §1 — a **fake +28 % speedup
that passed every gate** — is why the LOSSLESS gate exists.
