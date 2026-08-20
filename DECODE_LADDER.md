# Decode ladder — the ordered work list the autonomous loop executes

`scripts/decode_loop.sh` reads this file, picks the topmost `TODO`, does it, gates it, measures it,
and marks it. **Edit this file to steer the loop.** One item per iteration; one change per
measurement; bands not points.

## The stop condition — what "no faster combined decode is possible" means here

Combined decode is `tok/s = tau * 1000 / ms_per_forward`, and `ms_per_forward = a + b*ctx`
(`tools/decode_model.py`). Both terms have a **byte floor** at the measured 240 GB/s:

- **Term A floor** — forward bytes at the measured K=5 expert union (17.53 of 30) plus the draft
  side: **82.18 ms**. Term A was 136.44 ms at suspension = **60.2 % of achievable**.
- **Term B floor** — 42.0 MB at ctx 6592 = 0.509 ms/forward at tau 2.91. Term B was **198.1 ms**.

**Where the two terms stand now (1.0, 2026-08-20):** `a = 129.96 ms` = **1.58x** the 82.18 ms floor
(stop wants <= 1.25x); `b x 6592 = 26.41 ms`, down from 47.60 (stop wants <= 5.0 ms). Neither is met,
but **term B moved for the first time**: 1.0 took the context slope from `7.220 +/- 0.165` to
`4.006 +/- 0.210` ms per 1000, a 44 % cut, measured as a paired saving of `3.604 +/- 0.076`. Term A
is untouched and is now the larger distance from its floor. The fit reaches context 12,410, so the
stop check does not extrapolate.

**The loop STOPS when either:**
1. `a <= 1.25 * a_floor` **and** `b*6592 <= 5.0 ms` — i.e. both terms are within a quarter of their
   byte floors and the context term has stopped mattering; or
2. five consecutive iterations move the suite by less than **2 %**, which is below the measured
   3.5 % run-to-run spread and therefore unmeasurable; or
3. `DECODE_LOOP_STOP` exists, or the iteration/wall caps are hit.

Reaching (1) is the roofline. Reaching (2) means the remaining headroom is not addressable by the
items on this ladder and the loop should hand back rather than thrash.

## THE POINT IS FASTER DECODE, NOT BETTER INSTRUMENTS

Read this before picking an item. Measurement earns its place only when it changes what gets built
next, and this ladder has now spent **four of its first five items on instruments** (0.1-0.4). Each
was defensible in isolation and 0.4 in particular paid for the whole run -- it proved 0.2 had never
varied context at all, un-retired 1.2 and 1.5, and found in `draft:main_kv` a larger item than
either -- but the engine had not gotten faster since the warp top-k landed, that was five
iterations, and the whole context term was attributed with error bars.

**BROKEN 2026-08-20 by 1.0: +24.4 % tok/s at ctx 12,282, and the rule is what produced it.** 0.4
bought exactly one thing -- it named the largest term and predicted its size -- and the very next
iteration spent that prediction on a kernel and got 93 % of it back. That is the whole case for the
rule, and also its limit: **the remaining items are smaller and none of them needs a new
instrument.** `i:topk`, `i:score` and `cattn:sparse` are already attributed with error bars. Build
them. The one open instrument-shaped item, 1.9, is deliberately ranked below both kernel changes.

So, in order:

1. **Prefer an item that changes a kernel over an item that measures one.** If both are available,
   take the kernel.
2. **An instrument is justified only by naming the optimisation it unblocks**, in the ladder entry,
   before building it. "We need better data" is not a justification; "we cannot choose between 1.2
   and 1.5 without attributing the residual" is.
3. **A speedup that is not in the wiki did not happen.** The moment a kernel change is measured and
   kept, write it into `wiki/` in the same iteration -- `kernel-optimisations.md` for an adopted win
   (mechanism -> measured gain -> the gate that proved it), `negative-results.md` for a lever that
   was built and killed, `context-scaling.md` for anything that moves the context term, and
   `measurement-and-traps.md` for any new way a number turned out to be untrustworthy. Update
   `wiki/README.md`'s state table in the same commit. The wiki said "the M=1 kernel path is
   finished" for weeks while the largest term in decode had never been timed; a page that is not
   maintained as the work lands becomes confidently wrong, which is worse than absent.
4. **Every iteration that ships a kernel change must report the before/after on the same corpus**,
   with tau, as a band. That is the ratchet: a number that went up, on a fit that already exists.
5. **If two consecutive iterations produce no kernel change, say so at the top of the ladder entry**
   and take the highest-expected-value kernel item next even if it is less certain. Thrashing on
   measurement is the failure mode this section exists to prevent.

## Hard invariants — the loop must never violate these

1. **Bit-exactness or an explicit gate.** Any kernel change either produces byte-identical generated
   token ids, or ships behind the LOSSLESS gate with the deviation measured and recorded. A change
   that silently alters output is reverted, not debugged.

   **AMENDED 2026-08-20 by 1.0 — token ids are not a valid test at context, and this invariant was
   nearly used to revert a correct change.** The engine stops reproducing itself part-way through a
   long run: `build/decode` with `DSV4_MAINKV_CACHE=0` on **both** sides of the comparison, argmax,
   no seed, no HTTP, still diverged from itself at the identical id index 1067. The cached path
   never ran on either side, so the divergence cannot be the cache. **A token-id comparison is only
   evidence if the same-arm control was run and passed — run the control FIRST.** Where it fails,
   the invariant is satisfied instead by memcmp of the changed kernel's **entire output buffer**
   against the untouched reference, in situ, on every call; that is strictly stronger, because it
   proves the whole intermediate identical rather than one downstream consumer's output. See
   `wiki/measurement-and-traps.md` §12 and item 1.9.
2. **tau is reported in every A/B.** A byte-identical token sequence can still collapse acceptance
   3.12 -> 1.00 (`LOOP_LOG`), because acceptance is an exact draft/target comparison. Throughput
   without tau is not a measurement.
3. **Never restart the eval battery.** The watchdog and boot unit are disarmed on purpose. The GPU
   is single-tenant; a second client turns a scored item into a banked wrong answer.
4. **Never modify the checkpoint.** Weights are not ours to edit. The draft head is backed up at
   `~/model-backups/heads/shipped-dspark-0731reap/`.
5. **Clocks are a measurement variable.** Do not change `jetson_clocks` mid-comparison. It is its
   own ladder item with its own before/after.

---

## Phase 0 — instrument (must complete before anything is trusted)

- [x] **0.1** `timings` on `/v1/completions` — staged in `server.cpp`, built 2026-08-19.
- [x] **0.2** `DSV4_DPROF` diff at two contexts. **DONE 2026-08-19.** The headline below — "the
      context term is GONE" — is **SUPERSEDED BY 0.3, which measured it as 4.1x smaller, not gone.**
      Both runs here were at ctx <= 3000, where the residual term is 22 ms of a 153 ms forward and
      two single runs at a 3.5 % spread cannot see it. Read the retirements of 1.2/1.5 with 0.4.
      Two single-prompt runs on the same binary, ctx 480 and ctx 3000:

      | | base AR (M=1) | spec | i:topk (21 calls) | i:score | cattn:indexer |
      |---|---|---|---|---|---|
      | ctx 480 | 89.3 ms/tok | 65.9 ms/tok, 15.18 tok/s | 0.10 ms | 0.02 ms | 2.09 ms |
      | ctx 3000 | 89.4 ms/tok | 65.2 ms/tok, 15.33 tok/s | 0.10 ms | 0.02 ms | 2.05 ms |

      **Base AR moved 0.1 ms across a 6x context change.** The pre-fix fit predicted
      150.9 -> 226.6 ms over the same span (`136.44 + 30.053*(ctx/1000)`). The slope went from
      **30.05 ms per 1000 context to ~0.04**, and `i:topk` is flat at 0.10 ms across all 21 calls.
      Both LOSSLESS gates passed.

      **This retires 1.2 and 1.5 as decode levers.** A radix select replaces a kernel that now
      costs 0.10 ms, and the `index_score` GEMM attacks `i:score` at 0.02 ms. Neither is worth a
      bit-exactness risk. 1.3 and 1.4 remain as correctness items, not performance ones.
      The remaining cost is **ATTENTION 44 % and MoE 38 %**, which is Term A — and Term A is the
      term with ~1.5x of headroom, not 3x.
- [x] **0.3** Re-fit `tools/decode_model.py` on a post-fix run and record both coefficients.
      **DONE 2026-08-19. `a = 130.98 +/- 2.25 ms`, `b = 7.362 +/- 0.370 ms per 1000 context`**
      (95 % CI; n = 48, R^2 0.971, measured context **249 to 12,410** — the first fit this repo has
      ever had above 6.6k). Binary rebuilt from HEAD (d961544) so the fit describes shipped source;
      all six CPU gates pass. Server at seqmax 16384, `mem 120.0/122.8 GiB`, memguard armed.

      | | pre-fix (battery, n 2158) | post-fix (sweep, n 48) | |
      |---|---|---|---|
      | `a` | 136.44 +/- 0.52 ms | **130.98 +/- 2.25 ms** | unchanged, see caveat |
      | `b` | 30.051 +/- 0.241 ms/1000 | **7.362 +/- 0.370 ms/1000** | **4.08x smaller** |
      | `b x 6592` | 198.10 ms | **48.53 +/- 2.44 ms** | stop condition wants <= 5.0 |
      | context range | 71 - 6592 | 249 - 12,410 | |
      | tau p50 | 2.91 | 1.68 | different corpus — see caveat |

      **Neither stop condition is met and the loop continues.** `a` is **1.59x** its 82.18 ms byte
      floor against a target of 1.25x; `b x 6592` is **9.7x** the 5.0 ms threshold. The context term
      is now 10 % of a forward at ctx 2000 (was 31 %) and **41 % at ctx 12,410**.

      Per point, medians over 6 repeats (`evidence/decode_loop/fit/postfix.sweep.jsonl`):

      | ctx | ms/forward | spread | tau | mean verify width | tok/s |
      |---|---|---|---|---|---|
      | 249 | 127.51 | 5.2 % | 1.636 | 2.62 | 12.83 |
      | 889 | 136.77 | 2.6 % | 1.724 | 2.64 | 12.65 |
      | 1664 | 147.66 | 2.6 % | 1.701 | 2.68 | 11.52 |
      | 3197 | 160.54 | 3.0 % | 1.626 | 2.69 | 10.08 |
      | 6260 | 179.18 | 2.4 % | 1.718 | 2.66 | 9.58 |
      | 9341 | 204.68 | 7.4 % | 1.969 | 2.92 | 9.66 |
      | 12410 | 215.33 | 4.3 % | 1.652 | 2.66 | 7.68 |

      **The slope is not a speculation artifact.** `tau` and realised verify width both correlate
      +0.35 with context here, and width is what sets bytes per forward, so the raw slope could in
      principle have been an acceptance effect wearing a context costume. Regressing on both:
      `fwd = 70.74 + 7.029 x (ctx/1000) + 23.04 x width` (R^2 0.985, SE 0.145 and 3.50). Width does
      cost a real 23 ms per unit, and it absorbs **0.33 of the 7.36** — the context term survives at
      **7.03 +/- 0.28** with width held fixed.

      **The cheap prefill was checked, not assumed.** The sweep visits contexts descending through
      token-prefixes of one document, so the engine's longest-common-prefix cache turns all but the
      first prefill into a `rewind_to` (`prefill=0ms` on 47 of 48 legs, ~40 min of prefill saved).
      If `rewind_to` left the compressed or index caches in a cheaper state than a freshly built
      context, every number above would be measuring a path the engine does not ship. Four control
      legs built their context from a **different document** (shared prefix: 1 token) and so took
      the build path: **181.59 vs 179.18 ms/forward at ctx ~6250 (1.3 %)** and **150.09 vs 147.66 at
      ctx ~1660 (1.6 %)**, both inside the 3.5 % run-to-run spread. The shortcut is sound.

      **CAVEAT, and it decides what may be claimed.** This is a different population from the
      battery corpus: raw `/v1/completions` continuation of one technical document, versus chat
      benchmarks. tau is **1.68 here against 2.91 there**, so fewer positions are verified per
      forward, a smaller expert union is read, and `a` is *expected* to come in below the battery's
      136.44 for reasons that have nothing to do with any kernel. **`a` must therefore be read as
      unchanged, not improved** — the top-k fix never touched Term A, and 130.98 at tau 1.68 is if
      anything the optimistic end. The `b` comparison is the sound one: each number is a slope
      fitted *within* its own corpus, and the ~4x is far outside either interval.
      Likewise **the tok/s column above is not comparable to `PERF.md`'s 22.66 suite mean** — same
      engine, lower-tau workload.

      Instruments: `tools/decode_fit_probe.py` (new; detached, fatal on transport failure, writes no
      record for a leg that generated nothing, and now named in `detach_audit.sh`);
      `tools/decode_model.py --dir` plus standard errors on both coefficients, because once `b` is
      small R^2 is the wrong statistic — a genuinely flat slope scores R^2 ~ 0 however precisely the
      zero is known, and the stop condition is a claim about an interval on `b`.
- [x] **0.4** `DSV4_DPROF` at ctx 12k, and attribute the residual 7.36 ms/1000. **DONE
      2026-08-20. The residual is fully attributed and 0.2 was wrong at the root: 0.2 never varied
      context.**

      **NO KERNEL CHANGED IN 0.3 OR 0.4** — two consecutive measurement iterations, which the
      section above says must be declared. It is declared. The next item taken is **1.0**, a kernel
      change, and 0.4 is what made it findable.

      **0.2 DID NOT MEASURE TWO CONTEXTS. It measured the same context twice.** `src/decode.cu`
      takes its prompt from **argv[2]**; `DSV4_PROMPTS_FILE` only appends prompts 1..N, and the
      base-AR and K-sweep paths both run prompt 0. Both 0.2 runs passed a **2-id** argv[2], so both
      printed `prefill 1 positions`, `step 0 pos 1`, and `[spec] ... prompt=0 s=2`, and both emitted
      the identical 24 tokens `223 643 27 15 397 5029 515 260 ...`. The labels "ctx 480" and
      "ctx 3000" were `seqmax`, which is sized from the longest prompt in the FILE
      (`480+24+7+8=519`, `3000+24+7+8=3039`) — an allocation, not a context. Two runs of the same
      computation at ctx 1..9, reported as a 6x context sweep. That is the whole of the evidence
      that retired 1.2 and 1.5.

      **THE ATTRIBUTION** (`evidence/decode_loop/dprof_ctx_0p4.txt`, one server load, seqmax 16384,
      145 steady-state verify steps at four depths 772..12,406, `tools/dprof_ctx.py`). `b|VB` holds
      the realised verify width fixed, because width sets bytes-per-forward and correlates with
      context (0.3); it is the number to read.

      | mark | ms/1000 ctx | `b\|VB` | ms at ctx 768 | ms at 12,288 | share of step slope |
      |---|---|---|---|---|---|
      | **STEP (verify+draft)** | 6.488 +/- 0.332 | **6.969 +/- 0.112** | 126.3 | 203.9 | 100 % |
      | **`draft:main_kv`** | 3.867 +/- 0.001 | **3.867 +/- 0.001** | 3.30 | **47.87** | **55 %** |
      | `ATTENTION` | 3.021 +/- 0.146 | 3.173 +/- 0.109 | 52.3 | 87.4 | 46 % |
      | &nbsp;&nbsp;`cattn:indexer` | 1.472 +/- 0.040 | 1.515 +/- 0.029 | 5.80 | 22.20 | 22 % |
      | &nbsp;&nbsp;&nbsp;&nbsp;**`i:topk`** | 0.871 +/- 0.021 | **0.872 +/- 0.021** | 2.58 | **13.47** | **12.5 %** |
      | &nbsp;&nbsp;&nbsp;&nbsp;**`i:score`** | 0.609 +/- 0.029 | **0.644 +/- 0.018** | 0.92 | **6.58** | **9 %** |
      | &nbsp;&nbsp;`cattn:sparse` | 0.695 +/- 0.050 | 0.709 +/- 0.050 | 11.00 | 21.17 | 10 % |
      | `MoE` | -0.380 +/- 0.198 | **-0.079 +/- 0.029** | 35.8 (med) | 35.8 (med) | ~0 |
      | every GEMM (`q:wq_a/b`, `o:wo_a/b`, `moe:w1w3/w2`, `cattn:ogroup`) | | within 1 SE of 0 | | | ~0 |

      **The instrument reproduces the fit it was built to explain, which is the check that it is
      measuring the right thing.** 0.3's wall-clock fit: `b = 7.362 +/- 0.370`, width-controlled
      **7.029 +/- 0.28**. This run's dprof step total: `6.488 +/- 0.332`, width-controlled
      **6.969 +/- 0.112** — a 0.9 % difference on the number that matters. The per-mark slopes sum
      **exactly** to the step slope (the tool prints that identity rather than assuming it), so
      nothing in the step is unaccounted for. The wall-clock legs also reproduce 0.3 point for
      point (214.7/215.9 vs 215.33 at ctx 12.3k; 181.5/182.5 vs 179.2 at 6.1k; 160.8/162.6 vs
      160.5 at 3.1k), so **`DSV4_DPROF` costs nothing measurable** and this is a profile of the
      shipped cost, not of an instrumented one.

      **1.2 AND 1.5 ARE UN-RETIRED, and the numbers they were retired on were off by 100x.**
      0.2 reported `i:topk` 0.10 ms and `i:score` 0.02 ms. At ctx 12,288 they are **13.47 ms** and
      **6.58 ms** — **135x** and **330x** — and together they carry **1.52 of the 6.97 ms/1000**.
      They grow because they are O(context) by construction and 0.2 ran them at context 9.

      **AND A LARGER ITEM THAN EITHER WAS FOUND, because the draft side had never been timed at
      all.** Every dprof mark in this repo lived in the VERIFY stack, so `dprof_report`'s TOTAL has
      only ever described part of a decode step. `src/engine.cu` calls `dspark_main_kv` **NSTAGE=3
      times per step over the full `ctxlen`** — a from-scratch fp8 GEMM + rmsnorm + rope + quant
      over EVERY position in the context, recomputed on every single token. It is **47.87 ms at
      ctx 12,288, the largest single row in the step**, and **55 % of the entire context term** —
      more than `i:topk`, `i:score` and `cattn:sparse` combined. Its R^2 against context is
      **1.000**: it is not noisy, it is arithmetic. This is now item **1.0**.

      **What is NOT the context term.** MoE is flat (`-0.079 +/- 0.029` per 1000 with width fixed);
      its apparent -0.38 uncontrolled is a width artifact and disappears when width is held. Every
      dense GEMM is flat. `cattn:ogroup`, 15.8 ms and the second-largest verify row, is flat. So
      Term B is **not** a bytes problem — at ctx 6592 its byte floor is 0.509 ms/forward
      (DECODE_ZENITH_FINDINGS 1.2) against ~43 ms measured, still ~85x off roofline. It is serial
      and redundant work: one recompute (`draft:main_kv`) and one selection sort (`i:topk`).

      Instruments, both justified before building and both of which paid: (1) `src/engine.cu` now
      calls `dprof_report` once per verify, tagged with the context it was taken at — it had called
      `dprof_init` since the marks were written and never reported, so **every mark the server ever
      recorded was discarded**, which is why the only profiles that existed came from decode.cu's
      argv[2]-length K-sweep. (2) Three new draft-side ids (`draft:main_kv/block/head`) placed after
      `DP_ARGMAX` so historical tables stay comparable. (3) `tools/dprof_ctx.py` fits the ladder's
      own linear model per mark with standard errors, because once a mark is flat R^2 is the wrong
      statistic and the question is whether an interval covers zero. (4) `--targets/--no-control` on
      `decode_fit_probe.py`; (5) `scripts/dprof_ctx_run.sh`, detached, health-gated, refuses to run
      a binary older than the sources it must describe, and named in `detach_audit.sh`.

      Bit-exactness: nothing numeric changed. Every added call is `dprof_begin/end`, which returns
      on `!g_dprof_on`, and the reporting block is behind `DSV4_DPROF`. The four CPU gates pass in
      preflight on every server start.

      **One open lead, recorded and NOT chased here.** At the 6144 point `cattn:q_proj` is cleanly
      bimodal — 18 steps at 10.3 ms and 17 at 14.7, with `q:wq_a` 1.71 vs 5.47 **at the same VB=2
      and the same context**. A 3.2x swing in a GEMM at fixed shape is not explained by width or by
      context, it is flat in context (R^2 0.005) so it does not touch this attribution, and it is
      worth its own item.


## Phase 1 — the context term

- [x] **1.1** Warp-parallel top-k (all four kernels). **14.2x at ctx 6592, 24.7x at 24k**,
      bit-identical on nine shapes and three distributions (`tests/gate_topk_warp.cu`).
- [x] **1.0** **Cache the DSpark main-KV prefix instead of recomputing it every token.**
      **DONE 2026-08-20 (iteration 1). +24.4 % tok/s at ctx 12,282, bit-exact, tau unchanged.**
      Full write-up: `wiki/kernel-optimisations.md` §2.5 and `wiki/context-scaling.md`.

      **What shipped.** `dspark_main_kv_upto` (`kernels/dspark_attn.cu`) computes only rows
      `[valid, s)` and advances a high-water mark; `src/engine.cu` and `src/decode.cu` call it
      instead of rebuilding all `ctxlen` rows NSTAGE=3 times per token. The three paths that can
      invalidate a prefix — `prefill_full`, `extend`, `rewind_to` — clamp the mark down; everything
      else may only grow it. Per-step cost is O(tokens committed) instead of O(context); over a
      generation, O(n) instead of O(n^2). `DSV4_MAINKV_CACHE=0` restores the old path exactly and is
      the A/B arm.

      **Measured, PAIRED, same corpus, baseline arm first so drift penalises the cached arm.** On
      every point at ctx >= 1536 both arms produced **bit-identical tau and mean verify width
      rep-for-rep** (6/6 legs), so each rep pairs with its twin and the delta is the kernel alone:

      | ctx | before ms/fwd | after | paired delta | band | speedup | tau b / tau a | tok/s |
      |---|---|---|---|---|---|---|---|
      | 12,282 | 215.65 | 170.25 | **-45.40** | [-45.9, -38.5] | **1.267x** | 1.652 / 1.652 | 7.67 -> 9.54 (**+24.4 %**) |
      | 9,213 | 203.51 | 167.27 | -36.24 | [-36.9, -34.3] | 1.217x | 1.969 / 1.969 | 9.69 -> 11.79 (+21.7 %) |
      | 6,132 | 178.89 | 154.71 | -24.18 | [-24.7, -23.4] | 1.156x | 1.718 / 1.718 | 9.59 -> 11.10 (+15.8 %) |
      | 3,069 | 160.46 | 148.19 | -12.27 | [-12.7, -12.0] | 1.083x | 1.626 / 1.626 | 10.07 -> 10.91 (+8.3 %) |
      | 1,536 | 147.67 | 141.15 | -6.52 | [-6.9, -6.2] | 1.046x | 1.701 / 1.701 | 11.54 -> 12.07 (+4.5 %) |
      | 249 | 128.25 | 125.28 | -2.97 | [-7.7, +1.0] | 1.024x | 1.646 / 1.636 | 13.05 -> 12.92 |

      Bands are **+/-1 %** against the 3.5 % run-to-run spread, because pairing removes verify-width
      variance rather than averaging over it. `tau` is unchanged to three decimals wherever the leg
      is reproducible — the required signature of a change that alters no arithmetic.

      **The one disturbed leg, stated exactly:** `sweep-t12288-r3` came in at 204.87 against a
      same-arm cohort of 165.7-176.5 at identical tau and width, and its *before* twin (216.55) is
      unremarkable — so it is the after leg that was disturbed, and it is the whole of that point's
      22.7 % spread. It is **kept** in the medians above, which are robust to it, and **excluded**
      from the regression below, which is not. Both are reported: with it the slope is
      `3.190 +/- 0.278`, without it `3.604 +/- 0.076`. The conclusion is the same either way and the
      exclusion is not doing the work.
      Evidence: `evidence/decode_loop/fit_1p0_paired.txt`, `fit_1p0_base.txt`, `fit_1p0_cache.txt`.

      **The context term fell 44 %, and it is the term 0.4 predicted would fall.**
      `fwd = 132.24 + 7.220 x ctx/1000` (R^2 0.976) -> `129.96 + 4.006 x ctx/1000` (R^2 0.888).
      Regressing the paired saving directly on context over 30 of the 31 exactly-paired legs:
      **`3.604 +/- 0.076 ms per 1000`, R^2 0.988**, against the `3.867 +/- 0.001` that 0.4's dprof
      attributed to `draft:main_kv`. **93 % of it, and the 2 SE band [3.45, 3.76] does NOT cover the
      prediction** — 0.26 ms/1000 is unaccounted for and is recorded rather than rounded away. The
      residual per-step main-KV work (`acc+1` rows x 3 stages, 15 launches, 3 dkmalloc/sync/free
      triples) is context-*independent* and belongs in the intercept, which is where the -1.42 ms
      also went, so it does not explain a slope.

      **BIT-EXACTNESS: gated by memcmp on the whole buffer, 704 calls, 2.59 M retained rows, 0 FAIL.**

      | gate | scope | result |
      |---|---|---|
      | `tests/gate_mainkv_incr.cu` | no checkpoint, 2048 x 512 floats, 22 split points, both GEMM paths | PASS |
      | ^ negative control: rope offset dropped | same | **FAIL**, 130,624/1,048,576 floats |
      | ^ negative control: GEMM pin dropped | same | **FAIL at g_tc_fp8=1 only, 377 floats, last ulp** |
      | `build/dsv4-server` in situ, `DSV4_MAINKV_GATE=1` | 384 calls, ctx to 12,281, 2,023,320 rows kept, all 3 invalidation paths | 0 FAIL |
      | `build/decode` in situ | 320 calls, ctx to 3,130, 568,509 rows kept | 0 FAIL |
      | LOSSLESS gate (spec vs base AR) | every decode run | PASS |

      Each recomputes all `s` rows with the **untouched** `dspark_main_kv` into a private buffer and
      memcmps the full `[s, HEAD_DIM]` range, aborting on the first differing float. The load-bearing
      detail is the **GEMM pin**: `fp8_block_gemm` dispatches on M (M=1 GEMV, M in [2,8] NVFP4
      overlay, larger -> `tc_fp8_gemm`), and the incremental delta is 1-6 rows where the from-scratch
      call was thousands, so the naive version would compare a GEMV against a tensor-core tile and
      lose bit-exactness for a reason unrelated to caching. The incremental path pins to
      `tc_fp8_gemm`, whose dispatch depends only on N and K. Second trap, also covered: `cosT/sinT`
      must be offset by `r0` or a sub-range rotates every row by the wrong angle.

      **The token-id half of the invariant could NOT be tested, and that is a finding, not a gap.**
      The server A/B reported 21/52 completion hashes differing; on `build/decode` (argmax, no seed)
      2 of 3 points diverged. **Two same-arm controls settled it, and they agree.**

      *Control 1, `build/decode`:* `DSV4_MAINKV_CACHE=0` on BOTH sides still diverged from itself, at
      ctx 1,024 at the identical id index 1067 with the identical token pair (223 vs 5115) — the
      cached path never executed in either arm. Baseline-run-2 and the cache-on run agree with each
      other; the first baseline was the outlier.

      *Control 2, `build/dsv4-server`, a full second baseline sweep from an independent server start:*

      > **base-vs-base2 differs on 21 of 52 legs — the SAME 21 legs as base-vs-cache.** Not merely
      > the same count: the same set, both symmetric differences empty. The cache code path never ran
      > on either side of that comparison.

      Control 2 also validates the A/B **timing** measurement, which is the part that actually
      ratchets: with no change at all it reports **0.998x-1.000x** at ctx 12,410 / 9,341 / 6,260 /
      3,197 / 1,536, where the real A/B reported 1.267x / 1.217x / 1.156x / 1.083x / 1.046x. The
      harness reads "no speedup" when there is no speedup.

      So the engine is not run-to-run reproducible once a run is long enough, and token ids cannot
      test anything past that point. See the amendment to hard invariant 1, item **1.9**, and
      `wiki/measurement-and-traps.md` §12. Evidence: `genout_1p0_identity.txt`,
      `genout_1p0_determinism.txt`, `fit_1p0_determinism.txt`, `mainkv_determinism.log`.

      **Second-order, as predicted and free:** the per-step scratch `dkmalloc`/`dkfree` of
      `s*DIM + s*(DIM/128)*4` — 51.8 MB per stage, 155 MB per step at ctx 12,288 — is now sized by
      the delta, ~12 KB.

      Scripts, all detached and all named in `detach_audit.sh`: `mainkv_ab_run.sh` (the A/B),
      `mainkv_verify_run.sh` (token ids + server determinism), `mainkv_decodegate_run.sh` (in-situ
      memcmp on `build/decode`), `mainkv_determinism_run.sh` (the same-arm control).

- [ ] **1.2** **CONFIRMED BY 0.4, 2026-08-20 — this kernel is 12.5 % of the context term.**
      Retired on 0.2's "top-k is 0.10 ms". 0.4 measured `i:topk` at **13.47 ms at ctx 12,288** and
      a slope of **0.872 +/- 0.021 ms per 1000 context** (R^2 0.922, width held fixed). 0.2's
      0.10 ms was real but was taken at **context 9** — both of its runs decoded from position 1,
      see 0.4 — so it was 135x low. The retirement is void and the item is live. It ranks BELOW 1.0
      (0.87 against 3.87 ms/1000) and above everything else in this phase. Single-CTA radix select to replace the warp scan. Reference: SGLang's
      `deepseek_v4_topk.cu` (Apache-2.0) and TileLang `topk_selector.py`. **Restore descending
      order with a 512-element bitonic sort** — the reference emits in `atomicAdd` order, and
      `sparse_attn` sums selected rows in order, so without the sort this is not bit-exact.
- [ ] **1.3** `seq_len <= topk` early-out. Below ctx 2048 every row survives and the kernel still
      does the full scan to discover it.
- [ ] **1.4** `cudaFuncSetAttribute` opt-in for dynamic shared memory, or drop the requirement
      entirely (1.2 does). Removes a silent garbage-return above ~49k context.
- [ ] **1.5** **CONFIRMED BY 0.4, 2026-08-20 — this kernel is 9 % of the context term.**
      `i:score` measured **6.58 ms at ctx 12,288**, slope **0.644 +/- 0.018 ms per 1000 context**
      (R^2 0.750, width held fixed), against the 0.02 ms at context 9 it was retired on — 330x low.
      Ranks third, behind 1.0 and 1.2. `index_score` as a GEMM + fused epilogue. Measured 15.2x standalone
      (658 us -> 39 us at T=6000). **FP32/TF32 accumulation only** — an external ablation shows
      FP16 dropping perfect-recall rows from 99.99 % to 91.82 % on this exact operation.

- [ ] **1.9** **Find out why the engine stops reproducing itself part-way through a long run.**
      **Opened by 1.0, 2026-08-20, and it is a correctness item before it is a performance one.**
      **RANKED HERE, BELOW 1.2 AND 1.5, DELIBERATELY.** It is the most interesting thing on this
      page and it is still not the next thing to build: the ladder has spent four of its first six
      iterations on instruments, and 1.0 showed the buffer-memcmp gate is a *stronger* correctness
      instrument than the token ids it replaces — so the invariant has a working substitute and
      this is not blocking. Promote it above 1.2 the moment a kernel change cannot be gated by
      buffer memcmp, or if `tau` itself starts moving between identical runs.
      `build/decode` is argmax, unseeded, single-stream, no HTTP, and its width controller reads the
      draft's own margins (`while (VK < VKCAP && hmarg[VK-1] >= adaptK) ++VK`) rather than the clock
      — so "the faster arm picked a different width" does **not** explain it. Yet two runs of the
      identical binary with the identical env diverge.

      **State the shape precisely, because it is the diagnostic.** In BOTH binaries the divergence
      is a **clean suffix in run order**, not a property of a context:

      * `build/decode`: point 0 (ctx 6, 260 ids) identical in **three** separate runs; points 1
        (ctx 3,072) and 2 (ctx 1,024) differ in every pairing, first at generated token 20 and 43.
      * `build/dsv4-server`: legs 1-31 of 52 identical (including every leg at ctx 12,410, 9,341,
        6,260, 3,197 and 1,536, with `tau` matching to three decimals across three independent
        server starts), then leg 32 onward all differ.

      Generation is autoregressive, so **one** flipped token permanently de-synchronises everything
      after it. The observation is therefore not "context X is nondeterministic" but "there is a
      rare per-step event whose rate rises with context": zero occurrences in 3 x 260 steps at ctx 6,
      but within 20-43 steps at ctx 1,024-3,072. A context-linear reduction with a
      **non-deterministic accumulation order** fits exactly — more terms, more chances that a
      near-tie in an argmax flips. `atomicAdd`-ordered sums or split-K reductions in `i:topk`,
      `i:score` or `cattn:sparse` are the obvious family; note that 1.2's own reference "emits in
      `atomicAdd` order".

      Cheap first probe, in ONE process so there is no load-to-load variation to argue about: run the
      same prompt twice back to back and hash the logits every step, then bisect to the first step
      whose logits differ and dump the per-mark intermediates there. Second probe: re-run with each
      candidate kernel forced to a deterministic order and see which one makes the repeat stable.
      **Why it matters beyond tidiness:** it disables the ladder's primary correctness invariant for
      every remaining item, forcing each onto buffer-memcmp gates; and if a reduction really is
      order-nondeterministic, `tau` is being measured against a moving target.

- [ ] **1.7** `cattn:sparse` — **0.709 +/- 0.050 ms per 1000 context (10 % of the term), measured
      by 0.4**: 11.00 ms at ctx 768 rising to 21.17 ms at 12,288. Note the shape: it nearly
      DOUBLES over the first 3k and then almost flattens (19.22 at 3072, 19.90 at 6144, 21.17 at
      12,288), which is the signature of a top-`k` gather saturating once context exceeds `k`
      rather than of a full scan. So the lever here is the 11 ms FLOOR, not the slope, and it
      belongs to Term A as much as to Term B. Do not open this until 1.0/1.2/1.5 are done — it is
      the smallest of the four and the only one whose mechanism is not yet understood.
- [ ] **1.8** **Explain the `cattn:q_proj` bimodality 0.4 found.** At ctx 6144, at the SAME verify
      width VB=2, 18 steps ran `q:wq_a` at 1.71 ms and 17 at 5.47 ms — a 3.2x swing in one fp8 GEMM
      at fixed shape and fixed context, with `cattn:compress` splitting the same way (0.05 vs
      2.50 ms). It is flat in context (R^2 0.005) so it does not touch 0.4's attribution, but
      `cattn:q_proj` is 14.6 ms of every step and if the cheap mode is reachable on demand that is
      ~4.4 ms/step of Term A for free. Correctness item first: find out which mode is right.
- [x] **1.6** **RESOLVED as pre-existing — proven by a control build, not by argument.**
      Built `build/decode_prechange` from `1a33cfe^` (the two kernel files and the header, before the
      warp top-k) and ran it: **`err711=42`, `err820=21` — the same fault.** The launch error has
      nothing to do with this work. Two earlier calls of mine on this ("it is mine", then "it is
      pre-existing because mla_attn.cu changed") were both reasoning from circumstance; this is the
      experiment that should have been run first and it took one model load.

      Still latent: output correct, LOSSLESS gate passes on every run including the control, tau
      normal. Both sites are `dsync` on return paths of `ogroup_gemm_fp8`, and the bs==1 kernel
      launched there has no launch bounds, no static shared memory and a 1024-block grid, so it
      cannot fail on its own — the flag is stale from an unprobed launch, plausibly a side stream
      (Finding 55 forks one). **Tracked, not blocking.** Worth asking whether the failing launch is
      a dead branch: the NVFP4 wo_a overlay in that function is documented DEFAULT OFF.

      Free A/B from the control, ctx 480: base AR **92.6 -> 89.3 ms/tok**, spec **67.2 -> 65.9**.
      Small here by construction — at short context the top-k was never the cost. The win is the
      slope, and 0.2 measured that as flat.

## Phase 1b — bit-exact packing (`KV_PRECISION_FINDINGS.md`)

- [~] **1b.1** Pack the DSA index cache as real MXFP4: 128xE2M1 + 4xUE8M0 = **68 B** (was 512 B).
      **Primitives done and gated; wiring still to do.** `include/idx_pack.h` carries the layout and
      the exact inverse of the write path; `tests/gate_idx_pack.cu` proves
      **unpack(pack(x)) == x bit-for-bit on 524,288 elements, 0 mismatches**, across tiny (1e-6),
      large (1e5) and zero-heavy rows. 512 B -> 68 B, **7.53x** — the same 68 B/token SGLang ships
      for the DeepSeek-V4 FP4 indexer pool, which is an independent check on the arithmetic.

      **The gate immediately earned itself.** The first implementation took the sign from `v < 0`,
      but `round_e2m1` returns NEGATIVE ZERO for a negative input that rounds to zero, and
      `-0.0f < 0.0f` is false — so the sign was dropped. 16,011 mismatches at **worst |delta| = 0**:
      numerically perfect, bitwise wrong, and invisible to any tolerance. Fixed with `signbit`.

      Remaining: 6 allocation/memcpy sites bake in the 4-byte stride (`src/engine.cu:346`,
      `src/decode.cu:251/444/453/472/831`), the write path in `compressor_emit_group`, and both
      `index_score` read kernels. Acceptance stays `memcmp` on generated token ids.
- [ ] **1b.2** Pack KV dims 0-447 as real FP8 E4M3 + 7xUE8M0 = **711 B** (was 2048 B); RoPE stays
      FP32. Acceptance: `memcmp`. **Unlocks seqmax 32k-64k, which does not fit today.**

## Phase 2 — speculation (accuracy-neutral by construction)

- [ ] **2.1** Re-tune block width AFTER 1.1/1.2. Verify width is currently free only by accident of
      the broken kernel; with Term B small the optimum returns to ~7-9 from an apparent 11-13.
- [ ] **2.2** **Deploy `s3`.** It is promoted, archived, measured at tau 3.8438 / 25.53 tok/s
      against the shipped head's 3.5362 / 22.66 — and `promote_head.py` only archives, it never
      writes the live checkpoint, so **the server has never run it**. Needs a `--head` path in the
      server, or a staged checkpoint. Free ~13 % on the bench suite.
- [ ] **2.3** Use the confidence head at verify time (EVICT-style `argmax E[A(T_k)]/C(k)`). It
      exists and is unused.

## Phase 3 — clocks, then hand back

- [ ] **3.1** `jetson_clocks` — GPU 1386 -> 1575 MHz, EMC 2750 -> 4266. Measured in-repo at
      +3.0-6.4 %. Its own before/after; do not fold into another item.
- [ ] **3.2** Final `PERF.md` re-run and both coefficients recorded. Then STOP and hand back.

## After the roofline — the long-horizon pivot

Not part of the decode loop. When the loop stops, the next programme is agentic trace capture and a
distribution-matched draft-head fine-tune (`S5_RECIPE.md`, `DECODE_ZENITH_FINDINGS.md` Phase 2).
The highest-leverage change identified: **harvest `(h_40/41/42, p_target)` from live verify
forwards** — zero marginal compute, on-policy and distribution-matched by construction, and it
removes the 240-agentic-prompt ceiling that caps the current corpus.
