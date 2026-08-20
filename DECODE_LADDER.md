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

**Where the two terms stand now (0.3, 2026-08-19):** `a = 130.98 +/- 2.25 ms` = **1.59x** the
82.18 ms floor (stop wants <= 1.25x); `b x 6592 = 48.53 +/- 2.44 ms` (stop wants <= 5.0 ms).
Neither is met. The fit now reaches context 12,410, so the ladder's stop check no longer has to
extrapolate.

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
next, and this ladder has now spent two of its first three items on instruments. That was defensible
-- 0.3 caught a wrong conclusion in 0.2 and the fit now reaches 12,410 context -- but **the engine
has not gotten faster since the warp top-k landed**, and an instrument that does not lead to a
shipped speedup inside the next iteration is overhead.

So, in order:

1. **Prefer an item that changes a kernel over an item that measures one.** If both are available,
   take the kernel.
2. **An instrument is justified only by naming the optimisation it unblocks**, in the ladder entry,
   before building it. "We need better data" is not a justification; "we cannot choose between 1.2
   and 1.5 without attributing the residual" is.
3. **Every iteration that ships a kernel change must report the before/after on the same corpus**,
   with tau, as a band. That is the ratchet: a number that went up, on a fit that already exists.
4. **If two consecutive iterations produce no kernel change, say so at the top of the ladder entry**
   and take the highest-expected-value kernel item next even if it is less certain. Thrashing on
   measurement is the failure mode this section exists to prevent.

## Hard invariants — the loop must never violate these

1. **Bit-exactness or an explicit gate.** Any kernel change either produces byte-identical generated
   token ids, or ships behind the LOSSLESS gate with the deviation measured and recorded. A change
   that silently alters output is reverted, not debugged.
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
- [ ] **0.4** `DSV4_DPROF` again at **ctx 12k**, and attribute the residual 7.36 ms/1000. 0.2
      attributed the context term at ctx <= 3000 and found `i:topk` 0.10 ms and `i:score` 0.02 ms,
      and **retired 1.2 and 1.5 on those numbers**. 0.3 then measured 91 ms of context term at
      ctx 12,410 — 41 % of the forward — which those two marks cannot account for at any context
      unless they grow. So either something else carries it (`cattn:indexer` was 2.09 ms and is the
      obvious candidate, or the compressed-cache read itself), or the marks do grow past 3000 and
      the retirements were taken on an unrepresentative measurement. Do not act on 1.2/1.5 either
      way until this says which. One server load; the sweep harness already exists.

## Phase 1 — the context term

- [x] **1.1** Warp-parallel top-k (all four kernels). **14.2x at ctx 6592, 24.7x at 24k**,
      bit-identical on nine shapes and three distributions (`tests/gate_topk_warp.cu`).
- [ ] **1.2** **REOPENED 2026-08-19.** Retired on 0.2's "top-k is 0.10 ms", but 0.3 superseded 0.2:
      the context term is 4.1x smaller, **not gone**, and is still 41 % of a forward at ctx 12,410.
      0.2 could not see it because both its runs were at ctx <= 3000 against a 3.5 % spread. The
      retirement is therefore unsupported. Whether the residual is *this* kernel is 0.4's job --
      do not assume it, which is the exact error that produced the retirement. Single-CTA radix select to replace the warp scan. Reference: SGLang's
      `deepseek_v4_topk.cu` (Apache-2.0) and TileLang `topk_selector.py`. **Restore descending
      order with a 512-element bitonic sort** — the reference emits in `atomicAdd` order, and
      `sparse_attn` sums selected rows in order, so without the sort this is not bit-exact.
- [ ] **1.3** `seq_len <= topk` early-out. Below ctx 2048 every row survives and the kernel still
      does the full scan to discover it.
- [ ] **1.4** `cudaFuncSetAttribute` opt-in for dynamic shared memory, or drop the requirement
      entirely (1.2 does). Removes a silent garbage-return above ~49k context.
- [ ] **1.5** **REOPENED 2026-08-19**, same reason as 1.2: the 0.2 measurement it was retired on
      is superseded. `index_score` reads T x 128 per layer, which is genuinely O(context) and a
      real candidate for the residual 7.36 ms/1000. Gate on 0.4. `index_score` as a GEMM + fused epilogue. Measured 15.2x standalone
      (658 us -> 39 us at T=6000). **FP32/TF32 accumulation only** — an external ablation shows
      FP16 dropping perfect-recall rows from 99.99 % to 91.82 % on this exact operation.

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
