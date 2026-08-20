# Measurement methodology, and the traps that earned it

Every rule on this page exists because breaking it cost this project a cycle. `LEVERS.md` §6 carries
the full numbered trap list; this page is the reasoning behind it.

---

## 1. How a number becomes trustworthy here

A performance claim is admissible only if **all** of these hold:

1. **Correctness gate first.** A unit gate against a PyTorch oracle, or `memcmp` for changes that
   are bit-identical by construction.
2. **The LOSSLESS gate**, for any speculative number: `first 8 tokens match base AR -> PASS`.
3. **Full-model, end-to-end.** Not `gemm_bench`. The bench overstates systematically (F47).
4. **Clean run.** No `DSV4_DPROF`, `DSV4_KSWEEP`, `DSV4_SPECPROF`, `DSV4_BLKSWEEP` or `DSV4_MEMTRACE`
   in the binary. Clocks pinned (`jetson_clocks`), caches dropped.
5. **One change per measurement.**
6. **Paired**, if the effect is under ~1.5 %: nine spec verifies pairing 1:1 at identical `K` and
   identical accept counts. End-to-end tok/s alone cannot resolve sub-1 % effects.
7. **Bands, not points.**

---

## 2. The trap that created rule 2

**F68 — a fake +28 % that passed every gate.** Split-K changed the floating-point accumulation
order; the shifted numerics changed which draft tokens were accepted, producing a genuinely faster
run that was decoding a *different sequence*. Cosine gates cannot see this. Only comparing the
emitted tokens against base AR can.

## 3. The trap that created rule 3

**F76 — the extreme case.** A free, bit-identical, `gemm_bench`-verified **−7…−15 %** on the
`ogroup` kernel was **+0.1 %** in situ. The kernel spends 62.3 % of its 13.0 cycles between issues
stalled on an L1TEX scoreboard: it is *latency*-bound, and deleting instructions from a kernel
waiting on memory returns the memory latency, which is zero.

**Corollary:** before optimising a sub-roofline kernel, establish *why* it is sub-roofline.
Bandwidth-bound and latency-bound need opposite fixes.

## 4. The traps that created rule 6

**F79 → F81.** A single clean-vs-clean control suggested a ~1.5 % cross-run floor. A *second*
identical-arm pair changed the reading entirely: the instrument reproduces to ~0.2 %, and the 1.5 %
was an **occasional systematic shift**, not per-run noise.

So: a single cross-run claim under ~1.5 % is unsupportable (you cannot tell which pair you drew),
but a sub-1 % effect **is** resolvable with two pairs. This matters because everything remaining in
`LEVERS.md` §4 is sub-1 %.

## 5. The probe-distribution trap

**F65, F70, F59, F84.** A microbenchmark whose input distribution differs from production selects
the *wrong parameter*, confidently.

- `RB` was chosen twice from a probe whose grouping clamped rows-per-expert at 2, so no tile ever
  needed chunking and the sweep could not see what `RB` is for. On the real histogram the ranking
  **inverts**.
- Adaptive verify width was "worth 7×", measured on the one prompt where it does nothing, against a
  prefill that F62 later proved was broken. **A parameter fitted on broken data is fitted to the
  breakage.**

## 6. Counting the wrong quantity

**F84 → F85.** The prefill MoE redundancy was reported as 3.26× from a **tile** count. The weight
load sits inside the `rb` loop, so a tile costs `ceil(me/RB)` reads; the real figure is **11.26×**.
Worse, the fix implied by the wrong number — "use larger tiles" — would have moved **nothing**,
because traffic depends on `RB` and the tile cap does not enter it.

**How it was caught:** not by the instrument. By arithmetic. Backing out the prefill-only union gave
**173 experts against a hard maximum of 160**. An impossible intermediate is worth more than a
plausible one — a slightly-wrong 2.27× would have shipped unnoticed.

## 7. Host time is not critical-path time

**F82 → F83.** 10.19 ms/round measured inside `cudaMalloc`/`cudaFree` on a drained GPU, priced at
+5–7 %, delivered **+0.41 %**. The time was real; it was **host** time overlapping device work.

## 8. Instruments that cannot see themselves

- **`dprof_init()` was called only inside the `DSV4_KSWEEP` branch**, which runs *after* the prefill
  — so the first prefill profiling run silently dropped every mark and produced a log with no
  `[dprof]` line at all. That is the cheap failure. The expensive one is a report that looks
  complete because the marks it is missing never announce themselves.
- **The harness classified runs clean-vs-profiling by grepping `^[dprof]`** while the engine prints
  five instrument markers, so a `DSV4_SPECPROF` run (+1.01 %, measured) scored as a *clean
  re-baseline*. Fixed to match the class, not the instance.
- **The MoE union counters were cumulative and never reset**, so a prefill byte count swept up 400+
  bs=1 decode calls that are redundancy 1.0 by construction, diluting the average.

**Rule earned:** a guard that recognises one of five instruments is not a guard. Enumerate from the
source of truth.

## 9. Self-checks that cry wolf are still worth keeping

`dprof_report` has a parent/child consistency check. It fires an `*** INVALID` on the prefill table
because `DP_C_COMPRESS` was assigned to a region *outside* `DP_ATTN` — an artifact of an id choice,
not an inconsistency (the 99.98 % accounting settles it). **It was left firing**, because that same
check is what exposed the first run's missing marks.

## 10. Process traps in the harness itself

- **`pgrep -f <script>` matches the shell running the grep** — reported "executor RUNNING" twice
  while idle. Use `pgrep -x` or the lock the process actually holds.
- **The auditor reviewed a sha and pushed a ref.** `HEAD_SHA` was captured at the top, checks ran
  against it, then `git push origin main` published *whatever HEAD had become* — shipping a commit
  it never reviewed. The audit log recorded the contradiction in its own entry (header sha from the
  audit, subject line from a different commit) and read as normal. Now pushes the reviewed object.
- **A documented manual workaround for a harness bug is not a mitigation if the harness runs last.**
  The executor correctly hand-set a counter; the harness clobbered it thirty seconds later, leaving
  a state note describing a value no longer in the file.
- **An arena constant fitted to the 6-token gate prompt became a silent prompt-length ceiling** —
  invisible because the gate prompt never reaches it (F62's shape, twice).

## 11. Gate the shape range the engine spans

**Trap 1, and the origin of several others.** The canonical prompt is 6 tokens (`PSp=5`). It cannot
reach any `bs>16` code path — which is exactly how F62 (garbage prefill for every prompt ≥18 tokens)
survived every gate in the project.

`gate_prefill_len` now varies length deliberately. **A masked read is not a safe read**: F62's row
masks made the reads legal and hid a missing loop.

## 12. The engine stops reproducing itself part-way through a long run, so "bit-identical token ids" is not a test there

**Found proving ladder 1.0, 2026-08-20.** `DECODE_LADDER.md`'s standing invariant is that a kernel
change "either produces byte-identical generated token ids, or ships behind the LOSSLESS gate".
That invariant assumes the engine is a function of its input. **At context it is not.**

The A/B for 1.0 reported 21 of 52 completion hashes differing between arms. The obvious reading —
the cached prefix is wrong — was wrong, and it took three runs to establish that:

| run | arms compared | result |
|---|---|---|
| server A/B, temperature 1.0 | cache off vs cache on | 21/52 legs differ |
| `build/decode`, **argmax, no seed, no HTTP** | cache off vs cache on | point 0 identical (266 ids); ctx 3,072 and 1,024 diverge at generated token 20 and 43 |
| `build/decode`, **`DSV4_MAINKV_CACHE=0` on BOTH sides** | cache off vs cache off | **same divergence** |
| server, full second baseline sweep, independent start | cache off vs cache off | **the SAME 21 of 52 legs differ** — same set, both symmetric differences empty |

The last two runs are the whole finding. The cached code path never executed on either side, and the
binary **still disagreed with itself** — at ctx 1,024, at the *identical* id index 1067 with the
identical token pair (223 vs 5115). The first baseline run was the odd one out: baseline-run-2 and
the cache-on run agree with each other. A comparison whose control fails cannot convict anything.

**Describe the shape correctly or you will chase the wrong bug.** It is tempting to write this up
as "context ≥ 1024 is nondeterministic". It isn't. In both binaries the divergence is a **clean
suffix in run order** — `build/decode` point 0 identical in three runs then points 1 and 2 differ;
the server's legs 1–31 of 52 identical (with `tau` matching to three decimals at ctx 12,410 across
three independent server starts) then leg 32 onward all differ. Generation is autoregressive, so
**one** flipped token permanently de-synchronises everything after it. What is actually observed is
a **rare per-step event whose rate rises with context**: zero in 3 × 260 steps at ctx 6, but inside
20–43 steps at ctx 1,024–3,072.

That shape points somewhere specific — a context-linear reduction with a **non-deterministic
accumulation order**, where more terms mean more chances a near-tie in an argmax flips. It is also
**not timing-dependent in the obvious way**: `decode.cu`'s width controller reads the draft's own
margins (`while (VK < VKCAP && hmarg[VK-1] >= adaptK) ++VK`), not the clock, so "the faster arm
picked a different width" does not explain it. Cause unknown; it is now ladder item 1.9.

> **The rule.** A token-id comparison is only evidence if the *same-arm* control was run and passed.
> Run it first, not after the result surprises you. Where the control fails, fall back to comparing
> the **intermediate buffer** rather than the output.

**The same control earns its cost twice, because it also calibrates the timing half of the A/B.**
Running the baseline arm against itself end-to-end reported **0.998×–1.000×** at ctx 12,410 through
1,536 — where the real comparison reported 1.267×–1.046×. A null A/B that reads null is the cheapest
available evidence that a measured speedup is not an artifact of the harness, the corpus, or the
order the arms ran in, and it costs one extra arm.

**And the fallback is the stronger instrument anyway.** 1.0 was ultimately proved by memcmp'ing the
whole `[s, HEAD_DIM]` main-KV buffer against the untouched from-scratch function on every call —
704 calls across two binaries, contexts to 12,281, 2.59 M retained rows, zero mismatches, aborting
on the first differing float. That proves the changed function's *entire output* is identical, which
is strictly stronger than proving that one downstream consumer happened to emit the same tokens. A
buffer comparison is also immune to the nondeterminism above, because both sides are computed
inside the same process on the same step from the same `main_x`.

## 13. A gate that passed, reported as a failure, because the probe it rode on kept no records

Same item. Phase 1 of the A/B ran the in-situ bit-exactness gate and the harness printed:

```
[1.0] gate probe rc=1
[1.0] gate: 9 PASS lines, 0 FAIL lines
[1.0] FATAL: bit-exactness gate did not pass cleanly. STOPPING before spending three more loads.
```

The gate had passed — 384 checks, 2,023,320 retained rows, zero failures. What returned 1 was
`decode_fit_probe.py`, which **refuses to bank a record below 64 completion tokens** (correctly —
CLAUDE.md forbids writing records for items that generated nothing usable). The phase had asked it
for `--max-tokens 64`, the model emitted 32, so it wrote **0 usable records and exited 1** with the
gate underneath it perfectly healthy. The stop condition `[ "$rc" != "0" ]` conflated *the vehicle
failed to collect timings* with *the thing being gated is broken*, and cost a full run.

> **Rule.** When a gate rides inside a harness, the gate's verdict must come from the gate's own
> output, not from the harness's exit status. Check `nfail`/`npass` first and treat the carrier's
> `rc` as a separate, differently-named condition — they answer different questions.

## 14. A phase that ran nothing, and reported "finished"

Ladder 1.2, 2026-08-20, and it is §13's rule failing in the opposite direction — the harness reported
the *absence* of a check as a completed check.

Phase 4 of `topk_ab_run.sh` runs `build/decode` for the standing GATE and the LOSSLESS gate. It is
launched through `run_model.sh`, which correctly refuses to start a ~105 GiB load when less than
that is available. It ran **0 seconds after** the previous phase's `server_down` returned:

```
[1.2] 05:25:22 === PHASE 4: standing GATE + LOSSLESS GATE (build/decode, radix ON) ===
REFUSED: only 27 GiB available; a full-model load needs ~105 GiB.
[1.2] 05:25:22 decode finished; gates:
[1.2] 05:25:22 === PHASE 5: reports ===
```

`server_down` waits for the process to *exit*; the kernel reclaims its ~100 GiB of page cache
lazily, and 70 seconds later 106 GiB was available. So the phase printed "decode finished; gates:"
followed by nothing at all, and the script proceeded to the report phase and exited 0. **A missing
gate looked exactly like a passing one**, which is the same class of defect as CLAUDE.md's
`eval_bfcl_mt.py` "unconditional `return 0`" — a stage that completes against a dead engine.

Two fixes, both needed:

1. **Wait for the resource, do not assume it.** Poll `MemAvailable` up to 30 minutes before Phase 4.
2. **Assert the gate produced output.** `[ ! -s "$DEC_LOG" ] && exit 1`, then
   `grep -qE 'LOSSLESS GATE: .* -> PASS'`.

The second fix immediately caught a third defect in itself: the first pattern written was
`grep -q 'LOSSLESS GATE: PASS'`, and the actual line is
`[spec] LOSSLESS GATE: first 8 tokens match base AR -> PASS`. A gate that had genuinely passed was
reported FATAL.

> **Rule.** An unattended phase must fail loudly when it did **not run**, and its success pattern
> must be pasted from real output rather than remembered. "Finished" and "checked" are different
> claims, and only the log can tell them apart.

---

## 15. Put a leg the change *cannot* touch inside the same sweep

Ladder 1.3 was a ~4 µs/call kernel saving expected to be worth 0.07 % of a forward, against a
measured 3.5 % run-to-run spread. The usual write-up for that is "the delta is inside the noise",
which quotes a spread measured on some other day, on some other corpus, and invites the reader to
take it on faith.

The A/B was instead built so the noise was measured **inside the experiment**. The early-out fires
only at `T ≤ 512`, i.e. ctx ≤ 2048. Two of the seven targets were placed at ctx ~6,250, where
`T = 1565` and both arms therefore execute **provably identical code** — same kernel, same branch,
same instruction stream. Those two legs measured:

| leg | fires | delta ms/forward |
|---|---|---|
| control, ctx 6248 | **no — identical code** | **+0.24** |
| sweep, ctx 6260 | **no — identical code** | **+0.35** |
| sweep, ctx 1664 | yes | +0.10 |
| sweep, ctx 889 | yes | +0.11 |
| sweep, ctx 492 | yes | −0.53 |
| sweep, ctx 249 | yes | −0.21 |
| control, ctx 1656 | yes | −0.06 |

So the statement is not "the effect is within a spread I am quoting at you" but **"the effect is
the same size as the delta between two arms that ran the same code"** — an upper bound on the
instrument established by the instrument, in the same session, at the same clocks, on the same
corpus, with the same thermal history. It also rules out the failure mode a pure noise argument
cannot: a change that is *net negative* somewhere it was not supposed to reach would show as a
control leg moving differently from the affected ones.

> **Rule.** When you expect a small effect, choose at least one point in the sweep where the change
> is structurally inert, and report it in the same table. A null is only credible next to a
> measured zero.

The corollary is a design constraint on the sweep, not just the report: **an A/B whose every leg is
in the affected regime cannot distinguish a real saving from drift.** 1.3's script says so in its
header, before the run.

---

## 16. A fitted slope is only "cost per 1000 context" if the mark is linear in context

1.3's re-attribution fit `cattn:sparse` at **1.694 ± 0.065 ms per 1000 context** (R² 0.731, width
held fixed, 220 verify samples). Item 0.4 had fit the same mark at **0.709 ± 0.050**. Two clean fits
with tight standard errors, non-overlapping by 15 SE, on the same engine and the same kernel — and
taken at face value the new one promotes ladder item 1.7 above 1.5, because 1.694 is 2.7× `i:score`.

Neither fit is wrong and nothing drifted. They cover different context ranges:

| ctx | 768 | 1,536 | 3,072 | 6,144 | 12,288 |
|---|---|---|---|---|---|
| `cattn:sparse` ms | 9.35 | 15.72 | 19.22 | 19.89 | 21.17 |
| implied ms per 1000 across the leg | — | **8.29** | 2.28 | **0.905** | 0.208 |

One concave curve. 0.4 fit ctx 3k–12k, the flat part; 1.3 fit ctx 0.4k–6k, which is dominated by the
steep first leg. `cattn:sparse` is a top-`k` gather: it grows with context until context exceeds
`k × ratio` and then stops, because after that it always gathers `k` rows. A linear coefficient for
it is a property of the *fitting window*, not of the kernel, and comparing two such coefficients
across windows is meaningless.

The contrast in the same table is what makes this checkable rather than a caution. `i:score` over
the identical samples: 0.88 → 1.25 → 3.53 ms at ctx 768/1536/6144, i.e. **0.482 then 0.495 ms per
1000**. That is linear, its fitted `b` means what it says, and item 1.5 keeps its rank.

> **Rule.** Before ranking work on a fitted `b`, read the per-point medians and check the leg-to-leg
> slopes agree. If they do not, you have a shape, not a slope — and the lever is probably the
> *floor* (Term A) rather than the coefficient (Term B), which is a different item with a different
> mechanism.

This trap is the direct counterweight to rule 6 in `DECODE_LADDER.md`: re-attributing before you
pick is free and necessary, and it will hand you a number that re-ranks the list for the wrong
reason if you do not check the shape.
