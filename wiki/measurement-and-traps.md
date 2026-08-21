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

---

## 17. A launch that failed, and a `cudaStreamSynchronize` that called it success

DECODE_LADDER item 1.4, 2026-08-20. The four warp selection-sort top-k kernels stage the whole score
row in **dynamic** shared memory, so they are launched with `~4T` bytes. Measured on this box
(`/tmp/smem_probe.cu`, reproduced as a permanent leg of `tests/gate_topk_radix.cu`):

| T | dynamic smem | `cudaGetLastError()` after the launch | `cudaDeviceSynchronize()` | output buffer |
|---:|---:|---|---|---|
| 12,288 | 49,152 B | `cudaSuccess` | `cudaSuccess` | correct |
| 16,384 | 65,536 B | **`cudaErrorInvalidValue`** | **`cudaSuccess`** | **untouched** |

Read the third and fourth columns together. **The launch never happened, and the synchronize said
everything was fine.** A shared-memory over-request is caught at *launch configuration* time, on the
host, before anything is enqueued — so there is no device-side work to fault, nothing for the stream
to report, and the error sits in the thread's error slot until somebody calls `cudaGetLastError()`.
The output buffer keeps whatever it held. Here that is `zalloc`'s zeros, so `sparse_attn` would have
attended to KV row 0 for every head, at full speed, forever, with no diagnostic anywhere.

**This defeats every defence this repo had.** `dprobe()` is compiled into the path but inert unless
`DSV4_SYNCPROBE` is set. `dsync()` does read the error slot, but only *warns* unless
`DSV4_STRICT_LAUNCH` is set, and it skips its synchronize entirely while the arena is on. And the
bit-exactness gates cannot see it either: a gate that compares two paths would have had to run
*above* the ceiling to notice, and nothing did, because `seqmax` is 16,384 and the ceiling is
context 49,140.

That last point is the general one and it is why this sits next to §11:

> **Rule.** A launch-configuration failure is invisible to a stream synchronize. Check
> `cudaGetLastError()` immediately after any launch whose *configuration* — dynamic shared memory,
> block size, grid size — is computed from a runtime quantity. And note that the quantity here is
> **context**, which no gate in the suite varied past the engine's own `seqmax`: a limit that only
> bites outside the range your harness spans is a limit your harness certifies as absent.

The fix is `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, ...)`, which on
this device moves the ceiling from 49,152 B to 232,448 B — context 49,140 to 232,436. Above *that*
there is no opt-in on any device, so `topk_scan_smem_optin` (`include/indexer.h`) aborts with the
numbers rather than returning a size whose launch will fail silently. The abort is not defensive
decoration: it is exercised by a forked child in `tests/gate_topk_radix.cu`, which requires
`SIGABRT`. Error handling that is never executed is not error handling.

**Why the shipped path was never affected, and why the item was still worth closing.** Item 1.2's
radix select uses ~7 KiB of *static* shared memory whatever `T` is, so the engine's default path
cannot reach this at any context. What retained the defect was the `DSV4_TOPK_RADIX=0` A/B arm and
the `DSV4_TOPK_GATE=1` in-situ reference — the two instruments that exist to prove a *future* top-k
change still matches the original selection sort.

**CORRECTED 2026-08-20 WHEN THE DEFECT WAS FINALLY RUN THROUGH THE ENGINE, and the correction is the
interesting half.** This section used to end "an instrument that silently produces zeros above a
context nobody has tried yet is worse than no instrument, because it will agree with anything."
Both halves of that were wrong, and only running it found out. `scripts/gate_topk_smem_ctx.sh`
drives `compressed_decode_step_indexer` — the function the server calls once per decode step — at
context 49,207 with `DSV4_TOPK_SMEM_OPTIN=0`, which restores the pre-1.4 behaviour in the same
binary:

| | correct (opt-in on) | pre-1.4 (opt-in off) |
|---|---|---|
| launch | success | **`invalid argument`** |
| `cudaDeviceSynchronize` | success | **success** |
| process exit code | 0 | **0** |
| `out` | 4096/4096 nonzero, 0 NaN | **4096/4096 nonzero, 0 NaN** |
| hash of `out` | `2b36c09ec550fc1a` | **`8f24ad5745233320`** |

1. **It does not produce zeros.** `dtop` comes from `dmalloc`, a bump allocator that does not clear,
   so the consumer reads whatever the arena last held. The output is not obviously broken: a full
   4096-wide vector, every element nonzero, no NaN, no diagnostic, exit 0 — and simply the wrong
   answer. A wrong answer wearing the costume of a right one is a harder failure than zeros,
   because zeros at least look like a bug.
2. **The instrument does not "agree with anything" — it disagrees with everything.** Under the same
   switch the in-situ reference gate prints
   `[topk-gate] FAIL decode ctx=49206 first diff at slot 0: radix 55526 vs warp 0`. The *reference*
   is the arm that failed to launch, so above the ceiling the gate would have condemned the correct
   shipped path. An instrument that fails safe-looking is bad; one that raises a false alarm against
   the code it audits is how a good change gets reverted.

The general form is a rule about gates, not about shared memory: **a gate is evidence only inside
the range it has been run over, and a failure it reports outside that range may be its own.**


---

## 18. The device opt-in maximum is not the kernel's maximum, and the safety margin that failed a passing gate

DECODE_LADDER item 1.4, 2026-08-20, found while extending §17's fix to the third dynamic-shared
launch (`sdpa`, `kernels/attention.cu`). `cudaFuncSetAttribute(fn,
cudaFuncAttributeMaxDynamicSharedMemorySize, cudaDevAttrMaxSharedMemoryPerBlockOptin)` works on the
four top-k scan kernels and returns **`invalid argument`** on `sdpa_kernel`. The difference is not
the block size, the register count or the arch: `sdpa_kernel` also declares 1,040 B of **static**
shared memory (`red[ATT_THREADS]`, `m_sh`, `l_sh`), and the four scan kernels declare none.

Bisecting `cudaFuncSetAttribute` over a pair of kernels differing only in their static shared
memory, then launching at exactly the value found:

| kernel | static shared | max settable dynamic | launch at that size |
|---|---:|---:|---|
| no static shared | 0 B | **232,448 B** — the full opt-in attribute | success |
| `__shared__ float r[256]; float a, b;` | 1,024 B | **231,424 B** — opt-in − static | success |

So the rule on sm_110a is `settable = MaxSharedMemoryPerBlockOptin − the kernel's own static
shared`, and **`cudaDevAttrReservedSharedMemoryPerBlock` (1,024 B here) is NOT a second deduction** —
it is already inside the opt-in figure.

**The part worth carrying is what the wrong version cost.** The first fix subtracted `reserved` as
well, on the reasoning that a conservative bound cannot hurt. It lowered the ceiling by 1,024 B —
256 rows of context — and `tests/gate_topk_radix.cu` immediately aborted on its T = 58,045 leg,
which requests 232,192 B and *had passed minutes earlier*. The margin was not free, it was not
conservative, and it turned a correct launch into `FATAL ... Refusing to issue a launch`.

> **Rule.** A safety margin added by reasoning rather than by measurement is a behaviour change and
> needs the same gate as any other. Prefer `cudaFuncGetAttributes(&fa, fn).sharedSizeBytes`, which
> reports the kernel's actual static usage, over a constant you talked yourself into. If a defensive
> bound makes a passing gate fail, the bound is the bug.


---

## 19. The variance is BETWEEN checkpoint loads, not within them — 0.6 % vs 5.7 %

DECODE_LADDER item 1.4, 2026-08-20, and it was found by a control that nearly was not run.

1.4 changes no device code: `kernels/indexer.cu`, `kernels/compressed_decode.cu` and
`kernels/attention.cu` compile to byte-identical SASS, and the shipped radix path does not evaluate
any of the changed macros. Its standing `build/decode` leg nevertheless came back at **19.62 tok/s
against the 20.89 recorded for 1.3 three hours earlier — −6.1 %**, with `tau` at 2.87 in both and
the 66 generated token ids byte-identical. An impossible regression, on the same prompt, the same
parameters and the same box.

The control: rebuild the **pre-1.4** binary from `e4d7c6d` and run the identical leg *now*.

| | 1.3, as recorded (~04:00) | pre-1.4 binary, re-measured (~08:20) | 1.4 binary (~08:00) |
|---|---|---|---|
| tau | 2.87 | 2.87 / 2.87 / 2.87 | 2.87 / 2.87 / 2.87 |
| spec tok/s | **20.89** | **19.69 / 19.65 / 19.72** | **19.65 / 19.73 / 19.76** |
| base AR M=1 | 11.43 | 11.28 | 11.37 |

**The old binary does not reproduce its own number either.** Both arms sit inside 0.6 % of each
other and 5.7 % below a figure that this repo had already written down as a baseline.

Read the two spreads against each other, because that is the finding:

* **Three replicates inside ONE checkpoint load (`DSV4_BLKSWEEP="6:1:1.50,6:1:1.50,6:1:1.50"`,
  each point re-prefilling): 19.65 → 19.76, a 0.6 % spread.**
* **The same measurement across loads three hours apart: 5.7 %.**

So the dominant term is not per-run noise, it is whatever differs between one 100.4 GiB load and the
next — page placement of managed weights, thermal and DVFS history, what else has touched the unified
pool. The ladder's stated "3.5 % run-to-run spread" is roughly right *within* a session and
substantially understates the *between*-session term.

> **Rule.** A number from a previous iteration is not a valid before-arm. Both arms of an A/B must be
> measured in the same session, ideally in the same checkpoint load — that is what `DSV4_BLKSWEEP`
> and the paired-fit protocol exist for. When a comparison against a recorded baseline shows a gap
> you cannot explain mechanically, **re-measure the baseline before you believe the gap.** Rebuilding
> the old binary costs one build and one load; attributing a 6 % phantom to a change costs the
> ladder its ratchet.

This is the throughput-side twin of §12: there, token ids diverged run-to-run and the fix was to run
the same-arm control first. Here, throughput drifts load-to-load and the fix is the same shape.

## 20. A gate that does not compile is a gate that passes by never running (ladder 1.5, 2026-08-20)

`scripts/ixgemm_ab_run.sh` phase 1 refuses to spend a checkpoint load until the in-situ engine gates
pass. It could not start, because **four of them had not built since commit `5c1e047`**:

```
tests/gate_indexer_decode.cu(66): error: too few arguments in function call
tests/gate_compressed_decode.cu(62): error: too few arguments in function call
tests/gate_indexer_graph.cu, tests/gate_compressed_graph.cu — same
```

`5c1e047` ("xin ring: seqmax 8192 → 32768") split `x_full` into `(x_cur, x_full)` on the decode and
verify steps — `const float* xt = x_full + pos*DIM` became `const float* xt = x_cur` — so that `xin`
could be a small ring. Every engine call site was updated. Four **test** call sites were not, and
they are, precisely:

* `gate_indexer_decode` — prefill vs. `compressed_attn_cache_r4` + `compressed_decode_step_indexer`
  at ratio 4. **The in-situ gate for the DSA indexer decode path.**
* `gate_compressed_decode` — the strided twin.
* `gate_indexer_graph`, `gate_compressed_graph` — the **CUDA-graph capture** equivalence gates.

That is the entire in-situ correctness harness for the subsystem ladder items 1.2, 1.3, 1.4 and 1.5
all edit, dead for the whole ladder to date. Nothing reported it, because a build failure in
`build_gate.sh` is not a gate failure — it is an *absent* gate, and the run that consumes it either
skips it or tests a binary from before the change. §13 and §14 are the same shape at the stage
level; this is it at the build level, and it is the more dangerous version because `build/` still
contains a plausible-looking executable with an old timestamp.

**Compounding it, one line above:** `build_gate.sh` had a bare `nvcc` under `set -e`, so the first
link failure (`gate_units`, missing `kernels/nvfp4_dense.cu` since nvfp4 landed) killed the script
and **none of the twelve gates below it rebuilt either**. The file already contained a block written
to prevent exactly this — a `bg` helper that records failures and keeps going — and the raw `nvcc`
lines above that block reintroduced the failure mode the block was written for. Found by *running*
the script, not by reading it.

> **Rule.** Treat "did the gate build?" as part of the gate. `build_gate.sh` now exits non-zero when
> any binary is missing, and a stage that consumes a gate must refuse when the binary is absent or
> older than the source it tests — which is what phase 0/1 of the A/B scripts do. The general form:
> **a gate has three outcomes, PASS, FAIL and NOT RUN, and only the first two are ever printed.**
> Go looking for the third.

The fix here was mechanical (`x` → `x + pos*DIM, x`, restoring the pre-`5c1e047` semantics exactly)
and all four now pass — `cosine=1.00000000 rms=0.00e+00 maxabs=0.00e+00` on the decode pair, and
`cosine=1.0000000 maxabs/|o|=0.00e+00` on both **CUDA-graph replay** gates, which is the first time
the captured path has been tested against the sequential one since the ring landed. The cost of
*not* having them was four ladder items gated on strictly less evidence than their write-ups
claimed.

## 21. dprof reads a saving 15–20 % low, so it attributes but does not ratchet (ladder 1.5)

Same change, three instruments, one session:

| instrument | context-term saving |
|---|---|
| clean paired A/B, 16 legs | **0.572 ± 0.018 ms per 1000** |
| `STEP` mark under `DSV4_DPROF=1` | 0.484 ± 0.009 |
| `i:score` mark under `DSV4_DPROF=1` | 0.463 |

The two dprof numbers agree with each other and both sit ~15–20 % below the clean pair. That is the
expected direction: dprof inserts a per-mark synchronisation, which adds a roughly fixed cost to
every step and dilutes any *relative* difference between arms. It does not bias the *attribution* —
`i:score` still goes from 0.503 ± 0.006 to 0.040 ± 0.001 ms per 1000, and the saving still lands
entirely inside its parent mark `cattn:indexer` (−5.24 ms against −5.22) with the untouched `i:topk`
control flat at 0.72/0.72/0.76 — so dprof answers "where did it go" correctly and "how much was it"
low.

> **Rule.** Attribute with dprof, ratchet with the clean pair. Quoting a dprof delta as the
> throughput result understates a win and would, on a marginal item, turn a real gain into a null.

---

## 22. `tau` reproduced to four decimals; `tok/s` on the same protocol did not — and the promotion gate keys on the second one (ladder 2.2, 2026-08-20)

Deploying `s3` meant re-measuring it and the shipped head back to back on today's engine, because
their archived numbers came from revs `85dbea6c` and `2632540` — before 1.0/1.2/1.3/1.4/1.5. Both
arms ran the frozen 8-prompt protocol, same binary, same session. What came back:

| suite mean | archived | re-measured today | drift |
|---|---|---|---|
| shipped `tau` | 3.5362 | **3.5362** | **0.0000** |
| `s3` `tau` | 3.8438 | **3.8438** | **0.0000** |
| shipped tok/s | 22.655 | 22.1425 | −2.3 % |
| `s3` tok/s | 25.5312 | 24.2512 | −5.0 % |
| base AR tok/s | 13.76 / 13.80 | 11.41 / 11.33 | **−17.4 %** |

Not "close". **Identical to the last recorded digit**, eight days and five decode-kernel rewrites
apart, on both heads, and per-prompt as well as in the mean. That is not luck — `tau` is an exact
draft/target token comparison. It is a property of the head, the prompt and the acceptance rule, and
it is blind to clocks, thermals, page cache and to every kernel on the ladder. **On this protocol
`tau` has no measurement error at all.**

Everything else on the same lines moved.

### 22.1 The consequence for the selection rule

`tools/promote_head.py` fixes its criteria before any head exists, which is the right instinct, and
criterion 4 is:

> Suite-mean tau must EXCEED the incumbent's by more than the measured run-to-run spread (3.5 %).

The docstring says `tau`. **The code compares `suite_tok_s`** — and against an incumbent row read
out of `HEAD_REGISTRY.md`, i.e. a number recorded on whatever engine revision was current when that
head was measured. So the gate is differencing the one column that drifts, across the axis it drifts
along, and calling the result head quality.

It is not hypothetical. `s2` was refused at 24.7575 against s1's 24.515 × 1.035 = 25.373, a margin
of 0.6 tok/s — about 2.5 %. The engine drift between two arbitrary revisions measured here is 2.3 %,
5.0 % and 17.4 % on the three rows above. **A verdict that turns on 2.5 % was being decided by a
quantity that moves more than that for reasons having nothing to do with the head.** Three of the
registry's seven rows are `not promoted` verdicts of exactly this shape.

The fix is not to switch the comparison to `tau` and stop there — throughput is what a user gets,
and a head could in principle buy `tau` while costing draft time. The fix is that **the incumbent
must be re-measured in the same session as the candidate**, which is what ladder 2.2 did, and which
makes the drift common-mode instead of a term in the answer. `tau` is then the cross-session anchor
that says the re-measurement was faithful — as it did here, twice, exactly.

### 22.2 The 17 % that is not in the answer but should be in someone's notebook

The base-AR row above is the loudest number on the page and it is not this item's business, so it
is recorded rather than explained: **the `WARM decode` M=1 measurement fell from 72.5 ms/tok to
87.7 and 88.2 ms/tok** — measured twice today, in two independent loads, on the identical gate
prompt and the identical 7-step averaging.

The reason to be careful rather than alarmed is that **speculative throughput on the same suite in
the same runs is flat** (22.655 → 22.1425, −2.3 %, inside the 3.5 % spread). Spec decode runs the
same forward. A genuine 17 % regression in the forward cannot leave spec throughput unchanged, so
the likeliest reading is that the *measurement* changed, not the engine — a 7-step warm average is
short, and the ladder has since put a main-KV cache (1.0) on a path it warms.

What it definitely does mean is that **every "N.NNx vs base AR" figure in this repo is a ratio of
two numbers only one of which is stable**, and the shipped head's own ratio inflated from 1.6464 to
1.9406 across the ladder without the numerator moving. Quote the two throughputs, not the ratio.
Worth its own ladder item; not worth folding into a deployment.

## 23. A drift control that failed — and the agent that was contaminating its own arms (ladder 3.1, 2026-08-20)

`clocks_ab_run.sh` was built with a kill switch it wrote down before it saw any data: four checkpoint
loads, `A` governed / `B` pinned / `C` MAXN / **`A'` governed again, last**, and

> "If A' comes back to A, the ordering did not manufacture the result; if it does not, the whole
> comparison is void and says so."

It did not come back. **A' was +8.19 % above A on the suite mean — larger than the +5.70 % pinning
was supposed to be worth.** Both arms governed, both at the same nvpmodel, same binary, same
checkpoint, same 9 prompts, 25 minutes apart. By its own pre-registered rule the run was void, and
the +5.70 % it would otherwise have reported would have been session drift wearing a clock's
clothes.

**Two other things in the same run pointed the same way.**

* Arm A' returned `tau = 3.80` on suite prompt 3 where A, B and C had all returned **2.49** — the
  engine failing to reproduce itself (§12, ladder 1.9). That single leg is +42 % on one prompt of
  eight and roughly 5 of the 8.19 percentage points on its own.
* Arm C contains the only backwards leg in 36: prompt 4 at **22.97 -> 14.37 tok/s**, with the core
  clock pinned at a verified 1575 MHz for every sample either side of it. The clock trace is flat
  through it; what moves is **power, 34 W -> 19.5 W** — the GPU went idle mid-prompt.

**The common cause is the agent that launched the arms.** The driving Claude Code iteration was
still working during A, B and C — repo-wide `grep -rn`, file reads — and its ~GB node process
**exited at 11:08:23**, inside the 10-second sample window containing arm C's stalled prompt. Arm A'
ran after it was gone. The arms are ordered, monotonically, by how busy the box's *supervisor* was:
A 24.83 -> B 26.26 -> C 26.61 -> A' 26.01 tok/s (prompts 0 and 3 excluded).

**Re-run on a quiet box, everything tightens by an order of magnitude.** Same protocol, ABA bracket,
nothing else running:

| | first run (agent active) | re-run (quiet box) |
|---|---|---|
| governed-vs-governed drift control | **+8.19 %** | **+1.78 %** (paired +2.03 +/- 0.99 %) |
| `tau` reproduced on every prompt | **no** — prompt 3 flipped 2.49 -> 3.80 | **yes**, all 9, both arms |
| the effect being measured | +5.70 % (void) | +1.99 +/- 0.15 %, 8/8 legs |

> **Rule.** A measurement arm is a single-tenant workload and **the agent supervising it is a
> tenant.** Launch the arms detached, then do nothing until they finish — no repo-wide greps, no
> builds, and in particular do not let a large process exit mid-arm. This is not the same rule as
> "detach the workload" (CLAUDE.md); the arms here were correctly detached and still contaminated.
> "One model at a time" understates it: it is one *workload* at a time.

Note which control caught this and which did not. `tau`, the LOSSLESS gate and the first-token gate
all passed in all four arms — they are exact-output checks and contamination is a timing effect. The
only instrument that saw it was **the repeat of an arm already run**, which is the one thing an A/B
is always tempted to skip because it buys no new comparison.

## 24. The base-AR window times the governor, not the engine — and it is 21 % (ladder 3.1, 2026-08-20)

`WARM decode` is a 7-step average taken immediately after the checkpoint loads. On this box that is
**inside the DVFS ramp**: the GPU has been idle for the ~90 s of loading, both rails sit at their
idle rates (gpc 315 MHz, EMC 2750 MHz), and they take ~2 s of GPU-busy time to reach 1386 and 4266.
A 7-step average at ~88 ms/step is ~0.6 s long. It is measured almost entirely on the way up.

Six loads on 2026-08-20, same binary, same gate prompt, same protocol, differing only in the rails:

| machine state | base AR (`WARM decode`) |
|---|---|
| governed | **88.0 / 88.0 / 88.7 / 88.4 ms/tok** |
| pinned | **72.7 / 72.9 ms/tok** |

**+21.0 to +21.7 %, and it reproduces to a tenth of a millisecond in both states** — which is what
makes it dangerous. This is not noise that averages out; it is a bimodal, perfectly repeatable
number whose mode is set by a machine state nobody was recording. HARDWARE.md called this exact
figure — **+20.7 %** — on 2026-08-07 and named the mechanism, and it still cost the ladder an item:
**2.5 exists entirely because of it**, filed as "base AR fell 72.5 -> 87.7/88.2 and nothing explains
it", with a note that if it were real it would be "larger than everything the ladder has won put
together". It was a clock state.

The tell was in the entry that raised it: speculative throughput on the same suite in the same runs
was **flat**, and spec runs the same forward. A real forward regression cannot leave spec unchanged.
The reason spec was flat is that the suite runs *after* the ramp — the artefact is confined to the
one window measured before it.

> **Rule.** A short average taken immediately after a cold start measures the machine warming up.
> Either pin the clocks or state that you did not — and prefer pinning, which is why
> `run_model.sh` and `run_server.sh` now call `scripts/pin_clocks.sh` and drop a `<log>.clocks`
> sidecar next to every run. HARDWARE.md has required "every future measurement must state whether
> clocks were pinned" since 2026-08-07; **nothing enforced it, so nothing did it**, and every A/B on
> this ladder ran against a ramping governor without knowing. A rule that lives only in prose is
> followed when someone remembers.

---

## 25. A unit gate that stops six times short of the regime it is quoted about (ladder 1.9, 2026-08-20)

`tests/gate_scratch_init` was built to answer Finding 60's leading suspect — does the prefill chain
read uninitialised scratch? It poisons the scratch with `0x00`, `0xFF` and `0x3C`, includes the
arena, and runs `compressed_attn_forward` **at every length 1..29**. It came back bitwise identical
everywhere. Finding 61 then quoted it as the retraction that exonerated the prefill chain, and both
`kernels/compressed_attn.cu` and `kernels/indexer.cu` still carry the comment *"The prefill chain
does not read uninitialised scratch."*

Ladder 1.9 measured the same function's reproducibility as a function of prefill length. It is
byte-identical run-to-run at 16, 32, 64, 128, 129, 132, 136, 144 and 160 positions, and
**nondeterministic at 192 and at every length above, to 3071**. The gate's range stops at 29. It
never touched the regime its result was used to close.

Nothing about the gate was wrong. What was wrong was the sentence built on it: a bounded experiment
was quoted as an unbounded conclusion, and that conclusion then removed the prefill from the suspect
list for two more ladder items.

> **Rule.** State a gate's range in the sentence that quotes its verdict, not only in its source.
> "`compressed_attn_forward` is poison-independent" is a claim about the function; "…at lengths
> 1–29" is the measurement. When the two get separated the first one survives, and it is the one
> that is not true.

The same shape, stated positively: **1.9's own length ladder is the instrument the gate should have
been.** It costs two runs, it is `tools/lhash_compare.py` plus `DSV4_HASH=2`, and it converts "is
this deterministic?" from a yes/no into a threshold — which is what actually names a mechanism.

---

## 26. A field that was never computed compares equal, and equal reads as exonerated (ladder 1.9)

`DSV4_STEPHASH` writes one line per verify carrying the causal chain of that step, and
`DSV4_STEPHASH_LVL=1` skips the two expensive device hashes — `mkv` and `mx` — writing them as
zero. `tools/stephash_compare.py` walks the fields in dataflow order and reports the first one that
differs. On a level-1 pair it duly reported:

```
first differing field: draft
-> mkv, main_x and the draft input all MATCH and the draft's output does not --
   the DSpark draft chain (3 blocks + head) is the nondeterministic component
```

Every word of that is false, and the tool produced it from data that was correct. `mkv` and `mx`
matched because they were **both zero**, because level 1 does not compute them. Run at level 2 the
same comparison names `mkv` as the first differing field and points at the prefill instead — the
opposite subsystem.

The fix is in the tool: a field that is all-zero on both sides is dropped from the ordering, printed
as `NOT MEASURED`, and any verdict that rests on it carries an explicit "that reading ASSUMES … they
were NOT COMPUTED" line.

> **Rule.** An instrument with levels must distinguish *equal* from *not measured*, in the output,
> at the point of the claim. A sentinel that is a legal value of the thing it is a sentinel for
> (zero, for a hash) will be read as evidence exactly when the evidence is absent.

---

## 27. `build/decode` and `dsv4-server` do not share a prefill, and a bit-exactness result on one says nothing about the other (ladder 1.9)

Ladder 1.9's tension, and the thing that resolved it. `build/decode` fails to reproduce its own
prefill above ~192 positions. 1.5's A/B ran 16 legs per arm at contexts 1,656–12,410 across two
independent `dsv4-server` starts and every pair was byte-identical, `tau` to four decimals. Both are
correct, because they are different code:

| | prefill entry point | compressed layer takes | widest compressed call |
|---|---|---|---|
| `build/decode` | `run_layer(prefill=true)` → `cblock_prefill_cache` | `kernels/compressed_attn.cu` | the whole prompt, in one call |
| `dsv4-server` | `Engine::Impl::prefill_full` | `cblock_verify_step`, `kernels/compressed_decode.cu` | `EXT_CHUNK` = **64** rows |

The server chunks its prefill at 64 and never issues a compressed prefill call wide enough to reach
the threshold; `build/decode` issues one call as wide as the prompt. That predicts what was
observed on both binaries, and it means the two are not replicates of each other in any respect that
touches the prefill.

> **Rule.** Name the binary in every bit-exactness claim. "The engine is deterministic" is not a
> property of the checkpoint or of the kernels; it is a property of a call graph, and this project
> has two.

## 28. A tuning default that was a regression at its own design point, and the mechanism story that was wrong (ladder 1.7, 2026-08-20)

**Two failures, one kernel, and they compound.**

**First: the heuristic was never re-measured after the shape it was tuned on moved.** B9 added an
`HPB` (heads-per-block) parameter to `sparse_attn`, sized it on a 1022-token prefill, and shipped
`total >= 640 -> HPB=8`. That comment is careful and explicitly warns about the *opposite* error —
"HPB MUST FOLLOW THE BATCH … HPB=8 would launch 8 blocks onto 20 SMs and starve the machine". It
still shipped a default that, measured today at exactly the shape it was chosen on, is

    hpb=8/smem=0 at m=1022, topk=1277:  328.5 ms   against hpb=1's 263.2 ms   =  0.80x
    hpb=8/smem=0 at m=6,    topk=640:    1.581 ms  against hpb=1's 0.818 ms   =  0.52x

**A 20 % prefill regression and a 1.9x verify regression, live, under a comment that reasons
correctly about why they might happen.** Nothing had re-run the numbers since; every intervening
change to the verify width, the top-k cap and the clock state moved the shape underneath it. The
rule this produces is narrow and cheap: **a launch-configuration heuristic is a measurement, and it
expires like one.** If a default is a number chosen from a sweep, the sweep belongs in a gate
binary that runs in seconds, so re-running it is free — which is what `tests/gate_sparse_hpb.cu`
now is. `nvcc`-and-run cost 90 seconds and would have caught this at any point in the intervening
weeks.

**Second: the mechanism in the comment was the wrong one, and it was wrong in the flattering
direction.** Both B9's comment and `DECODE_LADDER` 1.7 attributed the kernel's cost to **reuse**:
`num_key_value_heads == 1`, so all 64 heads of a query gather the identical KV rows and 63/64ths of
the traffic is redundant. It is a correct, checkable, load-bearing-sounding observation. It is also
**not the binding constraint** — `hpb`, which is precisely the fix for it, measures **1.00x at
hpb=2 and hpb=4 at every shape**. L1 was already catching the reuse. The actual constraint was
instruction issue (32 scalar loads per warp per row on a kernel with 3–6 warps per SM), which no
one had written down anywhere.

What makes this a trap rather than an ordinary wrong guess is that **the redundancy arithmetic is
spectacular and unfalsifiable by inspection**: 168 MB per compressed layer per step, 63/64ths of it
"wasted". A number that large reads as a diagnosis. It was a description. The check that separated
them cost one column in a table — sweep the knob that fixes the hypothesised mechanism *alone*,
before building anything, and see whether it moves. Compare §16: the same discipline of testing the
attribution rather than the story.

**Corollary for gates: a control that never fails has not been shown to be able to.**
`gate_sparse_hpb`'s one-ulp negative control came back "0 of 65536 floats differ" on its fourth
shape — not because the gate was broken, but because the row it perturbed was outside *that*
shape's gathered set. A perturbation that the kernel provably cannot see is not a control. Fixed by
perturbing a sliding-window row, which every shape gathers by construction.

## 29. A dprof mark that timed another stream's work, and reported it as a 3.2x swing in a GEMM (ladder 1.8, 2026-08-20)

`cattn:q_proj` measured **1.71 ms on some verify steps and 5.47 ms on others, at the same verify
width, the same context and the same GEMM shape**. 0.4 recorded it as an open lead worth its own
item, on the reasoning that `cattn:q_proj` is 14.6 ms of every step and "if the cheap mode is
reachable on demand that is ~4.4 ms/step of Term A for free".

**There is no cheap mode. The mark is a stopwatch on a stream, not on a kernel.**
`compressed_verify_step_indexer` records `g_side_fork` and *then* issues `build_qKV` on the main
stream, so the two `compressor_emit_group` calls run concurrently with the q chain (Finding 55/56).
`dprof_begin/end` record CUDA events on the **main** stream, so the interval they measure contains
whatever the *other* stream was doing to the memory system at the time. The emits fire only for
groups completing inside the block — `for j in [pos,pos+K): if (j+1)%ratio==0` — so the mark times
a GEMM-plus-21-layers-of-compressor on some steps and a GEMM alone on others.

**The tell was free and sitting on disk.** The schedule is arithmetic, so it is predictable from the
dprof tag with no new run at all:

    g = #{ j in [ctx, ctx+VB) : (j+1) % 4 == 0 }        # ratio 4 = the 21 indexer layers

`g` classified 0.4's own **153 per-verify tables 153/153**, with a 2.80x gap and *no overlap between
the two populations*, on data collected before the hypothesis existed. The excess is **linear in
`g`**, not merely present: `cattn:q_proj + cattn:compress` is +6.97 ms at g=1 and +14.78 at g=2, so
a block wide enough to straddle two group boundaries pays twice. `tools/qproj_bimodal.py` runs this
on any dprof server log.

**And the correlation was still not the proof.** "The GEMM is intrinsically slower on those steps"
predicts the same classification. The separating experiment is the switch the code already carried,
`NO_ATTN_SPLIT=1`, at fixed VB=2 (`g` correlates with width by construction — P(g>=1) = min(1, VB/4)
— so §the width-control rule applies):

| arm | g | `q:wq_a` | `cattn:compress` | q_proj+compress |
|---|---|---|---|---|
| serial | 0 | 1.65 | 0.05 | 10.16 |
| serial | 1 | **1.66** | **8.34** | 18.46 |
| split | 0 | 1.66 | 0.04 | 10.16 |
| split | 1 | **5.54** | **2.50** | 17.18 |

The 3.2x swing **exists only when the fork does** (1.00–1.02x serial, 3.05x split, 174/174
separated), and the identical time reappears one mark over. The g=0 baseline is **10.16 ms in both
arms across two separate checkpoint loads**, which is what says the arms are comparable at all.

**Three lessons, in order of how much they cost.**

1. **A mark bounded by events on one stream is not a measurement of the kernels issued on that
   stream.** It is a measurement of an interval. Where the engine forks — `g_side` here, `g_side2`
   for the C1 kv chain, `moe.cu` for the shared expert — every enclosing mark on the main stream
   silently includes the side stream's pressure. This is not noise and it does not average out: it
   is a deterministic function of a schedule, which is exactly why it looked like a clean bimodal
   distribution rather than variance.
2. **"Cheap mode / expensive mode" in a mark means look for a conserved quantity before looking for
   a free lunch.** Summing the two marks that split *together* is the whole diagnosis: the sum is
   flat at g=0 in both arms and differs by only the overlap gain at g>=1. Finding 56 had already
   written down "cattn:compress 8.56 -> 0.51, cattn:q_proj 17.03 -> 23.85 (it absorbs the traffic)"
   — the answer was in a comment above the fork, four months before the question was asked.
3. **Outliers are schedule too.** Every `cattn:compress` sample above 20 ms — 8 of 8 across both
   arms, 43–50 ms each, plus the lone 43.90 ms outlier in 0.4's table — is a step whose block
   contains a **ratio-128** boundary, i.e. the 20 strided layers emitting through a path that uses
   no side stream at all. Same mechanism, different ratio, arrived at by asking the same question of
   the tail that had just been asked of the mode.

**What it cost to check: zero GPU time for the classification, two checkpoint loads for the cause.**
What it saved: a 4.4 ms/step phantom deleted from the ladder's headroom, and a first price on the
compressed-KV emit — **7.02 ms on the 64.9 % of forwards that carry one, 4.56 ms/forward amortised,
52 % of its 880 MB byte roofline** — which nothing had ever timed.

## 30. A two-arm gate sweep in which the arms were identical by construction (ladder 1b.2, 2026-08-20)

1b.2's battery ran eight in-situ engine gates **twice**, once with `DSV4_KV_PACK=0` and once with
`=1`, and reported sixteen passes. Fourteen of them were the same run twice.

The switch is read by `kv_pack_init()`, and `kv_pack_init()` is called by exactly two translation
units — `src/engine.cu` and `src/decode.cu`, i.e. the two binaries that own a KV allocation. A gate
binary links `kernels/` and calls the decode functions directly; it never calls `kv_pack_init()`, so
`g_kv_pack` stayed 0 no matter what the environment said. **The environment variable was set, the
process read no environment variable, and the log printed `pack=1 ... PASS`.**

Nothing was wrong with those gates and nothing they proved was false — they are a regression check on
the FP32 path, which is worth having. What was false was the *shape* of the evidence: a table with
two arms in it, where one arm did not exist. An A/B whose arms are set by a mechanism the binary
under test does not participate in will report agreement forever.

**The fix is the one that generalises: make the arm OBSERVABLE in the artefact.** The two graph
gates now call `kv_pack_init()` and print `[graph-gate] KV layout: PACKED (1b.2), 720 B/row` before
they run, so the log says which arm ran rather than which arm was requested — and with that in
place they became a real test of the device-pos store kernels (`kv_store_at`, `kv_store_comp`),
which nothing else exercises. `build/decode` prints the same line for the same reason, and it is how
the engine A/B is known to have actually switched layouts:

    [decode] KV layout: fp32, 2048 B/row, 0.007 GiB for 85 rows x 43 layers
    [decode] KV layout: PACKED fp8+ue8m0 (1b.2), 720 B/row, 0.002 GiB for 85 rows x 43 layers

Same lesson as §22's `tokens_per_verify` probe and 2.2's live proof: **prove which configuration is
resident from a behaviour it changes, not from the flag you passed.**

## 31. The resident-set line cannot price a cache that has not been touched (ladder 1b.2, 2026-08-20)

1b.2's whole justification is capacity, so the natural instrument is the engine's own
`[engine] ready. mem X/Y GiB` line, measured with the packed layout on and off at a fixed `seqmax`.
The arithmetic says the two arms differ by **1.04 GiB at seqmax 16384 and 2.08 GiB at 32768** — 43
window caches, 41 compressed caches and 3 draft main-KV caches, all at 2048 B/row against 720.

Measured: **112.4 vs 112.0** at 16384 and **115.3 vs 115.1** at 32768. A fifth of the predicted
difference, and in one case less.

Nothing is wrong with the arithmetic and nothing is wrong with the line. They measure different
things. `mem` is `cudaMemGetInfo` — physical pages — and on this unified pool a `cudaMalloc` costs
no physical page until something WRITES to it. At `[engine] ready` the KV caches have been
allocated and are empty; the first token of the first prompt is what starts faulting them in. **A
capacity claim measured at load time measures the allocator's promises, not the machine's memory.**

Two consequences, both of which apply beyond this item:

1. **Price a cache from the allocation arithmetic, or from a resident set taken after the cache is
   FULL** — not from the load-time line. The arithmetic here is exact and checkable at two lines of
   `src/engine.cu`; filling a 32,768-row cache to measure the same number costs a 32k-token prefill.
2. **The converse is the dangerous one: an OOM check at load time is not a safety margin.** A
   `seqmax` that loads may still exhaust the pool once the context is actually used, and on this box
   that failure is a whole-machine takedown with no oom-kill line
   ([`oom-and-memory-safety.md`](oom-and-memory-safety.md)). "It came up" is evidence about the
   loader, not about the run.

The same distinction is why `EVALS.md`'s seqmax ceiling was ever a real constraint: it binds when
the context is used, not when it is reserved.

## 32. The first pass through the arena reads scratch it never wrote, so run 1 of a process disagrees with every later run (ladder 1.11, 2026-08-20)

`gate_join_defer` was written to prove 1.11 bit-exact. Its first run said **FAIL: 4,096 floats
differ** — exactly one row, row 0, and row 0 is an **M=1 decode step**, which 1.11 does not touch.
Perfectly reproducible across three runs.

**The null control is the whole story.** Pin BOTH arms to the SAME join position and the row still
differs — including with the *pre-1.11* code in both arms, i.e. with the change out of the picture
entirely. Then duplicate the case: **instance 1 differs, instance 2 of byte-identical data is
clean.** It is not the change and it is not the data; it is being first.

    Case cases[] = { {480,6,2}, {480,6,2}, ... };   // same seed, same weights
    s0=480 decode=6 verify_K=2   4096 / 32768 floats differ   <<<<
    s0=480 decode=6 verify_K=2      0 / 32768 floats differ

**The mechanism.** The arena slab is `cudaMalloc`'d once and never zeroed (`kernels/dscratch.cu`),
and `dfree` is a no-op under it, so every pass hands out the same addresses. A kernel in the M=1
step reads some `dmalloc`'d scratch before writing all of it: on pass 1 that read sees driver
garbage, and on pass 2 and forever after it sees the *previous pass's* bytes, which repeat. The
symptom is therefore "run 1 is the odd one out", not "the results are random" — which is exactly why
it looked like a deterministic kernel difference rather than uninitialised memory.

**Three things to take from it.**

1. **A two-arm gate needs a NULL control — both arms pinned to the same setting — before its FAIL
   means anything.** Without it, a harness bug and a kernel bug produce the identical output, and
   the natural reading is the one that indicts the change you just made. This cost twenty minutes;
   reading it the other way would have reverted a real 0.54 ms/forward win.
2. **`--swap` is not that control.** Reversing the arm order still runs case 1 first, so it cannot
   see a first-in-process effect at all. `--swap` controls for *arm order*; `--same-X` controls for
   *the arm being the cause*. They are different controls and this gate now carries both.
3. **The bug is real and is now ladder item 1.14.** It is one token per process in the engine, so it
   is low on throughput — but every bit-exactness gate in this repo has been silently working
   around it, and gates are the instrument the first hard invariant rests on.

## 33. Pairing does not remove a between-load offset — one arm order could not resolve a 0.9 ms effect, and two could (ladder 1.11, 2026-08-20)

1.11's pre-registered ceiling was **1.29 ms/forward, ~0.9 % of a 140 ms forward**. Run it the
standard way — control arm first so drift penalises the change — and the answer is:

    n=18  paired mean -0.324 ms/forward   sd 0.734   2 SE band [-0.670, +0.022]
    VERDICT: the band COVERS ZERO — this is a NULL, not a win.

By the item's own pre-registration that is a negative result, and it would have been written up as
one. **It is not one.** Run the identical pair again with the ARM ORDER REVERSED and the same
18 legs give **−0.760, 2 SE [−1.109, −0.411]** — a clean win.

**Why pairing was not enough.** `decode_fit_probe.py` runs the same prompt with `seed=1000+rep` in
both arms, so leg `t6144-r3` before and after are the same sample. Pairing removes the variance
BETWEEN LEGS, which is large. It does nothing to a **constant offset between the two arms**, and
§19 of this page measured exactly that offset — the run-to-run spread is 0.6 % *within* a checkpoint
load and 5.7 % *between* loads. 0.6 % of 140 ms is **0.84 ms/forward**, which is the size of the
entire effect. A two-load A/B of a sub-1 ms change is a coin-flip dressed as a band.

**The fix is arithmetic and costs two more loads.** Run 1 measures `effect + drift`; run 2, with the
arms swapped, measures `effect − drift`. Average the two per-leg deltas and drift cancels; half
their difference *is* the drift:

| | paired mean | 2 SE band | legs faster |
|---|---|---|---|
| run 1 — control arm first | −0.324 | [−0.670, **+0.022**] | 13/18 |
| run 2 — change arm first | −0.760 | [−1.109, −0.411] | 14/18 |
| **pooled, drift-free** | **−0.542** | **[−0.852, −0.232]** | 12/18 |
| **drift itself** | **+0.218** | [+0.061, +0.375] | — |

**`+0.218 ± 0.157 ms/forward` is now a measured number: what the second checkpoint load of a session
costs relative to the first.** It is 40 % of the effect 1.11 was chasing. `tools/paired_band.py
--reversed` does the pooling and prints both.

**The standing rule this replaces.** "Run the control arm first so drift penalises the change" is a
*bound*, not a correction: it tells you a positive result is real and tells you nothing at all about
a null. Where the expected effect is under ~1 ms/forward — which is now most of what is left on this
ladder, since `a` needs 22 ms in pieces this size — **a null from one arm order is not a result, and
reporting it as one retires a real win.**

**And the reversed pair paid a second time.** With drift out, the by-context structure became
readable: −1.092 ± 0.146 at ctx 12,410 with 6/6 in *both* orders, against bands covering zero at
3,197 and 6,260. That contradicted 1.11's pre-registered "the saving must be flat in context", which
is a finding the single-order run could not have produced either way.


## 34. The arm switch never switched, and the tell was that the null was *too* clean (ladder 1.12, 2026-08-20)

1.12's first engine A/B spent **four checkpoint loads** and came back a flawless null:

    cattn:compress on ratio-128 emit steps   before 14.50 [14.23, 15.22]   after 14.47 [14.22, 15.16]   -0.03
    cattn:q_proj                             before 13.97                  after 13.98                  +0.01
    cattn:sparse                             before 15.38                  after 15.41                  +0.03

against a microbenchmark that said the targeted GEMM shape was **2.32× faster** and a call-site
trace that confirmed the engine reaches that shape 40 times per emit step. Exactly one of those can
be true.

**What happened.** The control arm is spelled `DSV4_F32MK_TILE=8x0` — `nn = 0` means "take the
pre-1.12 host-side chunk loop". The parser was

    if (sscanf(e,"%dx%d",&mm,&nn)==2 && mm>0 && nn>0) { g_f32mk_mm=mm; g_f32mk_nn=nn; }

so the guard rejected **the only value that selects the before-arm** and left the default in place.
Both arms ran the 4×4 tile. Every number above is correct; they are the run-to-run spread of two
loads of *the same code*.

**The tell is worth more than the bug.** A real null on this instrument does not look like ±0.03 ms
on a 3.5 % spread — that is not "no effect", it is "no difference", and the two are distinguishable
by how *small* the residual is. When an A/B returns deltas an order of magnitude below the harness's
own noise floor, the first hypothesis is that both arms are the same build, not that the change did
nothing. The same run's paired throughput band, which is the *noisy* instrument, said
`+0.512, 2 SE [−0.969, +1.993]` — perfectly consistent with a null and completely uninformative
about which kind.

**The structural fix, in two places, because a comment would not have caught this.**
1. The engine prints the tile it resolved, unconditionally, on the first `gemm_fp32` of the process:
   `[f32mk] tile 8x0 = PRE-1.12 host chunk loop (DSV4_F32MK_TILE=8x0)`.
2. `scripts/f32mkn_ab_run.sh` greps that line out of **both** server logs before it reports anything
   and **exits non-zero if the two arms resolved to the same tile**. An A/B that cannot read its own
   arm back out of the log is guessing.

**What the wasted run bought.** It is the best null control this ladder has: two full checkpoint
loads, 18 paired legs, identical binaries. The paired throughput band on identical code is
**+0.512, sd 3.142, 2 SE [−0.969, +1.993]** — so ~±1.5 ms/forward is the resolving floor of an
18-leg two-load design, which is why 1.12 went on to run the reversed replicate. And the conditional
mark reproduced to **0.03 ms across two loads**, which is what makes `tools/emit_spike.py` the right
instrument for a change whose amortised size is below the throughput noise floor.

**Generalises to:** every `NO_*`/`DSV4_*` arm switch in this repo. An env-var arm is a silent
default when it is misspelled, when the parser rejects it, when the process that reads it is not the
process you set it on, or when a caller pinned the value first. `NO_FP32MK`, `NO_ATTN_SPLIT`,
`NO_JOIN_DEFER`, `DSV4_KV_PACK` and `COMP_BF16` are all one typo away from this failure, and none of
them prints what it resolved to.

## 35. A byte-identity gate run past the length at which the engine is not byte-identical (ladder 1.12, 2026-08-20)

The same wasted run set `--max-tokens 256` on `decode_fit_probe.py`, to get two ratio-128 boundaries
per leg instead of one. `paired_band.py` then reported, for two loads of **the same binary**:

    token identity: 0/18 legs byte-identical, 18 differ, 0 unproven

**That is not a bit-exactness failure; it is ladder 1.9's race, being measured on purpose.** 1.9
established `build/decode` prefill as **byte-identical to 160 positions and racing at 192+**, and
this row of `wiki/README.md` has said so since. A 256-token completion runs the engine straight
through that boundary, so the identity column stops being a gate and starts being a coin flip — in
*both* arms, for *any* change.

**The rule.** The corpus length of an A/B is not free to choose: it is bounded above by the length
at which the engine is reproducible, and any instrument that reports identity must run below that
bound or report nothing. 1.12 dropped to `--max-tokens 128`, which still crosses exactly one
ratio-128 boundary per leg at every target it sweeps (ctx 12,282 → 12,410 contains 12,288), and the
identity column became meaningful again.

**And note which instrument survived the mistake.** The conditional dprof mark
(`tools/emit_spike.py`) is indifferent to the race — it classifies steps by an *arithmetic* property
of `(ctx, VB)` and reports a distribution, not an equality. When the reproducible window is shorter
than the phenomenon you need to sample, a distributional instrument still works where an equality
gate cannot.

**CORRECTED 2026-08-21 BY 1.10, AND THE DIAGNOSIS ABOVE IS WRONG IN BOTH HALVES.** The 0/18 was not
1.9's `build/decode` prefill race — the server does not use that code path at all (§27) — and
dropping to `--max-tokens 128` did not make the identity column meaningful. The saved runs say so
directly: **all four of 1.12's 128-token loads are 0/18 against each other**, while **all four of
1.11's 256-token loads are 18/18**. Completion length was never the variable. What actually changed
between 1.11 and 1.12 is that 1.12 altered `gemm_fp32`, therefore what shares an SM with the
compressor emit on `g_side`, therefore whether the aliased `hadamard(dst, dst, 1, d)` in that emit
fires its read/write race — 0/200 alone, **65/200 with anything else resident** (§36). With 1.10's
fix the identity column is meaningful again at 256 tokens: 15 of 15 load-pairs byte-identical.

**The rule as it should have been written.** "The identity column is a coin flip past length L" is a
statement about a mechanism, and it was adopted here without one — the number 160 was carried over
from a different binary's different code path because it was the only threshold on the page. **When
an instrument goes dark, the cheapest real test is the NULL CONTROL you already have: two loads of
the same arm.** 1.12 ran exactly that control, got 0/18, and read it as confirmation of the
inherited explanation rather than as a fact needing one of its own.

## 36. A kernel that reads a buffer it is writing — and why every instrument pointed at the wrong subsystem for two iterations (ladder 1.10, 2026-08-21)

**The defect, in four lines of CUDA that had been in the repo since the indexer was written.**

```c
__global__ void hadamard_kernel(float* y, const float* x, int rows, int D, float scale) {
    int idx = ..., r = idx / D, j = idx % D;
    const float* xr = x + (size_t)r * D;
    float acc = 0.f;
    for (int i = 0; i < D; ++i) acc += (__popc(i & j) & 1) ? -xr[i] : xr[i];   // reads the WHOLE row
    y[idx] = acc * scale;                                                      // writes ONE of it
}
```

Every thread of a row reads all `D` elements of that row and overwrites one of them. That is correct
exactly while `y != x`. Three call sites passed the same pointer twice —
`kernels/compressor.cu`'s `rotate` branch (`hadamard(out, out, groups, d)` in the prefill and
`hadamard(dst, dst, 1, d)` in the single-group emit) and `compressed_decode.cu`'s candidate emit —
so which value a reading thread got was a scheduling outcome. **Every other call site passes
distinct buffers, and one of them even carries the comment `// out!=in`.** The invariant was known;
it just was not enforceable from inside the kernel, and nothing checked it.

**IT IS A RATE, NOT A THRESHOLD, AND THAT IS WHY A LENGTH LADDER MISREAD IT.** 1.9 measured
`build/decode`'s prefill byte-identical at 128/129/132/136/144/160 positions and nondeterministic at
192 and up, and wrote the finding as *"the threshold is a prefill length between 160 and 192, and it
is SHARP"*. `tests/gate_hadamard_alias` drives the kernel directly, 200 repeats per row count, and
the boundary is not a step:

| rows (= `s/4` in prefill) | 1 | 2 | …40 | 41 | 42 | 44 | 48 | 56 | 64 | 96 | 128 | 256 | 768 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| blocks (256 thr, 2 rows each) | 1 | 1 | ≤20 | 21 | 21 | 22 | 24 | 28 | 32 | 48 | 64 | 128 | 384 |
| runs differing, /200 | 0 | 0 | **0** | **0** | 17 | 28 | 41 | 74 | 116 | 184 | **200** | 200 | 200 |
| distinct results | 1 | 1 | 1 | 1 | 15 | 28 | 38 | 64 | 111 | 179 | **200** | 200 | 200 |

Zero at every row count whose grid fits the 20 SMs once; first firing at **rows 42 — the first count
at which 21 *full* blocks exist** — and rising monotonically to every run distinct. **A step function
and a rising rate are indistinguishable at one sample per length, which is exactly what a length
ladder is.** 1.9's ladder was clean below the boundary because the rate is genuinely 0 there, and
"41 is clean, 42 is not" is inside its 160→192 gap; but its word *sharp* was an artefact of sampling
once. Nothing in this repo should carry a threshold from a one-sample-per-point sweep without the
repeat count next to it.

**THE SAME ARITHMETIC EXPLAINS WHY THE ENGINE LOOKED 100 % BROKEN.** At prefill 192, `rows = 48` and
the per-launch rate is 41/200 = 20.5 % — but a prefill runs **21 ratio-4 layers**, each issuing one
aliased `hadamard`, so P(some layer differs) = 1 − 0.795²¹ = **99.2 %**. That is why 1.9 saw 56 of 56
pairs diverge from a kernel that is deterministic four runs out of five. **A per-launch rate and a
per-forward rate differ by the layer count, and the number you can see is never the one you want.**

It also explains the shape 1.9 could not: `build/decode`'s prefill issues ONE call as wide as the
prompt, so it crosses the isolated-kernel threshold on grid size alone and needs no help from a
neighbour; the server never gets near that threshold and races anyway, through concurrency. **Two
different routes to the same defect, which is why the two binaries disagreed about it.**

**AND THE OBVIOUS COROLLARY — "so the decode path is safe" — IS FALSE, WHICH IS THE MOST
IMPORTANT THING ON THIS PAGE.** The two aliased sites decode reaches emit ONE group: `rows = 1`,
one block, four warps, **0 of 200 differ** in the sweep above. That reads as an exoneration and it
is an artefact of the harness: the sweep runs the kernel ALONE, and four warps of a lone block on an
idle GPU do issue in lockstep. `compressed_verify_step_indexer` forks the compressor emits onto
`g_side` and runs them CONCURRENTLY with main-stream work, so in the engine those four warps share
an SM with whatever else is resident. `gate_hadamard_alias --concurrent` puts a filler kernel on 80
blocks across the 20 SMs and re-runs exactly the same `rows = 1` call:

| arm | differ | distinct results |
|---|---|---|
| flat (pre-1.10) alone | **0/200** | 1 |
| **flat, with anything else resident** | **65/200** | **28** |
| staged, with anything else resident | **0/200** | 1 |

**A race that needs a co-resident block cannot be seen by a benchmark that runs the kernel alone,
and every microbenchmark in this repo runs kernels alone.** The isolated row sweep answers "how many
blocks does this kernel need before it fights itself"; the engine question is "what else is on the
SM", and the answer to that is set by other kernels entirely.

**THE ENGINE PAID FOR THAT DISTINCTION AND NOBODY NOTICED FOR AN ITERATION.** Twelve `dsv4-server`
loads are on disk with the same corpus and the same probe. Grouped by which `hadamard` they ran:

| | load-pairs fully byte-identical |
|---|---|
| **staged** — 1.10's two loads + all four of 1.11's | **15 of 15** |
| **flat** — 1.10's two control loads + all four of 1.12's | **0 of 15** |

1.11 ran the flat kernel and was perfectly reproducible; 1.12 ran the flat kernel and no two of its
four loads agree; **1.12 changed `gemm_fp32` and nothing in `indexer.cu`.** That is the co-residency
mechanism above, seen from the engine: 1.12 changed what shares the SM with the emit, and a race
that had been latent since the indexer was written started firing. **1.10's staged arm does not
merely reproduce itself — it reproduces 1.11's four loads leg for leg**, so the pre-1.12 values were
the correct ones and 1.12's arithmetic really was bit-exact, exactly as its entry claimed.

`tau` shows the same thing without any hashing: the staged arm's two loads both give **1.6445**, and
the flat arm's give **1.6508 and 1.6413**.

**WHAT THIS COSTS.** Every server A/B through 1.11 reported byte-identical legs and is unaffected —
1.5 (16/16), 1.7 (16/16), 1b.2 (16/16), 1.11 (44/44). **1.12's is the one that was measured against
a control that does not reproduce itself**, and it is the item whose paired band moved tracked
Term A. It needs re-running now that the engine is deterministic; that is ladder item **1.15**, not
a retraction, because a band measured against a moving target is unproven and not wrong.

**THE DISCRIMINATOR THAT NAMED IT WAS A HISTOGRAM, NOT AN ABLATION.** `tools/lhash_pairs.py` was
written for 1.10 because `lhash_compare --within` compared point 0 against every later point only —
and point 0 is not interchangeable with the rest (the R logs show `scratch0=…d2000000` at point 0
and `…d2200000` at every later one, i.e. the first point of a process allocates at a different
address). All-pairs removes that, and it prints **the compression ratio of the first differing
layer**. Layers alternate ratio 4 / ratio 128 from layer 2, and `compressed_attn_forward` runs
`indexer_forward` and the overlapping compressor **only at ratio 4**. Across 1.9's logs and 1.10's
six-arm campaign, **335 of 336 pairs diverged and every single first-differing layer was ratio 4 —
not one was ratio 128 or ratio 0.** That one line of read-out excluded, before a single ablation
ran, every kernel a ratio-128 layer also executes: the main compressor, `sparse_attn`, the whole
MoE, `mla_forward`.

**AND THE SIX-ARM ABLATION CAMPAIGN RETURNED ALL-NEGATIVE, WHICH IS THE RESULT IT WAS FOR.**
`scripts/lhash_ablate.sh` re-ran 1.9's R protocol under one env flag per candidate — each flag
swapping one kernel for a different implementation of the same maths, on the shipped binary:

| arm | what it swaps | verdict |
|---|---|---|
| `base` | nothing (control on today's binary) | 56/56 pairs diverge |
| `NO_IXGEMM=1` | `index_score` GEMM → tiled | 55/56 diverge |
| `NO_IXGEMM=1 NO_IXTILE=1` | → warp kernel | 56/56 diverge |
| `DSV4_TOPK_RADIX=0` | radix select → warp selection sort | 56/56 diverge |
| `NO_FP32MK=1` | 1.12's `gemm_fp32` warp tile → legacy | 55/56 diverge |
| `DSV4_SPARSE_HPB=1 DSV4_SPARSE_SMEM=0` | 1.7's `sparse_attn` staging → pre-1.7 | 56/56 diverge |

Six arms, six divergences, and that is what a bisection *is*: the arms that came back dirty are the
kernels that are now excluded by measurement rather than by argument. **The one that mattered was
not on the list**, and it was not on the list because the list was built from "which kernel is
complicated" rather than from "which buffer is written by the kernel that reads it".

**THE GENERALISABLE RULE.** Two pointers of the same type in one kernel signature, one `const`, one
not, is an aliasing contract with no enforcement anywhere in this repo. `__restrict__` makes it
worse: it tells the compiler the contract holds while the call site is breaking it. Grep for it —
`grep -n 'kernelname(\([a-z_]*\), *\1'` finds the obvious cases — and where a kernel genuinely must
support both, stage through shared memory so the read set is closed before the write set opens.
That is the shipped fix, and it is bit-identical wherever the buffers were already distinct because
it changes memory placement and not the order of the sum.

Evidence: `evidence/decode_loop/gate_hadamard_alias_1p10.log`,
`evidence/decode_loop/lhash_ablate_verdict.txt`, `evidence/decode_loop/lhash_verify_verdict.txt`.
Code: `kernels/indexer.cu` (`hadamard_stage_kernel`, `hadamard_set_stage`),
`tests/gate_hadamard_alias.cu`, `tools/lhash_pairs.py`, `scripts/lhash_ablate.sh`,
`scripts/lhash_verify.sh`, `scripts/hadamard_ab_run.sh`.
