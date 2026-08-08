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
