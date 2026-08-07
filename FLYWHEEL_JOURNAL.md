# FLYWHEEL_JOURNAL.md — one entry per autonomous iteration

Written by `scripts/flywheel.sh`, read by the human observer. Each entry: what was done, the number,
what it means, what comes next. `LOOP_LOG.md` holds the full reasoning; this is the watch log.

**Observer's checklist** — the failure modes this project has actually hit, in the order they cost
the most. Check each entry against them:

1. **Is there a number, and was it run this iteration?** Finding 33: a config was written up that had
   never been executed, and the number quoted belonged to the previous build.
2. **Does the number come from a full-model run, or only from `gemm_bench`?** Finding 47: the bench
   overstates end-to-end value by 2-4x. A bench-only adoption is not an adoption.
3. **Was the gate capable of failing?** Finding 41: a gate that allocates its own inputs with
   `cudaMalloc` always gets 256-byte alignment and cannot catch the misalignment that crashed prefill.
4. **Did one iteration change one thing?** Two changes in one measurement means neither was measured.
5. **Was a retired lever re-proposed?** The retired list in `LOOP_LOG.md` is binding.
6. **Predicted vs measured within 2x?** If not, the ranking model is wrong and must be fixed before
   the next lever — not the lever.
7. **Did it stop when it should have?** A gate failure or `GATE FAIL` must halt, not get built upon.

---

## Cycle 0 — seeded by hand, 2026-08-07

State at handover to the loop:

| | |
|---|---|
| speculative decode | **16.86 tok/s** (NGEN0=60, 18 verifies, mean 3.39 tokens/verify) |
| base AR | 92.5 ms/tok; **79.3 ms = 12.61 tok/s** with the full-step CUDA graph |
| M=5 verify | 167.5 ms, `c_v` 1.82 against a byte-model floor of 1.84 |
| session | 10.04 -> 16.86 tok/s speculative (**1.68x**), 9.98 -> 12.61 base (**1.26x**) |

Phase A, with `consecutive_sub_half_pct = 3` — i.e. the stopping rule is already armed. The queue
carries four entries but the honest expectation is that A goes dry quickly and the loop advances to
B, and then to D once `phaseB_top_history` repeats.

**What the observer should watch for first:** the ranking model currently says the top residual is
MoE latency, and ncu (Finding 47) says that phase is latency-bound at 25% memory throughput with 84%
of stall cycles on `long_scoreboard`. Two levers there have already measured under 1%. If cycle 1 or
2 proposes a third variation on the same phase, the loop is spinning and should be pushed to D.

## Cycle 1 — 2026-08-07 — HALTED, no measurement. Read this one before anything else.

**The loop cannot execute.** `scripts/flywheel.sh:131` launches the executor with
`--permission-mode acceptEdits`, which auto-approves file edits and **not Bash**. Denied this cycle:
`g++` (even `--version`), `./build/inspect_weights`, `git add -A`. Read-only inspection worked.
So: no build, no unit gate, no `run_model.sh`, **and no commit** — cycle 1's edits are sitting
uncommitted in the working tree. I produced no numbers, which is the only safe output available;
an executor that can write findings but cannot run them is Finding 33 with the safety off.

**What I did settle without running anything** (existence questions, answerable from source):
- **A6 retired.** No CUTLASS kernel is ever launched by this engine — no TU includes
  `cutlass_moe.h`, the only callers of `cutlass_nvfp4_gemm` are that file's own self-tests, and
  `build/cutlass_moe.o` is linked but dead. The upstream `reg_reconfig.h` patch would change an
  object that never runs. Expected end-to-end value: zero.
- **A1 pulled out of Phase A.** It is not lossless, `research/MOE_DECODE.md:98` already recorded
  MXFP4 block-32 as blocking sub-block skips (only `w3` is skippable), and §2 rule 1 says byte
  reduction does not pay on the MoE phase Finding 47 measured latency-bound. Re-scoped to §4 with a
  user decision attached — this is the third time the loop has aimed byte reduction at that phase.
- **Prepared R1** (the re-fit Finding 49 asked for) with `tools/encode_prompt.cpp`: 45 lines that
  turn text into argv ids from the checkpoint's own tokenizer, and refuse to print unless they
  reproduce `671,6102,294,8760,344`. **Never compiled, gate never fired.** I did not write the
  companion `src/decode.cu` multi-prompt change: an uncompilable edit to the measurement harness is
  worse than no edit.

**Next iteration, in order:** (1) fix permissions and commit this tree; (2) build
`tools/encode_prompt.cpp` and run its gate — a FAIL is a real result, it would mean the in-tree
gemma-4 tokenizer does not match this checkpoint and R1 needs another id source; (3) only then the
~20-line `DSV4_BLKSWEEP` prompt-index change and one multi-prompt adaptK A/B.
**Do not re-enter Phase D** on the armed counters: D ran on 2026-08-07 and this queue is still
consuming its output.

## Cycle 2 — 2026-08-07 — R1 attempted, BLOCKED. Two instrument defects, both measured. 0.0% end-to-end.

Bash works this cycle, so cycle 1's halt is cleared. I took queue[0] (R1, re-fit adaptK across
prompts), got as far as a 17-point full-model run, and am reporting **no adaptK number**, because
the instrument that produced it is unsound. That is the result.

- **Id source settled.** Built and fired cycle 1's `tools/encode_prompt.cpp` gate: **FAIL** —
  `The capital of France is` → `671 464 388 367 79666 ...`, not `671 6102 294 8760 344`.
  `include/tokenizer.h` is a gemma-4 sentencepiece encoder; this checkpoint is ByteLevel BPE. The
  gate refused to print, exactly as designed. Deleted it. `tools/encode_prompt.py` (already at HEAD
  from a quarantined cycle) passes the same gate plus 6/6 round-trip.
- **Run:** `~/cycle2.log`, base-AR gate **PASS** (11111), base AR 10.27 tok/s (adaptk3: 10.31).
  Canonical prompt at the shipped adaptK=1.5, three replicates: **16.77 / 16.94 / 16.95 tok/s** —
  the 16.86 baseline is confirmed, not moved.
- **D1 — `adaptK=0` in a sweep entry is not fixed width.** It falls through to the 1.5 default
  unless `NO_ADAPTK=1` is in the environment, and the table printed the *requested* value. All four
  of my control points ran at 1.5. Finding 49 survives: adaptk3.log set `NO_ADAPTK=1`.
- **D2 — the multi-prompt harness leaks state.** My designed leakage gate (canonical prompt first
  AND last, twelve points apart) **passed byte-identically** — and was not enough. On prompt 2, two
  points at identical effective settings emitted *different token sequences*, and points 9/10/11
  agree with each other while disagreeing with 8: deterministic contamination, not noise. Replicate
  spread at one setting: **1.1% canonical, 6.1% / 8.5% / 15.6% on the other three prompts.**

**Conclusion: every adaptK difference this run could have reported is smaller than the replicate
spread of the setting itself.** No prior number is impugned — the one prompt whose replicates are
stable is the one this project has always measured on — but R1's premise is false as implemented.

**Next iteration takes I1: make one process run the same point twice and get the same answer.** The
gate is written and already failing; `FLYWHEEL_STATE.json` carries three places to look, the best
being that the divergence is visible in the *draft* before a point's first verify. Then D1, then R1.
**Do not advance to B or D** — the counter says the stopping rule fired, but ranking residuals off
an instrument with a 15% error bar is how the loop buys a wrong answer twice.

Two process notes for the observer: I ran `git rm --cached` on the deleted tool before remembering
the harness owns git (harmless — it stages the same deletion `git add -A` would), and I overwrote
`tools/encode_prompt.py` before checking that a quarantined cycle had already committed a better
version. Restored it verbatim from HEAD and re-ran its gates.

## Cycle 3 — 2026-08-07 — I1 root-caused and fixed. 2 of 3 non-canonical prompts now reproduce. 0.0% end-to-end. HALT.

I took queue[0] (I1, make one process run the same sweep point twice and get the same answer) and
found the cause by reading code, before spending the run — which is the right order and it is the
first time this loop has managed it.

- **Finding 52.** `run_layer` (src/decode.cu:265) is a `[&]` lambda; its prefill branch passed `PS`.
  Lambda name lookup is lexical, so that bound to line 96's `argv[2].size()-1`. The sweep loop then
  declared its own `PS` for the point's prompt — a shadow the callee cannot see. **Every sweep point
  prefilled the KV caches to the argv prompt's length** while everything else used the point's.
  It predicts cycle 2's signature with no free parameters, including why prompt 0 was the one that
  always reproduced: it *is* the argv prompt.
- **Fix:** explicit `npre` parameter (8 call sites) + no shadowing at all in the sweep loop. A
  parameter cannot be shadowed out from under the callee.
- **New in-run gate, every point, before any measurement:** `KV[L].T == PSp/ratio` for all 21
  compressed layers, because prefill emits exactly `floor(PSp/ratio)` compressed rows — so `T` reads
  back the length the prefill *actually ran at*. Under the old code it would have read 1 instead of
  4. **PASS at all 11 points.** This is the assertion whose absence cost cycle 2 its whole run.
- **Result (`~/cycle3.log`, base-AR gate PASS 11111, base AR 10.33 tok/s):** replicate pairs
  separated by a different-length prompt, hashed over the emitted tokens *and* the full per-verify
  decision trace. Prompt 0 (s=6) **identical**, 16.83 / 16.90 tok/s. Prompt 1 (s=11) **identical**,
  15.45 / 15.45. Prompt 3 (s=15) **identical**, 11.10 / 11.11. Cycle 2 had **0 of 3**; this is 2 of 3
  plus canonical, and they are byte-identical, not merely close.
- **What still fails:** prompt 2 (s=18) — three points at one setting gave three different sequences
  and 11.05 / 14.62 / 16.36 tok/s — and the run then **died in that same prompt's 12th prefill**,
  `cuda kernels/indexer.cu:91 an illegal memory access`. Both symptoms are on the only prompt whose
  length equals SMAX. `mh_pre = (SMAX-1)*3*d*4` is the one zero-slack buffer in that path, but that
  is a lead, not a cause, and I did not spend a second run guessing.
- **Bonus worth keeping:** prompt 1 at adaptK 1.00 / 1.50 / 2.50 emitted the **identical 61-token
  sequence** over 23 / 23 / 26 verifies at 14.76 / **15.45** / 14.51 tok/s. Finding 49 argued
  adaptive verify width is lossless; this measures it. Not adopted — one prompt, and D1 means there
  is still no fixed-width control in any run.

**HALT set (invariant 7): a run that faults is a failed run.** Nothing on the shipped path is
implicated — the canonical prompt reproduced twice byte-identically at the baseline — so the halt is
about not stacking a second fix on an un-root-caused fault.

**Next iteration.** The queue head is now an externally-inserted Phase-D research pass, which needs
no build and no model run and is compatible with the halt. After it, **I2**, whose first action is a
falsification test that costs no extra run: append a prompt longer than 18 ids so `SMAX > 18` and
re-run the same 9 gate points. If prompt 2 then reproduces, the defect is "the prompt of length
SMAX" and the zero-slack buffer is the place to look; if it does not, SMAX is a coincidence. A
3-point sweep under `compute-sanitizer` would locate the fault directly and does not need NGEN0=60.

**Process note for the observer.** A commit landed out-of-band at 03:48 while this cycle was still
working: `469d9c2 "flywheel: research is not blocked by a broken instrument"`. Its message describes
only the Phase-D policy change, but its diff also swept my in-flight working tree — `src/decode.cu`,
`LOOP_LOG.md` and `FLYWHEEL_STATE.json`. **Nothing was lost, but cycle 3's Finding 52 fix and log are
committed under a message that does not mention them.** `.flywheel_commit_msg` holds the correct
cycle-3 message; the harness's commit at exit will apply it to `FLYWHEEL_JOURNAL.md` alone. If the
harness commits mid-cycle by design, it should either wait for the executor to exit or use
`.flywheel_commit_msg`, or history will keep attributing work to the wrong commit.

## Cycle 4 — 2026-08-07 — I2: candidate -> implemented. Finding 53. No model run.

- **Stage discipline.** I2 was at `candidate`, so this cycle builds it and gates it and does not
  measure it. **No full-model run was launched and the baseline is untouched at 16.86 spec / 12.61
  base tok/s** — those are still cycle 3's numbers, not mine.
- **Cycle 3's lead was wrong, and it cost nothing to say so.** `mh_pre = (SMAX-1)*3*d*4` is an exact
  fit: `k_tap_pool` writes exactly `PSp*3*d` floats and max `PSp` is `SMAX-1`. Discharged by reading
  `kernels/dspark_real.cu:44`, no run spent.
- **The gate this suite was missing: one that varies `s`.** Every existing gate runs at one fixed
  prompt length, which is exactly how a length-dependent defect survives. `tests/gate_prefill_len.cu`
  asserts prefix-invariance — prefill is causal, so for lengths s < S over the same x the first s rows
  must agree bit-exactly — over outputs and all three KV caches, sliding + ratio-4 + ratio-128,
  s = 1..20, weight alignment +0 and +4.
- **The invariant holds everywhere.** That is the more valuable half: the prefill is *not*
  length-dependent, `PSp=17` is not special, and cycle 3's whole framing was wrong.
- **What it caught instead.** Under compute-sanitizer: `Invalid __shared__ read of size 12 bytes at
  k_topk_offset ... indexer.cu:60` → `cuda kernels/indexer.cu:91 unspecified launch failure`. The
  kernel fills `extern __shared__ float sh[]` and scans it, and was launched with exactly `T*4` bytes;
  nvcc widens the scan into a 12-byte vectorised load, so **any T < 3 reads out of bounds**. Swept it:
  T=1 and T=2 fault (5, 6, 9 errors), T>=3 clean. Three sibling top-k kernels had the same undersizing.
- **Second defect, found chasing the attribution.** A gridDim-0 launch reports only through the
  last-error slot — `cudaDeviceSynchronize` returns success, measured here — and this engine never
  called `cudaGetLastError` anywhere, so **a kernel that never ran was indistinguishable from one that
  did**. It was firing on every prefill: `groups = s/128` is 0 for all 20 ratio-128 layers at every
  prompt this project has run. Fixed, and `dsync()` now drains and names the slot.
- **Verified:** gate_prefill_len passes plain and under memcheck (0 errors, 2m45s); gate_indexer_decode
  clean at s=4,5,8,12,16,17,18 where 4/5/8 used to fault; the whole existing suite passes; the four
  compressed/indexer equivalence gates are still bit-exact at rms 0.00e+00; `build_decode.sh` clean.
- **What I am NOT claiming.** The prompt that crashed cycle 3 had T=4, which does not over-read. Two of
  that run's four prompts did. This is a real fix on the exact line the fault named; its sufficiency is
  unproven.
- **Next iteration: I2 is at `implemented`, so it is ONE full-model run.** `NGEN0=60`, cycle 3's gate
  points, plus a prompt longer than 18 ids appended so `SMAX > 18` — which answers "does prompt 2
  reproduce", "does the fault recur" and "was SMAX ever the right frame" in a single run. Watch stderr
  for `[launch] file:line pending CUDA error`; that is new, and it is the engine's first voice for a
  kernel that failed to launch.
- **Counter caveat for the observer.** `consecutive_sub_half_pct` is now 6, but cycles 2, 3 and 4 were
  instrument and correctness repair, not levers. The Phase-A stopping rule should not be read as fired
  on that number alone.
