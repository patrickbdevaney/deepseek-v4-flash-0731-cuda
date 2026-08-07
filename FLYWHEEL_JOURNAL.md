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
