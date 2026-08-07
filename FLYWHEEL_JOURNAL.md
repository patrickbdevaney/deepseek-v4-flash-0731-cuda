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
