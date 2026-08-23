# DECODE_ENDGAME.md — the ladder from 28.38 to 35–42 tok/s

The target is **35–42 tok/s** speculative decode for `DeepSeek-V4-Flash-0731-REAP` on Jetson AGX
Thor, against **22.66** at the start of this project and **28.38** today. Every number is measured
unless marked **est.** The full ranking with mechanisms is [`ROADMAP.md`](ROADMAP.md).

## The ladder

| # | rung | worth | cost | state |
|---|---|---|---|---|
| — | **banked**: block width 5 + fine-tuned draft head | 22.66 → **28.38**, **+25.3 %** | done | ✅ |
| 1 | **corpus** — agentic-weighted, 2× size, 2× generation depth | +4–9 % est. | wall clock | **running** |
| 2 | **remaining arms** — anchor shape (`β·r^p`) | +0–2 % | wall clock | capped at 4 |
| 3 | **C(k) sweep** — widths 4–12 × 9 prompts | *prices rung 4* | wall clock | **queued** |
| 4 | **adaptive block width** | **+20–25 % est.** | CUDA | gated on rung 3 |
| 5 | **AR kernel headroom** | +5–10 % est. | CUDA, hard | deferred |
| — | **prefill to the roofline** | **6.6× TTFT**, no tok/s | CUDA | practicality, not throughput |

Rungs 1–3 need no supervision and are what the box does unattended. Rungs 4–5 are CUDA and need a
session. That split is deliberate: the phases that consume the most wall clock are exactly the ones
that need no one watching.

## Why the banked 25.3 % does not repeat

Ten arms at block 5 put their top five **within 1.3 %** of each other, and the promotion bar is
3.5 %. The knobs are spent — ce/tv swept three ways, β bracketed on **both** sides, HASS retired,
the confidence loss term retired. The exchange rate is **~13.8 tok/s per unit `tau`**, so an
excellent further arm is worth about **+1.5 tok/s** and most are worth nothing.

That is why rung 1 is data and rung 4 is architecture. More hyperparameter search is rung 2, and it
is capped at four arms for exactly this reason.

## Rung 3 decides rung 4, and it is the highest-leverage hour on the machine

**`tau`'s ceiling IS the draft width.** At a fixed width of 5 a *literally perfect* head scores 5.00
— worth only 1.30× over today — and perfect is unreachable, because acceptance is bounded by the
**entropy of the target distribution**, not by our ignorance. `code_gen` at `tau` 2.59 is not a
training failure; constructive generation is genuinely less predictable than reconstruction. Rungs
1 and 2 chase a bounded quantity, which is why they are small.

Varying the width removes the bound — but only if the best width actually differs by task shape,
and **that has never been tested.** It needs no kernel: the engine already accepts widths 4–12, and
tok/s at width k already equals `E[A(T_k)] / C(k)`, so the per-prompt argmax over k *is* the choice
an adaptive engine would make.

| outcome | consequence |
|---|---|
| **k\* varies by prompt** | rung 4 is real; the spread between each prompt's own k\* and the served 5 becomes a **measured** upper bound, replacing the estimate |
| **k\* = 5 everywhere** | the +20–25 % is **refuted**; rung 4 dies for one overnight run instead of a CUDA rewrite |

Ladder 2.1 found width 6 worse than 5 **at fixed width**, because verify is expensive. Rung 3 asks
the question 2.1 could not: is 5 best for *every* prompt, or only on average? Those have the same
answer only if acceptance is uniform — and it spans **2.7×** by category.

The expensive half of rung 4 is already paid for: the confidence head predicts acceptance at
**AUC 0.8795 overall and 0.84–0.87 within every draft position k**, and it needs no training
because it **ships in the base checkpoint**. What remains is the verify-time engine change.

## The arithmetic

| | today | endgame est. |
|---|---:|---:|
| base AR | 14.61 | **~16** (rung 5) |
| effective `tau` | 3.84 / 5, fixed | **5.0–5.5** (rung 4) |
| speculation multiplier | 1.94× | **~2.4–2.7×** |
| **speculative decode** | **28.38 tok/s** | **35–42 tok/s** |
| TTFT at 12 k | ~3.3 min | **< 30 s** (prefill) |

## What bounds it, and none of these are effort

1. **`B_tok` = 12.26 GB/token against ~240 GB/s achievable.** Base AR cannot exceed ~19.6 tok/s
   even if every byte moved at peak and nothing else took any time — and **22.3 ms of a 71.4 ms
   step is not bytes at all**. The long-quoted "19.0 tok/s roofline" is a *normalisation constant,
   not a target*; treating it as reachable cost this project real time.
2. **Acceptance is entropy-bounded.** Adaptive width raises the ceiling; it does not make
   constructive tokens predictable.
3. **The checkpoint is fixed** — no re-quantisation, no REAP work. `B_tok` is what it is.

**Decode is roughly two-thirds of the way to its practical limit.** The remaining third is one
unstarted engine change plus hard kernel work.

## Prefill is not on this ladder and may matter more than all of it

Prefill is **62.4 tok/s** — only 2.4× the decode rate, when a batched prefill should be one to two
orders of magnitude faster than token-at-a-time decode. **TTFT at 12 k is ~3.3 minutes.** It
contributes nothing to the tok/s column and is still probably the most valuable single item here,
because a three-minute time-to-first-token makes throughput academic for an agentic harness.

Two specific things have never been tried: `DSV4_DPROF` has **never been run on a PS=1023 prefill**
on this engine, and the m16 B-repack was rejected for decode because it needs 5.05 rows/expert
where decode presents 1.67 — **at prefill M≈1022 each of 160 experts sees ~50 rows, ten times the
threshold, and it has never been tested at prefill shapes.**

---

## Log: 2026-08-23 — C(k) swapped ahead of the corpus

**What changed.** The C(k) sweep (rung 3) now runs *before* the agentic corpus (rung 1), not after.

**Why.** C(k) is a ~30 minute measurement that *prices* rung 4, adaptive block width, at an estimated
+20–25 %. The corpus is a 31 hour generation worth an estimated +4–9 %. Running the cheap decisive
measurement behind the expensive incremental one meant the single most valuable question this
project still has — *is 5 the best width for every prompt, or only on average?* — would not be
answered until Monday evening, and every CUDA decision downstream of it was blocked on that.

Ordering is now **p25b → ck → corpus → arms → evals**.

**Why it cost nothing.** `decode` opens `DSV4_GENOUT` with `"a"` and `fclose()`s after *every*
sequence (`src/decode.cu:1457`), so `gen.txt` only ever holds whole lines. The interruption landed
between sequences: all **384** generated responses survived, each with its full 1024 tokens, zero
alignment mismatches. The earlier claim that pass 1 had a chunk seam every 491 prompts was **wrong**
— `S5_CHUNK` chunks *pass 2* (capture + train), not generation. It did not matter, because the
append-per-sequence behaviour makes any interruption point a safe one.

**How the splice is verified.** `scripts/resume_corpus_gen.sh` generates prompts `L+1..N` into the
same `gen.txt` and then checks *both* hypotheses:

| check | requirement |
|---|---|
| aligned at offset 0 | **0** mismatches |
| aligned at offset 1 | **large** — if this is 0 the splice is shifted |
| short sequences | 0 with fewer than `NGEN` generated tokens |
| trailing newline | present |

Checking offset 1 is the point. A splice silently shifted by one produces a corpus in which every
response answers the previous prompt — it trains without error and measures as noise. Before the
swap this was tested against the live file: **0 mismatches at offset 0, 383/383 at offset 1.** The
off-by-one hypothesis is refuted, not assumed away.

The resume also asserts its checkpoint equals `s5_session_auto.sh`'s, because pass 1 generates with
the **base** checkpoint (the S5 "target"), and resuming under the live head would make the two
halves of the corpus inhomogeneous — an arm whose result would mean nothing.

**Ordering is enforced from both sides**, as before: `dsv4-resume` waits on `dsv4-ck`, and
`dsv4-resume` was added to the autopilot's `other_chain()` guard so the arms and the eval battery
wait on the corpus finishing. `systemctl --user is-active --quiet A B` returns 0 if *any* unit is
active — verified on this box, not assumed from the man page.
