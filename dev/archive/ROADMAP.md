# ROADMAP.md — what is left in decode, ranked, with an endgame number

Written 2026-08-23, after the block-5 draft-head programme closed P2.5, P2.6 and 2.3's training
half. It states what remains, what each is worth, and what this box and this model can ultimately
decode at. Every number below is measured unless marked **est.**

## Where we are

| | measured |
|---|---|
| speculative decode, 8-prompt suite mean | **28.38 tok/s** |
| base AR decode | **14.61 tok/s** — speculation is **1.94×** |
| acceptance `tau` | **3.84 / 5**, 77 % of the width ceiling |
| prefill (PS=1022) | **62.4 tok/s**, ~3.3 min TTFT at 12 k |
| vs the stock head this project started from | **22.66 → 28.38, +25.3 %** |

## The ranking

| lever | worth | cost | state |
|---|---|---|---|
| **1. adaptive block width** (2.3 serving side) | **+20–25 % est.** | CUDA | **not started** — the measurement that justifies it is done |
| **2. prefill to the roofline** | **6.6× TTFT**, not decode | CUDA | not started |
| **3. AR kernel headroom** | +5–10 % est. | CUDA, hard | deferred items 1.7/1.8/1.11/1.12/1b.2 |
| 4. corpus (running) | +4–9 % est. | wall clock | in flight |
| 5. remaining hyperparameter arms | +0–2 % | wall clock | capped at 4 |

---

## 1. Adaptive block width — the only lever that raises the ceiling instead of approaching it

**`tau`'s ceiling IS the draft width.** At a fixed width of 5, a *literally perfect* draft head
scores 5.00 and nothing can score more. We are at 3.84, so even perfection is worth only 1.30× —
and perfection is unreachable, because acceptance is bounded by the **entropy of the target
distribution**, not by our ignorance. `code_gen` at 2.59 is not a training failure; constructive
generation is genuinely less predictable than reconstruction. That is why levers 4 and 5 are
small: they chase a bounded quantity.

Varying the width removes the bound. Acceptance spans **2.7×** by task shape — `long_context` runs
r ≈ 0.9, `code_gen` r ≈ 0.5 — so a fixed width simultaneously **overspends** on tokens that will
accept 2 and **stops short** on tokens that would have accepted 9. The engine should choose per
position:

    k* = argmax_k  E[A(T_k)] / C(k)

**The hard half is already done and it cost two GPU sessions.** The confidence head predicts
acceptance at **AUC 0.8795 overall, 0.84–0.87 within every draft position k** — and it needs no
training, because `mtp.2.confidence_head.proj.weight` **ships in the base checkpoint** and is
byte-identical in every head we serve. The per-k breakdown is what makes it usable: a head that had
only learned the k-prior would read ~0.5 within each k while looking strong pooled.

What remains is the verify-time engine change. Note ladder 2.1's result cuts the right way here —
it found width 6 *worse* than 5 **because verify is expensive**, which is an argument for spending
width selectively, not uniformly.

**Est. +20–25 %.** Routing high-r positions to width 8–10 and low-r to 2 plausibly puts effective
`tau` at 5.0–5.5 against today's 3.84. Marked est. because the verify-cost curve C(k) above width 6
has never been measured at this engine revision — **measuring C(k) is the first task, before any
kernel is written.**

## 2. Prefill to the roofline — the largest user-visible number in the system

Prefill is **62.4 tok/s**, only 2.4× the decode rate, when a batched prefill should be one to two
orders of magnitude faster than a token-at-a-time decode. TTFT at 12 k is **~3.3 minutes**. The
diagnosis is that weight reads are not amortised across the batch — prefill is running
decode-shaped kernels.

Target: **TTFT < 30 s at 12 k uncached ⇒ ≥ 410 tok/s, a 6.6×.**

Two specific unexplored items: `DSV4_DPROF` has **never been run on a PS=1023 prefill** on the
current engine, and the m16 B-repack was rejected for decode because it needs 5.05 rows/expert
where decode presents 1.67 — **at prefill M≈1022 over 160 experts each expert sees ~50 rows, ten
times the threshold, and it has never been tested at prefill shapes.**

This does not raise tok/s. It decides whether the server is usable by an agentic harness at all,
which is the actual goal.

## 3. The remaining AR kernel headroom

Base AR at 14.61 tok/s is bandwidth-bound and close to its realistic floor. `B_tok` is
**12.26 GB/token**; the byte-moving marks average **191 GB/s against a 240 GB/s achievable peak**,
and **22.3 ms of a 71.4 ms step is not bytes at all**. The long-quoted "19.0 tok/s AR roofline" is
a **normalisation constant, not a target** — assuming it was reachable cost this project real time.

Deferred items and their pre-registered ceilings:

| item | ceiling | note |
|---|---|---|
| 1.7 / 1.8 | context-term | worth +11.7 % at ctx 12.4 k, only +1.7 % at ctx 1.7 k |
| 1b.2 | KV FP8, 2048 B → 711 B | a **capacity** lever, not a throughput one |
| 1.11 | 0.9 % | deferred ATTN_SPLIT join |
| 1.12 | ~0.4 % throughput | but a 40 ms latency spike on 1 token in 75 |
| 3.1 clocks | **closed negative** | MAXN measured +1.68 ± 5.83 %, band covers zero |

**Est. +5–10 %**, concentrated at long context — which is where agentic sessions live, so it is
worth more than the suite-mean figure suggests.

---

## The endgame number

Composing the three, with base AR and the speculation multiplier treated separately:

| | today | endgame est. |
|---|---:|---:|
| base AR | 14.61 | **~16** (lever 3, +5–10 %) |
| effective `tau` | 3.84 / 5 fixed | **5.0–5.5** (lever 1, width varies) |
| speculation multiplier | 1.94× | **~2.4–2.7×** |
| **speculative decode** | **28.38 tok/s** | **31–35 tok/s** |
| TTFT at 12 k | ~3.3 min | **< 30 s** (lever 2) |

**31–35 tok/s is the realistic endgame for this model on this box**, against 22.66 at the start —
a **1.5–1.9× total programme gain**, of which +25.3 % is already banked.

Three things bound it and none of them are effort:

1. **`B_tok` = 12.26 GB/token against ~240 GB/s achievable.** Base AR cannot exceed ~19.6 tok/s
   even if every byte moved at peak and nothing else took any time, and ~22 ms/step is not bytes.
2. **Acceptance is bounded by the target's entropy.** Adaptive width raises the *ceiling*; it does
   not make constructive tokens predictable.
3. **The checkpoint is fixed** — no re-quantisation, no REAP work. `B_tok` is what it is.

The honest read: **decode is roughly two-thirds of the way to its practical limit, and the
remaining third is one unstarted engine change (lever 1) plus hard kernel work (lever 3).** Lever 2
does not appear in the tok/s column at all and is still probably the most valuable of the three for
the stated goal, because a 3.3-minute TTFT makes the throughput academic.

## Sequencing

Levers 1–3 are **CUDA work and cannot run unattended**; levers 4–5 are wall clock and are what the
box does overnight (`scripts/autopilot.sh`, then the eval battery). That split is deliberate: the
two phases that consume the most wall clock are exactly the two that need no supervision.

**Lever 1's first task needs no kernel and should be done first: measure C(k), the verify cost
curve, for k up to ~12.** If C(k) rises faster than E[A(T_k)] beyond width 6, the +20–25 % estimate
is wrong and the whole item should be re-priced before a line of CUDA is written.
