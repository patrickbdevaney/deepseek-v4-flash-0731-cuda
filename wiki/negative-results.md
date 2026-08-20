# Negative results — levers built, measured, and retired

This page is longer than the wins list and is more valuable. Every entry below cost a full cycle:
the idea was plausible, the implementation was correct, and the measurement said no. **They are
recorded so nobody builds them again.**

The project's own accounting: roughly **one adopted speed win per six cycles**. That ratio is not a
defect — it is what a well-run optimisation loop looks like near the end of a lever queue.

---

## 1. The most expensive lesson: a fake +28 % that passed every gate

**Finding 68 — split-K on the fp8 tile GEMM.**

Split-K was implemented, every unit gate passed, and the end-to-end number improved by **28 %**. It
was wrong. The split changed the floating-point accumulation order, and the resulting numerics
shifted the *draft distribution* enough to change which tokens were accepted — producing a faster
run that was decoding a different sequence.

**This is the origin of the LOSSLESS gate**, which is now mandatory for any claimed speculative
speedup:

```
[spec] LOSSLESS GATE: first 8 tokens match base AR -> PASS
```

A cosine-similarity gate cannot catch this. Only comparing the emitted token sequence against base
AR can. Every spec-decode number in this repo is now gated on it.

---

## 2. Retired by measurement

| lever | why it was plausible | what killed it |
|---|---|---|
| **Split-K** (F68) | classic GEMM technique | changed numerics → fake speedup; see above |
| **Draft refinement, NPASS>1** (F45) | more draft passes → better drafts | acceptance got **worse** |
| **Suffix-automaton drafting / S6** (F80) | retrieval drafting is cheap and free | **oracle ceiling +0.0 %, 0 wins in 21 verifies** — speculation hands a retrieval drafter its worst possible query, because the tokens most worth drafting are exactly the ones with no prior occurrence |
| **cp.async ring / B8-cpasync** (F81) | async weight streaming, textbook | hands back 16 registers, **raises** occupancy 4→5 blocks/SM, and is **15–53 % slower**. Depth is negative and cp-size 16 does not save it |
| **B7' — raise MoE GEMV occupancy via the register cap** (F67) | occupancy is the obvious knob | even at the optimum it runs 62 reg / 63 % occ / 155 GB/s = 67 % of roofline; the register budget is dominated by structure, not by the cap |
| **B6 — funnel partner shuffle** (F66) | avoid the misaligned load | the weights were *already* 8-byte aligned; the compensation was for a problem that did not exist |
| **`ogroup` instruction-count cures** (F76) | the kernel issues too many instructions | it is **latency**-bound: 62.3 % of 13.0 cycles between issues stalled on an L1TEX scoreboard. Deleting instructions from a kernel waiting on memory returns the memory latency, which is zero. **Closed the whole family** — the fp8x2 cvt pairing and the `exp2f`→bit-shift rewrite have the same shape and need not be built |
| **`OG_SMEM` activation staging** (F55, F79) | 8× less activation traffic, bit-exact | traffic was not the binding resource; **overlap** was. The `__syncthreads` pair forces 8 warps into lockstep and destroys the skew hiding load latency. −40 % at the shipped M=5/NR=4 |
| **Double-buffered fp8 tile chunk** (F78) | the natural next step after F74 | **+0.28 %**, killed by a four-register occupancy step |
| **B10 — draft path on the arena** (F82→F83) | priced at +5–7 % from a *measured* 10.19 ms/round | the measured time was **host** time overlapping device work. Returned **+0.41 %**, a 15× miss |
| **Tree / DDTree drafting** | standard in the literature | correct but depth-dominated; does not beat linear on this architecture |
| **Block size > 5** (F43) | longer drafts | conditionally retired — **re-opens after S5**, when acceptance justifies it |

---

## 3. The negatives that were really instrumentation failures

Worth separating, because the fix is different: these were not bad ideas, they were bad
measurements.

- **F69 — the MoE GEMV's `BN` sweep was measuring dead-code elimination.** The store was optimised
  away in some arms. It was also a latent wrong-answer bug.
- **F65/F70 — `RB` chosen twice from a probe whose grouping clamped rows-per-expert at 2**, so no
  tile ever needed chunking and the sweep could not see what `RB` is *for*. The real histogram
  inverts the ranking.
- **F59/F63 — adaptive verify width "worth 7×"**, measured on the one prompt where it does nothing,
  before F62 fixed the prefill it was measured against. Real value: +9–11 % where it engages.
- **F84/F85 — prefill MoE redundancy reported as 3.26×.** The counter counted *tiles*; the weight
  load sits inside the `rb` loop, so a tile costs `ceil(me/RB)` reads. Real figure **11.26×**, and
  the fix implied by the wrong number ("use larger tiles") would have moved **nothing** — traffic
  depends on `RB`, and the tile cap does not enter it.

---

## 4. Conditionally retired — re-open after the draft-head fine-tune

These are not dead. They were measured at acceptance 2.89 and lose there; they should be re-measured
once acceptance rises, because they all scale with draft length:

- block size > 5 (F43)
- draft refinement NPASS > 1 (F45)
- tree / DDTree drafting
- adaptive width threshold 1.5
- S4

The verify gets *cheaper per token* as `K` grows — `MLA wq_b` is 231 GB/s as an M=1 GEMV but
**351 GB/s at M=2** as an mma. Higher acceptance makes longer blocks profitable, which makes the
whole family live again.

---

## 4b. Killed inside the winning iteration — the radix top-k's two dead levers (ladder 1.2, 2026-08-20)

Both were measured while building the change in [`kernel-optimisations.md` §2.6](kernel-optimisations.md),
and both are here because "we tried the obvious optimisation and it lost" is the part that does not
survive in a diff.

**Warp-aggregated histogram: slower, and the theory behind it was right.** The radix select's cost
at T=3072 was 20.7 µs of passes against a 2.1 µs empty-kernel floor — not bandwidth (24 loads per
thread), so the shared `atomicAdd` contending on a handful of top-byte bins was the obvious suspect,
and real score rows *are* exponentially distributed, so the contention is real. Replacing it with
one `atomicAdd` per (warp, distinct digit) via `__match_any_sync`:

| T | 768 | 1,536 | 3,072 | 4,096 | 8,192 |
|---|---|---|---|---|---|
| naive shared `atomicAdd` µs | 35.0 | **26.8** | **39.0** | **45.2** | **49.3** |
| `__match_any_sync` aggregated µs | 35.2 | 28.7 | 43.0 | 49.3 | 57.5 |

Worse at every T that matters and **16 % worse at 8,192**, because `__match_any_sync` plus `__ffs`
plus `__popc` on *every* element costs more than the serialisation it removes on the minority that
collide. Reverted. The diagnosis was correct and the fix still lost — this is F76's rule
("instruction count is free in a latency-bound kernel") running the other way: instruction count is
*not* free when the thing you spend it to avoid was cheap.

**Block size was worth sweeping, and 256 was not the answer.** Same kernel, T=3072:
**128 → 53.3 µs, 256 → 38.9, 512 → 32.0, 1024 → 31.6.** 512 shipped; 1024 buys 1 % for half the
occupancy. Worth recording because the four kernels it replaced were all launched `<<<K,32>>>`, and
that 32 was never a decision — it was the warp the original one-thread selection sort happened to
sit in.

---

## 5. What the negatives taught

1. **A gate that passes is not a result that is true** (F68).
2. **A microbenchmark win is not an in-situ win** (F47, F76, F79). The bench overstates
   systematically.
3. **Occupancy is not throughput** (F81, F30). More blocks/SM can be slower.
4. **Instruction count is free in a latency-bound kernel** (F76).
5. **Host time is not critical-path time** (F83).
6. **A probe whose input distribution differs from production selects the wrong parameter** (F65,
   F70, F59).
