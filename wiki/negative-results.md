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

## 4c. Adopted, bit-exact, and worth nothing — the `lim <= topk` early-out (ladder 1.3, 2026-08-20)

This one is not a lever that lost a race. It won its race, by 22 %, and the race did not matter.

**The change.** `topk_radix_select` with `lim <= topk` cannot exclude anything — every candidate is
in the top-k before a score is read. The full path still spent one entire radix level discovering
that: clear 256 bins, one strided pass with a shared `atomicAdd` per surviving element, a serial
scan on thread 0, and the only conclusion is `hist[d] == need`. Skipping that level is a strict work
reduction with provably identical output (the threshold it would have computed is `<=` every
candidate's composite, so `thr = 0` picks the same set; the bitonic sort that orders it is
untouched). Standalone it is worth **+2.1 to +4.1 µs per call at T ≤ 512, and +0.00 ± 0.03 µs above
it** — clean, reproducible, and exactly where the theory says.

**In situ it is worth nothing, and this was stated before the run rather than after.** 21 ratio-4
layers × 4.05 µs ≈ **0.085 ms of a ~130 ms forward = 0.07 %**, against a 3.5 % run-to-run spread.
The paired sweep found deltas of −0.53 to +0.35 ms across seven targets with `tau` identical to
three decimals and 34/34 legs byte-identical; the fitted context term went `3.008 ± 0.241 →
3.036 ± 0.240`. The `i:topk` dprof mark — the only instrument that brackets the changed launch —
did see it, **0.42 → 0.28 ms at ctx 768 and 0.52 → 0.34 at 1536**, and 0.72 → 0.72 at 6144 where it
cannot fire. So the change works, at the size predicted, and that size is below the floor of every
end-to-end instrument this project has.

**It is kept, at default-on**, because it is less work for identical output behind an env arm
(`DSV4_TOPK_EARLY=0`) and reverting it would cost more than it saves. It is filed here rather than
in the wins list because **it is not a win**, and a wins list that admits 0.07 % stops meaning
anything.

**The actual finding is why it was built.** 1.3 was ranked above 1.5 on 0.4's attribution of
`i:topk` at **13.47 ms at ctx 12,288**. The iteration immediately before it took `i:topk` to
**0.72 ms at ctx 6144** — so by the time 1.3 was picked, its headroom was 0.5 ms and nobody
re-derived it. **A ranked work list is a function of a cost model, and the item above just changed
the cost model.** The re-check is free: every A/B here already runs a dprof pair, so the previous
iteration's attribution is sitting on disk. This produced ladder rule 6, and it generalises past
this repo — the more effective a queue of optimisations is, the faster its own ordering rots.

## 4d. Superseded before it shipped — the tiled `index_score`, and the cost of picking the wrong reference (ladder 1.5, 2026-08-20)

**A correct, bit-exact, 2x kernel that never ran in the engine**, because a better one existed one
assumption away — and the assumption was not about the hardware, it was about **which kernel the
bit-exactness claim was made against**.

**The change.** `index_score_warp_kernel` re-reads both operands from global on every head: 32 KiB
of `q` once per row, `kv[t]` once per head, 1.2 GB moved per call at the verify shape for 151 M MACs.
`index_score_tiled_kernel` fixes exactly that — `q` staged once per block into shared, `kv[t]` held
in registers by the warp that owns row `t`, `d` promoted to a template parameter so the inner loop
unrolls — while changing **nothing** about the arithmetic: same lane→element mapping, same serial
`dot +=`, same 5-step `__shfl_down_sync` tree, same `fmaxf`, same serial accumulation over heads. It
is bit-identical to the shipped kernel by construction and `memcmp`-gated. It is worth **2.0x**
(898.7 → 449.7 µs at S=6, T=3072).

**And 2.0x is its ceiling, structurally.** Per (row, head) the kernel does 4 useful FFMAs against
~16 instructions of overhead, half of which is the shuffle tree. SHFL retires at one
warp-instruction per SM per clock on this part, so 1.18 M (row, head) pairs × 5 steps over 20 SMs is
a **~200 µs floor** at the verify shape regardless of how the operands are staged — and the measured
tiled kernel is 451 µs against exactly that arithmetic. **The tree cannot be removed while the claim
is "bit-identical to the warp kernel", because the tree IS the warp kernel's summation order.**

**The fix was to aim the claim at the reference instead of at the incumbent.** `index_score_kernel`,
the scalar version `gate_units` checks against `ref/goldens`, accumulates **serially in d** — which
is precisely what a register-tiled GEMM does in k. So a GEMM can be bit-identical to the *reference*
while the *shipped* kernel is not, and it is **6.8x**. LOOP_LOG Finding 68 had adopted the warp
kernel as a deviation from that reference behind the LOSSLESS gate; 1.5 handed the deviation back
and got 3.4x more for it. [`kernel-optimisations.md` §2.7](kernel-optimisations.md).

**The tiled kernel is kept**, but only as the fallback for shapes the GEMM's tiling cannot serve
(`H % 8 != 0`), which the model never issues. On the shipped path it is dead code, and this entry is
what it is for.

**The generalisable form: "bit-exact" is a two-place relation and nobody says the second argument.**
A bit-exactness constraint is only as good as the kernel you point it at, and pointing it at the
*current* implementation silently inherits every reassociation that implementation ever made —
including ones adopted, as here, as explicit deviations. **Ask what the claim is against before
letting it bound the design.** The incumbent is not the reference; the reference is the reference.

---

## 5. What the negatives taught

1. **A gate that passes is not a result that is true** (F68).
2. **A microbenchmark win is not an in-situ win** (F47, F76, F79). The bench overstates
   systematically.
3. **Occupancy is not throughput** (F81, F30). More blocks/SM can be slower.
4. **Instruction count is free in a latency-bound kernel** (F76).
5. **A correct optimisation of a term that is already spent is a null result** (ladder 1.3). Rank on
   a cost model measured *after* the last thing you shipped, not the one that ordered the list.
5. **Host time is not critical-path time** (F83).
6. **A probe whose input distribution differs from production selects the wrong parameter** (F65,
   F70, F59).
7. **"Bit-exact" is a two-place relation, and the second argument is a design decision** (ladder
   1.5, §4d). Claiming it against the incumbent inherits the incumbent's reassociations and cost
   1.5 a factor of 3.4 until the claim was re-aimed at the reference.
