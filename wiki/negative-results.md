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

## 4e. The lever that was already applied — `jetson_clocks` (ladder 3.1, 2026-08-20)

The ladder carried this item from the beginning on a line in HARDWARE.md: pinning is worth
"**+6.4 % / +3.5 % / +3.0 %**", GPU 1386 -> 1575, EMC 2750 -> 4266. It was the last throughput item
before hand-back, and it looked like free money.

**Nobody had ever sampled a rail while a decode was in flight.** One `cat` of
`/sys/kernel/debug/bpmp/debug/clk/emc/rate` every two seconds, across three governed arms:

| arm | machine state | compute-window samples | EMC at 4266 | gpc at 1386 |
|---|---|---|---|---|
| A2 | governed | 43 | **97.7 %** (mean 4231 MHz) | **97.7 %** (mean 1378 MHz) |
| A2p | governed | 42 | **97.6 %** (mean 4241 MHz) | **97.6 %** (mean 1380 MHz) |
| B2 | pinned 120W | 89 | 100 % | 100 % |

**The governed box already runs at the pinned frequencies.** Both rails ramp to their ceiling within
~2 s of the GPU going busy and stay there. The famous "315 MHz idle / 2750 MHz EMC" state is real,
and it is the **~90 s checkpoint load** — when the GPU is idle and the CPU is reading 100 GiB off
disk. Reading an idle clock and calling it the operating clock is how a 2.3 % lever got written down
as a 6.4 % one.

**So the ceiling is not the lever; the ramp is.** What pinning removes is the 2.3 % of the compute
window spent climbing, and it measures as exactly that: **+1.99 +/- 0.15 % (8/8 legs positive)**
against the closest-in-time governed arm, **+2.96 %** against a time-interpolated one, with the
governed-vs-governed drift control at +2.03 +/- 0.99 %. Below the 3.5 % run-to-run spread. **Not
counted as a ladder win.**

**And the 1575 MHz half of the item is worth nothing at all**, which is the correct answer for a
bandwidth-bound engine asked about its core clock. Reaching 1575 needs `nvpmodel -m 0` (MAXN) —
`jetson_clocks` raises a rail to its *governor's* ceiling, and at nvpmodel 1 `/etc/nvpmodel.conf`
caps GPU MAX_FREQ at 1386000000, so the item's own headline was unreachable by the tool the item
named. Measured at a verified 1575 MHz for 18/18 samples: **+1.68 +/- 5.83 % paired** against
pinned-120W. The deployment pins at the current power mode and does not touch `nvpmodel`.

**It is adopted anyway, and not for the 2 %.** `scripts/pin_clocks.sh`, called by `run_model.sh` and
`run_server.sh`. The reason is [`measurement-and-traps.md` §24](measurement-and-traps.md): the
base-AR window is measured *inside* the ramp, so it reports 88 ms/tok governed and 72.8 pinned — a
**21 % artefact that reproduces to a tenth of a millisecond** and reads exactly like a regression.
It is the entirety of ladder item 2.5's unexplained "17.4 % base-AR fall". A free 2 % is the smaller
half of this entry.

**The generalisable part.** Three of this ladder's items — 1.3, and now both halves of 3.1 — were
ranked on a number measured against the wrong operating point: the wrong context (1.3), and the idle
clock (3.1). The check that would have caught 3.1 costs one sampler and no model load, and it is the
same check §16 asks for: **before spending an iteration closing a gap, measure the gap under the
conditions the workload actually runs in.**

## 4f. Two hypotheses killed on the way to localising the prefill race (ladder 1.9, 2026-08-20)

Neither of these was a performance lever; both were *causes* proposed for the engine's
nondeterminism, and both are now dead at the lengths where the defect actually lives. Recorded here
because a cause eliminated with a measurement is worth exactly as much as a lever retired with one,
and Finding 60/61 spent four hypotheses reaching a conclusion that was bounded to lengths 1–29.

**Uninitialised arena scratch — dead, again, and this time in the right regime.** Finding 61
exonerated the arena with `DSV4_ARENA_ZERO=1` against the point-to-point 5-cycle. 1.9 re-ran the
eight-point length ladder with `DSV4_ARENA_ZERO=1` **on both arms** and got the identical verdict:
clean at prefill 128–160, divergent at 192 and 256 (`evidence/decode_loop/lhash_Z.txt` against
`lhash_W.txt`). Zeroing the arena to its high-water mark on every reset changes nothing. The same
run also killed the weaker form of the hypothesis: with the arena zeroed, the *first differing
layer* moved (14 and 2, against 2 and 10 unzeroed), which is not what a deterministic wrong branch
does.

**"It is per-process state" — dead.** Every result up to this point compared two *processes*, which
leaves address layout and whatever a fresh `cudaMalloc` happens to contain as live explanations.
The R protocol runs the identical sweep point **four times inside one process**: at prefill 192 the
four prefills disagree with each other, first differing layer 2 / 4 / 12 / 28 across the pairs, and
the pattern is not the same in the second process either
(`evidence/decode_loop/lhash_R.txt`). Same process, same addresses, same buffer contents, four
different answers — it is a race or an order-dependent reduction, not state.

What survives is stated in the ladder entry: the divergence is confined to `compress_ratio != 0`
layers, so the shortlist is `compressor_forward`, `indexer_forward` and `sparse_attn` over the
combined index list, and nothing else in `compressed_attn_forward`.

## 4g. Retired on a null before it was built on — `hpb` alone, and the reuse hypothesis (ladder 1.7, 2026-08-20)

**The lever:** `sparse_attn` gathers the same `topk` latent KV rows for all 64 heads of a query
(`num_key_value_heads == 1`, so `topk_idxs` is indexed by `(b, m)` and not by head). At the
saturated shape that is `topk x 2 KB x m x h` = 168 MB per compressed layer per step, of which
63/64ths is the same rows re-read. `HPB` — put HPB heads of one query in one block so warps
2..HPB hit L1 instead of L2 — is the exact fix, it already existed in the kernel, and the launch
heuristic was giving `HPB=1` across the whole decode/verify regime.

**The number that killed it:** **1.00x.** `gate_sparse_hpb`, hpb ∈ {2,4} against hpb=1, at all six
shapes the engine issues:

    m=1  topk=640   0.737 -> 0.736 / 0.737     m=6    topk=640    0.818 -> 0.822 / 0.821
    m=2  topk=640   0.746 -> 0.754 / 0.754     m=256  topk=320   15.78 -> 15.76 / 15.83
    m=2  topk=320   0.375 -> 0.382 / 0.381     m=1022 topk=1277 263.2  -> 263.5 / 264.3

L1 was already catching the reuse; the redundancy was real and cost nothing. `HPB=8` is worse than
null — 0.80x at the 1022-token prefill and **0.52x** at the K=6 verify — which is separately why
the shipped default had been a live regression (`measurement-and-traps.md` §28).

**Why it is here and not in the wins list:** the *shape* `hpb` provides turned out to be necessary
for the win that did land — the smem staging in `kernel-optimisations.md` §2.9 needs several warps
per block to amortise one vector load of the row, and is itself **0.71x at hpb=1**. So `hpb` is
load-bearing scaffolding for a different mechanism, and worth exactly nothing for the one it was
built for. Two levers with the same knob and opposite stories: the honest record is that the
attribution was wrong and the parameter survived by accident.

**The generalisation:** *sweep the knob that fixes your hypothesised mechanism, alone, before
building on it.* One column in one table, 90 seconds, no checkpoint. The redundancy arithmetic was
large enough (168 MB, 63/64ths) to read as a diagnosis; it was a description.

## 4h. Headroom that was never there — the `cattn:q_proj` "free 4.4 ms" (ladder 1.8, 2026-08-20)

**The claim, from 0.4:** `cattn:q_proj` runs 1.71 ms on some verify steps and 5.47 on others at the
same width, shape and context; it is 14.6 ms of every step; *"if the cheap mode is reachable on
demand that is ~4.4 ms/step of Term A for free"*.

**Killed.** The two modes are the same GEMM with and without 21 layers of compressor emits running
beside it on `g_side`. A schedule term computed from the dprof tag alone,
`g = #{ j in [ctx,ctx+VB) : (j+1)%4 == 0 }`, separates them **153/153 on 0.4's own log and 174/174
on a fresh arm, with no overlap between the populations**; under `NO_ATTN_SPLIT=1` the swing goes to
**1.00–1.02x** and the identical time reappears in `cattn:compress` (2.50 -> 8.34 ms at fixed VB=2).
Mechanism and evidence: `measurement-and-traps.md` §29.

**Why it is worth a page rather than a line.** The refutation is *conservation*, and conservation is
the check that most "one mode is cheaper" findings need. `cattn:q_proj + cattn:compress` is
**10.16 ms at g=0 in both arms** — two separate checkpoint loads agreeing to two decimals — and at
g>=1 it is 18.46 serial against 17.18 split. So the 4.4 ms is not a mode that could be selected; it
is compressor traffic a correct engine must move, it is **already** overlapped, and the overlap is
**already** worth a measured 0.81 ms/forward (2 SE band [0.72, 0.90], 9/9 legs faster and
byte-identical, tau equal to three decimals). There was nothing left to take.

**The value of a negative here is a smaller denominator.** Term A needs to fall 22.4 ms to reach its
floor. Before 1.8, 4.4 of those ms had a named owner that would have absorbed an iteration and
returned nothing — the 1.3 failure mode, one item earlier in the queue. After it, the same
measurement prices what the emit actually costs (**4.56 ms/forward amortised at 52 % of its 880 MB
byte roofline**) and leaves two honestly-small follow-ups on the ladder (1.11, 1.12) with
pre-registered ceilings of 1.29 and 0.53 ms/forward. Deleting a phantom is not progress on the
clock, but it is progress on the ladder, and it is cheaper to do before the kernel work than after.

## 4i. Packing the KV cache is a CAPACITY lever, not a throughput one — and the kernel it feeds says why (ladder 1b.2, 2026-08-20)

**The change.** `KV_PRECISION_FINDINGS.md` found that the main KV cache already stores values that
sit exactly on the E4M3 grid with an exact power-of-two scale, and then keeps them in FP32 — 8 bits
of information in 32 bits per element. 1b.2 stores what was computed: 448 E4M3 bytes + 7 UE8M0 scale
bytes + the 64 FP32 RoPE dims = **711 B of payload in a 720 B row against 2048 B, 2.844x**, behind
`DSV4_KV_PACK=1`. It is BIT-EXACT, and that is not an argument, it is four memcmp gates and a
16-of-16-leg engine A/B (see [`kernel-optimisations.md` §3](kernel-optimisations.md)).

**What killed it as a throughput lever, measured in seconds and not in checkpoint loads.**
`gate_kv_pack --bench` times `sparse_attn` over the packed cache against `sparse_attn` over the FP32
cache holding the identical values, at the shapes the engine issues:

```
  shape                            fp32       packed     ratio
  m=1   topk=640 (base AR)       0.5098 ms   0.6565 ms   0.776x
  m=2   topk=640 (mean verify)   0.5540      0.6883      0.805x
  m=6   topk=640 (max verify)    0.7249      0.8373      0.866x
  m=2   topk=320 (ctx 768)       0.2830      0.3452      0.820x
  m=256 topk=320 (prefill)      13.169      14.561       0.904x
```

**A third fewer bytes per row, and the kernel got SLOWER at every shape.** That is
`COMPRESSION_PLAYBOOK.md` §0's standing rule paying out on itself — *byte reduction pays in
proportion to how bandwidth-bound you already are* — and 1.7 had already established that this
particular kernel is not bandwidth-bound but ISSUE-bound. Staging a 2048 B row costs one `float4`
load and one `float4` shared store per four elements; staging a 720 B row costs one 4 B load, one
scale byte, two `cvt.rn.f16x2.e4m3x2`, four multiplies and the same store. **Fewer bytes, more
instructions, on a kernel whose constraint is instructions.** §5 rule 4 ("instruction count is free
in a latency-bound kernel") turns out to have a converse.

**The retune was checked, not assumed.** A packed row has a different cost structure from an FP32
one, and the unpack is paid once per block per gathered row — so `hpb`, which is exactly how many
heads share a block, is the parameter that could amortise it. Swept: at the decode shapes the packed
optimum is `hpb=4, smem=2`, **the same launch 1.7 chose for FP32**, and `hpb=8` is worse in both
layouts. There is no packed-specific launch to recover.

**The first implementation was 30 % worse than that and the fix is worth recording**, because the
instinct it violates is a common one. The staging loop originally read a `uint4` — 16 codes per
thread, only 28 vector loads for the 448 codes, which is the *better* number by every metric a
roofline argument uses. But the block is 128 threads at `hpb=4`, so **28 threads did sixteen unpacks
each while 100 waited**, on a step that sits on the double-buffered prefetch's critical path:
0.699x, and +5.00 ± 0.37 ms per forward in the engine. Four elements per thread — 112 loads instead
of 28, i.e. *four times the load instructions* — put 112 of 128 threads to work for one round each
and took it to 0.776x. **Minimising the load count was the wrong objective; occupying the block was
the right one.**

**What it IS worth, and why it ships default OFF rather than not at all.** The saving is real and it
is capacity: **2.844x on every KV allocation** — 1.61 → 0.57 GiB at seqmax 16384, 3.21 → 1.13 GiB at
32768, 6.43 → 2.26 GiB at 65536 — with no accuracy exposure of any kind, because the stored values
do not change. And the byte reduction does buy something in the engine that the microbench cannot
see: the paired sweep splits into a **flat Term-A cost and a Term-B saving that grows with context**
(numbers in [`context-scaling.md`](context-scaling.md)), which is the L2-residency second-order
effect `KV_PRECISION_FINDINGS.md` §4 hypothesised, now measured. So the switch exists, it is proven
bit-exact, and it is the thing to reach for when `seqmax` is the binding constraint — which
`EVALS.md` records that it is for GPQA-Diamond. It is not the thing to reach for to make decode
faster at the contexts this ladder measures.

## 4j. Six ablation arms, six divergences — and a boundary prediction that was falsified in its own gate (ladder 1.10, 2026-08-21)

Two negatives from the iteration that finally named `compressed_attn_forward`'s race, both worth
keeping because both were *the plan* and neither worked.

**THE ABLATION CAMPAIGN NAMED NOTHING.** 1.10's own entry proposed the cheapest first step: re-run
1.9's repeat protocol under `DSV4_TOPK_RADIX=0`, then `NO_IXGEMM=1`, on the grounds that *"either
one coming back clean names the kernel outright"*. `scripts/lhash_ablate.sh` ran that and four more
arms — one env flag per candidate, each swapping one kernel for a different implementation of the
same maths on the shipped binary, 56 pairs of 43 layer-hashes per arm:

| arm | swaps | pairs diverging |
|---|---|---|
| `base` | nothing (control on today's binary) | 56/56 |
| `NO_IXGEMM=1` | `index_score` GEMM → tiled | 55/56 |
| `NO_IXGEMM=1 NO_IXTILE=1` | `index_score` → warp | 56/56 |
| `DSV4_TOPK_RADIX=0` | radix select → warp selection sort | 56/56 |
| `NO_FP32MK=1` | 1.12's `gemm_fp32` warp tile → legacy | 55/56 |
| `DSV4_SPARSE_HPB=1 DSV4_SPARSE_SMEM=0` | 1.7's `sparse_attn` staging → pre-1.7 | 56/56 |

**335 of 336 pairs diverged.** The culprit was an *aliased buffer* in `hadamard`, which has no env
flag and is not a "candidate kernel" in the sense the arm list was built on. The list came from
"which kernel is complicated" — split reductions, top-k, a new GEMM tile — and the answer came from
"which kernel writes the buffer it is reading". **An ablation sweep can only test hypotheses that
are already on the list, and the cost of the wrong list is the whole sweep.** What the campaign
*did* buy is real and is why it is recorded rather than deleted: five kernels are now excluded by
measurement instead of by argument, and the exclusions held when the true cause was found.

The thing that actually named it cost no GPU time at all: `tools/lhash_pairs.py` reading 1.9's
existing logs all-pairs and printing **the compression ratio of the first differing layer**. Every
one of 335 was ratio 4, none ratio 128 — and only ratio-4 layers run `indexer_forward` and the
`rotate` compressor. See [`measurement-and-traps.md` §36](measurement-and-traps.md).

**AND THE GATE'S OWN "DECODE IS SAFE" READING WAS A THIRD NEGATIVE, CAUGHT ONLY BY THE ENGINE.**
The row sweep found the two aliased call sites decode reaches — both `rows = 1` — clean at 200
repeats, and that was written down as an exoneration. It is an artefact of benchmarking the kernel
ALONE: put a filler kernel on the other 20 SMs and the same `rows = 1` call goes to **65/200
differing, 28 distinct results**. The engine forks those emits onto `g_side` deliberately. **Every
microbenchmark in this repo runs its kernel alone, so none of them can see a race that needs a
neighbour** — and the thing that did see it was a cross-run identity matrix over saved server logs
that cost nothing to compute. [`measurement-and-traps.md` §36](measurement-and-traps.md).

**THE BOUNDARY PREDICTION WAS WRONG IN THE SHAPE THAT MATTERED.** `tests/gate_hadamard_alias` was
written around a pre-registered claim: threads of a row race only when they are not co-scheduled,
one 256-thread block owns two rows, so while `blocks <= 20 SMs` every block has an SM to itself and
the defect cannot fire — a **step** at rows 40, i.e. prefill `s = 163/164`. At 8 repeats per row
count the gate printed `PREDICTION MISSED` four times: rows 41, 42, 48 and 56 clean, rows 44 racing.
At 200 repeats it is a **rate**, 0/200 up to 20 blocks and then 17, 28, 41, 74, 116, 184, 200/200 as
the grid grows. The direction and the scale survived; the word *sharp* did not, and it had been
inherited from 1.9, which had one sample per length and could not have seen the difference.
**A threshold measured at one sample per point is a lower bound on where the defect starts, never
the boundary itself.**

## 4k. Wider speculation blocks, and a prediction that named the right mechanism with the wrong sign (ladder 2.1, 2026-08-21)

**Killed: every block width above 5. 7 −1.70 %, 8 −2.93 %, 9 −6.30 %, 10 −9.36 %, 12 −10.86 %,
monotone, paired per prompt against the shipped 6.** The width above which extra proposals stop
paying is now measured rather than bounded, and the whole region 7–12 is closed.

**What makes this worth a page is that `tau` rises the entire way down.** Suite mean acceptance goes
3.48 → 3.69 → 3.85 → 3.94 → 4.07 → 4.16 → 4.19 → 4.36 as the block widens 4 → 12, while throughput
falls 10.5 %. Acceptance is the metric this project has optimised the hardest and it is the metric
that says "wider is better" at every single point. A wider block is a lever that buys the score and
charges more than the score is worth: each extra drafted position costs 3.324 ± 0.281 ms whether or
not the verify reaches it, and at BLK=12 **7.8 proposals per round are drafted and thrown away**,
24.5 ms of a 169 ms round. Reporting `tau` without `ms/token` beside it would have adopted this.

**The ladder's own prediction was wrong in the direction it named.** Item 2.1 was written as "with
Term B small the optimum returns to ~7-9 from an apparent 11-13" — i.e. the expensive verify would
pull the optimum down *into* 7–9. It pulls it to **5**, below the shipped 6, and the "apparent
11-13" was never a measurement anyone took here: the default was 6 and F94 had already closed 8. The
mechanism in the prediction was right (a verified position stopped being free, so the width had to
be re-decided); the arithmetic was a guess dressed as a prior. See
[`kernel-optimisations.md` §2.13](kernel-optimisations.md).

## 4l. HASS — training the draft head on its own predictions (P2.6, 2026-08-22)

**The idea, and why it was worth trying.** The draft head is trained by teacher forcing: at every
position it is fed the *ground-truth* previous token. At serve time it is fed **its own previous
prediction**. HASS closes that mismatch by feeding `prev = argmax(logits)` from position
`hass_from` onward during training.

It is unusually cheap in this architecture, which is the specific reason it looked attractive here:
the base `logits` do **not** depend on `ids_in` at all — only `markov_head`'s bias does — so
free-running costs one extra `argmax` per position and no second forward.

**What it measured.** Two arms, both at block width 5 against an incumbent of `tau` 3.8413:

| arm | `tau` | tok/s | vs the winning recipe |
|---|---:|---:|---|
| `s3recap-p25-b0.1` — the recipe, no HASS | **3.8413** | 28.3825 | — |
| `s3recap-hass1-p25` — recipe + HASS | 3.7950 | 27.8562 | **−0.046** |
| `s3recap-hass1` — HASS alone | 3.6225 | 26.7563 | −0.219 |

**Monotone in the wrong direction.** More HASS, worse head — alone it is worse than the plain
control (3.6250), and composed with the winning recipe it still subtracts. A lever that is
under-tuned produces a *non-monotone* sweep with a hidden optimum; a lever that is wrong produces
this. That is why P2.6 is retired rather than swept over `hass_from`.

The hold-out gate reached the same verdict independently, by a different instrument and a different
threshold:

    VERDICT: STOP
      - suite mean tau 3.5238 DROPPED below the 3.5362 baseline -- single-domain overfit

**The reading.** The train/serve mismatch HASS closes is real; it is simply not this head's binding
constraint. On a 1536-sequence corpus, free-running rollout compounds the head's own errors into the
training signal faster than it teaches robustness — the added variance costs more than the mismatch
does. This is consistent with the corpus-saturation arithmetic in `draft-head-finetuning.md` §5: at
this scale the recipe is data-limited, and HASS spends data to buy robustness.

**What it is worth keeping.** HASS was not a wasted arm — it is the only configuration that produces
**free-running acceptance labels**, and those gave the first honest measurement of the confidence
head's loss magnitude: **O(1)**, against the **10034** F100 measured under teacher forcing. That
number unblocks ladder 2.3, which had been deliberately stalled rather than run with a guessed
`a_conf`. See `draft-head-finetuning.md` §9.4.1.

**The general shape.** A lever borrowed from the literature can be correctly implemented, cheap, and
still wrong for the regime you are in — and the way to find out is to compose it with the recipe
that already works, not only to run it alone. Alone, `hass1` is 0.219 below the incumbent and could
be dismissed as a bad interaction with the control's loss weights. Composed, it still subtracts,
which is what makes the retirement safe.

---

### 4m. Adaptive block width: real, but worth +1.8 % not +20–25 % (2026-08-23)

**Claim tested.** The served draft width is a single constant (5). If the best width differed by
task shape, an engine that varied it per prompt would beat any fixed width. `ROADMAP.md` priced this
at **+20–25 %** and it was the largest rung on the decode ladder — large enough that the whole CUDA
programme was sequenced behind it.

**Measurement.** `scripts/measure_ck.sh`: widths {4,5,6,7,8,9,10,12} × 9 suite prompts × 200 tokens,
one checkpoint load, on the champion head `ckpt-head-s3recap-p25-b0.1`.

| | suite mean tok/s |
|---|---|
| best fixed width (5) | 27.982 |
| width 4 | 27.748 |
| width 6 | 26.872 |
| **oracle, per-prompt k\*** | **28.479** |

**Verdict: +1.77 %, and that is a ceiling nothing can reach.** The oracle picks each prompt's best
width *with hindsight*, from a table computed after the fact. A live engine must predict k\* per
position from the confidence head (AUC 0.88) and necessarily realises less. k\* does vary —
{4, 5, 7, 8} — so the mechanism is real; it is the magnitude that was wrong, by an order of
magnitude.

**Why the estimate was so far off.** It assumed the widths were far apart in value. They are not:
four of nine prompts are already at their optimum at 5, and the two directions of deviation cancel.
The agentic categories want **wider** (multi_turn 7 at 35.14, agentic_format 8 at 31.09); control,
code_gen and explanation want **narrower** (4). Averaging a +4.3 % and a +4.3 % that point opposite
ways across a nine-prompt suite leaves +1.8 %.

**What it cost to find out: 12 minutes.** This is the argument for measuring a lever before building
it. The alternative was a kernel rewrite of the draft loop to support a variable width, sequenced
ahead of the AR-kernel work, to buy under two percent.

**Two things it bought.**

1. **Width 5 re-confirmed as the best fixed width** on a fresh measurement, independently of ladder
   2.1 — 27.98 against 27.75 at 4, and monotone decay above 5.
2. **The remaining headroom is in the kernels, not the speculator.** `tau`'s ceiling *is* the draft
   width, so a fixed 5 caps even a perfect head at 1.30×. Varying the width was the one lever that
   removed that cap, and it has now been measured as nearly flat — meaning the width the target's
   *entropy* supports is close to 5 almost everywhere. AR kernel headroom (+5–10 % est.) is now the
   largest remaining decode lever, and prefill (6.6× TTFT) the largest practical one.

**Headline consequence.** The decode target dropped from **35–42 tok/s to 31–35 tok/s**. Recorded in
`README.md`, `DECODE_ENDGAME.md` and `ROADMAP.md` rather than quietly left standing.

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

## S6 / suffix drafting, closed a second time — on the workload it asked for (2026-08-26)

F80 retired S6 at an oracle ceiling of **+0.0 %** and scoped its own refutation: it was measured on a
period-8 degenerate repeating decode, and *"reopening needs a long-repeated-context prompt on which
`mlen` routinely reaches the block size."* `PHASE2_PLAN.md` §6.1 named that as the reopening
condition and called it the highest-value-per-effort item in the plan.

**The condition has now been met, and S6 still loses.** Three prompts built to the condition and
encoded through the checkpoint's own tokenizer — a copy-heavy manifest rewrite whose correct output
is almost entirely a verbatim span of the prompt, `longctx_001`'s needle-plus-repeated-filler, and a
tool-call schema echo — run under `DSV4_SUFFIXPROBE=1` at block 5, **217 verifies** against F80's 21.

| | copy-heavy edit | needle + filler | tool-call |
|---|---:|---:|---:|
| verifies | 68 | 75 | 74 |
| a suffix match existed | **68/68** | 72/75 | 69/74 |
| MTP (shipped) tok/verify | 5.926 | 5.360 | 5.405 |
| suffix only, sound lower bound | 5.162 (−12.9 %) | 3.600 (−32.8 %) | 2.838 (−47.5 %) |
| **ORACLE `max(MTP, suffix)`** | **+0.2 %** | **+0.5 %** | **+1.2 %** |
| suffix beat MTP | 1 | 2 | 5 |
| best cascade, any threshold | −12.4 % | −30.6 % | −45.0 % |

The reopening condition is unambiguously satisfied — a match existed in **209 of 217** verifies and
`mlen` ran 6–32 against a block size of 5, where F80 saw `mlen = 0` in 13 of 21. So this is not
trap 27 again; the drafter is being handed exactly the queries it wanted. **It still cannot use
them.** The oracle — a cascade that picks the better drafter for free, every time, with no selection
cost — tops out at **+1.2 %**, against a 3.5 % promotion bar. Every realisable threshold is negative.

**Why, and it is not the drafter's fault.** The MTP head reaches **5.926 of a 6.0 ceiling — 98.8 %**
— on the copy-heavy prompt, and 89–90 % on the other two. On precisely the shapes SuffixDecoding
reports wins for, the shipped head has already taken almost everything there is. A suffix drafter can
only win where the MTP is wrong, and on repetitive spans the MTP is not wrong.

**S6 is closed.** Not "closed on the wrong workload" — closed on its own stated reopening condition,
with 10x the evidence. Do not reopen it without a *new* mechanism, not a new workload.

**The wider reading:** this also prices the acceptance ceiling for the agentic regime generally. At
32.4–35.0 tok/s and 2.25–2.51x on these three prompts, the engine is at 89–99 % of the block-5
per-verify ceiling on the workloads it exists to serve. The suite mean of `tau` 3.84 is dragged by
other categories; where it matters most, there is nearly nothing left for *any* draft technique.

## Warps per block on the prefill `ogroup` — a null, and the reason it was a null (2026-08-26)

`tc_ogroup_fp8_mt_kernel` launches `<<<grid,32>>>` — one warp per block — which is exactly the shape
LOOP_LOG Finding 21 identified on the MoE GEMMs, where `moe_wpb()` (4 warps/block) was a real win:
"a 32-thread block still consumes a whole block slot, so half the SM's warp slots were unusable."

Packing 4 warps per block here, each taking its own n-block, changed **nothing**:

| | `cattn:ogroup` | prefill |
|---|---:|---:|
| `OG_WPB=1` (shipped) | 1544.80 ms | 91.0 tok/s |
| `OG_WPB=4` | 1542.56 ms | 90.8 tok/s |

Bit-exact (token streams identical), and inside run-to-run noise on both columns.

**Why.** Finding 21's fix pays when the SM is short of *resident work*. `ogroup` at prefill launches
`(R/8, G, ntile/MT) = 128 x 8 x 7 = 7168` blocks over 20 SMs; it is not short of blocks, and
occupancy was never what bound it. The traffic model says what does — per layer at bs=845:

```
weights, ideal        33.6 MB       activations, ideal      55.4 MB
weights, MT=8        234.9 MB       activations, NB=1     7088.4 MB   <- 97% of the traffic
```

The activation block is re-read once per n-block, 128 times, because a warp owns 8 of R's 1024
columns. That model predicts **1231 ms against 1543 measured**, so it accounts for the region. Warps
per block moves no bytes and therefore moved no time.

**The lesson is about transfer.** A fix that was large on one kernel was assumed to transfer to
another with the same launch shape. Launch shape is not the mechanism; *what is scarce* is. The MoE
was short of warp slots, `ogroup` is short of bandwidth, and the two want opposite work. The knob is
left in place, defaulted to 1, so the null does not have to be re-derived.
