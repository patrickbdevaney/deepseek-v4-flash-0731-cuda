# Research prompt — the decode zenith for DeepSeek-V4-Flash-0731-REAP on Jetson AGX Thor

**Status:** open. Written 2026-08-19 from 2,156 real generations of the evaluation battery.
**Question:** where is the remaining decode performance, and what is the correct methodology to
reach the physical roofline for this architecture on this hardware?

---

## 0. Why this prompt exists, and what makes it different from the last one

Every prior optimisation pass on this repo measured decode on **short prompts**. The evaluation
battery has now produced 2,156 generations at contexts from 300 to 24,000 tokens, and the picture
at length is not the picture at zero. Benchmarks reported 24–25 tok/s; the battery sustained
8–15 tok/s and dropped to ~10 tok/s on the long rows. That gap is not noise and it is not the
model: it is a term nobody has been optimising, because at the contexts previously measured it
barely existed.

This prompt is therefore **not** "make the MoE GEMV faster". It is: given the decomposition below,
what is the complete set of techniques — kernel, algorithmic, and speculative — that closes both
terms, and in what order?

---

## 1. Ground truth (measured on this box, not inherited)

| fact | value | source |
|---|---|---|
| device | Jetson AGX Thor, `sm_110a`, 122 GiB unified | — |
| achievable bandwidth | **240 GB/s** (212 GB/s contended) | `tools/bw_probe.cu` |
| model | DeepSeek-V4-Flash-0731-REAP, K160, native MXFP4 | `config.json` |
| weights resident | 100.400 GiB (`tensor_bytes` 107,803,320,952) | `MODEL_INVENTORY.md` |
| layers / hidden | 43 / 4096 | `config.json` |
| attention | MLA, `HEAD_DIM` 512, sliding `WINDOW` 128 | `include/deepseek_v4.h` |
| KV compression | ratio 4 on even layers 2–42 (compressor **+ DSA indexer**); ratio 128 on odd 3–41 (compressor only, strided); layers 0,1 pure sliding | `deepseek_v4.h:75` |
| DSA selection | `INDEX_TOPK` 512 of `context/4` compressed rows, `INDEX_HEAD_DIM` 128 | `deepseek_v4.h:57` |
| KV cache dtype | **FP32**, hard-coded (`HEAD_DIM * 4`) — 86 KiB/token across 43 layers | `src/engine.cu:331` |
| speculation | 3 chained embedded DSpark MTP stages | `ARCH_DELTA.md` |
| measured τ | p10 2.41, **p50 2.91**, p90 3.72 | battery, n=2156 |
| AR wall @240 GB/s | 21.42 tok/s = **46.70 ms per target forward** | `ROOFLINE.md` |

## 2. The central measurement

Least-squares over 2,156 generation legs, expressed **per target forward** so that speculation is
divided out and the result is directly comparable with the bandwidth wall. Recompute at any time
with `tools/decode_model.py`.

```
ms per target forward  =  136.44  +  30.053 x (context / 1000)      R^2 = 0.965, n = 2156
measured context range: 71 to 6592          <-- NOT extrapolated beyond this

  context        0     2000     4000     6592
  ms/forward   136.4    196.6    256.7    334.5
  context %       0%      31%      47%      59%
```

**A correction that matters, recorded so nobody repeats it.** The first version of this fit gave
`147.14 + 21.891x` with R^2 0.64 and was quoted out to 24,000 tokens. It was wrong twice. Extended
records carry the BASE leg's `timings` while their `usage` has been updated to the MERGED total, so
dividing one by the other understated cost on precisely the longest generations; correcting the
divisor moved R^2 from 0.640 to **0.965**. And the table was extrapolated far past the data. Both
are now structurally impossible in `decode_model.py`: it takes the divisor from the extension block
and refuses to print a row beyond the measured range.

**INSTRUMENTATION GAP — FIX THIS FIRST, IT IS CHEAP.** `/v1/completions` returns **no `timings`
object at all**, so every continuation leg -- the only decode this project has ever run above the
8k base budget -- recorded nothing. Decode above ~6.6k context is therefore **entirely unmeasured**.
Adding `timings` to that endpoint is a few lines and makes the 8k-24k regime observable for the
first time. Until then, treat every claim about 24k decode as a conjecture.

**Two terms, both with headroom, and they are different engineering problems.**

**Term A - the context-independent 136.44 ms/forward.** It runs at **60-68% of achievable**,
headroom **1.47-1.66x**.

This corrects a headline that was wrong here and in `PERF.md`. `B_tok` = 11,202 MB is defined for
an **M=1 step**; a speculative target forward verifies K positions and reads the **union** of the
experts they route to -- measured at **17.53 of a possible 30 at K=5** (`DSV4_MOEUNION`; the
instrument self-validates at K=1 -> exactly 6.00) -- plus the entire draft side. Dividing M=1 bytes
by a per-forward wall time understates efficiency by ~1.76x. `PERF.md:44` says so explicitly --
*"a lower bound on efficiency, not an estimate of it"* -- and the 34% that fell out of it was then
quoted as an estimate anyway, including by me, who then called reproducing the same construction
"independent corroboration". It was the same error twice, not two measurements.

LOOP_LOG **F137** had already retracted the premise from the other direction: scored in situ, the
MoE region runs at **94% of roofline** and the whole step at 77%. `LEVERS.md` §8 and
`wiki/moe-gemv-ceiling.md` still carry the superseded F67 figure and should be corrected.

**Term B - 30.053 ms per 1000 tokens of context.** It is 47% of a forward by 4k and 59% by 6.6k,
i.e. the majority of decode cost well inside the range we can actually see. Bytes that must move at
the deepest measured context:

```
at context 6592:  DSA index scoring 17.7 MB + top-512 attend 22.0 MB + strided 2.2 MB
                  = 42.0 MB  ->  0.175 ms at 240 GB/s  ->  0.509 ms per forward at tau 2.91
                  MEASURED: 198.1 ms per forward   ->  389x off its own roofline
```

Term A is 2.92x from its wall. Term B is **389x** from its wall, inside the measured range, and has
never been profiled. That asymmetry is the whole finding.

**The named suspect, already in the source and never followed up.** `src/engine.cu:320` records
that the context-dependent cost is *not* the KV copy, it is the DSA index path -- "which scores
every compressed row to select top-k -- O(context/ratio) by construction -- **with a `<<<1,32>>>`
top-k on one warp** on top of it. That is where the next measurement goes." It never went. A
single-warp selection over ~1,600 candidates at 6.6k context (~6,000 at 24k), on 21 layers, once
per forward, is a latency structure that would produce exactly this.

**A falsifiable prediction that has NOT yet been tested.** `INDEX_TOPK` is 512 and selection is
over `context/4` rows, so below context 2048 there is nothing to select -- every row survives. If
the indexer's *selection* is the cost, the slope should change at ~2048. Binned medians over the
current data do not show a clean knee there (marginal cost per 1000 context runs 28.2, 34.9, 29.1,
23.6, 18.5 across the range), which suggests the cost is the **O(context) scoring pass** rather
than the top-k selection itself -- or that both matter. Resolving this decides which of 4.1's
answers applies, and it is cheap: profile one decode step at two context lengths with `ncu`.

## 3. What is already ruled out — do not re-derive these

- **τ inflation.** `ms_per_forward = ms_per_token × τ` could manufacture a slope if τ rose with
  depth. `PERF.md` measured it: **τ falls with depth**, so the confound runs the other way.
- **Kernel launch overhead.** The full 43-layer decode is captured as one CUDA graph, bit-exact,
  and measured at parity with the uncaptured path ⇒ GPU-bound, not launch-bound.
- **The KV copy** as the context-dependent cost (`engine.cu:320`).
- **`xin` growth.** Was 656 KiB per token of context across 41 layers; now a fixed ~124 MiB ring.
- **Deeper REAP / lower bit-width as a decode lever.** `COMPRESSION_PLAYBOOK.md` §0: byte
  reduction pays only in proportion to how bandwidth-bound you are, and Term A is 32% of the wall
  — inefficiency, not bytes. Weight-changing work also obsoletes a six-day eval battery.

## 4. What we want back

Ordered by expected value, but the ordering is itself part of the question.

### 4.1 Term B — the context path (largest, least explored)
1. What is the **state of the art for top-k selection on GPU** at k=512 over ~6k candidates,
   batch-1, repeated 21× per forward? Radix select, bitonic partial sort, threshold-then-compact,
   two-pass histogram? What does each cost at this size, and which are latency- rather than
   bandwidth-bound at batch 1?
2. Can the selection be **avoided or amortised** rather than optimised? Reuse of the previous
   step's top-k with a correction, hierarchical/blocked scoring, early termination on a score
   bound, or maintaining the selection incrementally as context grows by one.
3. How do **vLLM, SGLang and FlashMLA** implement DeepSeek sparse-attention index selection, and
   what do they measure for it at batch 1? What is DeepSeek's own reference kernel doing?
4. Is fusing **score → select → gather → attend** into one kernel viable on `sm_110a`, and what
   does that do to occupancy given 21 layers × per-forward invocation?

### 4.2 Term A — the weight path (bounded 3.15×, well-characterised)
5. **MXFP4 MoE GEMV at batch 1**: what do the best current implementations achieve as a fraction
   of peak bandwidth, and by what technique? `sm_110a` has **no `tcgen05`**, but **does** have
   `cp.async` and hardware FP4×2 unpack (`dspark-decode-gap-research`). What is the best known
   dequant-and-multiply inner loop under exactly those constraints?
6. Well-written batch-1 decode kernels reach 70–80% of achievable bandwidth. We are at 32%.
   What specifically separates a 32% kernel from a 75% kernel here — vectorised loads, L2
   residency of the router/shared expert, `__nv_fp4` intrinsics, persistent-CTA scheduling,
   split-K over experts, async pipelining depth?
7. With top-6 of 160 experts, is expert **gather locality** worth engineering (sorting the routed
   set, co-locating frequently co-activated experts) given measured routing statistics?

### 4.3 Speculation
8. τ is 2.91 with 3 chained DSpark MTP stages. What is the **acceptance/latency optimum** — is a
   shallower tree with higher acceptance better than depth here, given that verifying K tokens
   reads the union of routed experts (~3.8× at K=4, top-6 of 160) while the context path costs
   scale with positions verified?
9. **Prompt-lookup / n-gram speculation** on agentic coding workloads: expected τ on traces with
   heavy verbatim copying, and can it compose with the MTP head rather than replace it?
10. Draft-head fine-tuning: what is current best practice for **distribution-matched** draft
    training (the deployment workload is agentic coding, not the math/science reasoning the eval
    battery contains), and what τ gain is realistically attributable to it?

### 4.4 The interaction nobody has costed
11. **FP8 KV cache is not primarily a memory win here — it is a 4× bandwidth cut on Term B.**
    KV is FP32 today (86 KiB/token). What is the accuracy cost of FP8 KV *specifically at
    8k–24k reasoning contexts*, where the literature's degradation is largest and where this model
    actually operates? Which formats (E4M3 vs E5M2, per-head vs per-tensor scaling) and which
    parts (K only, KV, the compressed cache only, the DSA index cache only) give the best
    accuracy-per-byte? Note the DSA index cache is 64 of the 93 MB and is used only for
    *selection* — it may tolerate far lower precision than the attended values.

## 5. What a good answer looks like

- **Cites measurements, not vendor claims**, and states the batch size and context length of each.
- Distinguishes **latency-bound** from **bandwidth-bound** at batch 1 — the two terms above fail
  differently and a technique that fixes one may not touch the other.
- Is **implementable on `sm_110a` without `tcgen05`**, in CUDA, in a from-scratch server.
- Proposes an **ordered plan with predicted speedups per step**, so each can be gated
  independently against `PERF.md` — one change per measurement, bands not points.
- Preserves **bit-exactness** wherever possible. Anything that changes numerics (FP8 KV, altered
  accumulation order) must be flagged, because the ten-row eval battery describes one engine and
  a numerics change splits the table.
- States what it does **not** know.

## 6. The instrument this prompt came from

`ms/forward = a + b·context` fitted over the battery's own records — every generation stores
`timings.decode_ms`, `timings.tokens_per_verify` and `usage`, so the decomposition is recomputable
at any time and should be re-run after every optimisation. **Report both coefficients, never a
single tok/s number**: a single number at one context is what hid this for the whole project.
