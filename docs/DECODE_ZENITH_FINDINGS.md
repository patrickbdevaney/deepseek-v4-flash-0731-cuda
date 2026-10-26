# Decode zenith — findings and implementation plan

**Written 2026-08-19** from a four-front research pass against `RESEARCH_PROMPT_DECODE_ZENITH.md`,
plus regression over the evaluation battery's own 2,156 generation legs. Three fronts have reported
(context path, weight path, speculation); the KV-precision front is still open.

Every number here is marked: **[M]** measured on this box, **[R]** recorded in this repo by a prior
session, **[X]** external with a citation, **[D]** derived arithmetic shown at the point of use.

---

## 0. The one-paragraph answer

The project has been optimising the smaller half of the problem, in the one regime where the larger
half is invisible. Decode cost is `136.44 + 30.053 x (context/1000)` ms per target forward **[M]**.
The constant term is at **60-68 % of achievable bandwidth** — roughly 1.5x of headroom, hard-won and
partly unexplained. The context-linear term is **198 ms at the deepest measured context against a
byte floor under 1 ms**, and it is a **single-thread selection sort**. Every prior decode measurement
in this repo was taken at short context, which is the only regime where that kernel is free.

---

## 1. The corrected cost model

**SUPERSEDED AS OF 2026-08-19 by ladder item 0.3, and left standing because the derivation below is
what the plan was built on.** After the warp top-k fix the same regression, re-fitted on a
controlled context sweep against the current binary, gives
`ms per target forward = 130.98 (+/- 2.25) + 7.362 (+/- 0.370) x (context/1000)`, R^2 0.971, n 48,
measured context **249-12,410** — `b` is **4.08x smaller** and the constant term is unchanged.
See `DECODE_LADDER.md` 0.3 for the per-point table, the acceptance caveat (that corpus runs at
tau 1.68, not 2.91) and the fresh-prefill controls. Everything below describes the PRE-FIX engine.

```
ms per target forward = 136.44 + 30.053 x (context/1000)     R^2 0.965, n 2156   [M]
measured context range 71-6592. Above that, decode is UNMEASURED (see 5.1).

context        0     2000     4000     6592
ms/forward   136.4    196.5    256.6    334.5
context %       0%      31%      47%      59%
```

Recompute with `tools/decode_model.py`. Report **both coefficients**; a single tok/s figure at one
context is exactly what hid this.

### 1.1 Term A — context-independent, 136.44 ms

| expert-union assumption | forward bytes | wall @240 GB/s | efficiency | headroom |
|---|---|---|---|---|
| measured 17.53 at K=5 **[R]** | 19,723 MB | 82.18 ms | **60.2 %** | 1.66x |
| fit-implied 22.0 **[R]** | 22,292 MB | 92.89 ms | **68.1 %** | 1.47x |

**The widely-quoted "34 % of achievable" is wrong and this repo already said so.** `B_tok` = 11,202 MB
is defined for an **M=1 step**; a target forward verifies K positions, reads the *union* of the
experts they route to — measured 17.53 of a possible 30 at K=5, on an instrument that self-validates
at K=1 by returning exactly 6.00 **[R]** — plus the whole draft side. `PERF.md:44` calls the 34 %
figure *"a lower bound on efficiency, not an estimate of it"*. It was then quoted as an estimate,
including by me, who compounded it by describing a second computation of the same construction as
independent corroboration.

`LOOP_LOG` **F137** retracted the same premise from the other direction: scored in situ, the **MoE
region runs at 94 % of roofline** and the whole M=1 step at 77 % **[R]**. External calibration on
this same silicon **[X]**: llama.cpp MXFP4 MoE 63.9 %, vLLM NVFP4 MoE 50.3 %, vLLM *dense* W4A16
97.9 %. **Every MoE stack on this box lands at 50-68 %; this engine is at the top of that band.**
The ~30-point MoE-vs-dense gap reproduces across three frameworks, three quantizations and two
architectures, and **no published work explains it.** It is the real ceiling on Term A.

`LEVERS.md` §8 and `wiki/moe-gemv-ceiling.md` still carry the superseded F67 figure and should be
corrected before they mislead a third time.

### 1.2 Term B — context-linear, 30.053 ms per 1000 tokens

At the deepest measured context (6592), the bytes that must move are ~42 MB = **0.509 ms per forward**
at tau 2.91 **[D]**. Measured: **198.1 ms**. Roughly **390x off its own roofline, inside the measured
range**, and never profiled.

---

## 2. Finding 1 — the context term is a one-thread selection sort

`kernels/compressed_decode.cu:550` (and `:47`, `:58`, `kernels/indexer.cu:120`), once per ratio-4
layer, 21 of them, per target forward:

```c
__global__ void k_topk_masked(int* out, const float* score, int Tmax, int topk, int winmax){
    if(threadIdx.x||blockIdx.x) return;                       // 1 thread of 32; 1 SM of 20
    for(int k=0;k<topk;++k){ float best=-1e29f; int bi=-1;
        for(int t=0;t<Tmax;++t) if(sh[t]>best){best=sh[t];bi=t;}   // O(512 x ctx/4), serial
```

**Three independent measurements agree:**

1. Regression over 2,156 real generations, R^2 0.965; tracks *generated* tokens (R^2 0.871) not
   prompt length (R^2 0.082), so the scan is bounded by live context **[M]**.
2. The kernel compiled from this source and benchmarked on this Thor at k=512 **[M]**:
   T=1024 → 9,672 us; T=2048 → 19,115 us; **T=6000 → 55,745 us**, against **26.13 us** for an exact
   single-CTA radix select — **2,134x**, index sets verified bit-identical at seven shapes.
3. Cycle arithmetic: **5.33 cycles per inner iteration, constant across a 3.3x context range** **[D]**.
   The inner loop alone accounts for the entire measured term.

**Why it was recorded as a null.** `LOOP_LOG` F71 profiled it at **T~19 — context ~76 tokens** —
where `topk` collapses to `min(512,19)` and the sort is 361 comparisons. That produced `i:topk =
0.12 ms`, and `DECODE_MAX_REPORT.md:130`'s claim of being "2.6x faster than SGLang's optimised
15 us" — against SGLang's figure for **K=512 over a 1M-token context**. Never comparable.
`LEVERS.md` B0's *"the class paid twice and is now dry"* rests on it.

**Two further defects in the same path:**

- **`index_score` runs at ~2 % of achievable bandwidth** **[M]**: 658 us at T=6000 against an 8.2 us
  pure-stream floor. Cost is exactly linear in H because each of the T warps re-reads the entire
  32 KB query tensor from global — **196.6 MB of L1/L2 traffic per layer** — plus a 6-deep dependent
  `__shfl` chain per head. It is a GEMM (`L[T,64] = Kc[T,128] x Q^T[128,64]`); cuBLAS TF32 does it in
  **28.75 us**, ~39 us with a fused relu/head-weight epilogue. **15.2x.**
- **A silent context ceiling at ~49k.** `topk_scan_smem(n)` requests ~4n bytes of dynamic shared
  memory with **no `cudaFuncSetAttribute` opt-in anywhere in `kernels/` or `src/`**, against a 48 KiB
  default (`sharedMemPerBlock` 49152, `sharedMemPerBlockOptin` 232448 **[M]**). At T=12288 the launch
  fails and returns garbage. The block-parallel replacement needs no dynamic shared memory, so this
  disappears as a side effect.

---

## 3. Finding 2 — tau falls with GENERATION length, not context, and agentic is the good end

I had repeated `PERF.md`'s "tau falls with depth" without separating the two variables. Regressed
over all 2,157 usable legs **[M]**:

```
corr(tau, prompt_tokens)     = -0.025      <-- null
corr(tau, completion_tokens) = -0.429

completion bin  mean tau      prompt bin (completion held 128-1024)  mean tau
     64-256       3.251            0-256                               3.174
   1024-2048      2.901          256-512                               3.233
   4096-8192      2.648         512-1024                               3.192
                                1024+                                  3.017
```

Monotone inside every task (aime24 3.30→2.71, humaneval 3.52→2.90, math500 3.51→2.82) **[M]**, so it
is not cross-task mixing.

**This is good news for the actual goal.** Agentic coding is **long prompt, short generation** — the
favourable end. `bfcl` at <=512 completion tokens measures **tau = 3.71**, the highest value anywhere
in the corpus, and `NORTH_STAR.md` already records the four fastest categories as long_context 30.77,
agentic_format 29.98, multi_turn 28.97, code_edit 26.81 tok/s against a suite mean of 22.66 **[R]**.
**The math/science battery understates agentic tau; it does not overstate it.**

**Caveat, and it is a real one.** Prompts in this corpus top out at 3,492 tokens (p50 282, p99 2,042)
**[M]**. The prompt-length null is established only to ~3.5k; the agentic target is 8k-100k. The
drafter is SWA-128 plus a pooled hidden state from target layers 40/41/42, so everything beyond 128
tokens reaches it through one vector — the mechanism BudgetDraft measures at a **15-25 % acceptance
penalty** for 512-token windows **[X]**. Unmeasured here, and it is the single largest open risk to
the agentic thesis.

### 3.1 Two premises in the research prompt were wrong

- **"3 chained MTP stages"**: `main_proj`/`main_norm` exist only on `mtp.0`, `markov_head`/
  `confidence_head` only on `mtp.2`. It is **one 3-layer draft backbone run in one forward**, plus a
  Markov head and a confidence head. `config.json` has `num_nextn_predict_layers: 1`. Draft cost is
  one pass, not five — **draft cost is not a lever**.
- **"Verify costs ~3.8x expert bytes at K=4"**: 3.78 is the *independence model*. F64 **measured**
  6.00/9.67/12.58/15.16/17.53 at K=1..5 **[R]** — **2.53x at K=4**, and since routed experts are only
  30.8 % of `B_tok`, the union tax on the whole forward is **1.47x**. Speculation is far cheaper here
  than assumed, and the marginal cost of one more verified position *falls* with M (11.0 → 5.3 ms).

### 3.2 A caution that ties the two findings together

Verify width is currently **free** in the context term, because `k_topk_verify<<<K,32>>>` launches K
blocks of one active thread on 20 SMs, so K<=20 costs the same as K=1. That is fortunate — about a
bug. **Fixing the top-k changes the optimal block size.** Decide block width *after* the kernel fix,
or you will tune to a transient.

**RESOLVED 2026-08-21 by ladder 2.1, and this caution was right — including about the transient.**
Re-tuned after 1.1/1.2 over 32 prompts in one load, the optimum moves **6 -> 5**, worth
**+3.91 +/- 1.65 % tok/s at unchanged `tau` (-0.052 +/- 0.084)**, and everything above 5 is now
closed by measurement (7 -1.70 %, 12 -10.86 %, monotone). Note the SIGN: with the verify no longer
free the optimum moves DOWN, not up. Priced out of 13,392 verify rounds, one more **drafted**
position costs 3.324 +/- 0.281 ms and one more **verified** position 15.184 +/- 0.396 ms, so the
block only pays through positions adaptK actually spends — and at BLK=6 the mean realised verify
width was 3.87 of a ceiling of 7. [`wiki/kernel-optimisations.md` §2.13]

---

## 4. The implementation plan

Ordered by expected value per unit risk. Each step is independently gated: one change per
measurement, bands not points, bit-exactness preserved unless explicitly noted.

### Phase 0 — free, and required before anything is trusted

| # | action | why | risk |
|---|---|---|---|
| 0.1 | **Add `timings` to `/v1/completions`** (staged in `server.cpp`, not yet built) | It returns none today, so every continuation leg — the only decode ever run above 8k — recorded nothing. Decode above ~6.6k is unmeasured. | none; needs one restart |
| 0.2 | **`DSV4_DPROF` at two contexts (512 and 6000), diff the marks** | Splits Term B between `i:topk`, `i:score` and `cattn:sparse`. Predicted signatures: `i:topk` and `i:score` linear in context; `cattn:sparse` saturating at ctx 2048; everything else flat. | none; minutes |
| 0.3 | **`jetson_clocks`** — GPU is at 1386 MHz with 1575 available, EMC 2750 of 4266 **[M]** | Measured in-repo at **+3.0-6.4 %** **[R]** | changes thermals and invalidates cross-run comparisons — **not while the battery runs** |

### Phase 1 — the context term (the large factor)

| # | action | expected | risk |
|---|---|---|---|
| 1.1 | **Use the 32 threads already launched.** Each lane scans T/32, warp-reduce the argmax, repeat. ~29k steps vs 844k. | **~29x on the kernel** for a five-line change | very low — no shared-memory or algorithm change; index set trivially bit-identical |
| 1.2 | **Single-CTA radix select** (3 passes of a 10-bit histogram over the float-as-sortable-uint key, parallel suffix scan over bins, two-sweep compact). Reference: SGLang's `deepseek_v4_topk.cu`, 372 lines, Apache-2.0, and TileLang's `topk_selector.py`. | **26 us at T=6000**, essentially flat in context — the remaining ~70x | low; exactness verifiable per shape |
| 1.3 | **Add the `seq_len <= topk` early-out.** Below context 2048 every row survives and the current kernel still burns 512 x T iterations to discover it. | free win at short context | none |
| 1.4 | **`cudaFuncSetAttribute` opt-in for dynamic shared memory** — or drop it entirely, which 1.2 does | removes the silent ~49k context failure | none |
| 1.5 | **Restructure `index_score` as a GEMM + fused epilogue**: load `kv_t` once into registers, loop heads over `q`, one reduction. | 658 us → ~39 us, **15.2x** | medium — reference is TileLang `fp8_lighting_indexer.py`, not a drop-in. **Use FP32/TF32 accumulation**: an external ablation shows FP16 dropping perfect-recall rows from 99.99 % to 91.82 % on this exact operation **[X]** |

**Do not** build temporal-reuse top-k (GVR/PRR/FlexiCache) yet. A *perfect* prediction that eliminates
the search entirely measures **6.1 us, of which ~4.8 us is the kernel launch floor** **[M]** — so the
whole prize beyond 1.2 is ~20 us per layer. GVR is *slower* than radix select below N=8192 **[X]**,
and our N at 24k context is ~6000.

### Phase 2 — speculation (accuracy-neutral by construction)

| # | action | expected | risk |
|---|---|---|---|
| 2.1 | **Re-tune block width AFTER 1.1/1.2**, not before | with Term B ~0 the optimum returns to M~7-9 from today's apparent 11-13 | none |
| 2.2 | **Retrain the head at gamma=8-10.** F94: the head generalises exactly one position past its trained width **[R]** | +3.6 % (ctx 0) to +17 % (ctx 6.6k) at today's tau | one S5 session |
| 2.3 | **Harvest `(h_40/41/42, p_target)` from live verify forwards** as the S5 corpus | Zero marginal compute — the verifier already computed both. On-policy and distribution-matched *by construction*. Removes the 240-agentic-prompt ceiling and 88 % of capture wall time. | plumbing + disk; guard the self-training feedback loop with a frozen eval suite and versioned shards |
| 2.4 | **Use the confidence head at verify time** (EVICT-style `argmax E[A(T_k)]/C(k)`) — it exists and is unused | +5-10 % over the fixed `adaptK` | low |

**Do not build a tree.** At p2~0.10 the second-best candidate at any position is worth far less than
the admission threshold everywhere short of ~6.6k context. Every production engine agrees: SGLang
auto-selects `eagle-topk=1` for *every* DeepSeek architecture, TensorRT-LLM's PyTorch backend supports
chain-only, vLLM has no tree speculation at all **[X]**. Our own DDTree result already said
"correct but depth-dominated".

### Phase 1b — the bit-exact packing (see `KV_PRECISION_FINDINGS.md`)

**Bigger than first reported: BOTH caches are already quantized and stored in FP32.**
`act_quant_fp8sim` (block 64, dims 0-447) runs at every KV write site and `act_quant_fp4sim`
(block 32, all 128 dims, post-Hadamard) at every index write site. Both compute
`scale = exp2f(ceilf(log2f(...)))` -- an **exact power of two**, losslessly representable in one
UE8M0 byte -- and both write the **dequantized** value back into an FP32 buffer. So the values sit
exactly on the E4M3 and E2M1 grids already, and packing is **bit-exact for the main KV too**, not
just the index.

| row | FP32 today | bit-exact packed |
|---|---|---|
| KV (512 dims), RoPE left FP32 | 2048 B | **711 B (2.88x)** |
| DSA index (128 dims) | 512 B | **68 B (7.53x)** |

This is the format DeepSeek's own tech report specifies (§2.3.4: BF16 RoPE, FP8 elsewhere, FP4
indexer; §5.2.1: FP4 indexer QK gives *"a 2x speedup for the top-k selector, while preserving a
99.7 % recall rate"*), and SGLang ships the DSV4 index pool at exactly 68 B/token. **We already
compute the paper's format and throw the saving away at the store.**

**The prize is capacity, not bandwidth.** seqmax 32768 needs 3.11 GiB today and does not fit;
packed it is 0.87 GiB. seqmax 65536 is 6.21 GiB (impossible) versus **1.74 GiB packed**. `EVALS.md`
records seqmax as *the* binding constraint on which eval items can run at all. Bandwidth-wise the
index path is 389x off its own roofline, so it is not bandwidth-bound and byte reduction buys little
until Phase 1 lands -- `COMPRESSION_PLAYBOOK.md` §0 applied to itself.

**Acceptance test is `memcmp` on generated token ids plus the existing lossless gate, not a
benchmark.** The battery cannot detect a 1-point regression (GPQA CI +/-5.0, AIME +/-15.6 at n=30);
delta=1pt unpaired needs ~20,000 items. At a flip rate of zero the required n is zero.

**One hazard that would otherwise be invisible:** `LOOP_LOG` records a dequant change that produced
a **byte-identical token sequence**, passed the lossless gate, and still collapsed DSpark acceptance
**3.12 -> 1.00** -- acceptance is an exact draft/target token comparison, so perturbed logits break
agreement even when the greedy output does not move. **Apply any format change atomically to target
and draft paths, and report tau in every A/B.**

### Phase 1b-old — the bit-exact FP4 index cache

`kernels/compressor.cu:500` already does this to every row it writes into the DSA index cache:

```c
if (rotate) { hadamard(out, out, groups, d, stream);
              act_quant_fp4sim(out, groups, d, 32, d, stream); }   // indexer compressor
```

**The index cache values are already on the E2M1 grid with a per-32 scale, and are then stored in
FP32.** Storing them as actual FP4 is therefore **bit-exact** — 4.25 bits of information currently
occupying 32. `scripts/build.sh:12` already builds with `-gencode arch=compute_110a,code=sm_110a`,
so the FP4 hardware path is live in the binary today.

That cache is **~64 of the 93 MB** of context-dependent bytes at 24k context, and it is read only to
*select*; it never contributes to an attention output. So this is a **4x cut on the largest
context-dependent component, with no accuracy gate to pass and no numerics change to the ten-row
battery.**

**Order it AFTER Phase 1.** Today the index path is latency-bound on the selection sort, not
bandwidth-bound, so byte reduction buys nothing until the sort is gone. Re-measure before building.

**Do not extend FP4 to the main KV.** That is a genuine accuracy change and the published evidence
is bad in exactly our regime: SGLang measures per-tensor FP4 KV at **-10.7 points AIME25** on
DeepSeek-R1 and **-40.0 points** on GPT-OSS-120B, while gsm8k is untouched — i.e. it is free on short
generations and expensive on long chain-of-thought **[X]**. FP8 is the right stop for the main KV,
and even that is deferred (Phase 3).

**Two traps to design against, both from production bug reports [X]:**
- NVFP4 block scales have **two incompatible layouts** (SM100 4x4 swizzle vs SM120 linear NHD).
  Wrong layout gives correct shapes, correct dtypes, no NaNs — and silently wrong values. Write the
  convention down and gate it with a round-trip test, since we write both producer and consumer.
- A naive per-token dequant loop **destroys the bandwidth win**. On DGX Spark (GB10 — Blackwell,
  aarch64, unified LPDDR5X, the closest public analogue to Thor), llama.cpp `q4_0` KV measured
  **-3.4 % at 8k, -18.5 % at 32k, -35.3 % at 64k** generation, and used *more* RSS than `f16`
  because scale metadata exceeded the compression. The escape is one fused launch reconstructing a
  whole window, not one launch per chunk.

### Phase 3 — deferred, and why

- **BFCL multi-turn** runs *after* Phase 1, because it is decode-bound across many sequential turns
  and a 2x decode win halves it. It is a new row, not a repair.
- **FP8 for the MAIN KV** is deferred to its own gate: it changes numerics, so the ten-row battery
  would no longer describe one engine. If it is done, the literature is unusually clear on how
  **[X]**: quantize K **per-channel** and V **per-token** (getting the axis wrong is a 60-point CoQA
  collapse, far worse than any bit-width choice); quantize K **pre-RoPE** (0.82 PPL free); keep a
  full-precision recent window (worth **+10.18 LongBench points**, the single largest lever in the
  whole method); group size 32-64, never 128. **We already satisfy the window requirement by
  construction** — every layer carries the `WINDOW=128` sliding branch — which is a reason not to
  accept any future scheme that weakens it.
- **Deeper REAP / lower bit-width**: still shelved (`COMPRESSION_PLAYBOOK.md` §0), now with a second
  reason. Term A is 60-68 % of its wall, not 34 %, so byte reduction pays even less than thought.

---

## 5. What is unknown

1. **Decode above ~6.6k context is entirely unmeasured** — `/v1/completions` returns no `timings`.
   Every 8k-24k claim in this document, mine included, is extrapolation. Phase 0.1 fixes it.
2. ~~**The split of Term B** between `i:topk`, `i:score` and `cattn:sparse` has never been measured.
   Phase 0.2 costs minutes and gates the ordering of 1.1 vs 1.5.~~ **CLOSED 2026-08-20 by ladder
   0.4, and all three named items have since been acted on** — `i:topk` (1.2), `i:score` (1.5) and
   `cattn:sparse`, whose cost turned out to be a ~20 ms floor rather than a slope. `b` is down 73 %
   from the 7.220 ms/1000 this document's phase 0 opened on, and what remains of the term is *not*
   attributed to any mark. See `DECODE_LADDER.md` and `wiki/context-scaling.md`; this section is
   the plan as written, kept for the record, not the current state.
3. **Whether tau holds above 3.5k prompt tokens.** The corpus stops there; the agentic target is
   8k-100k. The SWA-128-plus-pooled-hidden drafter is the hedge, and it is untested at that scale.
4. **Why MoE decode universally lands ~30 points below dense** on this hardware. Reproduces across
   llama.cpp, vLLM and this engine, on Thor and on B200. No published mechanism. This is the real
   ceiling on Term A.
5. **The in-situ vs isolation gap for the MLA GEMV** (F125: 225-246 GB/s standalone vs 168-195 in
   situ **[R]**) is the majority of Term A's remaining gap and its mechanism is named nowhere.
6. **Whether the generation-length tau decay is drift or difficulty selection.** Not separable from
   whole-generation records. Log tau per 256-token window within one generation to settle it.
7. **`tcgen05` IS available on `sm_110a`** — verified by assembling
   `tcgen05.mma.cta_group::1.kind::mxf4.block_scale.block32` **[M]**. Note the instruction that does
   *not* work: `mma.sync...kind::mxf4.block_scale` requires sm_120a, so **any FP4 kernel ported from
   DGX Spark or RTX 50 will not assemble on Thor** — the tcgen05+TMEM path is the only one. CUTLASS
   documents SM100 and SM120 and **has no SM110 row**, so prebuilt FP4 kernels lack an instantiation.
   Also: `-arch=sm_110` (no `a`) compiles `cuda_fp4.h` **silently** and emits *zero* hardware
   `cvt.rn.f16x2.e2m1x2` — software emulation with no warning. `build.sh` is correct today; an
   assert would keep it that way. — the earlier "not supported" finding was a target-flag
   error (bare `-arch=sm_110a` also emits a `compute_110` pass, and the error text says `sm_110`).
   Irrelevant to batch-1 decode; it reopens prefill.
8. **Hugepages / TLB behaviour for the expert table** (~22.5M PTEs over 85.7 GiB at 4 KB pages, on a
   part where the GPU shares the system page table). No published measurement anywhere. Cheap.
