# Kernel optimisations — AR and speculative decode

Every optimisation below was **adopted**: built, gated, measured end-to-end on the full model, and
kept. Findings that were built and *rejected* are in [`negative-results.md`](negative-results.md),
which is the longer and more useful list.

Session arc: **base AR 10.30 → 13.83 tok/s (+34.3 %)**, **speculative 16.86 → 22.15 tok/s (+31.4 %)**
— both at context ~9. At the contexts agentic work runs at, the arc that matters is §2.5:
**+24.4 % at ctx 12,282**, the first win here that is a function of context rather than of shape.
See [`context-scaling.md`](context-scaling.md) for why every number above has a context attached now.

---

## 1. The transformations that generalised

Three ideas account for most of the gain. Each was discovered once, then found again in three or
four other kernels — which is the main reason this page is organised by *mechanism* rather than by
kernel.

### 1.1 "Read B once" — amortise a weight read across every row that uses it

**Findings 40, 42, 43, 64.** The naive grouped-GEMV structure reads a weight matrix inside the row
loop, so an expert serving `m` rows re-reads its whole weight matrix `m` times. Hoisting the load
out of the row loop and accumulating `RB` rows against one load turns traffic from
`rows × weight_bytes` into `ceil(rows/RB) × weight_bytes`.

Applied to: `ogroup`, `lm_head`, `gemm_fp32` (F40/42/43), then the MoE grouped GEMV (F64), then the
ogroup **tensor-core** path (F88, prefill).

**The subtlety that cost two cycles: `RB` must follow rows-per-EXPERT, not batch size.**
`acc[RB][BN]` is live regardless of the real row count, so an oversized `RB` buys nothing and costs
occupancy. Measured histogram at K=5: ~70 % of experts serve exactly **one** row, ~88 % serve ≤2,
max 5. F65 shipped `RB` sized from `bs`; F70 corrected it to the measured histogram:

```
capped-at-2 probe:  RB=1 441.9  RB=2 423.8  RB=4 434.4  RB=8 497.3   -> picks RB=2
measured histogram: RB=1 559.1  RB=2 534.7  RB=4 499.8  RB=8 530.6   -> picks RB=4
```

The ranking **inverts** depending on which grouping the probe used. A microbenchmark whose input
distribution does not match production will confidently select the wrong parameter.

### 1.2 Thread-per-output is wrong at decode shapes

**Findings 71, 73.** Kernels written for large batches degrade into near-serial work at M=1.

- `index_score` was one thread per output on a **single block** — 6.05 ms to do 97 k MACs.
  Rewritten warp-per-output: **+4.4 % end-to-end**.
- `k_moe_prefix` and `k_build_tiles` were both `<<<1,1>>>` serial scans over `nr=160`.
  Parallelised: `moe:group` **−20.3 %**.

Both now sit at the **launch-latency floor** (0.24–0.25 ms across 43 layers, against 0.19 ms for a
trivially parallel neighbour), which is the signal that the class is exhausted. Re-opening it needs
a *new* kernel with bad geometry, not another pass over these.

### 1.3 Stage more per barrier

**Findings 74, 78.** The fp8 tile GEMM staged one 128-element K-block per `__syncthreads` pair.
Staging `KC` of them cut the barrier count proportionally: the four-mark GEMM block went **−13.5 %,
bit-identical**, worth **+6.1 % end-to-end** — the largest single adoption of the session.

Double-buffering that staged chunk (F78) was then worth only **+0.28 %**, and the reason is a
four-register occupancy step. That closed the family.

---

## 2. Structural wins

### 2.1 Full-step CUDA graph (F44)

Capture the entire decode step once, replay per token. **92.5 → 79.3 ms/tok** — the largest
structural win in the project's history, and the reason launch overhead is *not* on the open lever
list today.

Corollary from F46: a *verify* graph is worth only **1.05×** (2,788 nodes). The verify has no
launch gap to recover. Launch overhead was a decode problem, not a general one.

### 2.2 The arena (F44, F83)

Per-call `cudaMalloc`/`cudaFree`/`cudaStreamSynchronize` in every sub-function dominates at M=1.
`dscratch.h` replaces them with a bump allocator reset per layer.

**F83 is the cautionary half.** The DSpark *draft* path never got the arena; F82 measured
10.19 ms/round — 7.4 % of the spec cycle — inside raw `cudaMalloc`/`cudaFree` on a drained GPU and
priced the fix at +5–7 %. Built exactly as specified, it returned **+0.41 %**. The measured time was
**host** time overlapping **device** work; the GPU was the critical path throughout. Deleting 134
allocator calls and 10 syncs per round removed real work from the CPU and almost nothing from the
clock.

> **Rule earned:** host time is not recoverable time. Measure what is on the critical path, not what
> is expensive to execute.

### 2.3 Adaptive verify width (F49, F59, F63)

The speculation block length `K` is a decision variable, not a constant. Adapting it to the draft's
margin is worth **+9–11 % where it engages**.

This finding has an instructive history: F49 discovered it was being treated as a constant, F59
claimed 7×, and F63 — re-measured on a *correct* prefill after F62 — showed the 7× was inflated by
the same prefill bug. **A parameter fitted on broken data is fitted to the breakage.**

### 2.4 Fork/join the kv chain (F57)

The kv chain forked off the q chain, serialising two independent computations. Splitting them onto
separate streams recovered the overlap.

### 2.5 Don't recompute a prefix that cannot have changed — cache the main-KV (ladder 1.0, 2026-08-20)

**The first adopted win that is a function of context rather than of shape**, and the largest single
adoption of the programme at the lengths agentic work runs at: **+24.4 % tok/s at ctx 12,282**.

**Mechanism.** The decode loop opened with

```c
for (int st = 0; st < NSTAGE; ++st) dspark_main_kv(mkv[st], main_x, mb[st].attn, ctxlen, EPS);
```

— an `act_quant_fp8` + fp8 block GEMM + `rmsnorm` + `rope` + `act_quant_fp8sim` over **every
position in the context**, three stages, **on every token**. `main_x` is written exactly once per
position (prefill writes `[0,n)`; the accept path writes the `acc+1` committed rows and `cpos`
advances past them — rejected drafts never reach it), so every row below the committed position is
frozen and all but the last `acc+1` rows of that work reproduce bytes already in the buffer.
`dspark_main_kv_upto` computes only rows `[valid, s)` and advances the high-water mark. Per-step
cost goes from **O(context) to O(tokens committed)**; over a generation, from O(n²) to O(n).

**Measured, paired A/B** — same corpus, same binary, `DSV4_MAINKV_CACHE=0` vs default, 6 reps per
point, baseline arm run first so thermal drift penalises the cached arm. On every point at
ctx ≥ 1536 the two arms produced **bit-identical `tau` and mean verify width rep-for-rep**, so each
rep pairs exactly with its twin and the comparison is of the kernel and nothing else:

| ctx | before ms/fwd | after | paired Δ (median) | Δ band | speedup | tok/s |
|---|---|---|---|---|---|---|
| 12,282 | 215.65 | 170.25 | **−45.40** | [−45.9, −38.5] | **1.267×** | 7.67 → 9.54 (**+24.4 %**) |
| 9,213 | 203.51 | 167.27 | −36.24 | [−36.9, −34.3] | 1.217× | 9.69 → 11.79 (+21.7 %) |
| 6,132 | 178.89 | 154.71 | −24.18 | [−24.7, −23.4] | 1.156× | 9.59 → 11.10 (+15.8 %) |
| 3,069 | 160.46 | 148.19 | −12.27 | [−12.7, −12.0] | 1.083× | 10.07 → 10.91 (+8.3 %) |
| 1,536 | 147.67 | 141.15 | −6.52 | [−6.9, −6.2] | 1.046× | 11.54 → 12.07 (+4.5 %) |

The bands are **±1 %** against a 3.5 % run-to-run spread, because pairing removes verify-width
variance instead of averaging over it — see §4.3, this is that rule applied at context.

**It removes the term it was predicted to remove, and 93 % of it.** Regressing the paired saving on
context over 30 of the 31 exactly-paired legs (`sweep-t12288-r3` excluded as a disturbed leg — it is
kept in the medians above, which are robust to it; including it gives `3.190 ± 0.278`, the same
conclusion with a wider band):

```
saving ms/forward = -1.420 + 3.604 x (ctx/1000)      R^2 0.988,  SE(b) 0.076
dprof attribution of draft:main_kv:  3.867 +/- 0.001 ms per 1000
```

**The 2 SE band [3.45, 3.76] does not quite cover the predicted 3.867** — 0.26 ms/1000, 6.8 % of the
term, is unaccounted for and is written down here rather than rounded away. The remaining per-step
main-KV work (`acc+1` rows × 3 stages, plus 15 kernel launches and 3 `dkmalloc`/`dksync`/`dkfree`
triples) is context-*independent* and should land in the intercept, which is where the −1.42 ms
also is. Whatever the residual slope is, it is not that.

**Second-order and free:** the per-step scratch `dkmalloc` of `s*DIM + s*(DIM/128)*4` — **51.8 MB
per stage, 155 MB per step at ctx 12,288**, bumped off a 640 MB arena before the step's first
`arena_reset()` — is now sized by the delta instead, about 12 KB.

**The gate.** Bit-exactness, not approximate equality. Three independent gates, each memcmp'ing the
**whole** `[s, HEAD_DIM]` buffer against the untouched `dspark_main_kv` and aborting on the first
differing float:

| gate | scope | result |
|---|---|---|
| `tests/gate_mainkv_incr.cu` | no checkpoint, 2048 × 512 floats, **22 split points**, both GEMM paths | PASS, byte-identical |
| `build/dsv4-server`, in situ (`DSV4_MAINKV_GATE=1`) | 384 calls, ctx to 12,281, **2,023,320 retained rows**, all three invalidation paths | 0 FAIL |
| `build/decode`, in situ | 320 calls, **568,509 retained rows** | 0 FAIL |

**The gate was proved to have teeth.** Asserting that a gate would catch the bug is not the same as
watching it do so, so both failure modes were reintroduced on a scratch copy and rebuilt:

| negative control | `g_tc_fp8=0` | `g_tc_fp8=1` | floats differing |
|---|---|---|---|
| rope offset not advanced by `r0` | **FAIL** | **FAIL** | 130,624 / 1,048,576 |
| GEMM pin removed | PASS | **FAIL** | **377 / 1,048,576, at the last ulp** |

The second row is the whole argument for two of this gate's design choices. **0.036 % of floats
differing in the last ulp** (0.209164858 vs 0.209164843) is precisely what a tolerance-based check
waves through — memcmp is not pedantry here, it is the only instrument that sees it. And it appears
**only on the tensor-core path**, which is the one decode actually uses; a gate that ran only the
default `g_tc_fp8=0` would have reported a clean pass on a broken kernel. Note also that both
controls put the first differing float at column 448 = `NOPE_DIM`, which is mechanism rather than
coincidence: `act_quant_fp8sim` re-quantises columns `[0, NOPE_DIM)` and absorbs sub-ulp
differences, while the rope half above it stays fp32 and preserves them. **The NOPE half hides small
errors; look in the rope half first.**

**The load-bearing detail is the GEMM pin.** `fp8_block_gemm`'s dispatch is **M-dependent**: M=1
takes a GEMV, M∈[2,8] can take an NVFP4 overlay, larger M reaches `tc_fp8_gemm`. The incremental
delta is 1–6 rows where the from-scratch call was thousands, so going through that dispatch would
compare a GEMV against a tensor-core tile and lose bit-exactness *for a reason that has nothing to
do with caching*. The incremental path pins to `tc_fp8_gemm`, whose own dispatch depends only on N
and K — M only sets `grid.y`, and each output element accumulates over K in the same order
regardless of which 16-row tile it lands in — which makes the result independent of the split
point. The other trap is the rope offset: `cosT/sinT` must be advanced by `r0`, or a sub-range
rotates every row by the wrong angle. Both are covered by the 22 split points.

**Invalidation is the caller's job, and there are exactly three callers that need it**:
`prefill_full` (memsets `main_x`), `extend` (rewrites `[from,n)`), `rewind_to` (drops to `n`). Each
clamps the high-water mark down; everything else may only grow it.

---

### 2.6 A selection sort is not a top-k — single-CTA radix select (ladder 1.2, 2026-08-20)

**+8.2 % tok/s at ctx 12,410**, bit-exact, `tau` unchanged, and the second consecutive adoption that
is a function of context rather than of shape.

**Mechanism.** Four kernels — `k_topk_verify` and `k_topk_decode` (spec-decode verify and draft),
`k_topk_masked` (the CUDA-graph base-AR path) and `k_topk_offset` (prefill) — selected the DSA
indexer's top 512 compressed rows by running 512 **sequential** argmax rounds over the score row,
marking the winner `-1e30f` between rounds. §1.2's fix spread each *round* across the 32 lanes that
were already launched (14–28×, F71); the rounds themselves stayed serial, so at `topk=512` and
`T=3072` (ctx 12,288, ratio 4) the kernel was still 512 dependent rounds of 96 strided loads.
0.4 measured the survivor at **13.47 ms at ctx 12,288 — 12.5 % of the whole context term.**

The replacement (`include/topk_radix.h`) is O(T) work in a constant number of passes: an MSB-first
8-bit radix select, one gather, one bitonic sort of the ≤512 winners. Standalone, same block, same
input:

| T (= ctx/4) | 512 | 1,648 | 2,048 | **3,072 (ctx 12,288)** | 4,096 | 6,000 | 8,192 |
|---|---|---|---|---|---|---|---|
| warp scan (§1.2) µs | 258 | 502 | 459 | **592** | 725 | 1,076 | 1,259 |
| radix select µs | 18.4 | 24.6 | 24.5 | **32.8** | 37.3 | 35.6 | 37.0 |
| speedup | 14.1× | 20.4× | 18.7× | **18.0×** | 19.4× | 30.2× | **34.1×** |

**Bit-exactness is the design, not a property checked afterwards.** `sparse_attn` sums the selected
rows *in order*, so fp32 association makes the ORDER load-bearing and not just the set. Four things
carry it, and each has its own gate distribution:

1. **The composite key** `comp(v,t) = (ord(v) << 32) | ~t`, `ord` the standard order-preserving
   float→uint32 map. Sorting `comp` descending *is* (value descending, index ascending) — exactly
   what a serial ascending scan with a strict `>` produces. And because `~t` makes every composite
   distinct, **the select has no ties to break**: exactly `k_eff` elements satisfy
   `comp >= threshold`, always, so there is no equal-key special case to get wrong.
2. **Admission is the original's float compare** `v > floorv`, never a key-space test. `ord(NaN)`
   is larger than every finite key, but `NaN > best` is false — a key-space test would select
   something the original cannot.
3. **Signed zero is canonicalised.** `-0.0f == +0.0f` for the original's `>`, so they tie and the
   lower index wins; but `ord(-0.0) < ord(+0.0)` as raw bits. `index_score` can emit `-0.0` (relu
   gives exactly 0, times a negative head weight).
4. **The sentinel convention stays at the caller.** The selector returns raw source indices or −1;
   each caller applies its own offset/threshold rule, including `k_topk_offset`'s deliberate
   asymmetry (scan the full row, reject out-of-range picks only at the output, still consuming a
   slot).

**Measured in situ, paired A/B**, same corpus and binary, `DSV4_TOPK_RADIX=0` vs default, 6 reps per
point, baseline arm first so thermal drift penalises the radix arm. 35 of 52 legs paired exactly
(identical `tau` *and* identical mean verify width rep-for-rep); all six reps pair at every point at
ctx ≥ 1,664, and the unpaired legs are ctx 128/384 and the two controls — the known
non-reproducibility of [`measurement-and-traps.md` §12](measurement-and-traps.md), not this change.

| ctx | before ms/fwd | after | paired Δ | Δ band | speedup | tok/s |
|---|---|---|---|---|---|---|
| 12,410 | 167.18 | 154.66 | **−12.55** | [−12.92, −12.33] | **1.081×** | 9.88 → 10.69 (**+8.2 %**) |
| 9,341 | 166.77 | 156.54 | −10.47 | [−10.66, −10.15] | 1.065× | 11.81 → 12.60 (+6.7 %) |
| 6,260 | 154.62 | 146.23 | −8.58 | [−8.65, −8.30] | 1.057× | 11.12 → 11.77 (+5.8 %) |
| 3,197 | 147.85 | 141.23 | −6.72 | [−7.01, −6.28] | 1.047× | 10.93 → 11.46 (+4.8 %) |
| 1,664 | 140.91 | 136.41 | −4.53 | [−5.07, −4.40] | 1.033× | 12.08 → 12.48 (+3.3 %) |
| 889 | 132.80 | 130.30 | −2.34 | [−2.84, −2.18] | 1.019× | 12.95 → 13.19 (+1.9 %) |

**It removes the term it was predicted to remove, 91 % of it, plus a term nobody could have
predicted.** Regressing the paired saving on context over all 35 exactly-paired legs:

```
saving ms/forward = 3.122 + 0.793 x (ctx/1000)      R^2 0.951,  SE(b) 0.031
dprof attribution of i:topk:                        0.872 +/- 0.021 ms per 1000
```

The 2 SE band **[0.730, 0.856] does not cover 0.872** — 0.079 ms/1000, 9 % of the term, is
unaccounted for and is written down rather than rounded into agreement. The **3.12 ms
context-independent** saving is the interesting half: `i:topk` marks only `k_topk_verify`, and
`k_topk_decode` on the draft side **has never had a dprof mark at all** — the same blind spot that
hid `draft:main_kv` until 0.4 went looking. A win that arrives in the intercept is a hint that the
profile is still incomplete.

**The gate.** Bit-exactness on the whole index array, not the same set:

| gate | scope | result |
|---|---|---|
| `tests/gate_topk_radix.cu` | no checkpoint, 4 kernel shapes × 13 lengths × **6 distributions** — exact ties, signed zeros, floor-straddling rows, all-negative | PASS |
| ^ under `compute-sanitizer --tool memcheck` | same | 0 errors |
| `build/dsv4-server` in situ (`DSV4_TOPK_GATE=1`) | **11,008 calls, ctx to 12,282, 183,179,703 index slots**, prefill + extend + rewind | 0 FAIL |
| standing GATE + LOSSLESS gate (`build/decode`) | every decode run | PASS, τ 2.87 tok/verify |

The in-situ gate launches the **untouched** warp kernel with identical arguments into a private
buffer and memcmps the entire `[K, topkc]` array, aborting on the first differing slot. It is not
armed on the `_dp` graph path, which is CUDA-graph captured and cannot contain a host sync; that
kernel is covered by the unit gate and by the LOSSLESS gate on every `build/decode` run.

**Free side effect.** The originals asked for ~4T bytes of *dynamic* shared memory against a
48 KiB default limit, and above ~49k context the launch silently failed and returned garbage
(ladder 1.4). The radix select uses ~7 KiB of **static** shared memory whatever T is, so the
shipped path cannot hit that ceiling. 1.4 stays open for the fallback arm, which still asks.

**Two things measured and not shipped**, both in [`negative-results.md`](negative-results.md): a
`__match_any_sync` warp-aggregated histogram (slower — 43.0 vs 39.0 µs at T=3072), and block sizes
128/256/1024 (53.3 / 38.9 / 31.6 µs at T=3072 against 512's 32.0).

---

## 3. Precision and layout

| finding | change | result |
|---|---|---|
| F41 | m16 tile fetched every weight sector for half its payload | fixed; prerequisite for everything after |
| F66 | expert pointers are misaligned by a **constant 8 bytes** (43,470 of 44,436 tensors at `data_offset%16==8`) | the funnel shift was buying alignment that was already there |
| F72 | so: two `uint2` loads instead of two `uint4` + funnel shift | same instruction count, **half the bytes requested**, +1.4 % |
| F56 | the shared expert was never dependent on the routed ones | run them concurrently |

F66 → F72 is the cleanest example of the project's method: **measure the property you are
compensating for before writing the compensation.**

---

## 4. What every adopted change has in common

1. **A gate proved it before a benchmark did.** `gate_units`, `gate_grouped_moe`, `gate_ogroup_gemv`,
   `gate_prefill_len`, `gate_compressed_decode` and friends run against a PyTorch oracle. Several
   optimisations are *bit-identical* by construction and gated with `memcmp`.
2. **It was measured end-to-end on the full model**, not in `gemm_bench`. The bench overstates —
   F47 quantified this, and F76 is the extreme case: a free, bit-identical, `gemm_bench`-verified
   −7…−15 % on the kernel was **+0.1 %** in situ, because the kernel is latency-bound.
3. **The measurement was paired.** Nine spec verifies pairing 1:1 at identical `K` and identical
   accept counts is the instrument; end-to-end tok/s alone cannot resolve sub-1 % effects.

See [`measurement-and-traps.md`](measurement-and-traps.md) for why each of those is non-negotiable.
