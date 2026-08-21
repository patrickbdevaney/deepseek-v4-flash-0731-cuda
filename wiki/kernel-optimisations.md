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

**And every one of those five call sites tiled M only (ladder 1.12).** "Read B once" as written
above is a 1-D tile: `RB` rows of A against **one** row of B, so B traffic falls by `RB` and A
traffic does not fall at all. Giving the warp `NN` rows of B as well makes the traffic
`M*N*K*(1/NN + 1/MM)` instead of `M*N*K*(1/1 + 1/MM)`, and on `gemm_fp32`'s largest call site that
is 1.125 → 0.375 for the price of `MM*NN*4` registers. §2.11 has the sweep, the point at which it
stops paying, and the reason the guard below M=8 is not optional.

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

**2.4b The compressor fork (F55/F56), re-measured with a band by ladder 1.8, 2026-08-20.** The two
`compressor_emit_group` calls read `x_full` and nothing else, so they never depended on `build_qKV`
— they were merely issued after it on the same stream. Forking them onto `g_side` is the same
transformation as F57, one level up. F56 adopted it on a single-load pair (K=5 verify 155.22 →
153.48 ms, spec 17.32 → 17.48 tok/s); 1.8 measured it **paired on the current engine, SERIAL arm
first so drift penalises the fork**:

| | value |
|---|---|
| paired saving | **0.81 ms/forward**, sd 0.14, **2 SE band [0.72, 0.90]** |
| legs faster under the fork | **9 of 9** |
| token identity | **9/9 byte-identical**, `tau` and realised width equal to 3 d.p. on every leg |
| ctx range | 3,069 – 12,282, three points × three reps, one binary, `NO_ATTN_SPLIT=1` is the other arm |

**And it says what the fork does NOT do.** The emit costs **8.31 ms** serially at fixed VB=2 and
**7.02 ms** overlapped, so the fork hides **16–17 % of it**; the other 84 % is compressor traffic
that must move. Amortised over the 64.9 % of forwards that carry an emit that is **4.56 ms/forward
at 52 % of its 880 MB byte roofline**, which is the first price anything has put on the
compressed-KV emit. Two consequences are on the ladder as 1.11 (defer the join past `i:qidx`/`i:iw`,
1.99 ms of independent work currently sitting on the wrong side of it) and 1.12 (the ratio-128 emit
re-reads its weights 32×). The trap this measurement walked into on the way —
`cattn:q_proj` reporting a 3.2× swing that was the *other stream's* work — is
[`measurement-and-traps.md` §29](measurement-and-traps.md).

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
shipped path cannot hit that ceiling. **1.4 CLOSED 2026-08-20** for the fallback arm and the in-situ
reference, which still ask: they now opt in via `cudaFuncSetAttribute` and abort rather than issue a
launch that would fail silently. The item moved no throughput number and could not have — all three
touched translation units compile to byte-identical SASS, so the change is host-side launch
configuration only. What it bought is that the two arms which certify every *future* top-k change
now work above the context they were previously only ever run below: the engine's own
`compressed_decode_step_indexer` is gated at contexts 49,207 and 200,003
(`scripts/gate_topk_smem_ctx.sh`), and the defect is reproducible in the same binary under
`DSV4_TOPK_SMEM_OPTIN=0`. See [`measurement-and-traps.md` §17–18](measurement-and-traps.md).

**Two things measured and not shipped**, both in [`negative-results.md`](negative-results.md): a
`__match_any_sync` warp-aggregated histogram (slower — 43.0 vs 39.0 µs at T=3072), and block sizes
128/256/1024 (53.3 / 38.9 / 31.6 µs at T=3072 against 512's 32.0).

**One thing shipped inside this kernel and worth nothing** (ladder 1.3, 2026-08-20): a `lim <= topk`
early-out that skips the threshold search entirely below ctx 2048, bit-exact and worth +2.1 to
+4.1 µs per call standalone — **and 0.07 % of a forward in situ, which no end-to-end instrument
here can resolve.** It is default-on because it is strictly less work, but it is not a win and it is
written up in [`negative-results.md` §4c](negative-results.md), because §2.6 above had already taken
the money it was aimed at.

### 2.7 Change which kernel the bit-exactness claim is *against* — `index_score` as a register-tiled GEMM (ladder 1.5, 2026-08-20)

**Mechanism.** `index_score[s,t] = sum_h relu(q[s,h,:] . kv[t,:]) * w[s,h]` is what the DSA indexer
scores every compressed row with, once per ratio-4 layer, every forward. The shipped
`index_score_warp_kernel` put one warp on each (query, row) pair and read **both** operands from
global on every head: `q` for one query is `H*d = 64*128 = 8192` floats = 32 KiB and is re-read once
per row `t`; `kv[t]` is 128 floats and is re-read once per **head**. At the verify shape
(S=6, T=3072) that is **1.2 GB moved to do 151 M MACs — 0.5 FLOP/byte** on a part whose FFMA peak is
5.45 TFLOPS, and `d` is a runtime argument so the inner loop is a serial chain of dependent global
loads that cannot unroll.

The replacement is the same arithmetic as a GEMM: `P[(s,h),t] = q . kv^T`, M = H = 64, N = T,
K = d = 128, with the reduction over M in the epilogue. One block owns **all 64 heads of one query**
and 128 rows of `kv`; the k-loop is 8x8 register-tiled, so 16 shared loads feed 64 FFMAs (~4:1,
against the warp kernel's 1:1); `P` for the tile goes to shared and the epilogue walks `h` in order.
The register tiling is **strided, not blocked** (`t = tx + j*16`, `h = ty + i*8`) so consecutive
threads read consecutive shared floats with plain scalar loads and no bank conflict — no `float4`
and no padding games.

**THE WIN CAME FROM CHANGING WHICH KERNEL THE CLAIM WAS AGAINST.** An intermediate
`index_score_tiled_kernel` was built first, staging `q` in shared and holding `kv[t]` in registers,
and it is bit-identical to the shipped warp kernel — which capped it at **2.0x**. Bit-exactness with
the warp kernel mandates keeping its 5-step `__shfl_down_sync` tree, and SHFL retires at one
warp-instruction per SM per clock here: 1.18 M (row, head) pairs x 5 steps over 20 SMs is a ~200 us
floor at the verify shape *no matter how the operands are staged*. The GEMM instead is bit-identical
to `index_score_kernel`, the correctness-first **scalar reference** that `gate_units` checks against
`ref/goldens/unit_index_score.safetensors`, because a register-tiled GEMM accumulates serially in k
and that IS the reference's order. LOOP_LOG Finding 68 had adopted the warp kernel as a deviation
*from* that reference behind the LOSSLESS gate; 1.5 spends the deviation back and is 6.8x rather
than 2.0x for it. Full argument in [`negative-results.md` §4d](negative-results.md).

**Measured gain, standalone** — 30 repeats, arms interleaved in one process, H=64 d=128, median us:

| S, T | warp | tiled | GEMM | |
|---|---|---|---|---|
| 6, 768 | 239.7 | 138.3 | **45.0** | 5.33x |
| 6, 1536 | 462.9 | 237.2 | **72.6** | 6.37x |
| 6, 3072 | 898.7 | 449.7 | **132.5** | 6.78x |
| 6, 6144 | 1771.6 | 877.8 | **237.6** | 7.46x |
| 1, 6144 | 311.7 | 156.7 | **53.9** | 5.78x |

**Measured gain, in situ** — one server load per arm, control arm first so drift favours the
control, four reps per sweep context:

| ctx | tok/s before | tok/s after | paired | `tau` |
|---|---|---|---|---|
| 3,197 | 12.14 | **12.27** | +0.99 % | 1.736 both |
| 6,260 | 11.87 | **12.15** | +2.24 % | 1.788 both |
| 12,410 | 10.46 | **10.93** | **+4.57 %** | 1.615 both |

Regressing the 16 paired per-leg forward-time deltas on context gives
**-0.572 +/- 0.018 ms per 1000 context** (R^2 0.987), which is **89 % of the 0.644 +/- 0.018 that
ladder item 0.4 attributed to this mark** — prediction and delivery agree inside one SE. The
intercept is **+0.358 +/- 0.134 ms**: at zero context the GEMM is slightly *slower*, because it
stages 33 KiB of shared per block that needs rows to amortise over. Break-even is near ctx 625, and
that is visible as a **null** on `build/decode` at seqmax 85 — 21.03-21.24 tok/s against the warp
kernel's 21.02-21.26, three replicates each in one load per arm.

**The gate that proved it.** The claim is value equality, so the instrument is `memcmp`:
`tests/gate_index_score` sweeps 1130 shapes (S ∈ {1,2,5,6,17,129}, T from 1 to 6001, H ∈ {1,3,64},
d ∈ {32,64,96,128,256}, five input distributions including all-negative, near-tie and
signed-zero/denormal, output poisoned to `0xEE` before every launch) — **21,773,760 floats, 0
differing** on both claims. It also **prints what the change spends** against the shipped warp
kernel rather than asserting it is small: up to 88 % of elements differ, `max_abs` 4.88e-04,
`max_rel` 0.847 on near-zero scores. Those numbers are meaningless on their own, because this
kernel feeds a **selection** and a last-ulp flip near the k-th boundary changes which rows attention
sees. So the evidence that counts is downstream and at context: **16 of 16 legs emitted
byte-identical text at ctx up to 12,410, with `tau` and mean verify width identical to four
decimals**, plus `LOSSLESS GATE -> PASS` x3 and `gate_units` / `gate_indexer_decode` /
`gate_compressed_decode` / `gate_prefill_len` all green. See
[`context-scaling.md`](context-scaling.md) for what it did to the term.

### 2.8 The one adopted win that is not a kernel — actually serve the head you trained (ladder 2.2, 2026-08-20)

**+9.52 % on the frozen 8-prompt suite, `tau` 3.5362 -> 3.8438, with no code on the request path
changed at all.** The `s3` draft head was promoted on 2026-08-12 and had never been loaded by the
server: `tools/promote_head.py` archives and records but deliberately never writes the live
checkpoint, and `scripts/serve.sh` — which every server launcher in the repo execs — hardcoded the
base checkpoint path. The promotion pipeline terminated one step short of having an effect.

Mechanism: `scripts/stage_head.sh` builds a **symlink farm** beside the checkpoint — 45 shards and
the tokenizer linked to the read-only base at the same inode, the three `mtp.*` shards linked to the
archived head — and `config/live_ckpt`, a tracked one-line file, tells `serve.sh` which checkpoint to
load. A `--head DIR` flag on the server would have been the obvious alternative and is the wrong one:
`src/engine.cu` loads the embedded head out of the main `WeightStore` precisely because a second
store "would duplicate ~6.5 GiB against ~16 GiB of headroom", and a flag *adds* a mapping where
staging *replaces* one. Staging also costs no disk and keeps the binary bit-identical across the A/B.

The gate that proved it: both heads re-measured back to back on today's engine (the archived numbers
predate five ladder items and are not an admissible before-arm), frozen protocol, LOSSLESS and
first-token gates PASS on both arms, base AR within 0.70 % between the loads as a drift control. The
suite `tau` of **both** arms reproduced `HEAD_REGISTRY.md` to four decimal places while their tok/s
did not — see [`measurement-and-traps.md` §22](measurement-and-traps.md). The win is concentrated in
the prompts the shipped head was worst at: no suite prompt is below `tau` 2 any more, spread across
the suite falls 32.8 %, and prompt 7 goes 14.39 -> 26.33 tok/s. Deployment confirmed live —
`tokens_per_verify` 2.857 from the running server on the gate prompt, against the shipped head's
3.61 offline. Full write-up in
[`draft-head-finetuning.md` §8](draft-head-finetuning.md).

### 2.9 `sparse_attn` — stage the gathered row in shared memory, and vectorise the load without losing bit-exactness (ladder 1.7, 2026-08-20)

**+2.4–3.3 % tok/s across ctx 1,656–12,410, paired saving 4.227 ± 0.121 ms per forward over 16
legs, every leg faster, all 16 byte-identical.** The first item on this ladder to move **Term A**:
`a` 129.11 → 125.11 ms (1.571× → 1.522× the 82.18 ms byte floor).

**Mechanism — and the hypothesis it refuted.** `cattn:sparse` is ~20 ms of every forward above
ctx 2048. Both the ladder entry and B9's own comment in `kernels/mla_attn.cu` blamed **reuse**:
`num_key_value_heads == 1`, so `topk_idxs` is indexed by `(b, m)` and *not* by head, and all 64
heads of a query gather the identical latent KV rows — 63/64ths of the traffic redundant. The fix
for that is `hpb`, putting HPB heads of one query in one block so warps 2..HPB hit L1 instead of
L2. It already existed. **It is a measured null: 1.00× at hpb=2 and hpb=4, at every shape.** L1 was
already catching the reuse.

The binding constraint is **instruction issue**. Every warp issued 32 scalar loads per gathered row
— 16 for the dot product, 16 more for the accumulate — perfectly coalesced, so never a byte
problem, on a kernel with 3–6 warps per SM and a 5-deep `__shfl_down_sync` chain on the critical
path with nothing to hide them behind. Three changes, all value-preserving:

1. **The block stages the row into shared memory once per `t`**, with `float4` loads and double
   buffering, so the fetch for `t+1` is issued before `t`'s dot product. 4 vector loads per *block*
   replace 32 scalar loads per *warp*.
2. **`KREG` holds the row in registers across both consumers.** `__shfl_sync` sits between the dot
   product and the accumulate and blocks the compiler from reusing the first set of loads, so the
   row was being read from shared memory twice.
3. **`corr = expf(run_max − new_max)` is skipped when the running max did not move.** `expf(0.f)`
   is exactly `1.f`, so this is a value-preserving branch and not an approximation, and it is
   warp-uniform because `score` is the lane-0 broadcast.

**Why it is bit-exact, which is the part worth stealing.** Vectorising the *global* load normally
destroys bit-exactness: `float4` changes which lane owns which element and therefore the shape of
the partial-sum tree. **Staging through shared memory decouples the two.** The load pattern becomes
a pure memory-*placement* decision, while the reduction still sees lane `l` holding
`{l, l+32, …, l+480}`, the same 5-step shuffle tree, and the same online-softmax update order over
`t`. Every float operation, in every order, is the one the shipped kernel performed. This is the
opposite of §2.7's situation, where the speed required accepting a reassociation.

**The kernel band** (`gate_sparse_hpb`, at the shapes the engine issues, memcmp-clean against the
pre-1.7 launch):

| shape | pre-1.7 | hpb=8/smem=0 *(the old default)* | **hpb=4/smem=2** |
|---|---|---|---|
| m=1 topk=640 (base AR) | 0.737 ms | 0.771 (0.96×) | **0.506 (1.46×)** |
| m=2 topk=640 (mean verify) | 0.746 | 0.773 (0.97×) | **0.548 (1.36×)** |
| m=6 topk=640 (max verify) | 0.818 | 1.581 (0.52×) | **0.719 (1.14×)** |
| m=1022 topk=1277 (prefill) | 263.2 | 328.5 (0.80×) | 220.4 (1.19×) — *smem=1 is 205.1 (1.28×)* |

`KREG` wins where the kernel is latency-bound with few blocks and loses where it is
occupancy-bound with thousands, which is why the default switches on `total` and not on a constant:
`hpb=4`, `smem=2` below 1024 warps, `smem=1` above.

**The gate that proved it.** `tests/gate_sparse_hpb.cu` — memcmp of the **entire output buffer** of
every `(hpb, smem)` launch against the pre-1.7 launch at six engine shapes, 0 differing bytes, plus
a one-ulp negative control per shape that must fail (it caught the gate being blind at topk=320 on
its first run, because the row it perturbed was not in that shape's gathered set). Then `gate_units`,
`gate_indexer_decode`, `gate_compressed_decode`, `gate_prefill_len`, `gate_compressed_graph` and
`gate_indexer_graph` all at `maxabs = 0.00e+00`; `LOSSLESS GATE -> PASS` ×3; both `build/decode`
arms generating the identical ids; and **16 of 16 server legs emitting a byte-identical
`text_sha256` with `tau` equal to four decimals at ctx up to 12,282**. A cosine gate would have
passed a reordering here — `sparse_attn` sums the gathered rows *in order* under a non-associative
online softmax — which is why the instrument is memcmp, as in §2.5 and §2.6.

**dprof isolates it.** `cattn:sparse` 18.66 / 19.38 / 20.57 → **15.27 / 14.48 / 15.38** ms at ctx
3072 / 6144 / 12,288; the parent `ATTENTION` falls by 5.75 against the mark's 5.19 at ctx 12,288,
and `i:score`, `i:topk`, `o:rope` and every `moe:*` mark are unchanged — no work was relocated into
an unmarked region. The before-arm also **reproduces 0.4's original 19.22 / 19.90 / 21.17**, four
iterations and four kernel changes later.

See [`measurement-and-traps.md` §28](measurement-and-traps.md) for the old default being a
regression at its own design point, [`negative-results.md` §4g](negative-results.md) for the reuse
hypothesis, [`context-scaling.md`](context-scaling.md) for the 6 % of this that is Term B, and
[`prefill-optimisation.md` §7](prefill-optimisation.md) for what it gives prefill back.

---

### 2.10 Join a side stream where the data is needed, not where it was forked (ladder 1.11, 2026-08-20)

**Mechanism.** `compressed_verify_step_indexer` forks the two `compressor_emit_group` calls onto
`g_side` (§2.4's pattern, Finding 55/56) and then joined them straight back, immediately after
`build_qKV`. The join position was inherited, not chosen. **The first consumer of `idx_ckv` is
`index_score` and of `comp_kv` is the `kv_all` copy**, and the two marks in between — `i:qidx`
(a GEMM on `iqrq/iqrs`) and `i:iw` (a GEMM on `x_cur`) — read neither. 1.8 measured that window at
1.94 + 0.64 = **2.58 ms of independent main-stream work sitting behind a barrier that had no reason
to be in front of it**.

The fix keeps the `cudaEventRecord` where the emits end and moves only the `cudaStreamWaitEvent`:

    if(asplit) cudaEventRecord(g_side_join,cs);                 // "the emits are done"
    if(asplit && !jdefer) cudaStreamWaitEvent(stream,g_side_join,0);
    ...  i:qidx, i:iw ...
    if(jdefer) cudaStreamWaitEvent(stream,g_side_join,0);       // at index_score, the true dependency

**Measured gain.** Drift-free paired over **18 legs and four checkpoint loads**:
**−0.542 ± 0.310 ms/forward**, and **−1.092 ± 0.146 at ctx 12,410**, where it is 6/6 in each arm
order independently (+0.77 % tok/s there). The mark-level instrument agrees on the mechanism at
K=5: `cattn:compress` 2.34 → **0.07** ms as the barrier leaves it, `i:qidx` 1.94 → **2.81** as it
absorbs the emit traffic, verify TOTAL 127.33 → **125.89**; −1.44 ms on a step carrying one emit ×
1.8's 64.9 % emit rate predicts −0.93 ms/forward.

**The gate that proved it.** `tests/gate_join_defer.cu` — and it had to be written, because **every
existing gate that links these kernels does so without calling `arena_init()`, so `g_side` is null,
`asplit` is false, and the fork/join has never been under test at all.** It calls `arena_init()`
first and refuses to PASS if `g_side` comes back null. Both join positions in one process on
identical weights, four cases (1 emit, 2 emits, 0 emits, second shape): **0 of 143,360 floats
differ**, same under `--swap`, `--negctl` fires on every case. Engine: identical generated ids,
LOSSLESS PASS in both arms, **44 of 44 sweep legs byte-identical with `tau` equal to three
decimals**.

**What generalises.** §2.4 established *that* independent chains belong on two streams. 1.11 is the
second-order version: **the join is a free parameter and the default is the wrong one.** A fork
written as `fork; workA; join; workB; consume` costs nothing to rewrite as `fork; workA; workB;
join; consume` when `workB` does not read the fork's output, and the compiler will never do it for
you because the dependency lives in your head, not in the types. Every fork site in this repo —
`g_side` here, `g_side2` in `build_qKV`, `moe.cu`'s shared expert — is worth re-reading with the
question "what is the *first* thing that actually reads this?".

**And read the marks with care after doing it.** Deferring a join MOVES time between marks rather
than removing it: `cattn:compress` fell 97 % and that is not the win, it is the barrier leaving.
Only the conserved sum, and then the paired ms/forward, decides. See
[`measurement-and-traps.md` §29](measurement-and-traps.md) for the same effect misread as a 3.2×
swing in a GEMM, and **§33 for why one arm order could not resolve this item at all**.

### 2.11 One warp, a 2-D tile — `gemm_fp32` at the compressor emit (ladder 1.12, 2026-08-20)

> **CAVEAT ADDED 2026-08-21 (ladder 1.10/1.15).** The engine band below was measured across four
> `dsv4-server` loads of which **no two are byte-identical** — the aliased `hadamard` race was firing
> in every one of them, and 1.12 changing `gemm_fp32` is what started it (§36). The arms did not
> generate the same text, so the legs were not the same work and the SIZE of this saving is
> **unproven**, not wrong. Its BIT-EXACTNESS is now confirmed independently and more strongly than
> when it landed: with the race fixed, the engine reproduces 1.11's four saved loads leg for leg,
> which is only possible if this tile changed no value. Re-measured by ladder **1.15**.

**Mechanism.** §1.1's "read B once" was applied to `gemm_fp32` as a 1-D tile: one warp per
(8-row chunk of A, one row of B), with the chunk loop on the HOST. `compressor_emit_group` calls it
at `M = ratio = 128, N = d = 512, K = DIM = 4096` on the twenty ratio-128 layers, so `B =
[512,4096]` f32 = 8.39 MB was walked **sixteen times, in sixteen separate launches**, twice per
layer.

A 1-D tile is only half the transformation. A warp owning `MM` rows of A **and `NN` rows of B**
issues `MM + NN` loads for `MM*NN*4` FMAs, so total operand traffic is `M*N*K*(1/NN + 1/MM)` floats:

| tile | traffic factor | `[128,512]x[512,4096]` |
|---|---|---|
| pre-1.12: 8-row chunks, host loop | 1.125 | 0.5744 ms |
| `(8,1)` — the same tile, `gridDim.y` instead of 16 launches | 1.125 | 0.4472 ms — **1.28× from launches alone** |
| `(8,2)` | 0.625 | 0.2864 |
| **`(4,4)` — adopted** | **0.375** | **0.2477 ms, 2.32×** |
| `(6,6)` | 0.333 | 0.2849 — *lower traffic, slower* |
| `(4,8)`, `(8,4)` | 0.375 | 0.2830, 0.2937 |
| `(16,1)`, `(32,1)` | 1.06, 1.03 | 0.4649, 0.5016 — **slower than the baseline** |

Seventeen tiles, four engine shapes, `tools/f32mk_bench.cu`. `(6,6)` moving *less* traffic and
running *slower* than `(4,4)` is the boundary of the transformation: past ~128 accumulator registers
the kernel is register- and issue-bound, and buying traffic with occupancy stops paying.

**The tile is OFF below M=8, and that guard is load-bearing.** The indexer's `idx_weights_proj`
GEMV is `[5,64]x[64,4096]` and fires **21× on every forward**, against the emit shapes' once every 4
or 128 positions. At `(4,4)` it becomes a 4-row chunk plus a 1-row tail — two launches where the
legacy path templates one — and goes `0.0148 → 0.0205 ms`. Unguarded, the most frequent call site
would have paid for the rarest one.

**Measured gain — and the engine beat the bench 2:1.** The microbenchmark reuses one 8.39 MB `B`
across 200 reps, so 15 of its 16 passes are L2 hits on a 32 MB L2. The engine walks **20 layers × 2
matrices = 335 MB of distinct weights**, so the passes removed are DRAM. Predicted −13.07 ms per
emit step; measured **−26.26**.

| `tools/emit_spike.py`, pooled over four checkpoint loads | n | before | after | |
|---|---|---|---|---|
| `cattn:compress`, steps whose block contains a ratio-128 boundary | 69 | **40.76** [39.92, 45.22] | **14.50** [14.29, 14.64] | **−26.26 ms, 2.82×**, populations disjoint |
| verify `TOTAL` excess on those steps | 69 | +63.17 | **+34.26** | **−28.91 ms** |
| verify `TOTAL` excess on ratio-4 boundary steps (62.8 % of steps) | 1811 | +13.52 | **+11.03** | **−2.49 ms** |
| `cattn:q_proj` excess on ratio-4 boundary steps | 1811 | +4.61 | **+3.92** | **−0.69 ms** |

The last row is §2.4/§2.10's side stream showing up as a *second-order* effect: the ratio-4 emits
run on `g_side`, 1.8 established that their contention lands in `cattn:q_proj`, and making them
1.79–1.87× faster gives 0.69 ms of that back on nearly two steps in three. **That row, not the
40 ms spike, is where the throughput comes from** — and it is where the item's own ranking
("LOW on throughput") was wrong.

End to end, drift-free over 18 paired legs and four loads: **−1.963 ± 1.504 ms/forward, 14/18 legs
faster, 2 SE [−3.467, −0.459]**, flat `−2.348 ± 3.302` against `+0.0533 ± 0.4038` per 1000 context
(R² 0.004) — **Term A, as pre-registered**. So `a` 125.11 → **123.15 ms**. The mark-level
prediction `0.628 × 2.49 + 0.0239 × 26.42 = −2.20 ms/forward` falls inside the band: two instruments,
one number.

**The gate that proved it.** `tests/gate_bf16w.cu` sweeps **all 18 dispatchable `(MM,NN)` tiles
including the pre-1.12 `8x0` arm**, at 7 values of M and 2 of N — 252 comparisons against the M=1
warp-per-output path, **worst max|diff| = 0**. `N = 254` is there because a widened tile fails at
the guarded store, not in the middle. Negative control: perturb the epilogue by one part in 1e7 and
**17 of 18 tiles FAIL** while `8x0`, which does not use the new kernel, correctly does not.

**What generalises.**

1. **A 1-D tile is half a 2-D tile, and the half you skipped is usually the cheaper one.** §1.1 was
   applied to five call sites in this repo and every one of them tiled M only. `NN` costs `MM*NN*4`
   registers and nothing else.
2. **Where the m-loop lives is worth 1.28× on its own.** Sixteen launches → one `gridDim.y`, same
   arithmetic, same traffic. On this box a launch is not free and a host-side loop over chunks is a
   launch per chunk.
3. **Bit-exactness constrains the tile shape, not the tile size.** The claim is that lane `l`
   accumulates k4 indices `l, l+32, …` into four chains reduced by the same shuffle tree — a
   *warp-local* property. Any tile that keeps one warp per output is free; a classic
   shared-memory/k-split tile is not.
4. **Optimise the shape the engine issues, and verify that it issues it.** The ladder entry that
   opened this item had the shape wrong in three places (`ntok`, the matrix size, the pass count),
   and the first A/B ran both arms on the same build. `DSV4_GEMM_TRACE=1` prints a
   per-`(route, M, N, K)` call-site histogram with byte totals and settled all four questions in one
   90-second load. See [`measurement-and-traps.md` §34](measurement-and-traps.md).

**What it leaves.** `cattn:compress` on an emit step is still **14.50 ms against a 335 MB / 1.40 ms
byte floor — 4.3 % of roofline**. The next factor is not a bigger tile (`(6,6)` is slower); it is
tensor cores or a shared-memory B stage, and it is a new item.

### 2.12 The kernel that read the buffer it was writing — `hadamard`, staged through shared memory (ladder 1.10, 2026-08-21)

**The measured gain is not speed — it is that `dsv4-server` reproduces itself again, and the number
is 15 of 15 load-pairs byte-identical against 0 of 15.** The speed effect is a drift-free null,
**−0.085 ± 1.810 ms/forward over 18 paired legs**, and is stated as one. This entry is on the wins
page anyway, because the staging is load-bearing and looks like an optimisation somebody could
helpfully remove.

**Mechanism.** `hadamard_kernel` computed `y[r,j] = D^-0.5 * Σ_i x[r,i] * (-1)^popcount(i&j)` with
one thread per `(r, j)`:

```c
for (int i = 0; i < D; ++i) acc += (__popc(i & j) & 1) ? -xr[i] : xr[i];   // reads the WHOLE row
y[idx] = acc * scale;                                                      // writes ONE of it
```

Every thread of a row reads all `D` elements and overwrites one. Correct exactly while `y != x` —
and three call sites pass the same pointer twice, including `kernels/compressor.cu`'s `rotate`
branch. There is no in-place form of this transform: the direct `O(D²)` evaluation needs the whole
input row after part of it has been overwritten. The fix gives **one block one row**, stages the row
into shared memory, `__syncthreads()`, and reduces out of shared — the read set closes before the
write set opens.

**A race that needs a neighbour, which is why it hid for so long.** Four warps of a lone 128-thread
block issue in lockstep, so the isolated kernel only fights itself once the grid exceeds the 20 SMs
(0/200 differing to 20 blocks, then 17, 28, 41, 74, 116, 184, 200 of 200 as it grows). **Under
concurrency it needs no grid at all**: `compressed_verify_step_indexer` forks the compressor emits
onto `g_side`, and with a filler kernel resident the `rows = 1` emit goes from 0/200 to **65/200
differing, 28 distinct results**. That is the decode path, and it had been reading as exonerated.

**Why it is bit-identical wherever it was already correct.** The sum is still serial in
`i = 0..D-1` over the same values; only where the values come from changed. Measured, not argued:
with `y != x` the staged kernel reproduces the flat kernel bit for bit, **200/200 repeats at all 21
row counts on both arms**, `gate_units` still passes the hadamard golden, and — the strongest form —
**the fixed server reproduces 1.11's four saved loads leg for leg**, which also confirms 1.12's
`gemm_fp32` change was bit-exact as it claimed.

**Measured gain, kernel level** (`gate_hadamard_alias --bench`, 200 iterations, D = 128):

| rows | 1 | 64 | 256 | 384 | 768 |
|---|---|---|---|---|---|
| where | both aliased emit sites | q-side, per verify | prefill s=1,024 | q-side at K=6 | prefill s=3,072 |
| flat / staged | **1.49×** | 1.06× | 1.14× | 0.99× | 0.99× |

D global loads per thread become D shared loads per thread plus D/blockDim global loads per *block*,
which is why the small shapes gain and the large ones — already L1-resident — do not.

**Measured gain, engine level: a NULL, and it was pre-registered as one.** Drift-free over two arm
orderings and four checkpoint loads: **−0.085 ms/forward, 2 SE [−1.895, +1.724], 10/18 legs faster**;
`tau` 1.6445 staged against 1.6461 flat. Note what one ordering said on its own: **+1.731 ms/forward**
with a drift term of **+1.817** — traps §33's lesson arriving for the third time, and here it would
have manufactured a regression instead of hiding a win. `hadamard` runs ~63 times per forward at
rows ∈ {1, 64, ≤384}; the kernel-level table bounds the whole effect at ~0.1 ms of a ~130 ms forward,
an order of magnitude under the 3.5 % run-to-run spread.

**And the before-arm is a moving target, which is the honest caveat on that band.** The flat arm
does not reproduce itself — 0 of 15 load-pairs — so the paired legs compare a fixed engine against a
distribution. It is the only before-arm there is, the null is consistent with the microbenchmark
bound, and the alternative reading (that the change costs several ms) is excluded by both.

**The gates.** `tests/gate_hadamard_alias.cu` runs both arms in ONE process via
`hadamard_set_stage()` — same binary, same input, one dispatch different — and **returns non-zero if
the OFF arm ever stops reproducing the defect**, because a gate that can no longer see its own bug is
not a passing gate. `--control` flips one ulp in the reference and requires every comparison to see
it; `--concurrent` is the co-resident probe above. Then the engine: `scripts/lhash_verify.sh`
re-runs 1.9's R and W protocols (0/56 pairs and 0/8 lengths, against 56/56 and 2/8 before), and
`scripts/genout_within_run.sh` runs four identical points at ctx 1,023 × 256 generated tokens inside
one process on both arms — **6/6 pairs diverge with `DSV4_HADAMARD_STAGE=0`, first at generated
token 26/42/45; 0/6 with it unset.**

[`measurement-and-traps.md` §36](measurement-and-traps.md),
[`negative-results.md` §4j](negative-results.md).

### 2.13 A width that was tuned when the verify was free — block 6 -> 5 (ladder 2.1, 2026-08-21)

**+3.91 ± 1.65 % tok/s on 32 prompts at unchanged `tau` (−0.052 ± 0.084, band covers zero), 26/32
legs positive, from a one-line change to a default.** The speculation block had been 6 since F94,
which measured 6 beating 5 on 4 of 4 realistic prompts. That measurement was correct for its engine
and is not overturned: it was taken when `k_topk_verify<<<K,32>>>` ran **one active thread per
block**, so a verified position cost almost nothing and the only thing extra width bought was
acceptance. Ladder 1.1/1.2 replaced that kernel. `DECODE_ZENITH_FINDINGS.md` §3.2 said in advance
that fixing it "changes the optimal block size" and that the width must be re-decided afterwards
"or you will tune to a transient" — this is that re-decision, and the transient was 6.

Mechanism, priced out of the engine's own 13,392 verify rounds (`ms_round ~ c0 + cB·BLK + cK·K`,
fitted per prompt, banded across prompts):

| | cost | what it is |
|---|---|---|
| `cB` | **+3.324 ± 0.281 ms** | one more **drafted** position — the MTP draft runs at M=BLK whether or not the verify ever reaches it |
| `cK` | **+15.184 ± 0.396 ms** | one more **verified** position — a target forward at M=K |

A verified position costs **4.6×** a drafted one, so the block's job is only to have proposals
ready; adaptK decides how many get spent. And adaptK stops early far more often than the ceiling
suggests — at BLK=6 the mean realised verify width is **3.87 of a ceiling of 7**, and only 19.3 % of
rounds reach that ceiling:

| BLK | mean realised K | rounds at ceiling | tokens/round | ms/round | **ms/token** | drafted, never verified |
|---|---|---|---|---|---|---|
| 4 | 3.507 | 37.1 % | 2.809 | 122.17 | **43.49** | 1.493 (4.96 ms) |
| **5** | 3.736 | 27.1 % | 2.958 | 128.74 | **43.53** | 2.264 (7.53 ms) |
| 6 | 3.871 | 19.3 % | 2.972 | 134.14 | **45.14** | 3.129 (10.40 ms) |

**The sixth proposal buys 0.014 tokens per round and costs 5.40 ms**: 0.5 % more tokens for 4.2 %
more time. This table is computed from a different field of the same log than the paired tok/s
result and lands within a tenth of a percent of it, which is what makes the paired number an
engine property rather than an averaging artefact.

**5 is an interior optimum and lands exactly on the trained width.** 4 buys nothing over 5
(−0.12 ± 1.43 %) and does cost acceptance (`tau` −0.242 ± 0.102); 6 loses 3.58 ± 1.53 %; the first
sweep closes everything above — 7 −1.70, 8 −2.93, 9 −6.30, 10 −9.36, 12 −10.86 %, monotone, while
`tau` climbs monotonically 3.48 → 4.36. `config.json`'s `dspark_block_size` is 5. F94 pushed serving
one position past training while that position was nearly free; with it no longer free, serving
falls back onto training.

The gate that proved it: two sweeps, each in ONE checkpoint load, each palindromic per prompt so
drift linear in run order cancels (measured drift in the deciding sweep: **+0.000 ± 0.002 tok/s over
99 mirrored pairs**), adaptK frozen at 1.50, on the live `s3` checkpoint. BLK sets M in the verify
forward and therefore MoE atomic reduction order, so the sweep was pre-registered as *not*
bit-exact and instrumented to measure the divergence rather than assume it — and it measures
**none**: LOSSLESS gate **24/24 points PASS**, and across the 344 sequences `DSV4_GENOUT` recorded,
**every arm is id-for-id identical to the BLK=6 reference over every shared position — 258
comparisons at widths 4 through 12, zero differing tokens.** What differs is only where the
sequence *ends*: generation stops at the first verify that carries the count past NGEN, so a
narrower block overshoots 200 by −6 to +7 tokens. The analyser originally counted that as
divergence, which is [§35](measurement-and-traps.md)'s trap run backwards — comparing past the
length at which two arms can be expected to agree, and calling agreement a difference.
`include/dsv4_engine.h` carries the served default;
`scripts/serve.sh` passes no `--blk`, and the change was proven behaviourally — `[engine] loading …
(seqmax=8192, blk=5, adaptK=1.50)` from a real `run_server.sh` load. Every measurement script pins
its width explicitly, so no frozen protocol moved.

[`negative-results.md` §4k](negative-results.md),
[`measurement-and-traps.md` §37](measurement-and-traps.md).

## 3. Precision and layout

| finding | change | result |
|---|---|---|
| F41 | m16 tile fetched every weight sector for half its payload | fixed; prerequisite for everything after |
| F66 | expert pointers are misaligned by a **constant 8 bytes** (43,470 of 44,436 tensors at `data_offset%16==8`) | the funnel shift was buying alignment that was already there |
| F72 | so: two `uint2` loads instead of two `uint4` + funnel shift | same instruction count, **half the bytes requested**, +1.4 % |
| F56 | the shared expert was never dependent on the routed ones | run them concurrently |
| 1b.2 | the main KV cache stores E4M3-grid values with a power-of-two scale **in FP32** — 8 bits of information in 32 | packed to 448xE4M3 + 7xUE8M0 + 64 fp32 RoPE = **711 B of payload in a 720 B row, 2.844x**, BIT-EXACT |

F66 → F72 is the cleanest example of the project's method: **measure the property you are
compensating for before writing the compensation.**

**1b.2 is the cleanest example of a bit-exactness claim being cheap to PROVE rather than to argue.**
The stored value does not change — the write path already computed `e4m3(x/scale) * scale` and threw
the code away — so the acceptance test is `memcmp`, not a benchmark and not a tolerance, and the
required `n` is zero at a flip rate of zero. Four gates, none needing a checkpoint:
`gate_kv_pack` (round trip on 5.24 M floats per distribution across five distributions, plus the
reader against the FP32 reader at 11 launch shapes x 5 problem shapes), `gate_kv_pack_e2e` (the
~30 WIRING sites, driving the real decode/verify functions in both layouts and memcmping the
outputs), and the two graph gates for the device-pos store kernels. Every one runs a negative
control that must fail. The engine then agreed: **16 of 16 legs byte-identical with `tau` and mean
verify width equal on every leg**, identical generated ids on `build/decode`, LOSSLESS x3 on both
arms. A tolerance gate would have passed the sign-bit bug that `gate_idx_pack` caught in 1b.1 at
**worst |delta| = 0** — numerically perfect, bitwise wrong.

**It is not an adopted throughput win and it ships default OFF**; the measured trade is
+3.233 ± 0.203 ms/forward flat against −0.4996 ± 0.0290 per 1000 context, break-even ctx 6,471.
[`negative-results.md` §4i](negative-results.md) for why a 2.84x byte reduction made the reader
kernel *slower*, [`context-scaling.md`](context-scaling.md) for the half of it that is a win.

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
