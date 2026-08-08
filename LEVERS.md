# LEVERS.md — what has been tried, what is dead, and what is actually left

**Read this before proposing any decode optimisation.** `LOOP_LOG.md` is 3,200 lines and 62 findings;
this is the index. Its job is to stop a cycle spending a 15-minute checkpoint load re-deriving a
result that is already in the log with a number attached.

Rules of the road:

- A lever in **§3 (retired)** is closed. Re-opening one requires new *evidence*, not a new argument —
  and the evidence has to address the specific measurement that killed it.
- A lever in **§4/§5 (open)** has a stated expected value. If a measurement lands more than ~2x off
  that estimate, fix the model before spending the next run (that rule is what produced Finding 57).
- Everything here is measured on this box, this checkpoint, `sm_110a`. None of it transfers.

---

## 1. Where the time actually goes

Current baseline (prompt 0, clocks pinned, post-Finding-72): **20.44 tok/s speculative**,
**13.50 tok/s base AR** (74.1 ms/tok). `evidence/final6.log`.

The K=5 verify dprof TOTAL is **134.8 ms** and splits into two populations that behave completely
differently (`evidence/moescan.log`, post-F73, `DSV4_DPROF=1 DSV4_KSWEEP=1`, clocks pinned):

| group | ms | % | headroom |
|---|---|---|---|
| routed MoE — `moe:w1w3` 40.4 + `moe:w2` 20.2 | **60.6** | 45 % | **F64/F65/F70/F72 took 21 % off `w1w3` and 10 % off `w2`; still only ~76 % of roofline** |
| `cattn:ogroup` (`o:wo_a` 11.5 + `o:wo_b` 9.8 + rope) | 21.7 | 16 % | some |
| `cattn:q_proj` (`q:wq_b` 11.2 + `q:wq_a` 6.7 + glue) | 20.0 | 15 % | some |
| `lm_head` | 7.0 | 5 % | little |
| `cattn:indexer` (21 layers) | 3.9 | 3 % | **F71 took 57 % off this; `i:score` is now 0.76** |
| `hc_pre` attn+ffn, `rmsnorm`, `moe:router/group/act/combine` | ~10.2 | 8 % | latency-bound, not bytes |
| `cattn:compress`, `cattn:sparse`, misc | ~3.6 | 3 % | already forked |

**Ranking is now stable; the two fp8 GEMM blocks (`q_proj` + `ogroup` = 41.7 ms, 31 %) are the
largest untouched region.** Their four marks — `q:wq_b` 11.2, `o:wo_a` 11.5, `o:wo_b` 9.8, `q:wq_a`
6.7 — are all the same smem-staged fp8 tile at different N, and F67 already refuted the two obvious
reroutes. Nothing here is a geometry bug (F73 audited that); it is kernel efficiency.

**This table was WRONG for the whole project and Finding 64 fixed it.** The MoE was believed to be at
the roofline on the strength of a *modelled* expert union of 29.9 at K=5; the measured union is
**17.53** (`DSV4_MOEUNION=1`), and the kernel was re-reading each expert's weights once per row it
served. Amortising that took `moe:w1w3` down 20 %. **Treat every remaining "at the roofline" claim in
this repo as unverified until its bytes have been counted from the kernel, not from a model.**
`moe:w2` is now issue-bound rather than bandwidth-bound (K=2048 gives each lane two `kb` iterations),
so it did not move.

Achievable bandwidth is **~233–240 GB/s** (`tools/bw_probe.cu`, 235.6 GB/s with clocks pinned), and
`tools/footprint_probe.cu` proved it is reachable at scale — 230–246 GB/s at footprints from 0.5 GiB
to 64 GiB, both allocators. Streaming out of the 111 GiB managed pool is free.

---

## 2. Adopted — already in the engine, do not re-propose

| lever | gain | finding |
|---|---|---|
| smem-staged FP8 mma tile; `lm_head` M=K | 10.04 → 12.12 tok/s | 41, 42 |
| bf16 alignment fallback (`head_bf` is 4-byte aligned) | 12.12 → 14.36 | 41 |
| ogroup NR row-sharing; `gemm_fp32` M=K | 14.36 → 14.76 | 43 |
| weights → managed device-preferred memory | 14.76 → 15.49 | 44 |
| full-step CUDA graph, base AR only | 92.5 → 79.3 ms/tok | 44 |
| adaptive verify width (lossless) | **+9-11 %** where it engages; wash on prompts that never narrow | 49, 59, **63** |
| **intra-layer concurrency, 3 fork sites** | **+3.1 %** (36 matched points) | 55, 56, 57 |
| **clock pinning (`jetson_clocks`)** | **+3.0 % steady, +20.7 % cold** | 60 |
| **ogroup row-tile fix** | correctness, not speed — see §6 | **62** |
| **MoE row amortisation, RB=4 fitted to the measured histogram** | **+7.4 % spec, +2.3 % base AR; verify −9.3 %** | **64, 65, 70** |
| **`index_score` warp-per-output** | **+4.4 % spec, +7.1 % base AR** (`i:score` −87 %) | **71** |
| **MoE GEMV `uint2` weight loads (no funnel)** | **verify −2.8 %, `moe:w1w3` −7.0 %** | **72** |
| **MoE grouping scans parallelised (`k_moe_prefix`, `k_build_tiles` were `<<<1,1>>>`)** | **`moe:group` −20.3 %** (2.66 → 2.12 ms) = ~0.4 % of the verify. Bit-identical. End-to-end **not** claimed — see F73 | **73** |

---

## 3. Retired with a measurement — closed, with the number that closed it

**Speculation**

| lever | what killed it |
|---|---|
| block size > 5 | identical accept sequence at 5 and 8; acceptance is flat in block size (F43) |
| draft refinement (NPASS>1) | acceptance **3.00 → 2.08** — the MTP heads are trained with the noise token as placeholder, so feeding real proposals is off-distribution (F45) |
| DDTree / tree speculation | correct but does not beat linear; depth-dominated (prior project memory) |
| a reusable verify CUDA graph | `VERIFYGRAPH=1`: 2,788 nodes, **1.05x**. Launch overhead is not the verify's problem (F46) |

**Kernels**

| lever | what killed it |
|---|---|
| shared-A fp8 GEMV | slower at every M; trades L1 for 4x smem traffic (F41) |
| small-M fp8 GEMV as default at M≥2 | loses 1.5–2.3x to the smem-staged m16 tile (F41) |
| wave-quantisation on the M=1 GEMV | two schemes, both worse or in the noise (Opt #2) |
| software-pipelining the MoE tile loop | reverted, no gain (tc_moe_gemm.cu:407) |
| B6: skip the funnel when the weight pointer is 16B-aligned | **it never is.** Across all 48 shard headers, 43,470 of 44,436 expert tensors sit at `data_offset%16 == 8` and 966 at 12 — **none at 0**. The fast path could not fire (F66) |
| B6': supply the funnel partner by warp shuffle instead of a second load | lane L's `wa+16` IS lane L+1's `wa`, so it looked free. Measured **423.8 -> 650.6 us** at RB=2, stalls 3.90 -> 9.81, registers 62 -> 67: lane 31 still needs a real load and it is *predicated*, not branched away, so the warp pays for both paths plus 8 shuffles per kb (F66) |
| MoE GEMV `BN` > 2 | monotonically worse with a CORRECT store: 431.8 / 459.6 / 492.2 / 555.2 µs at BN=2/3/4/6, registers 64 -> 114, occupancy 63 % -> 32 %. The apparent BN=4 win was the compiler deleting half the weight loads because the store was hardcoded to two columns (F69) |
| split-K on the small-N fp8 GEMMs | `q:wq_a` genuinely improved 6.58 -> 5.69 ms, but the reduction-order change dropped the model into a **degenerate repeating loop** from token 6, which RAISED acceptance 2.90 -> 3.86 and produced a **fake +28 % tok/s** that every existing gate passed. Retired: not for speed, for numerics (F68) |
| align the weights so the fp8 `uint4` fast path fires | a per-shard pad makes 99.06 % of tensors 16B-aligned and bought **nothing** (`q:wq_a` 6.58 -> 6.52, spec 19.13 -> 18.59). `gemm_bench`'s own `m16+smem B+4` column already said alignment is worth 7 %, not 3.3x (F67) |
| route all fp8 GEMMs through the M=K GEMV | `q:wq_a` 6.58 -> 5.85 but `q:wq_b` 11.67 -> **21.26**, `o:wo_b` 9.24 -> **20.56**, TOTAL 144.94 -> 176.15. A crossover in N, not a better kernel (F67) |
| route only small-N through the M=K GEMV | net negative at BOTH thresholds. N<=2048 caught the shared expert on the side stream and cost `moe:w1w3` +4.80 ms; N<=1024 fixed that and was still worse (TOTAL 147.23 vs 144.94) (F67) |
| `__ldcs` evict-first on the ogroup weight stream | stall ratio 7.33 → 3.46 **and slower**: 0.197 → 0.257 ms. L2 hit rate *fell*; the weight line was not the evictor (F55) |
| smem staging of `o` in the ogroup M=K GEMV | bit-exact, 8x less activation traffic, and **a wash** at the NR the engine uses. The barrier destroys the warp skew that was hiding latency. Kept behind `OG_SMEM=1` (F55) |

**Whole hypothesis classes**

| hypothesis | what killed it |
|---|---|
| working-set size costs bandwidth (TLB/SMMU reach) | `footprint_probe`: 230–246 GB/s from 0.5 to 64 GiB, both allocators (F55) |
| the bench-vs-in-situ gap is launch overlap | `gemm_bench`'s `timeit` launches on **stream 0**, which is stream-ordered. It overlaps nothing. The gap was single-kernel under-saturation (F55) |
| the indexer fault is a length bug / an allocator accumulation | `:91` and `:96` are the **same `cudaStreamSynchronize`**, the first real sync in the layer — never a location. Free memory *rises* 10.36 → 11.61 GiB across 17 points. Three faithful reruns 17/17 clean (F58) |
| the nondeterminism is the concurrency / atomics / stale window / uninit scratch / caches | all five eliminated with measurements; the real cause was F62 (F60, F61, F62) |

---

## 4. Open — base AR decode (12.56 tok/s, 79.6 ms/tok)

Base AR reads the whole 12.26 GB weight set per token. At 233 GB/s the floor is **52.6 ms/tok =
19.0 tok/s**; graphed we are at 79.9 ms, so ~34 % of the base step is not bytes.

| # | lever | expected | why it might work / what to watch |
|---|---|---|---|
| ~~B0~~ | ~~audit every kernel for thread-per-output at decode shapes~~ | **NAMED LIST EXHAUSTED, F71 + F73** | `index_score` **+4.4 %** (F71). `k_moe_prefix` and `k_build_tiles`, both `<<<1,1>>>` over `nr=160`, fixed in F73: `moe:group` −20.3 %, and both now sit at the **launch-latency floor** (0.24–0.25 ms / 43 layers, vs 0.19 for a trivially parallel neighbour). The rest are measured under B0's own 0.5 ms bar: `k_topk_verify`/`k_topk_decode` → `i:topk` = **0.12 ms**; `k_dg`/`k_advance_T`/`k_incr` are genuinely one scalar's work. **The class paid twice and is now dry** — reopening needs a *new* kernel with a bad geometry, not another pass over these. |
| B8 | **the fp8 GEMM block: `q:wq_b` 11.2 + `o:wo_a` 11.5 + `o:wo_b` 9.8 + `q:wq_a` 6.7 = 39.2 ms (29 %)** | unknown; largest untouched region | Same smem-staged tile at four N. Not a geometry bug (F73 audited). F67 closed both reroutes (M=K GEMV at all N, and small-N only) and F68 closed split-K **on numerics, not speed**. What is *not* closed: nobody has counted this kernel's actual bytes from the source the way F64 did for the MoE, and §1's old "~80 GB/s" for `q_proj` was a region number over a region containing five kernels. **Falsify by counting bytes per mark first** — if `q:wq_b` is already at roofline this is dead and should be retired in one cycle without a build. |
| B1 | **more fork sites** (§5 pricing table) | ~0.3–1 %/pair | The three obvious independent chains are taken. Remaining pairs are small; check the partner is *not* already saturated or the gain collapses. |
| ~~B2~~ | ~~split-K on the small-N GEMMs~~ | **RETIRED, F68** | 512 rows = 32 m16 tiles = 32 blocks on 20 SMs. **Not bit-exact** — a K-split reduction changes accumulation order, so it needs a tolerance gate, not an equality gate. |
| B3 | **fuse `wq_a`+`wkv` into one launch** | ~0.5 % | Combined N = 1536 is still only 192 warps, so it barely moves `N/8` (F67). And `wkv` is already forked to a side stream by C1 and fully hidden (`q:kv_join` = 0.05 ms), so there is nothing left to overlap. Low value now. |
| B4 | ~~fuse the elementwise glue~~ | **≤0.4 %, killed (F67)** | Moves almost no bytes; pure launch/latency floor. The verify-graph result (1.05x) caps what graphing can return here. Sinkhorn is *already* one fused kernel — do not "fuse" it again. |
| B7 | **occupancy of the MoE GEMV** | smaller than it looks | The RB sweep found the shipped RB=2 optimal, and the `OGMK_BLOCKS_PER_SM` register-cap knob is also already at its optimum (BPS=4/NR=4 = 0.2044 ms beats 2, 3 and 6 at every NR). Both knobs are exhausted. |
| B7' | ~~raise MoE GEMV occupancy via the register cap~~ | **exhausted (F67)** | Even at the optimum RB=2 it runs 62 registers / 63 % occupancy / 155 GB/s = 67 % of roofline. The register budget is dominated by the funnel pair (B6) and `acc[RB][BN]`. B6 is the cheapest way in. |
| B5 | **FP4 for the MLA/dense weights** | moves the byte floor ~1.2x | Assessed and **not done**: it is pure weight transform, trivial on Thor, but costs accuracy and the constraint is *no additional quantization*. Requires an explicit decision to relax that. |

## 5. Open — speculation (18.13 tok/s, acceptance ~2.9 of 5)

`tok/s = tokens_per_cycle / cycle_ms`. The cycle is verify (~86 %) + draft (~14 %). Since the verify's
MoE half is at the roofline, **the numerator is the better lever than the denominator.**

| # | lever | expected | why it might work / what to watch |
|---|---|---|---|
| S1 | ~~re-fit the adaptive-width threshold~~ | **DONE, F63** | Re-run post-F62: adaptive is worth **+9-11 %** on the three prompts where it engages and a wash on the other two — not the +28 % prompt 2 showed on the garbage prefill. The threshold **1.0 / 1.5 / 1.75 remain not separable** (means 16.94/16.70/16.79 against a 15 % within-cell spread). 1.5 stays. Do not re-run without a reason to expect >5 % from a threshold move. |
| S2 | **raise acceptance** | the biggest single multiplier | Acceptance 2.9 → 3.5 is +21 % at constant cycle. But the two obvious routes are dead: block size (F43) and draft refinement (F45). This needs a *different* idea — the MTP heads are what they are, and retraining is out of scope. |
| S3 | **cascade / early-exit verify** | probably negative | Verifying 1–3 then 4–5 reads the 8.81 GB fixed weight set **twice**. That duplicated fixed cost (~92 ms) swamps the expert-union saving. Priced and rejected on arithmetic; do not implement without a byte model that beats it. |
| S4 | **reduce the expert union at K=5** | smaller than believed | The union is **17.53** measured, not 29.9 — the five positions already share most of their experts, so there is far less to win here than the old model implied (F64). This is the largest single block of time in the engine and the only untried structural idea against it. Any scheme must keep the verify *exact*: biasing the **draft** toward expert-overlapping tokens is lossless by construction (the verify still corrects), but it would cost acceptance, so it is a trade, not a free win. |

### Pricing model for fork-style levers (fitted, F56/F57)

Overlap is **cheaper than serialising, not free** — the hidden op's cost reappears in the partner
region because two concurrent kernels contend for SMs and L2, not only DRAM.

| pair | hidden | partner state | recovered |
|---|---|---|---|
| shared expert ∥ routed | 10.37 ms | at roofline | 2.78 = **27 %** |
| compressor ∥ `build_qKV` | 8.56 | mixed | 1.74 = **20 %** |
| kv chain ∥ q chain | 3.86 | both starved | 1.22 = **32 %** |

**Expect 20–32 % of the hidden op's standalone cost, rising the more starved both sides are.** The
naive byte model predicts ~80 % and is wrong by 3–4x. Do not use it.

---

### Reference: measured MoE shape at K=5 (`DSV4_MOEUNION=1`)

union **17.53** distinct experts over 30 rows, 43 layers. Rows per expert: **1 → 61.7 %, 2 → 20.7 %,
3 → 7.8 %, 4 → 4.5 %, 5 → 5.3 %**, max 5. Weight reads per expert by RB: 2 → 1.229, 4 → 1.053,
8 → 1.000. Any future MoE blocking decision should start from this table, not from `bs`.

## 6. Measurement traps — every one of these has cost this project a cycle

1. **The canonical prompt is 6 tokens.** `PSp=5`. It cannot reach any `bs>16` code path — which is
   exactly how Finding 62 (garbage prefill for every prompt ≥18 tokens) survived every gate in the
   project. **Gate the shape range the engine spans, not the one the canonical prompt uses.**
2. **A masked read is not a safe read.** F62's row masks made the reads legal and hid a missing loop.
3. **`gemm_bench` ranks kernels; it does not predict end-to-end gain.** It launches on stream 0, so it
   cannot see the concurrency effect at all, and it systematically undervalues anything that makes a
   kernel bigger or lets two run together. Its HOT-vs-COLD row is already 1.73x at M=5.
4. **Clocks are governed by default** (GPU 315/1386 MHz, **memory controller 2750/4266**). A long
   roofline probe ramps the governor by itself, which is why no roofline number here ever saw it. Pin
   with `jetson_clocks` and *say so* in the write-up.
5. **Sweep position matters.** Replicate 1 of a 36-point sweep ran ~14 % slower than replicates 2–3.
   Put the two arms of an A/B at **adjacent** sweep positions, or use ≥3 replicates and compare means.
6. **A non-bit-exact change can BUY acceptance by degrading output.** Split-K sent the model into a
   repeating loop, which is trivially predictable, so acceptance rose 2.90 → 3.86 and tok/s "improved"
   28 % — past a first-token argmax gate, a MATCH 5/5 gate and a cosine gate (F68). Any
   acceptance-based number is meaningless unless the emitted sequence is checked. The engine now runs
   a **lossless gate** (spec output == base-AR output) on every canonical-prompt sweep point; keep it
   in any run that carries a tok/s number.
7. **A shared bump allocator makes every allocation a global variable.** A `dmalloc` added in one
   kernel moved buffers in unrelated marks (`cattn:ogroup` +1.28 ms) purely by shifting arena offsets.
8. **A count of distinct values over 8 samples is not evidence about a mechanism.** That reasoning
   produced a wrong adoption in F60 which F61 had to retract.
9. **Check the harness can express the regime before sweeping the parameter.** Three findings are the
   same error: F65's probe modelled 1 row per expert, F69's store was hardcoded to 2 columns so a BN
   sweep measured dead-code elimination, F70's probe clamped rows at 2 so an RB sweep never chunked.
   Each time the apparatus encoded an assumption and the sweep confirmed it.
10. **A probe predicts ranking, not magnitude.** F70's probe said −6.5 %, in situ gave −1.3 %, because
   the kernel is at 76 % of roofline and a byte cut does not convert 1:1.
11. **A cached property must be keyed on the thing it describes.** F72 shipped a `static` flag computed
   from the main layers' weights and applied it to the MTP draft blocks — different tensors, different
   alignment, `misaligned address`. Process-wide was too coarse; per-call was correct but cost 3.05 ms
   of host stall *inside a GPU mark*. The struct was the right key.
12. **A change that is flat in K has a free within-run control: K=1.** F73's fix saved the same ~0.5 ms
   at every K (the `mg:*` marks are identical at K=1 and K=5). The run showed K=5 −3.2 %, base AR
   +4.4 %, spec +2.5 % — and K=1 **+0.2 %**. A K-flat change cannot produce a K-dependent delta, so
   the end-to-end movement was noise and was not claimed. **Before believing a cross-run end-to-end
   delta, find a mark inside the same run that the change must not have moved.**
13. **Check what a control was actually measuring before averaging it.** F73 nearly used three
   `uint2*.log` runs as a variance estimate. Two of them are F72's *buggy* per-call-align8
   intermediates (`moe:group` 6.41 and 5.79 vs the shipped 2.66) — averaging them would have
   manufactured a 3 % "control spread" out of a known bug. There was exactly **one** valid control.
14. **A single-thread loop over N elements is not N serial memory latencies.** F73 predicted 160
   dependent loads ≈ 35 µs/layer and measured ≈ 12. Only the *additions* chain; the `counts[e]` loads
   are independent, so the compiler pipelines them. Serial-scan geometry is worth ~2x here, not 6x —
   size the lever off the dependency graph, not the trip count.
15. **Line numbers in CUDA error messages are where the error was *collected*, not where it happened.**
   `dsync` is a no-op under the arena, so the first real sync in a layer absorbs ~20 launches' worth of
   asynchronous faults. Use `DSV4_SYNCPROBE=1`.

---

## 7. Instruments available (all off by default)

| knob / tool | what it answers |
|---|---|
| `DSV4_DPROF=1` + `DSV4_KSWEEP=1` | per-sub-op ms for K=1..5 — **the** ranking instrument |
| `DSV4_SPECPROF=1` | draft vs verify split of the spec cycle |
| `DSV4_HASH=1` / `=2` | prefill state hash per sweep point / per layer — bisects a divergence to a layer |
| `DSV4_SYNCPROBE=1` | checked sync after 21 individual launches; names the faulting launch |
| `DSV4_MEMTRACE=1`, `DSV4_BALLAST_GB` | free memory per sweep point; make headroom settable |
| `DSV4_ARENA_ZERO`, `NO_ZERO_SCRATCH`, `DSV4_ZERO_CACHES` | isolate uninitialised-memory hypotheses |
| `DSV4_BLKSWEEP="BLK:passes:adaptK:prompt"` | many configurations per checkpoint load; **negative adaptK = fixed width** |
| `tools/overlap_probe.cu` | does this kernel saturate memory alone, or does it need a partner? |
| `tools/footprint_probe.cu` | does working-set size cost bandwidth? (no) |
| `compute-sanitizer --tool initcheck` | uninitialised global reads — found F62 in one run |
| `tests/gate_scratch_init` | poisons prefill scratch and diffs; localises to one allocation |
| `tests/gate_forkjoin_graph` | a fork/join that breaks graph capture, caught in 2 s not 15 min |
