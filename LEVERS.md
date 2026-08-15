# LEVERS.md — what has been tried, what is dead, and what is actually left

> ## ⛔ THE LOOP IS STOPPED — `FLYWHEEL_STOP` created 2026-08-08 12:29:25 (F84)
>
> An operator set the kill switch. It is untracked, size 0, and **no script in this repo writes it**
> (`grep -rn FLYWHEEL_STOP scripts/` → three readers, zero writers). It was created 4m28s after
> `2ece9f9`, the commit that concluded *"two unrelated methods now agree the kernel path is
> finished"*. **Do not pick a lever, do not run a research phase, do not take a model run.** If you
> are an executor cycle reading this, you got past `flywheel.sh:61` because your tick's preflight
> started before the file existed (F84 has the timestamps) — halt and say so.
>
> **Queue: 0** open non-training levers ≥1 %. **The PIVOT is due and is being suppressed**:
> `open_nontraining_levers` = 0 and `.flywheel_openprev` = 0 satisfy `flywheel_audit.sh:42`, but the
> `FLYWHEEL_STOP` early-exit at line **25** returns first, so the banner can never be written.
> Not fixed here on purpose — see F84. The remaining lever is **S5**, and it is a training job.
>
> To resume: `rm FLYWHEEL_STOP`. `halt` in `FLYWHEEL_STATE.json` was deliberately left `false` so
> that is all it takes.

> ## ⚠ ONE LEVER REOPENS — and it is large. `PERF.md`, 2026-08-15
>
> **The queue is not 0.** §9's ledger, §1's baseline and every finding behind "the kernel path is
> finished" share one scope condition, stated in §1's own first line: **prompt 0**. A cost that
> scales with resident KV depth is identically zero there. The ledger is not wrong — it is correct
> within the regime it measured, and nothing below retires anything in §3.
>
> The eval battery ran **891 requests of real workload** at prompt depths from 151 to 6568 tokens.
> Fitting `ms_per_verify = W + K x kv_mid` over them (`tools/perf_report.py`):
>
> | term | estimate | 95 % CI (cluster bootstrap over tasks) |
> |---|---:|---|
> | `W` fixed per forward | 136.8 ms | [131.8, 143.1] |
> | **`K` per resident KV token** | **0.0307 ms** | **[0.0283, 0.0335]** |
>
> That is **31 % of the forward at 2k depth, 65 % at 8k, 79 % at 16k**. It clears trap 25's ~1.5 %
> cross-run floor by more than an order of magnitude, so it is not a timing artefact, and three
> in-tool artefact checks fail to kill it: `tau` FALLS with depth (corr −0.317) so the mechanical
> confound runs the wrong way; four workloads agree on the `tau`-free slope to within 4 %; and the
> slope reproduces on **exogenous** prompt-length variation at matched completion length. The
> intercept independently reproduces ROOFLINE §3's 126.7 ms AR step on the byte-identical 180B
> backbone, landing just above it as the expert-union argument requires.
>
> **It is not a bandwidth term.** At 240 GB/s, `K` would mean moving **7.36 MB per resident KV token
> per forward** against ~3.3 KB of actual per-position state, with `index_topk` = 512 bounding what
> attention may read at any depth. Depth-linear cost that bytes cannot explain is compute-,
> occupancy- or launch-bound. Ranked hypothesis, **not a diagnosis**: the DSA indexer must SCORE all
> D resident positions per layer per forward to select its top-512, so it is linear in depth by
> construction while the attention it feeds is not.
>
> This does **not** contradict F125/F126/F137. "MoE at 94 % of roofline" is a prompt-0 measurement of
> the MoE window and stays true; the depth term lives in a different region that the prompt-0
> instrument cannot see.
>
> **Instrument gap this exposes.** §7's instruments are rich but all are env-gated at process start
> and emit log output from bench harnesses, and `DSV4_DPROF` is documented as costing ~0.4–1.0 %
> ("never re-baseline from one"). None of them attribute *production* traffic. What is missing is a
> cheap always-on per-request path in the **server**: per-position acceptance hazard `h(j)` rather
> than only `tau`, realised verify width under `adaptK`, distinct experts touched per forward
> (which would settle the speculation/weight-traffic interaction directly), and KV depth per step
> rather than per request.
>
> **Do not treat this as a restart.** `FLYWHEEL_STOP` is untouched. This is the evidence §"Rules of
> the road" requires to reopen a lever, recorded where the next reader will see it.
>
> **One caution, on a byte lever this reopens the pricing of.** A cloud FP4 requant is worth
> costing, but not off `B_tok`: the routed experts are **already MXFP4**, so the FP4 mass is MLA
> (4599 MB, FP8), shared expert (1082, FP8), `lm_head` (1059, BF16), KV compressor (526, BF16) and
> the DSA indexer (275, BF16) — nominally `B_tok` 11202 -> 6848 MB, **38.9 %**. Two reasons that
> number is not the answer. (a) **F129**: `B_tok` overstates every byte lever on a speculative
> engine, because the verify reads each weight once for M tokens — so this must be re-derived
> against F129 before anyone spends the requant. (b) Even taken at face value it prices out very
> differently depending on what `W`'s non-transfer time is: at 240 GB/s `B_tok` is 46.7 ms of
> transfer against a measured `W` of 136.8 ms, so **90.1 ms is not transfer**. If that is additive
> overhead the requant buys **1.15x**; if it is bandwidth wasted by access pattern it buys
> **1.64x**. Same measurement, two decompositions. The `ncu` profile that settles the depth term
> settles this too — **profile before requanting**.

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

Current baseline (prompt 0, clocks pinned, **post-Finding-83, CLEAN**): **22.15 tok/s speculative**,
**13.78 tok/s base AR** (72.6 ms/tok), acceptance **2.89**, speedup **1.61x**.
`evidence/clean_post_f83.log` — no `DSV4_DPROF`, no `DSV4_KSWEEP`, no `DSV4_SPECPROF`, GATE PASS,
MATCH 5/5 and LOSSLESS GATE PASS; draft path on the arena (F83), output bit-identical to
`clean_post_f81.log`. The previous baseline was 22.06 / 13.83 (`evidence/clean_post_f81.log`) —
no `DSV4_DPROF`, no `DSV4_KSWEEP`, no `TCB_CPA`, GATE PASS and LOSSLESS
GATE PASS. F81's lever ships default OFF and this run's spec token sequence is **byte-identical** to
`clean_post_f79.log`, so it measures the same default arm — which is what makes it a variance control.

**Trap 25 is now measured twice and it does NOT read as a 1.5 % noise band.** Two identical-arm clean
pairs on the nine-paired-verify instrument: f76 → f79 was **−1.5 %, 9/9 in one direction**; f79 → f81 is
**+0.17 %, 2/9**, per-verify spread +0.00 % to +1.21 %, spec −0.05 %, base +0.36 %. So the instrument
*can* reproduce to ~0.2 %, and the 1.5 % was an occasional **systematic shift**. Practical rule: a
*single* cross-run claim under ~1.5 % is still unsupportable — you cannot tell which kind of pair you
drew — but a sub-1 % effect **is** resolvable with **two** pairs. That is the only route left to
anything in §4, all of which is sub-1 %.

The first of those two pairs is F79's, and it is the reason the floor exists at all: F79 also shipped
default OFF, so its clean run measured the same arm as `clean_post_f76.log` (21.76 / 13.64) — `ptxas -v`
confirmed `ogroup_gemv_mk_kernel<5,4,false>` unchanged at 64 registers / 44 bytes spill — and yet
**+1.4 % spec, +1.0 % base, and −1.5 % on all nine paired verifies at identical K and accept counts**.
The ±0.8 % quoted in F76/F78 is the spread *within* a matched pair, not across runs. F76 (+0.1 %) and
F78 (+0.28 %) were called null and stay null; F74's +6.1 % clears either pair 4x.

The previous re-baseline (Finding 77) settled two things the stale line could not. **Acceptance is unchanged at 2.89** against final6's 2.89 — F73 and F74 both
claimed bit-identical, and a pure kernel win must leave drafting untouched, so this is the claim
holding where it could have failed silently. And **dprof overhead is ~0.4 %** (21.76 clean vs 21.68
with dprof), inside the run-to-run band — the profiler is not distorting the marks it reports, which
is Finding 73's observation confirmed from the other side.

It cannot go stale by three cycles again: `scripts/flywheel.sh` now classifies each cycle's run by
whether its log carries `[dprof]` marks, maintains `baseline.dprof_runs_since_clean`, and at >= 2
step 6 makes the next run a mandatory clean re-baseline.

**THE TOP-LEVEL SPLIT WAS A VERIFY TABLE CALLING ITSELF A CYCLE TABLE (Finding 82).** Everything
below this paragraph is the **verify**. The cycle is verify + draft, and the draft's only attribution
in this project was `evidence/f47.log` — from before F64/F65/F70/F71/F72/F74, every one of which cut
the verify and none of which touched the draft. Re-measured at the current baseline
(`evidence/specprof_f82.log`, `DSV4_SPECPROF=1`, clocks pinned):

| region of ONE verify round | ms | % |
|---|---|---|
| verify, 43 layers | 113.50 | 82.6 % |
| **draft: 3 MTP blocks** | **13.47** | **9.8 %** |
| **draft: `fwd_head`** (device-side AR over 5 positions, F27) | **10.13** | **7.4 %** |
| draft: `main_kv` x3 | 0.25 | 0.2 % |
| **TOTAL** | **137.35** | |

**The draft half is 17.4 %, not the 14 % carried since F47** — it grew as a *share* because the
verify shrank under it. And **97.2 % of the draft half is host time inside the raw allocator**:
**`cudaMalloc`+`cudaFree` = 10.19 ms = 7.4 % of the whole cycle, 134 calls at 76 µs**, plus 9
`cudaStreamSynchronize` at 1.44 ms. **127 of the 134 run after their own function's sync, i.e. on a
drained GPU.** The verify path has been on `dscratch.h`'s arena since F44; the DSpark draft path
(`dspark_main_kv`, `dspark_attn_forward`, `dspark_block_forward`, `dspark_forward_head`,
`dspark_main_x`) never was. **That is lever B10 in §4 and it is the largest open item since F74.**
The sync time is NOT priced as waste — it is mostly the host awaiting real draft work; what it shows
is that the draft runs as ten serialised drain-points where the verify runs as one async chain, and
those drains exist *because* a raw free requires them.

**READ THE PARAGRAPH ABOVE WITH FINDING 83 ATTACHED. B10 WAS BUILT AND THE 7.4 % DID NOT EXIST.**
The draft path is now on the arena (default ON; `DSV4_DRAFT_RAW=1` restores the raw path), all 134
`cudaMalloc`/`cudaFree` and all 10 `cudaStreamSynchronize` are gone from it, the output is
bit-identical — and the round got **0.67 ms/round faster, 6.5 % of the priced 10.19**
(`evidence/clean_post_f83.log`, nine paired verifies at identical K: 1196.7 → 1190.7 ms = −0.50 %,
9/9 in one direction; spec 22.06 → 22.15). So **the 7.4 % row above is HOST time, not device-timeline
time, and 93.5 % of it was never on the critical path** — the mallocs are issued in a batch at the
top of each function while the GPU is still running the previous function's kernels, and only the
frees follow a drain. **Trap 34.** The remaining open question, which needs one profiling run and a
two-line instrument change (`g_raw_ms` currently lumps malloc and free): which half of the 134 was
the 76 µs mean, and was any of the free time absorbing side-stream (`g_side`) MoE work that
`cudaStreamSynchronize(0)` does not wait for but `cudaFree`'s implicit device sync does.

The K=5 verify dprof TOTAL is **127.2 ms** and splits into two populations that behave completely
differently (`evidence/kchunk.log`, post-F74, `DSV4_DPROF=1 DSV4_KSWEEP=1`, clocks pinned):

| group | ms | % | headroom |
|---|---|---|---|
| routed MoE — `moe:w1w3` 39.5 + `moe:w2` 19.2 | **58.7** | 46 % | **F64/F65/F70/F72 took 21 % off `w1w3` and 10 % off `w2`; still only ~76 % of roofline** |
| `cattn:ogroup` (`o:wo_a` 11.4 + `o:wo_b` 8.7 + rope) | 20.5 | 16 % | `o:wo_a` is the block's worst mark but **F76 priced it: latency-bound, both knobs at optima** — see B8 |
| `cattn:q_proj` (`q:wq_b` 8.0 + `q:wq_a` 5.8 + glue) | 16.3 | 13 % | **F74 took 29 %/14 % off these two** |
| `lm_head` | 6.6 | 5 % | little |
| `cattn:indexer` (21 layers) | 3.6 | 3 % | **F71 took 57 % off this; `i:score` is now 0.77** |
| `hc_pre` attn+ffn, `rmsnorm`, `moe:router/group/act/combine` | ~10.7 | 8 % | latency-bound, not bytes |
| `cattn:compress`, `cattn:sparse`, misc | ~3.5 | 3 % | already forked |

**The fp8 GEMM block (`q_proj` + `ogroup` = 33.8 ms, 27 %) is still the largest region outside the
MoE, and F74 proved it is not at the roofline. F78 closed the tile half and F79 closed `o:wo_a`, so
after F79 it had no untried named move; the cycle-15 research phase re-opened exactly one —
`B8-cpasync`, a ≥4-stage `cp.async` smem ring — and **F81 built it and retired it: bit-exact, 16
registers handed back, occupancy UP 4 → 5 blocks/SM, and 15–53 % SLOWER in the bench; +2 to +26 %
even when re-benched at the 16-byte alignment the engine does not have.** With that the whole fp8
GEMM block is closed on every named idea, and `open_nontraining_levers` is **0** (see §4).**

Its four marks are two different kernels: `q:wq_b`, `q:wq_a`, `o:wo_b` are the
smem-staged fp8 tile (F74 fixed their MLP; **F78 measured its double buffer at +0.28 %, the last
named move, and the mark it targeted, `o:wo_b`, went +2.8 % the wrong way**), and `o:wo_a` is
`ogroup_gemv_mk_kernel`. **F76 attributed `o:wo_a` and it is not the cheap one:** it is
*latency*-bound (8.1 of 13.0 cycles between issues on an L1TEX scoreboard, Memory Throughput 41 %,
Compute 49 %), it spills 44 bytes against a 64-register `__launch_bounds__` cap, and both of its
knobs — NR and `OGMK_BLOCKS_PER_SM` — are at measured optima. Deleting instructions from it returns
nothing, and **F79 closed the last idea for giving it registers back**: the lazy smem read moves its
spill 396 → 368 bytes at an unchanged 64 registers, because ptxas re-hoists what the source declined
to name. See §3.

**Use the K=1 column as the achievability bar, not the roofline.** At M=1 the same weight bytes go
through `fp8_gemv_m1_kernel`, so K=1 vs K=5 in one dprof report is a free, controlled measurement of
what these exact rows can do: 115 / 195 / 185 / 168 GB/s. That comparison is what found F74, and it
cost nothing but reading a column that had been printed for months.

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
| **fp8 tile stages KC K-blocks per barrier pair (was 1)** | **`q:wq_b` −28.6 %, `q:wq_a` −13.6 %, `o:wo_b` −11.2 %; K=5 verify −5.6 %; spec 20.43 → 21.68 tok/s (+6.1 %, 9/9 paired verifies)**. Bit-identical | **74** |

---

## 3. Retired with a measurement — closed, with the number that closed it

**Speculation**

| lever | what killed it |
|---|---|
| block size > 5 | identical accept sequence at 5 and 8; acceptance is flat in block size (F43) |
| draft refinement (NPASS>1) | acceptance **3.00 → 2.08** — the MTP heads are trained with the noise token as placeholder, so feeding real proposals is off-distribution (F45) |
| DDTree / tree speculation | correct but does not beat linear; depth-dominated (prior project memory) |
| **S6: suffix-automaton / prompt-lookup drafting ahead of the MTP** | **the ORACLE `max(MTP, suffix)` is +0.0 %** — 21 verifies, suffix-only 23 tokens vs MTP's 61 (**−62.3 %**), **0 wins / 17 losses**, best cascade is the threshold that never fires. Priced from one run by `DSV4_SUFFIXPROBE=1` without building the cascade. **Mechanism: speculation hands a retrieval drafter its worst possible query** — the anchor is always the *correction* token, the one the MTP just mispredicted, so it is selected to be the least repetitive token in the sequence; `mlen=0` in 13 of 21 verifies. Measured on a period-8 *degenerate repeating* decode, i.e. the best case such a drafter can be given (F80) |
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
| **any instruction-count cure for `ogroup_gemv_mk_kernel`** (WS1 scale hoist, and by the same argument the fp8x2 cvt pairing and `exp2f`→`__int_as_float(e<<23)`) | **the kernel is LATENCY-bound, so deleted instructions return nothing.** ncu: 8.1 of its 13.0 cycles between issues are L1TEX scoreboard stalls; Memory Throughput 41 %, Compute 49 %. WS1 removes NR−1 redundant scale loads and `exp2f` per k-block, is **72/72 bit-identical** (`gate_og_ws1`), and gemm_bench scored it −7.6 % at M=2/NR=2 and −14.8 % at M=3/NR=4 — in situ it is worth **+0.1 %** on the nine paired spec verifies (1219.4 → 1220.8 ms), spec 21.68 → 21.67 tok/s, every ksweep K inside ±0.5 %. Kept behind `OG_WS1=1`, default OFF. A change here must move the kernel's **memory** behaviour or it is priced at zero (F76) |
| **double-buffering the fp8 tile's staged K-chunk** (issue round n+1's global loads before round n's mma) | **bit-identical (`gate_tc_fp8_kc` 1134/1134) and +0.28 %** on the nine paired spec verifies (1219.4 → 1222.8 ms); spec 21.68 → 21.62, ksweep K=5 +0.1 %. The mark it targeted went the WRONG way at every K≥2: `o:wo_b` **+2.8 %** at K=5, matching the bench's +3.1 % in sign and size. Mechanism: the prefetch works, but one KC=2 round is 8 mma plus L1/L2-resident A loads — a few hundred cycles against a ~600-cycle miss — so it hides only part of the latency and pays a second live register array for the rest. **Unbounded it also costs 4 registers, 64 → 68, which at 256 threads is 4 → 3 blocks/SM and cost up to +38.5 % in the bench** (see the trap). Kept behind `TCB_DB=1`, default OFF (F78) |
| **B8': the `OG_SMEM` variant reading `o4[m]` LAZILY per m from shared memory** | **`ptxas -v` killed it before the bench, which is what the falsification criterion asked for.** The premise was that `float4 o4[M]` is 4M = 20 live registers at M=5 in a kernel already 8 over its 64-register cap, and that giving them back would cross an occupancy step (trap 21). ptxas says the live set barely moves: `ogroup_gemv_mk_smem_kernel<5,4>` goes **396 → 368 bytes of spill at an unchanged 64 registers (−7%, not −16 registers)**, because ptxas *re-hoists* the smem loads back to where `o4[M]` was — that is the better schedule and the source cannot forbid it. The bench agrees: at the shipped M=5/NR=4 it is **−1.1 %** against the `OG_SMEM` arm it modifies and still **+37.8 %** against the kernel that ships (0.2697 vs 0.1958 ms), and across every (M, NR) the lazy-vs-smem delta is inside ±3.4 %. Bit-identical at all 9 M (`gate_ogroup_gemv`, extended to sweep it). Kept behind `OG_SMEM_LAZY=1`, default OFF. **This was B8's last named idea, so `o:wo_a` now has none** (F79) |
| **B8-cpasync: a ≥4-stage `cp.async` smem ring for the fp8 tile** (one K-block per stage, no register staging array at all) | **the bench, by 15–53 %, after the two cheap steps both said go.** `ptxas -v` said the mechanism works — **48 registers / 36864 B / 5 blocks/SM** vs the shipped `smemB<8,2,false>`'s **64 / 17408 / 4**, i.e. 16 registers handed back and occupancy *up* 25 % — and sm_110a assembles the whole TMA family. It is **bit-exact** (`gate_tc_fp8_kc` 1512/1512, +378 new cases). Then `gemm_bench` COLD at M=5 vs `m16+smem B+4`: **wq_a +23.9 %, wq_b +38.4 %, wkv +52.8 %, wo_b +39.9 %, sw1/3 +14.9 %, sw2 +36.3 %**. Re-benched at B+0 so the ring could use **cp-size 16** instead of the cp-size 4 the engine's 4-byte-aligned weights force: recovers ~25 % (wo_b 0.1428 → 0.1067) and **still loses at 5 of 6 shapes** (wq_a +5.7, wq_b +11.7, wkv +11.1, wo_b +26.4, sw2 +19.6, sw1/3 −3.0 %). **NS=2 beats NS=4 on 4 of 6 shapes**, so depth is negative. **Occupancy rose and it lost anyway** — at 4 blocks/SM the kernel was already saturating DRAM, so the 5th block only adds contention. Default OFF behind `TCB_CPA` (F81) |
| hoisting a loop-invariant **pointer** in a register-starved kernel | not a lever, a trap, but it belongs here with its number: hoisting `wsc + (gr0/128)*scw` out of the k-loop took `<5,4>` from **44 to 60 bytes of spill** and cost 4.4 %. Two permanently-live registers beat three short-lived ones only when there are registers to spare. A 32-bit offset restored it (F76) |

**Whole hypothesis classes**

| hypothesis | what killed it |
|---|---|
| working-set size costs bandwidth (TLB/SMMU reach) | `footprint_probe`: 230–246 GB/s from 0.5 to 64 GiB, both allocators (F55) |
| the bench-vs-in-situ gap is launch overlap | `gemm_bench`'s `timeit` launches on **stream 0**, which is stream-ordered. It overlaps nothing. The gap was single-kernel under-saturation (F55) |
| the indexer fault is a length bug / an allocator accumulation | `:91` and `:96` are the **same `cudaStreamSynchronize`**, the first real sync in the layer — never a location. Free memory *rises* 10.36 → 11.61 GiB across 17 points. Three faithful reruns 17/17 clean (F58) |
| the nondeterminism is the concurrency / atomics / stale window / uninit scratch / caches | all five eliminated with measurements; the real cause was F62 (F60, F61, F62) |

---

## 4. Open — base AR decode (12.56 tok/s, 79.6 ms/tok)

> **START FROM `wiki/roofline-why-the-needle-wont-move.md`, NOT FROM THIS SECTION'S ARITHMETIC.**
> F137 corrected the MoE rate (155 → 195 GB/s, **94 % of roofline**, the old figure having divided
> routed bytes by a window that was also streaming the forked shared expert) and F138 took apart the
> last unexamined region. The whole K=1 step is **160 GB/s = 77 % of a measured 208.7**, and the
> remaining 23 % is four regions at 82-94 % plus 8.53 ms that is not bandwidth at all:
>
> | region | ms | % step | GB/s | % roofline |
> |---|---|---|---|---|
> | MoE (routed + shared, concurrent) | 23.20 | 33 % | **195** | **94 %** |
> | Attention (MLA + compressor + indexer) | 31.40 | 45 % | 172 | 82 % |
> | `lm_head` | 5.80 | 8 % | 183 | 88 % |
> | HC / rmsnorm / router / glue | 8.53 | 12 % | 25 | latency |
>
> **B7/B7' below are retired by F137** — the MoE GEMV is at the roofline at M=1 and 155 GB/s was
> never its rate. **B4 is confirmed and sized by F138**: the glue is 6.2 µs/call of a launch floor
> that is *flat in grid size* (2.07 µs at both `<<<1,32>>>` and `<<<24,256>>>`) plus 14.5 µs of a
> 20-iteration serial Sinkhorn on a 4x4 matrix. It is also **near-flat in K** — 12.1 % of the step at
> K=1, 8.2 % at K=5 — so at tau = 3.736 speculation has *already* removed 67 % of it. The one
> bit-exact move left is fusing `hc_pre`'s three launches cooperatively: **+0.5 %, untried**.

> **19.0 IS A NORMALISATION CONSTANT, NOT A TARGET — computed 2026-08-08, `tools/byte_floor.py`.**
> It assumes every kernel moves bytes at full DRAM bandwidth AND that the non-byte part of the step
> is zero. Neither holds. Byte-weighting the K=1 dprof marks against shapes read from the
> checkpoint's own `config.json` (self-checked: it reproduces this file's hand-derived 115/195/185/168
> to within 2.1 %) gives **22.3 ms of the 71.4 ms step that is not bytes at all** — launch floors,
> Sinkhorn, activations, the latency-bound `ogroup` of F76 — and a byte-mark average of **191 GB/s,
> not 233**. The only mark above the probe is `moe:w2` at 246 GB/s, which is L2 reuse of the expert
> set `w1w3` just touched, not a kernel beating DRAM, so it is excluded as a target rate.
>
> | floor | ms/tok | tok/s |
> |---|---|---|
> | every byte mark at 233 GB/s (optimistic) | 62.6 | **15.98** |
> | every byte mark at 198 GB/s (`q:wq_b` cold, realistic) | 69.8 | **14.33** |
> | now | 71.4 | 14.01 |
>
> **Total remaining kernel headroom on base AR: 2.2 % realistic, 12.3 % optimistic** — against 37 %
> implied by 19.0. Scaled by acceptance the spec ceiling is **23.2–25.9 tok/s, not 30.8**.
> Caveat that cuts the other way: the model covers 77 % of `B_tok`; the missing 23 % (indexer,
> compressor, norms, embed) is charged at measured time as if fixed, so these are pessimistic on
> that axis. **F83 is the independent confirmation** — B10 was the largest remaining item, priced at
> +5–7 %, and delivered +0.41 %. Two unrelated methods now agree the kernel path is done.

| # | lever | expected | why it might work / what to watch |
|---|---|---|---|
| ~~B0~~ | ~~audit every kernel for thread-per-output at decode shapes~~ | **NAMED LIST EXHAUSTED, F71 + F73** | `index_score` **+4.4 %** (F71). `k_moe_prefix` and `k_build_tiles`, both `<<<1,1>>>` over `nr=160`, fixed in F73: `moe:group` −20.3 %, and both now sit at the **launch-latency floor** (0.24–0.25 ms / 43 layers, vs 0.19 for a trivially parallel neighbour). The rest are measured under B0's own 0.5 ms bar: `k_topk_verify`/`k_topk_decode` → `i:topk` = **0.12 ms**; `k_dg`/`k_advance_T`/`k_incr` are genuinely one scalar's work. **The class paid twice and is now dry** — reopening needs a *new* kernel with a bad geometry, not another pass over these. |
| ~~B8~~ | ~~the fp8 GEMM block~~ | **FULLY CLOSED, F78 + F79** | **F78 closed the tile half; F79 closed `o:wo_a`.** What remains is `cp.async` staging (below), which is a different kernel, not a tuning of this one. Original note kept: **F78 closed the one named move on the three tile marks.** Double-buffering the staged chunk is bit-identical and worth **+0.28 %** in situ, with `o:wo_b` **+2.8 %** — the wrong way, in agreement with the bench. With F67 (both reroutes), F68 (split-K, numerics) and F74 (the win that was there), **the tile has no untried idea left except `cp.async` staging global→shared**, which removes the register array entirely but must clear the same occupancy bar F78 failed. What remains of B8 is the single item below on `o:wo_a`. |
| ~~B8'~~ | ~~`o:wo_a`: the `OG_SMEM=1` variant reading `o4[m]` LAZILY per m from shared memory~~ | **RETIRED, F79** | `ptxas -v` answered before the bench, exactly as the criterion demanded: spill **396 → 368 bytes at an unchanged 64 registers**, a −7 % move where ~16 registers were predicted, because ptxas re-hoists the smem loads back to where `o4[M]` was. Bench: **−1.1 %** vs the `OG_SMEM` arm, **+37.8 %** vs the shipped kernel. See §3. **`o:wo_a` now has no untried named idea at all**, and with it B8 is fully closed. |
| B8'' | **M-gate `OG_SMEM` on: F55's 40 % kill was measured at M=5 ONLY, and the sign flips below M=4** | **~0.4 % — SUB-1 %, does not count toward the pivot queue** | New evidence, not a new argument (F79). F55 swept NR at M=5 and retired the variant on that one column. Sweeping **M** as well, same COLD harness, 3 alternating replicates (`evidence/oglazy_bench.log`), against the NR the dispatch actually picks at each M: **M=2 0.1717 → 0.1533 (−10.7 %), M=3 0.1962 → 0.1729 (−11.9 %), M=5 0.1958 → 0.2729 (+39.4 %, F55's number reproduced), M=8 0.3610 → 0.2810 (−22.2 %)**. So `ogsmem` gated on `bs<=3` is a real candidate. **Priced honestly it is small**: `o:wo_a` is 10.74 ms of an 85.9 ms K=2 verify and 10.26 of 104.7 at K=3, and only 4 of the 9 verifies in a canonical run are K≤3 → **~0.4 % end-to-end**, before the trap-3 discount that F76 (bench −7.6 %→ in situ +0.1 %) and F78 (bench +3.1 % → in situ +2.8 %) both demand. M=4 is unmeasured — the bench skips it and it is on the crossover. |
| ~~B8-tile~~ | ~~the three tile marks `q:wq_b` + `o:wo_b` + `q:wq_a` = 22.5 ms~~ | **CLOSED, F78** | Kept for the achievability bar it established: **the target was never 233 GB/s** but the M=1 GEMV's own measured rate on the same weight bytes — `q:wq_a` 115, `q:wq_b` 195, `o:wo_b` 185, `o:wo_a` 168 GB/s (K=1 column, `evidence/kchunk.log`). **Bytes are counted (F74) and it is NOT at roofline.** The target is not 233 GB/s — it is the M=1 GEMV's own measured rate on the same weight bytes: `q:wq_a` 115, `q:wq_b` 195, `o:wo_b` 185, `o:wo_a` 168 GB/s (K=1 column, `evidence/kchunk.log`). Against that the three tile marks hold ~6.5 ms. **The one remaining move on them is to double-buffer the staged tile so round n+1's loads issue before round n's mma** — F67 closed both reroutes, F68 closed split-K on numerics. **`o:wo_a` (3.3 ms of the old estimate) is NO LONGER a cheap 7 %: F76 attributed it.** `ogroup_gemv_mk_kernel<5,4>` is *latency*-bound (8.1 of 13.0 cycles between issues on an L1TEX scoreboard), spills 44 bytes against a 64-register cap, and sits at a **measured local optimum in both knobs** — NR (1/2/4/8) and `OGMK_BLOCKS_PER_SM` (2/3/4, re-swept in F76 with the spill reduced). Instruction-count cures are retired as a family (§3). The only untried idea that moves ~16 registers instead of 3: **the `OG_SMEM=1` variant reading `o4[m]` lazily per m from shared memory** instead of holding M float4s live — smem makes lazy reads affordable where global loads need the MLP. F55 measured that variant a 40 % regression at NR=4 *with* the 20-register `o4[M]` still in it, so the register argument was never tested; falsify with `ptxas -v` before building anything. |
| ~~**B8-cpasync**~~ | ~~`cp.async` staging global→shared for the fp8 tile with a ≥4-stage smem ring~~ | **RETIRED, F81 — bench +15 to +53 %, and +2 to +26 % even at ideal alignment** | Built, **bit-exact** (`gate_tc_fp8_kc` **1512/1512**), and it **passed both cheap falsification steps and then lost the bench by a mile**. (1) `ptxas -v`: at UF=1 the ring is **48 regs / 36864 B smem / 5 blocks/SM** against the shipped `smemB<8,2,false>`'s **64 regs / 17408 B / 4** — the register array really is given back and occupancy *rises*; smem was never the binding resource (233 472 B/SM, the shipped kernel uses 30 % of it at 4 blocks). (2) `cp.async.bulk.tensor.2d`, `cp.async.bulk`, `mbarrier.init` and `mbarrier.try_wait.parity` all **assemble at `-arch=sm_110a`**. (3) `gemm_bench` COLD, arms adjacent, M=5 vs `m16+smem B+4`: wq_a **+23.9 %**, wq_b **+38.4 %**, wkv **+52.8 %**, wo_b **+39.9 %**, sw1/3 **+14.9 %**, sw2 **+36.3 %**. **The disambiguation is the valuable half**, because the engine's weights are 4-byte aligned so the ring had to use cp-size 4: re-benched at B+0 (cp-size 16) it recovers ~25 % (wo_b 0.1428 → 0.1067) **and still loses to the ordinary LDG path at 5 of 6 shapes** — wq_a +5.7 %, wq_b +11.7 %, wkv +11.1 %, wo_b **+26.4 %**, sw2 +19.6 %, sw1/3 −3.0 %. **So no alignment work rescues it**, which is what makes this a permanent close rather than a block on F67's shard pad. And **NS=2 beats NS=4 on 4 of 6 shapes** (wo_b 0.0873 vs 0.1067), so *depth is negative* — the direct refutation of the promoted claim. Mechanism, new trap 29: one ring stage is **1** K-block, so `wait_group`+`__syncthreads` runs **once per K-block** where the shipped KC=2 pays it once per two — and F74's win *was* raising bytes-per-barrier. The ring trades the one thing that pays for bytes-in-flight the LDG path already had. Not run in situ: the falsification order says stop at the first no, and trap 3's discount has never turned a large bench negative into a win (F76 −7.6 % → +0.1 %, F78 +3.1 % → +2.8 %). Kept behind `TCB_CPA=<stages>`, default OFF. See §3. |
| ~~**B10**~~ | ~~put the DSpark DRAFT path on the existing `dscratch.h` arena~~ | **BUILT AND SHIPPED, but the +5 to +7 % is RETIRED — F83 measured 0.67 ms/round recovered against a predicted 10.19, i.e. 6.5 % of the target. Real value ≤0.5 %, SUB-1 %, does NOT count toward the pivot queue** | Built exactly as specified and it is **bit-identical**: `evidence/clean_post_f83.log` reproduces `clean_post_f81.log`'s spec tokens, base-AR tokens, all 45 drafter margins, every K and every accept count, acceptance 2.89, GATE + MATCH 5/5 + LOSSLESS GATE PASS. All 134 raw `cudaMalloc`/`cudaFree` and all 10 `cudaStreamSynchronize` are gone from the draft path (`dkmalloc`/`dkfree`/`dksync` in `dscratch.h`; `DSV4_DRAFT_RAW=1` restores the raw path exactly). **And it bought 0.50 %.** Nine paired verifies at identical K: 1196.7 → 1190.7 ms, **9/9 in one direction** but only **−0.50 %** (−1.29/−0.09/−0.15/−0.26/−0.42/−0.94/−0.84/−0.32/−0.28 %); spec 22.06 → **22.15** (+0.41 %). **The kill number needs no timing resolution at all: F82 priced 10.19 ms/round of allocator host time; the whole round got 0.67 ms/round faster. 93.5 % of the priced time was never on the device timeline.** The lever is KEPT because it is free and never worse (9/9 direction, and the three draft-free control marks — cold prefill 206.1 → 206.9, base AR 72.3 → 72.6, warm prefill 137.1 → 137.9 — all moved the *other* way, so the pair drawn was a slow one), but it is now a sub-1 % item. **The lesson is trap 34: HOST time inside a driver call is not device-timeline time, even when the device is drained.** |
| B1 | **more fork sites** (§5 pricing table) | ~0.3–1 %/pair | The three obvious independent chains are taken. Remaining pairs are small; check the partner is *not* already saturated or the gain collapses. |
| ~~B2~~ | ~~split-K on the small-N GEMMs~~ | **RETIRED, F68** | 512 rows = 32 m16 tiles = 32 blocks on 20 SMs. **Not bit-exact** — a K-split reduction changes accumulation order, so it needs a tolerance gate, not an equality gate. |
| B3 | **fuse `wq_a`+`wkv` into one launch** | ~0.5 % | Combined N = 1536 is still only 192 warps, so it barely moves `N/8` (F67). And `wkv` is already forked to a side stream by C1 and fully hidden (`q:kv_join` = 0.05 ms), so there is nothing left to overlap. Low value now. |
| B4 | ~~fuse the elementwise glue~~ | **≤0.4 %, killed (F67)** | Moves almost no bytes; pure launch/latency floor. The verify-graph result (1.05x) caps what graphing can return here. Sinkhorn is *already* one fused kernel — do not "fuse" it again. |
| B7 | **occupancy of the MoE GEMV** | smaller than it looks | The RB sweep found the shipped RB=2 optimal, and the `OGMK_BLOCKS_PER_SM` register-cap knob is also already at its optimum (BPS=4/NR=4 = 0.2044 ms beats 2, 3 and 6 at every NR). Both knobs are exhausted. |
| B7' | ~~raise MoE GEMV occupancy via the register cap~~ | **exhausted (F67)** | Even at the optimum RB=2 it runs 62 registers / 63 % occupancy / 155 GB/s = 67 % of roofline. The register budget is dominated by the funnel pair (B6) and `acc[RB][BN]`. B6 is the cheapest way in. |
| B5 | **FP4 for the MLA/dense weights** | moves the byte floor ~1.2x | Assessed and **not done**: it is pure weight transform, trivial on Thor, but costs accuracy and the constraint is *no additional quantization*. Requires an explicit decision to relax that. |

## 5. Open — speculation (18.13 tok/s, acceptance ~2.9 of 5)

`tok/s = tokens_per_cycle / cycle_ms`. The cycle is verify **82.6 %** + draft **17.4 %** (F82,
`evidence/specprof_f82.log` — the 86/14 quoted here until now dated to `evidence/f47.log` and was
stale by six verify-side adoptions). Since the verify's MoE half is near the roofline, **the
numerator is the better lever than the denominator** — but the denominator is no longer exhausted:
~~**7.4 % of the cycle is raw-allocator host time in the draft half, see B10 in §4.**~~ **F83 removed
all of it and recovered 0.50 %.** The 7.4 % was HOST time, not device-timeline time — trap 34.
The draft half's remaining cost is real GPU work: 3 MTP blocks 13.47 ms + `fwd_head` 10.13 ms.

| # | lever | expected | why it might work / what to watch |
|---|---|---|---|
| S1 | ~~re-fit the adaptive-width threshold~~ | **DONE, F63** | Re-run post-F62: adaptive is worth **+9-11 %** on the three prompts where it engages and a wash on the other two — not the +28 % prompt 2 showed on the garbage prefill. The threshold **1.0 / 1.5 / 1.75 remain not separable** (means 16.94/16.70/16.79 against a 15 % within-cell spread). 1.5 stays. Do not re-run without a reason to expect >5 % from a threshold move. |
| **S5** | **fine-tune the DSpark MTP draft head** | **the largest lever in the project: +24 % at acceptance 3.6, +46 % at 4.0** | **Recipe now costed** (dossier, 2026-08-08): FastMTP (arXiv 2509.18362) recovered per-position acceptance 70/10/~0 % → **80/56/36 %** at k=1/2/3 with a *position-shared* head, 210.8M params (<3 % of backbone), **3 epochs on 389.4K self-distilled samples**, lifting speedup 1.21× → 2.03×. Nebius did the same for V3-0324 in **1 epoch on 660K on-policy pairs**. LK-Losses (arXiv 2602.23881) targets acceptance directly instead of KL and adds **+5.6 %** acceptance-length. Tess-4-27B reports an inherited head after backbone surgery driving acceptance to **~0 %** — our REAP prune + MXFP4 is a larger shift than a fine-tune, so 2.90/5 is consistent with an *unaligned* head, not a ceiling. Data must be regenerated **by our own pruned checkpoint** (on-policy), coding/agentic-heavy. Acceptance is 2.90 of 5 and multiplies the whole cycle. **It is LOSSLESS BY CONSTRUCTION**: the verify corrects every draft, so a better draft cannot change the output — only the speed. That makes it the only large lever here with *zero* quality risk, unlike F68. Feasible on-device in two phases: (1) run the frozen backbone forward and cache the teacher signal at the layer-40/41/42 taps — pure inference, the memory profile we already run; (2) train the head alone with the backbone unloaded. Only the non-expert ~0.55 GiB of the head is trainable on-device, which is the part that matters. Needs a training loop, not a kernel change — the largest build cost of anything open. |
| ~~S6~~ | ~~suffix-automaton / prompt-lookup drafting, ahead of the MTP~~ | **RETIRED, F80 — the oracle ceiling is exactly +0.0 %** | Executed at its own stated falsification (*"price it as acceptance, not as draft time saved"*), which is also the only way anything is resolvable under trap 25: acceptance is a **counted integer** and has no variance floor. `DSV4_SUFFIXPROBE=1` priced it from ONE checkpoint load without building the cascade. Over 21 verifies: MTP **61 tokens / 2.905 per verify**; suffix-only **23 / 1.095 (−62.3 %)**, sound bound and optimistic estimate coinciding; **ORACLE `max(MTP, suffix)` = 61 = +0.0 %**; **0 wins, 17 losses**; a match existed in only 8/21 and `mlen` never reached the block size, so the best cascade threshold is the one that never fires. **Mechanism (new trap 27): the anchor is selected against you.** The token a draft starts from is always the *correction* — the one token the MTP just mispredicted — because whenever the sequence is predictable the MTP accepts straight through it. So the drafter is always queried at the least repetitive point: `mlen=0` in 13 of 21 verifies, and where a long match did exist it only ever proposed the previous cycle's value of the one slot that varies (`16603` vs `14251`, `29585` vs `25062`, `3702` vs `17705`). Measured on a **period-8 degenerate repeating decode** — the most favourable input such a drafter can be handed. **Reopening needs a long-repeated-context prompt** (the shape of `scripts/prompt_suite.json`'s `longctx_001`, `filler_repeats: 400`, ids from `tools/encode_prompt.py`) on which `mlen` routinely reaches the block size — that is the agentic regime SuffixDecoding actually reports, and this measurement does not refute it, only S6 on the workload every number here is measured on. |
| **S7** | **mHC-aligned block drafter (HyperDFlash-style)** | τ 2.93 → 3.69, 2.25× → 2.80× *(paper, small-batch)* | The only published drafter built for **this** architecture: conditions on the target's **pre-collapse mHC residual states** and inherits the `hc_head` gate as a **65K-param reducer** (vs 67M for a generic linear one). We already have every piece it needs — `hc.cu`, `hc_sinkhorn.cu`, and the `hc_head` collapse — and the pre-collapse states are what our `dspark_tap_pool` already reads at layers 40/41/42. **Fallback if S5 plateaus, not before**: it is a larger build than re-aligning the existing head, and the paper reports **no batch-1 number and no named hardware**, so treat 2.80× as a small-batch-favourable upper bound. |
| **B9** | **optimise the PREFILL path** *(NOT a decode lever — see the note below)* | capture 3.4 days -> hours; also TTFT | Prefill measures **52.6 / 50.3 / 47.7 / 43.8 tok/s** at PS=255/511/1023/2047 — only **3.4x** the M=1 decode rate, when a batched forward that amortises 12.26 GB of weights over 255 positions should be compute-bound and far faster. Every kernel in the path was tuned for M=1: `RB` is fitted to a 1.71-rows-per-expert histogram that prefill does not have, the grouped GEMV is chosen over the mma GEMM on a decision that inverts at prefill row counts (the GEMM lost 416 vs 783 us at M=1 — at PS=255 the arithmetic intensity is ~150x higher), and `moe:group` runs per layer regardless. **First measurement to take: a `DSV4_DPROF` profile of a PS=1023 prefill** — nobody has ever looked at where prefill time goes. |
| S2 | **raise acceptance** | the biggest single multiplier | Acceptance 2.9 → 3.5 is +21 % at constant cycle. But the two obvious routes are dead: block size (F43) and draft refinement (F45). This needs a *different* idea — the MTP heads are what they are, and retraining is out of scope. |
| S3 | **cascade / early-exit verify** | probably negative | Verifying 1–3 then 4–5 reads the 8.81 GB fixed weight set **twice**. That duplicated fixed cost (~92 ms) swamps the expert-union saving. Priced and rejected on arithmetic; do not implement without a byte model that beats it. |
| S4 | **expert-overlap-aware draft selection (EcoSpec-style)** | bounded by the measured union | Choose drafted tokens by acceptance × **expert reuse**, and dedup the union before the grouped GEMM. Named prior art: EcoSpec (arXiv 2607.12696), AcceptMoE (2608.02989); Cohere measures 20-31 % of the union reclaimable from temporal routing correlation. **Our own measurement bounds the prize**: the K=5 union is already **17.53 of a possible 30**, i.e. the five positions share 42 % of their experts before any optimisation, so the remaining reclaimable fraction is smaller here than the papers' baselines. |
| ~~S4-old~~ | ~~reduce the expert union at K=5~~ | superseded by the row above | The union is **17.53** measured, not 29.9 — the five positions already share most of their experts, so there is far less to win here than the old model implied (F64). This is the largest single block of time in the engine and the only untried structural idea against it. Any scheme must keep the verify *exact*: biasing the **draft** toward expert-overlapping tokens is lossless by construction (the verify still corrects), but it would cost acceptance, so it is a trade, not a free win. |

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
15. **A profile you already have may contain its own control.** B8 sat open for cycles as "unknown;
   count the bytes". The bytes never needed counting — `q:wq_b` at K=1 (195 GB/s) and at K=5
   (123 GB/s) move the *same* weight bytes through *different kernels*, and that ratio was printed in
   every dprof report since the K-sweep existed. Before building an instrument, check whether an
   existing report already varies the one thing you want varied (F74).
16. **A gate harness that passes the wrong argv reports a failure that is about the harness.** Running
   `build/gate_*` in a loop with a shared `ref/goldens` argument makes `gate_prefill_len` read it as a
   sweep length, sweep s=0 and print **GATE FAIL** with eight "invalid argument" launches. Only
   `gate_units` and `gate_encoding` take a path. Also: `build_gate.sh` builds 10 of the 17 binaries —
   the other 7 go stale silently and must be rebuilt from the build line in their own headers (F74).
17. **Line numbers in CUDA error messages are where the error was *collected*, not where it happened.**
   `dsync` is a no-op under the arena, so the first real sync in a layer absorbs ~20 launches' worth of
   asynchronous faults. Use `DSV4_SYNCPROBE=1`.
18. **A kernel can be sub-roofline because it is LATENCY-bound, and then instruction count is free
   to cut and worth nothing.** `ogroup_gemv_mk_kernel` runs at 120 GB/s against the M=1 path's 168 on
   the same bytes, which reads as headroom. ncu says 8.1 of its 13.0 cycles between issues are L1TEX
   scoreboard stalls at 41 % Memory Throughput and 49 % Compute — it is parked on memory. F76's cut
   (NR−1 redundant scale loads and `exp2f` per k-block, bit-identical, −7.6 % to −14.8 % in
   `gemm_bench`) measured **+0.1 %** on the nine paired verifies. **Before optimising a sub-roofline
   kernel, read the stall reason, not just the achieved GB/s** — and note gemm_bench sees these wins
   precisely *because* a standalone launch with nothing to contend with is issue-limited and the
   in-situ launch is not. This is trap 3 with a mechanism attached.
19. **In a register-starved kernel, hoisting a loop-invariant *pointer* is a pessimisation.** A
   64-bit pointer live across the loop costs two permanently-allocated registers; the three
   short-lived loads it replaced cost less. `<5,4>` went 44 → 60 bytes of spill and lost 4.4 %. A
   32-bit offset fixed it. `ptxas -v` answers this in one recompile and should be read before the
   bench (F76).
20. **A gate whose `extern` declaration drifts from the engine fails at LINK time, which looks like a
   build problem rather than a gate that stopped testing anything.** `gate_fp4_gemv` had been two
   parameters behind `tc_fp4_grouped_gemv_e8m0` since F65/F72 while a stale binary sat in `build/`
   looking like a pass. `build_gate.sh` now builds all 19 binaries so a gate cannot go quiet (F76).

21. **Four registers can cost 25 % of your occupancy, and the step is invisible in the source.** The
   fp8 tile's double buffer takes `<8,2>` from **64 to 68** registers. At 256 threads/block that is
   65536/(256·64) = 4 blocks/SM versus 65536/(256·72, the 8-register granularity) = 3 — and the
   bench went **+38.5 %** on `wo_b` until `__launch_bounds__` put the 4 back (then +3.1 %). Compute
   the step, do not eyeball the delta: what matters is which side of `regfile/(threads·regs)` the new
   count lands on, not how many registers were added (F78).
22. **`__launch_bounds__(T, 1)` is not a no-op — it RAISES register use.** Adding it to a shared
   template to constrain one instantiation took the *other* instantiation from 64 to **84**
   registers, because `minBlocks=1` tells ptxas one resident block suffices and it stops economising.
   That instantiation was the control arm of the A/B being run. Put the body in a `__device__`
   function and give only the variant that needs the bound its own `__global__` entry, then re-read
   `ptxas -v` for BOTH (F78).

23. **You cannot free a register by not writing it down.** B8' proposed deleting a `float4 o4[M]`
   array and re-reading the same value from smem inside the loop that consumes it, to give ~16
   registers back to a kernel 8 over its cap. ptxas moved **396 → 368 bytes of spill at an unchanged
   64 registers**: `sh` is not written in that loop, so hoisting the loads back to exactly where the
   array had been is legal *and* is the lower-latency schedule, and ptxas does it. **Source-level
   register hints only bind when the transform changes what the compiler is ALLOWED to do** (an
   aliasing fact, a barrier, a different address). Trap 19 is the same lesson from the other
   direction: there, the source forced two registers to be live and ptxas could not undo it. Rewrites
   that merely *suggest* fewer live values are free to be ignored, and are (F79).
24. **A kernel retired on one shape's sweep is retired on one shape.** F55 killed `OG_SMEM` on a
   NR sweep taken at **M=5 only**, and the note has read "40 % regression" for eight findings. Adding
   the **M** axis to the same harness flips the sign: −10.7 % at M=2, −11.9 % at M=3, +39.4 % at M=5,
   −22.2 % at M=8. Nothing about the old measurement was wrong; it was one column of a two-axis
   table. **Before re-deriving a retired lever's mechanism, check which axes its killing measurement
   actually varied** — that is cheaper than any new idea, and here it cost one bench run (F79).
25. **Two CLEAN runs of the SAME binary differ by 1.5 % on the paired-verify instrument, in one
   direction.** F79 shipped default OFF, so its clean run measured the identical default arm as
   `clean_post_f76.log`, and the nine verifies pair 1:1 at identical K and accept counts for
   **1212.4 → 1194.7 ms (−1.5 %), all nine negative**; spec 21.76 → 22.07. Nothing changed. The
   "±0.8 % own spread" quoted in F76/F78 is the *within*-pair figure.
   **AMENDED BY F81, which supplied the second pair this trap was missing.** F81 also shipped default
   OFF, and f79 → f81 is **+0.17 % over the nine paired verifies, only 2/9 in one direction** (spread
   +0.00 % to +1.21 %), spec **−0.05 %**, base **+0.36 %**, with the spec token sequence
   byte-identical. So ~1.5 % is **not** a per-run noise band — the instrument reproduces to ~0.2 % when
   it behaves, and the F79 pair was an occasional **systematic shift** of everything in one direction.
   The rule to work by: a **single** cross-run comparison under ~1.5 % is still unsupportable, because
   you cannot tell from inside one pair which kind you drew — but a sub-1 % effect **is** resolvable
   with **two** pairs, which is the only route left to anything in §4 and was not obvious from F79
   alone (F79, F81).
26. **Do not pass one argument to every gate binary.** Running the suite as `for g in build/gate_*; do
   $g ref/goldens; done` produces **four** red gates that are all harness artefacts: `gate_encoding`
   cannot open `ref/goldens/test_input_1.json`, and `gate_compressed_decode`, `gate_indexer_decode`
   and `gate_prefill_len` take argv[1] as a *shape/offset* and go to an illegal memory access or a
   CUDA-error stage count on a path they were never meant to run. All four pass with **no
   arguments**; only `gate_units` wants `ref/goldens`. FLYWHEEL_STATE recorded two instances of this
   in cycle 14 and it recurred at four in cycle 15, so it is a standing property of the suite, not a
   one-off. **A red gate must be re-run on its own default before it is believed** — and equally,
   this must never be used to wave away a gate that is red on its own defaults (F80).
27. **The drafting anchor is selected against a retrieval drafter.** In speculative decoding the token
   a draft starts from is *always* the **correction** — the one token the draft head just
   mispredicted — because wherever the sequence is locally predictable the head accepts straight
   through it and the next draft begins after the surprise. So any drafter that conditions on the
   *surface* of the recent context (n-gram, prompt-lookup, suffix automaton) is systematically queried
   at the **least** repetitive point in the sequence: `mlen = 0` in 13 of 21 verifies, and the oracle
   ceiling over 21 verifies was exactly **+0.0 %** (F80). **Do not price a drafting idea on how
   repetitive the OUTPUT looks** — price it on how repetitive the sequence is *at the positions
   speculation actually asks about*, which is a different and much smaller set.
28. **Price what you can as a counted integer, and prefer a ceiling to a point estimate.** F79 left
   the end-to-end instrument with a ~1.5 % cross-run floor (trap 25), below which nothing in §4 is
   resolvable. Acceptance has no such floor — it is an integer count — and a **read-only
   counterfactual** probe alongside the shipped path priced a whole lever from ONE checkpoint load
   without building it (F80, and F67 before it). Measuring the **oracle** over both arms rather than
   one arm's point estimate is what made the result final: a ceiling of zero closes a lever, where a
   small negative would only have invited a better gating rule.
29. **On this fp8 tile the binding resource is BARRIERS PER BYTE, not bytes in flight — and F74 already
   said so.** F74's win was staging KC K-blocks per barrier pair instead of 1, i.e. *more bytes per
   barrier*. B8-cpasync built a 4-stage `cp.async` ring at **one** K-block per stage, which doubles
   the barrier count back to the pre-F74 rate while adding bytes in flight, and lost **15–53 %** in
   the bench — with **NS=2 beating NS=4 on 4 of 6 shapes**, so the depth axis is negative. A stage is
   64 rows x 128 B = 8 KB over 256 threads; there is not enough work per stage to amortise a
   `wait_group` plus a `__syncthreads`. **Before adding pipeline depth to a kernel, check which of
   (bytes per barrier, barriers per byte) its last adopted win moved, and do not move it back** (F81).
30. **More blocks/SM is not more throughput once the kernel already saturates DRAM.** The `cp.async`
   ring genuinely bought occupancy — `ptxas -v` says **48 registers vs 64, 5 blocks/SM vs 4**, a 25 %
   gain, exactly the bar F78 failed — and it was **still 15–53 % slower**. The shipped kernel at 4
   blocks/SM was already at 199 GB/s on `wo_b` against a 233–240 GB/s achievable; the 5th block adds
   contention, not bandwidth. Occupancy is a *means*, and trap 21's arithmetic tells you when you have
   lost it, not that winning it pays. **An occupancy win still has to be measured as time** (F81).
31. **`cp.async` at cp-size 4 costs ~25 % against cp-size 16, and this engine's weights force
   cp-size 4.** cp.async requires the copy size to equal the natural alignment, and F66 counted the
   fp8 weights at `data_offset % 16 == 8` (43,470 of 44,436) or 12 (966), **never 0**. The same
   alignment fact that made the `uint4` staging path need an `AL16` template makes the async path use
   eight 4-byte DMAs where it wants two 16-byte ones: `wo_b` M=5 is **0.1428 ms at cp-size 4 vs
   0.1067 at cp-size 16**. Measuring both was what turned F81 from "blocked on alignment" into a
   permanent close, since cp-size 16 **still lost**. **Any future `cp.async`/TMA idea here inherits
   this alignment tax and must be priced at B+4, not B+0** (F81).

32. **A `DSV4_SPECPROF` run is INVISIBLE to the harness's clean/profiling classifier, and it is not a
   clean run.** `scripts/flywheel.sh` decides whether a cycle's run was a re-baseline by grepping
   `^\[dprof\]`; `DSV4_SPECPROF` prints `[specprof]`, so a contaminated run scores as clean and would
   overwrite `baseline.spec_tok_s` with a number no clean run produced. It **is** contaminated: five
   `cudaEventRecord` on the null stream per round cost **+1.01 % on the nine paired verifies, 9/9 in
   one direction** (1196.7 → 1208.8 ms), spec 22.06 → 21.84, base 13.83 → 13.72 — a known cause with a
   consistent sign, which is exactly what an instrument's cost should look like. Set
   `dprof_runs_since_clean` by hand after one, and never re-baseline from one (F82).
   **FIXED 2026-08-08 in the harness, and the fix is wider than the bug.** `flywheel.sh` now matches
   `^\[(dprof|specprof|ksweep|blksweep|memtrace)\]` — five instruments, enumerated from the source of
   truth (`grep -rhoE '"\[[a-z]+\]' include/dprof.h src/*.cu`), not the one that happened to bite.
   A/B'd on real logs: `specprof_f82.log` flips CLEAN→DIRTY, `clean_post_f81.log` and
   `clean_post_f76.log` stay CLEAN. **The hand-set was never going to hold anyway** — the counter
   block runs *after* the executor writes `FLYWHEEL_STATE.json`, so cycle 17's correct hand-set 1 was
   clobbered back to 0 by the harness thirty seconds later, and the state note that documented the
   workaround described a value that no longer existed in the file it described. That is the general
   lesson: **a documented manual workaround for a harness bug is not a mitigation if the harness runs
   last.** The executor cannot patch this itself (`scripts/flywheel*.sh` is in its deny list), so
   harness bugs it *finds* must be escalated to the operator rather than worked around in state.

33. **A region's SHARE of the cycle goes stale even when nobody touches it.** The draft half was 14 %
   at F47 and is **17.4 %** now, unchanged in absolute terms while six adoptions cut the verify around
   it — so §1's table, which is a *verify* table, was being read as a *cycle* table and the draft was
   never a candidate. Re-measure the top-level split whenever the thing below it has moved (F82).
34. **HOST time inside a driver call is NOT device-timeline time, even on a drained device.** F82
   measured 10.19 ms/round of host time inside the draft path's `cudaMalloc`/`cudaFree` — a correct
   measurement — and priced it at 7.4 % of the cycle because 127 of the 134 calls follow their own
   function's `cudaStreamSynchronize`. F83 removed **all 134 of them plus all 10 syncs**, bit-identical,
   and the round got **0.67 ms faster: 6.5 % of the prediction, a 15x miss** (`clean_post_f83.log`,
   9/9 paired verifies, −0.50 %). The flaw is that "the device was drained when this call *started*"
   does not imply "the device was idle for the duration of this call": each function issues its whole
   malloc batch at the top, and the kernels of the *previous* function are still draining underneath.
   **The instrument you want is the gap between two device events, not a `steady_clock` bracket around
   a host call.** F82's own refusal to price the 12.99 ms of `cudaStreamSynchronize` as waste was the
   right instinct applied to only half of the data.

---

### 6b. Long-run pipeline traps — the S5 class, and the one preflight that closes it

Every S5 failure was in a code path that only executes hours into a run. None were hard bugs; all
were cheap to find and expensive to hit.

| # | what failed | when it surfaced | why no gate caught it |
|---|---|---|---|
| 1 | `trained` symlink was **absolute** | build step, after 2 h capture + clean train | it resolves on the host; only the container sees it dangle |
| 2 | build refused the 27 **fp32** tensors | build step, again | the dtype branch had only ever seen fp8 + bf16 |
| 3 | `\| head -1 \| ... \|\| echo D` under **pipefail** emitted the value *and* the fallback | secondary gate, after the eval it judged had run | SIGPIPE makes a correct pipeline report failure |
| 4 | `--resume` built with **`basename`**, pointing at a symlink created only *after* the chunk loop | hour 4 of a 19 h session | **never executed**: s1 and s2 both ran in one chunk |
| 5 | AdamW resume asserted `len(state) == len(params)` | same run, minutes later | lazy optimizer state never satisfies it; also never executed |

**The rule this yields: exercise the LATE stages EARLY, at toy scale, on real data.**
`scripts/s5_preflight.sh` runs the real `s5_session.sh` unmodified on 4 prompts with `S5_CHUNK=1`,
which forces a **2-chunk split** — the only way to execute the resume path — and asserts stage
*markers* rather than the quality verdict, because a head trained on two sequences is supposed to
fail its gates. ~20 minutes, and it would have caught all five of the above. **Run it before every
full session, and after any change to the session script or the trainer.**

Two further rules, each bought with a cycle:

- **Never edit a script that is currently executing.** bash reads a script by byte offset, so
  inserting lines into a running file shifts what it reads next. Editing `ce_tv_ablation.sh` mid-run
  killed it after arm 1 and cost ~20 minutes and a relaunch. `pgrep -f <script>` before any edit.
- **A toy-scale run must use real data.** Toy SCALE is the point; toy DATA would exercise a
  tokenisation path the real session never takes, and prove nothing about the path that matters.

## 7. Instruments available (all off by default)

| knob / tool | what it answers |
|---|---|
| `DSV4_DPROF=1` + `DSV4_KSWEEP=1` | per-sub-op ms for K=1..5 — **the** ranking instrument |
| `DSV4_SPECPROF=1` | draft vs verify split of the spec cycle, **plus (F82) the draft path's raw `cudaMalloc`/`cudaFree`/`cudaStreamSynchronize` host time and CALL COUNTS** (`rmalloc`/`rfree`/`rsync` in `dscratch.h`). Counts are integers, immune to trap 25. **Costs +1.0 % — see trap 32, never re-baseline from one** |
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
| `TCB_KC=<n>` | K-blocks staged per barrier pair in the fp8 tile. **`TCB_KC=1` is the pre-F74 kernel** — the A/B is one env var |
| `tests/gate_tc_fp8_kc` | is the fp8 tile still BIT-identical to KC=1? (equality, unlike the cosine gate next door) |
| `OG_WS1=1` | one scale load + one `exp2f` per k-block in the ogroup GEMV instead of NR identical ones. **Bit-identical and worth +0.1 %** — default OFF, see §3 (F76) |
| `tests/gate_og_ws1` | is `OG_WS1=1` still bit-identical to the default? memcmp over 72 (M, NR, shape) points |
| `tools/dprof_diff.sh a.log b.log [mark...]` | pairs two `DSV4_DPROF=1 DSV4_KSWEEP=1` logs by (K, sub-op) and prints the delta, including the ksweep row. **Use it to find the control the change must not have moved** (trap 12) |
| `nvcc -Xptxas -v` | registers and **spill bytes** per template instantiation. One recompile; it answers occupancy questions the bench can only rank (traps 18, 19) |
| `OG_SMEM_LAZY=1` | with `OG_SMEM=1`, re-read the staged activation float4 from smem per (r,m) instead of hoisting M of them into 4M registers. **Bit-identical and worth −1.1 % on an arm that is itself +37.8 %** — default OFF, see §3 (F79) |
| `TCB_DB=1` | prefetch the fp8 tile's next staged chunk before the current mma. **Bit-identical and worth +0.28 %** — default OFF, see §3 (F78). `TCB_DB_BPS` is the `__launch_bounds__` blocks/SM the DB entry is held to (4) |
| `DSV4_SUFFIXPROBE=1` | **prices LEVERS.md S6 without building it.** At every verify it runs `suffix_draft()` on the committed sequence and counts what the target would have accepted from a suffix/prompt-lookup draft, against what the MTP actually got — as a **counted integer**, which is the only way anything survives the ~1.5 % cross-run timing floor (trap 25). READ-ONLY: no device buffer, no engine state, so GATE and LOSSLESS GATE are unaffected. `DSV4_SUFFIX_MAXNG` caps the match length (32). Reports a **sound lower bound** and an optimistic estimate — `tam[i]` is only valid ground truth for `i <= acc` (F80) |
| `tests/gate_suffix_draft` | is the S6 matcher right? 8 host checks, no CUDA, no checkpoint — so the probe cannot retire S6 on a bug |
| `TCB_CPA=<stages>` | stage the fp8 tile's B rows with a `cp.async` ring `<stages>` deep, one K-block per stage, instead of the KC register array. **Bit-identical and 15–53 % SLOWER** (2–26 % even at cp-size 16) — default OFF, see §3 (F81). `TCB_CPA_UF` is the issue-loop unroll; **1 is required**, since unrolling keeps H addresses live and costs a block/SM (78 regs vs 48) |
| `tools/moe_gemv_bench.cu` | **prices the MoE GEMV with NO checkpoint load.** Runs the shipped `tc_fp4_grouped_gemv_e8m0` at both real shapes and both real groupings (M=1, and the measured K=5 histogram), pointers at residue 8, over a pool sized past L2, with a streaming roofline taken in the same binary at the same clocks. Also benches the mma path head to head. **Use `ms`, not the GB/s column, when varying RB** — the byte model charges re-reads that hit L1/L2 and reports 284 GB/s at RB=1 (F137) |
| `MOE_BN=<n>` (compile-time, `-DMOE_BN=`) | output columns per warp in the MoE GEMV. **Default 2 = the shipped kernel, SASS byte-identical.** F137 killed BN=1 on `ptxas -v` alone: at RB=4 it is 66 registers -> 7 blocks/SM against BN=2's 70 -> 7, so no occupancy step is crossed and the activation loads double for nothing |
| `tests/gate_tc_fp8_kc` (extended, F81) | now also sweeps `TCB_CPA ∈ {2,4,8}` against KC=1 at every M and both B offsets, **including a depth larger than KB** so the empty-`commit_group` padding that holds `wait_group` at a compile-time depth is exercised. Without that padding the mma reads a stage that has not landed — a race, and trap 9 says a sweep that cannot express the regime confirms itself |

---

## 8. The post-S5 queue — ranked against F122's ceiling arithmetic

S5 (draft-head fine-tuning) is close to exhausted: three sessions moved the suite from 22.66 to
24.52 tok/s shipped, the objective ablation moved nothing (F119), and the adaptK threshold is worth
a real but small +1.3 % (F121). What remains is ranked below **by whether it moves the ceiling or
merely approaches it**, which is the distinction F122 makes precise:

    verify_ms(K) = 69.9 + 17.11 * K
    intercept = the base AR step, amortised across the block  -> what speculation buys
    slope     = one more verified position                    -> what bounds every method

Block-6 perfect-acceptance ceiling **40.6 tok/s**; asymptote **58.4**; shipped 24.52 is **60 %** of
the block-6 ceiling.

| # | lever | moves | measured size | why it is where it is |
|---|---|---|---|---|
| 1 | **MoE GEMV efficiency** | intercept **and** slope | at ~55 % of achievable | The largest measured gap left. It is the only item that lifts base AR decode *and* every verify width at once, so it raises the whole table rather than one row. Occupancy and the register-cap knob are already exhausted (B7/B7'), so this needs a different kernel, not another launch-geometry tweak. |
| 2 | **m16 B-repack (M>=2 only)** | slope | M>=2 by construction | Attacks the 17.11 ms per-position term directly, which is the term that bounds *any* speculative method. No base-decode effect at all, so it is pure verify/draft win and is invisible to the base number. |
| 3 | **acceptance** | approaches the ceiling | tau 3.5362 -> 3.6712 over three sessions | Cannot exceed 40.6 at block 6 however good the drafter gets. Now measured to be a small lever, which is itself the S5 result. Any further work here needs a *different idea*, not more data — block size (F43), draft refinement (F45), K-selection (F110), gating margins (F118) and the CE/TV objective (F119) are all closed. |
| 4 | **prompt-lookup / n-gram speculation** | approaches the ceiling | untried | The one genuinely untried *technique*. It proposes tokens by copying spans from the prompt/context instead of running a drafter, so it is nearly free to produce and lossless under greedy verification. It does **not** reduce the slope — a copied token still consumes a verify position — so it cannot raise the ceiling; it can push acceptance toward it on copy-heavy spans, which is the `code_edit` / `long_context` / agentic-diff shape this engine exists to serve. Complements the MTP head rather than replacing it: use the lookup when a span matches, the drafter otherwise. |

**Do not re-propose** tree / multi-candidate speculation (Medusa, SpecInfer, EAGLE-2 style) without
new evidence about the slope. On a dense model extra tree nodes are nearly free in weight traffic,
which is the premise of the whole family; on this engine every candidate position costs the same
17.11 ms as a depth position while carrying lower marginal acceptance. F122 has the argument.

---

## 9. Post-NVFP4 ledger — what is actually left (supersedes §8)

§8 ranked MoE GEMV as lever #1 and priced the byte levers off `B_tok`. Both are now wrong. F125/F126
showed the GEMV kernels are not the limiter; F129 showed **`B_tok` overstates every byte lever on a
speculative engine**, because the verify reads each weight once for M tokens. The corrected ledger:

**Shipping: `s3` head, FP8, 24.73 tok/s suite.** Ceiling arithmetic unchanged (F122): block-6 with
perfect acceptance is 40.6, asymptote 58.4, so today is ~61 % of the block-6 ceiling.

| # | lever | expected | confidence | cost |
|---|---|---|---|---|
| 1 | **adaptK 1.5 -> 2.0** | **+1.3 %** | **measured 3x independently** (F121, s1/s2/s3) | one decision + a registry re-baseline |
| 2 | **s3 corpus at `a_ce 1.0`** | +0.37 tok/s (~1.5 %) | measured at t=4.17 (F124) | ~6 h: capture + train, `gen.txt` survives |
| ~~3~~ | ~~**m16 B-repack (M>=2)**~~ | **RETIRED, F137 — quantified at last, and it loses** | the mma path is 0.652 ms vs the GEMV's 0.442 at the **measured K=5 grouping**; fitted `GEMV 0.244+0.139R` vs `mma 0.504+0.087R` puts the crossover at **R = 5.05 rows per expert** and block 6 presents **1.67**. The dequant-once mechanism works (row cost 37 % lower) and is buried by 0.26 ms of intercept. Alive for **prefill** only (F85, +16.7 %) | — |
| 4 | **prompt-lookup / n-gram** | workload-dependent | untried | moderate |

> **F137 also retracts the premise of §8's lever #1 and of `wiki/moe-gemv-ceiling.md`.** The
> "MoE GEMV at 155 GB/s = 67 % of achievable" divided the **routed** expert bytes by a window in
> which the **shared** expert is concurrently streaming its own 1082.20 MB (forked `moe.cu:436`,
> joined `:516` — the fork F55 itself priced). Counted correctly, the M=1 MoE window runs
> **215.0 GB/s against a 202.5–214.9 GB/s streaming reference: at the roofline.** An isolated bench
> of the shipped kernel agrees (200.7 vs an in-binary 208.7). **There is no base-AR headroom in the
> MoE GEMV.** What is real is ~10 % at the K=5 *verify* grouping, attributed to the per-row inner
> loop (issue-bound, linear in rows at 0.139 ms/row on a 0.244 ms intercept; `ncu` Compute 30 % ->
> 48 %, occupancy 75 % -> 57 %) — ceiling **~+4 %** by two independent routes, and only if the row
> arithmetic became free. Tile count, the early-exit `grid.y` blocks, RB and BN/occupancy are each
> eliminated by their own control in `evidence/moe_gemv_bench.log`; BN died to `ptxas -v` alone
> (BN=1 is 66 regs -> 7 blocks/SM, BN=2 is 70 -> 7, no step crossed).
>
> **Corrected base-AR budget** (K=1, 70.03 ms step, `evidence/kchunk.log`, vs 208.7 GB/s):
>
> | region | ms | % | bytes | GB/s | % roofline |
> |---|---|---|---|---|---|
> | **MoE** (routed + shared, concurrent) | 23.20 | 33 % | 4531 MB | **195** | **94 %** |
> | **Attention** (MLA + compressor + indexer) | 31.40 | 45 % | 5400 MB | 172 | 82 % |
> | `lm_head` | 5.80 | 8 % | 1059 MB | 183 | 88 % |
> | HC / rmsnorm / router / glue | 8.53 | 12 % | 211 MB | 25 | latency |
>
> Whole step: 11.20 GB in 70.03 ms = **160 GB/s, 77 % of roofline**. The MoE is the best-optimised
> region in the engine, so **all** of the remaining 23 % is outside it — in the attention block
> (closed twice, F125/F126) and in 8.53 ms of glue moving 211 MB. Start here, not from `B_tok`.

**THE REFRAME THAT MATTERS.** F129 established that the suite is **verify-dominated**: NVFP4 at M=1
moved the number by -1.3 % while the same change at M=K moved it by -14 %. The M>=2 path is where the
time is. That makes **#3 the most under-explored item on this list** -- it targets exactly M>=2, it
has never been quantified, and every M=1 optimisation measured this session has come back inside
noise. It also means any future work should be A/B'd on the **suite with the trained head**, never on
the canonical prompt: those two traces disagreed in *sign* on NVFP4.

**Closed this session, each with the measurement that closed it:** MoE GEMV (~~at parity with a peer
engine, 67 % vs 70 % of respective rooflines~~ — **superseded by F137: it is at 94-100 % of the
roofline at M=1, and the 67 % was mis-accounted**); dense MLA GEMV kernel (225-246 GB/s isolated, above
the streaming reference -- not the limiter, F125); dependent layer sequence (~4 %, F126); activation
sparsity (dead at `moe_intermediate` 1024 *and* 2048, F123); tree/multi-candidate (width priced at
depth rates, F122); NVFP4 dense (no gain on the shipping metric, F129).

**Not a kernel lever:** batching. 50+ tok/s aggregate is reachable at batch 4-8 because dense/MLA
amortises fully across requests -- but it does not reduce per-request latency, and a coding agent's
hot path is serial. It belongs to Phase 6 (the server), not here.
