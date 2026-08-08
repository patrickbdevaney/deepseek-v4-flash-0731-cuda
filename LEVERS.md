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

Current baseline (prompt 0, clocks pinned, **post-Finding-81, CLEAN**): **22.06 tok/s speculative**,
**13.83 tok/s base AR** (72.3 ms/tok), acceptance **2.89**, speedup **1.60x**.
`evidence/clean_post_f81.log` — no `DSV4_DPROF`, no `DSV4_KSWEEP`, no `TCB_CPA`, GATE PASS and LOSSLESS
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

Base AR reads the whole 12.26 GB weight set per token. At 233 GB/s the floor is **52.6 ms/tok =
19.0 tok/s**; graphed we are at 79.9 ms, so ~34 % of the base step is not bytes.

| # | lever | expected | why it might work / what to watch |
|---|---|---|---|
| ~~B0~~ | ~~audit every kernel for thread-per-output at decode shapes~~ | **NAMED LIST EXHAUSTED, F71 + F73** | `index_score` **+4.4 %** (F71). `k_moe_prefix` and `k_build_tiles`, both `<<<1,1>>>` over `nr=160`, fixed in F73: `moe:group` −20.3 %, and both now sit at the **launch-latency floor** (0.24–0.25 ms / 43 layers, vs 0.19 for a trivially parallel neighbour). The rest are measured under B0's own 0.5 ms bar: `k_topk_verify`/`k_topk_decode` → `i:topk` = **0.12 ms**; `k_dg`/`k_advance_T`/`k_incr` are genuinely one scalar's work. **The class paid twice and is now dry** — reopening needs a *new* kernel with a bad geometry, not another pass over these. |
| ~~B8~~ | ~~the fp8 GEMM block~~ | **FULLY CLOSED, F78 + F79** | **F78 closed the tile half; F79 closed `o:wo_a`.** What remains is `cp.async` staging (below), which is a different kernel, not a tuning of this one. Original note kept: **F78 closed the one named move on the three tile marks.** Double-buffering the staged chunk is bit-identical and worth **+0.28 %** in situ, with `o:wo_b` **+2.8 %** — the wrong way, in agreement with the bench. With F67 (both reroutes), F68 (split-K, numerics) and F74 (the win that was there), **the tile has no untried idea left except `cp.async` staging global→shared**, which removes the register array entirely but must clear the same occupancy bar F78 failed. What remains of B8 is the single item below on `o:wo_a`. |
| ~~B8'~~ | ~~`o:wo_a`: the `OG_SMEM=1` variant reading `o4[m]` LAZILY per m from shared memory~~ | **RETIRED, F79** | `ptxas -v` answered before the bench, exactly as the criterion demanded: spill **396 → 368 bytes at an unchanged 64 registers**, a −7 % move where ~16 registers were predicted, because ptxas re-hoists the smem loads back to where `o4[M]` was. Bench: **−1.1 %** vs the `OG_SMEM` arm, **+37.8 %** vs the shipped kernel. See §3. **`o:wo_a` now has no untried named idea at all**, and with it B8 is fully closed. |
| B8'' | **M-gate `OG_SMEM` on: F55's 40 % kill was measured at M=5 ONLY, and the sign flips below M=4** | **~0.4 % — SUB-1 %, does not count toward the pivot queue** | New evidence, not a new argument (F79). F55 swept NR at M=5 and retired the variant on that one column. Sweeping **M** as well, same COLD harness, 3 alternating replicates (`evidence/oglazy_bench.log`), against the NR the dispatch actually picks at each M: **M=2 0.1717 → 0.1533 (−10.7 %), M=3 0.1962 → 0.1729 (−11.9 %), M=5 0.1958 → 0.2729 (+39.4 %, F55's number reproduced), M=8 0.3610 → 0.2810 (−22.2 %)**. So `ogsmem` gated on `bs<=3` is a real candidate. **Priced honestly it is small**: `o:wo_a` is 10.74 ms of an 85.9 ms K=2 verify and 10.26 of 104.7 at K=3, and only 4 of the 9 verifies in a canonical run are K≤3 → **~0.4 % end-to-end**, before the trap-3 discount that F76 (bench −7.6 %→ in situ +0.1 %) and F78 (bench +3.1 % → in situ +2.8 %) both demand. M=4 is unmeasured — the bench skips it and it is on the crossover. |
| ~~B8-tile~~ | ~~the three tile marks `q:wq_b` + `o:wo_b` + `q:wq_a` = 22.5 ms~~ | **CLOSED, F78** | Kept for the achievability bar it established: **the target was never 233 GB/s** but the M=1 GEMV's own measured rate on the same weight bytes — `q:wq_a` 115, `q:wq_b` 195, `o:wo_b` 185, `o:wo_a` 168 GB/s (K=1 column, `evidence/kchunk.log`). **Bytes are counted (F74) and it is NOT at roofline.** The target is not 233 GB/s — it is the M=1 GEMV's own measured rate on the same weight bytes: `q:wq_a` 115, `q:wq_b` 195, `o:wo_b` 185, `o:wo_a` 168 GB/s (K=1 column, `evidence/kchunk.log`). Against that the three tile marks hold ~6.5 ms. **The one remaining move on them is to double-buffer the staged tile so round n+1's loads issue before round n's mma** — F67 closed both reroutes, F68 closed split-K on numerics. **`o:wo_a` (3.3 ms of the old estimate) is NO LONGER a cheap 7 %: F76 attributed it.** `ogroup_gemv_mk_kernel<5,4>` is *latency*-bound (8.1 of 13.0 cycles between issues on an L1TEX scoreboard), spills 44 bytes against a 64-register cap, and sits at a **measured local optimum in both knobs** — NR (1/2/4/8) and `OGMK_BLOCKS_PER_SM` (2/3/4, re-swept in F76 with the spill reduced). Instruction-count cures are retired as a family (§3). The only untried idea that moves ~16 registers instead of 3: **the `OG_SMEM=1` variant reading `o4[m]` lazily per m from shared memory** instead of holding M float4s live — smem makes lazy reads affordable where global loads need the MLP. F55 measured that variant a 40 % regression at NR=4 *with* the 20-register `o4[M]` still in it, so the register argument was never tested; falsify with `ptxas -v` before building anything. |
| ~~**B8-cpasync**~~ | ~~`cp.async` staging global→shared for the fp8 tile with a ≥4-stage smem ring~~ | **RETIRED, F81 — bench +15 to +53 %, and +2 to +26 % even at ideal alignment** | Built, **bit-exact** (`gate_tc_fp8_kc` **1512/1512**), and it **passed both cheap falsification steps and then lost the bench by a mile**. (1) `ptxas -v`: at UF=1 the ring is **48 regs / 36864 B smem / 5 blocks/SM** against the shipped `smemB<8,2,false>`'s **64 regs / 17408 B / 4** — the register array really is given back and occupancy *rises*; smem was never the binding resource (233 472 B/SM, the shipped kernel uses 30 % of it at 4 blocks). (2) `cp.async.bulk.tensor.2d`, `cp.async.bulk`, `mbarrier.init` and `mbarrier.try_wait.parity` all **assemble at `-arch=sm_110a`**. (3) `gemm_bench` COLD, arms adjacent, M=5 vs `m16+smem B+4`: wq_a **+23.9 %**, wq_b **+38.4 %**, wkv **+52.8 %**, wo_b **+39.9 %**, sw1/3 **+14.9 %**, sw2 **+36.3 %**. **The disambiguation is the valuable half**, because the engine's weights are 4-byte aligned so the ring had to use cp-size 4: re-benched at B+0 (cp-size 16) it recovers ~25 % (wo_b 0.1428 → 0.1067) **and still loses to the ordinary LDG path at 5 of 6 shapes** — wq_a +5.7 %, wq_b +11.7 %, wkv +11.1 %, wo_b **+26.4 %**, sw2 +19.6 %, sw1/3 −3.0 %. **So no alignment work rescues it**, which is what makes this a permanent close rather than a block on F67's shard pad. And **NS=2 beats NS=4 on 4 of 6 shapes** (wo_b 0.0873 vs 0.1067), so *depth is negative* — the direct refutation of the promoted claim. Mechanism, new trap 29: one ring stage is **1** K-block, so `wait_group`+`__syncthreads` runs **once per K-block** where the shipped KC=2 pays it once per two — and F74's win *was* raising bytes-per-barrier. The ring trades the one thing that pays for bytes-in-flight the LDG path already had. Not run in situ: the falsification order says stop at the first no, and trap 3's discount has never turned a large bench negative into a win (F76 −7.6 % → +0.1 %, F78 +3.1 % → +2.8 %). Kept behind `TCB_CPA=<stages>`, default OFF. See §3. |
| **B10** | **put the DSpark DRAFT path on the existing `dscratch.h` arena** — `dspark_main_kv`, `dspark_attn_forward`, `dspark_block_forward`, `dspark_forward_head`, `dspark_main_x` all call raw `cudaMalloc`/`cudaFree` + a real `cudaStreamSynchronize` per invocation, inside the loop that runs once per verify round | **+5 to +7 % — MEASURED, not modelled (F82). The largest open item since F74** | **The target is a number, not a hypothesis: `evidence/specprof_f82.log` measures 10.19 ms/round = 7.4 % of the 137.35 ms cycle as HOST time inside `cudaMalloc`/`cudaFree`, 134 calls at 76 µs, and 127 of the 134 run after their own function's `cudaStreamSynchronize` — on a drained, idle GPU.** This is the identical change that took base AR 92.5 → 79.3 ms/tok at F44, applied to the one region that predates the arena; `dscratch.h`'s own first line says *"At M=1 the per-call cudaMalloc/cudaFree/cudaStreamSynchronize in every sub-function dominate."* Bit-identical by construction (an allocator swap; `dmalloc` bumps, `dfree`/`dsync` become no-ops). **Do NOT claim the 12.99 ms of sync as recoverable** — 9 calls at 1.44 ms, mostly the host awaiting real draft GPU work; the recoverable part is the launch-gap share that disappears when the draft becomes one async chain, and it is unknown until measured. **How to falsify / what to watch, in order: (1)** arena LIFETIME — `dspark_main_kv` is called *outside* the `for(pass)` loop that calls `arena_reset()`, and `mkv[st]` must survive it; check every buffer that crosses a reset or the draft reads freed bump space (this is the one way the change can be wrong and it will not show as a crash, it will show as wrong tokens, so gate on the byte-identical spec sequence). **(2)** arena CAPACITY — the draft adds ~5 MB/round (`logits` 2.59 MB, `bias` 517 KB, `res2` 327 KB, `kv_all` = ctx x HEAD_DIM x 4) against a 512 MB arena, so it should fit, but the arena **aborts** on overflow and that kills a 15-minute run: print `g_arena_hwm` before trusting it. **(3)** keep the raw path behind an env flag so the A/B is possible, and price it on the **paired per-verify** instrument (9 pairs at identical K), not on tok/s alone. |
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
**7.4 % of the cycle is raw-allocator host time in the draft half, see B10 in §4.**

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

33. **A region's SHARE of the cycle goes stale even when nobody touches it.** The draft half was 14 %
   at F47 and is **17.4 %** now, unchanged in absolute terms while six adoptions cut the verify around
   it — so §1's table, which is a *verify* table, was being read as a *cycle* table and the draft was
   never a candidate. Re-measure the top-level split whenever the thing below it has moved (F82).

---

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
| `tests/gate_tc_fp8_kc` (extended, F81) | now also sweeps `TCB_CPA ∈ {2,4,8}` against KC=1 at every M and both B offsets, **including a depth larger than KB** so the empty-`commit_group` padding that holds `wait_group` at a compile-time depth is exercised. Without that padding the mma reads a stage that has not landed — a race, and trap 9 says a sweep that cannot express the regime confirms itself |
