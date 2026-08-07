# OPTIMIZATION_LOG.md — measured A/B per lever

Discipline: profile -> hypothesis -> ONE change -> measure -> gate -> log. Correctness gate first;
a speed win with wrong numbers is a loss. Mechanism goes in `LOOP_LOG.md`, numbers here.

**Instrument policy (learned the hard way, see LOOP_LOG Opt #1):**
`ncu` finds the MECHANISM · `tools/gemm_bench.cu` predicts the MAGNITUDE · the full model CONFIRMS.
`ncu`'s isolated-launch duration exaggerates latency-bound wins (1.83x predicted vs 1.107x real).
And never trust `ncu`'s "Memory Throughput %" on Thor at all — it is L2 throughput (`HARDWARE.md` §3).

| # | lever | change | correctness | bench | **full model** | status |
|---|---|---|---|---|---|---|
| 2 | small-N GEMV wave quantisation | grid-stride + wave-aligned launch for `fp8_gemv_m1_kernel` | Gate K PASS | clamp-to-one-wave: `wq_b` 214.6 -> 186.3, `wo_b` 225.5 -> 194.7 (**worse**); round-down: within noise (`wq_a` 41.6 vs 156.8 across runs) | not run | **REVERTED — negative result.** Whole lever is <2% of `B_tok` (`wq_a` 1.6% + `wkv` 0.8%); could not justify risk to `wq_b`/`wo_b` (~26% at 89-94%) |
| 7 | **M=1 GEMV memory-level parallelism** | inner loop issued ONE 4-byte load per iteration and consumed it immediately (**ILP=1**). Unrolled by 4, all 8 loads issued before any use. kb accumulation order unchanged -> bit-exact. | Gate K PASS | on-box: ILP=1 streaming sustains 110-132 GB/s, ILP>=2 sustains 224-237 | **105.2 -> 101.7 ms/tok, 9.51 -> 9.83 tok/s (1.034x)** | **ADOPTED** |
| — | BF16-native compressor / indexer-compressor | `gemm_bf16w` instead of `Loader::bf16`'s f32 expansion; 526 MB/step read as 1052 | gate PASS, tokens byte-identical | — | **100.2 -> 103.5 ms/tok — SLOWER**, though memory 108.1 -> 107.6 GiB | **REJECTED (default off, `COMP_BF16=1`).** `gemm_fp32` is warp-per-column scalar = COMPUTE-bound, so halving bytes adds conversion ALU to an ALU-bound kernel. Re-try after a half2 rewrite of `gemm_fp32`. |
| 9 | **half2 dequant in the MoE GEMV** | `cvt.f16x2.e2m1x2` + `cvt.f16x2.e4m3x2` + `__hfma2` replacing 96 scalar ops per 16 B with ~32; half2 accumulate within a 32-elt block, f32 across blocks. GEMV promoted to default. | **TOLERANCE gate** (cosine>0.9999, rms_rel<1e-2, max_abs/\|o\|max<5e-3): cosine **0.9999999**, rms_rel 4.04e-04, rel_max 5.01e-04 | **M=1 121.5 -> 314.6 GB/s (2.59x)**; wins at every M | **101.7 -> 100.2 ms/tok, 9.83 -> 9.98 tok/s**; generated tokens byte-identical | **ADOPTED** — but costs DSpark acceptance 3.12 -> 1.00. `MOE_MMA=1` reverts. |
| 8 | **MoE GEMV output-column blocking BN=2** | one warp per output column re-loaded the activation for every column; now 2 columns/warp, activation loaded once, grid halved, 128 thr/block | `gate_fp4_gemv` **cosine 1.0000000**; Gate K MoE paths cosine 1.0000000 | **GEMV 91.0 -> 108.4 GB/s (+19%)** | **0** — GEMV still loses to the m16 mma (121.6) so it stays off by default | **BANKED, not adopted.** Revealed the GEMV is COMPUTE-bound (flat 108-109 GB/s across M=1..8): ~96 scalar ops per 16 B vs the reference kernel's ~32 via `cvt.f16x2` + `__hfma2`. Next step is a dequant rewrite, which needs a tolerance gate. |
| — | MoE g-loop software pipelining | prefetch iteration g+1's two `uint4` weight loads while consuming g | Gate K PASS, cosine 1.0000000 | bench 121.6 -> 118.0 GB/s | **102.9 vs 101.7 ms/tok — WORSE** | **REVERTED** — loop already at ILP=2; +8 registers against Block-Limit-Registers 48 loses more occupancy than it gains |
| — | lock EMC + GPU `min_freq` | default governors park EMC at 2750 MHz between bursts; worth +19% mean on *gapped* workloads | — | sustained bw 235 vs 240 (noise) | **106.8 vs 105.2 ms/tok — NO GAIN.** Our step is fully device-side with no host gaps to recover. | **REJECTED** |
| 6 | small-M coalesced FP8 GEMV | Templated `fp8_gemv_mkT_kernel<M>` (`acc[M]`, not `acc[16]` — the register pressure that made the prior project reject this path). Reads B fully coalesced (128 B/warp) vs the m16 tile's 8 rows strided by K. Adopted for M=2..4; `TC_MK=1` forces the tile. | Gate K PASS | **COLD** A/B: M=2 **2.00x**, M=3 1.49x, M=5 1.04x, M=8 0.72x | none by construction (dispatch is M>=2; base decode is M=1) | **ADOPTED for M=2..4** |
| 5 | device-side draft AR loop | `dspark_forward_head`'s greedy loop ran on the HOST: per block position a sync, a 517 KB logits D2H, and a 129k-element CPU argmax. Moved on-device (`k_seed_first`/`k_pick`/`k_argmax_row`/`k_store`), removed the internal sync in `dspark_markov`. | **draft tokens byte-identical**, acceptance unchanged at 3.12 | — | draft `fwd_head` **39.2 -> 32.4 ms**; round 353.5 -> 346.5; **spec-decode 0.98x -> 1.00x of base** | **ADOPTED** |
| 4 | BF16-native `lm_head` + markov heads | `Loader::bf16` was dequantising BF16 weights to f32, so `lm_head` was **read as 2118 MB instead of 1059 MB every step** (and `markov_w2` 5x per draft). New `gemm_bf16w`: native bf16x2 loads, 4 warps/block (the old `gemm_fp32` launched 646,400 one-warp blocks), runtime alignment check. | `gate_bf16w` vs the old f32 path: **cosine 1.000000000, argmax MATCH** on all 4 shapes | — | **108.4 -> 105.2 ms/tok, 9.26 -> 9.51 tok/s**; draft `fwd_head` 54.0 -> 39.2 ms; verify 308.7 -> 293.0; **memory 110.2 -> 108.1 GiB** | **ADOPTED** |
| 3 | HC: warp-parallel Sinkhorn + block-per-output `k_mixes` | `hc_sinkhorn_kernel` retired 31 of 32 lanes and kept a runtime-indexed `c[HCMAX*HCMAX]` in **local (DRAM) memory**; now one warp per token with the 4x4 in registers and `__shfl_xor` reductions. `k_mixes` one warp -> one block per output. `HC_SCALAR=1` restores the old path. | Gate K PASS, `comb` max_abs 1.19e-07 -> **8.94e-08** | sinkhorn 0.0862 -> 0.0165 ms (**5.2x**); `hc_pre` 0.1137 -> 0.0436 (2.6x); -6.1 ms/step | **115.8 -> 108.0 ms/tok, 8.63 -> 9.26 tok/s = 1.072x**; 96.7 -> 103.7 GB/s | **ADOPTED** |
| 1 | MoE grouped GEMM occupancy | `<<<grid,32>>>` (1 warp/block, 50% occupancy) -> `<<<grid,32*4>>>` with `n_block` derived from the warp id. Pure launch geometry; `MOE_WPB` env restores the old value. | Gate K PASS, MoE cosine **1.0000000** | 112 -> 122 GB/s @M=1 (1.09x) | **128.2 -> 115.8 ms/tok, 7.80 -> 8.63 tok/s = 1.107x**; 87.4 -> 96.7 GB/s (36.4% -> 40.3% of achievable) | **ADOPTED** |

## Baseline

| | ms/tok | tok/s | GB/s | % of 240 achievable |
|---|---:|---:|---:|---:|
| Ported as-is (= prior project's 180B result, 126.7 ms) | 128.2 | 7.80 | 87.4 | 36.4% |
| After Opt #1 | 115.8 | 8.63 | 96.7 | 40.3% |
| After Opt #3 | 108.0 | 9.26 | 103.7 | 43.2% |
| After Opt #4 | 105.2 | 9.51 | 116.6 | 48.6% |
| After Opt #7 (ILP) | 101.7 | 9.83 | 120.6 | 50.2% |
| After Opt #9 (half2) | 100.2 | 9.98 | 122.4 | 51.0% |
| After Opt #10 (gemm float4+ILP) | 99.9 | 10.01 | 122.7 | 51.1% |
| After Opt #12 (rsqrt fused) | 99.3 | 10.07 | 123.5 | 51.4% |
| After Opt #11 (ogroup uint32) | **96.9** | **10.32** | **126.5** | **52.7%** |
| Opt #13-15 (MoE ptr tables, router float4, hash-layer selected scoring) | 97.4 | 10.27 | — | — |
| **Cumulative** | **1.324x** | | | |

Opts #13–#15 measured **no gain** (96.9 → 97.4 ms, inside the ±1 ms run-to-run band). Kept: they
remove 258 H2D copies/token and 26x of router work on the hash layers, are strictly less work, and
gate clean — but they are recorded here as **zero**, not as a win, because that is what was measured.

### Measured sub-phase profile at K=1 (`~/dprof4.log`, 96.9 ms config)

This is the first profile that covers all 43 layers and the inside of MoE. Achieved bandwidth is
`phase_bytes / phase_ms`; the ceiling column is the on-box working-set curve for that phase's size.

| phase | ms | % | bytes | achieved GB/s | vs ceiling |
|---|---|---|---|---|---|
| `cattn:ogroup` (41 layers) | 16.63 | 21.5% | 2.75 GB | 165 | 79% |
| `moe:w1w3` | 12.40 | 16.0% | 2.44 GB | **196** | **94%** |
| `cattn:q_proj` (41 layers) | 11.35 | 14.7% | 1.63 GB | 144 | 69% |
| `moe:shared` | 7.55 | 9.7% | 1.08 GB | 143 | 69% |
| `moe:w2` | 5.79 | 7.5% | 1.22 GB | **210** | **at ceiling** |
| `hc_pre` ×2 | 4.74 | 6.1% | ~0.07 GB | latency-bound | — |
| `moe:router` + `moe:group` | 4.00 | 5.2% | ~0.11 GB | latency-bound | — |
| glue (`rmsnorm`, `hc_post`, `act`, `combine`, `sparse`, `kv xin`) | ~3.8 | 4.9% | small | latency-bound | — |
| unattributed inside ATTENTION (compressor + DSA indexer) | 9.02 | 11.6% | — | **not instrumented** | — |
| **unattributed outside the layer loop** (`lm_head`, head chain, host) | **20.1** | **20.6%** | ~1.1 GB | **never instrumented** | — |

**This resets the base-AR ceiling.** The three largest weight-streaming phases are already at
79–100% of achievable. The 126 GB/s *average* is not a bandwidth deficit spread evenly over the
step — it is ~20 ms of latency-bound glue moving almost no bytes, plus 20 ms outside the layer loop
that has never been measured at all. Eliminating every latency-bound phase entirely would reach
~78 ms ≈ 12.8 tok/s. **Realistic base AR is 13–14 tok/s, not the ~17 previously claimed.**

Two consequences:
1. The remaining base-AR levers are **fusion and launch-count**, not bandwidth. Byte reduction on
   `moe:w1w3`/`moe:w2` is worthless — they are already at the wall.
2. The **20.1 ms outside the layer loop is now the single largest unexamined block in the engine**,
   larger than any individual instrumented phase except `cattn:ogroup`. Instrument it next.

Note the GB/s column now uses the **engine's** `B_tok` of 12,261 MB, not the checkpoint's 11,202 MB.
`ROOFLINE.md` measured what is *stored*; Finding 26b showed the engine was reading `lm_head` at
double its stored size. After Opt #4 the two agree again.
| Target band (`ROOFLINE.md` §3) | 63–80 | 15–19 | 168–192 | 70–80% |
| Wall | 46.7 | 21.4 | 240 | 100% |

## Next levers (measurement-backed)

**The draft head is DONE as a lever** (Opt #4+#5: 75.7 -> 54.7 ms, 1.39x). Per the spec profile it is
now only ~15.8% of a verify round; **the verify is 84.2%**, so zeroing the rest of the draft would
move speculation only 1.00x -> ~1.10x.

1. ~~DSpark draft head~~ — ~6x off its own roofline; gates the entire speculative win (Finding 17).
2. ~~The M>=2 verify step penalty~~ — **ROOT-CAUSED (Finding 15 CLOSED)**. It is the attention
   **glue** (`act_quant`/`rmsnorm`/`rope`/KV-write around the projections): `q_proj` 2.78x,
   `kv_write` 2.62x, `ogroup` 3.02x from K=1 to K=2, while `sparse_attn` is FLAT at 1.00x. 68.6 ms
   over 43 layers, accounting for the whole ~70 ms. **Fusing that chain is now the #1 lever**:
   `c_v(5)` 2.61 -> ~1.9, speculation 1.00x -> ~1.6x, and it helps base decode too.
   (superseded entry:) old M>=2 note — ~+0.70 c_v at K=2, mechanism still OPEN after two refuted
   hypotheses. Bench `ogroup_gemm_fp8` and the HC/Sinkhorn path before proposing a third.
3. **MoE beyond occupancy** — still only ~51% of achievable after Opt #1.

Retired: MLA projection GEMVs (already 89-94%), MXFP4 hardware unpack (already implemented),
small-N GEMV wave quantisation (real mechanism, but the whole lever is <2% of `B_tok` — Opt #2).

**Rank by `bytes x (1 - efficiency)`, never by efficiency deficit alone.** Opt #2 was queued at #2
on its 19% efficiency without checking that the shape carries only 1.6% of the bytes.


---

## Session 2026-08-06/07 — speculation crosses 1.0x and the roofline gets corrected

Measured end-to-end, canonical prompt `0,671,6102,294,8760,344`, GATE PASS on every row, tokens
byte-identical unless noted.

| change | base AR | M=5 verify | spec-decode | note |
|---|---|---|---|---|
| start of session | 97.6 ms (10.24) | 279.4 ms | 10.04 tok/s (0.92x) | |
| S1 draft e8m0 scale fix | 97.4 | 279.4 | acceptance 1.00 -> **3.00** | Finding 39 |
| S2 ogroup M=K GEMV | 97.6 | 253.7 | 10.04 (0.98x) | Finding 40 |
| **F41 smem-staged fp8 tile + F42 lm_head M=K** | 97.4 | **200.6** | **12.12 (1.18x)** | first win |
| F42 bf16 alignment fallback | 97.7 | 179.4 | **14.36 (1.40x)** | `head_bf` is only 4-byte aligned |
| F43 ogroup NR + gemm_fp32 M=K | 96.5 | 173.5 | 14.76 (1.42x) | |
| **F44 weights -> managed device-preferred** | **92.8** | **168.8** | **15.49 (1.44x)** | + CUDA graph: 81.3 ms = 12.30 tok/s |

**1.54x on speculative decode; 1.05x on base AR, 1.23x with the CUDA graph.**

### What the numbers mean now

- `c_v` is **1.82** against a byte-model floor of **1.84** — the verify's K-scaling is optimal. All
  remaining loss is a uniform 1.89x shared by K=1 and K=5, so base-AR efficiency and speculation are
  now one lever, not two.
- The achievable bandwidth for this engine is **~233 GB/s strided** out of managed memory (was ~216
  out of mapped-host), **not** the 240 GB/s `cudaMalloc` figure every earlier "% of achievable" used.
- Cycle floor 106.6 ms at acceptance 3.00 = **28.1 tok/s**. 35 needs 3.70 tokens/verify; 50 needs
  5.30, above BLK=5's hard maximum of 5.

### Retired with a measurement this session

Block size > 5 (identical accept sequence at 5 and 8) · draft refinement (3.00 -> 2.08; the MTP
heads are trained with the noise token as placeholder) · shared-A fp8 GEMV (slower at every M) ·
the small-M fp8 GEMV as a default at M>=2 (loses 1.5-2.3x to the fixed m16 tile).
