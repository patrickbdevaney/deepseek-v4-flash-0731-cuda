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
| After Opt #4 | **105.2** | **9.51** | **116.6** | **48.6%** |
| **Cumulative** | **1.219x** | | | |

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
