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
| 1 | MoE grouped GEMM occupancy | `<<<grid,32>>>` (1 warp/block, 50% occupancy) -> `<<<grid,32*4>>>` with `n_block` derived from the warp id. Pure launch geometry; `MOE_WPB` env restores the old value. | Gate K PASS, MoE cosine **1.0000000** | 112 -> 122 GB/s @M=1 (1.09x) | **128.2 -> 115.8 ms/tok, 7.80 -> 8.63 tok/s = 1.107x**; 87.4 -> 96.7 GB/s (36.4% -> 40.3% of achievable) | **ADOPTED** |

## Baseline

| | ms/tok | tok/s | GB/s | % of 240 achievable |
|---|---:|---:|---:|---:|
| Ported as-is (= prior project's 180B result, 126.7 ms) | 128.2 | 7.80 | 87.4 | 36.4% |
| After Opt #1 | **115.8** | **8.63** | **96.7** | **40.3%** |
| Target band (`ROOFLINE.md` §3) | 63–80 | 15–19 | 168–192 | 70–80% |
| Wall | 46.7 | 21.4 | 240 | 100% |

## Next levers (measurement-backed, `LOOP_LOG.md` Findings 18-21)

1. **DSpark draft head** — ~6x off its own roofline; gates the entire speculative win (Finding 17).
2. **The M>=2 verify step penalty** — ~+0.70 c_v at K=2, mechanism still OPEN after two refuted
   hypotheses. Bench `ogroup_gemm_fp8` and the HC/Sinkhorn path before proposing a third.
3. **MoE beyond occupancy** — still only ~51% of achievable after Opt #1.

Retired: MLA projection GEMVs (already 89-94%), MXFP4 hardware unpack (already implemented),
small-N GEMV wave quantisation (real mechanism, but the whole lever is <2% of `B_tok` — Opt #2).

**Rank by `bytes x (1 - efficiency)`, never by efficiency deficit alone.** Opt #2 was queued at #2
on its 19% efficiency without checking that the shape carries only 1.6% of the bytes.
