# Decode ladder — the ordered work list the autonomous loop executes

`scripts/decode_loop.sh` reads this file, picks the topmost `TODO`, does it, gates it, measures it,
and marks it. **Edit this file to steer the loop.** One item per iteration; one change per
measurement; bands not points.

## The stop condition — what "no faster combined decode is possible" means here

Combined decode is `tok/s = tau * 1000 / ms_per_forward`, and `ms_per_forward = a + b*ctx`
(`tools/decode_model.py`). Both terms have a **byte floor** at the measured 240 GB/s:

- **Term A floor** — forward bytes at the measured K=5 expert union (17.53 of 30) plus the draft
  side: **82.18 ms**. Term A was 136.44 ms at suspension = **60.2 % of achievable**.
- **Term B floor** — 42.0 MB at ctx 6592 = 0.509 ms/forward at tau 2.91. Term B was **198.1 ms**.

**The loop STOPS when either:**
1. `a <= 1.25 * a_floor` **and** `b*6592 <= 5.0 ms` — i.e. both terms are within a quarter of their
   byte floors and the context term has stopped mattering; or
2. five consecutive iterations move the suite by less than **2 %**, which is below the measured
   3.5 % run-to-run spread and therefore unmeasurable; or
3. `DECODE_LOOP_STOP` exists, or the iteration/wall caps are hit.

Reaching (1) is the roofline. Reaching (2) means the remaining headroom is not addressable by the
items on this ladder and the loop should hand back rather than thrash.

## Hard invariants — the loop must never violate these

1. **Bit-exactness or an explicit gate.** Any kernel change either produces byte-identical generated
   token ids, or ships behind the LOSSLESS gate with the deviation measured and recorded. A change
   that silently alters output is reverted, not debugged.
2. **tau is reported in every A/B.** A byte-identical token sequence can still collapse acceptance
   3.12 -> 1.00 (`LOOP_LOG`), because acceptance is an exact draft/target comparison. Throughput
   without tau is not a measurement.
3. **Never restart the eval battery.** The watchdog and boot unit are disarmed on purpose. The GPU
   is single-tenant; a second client turns a scored item into a banked wrong answer.
4. **Never modify the checkpoint.** Weights are not ours to edit. The draft head is backed up at
   `~/model-backups/heads/shipped-dspark-0731reap/`.
5. **Clocks are a measurement variable.** Do not change `jetson_clocks` mid-comparison. It is its
   own ladder item with its own before/after.

---

## Phase 0 — instrument (must complete before anything is trusted)

- [x] **0.1** `timings` on `/v1/completions` — staged in `server.cpp`, built 2026-08-19.
- [x] **0.2** `DSV4_DPROF` diff at two contexts. **DONE 2026-08-19 — the context term is GONE.**
      Two single-prompt runs on the same binary, ctx 480 and ctx 3000:

      | | base AR (M=1) | spec | i:topk (21 calls) | i:score | cattn:indexer |
      |---|---|---|---|---|---|
      | ctx 480 | 89.3 ms/tok | 65.9 ms/tok, 15.18 tok/s | 0.10 ms | 0.02 ms | 2.09 ms |
      | ctx 3000 | 89.4 ms/tok | 65.2 ms/tok, 15.33 tok/s | 0.10 ms | 0.02 ms | 2.05 ms |

      **Base AR moved 0.1 ms across a 6x context change.** The pre-fix fit predicted
      150.9 -> 226.6 ms over the same span (`136.44 + 30.053*(ctx/1000)`). The slope went from
      **30.05 ms per 1000 context to ~0.04**, and `i:topk` is flat at 0.10 ms across all 21 calls.
      Both LOSSLESS gates passed.

      **This retires 1.2 and 1.5 as decode levers.** A radix select replaces a kernel that now
      costs 0.10 ms, and the `index_score` GEMM attacks `i:score` at 0.02 ms. Neither is worth a
      bit-exactness risk. 1.3 and 1.4 remain as correctness items, not performance ones.
      The remaining cost is **ATTENTION 44 % and MoE 38 %**, which is Term A — and Term A is the
      term with ~1.5x of headroom, not 3x.
- [ ] **0.3** Re-fit `tools/decode_model.py` on a post-fix run and record both coefficients.

## Phase 1 — the context term

- [x] **1.1** Warp-parallel top-k (all four kernels). **14.2x at ctx 6592, 24.7x at 24k**,
      bit-identical on nine shapes and three distributions (`tests/gate_topk_warp.cu`).
- [~] **1.2** RETIRED as a perf lever (0.2: top-k is now 0.10 ms). Single-CTA radix select to replace the warp scan. Reference: SGLang's
      `deepseek_v4_topk.cu` (Apache-2.0) and TileLang `topk_selector.py`. **Restore descending
      order with a 512-element bitonic sort** — the reference emits in `atomicAdd` order, and
      `sparse_attn` sums selected rows in order, so without the sort this is not bit-exact.
- [ ] **1.3** `seq_len <= topk` early-out. Below ctx 2048 every row survives and the kernel still
      does the full scan to discover it.
- [ ] **1.4** `cudaFuncSetAttribute` opt-in for dynamic shared memory, or drop the requirement
      entirely (1.2 does). Removes a silent garbage-return above ~49k context.
- [~] **1.5** RETIRED as a perf lever (0.2: i:score is 0.02 ms). `index_score` as a GEMM + fused epilogue. Measured 15.2x standalone
      (658 us -> 39 us at T=6000). **FP32/TF32 accumulation only** — an external ablation shows
      FP16 dropping perfect-recall rows from 99.99 % to 91.82 % on this exact operation.

- [ ] **1.6** **OPEN DEFECT — pre-existing, NOT caused by the top-k change.** `pending CUDA error:
      invalid argument`, 42x at `mla_attn.cu:711` and 42x at `:820`, both of which are the `dsync`
      on a return path of **`ogroup_gemm_fp8`** — a function this work never touched.

      **The "it is new" conclusion was wrong and is retracted.** It rested on older DPROF runs
      (`dbuf.log`, `kchunk.log`, 2026-08-08) being clean while the reporting was already live from
      2026-08-07. But `mla_attn.cu` was modified on **2026-08-11** (NVFP4 dense overlay) and
      **2026-08-12** (F128, ogroup M=K), i.e. *after* those logs, and no DPROF run has been taken
      on this binary since. The clean logs simply predate the change that introduced it.

      Confirmed by measurement, not argument: `DSV4_SYNCPROBE=1` against `build/decode_probe`,
      with a probe after **every** launch in `compressed_decode_step_indexer_dp` including
      `sparse_attn`, ran to completion with **no fault attributed**. The indexer decode path is
      clean; the fault is downstream of it.

      Ruled out so far, each by arithmetic rather than by guess: the bs==1 GEMV grid
      (`G*R*32/256` = 1024 blocks, no overflow); the `__launch_bounds__(256, 4)` = 1024 threads/SM
      against Thor's 1536 cap (satisfiable); and dynamic shared memory (the M=K launches pass 0).
      A bisecting probe after the o-rope is in flight to separate `rope_interleaved_dp` from the
      launches inside `ogroup_gemm_fp8` itself.

      **Severity: latent.** Output is correct, both LOSSLESS gates pass on every run, and tau is
      normal. It is a launch that fails and whose absence does not change the answer — which is
      its own question worth asking, since a kernel nobody needs is a kernel to delete.

## Phase 1b — bit-exact packing (`KV_PRECISION_FINDINGS.md`)

- [ ] **1b.1** Pack the DSA index cache as real MXFP4: 128xE2M1 + 4xUE8M0 = **68 B** (was 512 B).
      Values already on the E2M1 grid with exact pow2 scales. Acceptance: `memcmp`.
- [ ] **1b.2** Pack KV dims 0-447 as real FP8 E4M3 + 7xUE8M0 = **711 B** (was 2048 B); RoPE stays
      FP32. Acceptance: `memcmp`. **Unlocks seqmax 32k-64k, which does not fit today.**

## Phase 2 — speculation (accuracy-neutral by construction)

- [ ] **2.1** Re-tune block width AFTER 1.1/1.2. Verify width is currently free only by accident of
      the broken kernel; with Term B small the optimum returns to ~7-9 from an apparent 11-13.
- [ ] **2.2** **Deploy `s3`.** It is promoted, archived, measured at tau 3.8438 / 25.53 tok/s
      against the shipped head's 3.5362 / 22.66 — and `promote_head.py` only archives, it never
      writes the live checkpoint, so **the server has never run it**. Needs a `--head` path in the
      server, or a staged checkpoint. Free ~13 % on the bench suite.
- [ ] **2.3** Use the confidence head at verify time (EVICT-style `argmax E[A(T_k)]/C(k)`). It
      exists and is unused.

## Phase 3 — clocks, then hand back

- [ ] **3.1** `jetson_clocks` — GPU 1386 -> 1575 MHz, EMC 2750 -> 4266. Measured in-repo at
      +3.0-6.4 %. Its own before/after; do not fold into another item.
- [ ] **3.2** Final `PERF.md` re-run and both coefficients recorded. Then STOP and hand back.

## After the roofline — the long-horizon pivot

Not part of the decode loop. When the loop stops, the next programme is agentic trace capture and a
distribution-matched draft-head fine-tune (`S5_RECIPE.md`, `DECODE_ZENITH_FINDINGS.md` Phase 2).
The highest-leverage change identified: **harvest `(h_40/41/42, p_target)` from live verify
forwards** — zero marginal compute, on-policy and distribution-matched by construction, and it
removes the 240-agentic-prompt ceiling that caps the current corpus.
