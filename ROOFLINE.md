# ROOFLINE.md — Gate R1

**Status: COMPLETE.** Every number below is computed by `tools/inventory.py` and
`tools/verify_cost.py`, which read `docs/config.json` and `docs/hdrs/*.json` — verbatim copies of
the checkpoint's `config.json` and the safetensors header of all 48 shards, harvested by
`tools/fetch_headers.py`. No model constant is hardcoded anywhere in this document.

**Verification anchor.** The per-tensor byte sum reproduces the index's
`metadata.total_size = 107,803,320,952` **exactly**, and the derived activated-parameter count
comes out at **13.26 B** against the model card's "~13B active" — independently, from tensor
shapes. The accounting is right.

---

## 0. Headline — four of the directive's premises are wrong, and one of the corrections is large

> **1. `B_tok` is 11.20 GB/token, not the ~7 GB the directive's arithmetic implies.**
> §4.2 says "13B active params at MXFP4 (~4.3 bits average blended)". The blend is **6.76
> bits/active-param**, not 4.3, because **only the routed experts are MXFP4**. Attention, the
> shared expert, the compressor, the indexer, the router and `lm_head` are FP8/BF16 as shipped.
> The routed experts are 85.3% of the *stored* model but only 30.8% of the *per-token* read.

> **2. The AR wall is 17.9 tok/s at 200 GB/s (24.4 at 273), not ~46.** The directive's
> Qwen-3.5-122B active-param-scaling anchor (`50 × 12/13 ≈ 46`) is not merely "weakly anchored" —
> it lands above the memory-bandwidth wall of this model on this box and is therefore
> unreachable at any kernel quality. **Discard it.** §3 below replaces it with something far
> stronger.

> **3. MXFP4 is not new work.** The directive's `ARCH_DELTA` row —
> *"NVFP4 → MXFP4 (different block-scale format) → new dequant kernel — do not assume NVFP4
> kernel reuse"* — rests on a mislabel. The **prior 180B checkpoint was already MXFP4**:
> byte-identical expert layout (`I8 [2048,2048]` + `F8_E8M0 [2048,128]`), and
> `~/dspark-cuda-reap-finetune/kernels/moe.cu:25` already walks `KBw = K / 32` with e8m0 scales.
> Gate **G1 is effectively already passed** by existing gated code. (The confusion is DeepSeek's:
> `config.json` says only `expert_dtype: fp4`, and the prior project's notes wrote that down as
> "NVFP4". The tensors say otherwise.)

> **4. There is no REAP expert-ID remapping to apply.** §0 and G6 of the directive ask for a
> bit-for-bit replication of a router remap read from `reap_plan.json`. **`reap_plan.json` and
> `REAP_MANIFEST.json` contain no retained-expert ID list at all** — and they don't need to:
> `router_alignment.identity_nearest_all = true` and `hash_tid2eid_equal_fraction = [1,1,1]`,
> `gate.weight` ships as `[160, 4096]`, `gate.bias` as `[160]`, and experts are dense `0..159`.
> The checkpoint is already self-consistent. G6 becomes "validate top-6 index-exact vs the
> oracle", with no remap step.

**And one premise is right, emphatically:** §4.1's warning not to conflate "small KV cache" with
"small attention bandwidth". **MLA is the single largest per-token consumer at 41.1% — larger
than all six routed experts combined.** This inverts the directive's Phase-7 priority order
(§6 below).

---

## 1. What is on disk

| | value | source |
|---|---|---|
| Tensors | 45,821 | shard headers |
| Shards | 48 | index |
| Total tensor bytes | 107,803,320,952 = **100.400 GiB** | reconciles with `index.json` |
| Layers | 43 backbone + 3 DSpark MTP | `num_hidden_layers`, `mtp.{0,1,2}.*` |
| Routed experts | 160 (from 256), top-**6** | `n_routed_experts`, `num_experts_per_tok` |
| Hidden / head_dim | 4096 / 512, 64 heads, 1 KV head | `config.json` |
| MLA LoRA ranks | `q_lora_rank` 1024, `o_lora_rank` 1024, `o_groups` 8, `qk_rope_head_dim` 64 | `config.json` |
| DSA indexer | `index_n_heads` 64, `index_topk` **512**, `index_head_dim` 128 | `config.json` |
| KV compression | `compress_ratios`: 2 pure-sliding (0), 21 × ratio-4, 20 × ratio-128 | `config.json` |
| Sliding window | 128 | `config.json` |
| Context | `max_position_embeddings` 1,048,576, YaRN factor 16 over 65,536 | `config.json` |
| Vocab / tied | 129,280 / `tie_word_embeddings = **false**` | `config.json` — **do not carry gemma's tied-embedding assumption** |
| DSpark | `dspark_block_size` **5**, `markov_rank` 256, `noise_token_id` 128799, `target_layer_ids` [40,41,42] | `config.json` |

### Storage by subsystem

| subsystem | GiB | % |
|---|---:|---:|
| routed experts (MXFP4) | 85.664 | 85.3 |
| **MTP / DSpark heads** | **6.529** | **6.5** |
| MLA attn | 4.284 | 4.3 |
| shared expert (FP8) | 1.008 | 1.0 |
| embed | 0.986 | 1.0 |
| lm_head | 0.986 | 1.0 |
| KV compressor | 0.490 | 0.5 |
| DSA indexer | 0.256 | 0.3 |
| HC (hyper-connection) params | 0.126 | 0.1 |
| router | 0.070 | 0.1 |

### Quantisation, proved rather than assumed

```
layers.5.ffn.experts.0.w1.weight   I8      [2048, 2048]  -> logical [2048, 4096] E2M1
layers.5.ffn.experts.0.w1.scale    F8_E8M0 [2048,  128]
scale block along K = 4096 / 128 = 32
=> E2M1 data + E8M0 shared scale, block 32 = OCP MXFP4, 4.25 bits/param effective
```
Dense/attention tensors are `F8_E4M3` with `F8_E8M0` scales at `weight_block_size [128,128]`.
Norms, embeddings, `lm_head`, compressor and indexer projections are BF16. HC params are F32.

### Memory budget

```
weights                100.400 GiB
available pool         117     GiB
-------------------------------------
headroom                16.6   GiB   for KV + activations + CUDA context + the OS
```
Dropping the 3 MTP blocks (base AR only, no speculation) returns **6.529 GiB**, i.e. 23.1 GiB of
headroom. That is a real configuration knob, not a rounding detail.

---

## 2. B_tok — bytes read per M=1 decode step

| component | MB | % |
|---|---:|---:|
| **MLA attn** | **4599.48** | **41.1** |
| routed experts (top-6) | 3449.29 | 30.8 |
| shared expert | 1082.20 | 9.7 |
| lm_head | 1059.06 | 9.5 |
| KV compressor | 525.72 | 4.7 |
| DSA indexer | 275.35 | 2.5 |
| HC params | 135.28 | 1.2 |
| router | 75.00 | 0.7 |
| norms + hc_head + embed row | 0.97 | 0.0 |
| **B_tok** | **11,202.36 MB** | **10.433 GiB** |

**Why MLA is so large.** Per layer: `wq_b [32768,1024]` = 33.5 MB, `wo_a [8192,4096]` = 33.5 MB,
`wo_b [4096,8192]` = 33.5 MB, plus `wq_a` and `wkv` — ~107 MB per layer × 43 layers. The
compressed *KV cache* is tiny (§4); the *projections that produce and consume the latent* are
dense FP8 matrices read in full on every single decode step. This is exactly the trap §4.1 of
the directive warned about, and it is the dominant term.

### The autoregressive wall

Achievable bandwidth is **measured, not inherited** — `tools/bw_probe.cu` gives **240 GB/s**
(212 GB/s under memory contention), well above the ~200 GB/s planning figure carried from prior
projects. See `HARDWARE.md` §2.

```
@ 212 GB/s (contended)  :  18.92 tok/s   (52.8 ms/tok)
@ 240 GB/s (achievable) :  21.42 tok/s   (46.7 ms/tok)   <- the operative wall
@ 273 GB/s (spec peak)  :  24.37 tok/s   (41.0 ms/tok)
```
KV traffic is excluded from `B_tok` above and is negligible by comparison — at 8K context it adds
~27 MB/token-step of cache reads against 11.2 GB of weights, under 0.3%.

---

## 3. The anchor: a direct measurement, not a scaling argument

This is the single most useful finding of Phase 0.

**The prior project's 180B checkpoint has a bit-identical per-token byte profile to this one.**
Running the same accounting over the local `~/models/DeepSeek-V4-Flash-180B` shards gives
`B_tok = 11.202 GB` — the same number to the megabyte, component by component (MLA 4599.48,
experts 3449.29, shared 1082.20, …). The two checkpoints differ only in post-training and in the
3 embedded MTP blocks; the 43-layer backbone geometry is identical.

Therefore the prior project's **measured** result transfers directly, with no scaling assumption:

| | value |
|---|---|
| Measured warm decode, 180B, this box | **126.7 ms/tok = 7.89 tok/s** |
| Implied effective bandwidth | **88.4 GB/s = 37% of the 240 GB/s achievable** (32% of spec peak) |
| Optimisation history behind it | 9 gated wins, 0.50 → 7.89 tok/s (15.9×), full 43-layer CUDA graph captured bit-exact, measured at parity ⇒ GPU-bound, not launch-bound |

> Note: the prior project's own docs state "8.77 GB/tok → ~25% of peak". That figure undercounts;
> the correct `B_tok` is 11.20 GB and the correct efficiency is **32%**. The conclusion is
> unchanged — the gap is bandwidth efficiency, not algorithm — but the headroom is slightly
> smaller than the prior write-up claims. `tools/inventory.py --model-dir ~/models/DeepSeek-V4-Flash-180B`
> reproduces this.

**Well-written batch-1 decode kernels reach 70–80% of achievable bandwidth.** Closing
37% → 70–80% is a **1.9–2.2×** kernel-efficiency win, landing base AR decode at
**15.0–17.1 tok/s**, against a wall of 21.4 tok/s. Unlike the earlier 200 GB/s framing, the
target and the wall no longer coincide — there is genuine (if modest) room above the 70–80%
band, but **the base AR ceiling on this hardware is ~21 tok/s, and speculation is the only route
past it.**

### Projected band (base AR, no speculation)

| | tok/s | basis |
|---|---|---|
| Today, ported as-is | **~7.9** | direct measurement, identical `B_tok` |
| Realistic after kernel work | **15–19** | 70–80% of the measured 240 GB/s |
| Hard ceiling | **21.4** (240 GB/s measured) / 24.4 (273 spec) | roofline |

**Confidence: HIGH** — unusually so for a Phase-0 projection, and much higher than the directive
anticipated. This is not an MLA/DSA/DSpark extrapolation from a GQA model; it is the same
backbone geometry, the same box, the same toolchain, already measured. The residual uncertainty
is in the *0731 lineage's* post-training, which does not affect bandwidth at all.

---

## 4. KV cache — confirmed not to be the binding constraint

Marginal cost per token, derived from `compress_ratios`, `head_dim` (512 = MLA latent width),
`index_head_dim` (128) and `sliding_window` (128). Pure-sliding layers (0 and 1) have a fixed
128-token window and contribute **zero** marginal growth.

| KV dtype | marginal | fixed window | 1M context |
|---|---:|---:|---:|
| fp8 / int8 | 3.36 KiB/tok | 2.7 MiB | **3.21 GiB** |
| bf16 | 6.72 KiB/tok | 5.4 MiB | 6.41 GiB |
| fp32 (what the prior engine allocates) | 13.44 KiB/tok | 10.8 MiB | 12.83 GiB |

**Verdict: weights are the binding constraint, KV is not** — as the directive predicted, and for
the reason it gave. But two corrections to the numbers it carried forward:

- The directive's prior of **"~37.7 KB/token marginal"** (from the 180B project's vLLM-era
  measurement) is **~11× above** what this checkpoint's own compression schedule implies even at
  fp32. Do not budget against it. The likely explanation is that the measured figure came from a
  runtime that did not implement the ratio-4/ratio-128 compressor, so it paid full per-token KV.
- **The prior CUDA engine stores KV in fp32** (`~/dspark-cuda-reap-finetune/src/decode.cu:105-108,243`).
  At 16.6 GiB of headroom that caps usable context near ~1M and wastes bandwidth. **Moving the
  KV cache to fp8 is a straightforward 4× KV win** and should be an early item — it is
  memory-*negative*, which the hard constraint in `HARDWARE.md` §2 actively rewards.

At a realistic 32K serving context, fp8 KV costs **~105 MiB**. KV pressure is a non-issue here.

---

## 5. Speculative verify cost — `E_frac(k)`, `c_v(k)`, and where the break-even sits

**The structural fact that makes speculation attractive on this model:** 69.2% of `B_tok`
(7753 MB — MLA, compressor, indexer, shared expert, lm_head, norms) is **k-invariant**, read once
per verify pass regardless of how many tokens are being verified. Only the 30.8% routed-expert
term scales with the expert union. MLA being enormous *helps* here.

Union model, independent routing (an upper bound; adjacent tokens share experts, so reality is
smaller): `E[|union|](k) = 160·(1 − (1 − 6/160)^k)`.

| k | E[\|union\|] | `E_frac` | B_verify | `c_v` | per-token | needs `a ≥` to win |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6.00 | 1.000 | 11202 MB | 1.000 | 1.000 | 1.00 |
| 2 | 11.77 | 0.981 | 14522 MB | 1.296 | 0.648 | 1.30 |
| 3 | 17.33 | 0.963 | 17718 MB | 1.582 | 0.527 | 1.58 |
| 4 | 22.68 | 0.945 | 20793 MB | 1.856 | 0.464 | 1.86 |
| 5 | 27.83 | 0.928 | 23754 MB | 2.120 | 0.424 | 2.12 |
| 6 | 32.79 | 0.911 | 26603 MB | 2.375 | 0.396 | 2.37 |
| 7 | 37.56 | 0.894 | 29345 MB | 2.620 | 0.374 | 2.62 |
| 8 | 42.15 | 0.878 | 31985 MB | 2.855 | 0.357 | 2.86 |

Speedup `S(k) = a(k)/c_v(k)`, with `a(k) = Σ_{j=1..k} α^j + 1`:

| α \ k | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| 0.4 | 1.40 | 1.20 | 1.03 | 0.89 | 0.78 | 0.70 | 0.64 |
| 0.5 | 1.50 | 1.35 | 1.19 | 1.04 | 0.93 | 0.84 | 0.76 |
| 0.6 | 1.60 | 1.51 | 1.38 | 1.24 | 1.12 | 1.02 | 0.94 |
| 0.7 | 1.70 | **1.69** | 1.60 | 1.49 | 1.39 | 1.29 | 1.20 |
| 0.8 | 1.80 | **1.88** | **1.87** | 1.81 | 1.74 | 1.66 | 1.59 |

**Read-outs:**
- **Optimal k is small — 1 to 3, not 7.** The vLLM recipe's `num_speculative_tokens: 7` is a
  *bad* prior for this engine; it only pays at acceptance rates this model is unlikely to reach.
  `dspark_block_size = 5` is the mechanism's own natural width and is the value to sweep around,
  but expect the measured optimum to sit below it.
- **Realistic ceiling from speculation alone is ~1.5–1.9×**, not the 2.5–4× the prior project's
  planning documents hoped for. Combined with the base band (15–19 tok/s): **DSpark decode lands
  in the 22–36 tok/s range**, centred near ~28.
- The published external DSpark figure the directive cites (41.7 vs 26.1 tok/s ≈ 1.60×) sits
  **inside this band** and is consistent with it. Good corroboration, weakly weighted.

**Two caveats on this table, both flagged for empirical resolution in Phase 5:**
1. ~~It assumes the grouped MoE reads each activated expert exactly once~~ — **AUDITED AND
   CONFIRMED** (`LOOP_LOG.md` Finding 13). `k_build_tiles` groups rows by expert and each tile
   reads that expert's weights once; at M=5 no expert exceeds 5 rows so it is exactly one read
   per activated expert, giving the `|union| ≈ 28` the model assumes. **The inherited
   "expert-union dilation" hypothesis is refuted.** `c_v(5) = 2.62×` was reproduced here
   (336.1 / 128.1 ms) and the ~64 ms excess lies elsewhere — see caveat 2.
2. It prices **weight traffic only**, and says nothing about work that scales with M while
   weight traffic does not. This caveat originally guessed that better arithmetic intensity at
   M=k would make `c_v` come in **better** than modelled. **That guess was wrong** — measured
   `c_v(5)` is 2.62× against 2.120× modelled. With the MoE ruled out (caveat 1), the leading
   explanation is the **DSA indexer + sparse attention**, which run a top-512 selection and an
   irregular gather *per query position*: 5 positions cost ~5× the compute while re-reading the
   same weights once. At 36% of achievable bandwidth we are already substantially
   compute/latency-bound, so that term is exposed rather than hidden under the memory stream.

### The DSA verify-cost unknown — flagged, unresolved

`c_v(k)` above prices **weight traffic only**. DSA adds a term this model does not capture: for
each of the k verify positions the lightning indexer runs a selection over prior tokens and the
sparse attention gathers `index_topk = 512` KV entries. Whether the k positions' selections
overlap determines whether that gather is paid once or k times, and sparse gathers have
irregular access patterns that fight coalescing.

**This has no analog in `gemma-cuda-hybrid` or `laguna-s1-cuda-server`, and — importantly — the
prior 180B project never isolated it either.** It stays an explicit unknown, but it is no longer
merely hypothetical: with the MoE hypothesis refuted (caveat 1), **DSA is now the prime suspect
for the whole 2.62× vs 2.120× gap.** Resolution: per-kernel wall-clock at M=1 vs M=5 plus `ncu`
`Compute (SM) Throughput` (the metric that is honest on Thor — `HARDWARE.md` §3) on the
indexer/sparse-attention kernels. If those scale ~5× while the MLA and MoE GEMMs stay flat, the
fix is a batched indexer that shares selection work across verify positions.

---

## 6. Priority order — corrected by the numbers above

The directive's Phase-7 order leads with DSA sparse attention. The byte accounting says otherwise.

| rank | lever | why | evidence |
|---|---|---|---|
| **1** | **MLA projection GEMV bandwidth** (`wq_b`, `wo_a`, `wo_b`) | **41.1% of `B_tok`** — the largest single term, larger than all routed experts | §2 |
| **2** | **MXFP4 MoE GEMV with hardware `cvt.f16x2.e2m1x2` unpack** | 30.8% of `B_tok`; the prior project's rejected FP4 GEMV used *scalar* nibble decode, so the rejection does not cover the HW-unpack path — and `__nv_cvt_fp4x2_to_halfraw2` is confirmed working on `sm_110a` | §2, `HARDWARE.md` §3 |
| **3** | **Fuse the attention/indexer/compressor glue** (RoPE + norm + act-quant + KV-write) | currently separate kernels each paying a DRAM round-trip through the arena; vLLM reports 2–20× on exactly these fusions | prior `DECODE_GAP_RESEARCH.md` T1.2 |
| **4** | **fp8 KV cache** (from fp32) | 4× KV traffic and footprint, memory-*negative* | §4 |
| **5** | **M=k verify expert-union dedup** | resolves the 2.6× vs 2.12× discrepancy; flips speculation from marginal to worthwhile | §5 |
| **6** | **DSA sparse-attention kernel efficiency** | 2.5% of `B_tok` in weights — but an unmeasured latency term, and the one genuine unknown | §5 |

DSA drops from #1 to #6 **on weight bandwidth**, but it is deliberately kept on the list because
its *latency* contribution is unmeasured and its verify-step behaviour is the project's one real
blind spot. If measured decode lands below this band, DSA is the first place to look.

---

## 7. What Gate R1 leaves open

1. ~~Achievable streaming bandwidth unmeasured~~ — **RESOLVED**: 240 GB/s measured
   (`HARDWARE.md` §2). Re-run idle to finalise; the sweep was taken under download contention.
2. ~~`ncu` permission-blocked~~ — **RESOLVED**, with a sting in the tail. `sudo ncu` works, but
   **Thor exposes no DRAM counters, so `ncu`'s "Memory Throughput %" is L2 throughput and does
   not mean bandwidth utilisation** — a kernel measured at 89% of peak reports 30%
   (`HARDWARE.md` §3). **Bandwidth utilisation stays an analytical quantity** (byte model ÷
   wall-clock); `ncu` is for compute throughput, occupancy, warp stalls and cache hit rates.
   The prior project's planned "confirm Memory% vs Compute% per kernel" step would have been
   actively misled by this metric.
3. **The DSA verify cost** (§5) — no model, no measurement, no precedent.
4. **The 2.6× vs 2.12× verify discrepancy** (§5) — inherited unexplained from the prior project.
5. **0731-lineage acceptance rate is unknown.** The prior project measured 1.286/5 block
   acceptance with an *unpruned, separately-loaded 256-expert* head bolted onto a *different*
   checkpoint. This checkpoint's heads are embedded and REAP-pruned consistently with their own
   backbone, which should read higher — but "should" is not a measurement, and α is the single
   input the §5 table is most sensitive to.
