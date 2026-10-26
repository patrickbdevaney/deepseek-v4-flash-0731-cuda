# Deep-research prompt — maximum decode speed, from first principles

**Target system.** A from-scratch pure-CUDA inference server for `DeepSeek-V4-Flash-0731-REAP` on a
**single Jetson AGX Thor** (`sm_110a`, 20 SMs, 122.8 GiB unified LPDDR5X, CUDA 13.0). 43 MoE layers,
hidden 4096, 64 heads x head_dim 512, 160 routed experts top-6 + 1 shared, MLA + DSA attention,
hyper-connections with Sinkhorn. Embedded **DSpark** 3-block MTP draft head (`mtp.0/1/2`, tapping
layers 40/41/42). **Single stream, batch 1.** No Python on the hot path.

## 0. The first-principles frame — argue against this, do not restate it

```
base AR tok/s   = 1 / (B_tok / BW_eff + T_fixed)
spec tok/s      = tau / (T_draft + T_verify(block))
```

* `B_tok` = **12.26 GB/token** (measured; active experts only)
* `BW_eff` = **233 GB/s** achievable (measured, `tools/bw_probe.cu`)
* Compute peaks (measured, `tools/flops_probe.cu`): **FP32 5.45 / BF16 53.1 / FP8 92.2 TFLOPS**
* `T_fixed` = **22.3 ms of a 71.4 ms step is NOT bytes** (launch floors, Sinkhorn, activations)
* Byte-weighted realistic AR floor = **14.33 tok/s**; measured **13.83** = **97 % of it**
* Speculative: **22.15 tok/s** at `tau` = 2.89, block 5; efficiency factor **0.552** measured

**Therefore the only three ways past the wall are: (a) raise `tau`, (b) cut `B_tok`, (c) cut
`T_fixed`.** The research must be organised around those three, and must say which bucket each
technique falls in and how much it is worth *at batch 1 on one device*.

## 1. Hard constraints — a technique that violates one is disqualified, say so explicitly

1. **No additional quantisation.** The checkpoint is used as shipped (MLA/dense FP8 + block scales,
   routed experts OCP MXFP4). Techniques whose entire gain is "use fewer bits" are out.
2. **Single device, single stream.** Batching, tensor/pipeline parallel and multi-GPU are out.
   Weights are fully resident: **expert offload/caching techniques do not apply** (there is no miss).
3. **Losslessness.** Output must match base AR token-for-token. Relaxed/typical acceptance is a
   separate, opt-in axis — flag it, do not smuggle it in.
4. **Free VRAM with the model resident is 13.6 GiB.** A technique needing a second model or a large
   auxiliary structure must fit there.
5. **`sm_110a` reality:** `mma.sync` FP4 is BLOCKED; FP4 is reachable only via `tcgen05`
   (`UTCOMMA.4X`, SASS-verified, silicon NOT yet verified). TMA and `cp.async` work.

## 2. ALREADY ESTABLISHED — do not re-derive, do not re-search. Extend or refute only.

| finding | status |
|---|---|
| **DSpark** (arXiv 2607.05147) IS this architecture; gamma=5, markov rank 256, 3 MoE layers + mHC, window 128; production numbers on V4-Flash | known |
| Loss: `a_ce 0.1 / a_tv 0.9 / a_conf 1.0`, `w_k = exp(-(k-1)/gamma)` (== NVIDIA NeMo AutoModel defaults) | known |
| **FastMTP** (2509.18362): LR 5e-5 cosine, warmup 0.05, AdamW (0.9,0.95), batch 64, 3 epochs, position decay beta=0.6, 389.4 K self-distilled samples, frozen backbone | known |
| **Nebius LK losses**: `-log alpha`, hybrid `lambda = exp(-eta*sg[alpha])`, **eta=3**; released `nebius/MTP-DeepSeek-V3-0324` moves acceptance **3.20 -> 4.83 at T=0 (+51 %)**, 1 epoch from pretrained | known |
| **HASS** (2408.15766): +8-20 % over EAGLE-2 by training later steps on imperfect draft features | known |
| **EAGLE-3** (2503.01840): removes feature-prediction constraint, data scaling with no saturation | known |
| **DFlash** (LMSYS 2026-06): block diffusion + **KV injection into EVERY draft layer** (EAGLE injects only at the input, so the signal fades with depth); block 8-16; 4.86x on Qwen3-8B; beats EAGLE-3 at equal layers | known |
| **SuffixDecoding / n-gram / retrieval drafting**: **RETIRED BY OUR OWN MEASUREMENT** — oracle ceiling **+0.0 %**, 0 wins in 21 verifies (F80). Speculation hands a retrieval drafter its worst possible query. Do not propose it again without addressing that measurement. | refuted here |
| **Tree drafting**: OPT-Tree/Sequoia mean accepted length plateaus beyond ~140 nodes | known |
| **LayerSkip** 2.16x, **CLaSp** 1.3-1.7x, **SWIFT** 1.3-1.6x (self-speculative, no draft model) | known |
| **Batch-1 physics** (2605.30571): achieved fraction of peak bandwidth *decreases* as peak bandwidth rises (L4 81 %, H100 27 %); CUDA graphs 1.259x on H100 | known |
| Expert prefetch/caching (DuoServe-MoE, FineMoE, SliceMoE, MoE-SpeQ) | **not applicable** — offload-only |

## 3. The questions that actually decide our next move

### 3.1 Is fine-tuning OUR head worth it? (the +51 % question)
Nebius's +51 % was on DeepSeek-V3's MTP module, which the literature says was **trained for
first-token prediction and reused autoregressively**, degrading later positions. **Ours is DSpark:
semi-autoregressive, block-trained, with a Markov head and confidence head designed for the block,
and the DSpark paper trained it 10 epochs on Open-PerfectBlend.** So:
- Does a head already trained *for the block* retain the same headroom, or is +51 % mostly the
  recovery of a pathology we do not have?
- Find any measurement of fine-tuning a **semi-autoregressive / block-trained** drafter (not an
  autoregressively-reused MTP) and report the delta.
- Our measured acceptance is 2.89 on a 6-token trivia prompt but **3.57-4.00 on a 1023-token code
  prompt**. The DSpark paper reports 6.11 on math. How much of our gap is prompt/domain rather than
  head quality? What is the right way to measure acceptance so the number is comparable to papers?

### 3.2 What is the true ceiling of `tau x block` on ONE memory-bound device?
- Published acceptance-vs-block-size curves. Where does the verify cost overtake the acceptance gain
  at batch 1?
- **KV injection**: can it be retrofitted to an existing MTP head, or does it require training from
  scratch? What does it cost in memory and per-round latency?
- Is there a principled upper bound on `tau` for a given drafter capacity (scaling laws, 2505.07858)?

### 3.3 Cutting `B_tok` WITHOUT additional quantisation
This is the least-explored bucket and the most valuable if anything exists.
- Techniques that read **fewer expert bytes per token** while remaining lossless: expert pruning at
  inference, top-k reduction with compensation, shared-expert-only draft passes, activation-sparsity
  exploitation.
- **Weight streaming/reuse across speculative positions**: at block K the same expert weights serve K
  tokens — is anyone exploiting that beyond naive batching?
- Anything that makes the *draft* pass read fewer bytes than the verify.

### 3.4 Cutting `T_fixed` (22.3 ms of a 71.4 ms step)
- We already have a full-step CUDA graph (F44: 92.5 -> 79.3 ms/tok). What is left after graphing?
- Sinkhorn (20 iterations, hyper-connections) and other non-GEMM overheads: has anyone attacked
  these classes at batch 1?
- Persistent-kernel / megakernel decode: one launch for the whole step. Real numbers at batch 1?

### 3.5 Genuinely unexplored
- Anything published in the last ~12 months on **single-stream decode on unified-memory edge
  devices** (Jetson/Thor/GB10/Apple silicon) that we have not listed.
- Any technique that gets past the memory wall by a mechanism NOT in {speculation, quantisation,
  batching, sparsity}. Name it or state clearly that no such mechanism exists.

## 4. What a good answer looks like

1. A table: technique -> bucket (a/b/c) -> expected gain **at batch 1, one device** -> whether it
   violates a constraint in section 1 -> implementation cost.
2. **Ranked by expected gain per unit of engineering**, since engineering is the binding constraint.
3. Explicit disqualifications with the reason.
4. **Confidence markers**: published number / inferred from a figure / extrapolated. A wrong number
   copied confidently into a multi-day build is the expensive failure mode here — it has already
   happened once in this project (a head size taken from a paper about a different model).
5. Where a claim contradicts section 0's arithmetic, say so and show the arithmetic.
