# research/SPEC_VERIFY.md — speculative verify cost, SOTA survey (2026-08-06)

Sources at the bottom. Vendor-reported and unreproduced numbers are marked.

## The reframing that matters

My byte model for `c_v` assumed **zero expert overlap** between verify positions:
`c_v(K) = (8.81 GB fixed + 0.575 GB x U(K)) / 12.26 GB`, with `U(5) = 30` giving exactly the 2.13
I had been quoting. Cohere measured the real curve on a 128-expert/top-8 production MoE under vLLM:
unique experts at batch 1/2/4 = **8.0 / 14.7 / 25.4** vs an independence prediction of 8.0/15.5/29.1
— **38.1% overlap between consecutive tokens** (independence baseline 0.118), a structural property
of natural-language routing, not a workload artifact.

Applying that discount, `U(5) ≈ 25` and the byte floor is **c_v = 1.89**, not 2.13. Measured is
**2.77**. So the coalescing gap is *larger* than I thought: 2.77 → 1.89 is worth **1.29x** on
speculation, not 1.18x.

**Our grouped MoE already deduplicates** (device counting-sort over all `bs*na` rows → one weight
read per activated expert), so there is no code fix here — the correction is to my *model*. Worth
instrumenting the actual distinct-expert count per verify to confirm the 0.76 discount holds for
top-6/160.

## External validation: our c_v is normal, not a bug

**DraftExpert** (arXiv 2607.24434), end-device MoE with offload: verification without truncation
costs **2.35–3.30x a one-token step for K=4**. Our measured **2.77 at K=5 sits inside that band.**
So a bandwidth-bound MoE verify is *expected* to cost ~2.5-3x; our coalescing defect is a bug layered
on top of a real structural tax, not the whole story.

**"Lossless but Not Free"** (arXiv 2607.17283), Apple M3 unified memory: **3 of 5 configurations
DECELERATE** (0.33x, 0.50x, 0.52x) because the quantized backend does matrix-*vector* products per
position below batch 8. Best config 1.61x at K=6. This is the closest published analogue to our box.

**The general claim "verifying K tokens costs the same as 1 because decode is memory-bound" is true
only for DENSE models with resident weights.** For sparse MoE it is false — MoESD and Utility-Driven
SD (MICRO'25) state the correction: verify cost is set by the expert *union*, and can exceed the
dense baseline when expert diversity is high. **Nobody has written up the 20-SM / single-user /
resident-weights corner. Our measurements are in unwritten territory.**

## DSpark — the reference implementations

**Correction: arXiv 2606.19348 is the V4 model paper. The DSpark paper is arXiv 2607.05147.**

Mechanism matches our checkpoint exactly: (1) DFlash-style block-diffusion parallel drafter emitting
all γ=5 tokens in one pass via **KV injection** of the target's context hidden states — that is what
our tap of backbone layers 40/41/42 is; (2) **sequential Markov head** fixing "suffix decay", as a
low-rank `B = W1 W2`, `r = 256` — our rank-256 markov head, exactly.

**Confidence head**: `c_k = σ(wᵀ[h_k; W1[x_{k-1}]])` estimates **P(draft token k survives verify)`,
calibrated by Sequential Temperature Scaling (ECE 3-8% → ~1%). It drives a hardware-aware throughput
scheduler, not a simple early-stop. **We are not using it at all yet.**

**Verify is LINEAR, left-to-right, with EXACT rejection sampling** — DSpark explicitly refuses
typical-acceptance. Confirms our implementation choice.

**The reference checkpoint ships NO verify loop** (`inference/generate.py` is pure AR, 5.8 kB), and
**deepseek-ai/DeepSpec is training + offline eval only**. The only reference verify implementations
are vLLM's and SGLang's. **Our CUDA verify loop is the third in existence and the only single-GPU one.**

Vendor numbers (unreproduced): accepted length 3.29–3.64 on MT-Bench/Alpaca/Arena-Hard (Qwen3-4B),
6.11/5.70/4.89 on GSM8K/MATH/AIME25; V4-Flash production +51% throughput vs MTP-1; batch-1 peak
383.7 tok/s at accept≈5 on V4-Pro TP=8 B300. SGLang profile: **verify is 81% of the step** (7.3 ms
vs 1.7 ms) — matching our 84.2%.

## What fine-tuning the head would realistically buy

Our 3.12/5 is from an **un-fine-tuned head against a REAP-pruned backbone**. The literature on
misaligned drafters: EDA gets τ 1.19 → 5.19 on a math model with 127 MB of adapters and 2 GPU-hours;
but aggressive verifier pruning "induces catastrophic distributional shift, acceptance ≈ 1.03" — we
are at 3.12, so our REAP mismatch is **mild, which caps the upside**.

**Realistic: 3.12 → 4.0–4.5, not 4.8.** DSpark's own *chat-domain* accepted lengths (3.29–3.64) are
close to our 3.12; the 6.11 figures are math/code only. **We may already be near the
domain-appropriate ceiling for conversational text.** Recipe: on-policy Spec-AUF-style fine-tune of
the 3 MTP blocks + markov head against the pruned backbone's own outputs.

## Ranked levers for us

| # | lever | effect | note |
|---|---|---|---|
| 1 | **Fix m16 B-coalescing** (Finding 28; now TMA per Finding 29) | c_v 2.77 → 1.89, **1.29x** | our own #1, externally confirmed as the right target |
| 2 | **Reduced draft vocabulary + `d2t` map** (EAGLE-2/3) | draft 54.7 → ~32 ms | **the sleeper — see below** |
| 3 | REAP-repair fine-tune of the MTP heads | a 3.12 → ~4.2 | 2 GPU-h of training, data-gen is the real cost |
| 4 | Retune K: 5 → 4 | ~+5% | 5th position survives p≈0.20 but costs a full union increment |
| 5 | Confidence-head early stop | ≤5% at batch 1 | literature says +7-11% but **~0 at batch 1**; we already have the head |

**The sleeper (#2).** Our `forward_head` is 32.38 ms = 59% of the draft. `lm_head` is 1059 MB BF16 =
4.41 ms at 240 GB/s; **five sequential full-vocab evaluations** = 22.1 ms + markov ≈ 32.4 ms at ~68%
of achievable BW. That reproduces the measurement exactly. EAGLE-2/3 solve this with a **reduced
draft vocabulary and a `d2t` index back to the target vocab** — the drafter never materialises
full-V logits. A 32k draft vocab cuts the term ~4x. Cheap, self-contained, **touches no verify
bit-exactness**.

## DO NOT DO (evidence-backed)

- **Trees.** Every node is a fresh draw from a 160-way router — the most expensive token type in the
  literature. Our own prior DDTree result already agreed.
- **Draft/verify pipelining** (PipeSpec/FlowSpec/SpecPipe). Nothing to overlap into: 20 SMs, one
  DRAM, both phases bandwidth-bound. Concurrency would just split 240 GB/s.
- **Expert prefetch.** Only pays with off-device experts; ours are resident.
- **Lower-precision verify** (Quasar, ~50% weight-byte cut, 1.28x). Already spent — we are MXFP4/FP8.
- **Cascade / early-exit verify** (HiSpec 1.28x). Needs a trained intermediate head and still pays
  the union tax.
- **Typical acceptance.** DSpark itself refuses it. Literature consensus 1-2% task-accuracy loss,
  and **nobody publishes a KL/TV measurement of the induced distribution shift.**

## Ceiling arithmetic

`tok/s = a(K) / (T_draft(K) + c_v(K)·105.2 ms)`, union discount fitted to Cohere's curve.

| stage | T_draft | c_v(5) | round | a | tok/s | vs base |
|---|---:|---:|---:|---:|---:|---:|
| now | 54.7 | 2.77 | 346.5 | 3.12 | 9.0 | 0.95x |
| + coalescing fix | 54.7 | 2.13 | 279 | 3.12 | 11.2 | 1.18x |
| + real union (model correction) | 54.7 | 1.89 | 254 | 3.12 | 12.3 | 1.29x |
| + reduced draft vocab | ~32 | 1.89 | 231 | 3.12 | 13.5 | 1.42x |
| + REAP-repair fine-tune | ~32 | 1.89 | 231 | 4.2 | 18.2 | 1.91x |
| + retune K 5→4 | ~27 | 1.61 | 196 | 3.75 | 19.1 | **2.01x** |

**Ceiling ≈ 19 tok/s (2.0x base).** With a *perfect* drafter (a=K) at K=8 the model gives ~28.5
tok/s — i.e. **the MoE expert-union tax caps speculation at ~1.33x the 21.4 tok/s AR roofline no
matter how good the drafter gets.** That is a *bytes* limit, not an acceptance limit.

## Sources

EVICT arXiv 2605.00342 · DraftExpert 2607.24434 · Cohere MoE-SD blog · DSpark 2607.05147 ·
DSpark-in-SGLang (LMSYS 2026-07-06) · EcoSpec 2607.12696 · AcceptMoE 2608.02989 ·
Lossless-but-Not-Free 2607.17283 · Quasar 2603.01399 · HiSpec 2510.01336 · EAGLE-3 2503.01840 ·
EAGLE 3.1 (vLLM) · EDA 2603.09527 · Spec-AUF 2607.01893 · SpecDec++ 2405.19715 ·
Windowed-MTP 2607.21535 · CaDDTree 2606.01813 · MoESD 2505.19645 · Utility-Driven SD (MICRO'25)
2506.20675 · DFlash 2602.06036 · github deepseek-ai/DeepSpec · SGLang Spec V2 · Fuzzy SD 2502.20704
