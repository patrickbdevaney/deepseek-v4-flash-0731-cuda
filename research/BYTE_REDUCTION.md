# research/BYTE_REDUCTION.md — moving fewer bytes (2026-08-06)

## ⭐ NEW #1 RUNTIME-ONLY LEVER: intra-expert neuron activation sparsity

[arXiv 2605.08575](https://arxiv.org/abs/2605.08575): eight pretrained MoE models **1B–400B**, up to
**90% sparsity WITHIN each activated expert**, with *"no modification to the activation function or
model parameters"* — **training-free**, implemented as an execution-pipeline extension that skips
inactive neurons *after* expert selection. Measured 2.5x on the MoE layer, **1.2x end-to-end**.

Its motivating argument is exactly our situation: fine-grained expert granularity hits expert
collapse during training, so within-expert sparsity is the **complementary** axis to expert-level
sparsity. Corroborated by MoNE ([2510.05781](https://arxiv.org/html/2510.05781)): *"most neuron
activations within experts remain near zero"*, negligible loss to **60%** neuron pruning.

On our 4,531 MB of routed + shared expert bytes at 50% sparsity:

| realization | bytes saved | tok/s wall |
|---|---:|---:|
| ideal | 2,266 MB | 26.9 (+25.4%) |
| **71% realized** ([2511.04477](https://arxiv.org/abs/2511.04477): 1.55x measured vs 2x theoretical, zigzag layout aligning weight groups to sparsity patterns) | **1,609 MB** | **25.0 (+16.8%)** |
| 50% realized (conservative, our block-32 granularity) | 1,133 MB | 23.8 (+11.3%) |

**This is the only lever above +10% that needs NO opt-in artifact** — it does not touch the weights,
so it survives the project's no-additional-quantisation rule. Caveats stated plainly: no per-model
accuracy-vs-sparsity curve is published; and [2606.00567](https://arxiv.org/abs/2606.00567) measures
that element-level sparsity **overstates hardware-exploitable sparsity by up to 78 percentage
points**. Budget the conservative row until measured.

**WiSparse** ([2602.14452](https://arxiv.org/abs/2602.14452)) supersedes TEAL as the training-free
reference: **97% of Llama-3.1 dense performance at 50% sparsity, 21.4% end-to-end**, combining
activation magnitude with precomputed weight norms. Its criticism of TEAL — low-magnitude
activations can coincide with high-importance weights — is the failure mode to watch.

## Evidence that KILLS several tempting levers on THIS architecture

- **Fine-grained MoE is LESS robust to expert removal — DeepSeek say so themselves.** DeepSeekMoE
  Fig. 4: *"DeepSeekMoE exhibits greater sensitivity to the ratio of disabled top routed experts,
  indicating lower redundancy among routed experts."* **Our checkpoint is already 37.5% pruned on
  the architecture that tolerates it worst.**
- **Expert caching across tokens is dead, with a number.** Local Routing Consistency
  ([2505.16056](https://arxiv.org/html/2505.16056v2)) measures SRP across 20 MoE models:
  LLaMA-MoE-v2 78.2, Qwen3 54.1, Mixtral 49.4, **DeepSeek-V2-Lite 37.9, DeepSeekMoE 36.9 — among
  the lowest measured.** And **shared experts REDUCE local routing consistency**, so we are doubly
  disadvantaged. ~63% of expert selections change between adjacent 16-token windows.
- **Shared expert removal is catastrophic**: Pile loss 1.808 -> 2.414 at held-constant params and
  FLOPs — a bigger loss than removing 4 of 6 routed experts, for 1,082 MB. Worst bytes-per-nat in
  the report.
- **Expert-count pruning gives "<1% speedup"** despite the memory gain (TMLR
  [2406.02500](https://arxiv.org/html/2406.02500v3), Mixtral 12.5% Expert Drop). Our card confirms
  the framing: *"Top-k remains at 6, unchanged from the original."*
- **2:4 sparsity: arch-blocked AND arithmetically bad.** At 4 bits it stores 2x4 data + 4 index bits
  per group of 4 = 12 of 16 bits -> only **25%** saving (vs 44% at FP16). Wanda prices it at
  LLaMA-2-7B 59.71 -> 48.75. CUTLASS documents sparse GEMM for **SM100 only**; SM110 appears nowhere
  in its Blackwell docs; all block-scaled sparse variants are `tcgen05.mma.sp`.

## Levers that change the weights (opt-in artifact, not the shipped checkpoint)

- **MLA + shared expert -> MXFP4**: 2,663 MB, ~28.1 tok/s wall, **+0.59% PPL**, and it reuses
  `fp4_gemm.cu` machinery already gated. **Best bytes-per-unit-damage trade in the report.**
- **top-6 -> top-4**: DeepSeekMoE's own Fig. 5 (2B, 64 routed, top-6) shows **top-4 reaches Pile
  1.867 vs 1.808 — about +0.06 nats**. NAEE's skipping-only variant (gate renormalised over
  survivors) measures Mixtral 67.58 -> 66.37 (−1.21 pts) at 1.08x. Pure threshold dynamic-k has a
  low ceiling — Elbow-Based Routing buys only **5.3%**.
- **MoE-sublayer skipping on 8 of 43 layers**: 843 MB (+8.1%). MoE layers are markedly more
  redundant than dense (Mixtral drops 8 of 32 for ~1% MMLU vs 24.3% for dense Mistral), and Block
  Drop is far worse (−14.6 pts), which says **attention sublayers are the expensive half to remove**
  — directly relevant to our 41% MLA term. **But do not ship it**: pruning even 1–2 layers
  *"severely impairs test-time scaling, collapsing on long reasoning benchmarks"*
  ([2510.22228](https://arxiv.org/abs/2510.22228)), and deficits persist through 100B tokens of
  healing ([2602.01997](https://arxiv.org/abs/2602.01997)). The one existence proof needing
  distillation: E³-Pruner, Qwen3-32B, 25% of layers, MATH-500 96.0 vs 96.8.

## Two practical items for any artifact we do produce

1. **Run a router-only distillation pass after ANY requant or k-change.**
   [2603.02217](https://arxiv.org/abs/2603.02217): retraining-free compression induces
   **router–expert mismatch**, and Router-KD (distil the original next-token distribution on
   unlabeled data, update only the router) gives *"substantially larger gains in fine-grained MoEs"*.
   Our router is 160x4096 — hours, not days. Cheapest insurance available.
2. **SharQ** ([2606.26587](https://arxiv.org/abs/2606.26587)) is an accuracy-recovery tool for an
   MLA->MXFP4 artifact, not a competing lever: online sparse-dense decomposition, outliers to FP4,
   **no calibration data, no retraining**, recovers **43–63% of the NVFP4->FP16 accuracy gap**.
   First thing to try if the 4-bit MLA artifact lands short.

## Still open in the literature

- **No benchmark evaluation of any pruned DeepSeek checkpoint exists** — all cards state structural
  or smoke validation only, **including ours**: `0xSero` explicitly warns its importance scores came
  from *"a closely aligned earlier checkpoint, not a fresh 0731 observation"*.
- **No published head- or layer-pruning work on MLA-style attention at all.**
