# MODEL_INVENTORY.md — Gate A1 (part 1: static inventory)

> This file is the *incumbent*: what is on disk and what it is made of. For the forward-looking
> question — which other models fit this box, what they would decode at, and why the frontier
> open-weights do not fit at any 4-bit quantisation — see **[MODEL_SURVEY_APPENDIX.md](MODEL_SURVEY_APPENDIX.md)**
> (surveyed 2026-08-18).

`0xSero/DeepSeek-V4-Flash-0731-REAP` @ sha `ddc04540efda3d2a0788b129f1fad828ddc19b60`, lastModified 2026-07-31.

Checkpoint identity, from `REAP_MANIFEST.json` / `validation/structural-validation.json` (both copied to `docs/`):

| field | value |
|---|---|
| \`source_model\` | \`deepseek-ai/DeepSeek-V4-Flash-0731\` |
| \`source_revision\` | \`9e165c30e2704aec5d9d593cce3eebd58bbef1cb\` |
| \`prior_observation_model\` | \`deepseek-ai/DeepSeek-V4-Flash\` @ \`60d8d707\` (ranking transferred, not re-observed) |
| \`method\` | REAP transferred ranking with router identity alignment |
| \`kept_experts_per_layer\` | 160 of 256 |
| \`scopes\` / \`shard_count\` | 46 / 48 |
| \`tensor_count\` / \`tensor_bytes\` | 45,821 / 107,803,320,952 |
| structural validation | \`status: pass\`, `failures: []`, 12 protected + 92 router + 552 selected-expert tensors compared |
| license | MIT |

This is the **0731 lineage**, not the pre-0731 preview and not an NVFP4 requant. Directive §13.6 satisfied.

## Router alignment — no runtime remap required

\`\`\`
router_alignment.identity_nearest_all      = true
router_alignment.hash_tid2eid_equal_fraction = [1.0, 1.0, 1.0]   (the 3 hash-routed layers)
router_alignment.same_index_cos_mean_min  = 0.99237
router_alignment.same_index_cos_min_global = 0.94608
\`\`\`

Neither \`reap_plan.json\` nor \`REAP_MANIFEST.json\` contains a retained-expert ID list, and none is needed: \`gate.weight\` ships as \`[160, 4096]\`, \`gate.bias\` as \`[160]\`, expert tensors are dense \`0..159\`, and \`tid2eid\` for the hash layers is already remapped. **G6 has no remap step** — only an index-exact top-6 check against the oracle.

## Reference runtime smoke shipped with the checkpoint

From \`validation/runtime-smoke.json\` — note this was **not** run on Thor:

\`\`\`
hardware      NVIDIA DGX Spark GB10        engine        vllm 0.25.2.dev0+g752a3a504
moe_backend   flashinfer_b12x              router        vllm torch sqrtsoftplus fallback for K160
mtp_enabled   false                        max_model_len 8192
cuda_graphs   true                         status        pass
\`\`\`

The K160 torch-router fallback is a vLLM fused-kernel limitation (160 is not in its supported expert-count set) and **does not apply to us** — we write the router for whatever \`n_routed_experts\` reads from config. Also note \`mtp_enabled: false\`: the vendor smoke did **not** exercise the DSpark heads.

## Full tensor inventory

Regenerate with \`python3 tools/inventory.py\` (reads \`docs/config.json\` + \`docs/hdrs/*.json\`; add \`--model-dir\` to read a local checkout instead). Verbatim output:

```
==============================================================================
DeepSeek-V4-Flash-0731-REAP (K160) — inventory
==============================================================================
tensors            45,821
total tensor bytes 107,803,320,952  = 100.400 GiB = 107.803 GB
matches index.json total_size (107,803,320,952): True

--- bytes by dtype ---
  I8           86.250 GiB   85.9%
  F8_E4M3       5.871 GiB    5.8%
  F8_E8M0       5.391 GiB    5.4%
  BF16          2.730 GiB    2.7%
  F32           0.141 GiB    0.1%
  I64           0.017 GiB    0.0%

--- bytes by subsystem ---
  routed experts (MXFP4)        85.664 GiB   85.3%
  MTP (DSpark heads)             6.529 GiB    6.5%
  MLA attn                       4.284 GiB    4.3%
  shared expert (FP8)            1.008 GiB    1.0%
  embed                          0.986 GiB    1.0%
  lm_head                        0.986 GiB    1.0%
  KV compressor                  0.490 GiB    0.5%
  DSA indexer                    0.256 GiB    0.3%
  HC (hyper-connection) params     0.126 GiB    0.1%
  router                         0.070 GiB    0.1%
  norms                          0.001 GiB    0.0%
  hc_head                        0.000 GiB    0.0%

--- expert quantisation format (proof, not assumption) ---
  layers.5.ffn.experts.0.w1.weight  I8 [2048, 2048]  -> logical [2048, 4096] E2M1
  layers.5.ffn.experts.0.w1.scale   F8_E8M0 [2048, 128]
  scale block size along K = 4096 / 128 = 32
  => E2M1 data + E8M0 scale, block 32  =  OCP MXFP4  (4.25 bits/param effective)

--- REAP router layout (does a remap need applying at runtime?) ---
  gate.weight [160, 4096]  gate.bias [160]  experts present: 160
  config n_routed_experts = 160
  => router rows already compacted to the retained set; expert ids are dense 0..159.

--- B_tok: bytes read per M=1 decode step, 43 layers, top-6 ---
  MLA attn                        4599.48 MB   41.1%
  routed experts (top-6)          3449.29 MB   30.8%
  shared expert                   1082.20 MB    9.7%
  lm_head                         1059.06 MB    9.5%
  KV compressor                    525.72 MB    4.7%
  DSA indexer                      275.35 MB    2.5%
  HC params                        135.28 MB    1.2%
  router                            75.00 MB    0.7%
  norms                              0.70 MB    0.0%
  hc_head                            0.26 MB    0.0%
  embed (1 row)                      0.01 MB    0.0%
  B_tok TOTAL                    11202.36 MB  = 10.4330 GiB

--- autoregressive wall ---
  @   200 GB/s :  17.85 tok/s  ( 56.01 ms/tok)
  @   273 GB/s :  24.37 tok/s  ( 41.03 ms/tok)

--- active parameters per token (backbone, excl. embed lookup) ---
  13.26 B   (model card reports ~13B active)
  blended 6.76 bits/active-param  (NOT 4.25 — only the routed experts are MXFP4)

--- DSpark MTP blocks (embedded, REAP-pruned to 160 experts) ---
  mtp.0: resident  2.166 GiB | per-step non-expert  186.95 MB + top-6  80.22 MB =  267.16 MB
  mtp.1: resident  2.119 GiB | per-step non-expert  136.61 MB + top-6  80.22 MB =  216.82 MB
  mtp.2: resident  2.243 GiB | per-step non-expert  269.27 MB + top-6  80.22 MB =  349.48 MB
  MTP resident total 6.529 GiB

--- KV cache marginal cost per token (from config, per dtype) ---
  fp8 / int8                3.36 KiB/token marginal  +    2.7 MiB fixed window  -> 1M ctx =   3.21 GiB
  bf16                      6.72 KiB/token marginal  +    5.4 MiB fixed window  -> 1M ctx =   6.41 GiB
  fp32 (current engine)    13.44 KiB/token marginal  +   10.8 MiB fixed window  -> 1M ctx =  12.83 GiB
  (ratio-4 layers: 21, ratio-128 layers: 20, pure-sliding: 2)
```
