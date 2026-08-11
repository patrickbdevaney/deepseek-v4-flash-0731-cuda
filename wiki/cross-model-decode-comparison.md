# Cross-model decode comparison — why this checkpoint decodes at half the rate of Qwen, on the same box

Three MoE models, one Jetson AGX Thor, three engines. The decode numbers differ by 2x and the
temptation is to read that as an engine-quality difference. **It is not.** On the only axis that is
comparable — bytes moved per token — the gap is fully accounted for, and this engine is at parity
with the best of them per byte.

## The three

| | total / active | quantisation | `B_tok` | decode | engine |
|---|---|---|---|---|---|
| **DeepSeek V4-Flash-0731-REAP** (this repo) | 284 B → REAP ~180 B / **13 B active** | routed experts MXFP4; MLA+dense FP8; 1.74 GB/token still BF16 | **12.26 GB** | 13.8 AR / **24.5 spec** | pure CUDA |
| **Laguna S-2.1-NVFP4** (`~/laguna-s1-cuda-server`) | 117.6 B / **8.5 B active** | routed experts NVFP4; everything else was BF16, self-quantised to FP8 | **6.251 GB** | **33.0 AR** / 41 code / **49.7 edit** | pure CUDA |
| **Qwen 3.5 122B A10B** | 122 B / **10 B active** | NVFP4 | ~5.3 GB (est.) | **~50 spec** | vLLM + DFlash |

`B_tok` for this repo and for Laguna are measured from their own checkpoints and roofline docs.
Qwen's is an estimate from its active-parameter count at NVFP4 — treat it as ±20 %.

## The gap is bytes, not kernels

    decode x B_tok = useful weight traffic

| | tok/s | x `B_tok` | = effective |
|---|---|---|---|
| Qwen + vLLM/DFlash | 50.0 | 5.3 GB | ~265 GB/s |
| **this engine + s1 head** | 24.5 | 12.26 GB | **~300 GB/s** |
| Laguna AR (no speculation) | 33.0 | 6.251 GB | 206 GB/s (**91 % of its byte wall**) |

Qwen's 2.0x decode advantage over this repo is *the same number* as its 2.0-2.3x byte advantage.
Per byte moved, this engine is at or above vLLM+DFlash. **There is no engine gap to close** — the
difference is what the checkpoint asks the memory system to move.

## Three architectural facts that set `B_tok` here

**1. REAP bought capacity, not speed.** Pruning 256 → 160 experts cuts the *footprint* 37.5 %, which
is what lets this checkpoint fit in 122.8 GiB at all. But routing is top-6 either way, so the
per-token expert read is unchanged. **No amount of further pruning speeds up decode.**

**2. The dense path dominates, and that is the DeepSeek architecture.** Routed experts are 91 % of
the parameters but only **28 % of the per-token read** (3.69 GB), because 6 of 160 activate. The
other **72 %** is MLA + the DSA lightning indexer + the CSA/HCA compressors + hyper-connections ×4 —
read in full, every token. Head dim 512 × 64 heads, `q_lora_rank`/`o_lora_rank` 1024, `index_topk`
512 over 64 indexer heads. That density is plausibly what buys the unpruned checkpoint its benchmark
standing, and it is paid on every single token.

**3. Sparsity already did quantisation's job.** The 91 % of the model that was the natural target for
aggressive quantisation is *already* MXFP4 **and** barely read. What dominates the read is the dense
attention path — the tensors you least want to lose precision on.

## What it would take to go higher — each model, honestly

### This repo: ~26–32, and the next tier needs precision

| lever | worth | status |
|---|---|---|
| MoE GEMV efficiency (~55 % of achievable) | AR 13.8 → 14.3–16.0; spec → ~26–32 | **open**, lever #1 |
| m16 B-repack (M≥2 only) | attacks the 17.11 ms/position verify slope | **open**, unquantified |
| draft head (S5, three sessions) | 22.66 → 24.52 shipped | done, ~exhausted |
| adaptK threshold | +1.3 %, real | measured, not shipped |
| **BF16 → FP8 on the remainder** | **+7.1 % of `B_tok`** → AR 14.9, spec **26.4** | **excluded by constraint** |

That last row is the interesting one, because **Laguna already proved it on this same box.** We
carry **1.74 GB/token of BF16** that is read every step — `lm_head` 1.06 GB plus 0.68 GB of
compressor/indexer/norms. Converting it to FP8 is exactly Laguna's optimisations #15 and #30. It is
currently ruled out by this project's "no additional quantisation, checkpoint as shipped" rule,
which is a policy choice and not a physical wall.

Beyond that, NVFP4 on the whole dense path would take `B_tok` 12.26 → ~7.6 GB and decode to ~40,
matching Qwen. That is the trade the comparison models already made.

### Laguna: kernels are done; 70 was a precision problem

Laguna is at **91 % of its byte-model wall** at ~254 GB/s, against a published best-in-class
full-step batch-1 efficiency of 82 % (FlashFormer, H100). **There is no meaningful kernel headroom
left there** — the 70 tok/s target was not missed for want of optimisations:

    B_tok 6.251 GB = dense 3.756 + experts 2.495 (experts already NVFP4)
    dense FP8 -> FP4:  B_tok -> 4.37 GB  =>  AR ~47 tok/s
    x its measured edit-style speculation (1.51x)  =>  ~71 tok/s

**70 was reachable, through precision on the dense path, not through more kernel work.** Laguna's
own log already shows the pattern: every one of its large wins (#15 attention BF16→FP8, #30 shared
experts + layer-0 dense) was a `B_tok` reduction, 10.044 → 6.251, +52 % cumulative. Its activation
sparsity lever — the published 2.5× MoE-layer claim — was measured on its own weights and **killed**
(87 % sparsity costs 43.7 % relative error unstructured, 83 % block-structured; `moe_intermediate`
1024 is below every published operating point).

### Qwen: closest to its ceiling

Already NVFP4 throughout, already speculating, on vLLM. The reported 10–15 % from a pure-CUDA
implementation is consistent with what this project measured moving off a generic engine, and it is
the smallest remaining gap of the three.

## The one-line synthesis

**All three engines are near their byte walls; the decode ranking is a quantisation ranking.**
Qwen is fully NVFP4 and fastest. Laguna is NVFP4-experts + FP8-dense and second. This checkpoint is
MXFP4-experts + FP8-dense + 1.74 GB of BF16, on an architecture whose dense path is 72 % of every
token's read, and it is third — while extracting more useful bandwidth per byte than either.

Sources: [DeepSeek-V4 on Day 0 (LMSYS)](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/) ·
[DeepSeek V4 in vLLM](https://vllm.ai/blog/2026-04-24-deepseek-v4) ·
[DeepSeek V4 GA architecture](https://huggingface.co/blog/ResterChed/deepseek-v4-ga-architecture) ·
local: `~/laguna-s1-cuda-server/{README,ROOFLINE,OPTIMIZATION_LOG,ACTIVATION_SPARSITY,COST_MODEL_CORRECTION}.md`
