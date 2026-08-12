# If an NVFP4 REAP existed — what transfers, what is a translation, and what order to do it in

The cross-model comparison found this checkpoint decodes at half Qwen's rate on the same box because
it moves twice the bytes, and that the only lever with real volume behind it is precision on the
dense path. That raises a planning question with a real answer: **if the REAP were requantised to
NVFP4, does this project start over?**

**No. Most of it transfers untouched, the kernel work is a translation rather than a rewrite, and
the sequencing matters more than either.**

## 1. The sequencing, first, because it is the part that can waste months

**The dense MLA GEMV work and an NVFP4 requant attack the same term.** They are substitutes, not
complements. Priced against the measured `verify_ms(K) = 69.9 + 17.11K` and today's acceptance share:

| path | dense | AR step | AR | spec |
|---|---|---|---|---|
| today | 9.40 GB @ ~170 GB/s | 79.1 ms | 12.6 | **25.5** |
| **A:** dense GEMV first (fp8 → 228) | 9.40 GB @ 228 | 65.0 | 15.4 | ~29.9 |
| **B:** requant first (NVFP4 dense) | 4.70 GB @ ~170 | 51.5 | 19.4 | **~33.0** |
| B, then GEMV on the fp4 kernel | 4.70 GB @ 228 | 44.4 | 22.5 | ~34.8 |

The dense GEMV work is worth **14.1 ms at fp8 but only 7.0 ms after a requant** — half the bytes, and
a kernel that would have to be rewritten anyway. **Requant alone (~33) beats the entire dense GEMV
effort (~29.9).**

> **If an NVFP4 REAP is plausibly obtainable, requantise BEFORE doing the dense GEMV work.**
> Doing A then B wastes most of A.

## 2. Why the kernel work is a translation, not a rewrite

This is the load-bearing claim, and it is checkable in the source.

**The FP4 *element* format is identical.** MXFP4 and NVFP4 both store **E2M1 codes, two per byte**.
`kernels/tc_moe_gemm.cu:15` decodes them with the hardware instruction:

```
__nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)b, __NV_E2M1)
```

That line is already correct for NVFP4. So is the packing, the repack-at-load layout, the funnel, the
tile shape, the `mma_m16n8k16` path, the register budget and every occupancy result measured on it.

**Only the scale differs, in two mechanical ways:**

| | MXFP4 (ours) | NVFP4 |
|---|---|---|
| scale dtype | `F8_E8M0` — bare exponent, `exp2(b-127)` | `e4m3` — a real fp8 |
| group size | **32** | **16** |

In the kernel that is an index change (`k_tile/2` over `K/32` → `/16` over `K/16`) and a decode
change (`float` → `tcm_e4m3()`). **`tcm_e4m3` is already written — `tc_moe_gemm.cu:17` — sitting
unused in the MXFP4 kernel file.**

**And a full NVFP4 path already exists in this repo**: `kernels/cutlass_moe.cu`
(`nv_float4_t<float_e2m1_t>` with e4m3 block scales, plus `k_swizzle` converting linear e4m3
`[rows][K/16]` scales into CUTLASS's blocked SFA/SFB layout), `kernels/fp4_gemm.cu`,
`kernels/nvfp4_quant.cu`, `include/{fp4_gemm,cutlass_moe,nvfp4_quant}.h`, and `src/draft.cu` already
uses it.

## 3. The transfer matrix

### Transfers untouched — the majority of the elapsed time

- **The measurement machinery**: the frozen 8-prompt suite, the protocol (NGEN0 >= 200, clocks
  pinned, caches dropped, no instruments), `HEAD_REGISTRY.md`, `RUNS.md`, `head_off.sh`,
  `sweep_analyze.py`, and the **noise characterisation** — warm-up discarded, shuffled order, run
  index as a covariate, n=4 to resolve 1.5 %. That discipline caught three confident-but-wrong
  negatives (F119, F120, and the first adaptK sweep) and is entirely format-agnostic.
- **Every closed negative stays closed**: tree/multi-candidate (F122), activation sparsity (F123,
  dead at 1024 *and* 2048), block size (F43), draft refinement (F45), K-selection policy (F110).
- **The whole S5 pipeline**: `s5_session.sh`, the balanced corpus and `make_corpus.py`, the trainer,
  chunked resume, `s5_preflight.sh` — and **`gen.txt`, which cost 10.6 h of GPU and is pure token
  ids**. Requantisation cannot touch it.
- **The structural findings**: the `verify_ms(K) = a + bK` form, the acceptance-hazard method
  (`accept_profile.py`), the direction of the adaptK optimum (~2.0, confirmed 3x independently), and
  the ceiling arithmetic. The *constants* re-fit; the *form* holds.

### Transfers after re-fitting the numbers

`B_tok` (12.26 → ~7.6 if dense goes fp4), the `verify_ms(K)` constants, every per-kernel rate, the
roofline, and the adaptK optimum (expect ~2.0 again, but re-measure — it is cheap and this project
has been wrong about it before).

### Transfers with a re-quantisation step: **the fine-tuned draft head**

`build_trained_head.py` already re-quantises trained tensors into whatever formats the target
checkpoint uses, and it is self-checked (refuses any tensor whose round-trip exceeds 0.10). Pointing
it at NVFP4 formats is a modest change with `nvfp4_quant.cu` already present.

**The honest caveat**: `s3` was trained on activations captured from the **MXFP4** backbone.
Requantised, the backbone computes a slightly different function, so acceptance will drop by an
unknown amount. Recovery is a **short re-tune from s3's weights on a fresh capture** — the corpus and
`gen.txt` survive, so it is capture + train (~6 h), not another 19 h session.

### Needs real work

1. The hand-written MXFP4 expert GEMV → NVFP4: the scale translation above, plus re-tuning occupancy
   (twice as many scale loads at group 16 changes the register and bandwidth balance — the
   optimisation *findings* transfer, the optimal constants do not).
2. The **fp8 dense GEMV → fp4 dense GEMV**, which is a larger change than the MoE one: the element
   size halves (1 B → 0.5 B) and a scale appears where there was a 128x128 block scale. But the
   structural conclusions in `dense-mla-gemv.md` — warps/SM vs N, ILP and outstanding loads, the
   accumulator dependency chain, split-K for `wq_a`/`wkv` — are properties of GEMV on `sm_110a` and
   survive the format change intact.

## 4. What could make this not work

- **Nobody may ship an NVFP4 REAP.** This whole page is conditional on obtaining one, or on running
  the quantisation locally (`llmcompressor` and an `nvfp4-quantize` venv are both on this box; a 180 B
  requant needs calibration data and real compute, and has not been scoped here).
- **Quality is not free.** NVFP4 on the *dense/attention* path is the risky half — the routed experts
  are already 4-bit and quantising them further is not on the table. This project's whole comparison
  framework rests on `LOSSLESS: emitted tokens identical to base AR`, and that invariant is against
  *whatever checkpoint is loaded*. A requantised checkpoint is a **different model**, so every
  capability number would need re-measuring, not just every speed number.
- **The estimates above are extrapolations.** They assume the fp4 dense GEMV reaches the same GB/s as
  the fp8 one, which is the optimistic case: fewer bytes per element usually means *lower* achieved
  bandwidth, not the same.

## 5. The one-line answer

**Requantise first if you are going to at all; the kernel work is a scale-format translation on top
of an FP4 element path that is already correct and already partly built; and the corpus, the trained
head, the measurement discipline and every negative result carry over unchanged.**

## 6. SURVEYED: what is actually on HuggingFace, and why it does not change the picture

Section 4 said this page was conditional on obtaining an NVFP4 REAP. Surveyed 2026-08-11, and the
conditional **mostly fails** — for a reason that is worth stating precisely, because it is the
opposite of the intuition:

| candidate | pruning | experts | **dense / MLA** | note |
|---|---|---|---|---|
| **ours** (`0xSero/...-0731-REAP`) | K160 | **MXFP4** | **FP8 e4m3** | 107.8 GB |
| `0xSero/DeepSeek-V4-Flash-180B` | K160 | NVFP4/MXFP4 | **BF16** | ~103 GB, FP8 KV |
| `RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8` | none (163 B full) | NVFP4 | unspecified | "noticeably lower accuracy recovery" |
| `nvidia/DeepSeek-V4-Flash-NVFP4` | none (full) | NVFP4 | unspecified | too large for 122.8 GiB |

**"NVFP4" in these cards means the ROUTED EXPERTS are NVFP4.** Ours are already MXFP4. Both are
**4 bits per weight plus a block scale** — NVFP4 vs MXFP4 is a change of scale format (e4m3/group-16
vs E8M0/group-32), **not a change of size**. Swapping them moves `B_tok` by approximately nothing.

**And the bytes that actually matter are not quantised in any of them.** The dense/MLA path is
**72 % of `B_tok`**, and it is BF16 or FP8 in every available variant — never NVFP4. The `0xSero`
180 B is the closest lineage match to ours and its dense path is **BF16**, which is *worse* for
decode than the FP8 we already have.

So the honest conclusion is the inverse of the hope that motivated this page:

> **This checkpoint — MXFP4 experts + FP8 dense — is already the most decode-favourable
> quantisation publicly available for this model family.** There is no free swap. The variant that
> would change the picture is one with the *attention* path in NVFP4, and that is exactly the
> quantisation nobody ships, because it is where quality damage concentrates — the one card that
> comes closest reports "noticeably lower accuracy recovery than the base model".

Everything in sections 1-5 stays valid as a **plan for a requant we would have to do ourselves**
(`llmcompressor` and an `nvfp4-quantize` venv are both on this box). It is no longer a plan for a
download. And the sequencing rule survives intact: if that requant ever happens, it comes **before**
the dense GEMV work, not after.

Sources: [0xSero/DeepSeek-V4-Flash-180B](https://huggingface.co/0xSero/DeepSeek-V4-Flash-180B) ·
[RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8](https://huggingface.co/RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8) ·
[nvidia/DeepSeek-V4-Flash-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4-Flash-NVFP4)

**Caveat on this survey**: the per-layer formats above come from model cards read via web fetch, and
those cards are frequently vague about which components get which scheme (RedHat's explicitly is).
The `0xSero` 180 B BF16-dense claim in particular is surprising given its 103 GB size and deserves a
header-level check against the actual safetensors before anything is decided on it — the same check
`tools/` already does for our own checkpoint.
