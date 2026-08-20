# KV precision — the packing is already computed, only the storage is wrong

**2026-08-19.** Fourth research front. The headline is not a tradeoff to evaluate; it is a defect to
fix: **every value in the KV and DSA index caches already sits exactly on its target quantization
grid, with an exact power-of-two scale, and is then stored in FP32.** Packing them into real FP8 and
MXFP4 is **bit-exact**.

Marks: **[M]** verified by me on this box, **[R]** in-repo, **[X]** external with citation,
**[D]** derived arithmetic.

---

## 1. The finding, verified at the source

`kernels/mla_attn.cu:216` and `:862` **[M]**:

```c
// act_quant_fp8sim_kernel                                   // act_quant_fp4sim_kernel
float scale = exp2f(ceilf(log2f(amax/448.f)));  // pow2(ue8m0)  scale = exp2f(ceilf(log2f(amax/6.f)));
float q = clamp(xr[i]/scale, -448.f, 448.f);                  float q = clamp(xr[i]/scale, -6.f, 6.f);
xr[i] = (float)__nv_fp8_e4m3(q) * scale;                      xr[i] = round_e2m1(q) * scale;
```

Both write the **dequantized** value back into an FP32 buffer. So:

- the value is exactly on the **E4M3** grid (KV) or the **E2M1** grid (index), and
- the scale is `exp2f(ceilf(...))`, an **exact power of two**, representable losslessly in **one
  UE8M0 byte**.

Applied at every cache write path **[M]**:

| site | call | what it covers |
|---|---|---|
| `compressed_decode.cu` (10 sites), `mla_decode.cu`, `mla_forward.cu`, `dspark_attn.cu`, `compressor.cu:501` | `act_quant_fp8sim(x, rows, NOPE_DIM=448, 64, HEAD_DIM=512)` | KV dims **0-447** only |
| RoPE dims 448-511 | never passed to any `act_quant` | untouched FP32 |
| `compressor.cu:500`, `:541`; `compressed_decode.cu:510` | `hadamard(...)` then `act_quant_fp4sim(..., 32, 128)` | **all 128** index dims |
| `compressed_decode.cu:231`, `:435`, `:581` | same, on the index **query** | |

Caches are append-only (`k_commit_comp`, `k_append_at2`) **[M]**, so packed storage is trivially safe.

## 2. The model's own paper specifies exactly this

DeepSeek-V4 tech report **[X, arXiv 2606.19348]**, §2.3.4 verbatim:

> "we adopt a **mixed storage format for KV entries: BF16 precision is used for the rotary positional
> embedding (RoPE) dimensions, while FP8 precision is applied to the remaining dimensions.** …
> **attention computation within the lightning indexer is performed in FP4 precision**"

§5.2.1, on FP4 QAT: the indexer QK path is *"cached, loaded, and multiplied entirely in FP4"*, index
scores go FP32→BF16, and this *"achieves a **2x speedup for the top-k selector**, while preserving a
**99.7 % recall rate** of KV entries."*

**We already compute the paper's format and then throw the saving away at the store.** SGLang ships
the DSV4 index pool at **68 B/token** under `enable_deepseek_v4_fp4_indexer` **[X]** — which is
exactly the number below.

## 3. What packing buys

Per row **[D]**:

| row | FP32 today | bit-exact packed | with BF16 RoPE (lossy) |
|---|---:|---:|---:|
| KV (512 dims) | 2048 B | **711 B (2.88x)** | 583 B (3.51x) |
| DSA index (128 dims) | 512 B | **68 B (7.53x)** | — |

The byte model reproduces `EVALS.md`'s measured **99.4 KiB/token** exactly **[D]**, which is good
validation of the arithmetic. Packed: **27.9 KiB/token, 3.56x**.

**The prize is capacity, and it is decisive** **[D]**:

| seqmax | KV now | packed |
|---:|---:|---:|
| 8 192 (today) | 0.78 GiB | 0.22 GiB |
| 32 768 | **3.11 GiB — does not fit** | 0.87 GiB |
| 65 536 | 6.21 GiB — impossible | **1.74 GiB — fits** |

`EVALS.md` records seqmax as *the* binding constraint on which items can run at all (GPQA-Diamond
items that fit: **0 at seqmax 4096, 198 at 8192**) **[R]**. **Packing is what buys 32k-64k context**,
and it costs no accuracy at all.

## 4. Bandwidth is NOT the reason to do this

Context-dependent bytes at 24k go **89.8 MB → 18.0 MB (4.99x)**, i.e. 0.392 → 0.071 ms at 240 GB/s
**[D]** — against a fitted context term of hundreds of ms. `COMPRESSION_PLAYBOOK.md` §0 is the
standing rule and it applies to itself here: *byte reduction only pays in proportion to how
bandwidth-bound you already are*, and the index path is **389x off its own roofline**, i.e. not
bandwidth-bound at all. **Fix the kernels first; pack for capacity and for L2 residency.**

One real second-order effect: `sparse_attn_kernel` runs one warp per (b,m,head) and MLA shares one
latent row across all 64 heads, so each cache row is re-read **64x per layer** — ~2.6 GB of L1/L2
traffic per decoded token **[D]**. Packing the whole context-dependent working set to **16.2 MB**
brings it inside Thor's 24 MB persistable L2 window. **Hypothesis, not measurement.**

## 5. What is lossy, and the evidence against doing it

**Do not demote the main KV to FP4.** SGLang's measured per-tensor FP4 KV **[X]**, on DeepSeek-R1
(an MLA model, so this reads directly onto us):

| task | BF16 | FP8 | FP4 |
|---|---:|---:|---:|
| gsm8k | 0.9157 | 0.9154 | 0.9124 |
| **aime25** | 0.5067 | 0.4934 | **0.4000 (-10.7)** |
| gpqa_diamond | 0.7707 | 0.7697 | 0.7273 |

and on GPT-OSS-120B, aime25 **0.7533 → 0.3533 (-40.0)** while gsm8k moves 0.001. **GSM8K is blind to
a change that costs 40 points on AIME.** Any future KV-precision gate must be built on AIME-class
long-CoT tasks, never on GSM8K.

**RoPE dims are the fragile part, and we already keep them FP32.** SnapMLA measures the RoPE
component spanning ±10^3 with outlier tails while the content component sits within ±10^1, and FP8
on RoPE giving *"an order-of-magnitude increase in MSE"* **[X]**. vLLM's `fp8_ds_mla` layout keeps
RoPE in BF16 for exactly this reason — *"This part is not quantized for accuracy."* Demoting our
RoPE dims to BF16 (711 → 583 B) is the **only lossy step** on the table and should be decided on a
measurement, not on the byte count.

## 6. Two hazards specific to this engine

**tau is a hair-trigger numerics detector — and a trap.** `LOOP_LOG` records a half2 dequant change
that produced a **byte-identical** generated token sequence and passed the lossless gate, yet
collapsed DSpark acceptance **3.12 → 1.00** **[R]**, because acceptance is an exact token comparison
between draft and target and the perturbed logits broke agreement. **Any format change must be
applied atomically to the target and the MTP/draft paths, and every A/B must report tau.**

**Never materialize a dequantized copy of the pool.** SGLang #35291 measures exactly that mistake at
**43.7 % of decode step time** (13.77 of 31.52 ms) on R1 at 131K **[X]**; an AMD MLA case made the
kernel 34 % faster and the end-to-end **961 ms slower** because the cast kernels cost 8,594 ms
against 94 ms saved. Dequantize inside the inner loop, keep Q in BF16, accumulate FP32.

**And never store FP32 scales.** That is what made llama.cpp `q4_0` KV use *more* RSS than `f16` on
DGX Spark **[X]**. Our scales are already exact powers of two; UE8M0 stores them in one byte. FP32
scales would cost 24 % of the packed index row.

## 7. Validation — and why most of this needs none

The battery cannot detect a 1-point regression and never could. `EVALS.md`'s own CI table gives
GPQA-Diamond ±5.0 and AIME ±15.6 at n=30 **[R]**; the power formula needs **~20,000 items** for
delta=1 pt unpaired **[X, arXiv 2411.00640]**. A six-day re-run would tell us nothing.

**Which is the argument for the bit-exact path.** For the packing work the correct acceptance test
is `memcmp` on generated token ids plus the existing lossless gate — **not** a benchmark. Required n
scales with the flip rate, and at a flip rate of zero it is zero.

If a lossy step is ever taken, the proxy hierarchy is: **top-k selection overlap** (target: DeepSeek's
own **99.7 %**), then **KL** (Spearman 0.981 against answer-flip rate **[X, arXiv 2407.09141]**),
then a **generation-length KS test plus truncation count** to cover the free-running degeneration
that teacher-forced metrics are structurally blind to, then a **needle sweep at 2k/4k/8k/16k/24k/32k**
— because vLLM #52109 is precisely our exposure: **3/3 needles at 3,611 tokens, 0/3 at 5,294**,
silently, from a split-KV length heuristic.

## 8. Ranked

| # | change | bytes | accuracy risk | validation |
|---|---|---|---|---|
| **1** | Pack the DSA index cache as real MXFP4 (128xE2M1 + 4xUE8M0 = **68 B**, was 512 B) | **7.53x** | **none — bit-exact** | `memcmp` + lossless gate |
| **2** | Pack KV dims 0-447 as real FP8 E4M3 + 7xUE8M0 (**711 B**, was 2048 B); RoPE stays FP32 | **2.88x** | **none — bit-exact** | same |
| **3** | `index_score` → `tcgen05.mma...kind::mxf4.block_scale.block32` GEMM consuming #1 natively | — | not bit-exact (accumulation order) — gate it | first-8-token match + tau + top-k overlap |
| **4** | Parallelise `k_topk_masked` off its single thread | — | selection-order ties only | top-512 overlap |
| **5** | RoPE dims → BF16 (583 B) | +18 % | **small but real — the only lossy step** | full §7 machinery |
| — | main KV → FP4; per-layer mixed precision | — | **rejected** — §5, and §4 of the peer survey | |

**Sequencing.** #1 and #2 are pure storage refactors with a `memcmp` acceptance test and zero eval
exposure. #3 and #4 are the actual performance work. #5 waits until #3/#4 have made the engine
bandwidth-bound enough for it to be measurable at all.
