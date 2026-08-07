# research/MOE_DECODE.md — batch-1 MoE decode (2026-08-06)

> ## ⚠ CORRECTION APPLIED TO THIS REPORT
> The agent reported `tcgen05.* BLOCKED (confirms your finding)`. **That is wrong**, and it is the
> same target-flag trap as LOOP_LOG Finding 29/30: probing `sm_110` or building an executable with
> `-arch=sm_110a` (which also emits `compute_110` PTX) makes tcgen05 look absent.
> Verified here with `-gencode arch=compute_110a,code=sm_110a`:
> **`tcgen05.mma.kind::mxf4nvf4.block_scale` COMPILES and emits `UTCOMMA.4X`; alloc/dealloc RUN.**
> The agent's *other* probe results and all its measurements stand — it was working from the
> premise I gave it in the research prompt, which was wrong at the time.
> Everything below marked "arch-blocked because no tcgen05" must be re-read as **available**.

## ⭐ THE MEASUREMENT THAT MATTERS: our MoE shape hits 240 GB/s with output-row blocking

The agent built a kernel streaming exactly one layer's routed-expert working set in **our** layout
(6 experts x {w1,w3,w2}, MXFP4 E2M1 + E8M0 block-32, tile per (expert, n-block), 16 B `uint4`
loads, HW `__nv_cvt_fp4x2_to_halfraw2`, `__hfma2` accumulate, activation staged in shared once per
block). 76.5 MB moved. **`BN` = output rows per block:**

| BN | threads | blocks | GB/s | % of 240 |
|---|---|---|---|---|
| **1** | 128 | 36864 | **155-160** | 65-67% |
| **2** | **128** | 18432 | **242-249** | **101-104%** |
| 4 | 128 | 9216 | 222-235 | 93-98% |
| 8 | 128/256 | 4608 | 221-245 | 92-102% |
| 16 | 256 | 2304 | 211-222 | 88-92% |
| *(pure read, no dequant)* | 128 | 36864 | 240 | 100% |

**Mechanism:** at BN=1 every 16-byte weight load is paired with 16 shared-memory `LDS` fetches of
the activation. At BN>=2 the activation registers are reused across output rows and the instruction
count per weight byte roughly halves. Issue-rate budget: 20 SMs x 4 schedulers x 1.575 GHz =
**126 G warp-inst/s**; at 240 GB/s each warp's 512 B load must be consumed in <=~270 warp-inst, and
a BN=1 inner loop burns ~50 of them re-fetching X.

**Expected on our engine:** routed experts are 3449 MB/token; at our current 131 GB/s that is
26.3 ms of the 105.2 ms step, at 240 GB/s it is 14.4 ms. **-11.9 ms -> 9.51 -> 10.72 tok/s
(+12.7%)**; if the shared expert (1082 MB) uses the same kernel, **+17.5% -> 11.17 tok/s**.
Bit-exactness preserved if accumulation order within a row is unchanged.

**Caveat the agent flagged honestly:** a competing GPU workload held this box at 97-98% utilisation
throughout its session (the other research agents were benchmarking concurrently), so these are
LOWER bounds and absolute values are unstable. **Re-run on an idle box before committing.** The
BN=1 penalty is structural and will hold.

## Device facts measured on this box

L2 = **32 MB**, `persistingL2CacheMaxSize` = **24 MB**, `accessPolicyMaxWindowSize` = 128 MB,
**228 KB smem/SM**, 1536 threads/SM, 256-bit bus, `integrated=1`, `pageableMemoryAccess=1`,
EMC pinned at 4266 MHz (= 273 GB/s), `nvpmodel` = MAXN. **Clocks are already maxed — no free win there.**

Additional instruction findings (independent of the tcgen05 error):
- `cp.async.bulk.prefetch.L2.global` -> **`UBLKPF.L2`** — hardware async L2 prefetch exists.
- **`cp.async.bulk.tensor.2d ... tile::gather4`** compiles — SonicMoE's gather instruction is present.
- `ld.global.nc.L2::evict_first.v4.b64` -> **`LDG.E.EFL2.256.CONSTANT`** — 256-bit loads work, but
  only at `.v8.b32`/`.v4.b64` (32 B/thread); `.v4.u32` with L2 hints is rejected.
- `mma.m16n8k64.s4.s4.s32` "works" but ptxas **emulates it as 2x `IMMA.16832.S8.S8`** — INT4 is not
  native; INT8 is.

## What the GPT-OSS-120B MXFP4 stacks do, and the transferable lesson

- **llama.cpp**: no grouped GEMM at all — per-expert GEMV (`mmvq.cu`), `block_mxfp4` =
  `{uint8_t e; uint8_t qs[16]}` (**scale interleaved with data**, 17 B per 32 elements), FP4->int8
  via LUT, accumulate with **DP4A not tensor cores**. Reaches ~80% of SoL on a 5090. Also stores
  **`lm_head` as MXFP4** — worth 3.18 ms/token on GB10.
- **vLLM on GB10**: CUTLASS block-scaled FP8 activations x MXFP4 weights, 128x128x128 tile.
  Decode profile at 20.4 ms/token: **MoE GEMM 61%, QKV/O 20%, lm_head 6%, attention 4%, routing 9%.**
  Explicitly noted gap: *"No specialized small-M decode path (GEMV-based) for token-generation."*
- **SGLang on Spark**: MXFP4 routed experts, **FP8 for all dense/always-on linears and lm_head**.
- GB10 scoreboard (gpt-oss-120b): vLLM 60.0, llama.cpp 58, SGLang 52 tok/s.

**Transferable lesson: on this hardware class the DENSE path is the bigger prize.** Every stack that
won did it by taking dense linears and `lm_head` off BF16. Our shared expert is 1082 MB at FP8 while
routed experts are MXFP4 — **at MXFP4 it would be 575 MB, saving 507 MB/token = 4.1% of B_tok.**

## The 160-expert restriction: a router template, not a layout constraint

It lives in `vllm_topk_softplus_sqrt`, the fused *routing* kernel. Supported set
`{16,32,64,128,192,256,320,384,512}` = `experts_per_lane in {1/2,1,2,4,6,8,10,12,16}` over 32 lanes;
160 needs 5/lane, which was never instantiated. Note 192 and 320 are in the set, so it is not even
a power-of-two rule. **The GEMM is unaffected** ("Expert execution remains on the B12X MXFP4 MoE
backend"). Our router is 0.6% of `B_tok` and already device-side with no host sync — **ignore this
entirely.** The one real read: **GB10's 12-14 tok/s on this checkpoint is depressed by a torch
fallback router, so it is not a fair ceiling to measure ourselves against.**

## Levers that are DEAD for us, with the reason

- **Gather fusion / `tile::gather4` / token rounding / grouped-masked GEMM / MegaBlocks / ScatterMoE
  / Marlin-MoE** — all are *token-dimension* tile-efficiency fixes, degenerate at M=1. SonicMoE's
  redundant IO at T=1,K=6,d=4096 is 48 KB per layer against 80 MB of expert weights.
- **Expert prefetch / caching / sticky routing / hot-expert VRAM cache** — every published gain comes
  from avoiding a *slow link* we do not have. Two papers say so outright: SpecPrefetch ("under
  hot-cache or fast-storage settings... limited latency for prefetching to remove"), StickyMoE ("for
  already-resident weights, the routing consistency loss provides no latency benefit").
  Arithmetic kills L2 caching too: one expert is 13.4 MB, top-6 is 80 MB/layer vs a 32 MB L2, and
  expected cross-token expert overlap is 6^2/160 = **0.225 experts**.
- **Low-rank / SVD expert compression** — all published results are on FP16/BF16 experts; the factors
  do not stay accurate at 4.25 bits. Negative expected value.
- **Intra-expert activation sparsity** — MXFP4's 32-element blocks make sub-block skipping impossible
  without breaking the scale layout.

## Quality-cost levers (real, but they change the model)

- **Shared expert FP8 -> MXFP4**: -507 MB = -4.1% `B_tok` -> +4.3%.
- **top-6 -> top-4**: -1150 MB = -9.4% `B_tok` -> ~+10%. A third party working on *this exact model
  family* reports **top-4 is the floor** — below it "breaks coding quality".
- **AcceptMoE** (arXiv 2608.02989): commitment-weighted eligible expert sets at verify.
  **1.29x with all experts GPU-resident, batch 1**, -0.27 pp accuracy. Directly targets our
  `c_v` = 2.6x. The only technique measured at batch 1 *with resident weights*.

## Ranked

| # | lever | gain on this engine | cost |
|---|---|---|---|
| 1 | **MoE GEMM output-row blocking BN>=2, 128 threads, activation reused in registers** | **131 -> 240 GB/s measured on our shape; +12.7% routed, +17.5% with shared** | low, bit-exact per row |
| 2 | Shared expert FP8 -> MXFP4 | +4.3% | low kernel, accuracy risk |
| 3 | top-6 -> top-4 | ~+10% | trivial code, **real quality cost** |
| 4 | AcceptMoE expert-union constraint at verify | `c_v` 2.6 -> ~2.1 | medium, -0.27 pp |
| 5 | Fuse MoE epilogue (SwiGLU + weighted reduce + `route_scale`) into w2 writeback | -21.9% activation traffic in the analogous SGLang change | low |
| 6 | PDL (`griddepcontrol`) at layer boundaries + `UBLKPF.L2` at layer 0 | tail/ramp recovery, low single digits | low |
| 7 | Skip the router on the 3 hash-routed layers (expert ids are a pure function of token id, known before the forward) | 3 routers + 3 sorts off the critical path | low |
