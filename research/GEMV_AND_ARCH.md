# research/GEMV_AND_ARCH.md — batch-1 4-bit GEMV SOTA + the definitive Thor/GB10 arch delta

## Q1 SETTLED: Thor has NO FP4 tensor cores. GB10 does.

Tested every target suffix on this box (`nvcc --list-gpu-arch` shows `compute_110` and `compute_121`):

| target | `mma .kind::mxf4.block_scale` | plain `mma` e2m1 |
|---|---|---|
| `sm_110a` (Thor) | **BLOCKED** | **BLOCKED** |
| `sm_110f` (Thor, family target) | **BLOCKED** | **BLOCKED** |
| `sm_121a` (GB10 / DGX Spark) | **COMPILES** | — |
| `sm_121f` (GB10, family target) | **COMPILES** | — |

The `f` (family) suffix hypothesis — that FP4 is gated behind `sm_*f` rather than `sm_*a` — is
**refuted for Thor**. It does not matter which suffix you use; `sm_110` has no FP4 matmul.

**This is a permanent, structural disadvantage vs DGX Spark on exactly our model class.** GB10 can
feed MXFP4 expert weights straight into a block-scaled `mma`. Thor must convert FP4 → FP8/FP16
first (via `cvt.rn.f16x2.e2m1x2`, which we already use) and then run an FP8/FP16 `mma`. On a model
whose experts are 28% of per-token bytes, that is a real gap and no kernel work closes it.

What Thor DOES have, confirmed compiling *and running* (Finding 29): **tcgen05** (alloc/mma/ld) and
**TMA** (`cp.async.bulk`, `.tensor`), plus FP8 `mma`.

## THE headline number: llama.cpp already gets 85% on THIS box

| part | model | measured | % of achievable |
|---|---|---|---|
| **Jetson Thor** | Qwen2.5-Coder-7B **Q8_0** dense, 7.54 GiB, 25.26 tok/s | **204.5 GB/s** | **85.2% of our 240** |
| **Jetson Thor** | Gemma3-4B **Q4_0** dense, 2.35 GiB, 66.78 tok/s | **168.5 GB/s** | **70.2%** |
| DGX Spark | same Q8_0, 29.4 tok/s | 238 GB/s | 87% of 273 spec |
| DGX Spark | same Q4_0, 79.83 tok/s | 201 GB/s | 74% of 273 spec |

Source: jetsonhacks llama.cpp Spark-vs-Thor benchmark table.

**We are at 116.6 GB/s = 48.6%.** llama.cpp's plain MMVQ kernel reaches 85% on the same silicon.
**The gap is our engine, not the hardware** — that is now established on our exact part rather than
inferred. Caveat: those are *dense* models with a single weight stream; our MoE gathers 6 of 160
experts per layer, which is a harder access pattern. But 48.6% vs 85% is far too wide to be
explained by that alone.

## Q3 mechanism: it is REQUEST COUNT, not bytes — and Little's Law predicts our exact numbers

Sector granularity is 32 B, so the strided tile wastes no *bytes*; it wastes *requests*:
```
M=1 GEMV : 128 contiguous B per warp-instruction  ->  1 request  / 128 B
m16 tile : 8 rows x 32 B per mma                  ->  8 requests / 256 B   = 4x the requests/byte
```
`BW_achieved = (MSHRs x bytes_per_request) / latency`. Fix MSHR depth, quarter the bytes per
request, and you quarter bandwidth **unless latency also drops**. Measured Blackwell latencies:
L1/SMEM 33 cyc, **L2-near 79**, L2-far 180, **DRAM 372**.
- **Hot**: latency drops 372→79 = **4.7x**, which cancels the 4x request penalty → our measured **0.71x**.
- **Cold**: latency does not drop → we eat the full 4x → our measured **3.12x**.
Our numbers are the *predicted* values, not an anomaly. The 8 rows are also K bytes apart = 8
distinct LPDDR5X rows, so row-activate cost compounds it.

CUDA C++ Best Practices Guide §9.2.1 documents the same curve (`strideCopy`, Fig. 7).

## The fix, in order of preference

**(a) Never load `mma` fragments from global — stage through shared with a coalesced pattern.**
CUTLASS's whole structure is: coalesced `cp.async` gmem→smem → XOR-swizzle in smem → `ldmatrix`
smem→regs. Shared memory has no coalescing requirement, only bank conflicts.
Concretely for our FP8 B tile: stage **k=128 per row** so each 8-lane group issues ONE 128 B
contiguous request:
```
lanes 0..7   -> row n0+0, bytes [k0 .. k0+127]   (16 B/lane, 128 B contiguous)
lanes 8..15  -> row n0+1, ...
```
4 requests x 128 B per warp-instruction instead of 8 x 32 B → back to 1 request/128 B, then four
back-to-back `mma.m16n8k32` off the staged fragment. **Restores M=1 request efficiency with NO
offline repack, so it cannot change the weights and cannot break bit-exactness.**

Constraint to expect: **there is no `ldmatrix` shape for an 8x32 FP8 B fragment** (PTX 8.7+ has
`.m8n8` 16-bit only, `.m16n16` 8/6/4-bit, `.m8n16` 6/4-bit only). NVIDIA's own forum answer: *"You
do not have to use `ldmatrix`, you can also directly load the elements."* So plain per-thread `b32`
loads out of swizzled smem.

**(b) Offline repack into mma-fragment order** — only if smem→reg cost then shows up. References:
Marlin reshuffles 16x64 tiles, interleaving within an INT32 as `64207531`; TensorRT-LLM
`cutlass_preprocessors.cpp` gives the exact row permutations, of which **`W4_AFP8`:
`{0,1,2,3,16,17,18,19,4,5,6,7,20,21,22,23,8,...}`** is closest to our case; Machete pre-shuffles to
turn four 8-bit shared loads into one 128-bit load. **Our FP4 grouped MoE path already proves (b)
works here.**

**(c) No paper isolates coalescing loss at small M.** The nearest quantifications are the CUDA BPG
stride curve (mechanism only) and ML-SpecQD's 92%→81% BF16→MXFP4 drop. **Our 3.12x cold / 0.71x hot
pair on identical code appears to be a novel measurement.**

## Hot-benchmark folklore — where it IS written down

As harness code, not prose: **Triton `do_bench` zeroes a 256 MB buffer before every timed
iteration**; SOL-ExecBench does the same; jan.ai observed **SOL% > 100% — physically impossible —
before adding L2 flushing**. What is *not* written anywhere: that warm-L2 microbenchmarks mislead
specifically about **coalescing**, converting a request-count pathology into a non-event. That
sharp form appears to be unwritten, and it is exactly what bit us.

## L2 hints: mostly NEGATIVE at batch 1 on Blackwell — do not blanket-apply

Alpin's RTX 5090 persistent decode kernel, all at batch 1:

| technique | measured |
|---|---|
| `L1::no_allocate` on 128-bit weight loads | kept |
| **L2 `evict_first` on weights** | **-8%** ("premature eviction hurts same-layer reuse") |
| bulk L2 prefetch via `cp.async` | **-5%** vs plain `__ldg` |
| **`prefetch.global.L2` during the idle attention phase** | **+95 tok/s (+10.5%)** — his single biggest win |
| CUDA graphs | no improvement (matches our finding) |
| 256-bit vector loads | **not supported on sm_120** |

Marlin's `evict_first` is a *serving* optimisation protecting A and C in L2; for whole-model
streaming it measured negative. **The lever that paid was L2 PREFETCH issued during a phase that is
not bandwidth-bound.** We have exactly such phases — DSA indexer (0.05 ms/step) and HC/Sinkhorn.
Concrete, testable, +5-10%.

## Geometry on 20 SMs

Alpin (sm_120, batch 1): persistent, **regular launch, 128 blocks x 512 threads**; going to 170
blocks (= SM count) cost **-2%**. Per-layer barrier overhead was **9.2 us = 33.6% of per-layer
time**; replacing full barriers with flag-based partial barriers was worth +92 of his +506 tok/s.
Final: **71.2% of peak DRAM bandwidth at batch 1**, bf16, hand-written megakernel.

**Split-K is the real wave-quantisation remedy at decode shapes**, not grid clamping: fused W4A16
SplitK measured **+65% A100 / +124% H100** vs data-parallel decomposition (arXiv 2402.00025). Our
Opt #2 negative result was about *clamping grid sizes*, a different and much weaker intervention —
**split-K is untested here and is the stronger form.**

## Register-level dequant — we are already at SOTA, with two free checks

HW `cvt.rn.f16x2.e2m1x2` is the state of the art and we use it. Two possible free wins:
1. **Raw PTX instead of the C intrinsic.** GPU-MODE NVFP4 leaderboard winners found the intrinsic
   makes the compiler emit extra bit-ops; hand-written `cvt` was strictly better (32-45 registers vs
   80). **Check our SASS.**
2. **E8M0 scale as an exponent ADD, not a multiply.** E8M0 is exactly a power of two; add
   `(e-127)<<10` into the f16 exponent field with one `add.s32` instead of an `hmul2`. **Check we
   are not paying an FMA per element.**
Marlin's LOP3 trick and `prmt` LUTs are *superseded* on Thor by the HW cvt.

## Q2 answer: >70% at 4-bit is achieved, but nobody publishes it as %-of-bandwidth

Everyone reports speedup-vs-FP16 or TFLOPS. Reconstructed absolutes: llama.cpp Q4_0 on Thor 70%;
RTX 5090 Q4_0 ~76% of typical measured; Alpin's bf16 megakernel **71.2% of peak** (the best
*explicit* published %-of-peak at batch 1 on Blackwell); Marlin claims 3.87x of the ideal 4x
(~97% of the memory-bound ideal) but never states GB/s. AWQ+Marlin on L4 got only **1.38x** vs bf16
where ExLlamaV2 got **3.59x** on identical hardware — **kernel choice, not bit-width, decides
whether the bandwidth saving materialises.**

## Sources
Alpin RTX 5090 decode blog · jetsonhacks Spark-vs-Thor llama.cpp table · IST-DASLab/marlin +
arXiv 2408.11743 · TensorRT-LLM cutlass_preprocessors.cpp · Machete (Red Hat) · CUDA C++ Best
Practices §9.2.1 · PTX ISA 9.3 · NVIDIA forums "ldmatrix fp8 sm120" · Dissecting SM_120
(zartbot) · jan.ai kernel benchmarking · SOL-ExecBench 2603.19173 · Memory-Bound but Not
Bandwidth-Limited 2605.30571 · LiquidGEMM 2509.01229 · ML-SpecQD 2503.13565 · SplitK W4A16
2402.00025 · Colfax persistent/Stream-K tutorial · GPU-MODE NVFP4 leaderboard
