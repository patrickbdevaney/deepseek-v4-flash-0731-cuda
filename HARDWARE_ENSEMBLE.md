# The hardware ensemble: which box runs what, and the biggest model at acceptable decode

Surveyed 2026-08-18. Companion to `HARDWARE.md` (Thor alone), `MODEL_SURVEY_APPENDIX.md` (what
fits) and `COMPRESSION_PLAYBOOK.md` (how to make it fit). Same provenance convention: **measured**,
*searched*, (analysis).

## 1. Inventory, ordered by the only number that governs decode

| device | capacity | bandwidth | tier |
|---|---:|---:|---|
| RTX 3090 VRAM | 24 GB | **936 GB/s** (spec) | fastest available |
| Jetson AGX Thor | **117 GiB** | **240 GB/s** (**measured**) | best capacity x speed |
| Orin Nano Super | 8 GB (~6 usable) | 102 GB/s (*searched*) | + 1 TB NVMe |
| B550 AORUS ELITE AX V2 DDR4 | **128 GB max**, 4 DIMM, dual channel (*searched*) | ~45-51 GB/s | large but slow |
| NVMe (any host) | 1 TB+ | ~3-6 GB/s | 40-200x penalty |

The 3090 is Ampere (`sm_86`): **no FP4**, but INT4 via AWQ/GPTQ with Marlin kernels is mature.
Thor is `sm_110a` with FP4x2 hardware unpack. **Quantisation format is per-device, not global.**

## 2. A 20x bandwidth spread creates hard regime boundaries (analysis)

Since `tok/s ~= BW / bytes_per_forward`, the best host is a step function of model size:

| model footprint | best host | why |
|---|---|---|
| **<= 24 GB** | **RTX 3090** | 936 GB/s, 3.9x Thor |
| **24-117 GB** | **Thor** | the desktop would spill to 45 GB/s DDR4, 5x worse |
| **117-152 GB** | **desktop** | Thor must reach SSD at ~6 GB/s; DDR4 is ~7x faster than that |
| **> 152 GB** | none | streaming regime, 1-3 tok/s, offline only |

**The top row is the surprise and it changes the substitute plan.** Qwen3.8-27B at 4-bit is 13.5 GB
and fits entirely in 3090 VRAM: **69 tok/s roofline against 17.8 on Thor**, realistically 35-45
tok/s. Its 3:1 GatedDeltaNet hybrid keeps KV small enough that context is not the binding
constraint. **The 3090 is likely the better host for the Claude Code substitute than Thor is** —
Thor's advantage is capacity, not speed, and a 27B dense model does not need the capacity.

## 3. Orin Nano Super is genuinely useful, because streaming scales with ACTIVE params

8 GB looks hopeless until you notice that streaming cost is set by active parameters, not total.
Paired with its idle 1 TB NVMe (analysis, ~3 GB/s assumed and **worth measuring**):

| model | active | est. decode |
|---|---:|---:|
| Nemotron 3.5 Lightning 30B-**A3B** (15 GB @ 4-bit) | 1.5 GB/tok | **~9 tok/s** |
| Ling 3.0 Flash 124B-**A5.1B** (62 GB @ 4-bit) | 2.55 GB/tok | **~4 tok/s** |
| dense 4B, *searched* measured figure | — | ~9.5 tok/s |

**A 124B-parameter MoE at ~4 tok/s on an 8 GB board at 15-25 W** is a real capability. Useless for
interactive coding — see the serial-turn arithmetic in `MODEL_SURVEY_APPENDIX.md` §4b — and well
suited to exactly the asynchronous work this programme needs: **agentic trace generation**, RAG
indexing, summarisation, background agents. Streaming is lossless, so traces produced this way are
full-quality.

## 4. Pipeline parallelism across Thor + desktop: 269 GB of DRAM-or-better

**Activations are tiny.** A hidden state is ~10 KB per token, so a layer-boundary transfer is ~80 us
even over 1 GbE against a 50 ms token budget — **the network is not the bottleneck for pipeline
parallelism.** (Tensor parallelism would be hopeless; the distinction matters.)

Combined capacity is **117 + 24 + 128 = 269 GB**, which puts a new configuration in range:

**GLM-5.3-REAP-50% at plain 4-bit = 186 GB**, split across both machines (analysis):

* Thor holds ~117 GB (63% of the model) -> 12.6 GB active/token @ 240 GB/s = **52 ms**
* Desktop holds ~69 GB -> hottest 24 GB in VRAM, remainder in DDR4 = **~110 ms**
* Total ~162 ms -> **~6.2 tok/s base, ~15 tok/s at tau = 2.5**

**This sidesteps the two largest risks in `COMPRESSION_PLAYBOOK.md` simultaneously**: no sub-3-bit
quantisation, and no QTIP/AQLM kernel to write for `sm_110a`. It uses REAP at its validated 50% and
ordinary INT4/NVFP4. The cost is a two-machine heterogeneous pipeline engine spanning `sm_110a` and
`sm_86` with a different quantisation format on each side, and speculation across a pipeline
boundary is awkward. That is engineering work rather than a research bet on unvalidated pruning,
which is a meaningfully better risk profile.

**The limit that does not go away:** pipeline parallelism makes capacity additive *and latency
additive*. You are bounded by the slowest stage, which is DDR4. **There is no configuration in
which the desktop's 128 GB runs at Thor speeds.**

## 4b. The desktop alone, VRAM+DDR4 pooled: what it can actually run

24 GB VRAM + 128 GB DDR4 = **152 GB**, ~145 GB usable after OS, KV and activations. Pooling does
not make it one memory: the tiers differ by **20.8x**.

| tier | cost per GB read |
|---|---:|
| VRAM @ 936 GB/s | **1.07 ms** |
| DDR4 @ ~45 GB/s | **22.2 ms** |

**The design rule that follows:** a 20 tok/s target is a 50 ms budget, so you can afford
**<= 2.2 GB per token from DDR4**. Everything above that must hit VRAM.

### The inversion: Thor is capacity-constrained, the desktop is active-param-constrained

On Thor, unified 240 GB/s means *total footprint* binds and anything that fits runs at similar
speed. On the desktop total footprint barely matters — there is 152 GB — but **active parameters
dominate**, because every missed byte costs 22 ms. The two machines therefore want **opposite model
shapes**, which is a better division of labour than one simply being faster.

| model | active @ 4-bit | all-DDR4 worst case | with realistic VRAM caching |
|---|---:|---:|---:|
| Nemotron 3.5 Lightning 30B-**A3B** | 1.5 GB | 30 tok/s | 40-60 |
| **Ling 3.0 Flash 124B-A5.1B** | 2.55 GB | 17.6 tok/s | **40-90** |
| Nemotron 3 Super 120B-**A12B** | 6.0 GB | 7.5 tok/s | **~22** |
| DSV4-Flash-REAP 180B-**A13B** | 6.5 GB | 7 tok/s | ~17 |
| GLM-5.3-REAP-504B **A40B** @ 2-bit | 10 GB | 2.3 tok/s | **8-16** |
| Nemotron 3 Ultra 550B-**A55B** @ 2-bit | 13.8 GB | 1.7 tok/s | ~7 |

### Three answers, depending on what "best" means

* **Biggest, discounting speed — GLM-5.3-REAP-504B at 2-bit, 126 GB, ~8-16 tok/s.** A 744B-lineage
  frontier coding model, and the configuration **does not fit on Thor** (126 GB > 117 GiB), so it
  is the desktop's unique capability rather than a consolation prize. Needs only the shipping 34%
  REAP, not the unvalidated 50%.
* **Best overall tradeoff — Ling 3.0 Flash 124B-A5.1B at plain 4-bit, 62 GB, ~40-90 tok/s.** A 124B
  model at interactive-to-fast speed with **no pruning and no exotic quantisation**, fitting with
  80 GB to spare. A5.1B is small enough that even a poor hit rate still leaves ~17 tok/s. The risk
  is quality, not engineering: too few active parameters to reason confidently off the CoT rails.
* **Best quality-per-speed — Nemotron 3 Super 120B-A12B at 4-bit, 60 GB, ~22 tok/s.** A12B is in
  the range where forward passes are substantive and it still clears the 18-20 target.

### Consequence

**The desktop is the ideal host for ultra-sparse MoEs** — precisely the models whose low active
count makes them questionable as a sole driver but which the DDR4 tier can feed at speed. Thor is
the better host for denser active sets (A13B+) that would choke on DDR4.

**Out of reach:** unpruned GLM-5.3 at 4-bit is 372 GB, only 41% resident even in the full pool, so
it returns to the streaming regime at ~2 tok/s. And the expert-fanout objection from §4c applies to
the DDR4 tier at 20.8:1, so **speculation is marginal here too** unless the active set is largely
VRAM-resident — it should help the A3B/A5.1B configs and probably will not help the A40B one.

## 4c. Decision (2026-08-18): residency-first on Thor, desktop as the compression lab

**The pooled-desktop configuration in 4b is documented but NOT adopted.** Operator judgement,
supported by the arithmetic below: 8-16 tok/s is not worth the power draw or an unmeasured
capability delta.

### Power settles it

| config | decode | system power | tok/s per watt |
|---|---:|---:|---:|
| desktop, GLM-5.3-REAP-504B @ 2-bit | 8-16 | ~500 W | **0.024** |
| Thor, DSV4-Flash-0731-REAP @ NVFP4 | 10-24 (**measured**) | ~110 W | **0.155** |

**Thor is ~6x more efficient per token and faster at the same time.** The desktop path spends 500 W
and a two-tier memory engine to go slower. The capability being bought is also unmeasured and could
be negative: GLM-5.3-REAP-504B-at-2-bit stacks two lossy transforms (34% prune, then the riskiest
quantisation tier) on a base whose unpruned advantage over DeepSeek-V4-Flash-0731 is modest.

### The low-risk first milestone: requantise the incumbent

The checkpoint occupies **101 GiB for ~180B params — about 4.5 bits effective**, not 4 (MXFP4 plus
scales and overhead). Tightening it carries no capability risk at all:

| config | bits | resident | active bytes | roofline |
|---|---:|---:|---:|---:|
| today | ~4.5 | 101 GiB | 7.3 GB | 33 tok/s |
| tight 4-bit | 4.0 | 90 GB | 6.5 GB | 37 tok/s |
| **3.5-bit** | 3.5 | **79 GB** | **5.7 GB** | **42 tok/s** |

**~28% more decode and ~22 GiB freed, same weights lineage, no new evaluation burden.** The freed
memory then buys FP8 KV headroom *and* room for a larger draft head, which raises tau and compounds
on top of the 28%. See `wiki/nvfp4-migration.md`.

### The same pipeline unlocks the ambitious target

If compression reaches ~2.5 bits with negligible loss, **GLM-5.3-REAP-50% (372B) is 116 GB and
becomes resident on Thor**: A40B -> 12.5 GB/token -> **19 tok/s roofline, ~15-33 with speculation**.
Better than the desktop configuration on every axis at a fifth of the power, with both components
at published limits (REAP's validated 50%, QTIP's demonstrated 2-3 bit range). Residency-first is
therefore **strictly dominant**, not a consolation, and it has a milestone that pays before any new
model is touched.

### Ling on the desktop is rejected, for a stronger reason than speed

Ling 3.0 Flash fits Thor natively at 62 GB with a 94 tok/s roofline — **Thor runs it strictly
better than the desktop would**, so the desktop adds nothing. If A5.1B ever proves itself on
quality, it belongs on Thor.

### The desktop's role

**Not inference — the compression lab.** 24 GB VRAM + 128 GB DDR4 is a good workstation for
shard-by-shard GPTQ/AWQ/QTIP calibration passes, holding calibration activations, sweeping quant
configs against the KL metric of `COMPRESSION_PLAYBOOK.md` §5, and hosting the teacher-logit
pipeline. That serves the Thor plan rather than competing with it. The 3090 keeps one inference
role: anything <= 24 GB, i.e. Qwen3.8-27B at 35-45 tok/s, which is also power-sane.

**Gate unchanged: prototype the low-bit kernel on `sm_110a` before committing to a format.** If
trellis decode converts a bandwidth-bound decode into a compute-bound one, the whole path must
reroute to a format with hardware unpack, and that is cheap to discover first.

## 5. Biggest model at acceptable decode, by configuration

| configuration | biggest model | est. decode |
|---|---|---:|
| 3090 alone | Qwen3.8-27B @ 4-bit (13.5 GB) | **35-45 tok/s** |
| Thor alone | DSV4-Flash-0731-REAP 180B-A13B | 10-24 tok/s (**measured**) |
| Thor alone, compressed | GLM-5.3-REAP-50% @ 2.3-bit (107 GB) | ~22 tok/s roofline |
| **Thor + desktop pipeline** | **GLM-5.3-REAP-50% @ 4-bit (186 GB)** | **~15 tok/s with spec** |
| Orin Nano, async only | Ling 3.0 Flash 124B-A5.1B | ~4 tok/s |

## 6. The ensemble's actual advantage is concurrency, not size

Three independent devices are **three concurrent agentic workers**. Given that the frontier labs'
real structural edge is massively parallel agentic inference rather than single-stream speed —
single sequential agent latency being roughly comparable — a heterogeneous three-box fleet is
plausibly worth more than one marginally larger model:

* **3090** — the fast interactive coding agent (Qwen3.8-27B, 35-45 tok/s)
* **Thor** — the deep/breadth instrument for hard reasoning (DSV4-Flash-REAP)
* **Orin Nano** — always-on async: trace generation, indexing, background summarisation

## 7. Unmeasured quantities this analysis rests on

1. **NVMe sequential read on both the Orin Nano and Thor.** Assumed ~3-6 GB/s; every streaming
   number scales with it.
2. **Achievable DDR4 bandwidth** on the B550 at whatever kit is installed (assumed ~45 GB/s of a
   51.2 GB/s dual-channel DDR4-3200 theoretical).
3. **VRAM hit rate** for a hybrid VRAM/DDR4 MoE split, which drives every desktop estimate in §4.

## Sources

[B550 AORUS ELITE AX V2 specs](https://www.gigabyte.com/Motherboard/B550-AORUS-ELITE-AX-V2-rev-10/sp) ·
[Crucial compatibility](https://www.crucial.com/compatible-upgrade-for/gigabyte/b550-aorus-elite-ax-v2) ·
[Jetson Orin Nano Super dev kit](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/) ·
[Orin Nano Super boost](https://developer.nvidia.com/blog/nvidia-jetson-orin-nano-developer-kit-gets-a-super-boost/) ·
[Orin Nano LLM benchmarks](https://smolhub.com/posts/jetson-nano-super-benchmark-non-reasoning/)
