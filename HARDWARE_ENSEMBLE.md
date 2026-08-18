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
