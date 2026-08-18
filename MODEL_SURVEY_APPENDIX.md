# Appendix: candidate models for a local agentic-coding stack on Jetson AGX Thor

Survey date **2026-08-18**. Companion to `MODEL_INVENTORY.md` (what is on disk) and `HARDWARE.md`
(what the box actually does). This is the *forward-looking* list: what to run next, and why.

**Provenance of every number below, because they are not all the same kind of fact.**

| marking | means |
|---|---|
| **measured** | from this box, reproducible via the named tool |
| *searched* | from a public source dated 2026, cited at the bottom; several are aggregator sites, not primary evals — directional only |
| (analysis) | derived here from the roofline; not measured, and flagged where it is checkable |

The assistant's training cutoff is May 2026, so **every model named here post-dates reliable
parametric knowledge** and was taken from search rather than recall. Treat vendor and aggregator
benchmark claims as claims.

## 1. The envelope

**measured** (`HARDWARE.md`, `tools/bw_probe.cu`):

* unified memory **122 GiB total, 117 GiB available at rest** — `free` is authoritative on Thor,
  `nvidia-smi` reports `Not Supported` for memory
* **240 GB/s achievable streaming bandwidth**, 212 GB/s under contention, against 273 GB/s spec

Decode is bandwidth-bound, so the governing equation is

    tok/s  ~=  BW / bytes_read_per_forward  x  tau        (tau = accepted tokens per verify)

At NVFP4 (~0.5 bytes/param plus scale overhead) that fixes a hard ceiling per model before any
kernel work. Weights resident sets the *fit*; active params set the *speed*.

## 2. The correction that reframes MoE vs dense (analysis)

The usual claim is that an MoE buys big-model breadth at small-model speed. **That claim degrades
once a speculator is in the loop, and it degrades only for the MoE.**

A dense model reads the same weights whether it verifies 1 token or 5, so tau passes through
nearly intact. An MoE verifying K tokens must read the *union* of the experts those K tokens route
to — roughly `min(E, K*k)` experts — so bytes per forward grows with K until it saturates toward
reading most of the expert set. The MoE's speculative speedup is damped; the dense model's is not.

The supporting observation is local and is not a citation: **Qwen3.6-27B dense (17.8 tok/s
roofline) and Qwen3.5-122B-A12B (40 tok/s roofline) both land near 50 tok/s under DFlash.** The
model with the 2.2x worse roofline closed the entire gap. Expert fanout under speculation is the
most plausible explanation.

**This is checkable and should be checked** before it is trusted: the `spec_profile` histogram in
`build/dsv4-server.staged` gives the joint (realised verify width, accepted prefix length)
distribution. Bytes-per-verify plotted against verify width would show the saturation curve
directly.

**Counterweight specific to DSV4** — `dsv4-0731-cuda-server.md` records that **MLA is 41% of
B_tok, not the MoE**. Where attention is that large a share, expert fanout is not the dominant
term, and FP8 KV is a bigger lever than the fanout argument suggests. Do not generalise one
model's bottleneck to the class.

## 3. In-envelope candidates

Roofline is `240 / (active_B * 0.5)` (analysis). Resident is `total_B * 0.5` plus ~10% for scales
and embeddings (analysis). Benchmarks are *searched*.

| model | params | resident NVFP4 | roofline tau=1 | attention | context | note |
|---|---|---:|---:|---|---:|---|
| **Qwen3.8-27B** | 27B dense | ~15 GB | 17.8 tok/s | 3x Gated DeltaNet : 1x full GQA, 16 of 64 layers full | 262K, ~1M YaRN | multimodal; same backbone as a 2.4T MoE flagship |
| **Nemotron 3.5 Lightning** | 30B-A3B | ~17 GB | 160 tok/s | Mamba-2 + MoE + attention hybrid | 1M | **published vLLM + DSpark recipe tuned for GB10** |
| **Muse Glimmer** (Meta) | 30B dense | ~17 GB | 16 tok/s | not established | 120K+ | NVFP4 "coming soon" as of the NVIDIA post |
| **Nemotron 3 Super** | 120B-A12B | ~66 GB | 40 tok/s | not established | not established | released 2026-03-11, aimed at agentic |
| **Ling 3.0 Flash** | 124B-A5.1B | ~68 GB | **94 tok/s** | Kimi Delta Attention : gated MLA at 5:1 | 262K | highest roofline in the fit-on-Thor class |
| **Laguna S 2.1** | 118B-A8.5B | ~65 GB | 56 tok/s | not established | 1M | **NVFP4 published by Poolside**; Terminal-Bench 2.1 70.2 |
| **DSV4-Flash-0731-REAP** | ~180B-A13B | **101 GiB measured** | 37 tok/s | MLA / DSA | 32K at FP32 KV (**seqmax measured**) | the incumbent; **10-24 tok/s measured** |

Reported figures for Qwen3.8-27B (*searched*): LiveCodeBench v6 90.3, SWE-bench Pro 61.7,
Terminal-Bench 2.1 **73.0**, Artificial Analysis Agentic Index 51 — reportedly above Claude Opus
4.8 at maximum reasoning effort. Note it reportedly **beats Laguna S 2.1 on Terminal-Bench 2.1
(73.0 vs 70.2) at ~4.4x fewer total parameters**, which is evidence that training quality is
currently dominating parameter count in this band, and a caution against buying breadth by size
alone.

## 4. Out of envelope — and why the REAP line matters

The genuine frontier open-weights do not fit this box at any 4-bit quantisation (*searched*):

| model | params | resident at NVFP4 | verdict |
|---|---|---:|---|
| Kimi K3 | 2.78T, top-16 of 896 experts, 69 KDA + 24 gated MLA layers, native MXFP4 | ~1.4 TB | 12x over |
| Qwen3.8 flagship | 2.4T-A96B (big brother of the 27B dense) | ~1.2 TB | 10x over |
| DeepSeek V4 Pro | 1.6T-A49B | ~800 GB | 7x over |
| GLM-5.2 | 744B-A40B | ~372 GB | 3x over |
| Nemotron 3 Ultra | 550B-A55B | ~275 GB | 2.3x over |

**So expert pruning is not a curiosity on this hardware — it is the only path to frontier-lineage
breadth.** That is the strategic justification for the DSV4-Flash-0731-REAP programme independent
of how its eval lands.

REAP (Router-weighted Expert Activation Pruning, Cerebras + University of Calgary, ICLR 2026) is
one-shot and post-training, no fine-tuning. Reported fidelity at **50% expert pruning** on
Qwen3-480B-Coder-FP8: **97.6% of baseline non-agentic coding, 96.7% on agentic SWE-Bench**
(*searched*). The Cerebras HuggingFace collection carries pruned GLM-4.6, GLM-4.5-Air,
Qwen3-Coder-480B, Qwen3-Coder-30B, MiniMax-M2, Kimi-Linear and DeepSeek-V3.2.

Those published fidelity numbers are also the **honest prior for our own eval**: if the REAP under
test lands within a few points of the unpruned parent on matched harness and settings, that is the
expected result, not a surprise. A large gap would be the finding.

## 4b. SSD / NVMe weight streaming: why it does not rescue the out-of-envelope models

Three engines are current as of this survey (*searched*), all pure C, all streaming expert weights
from disk rather than requiring them resident:

| engine | what it runs | reported throughput |
|---|---|---|
| [Colibri](https://github.com/JustVugg/colibri) | GLM-5.2 744B-A40B | **8.3 tok/s** on a 128 GB M3 Max; 11.7 tok/s on a Ryzen 7950X + 990 Pro; 6.1 tok/s on a 32 GB M4 mini (with 200-500 ms stalls); ~0.05-0.1 tok/s on a weak laptop |
| [kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c) | Kimi K3 2.78T, 93 layers, 69 KDA + 24 gated MLA, 896 experts top-16, native MXFP4 | runs in **8.24 GB of RAM** on one CPU; trunk streaming makes the memory budget a dial, **byte-identical output from 8.24 GB to 224 GB** |
| [warp / WASTE](https://github.com/sqliteai/warp) | Kimi K3 2.78T from NVMe | **~0.6 tok/s on a 64 GB MacBook Pro** |

**These are lossless.** kimi-k3-in-c producing byte-identical output at every memory budget is the
important architectural point: streaming trades *throughput and memory* against each other and does
not touch quality. So the question is purely whether the resulting token rate supports the workload
— not whether the model gets dumber.

**Colibri's headline number is a cache-hit-rate result, not a bandwidth result** (analysis, and it
is what makes the numbers above look inconsistent until you decompose them). GLM-5.2 reads ~20 GB
of active weights per token at NVFP4. A Gen4 NVMe does 5-7 GB/s, so if that came off disk it would
be ~0.3 tok/s, not 8.3. To reach 8.3 the disk can supply at most ~0.7 GB/token — i.e. **~96% of
active weights are already resident**. Colibri pins attention, norms and embeddings and streams
only cold experts behind double-buffered prefetch. The SSD covers the tail; RAM does the work.

That makes **resident fraction** the governing variable, and it is exactly where the frontier
models fail on this box:

| | total NVFP4 | RAM | resident | active/token | observed |
|---|---:|---:|---:|---:|---|
| GLM-5.2 on the reported 128 GB systems | 372 GB | 128 GB | **34%** | 20 GB | 8.3 tok/s |
| Kimi K3 on a 64 GB MacBook (warp) | ~1.4 TB | 64 GB | **4.6%** | ~52 GB | **0.6 tok/s** |
| Qwen3.8-2.4T-A96B on Thor | ~1.2 TB | 117 GiB | **~10%** | 48 GB | ~0.6-3 tok/s (predicted) |

The warp datapoint is the useful one because it is the same regime: a trillion-scale MoE at single-
digit residency lands at **0.6 tok/s measured**. Thor has roughly twice that residency for a
slightly smaller model, so the estimate for Qwen3.8-2.4T here — 0.25-1.25 tok/s before speculation,
~0.6-3 tok/s granting DFlash's tau ~ 2.5 — is corroborated by an independent measurement rather
than resting on the roofline alone.

**Why that is disqualifying for agentic work specifically.** The threshold is not reading speed.
Agentic work multiplies tokens by turns, and the turns are strictly serial — turn N+1 depends on
turn N's tool result, so none of it parallelises. A 20-turn task at ~2000 tokens/turn is 40k tokens:

| rate | one agentic task |
|---:|---|
| 50 tok/s | 13 min |
| 8 tok/s | 1.4 h |
| 2 tok/s | 5.6 h |
| 0.6 tok/s | **18 h** |
| 0.3 tok/s | **37 h** |

At the bottom of that table the failure is not slowness. It is that you get one attempt per day,
cannot course-correct, and any wrong turn costs the entire run. That is a categorically different
activity from coding.

**The niche where it is still useful.** Streaming does not give you an agent; it can give you a
*teacher*. Trace generation is throughput-tolerant, fully offline, and has no serial feedback loop,
and this box is now configured to run unattended work for days (`scripts/eval_resume.sh`,
`dsv4-evals-watchdog.timer`, `CLAUDE.md` detachment rule). A trillion-scale model at 0.6 tok/s
still yields ~50k tokens overnight — a meaningful volume of frontier-quality reference traces for
draft-head training or distillation, and losslessly, per the byte-identical property above. That is
the same big-model-as-oracle pattern already used elsewhere in this programme. Prefill-heavy
single-answer work is the other survivor, since prefill is compute-bound and batches, so streaming
amortises far better there than in decode.

One further note of interest to the attention thesis in this document: **Kimi K3 is 69 KDA + 24
gated MLA layers with native MXFP4 weights**. The frontier is converging on the same hybrid
linear-attention design that makes Qwen3.8-27B and Ling 3.0 Flash attractive here, which is
evidence that the small-KV architectures are not merely an efficiency concession at small scale.

**Verdict: excluded from the interactive shortlist, retained as a candidate trace generator.**

## 5. Recommendation

**Run Qwen3.8-27B NVFP4 + DFlash as the Claude Code substitute candidate.** It wins on three axes
simultaneously rather than trading between them:

1. full dense forward pass every token — no routing approximation on the frontier-planning and
   novel-algorithm work that is the whole point
2. the 3:1 GDN hybrid means only 16 of 64 layers carry a growing KV cache, so 15 GB of weights
   leaves **~100 GiB for context and a draft model**; long-horizon agents stop being memory-bound
3. dense means speculation multiplies cleanly (§2), so the 50 tok/s already measured on the 3.6
   generation is a floor

Its honest cost is parametric breadth. 27B holds less biology, genetics, quantum and quantitative
finance than 180B-A13B, and that is exactly the spread the operator cares about.

**Keep DSV4-REAP as the breadth instrument, not the daily driver**, and finish its eval — it
answers the REAP-fidelity question, which is a prerequisite for the whole pruning strategy above.

**Add Ling 3.0 Flash to the shortlist.** At a 94 tok/s roofline with KDA + gated MLA it is the only
in-envelope model that could plausibly reach triple-digit decode here. A5.1B is a real quality risk
— it is the same shape as the Qwen3.6-35B-A3B failure mode, too few active parameters to reason
confidently off the CoT rails — but that is cheap to falsify and worth knowing.

**Nemotron 3.5 Lightning is the cheapest experiment on the list**, because NVIDIA published a vLLM
recipe with **DSpark speculative decoding tuned for GB10**. Thor is not GB10, but it is the closest
published tuning target to this hardware, and it is the only entry here that arrives with a
speculator already fitted.

## 6. The axis nobody benchmarks, and which this repo already measures

For an agentic substitute the figure of merit is not decode rate and not accuracy. It is

    effective throughput  =  tok/s  /  tokens_to_answer

A model at 50 tok/s that answers in 800 tokens beats one at 100 tok/s that spirals for 3000.
`tools/eval_suite.py --report` already emits the numerator's denominator as the `mean tok` column.
**measured**, DSV4 at low effort: math500 940, mmlu_pro 1948, gpqa_diamond 3717, lcb 5578. The lcb
row truncating at 59% is a CoT-efficiency failure, not a capability failure.

When the Qwen3.8-27B leg runs under matched local config — same harness, same effort, same item
set, same `eval_extend` policy — **that ratio is what should decide the comparison.** Record
tokens-to-answer as a first-class result alongside accuracy, not as a footnote.

## Sources

Qwen3.8-27B: [architecture](https://www.mindstudio.ai/blog/qwen3-8-27b-architecture-benchmarks) ·
[vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B) ·
[benchmarks](https://www.orcarouter.ai/blog/qwen-3-8-27b-benchmarks) ·
[VentureBeat](https://venturebeat.com/technology/qwen3-8-27b-runs-frontier-class-coding-agents-and-reasoning-locally-no-cloud-api-required).
Ling 3.0 Flash: [guide](https://www.aimadetools.com/blog/ling-3-0-flash-complete-guide/).
Laguna S 2.1: [NVFP4 weights](https://huggingface.co/poolside/Laguna-S-2.1-NVFP4) ·
[release](https://www.marktechpost.com/2026/07/21/poolside-releases-laguna-s-2-1/).
Nemotron: [3.5 Lightning](https://www.marktechpost.com/2026/08/11/nvidia-ai-releases-nemotron-3-5-lightning-and-nemo-switchyard/) ·
[NVIDIA local agents](https://blogs.nvidia.com/blog/local-ai-open-source-models-agents-nemotron/).
Frontier scale: [Kimi K3 vs DeepSeek V4 Pro vs GLM-5.2](https://www.marktechpost.com/2026/07/18/kimi-k3-vs-deepseek-v4-pro-vs-glm-5-2-open-trillion-scale-moe-models-compared-on-benchmarks-license-and-serving-cost/).
REAP: [Cerebras blog](https://www.cerebras.ai/blog/reap) ·
[repo](https://github.com/CerebrasResearch/reap) ·
[HF collection](https://huggingface.co/collections/cerebras/cerebras-reap).
Weight streaming: [Colibri](https://github.com/JustVugg/colibri) · [Colibri throughput](https://www.remio.ai/post/colibri-shows-how-to-run-glm-5-2-on-a-low-spec-pc-but-storage-sets-the-pace) · [Colibri on consumer hardware](https://wavect.io/blog/colibri-glm-5-2-consumer-hardware/) · [kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c) · [kimi-k3-in-c writeup](https://themenonlab.blog/blog/kimi-k3-in-c-trillion-parameter-model-8gb-ram/) · [warp / WASTE](https://github.com/sqliteai/warp).
