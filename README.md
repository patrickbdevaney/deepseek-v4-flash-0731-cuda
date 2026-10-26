# deepseek-v4-flash-0731-cuda

A from-scratch, pure-CUDA inference engine and OpenAI-compatible server for
[`0xSero/DeepSeek-V4-Flash-0731-REAP`](https://huggingface.co/0xSero/DeepSeek-V4-Flash-0731-REAP): DeepSeek-V4-Flash
pruned to 160 routed experts, with native MXFP4 experts. It runs on one NVIDIA Jetson AGX Thor (`sm_110a`), serves a
single agentic session at a time, and decodes losslessly with a fine-tuned DSpark speculative draft head at
**28.4 tok/s**. That is 1.97× its own autoregressive decode. Every emitted token is bit-identical to plain greedy
decoding, and every run checks it.

## Status (2026-09-11)

The programme is complete and frozen. Its final state was recorded 2026-08-25; the prefill campaign closed
2026-08-26; the last engine change (2026-09-11) discovers the routed-expert count instead of assuming 160. The engine
and server are usable for local agentic work on a Thor. They are not a general-purpose serving stack: one model,
one concurrent request, one GPU architecture.

| Component | State |
|---|---|
| Engine: MLA + DSA attention, MXFP4 MoE, hyper-connections, DSpark verify | Complete; every kernel gated against a PyTorch oracle |
| OpenAI-compatible server ([`server/`](server/)) | Complete: chat and completions, streaming, tool calls, thinking blocks, KV prefix cache |
| Checkpoint with the head installed: [`DeepSeek-V4-Flash-0731-REAP-DSpark`](https://huggingface.co/patrickbdevaney/DeepSeek-V4-Flash-0731-REAP-DSpark) | Public, 100.4 GiB; the model card carries the final evaluation |
| Draft head alone: [`dspark-mtp-draft-head-s3recap-p25-b0.1`](https://huggingface.co/patrickbdevaney/dspark-mtp-draft-head-s3recap-p25-b0.1) | Public; the served head |
| Capability evaluation | 14 rows; 12 complete, one partial (do not quote), two budget-limited ([below](#capability)) |
| Context | 128K tokens supported, 32,768 by default; KV packing switches on automatically above 32,768 |

## Results

All numbers were measured on one Jetson AGX Thor with the weights as shipped. Nothing was re-quantized.

| Metric | Value | Conditions |
|---|---|---|
| **Speculative decode** | **28.38 tok/s** | 8-prompt frozen suite mean, ≥ 200 generated tokens, draft width 5, head `s3recap-p25-b0.1` |
| Acceptance τ | 3.84 / 5 | 77% of the width ceiling; τ is not comparable across draft widths |
| Plain autoregressive decode | 14.61 tok/s | Same suite; speculation is 1.97× this |
| Prefill | 108.1 tok/s | Bit-exact; +75.2% over the start of the prefill campaign |
| Weights resident | 100.4 GiB | 12.26 GB read per decoded token |

**Speculative decoding by stage.**

| Stage | tok/s |
|---|---|
| Stock DSpark head shipped with the checkpoint | 22.66 |
| Draft width restored to the checkpoint's own `dspark_block_size` (6 → 5) | +3.9% at equal τ |
| Fine-tuned head (deficit-weighted loss, β = 0.1 anchor to the pre-training head) | **28.38** (+25.3% overall) |

The served head won against eleven other arms; their top five landed within 1.3% of each other against a 3.5%
promotion bar. Every candidate, rejects included, is in [`docs/HEAD_REGISTRY.md`](docs/HEAD_REGISTRY.md). The recipe,
with each hyperparameter's justification, is in [`docs/BEST_SETUP.md`](docs/BEST_SETUP.md).

**Where the speed comes from, and where it stops.** Multi-head latent attention (MLA) projections, not the experts,
dominate per-token bytes: 41% against 31% for the routed experts. A sweep of draft widths 4–12 put the gain from
adaptive width at **+1.8%** even with hindsight, down from an estimate of +20–25%. Decode speed is now limited by
the autoregressive kernels, not the speculator ([`wiki/roofline-why-the-needle-wont-move.md`](wiki/roofline-why-the-needle-wont-move.md)).

<a id="capability"></a>**Capability as served.** Sampling at temperature 1.0, top-p 0.95, reasoning effort `low`.
Brackets are 95% cluster-bootstrap intervals; "budget" is the `max_tokens` cap.

| Task | n | Budget | Accuracy | Truncated |
|---|---|---|---|---|
| MATH-500 | 100 | 8K | 95.0 [88.8, 97.8] | 1 |
| HumanEval | 164 | 8K | 95.1 [90.7, 97.5] | 6 |
| AIME 2024 (avg@2) | 60 | 24K | 91.7 [80.0, 100.0] | 6 |
| GPQA Diamond | 198 | 24K | 83.8 [78.1, 88.3] | 19 |
| MMLU-Pro (subset) | 150 | 24K | 74.7 [67.2, 81.0] | 4 |
| BFCL | 240 | 2K | 86.2 [81.3, 90.0] | 3 |
| BFCL live | 508 | 2K | 78.7 [75.0, 82.1] | 5 |
| LiveCodeBench | 175 | 8K | 46.9 [39.6, 54.2] | **104 (59%)**: budget-limited, not a capability number |
| SciCode | 291 | 8K | 30.2 [24.1, 36.9] | 54 (19%): budget-limited |

- **GPQA Diamond missed its prediction.** It came in at 83.8, below the pre-registered prediction of 87–93.
- **AIME 2025 at 24K is partial** (54 of 60) and must not be quoted. An extension finishes the easy traces first, so
  a partial file overstates accuracy.
- **Budget changes results.** At an 8K budget, GPQA Diamond reads 72.7 with 51 truncated; at 24K it reads 83.8.
  [`docs/EVAL_BUDGET_PROTOCOL.md`](docs/EVAL_BUDGET_PROTOCOL.md) explains why the budget has to be chosen at the
  ceiling.

## Engine

| Topic | Summary | Detail |
|---|---|---|
| Model | 43 MoE layers + 3 chained DSpark MTP blocks; hidden 4096; 64 heads × 512; Q/O-LoRA rank 1024; 160 routed experts, top-6, plus 1 shared; hyper-connections ×4 with 20 Sinkhorn iterations; sliding window 128 | [`docs/MODEL_INVENTORY.md`](docs/MODEL_INVENTORY.md), [`docs/ARCH_DELTA.md`](docs/ARCH_DELTA.md) |
| Formats | MLA and dense FP8 E4M3 with 128×128 E8M0 block scales; routed experts OCP MXFP4 with hardware `cvt.e2m1x2` unpack; norms, embedding and head BF16 | [`docs/ROOFLINE.md`](docs/ROOFLINE.md) |
| Decode kernels | Every adopted optimisation with its mechanism, measured gain and gate | [`wiki/kernel-optimisations.md`](wiki/kernel-optimisations.md) |
| Retired levers | Larger than the win list, each with the number that killed it | [`wiki/negative-results.md`](wiki/negative-results.md) |
| Prefill | Why prefill ran decode-shaped kernels, and the fixes | [`wiki/prefill-optimisation.md`](wiki/prefill-optimisation.md) |
| Draft-head training | Architecture, loss, data, feasibility, literature | [`wiki/draft-head-finetuning.md`](wiki/draft-head-finetuning.md), [`docs/S5_RECIPE.md`](docs/S5_RECIPE.md) |
| Measurement | How a number becomes trustworthy here, and 46 recorded traps | [`wiki/measurement-and-traps.md`](wiki/measurement-and-traps.md) |
| Hardware | Thor's measured bandwidth (240 GB/s achievable) and compute peaks; `sm_110a` ISA facts | [`wiki/hardware-sm110a.md`](wiki/hardware-sm110a.md), [`docs/HARDWARE.md`](docs/HARDWARE.md) |
| Memory safety | 100.4 GiB of weights in a 122 GiB unified pool: single tenancy and an OOM sentry | [`wiki/oom-and-memory-safety.md`](wiki/oom-and-memory-safety.md) |

Two measurement facts every number here depends on:
- **Acceptance needs at least 200 generated tokens.** τ is 1.39 over the first 32 tokens and rises to ~3.2 only
  after ~128, past the drafter's sliding window.
- **The "19.0 tok/s roofline" is a normalisation constant, not a target.** 22.3 ms of a 71.4 ms step moves no
  bytes, and byte-moving kernels average 191 GB/s, not the 233 GB/s peak.

## Getting started

Requires a Jetson AGX Thor (128 GB, CUDA 13) and the
[checkpoint with the head installed](https://huggingface.co/patrickbdevaney/DeepSeek-V4-Flash-0731-REAP-DSpark)
(100.4 GiB). Alternatively, use 0xSero's [0731-REAP checkpoint](https://huggingface.co/0xSero/DeepSeek-V4-Flash-0731-REAP)
plus the [draft head](https://huggingface.co/patrickbdevaney/dspark-mtp-draft-head-s3recap-p25-b0.1). Run nothing else that
loads a model: the weights take most of the unified memory pool.

```bash
bash scripts/build_server.sh       # unit gates, server, terminal client
bash scripts/serve.sh              # CPU gates as preflight, then listens on :8080
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Write a binary search in Python."}]}'
```

- **Endpoints:** `/v1/chat/completions`, streaming and not, `/v1/completions`, `/v1/models`, `/health`,
  `/metrics`, and a web UI at `/`.
- **Terminal client:** `build/dsv4-chat`.
- **Head selection:** `config/live_ckpt` selects the served head; `scripts/stage_head.sh <name> --activate`
  switches it.
- **Surface and design decisions:** [`docs/SERVER.md`](docs/SERVER.md).

For a standalone decode check, `scripts/run_model.sh` is the sanctioned launcher; it enforces single tenancy and
detaches:

```bash
bash scripts/build_decode.sh
scripts/run_model.sh <log> ./build/decode <ckpt> "0,671,6102,294,8760,344" 8   # BOS + "The capital of France is"
```

The expected first token is `11111`. A truncated id list still runs and reports a speed, but it gates against a
different sequence.

## Compatibility

| Platform | State |
|---|---|
| Jetson AGX Thor (`sm_110a`), CUDA 13 | Tested |
| Other REAP prunings of DeepSeek-V4-Flash with MXFP4 experts | Expected to load; the expert count is discovered since 2026-09-11. Not measured |
| Other GPUs or architectures | Not supported: kernels target `sm_110a` and Thor's unified memory |

## Limitations

- **Single request at a time.** There is no batching across clients; the KV prefix cache serves one agentic
  session.
- **Prefill is slow for long prompts.** At 108 tok/s, a 12K-token prompt takes about two minutes to process.
- **The draft head is checkpoint-specific.** It is trained against this 160-expert REAP. Another pruning needs its
  own head.
- **Held-out drafting.** Trained heads win the frozen suite but can lose on held-out continuations against a paired
  control. The β anchor addresses this mechanically, but the effect is not eliminated
  ([`wiki/draft-head-finetuning.md`](wiki/draft-head-finetuning.md)).
- **Only the live head is offsite.** The other ten trained heads exist as single local copies.

## Repository map

| Path | Contents |
|---|---|
| [`src/`](src/), [`kernels/`](kernels/), [`include/`](include/) | Engine: forward, decode, prefill, CUDA kernels |
| [`server/`](server/) | OpenAI-compatible server and terminal client |
| [`train/`](train/) | Draft-head fine-tuning |
| [`tests/`](tests/), [`protocol/`](protocol/) | Unit gates against PyTorch oracles; the frozen 8-prompt suite |
| [`tools/`](tools/), [`scripts/`](scripts/) | Benchmarks, evaluation harness, launchers, head staging and verification |
| [`docs/`](docs/) | Current reference: roofline, model inventory, server, best setup, head registry, evaluations, levers, decode ladder, final state |
| [`wiki/`](wiki/) | Topic pages synthesised from the programme; [`wiki/README.md`](wiki/README.md) is the index |
| [`evidence/`](evidence/) | Raw measurement logs and evaluation outputs behind the numbers |
| [`dev/archive/`](dev/archive/) | Programme history: research prompts, plans, handoffs, flywheel journals, `LOOP_LOG.md` (111 findings), superseded status pages |
| [`ref/`](ref/), [`original-fp8-path/`](original-fp8-path/), [`research/`](research/) | Reference implementations and early research |

Files kept at the root on purpose:
- [`CLAUDE.md`](CLAUDE.md) holds the operating rules for unattended work.
- `FLYWHEEL_STOP`, `FLYWHEEL_STATE.json`, `DECODE_LOOP_STOP` and `DECODE_PRIORITY` are the control files of the
  autonomous optimisation loops. `FLYWHEEL_STOP` is what keeps the hourly cron entry inert. Do not move it.

## Constraints the work held to

- **No additional quantization.** The checkpoint is used as shipped.
- **No invented constants.** Every model constant traces to `config.json`, `REAP_MANIFEST.json` or `reap_plan.json`.
- **Token ids come from the checkpoint's own tokenizer** (`tools/encode_prompt.py`).
- **Correctness before speed.** Correctness gates run before speed gates, with one change per measurement, reported
  as bands, not points.

## License

- **Code in this repository:** MIT ([`LICENSE`](LICENSE)).
- **Third-party code** keeps its own license: `include/third_party/httplib.h` and `json.hpp` (both MIT), and
  `ref/deepseek_v4_ref.py`, adapted from DeepSeek's reference inference code (MIT).
- **Weights:** the checkpoint is MIT, from `0xSero/DeepSeek-V4-Flash-0731-REAP`, derived from DeepSeek-V4-Flash.

## Citation

```bibtex
@misc{devaney2026dsv4cuda,
  title  = {deepseek-v4-flash-0731-cuda: lossless speculative decoding of DeepSeek-V4-Flash REAP on Jetson AGX Thor},
  author = {Devaney, Patrick},
  year   = {2026},
  url    = {https://github.com/patrickbdevaney/deepseek-v4-flash-0731-cuda}
}
```

Builds on DeepSeek-V4-Flash and its DSpark multi-token-prediction heads, 0xSero's REAP-160 pruning, and REAP (Lasby
et al., 2025, arXiv:2510.13999).
