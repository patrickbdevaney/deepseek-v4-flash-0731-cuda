# deepseek-v4-flash-0731-cuda

A from-scratch pure-CUDA inference server for **`0xSero/DeepSeek-V4-Flash-0731-REAP`** (K160, native
MXFP4) with embedded **DSpark** self-speculative decoding, MLA + DSA attention, hand-tuned for
**Jetson AGX Thor, `sm_110a`**.

No Python on the hot path. Every kernel gated against a PyTorch oracle before it is trusted.

---

## Where the numbers are today

| | measured | ceiling | |
|---|---|---|---|
| **speculative decode** | **22.15 tok/s** | 23.2–25.9 | 96 % of the realistic ceiling |
| **base AR decode** | **13.83 tok/s** | 14.33–15.98 | 97 % of the realistic ceiling |
| acceptance | 2.89 / 5 | — | the remaining 1.4× lives here |
| **prefill (PS=1022)** | **62.4 tok/s** | — | +30.3 % this session, still open |

All decode figures from `evidence/clean_post_f83.log`: clocks pinned, caches dropped, `GATE PASS`,
`LOSSLESS GATE PASS`, no profiling instruments in the binary.

**The single most important correction this project has made to its own model of itself:** the
long-quoted "19.0 tok/s AR roofline" is a *normalisation constant, not a target*. It assumes every
kernel moves bytes at full DRAM bandwidth and that the non-byte part of the step is zero. Neither
holds — 22.3 ms of a 71.4 ms step is not bytes at all, and the byte-moving marks average 191 GB/s,
not 233. See `wiki/measurement-and-traps.md`.

## The wiki

| page | what it holds |
|---|---|
| **[`wiki/kernel-optimisations.md`](wiki/kernel-optimisations.md)** | every adopted AR/spec-decode optimisation: mechanism, measured gain, and the gate that proved it |
| **[`wiki/negative-results.md`](wiki/negative-results.md)** | the levers that were built and **retired**, with the number that killed each. Larger than the win list, and more useful. |
| **[`wiki/prefill-optimisation.md`](wiki/prefill-optimisation.md)** | B9 — why prefill ran decode-shaped kernels, and the four fixes (+30.3 %) |
| **[`wiki/draft-head-finetuning.md`](wiki/draft-head-finetuning.md)** | S5 — the ML: architecture, loss, data, hyperparameters, feasibility arithmetic, and what the literature actually says |
| **[`wiki/measurement-and-traps.md`](wiki/measurement-and-traps.md)** | how a number becomes trustworthy here, and the 30+ ways one has failed to |
| **[`wiki/hardware-sm110a.md`](wiki/hardware-sm110a.md)** | Thor: measured bandwidth **and compute** peaks, and the `sm_110a` ISA facts already settled |

## Reference documents

| file | what it holds |
|---|---|
| `ROOFLINE.md` | the arithmetic that governs the project |
| `MODEL_INVENTORY.md` | checkpoint identity + every architectural constant, each traceable to a file |
| `HARDWARE.md` | the box, the memory constraint, `sm_110a` empirical facts |
| `LEVERS.md` | the implementation dedup ledger — what is open, what is closed, and why |
| `LOOP_LOG.md` | 88 findings, chronological. The primary source for everything in the wiki. |
| `RESEARCH_LOG.md` | the search dedup ledger |
| `S5_RECIPE.md` | the draft-head fine-tuning recipe |
| `DECODE_FLYWHEEL.md` | the autonomous optimisation loop's operating manual |

## The model

43 MoE backbone layers + 3 chained DSpark MTP blocks (layers 40/41/42, `mtp.0/1/2`). Hidden 4096,
64 heads × head_dim 512, Q-LoRA/O-LoRA rank 1024, 8 o-groups, 160 routed experts top-6 + 1 shared,
`moe_intermediate` 2048, hyper-connections ×4 with 20 Sinkhorn iterations, sliding window 128,
vocab 129280.

Quantisation **as shipped, never re-quantised**: MLA/dense FP8 e4m3 with F8_E8M0 128×128 block
scales; routed experts OCP MXFP4 (E2M1 + E8M0, block 32); norms/embed/lm_head/compressor/indexer
BF16. `B_tok` = **12.26 GB/token**.

## Hard constraints this repo operates under

- **No additional quantisation.** The checkpoint is used as shipped.
- **No invented model constants.** Every number traces to `config.json`, `REAP_MANIFEST.json` or
  `reap_plan.json`. This has been violated once — a head size copied from a *paper* about a
  different model — and the correction is recorded in `RESEARCH_LOG.md` §6(a).
- **Token ids come from the checkpoint's own tokenizer** (`tools/encode_prompt.py`, which gates
  itself on reproducing the canonical prompt). Inventing ids is the exact mistake the rule exists
  to stop.
- **Correctness gates before speed gates.** One change per measurement. Report bands, not points.
- **DSpark is not DFlash.** `~/gemma-cuda-hybrid` and `~/laguna-s1-cuda-server` are read-only
  references.

## Build and run

```bash
bash scripts/build_decode.sh                      # the engine
bash scripts/build_gate.sh                        # the unit gates

# the ONLY sanctioned launcher: enforces single-tenancy and detaches
scripts/run_model.sh <log> ./build/decode <ckpt> "0,671,6102,294,8760,344" 8
```

The six-id prompt is `BOS + "The capital of France is"` and the expected first decoded token is
`11111`. **Do not abbreviate it** — a truncated list still runs, still prints a tok/s, and silently
reports `GATE FAIL` against a different sequence.

Useful environment flags are catalogued in `LEVERS.md` §5. The two that matter most:
`MOE_MMA=1` (tensor-core MoE — right for prefill, **wrong for decode**) and `DSV4_DPROF=1`
(multi-level named GPU-phase timing, ~0.4 % overhead).
