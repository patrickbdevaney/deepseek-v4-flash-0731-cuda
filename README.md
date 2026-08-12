# deepseek-v4-flash-0731-cuda

A from-scratch pure-CUDA inference server for **`0xSero/DeepSeek-V4-Flash-0731-REAP`** (K160, native
MXFP4) with embedded **DSpark** self-speculative decoding, MLA + DSA attention, hand-tuned for
**Jetson AGX Thor, `sm_110a`**.

No Python on the hot path. Every kernel gated against a PyTorch oracle before it is trusted.

> ## Why
>
> **Local frontier-adjacent intelligence, on hardware you own, fast enough to do long-horizon
> agentic work unattended.** Neither half is rare alone — frontier capability is available through
> an API, fast local decode is available on small models. **Both at once, on one box, with no
> network in the loop, is not.**
>
> The four categories this engine runs *fastest* — long context 30.77, tool/JSON format 29.98,
> multi-turn 28.97, code edit 26.81 tok/s — are precisely the shapes agentic coding produces.
> Speculation pays where continuation is constrained, and agentic work is constrained by definition.
>
> **Every gain here is lossless**: emitted tokens identical to base AR, checked on every run. That
> invariant is what keeps "fast" and "frontier" from becoming a trade.
>
> **[`NORTH_STAR.md`](NORTH_STAR.md)** — the full argument, including the one number in it that is
> still inherited rather than measured.

---

## Where the numbers are today

| | measured | ceiling | |
|---|---|---|---|
| **speculative decode**, 8-prompt suite mean | **24.52 tok/s** (`s1`, shipped) | — | best category **32.42** |
| acceptance τ, suite mean | **3.58 / 7** | 7 at block 6 | the remaining headroom lives here |
| **base AR decode** | **13.76 tok/s** | 14.33–15.98 | **97 %** of the realistic floor |
| **prefill (PS=1022)** | **62.4 tok/s** | — | +30.3 % (F85/F86/F88) |

The shipped speculator is the **`s1` fine-tuned draft head**, promoted over the stock one at
**22.66 → 24.52 tok/s (+8.2 %)**. `s2` measures higher still (24.76) and is **not** shipped: the win
is inside the measured 3.5 % run-to-run spread and ties go to the incumbent. Every candidate,
rejects included, is in [`HEAD_REGISTRY.md`](HEAD_REGISTRY.md) with its weights archived.

From `evidence/s1_eval.log` (stock baseline: `evidence/baseline_blk6_suite.log`): clean run, clocks
pinned, caches dropped, `GATE PASS`, `LOSSLESS GATE PASS`, block 6, no profiling instruments in the
binary.

**One caveat that belongs next to the headline.** The trained heads win the frozen suite and *lose*
to the stock head on held-out continuation drafting — measured against a true paired control, `s1`
is −0.404 τ and `s2` −0.648 (F116). Training helps where the head is weak and hurts where it is
strong (F117). The suite number is the honest promotion criterion because the suite is the
representative mixed workload, but it is not the whole picture.

**The measurement protocol is part of the number.** τ is quoted as an **8-prompt suite mean at
NGEN0 ≥ 200** — past the drafter's 128-token sliding window. F92 measured τ at **1.39 over the first
32 generated tokens**, rising to ~3.2 only after ~128, so a short-generation acceptance figure is a
transient and is not comparable to anything, including this project's own earlier numbers. The
canonical 6-token prompt is retained as a **control only**; it is the worst case.

**The single most important correction this project has made to its own model of itself:** the
long-quoted "19.0 tok/s AR roofline" is a *normalisation constant, not a target*. It assumes every
kernel moves bytes at full DRAM bandwidth and that the non-byte part of the step is zero. Neither
holds — 22.3 ms of a 71.4 ms step is not bytes at all, and the byte-moving marks average 191 GB/s,
not 233. See `wiki/measurement-and-traps.md`.

## The wiki

| page | what it holds |
|---|---|
| **[`NORTH_STAR.md`](NORTH_STAR.md)** | why this project exists, what it is for, and the open capability question |
| **[`wiki/kernel-optimisations.md`](wiki/kernel-optimisations.md)** | every adopted AR/spec-decode optimisation: mechanism, measured gain, and the gate that proved it |
| **[`wiki/negative-results.md`](wiki/negative-results.md)** | the levers that were built and **retired**, with the number that killed each. Larger than the win list, and more useful. |
| **[`wiki/prefill-optimisation.md`](wiki/prefill-optimisation.md)** | B9 — why prefill ran decode-shaped kernels, and the four fixes (+30.3 %) |
| **[`wiki/draft-head-finetuning.md`](wiki/draft-head-finetuning.md)** | S5 — the ML: architecture, loss, data, hyperparameters, feasibility arithmetic, and what the literature actually says |
| **[`wiki/measurement-and-traps.md`](wiki/measurement-and-traps.md)** | how a number becomes trustworthy here, and the 30+ ways one has failed to |
| **[`wiki/hardware-sm110a.md`](wiki/hardware-sm110a.md)** | Thor: measured bandwidth **and compute** peaks, and the `sm_110a` ISA facts already settled |
| **[`wiki/cross-model-decode-comparison.md`](wiki/cross-model-decode-comparison.md)** | why this checkpoint decodes at half Qwen's rate on the same box — and why that is a quantisation ranking, not an engine ranking |
| **[`wiki/nvfp4-migration.md`](wiki/nvfp4-migration.md)** | if an NVFP4 REAP existed: what transfers, why the kernel work is a translation not a rewrite, and why requant must come BEFORE the dense GEMV work |
| **[`wiki/dense-mla-gemv.md`](wiki/dense-mla-gemv.md)** | the real lever — dense MLA GEMVs at 115–195 GB/s against a peer's 228–236 — and the bit-exactness invariant it collides with |

## Reference documents

| file | what it holds |
|---|---|
| `ROOFLINE.md` | the arithmetic that governs the project |
| `MODEL_INVENTORY.md` | checkpoint identity + every architectural constant, each traceable to a file |
| `HARDWARE.md` | the box, the memory constraint, `sm_110a` empirical facts |
| `LEVERS.md` | the implementation dedup ledger — what is open, what is closed, and why |
| `LOOP_LOG.md` | 111 findings, chronological. The primary source for everything in the wiki. |
| `RESEARCH_LOG.md` | the search dedup ledger |
| `S5_RECIPE.md` | the draft-head fine-tuning recipe |
| `S5_PROGRESSION.md` | the training session cadence, with stopping rules fixed before the data |
| `HEAD_REGISTRY.md` | every draft-head candidate and whether it was promoted — **rejects included** |
| `RUNS.md` | every fine-tune run and measurement with a link to its evidence log — generated, never hand-edited |
| `ARTIFACTS.md` | **where the draft-head weights live**, and which directory to upload |
| `protocol/suite_prompts.txt` | the frozen 8-prompt eval suite, as token ids |
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

### The server (Phase 6)

```bash
bash scripts/build_server.sh    # gates, server, terminal client
bash scripts/serve.sh           # CPU gates as preflight, then listen on :8080
build/dsv4-chat                 # terminal client
                                # web UI at http://localhost:8080/
```

OpenAI-compatible: `/v1/chat/completions` (streaming and not), `/v1/completions`, `/v1/models`,
`/health`, `/metrics`. Tool calls, thinking blocks, and a KV prefix cache for agentic turns. One
binary, no Python on the request path. See **`SERVER.md`** for the surface, the gates and the
design decisions — in particular why the tokenizer is not gemma's and why the engine is a separate
translation unit from `src/decode.cu`.
