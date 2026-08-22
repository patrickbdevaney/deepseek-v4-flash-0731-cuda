# The road to a production inference server

Four phases, in order, authorised 2026-08-21. Each ends with a measurement that can fail.
Reasoning and per-item detail: [`PHASE2_PLAN.md`](PHASE2_PLAN.md), [`DECODE_LADDER.md`](DECODE_LADDER.md).

**Why this order.** Decode and prefill are different bottlenecks (bandwidth at M=1, compute at
M>>1) and are optimised by different work, so they do not compete. The eval battery is ~3.46 M
tokens and takes over a day; running it last means running it once, at the configuration we
publish, which also removes the mixed-provenance problem instead of documenting it.

---

## Phase 1 — exhaust the draft head and speculative decode

`tau` is the only decode multiplier left: the AR floor is at the bandwidth roofline and the
remaining kernel items are bookkeeping. Status: RUNNING.

| item | what | state |
|---|---|---|
| control | s3's corpus re-captured post-1.10 | **done — tau 3.6250, NOT promoted** |
| ce/tv arms | `a_ce`/`a_tv` at 1.0/0.0 and 0.5/0.5 | running |
| P2.5 | deficit weighting + `beta*KL(q_new‖q_frozen)`, beta in {0, 0.1, 0.5} | queued |
| P2.6 | HASS: free-run the draft from step 2 | queued |
| 2.3 (train) | confidence head, `a_conf` set from the measured free-running magnitude | queued |
| 2.3 (serve) | use the confidence head at verify time, `argmax E[A(T_k)]/C(k)` | **not started — engine change** |
| P2.1 | the pattern labeller, then pattern-gated block width | **not started** |
| adaptK | re-tune the gate against whichever head wins | not started |

The control is already informative: re-capturing with the fixed `hadamard` did **not** lift `tau`
(3.6250 against the incumbent's 3.6888, inside the 3.5 % spread), so the racing capture was not
costing decode and the pre-1.10 heads are not compromised. One hypothesis retired.

**Gate:** every arm archived with its per-category table; the release rule in
[`HEAD_REGISTRY.md`](HEAD_REGISTRY.md) decides which is published.

## Phase 2 — prefill to the roofline

**Prefill is the weakest number in the system and it is not close to the hardware.** Best measured
end to end is **62.4 tok/s**, so a cold 12,282-token context is **~3.3 minutes to first token**.

The diagnostic is the ratio: prefill runs at only **~2.4x the decode rate**. It should be one to
two orders of magnitude. At M=1 decode re-reads every weight per token and is bandwidth-bound; at
M=1022 one weight read serves a thousand tokens, so prefill should be compute-bound. 2.4x means
the weight reads are **not being amortised across the batch**. The repo already named the cause:
*"seventeen cycles optimised decode inside functions that prefill shares, and prefill was never
measured."*

Attribution at PS=1022, 99.98 % accounted:

| region | % |
|---|---|
| MoE | 42.6 |
| `compressed_attn_forward` | 42.6 |
| KV cache population | 10.5 |
| everything else | 4.4 |

- **P2.7** — `DSV4_DPROF` on a PS=1023 prefill on the CURRENT engine. Never run. One load.
- **The m16 B-repack, at prefill shapes.** F129 rejected it for decode because it needs 5.05
  rows/expert and block 6 presents 1.67. At M~1022 across 160 experts each expert sees ~50 rows,
  **ten times the threshold**. The optimisation that does not pay at decode should pay heavily
  here, and it has never been tested at these shapes. This is the single largest named lever.
- **`compressed_attn_forward` at prefill shapes** — the other 42.6 %.

**Pre-registered target: TTFT < 30 s at 12 k uncached, i.e. >= 410 tok/s, a 6.6x.** If the cost
turns out diffuse rather than concentrated, say so and stop rather than grinding.

## Phase 3 — prefix caching, end to end, for agentic harnesses

The OpenAI surface already exists and is more complete than a first look suggests:
`POST /v1/chat/completions` (streaming and non-streaming, **tools**, thinking blocks),
`POST /v1/completions`, `GET /v1/models`, `/health`, `/metrics`.

The prefix cache exists too: `use_prefix_cache = true`, `Engine::extend`, a `cached_tokens`
counter, a `set_prefix_cache()` toggle for forcing a cold prefill, and a resolver that reports
exactly what `generate()` would reuse. So this phase is **verification and measurement, not
construction** — which is the right thing to be suspicious of, because an unexercised cache is
indistinguishable from a working one until the turn it silently misses.

- **P2.8** — a real 20-turn agentic session with a growing prefix, tool-call turns, and a
  **mid-context compression event at turn 10**, which invalidates every cache block after the edit
  point. That is the access pattern the eval battery never exercises and the one that turns an
  amortised cold prefill into a repeated one.
- Report **p50/p90 TTFT** across the session, cached vs `set_prefix_cache(false)`, and the
  `cached_tokens` hit rate per turn.
- Re-decide `1b.2`'s default against the PRODUCTION context distribution: it is net positive above
  ctx 6,471 and buys 3x the context in the same pool, which is a different calculation for a
  long-horizon agent than for a decode benchmark.

**Gate:** measured p50/p90 TTFT for a 20-turn session, reported as whatever they are.

## Phase 4 — the eval battery, once

Remaining: three extensions (aime25 21, gpqa_diamond 51, mmlu_pro 18 — all dry-run verified as
**0 not-continuable**) plus `bfcl_mt`, which has never run and starts clean. Driven by
`scripts/eval_stage.sh`: verify the archive, report, commit, stamp provenance, then
`eval_resume.sh`. **The battery does not start if the archive does not verify.**
