# STATUS

**Current state, 2026-08-22.** Everything below the divider is a **historical snapshot from
2026-08-06** kept for the record; its numbers were superseded many times over and must not be
quoted. The live scoreboard is [`README.md`](README.md); the sequencing is
[`PRODUCTION_PLAN.md`](PRODUCTION_PLAN.md).

## The four phases

| phase | state |
|---|---|
| 1. draft head + spec decode | **in flight** |
| 2. prefill to the roofline | not started |
| 3. prefix caching for agentic harnesses | not started |
| 4. the eval battery, once, at the final configuration | armed, deliberately not started |

## The scoreboard

| | 2026-08-06 (below) | today | |
|---|---:|---:|---|
| base AR decode | 9.51 tok/s | **14.61** | +53.6 % |
| speculative decode, suite mean | 9.48 (1.00× of base) | **28.38** | **1.94× of base** |
| acceptance τ | 3.12 / 5 | **3.84 / 5** | 77 % of the width ceiling |
| prefill (PS=1022) | not measured | **62.4 tok/s** | phase 2's target is ≥ 410 |
| `B_tok` | 11.202 GB/token | **12.26** | the early figure omitted terms |

Served head: **`s3recap-p25-b0.1`** at block width 5, `config/live_ckpt` →
`~/models/ckpt-head-s3recap-p25-b0.1`. Every emitted token is bit-identical to base AR, checked on
every run — the LOSSLESS gate has never been loosened.

## What the gate ledger below got right, and what it did not

The 2026-08-06 ledger's *gates* all still hold and several have been re-proven many times since.
What went stale is every **number**, and one framing: the AR wall of 21.42 tok/s was quoted as a
target. It is a **normalisation constant**, and treating it as reachable cost this project real
time. See `wiki/roofline-why-the-needle-wont-move.md` and `wiki/measurement-and-traps.md`.

Two of the five "not yet measured" items below are now answered: DSpark acceptance α (item 5) is the
subject of the entire draft-head programme and reads τ 3.84/5 today, and the `c_v(5) = 2.6×` verify
cost (item 4) is why the served block width moved from 6 to 5 — verify is expensive, so the optimum
width moves **down**, not up.

---

# Historical snapshot — 2026-08-06

*Preserved verbatim. Do not quote these numbers.*

**[2026-08-06] Phase 0 + Phase 1 complete. Gates H1 and R1 PASS. No kernel written yet — per the directive's
"report back before writing a single kernel".**

Last updated: 2026-08-06.

---

## Gate ledger

| Gate | What it demands | State |
|---|---|---|
| **H1** | hardware recorded before anything else | **PASS** — `HARDWARE.md` |
| **P0** | can cite the source file for every architectural constant | **PASS** — `MODEL_INVENTORY.md`, all from `docs/config.json` + 48 shard headers |
| **R1** | `ROOFLINE.md` with measured `B_tok`, AR wall, anchors, `E_frac(k)`, DSA unknown flagged | **PASS** — `ROOFLINE.md` |
| **A1** | oracle reproducible; `ARCH_DELTA` + `MODEL_INVENTORY` complete | **PARTIAL** — docs done, checkpoint down + integrity-checked, chat format specced (`CHAT_FORMAT.md`); oracle identified (`~/dspark-cuda-reap-finetune/ref/`) but not yet re-run against 0731 weights |
| **K** | unit kernel gates vs torch oracle | **PASS** — 20/20 on ported sources, goldens regenerated |
| **L1** | loads to device, peak < ~105 GB | **PASS** — 100.400 GiB / 45,821 tensors, zero-copy, GPU read verified vs file bytes |
| **G1–G8** | kernel gates + full AR forward | **PASS** — G1/G2/G5/G6/G7 via Gate K; **G3/G4/G8 via the full-model run**: "The capital of France is" → **" Paris. The capital of Spain is Madrid"** |
| G9 | CUDA graph capture | ported, not yet re-gated on 0731 |
| **D1** | DSpark spec-decode | **PASS** — embedded heads, memory-neutral, correct; at parity (0.97x) |
| **ENCODING** | chat encoder byte-exact vs vendor goldens | **PASS** — 4/4 vectors + 2 property checks |
| **API** | OpenAI request/response shaping | **PASS** — 30/30, incl. type-preserving DSML round-trip |
| S1 | full server | encoder + API shaping **done and gated**; HTTP transport + engine wiring pending |
| D1 | DSpark decode tok/s + acceptance | not started |
| S1 | multi-turn tool-calling session | not started |

## In flight

- **Checkpoint download: COMPLETE.** All 48 shards + metadata at
  `~/models/DeepSeek-V4-Flash-0731-REAP` (101 GiB on disk, 129 GiB free remaining), no
  `.incomplete` files. `sha256sum -c SHA256SUMS` (79 entries) running detached →
  `~/models/reap-0731-sha.log`.
- **Local artifact cross-checked against the remotely-harvested headers**: 45,821 tensors both
  sides, **keys identical, 0 dtype/shape mismatches**, byte total reconciles to
  107,803,320,952. `python3 tools/inventory.py --model-dir ~/models/DeepSeek-V4-Flash-0731-REAP`.
  The whole Phase-1 analysis is therefore validated against the real artifact, not just metadata.

## Headline numbers so far

| | value |
|---|---|
| Weights | 100.400 GiB of a 117 GiB pool → **16.6 GiB headroom** (23.1 without the MTP heads) |
| `B_tok` | **11.202 GB/token** |
| Achievable BW | **240 GB/s measured** (212 contended) — not the ~200 inherited |
| AR wall | **21.42 tok/s** @ 240 GB/s · 24.37 @ 273 spec |
| **MEASURED base AR decode (0731)** | **9.51 tok/s / 105.2 ms/tok = 116.6 GB/s = 48.6% of achievable** (Opts #1/#3/#4; was 7.80 = **1.219x**) |
| DSpark spec-decode | 9.48 tok/s = **1.00x of base** (was 0.93x). Draft head optimized 1.39x; the verify is now 84% of the round |
| Prior 180B anchor, for comparison | 7.89 tok/s / 126.7 ms/tok — **transferred to within 1.1%** |
| M=5 verify | 300.5 ms = **2.60×** an M=1 decode (model says 2.120×) — mechanism OPEN (Finding 15) |
| DSpark acceptance | **3.12 tokens/verify** of max 5 (α≈0.7) — acceptance is fine, cost is not |
| Base AR band after kernel work | **15–19 tok/s** (70–80% of achievable) |
| DSpark band (α-sensitive, k*≈2–3) | **22–36 tok/s**, centred ~28 |
| KV at fp8 | 3.36 KiB/token → 3.21 GiB at 1M context; **~105 MiB at 32K** |
| Expert quant | **OCP MXFP4** (E2M1 + E8M0, block 32) — same as the prior checkpoint |

## Not yet measured / open

1. ~~Achievable bandwidth~~ **RESOLVED: 240 GB/s measured.** Re-run idle to finalise (the sweep
   ran under download contention).
2. ~~`ncu` blocked~~ **RESOLVED: `sudo /usr/local/cuda-13.0/bin/ncu` works** — and revealed that
   Thor exposes no DRAM counters, so `ncu`'s "Memory Throughput %" is L2 throughput, not
   bandwidth utilisation (89%-of-peak kernel reports 30%). Bandwidth stays analytical.
3. DSA verify-step cost — no model, no measurement, no precedent.
4. `c_v(5) = 2.6×` measured vs 2.12× modelled — inherited unexplained.
5. 0731 DSpark acceptance rate α.

## Operating rules in force

- Memory-neutral optimisation only; never add persistent device allocations.
- Full-model binaries always run detached (`setsid nohup … > ~/run.log 2>&1 < /dev/null &`).
- Never loosen a gate. For deep fp8/fp4 compositions use relative-L2 + cosine + `max_abs/|o|max`,
  **not** per-element `max_rel` — the prior project proved `max_rel` is pathological there
  (rises 1.6% → 4.1% from seq16 → seq256 on a *correct* path while cosine stays 1.0000000).
- Single-tenant: only one process may hold the full model at a time.
- Bandwidth utilisation is measured analytically (byte model ÷ wall-clock), never from `ncu`'s
  "Memory Throughput %" — that metric is L2-derived on Thor and misleads. `HARDWARE.md` §3.
- No REAP pruning work, no additional quantisation, no invented model constants.
