# STATUS.md

**Phase 0 + Phase 1 complete. Gates H1 and R1 PASS. No kernel written yet — per the directive's
"report back before writing a single kernel".**

Last updated: 2026-08-06.

---

## Gate ledger

| Gate | What it demands | State |
|---|---|---|
| **H1** | hardware recorded before anything else | **PASS** — `HARDWARE.md` |
| **P0** | can cite the source file for every architectural constant | **PASS** — `MODEL_INVENTORY.md`, all from `docs/config.json` + 48 shard headers |
| **R1** | `ROOFLINE.md` with measured `B_tok`, AR wall, anchors, `E_frac(k)`, DSA unknown flagged | **PASS** — `ROOFLINE.md` |
| **A1** | oracle reproducible; `ARCH_DELTA` + `MODEL_INVENTORY` complete | **PARTIAL** — docs done; oracle identified (`~/dspark-cuda-reap-finetune/ref/`) but not yet re-run against 0731 weights |
| L1 | loads to device, peak < ~105 GB, cached restart < 60 s | not started (blocked on download) |
| G1–G9 | kernel gates | not started; **G1–G7 substantially pre-built and pre-gated** in the prior repo (see `ARCH_DELTA.md` §2) |
| D1 | DSpark decode tok/s + acceptance | not started |
| S1 | multi-turn tool-calling session | not started |

## In flight

- **Checkpoint download**: `0xSero/DeepSeek-V4-Flash-0731-REAP` → `~/models/DeepSeek-V4-Flash-0731-REAP`,
  running detached, log at `~/models/reap-0731-download.log`. 100.4 GiB total.
  Disk at survey: 230 GiB free → ~130 GiB after. Verify `SHA256SUMS` on completion.

## Headline numbers so far

| | value |
|---|---|
| Weights | 100.400 GiB of a 117 GiB pool → **16.6 GiB headroom** (23.1 without the MTP heads) |
| `B_tok` | **11.202 GB/token** |
| AR wall | **17.85 tok/s** @ 200 GB/s · 24.37 @ 273 GB/s |
| Direct anchor (identical `B_tok`, same box, measured) | **7.89 tok/s / 126.7 ms/tok = 32% of peak BW** |
| Base AR band after kernel work | **12–18 tok/s** |
| DSpark band (α-sensitive, k*≈2–3) | **18–34 tok/s**, centred ~25 |
| KV at fp8 | 3.36 KiB/token → 3.21 GiB at 1M context; **~105 MiB at 32K** |
| Expert quant | **OCP MXFP4** (E2M1 + E8M0, block 32) — same as the prior checkpoint |

## Not yet measured / open

1. Achievable streaming bandwidth on this build (200 GB/s inherited, unverified).
2. `ncu` blocked by `ERR_NVGPUCTRPERM` — **needs the user's sudo**; converts "32% of peak" from
   inference to per-kernel attribution.
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
- No REAP pruning work, no additional quantisation, no invented model constants.
