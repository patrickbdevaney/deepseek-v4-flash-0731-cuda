# Handoff — 2026-08-22, phase 1 running unattended

Written to survive a session ending mid-flight. Everything below is either running detached or is a
one-command resume. Nothing here needs a human.

Supersedes the 2026-08-19 handoff (the autonomous decode loop). **The kernel loop is finished and
deliberately stopped** — its remaining items are worth single digits and are listed at the bottom of
`DECODE_PRIORITY`. What runs now is the draft-head programme, which spends Thor time rather than
agent turns.

## What is running right now

| what | unit | state |
|---|---|---|
| the P2.6 HASS arms | `dsv4-p26` | arm 1 refused; arm 2 (`hass1-p25`) in its final measurement |
| the P2.5 sweep chain | `dsv4-chain`, `dsv4-supervise` | **complete** — inactive because they finished, not because they failed |
| the eval battery | `dsv4-evalstage` | **armed and deliberately disarmed**. Phase 4. Do not start it. |

**Watch it:** `tail -f evidence/chain/p26.log`. Per-arm detail lands in
`evidence/<arm>_{eval,holdout,gate}.log`. The registry row is the verdict:
`tail -5 HEAD_REGISTRY.md`.

## Where things actually stand

**Served head: `s3recap-p25-b0.1`, block width 5, τ 3.8413, 28.38 tok/s** — the first promotion of
the programme and +25.3 % over the stock head this project started from. `config/live_ckpt` is
tracked in git, so which head is in production is reviewable in a diff.

| | measured |
|---|---|
| speculative decode, 8-prompt suite mean | **28.38 tok/s** |
| acceptance τ | **3.84 / 5** (77 % of the width ceiling) |
| base AR decode | **14.61 tok/s** — at its realistic floor |
| prefill (PS=1022) | **62.4 tok/s**, ~3.3 min TTFT at 12 k — **the largest gap in the system** |

## The four phases, and what "done" means for each

Full text and gates: `PRODUCTION_PLAN.md`.

1. **Draft head + spec decode** *(in flight)*. Done when every acceptance lever is measured, not
   when one wins. Remaining: `hass1-p25`, then **2.3** (confidence head — needs a training arm
   *and* an engine change at verify time), **P2.1** (labeller → pattern-gated width), adaptK re-tune.
2. **Prefill to the roofline.** Target **TTFT < 30 s at 12 k uncached ⇒ ≥ 410 tok/s, a 6.6×**.
   First action is `DSV4_DPROF` on a PS=1023 prefill, which **has never been run**. See
   `PHASE2_PLAN.md`.
3. **Prefix caching.** A 20-turn agentic session with a growing prefix, tool-call turns and
   mid-context compression at turn 10; report p50/p90 TTFT cached vs `set_prefix_cache(false)`.
4. **The eval battery, once, at the final configuration.**

## The five things a fresh session must not get wrong

1. **ONE MODEL AT A TIME.** 100.4 GiB of weights in a 122 GiB pool; this box does not OOM
   gracefully (two whole-machine takedowns on 2026-08-12, no oom-kill line either time). Full-model
   runs go through `scripts/run_model.sh`, which enforces single-tenancy and arms a memguard.
2. **NEVER send a generation request to the engine while a benchmark is scoring.** `GET /metrics`
   is safe; nothing else is.
3. **`pgrep -f` MATCHES CLAUDE CODE'S OWN SHELLS.** The harness embeds the command text into the
   shell's command line, so `pgrep -f decode` matches the shell that is checking. Two `pkill`
   self-kills, one misread runtime, one memguard that adopted a shell as its victim. Match on `comm`.
4. **NEVER EDIT A RUNNING BASH SCRIPT IN PLACE.** Bash reads scripts by **byte offset**, so an edit
   to a running script executes garbage from the offset onward. Write a new file and `mv` it over,
   or fork it — this is why `scripts/s5_session_p25.sh` exists as a fork rather than an edit, and
   why the 2026-08-22 promoter fix went into the Python rather than into `chain_p26.sh`.
5. **DETACH UNATTENDED WORK VIA SYSTEMD, NOT `&`.** `setsid nohup … &` from a tool call is reaped
   when the call ends — it reports a live pid and then vanishes. Use `systemd-run --user`. Also:
   **journald has no journal files on this box**, so unit output must go to a file via
   `StandardOutput=append:`.

## τ, and the one way to misread it

**τ is not comparable across block widths.** It counts tokens committed per target forward and its
ceiling *is* the draft width. `s3` reads 3.8438 at width 6 and **3.6888 at width 5 — same weights.**
Any comparison must be at one width, against an incumbent measured at that width
(`scripts/baseline_tau.sh` produces one). `promote_head.py` enforces this: it filters the registry
by `DSV4_PROTOCOL_BLOCK` and **fails closed** if it can find no bar.

Also: the hold-out sweep prints a suite mean at **adaptK 2.0**; the registry number is the frozen
8-prompt protocol at **adaptK 1.50**. They are different instruments and the session log says so.
Do not read a promotion out of a sweep line.

## Safety state, deliberately set

- The checkpoint is `chmod -R a-w`. Revert with `chmod -R u+w ~/models/DeepSeek-V4-Flash-0731-REAP`.
- **Nothing is ever deleted from `~/model-backups/heads/`** — every arm of every sweep is archived
  whether or not it promoted. A refused head is still a measured point on the acceptance curve, and
  this project has twice discovered its **ruler** was wrong after the fact (ladder 2.4; traps §38).
  `python3 tools/verify_head_archive.py` proves the archive is intact; `--full` hashes ~30 GB and
  needs an idle box.
- `dsv4-evalstage.service` is armed and disarmed. It starts the battery. Leave it alone until
  phase 3 closes.
