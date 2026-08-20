# Handoff — 2026-08-19 23:10, decode-optimisation programme running unattended

Written to survive a session ending mid-flight. Everything below is either running detached or is a
one-command resume. Nothing here needs a human before morning.

## What is running right now

| what | pid/unit | state |
|---|---|---|
| `scripts/decode_loop.sh` | detached, ppid 1972 (`systemd --user`), tty `?` | iteration 2, ladder item **0.4** |
| the eval battery | **suspended on purpose** | `evidence/evals/SUSPENDED.md` |
| `dsv4-evals-watchdog.timer` / `.service` | **disabled on purpose** | re-enabling races the GPU |

**Watch it:** `tail -f evidence/decode_loop/live.log` — one line per tool call, with a `DONE` line
carrying turns, duration and cost. Coarse per-iteration record: `evidence/decode_loop/driver.log`.
Machine-readable: `evidence/decode_loop.jsonl`.

**Stop it:** `touch DECODE_LOOP_STOP` — it exits after the current iteration, cleanly.

## Where decode actually stands — the only scoreboard that matters

**One kernel change has shipped.** The warp-parallel top-k (`1a33cfe`), bit-identical, verified in
situ by the LOSSLESS gate on four independent runs.

```
                      pre-fix                     post-fix (n 48, R^2 0.971)
  ms/forward     136.44 + 30.053 x ctx/1000   130.98 + 7.362 x ctx/1000
  ctx  6,000          316.8 ms                     175.2 ms      1.81x
  ctx 12,000          497.1 ms                     219.3 ms      2.27x
  context range    71 - 6,592                   249 - 12,410
```

**Neither stop condition is met.** `a` is 1.59x its 82.18 ms byte floor (target 1.25x);
`b x 6592` is 48.53 ms against a 5.0 ms threshold. The loop continues.

Everything since the top-k has been measurement. That is the standing critique and it is written
into `DECODE_LADDER.md` as a rule the loop must read before picking an item: prefer a kernel change
over an instrument, and justify any instrument by naming the optimisation it unblocks.

## If the loop is dead when you return

```bash
cd ~/deepseek-v4-flash-0731-cuda
rm -f DECODE_LOOP_STOP
setsid nohup bash scripts/decode_loop.sh > evidence/decode_loop/driver.log 2>&1 < /dev/null &
```
It is idempotent: it re-reads `DECODE_LADDER.md`, takes the topmost unchecked item, and its preflight
refuses to start while a model is resident or memory is short.

## The three things a fresh session must not get wrong

1. **ONE MODEL AT A TIME.** 100.4 GiB of weights in a 122 GiB pool; this box does not OOM
   gracefully (two whole-machine takedowns on 2026-08-12, no oom-kill line either time). Full-model
   runs go through `scripts/run_model.sh`, which enforces single-tenancy and now arms a memguard on
   whatever it launches.
2. **`pgrep -f` MATCHES CLAUDE CODE'S OWN SHELLS.** The harness embeds the command text into the
   shell's command line, so `pgrep -f decode` matches the shell that is checking. This has cost two
   `pkill` self-kills, one misread runtime, and a memguard that adopted a shell as its victim. Match
   on `comm`.
3. **NEVER EDIT A RUNNING BASH SCRIPT IN PLACE.** Write a new file and `mv` it over: the running
   shell keeps its fd on the old inode. Truncating in place corrupts it mid-execution.

## Open items, in the order the ladder has them

- **0.4** (in flight) — attribute the residual 7.362 ms/1000 at ctx 12k. Decides between 1.2 and 1.5.
- **1.2 / 1.5** — REOPENED. Their retirement rested on 0.2, which 0.3 superseded. Whether the
  residual is those kernels is 0.4's job; do not assume it.
- **1.3 / 1.4** — correctness items. 1.4 removes a silent garbage-return above ~49k context
  (`topk_scan_smem` requests > 48 KiB dynamic shared memory with no `cudaFuncSetAttribute` opt-in).
- **1.6** — a pre-existing `invalid argument` launch fault inside `ogroup_gemm_fp8`, proven
  pre-existing by a control build from `1a33cfe^`. Latent: output correct, LOSSLESS passes, tau
  normal. Worth asking whether the failing launch is a dead branch.
- **1b.1** — MXFP4 index-cache packing. Primitives done and gated bit-exact
  (`tests/gate_idx_pack.cu`, 0 mismatches on 524,288 elements); the wiring is not done.

## Safety state, deliberately set

- The checkpoint is `chmod -R a-w`. The loop runs with `--dangerously-skip-permissions`, and while
  that does NOT override an explicit deny rule, the filesystem bit holds regardless. Revert with
  `chmod -R u+w ~/models/DeepSeek-V4-Flash-0731-REAP`.
- The live draft head is backed up at `~/model-backups/heads/shipped-dspark-0731reap/`, verified
  byte-exact. It is the head the server actually runs, and it was the ONE head not previously
  archived -- `promote_head.py` archives candidates and never writes the live checkpoint.
- `s3` (tau 3.8438 vs the shipped 3.5362) is archived and **has never been served**. Deploying it is
  ladder item 2.2 and is worth ~13 % on the bench suite.
