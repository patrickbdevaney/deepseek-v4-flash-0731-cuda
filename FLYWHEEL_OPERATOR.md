# FLYWHEEL_OPERATOR.md — running the loop, and stopping it

Two processes, deliberately separated.

| | worker | observer |
|---|---|---|
| what | `scripts/flywheel.sh` — one lever, then exits | a Claude session reviewing each cycle |
| trigger | **system cron, hourly at :17** (survives reboots and sessions) | session cron at :42, expires after 7 days |
| authority | edit / build / gate / ONE full-model run / **commit locally** | read-only; recommends, never edits |
| cannot | `git push`, run two model processes, touch the checkpoint, add model-changing work | — |

The split is the point: **the thing that makes changes cannot publish them, and the thing that
judges them cannot make them.** An autonomous loop that both edits and pushes has no check on a
plausible-but-wrong result, and this project has produced several (Findings 33, 39, 47).

## Stopping it

```bash
touch FLYWHEEL_STOP                 # immediate, survives everything, no session needed
crontab -l | grep -v flywheel_cron | crontab -    # permanent
```
The loop also halts itself on: any unit gate failing, a run printing `GATE FAIL`, corrupt state
JSON, or `halt: true` in `FLYWHEEL_STATE.json`. A halt is sticky — it waits for a human.

## What it costs per cycle

One `claude -p` invocation (bounded at 140 turns) and at most one full-model run: ~10 min to load
100.4 GiB plus a few minutes of decode. Budget ~30-40 min of wall clock and one build. The hourly
cadence leaves slack; `flock` makes an overrun skip the next tick rather than stack.

## Reading it

- `FLYWHEEL_JOURNAL.md` — the watch log, one entry per cycle, plus the observer's checklist.
- `LOOP_LOG.md` — full reasoning and every retired lever. **The retired list is the loop's most
  valuable output**: it is what stops the next cycle paying twice for the same negative result.
- `~/flywheel_cron.log` — preflight/postflight, halts, and what was committed.
- `.flywheel_last_run.txt` — the last agent transcript, when a cycle looks wrong.

## The honest expectation

At handover the loop is at **16.86 tok/s speculative / 12.61 tok/s base**, with
`consecutive_sub_half_pct = 3` — the Phase A stopping rule is *already armed*, and the last two
Phase-B rankings returned the same top entry. So the loop should advance to Phase D quickly. That is
the designed behaviour, not a failure: the levers this project can still name are each worth under
1%, and Finding 49 showed the value now comes from re-framing the bottleneck rather than grinding it.

**What the loop is genuinely for is the tail**: it runs while nobody is watching, it cannot forget a
retired lever, and it re-enters research on a mechanical trigger instead of when someone remembers
to. What it will not do is find the two things that actually gate further progress — MTP head repair
and MLA quantisation — because both are model changes that are explicitly the user's call.
