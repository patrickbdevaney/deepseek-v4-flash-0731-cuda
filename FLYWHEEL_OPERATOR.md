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

## Watching it

The agent runs with `--output-format stream-json --verbose`, so every cycle is a JSONL transcript of
the whole agentic loop. `scripts/flywheel_watch.py` renders it as colourised dialogue — assistant
prose, tool calls with a per-tool one-line summary of the input, tool results with line/byte counts,
and a closing block with turns, wall time, token usage and cost.

```bash
scripts/flywheel_watch.py                       # follow the LIVE cycle (survives cycle rollover)
scripts/flywheel_watch.py --list                # every captured cycle: turns, duration, cost
scripts/flywheel_watch.py .flywheel_cycles/cycle0003-*.jsonl     # replay one
scripts/flywheel_watch.py -q      <file>        # tool calls and results only, no prose
scripts/flywheel_watch.py --full  <file>        # do not truncate tool output
```

Tool output is trimmed in the middle by default (head + tail, with a count of what was elided),
because a single `dprof` or `gemm_bench` result is hundreds of lines and the point of this view is
the *shape* of the iteration — what it decided, what it ran, what came back. The raw output is in
`~/*.log` anyway. `--full` when you are debugging a specific step.

Raw JSON if you would rather use your own tooling:
```bash
jq -c 'select(.type=="assistant") | .message.content[] | select(.type=="tool_use") | {name, input}' \
   .flywheel_last_run.jsonl
```

**One thing you will go looking for and not find: thinking TEXT.** The CLI redacts it — the
`thinking_delta` events carry `{"thinking": "", "estimated_tokens": N}` and the completed block has
a signature with empty content. The renderer therefore shows *that* the agent deliberated and
roughly how much (`* thinking (~150 tok, content not exposed on the stream)`), which is enough to
see where it paused to reason, but the reasoning itself is not available to any consumer of
stream-json. An empty thinking block does not mean it did not think.

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
