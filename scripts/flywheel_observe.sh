#!/usr/bin/env bash
# flywheel_observe.sh — the headless observer. Audits the last cycle and can HALT the loop.
#
# The worker and the observer are separate processes on purpose: the thing that makes changes cannot
# publish them, and the thing that judges them cannot make them. This is the judging half, and it
# runs on system cron so oversight survives any interactive session ending.
#
# It has exactly two powers beyond reading: append to FLYWHEEL_OBSERVER.md, and set halt=true in
# FLYWHEEL_STATE.json. Halting is the point — if cycle N produced a number that cannot be traced to
# a run, cycle N+1 must not build on it. Everything else is denied by .claude/settings.json.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
LOCK=/tmp/dsv4-flywheel-observe.lock
log(){ printf '[observe %s] %s\n' "$(date -Is)" "$*"; }

exec 9>"$LOCK"; flock -n 9 || { log "already observing"; exit 0; }
[ -f "$ROOT/FLYWHEEL_STOP" ] && { log "FLYWHEEL_STOP present"; exit 0; }

# Do not audit a cycle that is still running — half a transcript reads like a failure.
if pgrep -f "scripts/flywheel.sh" >/dev/null || pgrep -f "build/decode" >/dev/null; then
    log "a cycle is in flight; skipping this pass"; exit 0
fi

LASTSEEN="$ROOT/.flywheel_observed"
HEAD_NOW=$(git rev-parse HEAD)
if [ -f "$LASTSEEN" ] && [ "$(cat "$LASTSEEN")" = "$HEAD_NOW" ]; then
    log "no new commit since last pass ($(git log -1 --format=%h)); nothing to audit"; exit 0
fi

PROMPT=$(cat <<'EOP'
You are the OBSERVER for the decode optimisation flywheel. An autonomous executor ran a cycle. Audit
it. You do not implement, you do not fix, you do not push. You read, you judge, and if the cycle is
unsound you HALT the loop so the next one cannot build on it.

Read: the newest entry in FLYWHEEL_JOURNAL.md, the commit it describes (`git show --stat` then the
substantive hunks), FLYWHEEL_STATE.json, and the tail of LOOP_LOG.md.

Audit against these, in order of how much they have historically cost this project:

 1. IS THERE A NUMBER, AND WAS IT ACTUALLY RUN? Take every performance number in the entry and grep
    for it in ~/*.log. If a number cannot be traced to a log file with a timestamp after the
    previous cycle, that is Finding 33 — a config written up that was never executed — and it is an
    automatic HALT.
 2. FULL-MODEL OR BENCH ONLY? gemm_bench overstates end-to-end value by 2-4x (Finding 47) because it
    relaunches a kernel on rotating weights while the engine serialises everything behind a data
    dependency. A lever ADOPTED on bench evidence alone is unsound. Flag it; halt if it changed a
    default.
 3. COULD THE GATE HAVE FAILED? A gate that allocates its own inputs with cudaMalloc always gets
    256-byte alignment and cannot catch the misalignment that crashed prefill (Finding 41). If a new
    kernel was added without a gate that could fail, flag it.
 4. ONE CHANGE PER MEASUREMENT? Two changes in one measurement means neither was measured.
 5. WAS A RETIRED LEVER RE-PROPOSED? The retired-with-a-measurement list in LOOP_LOG.md is binding.
 6. PREDICTED VS MEASURED WITHIN 2x? If not, the ranking model is wrong and should have been fixed
    before the next lever. Flag if it was not.
 7. IS THE LOOP SPINNING? If the last three cycles all touched the same phase for sub-1% results,
    the ranking model is exhausted and the loop should advance to Phase D. Say so explicitly.

Then APPEND to FLYWHEEL_OBSERVER.md (create if absent), at most 20 lines:

  ## <UTC timestamp> — cycle <N>, commit <short sha>
  **VERDICT: SOUND | UNSOUND | HALTED | SPINNING | NO-OP**
  what the cycle claims, in one sentence
  what you verified, and how (which log, which line)
  any checklist item it failed, by number
  what the next cycle should do

If the verdict is UNSOUND, set "halt": true and "halt_reason": "<observer: reason>" in
FLYWHEEL_STATE.json. Be conservative about halting a cycle that is merely unimpressive — a small or
negative result honestly measured is a GOOD cycle and should be marked SOUND. Halt for
unverifiability, not for disappointment.

Do not run git beyond read-only inspection. Do not edit sources. Finish by printing your verdict
line so it lands in the cron log.
EOP
)

# `< /dev/null` is load-bearing, not tidiness. Without it `claude -p` waits on stdin ("no stdin
# data received in 3s"), and a detached or cron-launched process has no terminal to supply one --
# the stream then aborts mid-turn with terminal_reason=aborted_streaming and
# "[Request interrupted by user]" in the transcript. It cost two whole cycles to find.
claude -p "$PROMPT" \
  --output-format stream-json --verbose \
  --permission-mode bypassPermissions \
  --max-turns 40 \
  --add-dir "$HOME" \
  --append-system-prompt "You are the observer, not the executor. You never implement or fix. Halt for unverifiability, not for disappointment." \
  > "$ROOT/.flywheel_last_observe.jsonl" 2>&1 < /dev/null
RC=$?
echo "$HEAD_NOW" > "$LASTSEEN"
log "observer exited rc=$RC; verdict: $(grep -o 'VERDICT: [A-Z-]*' "$ROOT/FLYWHEEL_OBSERVER.md" 2>/dev/null | tail -1)"
python3 -c "
import json;s=json.load(open('$ROOT/FLYWHEEL_STATE.json'))
print('[observe] loop halted by observer:', s.get('halt_reason')) if s.get('halt') else None" 2>/dev/null
