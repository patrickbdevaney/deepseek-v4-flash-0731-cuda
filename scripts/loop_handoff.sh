#!/usr/bin/env bash
# loop_handoff.sh — swap the running loop onto new driver code at a clean iteration boundary.
#
# bash parses a `for` loop's whole body before executing it, so an already-running decode_loop.sh
# keeps its old logic no matter what is written to the file. New driver code only takes effect on a
# fresh start -- but restarting mid-iteration throws away whatever the current one is measuring.
# So: ask it to stop (DECODE_LOOP_STOP is honoured at the top of the next iteration), wait for the
# current iteration to land, then start it again on the new code.
#
# DECODE_LOOP_STOP ALSO STOPS THE WATCHDOG, so leaving it behind wedges the programme silently.
# Every exit path removes it.
set -u
cd "$(dirname "$0")/.."
LOG=evidence/decode_loop/handoff.log
say(){ echo "[handoff $(date -Is)] $*" | tee -a "$LOG"; }
cleanup(){ rm -f DECODE_LOOP_STOP; }
trap cleanup EXIT INT TERM

running(){ ps -eo args= | grep -q "[d]ecode_loop.sh"; }

if ! running; then
  say "no loop running; starting one on the current code"
else
  say "asking the running loop to stop after its current iteration"
  touch DECODE_LOOP_STOP
  # 8 h: an iteration is capped at ITER_TIMEOUT=14400s (4 h) plus up to 4 h of post-check residency wait.
  for i in $(seq 1 2880); do
    running || break
    [ $((i % 30)) -eq 0 ] && say "still waiting for the current iteration ($((i/30)) x 5 min)"
    sleep 10
  done
  if running; then
    say "the loop did not stop in 8 h — leaving it alone rather than killing work mid-flight"
    exit 1
  fi
  say "loop stopped cleanly"
fi

cleanup   # remove DECODE_LOOP_STOP before starting, or the new loop exits immediately
sleep 2
say "starting the loop on the new driver code (priority: $(grep -vE '^\s*#|^\s*$' DECODE_PRIORITY | head -3 | tr '\n' ' '))"
setsid nohup bash scripts/decode_loop.sh >> evidence/decode_loop/driver.log 2>&1 < /dev/null &
sleep 5
if running; then say "loop is up"; else say "FAILED to start the loop — check driver.log"; exit 1; fi
