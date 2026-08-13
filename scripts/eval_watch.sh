#!/usr/bin/env bash
# eval_watch.sh — land each benchmark the moment it finishes, without touching the running battery.
#
# This is deliberately a SEPARATE process rather than a step inside run_evals.sh. bash reads a
# script incrementally as it executes, so editing run_evals.sh while a battery is running can change
# what a long-lived shell reads next and corrupt the run. Watching its log from outside cannot.
#
# Polls evidence/evals/run.log for the "<task> done" lines run_evals.sh emits, and calls
# eval_land.sh once per task. eval_land refuses to commit anything that does not re-derive, so a
# task that finished with no records (an aborted benchmark) is skipped and stays visible as missing
# rather than being published as complete.
#
#   nohup setsid bash scripts/eval_watch.sh > evidence/evals/watch.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
LOG="${LOG:-evidence/evals/run.log}"
POLL="${POLL:-60}"
declare -A landed

echo "[watch] started $(date -Is), polling $LOG every ${POLL}s"
while true; do
  if [ -f "$LOG" ]; then
    while read -r task; do
      [ -n "$task" ] || continue
      [ -n "${landed[$task]:-}" ] && continue
      landed[$task]=1
      echo "[watch] $(date -Is) $task finished — landing"
      bash scripts/eval_land.sh "$task" || echo "[watch] $task did not land (see above)"
    done < <(grep -oP '^===\s+\S+\s+\K\w+(?=\s+done)' "$LOG" 2>/dev/null | sort -u)
  fi
  # Stop once the battery is finished AND everything it produced has been landed.
  if grep -q "ALL TASKS COMPLETE" "$LOG" 2>/dev/null; then
    echo "[watch] battery complete, exiting $(date -Is)"; exit 0
  fi
  if ! pgrep -f run_evals.sh > /dev/null && ! pgrep -f "eval_suite.py --task" > /dev/null; then
    echo "[watch] no battery running, exiting $(date -Is)"; exit 0
  fi
  sleep "$POLL"
done
