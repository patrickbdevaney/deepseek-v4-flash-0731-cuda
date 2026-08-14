#!/usr/bin/env bash
# eval_doctor_loop.sh — run the doctor on a schedule so faults are caught in minutes, not days.
#
# The supervisor only re-examines the world when run_evals EXITS. On a six-day battery that can be
# days away, so anything that dies mid-run stays dead: the watcher stops landing finished
# benchmarks, the extension pass never happens, a half-written record sits waiting to break resume.
# Every one of those leaves the battery apparently healthy while the RESULT is being lost.
#
# So the invariants are checked on a clock instead. Repairs are conservative -- restart what is safe
# to restart, truncate a partial trailing record after taking a backup -- and anything riskier is
# printed as ESCALATE rather than acted on, because a watchdog that guesses is worse than one that
# reports. It never touches the engine: an unattended process should not be restarting a 113 GiB
# model load on a box that has been taken down by exactly that before.
#
#   nohup setsid bash scripts/eval_doctor_loop.sh > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
EVERY="${EVERY:-600}"
LOG="${LOG:-evidence/evals/doctor.log}"

while true; do
  python3 tools/eval_doctor.py >> "$LOG" 2>&1
  # Stop once there is nothing left to guard: no battery, and no extension pass pending.
  if ! pgrep -f "bash scripts/eval_supervise.sh" > /dev/null \
     && ! pgrep -f "bash scripts/eval_extend_all.sh" > /dev/null; then
    echo "[doctor-loop $(date -Is)] nothing left to supervise, exiting" >> "$LOG"
    exit 0
  fi
  sleep "$EVERY"
done
