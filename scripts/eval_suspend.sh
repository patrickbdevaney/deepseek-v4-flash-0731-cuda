#!/usr/bin/env bash
# eval_suspend.sh — stop the eval programme mid-flight, resumably, and prove it stayed stopped.
#
# WHY THIS IS A SCRIPT AND NOT A SEQUENCE OF KILLS. The programme is built to resurrect itself:
# `dsv4-evals-watchdog.timer` re-runs eval_resume.sh EVERY TEN MINUTES and `dsv4-evals.service`
# restarts everything at boot. Killing the stages without disarming those first buys ten minutes of
# quiet and then the whole battery comes back, holding the GPU that the kernel work needs. Disarm
# first, kill second, verify third -- in that order, every time.
#
# WHAT IS LOST. eval_extend.py flushes each record as it lands and skips ids already on disk, so the
# only casualty is the single item in flight. Everything else continues from its exact stored
# prefix. That is the whole reason it is safe to stop a six-day battery on a whim.
#
# WHAT IS WRITTEN. evidence/evals/SUSPENDED.md records what was running, what had landed, and what
# each row's truncation was at the moment of the stop -- because "resume later" is a promise that
# depends on somebody being able to reconstruct the state months from now, and process tables do
# not survive a reboot.
#
#   bash scripts/eval_suspend.sh          # stop
#   bash scripts/eval_resume.sh           # start again, exactly where it left off
set -u
cd "$(dirname "$0")/.."
say(){ echo "[suspend $(date -Is)] $*"; }

# ---- 1. DISARM THE RESURRECTION MACHINERY FIRST -------------------------------------------------
say "disarming the watchdog and the boot unit BEFORE killing anything"
systemctl --user stop    dsv4-evals-watchdog.timer 2>/dev/null
systemctl --user disable dsv4-evals-watchdog.timer 2>/dev/null
systemctl --user disable dsv4-evals.service        2>/dev/null
for u in dsv4-evals-watchdog.timer dsv4-evals.service; do
  state=$(systemctl --user is-enabled "$u" 2>&1)
  say "  $u -> $state"
  case "$state" in enabled) say "  WARNING: $u is STILL ENABLED — it will restart the battery";; esac
done

# ---- 2. SNAPSHOT THE STATE WHILE IT IS STILL TRUE -----------------------------------------------
say "recording state to evidence/evals/SUSPENDED.md"
python3 tools/eval_suite.py --report > /dev/null 2>&1
{
  echo "# Eval programme SUSPENDED — $(date -Is)"
  echo
  echo "Suspended deliberately to free the GPU for decode-kernel work. **Resume with"
  echo "\`bash scripts/eval_resume.sh\`**, which is idempotent and continues every stage from where"
  echo "it stopped. Re-enable the units too:"
  echo '```'
  echo "systemctl --user enable --now dsv4-evals-watchdog.timer"
  echo "systemctl --user enable dsv4-evals.service"
  echo '```'
  echo
  echo "## What was running at the moment of the stop"
  echo '```'
  ps -eo pid,etime,cmd | grep -E 'eval_|dsv4-server' | grep -v grep || echo "(nothing)"
  echo '```'
  echo
  echo "## Stage markers"
  for f in run.log extend.log extend_retry.log force.log bfcl_mt.log; do
    [ -e "evidence/evals/$f" ] || continue
    printf '| %-18s | ' "$f"
    grep -aoE 'ALL (TASKS|EXTENSIONS|EXTENSION RETRIES|FORCING|MULTI-TURN) COMPLETE' \
      "evidence/evals/$f" 2>/dev/null | tail -1 | tr -d '\n' || true
    echo " |"
  done
  echo
  echo "## Rows as of the stop"
  echo '```'
  python3 tools/eval_suite.py --report 2>/dev/null | tail -40
  echo '```'
  echo
  echo "## Extension progress (the resumable part)"
  echo '```'
  grep -a 'extend 2026\|^\[.*-> low24k\]\|^  \[[0-9]*/' evidence/evals/extend.log 2>/dev/null | tail -25
  echo '```'
} > evidence/evals/SUSPENDED.md
say "wrote evidence/evals/SUSPENDED.md"

# ---- 3. STOP THE STAGES, OUTERMOST FIRST --------------------------------------------------------
# Outermost first so a supervisor cannot notice its child dying and restart it.
for pat in eval_supervise.sh eval_watch.sh run_evals.sh \
           eval_bfcl_mt_run.sh eval_force_all.sh eval_extend_retry.sh eval_extend_all.sh \
           "eval_extend.py" "eval_suite.py --task" "eval_force.py" "eval_bfcl_mt.py"; do
  pids=$(pgrep -f "$pat" 2>/dev/null | grep -v "^$$\$" || true)
  [ -n "$pids" ] || continue
  say "stopping $pat -> $pids"
  # shellcheck disable=SC2086
  kill -TERM $pids 2>/dev/null || true
done
sleep 5
for pat in eval_supervise.sh eval_extend_all.sh "eval_extend.py"; do
  pids=$(pgrep -f "$pat" 2>/dev/null || true)
  [ -n "$pids" ] && { say "escalating to KILL for $pat -> $pids"; kill -KILL $pids 2>/dev/null; } || true
done

# ---- 4. THE ENGINE ------------------------------------------------------------------------------
# Kernel work means rebuilding and restarting anyway, and a resident 101 GiB engine is the single
# largest thing standing between here and a clean measurement.
if pgrep -f "dsv4-server --ckpt" > /dev/null; then
  say "stopping the engine (a rebuild would need this anyway)"
  pkill -TERM -f "dsv4-server --ckpt" 2>/dev/null || true
  for _ in $(seq 1 30); do pgrep -f "dsv4-server --ckpt" > /dev/null || break; sleep 1; done
  pgrep -f "dsv4-server --ckpt" > /dev/null && { say "engine did not exit; KILL"; pkill -KILL -f "dsv4-server --ckpt"; }
fi

# ---- 5. PROVE IT STAYED STOPPED -----------------------------------------------------------------
# The watchdog fires on a ten-minute cadence; a check taken one second after the kill proves nothing
# about the next tick. This proves the DISARM, which is the thing that actually holds.
sleep 3
say "verifying"
left=$(pgrep -af "eval_supervise|eval_extend|eval_force|eval_bfcl|run_evals|eval_suite.py --task|dsv4-server --ckpt" 2>/dev/null | grep -v "eval_suspend" || true)
if [ -n "$left" ]; then
  say "STILL RUNNING — investigate before starting kernel work:"; printf '%s\n' "$left"
else
  say "all eval stages and the engine are stopped"
fi
say "watchdog timer: $(systemctl --user is-active dsv4-evals-watchdog.timer 2>&1) / $(systemctl --user is-enabled dsv4-evals-watchdog.timer 2>&1)"
say "SUSPENDED. Resume with: bash scripts/eval_resume.sh  (and re-enable the units)"
