#!/usr/bin/env bash
# eval_supervise.sh — keep the battery running unattended, and never lie about why it stopped.
#
# The battery is ~20 hours and runs overnight with nobody watching. Three things can end it early,
# and all three have happened on this project:
#
#   * the server dies (a bad item, or memguard firing before the kernel takes the box down)
#   * run_evals.sh exits because it found the server down before a task
#   * the box reboots (four times so far; once from two concurrent model loads, hence the model lock)
#
# In every case the correct response is the same: bring the server back and resume. The harness is
# resumable by construction -- completed ids are skipped -- so a restart costs at most the item that
# was in flight. What must NOT happen is the loop restarting forever against a broken engine and
# turning a defect into hundreds of items scored wrong, so restarts are capped and each one is
# logged with the reason.
#
#   nohup setsid bash scripts/eval_supervise.sh > evidence/evals/supervise.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
MAX_RESTARTS="${MAX_RESTARTS:-6}"
EFFORT="${EFFORT:-low}"
SEQMAX="${SEQMAX:-32768}"
EXT_CHUNK="${EXT_CHUNK:-64}"
export TMPDIR="${TMPDIR:-/tmp}"
export EFFORT

say(){ echo "[supervise $(date -Is)] $*"; }

server_up(){ curl -s -m 10 -o /dev/null "http://localhost:8080/health" 2>/dev/null; }

start_server(){
  say "starting server (seqmax=$SEQMAX ext_chunk=$EXT_CHUNK) under the model lock"
  SEQMAX=$SEQMAX EXT_CHUNK=$EXT_CHUNK nohup setsid scripts/with_model_lock.sh \
      env SEQMAX=$SEQMAX EXT_CHUNK=$EXT_CHUNK bash scripts/serve.sh \
      > evidence/eval_server.log 2>&1 < /dev/null &
  for _ in $(seq 1 120); do server_up && { say "server up"; return 0; }; sleep 10; done
  say "server FAILED to come up in 20 min"; return 1
}

start_guard(){
  pgrep -f "bash scripts/memguard.sh" >/dev/null && return 0
  say "starting memguard"
  nohup setsid bash scripts/memguard.sh > /dev/null 2>&1 < /dev/null &
}

start_watcher(){
  pgrep -f "bash scripts/eval_watch.sh" >/dev/null && return 0
  say "starting watcher (lands + commits + pushes each finished benchmark)"
  EFFORT="$EFFORT" nohup setsid bash scripts/eval_watch.sh >> evidence/evals/watch.log 2>&1 < /dev/null &
}

for attempt in $(seq 0 "$MAX_RESTARTS"); do
  start_guard
  server_up || start_server || { say "cannot start server, giving up"; exit 1; }
  start_watcher

  say "battery attempt $attempt"
  SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}" EFFORT="$EFFORT" bash scripts/run_evals.sh \
      >> evidence/evals/run.log 2>&1
  rc=$?

  if grep -q "ALL TASKS COMPLETE" evidence/evals/run.log 2>/dev/null; then
    say "ALL TASKS COMPLETE — battery finished"
    python3 tools/eval_suite.py --report >> evidence/evals/run.log 2>&1
    exit 0
  fi
  say "run_evals exited rc=$rc without completing; resuming after a pause"
  sleep 60
done

say "hit MAX_RESTARTS=$MAX_RESTARTS without completing — stopping so a real defect is not"
say "converted into a battery of wrong answers. Check evidence/evals/run.log."
exit 1
