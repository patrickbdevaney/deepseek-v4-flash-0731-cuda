#!/usr/bin/env bash
# eval_resume.sh — bring the whole eval programme back after a power cut, from wherever it died.
#
# WHAT THIS HEDGES AGAINST, AND WHAT IT DOES NOT. Every stage of the programme is a bare
# `nohup setsid` process. eval_supervise.sh survives the *server* dying, but nothing survives the
# *box* dying, and this machine has lost power mid-run before. A cut on day three of an eight-day
# programme would otherwise end it with no notification and no log line saying so. This is started
# by a systemd user unit at boot so the chain re-converges by itself.
#
# IT IS IDEMPOTENT AND IT IS RESUMPTIVE. Running it when everything is already healthy does
# nothing: each stage is skipped if it is already running or has already printed its completion
# marker. Nothing here re-runs finished work -- eval_suite skips completed ids, eval_extend
# continues from stored prefixes, eval_force skips ids already written -- so the worst case is
# losing the single item that was in flight when the power went.
#
# ORDER IS LOAD-BEARING. The stages self-order by waiting on each other's process names, so this
# only has to start them in the right sequence and confirm each is visible to pgrep before starting
# its dependent. Starting the pair in one breath is a real race: eval_extend_all's first poll can
# miss a supervisor that has not exec'd yet, and it EXITS rather than extends when it sees no
# battery and no completion marker.
#
#   bash scripts/eval_resume.sh          # safe to run at any time, including right now
set -u
cd "$(dirname "$0")/.."
SEQMAX="${SEQMAX:-32768}"
EXT_CHUNK="${EXT_CHUNK:-64}"
EFFORT="${EFFORT:-low}"
export EFFORT

say(){ echo "[resume $(date -Is)] $*"; }
running(){ pgrep -f "bash scripts/$1" > /dev/null; }
done_marker(){ grep -aq "$2" "evidence/evals/$1" 2>/dev/null; }

# Launch a stage and do not return until pgrep can see it, so the next stage's wait loop cannot
# race past it. Ten seconds is generous for a bash exec; failing to appear is worth shouting about.
launch(){
  local script="$1" log="$2"
  nohup setsid bash "scripts/$script" >> "evidence/evals/$log" 2>&1 < /dev/null &
  for _ in $(seq 1 20); do
    running "$script" && { say "started $script"; return 0; }
    sleep 0.5
  done
  say "WARNING: $script did not appear in the process table"
  return 1
}

mkdir -p evidence/evals

if ! pgrep -f "bash scripts/memguard.sh" > /dev/null; then
  say "starting memguard"
  nohup setsid bash scripts/memguard.sh > /dev/null 2>&1 < /dev/null &
fi

# STAGE 1: the battery. eval_supervise starts the server itself, waits for it, and restarts it if
# it dies, so it is also how the engine comes back. If the battery is already complete we still
# need an engine for the later stages, so start one directly in that case.
if running eval_supervise.sh || pgrep -f "tools/eval_suite.py --task" > /dev/null; then
  say "battery already running"
elif done_marker run.log "ALL TASKS COMPLETE"; then
  say "battery already complete"
  if ! curl -s -m 10 -o /dev/null http://localhost:8080/health 2>/dev/null; then
    say "server is down and later stages need it — starting under the model lock"
    nohup setsid scripts/with_model_lock.sh \
        env SEQMAX=$SEQMAX EXT_CHUNK=$EXT_CHUNK bash scripts/serve.sh \
        > evidence/eval_server.log 2>&1 < /dev/null &
    for _ in $(seq 1 120); do
      curl -s -m 10 -o /dev/null http://localhost:8080/health 2>/dev/null && break
      sleep 10
    done
  fi
else
  say "battery incomplete — starting the supervisor (it brings the server up too)"
  nohup setsid bash scripts/eval_supervise.sh >> evidence/evals/supervise.log 2>&1 < /dev/null &
  for _ in $(seq 1 20); do running eval_supervise.sh && break; sleep 0.5; done
  say "started eval_supervise.sh"
fi

# STAGE 2: the extension. Repairs rows that are over the truncation gate.
if running eval_extend_all.sh; then
  say "extension already running"
elif done_marker extend.log "ALL EXTENSIONS COMPLETE"; then
  say "extension already complete"
else
  launch eval_extend_all.sh extend.log
fi

# STAGE 3: forcing. Rescues the rows the extension could not reach. Declines every row the
# extension already brought under the gate, so on a healthy run this does work on about two rows.
if running eval_force_all.sh; then
  say "forcing already running"
elif done_marker force.log "ALL FORCING COMPLETE"; then
  say "forcing already complete"
else
  launch eval_force_all.sh force.log
fi

# STAGE 4: BFCL multi-turn. A new row rather than a repair, so it goes last.
if running eval_bfcl_mt_run.sh; then
  say "multi-turn already running"
elif done_marker bfcl_mt.log "ALL MULTI-TURN COMPLETE"; then
  say "multi-turn already complete"
else
  launch eval_bfcl_mt_run.sh bfcl_mt.log
fi

say "resume complete — stages self-order from here"
