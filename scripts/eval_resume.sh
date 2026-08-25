#!/usr/bin/env bash
# eval_resume.sh — bring the whole eval programme back after a power cut, from wherever it died.
#
# WHAT THIS HEDGES AGAINST. eval_supervise.sh survives the *server* dying, but nothing survives
# the *supervisor* dying, and there are two ways that happens. The box loses power -- this machine
# has done it before -- which dsv4-evals.service covers by running this at boot. Or a stage was
# never detached in the first place: on 2026-08-17 the engine and eval_supervise were children of a
# Claude Code Bash invocation inside an SSH session, a laptop lid closed, SIGHUP killed both, and
# nothing noticed for sixteen hours with the box still up. dsv4-evals-watchdog.timer covers that by
# re-running this every ten minutes, and the audit at the bottom re-proves detachment each time.
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

# DETACHED MUST MEAN "OUTLIVES ITS PARENT", AND nohup+setsid DOES NOT GIVE YOU THAT UNDER SYSTEMD.
# setsid changes the SESSION; it does not change the CGROUP. When this script runs as a child of a
# systemd unit -- which is exactly how autopilot.sh's finalize() invokes it -- every stage launched
# here lands in that unit's cgroup, and systemd tears the cgroup down when the unit's main process
# exits. autopilot launches the battery and then exits immediately, so the whole battery would have
# been killed seconds after starting. That path had never been exercised end to end.
#
# MEASURED 2026-08-25 (and measured twice -- the first harness used `exec -a`, which renames argv[0]
# but not comm, so `pgrep -x` reported healthy processes as dead and produced a confidently wrong
# result): a nohup+setsid child of a transient unit IS killed when the unit's main process exits
# under the DEFAULT KillMode=control-group, and SURVIVES under KillMode=process. dsv4-autopilot runs
# with the default, so the battery would have died.
#
# KillMode=process on the launcher would therefore also work, but it is the weaker fix: it leaves
# the stages in a dead unit's cgroup, owned by nobody, invisible to systemctl, and still coupled to
# whoever happened to launch them. Own-unit-per-stage decouples them from that entirely.
#
# So each stage gets its OWN transient unit, with the stage itself as the main process. Its lifetime
# is then its own -- independent of this script, of autopilot, of any terminal, and of every other
# stage -- and it is inspectable with `systemctl --user status dsv4-ev-<stage>`.
# systemd requires an ABSOLUTE path for the executable; arguments stay relative to WorkingDirectory.
spawn(){   # $1 = unit suffix   $2 = log file, repo-relative, appended   rest = argv
  local unit="dsv4-ev-$1" log="$2"; shift 2
  systemctl --user reset-failed "$unit" 2>/dev/null
  systemd-run --user --unit="$unit" \
      --property=WorkingDirectory="$PWD" \
      --property=StandardOutput=append:"$PWD/$log" \
      --property=StandardError=append:"$PWD/$log" \
      "$@" >/dev/null 2>&1
}

# Launch a stage and do not return until pgrep can see it, so the next stage's wait loop cannot
# race past it. Ten seconds is generous for a bash exec; failing to appear is worth shouting about.
launch(){
  local script="$1" log="$2"
  spawn "${script%.sh}" "evidence/evals/$log" /usr/bin/bash "scripts/$script"
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
  spawn memguard evidence/evals/memguard_resume.log /usr/bin/bash scripts/memguard.sh
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
    spawn server evidence/eval_server.log "$PWD/scripts/with_model_lock.sh" \
        env SEQMAX=$SEQMAX EXT_CHUNK=$EXT_CHUNK bash scripts/serve.sh
    for _ in $(seq 1 120); do
      curl -s -m 10 -o /dev/null http://localhost:8080/health 2>/dev/null && break
      sleep 10
    done
  fi
else
  say "battery incomplete — starting the supervisor (it brings the server up too)"
  spawn supervise evidence/evals/supervise.log /usr/bin/bash scripts/eval_supervise.sh
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

# STAGE 2b: the extension RETRY. The sweep above visits each task exactly once and never looks
# back, so a task whose extension died mid-way is left with a partial merged file and the chain
# walks on. This finishes them. It must run BEFORE forcing, and forcing blocks on it.
if running eval_extend_retry.sh; then
  say "extension retry already running"
elif done_marker extend_retry.log "ALL EXTENSION RETRIES COMPLETE"; then
  say "extension retry already complete"
else
  launch eval_extend_retry.sh extend_retry.log
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

# THE RULE IS ONLY REAL IF SOMETHING CHECKS IT. A session-bound stage looks perfectly healthy
# until the connection drops, so every resume -- including every watchdog tick -- re-proves that
# what is running is actually detached. Quiet when clean; loud when not.
if ! out=$(bash scripts/detach_audit.sh 2>&1); then
  say "DETACHMENT FAULT — something is session-bound and will die with its terminal:"
  printf '%s\n' "$out"
fi

say "resume complete — stages self-order from here"
