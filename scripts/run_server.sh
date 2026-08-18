#!/usr/bin/env bash
# run_server.sh — the sanctioned detached launcher for the SERVER, mirroring run_model.sh.
#
# WHY THIS EXISTS. scripts/run_model.sh has said since it was written that the benchmark must be
# "DETACHED. setsid+nohup so an SSH drop or a killed shell never leaves the GPU wedged". That
# lesson was learned and written down -- and then the server path was built without it:
# `grep -c 'setsid|nohup' scripts/serve.sh scripts/with_model_lock.sh` returns 0 and 0. Detachment
# for the server depended entirely on how the caller happened to invoke it.
#
# On 2026-08-17T21:55 a caller did not, the SSH session carrying it ended, SIGHUP killed the engine
# and eval_supervise with it, and the programme lost ~16 hours with the box still up. This closes
# that gap structurally rather than by remembering.
#
# serve.sh is deliberately NOT changed: it is still useful in the foreground for debugging. The
# rule is that anything long-running goes through a launcher that detaches, and this is that
# launcher for the server.
#
#   bash scripts/run_server.sh                 # SEQMAX/EXT_CHUNK from the environment
#
# For the full eval programme use scripts/eval_resume.sh instead -- it starts the server *and*
# every dependent stage detached, and is idempotent.
set -u
cd "$(dirname "$0")/.."
SEQMAX="${SEQMAX:-32768}"
EXT_CHUNK="${EXT_CHUNK:-64}"
LOG="${LOG:-evidence/eval_server.log}"

if curl -s -m 10 -o /dev/null "http://localhost:8080/health" 2>/dev/null; then
  echo "a server is already healthy on :8080 — refusing to start a second one."
  echo "Two concurrent model loads have taken this box down before (see memguard.sh)."
  exit 1
fi

echo "[run_server] starting detached, seqmax=$SEQMAX ext_chunk=$EXT_CHUNK, log -> $LOG"
setsid nohup scripts/with_model_lock.sh \
    env SEQMAX="$SEQMAX" EXT_CHUNK="$EXT_CHUNK" bash scripts/serve.sh \
    >> "$LOG" 2>&1 < /dev/null &
disown 2>/dev/null || true

# PROVE IT, DO NOT ASSERT IT. Detachment is a property of the process tree, so verify the tree.
for _ in $(seq 1 30); do
  pid=$(pgrep -f "build/dsv4-server --ckpt" | head -1)
  [ -n "$pid" ] && break
  sleep 1
done
if [ -z "${pid:-}" ]; then
  echo "[run_server] server process did not appear — check $LOG"; exit 1
fi
ppid=$(ps -o ppid= -p "$pid" | tr -d ' ')
tty=$(ps -o tty=  -p "$pid" | tr -d ' ')
echo "[run_server] pid=$pid ppid=$ppid tty=$tty"
if [ "$tty" != "?" ]; then
  echo "[run_server] WARNING: server has a controlling terminal — it will die with that terminal."
  exit 1
fi
echo "[run_server] detached. Loading ~101 GiB; watch: tail -f $LOG"
echo "[run_server] verify the whole chain any time with: bash scripts/detach_audit.sh"
