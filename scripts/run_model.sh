#!/usr/bin/env bash
# run_model.sh — the ONLY sanctioned way to launch a full-model binary.
#
# Enforces the two hard rules from HARDWARE.md §4 mechanically instead of by discipline:
#
#   1. SINGLE-TENANT. The weights are 100.4 GiB in a 117 GiB unified pool shared with the OS.
#      Two concurrent full-model processes exhaust memory and thrash swap. `flock` makes a second
#      launch fail loudly instead of silently starving the box.
#      (This is not hypothetical: on 2026-08-06 a failed-looking launch had in fact succeeded, the
#      relaunch created a second loader, and the pair drove `available` to 0 with swap in use.
#      See LOOP_LOG.md Finding 22.)
#
#   2. DETACHED. setsid+nohup so an SSH drop or a killed shell never leaves the GPU wedged or the
#      user locked out.
#
# Usage:  scripts/run_model.sh <logfile> <binary> [args...]
#   e.g.  scripts/run_model.sh ~/opt1.log ./build/decode ~/models/DeepSeek-V4-Flash-0731-REAP "0,671,..." 8
# Env passthrough works normally:  DSV4_KSWEEP=1 scripts/run_model.sh ...
set -euo pipefail

LOCK=/tmp/dsv4-fullmodel.lock

if [ $# -lt 2 ]; then
    echo "usage: $0 <logfile> <binary> [args...]" >&2
    exit 2
fi
LOG="$1"; shift

# Fail fast if the lock is already held, rather than queueing behind a 10-minute load.
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "REFUSED: another full-model process holds $LOCK" >&2
    echo "  running: $(ps -o pid,etime,args -C decode -C forward -C load_device --no-headers 2>/dev/null || echo '(none named)')" >&2
    exit 1
fi

AVAIL_GB=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
if [ "$AVAIL_GB" -lt 105 ]; then
    echo "REFUSED: only ${AVAIL_GB} GiB available; a full-model load needs ~105 GiB." >&2
    echo "  Free memory first (the page cache is reclaimable: sync; echo 3 | sudo tee /proc/sys/vm/drop_caches)." >&2
    exit 1
fi

echo "[run_model] ${AVAIL_GB} GiB available; launching detached -> $LOG"
# The lock is held by THIS shell (fd 9); the child inherits it and holds it for its lifetime.
setsid nohup "$@" > "$LOG" 2>&1 < /dev/null &
CHILD=$!
echo "[run_model] pid $CHILD   watch: tail -f $LOG"
# Keep fd 9 open for the child's lifetime without blocking the caller.
( while kill -0 "$CHILD" 2>/dev/null; do sleep 5; done ) >/dev/null 2>&1 &
disown 2>/dev/null || true
