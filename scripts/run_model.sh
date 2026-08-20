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
# The GATE prompt is EXACTLY these six ids (BOS + "The capital of France is"); the expected first
# decoded token 11111 is only meaningful for this prompt. Do NOT abbreviate it — a truncated list
# still runs, still prints a tok/s, and silently reports GATE FAIL against a different sequence.
#   scripts/run_model.sh ~/opt1.log ./build/decode ~/models/DeepSeek-V4-Flash-0731-REAP "0,671,6102,294,8760,344" 8
# Env passthrough works normally:  DSV4_KSWEEP=1 scripts/run_model.sh ...
set -euo pipefail

LOCK=/tmp/dsv4-fullmodel.lock

if [ $# -lt 2 ]; then
    echo "usage: $0 <logfile> <binary> [args...]" >&2
    exit 2
fi
LOG="$1"; shift

# Bounded wait, not fail-fast. The lock is released by the watcher subshell below, which notices the
# child's exit on a 5 s poll -- while callers wait for the run to end with `while pgrep -x decode;
# do sleep 30; done`. Those are different conditions: for up to ~5 s decode is gone but the lock is
# still held. `flock -n` turned that window into an instant failure, and under `set -e` it killed
# session 3 outright at 18:07 after 19 h of work, between two adaptK sweep points.
#
# 60 s preserves the original intent -- do not queue behind a 10-minute model load -- while covering
# the handoff window by an order of magnitude. A caller that genuinely collides with a long run still
# gets refused, just 60 s later.
exec 9>"$LOCK"
if ! flock -w 60 9; then
    echo "REFUSED: another full-model process held $LOCK for the full 60s wait" >&2
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

# GUARD WHAT WE LAUNCHED, NOT JUST THE SERVER. memguard.sh defaults to PAT=dsv4-server and exits
# when its victim does, so every benchmark run here -- which loads the same 100.4 GiB into the same
# 122 GiB unified pool -- was running unguarded. On 2026-08-19 a two-prompt decode run reached
# `mem 116.5/122.8 GiB` and died with nothing watching it. This box does not OOM gracefully (see
# memguard.sh: two whole-machine takedowns on 2026-08-12, no oom-kill line in dmesg either time),
# and the normal operating point is under 3 GiB available -- so the guard is not optional, it is
# the only thing standing between a bad --seqmax and a reboot.
GUARD_PAT="$(basename "$1")"
# `9<&-` IS LOAD-BEARING. The single-tenancy lock is held by THIS shell on fd 9 and every child
# inherits it. The guard outlives the launch by design, so without closing fd 9 the guard holds the
# lock after the model has exited and the NEXT run_model.sh refuses with "another full-model process
# held the lock" -- which is exactly what happened on the first attempt.
setsid nohup env PAT="$GUARD_PAT" bash "$(dirname "$0")/memguard.sh" \
    > "${LOG%.log}.memguard.log" 2>&1 < /dev/null 9<&- &
echo "[run_model] memguard armed on '$GUARD_PAT' -> ${LOG%.log}.memguard.log"
# Keep fd 9 open for the child's lifetime without blocking the caller.
( while kill -0 "$CHILD" 2>/dev/null; do sleep 5; done ) >/dev/null 2>&1 &
disown 2>/dev/null || true
