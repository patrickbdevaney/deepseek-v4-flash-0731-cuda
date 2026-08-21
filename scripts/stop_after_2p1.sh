#!/usr/bin/env bash
# stop_after_2p1.sh — touch DECODE_LOOP_STOP once ladder 2.1 is checked off.
#
# WHY A WATCHER AND NOT A POLL. decode_loop.sh tests for DECODE_LOOP_STOP at the TOP of each
# iteration, before it picks an item. Touching the file while 2.1 is still unopened would skip 2.1
# entirely; touching it by hand at the right moment costs a human (or an agent) a poll loop. 2.1 is
# the last kernel item the programme needs -- everything else defers past the heads and the evals --
# so the correct trigger is "2.1 is done", which is exactly `- [x] **2.1` appearing in the ladder.
#
# The loop then finishes whatever iteration it is in and stops cleanly at the next top-of-loop.
set -euo pipefail
cd "$(dirname "$0")/.."
while :; do
    if grep -q '^- \[x\] \*\*2\.1' DECODE_LADDER.md; then
        touch DECODE_LOOP_STOP
        printf '[stop_after_2p1 %s] 2.1 is checked off; DECODE_LOOP_STOP set\n' "$(date -Is)"
        exit 0
    fi
    if ! pgrep -x -f 'bash scripts/decode_loop.sh' >/dev/null 2>&1; then
        printf '[stop_after_2p1 %s] loop is gone; nothing to stop\n' "$(date -Is)"
        exit 0
    fi
    sleep 120
done
