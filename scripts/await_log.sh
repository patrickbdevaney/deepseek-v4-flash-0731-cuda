#!/usr/bin/env bash
# await_log.sh LOG REGEX [MAX_SECONDS] — block until REGEX appears in LOG, or time out.
#
# Why this exists: a full-model measurement takes 10-20 minutes (a 100.4 GiB load plus decode), and
# run_model.sh detaches so the launching command returns immediately. The agent's Bash tool caps at
# 600 s, so no single call can span the run. This waits in a bounded window and reports which
# happened, so the caller loops:
#
#   scripts/run_model.sh ~/x.log ./build/decode "$MODEL" "0,671,6102,294,8760,344" 8
#   scripts/await_log.sh  ~/x.log 'SPEC-DECODE|GATE FAIL|cuda .*:'      # repeat until rc != 2
#
# exit 0  pattern found (the matching line is printed)
# exit 2  not yet — call again; the run is still going
# exit 3  the run died without matching (no decode process left and no match)
set -uo pipefail
LOG="${1:?usage: await_log.sh LOG REGEX [MAX_SECONDS]}"
PAT="${2:?}"
MAX="${3:-540}"                       # stay under the 600 s tool timeout
END=$(( $(date +%s) + MAX ))
while [ "$(date +%s)" -lt "$END" ]; do
    if [ -f "$LOG" ] && grep -qE "$PAT" "$LOG"; then
        grep -nE "$PAT" "$LOG" | tail -3
        exit 0
    fi
    if [ -f "$LOG" ] && ! pgrep -x decode >/dev/null; then
        sleep 3                        # settle: the process may be mid-exit with output buffered
        if grep -qE "$PAT" "$LOG"; then grep -nE "$PAT" "$LOG" | tail -3; exit 0; fi
        echo "run exited without matching /$PAT/; tail follows:" >&2
        tail -15 "$LOG" >&2
        exit 3
    fi
    sleep 10
done
echo "still running after ${MAX}s ($(wc -l < "$LOG" 2>/dev/null || echo 0) lines so far); call again" >&2
exit 2
