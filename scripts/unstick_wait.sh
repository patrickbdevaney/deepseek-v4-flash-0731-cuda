#!/usr/bin/env bash
# unstick_wait.sh — free an agent that is waiting on `pgrep -f <pattern>` and therefore on itself.
#
#   scripts/unstick_wait.sh <pattern> [max_wait_s]
#
# `pgrep -f X` matches any process whose COMMAND LINE contains X, and Claude Code's bash wrapper
# embeds the command text into its own command line. So `until ! pgrep -f "foo.sh"` keeps matching
# the shell running it long after foo.sh is gone, and waits forever. scripts/wait_build.sh and the
# loop prompt prevent new instances; this rescues one already stuck.
#
# It distinguishes the REAL process from the wrappers by the thing that actually differs: a wrapper's
# command line contains `shell-snapshots`. So it waits for every non-wrapper match to exit, then
# kills the wrappers -- which makes the agent's Bash call return and the iteration continue.
set -u
PAT="${1:?usage: unstick_wait.sh <pattern> [max_wait_s]}"
MAX="${2:-14400}"
cd "$(dirname "$0")/.."
LOG=evidence/decode_loop/unstick.log
say(){ echo "[unstick $(date -Is)] $*" | tee -a "$LOG"; }

real_pids(){ ps -eo pid=,args= | grep -F -- "$PAT" | grep -v shell-snapshots | grep -v "unstick_wait" | awk '{print $1}'; }
wrap_pids(){ ps -eo pid=,args= | grep -F -- "$PAT" | grep    shell-snapshots | awk '{print $1}'; }

say "watching for '$PAT' — real: $(real_pids | tr '\n' ' ') wrappers: $(wrap_pids | tr '\n' ' ')"
end=$(( $(date +%s) + MAX ))
while [ "$(date +%s)" -lt "$end" ]; do
  if [ -z "$(real_pids)" ]; then
    w="$(wrap_pids)"
    if [ -n "$w" ]; then
      say "'$PAT' has finished but these wrappers are still waiting on themselves: $(echo $w) — killing"
      # shellcheck disable=SC2086
      kill -9 $w 2>/dev/null
      say "done; the agent's Bash call returns and the iteration continues"
    else
      say "'$PAT' finished and nothing is stuck; nothing to do"
    fi
    exit 0
  fi
  sleep 15
done
say "gave up after ${MAX}s with '$PAT' still running"
exit 1
