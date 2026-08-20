#!/usr/bin/env bash
# wait_build.sh — block until no compiler is running. Use this instead of `pgrep -f <script>.sh`.
#
# `pgrep -f build_decode.sh` MATCHES THE SHELL THAT IS TYPING IT. Claude Code's bash wrapper embeds
# the text of the command into its own command line, so `until ! pgrep -f "build_decode.sh"` waits
# on itself and never returns. Iteration 8 hung this way for 10 minutes with zero compilers running
# and would have burned its whole 4 h timeout. The same trap has bitten memguard's victim selection
# and the watchdog's residency check; both were fixed by matching `comm` instead.
#
# `comm` is the executable name as the kernel reports it, so a shell is `bash` no matter what it is
# typing, and nvcc/cicc/ptxas/cudafe++ are the things that actually mean "a build is running".
#
#   scripts/wait_build.sh [timeout_s]     default 3600
set -u
TIMEOUT="${1:-3600}"
end=$(( $(date +%s) + TIMEOUT ))
compilers(){ ps -eo comm= | grep -cE '^(nvcc|cicc|ptxas|cudafe\+\+|cc1plus|g\+\+)$'; }
# settle: a build script between two compiler invocations shows zero for a moment
quiet=0
while [ "$(date +%s)" -lt "$end" ]; do
  if [ "$(compilers)" -eq 0 ]; then
    quiet=$((quiet+1))
    [ "$quiet" -ge 3 ] && { echo "wait_build: no compiler running ($(date -Is))"; exit 0; }
  else
    quiet=0
  fi
  sleep 5
done
echo "wait_build: TIMED OUT after ${TIMEOUT}s with $(compilers) compiler(s) still running" >&2
exit 1
