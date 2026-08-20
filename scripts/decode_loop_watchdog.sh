#!/usr/bin/env bash
# decode_loop_watchdog.sh — restart the decode loop if it dies, and ONLY if that is safe.
#
# The eval programme learned this the expensive way: a stage that dies unattended costs the whole
# night, and nothing noticed for sixteen hours. The decode loop has the same exposure -- it runs
# detached with no supervisor -- so it gets the same treatment the battery got.
#
# BUT A WATCHDOG THAT RESTARTS A FAILING THING IS WORSE THAN NO WATCHDOG. It converts one failure
# into a thrash that burns the token budget and fills the repo with half-finished commits. So every
# condition below is a REASON NOT TO RESTART, and the default is to do nothing:
#
#   * DECODE_LOOP_STOP exists            -> a human or the loop itself asked it to stop. Respect it.
#   * the loop is already running        -> nothing to do.
#   * a model is resident                -> the loop's own preflight would refuse anyway; do not
#                                           add a second process waiting on the GPU.
#   * MemAvailable below the floor       -> same.
#   * the CPU gates fail                 -> the tree is broken. Restarting would build on it.
#   * it has already been restarted N times this window -> something is wrong that restarting will
#                                           not fix, and the loop is meant to be watched, not nursed.
set -u
cd "$(dirname "$0")/.."
LOG=evidence/decode_loop/watchdog.log
STAMP=evidence/decode_loop/.watchdog_restarts
MAX_RESTARTS="${MAX_RESTARTS:-4}"
FLOOR_GB="${FLOOR_GB:-100}"
mkdir -p evidence/decode_loop
say(){ echo "[watchdog $(date -Is)] $*" >> "$LOG"; }

[ -e DECODE_LOOP_STOP ] && exit 0
# comm-based, because pgrep -f matches any shell whose command line mentions the script.
ps -eo args= | grep -q "[d]ecode_loop.sh" && exit 0
ps -eo comm= | grep -qE '^(dsv4-server|decode|decode_probe|decode_prechange)$' && { say "a model is resident; not restarting"; exit 0; }

avail=$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo)
[ "$avail" -ge "$FLOOR_GB" ] || { say "only ${avail} GiB available; not restarting"; exit 0; }

for g in gate_tokenizer gate_encoding gate_api gate_topk_warp gate_topk_radix gate_idx_pack; do
  [ -x "build/$g" ] || continue
  ./build/"$g" > /dev/null 2>&1 || { say "GATE FAIL: $g — refusing to restart onto a broken tree"; exit 0; }
done

# A restart that failed on a usage limit did no work and must not count against the cap -- that is
# what burned all four restarts overnight while the account was rate limited.
if [ -f "$LOG" ] && tail -40 evidence/decode_loop/driver.log 2>/dev/null | grep -qiE "usage/session limit"; then
  say "last stop was a usage limit, not a fault; clearing the restart counter"
  rm -f "$STAMP"
fi
n=$(cat "$STAMP" 2>/dev/null || echo 0)
if [ "$n" -ge "$MAX_RESTARTS" ]; then
  say "already restarted $n times; standing down. Clear $STAMP to re-arm."
  exit 0
fi
echo $((n+1)) > "$STAMP"
say "loop is gone and everything checks out — restart $((n+1))/$MAX_RESTARTS"
setsid nohup bash scripts/decode_loop.sh >> evidence/decode_loop/driver.log 2>&1 < /dev/null &
