#!/usr/bin/env bash
# memguard.sh — kill the server before the kernel kills the box.
#
# WHY THE KERNEL NEVER HELPS, MEASURED 2026-08-20. A 2048 MiB `cudaMalloc`, touched, charges
# **42 MiB** to the calling process's cgroup (tools: scratchpad `cgtest.cu`). Tegra unified memory
# is handed out by the driver's allocator, not the page allocator, so the model's ~100 GiB is
# invisible to every kernel memory-accounting path. Three consequences, and they explain every
# takedown this box has had:
#
#   * the OOM killer scores by RSS/memcg, so a 100 GiB model looks ~42 MiB "large" and is NEVER
#     selected -- hence no oom-kill line in dmesg, three takedowns running;
#   * the pages are driver-pinned: not page cache, not swappable, not reclaimable;
#   * so cgroup `memory.max` CANNOT bound this process, and `systemd-oomd` cannot run at all
#     (this kernel has no PSI -- /proc/pressure/memory does not exist).
#
# Polling MemAvailable is therefore not the lazy option, it is the ONLY option.
#
# WHY THE OLD POLL RATE COULD NOT FIRE IN TIME. A loaded server sits at `mem 119.6/122.8 GiB`, and
# the measured healthy low-water across 13 runs is 2389 MB. Against a 1500 MB floor that is 889 MB
# of separation, and the load ramp climbs ~100 GiB in ~60 s = ~1710 MB/s -- so the gap between
# "healthy plateau" and "dead" is **0.52 seconds**. The old settings needed POLL_S=2 x BREACHES=3
# = 6 s to decide, i.e. ~10 GiB of overshoot past the floor. The guard was not slow, it was
# arithmetically incapable of ever firing before the box died. That is the whole bug.
#
# THE FIX IS SLOPE, NOT JUST LEVEL. The floor cannot simply be raised: the healthy plateau is
# 2389 MB and this config's normal operating point is under 3 GiB, so any floor high enough to catch
# the ramp early also kills servers that have loaded fine -- which it did once already, costing a
# run. But the ramp and the plateau differ in a way a level test cannot see and a slope test cannot
# miss: a healthy server that has finished loading has slope ~0, while a runaway at the same
# MemAvailable is still falling at ~1700 MB/s. So there are two independent triggers:
#
#   LEVEL  avail < FLOOR_MB for BREACHES consecutive samples        (the original test, kept)
#   SLOPE  avail < DANGER_MB *and* falling faster than RATE_MB_S    (the ramp, which is the killer)
#
# and the poll goes to 0.2 s so the level test has ~1 GiB of overshoot instead of ~10 GiB.
#
# DANGER_MB MUST SIT BELOW THE HEALTHY LOW-WATER, and getting this wrong is how the guard kills a
# server that loaded fine. A healthy ramp does not decelerate into its plateau -- it falls at ~1710
# MB/s right up to the moment loading completes and then stops dead. So at any level ABOVE the
# plateau, "falling fast" describes the healthy case exactly as well as the runaway, and the slope
# test cannot tell them apart there. It only separates them below where a healthy run ever reaches:
# the measured minimum across 13 runs is 2389 MB, so the trigger arms at 2000 MB -- 389 MB of margin
# on the healthy side, and 2000/1710 = 1.2 s of warning on the fatal side, against the 0.3 s the
# level test alone would leave.
#
#   FLOOR_MB    kill below this MemAvailable             (default 1500)
#   BREACHES    consecutive samples under floor to kill  (default 3  -> 0.6 s)
#   DANGER_MB   slope trigger only armed below this      (default 2000, under the 2389 low-water)
#   RATE_MB_S   kill if falling faster than this         (default 400)
#   SLOPE_N     consecutive slope samples to kill        (default 2  -> 0.4 s)
#   MEMINFO     meminfo path, overridable for testing    (default /proc/meminfo)
#   POLL_S      seconds between samples                  (default 0.2)
#   PAT         victim's `comm`                          (default dsv4-server)
# THE GUARD MUST NOT PERTURB WHAT IT WATCHES. At POLL_S=0.2 a naive loop spawns `awk` and `ps` five
# times a second -- ~9000 processes per 30-minute model run -- and this project's run-to-run spread is
# 3.5 %. So the hot path is spawn-free: MemAvailable is read with bash builtins, and the victim lookup
# (the expensive part, a full `ps`) is refreshed on a slow cadence and forced immediately on a trip.
set -u
FLOOR_MB="${FLOOR_MB:-1500}"
BREACHES="${BREACHES:-3}"
DANGER_MB="${DANGER_MB:-2000}"
RATE_MB_S="${RATE_MB_S:-400}"
POLL_S="${POLL_S:-0.2}"
POLL_MS=$(awk -v p="$POLL_S" 'BEGIN{printf "%d",p*1000}')   # once, at startup
SLOPE_N="${SLOPE_N:-2}"
MEMINFO="${MEMINFO:-/proc/meminfo}"
PAT="${PAT:-dsv4-server}"
LOG="${LOG:-evidence/memguard.log}"

cd "$(dirname "$0")/.."
# SPAWN-FREE WAIT. `sleep` is not a bash builtin, so one per poll is 5 forks/s for the whole run.
# `read -t` on a fifo nobody writes to blocks for exactly the timeout and costs nothing.
_FIFO=$(mktemp -u); mkfifo "$_FIFO" 2>/dev/null && exec 8<>"$_FIFO" && rm -f "$_FIFO"
napp(){ if [ -e /proc/self/fd/8 ]; then read -t "$POLL_S" -u 8 _x 2>/dev/null; else sleep "$POLL_S"; fi; return 0; }
echo "[memguard] floor=${FLOOR_MB}MB x${BREACHES} slope=<-${RATE_MB_S}MB/s below ${DANGER_MB}MB poll=${POLL_S}s pat=${PAT} started $(date -Is)" | tee -a "$LOG"

low=999999999
bad=0
srate=0
tick=0
pid=""
VICTIM_EVERY="${VICTIM_EVERY:-10}"   # 10 x 0.2 s = 2 s
prev=""
maxrate=0
kill_victim(){
  echo "[memguard] !!! $1 — KILLING pid $2 $(date -Is)" | tee -a "$LOG"
  kill -9 "$2" 2>/dev/null
  sleep 5
  echo "[memguard] killed. MemAvailable now $(awk '/MemAvailable/{print int($2/1024)}' "$MEMINFO") MB" | tee -a "$LOG"
  exit 1
}
while true; do
  avail=""
  while read -r _k _v _; do
    if [ "$_k" = "MemAvailable:" ]; then avail=$((_v/1024)); break; fi
  done < "$MEMINFO"
  [ -n "$avail" ] || { echo "[memguard] meminfo exhausted, exiting" | tee -a "$LOG"; exit 0; }
  # SELECT THE VICTIM BY `comm`, NOT BY COMMAND LINE. `pgrep -f decode` matches every shell whose
  # command line merely CONTAINS "decode" -- and Claude Code's bash wrapper embeds the text of the
  # command it is running into its own command line. So the old form could adopt an interactive
  # shell as its "victim", never exit when the real model did, and -- since this script's whole job
  # is `kill -9` -- eventually kill that shell instead of the loader. `comm` is the executable name
  # as the kernel reports it; a shell is `bash` regardless of what it is typing.
  # Refresh at most every VICTIM_EVERY polls; a `ps` per poll is the expensive part.
  if [ $((tick % VICTIM_EVERY)) -eq 0 ] || { [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; }; then
    pid=$(ps -eo pid=,comm= | awk -v p="$PAT" '$2==p {print $1; exit}')
    [ -n "$pid" ] || pid=$(ps -eo pid=,comm= | awk -v p="$(basename "$PAT")" '$2==p {print $1; exit}')
  fi
  tick=$((tick+1))

  if [ -z "$pid" ]; then
    if [ "${seen:-0}" = "1" ]; then
      echo "[memguard] victim gone, exiting. low-water ${low} MB, peak fall rate ${maxrate} MB/s $(date -Is)" | tee -a "$LOG"
      exit 0
    fi
  else
    if [ "${seen:-0}" != 1 ]; then
      echo "[memguard] watching pid $pid ($PAT)" | tee -a "$LOG"
    fi
    seen=1
    [ "$avail" -lt "$low" ] && low=$avail

    # SLOPE. Rate over one poll interval, in MB/s, positive = falling.
    if [ -n "$prev" ]; then
      rate=$(( (prev - avail) * 1000 / POLL_MS ))
      [ "$rate" -gt "$maxrate" ] && maxrate=$rate
      if [ "$avail" -lt "$DANGER_MB" ] && [ "$rate" -gt "$RATE_MB_S" ]; then srate=$((srate+1)); else srate=0; fi
      if [ "$srate" -ge "$SLOPE_N" ]; then
        kill_victim "MemAvailable ${avail} MB falling ${rate} MB/s (< ${DANGER_MB} MB and faster than ${RATE_MB_S} MB/s): ~$(( avail * 10 / rate )) tenths of a second from zero" "$pid"
      fi
    fi
    prev=$avail

    # LEVEL.
    if [ "$avail" -lt "$FLOOR_MB" ]; then bad=$((bad+1)); else bad=0; fi
    [ "$bad" -ge "$BREACHES" ] && kill_victim "MemAvailable ${avail} MB < floor ${FLOOR_MB} MB x${bad}" "$pid"
  fi
  napp
done
