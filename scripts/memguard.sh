#!/usr/bin/env bash
# memguard.sh — kill the server before the kernel kills the box.
#
# This box does not OOM gracefully. Twice on 2026-08-12 a server launched with too large a --seqmax
# took the whole machine down mid-load (reboots at 20:02 and 20:11, `last -x reboot`); there is no
# oom-kill line in dmesg either time, so the Tegra unified-memory path brings the system down rather
# than sacrificing the offending process. `vm.panic_on_oom` is already 0 — this is not something a
# sysctl fixes. The only defence is to notice the approach and kill the process ourselves, with
# enough margin left that the kernel never has to make the decision.
#
# The floor cannot be generous, and finding that out cost a run. A healthy seqmax=4096 server
# reports `mem 119.9/122.8 GiB` and leaves ~2.9 GiB of MemAvailable — so a 3500 MB floor kills a
# server that has already loaded, printed `ready`, and started listening. It did exactly that on the
# first attempt. **This config's normal operating point is under 3 GiB available**; anything that
# treats 3 GiB as an emergency is measuring the design, not a fault.
#
# What the guard is actually for is the *load ramp*, which is where both crashes happened: the
# footprint climbs ~100 GiB in about 60 s and then goes flat, because decode allocates out of the
# pre-sized arena rather than growing. So the shape to catch is a fast monotonic run at the ceiling,
# not a low-but-stable reading. Hence a floor well under the steady state plus a consecutive-breach
# requirement: a runaway crosses 1500 MB and keeps going within one or two polls, while the healthy
# plateau never crosses it at all.
#
#   FLOOR_MB   kill below this MemAvailable            (default 1500)
#   BREACHES   consecutive samples under floor to kill (default 3)
#   POLL_S     seconds between samples                 (default 2)
#   PAT        pgrep -f pattern for the victim         (default dsv4-server)
#
#   ./scripts/memguard.sh &            # runs until the server exits
set -u
FLOOR_MB="${FLOOR_MB:-1500}"
BREACHES="${BREACHES:-3}"
POLL_S="${POLL_S:-2}"
PAT="${PAT:-dsv4-server}"
LOG="${LOG:-evidence/memguard.log}"

cd "$(dirname "$0")/.."
echo "[memguard] floor=${FLOOR_MB} MB x${BREACHES} poll=${POLL_S}s pat=${PAT} started $(date -Is)" | tee -a "$LOG"

low=999999999
bad=0
while true; do
  avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  pid=$(pgrep -f "$PAT" | head -1)

  if [ -z "$pid" ]; then
    # Only exit once the server has been seen and then gone; before it appears, keep waiting.
    if [ "${seen:-0}" = "1" ]; then
      echo "[memguard] victim gone, exiting. low-water was ${low} MB $(date -Is)" | tee -a "$LOG"
      exit 0
    fi
  else
    seen=1
    [ "$avail" -lt "$low" ] && low=$avail
    if [ "$avail" -lt "$FLOOR_MB" ]; then bad=$((bad+1)); else bad=0; fi
    if [ "$bad" -ge "$BREACHES" ]; then
      echo "[memguard] !!! MemAvailable ${avail} MB < floor ${FLOOR_MB} MB x${bad} — KILLING pid $pid $(date -Is)" | tee -a "$LOG"
      kill -9 "$pid" 2>/dev/null
      sleep 5
      echo "[memguard] killed. MemAvailable now $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) MB" | tee -a "$LOG"
      exit 1
    fi
  fi
  sleep "$POLL_S"
done
