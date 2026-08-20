#!/usr/bin/env bash
# oomsentry.sh — the always-on backstop. memguard is armed BY run_model.sh; this is armed by nothing
# and watches always, so a model started outside the sanctioned path is still covered.
#
# It exists because the kernel cannot do this job on this box (measured 2026-08-20): a touched
# 2048 MiB cudaMalloc charges 42 MiB to the cgroup, so Tegra unified memory is invisible to the OOM
# killer, unreclaimable, and unbounded by memory.max -- and this kernel has no PSI, so systemd-oomd
# cannot run either. See scripts/memguard.sh for the full accounting.
#
# IT CAN ONLY EVER KILL A NAMED MODEL BINARY. The victim must match `comm` exactly against ALLOW
# below. If memory is disappearing and nothing in ALLOW is running, it logs and does NOTHING --
# because the alternative, picking the biggest RSS, would pick a shell or an editor, and RSS is
# exactly the number we just proved does not reflect who is holding the memory. A guard whose whole
# job is `kill -9` gets the narrowest possible license.
#
# Floors sit BELOW memguard's so the armed guard always acts first and this only catches what it
# missed. Same slope-vs-level reasoning as memguard.
set -u
ALLOW="${ALLOW:-dsv4-server decode decode_probe decode_prechange decode_pre1p4_ref forward load_device}"
FLOOR_MB="${FLOOR_MB:-1200}"
DANGER_MB="${DANGER_MB:-1800}"
RATE_MB_S="${RATE_MB_S:-400}"
BREACHES="${BREACHES:-3}"
SLOPE_N="${SLOPE_N:-2}"
POLL_S="${POLL_S:-0.2}"
MEMINFO="${MEMINFO:-/proc/meminfo}"
LOG="${LOG:-evidence/oomsentry.log}"
cd "$(dirname "$0")/.."
mkdir -p "$(dirname "$LOG")"
say(){ echo "[oomsentry $(date -Is)] $*" >> "$LOG"; }
say "started: floor=${FLOOR_MB}MB slope=<-${RATE_MB_S}MB/s below ${DANGER_MB}MB poll=${POLL_S}s allow='${ALLOW}'"

bad=0; srate=0; prev=""; warned=0
while true; do
  avail=$(awk '/MemAvailable/{print int($2/1024)}' "$MEMINFO" 2>/dev/null) || avail=""
  if [ -z "$avail" ]; then sleep "$POLL_S"; continue; fi

  trip=""
  if [ -n "$prev" ]; then
    rate=$(awk -v a="$prev" -v b="$avail" -v p="$POLL_S" 'BEGIN{printf "%d",(a-b)/p}')
    if [ "$avail" -lt "$DANGER_MB" ] && [ "$rate" -gt "$RATE_MB_S" ]; then srate=$((srate+1)); else srate=0; fi
    [ "$srate" -ge "$SLOPE_N" ] && trip="MemAvailable ${avail} MB falling ${rate} MB/s"
  fi
  prev=$avail
  if [ "$avail" -lt "$FLOOR_MB" ]; then bad=$((bad+1)); else bad=0; fi
  [ "$bad" -ge "$BREACHES" ] && trip="MemAvailable ${avail} MB < floor ${FLOOR_MB} MB x${bad}"

  if [ -n "$trip" ]; then
    pid=""; who=""
    for c in $ALLOW; do
      pid=$(ps -eo pid=,comm= | awk -v p="$c" '$2==p {print $1; exit}')
      [ -n "$pid" ] && { who=$c; break; }
    done
    if [ -n "$pid" ]; then
      say "!!! $trip — KILLING $who pid $pid"
      kill -9 "$pid" 2>/dev/null
      sleep 5
      say "killed. MemAvailable now $(awk '/MemAvailable/{print int($2/1024)}' "$MEMINFO") MB"
      bad=0; srate=0; prev=""
    elif [ "$warned" = 0 ]; then
      say "WARN $trip — but nothing in ALLOW is running, so there is no sanctioned victim. Doing nothing."
      warned=1
    fi
  else
    warned=0
  fi
  sleep "$POLL_S"
done
