#!/usr/bin/env bash
# gate_memguard.sh — prove the guard fires on a runaway ramp and does NOT fire on a healthy one.
#
# The guard that was shipped could not fire in time (6 s to decide against a 0.52 s window) and
# nobody knew, because it had never been driven. This drives the real polling loop -- same file,
# same triggers, same timing -- against a synthetic MemAvailable trace via $MEMINFO.
#
# Both arms start at 8000 MB: above DANGER_MB the guard is trivially inert, so the interesting
# region is the approach. Step is 342 MB per 0.2 s poll = the measured 1710 MB/s load ramp.
#   HEALTHY  8000 -> 2400 then flat   (2400 is the measured 2389 MB low-water, rounded up)
#   RUNAWAY  8000 -> 0 without stopping
set -u
cd "$(dirname "$0")/.."
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
fail=0

# The kill level is the whole point: a guard that kills at MemAvailable 1502 MB saved the box, one
# that kills at -8000 MB fired into a corpse. So arms assert the LEVEL, not just the fact.
run_arm(){
  local name="$1" floor_stop="$2" expect="$3" minlevel="${4:-}" ; shift $(( $# > 3 ? 4 : 3 ))
  local mi="$TMP/meminfo.$name" glog="$TMP/guard.$name.log"
  echo "MemAvailable: $((8000*1024)) kB" > "$mi"

  # the victim. `comm` is `sleep`, so PAT=sleep selects it and nothing else.
  setsid sleep 300 & local vic=$!
  sleep 0.3

  # the ramp writer: atomic rename each step so the guard never reads a torn file.
  ( av=8000
    while [ "$av" -gt "$floor_stop" ]; do
      av=$((av-342)); [ "$av" -lt "$floor_stop" ] && av=$floor_stop
      echo "MemAvailable: $((av*1024)) kB" > "$mi.tmp"; mv "$mi.tmp" "$mi"
      sleep 0.2
    done
    sleep 3 ) &
  local writer=$!

  MEMINFO="$mi" PAT=sleep LOG="$glog" "$@" bash scripts/memguard.sh > /dev/null 2>&1 &
  local guard=$!
  wait $writer 2>/dev/null
  sleep 0.5

  local alive=no; kill -0 "$vic" 2>/dev/null && alive=yes
  kill -9 "$vic" "$guard" 2>/dev/null; wait 2>/dev/null

  local got=killed; [ "$alive" = yes ] && got=survived
  if ! grep -q "watching pid" "$glog" 2>/dev/null; then
    echo "  FAIL  $name: the guard never adopted a victim, so this arm proves nothing"; fail=1; return
  fi
  local lvl; lvl=$(grep -ho "MemAvailable -\?[0-9]* MB" "$glog" 2>/dev/null | head -1 | awk '{print $2}')
  if [ "$got" = "$expect" ] && { [ -z "$minlevel" ] || { [ -n "$lvl" ] && [ "$lvl" -ge "$minlevel" ]; }; }; then
    echo "  PASS  $name: victim $got (expected $expect)${lvl:+, killed at MemAvailable ${lvl} MB${minlevel:+ >= ${minlevel}}}"
    grep -h "KILLING" "$glog" 2>/dev/null | sed 's/^/        /'
  elif [ "$got" = "$expect" ]; then
    echo "  FAIL  $name: victim $got, but killed at MemAvailable ${lvl:-?} MB -- needed >= ${minlevel} (fired into a corpse)"; fail=1
    grep -h "KILLING" "$glog" 2>/dev/null | sed 's/^/        /'
  else
    echo "  FAIL  $name: victim $got, expected $expect"; sed 's/^/        /' "$glog" 2>/dev/null; fail=1
  fi
}

echo "gate_memguard: driving scripts/memguard.sh against synthetic MemAvailable traces"
run_arm healthy 2400 survived
# The trace runs PAST zero, so a guard that decides too slowly kills at a negative level -- which is
# the box already dead. Shipped settings must kill while MemAvailable is still positive.
run_arm runaway -20000 killed 1

# CONTROL: the settings that were shipped until 2026-08-20 (POLL_S=2, BREACHES=3, no slope test),
# on the identical trace. It is not a close call -- 6 s of decision against a 1710 MB/s fall.
echo "  -- control: the pre-2026-08-20 settings on the same trace --"
run_arm oldsettings -20000 killed "" env POLL_S=2 BREACHES=3 DANGER_MB=0

[ "$fail" = 0 ] && { echo "gate_memguard: PASS"; exit 0; } || { echo "gate_memguard: FAIL"; exit 1; }
