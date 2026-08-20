#!/usr/bin/env bash
# pin_clocks.sh — hold the GPU and memory rails at the ceiling for the whole of a run, and be able
# to put them back. DECODE_LADDER item 3.1.
#
#   bash scripts/pin_clocks.sh pin       # jetson_clocks at the CURRENT power mode (nvpmodel untouched)
#   bash scripts/pin_clocks.sh restore   # back to the governed state stored on the first pin
#   bash scripts/pin_clocks.sh show      # what the rails are right now
#   bash scripts/pin_clocks.sh pin-maxn  # + nvpmodel -m 0. MEASURED WORTHLESS -- see below.
#
# WHAT 3.1 MEASURED, AND WHY THE DEFAULT IS *NOT* MAXN. The ladder item read "`jetson_clocks` — GPU
# 1386 -> 1575 MHz, EMC 2750 -> 4266", and both halves of that turned out to be wrong about the
# workload:
#
#   * **The GPU core rail is already at 1386 during decode without any pin.** Sampled every 2 s
#     across three governed arms: 97.7 % of compute-window samples are at 1386 MHz, the governor's
#     ceiling. The 315 MHz samples everyone assumed were the run are the ~90 s checkpoint load, when
#     the GPU is idle and the CPU is reading 100 GiB.
#   * **EMC is already at 4266 too** — 97.7 % of compute samples, ramping from 2750 within ~2 s of
#     the GPU going busy. 2750 is the *idle* rate, not the rate decode runs at.
#   * **1575 MHz (MAXN) is worth nothing.** Arm C ran at a verified 1575 for 18/18 samples and
#     measured +1.68 +/- 5.83 % paired against the pinned-120W arm — nothing, which is what a
#     bandwidth-bound engine should say about a core-clock raise. `pin-maxn` is kept only so the
#     measurement can be reproduced.
#
# So what pinning actually buys is the **ramp**, not the ceiling: the 2.3 % of the compute window
# spent climbing, worth **+2.0 to +3.0 % on the suite** (+1.99 +/- 0.15 % against the closest-in-time
# governed arm, +2.96 % against a time-interpolated one, 8/8 legs positive, `tau` identical). That is
# below the ladder's 3.5 % run-to-run spread and is NOT claimed as a ladder win.
#
# THE REASON TO PIN ANYWAY IS MEASUREMENT, AND IT IS WORTH MORE THAN THE 2 %. The `WARM decode`
# base-AR window is taken immediately after load, inside the ramp, so it times the governor and not
# the engine: **88.0 / 88.0 / 88.7 / 88.4 ms/tok across four governed loads, 72.7 / 72.9 pinned** —
# a 21 % artefact that reproduces to the tenth of a millisecond and looks exactly like a regression.
# It is the whole of ladder item 2.5's unexplained "17.4 % base-AR fall".
#
# EVERY PATH IS BEST-EFFORT. A launcher must not fail because sudo is unavailable or a Jetson tool
# moved; it must say so and launch anyway. A refused pin costs 2 %, an aborted launch costs the run.
set -u
cd "$(dirname "$0")/.."

# Must survive reboots and /tmp cleaners, or `restore` silently degrades to "whatever jetson_clocks
# last wrote" — which is still pinned.
STORE="${DSV4_CLOCK_STORE:-/var/tmp/dsv4_clocks_governed.conf}"
say(){ echo "[clocks] $*"; }

show(){
  sudo -n nvpmodel -q 2>/dev/null | tr '\n' ' '; echo
  sudo -n jetson_clocks --show 2>/dev/null | grep -E 'gpu-gpc-0|gpu-nvd-0|EMC'
  nvidia-smi --query-gpu=clocks.gr,temperature.gpu,power.draw --format=csv,noheader 2>/dev/null | head -1
}

# "Already pinned" is Min==Max on the graphics rail — the same discriminator clocks_ab_run.sh uses to
# refuse to call a pinned box a baseline.
gpc_min(){ sudo -n jetson_clocks --show 2>/dev/null | sed -n 's/.*gpu-gpc-0.*MinFreq=\([0-9]*\).*/\1/p'; }
gpc_max(){ sudo -n jetson_clocks --show 2>/dev/null | sed -n 's/.*gpu-gpc-0.*MaxFreq=\([0-9]*\).*/\1/p'; }

do_pin(){
  local mn mx
  mn="$(gpc_min)"; mx="$(gpc_max)"
  if [ -z "$mn" ] || [ -z "$mx" ]; then
    say "WARNING: cannot read gpu-gpc-0 (no passwordless sudo?) — leaving the rails alone"; return 0
  fi
  if [ "$mn" = "$mx" ]; then say "already pinned at $(( mn / 1000000 )) MHz"; return 0; fi
  # STORE ONCE, AND ONLY FROM A GOVERNED BOX. Storing an already-pinned state makes `restore` a
  # no-op that reports success — the same shape of bug as an arm A that is secretly arm B.
  # --store also refuses to overwrite and prompts on stdin, which under nohup hangs forever.
  if [ ! -s "$STORE" ]; then
    timeout 120 sudo -n jetson_clocks --store "$STORE" >/dev/null 2>&1 \
      && say "stored the governed state -> $STORE"
  fi
  timeout 300 sudo -n jetson_clocks >/dev/null 2>&1 || { say "WARNING: jetson_clocks failed"; return 0; }
  sleep 2
  mn="$(gpc_min)"; mx="$(gpc_max)"
  if [ -n "$mn" ] && [ "$mn" = "$mx" ]; then say "pinned: gpu-gpc-0 $(( mn / 1000000 )) MHz"
  else say "WARNING: gpu-gpc-0 still governed (Min=$mn Max=$mx) — running UNPINNED"; fi
}

case "${1:-pin}" in
  pin) do_pin ;;

  pin-maxn)
    # nvpmodel FIRST: it resets the caps it manages, so jetson_clocks after the switch pins to the
    # new ceiling. The other order pins to the old one and reports success at 1386.
    timeout 300 sudo -n nvpmodel -m 0 >/dev/null 2>&1 || say "WARNING: nvpmodel -m 0 (MAXN) failed"
    sleep 3
    do_pin
    show
    ;;

  restore)
    timeout 120 sudo -n nvpmodel -m 1 >/dev/null 2>&1 || say "WARNING: nvpmodel -m 1 failed"
    sleep 3
    if [ -s "$STORE" ]; then
      timeout 120 sudo -n jetson_clocks --restore "$STORE" >/dev/null 2>&1 \
        || say "WARNING: jetson_clocks --restore $STORE failed"
    else
      say "WARNING: no stored governed state at $STORE — nvpmodel 1 only; the rails stay pinned"
    fi
    sleep 2
    show
    ;;

  show) show ;;
  *) echo "usage: $0 {pin|pin-maxn|restore|show}" >&2; exit 2 ;;
esac
