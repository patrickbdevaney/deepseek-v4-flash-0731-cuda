#!/usr/bin/env bash
# clocks_emc_probe.sh — DECODE_LADDER 3.1, phase 6: the arm the first run was missing.
#
#   nohup setsid bash scripts/clocks_emc_probe.sh > evidence/decode_loop/clocks_emc.log 2>&1 </dev/null &
#
# WHY THERE IS A SECOND RUN AT ALL. clocks_ab_run.sh pre-registered its own kill switch — "A' repeats
# arm A's machine state LAST; if A' does not come back to A the whole comparison is void" — and A'
# came back **+8.19 %** above A on the suite mean, which is LARGER than the +5.70 % that pinning was
# supposed to be worth. By the rule that run wrote before it saw its data, the suite comparison is
# void. This run is not a re-run for a better number; it is a run for the variable the first one
# never recorded.
#
# THE VARIABLE. The first run sampled `nvidia-smi clocks.gr` and nothing else, and it shows the thing
# the item was built on is not true: in BOTH governed arms the GPU core rail sat at **1386 MHz — the
# governor's ceiling — for every sample in the compute window** (the 315 MHz samples are the ~90 s
# checkpoint load, when the GPU is idle and the CPU is reading 100 GiB). So `jetson_clocks` does not
# change the GPU core clock during decode; the governor has already taken it to the same 1386 the pin
# would write. And MAXN's 1575 MHz measured +2.11 % on the suite against a 3.5 % spread, which is
# what a bandwidth-bound engine should say about a core-clock raise.
#
# That leaves EMC — 2750 governed vs 4266 pinned at IDLE — as the only rail that can carry this item,
# and **no run in this repo has ever sampled EMC while a decode was in flight**. `bwmgr` is a
# governor too. If it also ramps under load, 3.1 is a null by construction and no A/B will ever show
# it; if it stays at 2750 while the engine is bandwidth-bound, the item is real and the first run's
# noise buried it. One `cat` of a debugfs file per two seconds answers it either way, and it is
# cheaper than any throughput arm.
#
# ABA, NOT AB. The failure mode of the first run was between-load drift, so the governed state is
# measured on BOTH sides of the pinned arm. A pinned arm inside the governed bracket is a null no
# matter which single governed arm you compare it to.
#
# QUIET BOX. The first run's arms were measured while the agent that launched them kept working —
# repo-wide greps, and a ~GB node process exiting mid-arm C, which lands inside the one prompt of 36
# that went backwards (C prompt 4, 22.97 -> 14.37 tok/s, with the core clock pinned at 1575 for every
# sample). The arms here are detached and nothing else may run against them.
set -u
cd "$(dirname "$0")/.."
export PATH=/usr/local/cuda/bin:$PATH

EV=evidence/decode_loop
CKPT="${CKPT:-$(cat config/live_ckpt)}"
GATE_PROMPT="0,671,6102,294,8760,344"
NGEN0="${NGEN0:-200}"
ARM_TIMEOUT="${ARM_TIMEOUT:-4200}"
STORE=/tmp/clocks_3p1_before.conf

mkdir -p "$EV"
say(){ echo "[3.1p6] $(date -Is) $*"; }

SUITE="$(cat protocol/suite_prompts.txt)"
SW="6:1:1.5:0,6:1:1.5:1,6:1:1.5:2,6:1:1.5:3,6:1:1.5:4,6:1:1.5:5,6:1:1.5:6,6:1:1.5:7,6:1:1.5:8"
say "checkpoint: $CKPT"
say "suite sha256: $(printf '%s' "$SUITE" | sha256sum | cut -c1-16)"

restore_box(){
  say "restoring: nvpmodel 1 + stored governed clocks"
  timeout 120 sudo -n nvpmodel -m 1 >/dev/null 2>&1 || true
  sleep 3
  [ -s "$STORE" ] && { timeout 120 sudo -n jetson_clocks --restore "$STORE" >/dev/null 2>&1 || true; }
  sleep 2
}
trap 'restore_box' EXIT INT TERM

# Both rails, 2 s, straight out of debugfs — `jetson_clocks --show` forks a shell script per sample.
sampler_start(){
  ( while :; do
      printf '%s emc=%s gpc=%s\n' "$(date -Is)" \
        "$(sudo -n cat /sys/kernel/debug/bpmp/debug/clk/emc/rate 2>/dev/null)" \
        "$(nvidia-smi --query-gpu=clocks.gr --format=csv,noheader 2>/dev/null | awk '{print $1}')"
      sleep 2
    done ) > "$EV/clocks_emc_$1.samples" 2>/dev/null &
  SAMPLER_PID=$!
}
sampler_stop(){ [ -n "${SAMPLER_PID:-}" ] && kill "$SAMPLER_PID" 2>/dev/null; SAMPLER_PID=""; }

decode_running(){ ps -eo comm= | grep -qx decode; }

run_arm(){
  local tag="$1"
  local label="$2"
  local log="$EV/clocks_emc_$tag.log"
  local t0 waited
  say "=== ARM $tag ($label) ==="
  if decode_running; then say "FATAL: a decode is already running before arm $tag"; return 1; fi
  sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true
  sleep 2
  : > "$log"
  DSV4_PIN_CLOCKS=0 DSV4_PROMPTS="$SUITE" DSV4_BLKSWEEP="$SW" \
    bash scripts/run_model.sh "$log" ./build/decode "$CKPT" "$GATE_PROMPT" 8 "" "$NGEN0" \
    || { say "FATAL: run_model refused arm $tag"; return 1; }
  sampler_start "$tag"
  t0=$(date +%s)
  for _ in $(seq 1 60); do decode_running && break; sleep 2; done
  while decode_running; do
    waited=$(( $(date +%s) - t0 ))
    if [ "$waited" -gt "$ARM_TIMEOUT" ]; then sampler_stop; say "FATAL: arm $tag hung"; return 1; fi
    sleep 15
  done
  sleep 15
  sampler_stop
  say "arm $tag finished in $(( $(date +%s) - t0 ))s"
  [ -s "$log" ] || { say "FATAL: arm $tag produced no log"; return 1; }
  grep -qE 'LOSSLESS GATE: .* -> PASS' "$log" || { say "FATAL: arm $tag has no passing LOSSLESS gate"; return 1; }
  grep -qE 'first decoded token argmax = 11111 .*GATE PASS' "$log" || { say "FATAL: arm $tag failed the first-token gate"; return 1; }
  say "arm $tag OK"
  return 0
}

# THE COMPUTE WINDOW IS THE ONLY WINDOW THAT COUNTS. A rail's median over the whole arm is dominated
# by the ~90 s idle load, which is how "median 315 MHz" got printed for an arm that decoded entirely
# at 1386. Split on GPU busy (gpc > 315) and report the rails separately for the two phases.
emc_summary(){
  local f="$EV/clocks_emc_$1.samples"
  [ -s "$f" ] || { echo "arm $1: no samples"; return 0; }
  awk -v t="$1" '
    { split($2,a,"="); split($3,b,"="); emc=a[2]/1e6; gpc=b[2]+0
      if (gpc > 315) { n++; e[n]=emc; g[n]=gpc; if(emc>emx||n==1)emx=emc; if(emc<emn||n==1)emn=emc; es+=emc; gs+=gpc }
      else { m++; ie+=emc } }
    END{ if(n) printf "arm %-3s COMPUTE (%3d samples): EMC mean %6.0f  min %6.0f  max %6.0f MHz | gpc mean %6.0f MHz\n", t,n,es/n,emn,emx,gs/n
         if(m) printf "arm %-3s idle/load (%3d samples): EMC mean %6.0f MHz\n", t,m,ie/m }' "$f"
}

# ---------------- ARM A2: governed ---------------------------------------------------------------
GPC_LINE="$(sudo -n jetson_clocks --show 2>/dev/null | grep 'gpu-gpc-0')"
GPC_MIN="$(printf '%s' "$GPC_LINE" | sed -n 's/.*MinFreq=\([0-9]*\).*/\1/p')"
GPC_MAX="$(printf '%s' "$GPC_LINE" | sed -n 's/.*MaxFreq=\([0-9]*\).*/\1/p')"
[ -n "$GPC_MIN" ] && [ -n "$GPC_MAX" ] || { say "FATAL: cannot read gpu-gpc-0"; exit 1; }
[ "$GPC_MIN" != "$GPC_MAX" ] || { say "FATAL: box is already pinned; arm A2 would not be governed"; exit 1; }
say "governed baseline confirmed (Min=$GPC_MIN Max=$GPC_MAX)"
[ -s "$STORE" ] || timeout 120 sudo -n jetson_clocks --store "$STORE" >/dev/null 2>&1
run_arm A2 "governed, quiet box" || exit 1

# ---------------- ARM B2: pinned at 120W ---------------------------------------------------------
say "=== pinning: jetson_clocks at nvpmodel 1 ==="
timeout 300 sudo -n jetson_clocks 2>&1 | head -3
sleep 5
run_arm B2 "pinned 120W — gpc floor 1386, EMC 4266" || exit 1

# ---------------- ARM A2p: governed again, the bracket -------------------------------------------
say "=== restoring governed state for the closing bracket ==="
restore_box
sleep 30
run_arm A2p "governed AGAIN — closes the bracket around B2" || exit 1

# ---------------- report -------------------------------------------------------------------------
say "=== phase 6 report ==="
{
  echo "###############  3.1 phase 6 — EMC under load + ABA bracket — $(date -Is)"
  echo "checkpoint: $CKPT"
  echo
  echo "=========== WHAT THE RAILS ACTUALLY DID WHILE TOKENS WERE BEING PRODUCED ==========="
  for t in A2 B2 A2p; do emc_summary "$t"; done
  echo
  echo "=========== B2 vs A2 : pinned against the governed arm BEFORE it ==========="
  python3 tools/head_ab.py --a "$EV/clocks_emc_A2.log" --b "$EV/clocks_emc_B2.log" \
          --label-a "A2-governed" --label-b "B2-pinned120W"
  echo
  echo "=========== B2 vs A2p : pinned against the governed arm AFTER it ==========="
  python3 tools/head_ab.py --a "$EV/clocks_emc_A2p.log" --b "$EV/clocks_emc_B2.log" \
          --label-a "A2p-governed" --label-b "B2-pinned120W"
  echo
  echo "=========== A2p vs A2 : THE DRIFT CONTROL AGAIN, on a quiet box ==========="
  python3 tools/head_ab.py --a "$EV/clocks_emc_A2.log" --b "$EV/clocks_emc_A2p.log" \
          --label-a "A2-governed" --label-b "A2p-governed"
  echo
  echo "=========== base AR (the window that measures the governor ramp, not the engine) ==========="
  for t in A2 B2 A2p; do printf '%-4s %s\n' "$t" "$(grep -h 'WARM decode' "$EV/clocks_emc_$t.log" | tail -1)"; done
} | tee "$EV/clocks_emc_report.txt"
say "done -> $EV/clocks_emc_report.txt"
