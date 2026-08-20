#!/usr/bin/env bash
# clocks_ab_run.sh — DECODE_LADDER item 3.1, end to end.
#
#   nohup setsid bash scripts/clocks_ab_run.sh > evidence/decode_loop/clocks_ab.log 2>&1 </dev/null &
#
# WHY A SCRIPT AND NOT A SEQUENCE OF COMMANDS. Four checkpoint loads that need no steering once
# started, so CLAUDE.md says detach before starting. Named in detach_audit.sh's PATTERNS in the same
# commit, because a stage the audit cannot see reports as "all detached" by never being looked at.
#
# THE LADDER ITEM IS TWO LEVERS, NOT ONE, AND THE ITEM TEXT HIDES THAT. 3.1 reads
# "`jetson_clocks` — GPU 1386 -> 1575 MHz, EMC 2750 -> 4266", but on this box `jetson_clocks` alone
# CANNOT reach 1575: at the current power model (`nvpmodel` ID 1, "120W") /etc/nvpmodel.conf pins
# `GPU MAX_FREQ 1386000000`, and `jetson_clocks` only raises a clock to its governor's current
# ceiling. 1575 MHz is unlocked by `nvpmodel -m 0` (MAXN, `GPU MAX_FREQ -1`) — verified live before
# this run: mode 0 reports `gpu-gpc-0 MaxFreq=1575000000`, mode 1 reports 1386000000, and the switch
# needs no reboot. So the item is measured as three arms, each ONE change from the last:
#
#   A  120W, governed   (the state every measurement on this ladder was taken in)
#   B  120W, pinned     (`jetson_clocks`: gpc 315 -> 1386, EMC 2750 -> 4266)   <- the in-repo +3.0-6.4 %
#   C  MAXN, pinned     (`nvpmodel -m 0` then `jetson_clocks`: gpc ceiling 1386 -> 1575)
#
# ARM A' IS THE DRIFT CONTROL AND IT IS THE REASON THIS RUN IS FOUR LOADS AND NOT THREE. Arms B and
# C run hotter by construction, and every arm here is a separate checkpoint load, where
# measurement-and-traps §19 puts the spread at 5.7 % against 0.6 % within a load. A monotone drift
# over an hour — thermal, page cache, anything — would look exactly like a clock gain. A' repeats
# arm A's machine state LAST, after the box has been at pinned MAXN for two loads. If A' comes back
# to A, the ordering did not manufacture the result; if it does not, the whole comparison is void
# and says so.
#
# ORDER ALSO MAKES THE DRIFT CONSERVATIVE. Ungoverned-first means any warming over the session
# penalises the pinned arms, so B and C are measured against their best-case rival.
#
# THE PROTOCOL IS FROZEN AND NOT NEGOTIABLE HERE (F96, and 2.2 used exactly this): the 9-prompt
# suite from protocol/suite_prompts.txt, block 6, one draft pass, adaptK 1.50, NGEN0 200, the same
# gate prompt, caches dropped before each arm, the LIVE checkpoint from config/live_ckpt. That makes
# these numbers comparable to 2.2's table row for row, and it makes tools/head_ab.py the analyser —
# it is paired per prompt, it reports tau, and it refuses to compare arms whose LOSSLESS gate did
# not pass.
#
# NOTHING HERE TOUCHES A KERNEL, SO tau IS A NEGATIVE CONTROL AND NOT JUST A REQUIREMENT. Acceptance
# is an exact draft/target token comparison; clocks cannot change which token wins an argmax. If tau
# moves between arms, the arms are not measuring what this script claims and the run is void.
#
# THE BOX IS LEFT AS IT WAS FOUND. A trap restores nvpmodel 1 and the stored governed clock state on
# every exit path, including a kill. Deploying a clock policy is a separate, deliberate edit to the
# launchers — not a side effect of measuring one.
set -u
cd "$(dirname "$0")/.."
export PATH=/usr/local/cuda/bin:$PATH

EV=evidence/decode_loop
CKPT="${CKPT:-$(cat config/live_ckpt)}"
GATE_PROMPT="0,671,6102,294,8760,344"
NGEN0="${NGEN0:-200}"
ARM_TIMEOUT="${ARM_TIMEOUT:-4200}"        # 70 min per arm; a load alone is ~10
STORE=/tmp/clocks_3p1_before.conf
PHASE_FROM="${PHASE_FROM:-0}"

mkdir -p "$EV"
say(){ echo "[3.1] $(date -Is) $*"; }

# ---- the arms differ ONLY in machine state; the suite is byte-identical across all four ----------
SUITE="$(cat protocol/suite_prompts.txt)"
SW="6:1:1.5:0,6:1:1.5:1,6:1:1.5:2,6:1:1.5:3,6:1:1.5:4,6:1:1.5:5,6:1:1.5:6,6:1:1.5:7,6:1:1.5:8"
say "checkpoint: $CKPT"
say "suite sha256: $(printf '%s' "$SUITE" | sha256sum | cut -c1-16)  blksweep: $SW"

# ================= machine state ================================================================
clock_state(){   # $1 = label
  echo "### clock state [$1] $(date -Is)"
  sudo -n nvpmodel -q 2>&1 | tr '\n' ' '; echo
  sudo -n jetson_clocks --show 2>&1 | grep -E 'gpu-gpc-0|EMC|gpu-nvd-0'
  nvidia-smi --query-gpu=clocks.gr,temperature.gpu,power.draw --format=csv,noheader 2>&1 | head -1
}

restore_box(){
  say "restoring machine state: nvpmodel 1 + stored governed clocks"
  timeout 120 sudo -n nvpmodel -m 1 >/dev/null 2>&1 || true
  sleep 3
  [ -f "$STORE" ] && { timeout 120 sudo -n jetson_clocks --restore "$STORE" >/dev/null 2>&1 || true; }
  sleep 2
  clock_state "restored"
}
trap 'restore_box' EXIT INT TERM

# A SAMPLER, BECAUSE "THE CLOCK WAS PINNED" IS A CLAIM ABOUT THE WHOLE RUN AND NOT ABOUT THE INSTANT
# BEFORE IT. jetson_clocks writes a floor; a thermal cap can still pull the clock down mid-decode,
# and that is precisely the failure mode arm C is most exposed to. Sample throughout every arm so
# the write-up can say what the clock ACTUALLY was while the tokens were being produced.
sampler_start(){
  ( while :; do
      printf '%s %s\n' "$(date -Is)" \
        "$(nvidia-smi --query-gpu=clocks.gr,temperature.gpu,power.draw --format=csv,noheader 2>/dev/null | head -1)"
      sleep 10
    done ) > "$EV/clocks_3p1_$1.samples" 2>/dev/null &
  SAMPLER_PID=$!
}
sampler_stop(){ [ -n "${SAMPLER_PID:-}" ] && kill "$SAMPLER_PID" 2>/dev/null; SAMPLER_PID=""; }

# `sort -n` and not awk's asort(): asort is a gawk extension and this box's /usr/bin/awk is mawk,
# where it is a silent parse error inside a pipeline that would otherwise look like "no samples".
clock_summary(){  # $1 = tag
  local f="$EV/clocks_3p1_$1.samples"
  [ -s "$f" ] || { echo "[3.1] arm $1: no clock samples"; return 0; }
  awk '{gsub(/,/,""); print $2}' "$f" | sort -n | awk -v t="$1" '
    {v[NR]=$1}
    END{ if(NR) printf "[3.1] arm %-7s gpc clock over %3d samples: min %6.0f  median %6.0f  max %6.0f MHz\n",
                       t, NR, v[1], v[int((NR+1)/2)], v[NR] }'
}

# ================= one arm ======================================================================
# NEVER `pgrep -f` TO WAIT. Claude Code's bash wrapper — and this script's own command line — can
# contain the pattern, so a -f match can match the waiter itself and never return (LOOP_LOG,
# iteration 8). Match the process NAME via `ps -eo comm=`, and give every wait a timeout.
decode_running(){ ps -eo comm= | grep -qx decode; }

run_arm(){       # $1 = tag, $2 = human label
  # SPLIT, NOT ONE `local`. Bash expands ALL of a builtin's arguments before running it, so
  # `local tag="$1" log="...$tag..."` expands `$tag` while it is still unset and `set -u` kills the
  # script. Caught by the first launch of this script doing exactly that.
  local tag="$1"
  local label="$2"
  local log="$EV/clocks_3p1_$tag.log"
  local t0 waited
  say "=== ARM $tag ($label) ==="
  clock_state "arm $tag before"
  if decode_running; then say "FATAL: a decode is already running before arm $tag"; return 1; fi
  sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true
  sleep 2
  : > "$log"
  DSV4_PROMPTS="$SUITE" DSV4_BLKSWEEP="$SW" \
    bash scripts/run_model.sh "$log" ./build/decode "$CKPT" "$GATE_PROMPT" 8 "" "$NGEN0" \
    || { say "FATAL: run_model refused arm $tag"; return 1; }
  sampler_start "$tag"
  t0=$(date +%s)
  # Wait for the process to APPEAR first: run_model returns as soon as it has forked, and a wait
  # that starts by checking "is it gone?" would answer yes before it ever started.
  for _ in $(seq 1 60); do decode_running && break; sleep 2; done
  while decode_running; do
    waited=$(( $(date +%s) - t0 ))
    if [ "$waited" -gt "$ARM_TIMEOUT" ]; then
      sampler_stop; say "FATAL: arm $tag still running after ${waited}s"; return 1
    fi
    sleep 15
  done
  sleep 15                    # run_model's watcher holds the single-tenancy lock ~5 s past the exit
  sampler_stop
  clock_state "arm $tag after"
  say "arm $tag finished in $(( $(date +%s) - t0 ))s"
  # A STAGE THAT "COMPLETES" AGAINST A DEAD ENGINE IS WORSE THAN ONE THAT DIES (CLAUDE.md). An arm
  # with no LOSSLESS gate is not a slow arm, it is not an arm at all, and head_ab.py voids the
  # comparison anyway — fail here rather than three loads later.
  if [ ! -s "$log" ]; then say "FATAL: arm $tag produced no log"; return 1; fi
  grep -E 'GATE|WARM decode|LOSSLESS|tok/verify' "$log" | tail -8
  grep -qE 'LOSSLESS GATE: .* -> PASS' "$log" \
    || { say "FATAL: arm $tag has no passing LOSSLESS gate"; tail -20 "$log"; return 1; }
  if ! grep -qE 'first decoded token argmax = 11111 .*GATE PASS' "$log"; then
    say "FATAL: arm $tag failed the first-token gate"; return 1; fi
  say "arm $tag OK"
  clock_summary "$tag"
  return 0
}

# ================= PHASE 0 — record what we are starting from ===================================
if [ "$PHASE_FROM" -le 0 ]; then
say "=== PHASE 0: baseline machine state, stored so it can be put back ==="
# ARM A IS ONLY A BASELINE IF THE BOX IS ACTUALLY GOVERNED WHEN IT RUNS. `jetson_clocks` pins by
# writing MinFreq = MaxFreq, so MinFreq == MaxFreq on gpu-gpc-0 is the exact discriminator for
# "already pinned" — and a pinned arm A would silently measure arm B twice and report the delta as
# zero. Refuse rather than produce a null result that looks like a finding.
GPC_LINE="$(sudo -n jetson_clocks --show 2>/dev/null | grep 'gpu-gpc-0')"
GPC_MIN="$(printf '%s' "$GPC_LINE" | sed -n 's/.*MinFreq=\([0-9]*\).*/\1/p')"
GPC_MAX="$(printf '%s' "$GPC_LINE" | sed -n 's/.*MaxFreq=\([0-9]*\).*/\1/p')"
if [ -z "$GPC_MIN" ] || [ -z "$GPC_MAX" ]; then
  say "FATAL: could not read the gpu-gpc-0 clock state; refusing to change clocks"; exit 1; fi
if [ "$GPC_MIN" = "$GPC_MAX" ]; then
  say "FATAL: gpu-gpc-0 is ALREADY pinned (Min=Max=$GPC_MIN). Arm A would not be a baseline."
  say "       Unpin first (sudo jetson_clocks --restore <conf>) and re-run."; exit 1; fi
say "gpu-gpc-0 is governed (Min=$GPC_MIN Max=$GPC_MAX) — arm A is a real baseline"
# --store REFUSES to overwrite and asks on stdin, which under nohup is an instant failure. A stale
# file here is from an earlier attempt of THIS script, and the check above has just proven the
# current state is the governed one, so re-storing is right and reusing a stale file is not.
rm -f "$STORE"
timeout 120 sudo -n jetson_clocks --store "$STORE" >/dev/null 2>&1
[ -s "$STORE" ] || { say "FATAL: could not store the current clock state; refusing to change clocks"; exit 1; }
say "stored -> $STORE  ($(wc -l < "$STORE") entries)"
clock_state "baseline" | tee "$EV/clocks_3p1_state.txt"
fi

# ================= PHASE 1 — ARM A: governed, 120W (the status quo) =============================
if [ "$PHASE_FROM" -le 1 ]; then
run_arm A "120W governed — the state every ladder measurement so far was taken in" || exit 1
fi

# ================= PHASE 2 — ARM B: pinned, 120W ================================================
if [ "$PHASE_FROM" -le 2 ]; then
say "=== pinning clocks: jetson_clocks at nvpmodel 1 ==="
timeout 300 sudo -n jetson_clocks 2>&1 | head -5
sleep 5
clock_state "pinned 120W" | tee -a "$EV/clocks_3p1_state.txt"
run_arm B "120W pinned — jetson_clocks, gpc floor 1386, EMC 4266" || exit 1
fi

# ================= PHASE 3 — ARM C: pinned, MAXN ================================================
# nvpmodel resets the clock caps it manages, so jetson_clocks MUST be re-applied after the mode
# switch. Doing it in the other order pins to the OLD ceiling and measures arm B twice under a
# different name — the same trap 2.2's staged-checkpoint gate was built to catch.
if [ "$PHASE_FROM" -le 3 ]; then
say "=== switching to MAXN and re-pinning ==="
timeout 300 sudo -n nvpmodel -m 0 2>&1 | head -5
sleep 5
timeout 300 sudo -n jetson_clocks 2>&1 | head -5
sleep 5
clock_state "pinned MAXN" | tee -a "$EV/clocks_3p1_state.txt"
if ! sudo -n jetson_clocks --show 2>/dev/null | grep -q 'gpu-gpc-0 .*MaxFreq=1575000000'; then
  say "WARNING: MAXN did not raise the gpc ceiling to 1575 MHz — arm C may be a duplicate of arm B"
fi
run_arm C "MAXN pinned — gpc ceiling 1575, EMC 4266" || exit 1
fi

# ================= PHASE 4 — ARM A': governed, 120W again (the drift control) ===================
if [ "$PHASE_FROM" -le 4 ]; then
say "=== restoring 120W + governed clocks for the drift control ==="
restore_box
sleep 30                       # let the governor settle back down before measuring it
run_arm Aprime "120W governed AGAIN, last — if this != arm A the ordering made the result" || exit 1
fi

# ================= PHASE 5 — the ratchet ========================================================
say "=== PHASE 5: paired reports (tools/head_ab.py — paired per prompt, tau reported) ==="
{
  echo "###############  3.1 clocks A/B — $(date -Is)"
  echo "checkpoint: $CKPT"
  echo
  echo "=========== B vs A : what jetson_clocks alone is worth at 120W ==========="
  python3 tools/head_ab.py --a "$EV/clocks_3p1_A.log" --b "$EV/clocks_3p1_B.log" \
          --label-a "A-governed" --label-b "B-pinned120W"
  echo
  echo "=========== C vs B : what MAXN adds on top of pinning ==========="
  python3 tools/head_ab.py --a "$EV/clocks_3p1_B.log" --b "$EV/clocks_3p1_C.log" \
          --label-a "B-pinned120W" --label-b "C-pinnedMAXN"
  echo
  echo "=========== C vs A : the whole item, end to end ==========="
  python3 tools/head_ab.py --a "$EV/clocks_3p1_A.log" --b "$EV/clocks_3p1_C.log" \
          --label-a "A-governed" --label-b "C-pinnedMAXN"
  echo
  echo "=========== A' vs A : THE DRIFT CONTROL. Same machine state, one hour later. ==========="
  echo "=========== If this is not ~0, everything above is drift and not clocks.     ==========="
  python3 tools/head_ab.py --a "$EV/clocks_3p1_A.log" --b "$EV/clocks_3p1_Aprime.log" \
          --label-a "A-governed" --label-b "Aprime-governed"
  echo
  echo "=========== gpc clock actually observed during each arm ==========="
  for t in A B C Aprime; do clock_summary "$t"; done
  echo
  echo "=========== GPU temperature over each arm (thermal drift is the alternative explanation) ==="
  for t in A B C Aprime; do
    f="$EV/clocks_3p1_$t.samples"
    [ -s "$f" ] || { echo "$t: no samples"; continue; }
    awk -v t="$t" '{gsub(/,/,""); c[NR]=$4+0}
      END{ if(NR) printf "%-8s first %3.0f C  last %3.0f C\n", t, c[1], c[NR] }' "$f"
  done
} | tee "$EV/clocks_3p1_ab.txt"
say "done -> $EV/clocks_3p1_ab.txt"
