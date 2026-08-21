#!/usr/bin/env bash
# genout_within_run.sh -- DECODE_LADDER 1.10. The ladder's PRIMARY invariant, asked in ONE load.
#
#   nohup setsid bash scripts/genout_within_run.sh > evidence/decode_loop/genout_within.log 2>&1 </dev/null &
#
# Named in detach_audit.sh's PATTERNS in the same commit.
#
# 1.0's observation was "two runs of the identical binary with the identical env emit different
# token ids", and 1.9 chased it with a two-process protocol. Two processes means two 100.40 GiB
# loads, and on this box that is BOTH expensive and fragile: 1.10's first two attempts at it each
# lost an arm to the memguard, which fires on the fall RATE as well as the floor and has ~7 GiB of
# headroom to work with. It is also the weaker question -- two loads differ in allocator state, so a
# difference between them is not necessarily about the computation.
#
# Repeat the SAME sweep point inside ONE process instead. Every pair is then the same input on the
# same state, which is what 1.9's `--within` protocol established as the decisive form, and one load
# answers for both. Both arms are the SAME BINARY: `DSV4_HADAMARD_STAGE=0` is the pre-1.10 aliased
# `hadamard`, unset is the staged one, so the before-arm is the real prior code and not a memory of
# it. The OFF arm is expected to DIVERGE -- if it does not, this protocol cannot see the defect and
# a clean ON arm would mean nothing.
set -u
cd "$(dirname "$0")/.."
CKPT="${CKPT:-/home/patrickd/models/DeepSeek-V4-Flash-0731-REAP}"
EV=evidence/decode_loop
PROMPTS="${PROMPTS:-$EV/mainkv_prompts.txt}"
# prompt index 2 = 1,023 positions, FOUR repeats -> C(4,2) = 6 within-process pairs.
#
# WHY NOT ALSO 3,071, WHICH IS THE OTHER POINT 1.0 SAW DIVERGE. `compressed_attn_forward` allocates
# its scratch with raw cudaMalloc, per layer, sized in `s`: at s = 3,071 that is ~400 MB each for `q`
# and `o` alone. With 100.40 GiB of weights already resident in a 122.8 GiB pool the burst drives
# MemAvailable to ~1 GB and the memguard's SLOPE rule fires -- three separate attempts died at
# exactly the same place, immediately after `structs built`. That is a pre-existing property of the
# prefill allocator, not of anything 1.10 changed, and the right response is to ask the question at a
# length that fits rather than to disarm the guard. 1,023 is above 1.9's threshold by a factor of six
# (255 compressor rows = 128 blocks, where the kernel-level rate is 200/200), and it is one of the
# two points 1.0 originally saw diverge -- at generated token 43.
SWEEP="${SWEEP:-6:1:1.50:2,6:1:1.50:2,6:1:1.50:2,6:1:1.50:2}"
NGEN="${NGEN:-256}"
TRIES="${TRIES:-3}"
OUT=$EV/genout_within_verdict.txt
say(){ echo "[1.10-within] $(date -Is) $*"; }

[ -s "$PROMPTS" ] || { say "REFUSING: $PROMPTS missing."; exit 1; }
for f in kernels/indexer.cu include/indexer.h; do
  [ "$f" -nt build/decode ] && { say "REFUSING: build/decode is older than $f."; exit 1; }
done
: > "$OUT"
{ echo "genout_within -- ladder 1.10, $(date -Is)"
  echo "binary: build/decode $(date -Is -r build/decode)"
  echo "sweep:  $SWEEP  NGEN=$NGEN"; echo; } >> "$OUT"

run_arm(){   # $1 = tag, $2 = extra env ("" or "DSV4_HADAMARD_STAGE=0")
  local tag="$1" extra="$2" try
  for try in $(seq 1 "$TRIES"); do
    local LOG=$EV/gw_$tag.log GO=$EV/gw_genout_$tag.txt
    rm -f "$GO"                                        # DSV4_GENOUT appends
    # THE MEMGUARD IS NOT THE ENEMY AND IS NOT DISABLED. It fires because 100.40 GiB of weights in a
    # 122.8 GiB pool leaves the loader ~7 GiB, and whether the allocation burst outruns its slope
    # rule is a coin flip. Wait for the page cache to actually come back, then retry the toss.
    local w a
    for w in $(seq 1 40); do
      a=$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo)
      [ "$a" -ge 100 ] && break
      say "$tag try $try: MemAvailable ${a} GiB, waiting for 100"; sleep 15
    done
    say "$tag try $try: launching (env '${extra}', MemAvailable ${a} GiB)"
    env DSV4_PROMPTS_FILE="$PROMPTS" DSV4_BLKSWEEP="$SWEEP" DSV4_GENOUT="$GO" \
        DSV4_MAINKV_CACHE=0 $extra \
        bash scripts/run_model.sh "$LOG" ./build/decode "$CKPT" \
        "0,671,6102,294,8760,344" 16 "" "$NGEN" || { say "run_model refused ($tag)"; continue; }
    for w in $(seq 1 60);  do pgrep -x decode > /dev/null && break; sleep 2;  done
    for w in $(seq 1 360); do pgrep -x decode > /dev/null || break; sleep 10; done
    if grep -q "KILLING" "${LOG%.log}.memguard.log" 2>/dev/null; then
      say "$tag try $try: the memguard killed the load. Retrying."; continue; fi
    local n; n=$(wc -l < "$GO" 2>/dev/null || echo 0)
    say "$tag try $try: $n genout point(s)"
    [ "$n" -ge 4 ] && return 0
    say "$tag try $try: incomplete ($n of 4 points). Retrying."
  done
  say "$tag: FAILED after $TRIES tries"; return 1
}

for spec in "off|DSV4_HADAMARD_STAGE=0" "on|"; do
  tag="${spec%%|*}"; extra="${spec#*|}"
  if run_arm "$tag" "$extra"; then
    { echo "ARM $tag  env='${extra:-unset}'  ($([ "$tag" = off ] && echo 'pre-1.10 aliased hadamard -- MUST diverge' || echo 'shipped staged hadamard -- must be clean'))"
      python3 tools/genout_within.py "$EV/gw_genout_$tag.txt" "$EV/gw_$tag.log"
      echo; } >> "$OUT"
  else
    { echo "ARM $tag  env='${extra:-unset}'"; echo "  FAILED to produce a complete sweep -- NOT evidence"; echo; } >> "$OUT"
  fi
  tail -8 "$OUT"
done

say "=== VERDICT ==="; cat "$OUT"; say "done."
