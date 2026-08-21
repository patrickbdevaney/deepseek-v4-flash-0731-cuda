#!/usr/bin/env bash
# lhash_verify.sh -- DECODE_LADDER 1.10, the AFTER arm. Does the prefill reproduce itself now?
#
#   nohup setsid bash scripts/lhash_verify.sh > evidence/decode_loop/lhash_verify.log 2>&1 </dev/null &
#
# Named in detach_audit.sh's PATTERNS in the same commit.
#
# THREE PROTOCOLS, EACH OF WHICH 1.9 RAN AND WHICH FAILED, re-run on the fixed binary. Re-running
# the SAME protocols rather than new ones is the point: a new probe that comes back clean cannot be
# distinguished from a probe that cannot see the defect.
#
#   R  the repeat protocol -- eight identical sweep points, four at prefill 192 and four at 256,
#      inside ONE process. 1.9's own logs give 56 of 56 pairs divergent (tools/lhash_pairs.py).
#   W  the LENGTH ladder 128 -> 256, which is where 1.9 put the threshold between 160 and 192.
#      The prediction is not merely "clean": it is that the ladder is clean at EVERY length now,
#      including the ones that were already clean, i.e. nothing regressed below the boundary.
#   A  1.0's divergent arm at full length -- NGEN 256, ctx 5 / 3,071 / 1,023, DSV4_STEPHASH_LVL=2 --
#      which is the one that says the ladder's PRIMARY correctness invariant (byte-identical
#      generated token ids) is back. 1.9 measured points 1 and 2 diverging at verify 0 with `mkv`
#      and `mx` already different before the first draft ran.
set -u
cd "$(dirname "$0")/.."
EV=evidence/decode_loop
OUT=$EV/lhash_verify_verdict.txt
say(){ echo "[1.10-verify] $(date -Is) $*"; }

[ -x build/decode ] || { say "REFUSING: build/decode missing"; exit 1; }
for f in kernels/indexer.cu include/indexer.h; do
  [ "$f" -nt build/decode ] && { say "REFUSING: build/decode is older than $f -- rebuild first."; exit 1; }
done
: > "$OUT"
{ echo "lhash_verify -- ladder 1.10 AFTER arm, $(date -Is)"
  echo "binary: build/decode $(date -Is -r build/decode)"; echo; } >> "$OUT"
say "binary: build/decode $(date -Is -r build/decode)"

RSWEEP="6:1:1.50:7,6:1:1.50:7,6:1:1.50:7,6:1:1.50:7,6:1:1.50:8,6:1:1.50:8,6:1:1.50:8,6:1:1.50:8"
WSWEEP="6:1:1.50:1,6:1:1.50:2,6:1:1.50:3,6:1:1.50:4,6:1:1.50:5,6:1:1.50:6,6:1:1.50:7,6:1:1.50:8"

# ---- R: the repeat protocol -------------------------------------------------------------------
say "=== R: eight identical points, prefill 192 and 256, DSV4_HASH=2 ==="
ARMS="fixR1 fixR2" XENV="DSV4_HASH=2" PROMPTS=$EV/winladder_prompts.txt \
  SWEEP="$RSWEEP" NGEN=4 LVL=1 bash scripts/stephash_run.sh > $EV/lhash_fixR.log 2>&1
{ echo "R  -- repeats at prefill 192 / 256, one process each, two processes"
  python3 tools/lhash_pairs.py "fixR1=$EV/stephash_fixR1.log" "fixR2=$EV/stephash_fixR2.log"
  echo; } >> "$OUT"
tail -8 "$OUT"

# ---- W: the length ladder ---------------------------------------------------------------------
say "=== W: the length ladder 128 -> 256, the regime 1.9 put the threshold in ==="
ARMS="fixW1 fixW2" XENV="DSV4_HASH=2" PROMPTS=$EV/winladder_prompts.txt \
  SWEEP="$WSWEEP" NGEN=4 LVL=1 bash scripts/stephash_run.sh > $EV/lhash_fixW.log 2>&1
{ echo "W  -- length ladder 128,129,132,136,144,160,192,256 across two processes"
  echo "     (every PSp here is a group of ONE per process, so the numbers that matter are the"
  echo "      cross-process pairs; a within-process count of 0/0 is expected, not a pass)"
  python3 tools/lhash_pairs.py "fixW1=$EV/stephash_fixW1.log" "fixW2=$EV/stephash_fixW2.log"
  echo; } >> "$OUT"
tail -14 "$OUT"

# ---- A: 1.0's divergent arm, full length, LVL=2 -----------------------------------------------
say "=== A: 1.0's divergent arm, NGEN=256, ctx 5 / 3071 / 1023, STEPHASH LVL=2 ==="
ARMS="fixA fixB" XENV="" PROMPTS=$EV/mainkv_prompts.txt \
  SWEEP="6:1:1.50:0,6:1:1.50:1,6:1:1.50:2" NGEN=256 LVL=2 bash scripts/stephash_run.sh \
  > $EV/lhash_fixA.log 2>&1
{ echo "A  -- 1.0's divergent arm, token ids and the per-step causal chain"
  sed -n '/token-id level/,$p' $EV/lhash_fixA.log
  echo; } >> "$OUT"

say "=== VERDICT ==="
cat "$OUT"
say "done."
