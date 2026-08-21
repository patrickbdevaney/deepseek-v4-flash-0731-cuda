#!/usr/bin/env bash
# lhash_ablate.sh -- ladder 1.10. NAME THE KERNEL INSIDE compressed_attn_forward THAT IS RACING.
#
# 1.9 bounded the fault to one function and proved it is a RACE (identical sweep points inside ONE
# process disagree with each other). It did not say WHICH of `compressor_forward`, `indexer_forward`
# or `sparse_attn`. This runs 1.9's R protocol -- eight identical sweep points, four at prefill 192
# and four at 256, `DSV4_HASH=2` hashing the hidden state after each of the 43 prefill layers -- once
# per HYPOTHESIS, where a hypothesis is one existing env flag that swaps one kernel for a different
# implementation of the same maths. An arm that comes back CLEAN names its kernel.
#
# THE PRIOR THIS IS TESTING, and it comes free out of 1.9's own logs (tools/lhash_pairs.py on
# stephash_R1/R2): all 56 pairs diverge and ALL 56 first-differing layers have `compress_ratio 4`.
# Layers alternate 4 / 128 from layer 2 up, and `compressed_attn_forward` runs `has_indexer` and
# `overlap` ONLY when ratio == 4 -- so ratio-128 layers execute `compressor_forward` (non-overlap)
# and `sparse_attn` and never once broke first. That points at the indexer and at the overlap
# pooling, and away from sparse_attn; the arms below are ordered to test the strongest first.
#
# NO KERNEL IS EDITED HERE. Every arm is an environment variable on the SHIPPED binary, which is
# what makes a clean arm attributable: same weights, same input, same everything but one dispatch.
#
#   nohup setsid bash scripts/lhash_ablate.sh > evidence/decode_loop/lhash_ablate.log 2>&1 </dev/null &
set -u
cd "$(dirname "$0")/.."
EV=evidence/decode_loop
PROMPTS="${PROMPTS:-$EV/winladder_prompts.txt}"
# prompt 7 = 192 positions, prompt 8 = 256. Four repeats of each: C(4,2)=6 within-process pairs per
# group per process, 12 per arm, plus 16 cross-process pairs. 1.9's R sweep, unchanged.
SWEEP="${SWEEP:-6:1:1.50:7,6:1:1.50:7,6:1:1.50:7,6:1:1.50:7,6:1:1.50:8,6:1:1.50:8,6:1:1.50:8,6:1:1.50:8}"
OUT=$EV/lhash_ablate_verdict.txt
say(){ echo "[1.10] $(date -Is) $*"; }

# name|extra env. `base` MUST be first: the binary has changed twice (1.11, 1.12) since 1.9 measured
# this, and an ablation campaign whose control is a stale log measures nothing.
ARMLIST="${ARMLIST:-
base|
ixg0|NO_IXGEMM=1
rdx0|DSV4_TOPK_RADIX=0
ixg0tile0|NO_IXGEMM=1 NO_IXTILE=1
f32mk0|NO_FP32MK=1
sparse0|DSV4_SPARSE_HPB=1 DSV4_SPARSE_SMEM=0
}"

[ -s "$PROMPTS" ] || { say "REFUSING: $PROMPTS missing."; exit 1; }
: > "$OUT"
say "sweep=$SWEEP  prompts=$PROMPTS"

echo "lhash_ablate -- ladder 1.10, $(date -Is)" >> "$OUT"
echo "binary: build/decode $(date -Is -r build/decode)" >> "$OUT"
echo >> "$OUT"

while IFS='|' read -r name extra; do
  [ -n "${name:-}" ] || continue
  say "=== ARM $name  env='${extra}' ==="
  ARMS="${name}1 ${name}2" XENV="DSV4_HASH=2 ${extra}" PROMPTS="$PROMPTS" \
    SWEEP="$SWEEP" NGEN=4 LVL=1 bash scripts/stephash_run.sh > "$EV/lhash_${name}.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    say "ARM $name FAILED (rc=$rc) -- see $EV/lhash_${name}.log"
    { echo "ARM $name  env='${extra}'"; echo "  FAILED rc=$rc -- NOT a clean arm, NOT evidence"; echo; } >> "$OUT"
    continue
  fi
  { echo "ARM $name  env='${extra}'"
    python3 tools/lhash_pairs.py "${name}1=$EV/stephash_${name}1.log" "${name}2=$EV/stephash_${name}2.log"
    echo; } >> "$OUT"
  tail -8 "$OUT"
done <<< "$ARMLIST"

say "=== VERDICT TABLE ==="
cat "$OUT"
say "done."
