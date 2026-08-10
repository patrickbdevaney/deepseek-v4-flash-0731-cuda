#!/usr/bin/env bash
# suite_adaptk_sweep.sh -- sweep adaptK on the FROZEN SUITE, which is where the gate actually binds.
#
# Every adaptK re-fit so far ran on the hold-out. That was the wrong instrument: the hold-out is the
# model's own long continuation text, the trained head's margins there are so high that the gate
# barely engages (mean K 5.26 at threshold 0.0 vs 5.27 at 1.5), and a threshold cannot be fitted on
# a workload where it does nothing. The suite is bimodal by comparison -- 38 % of verifies are K=2
# and 24 % are K=7 -- so it is where the threshold has leverage.
#
# The economics of the two modes, measured on s1:
#     K=2 verifies: 1.76 tokens per 108.8 ms = 16.2 tok/s
#     K=7 verifies: 5.92 tokens per 193.1 ms = 30.7 tok/s
# The narrow ones are the drag. Whether widening them pays is exactly what a threshold sweep
# answers, and it is unanswerable from the shipped logs because a position past the chosen K was
# never verified.
#
# NOTE this is a MEASUREMENT, not a promotion. The registry number stays pinned at adaptK 1.50 for
# comparability with every head already in it; changing the shipped threshold would require
# re-baselining the incumbent at the new value, which is a decision, not a side effect.
#
#   scripts/suite_adaptk_sweep.sh <head-dir|""> <tag> [thresholds...]
set -euo pipefail
ROOT=/home/patrickd/deepseek-v4-flash-0731-cuda
CKPT=/home/patrickd/models/DeepSeek-V4-Flash-0731-REAP
HEAD="${1:?usage: suite_adaptk_sweep.sh <head-dir|\"\"> <tag> [thresholds...]}"
TAG="${2:?tag}"; shift 2
THRS=("$@"); [ "${#THRS[@]}" -gt 0 ] || THRS=(0.0 0.5 1.0 1.5 2.0 3.0)
cd "$ROOT"

for THR in "${THRS[@]}"; do
    OUT="$ROOT/evidence/suiteK_${TAG}_${THR}.log"
    [ -s "$OUT" ] && { echo "[sweep] $THR already measured"; continue; }
    echo "[sweep] $TAG adaptK $THR"
    sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; sleep 2
    SUITE=$(cat "$ROOT/protocol/suite_prompts.txt")
    DSV4_PROMPTS="$SUITE" \
        DSV4_BLKSWEEP="$(python3 -c "print(','.join(f'6:1:$THR:{i}' for i in range(0,9)))")" \
        scripts/run_model.sh "$OUT" ./build/decode \
        "$CKPT" "0,671,6102,294,8760,344" 8 "$HEAD" 200
    while pgrep -x decode >/dev/null; do sleep 20; done
    printf "  adaptK %-4s " "$THR"; python3 tools/holdout_rate.py --log "$OUT" --quiet || true
done

echo
echo "[sweep] $TAG summary (pooled tok/s over the 9 suite prompts):"
for THR in "${THRS[@]}"; do
    OUT="$ROOT/evidence/suiteK_${TAG}_${THR}.log"
    [ -s "$OUT" ] || continue
    printf "  %-5s " "$THR"; python3 tools/holdout_rate.py --log "$OUT" | tail -2 | head -1
done
