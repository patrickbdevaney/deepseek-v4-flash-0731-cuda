#!/usr/bin/env bash
# suite_adaptk_sweep.sh -- sweep adaptK on the FROZEN SUITE, which is where the gate actually binds.
#
# Every adaptK re-fit before F120 ran on the hold-out. That was the wrong instrument: the trained
# head's margins on its own continuation text are high enough that the gate barely engages (mean K
# 5.26 at threshold 0.0 vs 5.27 at 1.5), and a threshold cannot be fitted on a workload where it
# does nothing. The suite is bimodal -- ~38 % of verifies K=2, ~24 % K=7 -- so the threshold has
# leverage there.
#
# DESIGN, rebuilt after F120 invalidated the first attempt. That version ran one measurement per
# threshold, in increasing order, on a box whose decode rate drifts UPWARD ~6 % with run order. It
# produced an apparent optimum at 2.0 that was exactly the size of the drift and could not be
# separated from it. Three things fix that:
#
#   1. WARM-UP, discarded. The first run of a batch was the slowest in both batches measured.
#   2. REPLICATES per threshold, all shuffled together into one seeded order, so repeats of the same
#      threshold land at different points in the batch and drift becomes noise rather than bias.
#   3. RUN INDEX RECORDED per measurement, so the analysis can fit rate ~ threshold + run_index and
#      subtract the drift instead of hoping it averaged out. With sd ~2.1 % per measurement, n=1 per
#      arm cannot resolve the ~1.5 % differences at stake; this is the minimum design that can.
#
# Still a MEASUREMENT, not a promotion. The registry number stays pinned at adaptK 1.50 for
# comparability with every head already in it; changing the shipped threshold would require
# re-baselining the incumbent at the new value, which is a decision, not a side effect.
#
#   scripts/suite_adaptk_sweep.sh <head-dir|""> <tag> [thresholds...]
#   REPS=4 scripts/suite_adaptk_sweep.sh ...
set -euo pipefail
ROOT=/home/patrickd/deepseek-v4-flash-0731-cuda
CKPT=/home/patrickd/models/DeepSeek-V4-Flash-0731-REAP
HEAD="${1:?usage: suite_adaptk_sweep.sh <head-dir|\"\"> <tag> [thresholds...]}"
TAG="${2:?tag}"; shift 2
THRS=("$@"); [ "${#THRS[@]}" -gt 0 ] || THRS=(0.5 1.0 1.5 2.0 3.0)
REPS="${REPS:-4}"
SEED="${SEED:-20260810}"
cd "$ROOT"
SUITE=$(cat "$ROOT/protocol/suite_prompts.txt")
BLK(){ python3 -c "print(','.join(f'6:1:$1:{i}' for i in range(0,9)))"; }

run_one(){  # <threshold> <outfile>
    sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; sleep 2
    DSV4_PROMPTS="$SUITE" DSV4_BLKSWEEP="$(BLK "$1")" \
        scripts/run_model.sh "$2" ./build/decode \
        "$CKPT" "0,671,6102,294,8760,344" 8 "$HEAD" 200 >/dev/null 2>&1 || true
    while pgrep -x decode >/dev/null; do sleep 15; done
}

# NOTE 0.0 is deliberately not a default threshold: the engine CLAMPS it to the 1.50 default
# (`0.00 -> 1.50` in the eff column), so it silently measures the shipped policy under another
# label. That was useful exactly once, as the accidental replicate that produced F120.
echo "[sweep] $TAG: ${#THRS[@]} thresholds x $REPS reps, shuffled (seed $SEED), plus a discarded warm-up"
echo "[sweep] warm-up (discarded)"
run_one 1.5 "$ROOT/evidence/suiteK_${TAG}_warmup.log"

mapfile -t PLAN < <(python3 -c "
import random
thrs = '''${THRS[*]}'''.split()
plan = [(t, r) for t in thrs for r in range($REPS)]
random.Random($SEED).shuffle(plan)
print('\n'.join(f'{t} {r}' for t, r in plan))")
echo "[sweep] order: $(printf '%s ' "${PLAN[@]}" | tr '\n' ' ')"

CSV="$ROOT/evidence/suiteK_${TAG}_measurements.csv"
echo "run_index,threshold,rep,tok_s,log" > "$CSV"
i=0
for P in "${PLAN[@]}"; do
    THR="${P%% *}"; REP="${P##* }"
    OUT="$ROOT/evidence/suiteK_${TAG}_${THR}_r${REP}.log"
    if [ ! -s "$OUT" ]; then run_one "$THR" "$OUT"; fi
    R=$(python3 tools/holdout_rate.py --log "$OUT" --quiet 2>/dev/null || echo 0)
    echo "$i,$THR,$REP,$R,$OUT" >> "$CSV"
    printf "  [%2d] adaptK %-4s rep %s : %s tok/s\n" "$i" "$THR" "$REP" "$R"
    i=$((i+1))
done

echo
echo "[sweep] $TAG complete -> $CSV"
python3 tools/sweep_analyze.py --csv "$CSV" || true
