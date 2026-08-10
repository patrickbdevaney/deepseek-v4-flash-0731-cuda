#!/usr/bin/env bash
# holdout_control.sh -- the PAIRED untrained control the session was missing.
#
# The session registers its untrained hold-out baseline by re-reading pass 1 with
# `holdout_tau.py --only-last 32`. That is a proxy, not a control, and the two runs sit in
# different context regimes:
#
#   pass 1        short seed prompt, generate 512      -> tau averaged over positions 0..512
#   hold-out eval 512-token prompt, generate 220       -> tau over positions 512..732
#
# So "tau rose from 3.58 to 4.71" mixes the training effect with a change of measurement regime,
# and the sign of that confound is not obvious a priori. This runs the UNTRAINED head over the
# EXACT hold-out prompts, budget and thresholds the trained head was measured on -- same context,
# same prompts, same sweep -- so the difference is attributable to training alone.
#
#   scripts/holdout_control.sh [name] [thresholds...]
set -euo pipefail
ROOT=/home/patrickd/deepseek-v4-flash-0731-cuda
CKPT=/home/patrickd/models/DeepSeek-V4-Flash-0731-REAP
NAME="${1:-s1}"; shift || true
THRS=("$@"); [ "${#THRS[@]}" -gt 0 ] || THRS=(1.5 0.5)
WORK=/home/patrickd/s5-capture/$NAME
cd "$ROOT"
[ -s "$WORK/holdout.txt" ] || { echo "no hold-out at $WORK/holdout.txt"; exit 1; }

for THR in "${THRS[@]}"; do
    OUT="$ROOT/evidence/${NAME}_holdout_untrained_${THR}.log"
    echo "[control] untrained head, hold-out prompts, adaptK $THR -> $OUT"
    sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; sleep 2
    # head argument is "" -- the checkpoint's own untrained MTP blocks, everything else identical
    DSV4_PROMPTS_FILE="$WORK/holdout.txt" \
        DSV4_BLKSWEEP="$(python3 -c "print(','.join(f'6:1:$THR:{i}' for i in range(1,9)))")" \
        scripts/run_model.sh "$OUT" ./build/decode \
        "$CKPT" "0,671,6102,294,8760,344" 8 "" 220
    while pgrep -x decode >/dev/null; do sleep 30; done
    python3 tools/holdout_tau.py --log "$OUT" 2>/dev/null | tail -3 || true
done
echo "[control] done"
