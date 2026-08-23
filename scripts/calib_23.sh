#!/usr/bin/env bash
# calib_23.sh — produce the (conf, accepted) dumps that decide ladder 2.3.
#
# The 2.3 arms trained before --calib-out existed, so the dumps must be made after the fact. This
# is a PURE INFERENCE PASS: --lr 0 means the optimizer step moves nothing, --resume loads the arm's
# final weights, and every position's `accepted` label and `cvec` prediction are written out. It is
# NOT a training run and its --out is discarded.
#
# --hass-from 1 is REQUIRED and is not a choice: `accepted` is only a free-running label when the
# draft feeds its own previous token (train_head.py:800, F100). Scoring the confidence head against
# teacher-forced labels would measure it against a condition it will never see at serve time.
#
# All three passes use the SAME capture chunk so arm and control are paired.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
CKPT=/home/patrickd/models/DeepSeek-V4-Flash-0731-REAP
IMG=$(grep -oP 'IMG=\K\S+' scripts/s5_session_p25.sh | head -1 | tr -d '"')
DOCK=(sudo docker run --rm --runtime nvidia -v "$ROOT":/work -v "$CKPT":/ckpt:ro
      -v /home/patrickd/s5-capture:/cap -w /work "$IMG")
CHUNK=c2
LOG(){ printf '[calib %s] %s\n' "$(date -Is)" "$*"; }

pass(){ # name a_conf
    local name="$1" aconf="$2"
    local out="evidence/calib_${name}.jsonl"
    [ -s "$out" ] && { LOG "$name already dumped ($(wc -l < "$out") rows), skipping"; return 0; }
    local trained="/home/patrickd/s5-capture/$name/$CHUNK/trained"
    [ -s "$trained/mtp_trained.safetensors" ] || { LOG "$name: no trained weights at $trained, SKIP"; return 1; }
    LOG "$name: inference pass over $CHUNK, a_conf=$aconf, lr 0"
    rm -f "$out"
    "${DOCK[@]}" python3 -u train/train_head.py \
        --capture "/cap/s3recap/$CHUNK/cap" --ckpt /ckpt \
        --out "/tmp/calib_discard_$name" --resume "/cap/$name/$CHUNK/trained" \
        --pos-per-seq 16 --block 5 --lr 0 --hass-from 1 \
        --a-ce 0.1 --a-tv 0.9 --deficit --beta 0.1 --a-conf "$aconf" \
        --calib-out "/work/$out" \
        2>&1 | tee "evidence/calib_${name}.log" | tail -3
    LOG "$name: $(wc -l < "$out" 2>/dev/null || echo 0) rows -> $out"
}

pass s3recap-hass1-p25 0      # CONTROL: identical recipe, confidence head never got a gradient
pass s3recap-conf1.0   1.0
pass s3recap-conf0.1   0.1

LOG "gate:"
for arm in s3recap-conf1.0 s3recap-conf0.1; do
    echo "=================================================== $arm"
    python3 tools/conf_calibration.py "evidence/calib_${arm}.jsonl" \
            --control evidence/calib_s3recap-hass1-p25.jsonl
done
LOG "calib complete:"
