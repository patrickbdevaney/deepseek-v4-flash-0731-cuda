#!/usr/bin/env bash
# ce_tv_ablation.sh -- F112: is the training OBJECTIVE aligned with the metric?
#
# The engine VERIFIES GREEDILY: a draft token is accepted iff it equals the target's argmax, and the
# adaptK controller extends the verify on the draft's top1-top2 MARGIN. Both are properties of the
# draft's ARGMAX and its sharpness. The loss is 0.1*CE + 0.9*TV -- 90 % weight on matching the
# target's full distribution, which is a different objective and can be satisfied by a flatter draft.
#
# The training diagnostics say that is exactly what happens:
#
#     draft-argmax == target-argmax : 57.8 %
#     mean top-1 prob   target 0.841   draft 0.715      <- the draft is FLATTER than the target
#
# A flatter draft has smaller margins, the gate extends less often, and verifies get narrower --
# which is the mechanism behind the two measured regressions (s1 and s2 both LOSE to the untrained
# head on their paired hold-out control, -0.40 and -0.65 tau, while winning the frozen suite).
#
# This re-trains from the SAME captured data with different CE/TV weights. No re-capture: the
# capture is already on disk, so each arm costs a train + build + eval (~25 min), not a session.
# One variable, one measurement, same protocol as every other head.
#
#   scripts/ce_tv_ablation.sh [session] [arms...]        arms are "a_ce:a_tv"
set -euo pipefail
ROOT=/home/patrickd/deepseek-v4-flash-0731-cuda
CKPT=/home/patrickd/models/DeepSeek-V4-Flash-0731-REAP
IMG=ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor
NAME="${1:-s2}"; shift || true
ARMS=("$@"); [ "${#ARMS[@]}" -gt 0 ] || ARMS=(0.5:0.5 0.9:0.1 1.0:0.0)
WORK=/home/patrickd/s5-capture/$NAME
CAP="$WORK/c0/cap"
cd "$ROOT"
LOG(){ echo "[abl $(date -Is)] $*"; }
DOCK=(sudo docker run --rm --runtime nvidia -v "$ROOT":/work -v "$CKPT":/ckpt:ro
      -v /home/patrickd/s5-capture:/cap -w /work "$IMG")

[ -s "$CAP/manifest.jsonl" ] || { echo "no capture at $CAP"; exit 1; }
NTRAIN=$(wc -l < "$CAP/manifest.jsonl")
LOG "capture $CAP has $NTRAIN sequences; arms: ${ARMS[*]}"

for ARM in "${ARMS[@]}"; do
    CE="${ARM%%:*}"; TV="${ARM##*:}"
    TAG="ce${CE}_tv${TV}"
    OUT="$WORK/abl_$TAG"
    if [ -s "$ROOT/evidence/${NAME}_abl_${TAG}_eval.log" ]; then
        LOG "$TAG: eval already exists, skipping"; continue
    fi
    LOG "$TAG: training (a_ce=$CE a_tv=$TV) on the SAME $NTRAIN captured sequences"
    "${DOCK[@]}" python3 -u train/train_head.py --capture "/cap/$NAME/c0/cap" --ckpt /ckpt \
        --out "/cap/$NAME/abl_$TAG" --pos-per-seq 16 --a-ce "$CE" --a-tv "$TV" \
        --total-steps "$NTRAIN" --metrics-out "/cap/$NAME/abl_$TAG/train_metrics.json" \
        2>&1 | tee "$ROOT/evidence/${NAME}_abl_${TAG}_train.log" | tail -4
    [ -s "$OUT/mtp_trained.safetensors" ] || { LOG "$TAG: no weights, skipping"; continue; }

    LOG "$TAG: building head"
    "${DOCK[@]}" python3 -u tools/build_trained_head.py --base /ckpt \
        --trained "/cap/$NAME/abl_$TAG/mtp_trained.safetensors" \
        --out "/cap/$NAME/abl_${TAG}_head" 2>&1 | tail -2
    [ -s "$WORK/abl_${TAG}_head/model.safetensors.index.json" ] || { LOG "$TAG: build failed"; continue; }

    LOG "$TAG: eval on the FROZEN suite, adaptK 1.5, NGEN0=200 -- identical protocol to every head"
    sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; sleep 2
    SUITE=$(cat "$ROOT/protocol/suite_prompts.txt")
    DSV4_PROMPTS="$SUITE" \
        DSV4_BLKSWEEP="6:1:1.5:0,6:1:1.5:1,6:1:1.5:2,6:1:1.5:3,6:1:1.5:4,6:1:1.5:5,6:1:1.5:6,6:1:1.5:7,6:1:1.5:8" \
        scripts/run_model.sh "$ROOT/evidence/${NAME}_abl_${TAG}_eval.log" ./build/decode \
        "$CKPT" "0,671,6102,294,8760,344" 8 "$WORK/abl_${TAG}_head" 200
    while pgrep -x decode >/dev/null; do sleep 30; done

    python3 tools/session_gate.py --eval "$ROOT/evidence/${NAME}_abl_${TAG}_eval.log" \
        --json-out "$WORK/abl_${TAG}_verdict.json" || true
    python3 tools/holdout_rate.py --log "$ROOT/evidence/${NAME}_abl_${TAG}_eval.log" || true

    # DISK: the head is 6.6 GB and rebuildable from the 1 GB weights in ~1 min. Keep the weights and
    # the eval, drop the head and the 2.4 GB optimizer state; rebuild only whichever arm wins.
    # sudo: the container writes these as root, so a plain rm silently fails and each arm leaks
    # ~9 GB. Three arms would have been 27 GB against 70 GB free.
    sudo rm -rf "$WORK/abl_${TAG}_head" "$OUT/opt_state.pt"
    LOG "$TAG: done, $(df -BG --output=avail /home/patrickd | tail -1 | tr -d ' ') free"
done

LOG "ABLATION COMPLETE"
python3 tools/ablation_report.py --session "$NAME" || true
