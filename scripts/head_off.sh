#!/usr/bin/env bash
# head_off.sh -- decide WHICH head is fastest, with enough replicates to mean something.
#
# The registry ranks heads on one suite run each. That was adequate when the question was "does this
# beat the incumbent by more than 3.5 %", and it is not adequate for "which of these four is
# fastest", because they sit within ~2 % of each other. F121 established what a properly controlled
# measurement on this box actually costs and buys: with a discarded warm-up and a shuffled order the
# residual sd is ~0.12 tok/s (0.5 %), which resolves 1 % differences at n=4. Without those controls
# the first adaptK sweep manufactured a result out of run-order drift.
#
# So this runs every candidate head REPS times, shuffles all runs into one seeded order so each
# head's repeats land at different points in the batch, records the run index, and hands the result
# to tools/sweep_analyze.py to fit rate ~ head + run_index and report each head against the
# reference with a standard error.
#
# Heads are given by archive name (under ~/model-backups/heads) or by an explicit directory. An
# archive holding only trained tensors is materialised first with tools/build_trained_head.py, which
# is why this can compare arms whose 7 GB heads were deleted after their eval.
#
#   scripts/head_off.sh <tag> <adaptK> <head1> [head2 ...]
#   REPS=4 scripts/head_off.sh final 2.0 s1 s2 s2-abl-ce1.0_tv0.0 s3
set -euo pipefail
ROOT=/home/patrickd/deepseek-v4-flash-0731-cuda
CKPT=/home/patrickd/models/DeepSeek-V4-Flash-0731-REAP
IMG=ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor
STORE=$HOME/model-backups/heads
WORKDIR=/home/patrickd/s5-capture/headoff
TAG="${1:?usage: head_off.sh <tag> <adaptK> <head...>}"
THR="${2:?adaptK}"; shift 2
HEADS=("$@"); [ "${#HEADS[@]}" -ge 2 ] || { echo "need at least two heads"; exit 1; }
REPS="${REPS:-4}"; SEED="${SEED:-20260810}"
cd "$ROOT"; mkdir -p "$WORKDIR"
SUITE=$(cat "$ROOT/protocol/suite_prompts.txt")

# ---- resolve each name to a loadable head directory, materialising compact archives ------------
declare -A DIR
for H in "${HEADS[@]}"; do
    if [ -s "$H/model.safetensors.index.json" ]; then DIR[$H]="$H"; continue; fi
    if [ -s "$STORE/$H/model.safetensors.index.json" ]; then DIR[$H]="$STORE/$H"; continue; fi
    if [ -s "/home/patrickd/s5-capture/$H/head/model.safetensors.index.json" ]; then
        DIR[$H]="/home/patrickd/s5-capture/$H/head"; continue; fi
    TR="$STORE/$H/mtp_trained.safetensors"
    [ -s "$TR" ] || { echo "cannot resolve head '$H' (no loadable head, no trained tensors)"; exit 1; }
    OUT="$WORKDIR/$H"
    if [ ! -s "$OUT/model.safetensors.index.json" ]; then
        echo "[headoff] materialising $H from its trained tensors"
        sudo docker run --rm --runtime nvidia -v "$ROOT":/work -v "$CKPT":/ckpt:ro \
            -v "$STORE":/store -v "$WORKDIR":/out -w /work "$IMG" \
            python3 -u tools/build_trained_head.py --base /ckpt \
            --trained "/store/$H/mtp_trained.safetensors" --out "/out/$H" 2>&1 | tail -2
    fi
    DIR[$H]="$OUT"
done

run_one(){  # <head-dir> <outfile>
    sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; sleep 2
    DSV4_PROMPTS="$SUITE" \
        DSV4_BLKSWEEP="$(python3 -c "print(','.join(f'6:1:$THR:{i}' for i in range(0,9)))")" \
        scripts/run_model.sh "$2" ./build/decode \
        "$CKPT" "0,671,6102,294,8760,344" 8 "$1" 200 >/dev/null 2>&1 || true
    while pgrep -x decode >/dev/null; do sleep 15; done
}

echo "[headoff] $TAG: ${#HEADS[@]} heads x $REPS reps at adaptK $THR, shuffled, plus a warm-up"
run_one "${DIR[${HEADS[0]}]}" "$ROOT/evidence/headoff_${TAG}_warmup.log"

mapfile -t PLAN < <(python3 -c "
import random
heads = '''${HEADS[*]}'''.split()
plan = [(h, r) for h in heads for r in range($REPS)]
random.Random($SEED).shuffle(plan)
print('\n'.join(f'{h} {r}' for h, r in plan))")

CSV="$ROOT/evidence/headoff_${TAG}_measurements.csv"
echo "run_index,threshold,rep,tok_s,log" > "$CSV"
i=0
for P in "${PLAN[@]}"; do
    H="${P%% *}"; REP="${P##* }"
    OUT="$ROOT/evidence/headoff_${TAG}_${H}_r${REP}.log"
    [ -s "$OUT" ] || run_one "${DIR[$H]}" "$OUT"
    # LOSSLESS is a precondition, not a tiebreak: a head that changes the emitted sequence is not a
    # candidate at any speed, so record it and let the analysis see a zero rather than a fast lie.
    if ! grep -q "LOSSLESS GATE: first 8 tokens match base AR -> PASS" "$OUT"; then
        echo "  [$i] $H rep $REP : LOSSLESS FAIL -- excluded"
        echo "$i,$H,$REP,0,$OUT" >> "$CSV"; i=$((i+1)); continue
    fi
    R=$(python3 tools/holdout_rate.py --log "$OUT" --quiet 2>/dev/null || echo 0)
    echo "$i,$H,$REP,$R,$OUT" >> "$CSV"
    printf "  [%2d] %-26s rep %s : %s tok/s\n" "$i" "$H" "$REP" "$R"
    i=$((i+1))
done

echo
python3 tools/sweep_analyze.py --csv "$CSV" --ref "${HEADS[0]}" || true
echo
echo "[headoff] the winner is the head with the largest POSITIVE effect at |t| >= 2."
echo "[headoff] package it with:  python3 tools/package_release.py --head <name> --out <dir>"
