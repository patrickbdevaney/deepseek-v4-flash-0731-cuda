#!/usr/bin/env bash
# chain_23.sh — ladder 2.3, the confidence head. Runs after P2.6 closed HASS.
#
# WHY THIS ARM LOOKS LIKE A LOSS AND IS NOT.
#
# The confidence head predicts, per draft position k, whether that token will be accepted. It is
# trained by a BCE term against `accepted = (draft_argmax == target)`, computed at train/train_head.py:800.
# That label is only meaningful when the draft is FREE-RUNNING -- under teacher forcing almost every
# position is marked accepted and the target is near-constant (F100). The only way to get free-running
# labels in this trainer is `--hass-from 1`.
#
# So 2.3 is forced to carry HASS, and P2.6 measured HASS at -0.046 tau on top of the winning recipe.
# These arms will therefore score BELOW the incumbent on the frozen 8-prompt protocol, and
# promote_head.py will refuse them. **That refusal is correct and expected, and it is not the gate
# for this item.**
#
# The protocol holds the draft width FIXED at 5. The confidence head's entire purpose is to let the
# ENGINE VARY it -- spend 5 drafts where acceptance is high and 1-2 where it is not, per
# argmax_k E[A(T_k)]/C(k). A fixed-width instrument cannot see a variable-width win by construction.
# Measuring 2.3 on it would be measurement-and-traps.md's recurring failure: grading with the ruler
# that happens to be lying around.
#
# THE ACTUAL GATE, and it needs no engine change: does the trained confidence head PREDICT
# ACCEPTANCE? `tools/conf_calibration.py` scores AUC of predicted confidence against realised
# acceptance on the held-out sequences, with `s3recap-hass1-p25` (identical recipe, a_conf=0,
# confidence head therefore untrained) as the paired control.
#
#   AUC ~ 0.5  -> the head carries no signal. STOP. The verify-time engine work is worthless and
#                 must not be written. 2.3 closes as a negative and the whole item costs two GPU
#                 sessions instead of a CUDA rewrite.
#   AUC >> 0.5 -> the signal is real. THEN write the verify-time change, and re-measure with
#                 adaptive width against the fixed-width 3.8413 incumbent -- which is the
#                 comparison that answers the question actually being asked.
#
# a_conf IS NOT GUESSED. F100 measured conf = 10034 against ce 10.43 under teacher forcing. The P2.6
# arms measured it free-running at 0.4196 / 0.8377 / 1.4622 / 0.6955 / 0.0359 -- O(1). The DSpark
# paper's value is 1.0 and is now defensible rather than blind; 0.1 brackets it from below.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG(){ printf '[c23 %s] %s\n' "$(date -Is)" "$*"; }
S5=/home/patrickd/s5-capture

run_arm(){ # name a_ce a_tv deficit beta hass a_conf
    local name="$1"
    if [ -d "$HOME/model-backups/heads/$name" ]; then LOG "$name already archived, skipping"; return 0; fi
    LOG "arm $name: deficit=$4 beta=${5:-0} hass_from=${6:-off} a_conf=${7:-0}"
    mkdir -p "$S5/$name"; ln -sfn ../s3/gen.txt "$S5/$name/gen.txt"
    for ci in 0 1 2; do
        mkdir -p "$S5/$name/c$ci"
        ln -sfn ../../s3recap/c$ci/cap "$S5/$name/c$ci/cap"
        lo=$(( ci * 491 + 1 )); hi=$(( ci * 491 + 491 )); [ "$hi" -gt 1472 ] && hi=1472
        sed -n "${lo},${hi}p" "$S5/s3/gen.txt" > "$S5/$name/c$ci/gen.txt"
    done
    S5_GEN="$S5/s3/gen.txt" S5_HOLDOUT=64 S5_CHUNK=491 S5_KEEP_CAP=1 \
    S5_BLOCK="${S5_BLOCK:-5}" S5_INCUMBENT_TAU="$(cat evidence/baseline_tau.value 2>/dev/null)" \
    S5_ACE="$2" S5_ATV="$3" S5_DEFICIT="$4" S5_BETA="${5:-}" S5_HASS="${6:-}" S5_ACONF="${7:-}" \
        bash scripts/s5_session_p25.sh "$name" "$S5/mixed_prompts_s3.txt" 1536 512 \
        || LOG "arm $name failed (rc=$?), continuing"
}

# The winning P2.5 recipe (ce 0.1 / tv 0.9 / deficit / beta 0.1), plus HASS for the labels,
# plus the confidence term at the paper's value and one decade below it.
run_arm s3recap-conf1.0  0.1 0.9 1 0.1 1 1.0
run_arm s3recap-conf0.1  0.1 0.9 1 0.1 1 0.1

LOG "arms done. Registry (expect REFUSALS -- see the header of this script):"
tail -4 HEAD_REGISTRY.md

LOG "the gate that actually decides 2.3:"
LOG "  python3 tools/conf_calibration.py s3recap-conf1.0 s3recap-conf0.1 --control s3recap-hass1-p25"
LOG "c23 complete:"
