#!/usr/bin/env bash
# chain_p25b.sh — P2.5b: reshape the anchor so the reconstructive give-back stops.
#
# THE FINDING THIS ARM ANSWERS. The release rule, evaluable at block 5 for the first time on
# 2026-08-23, showed that ALL fine-tuning on this corpus costs long_context acceptance:
#
#     run-0 (stock)          5.00
#     s3    (no anchor)      4.00      -1.00
#     p25-b0.1 (beta 0.1)    4.53      -0.47   <- the anchor recovered HALF the loss
#
# So the anchor is the right mechanism and it is simply not strong enough where it matters. But
# raising it uniformly is NOT the fix and that is already measured: beta=0.5 scored 3.6738, BELOW
# the incumbent, because it also froze the weak positions that P2.5's entire gain came from.
#
# The anchor must be RESHAPED, not scaled. beta * r**p * KL, with r the block's accepted fraction:
#
#     p=1, beta=0.10   (the winner)   r=0.9 -> 0.090    r=0.3 -> 0.030
#     p=2, beta=0.30                  r=0.9 -> 0.243    r=0.3 -> 0.027
#     p=2, beta=0.20                  r=0.9 -> 0.162    r=0.3 -> 0.018
#
# 2.7x the anchor on positions that already accept well, and the SAME or LESS on the ones that do
# not. Same knob, opposite effects at the two ends. That is the shape the measurement asks for.
#
# GRADED ON BOTH GATES. A suite-mean win alone is what produced a head that fails 3 of 6 floors, so
# every arm here is run through tools/release_rule.py as well as promote_head.py. An arm that wins
# the mean and still fails the floors has NOT solved the problem it was built for.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG(){ printf '[p25b %s] %s\n' "$(date -Is)" "$*"; }
S5=/home/patrickd/s5-capture

run_arm(){ # name a_ce a_tv deficit beta anchor_pow
    local name="$1"
    if [ -d "$HOME/model-backups/heads/$name" ]; then LOG "$name already archived, skipping"; return 0; fi
    LOG "arm $name: deficit=$4 beta=$5 anchor_pow=$6"
    mkdir -p "$S5/$name"; ln -sfn ../s3/gen.txt "$S5/$name/gen.txt"
    for ci in 0 1 2; do
        mkdir -p "$S5/$name/c$ci"
        ln -sfn ../../s3recap/c$ci/cap "$S5/$name/c$ci/cap"
        lo=$(( ci * 491 + 1 )); hi=$(( ci * 491 + 491 )); [ "$hi" -gt 1472 ] && hi=1472
        sed -n "${lo},${hi}p" "$S5/s3/gen.txt" > "$S5/$name/c$ci/gen.txt"
    done
    S5_GEN="$S5/s3/gen.txt" S5_HOLDOUT=64 S5_CHUNK=491 S5_KEEP_CAP=1 \
    S5_BLOCK="${S5_BLOCK:-5}" S5_INCUMBENT_TAU="$(cat evidence/baseline_tau.value 2>/dev/null)" \
    S5_ACE="$2" S5_ATV="$3" S5_DEFICIT="$4" S5_BETA="$5" S5_ANCHOR_POW="$6" \
        bash scripts/s5_session_p25.sh "$name" "$S5/mixed_prompts_s3.txt" 1536 512 \
        || LOG "arm $name failed (rc=$?), continuing"

    LOG "release rule for $name:"
    python3 tools/release_rule.py "evidence/${name}_eval.log" \
        --run0 evidence/run0_tau_blk5.log --incumbent evidence/baseline_tau_blk5.log 2>&1 || true
}

run_arm s3recap-p25b-p2-b0.3  0.1 0.9 1 0.3 2
run_arm s3recap-p25b-p2-b0.2  0.1 0.9 1 0.2 2

LOG "arms done. Registry:"
tail -3 HEAD_REGISTRY.md
LOG "NOTE: a promotion here is necessary but NOT sufficient. The head that ships must clear the"
LOG "      per-category floors as well -- that is the whole reason this arm exists."
LOG "p25b complete:"
