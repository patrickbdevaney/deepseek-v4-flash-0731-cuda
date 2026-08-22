#!/usr/bin/env bash
# chain_p26.sh — P2.6 (HASS) and 2.3 (the confidence head), after the P2.5 sweep.
#
# WHY THESE ARE LAST AND WHY THEY ARE TOGETHER. The head TRAINS teacher-forced and RUNS
# free-running: at inference the engine feeds the draft's own previous token because there is
# nothing else to feed. Step 1 is identical either way, so the whole mismatch lives in steps 2..K,
# which is exactly where acceptance decays and exactly what P2.6 scopes.
#
# They are one stage because HASS is what makes 2.3 possible at all. a_conf has been pinned at 0
# since F100 for a measured reason -- the confidence term came in ~1000x the others (ce 10.43,
# tv 0.93, conf 10034) partly because it was trained to predict acceptance under free-running
# drafting while teacher forcing marks almost every position accepted. HASS removes that half of
# the cause. The other half, the un-normalised x with std ~190, is a SCALE question this stage
# measures rather than guesses: every arm records parts["conf"] even when a_conf = 0, so the
# free-running magnitude is read off arm 1 before any arm is trained on it.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG(){ printf '[p26 %s] %s\n' "$(date -Is)" "$*"; }
S5=/home/patrickd/s5-capture

LOG "waiting for the P2.5 sweep to finish"
while ! grep -q "chain complete:" evidence/chain/chain.log 2>/dev/null; do
    systemctl --user is-active --quiet dsv4-chain || {
        grep -q "chain complete:" evidence/chain/chain.log 2>/dev/null || {
            LOG "dsv4-chain stopped without completing -- not starting P2.6 on top of it"; exit 1; }
    }
    sleep 120
done
LOG "P2.5 sweep complete; starting HASS"

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

# Wave 1: HASS alone, so its effect is attributable. Recipe otherwise identical to the incumbent.
run_arm s3recap-hass1        0.1 0.9 0 "" 1 ""
# Wave 2: HASS with P2.5's weighting, since both act on the same positions and the question is
# whether they compose or fight.
run_arm s3recap-hass1-p25    0.1 0.9 1 0.1 1 ""

LOG "HASS arms done. Free-running confidence magnitude, from the arm logs:"
grep -hoE 'conf=[0-9.]+' evidence/s3recap-hass1_c*_train.log 2>/dev/null | tail -5 || LOG "  (none parsed)"
LOG "2.3's a_conf is NOT set blind -- read the magnitude above, then run:"
LOG "  S5_ACONF=<value> scripts/chain_p26.sh   (or add an arm here)"
LOG "p26 complete:"
