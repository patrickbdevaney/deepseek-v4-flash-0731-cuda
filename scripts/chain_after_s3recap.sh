#!/usr/bin/env bash
# chain_after_s3recap.sh — carry the programme from the capture-fix control to the eval battery
# with no agent in the loop.
#
#   wait for s3recap -> recipe arms on its RETAINED capture -> stage the best head -> eval battery
#
# WHY RECIPE ARMS AND NOT A BIGGER CORPUS. The obvious session 2 is "more data", and it was the
# plan until make_corpus.py priced it: a category-balanced corpus needs n/8 per category and the
# sources hold code_edit 249, code_gen 374. The balanced ceiling is ~1992 against s3's 1536 -- +30 %
# for ~12 h of generation, because pass 1 runs every prompt through ./build/decode at decode speed.
#
# The retained capture changes the economics completely. s3recap keeps its ~28 GB capture (279 GB
# free), so an arm that changes only the LOSS MIX costs training alone: no generation, no capture.
# That is the same reason the ablation family exists -- and the ruler fix makes it worth redoing,
# because s2-abl-ce1.0_tv0.0 at tau 3.6712 was refused against an incumbent (s1, 3.5762) that the
# corrected rule says should never have been promoted. Under the fixed comparison it would have
# cleared the baseline's 3.6600. That arm was rejected by the broken ruler, not by the evidence.
#
# a_ce/a_tv are the right knob to sweep and the docstring in train_head.py says why: this engine
# VERIFIES GREEDILY (F112), so acceptance is `draft[i] == target_argmax[i]` -- an argmax match, not
# a distribution match -- while the DSpark defaults (ce 0.1 / tv 0.9) come from a formulation where
# the whole distribution decides. TV still earns its place, since it keeps the draft MARGIN
# calibrated and the margin is what the adaptK gate reads. So: an ablation, not a knob to turn.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG(){ printf '[chain %s] %s\n' "$(date -Is)" "$*"; }
DIE(){ LOG "HALT: $*"; exit 1; }
S5=/home/patrickd/s5-capture

# ---------------------------------------------------------------- 1. wait for the control
LOG "waiting for dsv4-s3recap"
while systemctl --user is-active --quiet dsv4-s3recap; do sleep 120; done
RC=$(systemctl --user show -p ExecMainStatus --value dsv4-s3recap)
LOG "dsv4-s3recap exited status=$RC"
[ "$RC" = "0" ] || DIE "control session failed (status $RC); an arm would inherit it. \
Read: journalctl --user -u dsv4-s3recap"
[ -d "$HOME/model-backups/heads/s3recap" ] || DIE "s3recap left no archive directory"

# The arms are only cheap if the capture really was retained. If it was not, say so and stop
# rather than silently re-capturing three times.
for ci in 0 1 2; do
    [ -s "$S5/s3recap/c$ci/cap/manifest.jsonl" ] \
        || DIE "s3recap/c$ci/cap is missing -- S5_KEEP_CAP did not take effect, so each arm would \
re-capture (~1.2 h each). Fix the retention before spending that three times."
done
LOG "all three capture chunks retained; arms will be training-only"

# ---------------------------------------------------------------- 2. the loss-mix arms
# Order matters: ce1.0/tv0.0 first, because it is the arm the broken ruler rejected and therefore
# the one with a prior. If an arm fails, the chain continues -- one bad arm is not the programme.
run_arm(){
    local name="$1" ace="$2" atv="$3"
    if [ -d "$HOME/model-backups/heads/$name" ]; then LOG "$name already archived, skipping"; return 0; fi
    LOG "arm $name: a_ce=$ace a_tv=$atv (reusing s3recap's capture, no generation, no capture)"
    mkdir -p "$S5/$name"
    ln -sfn ../s3recap/gen.txt "$S5/$name/gen.txt"
    for ci in 0 1 2; do
        mkdir -p "$S5/$name/c$ci"
        # RELATIVE, because the trainer runs in a container with -v $S5:/cap. An absolute symlink
        # into /home/patrickd/s5-capture does not resolve as /cap inside it.
        ln -sfn ../../s3recap/c$ci/cap "$S5/$name/c$ci/cap"
    done
    S5_GEN="$S5/s3recap/gen.txt" S5_HOLDOUT=64 S5_CHUNK=491 S5_KEEP_CAP=1 \
    S5_ACE="$ace" S5_ATV="$atv" \
        bash scripts/s5_session.sh "$name" "$S5/mixed_prompts_s3.txt" 1536 512 \
        || LOG "arm $name failed (rc=$?), continuing"
}
run_arm s3recap-ce1.0 1.0 0.0
run_arm s3recap-ce0.5 0.5 0.5

# ---------------------------------------------------------------- 3. stage the best PROMOTED head
BEST=$(python3 - <<'PY'
import re
best=None
for line in open('HEAD_REGISTRY.md'):
    m=re.match(r"\|\s*`([^`]+)`\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|.*\|\s*(PROMOTED|baseline)\s*\|", line)
    if m and (best is None or float(m.group(2))>best[1]): best=(m.group(1), float(m.group(2)))
print(best[0] if best else "")
PY
)
[ -n "$BEST" ] || DIE "no PROMOTED head in HEAD_REGISTRY.md"
LOG "best promoted head by suite tau: $BEST"
bash scripts/stage_head.sh "$BEST" --activate || DIE "staging $BEST failed"
LOG "staged $BEST"

# ---------------------------------------------------------------- 4. the eval battery
# Authorised by the operator 2026-08-21 for exactly this point: "after you could rewire the systemd
# evals again for another now decode optimized eval battery". The standing rule against restarting
# it held while the GPU was needed for kernel work; that phase is over.
LOG "resuming the eval battery on $BEST"
bash scripts/eval_resume.sh; LOG "eval_resume.sh returned $?"
