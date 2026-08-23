#!/usr/bin/env bash
# measure_ck.sh — the gating measurement for adaptive block width (ROADMAP.md lever 1).
#
# WHY THIS IS THE WHOLE QUESTION, AND WHY IT NEEDS NO CUDA. Adaptive width is priced at +20-25 %
# on the assumption that the best width DIFFERS BY TASK SHAPE. That assumption has never been
# tested. It is testable right now: the engine already accepts widths 4-12 (DECODE_LADDER 1978
# swept exactly that range), so sweeping every width against every suite prompt answers it directly.
#
#   for each prompt, k* = argmax_k tok/s(k)
#
# tok/s already IS E[A(T_k)] / C(k) -- accepted tokens per verify over the cost of that verify --
# so the per-prompt argmax is exactly the quantity an adaptive engine would be choosing, measured
# rather than modelled.
#
# THE TEST IS DECISIVE IN BOTH DIRECTIONS, which is the point:
#
#   k* varies by prompt   -> adaptive width is real, and the SPREAD in tok/s between k*=5 and each
#                            prompt's own k* is the honest upper bound on what the engine change
#                            can return. No modelling, no estimate.
#   k* = 5 everywhere     -> the +20-25 % is WRONG, lever 1 is dead, and it cost one overnight run
#                            instead of a CUDA rewrite. That is the better outcome to discover now.
#
# Ladder 2.1 already found width 6 worse than 5 AT FIXED WIDTH because verify is expensive. This
# asks the different question 2.1 could not: is 5 best for EVERY prompt, or only on average?
set -uo pipefail
cd "$(dirname "$0")/.."
LOG(){ printf '[ck %s] %s\n' "$(date -Is)" "$*"; }
OUT=evidence/ck_sweep.log

LOG "waiting for the corpus and arm chains"
# Waits for the CORPUS and ARM chains only -- deliberately NOT for the autopilot, which this now
# runs AHEAD of. The autopilot's finalize() launches the eval battery and then exits, so waiting on
# it would have put this sweep on the GPU at the same time as the battery. Ordering is
# p25b -> corpus -> ck -> arms -> evals, and it is enforced from both sides.
while systemctl --user is-active --quiet dsv4-corpus dsv4-p25b; do sleep 300; done

CKPT=$(cat config/live_ckpt)
SUITE=$(cat protocol/suite_prompts.txt)
# 9 widths x 9 prompts. Prompt 0 is the control and is kept -- it is the worst case and its k*
# is informative on its own.
SW=$(python3 -c "print(','.join(f'{k}:1:1.5:{i}' for k in (4,5,6,7,8,9,10,12) for i in range(0,9)))")
LOG "sweeping widths 4-12 x 9 prompts on $CKPT"
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; sleep 2
DSV4_PROMPTS="$SUITE" DSV4_BLKSWEEP="$SW" \
    scripts/run_model.sh "$OUT" ./build/decode "$CKPT" "0,671,6102,294,8760,344" 8 "" 200
while pgrep -x decode >/dev/null; do sleep 30; done

LOG "per-prompt optimum:"
python3 tools/analyze_ck.py "$OUT" 2>&1 | tee -a evidence/chain/ck.log
LOG "ck complete:"
