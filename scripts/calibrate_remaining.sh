#!/usr/bin/env bash
# calibrate_remaining.sh — measure the completion-length tail of every benchmark still queued,
# BEFORE it runs, so no budget is set by guess a second time.
#
# WHY THIS EXISTS. GPQA's 8000-token cap came from a calibration that ran 8 items and reported "the
# terminating distribution tops out at 6356 tokens, so 8000 clears every one". That inference is
# invalid: the maximum of the TERMINATING subset is a censored statistic, bounded by the budget by
# construction, so it always looks reassuring no matter how heavy the real tail is. The signal that
# mattered was the 2 of 8 that did not terminate -- written off at the time as a property of the
# model rather than of the budget. It became 31.7 % truncation across 120 items, and a benchmark
# that measures max_tokens instead of the model.
#
# So each task is probed at a DELIBERATELY GENEROUS budget (uncensored, 16000) and the truncation
# rate is read off directly rather than inferred from the part of the distribution that fit.
#
# Runs sequentially: every probe serialises on the engine's lock behind whatever the battery is
# doing, so running these in parallel would not make them finish sooner, it would just interleave
# them into the battery's own queue.
#
#   nohup setsid bash scripts/calibrate_remaining.sh > evidence/evals/calib_remaining.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
N="${N:-10}"
BUDGET="${BUDGET:-16000}"
EFFORT="${EFFORT:-low}"

say(){ echo "[calib $(date -Is)] $*"; }

# Wait out any calibration already in flight (BFCL was launched first, being next in the battery).
while pgrep -f "eval_calibrate.py --task bfcl" > /dev/null; do sleep 60; done
say "bfcl calibration finished, continuing with the rest"

for task in mmlu_pro humaneval lcb math500 aime24; do
  out="evidence/evals/calib_${task}_${EFFORT}.log"
  if [ -s "$out" ] && grep -q "recommend" "$out" 2>/dev/null; then
    say "$task already calibrated, skipping"; continue
  fi
  say "calibrating $task (n=$N, uncensored budget=$BUDGET)"
  python3 tools/eval_calibrate.py --task "$task" --n "$N" --budget "$BUDGET" \
      --effort "$EFFORT" > "$out" 2>&1
  say "$task done: $(tail -3 "$out" | tr -d '\000' | tr '\n' ' ')"
done

say "ALL CALIBRATIONS COMPLETE — budgets can now be set from measurement"
