#!/usr/bin/env bash
# run_evals.sh — drive the whole published suite against a running server, unattended.
#
# ORDER IS DELIBERATE, and it is ordered by STATISTICAL DEFENSIBILITY, not by cost or by appeal.
# Measured on this box a reasoning item costs ~1.8-4 min, so this plan will very likely be
# interrupted; what matters is that the results that can actually carry a published claim are
# banked FIRST. The Wilson half-width at each planned n, assuming the accuracy in brackets:
#
#   GPQA-Diamond  n=198 [~85%]  +-5.0   <- headline, and the one with a reference number
#   MMLU-Pro      n=200 [~80%]  +-5.5   <- headline, and the one with a reference number
#   HumanEval     n=164 [~85%]  +-5.5
#   MATH-500      n=120 [~90%]  +-5.4
#   GSM8K         n=120 [~92%]  +-4.9
#   AIME 24/25    n=30  [~70%]  +-15.6  <- NOT PUBLISHABLE AT reps=1
#
# That last row is why AIME runs at `reps=4`. Thirty problems sampled once at temperature 1.0 is
# not a measurement -- the interval spans 52-83 %, which cannot distinguish a good model from a
# mediocre one. Published AIME numbers are avg@16 to avg@64 for exactly this reason. avg@4 takes
# the half-width to +-8.1, which is the least this can be run at and still be quoted. Reps are
# additive (rep 0 keeps the bare id), so k can be raised later without regenerating anything.
#
# SUBSET SIZES are a time budget, not a statistical preference: at ~20 tok/s the full MMLU-Pro test
# split is 12032 items and would take a week. `--n` draws a fixed deterministic subset (see
# `_sample`), the report always prints n-scored against n-in-benchmark, and EVALS.md quotes the
# Wilson interval, so a subset can never be read as the whole benchmark.
#
# Every task resumes: rerunning this script skips ids already in evidence/evals/<task>.jsonl.
#
#   nohup bash scripts/run_evals.sh > evidence/evals/run.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
export TMPDIR="${TMPDIR:-/tmp}"
HOST="${HOST:-localhost:8080}"

# task:n:reps  (n=0 = whole benchmark, reps = independent samples per item for avg@k)
EFFORT="${EFFORT:-low}"
# AIME runs at reps=4 and LAST, on the preflight's insistence. At reps=1 its 30 items give a
# +-13.9 point interval, which is not a measurement; avg@4 brings it to +-7.1, the least it can
# be quoted at, and costs 4x. Last because it is the only benchmark here with NO published
# reference number -- so if the run is cut short AIME is simply ABSENT rather than present and
# unquotable, and the benchmarks carrying the head-to-head have already landed.
# Ordered cheapest-and-most-informative first, so an interrupted battery still leaves the widest
# coverage on disk. bfcl and bfcl_live together cost 0.17 of a day; scicode is the keystone (the
# only task needing knowledge, reasoning and code in one item) and is placed before the long
# single-axis coding tasks. AIME sits last at reps=2, not 4: tools/eval_budget.py puts the full plan
# at exactly the 7-day ceiling with most rows still ASSUMED, and the nested bootstrap shows extra
# reps narrow the interval only when the model is genuinely stochastic on those problems -- so reps
# 4 spends a day of wall clock on a width improvement that may be zero. Raising k later is purely
# additive (rep 0 keeps the bare id), so this is reversible once the variance split is measured.
PLAN="${PLAN:-gpqa_diamond:0:1 bfcl:0:1 bfcl_live:0:1 scicode:0:1 mmlu_pro:150:1 humaneval:0:1 math500:100:1 lcb:0:1 aime24:0:2 aime25:0:2}"

# The battery does not start unless the preflight says GO. Its checks are not advisory: the first
# run of this suite lost an entire benchmark, mis-scored another, and quoted a third from a sample
# too small to mean anything -- and printed a clean table for all three.
if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
  echo "=== preflight $(date -Is)"
  if ! python3 -u tools/eval_preflight.py --host "$HOST"; then
    echo "=== PREFLIGHT NO-GO — not running the battery. Fix the failures above and rerun."; exit 1
  fi
fi

for spec in $PLAN; do
  IFS=: read -r task n reps <<< "$spec"; reps="${reps:-1}"
  # A dead server turns the rest of the plan into hundreds of instant failures that look like a
  # capability result. Check before each task and stop instead.
  if ! curl -s -m 10 -o /dev/null "http://${HOST}/v1/models" && \
     ! curl -s -m 10 -o /dev/null "http://${HOST}/"; then
    echo "=== SERVER DOWN before $task — stopping. Restart it and rerun to resume." ; exit 1
  fi
  echo "=== $(date -Is)  $task  n=$n reps=$reps effort=$EFFORT"
  python3 tools/eval_suite.py --task "$task" --n "$n" --reps "$reps" --effort "$EFFORT" --host "$HOST"
  echo "=== $(date -Is)  $task done"
done

echo "=== ALL TASKS COMPLETE $(date -Is)"
python3 tools/eval_suite.py --report
