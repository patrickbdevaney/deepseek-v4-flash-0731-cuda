#!/usr/bin/env bash
# run_evals.sh — drive the whole published suite against a running server, unattended.
#
# ORDER IS DELIBERATE, and it is ordered by VALUE, not by cost. Measured on this box a reasoning
# item costs ~1.8 min (~2000 completion tokens at ~20 tok/s), so the plan below is roughly 26 hours
# and will very likely be interrupted. Every task is complete-or-absent in the report rather than
# half-done-and-averaged, so what matters is that the most valuable results are banked FIRST:
#
#   1-2. AIME 24/25 -- complete standard benchmarks, 30 items each, and the most widely published
#        peer numbers in existence. Cheap enough to finish inside two hours.
#   3-4. GPQA-Diamond and MMLU-Pro -- the two benchmarks the unpruned 0731 has a published number
#        for (88.10 and 86.40), so they are the actual head-to-head. Also ~7 hours each.
#     5. HumanEval -- complete standard benchmark, exec-scored.
#     6. MATH-500 subset.
#     7. GSM8K last: this repo already has a number for it (F131, 91.7 %), so it confirms rather
#        than discovers.
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

# task:n  (n=0 means the whole benchmark)
PLAN="${PLAN:-aime24:0 aime25:0 gpqa_diamond:0 mmlu_pro:250 humaneval:0 math500:150 gsm8k:150}"

for spec in $PLAN; do
  task="${spec%%:*}"; n="${spec##*:}"
  # A dead server turns the rest of the plan into hundreds of instant failures that look like a
  # capability result. Check before each task and stop instead.
  if ! curl -s -m 10 -o /dev/null "http://${HOST}/v1/models" && \
     ! curl -s -m 10 -o /dev/null "http://${HOST}/"; then
    echo "=== SERVER DOWN before $task — stopping. Restart it and rerun to resume." ; exit 1
  fi
  echo "=== $(date -Is)  $task  n=$n"
  python3 tools/eval_suite.py --task "$task" --n "$n" --host "$HOST"
  echo "=== $(date -Is)  $task done"
done

echo "=== ALL TASKS COMPLETE $(date -Is)"
python3 tools/eval_suite.py --report
