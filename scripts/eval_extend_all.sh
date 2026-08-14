#!/usr/bin/env bash
# eval_extend_all.sh — after the battery, give every budget-limited benchmark its real number.
#
# A truncated item is scored WRONG, so a benchmark that hit its cap often is measuring max_tokens
# rather than the model. GPQA at 8000 truncates ~33 % and reads 64.8 % pooled against 91.9 % on the
# traces that terminated. Publishing the pooled figure against a reference run at a larger budget
# would be a 27-point self-inflicted wound, so every row past the 5 % gate gets extended.
#
# IT CONTINUES, IT DOES NOT RERUN. Selecting items by the outcome of a first draw and then redrawing
# them unconditionally is biased -- the redraw puts fresh mass below the old cap that the kept traces
# already account for. Continuing each truncated trace from its exact stored prefix recomposes the
# target distribution exactly, and costs half the decode. tools/eval_extend.py proves the prefix is
# token-exact per item and aborts if it is not.
#
# Runs unattended after "ALL TASKS COMPLETE" so the extension does not depend on anyone remembering.
# No rebuild and no server restart: /v1/completions already exists in the running binary, and taking
# the engine down mid-programme has been the single most expensive class of mistake on this box.
#
#   nohup setsid bash scripts/eval_extend_all.sh > evidence/evals/extend.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
EFFORT="${EFFORT:-low}"
BUDGET="${BUDGET:-24000}"
GATE="${GATE:-5}"                      # percent truncation above which a row must be extended
POLL="${POLL:-300}"
TASKS="${TASKS:-gpqa_diamond aime24 aime25 lcb math500 scicode}"

say(){ echo "[extend $(date -Is)] $*"; }

say "waiting for the battery to finish before extending anything"
while ! grep -aq "ALL TASKS COMPLETE" evidence/evals/run.log 2>/dev/null; do
  if ! pgrep -f "eval_supervise.sh" > /dev/null && ! pgrep -f "run_evals.sh" > /dev/null; then
    say "no battery and no supervisor running — stopping rather than extending a partial run"
    exit 1
  fi
  sleep "$POLL"
done
say "battery complete; deciding which rows are budget-limited"

python3 tools/eval_suite.py --report > /dev/null 2>&1

for task in $TASKS; do
  jsonl="evidence/evals/${task}.${EFFORT}.jsonl"
  [ -s "$jsonl" ] || { say "$task: no records, skipping"; continue; }

  read -r rate n complete <<EOF
$(python3 - "$task" "$EFFORT" <<'PY'
import json, os, sys
task, eff = sys.argv[1], sys.argv[2]
rows = json.load(open('evidence/evals/summary.json'))
r = next((x for x in rows if x['task'] == task and x.get('effort') == eff), None)
if not r:
    print('0 0 0')
else:
    n = r['n'] or 0
    print(f"{100.0*r.get('truncated',0)/n if n else 0:.1f} {n} "
          f"{1 if (r.get('n_total') and n >= r['n_total']) else 0}")
PY
)
EOF

  if [ "$complete" != "1" ]; then
    say "$task: only $n records and the task is incomplete — refusing to extend a partial run"
    continue
  fi
  over=$(python3 -c "print(1 if float('$rate') > float('$GATE') else 0)")
  if [ "$over" != "1" ]; then
    say "$task: ${rate}% truncated, at or under the ${GATE}% gate — no extension needed"
    continue
  fi

  say "$task: ${rate}% truncated over $n records — continuing truncated traces to ${BUDGET}"
  if python3 tools/eval_extend.py --task "$task" --effort "$EFFORT" --budget "$BUDGET"; then
    tag="${EFFORT}$((BUDGET / 1000))k"
    say "$task: extension finished, landing as ${task}@${tag}"
    bash scripts/eval_land.sh "$task" "$tag" || say "$task: extension did not land (see above)"
  else
    say "$task: EXTENSION FAILED — the base row stays flagged NOT QUOTABLE, which is correct"
  fi
done

say "ALL EXTENSIONS COMPLETE"
python3 tools/eval_suite.py --report
