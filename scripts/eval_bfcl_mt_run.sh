#!/usr/bin/env bash
# eval_bfcl_mt_run.sh — run BFCL multi-turn once the engine is free, and not one moment before.
#
# WHY IT WAITS. There is one engine and one lock. A second client generating while a benchmark is
# being scored does not merely slow it down -- it turns a scored item into a timeout, which is
# banked as a WRONG ANSWER. That is not hypothetical here: gpqa-0153 was scored incorrect because a
# calibration probe held the lock, and recovering it cost a drop-and-rerun. So this blocks until
# the battery AND the extension pass are both finished, then takes the engine alone.
#
# ORDER. Extension first, then forcing, then multi-turn. Both repair rows that are currently NOT
# QUOTABLE, which is worth more than a new row: a benchmark nobody can cite is worth less than one
# that is merely absent. eval_force_all.sh wakes on the same condition this script used to wake on,
# so without the third clause in the wait loop below the two would start together and put a second
# client on the engine -- the precise failure described above.
#
# SMOKE GATE. The checker, the stateful backends, the prompt assembly and the state comparison are
# all validated offline -- ground truth replayed as model output scores 200/200 on both categories.
# What CANNOT be validated offline is the HTTP generation loop against a live engine, because
# testing it would mean generating, which is the thing this script exists to avoid doing early. So
# the first action after the wait is a 2-item run, and a failure there aborts before committing
# 20 hours to a broken loop.
#
#   nohup setsid bash scripts/eval_bfcl_mt_run.sh > evidence/evals/bfcl_mt.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
EFFORT="${EFFORT:-low}"
POLL="${POLL:-300}"
PY=".venv-bfcl/bin/python"
CATS="${CATS:-base miss_func}"

say(){ echo "[bfcl-mt $(date -Is)] $*"; }

[ -x "$PY" ] || { say "no $PY — run: python3 -m venv .venv-bfcl && see tools/eval_bfcl_mt.py"; exit 1; }

say "waiting for the battery, the extension pass and the forcing pass to finish"
while pgrep -f "bash scripts/eval_supervise.sh" > /dev/null \
   || pgrep -f "bash scripts/run_evals.sh" > /dev/null \
   || pgrep -f "bash scripts/eval_extend_all.sh" > /dev/null \
   || pgrep -f "bash scripts/eval_force_all.sh" > /dev/null; do
  sleep "$POLL"
done
say "engine is free"

# Re-run the offline gate on a sample. Cheap, touches no engine, and catches a package or data
# change since the gate was last green.
if ! $PY tools/eval_bfcl_mt.py --self-gate --category base --n 25; then
  say "SELF-GATE FAILED — refusing to score a model with a checker that cannot score ground truth"
  exit 1
fi

for cat in $CATS; do
  say "$cat: smoke run, 2 items"
  if ! $PY tools/eval_bfcl_mt.py --category "$cat" --n 2 --effort "smoke"; then
    say "$cat: SMOKE FAILED — the generation loop is broken against the live engine, stopping"
    exit 1
  fi
  rm -f "evidence/evals/bfcl_mt_${cat}.smoke.jsonl"

  say "$cat: full run, 200 items"
  $PY tools/eval_bfcl_mt.py --category "$cat" --n 0 --effort "$EFFORT" \
      || { say "$cat: run exited non-zero"; continue; }
  bash scripts/eval_land.sh "bfcl_mt_${cat}" "$EFFORT" || say "$cat: did not land"
done

say "ALL MULTI-TURN COMPLETE"
python3 tools/eval_suite.py --report
