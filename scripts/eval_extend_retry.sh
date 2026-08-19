#!/usr/bin/env bash
# eval_extend_retry.sh — finish the extensions that the main sweep left unfinished.
#
# WHY THIS EXISTS. eval_extend_all.sh sweeps each task exactly once and never looks back, so an
# extension that dies mid-task is never retried and the chain walks on. On 2026-08-19 GPQA-Diamond
# died at item 10 of 51 on a token-exactness gate that could not be satisfied (see
# tools/eval_extend.py), and the sweep moved to mmlu_pro leaving a 147-of-198 merged file behind.
# That file is NON-EMPTY, which is the only thing eval_force_all.sh used to test, so forcing would
# have selected it, read 0 % truncation out of a file containing only the traces that terminated,
# declined, and left the battery's most truncated row NOT QUOTABLE with every stage log green.
#
# THE PARTIAL FILE IS AN ASSET, NOT RUBBISH. eval_extend.py skips ids already present in the output,
# so re-running it against a half-written file resumes rather than restarts: the 147 carried records
# stand and only the 51 truncated traces are continued. Deleting the partial file would throw away
# nothing but would also lose the resume, so this re-runs in place.
#
# IT DECIDES FROM THE FILES, NOT FROM THE LOG. A log says what a stage believed; the record files
# say what is actually on disk. A task needs another attempt when its base row is over the gate and
# its merged file is not complete -- fewer records than the base run, or no meta written, which is
# the last thing eval_extend.py does and therefore the marker that it finished.
#
# IT TAKES THE ENGINE ALONE, between the sweep and the forcing pass, and eval_force_all.sh blocks
# on it. Two clients generating at once turn a scored item into a timeout banked as a wrong answer.
#
#   nohup setsid bash scripts/eval_extend_retry.sh > evidence/evals/extend_retry.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
EFFORT="${EFFORT:-low}"
BUDGET="${BUDGET:-24000}"
GATE="${GATE:-5}"
POLL="${POLL:-300}"
MAX_PASSES="${MAX_PASSES:-2}"
TASKS="${TASKS:-gpqa_diamond mmlu_pro aime24 aime25 lcb math500 humaneval scicode}"
TAG="${EFFORT}$((BUDGET / 1000))k"

say(){ echo "[retry $(date -Is)] $*"; }

say "waiting for the battery and the main extension sweep to finish"
while pgrep -f "bash scripts/eval_supervise.sh" > /dev/null \
   || pgrep -f "bash scripts/run_evals.sh" > /dev/null \
   || pgrep -f "bash scripts/eval_extend_all.sh" > /dev/null; do
  sleep "$POLL"
done

# Needs a report to read truncation rates from, and an engine to continue traces with.
python3 tools/eval_suite.py --report > /dev/null 2>&1
if ! curl -s -m 10 -o /dev/null http://localhost:8080/health 2>/dev/null; then
  say "the engine is down — refusing to score anything against it. Run scripts/eval_resume.sh."
  exit 1
fi

# ONE DEFINITION OF "FINISHED", ASKED FOR RATHER THAN RE-IMPLEMENTED. eval_force_all.sh asks the
# same tool the same question; two copies of this rule drifting apart is exactly how a chain ends up
# reporting success over an amputated file.
needs_retry(){ python3 tools/eval_ext_complete.py "$1" "$EFFORT" "$TAG" --needs-retry "$GATE"; }

for pass_no in $(seq 1 "$MAX_PASSES"); do
  pending=0
  for task in $TASKS; do
    state=$(needs_retry "$task") || continue
    pending=1
    say "pass $pass_no: $task is incomplete ($state) — resuming its extension"
    if python3 tools/eval_extend.py --task "$task" --effort "$EFFORT" --budget "$BUDGET"; then
      say "$task: extension finished, landing as ${task}@${TAG}"
      bash scripts/eval_land.sh "$task" "$TAG" || say "$task: retried extension did not land"
    else
      say "$task: RETRY FAILED — leaving it for the forcing pass, which will see the residue"
    fi
  done
  [ "$pending" = "1" ] || { say "pass $pass_no: nothing left to retry"; break; }
done

# SAY WHAT IS STILL BROKEN. A stage that ends silently on an unfinished file is the failure mode
# this whole script exists to remove, so name every task that is still short before handing over.
still=0
for task in $TASKS; do
  if state=$(needs_retry "$task"); then
    say "STILL INCOMPLETE: $task ($state) — forcing will work from the base row instead"
    still=1
  fi
done
[ "$still" = "1" ] || say "every row over the gate now has a complete ${TAG} file"

say "ALL EXTENSION RETRIES COMPLETE"
python3 tools/eval_suite.py --report
