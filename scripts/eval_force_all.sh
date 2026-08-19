#!/usr/bin/env bash
# eval_force_all.sh — after the extension, rescue the rows the extension could not reach.
#
# WHY THIS EXISTS AS A SCRIPT AND NOT AS A NOTE TO SELF. tools/eval_force.py was built to be driven
# by hand after looking at the extension results, which means the unattended chain ended with GPQA
# and LCB still over the 5 % truncation gate and flagged NOT QUOTABLE for ever. A benchmark nobody
# can cite was the one outcome the whole battery existed to avoid, so the last repair step is now
# part of the chain rather than a thing somebody has to remember.
#
# IT FORCES THE EXTENDED FILE, NOT THE BASE ONE -- BUT ONLY IF THAT FILE IS FINISHED. Forcing `low`
# would close the thinking block at 8000 tokens for items that would have terminated on their own at
# 19k, throwing away the extension's work and scoring the row worse than it deserves. So each task is
# forced at `low24k` where that file exists and `low` only where it does not.
#
# "EXISTS" USED TO MEAN `[ -s ]`, AND THAT WAS A TRAP THAT SPRUNG. When an extension dies part way
# it has already copied across the traces that TERMINATED -- the easy half -- so it leaves a
# non-empty file containing no truncated items at all. GPQA-Diamond did exactly this on 2026-08-19:
# 147 of 198 records, all of them terminated. `[ -s ]` selected it, eval_force.py measured 0 %
# truncation in it, `--only-if-over 0.05` declined, and the battery's most truncated row would have
# ended the chain NOT QUOTABLE while this log said "declined by the gate -- no forcing needed". A
# stage that reports success over an amputated file is worse than one that crashes.
#
# So completeness is now tested, not assumed: the merged file counts at least as many records as the
# base run AND carries the meta that eval_extend.py writes as its very last act. Anything short of
# that falls back to the base file, which is truncated, over the gate, and therefore actually forced.
#
# THE GATE IS THE POINT. Every forced row costs a paragraph of published caveat -- forcing makes
# truncation ~0 % *by construction*, which the quotability gate cannot detect, so eval_publish.py
# marks it explicitly. `--only-if-over 0.05` means a row the extension already repaired declines
# itself and sends nothing. Expect this script to do real work on exactly two rows and nothing at
# all on the rest; a sweep that forces everything would be a bug, not thoroughness.
#
# IT TAKES THE ENGINE ALONE. Forcing generates, and a second client on the engine turns a scored
# item into a timeout that is banked as a wrong answer -- that is how gpqa-0153 was lost. This
# blocks on the battery and the extension, and eval_bfcl_mt_run.sh blocks on this.
#
#   nohup setsid bash scripts/eval_force_all.sh > evidence/evals/force.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
EFFORT="${EFFORT:-low}"
BUDGET="${BUDGET:-24000}"
GATE="${GATE:-0.05}"                   # fraction truncated above which a row is forced
POLL="${POLL:-300}"
TASKS="${TASKS:-gpqa_diamond mmlu_pro scicode lcb humaneval math500 aime24 aime25}"

say(){ echo "[force $(date -Is)] $*"; }

say "waiting for the battery, the extension sweep and its retry pass to finish"
while pgrep -f "bash scripts/eval_supervise.sh" > /dev/null \
   || pgrep -f "bash scripts/run_evals.sh" > /dev/null \
   || pgrep -f "bash scripts/eval_extend_all.sh" > /dev/null \
   || pgrep -f "bash scripts/eval_extend_retry.sh" > /dev/null; do
  sleep "$POLL"
done

# REFUSE ON A PARTIAL EXTENSION. If the extension died half way -- power cut, OOM, a failed row --
# then some tasks have a low24k file and some do not, and forcing now would quietly force the 8k
# base file for every task that had not been reached yet. That is the amputation this script exists
# to avoid, and it would be invisible in the published table. Wait for the resume instead.
if ! grep -aq "ALL EXTENSIONS COMPLETE" evidence/evals/extend.log 2>/dev/null; then
  say "the extension pass is not running and never printed ALL EXTENSIONS COMPLETE."
  say "Forcing now would force 8k base files for whichever tasks it had not reached yet."
  say "Refusing. Re-run scripts/eval_extend_all.sh first (it resumes), then this."
  exit 1
fi
# The retry pass is what turns a half-finished sweep into a finished one, so its marker is the real
# end of the extension phase. Absent entirely means an older chain that predates it -- carry on.
if [ -e evidence/evals/extend_retry.log ] \
   && ! grep -aq "ALL EXTENSION RETRIES COMPLETE" evidence/evals/extend_retry.log 2>/dev/null; then
  say "the retry pass started but never printed ALL EXTENSION RETRIES COMPLETE."
  say "Refusing. Re-run scripts/eval_extend_retry.sh first (it resumes), then this."
  exit 1
fi
say "extension complete; sweeping for rows still over the ${GATE} gate"

python3 tools/eval_suite.py --report > /dev/null 2>&1

ext="${EFFORT}$((BUDGET / 1000))k"
forced_any=0
for task in $TASKS; do
  # Prefer the extended row, but only when it is COMPLETE. A task with no extended file was under
  # the gate and never extended; naming its base file here is harmless because --only-if-over
  # declines it without sending. A task with a PARTIAL extended file is the dangerous case: see the
  # header. Fall back to the base file there, so the row is forced rather than silently declined.
  eff="$EFFORT"
  if [ -s "evidence/evals/${task}.${ext}.jsonl" ]; then
    if state=$(python3 tools/eval_ext_complete.py "$task" "$EFFORT" "$ext"); then
      eff="$ext"
    else
      say "$task: ${ext} is INCOMPLETE (${state}) — forcing the base ${EFFORT} row instead."
      say "$task: that row is still over the gate, which is why this is the safe fallback."
    fi
  fi
  [ -s "evidence/evals/${task}.${eff}.jsonl" ] || { say "$task: no records, skipping"; continue; }

  # ASK THE TOOL FOR THE TAG, DO NOT GUESS IT. eval_force.py derives the output tag from the base
  # meta's base_effort and max_tokens, not by appending to --effort: `low24k` becomes
  # `low24kforced`, but an unextended `low` at an 8000-token cap becomes `low8kforced`, NOT
  # `lowforced`. Guessing coincidentally works for the extended file and silently fails for the
  # other -- the landing step would look for a file that does not exist and report the row as
  # "declined by the gate", which is a lie that buries a forced row nobody ever publishes.
  tag=$(python3 -c "
import json, os, sys
sys.path.insert(0, 'tools')
import eval_force as F, eval_suite as E
p = 'evidence/evals/${task}.${eff}.meta.json'
m = json.load(open(p)) if os.path.exists(p) else {}
print(F.tag_for(m.get('base_effort') or '${eff}', m.get('max_tokens') or E.MAXTOK['${task}']))
" 2>/dev/null) || { say "$task: could not derive the forced tag, skipping"; continue; }
  [ -n "$tag" ] || { say "$task: empty forced tag, skipping"; continue; }

  say "$task: considering ${task}.${eff} -> ${tag}"
  if python3 tools/eval_force.py --task "$task" --effort "$eff" \
       --only-if-over "$GATE" --host localhost:8080; then
    if [ -s "evidence/evals/${task}.${tag}.jsonl" ]; then
      forced_any=1
      say "$task: forced, landing as ${task}@${tag}"
      bash scripts/eval_land.sh "$task" "$tag" || say "$task: forced run did not land"
    else
      say "$task: declined by the gate or nothing at the cap — no forcing needed"
    fi
  else
    say "$task: FORCING FAILED — the row stays flagged NOT QUOTABLE, which is correct"
  fi
done

[ "$forced_any" = "1" ] || say "no row needed forcing — the extension repaired everything"
say "ALL FORCING COMPLETE"
python3 tools/eval_suite.py --report
