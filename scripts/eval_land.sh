#!/usr/bin/env bash
# eval_land.sh — land ONE finished benchmark: verify it, publish it, commit it, push it.
#
# Each benchmark is published the moment it finishes rather than at the end of the battery, so the
# repo always reflects exactly what has actually been measured. A run that is interrupted after four
# of seven tasks leaves four complete, verified, pushed results and no half-written table.
#
# NOTHING IS PUBLISHED THAT HAS NOT BEEN RE-DERIVED. `tools/eval_verify.py` re-reads every stored
# generation, re-extracts the answer from the raw text, re-derives the gold from the pinned dataset
# rather than from the record, re-scores, and requires the result to equal the published number. If
# that fails, this script refuses to commit. A number in EVALS.md is therefore always reproducible
# from evidence/evals/<task>.jsonl by anyone, with no GPU and no model.
#
#   bash scripts/eval_land.sh gpqa_diamond
set -u
cd "$(dirname "$0")/.."
TASK="${1:?usage: eval_land.sh <task>}"
JSONL="evidence/evals/${TASK}.jsonl"

[ -s "$JSONL" ] || { echo "[land] $TASK: no records at $JSONL — nothing to land"; exit 1; }

python3 tools/eval_suite.py --report > /dev/null || exit 1
python3 tools/eval_publish.py  > /dev/null || exit 1

# Verify AFTER publishing, so the check compares against the number that is actually in the file.
if ! python3 tools/eval_verify.py --task "$TASK"; then
  echo "[land] $TASK: VERIFICATION FAILED — refusing to commit a number that does not reproduce"
  exit 1
fi

N=$(wc -l < "$JSONL")
ACC=$(python3 -c "
import json,sys
rows={r['task']:r for r in json.load(open('evidence/evals/summary.json'))}
r=rows.get('$TASK')
print(f\"{r['acc']:.1f}% [{r['ci'][0]}, {r['ci'][1]}] n={r['n']}/{r['n_total']} trunc={r['truncated']} err={r.get('errors',0)}\" if r else 'n/a')")

git add -A
git commit -q -F - <<MSG
eval: ${TASK} = ${ACC}

Landed from a completed run of tools/eval_suite.py against the shipping CUDA server at
seqmax 8192, temperature 1.0, top_p 0.95, reasoning_effort high.

Evidence, all in this commit:
  evidence/evals/${TASK}.jsonl        ${N} records — the full generation, reasoning trace,
                                      extracted answer, gold, timings and usage for every item
  evidence/evals/${TASK}.meta.json    dataset, pinned snapshot sha, sampling params, max_tokens
  evidence/evals/summary.json         the scored table
  evidence/evals/verification.json    the independent re-derivation of that table
  evidence/evals/provenance.json      the published facts about the dataset, checked on disk
  evidence/evals/preflight.log        the go/no-go run that permitted this battery

Reproducible with no GPU and no model:
  python3 tools/eval_verify.py --task ${TASK}
re-extracts every answer from the raw stored text, re-derives every gold from the pinned
dataset rather than from the record, re-scores, and requires the published number to match.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG

if git push -q origin main 2>&1; then
  echo "[land] $TASK: ${ACC} — verified, committed, pushed"
else
  echo "[land] $TASK: ${ACC} — verified and committed, PUSH FAILED (commit is safe locally)"
fi
