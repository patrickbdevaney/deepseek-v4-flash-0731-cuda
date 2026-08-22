#!/usr/bin/env bash
# eval_stage.sh — after the arms: prove the archive, report, commit, then run the battery ONCE.
#
# Ordered deliberately. The battery is ~3.46 M tokens and takes over a day; it is worth running
# only at the configuration we intend to publish, and only after the heads it measures are proven
# to be backed up. A battery finished against a head that was never archived is a number with no
# artifact behind it.
#
# It runs as its own unit rather than as a tail on chain_after_s3recap.sh because that script is
# mid-execution and bash reads a script by byte offset -- appending to a running script is how you
# get it to execute garbage halfway through.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG(){ printf '[evalstage %s] %s\n' "$(date -Is)" "$*"; }

LOG "waiting for the arms to finish"
while ! grep -q "chain complete:" evidence/chain/chain.log 2>/dev/null; do
    systemctl --user is-active --quiet dsv4-chain || {
        grep -q "chain complete:" evidence/chain/chain.log 2>/dev/null || {
            LOG "dsv4-chain is not running and never reported completion -- STOPPING rather than"
            LOG "evaluating a configuration nobody finished choosing."; exit 1; }
    }
    sleep 120
done
LOG "arms complete"

# ---------------------------------------------------------------- 1. the archive must be real
# The weights are not in git, so nothing about them is recoverable from a clone. This is the
# operator's standing requirement that every draft head be backed up before we move on.
LOG "verifying the head archive"
python3 tools/verify_head_archive.py --json evidence/head_archive_after_arms.json | tail -20
ARCHIVE_RC=${PIPESTATUS[0]}
[ "$ARCHIVE_RC" = "0" ] || LOG "ARCHIVE VERIFY FAILED (rc=$ARCHIVE_RC) -- continuing to report, but"
[ "$ARCHIVE_RC" = "0" ] || LOG "the battery will NOT start on an unproven archive."

# ---------------------------------------------------------------- 2. report
LOG "registry after the arms:"
grep -E '^\| `[a-z0-9._-]+` \| [0-9.]+ \|' HEAD_REGISTRY.md | tail -12
LIVE=$(basename "$(cat config/live_ckpt)" | sed 's/^ckpt-head-//')
BLK=$(grep -oP 'int\s+blk\s*=\s*\K[0-9]+' include/dsv4_engine.h | head -1)
LOG "serving head=$LIVE block=$BLK"

# ---------------------------------------------------------------- 3. commit the results
git add -A HEAD_REGISTRY.md evidence/chain evidence/head_archive_after_arms.json \
           evidence/*_eval.log evidence/*_train.log 2>/dev/null
if ! git diff --cached --quiet; then
    git commit -q -m "results: the draft-head arms, measured and archived

Written by scripts/eval_stage.sh at the end of the arm sweep. Every head is archived whether
or not it was promoted -- a rejected head is still a measured point on the acceptance curve.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>" && LOG "committed $(git rev-parse --short HEAD)"
else
    LOG "nothing to commit"
fi

[ "$ARCHIVE_RC" = "0" ] || { LOG "STOPPING: archive unproven."; exit 1; }

# ---------------------------------------------------------------- 4. the battery, once
# eval_resume.sh is idempotent and resumptive: eval_suite skips completed ids, eval_extend
# continues from stored prefixes, eval_force skips ids already written. The three unfinished
# extensions were dry-run verified as 0-not-continuable (aime25 21, gpqa_diamond 51, mmlu_pro 18),
# and bfcl_mt has never run at all, so it starts clean.
python3 tools/stamp_eval_provenance.py "$LIVE" "$BLK" || LOG "provenance stamp failed (non-fatal)"
LOG "starting the eval battery: extensions, forcing, and bfcl multi-turn"
bash scripts/eval_resume.sh
LOG "eval_resume.sh returned $?"
