#!/usr/bin/env bash
# hf_archive_arms.sh — upload the losing arms as `arms/<name>/`, keeping the champion at the root.
#
# WHY THE LOSERS ARE WORTH PUBLISHING. Eleven heads were measured and one promoted. The other ten
# are the acceptance-vs-recipe curve: they are what makes the winner's numbers interpretable, and
# this project has TWICE had to re-adjudicate refusals after discovering its own ruler was wrong
# (ladder 2.4, traps 38). A refusal with its weights and its unedited log is a reproducible data
# point; a refusal recorded only as a number in a table is a claim.
#
# Layout keeps the champion unambiguous: root = the head you want, arms/ = how it was found.
set -uo pipefail
cd "$(dirname "$0")/.."
REPO=patrickbdevaney/dspark-mtp-draft-head-s3recap-p25-b0.1
CHAMP=s3recap-p25-b0.1
LOG(){ printf '[hfarch %s] %s\n' "$(date -Is)" "$*"; }

for d in ~/model-backups/heads/*/; do
    name=$(basename "$d")
    [ "$name" = "$CHAMP" ] && continue
    [ -s "$d/mtp_trained.safetensors" ] || { LOG "$name: no source, skip"; continue; }
    LOG "uploading arms/$name"
    for f in mtp_trained.safetensors head_card.json train_metrics.json eval.log; do
        [ -s "$d/$f" ] || continue
        timeout 1800 hf upload "$REPO" "$d/$f" "arms/$name/$f" --repo-type model \
            --commit-message "archive arm $name: $f" >/dev/null 2>&1 \
            || LOG "  !! $name/$f FAILED"
    done
    LOG "  done $name"
done
LOG "archive upload complete:"
