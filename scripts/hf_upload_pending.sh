#!/usr/bin/env bash
# hf_upload_pending.sh — upload only the heads that are not on HuggingFace yet.
# Idempotent and resumable: it asks the repo what it already has, so re-running costs one API call
# per head and re-uploads nothing. Safe to run while the GPU is capturing -- this is network and
# disk, never the GPU, and capture is not a scored measurement.
set -uo pipefail
cd "$(dirname "$0")/.."
REPO=patrickbdevaney/dspark-mtp-draft-head-s3recap-p25-b0.1
CHAMP=$(basename "$(cat config/live_ckpt)"); CHAMP=${CHAMP#ckpt-head-}
LOG(){ printf '[hfpend %s] %s\n' "$(date -Is)" "$*"; }
LOG "champion (stays at repo root): $CHAMP"

HAVE=$(python3 -c "
from huggingface_hub import HfApi
print('\n'.join(HfApi().list_repo_files('$REPO')))" 2>/dev/null)
[ -n "$HAVE" ] || { LOG "cannot list repo -- aborting rather than blind-uploading"; exit 1; }

for d in "$HOME"/model-backups/heads/*/; do
    a=$(basename "$d")
    [ -s "$d/mtp_trained.safetensors" ] || { LOG "$a: no source, skip"; continue; }
    if [ "$a" = "$CHAMP" ]; then pre=""; else pre="arms/$a/"; fi
    for f in mtp_trained.safetensors head_card.json train_metrics.json eval.log; do
        [ -s "$d/$f" ] || continue
        grep -qxF "$pre$f" <<<"$HAVE" && continue
        LOG "uploading $pre$f ($(du -h "$d/$f"|cut -f1))"
        timeout 3600 hf upload "$REPO" "$d/$f" "$pre$f" --repo-type model \
            --commit-message "archive $a: $f" >/dev/null 2>&1 \
            && LOG "  ok" || LOG "  !! FAILED $pre$f"
    done
done
LOG "pending upload complete"
