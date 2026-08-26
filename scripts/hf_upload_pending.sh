#!/usr/bin/env bash
# hf_upload_pending.sh — upload only the heads that are not on HuggingFace yet.
# Idempotent and resumable: it asks the repo what it already has, so re-running costs one API call
# per head and re-uploads nothing. Safe to run while the GPU is capturing -- this is network and
# disk, never the GPU, and capture is not a scored measurement.
set -uo pipefail
HF_FORCE_ARMS="${HF_FORCE_ARMS:-}"
HF_SHARDS_ONLY="${HF_SHARDS_ONLY:-shipped-dspark-0731reap}"
cd "$(dirname "$0")/.."
REPO=patrickbdevaney/dspark-mtp-draft-head-s3recap-p25-b0.1
CHAMP=$(basename "$(cat config/live_ckpt)"); CHAMP=${CHAMP#ckpt-head-}
LOG(){ printf '[hfpend %s] %s\n' "$(date -Is)" "$*"; }
LOG "champion (stays at repo root): $CHAMP"
BROKEN=""

HAVE=$(python3 -c "
from huggingface_hub import HfApi
print('\n'.join(HfApi().list_repo_files('$REPO')))" 2>/dev/null)
[ -n "$HAVE" ] || { LOG "cannot list repo -- aborting rather than blind-uploading"; exit 1; }

for d in "$HOME"/model-backups/heads/*/; do
    a=$(basename "$d")
    # NOT A ROUTINE SKIP. An archived head with no BF16 master is a BROKEN ARCHIVE, not a head that
    # happens to have nothing to upload -- promote_head.py:archive() copies whatever sits in --head, so
    # when --head is the STAGED dir (FP8 shards) the master is never archived and this loop silently
    # walks past it. On 2026-08-26 that had hidden the three most recent arms from HuggingFace,
    # including `auto-ace0p5`, the highest-tau head ever measured on this box. Silence is what made it
    # survive; say it loudly and fail the run so the exit status carries it.
    # SHARDS_ONLY names have no BF16 master BY NATURE, not by bug: `shipped-dspark-0731reap` is the
    # stock head as it came in the checkpoint, never fine-tuned, so no trainer ever wrote a master for
    # it. It is also the only copy of the rollback baseline. Let it through to upload its card; only
    # a head that SHOULD have a master and does not is a broken archive.
    if [ ! -s "$d/mtp_trained.safetensors" ]; then
        case " ${HF_SHARDS_ONLY} " in
            *" $a "*) LOG "$a: shards-only baseline, no BF16 master expected" ;;
            *) LOG "$a: !! BROKEN ARCHIVE -- no mtp_trained.safetensors. Back-fill from the arm's trained/ dir."
               BROKEN="$BROKEN $a"; continue ;;
        esac
    fi
    if [ "$a" = "$CHAMP" ]; then pre=""; else pre="arms/$a/"; fi
    # HF_FORCE_ARMS: space- or comma-separated arm names whose files are re-uploaded even though the
    # path already exists on the repo. Existence is normally a sound proxy for "already uploaded",
    # but it is NOT when a path's CONTENT has been superseded -- e.g. a partial head archived under
    # a name that a later, complete run then reuses. Without this, the finished head is silently
    # skipped and the repo keeps serving the partial under the good name (2026-08-25: the 3-of-7
    # agentic head was archived and uploaded as `agentic-p25-b0.1` before the run was finished).
    force=0
    case " ${HF_FORCE_ARMS//,/ } " in *" $a "*) force=1; LOG "$a: FORCED re-upload (content superseded)";; esac
    for f in mtp_trained.safetensors head_card.json train_metrics.json eval.log; do
        [ -s "$d/$f" ] || continue
        if [ "$force" = 0 ] && grep -qxF "$pre$f" <<<"$HAVE"; then continue; fi
        LOG "uploading $pre$f ($(du -h "$d/$f"|cut -f1))"
        timeout 3600 hf upload "$REPO" "$d/$f" "$pre$f" --repo-type model \
            --commit-message "archive $a: $f" >/dev/null 2>&1 \
            && LOG "  ok" || LOG "  !! FAILED $pre$f"
    done
done
if [ -n "$BROKEN" ]; then
    LOG "INCOMPLETE: these archives have no BF16 master and were NOT uploaded:$BROKEN"
    LOG "pending upload finished with broken archives -- exit 3"
    exit 3
fi
LOG "pending upload complete"
