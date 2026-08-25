#!/usr/bin/env bash
# chain_corpus.sh — train the winning recipe on a CORPUS built for the deployment distribution.
#
# WHY THIS OUTRANKS MORE HYPERPARAMETER ARMS. Ten arms at block 5 put their top five within 1.3 %
# of each other, and the exchange rate is ~13.8 tok/s per unit tau -- so an excellent further arm
# is worth about +1.5 tok/s and most are worth nothing. The knobs are spent: ce/tv swept three
# ways, beta bracketed on both sides, HASS retired, the confidence term retired.
#
# Three independent results all say the CORPUS is the binding constraint, not the knobs:
#   - the release rule showed ALL fine-tuning costs long_context (5.00 -> 4.00 unanchored,
#     -> 4.53 anchored). That is composition, and no knob fixes it.
#   - HASS failed because on 1,536 sequences free-running compounds the head's own errors faster
#     than it teaches robustness -- data-limited.
#   - draft-head-finetuning.md 5 prices the recipe as data-limited at this size.
#
# WHAT IS DIFFERENT ABOUT THIS CORPUS. 3,071 prompts against 1,536, and 72 % reconstructive rather
# than balanced 8-way. Balance was the correct answer to session 1's single-domain corpus and it is
# NOT the deployment distribution: an agentic coding harness is long context, tool/JSON format,
# file edits and multi-turn. The weight is placed where the measured deficit is -- long_context
# (-0.47 against the stock head) and agentic_format (-0.25) -- and NOT on code_edit, which passed
# its floor and whose pool of 249 usable prompts would otherwise cap the whole corpus.
#
# SEQUENCE LENGTH IS PROMPT + NGEN, AND THE PROMPT HALF IS AT ITS SOURCE CEILING. Measured on
# dolly.jsonl, the long_context source: context fields run p50 130 words, p90 365, p99 1277,
# max 3796. The 1280-token cap binds only on the top ~10 %, so no reweighting extracts 4-12k
# prompts from this dataset -- genuinely long PROMPTS need a different source, and that is a
# roadmap item (ROADMAP.md), not a knob.
#
# The GENERATED half has no such ceiling, and it is the better half to lengthen. NGEN 512 -> 1024
# doubles the context depth every sampled position sees, and the context it adds is the model's OWN
# OUTPUT -- which is precisely what an agentic harness reconstructs from: its prior turns and tool
# results, not a pasted document. Cost is wall clock only: ~32 h of generation instead of ~16, and
# peak capture per 491-sequence chunk goes 8.6 -> 17.2 GB against ~106 GB free
# -- which holds ONLY because each consumed chunk is deleted (see the S5_KEEP_CAP note below).
set -uo pipefail
cd "$(dirname "$0")/.."
LOG(){ printf '[corpus %s] %s\n' "$(date -Is)" "$*"; }
S5=/home/patrickd/s5-capture
NAME=agentic-p25-b0.1

LOG "waiting for the anchor-shape arms to finish"
while systemctl --user is-active --quiet dsv4-p25b; do sleep 120; done
LOG "p25b done; anchor-shape results:"; tail -3 HEAD_REGISTRY.md

[ -d "$HOME/model-backups/heads/$NAME" ] && { LOG "$NAME already archived"; exit 0; }

# The WINNING recipe, unchanged, so the corpus is the only variable. That is the whole point:
# one change per measurement, and here the change is the data.
LOG "arm $NAME: winning recipe (deficit, beta 0.1, anchor_pow 1) on the agentic corpus"
LOG "pass 1 generates 3071 sequences x 1024 tokens on-box -- the long pole, ~32 h, no tokens"
# S5_KEEP_CAP is DELIBERATELY UNSET HERE, unlike every s3recap-derived arm above it. Retention is
# right when a LATER arm reuses the capture -- autopilot.sh symlinks s3recap/c$ci/cap into each new
# arm, so those 9.7 GB chunks earn their disk many times over. NOTHING reuses this corpus: it is a
# one-off 3,071-prompt agentic set at NGEN 1024, and its chunks are 19 GB, not 9.7. Keeping all
# seven costs 133 GB to serve exactly one consumer. Copying the flag down from the s3 chains is what
# killed the 2026-08-25 run at chunk 3; the header note below reasons about PER-CHUNK peak, which is
# only true with the flag off. Peak is now one chunk, as that note assumes.
S5_HOLDOUT=64 S5_CHUNK=491 S5_BLOCK=5 \
S5_INCUMBENT_TAU="$(cat evidence/baseline_tau.value 2>/dev/null)" \
S5_ACE=0.1 S5_ATV=0.9 S5_DEFICIT=1 S5_BETA=0.1 S5_ANCHOR_POW=1.0 \
    bash scripts/s5_session_auto.sh "$NAME" "$S5/mixed_prompts_agentic.txt" 3071 1024 \
    || LOG "corpus arm failed (rc=$?)"

LOG "release rule for $NAME:"
python3 tools/release_rule.py "evidence/${NAME}_eval.log" \
    --run0 evidence/run0_tau_blk5.log --incumbent evidence/baseline_tau_blk5.log 2>&1 || true
LOG "registry:"; tail -2 HEAD_REGISTRY.md
LOG "corpus chain complete:"
