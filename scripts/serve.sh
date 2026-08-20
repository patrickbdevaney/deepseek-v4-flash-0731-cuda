#!/usr/bin/env bash
# Start the production server. Runs the CPU-only gates first (seconds, no GPU, no checkpoint) so a
# broken tokenizer or encoder is caught before a ~10-minute weight load, not after it.
set -e; cd "$(dirname "$0")/.."

# WHICH CHECKPOINT THE SERVER SERVES IS A TRACKED DECISION, NOT AN AMBIENT ONE.
# Every server start in this repo goes through this script (run_server.sh, eval_resume.sh,
# eval_supervise.sh and deploy_staged_server.sh all exec it), so this line is the single place the
# engine's weights are chosen -- and it used to hardcode the base checkpoint. That is why the
# promoted `s3` draft head sat archived and unserved for eight days: promote_head.py deliberately
# only archives, and nothing else could name a different checkpoint.
#
# `config/live_ckpt` is that name, and it is IN GIT, so which head is in production is reviewable
# in the same diff as everything else instead of living in one operator's shell history. It is
# written by scripts/stage_head.sh --activate. If it names a directory that no longer exists we
# fall back to the base checkpoint LOUDLY rather than refusing to start -- a missing symlink farm
# must not be able to take the server down.
CKPT_DEFAULT=/home/patrickd/models/DeepSeek-V4-Flash-0731-REAP
if [ -z "${CKPT:-}" ] && [ -s config/live_ckpt ]; then
  PINNED=$(sed -n '1{s/[[:space:]]*$//;p}' config/live_ckpt)
  if [ -d "$PINNED" ]; then
    CKPT="$PINNED"
    echo "  live checkpoint pinned by config/live_ckpt -> $CKPT"
  else
    echo "  WARNING: config/live_ckpt names '$PINNED', which is not a directory."
    echo "  WARNING: falling back to $CKPT_DEFAULT — the promoted head is NOT being served."
  fi
fi
CKPT="${CKPT:-$CKPT_DEFAULT}"
PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"
SEQMAX="${SEQMAX:-8192}"
EXT_CHUNK="${EXT_CHUNK:-64}"

[ -x build/dsv4-server ] || { echo "build/dsv4-server missing — run scripts/build_server.sh"; exit 1; }

for g in gate_tokenizer gate_encoding gate_api gate_stream; do
  [ -x "build/$g" ] || { echo "build/$g missing — run scripts/build_server.sh"; exit 1; }
  if ! out=$(./build/$g 2>&1); then
    echo "$out"; echo "PREFLIGHT FAILED: $g"; exit 1
  fi
  echo "  preflight $g ok"
done

exec ./build/dsv4-server --ckpt "$CKPT" --host "$HOST" --port "$PORT" --seqmax "$SEQMAX" \
     --ext-chunk "$EXT_CHUNK" "$@"
