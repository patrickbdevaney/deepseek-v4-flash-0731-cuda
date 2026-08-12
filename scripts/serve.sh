#!/usr/bin/env bash
# Start the production server. Runs the CPU-only gates first (seconds, no GPU, no checkpoint) so a
# broken tokenizer or encoder is caught before a ~10-minute weight load, not after it.
set -e; cd "$(dirname "$0")/.."

CKPT="${CKPT:-/home/patrickd/models/DeepSeek-V4-Flash-0731-REAP}"
PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"
SEQMAX="${SEQMAX:-8192}"

[ -x build/dsv4-server ] || { echo "build/dsv4-server missing — run scripts/build_server.sh"; exit 1; }

for g in gate_tokenizer gate_encoding gate_api gate_stream; do
  [ -x "build/$g" ] || { echo "build/$g missing — run scripts/build_server.sh"; exit 1; }
  if ! out=$(./build/$g 2>&1); then
    echo "$out"; echo "PREFLIGHT FAILED: $g"; exit 1
  fi
  echo "  preflight $g ok"
done

exec ./build/dsv4-server --ckpt "$CKPT" --host "$HOST" --port "$PORT" --seqmax "$SEQMAX" "$@"
