#!/usr/bin/env bash
# with_model_lock.sh — only ONE process may hold the model resident at a time.
#
# This exists because of a reboot on 2026-08-13 at 18:01. The profiled server was left resident at
# seqmax 32768 (~100 GiB) and `gate_engine` was started on top of it, which loads another ~100 GiB.
# Two model loads on a 122 GiB box is not an out-of-memory error here -- the Tegra unified-memory
# path takes the WHOLE MACHINE down, with no oom-kill line in dmesg. That is the fourth reboot of
# this project and the first one caused purely by starting a second process while the first was
# still up, so a checklist is not sufficient: it has to be impossible.
#
# `flock` on a fixed lockfile. Whoever holds it owns the GPU; anyone else fails immediately with a
# message naming the holder, instead of loading 100 GiB and taking the box with it.
#
#   scripts/with_model_lock.sh ./build/gate_engine
#   scripts/with_model_lock.sh env SEQMAX=32768 bash scripts/serve.sh
#
# WAIT=1 blocks until the lock frees instead of failing.
set -u
cd "$(dirname "$0")/.."
LOCK="${MODEL_LOCK:-/tmp/dsv4-model.lock}"
touch "$LOCK" 2>/dev/null || true

exec 9>"$LOCK"
FLAGS="-n"
[ "${WAIT:-0}" = "1" ] && FLAGS=""

if ! flock $FLAGS 9; then
  holder=$(ps -eo pid,etime,cmd | grep -E "[g]ate_|[b]uild/dsv4-server" | head -1)
  echo "[model-lock] REFUSED: another process already holds the model resident."
  echo "[model-lock] holder: ${holder:-<unknown, but the lock is held>}"
  echo "[model-lock] Loading a second copy takes this box DOWN (reboot 2026-08-13 18:01)."
  echo "[model-lock] Stop the holder first, or run with WAIT=1 to queue behind it."
  exit 1
fi

# Belt and braces: the lock can only be as good as its coverage, so also refuse if a model process
# is somehow resident without holding it (e.g. started before this script existed).
resident=$(ps -eo rss,cmd | grep -E "[g]ate_engine|[b]uild/dsv4-server" | awk '$1 > 4194304' | head -1)
if [ -n "$resident" ]; then
  echo "[model-lock] REFUSED: a model is already resident without the lock:"
  echo "[model-lock]   $resident"
  exit 1
fi

echo "[model-lock] acquired ($$) — $(free -g | awk '/Mem:/{print $7}') GiB available"
exec "$@"
