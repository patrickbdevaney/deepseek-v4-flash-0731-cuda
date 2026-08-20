#!/usr/bin/env bash
# stage_head.sh — make a PROMOTED draft head the one the server actually runs, without ever
# touching the checkpoint.
#
# THE GAP THIS CLOSES. `tools/promote_head.py` archives a winning head to
# ~/model-backups/heads/<name>/ and writes it into HEAD_REGISTRY.md. It deliberately never writes
# the live checkpoint — promotion is a judgement about weights, not a deployment. The consequence
# was that s3 sat promoted and measured (tau 3.8438) for eight days while every server start loaded
# the shipped head, because nothing bridged the two. A head nobody serves is not a speedup.
#
# WHY A SYMLINK FARM AND NOT A --head FLAG IN THE SERVER.
#   * The engine's own comment (src/engine.cu, the mtp block) is the argument: the 0731-REAP
#     checkpoint EMBEDS mtp.0/1/2 in shards 46-48, and a second WeightStore "would duplicate
#     ~6.5 GiB against ~16 GiB of headroom". A --head flag adds a head mmap on top of the base head
#     mmap the main store already made. A staged dir REPLACES those three shards, so the resident
#     set is bit-for-bit the production one.
#   * It keeps the binary identical across the A/B. The only difference between the two arms is
#     three shard files, which is the cleanest comparison this deployment can be given.
#   * It costs no disk. 45 shards, the tokenizer, config and the encoding dir are symlinks to the
#     read-only checkpoint; only the head's three shards point elsewhere.
#
# The checkpoint itself is never written. It is mode 0444 in a read-only directory and it stays
# that way; this builds a NEW directory beside it.
#
#   bash scripts/stage_head.sh s3                 # -> ~/models/ckpt-head-s3, verified, not live
#   bash scripts/stage_head.sh s3 --activate      # ...and point config/live_ckpt at it
#
# --activate only rewrites the tracked pointer file `config/live_ckpt`, which scripts/serve.sh
# reads. It does NOT restart anything: bringing the engine down is a ~10-15 minute reload of a
# 101 GiB checkpoint and is documented in this repo as the most expensive class of mistake, so it
# is always a separate, deliberate act.
set -euo pipefail
cd "$(dirname "$0")/.."

BASE="${BASE_CKPT:-/home/patrickd/models/DeepSeek-V4-Flash-0731-REAP}"
STORE="${HEAD_STORE:-$HOME/model-backups/heads}"
NAME="${1:-}"
ACTIVATE=0
[ "${2:-}" = "--activate" ] && ACTIVATE=1

if [ -z "$NAME" ]; then
  echo "usage: $0 <head-name-from-HEAD_REGISTRY.md> [--activate]" >&2
  echo "available:" >&2; ls "$STORE" 2>/dev/null | sed 's/^/  /' >&2
  exit 2
fi
HEAD="$STORE/$NAME"
OUT="${OUT_DIR:-$HOME/models/ckpt-head-$NAME}"

[ -d "$BASE" ] || { echo "no base checkpoint at $BASE" >&2; exit 1; }
[ -d "$HEAD" ] || { echo "no archived head at $HEAD" >&2; exit 1; }
[ -f "$HEAD/model.safetensors.index.json" ] || { echo "$HEAD has no index.json — not a loadable head" >&2; exit 1; }

# ONLY PROMOTED HEADS. The registry is the ledger; a head that was archived but refused promotion
# is a data point, not a deployment candidate, and staging one would quietly ship a head that lost.
if ! grep -qE "^\| \`$NAME\` \|.*\| *(PROMOTED|baseline) *\|" HEAD_REGISTRY.md; then
  echo "REFUSING: '$NAME' is not PROMOTED in HEAD_REGISTRY.md." >&2
  grep -E "^\| \`$NAME\`" HEAD_REGISTRY.md >&2 || echo "  (not in the registry at all)" >&2
  exit 1
fi

# REFUSE TO WRITE ANYTHING THAT IS NOT ALREADY A SYMLINK FARM. This script rm -rf's OUT to rebuild
# it. A typo in OUT_DIR pointing at a real checkpoint would delete 100 GiB of weights that take
# hours to re-download. A farm is all symlinks; a checkpoint is not. Check before deleting.
if [ -e "$OUT" ]; then
  if [ -n "$(find "$OUT" -maxdepth 1 ! -type l ! -name "$(basename "$OUT")" -print -quit)" ]; then
    echo "REFUSING: $OUT exists and contains real files, not just symlinks." >&2
    echo "  This script only rebuilds symlink farms. Move it aside yourself if you meant it." >&2
    exit 1
  fi
  rm -rf "$OUT"
fi
mkdir -p "$OUT"

# every entry of the base checkpoint, linked
n_link=0
for p in "$BASE"/* "$BASE"/.[!.]*; do
  [ -e "$p" ] || continue
  ln -s "$p" "$OUT/$(basename "$p")"
  n_link=$((n_link + 1))
done

# then the head's shards on top. The head archive carries a PARTIAL index (mtp tensors only) that
# maps to the same shard filenames as the base, so the base index — already linked above — resolves
# them correctly and does not need replacing. Link only the files that index actually names.
HEAD_FILES=$(python3 -c "
import json,sys
wm=json.load(open('$HEAD/model.safetensors.index.json'))['weight_map']
print(' '.join(sorted(set(wm.values()))))")
n_head=0
for f in $HEAD_FILES; do
  [ -f "$HEAD/$f" ] || { echo "head index names $f but it is not in $HEAD" >&2; exit 1; }
  rm -f "$OUT/$f"
  ln -s "$HEAD/$f" "$OUT/$f"
  n_head=$((n_head + 1))
done
echo "[stage_head] $OUT: $n_link link(s) from base, $n_head replaced by head '$NAME'"

# PROVE IT BEFORE ANYONE LOADS IT. Seconds on the CPU against ~10 minutes of weight load.
python3 tools/verify_staged_ckpt.py --staged "$OUT" --base "$BASE" --head-store "$HEAD"

if [ "$ACTIVATE" = "1" ]; then
  mkdir -p config
  printf '%s\n' "$OUT" > config/live_ckpt
  echo "[stage_head] config/live_ckpt -> $OUT"
  echo "[stage_head] NOT restarted. The next server start picks this up:  bash scripts/run_server.sh"
else
  echo "[stage_head] staged but NOT live. Activate with: $0 $NAME --activate"
fi
