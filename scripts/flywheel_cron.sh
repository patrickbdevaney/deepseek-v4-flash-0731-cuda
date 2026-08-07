#!/usr/bin/env bash
# flywheel_cron.sh — the crontab entry point. cron gives a near-empty environment; `claude` lives in
# ~/.local/bin and needs the user's PATH and credentials, so source the profile before dispatching.
export HOME=/home/patrickd
[ -f "$HOME/.profile" ] && . "$HOME/.profile" >/dev/null 2>&1
[ -f "$HOME/.bashrc" ]  && . "$HOME/.bashrc"  >/dev/null 2>&1
export PATH="$HOME/.local/bin:/usr/local/cuda-13.0/bin:$PATH"
# The timer can fire while the script is being edited. Running a half-written file produced
# "unexpected EOF while looking for matching quote" and a lost tick; parse it first, and snapshot it
# so an edit landing mid-cycle cannot change what is executing.
SRC="$HOME/deepseek-v4-flash-0731-cuda/scripts/flywheel.sh"
SNAP="$(mktemp /tmp/flywheel-XXXXXX.sh)"
cp "$SRC" "$SNAP"
if ! bash -n "$SNAP" 2>>"$HOME/flywheel_cron.log"; then
    echo "[flywheel $(date -Is)] flywheel.sh does not parse (mid-edit?); skipping this tick" \
        >> "$HOME/flywheel_cron.log"
    rm -f "$SNAP"; exit 0
fi
trap 'rm -f "$SNAP"' EXIT
exec /usr/bin/flock -n /tmp/dsv4-flywheel-cron.lock \
     bash "$SNAP" >> "$HOME/flywheel_cron.log" 2>&1
