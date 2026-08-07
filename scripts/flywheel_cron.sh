#!/usr/bin/env bash
# flywheel_cron.sh — the crontab entry point. cron gives a near-empty environment; `claude` lives in
# ~/.local/bin and needs the user's PATH and credentials, so source the profile before dispatching.
export HOME=/home/patrickd
[ -f "$HOME/.profile" ] && . "$HOME/.profile" >/dev/null 2>&1
[ -f "$HOME/.bashrc" ]  && . "$HOME/.bashrc"  >/dev/null 2>&1
export PATH="$HOME/.local/bin:/usr/local/cuda-13.0/bin:$PATH"
exec /usr/bin/flock -n /tmp/dsv4-flywheel-cron.lock \
     bash "$HOME/deepseek-v4-flash-0731-cuda/scripts/flywheel.sh" \
     >> "$HOME/flywheel_cron.log" 2>&1
