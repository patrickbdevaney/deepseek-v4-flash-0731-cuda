#!/usr/bin/env bash
# supervise_pipeline.sh — make the draft-head run actually finish.
#
# WHAT WAS MISSING. dsv4-s3recap and dsv4-chain are one-shot units: any stage failure exits and the
# whole programme stops until a human looks. This run has already proved that matters -- the first
# attempt died in the equivalence gate at 13:42 and sat idle for three hours.
#
# WHY A SUPERVISOR AND NOT Restart=on-failure. Adding Restart= to a unit requires recreating it, and
# dsv4-s3recap is mid-capture: recreating it now would discard the chunk in flight (a chunk without
# manifest.jsonl re-captures from scratch, ~40 min). A separate watcher costs nothing and disturbs
# nothing.
#
# WHY RETRYING IS SAFE. s5_session.sh is resumable BY CONSTRUCTION and this run is configured to
# stay that way: a chunk whose trained weights exist is skipped, a chunk whose capture manifest
# exists is not re-captured, and S5_KEEP_CAP=1 keeps the captures. So a retry resumes at the failed
# stage rather than restarting the session. That is checkpointing -- it is already in the design;
# what was missing was something to USE it.
#
# WHY IT IS BOUNDED. A deterministic failure retried forever is a busy-loop that looks like
# progress. Each unit gets MAX_TRIES, and a fingerprint of the last failure is compared with the
# previous one: the same error twice in a row is not a transient, so it stops and says so.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG(){ printf '[supervise %s] %s\n' "$(date -Is)" "$*"; }
MAX_TRIES="${MAX_TRIES:-3}"
ARCHIVE="$HOME/model-backups/heads/s3recap"

fingerprint(){  # last line that looks like a failure, normalised
    tail -80 "$1" 2>/dev/null | grep -iE 'HALT|FAIL|Error|Traceback|rc=[1-9]' | tail -1 |
        sed 's/[0-9]\{2\}:[0-9]\{2\}:[0-9]\{2\}//g; s/[0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\}//g'
}

restart_unit(){ # $1 unit, $2 script, $3 logfile
    systemctl --user reset-failed "$1" 2>/dev/null
    systemd-run --user --unit="$1" --working-directory="$PWD" --property=Restart=no \
        --property=StandardOutput=append:$PWD/$3 --property=StandardError=append:$PWD/$3 \
        bash "$2" >/dev/null 2>&1
}

declare -A TRIES=() LASTFP=()
watch_unit(){ # $1 unit, $2 script, $3 logfile, $4 "done test" command
    local u="$1" sc="$2" lg="$3" donetest="$4"
    if eval "$donetest"; then return 0; fi
    if systemctl --user is-active --quiet "$u"; then return 1; fi
    # inactive and not done -> it failed or was never started
    local fp; fp=$(fingerprint "$lg")
    local n=${TRIES[$u]:-0}
    if [ "$n" -ge "$MAX_TRIES" ]; then
        LOG "$u: $n attempts exhausted; STOPPING. Last failure: ${fp:-<none found>}"; return 2
    fi
    if [ -n "$fp" ] && [ "$fp" = "${LASTFP[$u]:-}" ]; then
        LOG "$u: identical failure twice -- not a transient. STOPPING. $fp"; return 2
    fi
    LASTFP[$u]="$fp"; TRIES[$u]=$((n+1))
    LOG "$u: attempt $((n+1))/$MAX_TRIES after failure: ${fp:-<none found>}"
    LOG "$u: resuming (trained chunks and retained captures are skipped, not redone)"
    restart_unit "$u" "$sc" "$lg"
    return 1
}

LOG "supervising dsv4-s3recap then dsv4-chain (max $MAX_TRIES attempts each)"
while :; do
    watch_unit dsv4-s3recap scripts/session_s3recap.sh evidence/chain/s3recap.log \
        '[ -d "$ARCHIVE" ]'
    case $? in 2) exit 1;; 1) sleep 120; continue;; esac

    # The control is done. From here the chain owns the run; it is finished once the eval battery
    # is up, which is the last thing it starts.
    # The chain's completion marker, NOT the eval battery. The chain used to end by starting the
    # battery; it now stops after staging so the remaining decode levers can run first. Watching
    # for the battery here would mean a chain that SUCCEEDED never looks done, and the supervisor
    # would restart finished work.
    watch_unit dsv4-chain scripts/chain_after_s3recap.sh evidence/chain/chain.log \
        'grep -q "chain complete:" evidence/chain/chain.log 2>/dev/null'
    case $? in 2) exit 1;; 0) LOG "chain complete (arms measured, best head staged); supervision complete"; exit 0;; esac
    sleep 120
done
