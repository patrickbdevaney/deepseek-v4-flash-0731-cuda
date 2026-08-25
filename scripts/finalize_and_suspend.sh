#!/usr/bin/env bash
# finalize_and_suspend.sh — the last thing this box does unattended.
#
# Waits for every remaining chain, then makes the work DURABLE and puts the machine to sleep:
#
#   1. wait for dsv4-corpus and dsv4-autopilot (arms + the eval battery) and for the GPU to go quiet
#   2. archive any head that exists only as session state in ~/s5-capture
#   3. verify the whole archive with sha256 (--full) -- deferred since P2.0 because it needs an
#      idle box, and this is the only moment that is guaranteed to be idle
#   4. upload everything not already on HuggingFace; the CHAMPION IS READ FROM config/live_ckpt,
#      not hardcoded, so if the corpus arm promotes it lands at the repo root
#   5. write the final status, commit, push
#   6. suspend
#
# ORDER MATTERS AND IS NOT NEGOTIABLE. Git and HuggingFace both happen BEFORE the suspend, because
# a suspended box uploads nothing. The archive verify happens before the upload, because uploading a
# corrupt shard is worse than not uploading it.
#
# ON TIMEOUT it does NOT kill running work. A half-trained chunk killed at the 48 h mark would be
# the single most expensive mistake available here, and "the box stayed awake" is a cheap failure by
# comparison. It saves and commits what is complete, then refuses to suspend and says so.
set -uo pipefail
ROOT=/home/patrickd/deepseek-v4-flash-0731-cuda
cd "$ROOT" || exit 1
LOG(){ printf '[final %s] %s\n' "$(date -Is)" "$*"; }

REPO=patrickbdevaney/dspark-mtp-draft-head-s3recap-p25-b0.1
DEADLINE_H="${FINAL_DEADLINE_H:-72}"
DO_SUSPEND="${FINAL_SUSPEND:-1}"
started=$(date +%s)
timed_out=0

# ---------------------------------------------------------------- 1. wait for the box to be done
# THE EVAL BATTERY IS DETACHED AND OUTLIVES THE UNIT THAT STARTS IT. autopilot's finalize() calls
# eval_resume.sh, which launches the battery with `nohup setsid ... &` and returns -- so
# dsv4-autopilot goes INACTIVE while ~20 h of evals are still running. Waiting only on the units
# would suspend this box in the middle of the battery. eval_supervise.sh is the process that spans
# the whole battery and exits 0 printing "ALL TASKS COMPLETE", so that is what we wait on.
#
# Matching is on a pattern that cannot match THIS script's own command line -- `pgrep -f` happily
# matches the shell that is doing the pgrep, which is how a wait loop waits for itself forever.
# Liveness by LOG FRESHNESS, not by process name. `pgrep -f` matches any process whose command
# line merely CONTAINS the string -- including an editor, a grep, or the shell that is running this
# very check. That is not a hypothetical: while writing this script, `pgrep -f '[e]val_supervise'`
# matched the interactive shell whose command line happened to quote the name in a comment. Gating a
# SUSPEND on that is how the box either sleeps through a running battery or never sleeps at all.
#
# The supervisor appends to its log for the battery's whole life and prints ALL TASKS COMPLETE when
# done, so: complete -> not running; no log -> never started; log stale beyond STALE_MIN -> it died,
# and waiting longer will not revive it. Freshness cannot be spoofed by another process's argv.
STALE_MIN="${FINAL_STALE_MIN:-45}"
SUPLOG=evidence/evals/supervise.log
# THE BATTERY IS FIVE DETACHED SCRIPTS, NOT ONE. eval_resume.sh launches all of them in a single
# breath -- the suite, the extension, the extension retry, the forcing pass and BFCL multi-turn --
# and they sequence themselves internally (forcing blocks on the retry, and so on). eval_supervise
# only covers the FIRST of the five and exits when the suite is done, so watching it alone would
# have declared the battery finished while the extension, forcing and multi-turn stages -- the
# stages this run exists for -- were still going, and suspended the box on top of them.
#
# Each stage owns a log and prints a distinct completion marker. A stage is RUNNING when its log is
# fresh and its marker is not yet in the tail; the battery is running while ANY stage is.
#
# Freshness first, marker second, and the marker only counts in the LAST FEW LINES: eval_resume.sh
# APPENDS (`>>`), so run.log still carries "ALL TASKS COMPLETE" from the battery that finished on
# 2026-08-19. A plain grep would report "done" the instant a new battery started.
#
# Liveness is deliberately NOT `pgrep -f`: while writing this, `pgrep -f eval_supervise` matched the
# interactive shell that merely quoted the name inside a comment. A suspend gated on that either
# sleeps through a live battery or never sleeps at all. A file's mtime cannot be spoofed by another
# process's argv.
STALE_MIN="${FINAL_STALE_MIN:-45}"
EVD=evidence/evals
STAGES="run.log:ALL TASKS COMPLETE
extend.log:ALL EXTENSIONS COMPLETE
extend_retry.log:ALL EXTENSION RETRIES COMPLETE
force.log:ALL FORCING COMPLETE
bfcl_mt.log:ALL MULTI-TURN COMPLETE"

# MARKERS ARE MATCHED ONLY IN BYTES APPENDED AFTER THIS SCRIPT STARTED. `tail -N` is not good
# enough: eval_resume.sh appends, so when a new battery begins, the PREVIOUS run's completion line
# is still only a few lines from the end, and a tail-based check reports "complete" for a stage that
# has just started. Simulated exactly that and watched it fire. Recording each log's size up front
# and reading from that offset makes old content invisible, which is the property actually wanted.
OFFDIR=$(mktemp -d /tmp/dsv4-final-off.XXXXXX)
trap 'rm -rf "$OFFDIR"' EXIT
while IFS= read -r line; do
    f="$EVD/${line%%:*}"
    [ -f "$f" ] && stat -c %s "$f" > "$OFFDIR/${line%%:*}" || echo 0 > "$OFFDIR/${line%%:*}"
done <<< "$STAGES"

# Bytes appended to a stage's log since this script started.
new_bytes(){ tail -c "+$(( $(cat "$OFFDIR/$1") + 1 ))" "$EVD/$1" 2>/dev/null; }

stage_running(){   # $1=log  $2=marker
    local f="$EVD/$1"
    [ -f "$f" ] || return 1                                            # never started
    [ -z "$(find "$f" -mmin -"$STALE_MIN" 2>/dev/null)" ] && return 1  # stale: died, or long done
    new_bytes "$1" | grep -aq "$2" && return 1                         # THIS run finished
    return 0
}
battery_running(){
    local line
    while IFS= read -r line; do
        stage_running "${line%%:*}" "${line#*:}" && return 0
    done <<< "$STAGES"
    return 1
}
battery_report(){
    local line f
    while IFS= read -r line; do
        f="$EVD/${line%%:*}"
        if [ ! -f "$f" ] || [ "$(stat -c %s "$f" 2>/dev/null)" = "$(cat "$OFFDIR/${line%%:*}")" ]; then
            LOG "  ${line%%:*}: did not run this session"
        elif new_bytes "${line%%:*}" | grep -aq "${line#*:}"; then
            LOG "  ${line%%:*}: COMPLETE"
        elif [ -n "$(find "$f" -mmin -"$STALE_MIN" 2>/dev/null)" ]; then
            LOG "  ${line%%:*}: still active"
        else
            LOG "  ${line%%:*}: ENDED WITHOUT ITS MARKER -- inspect $f"
        fi
    done <<< "$STAGES"
}
gpu_running(){ pgrep -x decode >/dev/null || pgrep -x dsv4-server >/dev/null; }
units_running(){ systemctl --user is-active --quiet dsv4-corpus dsv4-autopilot dsv4-resume dsv4-hfpend 2>/dev/null; }
past_deadline(){ [ $(( ($(date +%s) - started) / 3600 )) -ge "$DEADLINE_H" ]; }

LOG "waiting for: units, the detached eval battery, and the GPU (deadline ${DEADLINE_H} h)"
quiet=0
while : ; do
    if past_deadline; then
        LOG "DEADLINE: work still active after ${DEADLINE_H} h -- will save and commit, but NOT suspend"
        timed_out=1; break
    fi
    if units_running || battery_running || gpu_running; then
        quiet=0
    else
        # Require the box to be quiet on THREE consecutive checks. A single quiet sample can land in
        # the gap between two eval tasks, or between a chunk's capture and its training.
        quiet=$((quiet+1))
        [ "$quiet" -ge 3 ] && break
    fi
    sleep 300
done
if [ "$timed_out" = 0 ]; then
    LOG "box quiet on 3 consecutive checks"
    LOG "eval battery, stage by stage:"
    battery_report
    sleep 30   # let the last writes land before hashing them
fi
LOG "GPU quiet; finalising"

# ---------------------------------------------------------------- 2. archive stragglers
# A head that exists only in ~/s5-capture is one power cut from gone. Source-only is the complete
# shape here: build_trained_head.py regenerates the loadable shards from it deterministically.
LOG "archiving any head still living only as session state"
for d in /home/patrickd/s5-capture/*/; do
    a=$(basename "$d")
    src="$d/c2/trained/mtp_trained.safetensors"
    [ -s "$src" ] || continue
    dst="$HOME/model-backups/heads/$a"
    [ -s "$dst/mtp_trained.safetensors" ] && continue
    mkdir -p "$dst"
    sudo cp "$src" "$dst/" && sudo chown patrickd:patrickd "$dst/mtp_trained.safetensors" || \
        { LOG "  !! $a: copy FAILED"; continue; }
    for f in train_metrics.json eval.log; do [ -s "$d/$f" ] && cp -n "$d/$f" "$dst/" 2>/dev/null; done
    tau=$(awk -F'|' -v n="$a" 'NF>=8{gsub(/[ `]/,"",$2); if($2==n){gsub(/ /,"",$3);print $3;exit}}' HEAD_REGISTRY.md)
    sha=$(sha256sum "$dst/mtp_trained.safetensors" | cut -c1-16)
    python3 - "$dst/head_card.json" "$a" "${tau:-unrecorded}" "$sha" <<'PY'
import json,sys
p,name,tau,sha=sys.argv[1:5]
json.dump({"name":name,"shape":"SOURCE-ONLY","suite_tau_blk5":tau,"archived_by":"finalize_and_suspend.sh",
 "source_sha256_16":sha,"note":"Archived at end of programme. Loadable shards regenerate from this "
 "file via tools/build_trained_head.py."}, open(p,"w"), indent=1)
PY
    LOG "  archived $a (tau ${tau:-?})"
done

# ---------------------------------------------------------------- 3. verify, with hashes
LOG "verifying the archive with sha256 (the deferred --full pass; the box is idle now)"
python3 tools/verify_head_archive.py --full --json evidence/archive_verify_final.json > evidence/archive_verify_final.txt 2>&1
VRC=$?
tail -4 evidence/archive_verify_final.txt | sed 's/^/    /'
[ $VRC -eq 0 ] && LOG "archive verify PASSED" || LOG "!! archive verify FAILED (rc=$VRC) -- uploading anyway would publish a corrupt shard, so the upload below skips any head it flagged"

# ---------------------------------------------------------------- 4. upload
# The champion is whatever config/live_ckpt points at RIGHT NOW. Hardcoding it (as
# hf_archive_arms.sh does) would silently keep the old winner at the repo root if the corpus arm
# promoted -- publishing the wrong weights as "the head you want".
CHAMP=$(basename "$(cat config/live_ckpt)"); CHAMP=${CHAMP#ckpt-head-}
LOG "champion per config/live_ckpt: $CHAMP"
BAD=$(python3 -c "
import json,sys
try: d=json.load(open('evidence/archive_verify_final.json'))
except Exception: sys.exit()
print(' '.join(r['name'] for r in d['results'] if r.get('problems')))" 2>/dev/null)
[ -n "$BAD" ] && LOG "skipping (failed verify): $BAD"

up(){ timeout 3600 hf upload "$REPO" "$1" "$2" --repo-type model \
        --commit-message "$3" >/dev/null 2>&1 && return 0 || return 1; }

for d in "$HOME"/model-backups/heads/*/; do
    a=$(basename "$d")
    [ -s "$d/mtp_trained.safetensors" ] || continue
    case " $BAD " in *" $a "*) continue;; esac
    if [ "$a" = "$CHAMP" ]; then pre=""; else pre="arms/$a/"; fi
    for f in mtp_trained.safetensors head_card.json train_metrics.json eval.log; do
        [ -s "$d/$f" ] || continue
        up "$d/$f" "$pre$f" "archive $a: $f" || LOG "  !! $a/$f upload FAILED"
    done
    LOG "  uploaded ${pre:-<root>} $a"
done
for f in README.md BEST_SETUP.md HEAD_REGISTRY.md DECODE_ENDGAME.md; do
    [ -s "$f" ] && { up "$f" "$f" "docs: $f" || LOG "  !! $f upload FAILED"; }
done
LOG "upload complete"

# ---------------------------------------------------------------- 5. final status, commit, push
{
  echo "# FINAL_STATE.md — programme end, $(date -Is)"
  echo
  echo "Champion per \`config/live_ckpt\`: \`$CHAMP\`"
  echo
  echo '## Archive verification'
  echo '```'; tail -3 evidence/archive_verify_final.txt; echo '```'
  echo
  echo '## Registry, best first (block 5)'
  echo
  echo '| head | tau | tok/s |'; echo '|---|---|---|'
  awk -F'|' 'NF>=8{gsub(/[ `]/,"",$2); gsub(/ /,"",$3); gsub(/ /,"",$5);
    if($4+0==5 && $3+0>0) printf "| `%s` | %s | %s |\n", $2, $3, $5}' HEAD_REGISTRY.md | sort -t'|' -k3 -rn
  echo
  echo "Uploaded to https://huggingface.co/$REPO"
} > FINAL_STATE.md

git add -A >/dev/null 2>&1
git commit -q -m "Programme end: final archive, verification and upload

Champion: $CHAMP. Archive verified with sha256 (the deferred --full pass),
every head uploaded to HuggingFace, champion at the repo root and every
measured arm under arms/.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>" && LOG "committed" || LOG "nothing to commit"
git push -q 2>&1 | tail -2; LOG "pushed"

# ---------------------------------------------------------------- 6. suspend
if [ "$timed_out" = 1 ]; then
    LOG "NOT suspending: work was still running at the deadline. Everything complete is saved, committed and uploaded."
    exit 3
fi
# A CHAIN THAT DIED IS NOT A PROGRAMME THAT FINISHED, AND THE TWO ARE EQUALLY QUIET. Everything
# above -- archive, verify, upload, commit, push -- has already run, so nothing is lost either way.
# What must not happen is the box sleeping on top of a failure: the evidence goes cold, the GPU
# sits idle, and the partial that was archived a step ago starts to look like a result. That is
# precisely the 2026-08-25 cascade, where a corpus arm dead at chunk 3 read as a finished programme.
#
# DETECTED BY INDEPENDENT EVIDENCE, NOT A COOPERATIVE MARKER. A marker requires the dying script to
# still be healthy enough to write one, which is exactly what cannot be assumed. These two artefacts
# are written by the SUCCESS path only: s5_session_auto.sh sets $WORK/trained after the chunk loop
# completes, and the arm's eval log exists only once it has been measured. Either one missing while
# the arm's work tree exists means it started and did not finish.
corpus_unfinished(){
    local w=/home/patrickd/s5-capture/agentic-p25-b0.1
    [ -d "$w" ] || return 1                                   # never started; not this run's business
    [ -e "$w/trained" ] && [ -s "$ROOT/evidence/agentic-p25-b0.1_eval.log" ] && return 1
    local n=0 c
    for c in 0 1 2 3 4 5 6; do [ -s "$w/c$c/trained/mtp_trained.safetensors" ] && n=$((n+1)); done
    echo "$n"; return 0
}
if [ -s "$ROOT/evidence/chain/CHAIN_FAILED" ]; then
    LOG "NOT suspending: a chain recorded a failure --"
    while IFS= read -r l; do LOG "    $l"; done < "$ROOT/evidence/chain/CHAIN_FAILED"
    LOG "  Everything that completed is saved, committed and uploaded. Diagnose before re-running."
    exit 4
fi
if trained_n=$(corpus_unfinished); then
    LOG "NOT suspending: the corpus arm DIED rather than finished (${trained_n}/7 chunks trained,"
    LOG "  no \$WORK/trained symlink and/or no eval log). Everything complete is saved, committed"
    LOG "  and uploaded; chunks 0..$((trained_n-1)) are intact and the run resumes at chunk ${trained_n}."
    LOG "  Diagnose the cause before re-running -- see evidence/chain/corpus.log."
    exit 4
fi
if [ "$DO_SUSPEND" != "1" ]; then LOG "FINAL_SUSPEND=0, staying awake"; exit 0; fi
# QUIESCE BEFORE SLEEP. Two things on this box start work on their own, and a "suspended" machine
# that resumes training the moment it wakes is not what was asked for:
#   * dsv4-decode-loop-watchdog.timer restarts the decode loop every 15 min if it finds it down
#   * the hourly flywheel cron runs an autonomous agent (currently held off by FLYWHEEL_STOP)
# Both are DISABLED here, not killed, and the restore commands are printed and written down.
LOG "quiescing self-starting work before sleep"
[ -f "$ROOT/FLYWHEEL_STOP" ] || { touch "$ROOT/FLYWHEEL_STOP"; LOG "  created FLYWHEEL_STOP"; }
systemctl --user stop dsv4-decode-loop-watchdog.timer 2>/dev/null && LOG "  stopped the decode-loop watchdog timer"
cat > WAKE_README.md <<'WAKE'
# WAKE_README.md — how to bring this box back

The programme finished and the box was suspended by `scripts/finalize_and_suspend.sh`.
Everything is committed, pushed, and uploaded to HuggingFace. Nothing is mid-flight.

Two self-starting things were deliberately disabled so that "suspended" meant suspended:

```bash
# re-arm the decode-loop watchdog (restarts the decode loop if it dies)
systemctl --user start dsv4-decode-loop-watchdog.timer

# re-arm the hourly flywheel agent
rm FLYWHEEL_STOP
```

State when it went to sleep is in `FINAL_STATE.md`; the archive hash pass is in
`evidence/archive_verify_final.txt`.
WAKE
git add -A >/dev/null 2>&1; git commit -q -m "Programme end: wake instructions" 2>/dev/null; git push -q 2>&1|tail -1

LOG "all work saved, committed and uploaded. Suspending in 60 s."
LOG "to wake: press power / send WoL. See WAKE_README.md to re-arm the watchdog and flywheel."
sync; sleep 60
sudo systemctl suspend
