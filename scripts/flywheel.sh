#!/usr/bin/env bash
# flywheel.sh — ONE iteration of the decode optimisation loop, run headless.
#
# This is the mechanical executor for DECODE_FLYWHEEL.md. It is triggered on a timer, does exactly
# one lever, and stops. It is deliberately NOT a long-running agent: one lever per invocation means
# every iteration is independently reviewable, and a bad one costs one iteration.
#
# AUTHORITY THIS SCRIPT GRANTS THE AGENT (read this before enabling the timer):
#   - edit sources under this repo, build, run unit gates, run ONE full-model measurement
#   - commit LOCALLY, and run sudo (needed for drop_caches between model loads)
# AUTHORITY IT DOES NOT GRANT — enforced by `.claude/settings.json` deny rules, which are honoured
# even under bypassPermissions (verified empirically: sudo runs, `git push` is refused):
#   - `git push` / `git remote` (the observer publishes after review)
#   - `crontab` (the loop must not reschedule itself), shutdown/reboot/systemctl, mkfs/dd
#   - `curl`/`wget` (research goes through WebFetch, which is auditable in the transcript)
#   - any write under the model directory
#   - any edit to `.claude/settings.json` or `scripts/flywheel*.sh` — the loop cannot widen its
#     own permissions or disable its own guards
#
# PERMISSION MODE. The first cron-fired cycle ran under `acceptEdits` and produced 13 tool errors in
# ~20 calls: compound commands ("the following parts require approval"), reads of /proc (outside the
# allowed working directories), and outright "this command requires approval". An unattended agent
# cannot answer a prompt, so every one of those is a dead turn. bypassPermissions + a deny list
# moves the boundary from "ask about everything" to "refuse a named few", which is the only shape
# that works headless.
#
# HALT CONDITIONS — the loop stops itself and waits for a human:
#   - any unit gate fails
#   - a full-model run prints GATE FAIL
#   - a measured regression worse than 3% against the recorded baseline
#   - FLYWHEEL_STATE.json .halt == true
set -uo pipefail
# The cron wrapper runs a SNAPSHOT of this script from /tmp (so an edit cannot race a firing timer),
# which makes $(dirname "$0")/.. resolve to "/" and every path wrong -- the loop then halted every
# hour on `no FLYWHEEL_STATE.json` at "//FLYWHEEL_STATE.json". The root is passed explicitly.
cd "${FLYWHEEL_ROOT:-$(dirname "$0")/..}" || exit 1
ROOT="$PWD"
[ -f "$ROOT/FLYWHEEL_STATE.json" ] || { echo "[flywheel] wrong root: $ROOT"; exit 1; }
STATE="$ROOT/FLYWHEEL_STATE.json"
JOURNAL="$ROOT/FLYWHEEL_JOURNAL.md"
LOCK=/tmp/dsv4-flywheel.lock
MODEL="$HOME/models/DeepSeek-V4-Flash-0731-REAP"
TURNS="${FLYWHEEL_TURNS:-140}"

log(){ printf '[flywheel %s] %s\n' "$(date -Is)" "$*"; }
halt(){ log "HALT: $*"
        python3 - "$STATE" "$*" <<'PY'
import json,sys
p,r=sys.argv[1],sys.argv[2]
s=json.load(open(p)); s["halt"]=True; s["halt_reason"]=r
json.dump(s,open(p,"w"),indent=2)
PY
        exit 3; }

# ---- single instance -------------------------------------------------------------------------
exec 9>"$LOCK"
flock -n 9 || { log "another iteration is running; exiting"; exit 0; }

# ---- preflight -------------------------------------------------------------------------------
# Kill switch that needs no JSON edit and no running session: `touch FLYWHEEL_STOP`.
[ -f "$ROOT/FLYWHEEL_STOP" ] && { log "FLYWHEEL_STOP present; exiting"; exit 0; }
[ -f "$STATE" ] || halt "no FLYWHEEL_STATE.json"
python3 -c "import json;s=json.load(open('$STATE'));exit(1 if s.get('halt') else 0)" \
  || { log "state is halted: $(python3 -c "import json;print(json.load(open('$STATE')).get('halt_reason',''))")"; exit 0; }
# `pgrep -f build/decode` matches ANY process whose command line mentions the path -- a
# monitoring shell, a grep, an editor -- and a false positive makes the loop skip the cycle.
# `pgrep -x decode` matches the executable name, which is what we actually mean.
pgrep -x decode >/dev/null && { log "a full-model process is already running; exiting"; exit 0; }
[ -d "$MODEL" ] || halt "model dir missing: $MODEL"

# The 100.4 GiB checkpoint needs headroom; the page cache is reclaimable and run_model.sh refuses
# below ~105 GiB. Reclaim here rather than letting the agent discover it mid-iteration.
AVAIL=$(free -g | awk '/^Mem:/{print $7}')
if [ "$AVAIL" -lt 110 ]; then sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; sleep 2; fi
AVAIL=$(free -g | awk '/^Mem:/{print $7}')
[ "$AVAIL" -ge 105 ] || halt "only ${AVAIL} GiB available after reclaim; a full-model load needs ~105"

# Prove the environment before spending agent turns in it. Cycle 1 burned a whole iteration
# discovering Bash was not approved; that class of failure is checked here, once, cheaply.
if ! bash "$ROOT/scripts/flywheel_selftest.sh" > "$ROOT/.flywheel_selftest.txt" 2>&1; then
    tail -30 "$ROOT/.flywheel_selftest.txt"
    halt "environment selftest failed — see .flywheel_selftest.txt"
fi
log "selftest PASS"

CYCLE=$(python3 -c "import json;print(json.load(open('$STATE'))['cycle'])")
log "cycle $CYCLE starting; ${AVAIL} GiB free; HEAD $(git rev-parse --short HEAD)"
BEFORE=$(git rev-parse HEAD)

# ---- the iteration ---------------------------------------------------------------------------
PROMPT=$(cat <<'EOP'
You are the executor for the decode optimisation flywheel in this repository. Do EXACTLY ONE lever
this iteration, then stop. Another invocation follows; you do not need to finish everything.

READ FIRST, in this order:
  DECODE_FLYWHEEL.md   — the loop, the phases, the entry/stopping conditions, the ranking model
  FLYWHEEL_STATE.json  — machine state: current phase, queue, baseline, counters
  LOOP_LOG.md          — tail it. The "retired with a measurement" entries are binding: re-proposing
                         one of them is the single most expensive mistake available to you.

THEN follow the phase in FLYWHEEL_STATE.json:
  A EXPLOIT  — take queue[0]. Implement, gate, measure, adopt-or-reject ON THE NUMBER, log, commit.
  B EXHAUST  — DSV4_KSWEEP=1 DSV4_DPROF=1 profile, re-rank by phase_ms x (1 - achieved/ceiling),
               rewrite queue, record the top entry in phaseB_top_history, go to A next iteration.
  C EXPLORE  — one structural item. Ends in a measured adopt/reject OR a written impossibility.
  D RESEARCH — write RESEARCH_PROMPT_v<N+1>.md against the CURRENT measured residual, query the
               arXiv API directly with WebFetch (export.arxiv.org/api/query?search_query=abs:"..."
               +AND+abs:"...", sortBy=submittedDate) — it is complete, dated and not SEO-shaped,
               unlike a web search. Fetch the abstracts of the top hits; an abstract usually states
               losslessness and the headline number, which is enough to rank. Convert hits to levers
               with falsification tests, refill the queue, go to A next iteration.

RESEARCH IS NOT BLOCKED BY A BROKEN INSTRUMENT. Phase D produces CANDIDATE LEVERS, not numbers, so
it does not need a working measurement to be worth doing — and it is the cheapest phase (no model
run, no build). If the queue is thin, or the top entry is blocked, or two Phase-B rankings returned
the same answer, do a D pass INSTEAD of idling. The one thing D must not do is adopt anything: a
lever that arrives from a paper still has to be implemented, gated and measured in a later A cycle
before a single number about it is written down. Finding 49 came from a D pass and it re-framed a
residual the loop had spent seven rounds treating as physics.

YOUR JOB IS CUDA, MEASUREMENT AND RESEARCH. NOTHING ELSE. The harness has already, before you
started: verified nvcc/g++/the GPU/a trivial sm_110a compile-and-launch, built and run every unit
gate, reclaimed page cache, confirmed ~105+ GiB is free, and confirmed no other model process is
running. It will commit your work after you exit. So:
  - Do NOT run git (no add, commit, push, checkout). Write `.flywheel_commit_msg`; the harness does
    the rest. Read-only `git log`/`git diff`/`git show` for context is fine and allowed.
  - Do NOT run sudo, touch the crontab, or manage processes. If you think you need to, you have
    misread the task — say so in the journal and halt instead.
  - Do NOT re-verify the toolchain. It was checked this minute; .flywheel_selftest.txt has the
    output if you want to see it.
Spend every turn on: reading the profile, writing a kernel or a gate, running a measurement,
reading a paper. If you find yourself debugging the environment, something is wrong with the
HARNESS — record that in the journal and halt, so it gets fixed once instead of every cycle.

HARD INVARIANTS — every one of these was paid for with a wrong result in this project:
 1. ONE change per measurement. If you change two things you have measured neither.
 2. Gate BEFORE you measure speed. Unit gate on the exact kernel at the exact M decode uses.
    A gate that allocates its own inputs cannot test alignment — real weights are 4-byte-aligned
    pointers into a mapped file, cudaMalloc is always 256. Reproduce the real alignment.
 3. NEVER write a number you did not just run. Not in the log, not in a comment, not in a commit
    message. If you did not run it this iteration, it does not exist.
 4. Microbenchmarks (gemm_bench) overstate end-to-end value by 2-4x — they relaunch a kernel on
    rotating weights so launches overlap, while the engine serialises everything behind a data
    dependency. gemm_bench RANKS two kernels; it does not PREDICT end-to-end gain. Confirm in situ.
 5. All full-model runs go through scripts/run_model.sh. Exactly ONE per iteration. The canonical
    prompt is "0,671,6102,294,8760,344" and the expected first token is 11111. Use NGEN0=60 for
    speculation numbers — 24-token runs understate steady state by ~10%.
    A run takes 10-20 minutes and run_model.sh DETACHES, so the launching command returns at once
    and your Bash tool caps at 600s. Do not try to wait in one call, and do not poll with bare
    sleeps. Use the helper, which is built for exactly this and tells you which case you are in:
        scripts/run_model.sh ~/cycleN.log ./build/decode ~/models/DeepSeek-V4-Flash-0731-REAP \
            "0,671,6102,294,8760,344" 8 "" 60
        scripts/await_log.sh ~/cycleN.log 'SPEC-DECODE|GATE FAIL|cuda .*:'
    await_log.sh exits 0 = found (it prints the lines), 2 = still running so CALL IT AGAIN,
    3 = the run died without matching (it prints the tail; treat as a failure and investigate).
    Expect to call it 2-3 times. Each call is one turn; you have plenty.
 6. Report bands, not points. Cross-run noise is +/-1%. A within-run A/B (sweep several settings in
    one checkpoint load, as DSV4_BLKSWEEP does) is worth far more than several separate runs.
 7. If a gate fails or a run prints GATE FAIL: STOP. Set halt=true with the reason in
    FLYWHEEL_STATE.json, commit what you have, and write it up. Do not build on top of it.
 8. Predicted vs measured off by more than 2x means the RANKING MODEL is wrong, not the lever.
    Fix the model in DECODE_FLYWHEEL.md before taking the next lever.
 9. Do NOT `git push`. Commit locally only. A human reviews and publishes.
10. Do NOT modify the checkpoint, quantise anything further, or touch the REAP artifact. Do not
    add model-changing work (MLA FP4, MTP fine-tune) — those are the user's decisions, not yours.

FINISH BY, in this order:
  a. appending your result to LOOP_LOG.md — including negative results, WITH the numbers
  b. updating FLYWHEEL_STATE.json: cycle+1, phase, queue, counters, last_result, and
     consecutive_sub_half_pct (increment if this lever moved end-to-end < 0.5%, else reset to 0)
  c. writing your commit message to `.flywheel_commit_msg` (first line a subject, then a blank
     line, then the body). THE HARNESS COMMITS FOR YOU — do not run git yourself.
  d. appending 5-15 lines to FLYWHEEL_JOURNAL.md: what you did, the number, what you concluded,
     and what the next iteration should do. This is what the human observer reads.

If the phase's stopping rule has fired (three consecutive levers under 0.5% AND the last two
phaseB_top_history entries are equal), advance the phase per DECODE_FLYWHEEL.md and say so.
EOP
)

# stream-json so the iteration is WATCHABLE, not just auditable after the fact. `--verbose` is
# required for stream-json under -p. Rendered by scripts/flywheel_watch.py; kept per cycle so a
# finished iteration can be replayed rather than only tailed.
mkdir -p "$ROOT/.flywheel_cycles"
CYCLE_JSONL="$ROOT/.flywheel_cycles/$(printf 'cycle%04d-%s' "$CYCLE" "$(date +%Y%m%d-%H%M)").jsonl"
LIVE="$ROOT/.flywheel_last_run.jsonl"
: > "$CYCLE_JSONL"; ln -sf "$CYCLE_JSONL" "$LIVE"

set +e
# `< /dev/null` is load-bearing, not tidiness. Without it `claude -p` waits on stdin ("no stdin
# data received in 3s"), and a detached or cron-launched process has no terminal to supply one --
# the stream then aborts mid-turn with terminal_reason=aborted_streaming and
# "[Request interrupted by user]" in the transcript. It cost two whole cycles to find.
claude -p "$PROMPT" \
  --output-format stream-json --verbose --include-partial-messages \
  --permission-mode bypassPermissions \
  --max-turns "$TURNS" \
  --add-dir "$HOME" \
  --append-system-prompt "You are running headless and unattended on a Jetson Thor. Never run git push. Never run a second full-model process. Prefer stopping and recording a halt over guessing." \
  >> "$CYCLE_JSONL" 2>&1 < /dev/null
RC=$?
set -e
# plain-text mirror of the final answer, for grepping without the renderer
python3 - "$CYCLE_JSONL" > "$ROOT/.flywheel_last_run.txt" 2>/dev/null <<'PY'
import json,sys
for ln in open(sys.argv[1]):
    if ln.startswith('{"type":"result'):
        try: print(json.loads(ln).get("result",""))
        except Exception: pass
PY
log "agent exited rc=$RC"

# ---- commit on the agent's behalf --------------------------------------------------------------
# Clerical work belongs to the harness, not to the agent's turn budget. The agent writes a message;
# we stage and commit. Nothing is pushed — that stays a human decision.
if [ -n "$(git status --porcelain)" ]; then
    MSG="$ROOT/.flywheel_commit_msg"
    if [ -s "$MSG" ]; then
        git add -A
        git -c user.name="flywheel" -c user.email="flywheel@localhost" commit -q -F "$MSG" \
            -m "" --cleanup=verbatim 2>/dev/null \
          || git add -A && git commit -q -F "$MSG"
        log "committed cycle $CYCLE: $(head -1 "$MSG")"
        : > "$MSG"
    elif [ "$RC" -ne 0 ]; then
        # A cycle that ERRORED must not leave edits behind: the next cycle would build on unverified
        # code and could report it as its own. Preserve the work as a patch, restore a clean tree.
        mkdir -p "$ROOT/.flywheel_quarantine"
        Q="$ROOT/.flywheel_quarantine/cycle${CYCLE}-$(date +%Y%m%d-%H%M).patch"
        git diff > "$Q"; git status --porcelain > "$Q.status"
        git checkout -- . 2>/dev/null
        log "cycle failed (rc=$RC) and left a dirty tree; quarantined to $Q and restored HEAD"
    else
        log "WARNING: working tree dirty but no .flywheel_commit_msg — leaving uncommitted for review"
    fi
fi

# two failed cycles in a row is a systemic problem, not bad luck
if [ "$RC" -ne 0 ]; then
    N=$(cat "$ROOT/.flywheel_failstreak" 2>/dev/null || echo 0); N=$((N+1))
    echo "$N" > "$ROOT/.flywheel_failstreak"
    [ "$N" -ge 2 ] && halt "two consecutive cycles failed (rc=$RC); see .flywheel_last_run.jsonl"
else
    echo 0 > "$ROOT/.flywheel_failstreak"
fi

# ---- postflight ------------------------------------------------------------------------------
# Enforce the invariants the agent was asked to respect, rather than trusting that it did.
AFTER=$(git rev-parse HEAD)
if [ "$BEFORE" = "$AFTER" ]; then log "no commit made this iteration"; fi

if git log "$BEFORE..$AFTER" --oneline 2>/dev/null | grep -q .; then
  log "committed: $(git log "$BEFORE..$AFTER" --oneline | head -3 | tr '\n' ' ')"
fi

# invariant 9: nothing may have been pushed
if [ -n "$(git log origin/main..HEAD --oneline 2>/dev/null)" ]; then
  log "OK: $(git log origin/main..HEAD --oneline | wc -l) commit(s) held locally for review"
fi

# invariant 7: gates must still pass after the iteration
FAILED=0
for g in gate_units gate_bf16w gate_ogroup_gemv gate_tc_fp8_smem; do
  [ -x "build/$g" ] || continue
  if ./build/$g 2>&1 | grep -qi "FAIL"; then log "GATE FAILURE in $g"; FAILED=1; fi
done
[ "$FAILED" -eq 0 ] || halt "a unit gate fails after cycle $CYCLE"

python3 -c "import json;json.load(open('$STATE'))" 2>/dev/null || halt "agent corrupted FLYWHEEL_STATE.json"
NEW=$(python3 -c "import json;print(json.load(open('$STATE'))['cycle'])")
[ "$NEW" != "$CYCLE" ] || log "WARNING: agent did not advance the cycle counter"

log "cycle $CYCLE done"
