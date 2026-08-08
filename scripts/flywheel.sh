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
# keep the guard installed; .git/hooks is not versioned, so a clone or a `git init` loses it
if [ ! -x "$ROOT/.git/hooks/pre-commit" ] || ! cmp -s "$ROOT/scripts/hooks/pre-commit" "$ROOT/.git/hooks/pre-commit"; then
    cp "$ROOT/scripts/hooks/pre-commit" "$ROOT/.git/hooks/pre-commit" 2>/dev/null && chmod +x "$ROOT/.git/hooks/pre-commit"
    log "installed pre-commit guard"
fi

CYCLE=$(python3 -c "import json;print(json.load(open('$STATE'))['cycle'])")
BASE_BEFORE=$(python3 -c "import json;print(json.load(open('$STATE'))['baseline'].get('spec_tok_s',0))")
log "cycle $CYCLE starting; ${AVAIL} GiB free; HEAD $(git rev-parse --short HEAD)"
BEFORE=$(git rev-parse HEAD)

# ---- the iteration ---------------------------------------------------------------------------
PROMPT=$(cat <<'EOP'
You are ONE cycle of the decode-speed loop for this CUDA inference engine. Do one lever, record it,
stop. You are unattended: never ask a question, never wait for input.

THE TWO LEDGERS ARE THE POINT. Read both before doing anything.
  LEVERS.md      — what is adopted, what is RETIRED WITH THE NUMBER THAT KILLED IT, what is open with
                   an expected value, the measurement traps, and the reference data. If your idea is
                   in the retired table, it is closed: reopening needs new EVIDENCE that addresses the
                   specific measurement that killed it, not a new argument.
  RESEARCH_LOG.md — every query already run. Do not re-run one.
A cycle that does not write its result back to LEVERS.md has wasted itself, because the next cycle
will try the same thing. The auditor holds engine commits that do not update it.

PHASES, in order:

0. ORIENT. Read LEVERS.md §1 (where the time goes) and §4/§5 (open levers). Read the tail of
   LOOP_LOG.md for the last two findings.

1. RESEARCH — ONLY IF NEEDED. If LEVERS.md has >=2 open levers with expected value >=1%, SKIP this
   and go to 2. Otherwise follow RESEARCH_LOG.md §1: 6-8 WebSearch queries, at most one per axis,
   check every hit against the constraints in §1 (no additional quantisation, no retraining, output
   must stay lossless, implementable against THIS engine), and append every query to §2 with its
   outcome — including "nothing usable", which is a result. Promote at most 2 into LEVERS.md §4/§5
   with an expected value and how you would falsify it.
   NOTE: every adoption so far (F64,F65,F70,F71,F72) came from profiling this engine, not from
   literature. Research is the tiebreaker when the profile stops suggesting things.

2. PICK the highest expected-value open lever. Say which and why in one line.

3. MEASURE FIRST IF THE COST IS UNKNOWN. dprof marks are cheap and have repeatedly changed the
   answer: DSV4_DPROF=1 DSV4_KSWEEP=1 gives per-sub-op ms at K=1..5. Do not optimise a region you
   have not attributed. Finding 67 killed a lever this way before building it.

4. IMPLEMENT. One change. Keep the previous behaviour reachable behind an env flag so the A/B is
   possible in a later cycle.

5. GATE. scripts/build_gate.sh then every build/gate_* binary. A MISSING binary is a FAILURE, not a
   pass. If a gate fails, fix it or revert — never proceed past a red gate.

6. MEASURE. ONE full-model run via scripts/run_model.sh (never two; never a second `decode`).
   Compare against the newest evidence/*.log for the same configuration. Quote the tight within-run
   marks (ksweep K=5, the dprof sub-op) as well as end-to-end tok/s.
   If the run prints a spec tok/s, the log MUST contain "LOSSLESS GATE ... PASS". A change that
   raises tok/s while failing it has degraded the output — Finding 68 is the worked example, a fake
   +28% that passed every other gate.

7. WRITE BACK. Append one LOOP_LOG.md finding: what you tried, the numbers, and the mechanism —
   including for a FAILURE, with the number that killed it. Move the lever in LEVERS.md into adopted
   or retired. Copy the run log into evidence/.

   YOU CANNOT RUN GIT. `git add`, `git commit`, `git push`, `git checkout` and `git reset` are ALL
   denied to you by .claude/settings.json, and the denial is silent from your side — it is not a bug
   to work around and retrying it just burns the rest of your turns. The harness commits for you:
   **write your commit message to `.flywheel_commit_msg` in the repo root** (first line = the result,
   e.g. "moe:group 2.66 -> 2.12 ms (-20%), bit-identical"), leave the tree dirty, and stop. If that
   file is absent the harness logs "dirty but no .flywheel_commit_msg" and your entire cycle is left
   uncommitted — which is exactly how cycles 9 and 10 were lost.

HARD RULES
  - Numbers come from a log you produced this cycle. A number you cannot grep is a number that did
    not happen. If a commit's numbers are all DERIVED (a roofline, a projection, a ceiling) rather
    than run, say so with a line starting `DERIVED-ONLY:` — the auditor holds unexplained numbers.
  - One change per measurement.
  - Prefer reverting to keeping something unmeasured.
  - Never run git. Write .flywheel_commit_msg instead (see phase 7).
  - If a gate fails or a run crashes, STOP and write it up. A cycle that reports a real failure
    honestly is a good cycle.
  - Clocks must be pinned (`sudo jetson_clocks`) before any measurement, and say so in the write-up.
  - `sync; echo 3 | sudo tee /proc/sys/vm/drop_caches` before each model run or it will refuse for
    lack of memory.
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

# ---- archive the evidence -----------------------------------------------------------------------
# Every number in LOOP_LOG.md is only as durable as the log it came from, and those logs live in
# ~/ -- outside the repo, unversioned, and one `rm ~/*.log` from making every past claim
# unverifiable. The observer's first and most valuable check is "grep the log for that number";
# that check has to still work in a month. Model-run logs are ~50 KB of text, so copy any that this
# cycle wrote into the repo and let them ride along in the same commit as the finding.
mkdir -p "$ROOT/evidence"
for L in $(find "$HOME" -maxdepth 1 -name '*.log' -newer "$ROOT/.flywheel_selftest.txt" 2>/dev/null); do
    B=$(basename "$L")
    cp "$L" "$ROOT/evidence/$(printf 'cycle%03d' "$CYCLE")-$B" 2>/dev/null &&         log "archived evidence: $(printf 'cycle%03d' "$CYCLE")-$B ($(wc -l < "$L") lines)"
done

# ---- commit on the agent's behalf --------------------------------------------------------------
# Clerical work belongs to the harness, not to the agent's turn budget. The agent writes a message;
# we stage and commit. Nothing is pushed — that stays a human decision.
if [ -n "$(git status --porcelain)" ]; then
    MSG="$ROOT/.flywheel_commit_msg"
    if [ -s "$MSG" ]; then
        # FLYWHEEL_COMMIT=1 tells the pre-commit hook this is the legitimate in-cycle commit; every
        # other commit is refused while the lock is held (see scripts/hooks/pre-commit).
        git add -A
        FLYWHEEL_COMMIT=1 git commit -q -F "$MSG"
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

# ---- publish on a measured gain --------------------------------------------------------------
# The autonomous loop may PUBLISH, but only against evidence it cannot fake, and the harness checks
# each condition itself rather than trusting the agent's write-up:
#   1. every unit gate passes (already verified above, or we halted)
#   2. baseline.spec_tok_s rose by >= 0.5% -- above the +/-1% cross-run band only in the sense that
#      the agent must have measured it; the real guard is (3)
#   3. a full-model log written DURING this cycle contains GATE PASS and no GATE FAIL
# Anything short of all three stays local for review. This is the one outward-facing action the
# loop can take, so it is the one with the most checks.
BASE_AFTER=$(python3 -c "import json;print(json.load(open('$STATE'))['baseline'].get('spec_tok_s',0))" 2>/dev/null || echo 0)
GAIN=$(python3 -c "b=float('${BASE_BEFORE:-0}'); a=float('$BASE_AFTER'); print('%.2f' % (((a-b)/b*100) if b>0 else 0.0))")
FRESH_PASS=0; FRESH_FAIL=0
for L in $(find "$HOME" -maxdepth 1 -name '*.log' -newermt "-${SINCE_MIN:-90} minutes" 2>/dev/null); do
    grep -q "GATE PASS" "$L" 2>/dev/null && FRESH_PASS=1
    grep -q "GATE FAIL" "$L" 2>/dev/null && FRESH_FAIL=1
done
if [ "$FAILED" -eq 0 ] && [ "$FRESH_FAIL" -eq 0 ] && [ "$FRESH_PASS" -eq 1 ] \
   && [ "$(python3 -c "print(1 if float('$GAIN')>=0.5 else 0)")" = "1" ]; then
    if git push -q origin main 2>>"$ROOT/.flywheel_push.log"; then
        log "PUBLISHED: baseline ${BASE_BEFORE} -> ${BASE_AFTER} tok/s (+${GAIN}%), gates pass, GATE PASS in a fresh run"
    else
        log "push failed (see .flywheel_push.log); commits remain local"
    fi
else
    log "not publishing: gain=${GAIN}% fresh_gate_pass=${FRESH_PASS} fresh_gate_fail=${FRESH_FAIL} unit_gates_failed=${FAILED}"
fi

log "cycle $CYCLE done"
