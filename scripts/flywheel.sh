#!/usr/bin/env bash
# flywheel.sh — ONE iteration of the decode optimisation loop, run headless.
#
# This is the mechanical executor for DECODE_FLYWHEEL.md. It is triggered on a timer, does exactly
# one lever, and stops. It is deliberately NOT a long-running agent: one lever per invocation means
# every iteration is independently reviewable, and a bad one costs one iteration.
#
# AUTHORITY THIS SCRIPT GRANTS THE AGENT (read this before enabling the timer):
#   - edit sources under this repo, build, run unit gates, run ONE full-model measurement
#   - commit LOCALLY
# AUTHORITY IT DOES NOT GRANT:
#   - `git push` (the observer pushes after review — an autonomous loop must not publish)
#   - anything outside this repo except the model dir (read-only) and ~/*.log
#   - a second concurrent full-model run (scripts/run_model.sh already flocks)
#
# HALT CONDITIONS — the loop stops itself and waits for a human:
#   - any unit gate fails
#   - a full-model run prints GATE FAIL
#   - a measured regression worse than 3% against the recorded baseline
#   - FLYWHEEL_STATE.json .halt == true
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
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
pgrep -f "build/decode" >/dev/null && { log "a full-model process is already running; exiting"; exit 0; }
[ -d "$MODEL" ] || halt "model dir missing: $MODEL"

# The 100.4 GiB checkpoint needs headroom; the page cache is reclaimable and run_model.sh refuses
# below ~105 GiB. Reclaim here rather than letting the agent discover it mid-iteration.
AVAIL=$(free -g | awk '/^Mem:/{print $7}')
if [ "$AVAIL" -lt 110 ]; then sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; sleep 2; fi
AVAIL=$(free -g | awk '/^Mem:/{print $7}')
[ "$AVAIL" -ge 105 ] || halt "only ${AVAIL} GiB available after reclaim; a full-model load needs ~105"

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
               arXiv API directly (export.arxiv.org/api/query, sorted by submittedDate), convert
               hits to levers with falsification tests, refill the queue, go to A next iteration.

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
  c. `git add -A && git commit` with a message that states what was measured and what it means
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
claude -p "$PROMPT" \
  --output-format stream-json --verbose --include-partial-messages \
  --permission-mode acceptEdits \
  --max-turns "$TURNS" \
  --add-dir "$HOME" \
  --append-system-prompt "You are running headless and unattended on a Jetson Thor. Never run git push. Never run a second full-model process. Prefer stopping and recording a halt over guessing." \
  >> "$CYCLE_JSONL" 2>&1
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
