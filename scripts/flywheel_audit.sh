#!/usr/bin/env bash
# flywheel_audit.sh — the reviewer. Deterministic, no model, and the ONLY thing that pushes.
#
# The previous design ran a second headless Claude as an "observer". That was the overengineering:
# an LLM was being asked to re-derive judgements that are now mechanical, and it cost a whole extra
# agent invocation per cycle to do it less reliably. Every check below is a grep or an exit code.
#
# The one judgement that genuinely needed intelligence — "is this speedup real or did the change
# degrade the output into something more predictable?" — stopped needing it when Finding 68 forced
# the LOSSLESS gate into the engine. That single line is now the correctness verdict.
#
# Contract with the executor (scripts/flywheel.sh):
#   the executor may edit, build, gate, run ONE model measurement, and commit LOCALLY.
#   it may NOT push (denied in .claude/settings.json).
#   this script decides whether the local commit becomes public.
#
# exit 0 = audited (pushed or deliberately held). Non-zero = something is wrong with the audit itself.
set -uo pipefail
cd "${FLYWHEEL_ROOT:-$(dirname "$0")/..}" || exit 1
ROOT="$PWD"
say(){ printf '[audit %s] %s\n' "$(date -Is)" "$*"; }
FAIL=""; note(){ FAIL="${FAIL}\n  - $*"; }

exec 9>/tmp/dsv4-flywheel-audit.lock; flock -n 9 || { say "already auditing"; exit 0; }
[ -f "$ROOT/FLYWHEEL_STOP" ] && { say "FLYWHEEL_STOP present"; exit 0; }
# Never audit mid-cycle: a half-written tree looks exactly like a broken one.
flock -n /tmp/dsv4-flywheel.lock true 2>/dev/null || { say "cycle in flight"; exit 0; }
pgrep -x decode >/dev/null && { say "model still running"; exit 0; }

HEAD_SHA=$(git rev-parse HEAD)
SEEN=$(cat "$ROOT/.flywheel_audited" 2>/dev/null || echo "")
[ "$HEAD_SHA" = "$SEEN" ] && { say "no new commit"; exit 0; }
UNPUSHED=$(git log --oneline origin/main..HEAD 2>/dev/null | wc -l)
[ "$UNPUSHED" -eq 0 ] && { echo "$HEAD_SHA" > "$ROOT/.flywheel_audited"; say "nothing unpushed"; exit 0; }

# ---- 1. gates. A cycle that leaves the tree failing must never be published. ----
bash scripts/build_gate.sh >/tmp/dsv4-audit-build.log 2>&1 || note "build_gate.sh failed"
for g in gate_units gate_bf16w gate_ogroup_gemv gate_tc_fp8_smem gate_forkjoin_graph gate_prefill_len gate_scratch_init; do
    if [ ! -x "build/$g" ]; then note "gate binary missing: $g"; continue; fi     # MISSING is not PASS
    ./build/"$g" >/tmp/dsv4-audit-$g.log 2>&1
    grep -qiE "GATE FAIL|FAIL$|FAIL " /tmp/dsv4-audit-$g.log && note "gate FAILED: $g"
done

# ---- 2. the ledger. A lever whose result is not written back will be tried again. Only enforced on
# commits that actually touch the engine — an infrastructure or docs commit owes nothing to LEVERS.md,
# and holding those would train the loop to write a filler entry to get past the auditor. ----
TOUCHED=$(git show --stat --format= HEAD | awk '{print $1}')
if echo "$TOUCHED" | grep -qE '^(kernels|src|include)/'; then
    echo "$TOUCHED" | grep -q "LEVERS.md"  || note "engine changed but LEVERS.md not updated"
    echo "$TOUCHED" | grep -q "LOOP_LOG.md" || note "engine changed but no LOOP_LOG entry"
fi

# ---- 3. the measurement. Only applies if the cycle claims a tok/s number. ----
CLAIM=$(git log -1 --format=%B | grep -oE '[0-9]+\.[0-9]+ tok/s' | head -1 | grep -oE '[0-9]+\.[0-9]+' || true)
NEWEST=$(ls -t evidence/*.log 2>/dev/null | head -1)
if [ -n "$CLAIM" ]; then
    if [ -z "$NEWEST" ]; then note "commit claims $CLAIM tok/s but evidence/ has no log"
    else
        # Finding 33: a number that cannot be traced to a run did not happen.
        grep -q "$CLAIM" "$NEWEST" || note "claimed $CLAIM tok/s not found in $(basename "$NEWEST")"
        # Finding 68: a speedup that is a quality collapse passes every other gate.
        if grep -q "SPEC-DECODE" "$NEWEST"; then
            grep -q "LOSSLESS GATE.*PASS" "$NEWEST" || note "no passing LOSSLESS gate in $(basename "$NEWEST")"
        fi
        grep -q "GATE FAIL" "$NEWEST" && note "in-run GATE FAIL in $(basename "$NEWEST")"
    fi
fi

# ---- 4. spinning. Three cycles touching the same lever with nothing adopted means the ranking is
# exhausted and the loop owes itself a research phase, not another attempt at the same thing. ----
LAST3=$(git log -3 --format=%s | sed 's/[^a-z]//g' | sort -u | wc -l)
[ "$LAST3" -eq 1 ] && note "last 3 commit subjects identical — loop may be spinning"

echo "$HEAD_SHA" > "$ROOT/.flywheel_audited"
if [ -n "$FAIL" ]; then
    say "HELD (not pushing):$(printf "$FAIL")"
    { echo "## $(date -Is) — HELD $HEAD_SHA"; printf "%b\n" "$FAIL"; } >> "$ROOT/FLYWHEEL_AUDIT.md"
    exit 0
fi
say "PASS — publishing $HEAD_SHA"
if git push -q origin main 2>/tmp/dsv4-audit-push.log; then
    { echo "## $(date -Is) — PUSHED $HEAD_SHA"; git log -1 --format='  %s'; } >> "$ROOT/FLYWHEEL_AUDIT.md"
else
    say "push failed: $(tail -1 /tmp/dsv4-audit-push.log)"
    { echo "## $(date -Is) — PUSH FAILED $HEAD_SHA"; } >> "$ROOT/FLYWHEEL_AUDIT.md"
fi
exit 0
