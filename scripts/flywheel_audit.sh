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

# ---- 5. PIVOT. Kernel work and the draft-head fine-tune compete for the same attention, so the loop
# declares exhaustion mechanically instead of grinding sub-1% levers forever. See
# FLYWHEEL_STATE.json:pivot_criterion. Two consecutive audits with an empty non-training queue. ----
OPEN=$(python3 -c "import json;print(json.load(open('FLYWHEEL_STATE.json')).get('pivot_criterion',{}).get('open_nontraining_levers',99))" 2>/dev/null || echo 99)
PREV=$(cat "$ROOT/.flywheel_openprev" 2>/dev/null || echo 99)
echo "$OPEN" > "$ROOT/.flywheel_openprev"
if [ "$OPEN" = "0" ] && [ "$PREV" = "0" ] && [ ! -f "$ROOT/FLYWHEEL_PIVOT" ]; then
    touch "$ROOT/FLYWHEEL_PIVOT"
    { echo "## $(date -Is) — ** PIVOT **"
      echo "  Two consecutive audits with no open non-training lever >=1%."
      echo "  Kernel optimisation is exhausted for this checkpoint. The remaining lever is S5:"
      echo "  re-align the DSpark MTP draft head. That is a TRAINING job -- the loop cannot do it."
      echo "  Position: $(grep -oE '\"spec_tok_s\": [0-9.]+' FLYWHEEL_STATE.json | head -1) vs a ceiling of 30.8 at acceptance 2.90."
    } >> "$ROOT/FLYWHEEL_AUDIT.md"
    say "** PIVOT ** kernel work exhausted; the remaining lever is training (S5). See FLYWHEEL_AUDIT.md"
fi

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
# "newest evidence" must be an actual MODEL RUN, not any file that landed in evidence/. The first
# version took `ls -t evidence/*.log | head -1` and picked up an archived CRON log, then reported the
# commit as unverifiable. Select on content, not on mtime and a glob.
CLAIM=$(git log -1 --format=%B | grep -oE '[0-9]+\.[0-9]+ tok/s' | head -1 | grep -oE '[0-9]+\.[0-9]+' || true)
NEWEST=$(grep -lE "SPEC-DECODE|WARM decode" evidence/*.log 2>/dev/null | xargs -r ls -t 2>/dev/null | head -1)
# Some commits legitimately carry numbers that were DERIVED, not run: a roofline, a projection, a
# ceiling. Trying to tell those apart by keyword failed twice (a commit about "MEASURED bytes" is
# about arithmetic on measured bytes, not about a run). So the executor DECLARES it: a commit whose
# numbers are all derived must contain a line starting `DERIVED-ONLY:` with the reason. Explicit,
# auditable, and impossible to trip accidentally — which a keyword heuristic is not.
git log -1 --format=%B | grep -q "^DERIVED-ONLY:" && CLAIM=""
if [ -n "$CLAIM" ]; then
    if [ -z "$NEWEST" ]; then note "commit claims $CLAIM tok/s but evidence/ has no log"
    else
        # Finding 33: a number that cannot be traced to a run did not happen.
        # Traceable to ANY archived run, not only the newest — a cycle may legitimately quote its
        # control. Finding 33 requires the number to have happened, not to be the most recent thing.
        grep -ql "$CLAIM" evidence/*.log 2>/dev/null || note "claimed $CLAIM tok/s not found in any evidence/ log"
        # Finding 68: a speedup that is a quality collapse passes every other gate.
        # Demand the LOSSLESS gate only when the CLAIMED number is itself a spec-decode rate. Finding
        # 68's failure mode is "spec decode got faster by degrading the output", which always shows up
        # on a SPEC-DECODE line -- so keying on the claim loses none of that protection. Keying on the
        # mere PRESENCE of a SPEC-DECODE line held Finding 75, whose claim was a PREFILL rate, because
        # its instrumentation run also happened to emit a degenerate `SPEC-DECODE: 0.00 tok/s` from a
        # 4-token sweep. An auditor that blocks a measurement for the contents of a line it is not
        # citing teaches the loop to launder evidence into a second file.
        if grep -E "SPEC-DECODE" "$NEWEST" | grep -q "$CLAIM"; then
            grep -q "LOSSLESS GATE.*PASS" "$NEWEST" || note "claimed $CLAIM tok/s is a SPEC-DECODE rate with no passing LOSSLESS gate in $(basename "$NEWEST")"
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
