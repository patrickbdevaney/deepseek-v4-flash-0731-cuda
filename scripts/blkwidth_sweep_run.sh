#!/usr/bin/env bash
# blkwidth_sweep_run.sh -- DECODE_LADDER 2.1. Re-tune the speculation block width, in ONE load.
#
#   nohup setsid bash scripts/blkwidth_sweep_run.sh > evidence/decode_loop/blkwidth.log 2>&1 </dev/null &
#
# Named in detach_audit.sh's PATTERNS in the same commit.
#
# WHAT IS BEING RE-TUNED AND WHY IT IS NOT A SETTLED NUMBER. The shipped default is BLK=6
# (`src/decode.cu`, `include/dsv4_engine.h`), and the comment that defends it cites F94: 6 beat 5 on
# 4 of 4 realistic prompts and 8 LOST to 6 on 3 of 4. Both halves of that measurement were taken
# against premises that have since changed:
#
#   * `k_topk_verify<<<K,32>>>` launched K blocks of ONE active thread, so verify width was very
#     nearly free in the context term (DECODE_ZENITH_FINDINGS.md 3.2). Ladder 1.1/1.2 replaced that
#     kernel. Width is no longer free, and 3.2 said in advance that fixing it "changes the optimal
#     block size" and that width must be decided AFTER the fix "or you will tune to a transient".
#   * F94 ran on the head that was shipped then. Ladder 2.2 deployed `s3`, which lifted suite tau
#     3.5362 -> 3.8438 and, more to the point here, lifted the FLOOR: no suite prompt is below tau 2
#     any more. F130 already measured the coupling in the other direction -- a better drafter earns
#     WIDER verifies, so its optimal adaptK is LOWER -- and the same coupling applies to the block.
#     At BLK=6 (ceiling BLK+1=7) suite prompt 6 already realises tau 6.09, i.e. it is pressing the
#     ceiling the block imposes.
#
# ONE LOAD, PALINDROMIC ORDER. traps 19: three replicates inside one checkpoint load spread 0.6 %,
# the same measurement across loads 5.7 %. So every width must be measured in ONE process or the
# between-load term swamps the effect. Within a load the rate still drifts (suite_adaptk_sweep.sh
# measured ~6 % UPWARD with run order), so each prompt's widths are run ASCENDING then DESCENDING:
# the two occurrences of every width have run indices that sum to a constant, which cancels any
# linear drift exactly when the pair is averaged. The palindrome is per PROMPT, not per run, so the
# cancellation window is ~2.5 minutes rather than ~25, and a run that dies part-way still leaves
# every completed prompt with a complete, drift-cancelled design.
#
# NOT BIT-EXACT BY CONSTRUCTION, AND THAT IS PRE-REGISTERED. Changing BLK changes M in the verify
# forward, which changes MoE atomic reduction order -- the engine's own gate already tolerates that
# ("diffs = MoE-atomic near-ties"). So this ships under invariant 1's SECOND branch: the LOSSLESS
# gate (first 8 tokens vs base AR) must PASS at every width on the canonical prompt, and
# DSV4_GENOUT records the full emitted id sequence of every point so the divergence profile across
# widths is MEASURED rather than assumed.
set -u
cd "$(dirname "$0")/.."
EV=evidence/decode_loop
# The LIVE checkpoint, not the base one: 2.2 pinned the deployed head in config/live_ckpt and a
# width tuned against a head that is not being served is a width tuned for nothing.
CKPT="${CKPT:-$(cat config/live_ckpt 2>/dev/null || echo /home/patrickd/models/DeepSeek-V4-Flash-0731-REAP)}"
WIDTHS="${WIDTHS:-4 5 6 7 8 9 10 12}"
NPROMPT="${NPROMPT:-9}"          # argv prompt 0 (canonical/control) + the 8 frozen suite prompts
ADAPTK="${ADAPTK:-1.50}"         # FROZEN. One change per measurement; the width is the change.
NGEN="${NGEN:-200}"              # the frozen protocol's NGEN0
TRIES="${TRIES:-3}"
LOG=$EV/blkwidth_sweep.log
GO=$EV/blkwidth_genout.txt
OUT=$EV/blkwidth_verdict.txt
say(){ echo "[2.1] $(date -Is) $*"; }

[ -s protocol/suite_prompts.txt ] || { say "REFUSING: protocol/suite_prompts.txt missing."; exit 1; }
[ -x build/decode ] || { say "REFUSING: build/decode missing."; exit 1; }
for f in src/decode.cu include/indexer.h include/topk_radix.h; do
  [ "$f" -nt build/decode ] && { say "REFUSING: build/decode is older than $f."; exit 1; }
done
SUITE=$(cat protocol/suite_prompts.txt)

# The sweep string. Warm-up point first and DISCARDED by the analysis (traps: the first run of a
# batch is the slowest in every batch this repo has measured), then, per prompt, the palindrome.
SWEEP=$(python3 - "$ADAPTK" "$NPROMPT" <<PY
import sys
ak, npr = sys.argv[1], int(sys.argv[2])
w = [int(x) for x in "$WIDTHS".split()]
pts = ["6:1:%s:0" % ak]                       # warm-up, discarded
for p in range(npr):
    for b in w + w[::-1]:
        pts.append("%d:1:%s:%d" % (b, ak, p))
print(",".join(pts))
PY
)
NPTS=$(awk -F, '{print NF}' <<< "$SWEEP")
say "checkpoint  : $CKPT"
say "widths      : $WIDTHS   prompts: $NPROMPT   adaptK: $ADAPTK   NGEN0: $NGEN"
say "sweep       : $NPTS points (1 warm-up + $NPROMPT x $(( (NPTS-1)/NPROMPT )) palindromic)"

# GATE THE TABLE BEFORE SPENDING A LOAD. A mislabelled 145-point table discovered after a
# ten-minute load is the expensive failure; DSV4_PARSE_ONLY answers in milliseconds.
if ! DSV4_PARSE_ONLY=1 DSV4_PROMPTS="$SUITE" DSV4_BLKSWEEP="$SWEEP" \
     ./build/decode "$CKPT" "0,671,6102,294,8760,344" 8 "" "$NGEN" > "$EV/blkwidth_parse.log" 2>&1; then
    say "PARSE GATE FAIL -- see $EV/blkwidth_parse.log"; tail -5 "$EV/blkwidth_parse.log"; exit 1; fi
PARSED=$(grep -c '^\[parse\] point' "$EV/blkwidth_parse.log")
[ "$PARSED" = "$NPTS" ] || { say "PARSE GATE FAIL: $PARSED points parsed, $NPTS requested"; exit 1; }
say "PARSE GATE PASS: $PARSED points, $(grep -c '^\[parse\] prompt' "$EV/blkwidth_parse.log") prompts"

for try in $(seq 1 "$TRIES"); do
    rm -f "$GO"                                   # DSV4_GENOUT appends
    for w in $(seq 1 60); do
        a=$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo)
        [ "$a" -ge 100 ] && break
        say "try $try: MemAvailable ${a} GiB, waiting for 100"; sleep 15
    done
    sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true; sleep 2
    say "try $try: launching detached (MemAvailable $(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo) GiB)"
    env DSV4_PROMPTS="$SUITE" DSV4_BLKSWEEP="$SWEEP" DSV4_GENOUT="$GO" \
        bash scripts/run_model.sh "$LOG" ./build/decode "$CKPT" \
        "0,671,6102,294,8760,344" 8 "" "$NGEN" || { say "run_model refused"; sleep 60; continue; }
    for w in $(seq 1 60);   do pgrep -x decode > /dev/null && break; sleep 2;  done
    for w in $(seq 1 1080); do pgrep -x decode > /dev/null || break; sleep 10; done   # <= 3 h
    if grep -q "KILLING" "${LOG%.log}.memguard.log" 2>/dev/null; then
        say "try $try: the memguard killed the load. Retrying."; continue; fi
    DONE=$(grep -c '^\[spec\] SPEC-DECODE:' "$LOG" 2>/dev/null || echo 0)
    say "try $try: $DONE of $NPTS points completed"
    [ "$DONE" -ge "$NPTS" ] && break
    say "try $try: INCOMPLETE. Retrying."
done

say "=== analysis ==="
python3 tools/blkwidth_ab.py --log "$LOG" --genout "$GO" --baseline 6 | tee "$OUT"
say "done -> $OUT"
