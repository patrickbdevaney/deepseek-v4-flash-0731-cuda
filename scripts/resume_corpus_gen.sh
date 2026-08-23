#!/usr/bin/env bash
# resume_corpus_gen.sh — finish a pass-1 corpus generation that was interrupted, then hand the
# corpus chain back its own job.
#
# WHY THIS EXISTS. On 2026-08-23 the C(k) sweep was moved ahead of the agentic corpus: C(k) is a
# ~30 min measurement that PRICES a +20-25 % lever (adaptive block width), and it was queued behind
# a 31 h generation that yields +4-9 %. Cheap-and-decisive should not wait on expensive-and-
# incremental. Swapping them costs nothing only because pass 1 is resumable -- this script is the
# proof of that claim, and the thing that makes the swap reversible.
#
# WHY A SPLICE IS SAFE HERE, precisely:
#   * decode opens DSV4_GENOUT with "a" and fclose()s after EVERY sequence (src/decode.cu:1457).
#     So gen.txt only ever contains whole lines; an interrupted run loses at most the in-flight
#     sequence, never a partial line, and never the lines already written.
#   * The sweep is one point per prompt in order, and prompts[0] is argv[2] while the file's prompts
#     are 1..N (src/decode.cu:190-192). So gen.txt line i corresponds to prompt-file line i.
#     This is not assumed -- it is checked below, and the OFF-BY-ONE ALTERNATIVE IS CHECKED TOO,
#     because a splice that is silently shifted by one produces a corpus in which every response
#     answers the previous prompt, which trains fine and measures as noise.
#   * Pass 1 generates with the BASE checkpoint (s5_session_auto.sh:32), not the live head, so the
#     resumed half is drawn from the same distribution as the first half. If that ever stops being
#     true, the corpus is inhomogeneous and the arm means nothing -- hence the assertion below.
#
# It is a LOOP, not a single shot: if the box is interrupted again, running this again finishes the
# remainder of the remainder. Idempotent -- with a complete gen.txt it does nothing but verify.
set -u
ROOT=/home/patrickd/deepseek-v4-flash-0731-cuda
cd "$ROOT" || exit 1
LOG(){ printf '[resume %s] %s\n' "$(date -Is)" "$*"; }
DIE(){ LOG "FATAL: $*"; exit 1; }

S5=/home/patrickd/s5-capture
NAME=agentic-p25-b0.1
PROMPTS="$S5/mixed_prompts_agentic.txt"
GEN="$S5/$NAME/gen.txt"
N=3071
NGEN=1024
BLK=5
# Must match s5_session_auto.sh:32 exactly. Pass 1 is the TARGET's own output.
CKPT=/home/patrickd/models/DeepSeek-V4-Flash-0731-REAP
WANT_CKPT=$(grep -m1 -oE '^CKPT=.*' scripts/s5_session_auto.sh | cut -d= -f2-)
[ "$CKPT" = "$WANT_CKPT" ] || DIE "checkpoint drift: this script says $CKPT, s5_session_auto.sh says $WANT_CKPT.
Resuming with a different checkpoint than generated the first half would make the corpus inhomogeneous."

# ---------------------------------------------------------------- wait for the box to be ours
# Single tenancy: 100.4 GiB of weights in a 122 GiB pool. Two models is an OOM, and an OOM in the
# middle of generation costs the whole remainder.
for u in dsv4-ck dsv4-p25b dsv4-autopilot; do
    while systemctl --user is-active --quiet "$u"; do
        LOG "waiting for $u"; sleep 120
    done
done
while pgrep -x decode >/dev/null || pgrep -x main >/dev/null; do LOG "waiting for the GPU"; sleep 120; done

# ---------------------------------------------------------------- generate the remainder, in a loop
for attempt in 1 2 3; do
    L=$(wc -l < "$GEN" 2>/dev/null || echo 0)
    [ "$L" -ge "$N" ] && { LOG "gen.txt already complete ($L/$N)"; break; }
    M=$(( N - L ))
    LOG "attempt $attempt: $L/$N done, generating the remaining $M"

    # The remainder file is rebuilt every attempt from the CURRENT line count, so a partial attempt
    # simply shrinks the next remainder rather than duplicating or skipping anything.
    REM="$S5/$NAME/prompts_remainder.txt"
    tail -n +$(( L + 1 )) "$PROMPTS" > "$REM"
    [ "$(wc -l < "$REM")" -eq "$M" ] || DIE "remainder is $(wc -l < "$REM") lines, expected $M"

    # 1-based: prompts[0] is argv[2], the file's prompts are 1..M.
    SW=$(python3 -c "print(','.join(f'$BLK:1:1.5:{i}' for i in range(1,$M+1)))")

    sudo jetson_clocks >/dev/null 2>&1 || true
    sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; sleep 2
    DSV4_PROMPTS_FILE="$REM" DSV4_GENOUT="$GEN" DSV4_BLKSWEEP="$SW" \
        scripts/run_model.sh "$ROOT/evidence/${NAME}_pass1_resume${attempt}.log" ./build/decode \
        "$CKPT" "0,671,6102,294,8760,344" 8 "" "$NGEN"
    sleep 30
    while pgrep -x decode >/dev/null; do sleep 60; done
    LOG "attempt $attempt ended with $(wc -l < "$GEN")/$N"
done

L=$(wc -l < "$GEN" 2>/dev/null || echo 0)
[ "$L" -ge "$N" ] || DIE "gen.txt is $L/$N after 3 attempts; not handing an incomplete corpus to the chain"

# ---------------------------------------------------------------- prove the splice, both hypotheses
python3 - "$PROMPTS" "$GEN" "$NGEN" <<'PY' || DIE "splice verification failed"
import sys
seed=[l.strip() for l in open(sys.argv[1])]
gen =[l.strip() for l in open(sys.argv[2])]
ngen=int(sys.argv[3])
if open(sys.argv[2],'rb').read()[-1:]!=b'\n': print("FAIL: gen.txt does not end in a newline"); sys.exit(1)
if len(gen)<len(seed): print("FAIL: %d gen lines < %d prompts"%(len(gen),len(seed))); sys.exit(1)
bad =[i for i,(s,g) in enumerate(zip(seed,gen)) if g.split(',')[:len(s.split(','))]!=s.split(',')]
bad1=[i for i,(s,g) in enumerate(zip(seed[1:],gen)) if g.split(',')[:len(s.split(','))]!=s.split(',')]
short=[i for i,(s,g) in enumerate(zip(seed,gen)) if len(g.split(','))-len(s.split(','))<ngen]
print("  lines            : %d / %d"%(len(gen),len(seed)))
print("  aligned  offset 0: %d mismatches   <- must be 0"%len(bad))
print("  aligned  offset 1: %d mismatches   <- must be LARGE; if 0 the splice is shifted"%len(bad1))
print("  short sequences  : %d (< %d generated tokens)"%(len(short),ngen))
ok = not bad and len(bad1)>len(gen)//2 and not short
print("  VERDICT:", "SPLICE SOUND" if ok else "SPLICE UNSOUND")
sys.exit(0 if ok else 1)
PY

LOG "pass 1 complete and verified ($L/$N); handing back to the corpus chain"
LOG "the chain skips pass 1 because gen.txt is non-empty and complete, and goes straight to capture+train"
systemctl --user start dsv4-corpus && LOG "dsv4-corpus started" || LOG "FAILED to start dsv4-corpus"
