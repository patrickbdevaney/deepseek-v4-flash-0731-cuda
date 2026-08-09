#!/usr/bin/env bash
# s5_session.sh — run ONE S5 training session end to end, and chain to the next only if it passes.
#
#   pass 1 generate -> pass 2 capture -> train -> build loadable head -> eval -> archive -> promote
#
# WHY IT IS ONE SCRIPT. Every step here has already failed at least once in a way that produced a
# plausible artifact: a truncated token dump that looked like data (F101), a v1 capture whose TV term
# was silently zero (F100), a port whose forward agreed with the engine 0% of the time (F102). Each
# step therefore VERIFIES its own output before the next begins, and the script stops rather than
# passing a bad artifact forward.
#
# ARCHIVE ALWAYS, PROMOTE ON MERIT. Every head this produces is saved with its sha256, its eval log
# and its training metrics -- whether or not it beats the incumbent. Promotion is a judgement;
# archiving is preservation, and a rejected head is still a measured point on the
# acceptance-vs-corpus curve that session 3's size is chosen from.
#
#   scripts/s5_session.sh <name> <prompts_file> <n> <ngen> [next_name] [next_prompts] [next_n]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
NAME="${1:?session name}"; PROMPTS="${2:?prompts file}"; N="${3:?n samples}"; NGEN="${4:-512}"
NEXT_NAME="${5:-}"; NEXT_PROMPTS="${6:-}"; NEXT_N="${7:-}"

CKPT=/home/patrickd/models/DeepSeek-V4-Flash-0731-REAP
WORK=/home/patrickd/s5-capture/$NAME
IMG=ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor
mkdir -p "$WORK"
# S5_GEN lets a session adopt a pass-1 dump produced outside this script (session 1's was already
# running when the orchestrator was written). Pass 1 is the expensive step; never redo it blindly.
GEN="${S5_GEN:-$WORK/gen.txt}"
LOG(){ printf '[s5:%s %s] %s\n' "$NAME" "$(date -Is)" "$*"; }
DIE(){ LOG "HALT: $*"; exit 1; }

SW=$(python3 -c "print(','.join(f'6:1:1.5:{i}' for i in range(1,$N+1)))")

# ---------------------------------------------------------------- preflight: will this fit on disk?
# Measured, not guessed: trial captures ran 6.8 MB per ~200-token sequence = ~33 KB/token, and the
# mtp shards are 7.01 GB. Running out of disk in the middle of a safetensors write leaves a
# truncated file that may still partially load -- far worse than refusing to start.
FREE_GB=$(df -BG --output=avail /home/patrickd | tail -1 | tr -dcs '0-9' ' ' | tr -d ' ')
CHUNK="${S5_CHUNK:-$N}"                      # default: one chunk, i.e. capture the whole corpus
NEED_GB=$(python3 -c "
# Peak is ONE CHUNK of capture, because the interleaved loop below deletes each chunk once the
# trainer has consumed it. Plus the per-chunk trained bf16 + optimizer state, the built head, and
# the archive copy of it.
cap = min($CHUNK, $N) * ($NGEN + 40) * 33e3 / 1e9
print(int(cap + 2 + 7 + 7 + 4))")
LOG "preflight: need ~${NEED_GB} GB (peak = one chunk of ${CHUNK}), have ${FREE_GB} GB free"
[ "$FREE_GB" -ge "$NEED_GB" ] || DIE "insufficient disk: need ~${NEED_GB} GB, have ${FREE_GB} GB. \
Set S5_CHUNK smaller (peak scales with the chunk, not with N) -- a truncated safetensors write is \
worse than not starting."

# ---------------------------------------------------------------- pass 1: regenerate with the target
if [ ! -s "$GEN" ]; then
    LOG "pass 1: regenerating $N responses x $NGEN tokens"
    sudo jetson_clocks >/dev/null 2>&1 || true
    sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; sleep 2
    DSV4_PROMPTS_FILE="$PROMPTS" DSV4_GENOUT="$GEN" DSV4_BLKSWEEP="$SW" \
        scripts/run_model.sh "$ROOT/evidence/${NAME}_pass1.log" ./build/decode \
        "$CKPT" "0,671,6102,294,8760,344" 8 "" "$NGEN"
    while pgrep -x decode >/dev/null; do sleep 60; done
fi
GOT=$(wc -l < "$GEN" 2>/dev/null || echo 0)
[ "$GOT" -ge "$N" ] || DIE "pass 1 produced $GOT/$N sequences"
# every generated line must begin with its seed prompt, or the round-trip mangled ids
python3 - "$PROMPTS" "$GEN" <<'PY' || DIE "pass 1 round-trip check failed"
import sys
seed=[l.strip() for l in open(sys.argv[1])]; gen=[l.strip() for l in open(sys.argv[2])]
bad=sum(1 for s,g in zip(seed,gen) if g.split(',')[:len(s.split(','))]!=s.split(','))
print(f"  round-trip: {len(gen)-bad}/{len(gen)} lines start with their exact seed")
sys.exit(1 if bad else 0)
PY
LOG "pass 1 OK ($GOT sequences)"

DOCK=(sudo docker run --rm --runtime nvidia -v "$ROOT":/work -v "$CKPT":/ckpt:ro
      -v /home/patrickd/s5-capture:/cap -w /work "$IMG")

# ---------------------------------------------------------------- pass 2 + train, INTERLEAVED
# S5_PROGRESSION §0 always priced storage as "capture a shard, train on it, delete it" -- but the
# first version of this script captured the entire corpus before training started, which at 5 000
# samples is ~123 GB against 120 GB free. Interleaving makes peak disk ONE CHUNK regardless of N.
#
# A chunked session is only equivalent to a continuous one if BOTH the weights and the AdamW
# moments carry across chunks, and the LR schedule spans the session rather than restarting inside
# each chunk. `--resume` + `--total-steps`/`--step-offset` do that; without them chunking would be
# N little trainings wearing one session's name.
# HOLD-OUT (F108). The last S5_HOLDOUT sequences are reserved for the secondary gate and are never
# trained on. Measuring the trained head on sequences it trained on would report memorisation, and
# the whole point of the secondary gate is that it estimates the CATEGORY rather than one prompt.
HOLD="${S5_HOLDOUT:-32}"
NTRAIN=$(( N - HOLD ))
[ "$NTRAIN" -gt 0 ] || DIE "N=$N leaves nothing to train on after a $HOLD-sequence hold-out"
sed -n "$((NTRAIN+1)),${N}p" "$GEN" > "$WORK/holdout.txt"
LOG "hold-out: $(wc -l < "$WORK/holdout.txt") sequence(s) reserved, $NTRAIN for training"

CHUNK="${S5_CHUNK:-$NTRAIN}"                 # default: one chunk, i.e. the original behaviour
NCHUNK=$(( (NTRAIN + CHUNK - 1) / CHUNK ))
LOG "training in $NCHUNK chunk(s) of up to $CHUNK sequence(s); peak capture on disk is one chunk"
PREV=""
OFF=0
for (( ci=0; ci<NCHUNK; ci++ )); do
    LO=$(( ci * CHUNK )); HI=$(( LO + CHUNK )); [ "$HI" -gt "$NTRAIN" ] && HI=$NTRAIN
    CDIR="$WORK/c$ci"; mkdir -p "$CDIR"
    if [ ! -s "$CDIR/cap/manifest.jsonl" ]; then
        sed -n "$((LO+1)),${HI}p" "$GEN" > "$CDIR/gen.txt"
        CN=$(wc -l < "$CDIR/gen.txt")
        [ "$CN" -gt 0 ] || DIE "chunk $ci is empty"
        LOG "chunk $ci: capturing $CN sequence(s)"
        mkdir -p "$CDIR/cap"
        sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; sleep 2
        DSV4_PROMPTS_FILE="$CDIR/gen.txt" DSV4_CAPTURE="$CDIR/cap" DSV4_NPROBE=16 \
            DSV4_BLKSWEEP="$(python3 -c "print(','.join(f'6:1:1.5:{i}' for i in range(1,$CN+1)))")" \
            scripts/run_model.sh "$ROOT/evidence/${NAME}_c${ci}_cap.log" ./build/decode \
            "$CKPT" "0,671,6102,294,8760,344" 8 "" 8
        while pgrep -x decode >/dev/null; do sleep 60; done
    fi
    python3 tools/read_capture.py "$CDIR/cap" | tail -2
    python3 tools/read_capture.py "$CDIR/cap" | grep -q "VALIDATE: PASS" \
        || DIE "chunk $ci capture validation failed"

    # The equivalence gate runs ONCE, on the first chunk. It tests the port against the engine, a
    # property of the code and not of the data, so re-running it per chunk would only re-pay a
    # weight load. It runs BEFORE any training: training against a port that disagrees with the
    # engine aligns the head to a function the server does not run (F101/F105).
    if [ "$ci" -eq 0 ]; then
        LOG "equivalence gate: port vs engine drafts"
        "${DOCK[@]}" python3 -u train/train_head.py --capture "/cap/$NAME/c0/cap" --ckpt /ckpt \
            --out "/cap/$NAME/x" --gate 2>&1 | tee "$ROOT/evidence/${NAME}_gate.log" | tail -2
        AGREE=$(grep -oE 'agreement: [0-9]+/[0-9]+ = [0-9.]+%' "$ROOT/evidence/${NAME}_gate.log" |
                tail -1 | grep -oE '[0-9.]+%$' | tr -d '%')
        python3 -c "import sys; sys.exit(0 if float('${AGREE:-0}') >= 90 else 1)" \
            || DIE "port/engine agreement ${AGREE:-?}% < 90% -- training on this would align the head to a function the server does not run (F101)"
        LOG "gate OK (${AGREE}%)"
    fi

    LOG "chunk $ci: training (ce+tv; a_conf 0 until free-running labels exist -- F100)"
    RES=(); [ -n "$PREV" ] && RES=(--resume "/cap/$NAME/$(basename "$PREV")")
    "${DOCK[@]}" python3 -u train/train_head.py --capture "/cap/$NAME/c$ci/cap" --ckpt /ckpt \
        --out "/cap/$NAME/c$ci/trained" --pos-per-seq 16 "${RES[@]}" \
        --total-steps "$NTRAIN" --step-offset "$OFF" \
        --metrics-out "/cap/$NAME/c$ci/train_metrics.json" \
        2>&1 | tee "$ROOT/evidence/${NAME}_c${ci}_train.log" | tail -4
    [ -s "$CDIR/trained/mtp_trained.safetensors" ] || DIE "chunk $ci produced no weights"
    OFF=$(( OFF + $(wc -l < "$CDIR/gen.txt") ))

    # Delete the consumed capture -- the point of chunking. gen.txt is kept, so a re-capture costs
    # only the prefill (~0.4 h per 500), never the generate pass that dominates the session.
    if [ "$NCHUNK" -gt 1 ]; then
        rm -rf "$CDIR/cap"
        LOG "chunk $ci: capture deleted, $(df -BG --output=avail /home/patrickd | tail -1 | tr -d ' ') free"
    fi
    PREV="$CDIR/trained"
done
ln -sfn "$PREV" "$WORK/trained"
python3 tools/merge_metrics.py "$WORK"/c*/train_metrics.json --out "$WORK/train_metrics.json" || true

# ---------------------------------------------------------------- write a head the ENGINE can load
LOG "building loadable head (re-quantising trained tensors to their ORIGINAL formats)"
"${DOCK[@]}" python3 -u tools/build_trained_head.py --base /ckpt \
    --trained "/cap/$NAME/trained/mtp_trained.safetensors" --out "/cap/$NAME/head" 2>&1 | tail -3
[ -s "$WORK/head/model.safetensors.index.json" ] || DIE "head build failed"

# ---------------------------------------------------------------- eval on the FROZEN protocol
LOG "eval: 8-prompt suite, NGEN0=200, block 6, clean"
SUITE=$(cat "$ROOT/protocol/suite_prompts.txt")   # frozen in-repo; a temp-dir protocol is not a protocol
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; sleep 2
DSV4_PROMPTS="$SUITE" DSV4_BLKSWEEP="6:1:1.5:0,6:1:1.5:1,6:1:1.5:2,6:1:1.5:3,6:1:1.5:4,6:1:1.5:5,6:1:1.5:6,6:1:1.5:7,6:1:1.5:8" \
    scripts/run_model.sh "$ROOT/evidence/${NAME}_eval.log" ./build/decode \
    "$CKPT" "0,671,6102,294,8760,344" 8 "$WORK/head" 200
while pgrep -x decode >/dev/null; do sleep 60; done
grep -q "LOSSLESS GATE: first 8 tokens match base AR -> PASS" "$ROOT/evidence/${NAME}_eval.log" \
    || DIE "LOSSLESS gate failed -- the fine-tuned head changes the emitted sequence"

# ---------------------------------------------------------------- secondary gate (F108)
# The pre-registered gate reads tau off ONE suite prompt, which F108 measured to be below the
# minimum of 63 samples of its own category. This runs the SAME hold-out prompts the head never
# trained on, with the trained head, and pairs them against the untrained numbers already in the
# pass-1 log -- same prompts, same budget, one variable.
LOG "secondary gate: $HOLD-prompt reasoning hold-out, trained head"
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; sleep 2
DSV4_PROMPTS_FILE="$WORK/holdout.txt" \
    DSV4_BLKSWEEP="$(python3 -c "print(','.join(f'6:1:1.5:{i}' for i in range(1,$HOLD+1)))")" \
    scripts/run_model.sh "$ROOT/evidence/${NAME}_holdout.log" ./build/decode \
    "$CKPT" "0,671,6102,294,8760,344" 8 "$WORK/head" 220
while pgrep -x decode >/dev/null; do sleep 60; done
BASE_MED=$(python3 tools/holdout_tau.py --log "$ROOT/evidence/${NAME}_pass1.log" \
             --only-last "$HOLD" --json-out "$WORK/holdout_untrained.json" 2>/dev/null |
           grep -oE 'median [0-9.]+' | head -1 | grep -oE '[0-9.]+' || echo 3.483)
LOG "hold-out untrained baseline (paired, from pass 1): $BASE_MED"
set +e
python3 tools/holdout_tau.py --log "$ROOT/evidence/${NAME}_holdout.log" --baseline "$BASE_MED" \
    --json-out "$WORK/holdout_trained.json" | tee -a "$ROOT/evidence/${NAME}_verdict.log"
V2=${PIPESTATUS[0]}
set -e

# ---------------------------------------------------------------- archive ALWAYS
# Unconditional, and BEFORE any judgement. A rejected head is still a measured point on the
# acceptance-vs-corpus curve that session 3's size gets chosen from; losing it because it did not
# win would mean re-paying its capture to learn the same thing twice.
python3 tools/promote_head.py archive --head "$WORK/head" --eval "$ROOT/evidence/${NAME}_eval.log" \
    --name "$NAME" --metrics "$WORK/train_metrics.json" --notes "auto: $N samples, ngen $NGEN"

# ---------------------------------------------------------------- promote (may fail; not the chain rule)
python3 tools/promote_head.py promote --head "$WORK/head" --eval "$ROOT/evidence/${NAME}_eval.log" \
    --name "$NAME" --metrics "$WORK/train_metrics.json" && LOG "PROMOTED" || LOG "not promoted (archived)"

# ---------------------------------------------------------------- the SESSION decision
# Thresholds live in S5_PROGRESSION.md §2 and were fixed before the data. Note this is deliberately
# NOT the promotion rule: session 1 is a single-domain proof and can legitimately say GO while
# failing a suite-wide >3.5% promotion bar.
set +e
python3 tools/session_gate.py --eval "$ROOT/evidence/${NAME}_eval.log" \
    --json-out "$WORK/session_verdict.json" | tee -a "$ROOT/evidence/${NAME}_verdict.log"
V=${PIPESTATUS[0]}
set -e
# COMBINING THE TWO GATES. The primary (pre-registered, one suite prompt) stays authoritative for
# anything it PASSES -- a rule can only be trusted if it is not renegotiated when inconvenient. The
# secondary can rescue a primary STOP, because F108 showed the primary's prompt is below the
# minimum of its own category and a category-level estimate is the better evidence. It cannot do
# the reverse: a secondary GO against a primary that saw the suite mean DROP or the LOSSLESS gate
# fail is not a rescue, it is overfitting to the hold-out, so those two reasons are absolute.
# Every hard reason session_gate.py can write, matched literally. Getting this list wrong in the
# permissive direction lets the secondary rescue a run that changed the emitted sequence -- the one
# thing this project never trades (F68). Checked against the strings the tool actually emits.
HARD=$(grep -cE "LOSSLESS|GATE FAIL|suite mean tau .* DROPPED|instruments|no reasoning-category" \
       "$WORK/session_verdict.json" 2>/dev/null || echo 0)
if [ "$V" = "3" ] && [ "$V2" = "0" ] && [ "$HARD" = "0" ]; then
    LOG "primary STOP but secondary GO on the $HOLD-prompt hold-out, and no hard failure -- \
proceeding per F108: the primary reads one prompt that sits below its own category's minimum"
    V=2
fi
case "$V" in
  0) LOG "GO (primary); secondary exit $V2" ;;
  2) LOG "GO_REPRICE -- proceeding, but the next session's value estimate is halved" ;;
  *) LOG "STOP: primary $V, secondary $V2. Head and metrics are archived. Chain ends."; exit 3 ;;
esac

if [ -n "$NEXT_NAME" ]; then
    LOG "chaining to session $NEXT_NAME"
    exec "$0" "$NEXT_NAME" "$NEXT_PROMPTS" "$NEXT_N" "$NGEN"
fi
LOG "no next session configured; stopping here"
