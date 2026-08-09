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
NEED_GB=$(python3 -c "
cap = $N * ($NGEN + 40) * 33e3 / 1e9    # taps + lm_in over prompt+generated
print(int(cap + 1 + 7 + 7 + 4))          # + trained bf16 + built head + archive copy + slack")
LOG "preflight: need ~${NEED_GB} GB, have ${FREE_GB} GB free"
[ "$FREE_GB" -ge "$NEED_GB" ] || DIE "insufficient disk: need ~${NEED_GB} GB, have ${FREE_GB} GB. \
Free space or reduce N before starting -- a truncated safetensors write is worse than not starting."

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

# ---------------------------------------------------------------- pass 2: capture taps + lm_in + probes
if [ ! -s "$WORK/cap/manifest.jsonl" ]; then
    LOG "pass 2: capturing taps, lm_head input and engine draft probes"
    mkdir -p "$WORK/cap"
    sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; sleep 2
    DSV4_PROMPTS_FILE="$GEN" DSV4_CAPTURE="$WORK/cap" DSV4_NPROBE=16 DSV4_BLKSWEEP="$SW" \
        scripts/run_model.sh "$ROOT/evidence/${NAME}_pass2.log" ./build/decode \
        "$CKPT" "0,671,6102,294,8760,344" 8 "" 8
    while pgrep -x decode >/dev/null; do sleep 60; done
fi
python3 tools/read_capture.py "$WORK/cap" | tail -3
python3 tools/read_capture.py "$WORK/cap" | grep -q "VALIDATE: PASS" || DIE "capture validation failed"
LOG "pass 2 OK"

DOCK=(sudo docker run --rm --runtime nvidia -v "$ROOT":/work -v "$CKPT":/ckpt:ro
      -v /home/patrickd/s5-capture:/cap -w /work "$IMG")

# ---------------------------------------------------------------- equivalence gate BEFORE training
LOG "equivalence gate: port vs engine drafts"
"${DOCK[@]}" python3 -u train/train_head.py --capture "/cap/$NAME/cap" --ckpt /ckpt \
    --out /cap/$NAME/x --gate 2>&1 | tee "$ROOT/evidence/${NAME}_gate.log" | tail -2
AGREE=$(grep -oE 'agreement: [0-9]+/[0-9]+ = [0-9.]+%' "$ROOT/evidence/${NAME}_gate.log" | tail -1 |
        grep -oE '[0-9.]+%$' | tr -d '%')
python3 -c "import sys; sys.exit(0 if float('${AGREE:-0}') >= 90 else 1)" \
    || DIE "port/engine agreement ${AGREE:-?}% < 90% -- training on this would align the head to a function the server does not run (F101)"
LOG "gate OK (${AGREE}%)"

# ---------------------------------------------------------------- train
LOG "training (1 epoch, ce+tv; a_conf 0 until free-running labels exist -- F100)"
"${DOCK[@]}" python3 -u train/train_head.py --capture "/cap/$NAME/cap" --ckpt /ckpt \
    --out "/cap/$NAME/trained" --pos-per-seq 16 --metrics-out "/cap/$NAME/train_metrics.json" \
    2>&1 | tee "$ROOT/evidence/${NAME}_train.log" | tail -4
[ -s "$WORK/trained/mtp_trained.safetensors" ] || DIE "trainer produced no weights"

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
case "$V" in
  0) LOG "GO" ;;
  2) LOG "GO_REPRICE -- proceeding, but the next session's value estimate is halved" ;;
  *) LOG "STOP by the pre-registered rule. Head and metrics are archived. Chain ends."; exit 3 ;;
esac

if [ -n "$NEXT_NAME" ]; then
    LOG "chaining to session $NEXT_NAME"
    exec "$0" "$NEXT_NAME" "$NEXT_PROMPTS" "$NEXT_N" "$NGEN"
fi
LOG "no next session configured; stopping here"
