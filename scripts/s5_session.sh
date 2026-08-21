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

# THE BLOCK WIDTH, IN ONE PLACE. Ladder 2.1 shipped 6 -> 5 on 2026-08-21 (+3.91 +/- 1.65 % at
# equal tau, ids bit-identical). It was hardcoded as `6` in seven sweep strings here and gated as
# `[6]` in promote_head.py. It has to move together everywhere or the session trains a head for one
# width, evaluates it at another, and grades it against archived numbers taken at a third.
#
# tau IS NOT COMPARABLE ACROSS WIDTHS -- its ceiling is the width. Every tau in HEAD_REGISTRY.md was
# taken at block 6 and a block-5 tau cannot be measured against it. That is what --incumbent-tau on
# promote_head.py is for, and S5_INCUMBENT_TAU below carries a value re-measured at THIS width.
BLK="${S5_BLOCK:-5}"
SW=$(python3 -c "print(','.join(f'$BLK:1:1.5:{i}' for i in range(1,$N+1)))")

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
    # RESUMABLE. A chunk whose weights already exist is done; re-running the session then costs the
    # build onward, not the ~2 h capture. Without this, a failure in ANY later stage (build, refit,
    # eval) forced a full re-capture to retry a step that had nothing to do with the data.
    if [ -s "$CDIR/trained/mtp_trained.safetensors" ]; then
        LOG "chunk $ci: already trained, reusing (rm -rf $CDIR/trained to force a re-run)"
        OFF=$(( OFF + $(wc -l < "$CDIR/gen.txt" 2>/dev/null || echo 0) ))
        PREV="c$ci/trained"; continue
    fi
    if [ ! -s "$CDIR/cap/manifest.jsonl" ]; then
        sed -n "$((LO+1)),${HI}p" "$GEN" > "$CDIR/gen.txt"
        CN=$(wc -l < "$CDIR/gen.txt")
        [ "$CN" -gt 0 ] || DIE "chunk $ci is empty"
        LOG "chunk $ci: capturing $CN sequence(s)"
        mkdir -p "$CDIR/cap"
        sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; sleep 2
        DSV4_PROMPTS_FILE="$CDIR/gen.txt" DSV4_CAPTURE="$CDIR/cap" DSV4_NPROBE=16 \
            DSV4_BLKSWEEP="$(python3 -c "print(','.join(f'$BLK:1:1.5:{i}' for i in range(1,$CN+1)))")" \
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
            --out "/cap/$NAME/x" --block "$BLK" --gate 2>&1 | tee "$ROOT/evidence/${NAME}_gate.log" | tail -2
        AGREE=$(grep -oE 'agreement: [0-9]+/[0-9]+ = [0-9.]+%' "$ROOT/evidence/${NAME}_gate.log" |
                tail -1 | grep -oE '[0-9.]+%$' | tr -d '%')
        python3 -c "import sys; sys.exit(0 if float('${AGREE:-0}') >= 90 else 1)" \
            || DIE "port/engine agreement ${AGREE:-?}% < 90% -- training on this would align the head to a function the server does not run (F101)"
        LOG "gate OK (${AGREE}%)"
    fi

    LOG "chunk $ci: training (ce+tv; a_conf 0 until free-running labels exist -- F100)"
    # PREV is already chunk-relative ("c0/trained"); `basename` collapsed it to "trained" and
    # pointed --resume at /cap/$NAME/trained, which does not exist DURING the loop -- that symlink is
    # created after it. Every prior session ran in a single chunk (NCHUNK=1, so PREV stayed empty and
    # this line never fired), which is how a broken resume survived two sessions unexercised and then
    # killed s3 four hours in, after chunk 0 had trained fine and chunk 1 had captured and validated.
    RES=(); [ -n "$PREV" ] && RES=(--resume "/cap/$NAME/$PREV")
    # S5_ACE/S5_ATV expose the loss mix so a recipe sweep can reuse a RETAINED capture instead of
    # re-capturing per arm. Unset leaves train_head.py's own defaults (0.1 / 0.9) untouched, so the
    # sessions already recorded in HEAD_REGISTRY.md remain exactly reproducible by omitting them.
    LOSSW=()
    [ -n "${S5_ACE:-}" ] && LOSSW+=(--a-ce "$S5_ACE")
    [ -n "${S5_ATV:-}" ] && LOSSW+=(--a-tv "$S5_ATV")
    if [ -n "$PREV" ] && [ ! -s "$WORK/$PREV/mtp_trained.safetensors" ]; then
        DIE "chunk $ci: --resume target $WORK/$PREV/mtp_trained.safetensors is missing; refusing to \
train a chunk as if it were a fresh session (that would silently discard every earlier chunk)"
    fi
    "${DOCK[@]}" python3 -u train/train_head.py --capture "/cap/$NAME/c$ci/cap" --ckpt /ckpt \
        --out "/cap/$NAME/c$ci/trained" --pos-per-seq 16 --block "$BLK" "${RES[@]}" "${LOSSW[@]}" \
        --total-steps "$NTRAIN" --step-offset "$OFF" \
        --metrics-out "/cap/$NAME/c$ci/train_metrics.json" \
        2>&1 | tee "$ROOT/evidence/${NAME}_c${ci}_train.log" | tail -4
    [ -s "$CDIR/trained/mtp_trained.safetensors" ] || DIE "chunk $ci produced no weights"
    OFF=$(( OFF + $(wc -l < "$CDIR/gen.txt") ))

    # Delete the consumed capture -- the point of chunking. gen.txt is kept, so a re-capture costs
    # only the prefill (~0.4 h per 500), never the generate pass that dominates the session.
    # S5_KEEP_CAP=1 retains the capture. Deleting it is right when disk is the binding constraint
    # (5 000 samples is ~123 GB against 120 GB free, which is why chunking exists at all). It is
    # WRONG when it is not: a retained capture makes every later RECIPE variant -- a_ce/a_tv, lr,
    # pos-per-seq -- cost training only, and re-capturing is ~0.4 h per 500 sequences that need not
    # be spent. Check the actual headroom rather than assuming the 2026-08 corpus sizes.
    if [ "$NCHUNK" -gt 1 ] && [ "${S5_KEEP_CAP:-0}" != "1" ]; then
        rm -rf "$CDIR/cap"
        LOG "chunk $ci: capture deleted, $(df -BG --output=avail /home/patrickd | tail -1 | tr -d ' ') free"
    fi
    PREV="c$ci/trained"
done
# RELATIVE, deliberately. $WORK is bind-mounted into the container at /cap/$NAME, so an ABSOLUTE
# symlink to /home/patrickd/... resolves on the host and dangles inside the container -- which is
# exactly how the first s1 run died, after 2 h of capture and a clean train, at the build step.
ln -sfn "$PREV" "$WORK/trained"
[ -s "$WORK/trained/mtp_trained.safetensors" ] || DIE "trained symlink does not resolve: $PREV"
python3 tools/merge_metrics.py "$WORK"/c*/train_metrics.json --out "$WORK/train_metrics.json" || true

# ---------------------------------------------------------------- write a head the ENGINE can load
LOG "building loadable head (re-quantising trained tensors to their ORIGINAL formats)"
"${DOCK[@]}" python3 -u tools/build_trained_head.py --base /ckpt \
    --trained "/cap/$NAME/trained/mtp_trained.safetensors" --out "/cap/$NAME/head" 2>&1 | tail -3
[ -s "$WORK/head/model.safetensors.index.json" ] || DIE "head build failed"

# ---------------------------------------------------------------- re-fit adaptK (F111)
# REQUIRED, not optional. The gate's job is to skip verify work that would be rejected, so the
# optimal threshold is a function of drafter reliability -- and this session just changed drafter
# reliability. Evaluating the trained head at a threshold fitted to the OLD drafter charges it for
# a setting it did not choose, and F111's sweep says the optimum moves DOWN as the head improves.
#
# Fitted on the hold-out, never on the eval suite: fitting a hyperparameter on the set you then
# report is the oldest way to manufacture a win.
LOG "re-fitting adaptK on the trained head (F111: the optimum moves with drafter quality)"
BEST=1.5; BESTR=0
for THR in 0.0 0.5 1.0 1.5 2.0; do
    sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; sleep 2
    DSV4_PROMPTS_FILE="$WORK/holdout.txt" \
        DSV4_BLKSWEEP="$(python3 -c "print(','.join(f'$BLK:1:$THR:{i}' for i in range(1,9)))")" \
        scripts/run_model.sh "$ROOT/evidence/${NAME}_adaptk_${THR}.log" ./build/decode \
        "$CKPT" "0,671,6102,294,8760,344" 8 "$WORK/head" 220
    while pgrep -x decode >/dev/null; do sleep 30; done
    # SELECT ON RATE, NOT tau. tau is tokens per verify; the objective is tokens per second, and
    # rate = tau / (ms per verify). Lowering the threshold widens the verify, buying tau at a
    # more-than-proportional cost in ms, so the two anti-correlate. On s1 the tau criterion selected
    # adaptK 0.5, which measured the WORST rate of the five swept (25.12 vs 26.78 at 1.5) -- it did
    # not just miss the optimum, it walked away from it. tau is still logged: it is the acceptance
    # half of the story and the secondary gate is defined on it.
    # same pipefail/SIGPIPE trap as the secondary gate below -- go through the JSON, not a grep chain
    python3 tools/holdout_tau.py --log "$ROOT/evidence/${NAME}_adaptk_${THR}.log" \
        --json-out "$WORK/tau_${THR}.json" >/dev/null 2>&1 || true
    T=$(python3 -c "
import json
try: print(float(json.load(open('$WORK/tau_${THR}.json'))['median']))
except Exception: print(0)" 2>/dev/null || echo 0)
    R=$(python3 tools/holdout_rate.py --log "$ROOT/evidence/${NAME}_adaptk_${THR}.log" \
          --quiet 2>/dev/null || echo 0)
    LOG "  adaptK $THR -> hold-out $R tok/s pooled (median tau $T)"
    python3 -c "import sys; sys.exit(0 if float('$R') > float('$BESTR') else 1)" \
        && { BEST=$THR; BESTR=$R; }
done
LOG "adaptK re-fit: $BEST ($BESTR tok/s pooled on the hold-out); shipped default was 1.5"
echo "{\"adaptk\": $BEST, \"holdout_tok_s\": $BESTR, \"criterion\": \"pooled tok/s\"}" \
    > "$WORK/adaptk.json"

# ---------------------------------------------------------------- eval on the FROZEN protocol
LOG "eval: 8-prompt suite, NGEN0=200, block $BLK, clean"
SUITE=$(cat "$ROOT/protocol/suite_prompts.txt")   # frozen in-repo; a temp-dir protocol is not a protocol
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; sleep 2
DSV4_PROMPTS="$SUITE" DSV4_BLKSWEEP="$(python3 -c "print(','.join(f'$BLK:1:1.5:{i}' for i in range(0,9)))")" \
    scripts/run_model.sh "$ROOT/evidence/${NAME}_eval.log" ./build/decode \
    "$CKPT" "0,671,6102,294,8760,344" 8 "$WORK/head" 200
while pgrep -x decode >/dev/null; do sleep 60; done
grep -q "LOSSLESS GATE: first 8 tokens match base AR -> PASS" "$ROOT/evidence/${NAME}_eval.log" \
    || DIE "LOSSLESS gate failed -- the fine-tuned head changes the emitted sequence"

# THE PROTOCOL EVAL ABOVE STAYS AT adaptK 1.50, always. It is the number that goes in the registry,
# because comparability with the F96 baseline is the whole point of freezing a protocol and a
# re-fitted threshold would silently change what "suite mean" means. If the re-fit found something
# better, it is measured HERE, separately, and shipping it would require re-baselining the
# incumbent at the same threshold -- which is a decision, not a side effect.
if [ "$BEST" != "1.5" ]; then
    LOG "second eval at the re-fitted adaptK $BEST (reported separately; NOT the registry number)"
    sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; sleep 2
    DSV4_PROMPTS="$SUITE" \
        DSV4_BLKSWEEP="$(python3 -c "print(','.join(f'$BLK:1:$BEST:{i}' for i in range(0,9)))")" \
        scripts/run_model.sh "$ROOT/evidence/${NAME}_eval_adaptk${BEST}.log" ./build/decode \
        "$CKPT" "0,671,6102,294,8760,344" 8 "$WORK/head" 200
    while pgrep -x decode >/dev/null; do sleep 30; done
    python3 tools/session_gate.py --eval "$ROOT/evidence/${NAME}_eval_adaptk${BEST}.log" \
        --json-out "$WORK/session_verdict_adaptk${BEST}.json" || true
    LOG "NOTE: promoting at adaptK $BEST requires re-measuring the incumbent at $BEST first."
fi

# ---------------------------------------------------------------- secondary gate (F108)
# The pre-registered gate reads tau off ONE suite prompt, which F108 measured to be below the
# minimum of 63 samples of its own category. This runs the SAME hold-out prompts the head never
# trained on, with the trained head, and pairs them against the untrained numbers already in the
# pass-1 log -- same prompts, same budget, one variable.
LOG "secondary gate: $HOLD-prompt reasoning hold-out, trained head"
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; sleep 2
DSV4_PROMPTS_FILE="$WORK/holdout.txt" \
    DSV4_BLKSWEEP="$(python3 -c "print(','.join(f'$BLK:1:1.5:{i}' for i in range(1,$HOLD+1)))")" \
    scripts/run_model.sh "$ROOT/evidence/${NAME}_holdout.log" ./build/decode \
    "$CKPT" "0,671,6102,294,8760,344" 8 "$WORK/head" 220
while pgrep -x decode >/dev/null; do sleep 60; done
# A TRUE PAIRED CONTROL, NOT A PROXY. This gate used to read its untrained baseline out of the
# pass-1 log. That is the same prompts but NOT the same measurement: pass 1 drafts from a short seed
# over positions 0..512, while this eval prefills a 512-token prompt and drafts positions 512..732.
# On s1 the regime gap was worth about +1.36 tau -- far larger than the training effect it was
# supposed to measure -- and it flipped the verdict. Proxy said tau 3.584 -> 4.536, "+0.952, GO".
# The real control, untrained head over these exact prompts and budget, measured 4.940: training had
# made the hold-out WORSE by 0.30 tau, and the gate reported a pass.
#
# The control depends only on the hold-out file and the UNTRAINED head, so it is identical for every
# session over the same corpus -- run it once, cache it, reuse it.
if [ ! -s "$WORK/holdout_control.json" ]; then
    LOG "paired control: UNTRAINED head over the same hold-out (once per corpus, then cached)"
    sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; sleep 2
    DSV4_PROMPTS_FILE="$WORK/holdout.txt" \
        DSV4_BLKSWEEP="$(python3 -c "print(','.join(f'$BLK:1:1.5:{i}' for i in range(1,$HOLD+1)))")" \
        scripts/run_model.sh "$ROOT/evidence/${NAME}_holdout_control.log" ./build/decode \
        "$CKPT" "0,671,6102,294,8760,344" 8 "" 220
    while pgrep -x decode >/dev/null; do sleep 30; done
    python3 tools/holdout_tau.py --log "$ROOT/evidence/${NAME}_holdout_control.log" \
        --json-out "$WORK/holdout_control.json" >/dev/null 2>&1 || true
fi
BASE_MED=$(python3 -c "
import json
try: print(float(json.load(open('$WORK/holdout_control.json'))['median']))
except Exception: print(0)" 2>/dev/null || echo 0)
[ "$BASE_MED" = "0" ] && DIE "paired control produced no baseline -- refusing to judge the trained \
head against a proxy measured in a different context regime (that is exactly how s1 passed a gate \
it should have failed)"
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
INC=(); [ -n "${S5_INCUMBENT_TAU:-}" ] && INC=(--incumbent-tau "$S5_INCUMBENT_TAU")
python3 tools/promote_head.py promote "${INC[@]}" --head "$WORK/head" --eval "$ROOT/evidence/${NAME}_eval.log" \
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
