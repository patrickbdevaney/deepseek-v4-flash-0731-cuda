#!/usr/bin/env bash
# s5_preflight.sh -- run the ENTIRE session pipeline at toy scale before paying for the real one.
#
# WHY THIS EXISTS. Every failure S5 has had was in a code path that only executes hours into a run:
#
#   s1  the `trained` symlink was ABSOLUTE, so it resolved on the host and dangled inside the
#       container -- found at the build step, after 2 h of capture and a clean train
#   s1  build_trained_head refused the 27 fp32 tensors -- found at the build step, again
#   s1  `... | head -1 | ... || echo D` under pipefail emitted the value AND the fallback -- found at
#       the secondary gate, after the eval it was meant to judge had already run
#   s3  --resume was built with `basename`, pointing at a symlink created only AFTER the chunk loop
#   s3  the AdamW resume asserted len(state) == len(params), which lazy state never satisfies
#
# None were hard bugs. All five were cheap to find and expensive to hit, and the last two had NEVER
# executed before -- s1 and s2 both ran in a single chunk, so the entire resume path was dead code
# that first ran four hours into a nineteen-hour session.
#
# The general fix is not more care, it is EXERCISING THE LATE STAGES EARLY. This runs the real
# `s5_session.sh`, unmodified, on 4 prompts with a 2-chunk split -- so the multi-chunk resume, the
# container path resolution, the head build, the re-fit, both evals, the gate parse and the archive
# all execute in ~20 minutes instead of at hour 15.
#
# WHAT "PASS" MEANS HERE. That every stage RAN, not that the head is good. A head trained on two
# sequences is garbage and its quality gates will fail; that is expected and is not a preflight
# failure. The preflight asserts stage MARKERS in the log, and ignores the verdict.
#
#   scripts/s5_preflight.sh [corpus] [name]
set -uo pipefail
ROOT=/home/patrickd/deepseek-v4-flash-0731-cuda
CORPUS="${1:-/home/patrickd/s5-capture/mixed_prompts_s3.txt}"
NAME="${2:-preflight}"
WORK=/home/patrickd/s5-capture/$NAME
cd "$ROOT"

[ -s "$CORPUS" ] || { echo "no corpus at $CORPUS"; exit 1; }

echo "[preflight] tearing down any previous run"
sudo rm -rf "$WORK" "$HOME/model-backups/heads/$NAME"
mkdir -p "$WORK"

# Real prompts from the real corpus -- toy SCALE, never toy DATA. A synthetic prompt would exercise
# a tokenisation path the session never takes.
head -4 "$CORPUS" > "$WORK/prompts.txt"
echo "[preflight] 4 real prompts, 16 generated tokens, hold-out 2, chunk 1 -> 2 chunks"
echo "[preflight]   the 2-chunk split is the point: it is the only way to execute --resume"

START=$(date +%s)
S5_HOLDOUT=2 S5_CHUNK=1 timeout 5400 scripts/s5_session.sh "$NAME" "$WORK/prompts.txt" 4 16 \
    > "$ROOT/evidence/${NAME}_run.log" 2>&1
RC=$?
EL=$(( ($(date +%s) - START) / 60 ))
LOG="$ROOT/evidence/${NAME}_run.log"

# Stage markers, in the order the pipeline must reach them. The session's own quality verdict is
# deliberately NOT one of them.
declare -a STAGES=(
  "pass 1 OK|pass 1 generate + round-trip check"
  "chunk 0: capturing|chunk 0 capture"
  "VALIDATE: PASS|capture validation"
  "chunk 0: training|chunk 0 train"
  "chunk 1: training|chunk 1 train (THE RESUME PATH)"
  "resumed AdamW state|AdamW moments carried across chunks"
  "building loadable head|head build"
  "re-quantised|head re-quantisation + round-trip gate"
  "re-fitting adaptK|adaptK re-fit"
  "LOSSLESS GATE|lossless gate evaluated"
  "secondary gate|secondary gate"
  "paired control|paired untrained control"
  "ARCHIVED|archive"
)
echo
echo "[preflight] stage coverage after ${EL} min (session exit $RC):"
MISS=0
for S in "${STAGES[@]}"; do
    PAT="${S%%|*}"; DESC="${S##*|}"
    if grep -qF "$PAT" "$LOG" 2>/dev/null || grep -qF "$PAT" "$ROOT/evidence/${NAME}"_*.log 2>/dev/null; then
        printf "  \033[32mOK  \033[0m %s\n" "$DESC"
    else
        printf "  \033[31mMISS\033[0m %s   (marker: %s)\n" "$DESC" "$PAT"
        MISS=$((MISS+1))
    fi
done

# The packaging path is downstream of the session and has its own failure modes, so exercise it too.
if [ -s "$WORK/c1/trained/mtp_trained.safetensors" ]; then
    if python3 tools/package_release.py --head "$WORK/c1/trained" \
         --out "$WORK/release_test" --name "$NAME" >/dev/null 2>&1; then
        printf "  \033[32mOK  \033[0m release packaging\n"
    else
        printf "  \033[31mMISS\033[0m release packaging\n"; MISS=$((MISS+1))
    fi
fi

echo
if [ "$MISS" -eq 0 ]; then
    echo "[preflight] PASS -- every stage executed. The pipeline is safe to run at full scale."
    echo "[preflight] (the trained head itself is garbage: 2 sequences. That is not what this tests.)"
else
    echo "[preflight] FAIL -- $MISS stage(s) never executed. Fix before spending a full session;"
    echo "[preflight] a stage that does not run here will not run at hour 15 either."
    echo "[preflight] log: $LOG"
fi
sudo rm -rf "$HOME/model-backups/heads/$NAME"        # never leave a toy head in the registry store
exit $MISS
