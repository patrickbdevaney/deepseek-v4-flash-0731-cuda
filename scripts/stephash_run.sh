#!/usr/bin/env bash
# stephash_run.sh — ladder 1.9. WHY THE ENGINE STOPS REPRODUCING ITSELF PART-WAY THROUGH A RUN.
#
# THE SHAPE OF THE THING, restated because it is the whole design of this probe. Two runs of the
# SAME binary with the SAME env diverge: point 0 (ctx 6, 260 ids) is identical across three runs;
# point 1 (ctx 3,072) and point 2 (ctx 1,024) differ in every pairing, first at generated token 25
# and 43 (evidence/decode_loop/genout_1p0_determinism.txt). Generation is autoregressive, so ONE
# flipped token hides everything after it -- a token-id diff can name the STEP and never the
# QUANTITY. `DSV4_STEPHASH` (src/decode.cu, ladder 1.9) writes one line per verify carrying the
# whole causal chain of that step in dataflow order:
#
#     mkv -> mx -> din -> draft -> lg -> acc/corr
#
# so `diff` of two runs lands on the first FIELD that differs and that field names the link:
#   * mkv/mx differ first          -> the persistent draft inputs already drifted (upstream)
#   * mkv+mx match, draft differs  -> the DSpark draft chain is order-nondeterministic
#   * everything to draft matches, lg differs -> the 43-layer target verify is
#   * lg matches, acc differs      -> host-side accept logic, i.e. not the GPU at all
#
# THE CONFIG IS NOT A CHOICE. It is byte-for-byte the arm that produced the known divergence in
# 1.0: same prompts file, same sweep, NDEC=16, NGEN=256, DSV4_MAINKV_CACHE=0 on BOTH sides. Using
# the cached path here would put a new variable next to the one being measured, and using a shorter
# NGEN would risk a probe that reports clean because it never reached the divergence.
#
#   nohup setsid bash scripts/stephash_run.sh > evidence/decode_loop/stephash.log 2>&1 </dev/null &
set -u
cd "$(dirname "$0")/.."
CKPT="${CKPT:-/home/patrickd/models/DeepSeek-V4-Flash-0731-REAP}"
EV=evidence/decode_loop
PROMPTS="${PROMPTS:-$EV/mainkv_prompts.txt}"
SWEEP="${SWEEP:-6:1:1.50:0,6:1:1.50:1,6:1:1.50:2}"
NDEC="${NDEC:-16}"; NGEN="${NGEN:-256}"
LVL="${LVL:-2}"
# ARMS/XENV let the same protocol be re-run under a hypothesis without editing it. XENV is a
# space-separated list of VAR=VAL prepended to both arms, so the two arms stay identical to each
# other by construction -- which is the whole point of a self-vs-self comparison.
ARMS="${ARMS:-A B}"
XENV="${XENV:-}"
say(){ echo "[1.9] $(date -Is) $*"; }

[ -s "$PROMPTS" ] || { say "REFUSING: $PROMPTS missing."; exit 1; }
for f in src/decode.cu kernels/dspark_attn.cu kernels/dspark.cu include/dspark.h; do
  [ -e "$f" ] && [ "$f" -nt build/decode ] && { say "REFUSING: build/decode is older than $f."; exit 1; }
done
say "binary: decode $(date -Is -r build/decode)  sweep=$SWEEP NGEN=$NGEN level=$LVL"

run_one(){   # $1 = tag
  local tag="$1"
  local SH=$EV/stephash_$tag.txt LOG=$EV/stephash_$tag.log GO=$EV/genout_1p9_$tag.txt
  rm -f "$SH" "$GO"                                   # DSV4_GENOUT appends
  env DSV4_PROMPTS_FILE="$PROMPTS" DSV4_BLKSWEEP="$SWEEP" DSV4_GENOUT="$GO" \
      DSV4_MAINKV_CACHE=0 DSV4_STEPHASH="$SH" DSV4_STEPHASH_LVL="$LVL" $XENV \
      bash scripts/run_model.sh "$LOG" ./build/decode "$CKPT" \
      "0,671,6102,294,8760,344" "$NDEC" "" "$NGEN" || { say "FATAL: run_model refused ($tag)"; return 1; }
  local i
  for i in $(seq 1 60);  do pgrep -x decode > /dev/null && break; sleep 2;  done
  pgrep -x decode > /dev/null || { say "FATAL: decode never started ($tag)."; tail -30 "$LOG"; return 1; }
  for i in $(seq 1 1080); do pgrep -x decode > /dev/null || break; sleep 10; done   # 3 h cap
  pgrep -x decode > /dev/null && { say "FATAL: decode still running after 3 h ($tag)."; return 1; }
  local n; n=$(wc -l < "$SH" 2>/dev/null || echo 0)
  say "$tag: $n stephash lines, $(wc -l < "$GO" 2>/dev/null || echo 0) genout lines"
  [ "$n" -gt 0 ] || { say "FATAL: no stephash lines from $tag"; tail -30 "$LOG"; return 1; }
  return 0
}

set -- $ARMS
T1="$1"; T2="$2"
run_one "$T1" || exit 1
run_one "$T2" || exit 1

say "=== token-id level (what 1.0 saw) ==="
python3 tools/genout_compare.py $EV/genout_1p9_$T1.txt $EV/genout_1p9_$T2.txt | tee $EV/stephash_genout_$T1$T2.txt
say "=== per-step field level (what this probe adds) ==="
python3 tools/stephash_compare.py $EV/stephash_$T1.txt $EV/stephash_$T2.txt | tee $EV/stephash_verdict_$T1$T2.txt
say "done."
