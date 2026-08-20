#!/usr/bin/env bash
# mainkv_determinism_run.sh — ladder 1.0, the control that decides whether the token-id
# divergence can be attributed to the cache at all.
#
# THE STATE OF THE EVIDENCE. The cached main-KV prefix is byte-identical to the from-scratch one:
#   * unit gate, no checkpoint: 2048 x 512 floats identical across 22 split points, BOTH GEMM paths
#   * build/dsv4-server, in situ: 384 calls, contexts to 12,281, 2,023,320 rows kept, 0 FAIL
#   * build/decode,     in situ: 320 calls, contexts to 3,130,     568,509 rows kept, 0 FAIL
# All three memcmp the WHOLE [s, HEAD_DIM] buffer against the untouched dspark_main_kv and abort on
# the first differing float. Nothing downstream of the head reads anything else from that call.
#
# AND YET the two arms emitted different token ids on 2 of 3 points. Both explanations are still
# open and they are not distinguishable from the A/B alone:
#   (a) something downstream is sensitive to the cache in a way the buffer comparison misses
#   (b) this binary does not reproduce itself run-to-run, and the A/B was measuring that
#
# ONLY (b) IS TESTABLE WITHOUT ASSUMING THE ANSWER, and it is testable directly: run the BASELINE
# arm a second time -- same binary, same env, DSV4_MAINKV_CACHE=0 on both sides, the cached code
# path never executed -- and compare it to the baseline ids already on disk. Same arm on both
# sides, so every difference is the binary disagreeing with ITSELF and none of it can be the cache.
# If this run is identical to the first baseline, (b) is dead and the divergence is real and mine.
#
#   nohup setsid bash scripts/mainkv_determinism_run.sh > evidence/decode_loop/mainkv_determinism.log 2>&1 </dev/null &
set -u
cd "$(dirname "$0")/.."
CKPT="${CKPT:-/home/patrickd/models/DeepSeek-V4-Flash-0731-REAP}"
EV=evidence/decode_loop
PROMPTS=$EV/mainkv_prompts.txt
SWEEP="${SWEEP:-6:1:1.50:0,6:1:1.50:1,6:1:1.50:2}"
NDEC="${NDEC:-16}"; NGEN="${NGEN:-256}"      # IDENTICAL to the arm it is compared against
REF=$EV/genout_1p0_off.txt
GO=$EV/genout_1p0_off2.txt
LOG=$EV/decode_1p0_off2.log
say(){ echo "[1.0d] $(date -Is) $*"; }

[ -s "$REF" ] || { say "REFUSING: $REF missing; nothing to compare against."; exit 1; }
for f in src/decode.cu kernels/dspark_attn.cu include/dspark_attn.h; do
  [ "$f" -nt build/decode ] && { say "REFUSING: build/decode is older than $f."; exit 1; }
done
say "binary: decode $(date -Is -r build/decode)   reference: $REF"
rm -f "$GO"                                  # DSV4_GENOUT APPENDS

env DSV4_PROMPTS_FILE="$PROMPTS" DSV4_BLKSWEEP="$SWEEP" DSV4_GENOUT="$GO" DSV4_MAINKV_CACHE=0 \
    bash scripts/run_model.sh "$LOG" ./build/decode "$CKPT" \
    "0,671,6102,294,8760,344" "$NDEC" "" "$NGEN" || { say "FATAL: run_model refused"; exit 1; }
for i in $(seq 1 60); do pgrep -x decode > /dev/null && break; sleep 2; done
pgrep -x decode > /dev/null || { say "FATAL: decode never started."; tail -30 "$LOG"; exit 1; }
for i in $(seq 1 900); do pgrep -x decode > /dev/null || break; sleep 10; done
n=$(wc -l < "$GO" 2>/dev/null || echo 0)
[ "$n" = "3" ] || { say "FATAL: got $n genout lines, expected 3"; tail -20 "$LOG"; exit 1; }

say "=== baseline vs baseline, same arm, two independent runs ==="
python3 tools/genout_compare.py "$REF" "$GO" | tee $EV/genout_1p0_determinism.txt
rc=${PIPESTATUS[0]}
say "determinism comparison rc=$rc"
if [ "$rc" = "0" ]; then
  say "VERDICT: build/decode REPRODUCES ITSELF exactly. Run-to-run nondeterminism is ruled out,"
  say "so the off-vs-on divergence is a real difference and 1.0 must be reverted or fixed."
else
  say "VERDICT: build/decode does NOT reproduce itself. The cached code path never ran in either"
  say "side of this comparison, so the off-vs-on divergence cannot be attributed to the cache."
fi
say "done."
