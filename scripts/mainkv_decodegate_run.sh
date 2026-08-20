#!/usr/bin/env bash
# mainkv_decodegate_run.sh — ladder 1.0, locating the divergence the token-id comparison found.
#
# WHAT WE KNOW. The incremental main-KV is byte-identical in `build/dsv4-server`: DSV4_MAINKV_GATE=1
# memcmp'd the whole [s, HEAD_DIM] buffer against the untouched from-scratch function on every call
# for 384 calls at contexts up to 12,281, keeping 2,023,320 rows, with zero failures
# (evidence/decode_loop/server_1p0_gate.log). The unit gate agrees across 22 split points on both
# GEMM paths.
#
# WHAT WE DO NOT KNOW. On `build/decode` the two arms emitted DIFFERENT token ids: point 0 (6-id
# prompt) identical over 266 ids, but point 1 (3,072-id prompt) diverged at generated token 20 and
# point 2 (1,024-id prompt) at generated token 43. The prompt region matched in both, so the two
# arms were given identical input. decode.cu's width controller reads the DRAFT'S OWN MARGINS
# (`while(VK < VKCAP && hmarg[VK-1] >= adaptK) ++VK`) and not the clock, so this binary is not
# obviously timing-dependent and the divergence cannot be waved away as jitter.
#
# THE QUESTION THIS ANSWERS, and why it is the right next measurement rather than another A/B: is
# the CACHED mkv BUFFER different from the from-scratch one in THIS binary, or is the buffer
# identical and the divergence downstream? The in-situ gate answers exactly that and, if it fails,
# prints the first differing row and column instead of a hash. A second token-id A/B would only
# re-report "they differ".
#
#   nohup setsid bash scripts/mainkv_decodegate_run.sh > evidence/decode_loop/mainkv_decodegate.log 2>&1 </dev/null &
set -u
cd "$(dirname "$0")/.."
CKPT="${CKPT:-/home/patrickd/models/DeepSeek-V4-Flash-0731-REAP}"
EV=evidence/decode_loop
PROMPTS=$EV/mainkv_prompts.txt
SWEEP="${SWEEP:-6:1:1.50:0,6:1:1.50:1,6:1:1.50:2}"
NDEC="${NDEC:-16}"
# Past BOTH divergence points (generated token 20 and 43) with margin, and short enough that the
# gate's three extra from-scratch recomputes per token do not make this an hour.
NGEN="${NGEN:-96}"
LOG=$EV/decode_1p0_gate.log
say(){ echo "[1.0g] $(date -Is) $*"; }

for f in src/decode.cu kernels/dspark_attn.cu include/dspark_attn.h; do
  [ "$f" -nt build/decode ] && { say "REFUSING: build/decode is older than $f."; exit 1; }
done
say "binary: decode $(date -Is -r build/decode)"

env DSV4_PROMPTS_FILE="$PROMPTS" DSV4_BLKSWEEP="$SWEEP" DSV4_MAINKV_GATE=1 \
    bash scripts/run_model.sh "$LOG" ./build/decode "$CKPT" \
    "0,671,6102,294,8760,344" "$NDEC" "" "$NGEN" || { say "FATAL: run_model refused"; exit 1; }
for i in $(seq 1 60); do pgrep -x decode > /dev/null && break; sleep 2; done
pgrep -x decode > /dev/null || { say "FATAL: decode never started."; tail -30 "$LOG"; exit 1; }
for i in $(seq 1 900); do pgrep -x decode > /dev/null || break; sleep 10; done

npass=$(grep -c 'mainkv-gate. PASS' "$LOG" 2>/dev/null || true)
nfail=$(grep -c 'mainkv-gate. FAIL' "$LOG" 2>/dev/null || true)
say "gate: ${npass:-0} PASS lines, ${nfail:-0} FAIL lines"
grep 'mainkv-gate' "$LOG" | tail -12
if [ "${nfail:-0}" != "0" ]; then
  say "VERDICT: the cached mkv buffer DIFFERS from from-scratch in build/decode. 1.0 is not"
  say "bit-exact on this path and must be reverted or fixed. First differing row is above."
else
  say "VERDICT: the cached mkv buffer is byte-identical to from-scratch on every call in this"
  say "binary too. The token divergence is therefore NOT the cached prefix; it is downstream."
fi
say "done."
