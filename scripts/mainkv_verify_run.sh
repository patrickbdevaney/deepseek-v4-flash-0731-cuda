#!/usr/bin/env bash
# mainkv_verify_run.sh — ladder 1.0, the BIT-EXACTNESS half, after the A/B raised a question.
#
# WHAT WENT WRONG WITH THE FIRST ATTEMPT. `mainkv_ab_run.sh` proved the speedup (b: 7.220 ->
# 4.006 ms per 1000 context) and then reported that 21 of 52 completion hashes differed between the
# arms. The differing legs are not scattered: in RUN ORDER they are a clean suffix.
#
#     t12288 t9216 t6144 t3072 t1536 (30 legs) SAME, t768-r0 SAME,
#     t768-r1 ... t128-r5 and both controls (21 legs) DIFFER.
#
# One divergence event at leg 32 of 52 and never identical again. A row-split bug does not switch on
# at leg 32 of a descending sweep -- it would fire at the deepest context, not the shallowest -- so
# the shape says "the two server runs came apart once" rather than "the cached prefix is wrong".
# But shape is an argument, and DECODE_LADDER.md's invariant is not satisfied by arguments (item 1.6
# was resolved with a control BUILD for exactly this reason). So this script measures it.
#
# THE INSTRUMENT THE FIRST ATTEMPT SHOULD HAVE USED. `decode_fit_probe.py` samples at temperature
# 1.0 / top_p 0.95 and leans on `seed=1000+rep` for reproducibility, so its completion hash is only
# a bit-exactness instrument if the whole server is reproducible run-to-run -- which is the very
# thing in question. `build/decode` has no such dependency: it is ARGMAX end to end, no seed, no
# HTTP, no prefix cache, and `DSV4_GENOUT` writes the raw prompt+generated token IDS. That is
# DECODE_LADDER.md's invariant *literally* ("byte-identical generated token ids"), measured on the
# binary that also carries the standing LOSSLESS gate.
#
# PHASES 1-3 are that comparison, at REAL context: prompt 0 is the canonical 6-id gate prompt, and
# prompts 1-2 are 3,072- and 1,024-token prefixes of LOOP_LOG.md -- the SAME document the sweep
# reads, so the cached prefix is exercised over thousands of rows rather than a handful.
#
# WHY 3,072 AND NOT 6,132, WHICH IS WHAT THE FIRST ATTEMPT ASKED FOR. `build/decode` is not the
# server and does not have the server's memory profile: it allocates a per-layer `xin` of
# seqmax*DIM*4 across all 43 layers plus h0/h/h2, which measured 18.7 GiB of buffers at seqmax=6403
# -- 2.99 MB per seqmax token -- on top of 100.4 GiB of weights in a 122.8 GiB pool. The first
# attempt reached "structs built. mem 119.1/122.8 GiB" and the memguard killed it 105 s later at
# MemAvailable 219 MB, during prefill. That is the memguard doing its job, not a fault. At
# seqmax 3343 the same arithmetic gives ~110 GiB resident and ~10 GiB of headroom.
#
# DEPTH IS NOT WHAT THIS PHASE IS FOR. The cached prefix has already been proven byte-identical at
# 6,131 and 12,281 retained rows, in situ, by the DSV4_MAINKV_GATE=1 memcmp over the whole
# [s, HEAD_DIM] buffer (evidence/decode_loop/server_1p0_gate.log: 384 checks, 0 FAIL, 2,023,320
# rows kept). What that gate cannot produce is a TOKEN ID, because the server samples. This phase
# supplies the ids; 3,072 retained rows is far past the point where a row-split or rope-offset
# error would still be hiding.
#
# PHASE 4-5 are the determinism control, and they are the half that explains the 21 legs: the
# BASELINE arm sweep, re-run from a second server start with the SAME binary and the SAME env, then
# compared against the baseline records already on disk. Same arm on both sides, so any leg that
# differs is the server disagreeing with ITSELF and cannot be attributed to the cache.
#
#   nohup setsid bash scripts/mainkv_verify_run.sh > evidence/decode_loop/mainkv_verify.log 2>&1 </dev/null &
set -u
cd "$(dirname "$0")/.."

CKPT="${CKPT:-/home/patrickd/models/DeepSeek-V4-Flash-0731-REAP}"
SEQMAX="${SEQMAX:-16384}"
EV=evidence/decode_loop
PROMPTS=$EV/mainkv_prompts.txt
BASE_OUT=$EV/fit_1p0_base
BASE2_OUT=$EV/fit_1p0_base2
TARGETS="${TARGETS:-12288,9216,6144,3072,1536,768,384,128}"
REPS="${REPS:-6}"
# BLK=6 and adaptK=1.50 are the shipped serving settings (see decode.cu's F94 note and the server's
# own banner), so the three points differ ONLY in which prompt they run.
SWEEP="${SWEEP:-6:1:1.50:0,6:1:1.50:1,6:1:1.50:2}"
NDEC="${NDEC:-16}"      # base-AR tokens; feeds the SPEC-vs-BASE LOSSLESS gate on prompt 0
NGEN="${NGEN:-256}"     # spec-decode tokens per point
PHASE_FROM="${PHASE_FROM:-1}"

say(){ echo "[1.0v] $(date -Is) $*"; }

# ---- the binary must be the one we are about to reason about --------------------------------
for f in src/engine.cu src/decode.cu kernels/dspark_attn.cu include/dspark_attn.h; do
  for b in build/dsv4-server build/decode; do
    if [ "$f" -nt "$b" ]; then say "REFUSING: $b is older than $f. Rebuild first."; exit 1; fi
  done
done
[ -f "$PROMPTS" ] || { say "REFUSING: $PROMPTS missing (the long-context prompt ids)"; exit 1; }
say "binaries: server $(date -Is -r build/dsv4-server), decode $(date -Is -r build/decode)"

# ---- THE SEQMAX GATE, in milliseconds, before a 100 GiB load ---------------------------------
# The first attempt discovered its prompts were too big for `build/decode` by loading the
# checkpoint, prefilling, and getting shot by the memguard -- ~5 minutes to learn a number that
# decode.cu computes before it opens the checkpoint. DSV4_PARSE_ONLY=1 prints exactly that number
# and exits, so the run refuses here instead of at 119.1/122.8 GiB.
#   2.99 MB/seqmax-token measured at seqmax=6403 (18.7 GiB of buffers); the cap below leaves
#   ~10 GiB of headroom over the 100.4 GiB of weights and the prefill arena.
SEQMAX_CAP="${SEQMAX_CAP:-3600}"
parse=$(DSV4_PROMPTS_FILE="$PROMPTS" DSV4_BLKSWEEP="$SWEEP" DSV4_PARSE_ONLY=1 \
        ./build/decode "$CKPT" "0,671,6102,294,8760,344" "$NDEC" "" "$NGEN" 2>&1) || {
  say "REFUSING: DSV4_PARSE_ONLY failed:"; echo "$parse"; exit 1; }
echo "$parse" | grep -E '^\[parse\] [0-9]+ prompt|^\[parse\] point' | sed 's/^/[1.0v]   /'
dsq=$(echo "$parse" | sed -n 's/.*seqmax=\([0-9]*\).*/\1/p' | head -1)
[ -n "$dsq" ] || { say "REFUSING: could not read seqmax out of DSV4_PARSE_ONLY"; exit 1; }
say "decode seqmax=$dsq (cap $SEQMAX_CAP, ~$((dsq*299/100000+100)) GiB resident incl. weights)"
[ "$dsq" -le "$SEQMAX_CAP" ] || {
  say "REFUSING: seqmax $dsq > cap $SEQMAX_CAP. build/decode would be memguard-killed during"
  say "prefill, exactly as it was at 02:28:55. Shorten $PROMPTS or raise SEQMAX_CAP knowingly."
  exit 1; }

# run_decode <armlabel> <cache_env_value|unset> <genout> <log>
run_decode(){
  local label="$1" cache="$2" go="$3" log="$4"
  rm -f "$go"                      # DSV4_GENOUT APPENDS; a stale file would be compared as if fresh
  say "=== decode arm [$label] -> $log (genout $go) ==="
  local envs=(DSV4_PROMPTS_FILE="$PROMPTS" DSV4_BLKSWEEP="$SWEEP" DSV4_GENOUT="$go")
  [ "$cache" != "unset" ] && envs+=(DSV4_MAINKV_CACHE="$cache")
  env "${envs[@]}" bash scripts/run_model.sh "$log" ./build/decode "$CKPT" \
      "0,671,6102,294,8760,344" "$NDEC" "" "$NGEN" || { say "FATAL: run_model refused"; return 1; }
  # A DEAD LOADER MUST NOT LOOK LIKE A SLOW ONE (CLAUDE.md). Wait for the process to APPEAR, then
  # for it to leave; a fixed sleep would race the 100 GiB load at one end and truncate at the other.
  local i
  for i in $(seq 1 60); do pgrep -x decode > /dev/null && break; sleep 2; done
  pgrep -x decode > /dev/null || { say "FATAL: decode never started. Tail of $log:"; tail -30 "$log"; return 1; }
  for i in $(seq 1 900); do pgrep -x decode > /dev/null || break; sleep 10; done
  if pgrep -x decode > /dev/null; then say "FATAL: decode still running after 2.5 h"; return 1; fi
  say "decode [$label] finished; gates and headline:"
  grep -E 'GATE|LOSSLESS|tokens/verify|SPEC-DECODE|genout' "$log" | tail -12
  local n; n=$(wc -l < "$go" 2>/dev/null || echo 0)
  say "genout lines: $n (expected 3)"
  [ "$n" = "3" ] || { say "FATAL: arm [$label] did not emit one id line per sweep point"; return 1; }
}

# ================= PHASES 1-2 — the two arms of the token-id comparison =======================
# BASELINE FIRST, so any thermal drift over the run makes the cached arm look slower, not faster --
# the same drift control mainkv_ab_run.sh uses.
if [ "$PHASE_FROM" -le 1 ]; then
  run_decode base-off 0 "$EV/genout_1p0_off.txt" "$EV/decode_1p0_off.log" || exit 1
fi
if [ "$PHASE_FROM" -le 2 ]; then
  run_decode cache-on unset "$EV/genout_1p0_on.txt" "$EV/decode_1p0_on.log" || exit 1
fi

# ================= PHASE 3 — the invariant, stated as ids and not as a hash ===================
if [ "$PHASE_FROM" -le 3 ]; then
say "=== PHASE 3: token-id identity, from-scratch vs cached ==="
python3 tools/genout_compare.py "$EV/genout_1p0_off.txt" "$EV/genout_1p0_on.txt" \
        | tee $EV/genout_1p0_identity.txt
idrc=${PIPESTATUS[0]}
say "token-id comparison rc=$idrc"
# The timing side of the same two runs, for a decode-binary cross-check on the server A/B.
say "--- ms/tok and tokens/verify, both arms ---"
for a in off on; do
  echo "[$a]"; grep -E '^\[spec\] (generated|SPEC-DECODE)' "$EV/decode_1p0_$a.log" || true
done
if [ "$idrc" != "0" ]; then
  say "TOKEN IDS DIFFER between the arms on the deterministic binary. That is the invariant"
  say "failing on its own terms -- the cached prefix is NOT bit-exact and 1.0 must be reverted."
  say "STOPPING: the determinism control below cannot exonerate this."
  exit 1
fi
say "token ids are byte-identical across the arms on all 3 points."
fi

# ================= PHASE 4 — the determinism control =========================================
# Same arm on both sides. If this diverges, the server is not reproducible run-to-run and the 21
# differing completion hashes in the A/B are that, not the cache.
if [ "$PHASE_FROM" -le 4 ]; then
say "=== PHASE 4: BASELINE sweep, second independent server start (determinism control) ==="
if curl -s -m 10 -o /dev/null http://localhost:8080/health 2>/dev/null; then
  say "FATAL: a server is already healthy on :8080"; exit 1; fi
rm -rf "$BASE2_OUT"; mkdir -p "$BASE2_OUT"
export DSV4_MAINKV_CACHE=0
BASE2_LOG=$EV/server_1p0_base2.log
: > "$BASE2_LOG"
SEQMAX="$SEQMAX" LOG="$BASE2_LOG" bash scripts/run_server.sh || { say "run_server refused"; exit 1; }
setsid nohup env PAT=dsv4-server bash scripts/memguard.sh \
    > "${BASE2_LOG%.log}.memguard.log" 2>&1 < /dev/null &
say "memguard armed -> ${BASE2_LOG%.log}.memguard.log"
ok=0
for _ in $(seq 1 360); do
  curl -s -m 10 -o /dev/null http://localhost:8080/health 2>/dev/null && { ok=1; break; }
  pgrep -f 'build/dsv4-server --ckpt' > /dev/null || {
    say "FATAL: server process is gone while we waited. Tail of $BASE2_LOG:"; tail -30 "$BASE2_LOG"; exit 1; }
  sleep 5
done
[ "$ok" = "1" ] || { say "FATAL: server never became healthy"; tail -30 "$BASE2_LOG"; exit 1; }
say "healthy [base2]: $(curl -s -m 10 http://localhost:8080/health)"
python3 tools/decode_fit_probe.py --outdir "$BASE2_OUT" --targets "$TARGETS" --reps "$REPS" --ckpt "$CKPT"
rc=$?; say "base2 probe rc=$rc"
pkill -f 'build/dsv4-server --ckpt' 2>/dev/null || true
for _ in $(seq 1 60); do pgrep -f 'build/dsv4-server --ckpt' > /dev/null || break; sleep 5; done
say "server down"
[ "$rc" = "0" ] || { say "FATAL: base2 sweep failed"; exit 1; }
fi

# ================= PHASE 5 — what the 21 legs actually were ==================================
say "=== PHASE 5: baseline vs baseline (same arm, two server starts) ==="
python3 tools/mainkv_ab_compare.py "$BASE_OUT" "$BASE2_OUT" | tee $EV/fit_1p0_determinism.txt
say "determinism comparison rc=${PIPESTATUS[0]}"
say "done."
