#!/usr/bin/env bash
# mainkv_ab_run.sh — ladder item 1.0, end to end: the bit-exactness gate, both arms of the A/B, and
# the standing LOSSLESS gate, in one unattended run.
#
# WHY A SCRIPT AND NOT A SEQUENCE OF COMMANDS. This is ~2.5 hours of four model loads and two full
# context sweeps that need no steering once started, so CLAUDE.md says it must be detached before it
# starts rather than babysat by whatever session launched it. It also has to run the phases in a
# fixed order with a health gate and a full teardown between them -- two model loads on this box is
# not an OOM, it is a reboot (with_model_lock.sh) -- and the correctness phase has to come FIRST so
# a bit-exactness failure costs one load instead of four.
#
#   nohup setsid bash scripts/mainkv_ab_run.sh > evidence/decode_loop/mainkv_ab.log 2>&1 </dev/null &
#
# THE ARM IS AN ENVIRONMENT VARIABLE ON ONE BINARY. DSV4_MAINKV_CACHE=0 restores the from-scratch
# `dspark_main_kv` per token; unset takes the incremental path. Both arms therefore run the SAME
# build, the same corpus (LOOP_LOG.md, sha recorded in every record), the same targets, the same
# seeds -- so the only difference between the two record sets is the kernel path.
#
# PHASE ORDER IS THE DRIFT CONTROL. Baseline runs BEFORE cached, so any thermal drift over the
# evening makes the cached arm look SLOWER, not faster. The reported win is therefore a lower bound
# with respect to drift.
set -u
cd "$(dirname "$0")/.."

CKPT="${CKPT:-/home/patrickd/models/DeepSeek-V4-Flash-0731-REAP}"
SEQMAX="${SEQMAX:-16384}"
EV=evidence/decode_loop
GATE_OUT=$EV/fit_1p0_gate
BASE_OUT=$EV/fit_1p0_base
CACHE_OUT=$EV/fit_1p0_cache
TARGETS="${TARGETS:-12288,9216,6144,3072,1536,768,384,128}"
REPS="${REPS:-6}"
GATE_TARGETS="${GATE_TARGETS:-12288,6144,1536}"
# PHASE_FROM lets a run resume without re-proving what is already on disk. Each phase costs a
# ~10-minute 100 GiB load, so re-running a passed correctness gate to get to the sweeps is a real
# cost, and the alternative -- hand-running the remaining phases -- is exactly the session-bound
# launch CLAUDE.md exists to prevent.
PHASE_FROM="${PHASE_FROM:-0}"

say(){ echo "[1.0] $(date -Is) $*"; }

# ---- the binary must be the one we are about to reason about ---------------------------------
for f in src/engine.cu kernels/dspark_attn.cu include/dspark_attn.h; do
  if [ "$f" -nt build/dsv4-server ]; then
    say "REFUSING: build/dsv4-server is older than $f. Run scripts/build_server.sh."; exit 1; fi
done
if [ src/decode.cu -nt build/decode ] || [ kernels/dspark_attn.cu -nt build/decode ]; then
  say "REFUSING: build/decode is older than its sources. Run scripts/build_decode.sh."; exit 1; fi
say "binaries: server $(date -Is -r build/dsv4-server), decode $(date -Is -r build/decode)"

server_down(){
  pkill -f 'build/dsv4-server --ckpt' 2>/dev/null || true
  for _ in $(seq 1 60); do
    pgrep -f 'build/dsv4-server --ckpt' > /dev/null || { say "server down"; sleep 5; return 0; }
    sleep 5
  done
  say "FATAL: server would not die"; return 1
}

# start_server <logfile> <label>   -- env for the arm is exported by the caller
start_server(){
  local log="$1" label="$2"
  if curl -s -m 10 -o /dev/null http://localhost:8080/health 2>/dev/null; then
    say "FATAL: a server is already healthy on :8080 and it is not ours (wrong arm env)"; return 1
  fi
  say "starting server [$label] seqmax=$SEQMAX -> $log"
  SEQMAX="$SEQMAX" LOG="$log" bash scripts/run_server.sh || { say "run_server refused"; return 1; }
  # run_server.sh detaches but arms no memory guard; seqmax 16384 sits at 120.0/122.8 GiB.
  setsid nohup env PAT=dsv4-server bash scripts/memguard.sh \
      > "${log%.log}.memguard.log" 2>&1 < /dev/null &
  say "memguard armed -> ${log%.log}.memguard.log"
  for _ in $(seq 1 360); do
    curl -s -m 10 -o /dev/null http://localhost:8080/health 2>/dev/null && {
      say "healthy [$label]: $(curl -s -m 10 http://localhost:8080/health)"; return 0; }
    # A DEAD LOADER MUST NOT LOOK LIKE A SLOW ONE (CLAUDE.md).
    pgrep -f 'build/dsv4-server --ckpt' > /dev/null || {
      say "FATAL: server process is gone while we waited. Tail of $log:"; tail -30 "$log"; return 1; }
    sleep 5
  done
  say "FATAL: server never became healthy"; tail -30 "$log"; return 1
}

mkdir -p "$GATE_OUT" "$BASE_OUT" "$CACHE_OUT"

# ================= PHASE 0 — the unit gate, before any weights are loaded =====================
# Seconds, no checkpoint. Everything that can be wrong about a row split is shape logic (the rope
# offset, the M-dependent GEMM dispatch, the rewind clamp) and none of it needs real weights, so
# finding it here costs 2 seconds instead of a 10-minute 100 GiB load.
if [ "$PHASE_FROM" -le 0 ]; then
say "=== PHASE 0: gate_mainkv_incr (no checkpoint) ==="
if [ ! -x build/gate_mainkv_incr ] || [ kernels/dspark_attn.cu -nt build/gate_mainkv_incr ]; then
  say "REFUSING: build/gate_mainkv_incr missing or stale. Run scripts/build_gate.sh."; exit 1; fi
./build/gate_mainkv_incr || { say "FATAL: incremental main-KV is not bit-identical. Nothing else runs."; exit 1; }

fi

# ================= PHASE 1 — bit-exactness, in situ, at real context ==========================
# DSV4_MAINKV_GATE=1 makes every incremental call also rebuild all `s` rows from scratch with the
# UNTOUCHED dspark_main_kv and memcmp the whole [s, HEAD_DIM] buffer, aborting on the first
# differing float. The point list is descending from 12,288 and the CONTROL legs are kept (they use
# a different document), so this exercises prefill_full, rewind_to AND extend -- the three paths
# that can invalidate a cached prefix -- rather than just the easy one.
if [ "$PHASE_FROM" -le 1 ]; then
say "=== PHASE 1: bit-exactness gate (DSV4_MAINKV_GATE=1, cache ON) ==="
export DSV4_MAINKV_GATE=1
unset DSV4_MAINKV_CACHE
GATE_LOG=$EV/server_1p0_gate.log
: > "$GATE_LOG"
start_server "$GATE_LOG" gate || exit 1
# 64 tokens, not fewer: `decode_fit_probe.py` refuses to bank a record below 64 completion tokens
# (CLAUDE.md -- nothing is written for an item that generated nothing usable), so a shorter leg
# makes the probe return 1 with the gate itself perfectly healthy. Measured the hard way.
python3 tools/decode_fit_probe.py --outdir "$GATE_OUT" --targets "$GATE_TARGETS" --reps 1 \
        --max-tokens 64 --ckpt "$CKPT"
rc=$?
say "gate probe rc=$rc"
server_down
nfail=$(grep -c 'mainkv-gate. FAIL' "$GATE_LOG" 2>/dev/null || true)
npass=$(grep -c 'mainkv-gate. PASS' "$GATE_LOG" 2>/dev/null || true)
say "gate: $npass PASS lines, $nfail FAIL lines"
grep 'mainkv-gate' "$GATE_LOG" | tail -8
if [ "$rc" != "0" ] || [ "${nfail:-1}" != "0" ] || [ "${npass:-0}" = "0" ]; then
  say "FATAL: bit-exactness gate did not pass cleanly. STOPPING before spending three more loads."
  exit 1
fi
unset DSV4_MAINKV_GATE

fi

# ================= PHASE 2 — the BEFORE arm ==================================================
if [ "$PHASE_FROM" -le 2 ]; then
say "=== PHASE 2: baseline sweep (DSV4_MAINKV_CACHE=0, from-scratch every token) ==="
export DSV4_MAINKV_CACHE=0
BASE_LOG=$EV/server_1p0_base.log
: > "$BASE_LOG"
start_server "$BASE_LOG" base || exit 1
python3 tools/decode_fit_probe.py --outdir "$BASE_OUT" --targets "$TARGETS" --reps "$REPS" --ckpt "$CKPT"
rc=$?; say "baseline probe rc=$rc"
server_down
[ "$rc" = "0" ] || { say "FATAL: baseline sweep failed"; exit 1; }

fi

# ================= PHASE 3 — the AFTER arm ===================================================
if [ "$PHASE_FROM" -le 3 ]; then
say "=== PHASE 3: cached sweep (incremental main-KV) ==="
unset DSV4_MAINKV_CACHE
CACHE_LOG=$EV/server_1p0_cache.log
: > "$CACHE_LOG"
start_server "$CACHE_LOG" cache || exit 1
python3 tools/decode_fit_probe.py --outdir "$CACHE_OUT" --targets "$TARGETS" --reps "$REPS" --ckpt "$CKPT"
rc=$?; say "cached probe rc=$rc"
server_down
[ "$rc" = "0" ] || { say "FATAL: cached sweep failed"; exit 1; }

fi

# ================= PHASE 4 — the standing LOSSLESS gate ======================================
# The memcmp gate and the completion hashes below are the real bit-exactness evidence; this is the
# frozen protocol DECODE_LADDER.md names, on the binary that carries it.
if [ "$PHASE_FROM" -le 4 ]; then
say "=== PHASE 4: standing GATE + LOSSLESS GATE (build/decode, cache ON) ==="
DEC_LOG=$EV/decode_1p0_lossless.log
bash scripts/run_model.sh "$DEC_LOG" ./build/decode "$CKPT" "0,671,6102,294,8760,344" 8 "" 64
for _ in $(seq 1 240); do pgrep -f 'build/decode' > /dev/null || break; sleep 10; done
say "decode finished; gates:"
grep -E 'GATE|tok/verify|tok/s|LOSSLESS' "$DEC_LOG" | tail -20

fi

# ================= PHASE 5 — the ratchet =====================================================
say "=== PHASE 5: reports ==="
say "--- BEFORE (from-scratch) ---"
python3 tools/decode_fit_probe.py --outdir "$BASE_OUT" --report  | tee $EV/fit_1p0_base.txt
say "--- AFTER (cached) ---"
python3 tools/decode_fit_probe.py --outdir "$CACHE_OUT" --report | tee $EV/fit_1p0_cache.txt
say "--- token identity: completion sha256 per (target, rep), both arms ---"
python3 tools/mainkv_ab_compare.py "$BASE_OUT" "$CACHE_OUT" | tee $EV/fit_1p0_identity.txt
say "done."
