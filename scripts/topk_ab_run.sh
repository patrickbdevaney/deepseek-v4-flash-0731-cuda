#!/usr/bin/env bash
# topk_ab_run.sh — ladder item 1.2, end to end: the unit gate, the in-situ bit-exactness gate, both
# arms of the A/B, and the standing LOSSLESS gate, in one unattended run.
#
# WHY A SCRIPT AND NOT A SEQUENCE OF COMMANDS. This is ~2.5 hours of four model loads and two full
# context sweeps that need no steering once started, so CLAUDE.md says it must be detached before it
# starts rather than babysat by whatever session launched it. It also has to run the phases in a
# fixed order with a health gate and a full teardown between them -- two model loads on this box is
# not an OOM, it is a reboot -- and the correctness phases have to come FIRST so a bit-exactness
# failure costs one load instead of four.
#
#   nohup setsid bash scripts/topk_ab_run.sh > evidence/decode_loop/topk_ab.log 2>&1 </dev/null &
#
# THE ARM IS AN ENVIRONMENT VARIABLE ON ONE BINARY. DSV4_TOPK_RADIX=0 restores the warp selection
# sort (item 1.1); unset takes the single-CTA radix select. Both arms run the SAME build, the same
# corpus, the same targets, the same seeds -- so the only difference between the two record sets is
# the kernel path.
#
# PHASE ORDER IS THE DRIFT CONTROL. The baseline (warp) arm runs BEFORE the radix arm, so any
# thermal drift over the run makes the radix arm look SLOWER, not faster. The reported win is a
# lower bound with respect to drift.
set -u
cd "$(dirname "$0")/.."

CKPT="${CKPT:-/home/patrickd/models/DeepSeek-V4-Flash-0731-REAP}"
SEQMAX="${SEQMAX:-16384}"
EV=evidence/decode_loop
GATE_OUT=$EV/fit_1p2_gate
BASE_OUT=$EV/fit_1p2_base
RADIX_OUT=$EV/fit_1p2_radix
TARGETS="${TARGETS:-12288,9216,6144,3072,1536,768,384,128}"
REPS="${REPS:-6}"
GATE_TARGETS="${GATE_TARGETS:-12288,6144,1536}"
PHASE_FROM="${PHASE_FROM:-0}"

say(){ echo "[1.2] $(date -Is) $*"; }

# ---- the binary must be the one we are about to reason about ---------------------------------
for f in kernels/compressed_decode.cu kernels/indexer.cu include/topk_radix.h; do
  if [ "$f" -nt build/dsv4-server ]; then
    say "REFUSING: build/dsv4-server is older than $f. Run scripts/build_server.sh."; exit 1; fi
  if [ "$f" -nt build/decode ]; then
    say "REFUSING: build/decode is older than $f. Run scripts/build_decode.sh."; exit 1; fi
done
say "binaries: server $(date -Is -r build/dsv4-server), decode $(date -Is -r build/decode)"

server_down(){
  pkill -f 'build/dsv4-server --ckpt' 2>/dev/null || true
  for _ in $(seq 1 60); do
    pgrep -f 'build/dsv4-server --ckpt' > /dev/null || { say "server down"; sleep 5; return 0; }
    sleep 5
  done
  say "FATAL: server would not die"; return 1
}

start_server(){
  local log="$1" label="$2"
  if curl -s -m 10 -o /dev/null http://localhost:8080/health 2>/dev/null; then
    say "FATAL: a server is already healthy on :8080 and it is not ours (wrong arm env)"; return 1
  fi
  say "starting server [$label] seqmax=$SEQMAX -> $log"
  SEQMAX="$SEQMAX" LOG="$log" bash scripts/run_server.sh || { say "run_server refused"; return 1; }
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

mkdir -p "$GATE_OUT" "$BASE_OUT" "$RADIX_OUT"

# ================= PHASE 0 — the unit gate, before any weights are loaded =====================
# Seconds, no checkpoint. Everything that can be wrong about a selection is order and tie-breaking,
# and none of it needs real weights -- so finding it here costs 2 seconds instead of a 10-minute
# 100 GiB load. Six distributions including exact ties, signed zeros and floor-straddling rows.
if [ "$PHASE_FROM" -le 0 ]; then
say "=== PHASE 0: gate_topk_radix (no checkpoint) ==="
if [ ! -x build/gate_topk_radix ] || [ include/topk_radix.h -nt build/gate_topk_radix ]; then
  say "REFUSING: build/gate_topk_radix missing or stale. Run scripts/build_gate.sh."; exit 1; fi
./build/gate_topk_radix || { say "FATAL: radix select is not bit-identical. Nothing else runs."; exit 1; }
say "=== PHASE 0b: gate_topk_warp (the arm this A/B calls 'before' must still be correct) ==="
[ -x build/gate_topk_warp ] && { ./build/gate_topk_warp > /dev/null || { say "FATAL: gate_topk_warp"; exit 1; }; }
fi

# ================= PHASE 1 — bit-exactness, in situ, at real context ==========================
# DSV4_TOPK_GATE=1 makes every radix call also run the UNTOUCHED warp selection sort into a private
# buffer and memcmp the whole index array, aborting on the first differing slot. This is the
# substitute hard invariant 1 names after 1.0's amendment: token ids are not a valid test at context
# because the engine stops reproducing itself part-way through a long run, and a whole-buffer memcmp
# of the changed kernel's output is strictly stronger anyway.
if [ "$PHASE_FROM" -le 1 ]; then
say "=== PHASE 1: bit-exactness gate (DSV4_TOPK_GATE=1, radix ON) ==="
export DSV4_TOPK_GATE=1
unset DSV4_TOPK_RADIX
GATE_LOG=$EV/server_1p2_gate.log
: > "$GATE_LOG"
start_server "$GATE_LOG" gate || exit 1
# 64 tokens, not fewer: decode_fit_probe.py refuses to bank a record below 64 completion tokens.
python3 tools/decode_fit_probe.py --outdir "$GATE_OUT" --targets "$GATE_TARGETS" --reps 1 \
        --max-tokens 64 --ckpt "$CKPT"
rc=$?
say "gate probe rc=$rc"
server_down
nfail=$(grep -c 'topk-gate. FAIL' "$GATE_LOG" 2>/dev/null || true)
npass=$(grep -c 'topk-gate. PASS' "$GATE_LOG" 2>/dev/null || true)
say "gate: $npass PASS lines, $nfail FAIL lines"
grep 'topk-gate' "$GATE_LOG" | tail -8
if [ "$rc" != "0" ] || [ "${nfail:-1}" != "0" ] || [ "${npass:-0}" = "0" ]; then
  say "FATAL: bit-exactness gate did not pass cleanly. STOPPING before spending three more loads."
  exit 1
fi
unset DSV4_TOPK_GATE
fi

# ================= PHASE 2 — the BEFORE arm (warp selection sort) =============================
if [ "$PHASE_FROM" -le 2 ]; then
say "=== PHASE 2: baseline sweep (DSV4_TOPK_RADIX=0, warp selection sort) ==="
export DSV4_TOPK_RADIX=0
BASE_LOG=$EV/server_1p2_base.log
: > "$BASE_LOG"
start_server "$BASE_LOG" base || exit 1
python3 tools/decode_fit_probe.py --outdir "$BASE_OUT" --targets "$TARGETS" --reps "$REPS" --ckpt "$CKPT"
rc=$?; say "baseline probe rc=$rc"
server_down
[ "$rc" = "0" ] || { say "FATAL: baseline sweep failed"; exit 1; }
fi

# ================= PHASE 3 — the AFTER arm (radix select) ====================================
if [ "$PHASE_FROM" -le 3 ]; then
say "=== PHASE 3: radix sweep ==="
unset DSV4_TOPK_RADIX
RADIX_LOG=$EV/server_1p2_radix.log
: > "$RADIX_LOG"
start_server "$RADIX_LOG" radix || exit 1
python3 tools/decode_fit_probe.py --outdir "$RADIX_OUT" --targets "$TARGETS" --reps "$REPS" --ckpt "$CKPT"
rc=$?; say "radix probe rc=$rc"
server_down
[ "$rc" = "0" ] || { say "FATAL: radix sweep failed"; exit 1; }
fi

# ================= PHASE 4 — the standing LOSSLESS gate ======================================
if [ "$PHASE_FROM" -le 4 ]; then
say "=== PHASE 4: standing GATE + LOSSLESS GATE (build/decode, radix ON) ==="
# WAIT FOR THE PAGE CACHE, DO NOT ASSUME IT. run_model.sh refuses below ~105 GiB available, and a
# server that has just exited leaves ~100 GiB of checkpoint in the page cache that the kernel
# reclaims lazily. On the first run of this script that refusal fired 0 seconds after `server_down`
# returned and PHASE 4 was skipped silently -- the phase "completed" having run nothing, which is
# exactly the failure CLAUDE.md's second rule is about.
for _ in $(seq 1 120); do
  avail=$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo)
  [ "$avail" -ge 105 ] && break
  say "waiting for the page cache: ${avail} GiB available, need 105"
  sleep 15
done
DEC_LOG=$EV/decode_1p2_lossless.log
bash scripts/run_model.sh "$DEC_LOG" ./build/decode "$CKPT" "0,671,6102,294,8760,344" 8 "" 64
for _ in $(seq 1 240); do pgrep -f 'build/decode' > /dev/null || break; sleep 10; done
say "decode finished; gates:"
if [ ! -s "$DEC_LOG" ]; then say "FATAL: PHASE 4 produced no log -- the decode run never started."; exit 1; fi
grep -E 'GATE|tok/verify|tok/s|LOSSLESS' "$DEC_LOG" | tail -20
grep -qE 'LOSSLESS GATE: .* -> PASS' "$DEC_LOG" || { say "FATAL: LOSSLESS gate did not pass"; exit 1; }
fi

# ================= PHASE 5 — the ratchet =====================================================
say "=== PHASE 5: reports ==="
say "--- BEFORE (warp selection sort) ---"
python3 tools/decode_fit_probe.py --outdir "$BASE_OUT" --report  | tee $EV/fit_1p2_base.txt
say "--- AFTER (radix select) ---"
python3 tools/decode_fit_probe.py --outdir "$RADIX_OUT" --report | tee $EV/fit_1p2_radix.txt
say "--- paired, per (target, rep), tau and width reported per leg ---"
python3 tools/mainkv_ab_compare.py "$BASE_OUT" "$RADIX_OUT" | tee $EV/fit_1p2_identity.txt
say "done."
