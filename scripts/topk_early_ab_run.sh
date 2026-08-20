#!/usr/bin/env bash
# topk_early_ab_run.sh — ladder item 1.3, end to end: the unit gate, the in-situ bit-exactness gate,
# both arms of the tok/s A/B, both arms of a dprof attribution, and the standing LOSSLESS gate.
#
#   nohup setsid bash scripts/topk_early_ab_run.sh > evidence/decode_loop/topk_early_ab.log 2>&1 </dev/null &
#
# WHY A SCRIPT AND NOT A SEQUENCE OF COMMANDS. Six checkpoint loads that need no steering once
# started, so CLAUDE.md says detach it before it starts. Named in detach_audit.sh's PATTERNS in the
# same commit, because a stage the audit cannot see reports as "all detached" by never being looked
# at.
#
# THE ARM IS AN ENVIRONMENT VARIABLE ON ONE BINARY. DSV4_TOPK_EARLY=0 restores the full MSB-first
# threshold search; unset takes the `lim <= topk` early-out. Same build, same corpus, same targets.
#
# WHY THERE IS A DPROF PAIR AND NOT JUST A tok/s PAIR, stated BEFORE the run so it cannot be a
# post-hoc excuse. The kernel band (tests/gate_topk_radix) puts the saving at ~4.05 us per
# k_topk_decode call at T=384-512 with a min/max spread under 0.1 us. There are 21 ratio-4 layers,
# so one verify forward saves ~0.085 ms of a ~130 ms step -- 0.07 %, against a measured run-to-run
# spread of 3.5 %. THE tok/s SWEEP CANNOT RESOLVE THIS AND IS NOT EXPECTED TO. It is run anyway
# because the ladder requires tau and a band in every A/B and because a *regression* at that scale
# would show; the `i:topk` dprof mark is the instrument that can actually see the change, since it
# brackets the top-k launch itself rather than the whole step.
#
# TARGETS ARE CHOSEN SO THE CONTROL IS INSIDE THE RUN. `lim <= topk` is T <= 512, i.e. ctx <= 2048 at
# ratio 4 with INDEX_TOPK 512. 6144 is above it and MUST be unchanged; 1536/768/384/128 are below it
# and are where any change has to appear. An A/B whose every leg is in the affected regime cannot
# tell a real saving from a drift.
#
# PHASE ORDER IS THE DRIFT CONTROL. The OFF (full-search) arm runs BEFORE the ON arm in both pairs,
# so thermal drift over the run makes the early-out look SLOWER, not faster.
set -u
cd "$(dirname "$0")/.."
export PATH=/usr/local/cuda/bin:$PATH

CKPT="${CKPT:-/home/patrickd/models/DeepSeek-V4-Flash-0731-REAP}"
SEQMAX="${SEQMAX:-16384}"
EV=evidence/decode_loop
OFF_OUT=$EV/fit_1p3_off
ON_OUT=$EV/fit_1p3_on
GATE_OUT=$EV/fit_1p3_gate
DPOFF_OUT=$EV/dprof_1p3_off
DPON_OUT=$EV/dprof_1p3_on
TARGETS="${TARGETS:-6144,1536,768,384,128}"
REPS="${REPS:-6}"
GATE_TARGETS="${GATE_TARGETS:-1536,768,384}"
DPROF_TARGETS="${DPROF_TARGETS:-6144,1536,768,384}"
DPROF_REPS="${DPROF_REPS:-3}"
PHASE_FROM="${PHASE_FROM:-0}"

say(){ echo "[1.3] $(date -Is) $*"; }

# ---- the binaries must be the ones we are about to reason about -------------------------------
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

# Wait for the page cache before any load. run_server/run_model refuse below ~105 GiB available and
# a server that has just exited leaves ~100 GiB of checkpoint that the kernel reclaims lazily; a
# refusal here is a phase that "completes" having run nothing.
wait_mem(){
  for _ in $(seq 1 120); do
    avail=$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo)
    [ "$avail" -ge 105 ] && return 0
    say "waiting for the page cache: ${avail} GiB available, need 105"
    sleep 15
  done
  say "FATAL: never got 105 GiB back"; return 1
}

start_server(){
  local log="$1" label="$2"
  if curl -s -m 10 -o /dev/null http://localhost:8080/health 2>/dev/null; then
    say "FATAL: a server is already healthy on :8080 and it is not ours (wrong arm env)"; return 1
  fi
  wait_mem || return 1
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

mkdir -p "$OFF_OUT" "$ON_OUT" "$GATE_OUT" "$DPOFF_OUT" "$DPON_OUT"

# ================= PHASE 0 — the unit gate, before any weights are loaded =====================
# Seconds, no checkpoint. Since 1.3 it compares BOTH arms of the early-out against the untouched
# warp selection sort, at every T and every distribution -- including T > topk, where the arm must
# be inert. It also prints the item-1.3 timing band.
if [ "$PHASE_FROM" -le 0 ]; then
say "=== PHASE 0: gate_topk_radix (no checkpoint, both early arms) ==="
if [ ! -x build/gate_topk_radix ] || [ include/topk_radix.h -nt build/gate_topk_radix ]; then
  say "REFUSING: build/gate_topk_radix missing or stale. Run scripts/build_gate.sh."; exit 1; fi
./build/gate_topk_radix | tee $EV/gate_topk_radix_1p3.log
[ "${PIPESTATUS[0]}" = "0" ] || { say "FATAL: the early-out is not bit-identical. Nothing else runs."; exit 1; }
fi

# ================= PHASE 1 — bit-exactness, in situ, at real context ==========================
# DSV4_TOPK_GATE=1 makes every radix call also run the UNTOUCHED warp selection sort into a private
# buffer and memcmp the whole index array, aborting on the first differing slot. With the early-out
# ON this is transitive proof that early == warp on real score rows, which is what hard invariant 1
# asks for after 1.0's amendment (token ids are not a valid test at context).
# TARGETS ARE ALL IN THE EARLY REGIME on purpose: a gate run at ctx 6144 would exercise the arm's
# inert branch and prove nothing about the branch that changed.
if [ "$PHASE_FROM" -le 1 ]; then
say "=== PHASE 1: bit-exactness gate (DSV4_TOPK_GATE=1, early ON) ==="
export DSV4_TOPK_GATE=1
unset DSV4_TOPK_EARLY
GATE_LOG=$EV/server_1p3_gate.log
: > "$GATE_LOG"
start_server "$GATE_LOG" gate || exit 1
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
  say "FATAL: bit-exactness gate did not pass cleanly. STOPPING before spending four more loads."
  exit 1
fi
unset DSV4_TOPK_GATE
fi

# ================= PHASE 2 — tok/s, OFF arm (full threshold search) ==========================
if [ "$PHASE_FROM" -le 2 ]; then
say "=== PHASE 2: tok/s sweep, DSV4_TOPK_EARLY=0 (full threshold search) ==="
export DSV4_TOPK_EARLY=0
OFF_LOG=$EV/server_1p3_off.log
: > "$OFF_LOG"
start_server "$OFF_LOG" off || exit 1
python3 tools/decode_fit_probe.py --outdir "$OFF_OUT" --targets "$TARGETS" --reps "$REPS" --ckpt "$CKPT"
rc=$?; say "off probe rc=$rc"
server_down
[ "$rc" = "0" ] || { say "FATAL: OFF sweep failed"; exit 1; }
fi

# ================= PHASE 3 — tok/s, ON arm (early-out) =======================================
if [ "$PHASE_FROM" -le 3 ]; then
say "=== PHASE 3: tok/s sweep, early-out ON ==="
unset DSV4_TOPK_EARLY
ON_LOG=$EV/server_1p3_on.log
: > "$ON_LOG"
start_server "$ON_LOG" on || exit 1
python3 tools/decode_fit_probe.py --outdir "$ON_OUT" --targets "$TARGETS" --reps "$REPS" --ckpt "$CKPT"
rc=$?; say "on probe rc=$rc"
server_down
[ "$rc" = "0" ] || { say "FATAL: ON sweep failed"; exit 1; }
fi

# ================= PHASE 4 — dprof, OFF then ON ==============================================
# The instrument that can see 4 us. Both arms carry the identical dprof overhead, so the `i:topk`
# medians are comparable to each other; they are NOT comparable to PHASE 2/3's clean tok/s, and
# nothing here should be quoted as a throughput number.
if [ "$PHASE_FROM" -le 4 ]; then
export DSV4_DPROF=1
export DSV4_DPROF_EVERY="${DSV4_DPROF_EVERY:-4}"
for arm in off on; do
  say "=== PHASE 4$arm: dprof attribution, early=$arm ==="
  if [ "$arm" = "off" ]; then export DSV4_TOPK_EARLY=0; OUT=$DPOFF_OUT; else unset DSV4_TOPK_EARLY; OUT=$DPON_OUT; fi
  DP_LOG=$EV/server_1p3_dprof_$arm.log
  : > "$DP_LOG"
  start_server "$DP_LOG" "dprof-$arm" || exit 1
  python3 tools/decode_fit_probe.py --outdir "$OUT" --targets "$DPROF_TARGETS" --reps "$DPROF_REPS" \
          --max-tokens 128 --no-control --ckpt "$CKPT"
  rc=$?; say "dprof-$arm probe rc=$rc"
  server_down
  [ "$rc" = "0" ] || { say "FATAL: dprof $arm sweep failed"; exit 1; }
  python3 tools/dprof_ctx.py "$DP_LOG" | tee $EV/dprof_ctx_1p3_$arm.txt
done
unset DSV4_DPROF DSV4_TOPK_EARLY
fi

# ================= PHASE 5 — the standing LOSSLESS gate ======================================
if [ "$PHASE_FROM" -le 5 ]; then
say "=== PHASE 5: standing GATE + LOSSLESS GATE (build/decode, early ON) ==="
wait_mem || exit 1
DEC_LOG=$EV/decode_1p3_lossless.log
bash scripts/run_model.sh "$DEC_LOG" ./build/decode "$CKPT" "0,671,6102,294,8760,344" 8 "" 64
for _ in $(seq 1 240); do pgrep -f 'build/decode' > /dev/null || break; sleep 10; done
say "decode finished; gates:"
if [ ! -s "$DEC_LOG" ]; then say "FATAL: PHASE 5 produced no log -- the decode run never started."; exit 1; fi
grep -E 'GATE|tok/verify|tok/s|LOSSLESS' "$DEC_LOG" | tail -20
grep -qE 'LOSSLESS GATE: .* -> PASS' "$DEC_LOG" || { say "FATAL: LOSSLESS gate did not pass"; exit 1; }
fi

# ================= PHASE 6 — the ratchet =====================================================
say "=== PHASE 6: reports ==="
say "--- BEFORE (full threshold search) ---"
python3 tools/decode_fit_probe.py --outdir "$OFF_OUT" --report | tee $EV/fit_1p3_off.txt
say "--- AFTER (lim <= topk early-out) ---"
python3 tools/decode_fit_probe.py --outdir "$ON_OUT"  --report | tee $EV/fit_1p3_on.txt
say "--- paired, per (target, rep), tau and width reported per leg ---"
python3 tools/mainkv_ab_compare.py "$OFF_OUT" "$ON_OUT" | tee $EV/fit_1p3_paired.txt
say "--- i:topk, the mark that brackets the changed launch ---"
for arm in off on; do
  echo "### early=$arm"; sed -n '/PER-POINT MEDIANS/,/^$/p' $EV/dprof_ctx_1p3_$arm.txt | grep -E 'mark|i:topk|cattn:indexer|STEP'
done | tee $EV/fit_1p3_itopk.txt
say "done."
