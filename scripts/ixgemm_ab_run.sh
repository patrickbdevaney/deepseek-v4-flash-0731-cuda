#!/usr/bin/env bash
# ixgemm_ab_run.sh — DECODE_LADDER item 1.5, end to end: the memcmp gate, the in-situ engine gates,
# both arms of the tok/s A/B, both arms of a dprof attribution on the `i:score` mark, and the
# standing LOSSLESS gate.
#
#   nohup setsid bash scripts/ixgemm_ab_run.sh > evidence/decode_loop/ixgemm_ab.log 2>&1 </dev/null &
#
# WHY A SCRIPT AND NOT A SEQUENCE OF COMMANDS. Five checkpoint loads that need no steering once
# started, so CLAUDE.md says detach it before it starts. Named in detach_audit.sh's PATTERNS in the
# same commit, because a stage the audit cannot see reports as "all detached" by never being looked
# at.
#
# THE ARM IS AN ENVIRONMENT VARIABLE ON ONE BINARY. The control arm is `NO_IXGEMM=1 NO_IXTILE=1`,
# which restores `index_score_warp_kernel` -- the kernel that shipped before 1.5, bit for bit. It is
# NOT "1.4's binary": per measurement-and-traps §19 a number from a previous iteration is not a
# valid before-arm, and both arms here are measured in the same session on the same build.
#
# WHY THERE IS A DPROF PAIR AND NOT JUST A tok/s PAIR, stated BEFORE the run so it cannot be a
# post-hoc excuse. `tests/gate_index_score` puts the kernel band at 898.8 us -> 132.7 us at the
# verify shape (S=6, T=3072, i.e. ctx 12,288 at ratio 4), a 6.77x saving. 0.4 measured the `i:score`
# MARK -- which is the sum over all ratio-4 layers of one forward -- at 6.58 ms at ctx 12,288. If
# the mark scales with the kernel, 1.5 is worth ~5.6 ms of a ~155 ms forward: 3.6 %, against a
# run-to-run spread this ladder quotes as 3.5 % and measurement-and-traps §19 puts nearer 5.7 %
# between loads. SO THE tok/s SWEEP IS EXPECTED TO BE MARGINAL AND THE dprof PAIR IS THE INSTRUMENT
# THAT CAN ACTUALLY RESOLVE THIS. Both are run: tok/s because the ladder's ratchet is throughput
# with tau, dprof because `i:score` brackets exactly the launch that changed.
#
# PHASE ORDER IS THE DRIFT CONTROL. The control (warp) arm runs BEFORE the GEMM arm in both pairs,
# so thermal drift over the run makes the new kernel look SLOWER, not faster.
#
# 1.5 IS NOT BIT-EXACT WITH THE ARM IT REPLACES AND THAT IS THE POINT OF PHASE 0 AND PHASE 5.
# IXS_GEMM is bit-identical to `index_score_kernel`, the correctness-first scalar REFERENCE that
# gate_units checks against ref/goldens; the warp kernel it replaces was itself adopted (Finding 68)
# as a deviation from that reference, behind the LOSSLESS gate. So the invariant this run has to
# satisfy is the second branch of the ladder's rule: ship behind the LOSSLESS gate with the
# deviation measured. Phase 0 prints the deviation per input distribution; phase 5 is the gate.
set -u
cd "$(dirname "$0")/.."
export PATH=/usr/local/cuda/bin:$PATH

CKPT="${CKPT:-/home/patrickd/models/DeepSeek-V4-Flash-0731-REAP}"
SEQMAX="${SEQMAX:-16384}"
EV=evidence/decode_loop
OFF_OUT=$EV/fit_1p5_off          # control: warp kernel
ON_OUT=$EV/fit_1p5_on            # 1.5: GEMM
DPOFF_OUT=$EV/dprof_1p5_off
DPON_OUT=$EV/dprof_1p5_on
TARGETS="${TARGETS:-12288,6144,3072}"
REPS="${REPS:-4}"
DPROF_TARGETS="${DPROF_TARGETS:-12288,6144,3072}"
DPROF_REPS="${DPROF_REPS:-3}"
PHASE_FROM="${PHASE_FROM:-0}"

say(){ echo "[1.5] $(date -Is) $*"; }

# ---- the binaries must be the ones we are about to reason about -------------------------------
for f in kernels/indexer.cu include/indexer.h; do
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
  say "starting server [$label] seqmax=$SEQMAX NO_IXGEMM=${NO_IXGEMM:-unset} NO_IXTILE=${NO_IXTILE:-unset} -> $log"
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

mkdir -p "$OFF_OUT" "$ON_OUT" "$DPOFF_OUT" "$DPON_OUT"

# ================= PHASE 0 — the memcmp gate, before any weights are loaded ====================
# Seconds, no checkpoint. TWO claims: IXS_TILED == IXS_WARP and IXS_GEMM == IXS_SCALAR. It also
# prints the deviation IXS_GEMM spends against the shipped warp kernel, per input distribution.
if [ "$PHASE_FROM" -le 0 ]; then
say "=== PHASE 0: gate_index_score (no checkpoint) ==="
if [ ! -x build/gate_index_score ] || [ kernels/indexer.cu -nt build/gate_index_score ]; then
  say "REFUSING: build/gate_index_score missing or stale. Run scripts/build_gate.sh."; exit 1; fi
./build/gate_index_score 2>&1 | tee $EV/gate_index_score_1p5.log
[ "${PIPESTATUS[0]}" = "0" ] || { say "FATAL: index_score is not bit-exact against its reference. Nothing else runs."; exit 1; }
fi

# ================= PHASE 1 — the in-situ engine gates, still no checkpoint ====================
# gate_units is the COSINE gate against ref/goldens/unit_index_score.safetensors: 1.5 restores the
# reference's own summation order, so this must pass at least as tightly as before. The other three
# drive `index_score` through real engine entry points (the decode step, the prefill chain, the
# ratio-4 compressed decode) with random weights and no checkpoint -- which is where a shape the
# gate sweep did not think of shows up as a wrong number rather than a wrong idea.
if [ "$PHASE_FROM" -le 1 ]; then
say "=== PHASE 1: in-situ engine gates (no checkpoint) ==="
rc1=0
for g in gate_units gate_indexer_decode gate_compressed_decode gate_prefill_len; do
  if [ ! -x build/$g ]; then say "REFUSING: build/$g missing. Run scripts/build_gate.sh."; exit 1; fi
  if [ "$g" = gate_units ]; then ./build/$g ref/goldens > $EV/${g}_1p5.log 2>&1; else ./build/$g > $EV/${g}_1p5.log 2>&1; fi
  r=$?; say "$g rc=$r"; grep -E 'FAIL|PASS' $EV/${g}_1p5.log | tail -4
  [ "$r" = "0" ] || rc1=1
done
[ "$rc1" = "0" ] || { say "FATAL: an in-situ gate failed. STOPPING before spending five loads."; exit 1; }
fi

# ================= PHASE 2 — tok/s, CONTROL arm (warp kernel, i.e. pre-1.5) ==================
if [ "$PHASE_FROM" -le 2 ]; then
say "=== PHASE 2: tok/s sweep, CONTROL (NO_IXGEMM=1 NO_IXTILE=1 -> warp kernel) ==="
export NO_IXGEMM=1 NO_IXTILE=1
OFF_LOG=$EV/server_1p5_off.log
: > "$OFF_LOG"
start_server "$OFF_LOG" off || exit 1
python3 tools/decode_fit_probe.py --outdir "$OFF_OUT" --targets "$TARGETS" --reps "$REPS" --ckpt "$CKPT"
rc=$?; say "control probe rc=$rc"
server_down
[ "$rc" = "0" ] || { say "FATAL: control sweep failed"; exit 1; }
unset NO_IXGEMM NO_IXTILE
fi

# ================= PHASE 3 — tok/s, 1.5 arm (GEMM) ===========================================
if [ "$PHASE_FROM" -le 3 ]; then
say "=== PHASE 3: tok/s sweep, 1.5 (GEMM, default) ==="
unset NO_IXGEMM NO_IXTILE
ON_LOG=$EV/server_1p5_on.log
: > "$ON_LOG"
start_server "$ON_LOG" on || exit 1
python3 tools/decode_fit_probe.py --outdir "$ON_OUT" --targets "$TARGETS" --reps "$REPS" --ckpt "$CKPT"
rc=$?; say "gemm probe rc=$rc"
server_down
[ "$rc" = "0" ] || { say "FATAL: gemm sweep failed"; exit 1; }
fi

# ================= PHASE 4 — dprof, CONTROL then 1.5 =========================================
# The instrument that can see the `i:score` mark itself. Both arms carry the identical dprof
# overhead, so their medians are comparable to each other; they are NOT comparable to phase 2/3's
# clean tok/s and nothing here should be quoted as a throughput number.
if [ "$PHASE_FROM" -le 4 ]; then
export DSV4_DPROF=1
export DSV4_DPROF_EVERY="${DSV4_DPROF_EVERY:-4}"
for arm in off on; do
  say "=== PHASE 4$arm: dprof attribution, arm=$arm ==="
  if [ "$arm" = "off" ]; then export NO_IXGEMM=1 NO_IXTILE=1; OUT=$DPOFF_OUT
  else unset NO_IXGEMM NO_IXTILE; OUT=$DPON_OUT; fi
  DP_LOG=$EV/server_1p5_dprof_$arm.log
  : > "$DP_LOG"
  start_server "$DP_LOG" "dprof-$arm" || exit 1
  python3 tools/decode_fit_probe.py --outdir "$OUT" --targets "$DPROF_TARGETS" --reps "$DPROF_REPS" \
          --max-tokens 128 --no-control --ckpt "$CKPT"
  rc=$?; say "dprof-$arm probe rc=$rc"
  server_down
  [ "$rc" = "0" ] || { say "FATAL: dprof $arm sweep failed"; exit 1; }
  python3 tools/dprof_ctx.py "$DP_LOG" | tee $EV/dprof_ctx_1p5_$arm.txt
done
unset DSV4_DPROF NO_IXGEMM NO_IXTILE
fi

# ================= PHASE 5 — the standing GATE and LOSSLESS GATE =============================
# This is the branch of the ladder's bit-exactness invariant that 1.5 has to satisfy: the change is
# NOT byte-identical to the arm it replaces, so it ships behind this gate. Three replicates in ONE
# checkpoint load (measurement-and-traps §19: the dominant variance is between loads), so tau and
# the generated ids are comparable within the run.
if [ "$PHASE_FROM" -le 5 ]; then
say "=== PHASE 5: standing GATE + LOSSLESS GATE (build/decode, GEMM on) ==="
wait_mem || exit 1
DEC_LOG=$EV/decode_1p5_lossless.log
DSV4_BLKSWEEP="6:1:1.50,6:1:1.50,6:1:1.50" \
  bash scripts/run_model.sh "$DEC_LOG" ./build/decode "$CKPT" "0,671,6102,294,8760,344" 8 "" 64
for _ in $(seq 1 360); do pgrep -f 'build/decode' > /dev/null || break; sleep 10; done
say "decode finished; gates:"
if [ ! -s "$DEC_LOG" ]; then say "FATAL: PHASE 5 produced no log -- the decode run never started."; exit 1; fi
grep -E 'GATE|tok/verify|tok/s|LOSSLESS' "$DEC_LOG" | tail -30
grep -qE 'LOSSLESS GATE: .* -> PASS' "$DEC_LOG" || { say "FATAL: LOSSLESS gate did not pass"; exit 1; }

# THE SAME-ARM CONTROL, in the same session (measurement-and-traps §19). Without it a tau or tok/s
# difference against phase 5 could be the kernel or could be the load.
say "=== PHASE 5b: same three replicates, CONTROL arm (warp) ==="
wait_mem || exit 1
DECC_LOG=$EV/decode_1p5_control.log
NO_IXGEMM=1 NO_IXTILE=1 DSV4_BLKSWEEP="6:1:1.50,6:1:1.50,6:1:1.50" \
  bash scripts/run_model.sh "$DECC_LOG" ./build/decode "$CKPT" "0,671,6102,294,8760,344" 8 "" 64
for _ in $(seq 1 360); do pgrep -f 'build/decode' > /dev/null || break; sleep 10; done
if [ ! -s "$DECC_LOG" ]; then say "FATAL: PHASE 5b produced no log."; exit 1; fi
grep -E 'GATE|tok/verify|tok/s|LOSSLESS' "$DECC_LOG" | tail -30
fi

# ================= PHASE 6 — the ratchet =====================================================
say "=== PHASE 6: reports ==="
say "--- BEFORE (warp kernel) ---"
python3 tools/decode_fit_probe.py --outdir "$OFF_OUT" --report | tee $EV/fit_1p5_off.txt
say "--- AFTER (register-tiled GEMM) ---"
python3 tools/decode_fit_probe.py --outdir "$ON_OUT"  --report | tee $EV/fit_1p5_on.txt
say "--- paired, per (target, rep), tau and width reported per leg ---"
python3 tools/mainkv_ab_compare.py "$OFF_OUT" "$ON_OUT" | tee $EV/fit_1p5_paired.txt
say "--- i:score, the mark that brackets the changed launch ---"
for arm in off on; do
  echo "### arm=$arm"; sed -n '/PER-POINT MEDIANS/,/^$/p' $EV/dprof_ctx_1p5_$arm.txt | grep -E 'mark|i:score|i:topk|cattn:indexer|STEP'
done | tee $EV/fit_1p5_iscore.txt
say "--- generated ids and tau, both arms, same session ---"
for f in $EV/decode_1p5_control.log $EV/decode_1p5_lossless.log; do
  echo "### $f"; grep -E 'generated:|tok/verify|WARM decode|spec\] decoded|LOSSLESS' "$f" | tail -12
done | tee $EV/fit_1p5_tau.txt
say "done."
