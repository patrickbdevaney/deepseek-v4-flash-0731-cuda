#!/usr/bin/env bash
# sparse_ab_run.sh — DECODE_LADDER item 1.7, end to end: the memcmp gate, the in-situ engine gates,
# both arms of the tok/s A/B, both arms of a dprof attribution on the `cattn:sparse` mark, and the
# standing LOSSLESS gate.
#
#   nohup setsid bash scripts/sparse_ab_run.sh > evidence/decode_loop/sparse_ab.log 2>&1 </dev/null &
#
# WHY A SCRIPT AND NOT A SEQUENCE OF COMMANDS. Five checkpoint loads that need no steering once
# started, so CLAUDE.md says detach it before it starts. Named in detach_audit.sh's PATTERNS in the
# same commit, because a stage the audit cannot see reports as "all detached" by never being looked
# at.
#
# THE ARM IS AN ENVIRONMENT VARIABLE ON ONE BINARY. The control arm is
# `DSV4_SPARSE_HPB=1 DSV4_SPARSE_SMEM=0`, which restores the launch that shipped before 1.7 -- the
# same `sparse_attn_kernel_t<16,1>` on a 32-thread block -- bit for bit. It is NOT "1.5's binary":
# per measurement-and-traps §19 a number from a previous iteration is not a valid before-arm, and
# both arms here are measured in the same session on the same build.
#
# 1.7 IS BIT-EXACT, WHICH IS WHY THE FIRST PHASE IS memcmp AND NOT A TOLERANCE. `gate_sparse_hpb`
# compares the ENTIRE output buffer of every (hpb, smem) launch against the hpb=1 launch, byte for
# byte, at six engine shapes, and runs a one-ulp negative control so the gate is shown to be able to
# fail. `sparse_attn` sums the gathered rows IN ORDER under a non-associative online softmax, so a
# cosine gate would pass exactly the defect this must catch -- the same reasoning that put
# gate_topk_radix, gate_tc_fp8_kc and gate_og_ws1 on memcmp.
#
# WHY THERE IS A dprof PAIR AND NOT JUST A tok/s PAIR, stated BEFORE the run so it cannot be a
# post-hoc excuse. `gate_sparse_hpb` puts the kernel at 0.746 -> 0.548 ms at the mean verify shape,
# a 1.36x saving; 0.4 measured the `cattn:sparse` MARK at 21.17 ms at ctx 12,288. If the mark scales
# with the kernel, 1.7 is worth ~5.6 ms of a ~150 ms forward: 3.7 %, against a run-to-run spread
# this ladder quotes as 3.5 % and measurement-and-traps §19 puts nearer 5.7 % between loads. SO THE
# tok/s SWEEP IS EXPECTED TO BE MARGINAL AND THE dprof PAIR IS THE INSTRUMENT THAT CAN ACTUALLY
# RESOLVE THIS. Both are run: tok/s because the ladder's ratchet is throughput with tau, dprof
# because `cattn:sparse` brackets exactly the launch that changed.
#
# PHASE ORDER IS THE DRIFT CONTROL. The control arm runs BEFORE the 1.7 arm in both pairs, so
# thermal drift over the run makes the new kernel look SLOWER, not faster.
set -u
cd "$(dirname "$0")/.."
export PATH=/usr/local/cuda/bin:$PATH

CKPT="${CKPT:-/home/patrickd/models/DeepSeek-V4-Flash-0731-REAP}"
SEQMAX="${SEQMAX:-16384}"
EV=evidence/decode_loop
OFF_OUT=$EV/fit_1p7_off          # control: pre-1.7 launch
ON_OUT=$EV/fit_1p7_on            # 1.7: smem-staged
DPOFF_OUT=$EV/dprof_1p7_off
DPON_OUT=$EV/dprof_1p7_on
TARGETS="${TARGETS:-12288,6144,3072}"
REPS="${REPS:-4}"
DPROF_TARGETS="${DPROF_TARGETS:-12288,6144,3072}"
DPROF_REPS="${DPROF_REPS:-3}"
PHASE_FROM="${PHASE_FROM:-0}"

say(){ echo "[1.7] $(date -Is) $*"; }

# ---- the binaries must be the ones we are about to reason about -------------------------------
for f in kernels/mla_attn.cu include/mla_attn.h; do
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
  say "starting server [$label] seqmax=$SEQMAX HPB=${DSV4_SPARSE_HPB:-default} SMEM=${DSV4_SPARSE_SMEM:-default} -> $log"
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
# Seconds, no checkpoint. Every (hpb, smem) launch vs the pre-1.7 launch, byte for byte, at six
# engine shapes, plus a one-ulp negative control per shape.
if [ "$PHASE_FROM" -le 0 ]; then
say "=== PHASE 0: gate_sparse_hpb (no checkpoint) ==="
if [ ! -x build/gate_sparse_hpb ] || [ kernels/mla_attn.cu -nt build/gate_sparse_hpb ]; then
  say "REFUSING: build/gate_sparse_hpb missing or stale. Run scripts/build_gate.sh."; exit 1; fi
./build/gate_sparse_hpb --control 2>&1 | tee $EV/gate_sparse_hpb_1p7.log
[ "${PIPESTATUS[0]}" = "0" ] || { say "FATAL: sparse_attn is not bit-exact against the pre-1.7 launch. Nothing else runs."; exit 1; }
fi

# ================= PHASE 1 — the in-situ engine gates, still no checkpoint ====================
# gate_units is the COSINE gate against ref/goldens/unit_sparse_attn.safetensors. The other three
# drive `sparse_attn` through real engine entry points (the compressed decode step, the ratio-4
# indexer decode, the prefill chain) with random weights and no checkpoint -- which is where a shape
# the gate sweep did not think of shows up as a wrong number rather than a wrong idea. gate_prefill_len
# in particular is the one that varies `s`, i.e. the `m` this kernel's block mapping depends on.
if [ "$PHASE_FROM" -le 1 ]; then
say "=== PHASE 1: in-situ engine gates (no checkpoint) ==="
rc1=0
for g in gate_units gate_indexer_decode gate_compressed_decode gate_prefill_len gate_mla_decode gate_compressed_graph gate_indexer_graph; do
  if [ ! -x build/$g ]; then say "SKIP $g (no binary)"; continue; fi
  if [ "$g" = gate_units ]; then ./build/$g ref/goldens > $EV/${g}_1p7.log 2>&1; else ./build/$g > $EV/${g}_1p7.log 2>&1; fi
  r=$?; say "$g rc=$r"; grep -E 'FAIL|PASS' $EV/${g}_1p7.log | tail -4
  [ "$r" = "0" ] || rc1=1
done
[ "$rc1" = "0" ] || { say "FATAL: an in-situ gate failed. STOPPING before spending five loads."; exit 1; }
fi

# ================= PHASE 2 — tok/s, CONTROL arm (pre-1.7 launch) =============================
if [ "$PHASE_FROM" -le 2 ]; then
say "=== PHASE 2: tok/s sweep, CONTROL (DSV4_SPARSE_HPB=1 DSV4_SPARSE_SMEM=0) ==="
export DSV4_SPARSE_HPB=1 DSV4_SPARSE_SMEM=0
OFF_LOG=$EV/server_1p7_off.log
: > "$OFF_LOG"
start_server "$OFF_LOG" off || exit 1
python3 tools/decode_fit_probe.py --outdir "$OFF_OUT" --targets "$TARGETS" --reps "$REPS" --ckpt "$CKPT"
rc=$?; say "control probe rc=$rc"
server_down
[ "$rc" = "0" ] || { say "FATAL: control sweep failed"; exit 1; }
unset DSV4_SPARSE_HPB DSV4_SPARSE_SMEM
fi

# ================= PHASE 3 — tok/s, 1.7 arm (smem-staged, default) ===========================
if [ "$PHASE_FROM" -le 3 ]; then
say "=== PHASE 3: tok/s sweep, 1.7 (smem-staged, default) ==="
unset DSV4_SPARSE_HPB DSV4_SPARSE_SMEM
ON_LOG=$EV/server_1p7_on.log
: > "$ON_LOG"
start_server "$ON_LOG" on || exit 1
python3 tools/decode_fit_probe.py --outdir "$ON_OUT" --targets "$TARGETS" --reps "$REPS" --ckpt "$CKPT"
rc=$?; say "1.7 probe rc=$rc"
server_down
[ "$rc" = "0" ] || { say "FATAL: 1.7 sweep failed"; exit 1; }
fi

# ================= PHASE 5 — the standing GATE and LOSSLESS GATE =============================
# 1.7 IS bit-exact, so unlike 1.5 this phase expects BYTE-IDENTICAL generated ids between the arms,
# and the ladder's primary invariant applies rather than its lossless-gate fallback. The gate prompt
# prefills 6 ids, far below the 160-position threshold 1.9 measured, so the token-id comparison is
# valid here (invariant 1, as bounded by 1.9). Three replicates in ONE load per arm.
if [ "$PHASE_FROM" -le 5 ]; then
say "=== PHASE 5: standing GATE + LOSSLESS GATE (build/decode, 1.7 on) ==="
wait_mem || exit 1
DEC_LOG=$EV/decode_1p7_on.log
DSV4_BLKSWEEP="6:1:1.50,6:1:1.50,6:1:1.50" \
  bash scripts/run_model.sh "$DEC_LOG" ./build/decode "$CKPT" "0,671,6102,294,8760,344" 8 "" 64
for _ in $(seq 1 360); do pgrep -f 'build/decode' > /dev/null || break; sleep 10; done
if [ ! -s "$DEC_LOG" ]; then say "FATAL: PHASE 5 produced no log -- the decode run never started."; exit 1; fi
grep -E 'GATE|tok/verify|tok/s|LOSSLESS' "$DEC_LOG" | tail -30
grep -qE 'LOSSLESS GATE: .* -> PASS' "$DEC_LOG" || { say "FATAL: LOSSLESS gate did not pass"; exit 1; }

say "=== PHASE 5b: same three replicates, CONTROL arm (pre-1.7 launch) ==="
wait_mem || exit 1
DECC_LOG=$EV/decode_1p7_control.log
DSV4_SPARSE_HPB=1 DSV4_SPARSE_SMEM=0 DSV4_BLKSWEEP="6:1:1.50,6:1:1.50,6:1:1.50" \
  bash scripts/run_model.sh "$DECC_LOG" ./build/decode "$CKPT" "0,671,6102,294,8760,344" 8 "" 64
for _ in $(seq 1 360); do pgrep -f 'build/decode' > /dev/null || break; sleep 10; done
if [ ! -s "$DECC_LOG" ]; then say "FATAL: PHASE 5b produced no log."; exit 1; fi
grep -E 'GATE|tok/verify|tok/s|LOSSLESS' "$DECC_LOG" | tail -30
fi

# ================= PHASE 6 — dprof, CONTROL then 1.7 =========================================
# The instrument that can see the `cattn:sparse` mark itself. Both arms carry the identical dprof
# overhead, so their medians are comparable to each other; they are NOT comparable to phase 2/3's
# clean tok/s and nothing here should be quoted as a throughput number.
if [ "$PHASE_FROM" -le 6 ]; then
export DSV4_DPROF=1
export DSV4_DPROF_EVERY="${DSV4_DPROF_EVERY:-4}"
for arm in off on; do
  say "=== PHASE 6$arm: dprof attribution, arm=$arm ==="
  if [ "$arm" = "off" ]; then export DSV4_SPARSE_HPB=1 DSV4_SPARSE_SMEM=0; OUT=$DPOFF_OUT
  else unset DSV4_SPARSE_HPB DSV4_SPARSE_SMEM; OUT=$DPON_OUT; fi
  DP_LOG=$EV/server_1p7_dprof_$arm.log
  : > "$DP_LOG"
  start_server "$DP_LOG" "dprof-$arm" || exit 1
  python3 tools/decode_fit_probe.py --outdir "$OUT" --targets "$DPROF_TARGETS" --reps "$DPROF_REPS" \
          --max-tokens 128 --no-control --ckpt "$CKPT"
  rc=$?; say "dprof-$arm probe rc=$rc"
  server_down
  [ "$rc" = "0" ] || { say "FATAL: dprof $arm sweep failed"; exit 1; }
  python3 tools/dprof_ctx.py "$DP_LOG" | tee $EV/dprof_ctx_1p7_$arm.txt
done
unset DSV4_DPROF DSV4_SPARSE_HPB DSV4_SPARSE_SMEM
fi

# ================= PHASE 7 — the ratchet =====================================================
say "=== PHASE 7: reports ==="
say "--- BEFORE (pre-1.7 launch) ---"
python3 tools/decode_fit_probe.py --outdir "$OFF_OUT" --report | tee $EV/fit_1p7_off.txt
say "--- AFTER (smem-staged) ---"
python3 tools/decode_fit_probe.py --outdir "$ON_OUT"  --report | tee $EV/fit_1p7_on.txt
say "--- paired, per (target, rep), tau and width reported per leg ---"
python3 tools/mainkv_ab_compare.py "$OFF_OUT" "$ON_OUT" | tee $EV/fit_1p7_paired.txt
say "--- cattn:sparse, the mark that brackets the changed launch ---"
for arm in off on; do
  echo "### arm=$arm"; sed -n '/PER-POINT MEDIANS/,/^$/p' $EV/dprof_ctx_1p7_$arm.txt | grep -E 'mark|cattn:sparse|cattn:ogroup|cattn:q_proj|STEP'
done | tee $EV/fit_1p7_sparse.txt
say "--- generated ids and tau, both arms, same session ---"
for f in $EV/decode_1p7_control.log $EV/decode_1p7_on.log; do
  echo "### $f"; grep -E 'generated:|tok/verify|WARM decode|spec\] decoded|LOSSLESS' "$f" | tail -12
done | tee $EV/fit_1p7_tau.txt
say "done."
