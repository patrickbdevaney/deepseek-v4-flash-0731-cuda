#!/usr/bin/env bash
# joindefer_ab_run.sh — DECODE_LADDER item 1.11, end to end.
#
#   nohup setsid bash scripts/joindefer_ab_run.sh > evidence/decode_loop/joindefer_ab.log 2>&1 </dev/null &
#
# Named in detach_audit.sh's PATTERNS in the same commit: a stage the audit cannot see reports as
# "all detached" by never being looked at.
#
# THE CHANGE. `compressed_verify_step_indexer` forks the two compressor emits onto `g_side`
# (Finding 56 / ladder 1.8) and joined them back immediately after `build_qKV`. Nothing between that
# join and `index_score` reads what the emits write -- the first consumer of `idx_ckv` is
# `index_score` and of `comp_kv` is the `kv_all` copy, while `i:qidx` and `i:iw` read neither. 1.11
# moves the `cudaStreamWaitEvent` to the true dependency and leaves the `cudaEventRecord` where the
# emits end. 1.8 measured 1.54 + 0.45 = 1.99 ms of main-stream work in that window.
#
# THE ARM IS AN ENVIRONMENT VARIABLE ON ONE BINARY. `NO_JOIN_DEFER=1` restores the 1.8 join
# position; unset is 1.11. Both arms are the same build, measured in the same session, because per
# measurement-and-traps §19 a number from a previous iteration is not a valid before-arm.
#
# PRE-REGISTERED, BEFORE THE RUN, SO NEITHER OUTCOME CAN BE RATIONALISED AFTERWARDS.
#   * CEILING is 1.99 ms x 64.9 % of forwards that carry an emit = 1.29 ms/forward, ~0.9 % of a
#     140 ms forward. The run-to-run spread is 3.5 %, so this is ONLY resolvable paired.
#   * 1.8 DOES NOT ESTABLISH THE CEILING WILL BE RECOVERED. The deferred window's own traffic
#     contends with the same emits on the same memory system, which is exactly why the existing
#     overlap recovers only 16 % of what it hides. A PAIRED BAND THAT COVERS ZERO IS THE EXPECTED
#     NULL and gets written into wiki/negative-results.md as a negative, not rounded into a win.
#   * TERM A ONLY, and that is a prediction, not a hope: an emit reads `x_full` for one group of
#     `ratio` tokens and `i:qidx`/`i:iw` are K x QD and K x nH -- every byte in this window is
#     context-independent. If the paired split shows the saving in the CONTEXT term instead, the
#     mechanism claimed here is wrong and the item says so.
#   * BIT-EXACT IS NOT A HOPE EITHER, it is the construction: kernels move between streams,
#     dependency order is unchanged, no arithmetic is touched. Acceptance is memcmp on generated
#     token ids plus identical completion hashes on every sweep leg. A tolerance would be the wrong
#     gate here for the same reason it was in 1b.1.
#
# READ THE dprof MARKS WITH CARE. Deferring the join MOVES time between marks: `cattn:compress`
# collapses toward zero on the main stream and the emit traffic reappears inside `i:qidx`/`i:iw`.
# A shrinking `cattn:compress` IS NOT THE WIN -- it is what the barrier moving looks like, and 1.8
# already showed the same time teleporting between marks under NO_ATTN_SPLIT=1. Only the paired
# ms/forward decides this item. Phases 1-2 collect the marks; phases 3-4 decide.
#
# PHASE ORDER IS THE DRIFT CONTROL. The CONTROL arm (NO_JOIN_DEFER=1, i.e. the 1.8 join) runs BEFORE
# the deferred arm in every pair, so thermal drift over the run makes 1.11 look SLOWER, not faster.
set -u
cd "$(dirname "$0")/.."
export PATH=/usr/local/cuda/bin:$PATH

CKPT="${CKPT:-/home/patrickd/models/DeepSeek-V4-Flash-0731-REAP}"
SEQMAX="${SEQMAX:-16384}"
EV=evidence/decode_loop
OFF_OUT=$EV/fit_1p11_off         # control: the 1.8 join, immediately after build_qKV
ON_OUT=$EV/fit_1p11_on           # 1.11:    the join deferred to index_score
OFF2_OUT=$EV/fit_1p11_off2       # the same two arms AGAIN with the arm order REVERSED, so that
ON2_OUT=$EV/fit_1p11_on2         # drift enters the second pair with the opposite sign
TARGETS="${TARGETS:-12288,6144,3072}"
REPS="${REPS:-6}"
PHASE_FROM="${PHASE_FROM:-0}"
PHASE_TO="${PHASE_TO:-9}"

say(){ echo "[1.11] $(date -Is) $*"; }
phase(){ [ "$PHASE_FROM" -le "$1" ] && [ "$PHASE_TO" -ge "$1" ]; }

for f in kernels/compressed_decode.cu kernels/compressor.cu kernels/dscratch.cu src/decode.cu src/engine.cu; do
  [ -e "$f" ] || { say "REFUSING: $f missing"; exit 1; }
  if [ "$f" -nt build/dsv4-server ]; then say "REFUSING: build/dsv4-server older than $f"; exit 1; fi
  if [ "$f" -nt build/decode ] && [ "$f" != "src/engine.cu" ]; then
    say "REFUSING: build/decode older than $f"; exit 1; fi
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
    say "waiting for the page cache: ${avail} GiB available, need 105"; sleep 15
  done
  say "FATAL: never got 105 GiB back"; return 1
}
start_server(){
  local log="$1" label="$2"
  if curl -s -m 10 -o /dev/null http://localhost:8080/health 2>/dev/null; then
    say "FATAL: a server is already healthy on :8080 and it is not ours (wrong arm env)"; return 1; fi
  wait_mem || return 1
  say "starting server [$label] seqmax=$SEQMAX NO_JOIN_DEFER=${NO_JOIN_DEFER:-unset} -> $log"
  SEQMAX="$SEQMAX" LOG="$log" bash scripts/run_server.sh || { say "run_server refused"; return 1; }
  setsid nohup env PAT=dsv4-server bash scripts/memguard.sh \
      > "${log%.log}.memguard.log" 2>&1 < /dev/null &
  for _ in $(seq 1 480); do
    curl -s -m 10 -o /dev/null http://localhost:8080/health 2>/dev/null && {
      say "healthy [$label]: $(curl -s -m 10 http://localhost:8080/health)"; return 0; }
    pgrep -f 'build/dsv4-server --ckpt' > /dev/null || {
      say "FATAL: server process is gone while we waited. Tail of $log:"; tail -40 "$log"; return 1; }
    sleep 5
  done
  say "FATAL: server never became healthy"; tail -40 "$log"; return 1
}
wait_decode(){ for _ in $(seq 1 720); do pgrep -x decode > /dev/null || return 0; sleep 10; done
               say "FATAL: build/decode never exited"; return 1; }

mkdir -p "$OFF_OUT" "$ON_OUT" "$OFF2_OUT" "$ON2_OUT"

# ============ PHASE 0 — the no-checkpoint gates, BOTH ARMS ====================================
# These are the gates that actually drive compressed_verify_step_indexer and the graph capture that
# the deferred wait has to survive. gate_forkjoin_graph is the one that exists precisely because a
# fork/join across streams inside a capture is the thing that can silently rewire a graph.
if phase 0; then
say "=== PHASE 0: in-situ gates, BOTH arms (no checkpoint) ==="
rc0=0
# 0a. THE TARGETED GATE. gate_join_defer is the only gate in this repo that calls arena_init()
# before driving these kernels, so it is the only one for which `g_side` is non-null and the
# fork/join is reachable at all. It runs both join positions in ONE process on the same weights and
# memcmps; --swap controls for arm order and --negctl proves the memcmp is live. It sets and unsets
# NO_JOIN_DEFER itself, so it must NOT run inside the arm loop below.
env -u NO_JOIN_DEFER ./build/gate_join_defer                > $EV/gate_join_defer_1p11.log 2>&1  || rc0=1
env -u NO_JOIN_DEFER ./build/gate_join_defer --swap       >> $EV/gate_join_defer_1p11.log 2>&1  || rc0=1
# The two NULL controls: both arms pinned to the SAME join position. These are what turned the
# gate's first result -- one differing row -- from "1.11 is not bit-exact" into "the first run_arm()
# in a process reads uninitialised arena scratch in the M=1 step". They must be CLEAN, and they are
# the reason the PASS above means anything.
env -u NO_JOIN_DEFER ./build/gate_join_defer --same-join  >> $EV/gate_join_defer_1p11.log 2>&1  || rc0=1
env -u NO_JOIN_DEFER ./build/gate_join_defer --same-defer >> $EV/gate_join_defer_1p11.log 2>&1  || rc0=1
env -u NO_JOIN_DEFER ./build/gate_join_defer --negctl     >> $EV/gate_join_defer_1p11.log 2>&1  || rc0=1
cat $EV/gate_join_defer_1p11.log
[ "$rc0" = "0" ] || { say "FATAL: gate_join_defer failed. 1.11 is not bit-exact; nothing else runs."; exit 1; }

# 0b. THE ARM-INVARIANT GATES. These link without an arena and therefore take the single-stream
# path; they are here to prove 1.11 broke nothing ELSE, not to prove 1.11 itself.
for arm in defer nodefer; do
  if [ "$arm" = nodefer ]; then export NO_JOIN_DEFER=1; else unset NO_JOIN_DEFER; fi
  for g in gate_units gate_indexer_decode gate_compressed_decode gate_compressed_graph \
           gate_indexer_graph gate_compressor_emit gate_forkjoin_graph gate_index_score \
           gate_prefill_len gate_topk_smem_ctx; do
    [ -x build/$g ] || { say "SKIP $g (no binary)"; continue; }
    if [ "$g" = gate_units ]; then ./build/$g ref/goldens > $EV/${g}_1p11_$arm.log 2>&1
    else                           ./build/$g            > $EV/${g}_1p11_$arm.log 2>&1; fi
    r=$?; say "arm=$arm $g rc=$r"; grep -E 'FAIL|PASS' $EV/${g}_1p11_$arm.log | tail -3
    [ "$r" = "0" ] || rc0=1
  done
done
unset NO_JOIN_DEFER
[ "$rc0" = "0" ] || { say "FATAL: an in-situ gate failed. STOPPING before spending any loads."; exit 1; }
fi

# ============ PHASE 1 — build/decode, CONTROL arm (the 1.8 join) ==============================
# The gate prompt prefills 6 ids, far below the 160-position threshold 1.9 measured, so the
# token-id comparison is a VALID gate here (invariant 1 as bounded by 1.9).
# DSV4_KSWEEP + DSV4_DPROF ride along: the K-sweep runs AFTER generation and cannot touch the ids,
# and it is the only place that prints a per-K mark table without a second checkpoint load. Every
# byte the 1.11 window moves is context-independent (one group of `ratio` tokens for the emits,
# K x QD and K x nH for i:qidx / i:iw), so a PS=6 mark table is representative of the mechanism --
# it is NOT representative of throughput, which is what phases 3-4 are for.
if phase 1; then
say "=== PHASE 1: build/decode, CONTROL (1.8 join, NO_JOIN_DEFER=1) ==="
wait_mem || exit 1
NO_JOIN_DEFER=1 DSV4_KSWEEP=1 DSV4_DPROF=1 \
  bash scripts/run_model.sh $EV/decode_1p11_off.log ./build/decode "$CKPT" "0,671,6102,294,8760,344" 8 "" 64
wait_decode || exit 1
[ -s "$EV/decode_1p11_off.log" ] || { say "FATAL: control decode produced no log."; exit 1; }
grep -E 'GATE|tok/verify|tok/s|LOSSLESS|generated:|mem ' $EV/decode_1p11_off.log | tail -30
fi

# ============ PHASE 2 — build/decode, DEFERRED arm ============================================
if phase 2; then
say "=== PHASE 2: build/decode, DEFERRED (1.11) ==="
wait_mem || exit 1
DSV4_KSWEEP=1 DSV4_DPROF=1 \
  bash scripts/run_model.sh $EV/decode_1p11_on.log ./build/decode "$CKPT" "0,671,6102,294,8760,344" 8 "" 64
wait_decode || exit 1
[ -s "$EV/decode_1p11_on.log" ] || { say "FATAL: deferred decode produced no log."; exit 1; }
grep -E 'GATE|tok/verify|tok/s|LOSSLESS|generated:|mem ' $EV/decode_1p11_on.log | tail -30
say "--- generated ids, both arms (must be IDENTICAL: 1.11 is bit-exact by construction) ---"
a=$(grep -m1 'generated:' $EV/decode_1p11_off.log); b=$(grep -m1 'generated:' $EV/decode_1p11_on.log)
echo "off: $a"; echo " on: $b"
[ "$a" = "$b" ] && say "TOKEN-ID GATE: PASS (identical)" || say "TOKEN-ID GATE: FAIL (arms diverged)"
fi

# ============ PHASE 3 — tok/s sweep, CONTROL arm =============================================
if phase 3; then
say "=== PHASE 3: tok/s sweep, CONTROL (1.8 join) ==="
export NO_JOIN_DEFER=1
: > $EV/server_1p11_off.log
start_server $EV/server_1p11_off.log off || exit 1
python3 tools/decode_fit_probe.py --outdir "$OFF_OUT" --targets "$TARGETS" --reps "$REPS" --ckpt "$CKPT"
rc=$?; say "control probe rc=$rc"; server_down
[ "$rc" = "0" ] || { say "FATAL: control sweep failed"; exit 1; }
unset NO_JOIN_DEFER
fi

# ============ PHASE 4 — tok/s sweep, DEFERRED arm ============================================
if phase 4; then
say "=== PHASE 4: tok/s sweep, DEFERRED (1.11) ==="
unset NO_JOIN_DEFER
: > $EV/server_1p11_on.log
start_server $EV/server_1p11_on.log on || exit 1
python3 tools/decode_fit_probe.py --outdir "$ON_OUT" --targets "$TARGETS" --reps "$REPS" --ckpt "$CKPT"
rc=$?; say "deferred probe rc=$rc"; server_down
[ "$rc" = "0" ] || { say "FATAL: deferred sweep failed"; exit 1; }
fi

# ============ PHASE 5 — the ratchet ==========================================================
if phase 5; then
say "=== PHASE 5: reports ==="
say "--- BEFORE (1.8 join) ---"
python3 tools/decode_fit_probe.py --outdir "$OFF_OUT" --report | tee $EV/fit_1p11_off.txt
say "--- AFTER (1.11 deferred join) ---"
python3 tools/decode_fit_probe.py --outdir "$ON_OUT"  --report | tee $EV/fit_1p11_on.txt
say "--- paired, per (target, rep), tau and completion hash reported per leg ---"
python3 tools/mainkv_ab_compare.py "$OFF_OUT" "$ON_OUT" | tee $EV/fit_1p11_paired.txt
say "--- dprof mark tables, both build/decode arms: WHERE the time moved, not whether it shrank ---"
for f in $EV/decode_1p11_off.log $EV/decode_1p11_on.log; do
  echo "### $f"; grep -E 'dprof|cattn:|i:qidx|i:iw|i:score|generated:|tok/verify|LOSSLESS' "$f" | tail -80
done | tee $EV/dprof_1p11_marks.txt
fi
# ============ PHASES 6-7 — THE REVERSED-ORDER REPLICATE ======================================
# WHY. Phases 3-4 put the CONTROL arm first, so drift penalises 1.11. That is the right default, but
# it only bounds the answer when the effect is larger than the between-LOAD offset, and
# measurement-and-traps §19 measures that offset at 0.6 % = ~0.8 ms/forward -- the same size as this
# item's whole pre-registered ceiling. Pairing removes between-LEG variance; it does NOT remove a
# constant offset between two checkpoint loads.
#
# So run the pair again with the ARM ORDER REVERSED: deferred first, joined second. Drift now
# penalises the CONTROL. Writing d1 = effect + drift and d2 = effect - drift, the average of the two
# paired means is drift-free and their difference measures the drift itself. Two more loads, and it
# is the difference between "the band covers zero" and "the band covers zero BECAUSE the design
# cannot resolve it".
if phase 6; then
say "=== PHASE 6: REVERSED replicate, DEFERRED arm FIRST ==="
unset NO_JOIN_DEFER
: > $EV/server_1p11_on2.log
start_server $EV/server_1p11_on2.log on2 || exit 1
python3 tools/decode_fit_probe.py --outdir "$ON2_OUT" --targets "$TARGETS" --reps "$REPS" --ckpt "$CKPT"
rc=$?; say "reversed deferred probe rc=$rc"; server_down
[ "$rc" = "0" ] || { say "FATAL: reversed deferred sweep failed"; exit 1; }
fi

if phase 7; then
say "=== PHASE 7: REVERSED replicate, CONTROL arm SECOND ==="
export NO_JOIN_DEFER=1
: > $EV/server_1p11_off2.log
start_server $EV/server_1p11_off2.log off2 || exit 1
python3 tools/decode_fit_probe.py --outdir "$OFF2_OUT" --targets "$TARGETS" --reps "$REPS" --ckpt "$CKPT"
rc=$?; say "reversed control probe rc=$rc"; server_down
[ "$rc" = "0" ] || { say "FATAL: reversed control sweep failed"; exit 1; }
unset NO_JOIN_DEFER
fi

if phase 8; then
say "=== PHASE 8: the drift-free estimate ==="
say "--- run 1: control arm first (drift penalises 1.11) ---"
python3 tools/paired_band.py "$OFF_OUT" "$ON_OUT" --label-before joined --label-after deferred \
  | tee $EV/fit_1p11_band.txt
say "--- run 2: deferred arm first (drift penalises the control) ---"
python3 tools/paired_band.py "$OFF2_OUT" "$ON2_OUT" --label-before joined --label-after deferred \
  | tee $EV/fit_1p11_band_rev.txt
say "--- pooled, drift-free ---"
python3 tools/paired_band.py "$OFF_OUT" "$ON_OUT" --reversed "$OFF2_OUT" "$ON2_OUT" \
  --label-before joined --label-after deferred | tee $EV/fit_1p11_band_pooled.txt
fi
say "done."
