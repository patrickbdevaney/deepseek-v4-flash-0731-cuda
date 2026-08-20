#!/usr/bin/env bash
# kvpack_ab_run.sh — DECODE_LADDER item 1b.2, end to end: the two memcmp gates, the in-situ engine
# gates, both arms of the build/decode token-id + tau gate, both arms of the tok/s sweep, and the
# capacity measurement that is the actual point of the item.
#
#   nohup setsid bash scripts/kvpack_ab_run.sh > evidence/decode_loop/kvpack_ab.log 2>&1 </dev/null &
#
# Named in detach_audit.sh's PATTERNS in the same commit: a stage the audit cannot see reports as
# "all detached" by never being looked at.
#
# THE ARM IS AN ENVIRONMENT VARIABLE ON ONE BINARY. `DSV4_KV_PACK=1` puts the main KV caches in the
# 720 B FP8+UE8M0 layout of include/kv_pack.h; unset is the FP32 layout that shipped before 1b.2,
# byte for byte -- `kv_stage` returns the cache row itself and `kv_commit` is the same
# `act_quant_fp8sim` call that was already there. Both arms are the same build, measured in the same
# session, because per measurement-and-traps §19 a number from a previous iteration is not a valid
# before-arm.
#
# 1b.2 IS BIT-EXACT, WHICH IS WHY EVERY GATE HERE IS memcmp AND NOT A TOLERANCE. Both formats hold
# the identical float values; only the storage differs. A tolerance gate passes a dropped sign bit
# at |delta| = 0, which is exactly the bug 1b.1's first implementation had (16,011 mismatches, worst
# |delta| 0). So: `gate_kv_pack` for the format and the reader, `gate_kv_pack_e2e` for the WIRING
# (the ~30 call sites where a row stride, a memcpy length or a staging buffer had to change -- a
# wrong one produces a plausible number, not a crash), and byte-identical generated ids on
# build/decode for the engine.
#
# WHAT IS EXPECTED AND WHAT IS NOT, STATED BEFORE THE RUN so neither can be a post-hoc excuse.
#   * CAPACITY is the claim KV_PRECISION_FINDINGS.md §3 makes and it is arithmetic: 2048 -> 720 B
#     per row over 43 window caches, 41 compressed caches and 3 draft main-KV caches. Phase 6
#     measures it from the engine's own `mem X/Y GiB` line at a FIXED seqmax, which is a
#     measurement of the resident set and not of the formula.
#   * THROUGHPUT IS NOT PROMISED. §4 of the same document is explicit that the index path is 389x
#     off its own roofline, i.e. not bandwidth-bound, so byte reduction need not pay. The
#     mechanism that COULD pay is `sparse_attn`: 1.7 showed that kernel is issue-bound on the
#     gathered row, and a packed row is 44 vector loads instead of 128 and 720 B instead of 2048.
#     The paired sweep is what decides it, and a band covering zero is a legitimate result to
#     report -- as a negative in wiki/negative-results.md, not as a rounding.
#   * The pack kernel ADDS a staging buffer and a second pass on the prefill write path. If the
#     sweep is negative that is the first place to look.
#
# PHASE ORDER IS THE DRIFT CONTROL. The FP32 (control) arm runs BEFORE the packed arm in every
# pair, so thermal drift over the run makes the new layout look SLOWER, not faster.
#
# WHAT THIS DELIBERATELY DOES NOT DO. It does not run the FP32 arm at seqmax 32768 to show it does
# not fit. This box does not OOM gracefully -- memguard.sh records two whole-machine takedowns with
# no oom-kill line in dmesg -- and deliberately driving it into that state to produce a failure log
# is not worth a machine. The capacity claim is made from the MEASURED per-arm resident set at a
# seqmax both arms fit, plus the packed arm actually running at the larger one.
set -u
cd "$(dirname "$0")/.."
export PATH=/usr/local/cuda/bin:$PATH

CKPT="${CKPT:-/home/patrickd/models/DeepSeek-V4-Flash-0731-REAP}"
SEQMAX="${SEQMAX:-16384}"
BIGSEQ="${BIGSEQ:-32768}"
EV=evidence/decode_loop
OFF_OUT=$EV/fit_1b2_off          # control: FP32 KV rows
ON_OUT=$EV/fit_1b2_on            # 1b.2: packed KV rows
TARGETS="${TARGETS:-12288,6144,3072}"
REPS="${REPS:-4}"
PHASE_FROM="${PHASE_FROM:-0}"
PHASE_TO="${PHASE_TO:-9}"

say(){ echo "[1b.2] $(date -Is) $*"; }
phase(){ [ "$PHASE_FROM" -le "$1" ] && [ "$PHASE_TO" -ge "$1" ]; }

for f in include/kv_pack.h kernels/mla_attn.cu kernels/compressed_decode.cu kernels/mla_decode.cu \
         kernels/compressor.cu kernels/dspark_attn.cu src/engine.cu src/decode.cu; do
  [ -e "$f" ] || { say "REFUSING: $f missing"; exit 1; }
  if [ "$f" -nt build/dsv4-server ]; then say "REFUSING: build/dsv4-server older than $f"; exit 1; fi
  if [ "$f" -nt build/decode ]     && [ "$f" != "src/engine.cu" ]; then
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
  local log="$1" label="$2" sm="${3:-$SEQMAX}"
  if curl -s -m 10 -o /dev/null http://localhost:8080/health 2>/dev/null; then
    say "FATAL: a server is already healthy on :8080 and it is not ours (wrong arm env)"; return 1; fi
  wait_mem || return 1
  say "starting server [$label] seqmax=$sm KV_PACK=${DSV4_KV_PACK:-unset} -> $log"
  SEQMAX="$sm" LOG="$log" bash scripts/run_server.sh || { say "run_server refused"; return 1; }
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

mkdir -p "$OFF_OUT" "$ON_OUT"

# ============ PHASE 0 — the two memcmp gates, before any weights are loaded ====================
if phase 0; then
say "=== PHASE 0: gate_kv_pack + gate_kv_pack_e2e + gate_sparse_hpb (no checkpoint) ==="
rc0=0
./build/gate_kv_pack --control       > $EV/gate_kv_pack_1b2.log 2>&1      || rc0=1
./build/gate_kv_pack_e2e             > $EV/gate_kv_pack_e2e_1b2.log 2>&1  || rc0=1
./build/gate_kv_pack_e2e --swap     >> $EV/gate_kv_pack_e2e_1b2.log 2>&1  || rc0=1
# gate_sparse_hpb is the control on the OTHER side: adding the PACKED template parameter must not
# have moved the FP32 launch, which it memcmps against the pre-1.7 reference.
./build/gate_sparse_hpb --control    > $EV/gate_sparse_hpb_1b2.log 2>&1   || rc0=1
grep -hE 'GATE .*(PASS|FAIL)' $EV/gate_kv_pack_1b2.log $EV/gate_kv_pack_e2e_1b2.log $EV/gate_sparse_hpb_1b2.log
[ "$rc0" = "0" ] || { say "FATAL: a memcmp gate failed. Nothing else runs."; exit 1; }
fi

# ============ PHASE 1 — the in-situ engine gates, still no checkpoint =========================
if phase 1; then
say "=== PHASE 1: in-situ engine gates, BOTH arms (no checkpoint) ==="
rc1=0
for arm in 0 1; do
  for g in gate_units gate_indexer_decode gate_compressed_decode gate_prefill_len gate_mainkv_incr \
           gate_compressed_graph gate_indexer_graph gate_compressor_emit; do
    [ -x build/$g ] || { say "SKIP $g (no binary)"; continue; }
    if [ "$g" = gate_units ]; then DSV4_KV_PACK=$arm ./build/$g ref/goldens > $EV/${g}_1b2_p$arm.log 2>&1
    else                           DSV4_KV_PACK=$arm ./build/$g            > $EV/${g}_1b2_p$arm.log 2>&1; fi
    r=$?; say "pack=$arm $g rc=$r"; grep -E 'FAIL|PASS' $EV/${g}_1b2_p$arm.log | tail -3
    [ "$r" = "0" ] || rc1=1
  done
done
[ "$rc1" = "0" ] || { say "FATAL: an in-situ gate failed. STOPPING before spending any loads."; exit 1; }
fi

# ============ PHASE 2 — build/decode, CONTROL arm (FP32 rows) ================================
# The gate prompt prefills 6 ids, far below the 160-position threshold 1.9 measured, so the
# token-id comparison is a VALID gate here (invariant 1 as bounded by 1.9).
if phase 2; then
say "=== PHASE 2: build/decode, CONTROL (FP32 KV rows) ==="
wait_mem || exit 1
DSV4_BLKSWEEP="6:1:1.50,6:1:1.50,6:1:1.50" DSV4_KV_PACK=0 \
  bash scripts/run_model.sh $EV/decode_1b2_off.log ./build/decode "$CKPT" "0,671,6102,294,8760,344" 8 "" 64
wait_decode || exit 1
[ -s "$EV/decode_1b2_off.log" ] || { say "FATAL: control decode produced no log."; exit 1; }
grep -E 'KV layout|GATE|tok/verify|tok/s|LOSSLESS|generated:|mem ' $EV/decode_1b2_off.log | tail -30
fi

# ============ PHASE 3 — build/decode, PACKED arm =============================================
if phase 3; then
say "=== PHASE 3: build/decode, PACKED (1b.2) ==="
wait_mem || exit 1
DSV4_BLKSWEEP="6:1:1.50,6:1:1.50,6:1:1.50" DSV4_KV_PACK=1 \
  bash scripts/run_model.sh $EV/decode_1b2_on.log ./build/decode "$CKPT" "0,671,6102,294,8760,344" 8 "" 64
wait_decode || exit 1
[ -s "$EV/decode_1b2_on.log" ] || { say "FATAL: packed decode produced no log."; exit 1; }
grep -E 'KV layout|GATE|tok/verify|tok/s|LOSSLESS|generated:|mem ' $EV/decode_1b2_on.log | tail -30
say "--- generated ids, both arms (must be IDENTICAL: 1b.2 is bit-exact) ---"
a=$(grep -m1 'generated:' $EV/decode_1b2_off.log); b=$(grep -m1 'generated:' $EV/decode_1b2_on.log)
echo "off: $a"; echo " on: $b"
[ "$a" = "$b" ] && say "TOKEN-ID GATE: PASS (identical)" || say "TOKEN-ID GATE: FAIL (arms diverged)"
fi

# ============ PHASE 4 — tok/s sweep, CONTROL arm =============================================
if phase 4; then
say "=== PHASE 4: tok/s sweep, CONTROL (FP32 KV rows) ==="
export DSV4_KV_PACK=0
: > $EV/server_1b2_off.log
start_server $EV/server_1b2_off.log off || exit 1
python3 tools/decode_fit_probe.py --outdir "$OFF_OUT" --targets "$TARGETS" --reps "$REPS" --ckpt "$CKPT"
rc=$?; say "control probe rc=$rc"; server_down
[ "$rc" = "0" ] || { say "FATAL: control sweep failed"; exit 1; }
unset DSV4_KV_PACK
fi

# ============ PHASE 5 — tok/s sweep, PACKED arm =============================================
if phase 5; then
say "=== PHASE 5: tok/s sweep, PACKED (1b.2) ==="
export DSV4_KV_PACK=1
: > $EV/server_1b2_on.log
start_server $EV/server_1b2_on.log on || exit 1
python3 tools/decode_fit_probe.py --outdir "$ON_OUT" --targets "$TARGETS" --reps "$REPS" --ckpt "$CKPT"
rc=$?; say "packed probe rc=$rc"; server_down
[ "$rc" = "0" ] || { say "FATAL: packed sweep failed"; exit 1; }
unset DSV4_KV_PACK
fi

# ============ PHASE 6 — CAPACITY, which is the point of the item =============================
# The `[engine] ready. mem X/Y GiB` line of phases 4 and 5 is the paired resident-set measurement at
# a FIXED seqmax. Then the packed arm alone is taken to $BIGSEQ, which is where the item's claim
# lives. The FP32 arm is deliberately NOT taken there; see the header.
if phase 6; then
say "=== PHASE 6: capacity ==="
for arm in off on; do
  echo "### seqmax=$SEQMAX arm=$arm"; grep -E '^\[engine\] ready' $EV/server_1b2_$arm.log
done | tee $EV/kvpack_1b2_mem.txt
export DSV4_KV_PACK=1
: > $EV/server_1b2_big.log
if start_server $EV/server_1b2_big.log "packed-$BIGSEQ" "$BIGSEQ"; then
  grep -E '^\[engine\] ready' $EV/server_1b2_big.log | tee -a $EV/kvpack_1b2_mem.txt
  curl -s -m 300 http://localhost:8080/v1/completions -H 'Content-Type: application/json' \
    -d '{"model":"dsv4","prompt":"The capital of France is","max_tokens":16,"temperature":0}' \
    | tee -a $EV/kvpack_1b2_mem.txt; echo
  server_down
else
  say "packed arm did NOT come up at seqmax $BIGSEQ -- recorded as a negative, not hidden"
  tail -20 $EV/server_1b2_big.log | tee -a $EV/kvpack_1b2_mem.txt
  server_down
fi
unset DSV4_KV_PACK
fi

# ============ PHASE 7 — the ratchet =========================================================
if phase 7; then
say "=== PHASE 7: reports ==="
say "--- BEFORE (FP32 KV rows) ---"
python3 tools/decode_fit_probe.py --outdir "$OFF_OUT" --report | tee $EV/fit_1b2_off.txt
say "--- AFTER (packed KV rows) ---"
python3 tools/decode_fit_probe.py --outdir "$ON_OUT"  --report | tee $EV/fit_1b2_on.txt
say "--- paired, per (target, rep), tau and width reported per leg ---"
python3 tools/mainkv_ab_compare.py "$OFF_OUT" "$ON_OUT" | tee $EV/fit_1b2_paired.txt
say "--- generated ids and tau, both build/decode arms ---"
for f in $EV/decode_1b2_off.log $EV/decode_1b2_on.log; do
  echo "### $f"; grep -E 'KV layout|generated:|tok/verify|WARM decode|LOSSLESS|mem ' "$f" | tail -12
done | tee $EV/fit_1b2_tau.txt
fi
say "done."
