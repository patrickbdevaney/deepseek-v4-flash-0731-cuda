#!/usr/bin/env bash
# f32mkn_ab_run.sh — DECODE_LADDER item 1.12, end to end.
#
#   nohup setsid bash scripts/f32mkn_ab_run.sh > evidence/decode_loop/f32mkn_ab.log 2>&1 </dev/null &
#
# Named in detach_audit.sh's PATTERNS in the same commit: a stage the audit cannot see reports as
# "all detached" by never being looked at.
#
# THE CHANGE. `gemm_fp32`'s M=K path was one warp per (8-row chunk, n), launched once per chunk from
# the HOST. `compressor_emit_group` calls it at M = ratio = 128, N = d = 512, K = DIM = 4096 on the
# twenty ratio-128 layers, so B = [512,4096] f32 = 8.39 MB was walked SIXTEEN times per GEMM, twice
# per layer, in sixteen separate launches. 1.12 gives each warp a (4x4) tile of the output and moves
# the chunk loop into gridDim.y: operand traffic goes from M*N*K*(1/1 + 1/8) to M*N*K*(1/4 + 1/4)
# floats -- 3x less -- and 16 launches become 1.
#
# ALREADY MEASURED, WITHOUT A CHECKPOINT (tools/f32mk_bench.cu, evidence/decode_loop/f32mk_bench_1p12.txt):
#   strided emit  [128,512]x[512,4096]   0.5744 -> 0.2477 ms   2.32x   (x40 per emit step: 22.98 -> 9.91 ms)
#   indexer main  [8,1024]x[1024,4096]   0.0729 -> 0.0407 ms   1.79x   (x42:                 3.06 -> 1.71 ms)
#   indexer idx   [8,256] x[256,4096]    0.0331 -> 0.0177 ms   1.87x   (x42:                 1.39 -> 0.74 ms)
#   indexer iw    [5,64]  x[64,4096]     0.0143 -> 0.0144 ms   1.00x   -- below M=8 the tile is OFF
# and every one of the 18 dispatchable tiles is max|diff| == 0 against the M=1 path, at 14 shapes
# including the N and M tails (gate_bf16w).
#
# THE ARM IS AN ENVIRONMENT VARIABLE ON ONE BINARY. `DSV4_F32MK_TILE=8x0` selects the EXACT pre-1.12
# code -- the host-side chunk loop calling gemm_fp32_mk_kernel<8>, not a re-expression of it -- and
# unset is 1.12's 4x4. Same build, same session, because per measurement-and-traps §19 a number from
# a previous iteration is not a valid before-arm.
#
# PRE-REGISTERED, BEFORE THE RUN, SO NEITHER OUTCOME CAN BE RATIONALISED AFTERWARDS.
#   * THE HEADLINE IS THE SPIKE, NOT tok/s, AND THE LADDER SAYS SO. A ratio-128 boundary falls in a
#     verify block on ~VB/128 of steps, ~4 % at VB=5. 13 ms off 4 % of steps is ~0.5 ms/forward
#     against a 3.5 % run-to-run spread on a ~150 ms forward. THE PAIRED ms/forward BAND IS EXPECTED
#     TO COVER ZERO and that is not a failure of the item -- it is the item's own ranking. What must
#     move, and by a margin no noise can produce, is `cattn:compress` CONDITIONAL ON g >= 1:
#     predicted 43-50 -> 30-37 ms (tools/emit_spike.py). If THAT does not move, the change did not
#     reach the engine and the item is a negative result.
#   * TERM A ONLY. An emit reads `x_full` for one group of `ratio` tokens and two weight matrices;
#     not one byte of it scales with context. If the paired split puts the saving in the CONTEXT
#     term, the mechanism claimed here is wrong and the item says so.
#   * BIT-EXACT IS THE CONSTRUCTION, NOT A HOPE. Every output is still produced by ONE warp with
#     lane l accumulating k4 indices l, l+32, ... into the same four chains and the same shuffle
#     tree; widening the tile changes which warp owns an output, not the order of any dot product.
#     Acceptance is memcmp on generated token ids plus identical completion hashes on every leg.
#     A tolerance would be the wrong gate for the same reason it was in 1b.1.
#   * DSV4_DPROF_EVERY=1, NOT 4. The spike population is ~4 % of steps; at EVERY=4 a 128-token leg
#     yields ONE candidate table and might yield none. 1.8's "8 of 8" was 8 samples in total. The
#     instrument tax is paid in BOTH arms and the paired band is a relative comparison, so the tax
#     cancels -- but the absolute ms/forward printed here is an instrumented number and is NOT the
#     shipped throughput.
#
# PHASE ORDER IS THE DRIFT CONTROL. The CONTROL arm (DSV4_F32MK_TILE=8x0) runs BEFORE the 1.12 arm
# in every pair, so thermal drift over the run makes 1.12 look SLOWER, not faster.
set -u
cd "$(dirname "$0")/.."
export PATH=/usr/local/cuda/bin:$PATH

CKPT="${CKPT:-/home/patrickd/models/DeepSeek-V4-Flash-0731-REAP}"
SEQMAX="${SEQMAX:-16384}"
EV=evidence/decode_loop
OFF_OUT=$EV/fit_1p12_off         # control: the pre-1.12 host-side 8-row chunk loop
ON_OUT=$EV/fit_1p12_on           # 1.12:    the (4,4) warp tile
OFF2_OUT=$EV/fit_1p12_off2       # the same two arms AGAIN with the arm order REVERSED
ON2_OUT=$EV/fit_1p12_on2
TARGETS="${TARGETS:-12288,6144,3072}"
REPS="${REPS:-6}"
# 128, NOT 256. Ladder 1.9 measured `build/decode` prefill byte-identical to 160 positions and
# RACING at 192+, and the first run of this script set 256 and got "0/18 legs byte-identical" out of
# two loads of THE SAME BINARY -- the text-identity gate was measuring 1.9's race, not this change.
# 128 completion tokens still crosses exactly one ratio-128 boundary per leg at every target here
# (12282 -> 12410 contains 12288), so the spike population is ~18 per arm, and the conditional mark
# reproduced to 0.03 ms across the null-control pair.
MAXTOK="${MAXTOK:-128}"
PHASE_FROM="${PHASE_FROM:-0}"
PHASE_TO="${PHASE_TO:-9}"
PRE=8x0                          # the exact before-arm tile spec

say(){ echo "[1.12] $(date -Is) $*"; }
phase(){ [ "$PHASE_FROM" -le "$1" ] && [ "$PHASE_TO" -ge "$1" ]; }

for f in kernels/compressor.cu kernels/compressed_decode.cu src/decode.cu src/engine.cu; do
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
  say "starting server [$label] seqmax=$SEQMAX DSV4_F32MK_TILE=${DSV4_F32MK_TILE:-unset} -> $log"
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
if phase 0; then
say "=== PHASE 0: in-situ gates, BOTH arms (no checkpoint) ==="
rc0=0
# 0a. THE TARGETED GATE. gate_bf16w sweeps all 18 dispatchable (MM,NN) tiles INCLUDING the pre-1.12
# 8x0 arm, at 7 values of M and 2 of N, against the warp-per-output-element path -- 252 comparisons,
# each of which must be max|diff| == 0. The N=254 column exists because a guarded store is exactly
# where a widened tile gets the tail wrong, and it costs nothing to run.
./build/gate_bf16w > $EV/gate_bf16w_1p12.log 2>&1 || rc0=1
grep -E 'gemm_fp32 M=|Gate BF16W' $EV/gate_bf16w_1p12.log
# 0b. THE BENCH, which is also a gate: it re-checks every tile against the M=1 path at the ENGINE'S
# shapes and returns non-zero if any differs.
./build/f32mk_bench 200 > $EV/f32mk_bench_1p12.txt 2>&1 || rc0=1
grep -E '^==|8x0|  4x4 |f32mk_bench:' $EV/f32mk_bench_1p12.txt
# 0c. THE ARM-INVARIANT GATES, in BOTH arms.
for arm in tile pre; do
  if [ "$arm" = pre ]; then export DSV4_F32MK_TILE=$PRE; else unset DSV4_F32MK_TILE; fi
  for g in gate_units gate_indexer_decode gate_compressed_decode gate_compressed_graph \
           gate_indexer_graph gate_compressor_emit gate_index_score gate_prefill_len \
           gate_kv_pack_e2e gate_join_defer gate_topk_smem_ctx; do
    [ -x build/$g ] || { say "SKIP $g (no binary)"; continue; }
    if [ "$g" = gate_units ]; then ./build/$g ref/goldens > $EV/${g}_1p12_$arm.log 2>&1
    else                           ./build/$g            > $EV/${g}_1p12_$arm.log 2>&1; fi
    r=$?; say "arm=$arm $g rc=$r"; grep -E 'FAIL|PASS' $EV/${g}_1p12_$arm.log | tail -3
    [ "$r" = "0" ] || rc0=1
  done
done
unset DSV4_F32MK_TILE
[ "$rc0" = "0" ] || { say "FATAL: an in-situ gate failed. STOPPING before spending any loads."; exit 1; }
fi

# ============ PHASE 1 — build/decode, CONTROL arm ============================================
if phase 1; then
say "=== PHASE 1: build/decode, CONTROL (pre-1.12, DSV4_F32MK_TILE=$PRE) ==="
wait_mem || exit 1
DSV4_F32MK_TILE=$PRE DSV4_KSWEEP=1 DSV4_DPROF=1 \
  bash scripts/run_model.sh $EV/decode_1p12_off.log ./build/decode "$CKPT" "0,671,6102,294,8760,344" 8 "" 64
wait_decode || exit 1
[ -s "$EV/decode_1p12_off.log" ] || { say "FATAL: control decode produced no log."; exit 1; }
grep -E 'GATE|tok/verify|tok/s|LOSSLESS|generated:|mem ' $EV/decode_1p12_off.log | tail -30
fi

# ============ PHASE 2 — build/decode, 1.12 arm ===============================================
if phase 2; then
say "=== PHASE 2: build/decode, 1.12 (4x4 warp tile) ==="
wait_mem || exit 1
DSV4_KSWEEP=1 DSV4_DPROF=1 \
  bash scripts/run_model.sh $EV/decode_1p12_on.log ./build/decode "$CKPT" "0,671,6102,294,8760,344" 8 "" 64
wait_decode || exit 1
[ -s "$EV/decode_1p12_on.log" ] || { say "FATAL: 1.12 decode produced no log."; exit 1; }
grep -E 'GATE|tok/verify|tok/s|LOSSLESS|generated:|mem ' $EV/decode_1p12_on.log | tail -30
say "--- generated ids, both arms (must be IDENTICAL: 1.12 is bit-exact by construction) ---"
a=$(grep -m1 'generated:' $EV/decode_1p12_off.log); b=$(grep -m1 'generated:' $EV/decode_1p12_on.log)
echo "off: $a"; echo " on: $b"
[ "$a" = "$b" ] && say "TOKEN-ID GATE: PASS (identical)" || { say "TOKEN-ID GATE: FAIL (arms diverged)"; exit 1; }
fi

export DSV4_DPROF=1
export DSV4_DPROF_EVERY="${DSV4_DPROF_EVERY:-1}"

# ============ PHASE 3 — server sweep, CONTROL arm ============================================
if phase 3; then
say "=== PHASE 3: server sweep, CONTROL (pre-1.12) ==="
rm -f $OFF_OUT/*.jsonl
export DSV4_F32MK_TILE=$PRE
: > $EV/server_1p12_off.log
start_server $EV/server_1p12_off.log off || exit 1
python3 tools/decode_fit_probe.py --outdir "$OFF_OUT" --targets "$TARGETS" --reps "$REPS" \
        --max-tokens "$MAXTOK" --no-control --ckpt "$CKPT"
rc=$?; say "control probe rc=$rc"; server_down
[ "$rc" = "0" ] || { say "FATAL: control sweep failed"; exit 1; }
unset DSV4_F32MK_TILE
fi

# ============ PHASE 4 — server sweep, 1.12 arm ===============================================
if phase 4; then
say "=== PHASE 4: server sweep, 1.12 ==="
rm -f $ON_OUT/*.jsonl
unset DSV4_F32MK_TILE
: > $EV/server_1p12_on.log
start_server $EV/server_1p12_on.log on || exit 1
python3 tools/decode_fit_probe.py --outdir "$ON_OUT" --targets "$TARGETS" --reps "$REPS" \
        --max-tokens "$MAXTOK" --no-control --ckpt "$CKPT"
rc=$?; say "1.12 probe rc=$rc"; server_down
[ "$rc" = "0" ] || { say "FATAL: 1.12 sweep failed"; exit 1; }
fi

# ============ PHASE 5 — the verdict ==========================================================
if phase 5; then
say "=== PHASE 5: reports ==="
# THE ARM ASSERTION, FIRST, BECAUSE THE FIRST RUN OF THIS SCRIPT DID NOT HAVE IT. `DSV4_F32MK_TILE=8x0`
# was silently rejected by a `nn > 0` guard in f32mk_tile_from_env(), both arms ran 4x4, and the
# result was a flawless null that looked exactly like a real negative. The engine now prints the
# resolved tile on every start; if the two arms do not disagree there is nothing to report.
aoff=$(grep -m1 '\[f32mk\] tile' $EV/server_1p12_off.log || true)
aon=$( grep -m1 '\[f32mk\] tile' $EV/server_1p12_on.log  || true)
say "control arm resolved: ${aoff:-MISSING}"
say "   1.12 arm resolved: ${aon:-MISSING}"
if [ -z "$aoff" ] || [ -z "$aon" ] || [ "$aoff" = "$aon" ]; then
  say "FATAL: the two arms resolved to the SAME tile (or did not report one). This A/B is comparing"
  say "       a build against itself and any number it prints is the run-to-run spread. STOPPING."
  exit 1
fi
say "--- THE HEADLINE: cattn:compress conditional on a ratio-128 boundary in the block ---"
python3 tools/emit_spike.py --before $EV/server_1p12_off.log --after $EV/server_1p12_on.log \
        --ratio 128 | tee $EV/emit_spike_1p12.txt
say "--- the same classifier at ratio 4, where the emits run on the side stream and are hidden ---"
python3 tools/emit_spike.py --before $EV/server_1p12_off.log --after $EV/server_1p12_on.log \
        --ratio 4 | tee $EV/emit_spike_1p12_r4.txt
say "--- full attribution, both arms ---"
for arm in off on; do
  python3 tools/dprof_ctx.py $EV/server_1p12_$arm.log | tee $EV/dprof_ctx_1p12_$arm.txt
done
say "--- BEFORE (pre-1.12) ---"
python3 tools/decode_fit_probe.py --outdir "$OFF_OUT" --report | tee $EV/fit_1p12_off.txt
say "--- AFTER (1.12) ---"
python3 tools/decode_fit_probe.py --outdir "$ON_OUT"  --report | tee $EV/fit_1p12_on.txt
say "--- paired, per (target, rep), tau and completion hash per leg ---"
python3 tools/mainkv_ab_compare.py "$OFF_OUT" "$ON_OUT" | tee $EV/fit_1p12_paired.txt
say "--- the paired band, flat / per-1000-context split ---"
python3 tools/paired_band.py "$OFF_OUT" "$ON_OUT" --label-before pre1p12 --label-after tile4x4 \
  | tee $EV/fit_1p12_band.txt
fi

# ============ PHASES 6-7 — THE REVERSED-ORDER REPLICATE ======================================
# Only worth the two extra loads if phase 5's paired band is close to resolving: pairing removes
# between-LEG variance and does NOTHING to a between-LOAD offset, which traps §19 measures at 0.6 %
# = ~0.8 ms/forward -- the size of this item's whole pre-registered ceiling. Run with PHASE_FROM=6.
if phase 6; then
say "=== PHASE 6: REVERSED replicate, 1.12 arm FIRST ==="
rm -f $ON2_OUT/*.jsonl
unset DSV4_F32MK_TILE
: > $EV/server_1p12_on2.log
start_server $EV/server_1p12_on2.log on2 || exit 1
python3 tools/decode_fit_probe.py --outdir "$ON2_OUT" --targets "$TARGETS" --reps "$REPS" \
        --max-tokens "$MAXTOK" --no-control --ckpt "$CKPT"
rc=$?; say "reversed 1.12 probe rc=$rc"; server_down
[ "$rc" = "0" ] || { say "FATAL: reversed 1.12 sweep failed"; exit 1; }
fi
if phase 7; then
say "=== PHASE 7: REVERSED replicate, CONTROL arm SECOND ==="
rm -f $OFF2_OUT/*.jsonl
export DSV4_F32MK_TILE=$PRE
: > $EV/server_1p12_off2.log
start_server $EV/server_1p12_off2.log off2 || exit 1
python3 tools/decode_fit_probe.py --outdir "$OFF2_OUT" --targets "$TARGETS" --reps "$REPS" \
        --max-tokens "$MAXTOK" --no-control --ckpt "$CKPT"
rc=$?; say "reversed control probe rc=$rc"; server_down
[ "$rc" = "0" ] || { say "FATAL: reversed control sweep failed"; exit 1; }
unset DSV4_F32MK_TILE
fi
if phase 8; then
say "=== PHASE 8: the drift-free estimate ==="
python3 tools/paired_band.py "$OFF_OUT" "$ON_OUT" --reversed "$OFF2_OUT" "$ON2_OUT" \
  --label-before pre1p12 --label-after tile4x4 | tee $EV/fit_1p12_band_pooled.txt
python3 tools/emit_spike.py --before $EV/server_1p12_off.log $EV/server_1p12_off2.log \
        --after $EV/server_1p12_on.log $EV/server_1p12_on2.log --ratio 128 \
  | tee $EV/emit_spike_1p12_pooled.txt
fi
say "done."
