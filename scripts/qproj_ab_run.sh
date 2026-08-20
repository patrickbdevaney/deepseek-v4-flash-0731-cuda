#!/usr/bin/env bash
# qproj_ab_run.sh — DECODE_LADDER item 1.8: the CAUSAL test of the `cattn:q_proj` bimodality.
#
#   nohup setsid bash scripts/qproj_ab_run.sh > evidence/decode_loop/qproj_ab.log 2>&1 </dev/null &
#
# WHY A SCRIPT AND NOT A SEQUENCE OF COMMANDS. Two checkpoint loads that need no steering once
# started, so CLAUDE.md says detach before starting. Named in detach_audit.sh's PATTERNS in the same
# commit, because a stage the audit cannot see reports as "all detached" by never being looked at.
#
# WHAT IS ALREADY ESTABLISHED, FOR FREE, BEFORE ANY GPU TIME (phase 0, tools/qproj_bimodal.py on
# 0.4's own log). `compressed_verify_step_indexer` forks the two `compressor_emit_group` calls onto
# `g_side` and then issues `build_qKV` on the main stream (Finding 55/56, ATTN_SPLIT). The emits
# fire only for groups COMPLETING in the block: `for j in [pos,pos+K): if (j+1)%4==0`. Define
# g = #{ j in [ctx,ctx+VB) : (j+1)%4==0 } from the dprof tag alone. Over 0.4's 153 per-verify
# tables g classifies the bimodality **153/153 with a 2.80x gap and no overlap**, and the excess is
# linear in g (q_proj+compress +6.97 ms at g=1, +14.78 at g=2).
#
# THAT IS A CORRELATION WITH A MECHANISM, NOT YET A CAUSE. Both "the GEMM is intrinsically slower on
# those steps" and "the mark absorbed concurrent side-stream work" predict it. This run separates
# them with the switch the code already carries, NO_ATTN_SPLIT=1, which restores the serial order.
#
# PRE-REGISTERED PREDICTIONS — written before the run so they cannot become post-hoc excuses:
#   P1  SERIAL arm: `q:wq_a` is UNIMODAL near the g=0 value (~1.7 ms). The classifier that separated
#       153/153 in the split arm must separate ~0 of N here.  <- if it stays bimodal the mechanism
#       is REFUTED and 1.8 stays unchecked.
#   P2  SERIAL arm: `cattn:compress` stays bimodal and its g>=1 value RISES to roughly the whole
#       excess (~7 ms at g=1), because the same emits now run inside that mark on the main stream.
#   P3  `cattn:q_proj + cattn:compress` at g=0 is EQUAL in both arms (nothing to overlap), and at
#       g>=1 the SPLIT arm is <= the SERIAL arm (overlap can only help). This is the conservation
#       check: it says the 3.2x swing is a relabelling of real compressor work, not free time.
#   P4  tok/s and tau: SPLIT >= SERIAL (Finding 56 measured K=5 verify 155.22 -> 153.48). If SPLIT
#       is NOT faster at these contexts the overlap has stopped paying, and that is the finding.
#
# ARM ORDER IS THE DRIFT CONTROL. SERIAL runs FIRST, so thermal drift over the run makes the shipped
# SPLIT arm look SLOWER, not faster.
#
# BIT-EXACTNESS. Nothing numeric changes in either arm: NO_ATTN_SPLIT only moves kernels between
# streams, and the fork/join makes the dependency order identical. The generated ids are compared
# between the arms anyway, per invariant 1 — at these contexts via the server, whose EXT_CHUNK=64
# prefill is reproducible (1.9), and tau is reported per leg.
set -u
cd "$(dirname "$0")/.."
export PATH=/usr/local/cuda/bin:$PATH

CKPT="${CKPT:-/home/patrickd/models/DeepSeek-V4-Flash-0731-REAP}"
SEQMAX="${SEQMAX:-16384}"
EV=evidence/decode_loop
SER_OUT=$EV/fit_1p8_serial
SPL_OUT=$EV/fit_1p8_split
TARGETS="${TARGETS:-12288,6144,3072}"
REPS="${REPS:-3}"
PHASE_FROM="${PHASE_FROM:-0}"

say(){ echo "[1.8] $(date -Is) $*"; }

if [ ! -x build/dsv4-server ]; then say "FATAL: build/dsv4-server missing"; exit 1; fi
for f in kernels/compressed_decode.cu kernels/dprof.cu include/dprof.h src/engine.cu kernels/dscratch.cu; do
  if [ "$f" -nt build/dsv4-server ]; then
    say "REFUSING: build/dsv4-server is older than $f. Run scripts/build_server.sh."; exit 1; fi
done
say "binary: server $(date -Is -r build/dsv4-server)"

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
  say "starting server [$label] seqmax=$SEQMAX NO_ATTN_SPLIT=${NO_ATTN_SPLIT:-unset} -> $log"
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

mkdir -p "$SER_OUT" "$SPL_OUT"

# ============ PHASE 0 — the schedule test on data that already exists (no GPU) ================
if [ "$PHASE_FROM" -le 0 ]; then
say "=== PHASE 0: schedule classifier on 0.4's log (no checkpoint, no GPU) ==="
python3 tools/qproj_bimodal.py $EV/server_0p4.log | tee $EV/qproj_1p8_schedule.txt
fi

export DSV4_DPROF=1
export DSV4_DPROF_EVERY="${DSV4_DPROF_EVERY:-4}"

# ============ PHASE 1 — SERIAL arm (NO_ATTN_SPLIT=1), first, so drift penalises SPLIT ========
if [ "$PHASE_FROM" -le 1 ]; then
say "=== PHASE 1: SERIAL arm (NO_ATTN_SPLIT=1) ==="
export NO_ATTN_SPLIT=1
SER_LOG=$EV/server_1p8_serial.log
: > "$SER_LOG"
start_server "$SER_LOG" serial || exit 1
python3 tools/decode_fit_probe.py --outdir "$SER_OUT" --targets "$TARGETS" --reps "$REPS" \
        --max-tokens 128 --no-control --ckpt "$CKPT"
rc=$?; say "serial probe rc=$rc"
server_down
[ "$rc" = "0" ] || { say "FATAL: serial sweep failed"; exit 1; }
unset NO_ATTN_SPLIT
fi

# ============ PHASE 2 — SPLIT arm (shipped default) ==========================================
if [ "$PHASE_FROM" -le 2 ]; then
say "=== PHASE 2: SPLIT arm (shipped default) ==="
unset NO_ATTN_SPLIT
SPL_LOG=$EV/server_1p8_split.log
: > "$SPL_LOG"
start_server "$SPL_LOG" split || exit 1
python3 tools/decode_fit_probe.py --outdir "$SPL_OUT" --targets "$TARGETS" --reps "$REPS" \
        --max-tokens 128 --no-control --ckpt "$CKPT"
rc=$?; say "split probe rc=$rc"
server_down
[ "$rc" = "0" ] || { say "FATAL: split sweep failed"; exit 1; }
fi

# ============ PHASE 3 — the verdict ==========================================================
say "=== PHASE 3: reports ==="
for arm in serial split; do
  say "--- q:wq_a vs the emit schedule, arm=$arm (P1/P2/P3) ---"
  python3 tools/qproj_bimodal.py $EV/server_1p8_$arm.log | tee $EV/qproj_1p8_$arm.txt
  say "--- full attribution, arm=$arm ---"
  python3 tools/dprof_ctx.py $EV/server_1p8_$arm.log | tee $EV/dprof_ctx_1p8_$arm.txt
done
say "--- tok/s and tau, paired per (target, rep)  (P4) ---"
python3 tools/mainkv_ab_compare.py "$SER_OUT" "$SPL_OUT" | tee $EV/fit_1p8_paired.txt
say "--- SERIAL report ---"; python3 tools/decode_fit_probe.py --outdir "$SER_OUT" --report | tee $EV/fit_1p8_serial.txt
say "--- SPLIT report  ---"; python3 tools/decode_fit_probe.py --outdir "$SPL_OUT" --report | tee $EV/fit_1p8_split.txt
say "done."
