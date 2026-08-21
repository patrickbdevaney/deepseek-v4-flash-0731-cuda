#!/usr/bin/env bash
# hadamard_ab_run.sh -- DECODE_LADDER item 1.10, end to end.
#
#   nohup setsid bash scripts/hadamard_ab_run.sh > evidence/decode_loop/hadamard_ab.log 2>&1 </dev/null &
#
# Named in detach_audit.sh's PATTERNS in the same commit: a stage the audit cannot see reports as
# "all detached" by never being looked at.
#
# THE CHANGE. `hadamard_kernel` gives every thread of a row the whole row to read and one element of
# it to overwrite, which is correct only while `y != x`; three call sites pass the same pointer
# twice. The shipped kernel stages the row in shared memory first. It is BIT-IDENTICAL wherever the
# buffers were already distinct -- same sum order i = 0..D-1, same values -- and it is the first
# thing on this ladder that makes `build/decode`'s prefill reproduce itself above 160 positions.
#
# THE ARM IS AN ENVIRONMENT VARIABLE ON ONE BINARY. `DSV4_HADAMARD_STAGE=0` is the pre-1.10 flat
# kernel; unset is 1.10. Both arms are the same build in the same session, per
# measurement-and-traps §19: a number from a previous iteration is not a valid before-arm.
#
# PRE-REGISTERED, BEFORE THE RUN.
#   * THE CORRECTNESS RESULT IS THE ITEM; the speed result is the ratchet's price of admission and
#     the expected answer is a NULL. In decode, `hadamard` runs at rows = 1 (two emit call sites),
#     rows = nH = 64 and rows = K*nH <= 384, on D = 128 -- a few microseconds of a ~130 ms forward.
#     A band covering zero is written into the ladder as a null and is NOT rounded into a win.
#   * THE ARMS ARE NOT BIT-COMPARABLE IN PREFILL AND THAT IS THE POINT. The OFF arm's prefill is
#     nondeterministic above 160 positions (1.9), so "the two arms emitted different text at ctx
#     12,288" is the DEFECT, not a failure of this change. The bit-exactness claim is made where it
#     is checkable: the no-checkpoint gates, and `build/decode` at a 6-id prompt (below 1.9's
#     threshold, so token ids are a valid instrument there -- see 1.9 point 8).
#   * tau IS REPORTED FOR BOTH ARMS AT EVERY LEG. A byte-identical sequence can still collapse
#     acceptance, and here the arms are not byte-identical by construction, so tau is the only
#     thing that says the change did not buy speed by drafting worse.
#
# PHASE ORDER IS THE DRIFT CONTROL. The CONTROL arm (DSV4_HADAMARD_STAGE=0, the pre-1.10 kernel)
# runs BEFORE the staged arm in every pair, so thermal drift over the run makes 1.10 look SLOWER.
set -u
cd "$(dirname "$0")/.."
export PATH=/usr/local/cuda/bin:$PATH

CKPT="${CKPT:-/home/patrickd/models/DeepSeek-V4-Flash-0731-REAP}"
SEQMAX="${SEQMAX:-16384}"
EV=evidence/decode_loop
OFF_OUT=$EV/fit_1p10_off         # control: the pre-1.10 flat kernel
ON_OUT=$EV/fit_1p10_on           # 1.10:    the shared-memory staged kernel
OFF2_OUT=$EV/fit_1p10_off2       # the same two arms AGAIN with the arm order REVERSED, so drift
ON2_OUT=$EV/fit_1p10_on2         # enters the second pair with the opposite sign
TARGETS="${TARGETS:-12288,6144,3072}"
REPS="${REPS:-6}"
PHASE_FROM="${PHASE_FROM:-0}"
PHASE_TO="${PHASE_TO:-9}"

say(){ echo "[1.10] $(date -Is) $*"; }
phase(){ [ "$PHASE_FROM" -le "$1" ] && [ "$PHASE_TO" -ge "$1" ]; }

for f in kernels/indexer.cu include/indexer.h; do
  [ -e "$f" ] || { say "REFUSING: $f missing"; exit 1; }
  [ "$f" -nt build/decode ] && { say "REFUSING: build/decode older than $f"; exit 1; }
done
say "binaries: decode $(date -Is -r build/decode), server $(date -Is -r build/dsv4-server 2>/dev/null || echo none)"

server_down(){
  pkill -f 'build/dsv4-server --ckpt' 2>/dev/null || true
  for _ in $(seq 1 60); do
    pgrep -x dsv4-server > /dev/null || { say "server down"; sleep 5; return 0; }
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
  say "starting server [$label] seqmax=$SEQMAX DSV4_HADAMARD_STAGE=${DSV4_HADAMARD_STAGE:-unset} -> $log"
  SEQMAX="$SEQMAX" LOG="$log" bash scripts/run_server.sh || { say "run_server refused"; return 1; }
  setsid nohup env PAT=dsv4-server bash scripts/memguard.sh \
      > "${log%.log}.memguard.log" 2>&1 < /dev/null &
  for _ in $(seq 1 480); do
    curl -s -m 10 -o /dev/null http://localhost:8080/health 2>/dev/null && {
      say "healthy [$label]: $(curl -s -m 10 http://localhost:8080/health)"; return 0; }
    pgrep -x dsv4-server > /dev/null || {
      say "FATAL: server process is gone while we waited. Tail of $log:"; tail -40 "$log"; return 1; }
    sleep 5
  done
  say "FATAL: server never became healthy"; tail -40 "$log"; return 1
}
wait_decode(){ for _ in $(seq 1 720); do pgrep -x decode > /dev/null || return 0; sleep 10; done
               say "FATAL: build/decode never exited"; return 1; }

mkdir -p "$OFF_OUT" "$ON_OUT" "$OFF2_OUT" "$ON2_OUT"

# ============ PHASE 0 -- the no-checkpoint gates, BOTH ARMS ===================================
if phase 0; then
say "=== PHASE 0: in-situ gates, BOTH arms (no checkpoint) ==="
rc0=0
# 0a. THE TARGETED GATE. Runs both arms itself via hadamard_set_stage(), so it must NOT run inside
# the arm loop below. It returns non-zero if the OFF arm stops reproducing the defect, which is what
# stops a green PASS from meaning "the harness went blind".
env -u DSV4_HADAMARD_STAGE ./build/gate_hadamard_alias --repeats 200 > $EV/gate_hadamard_alias_1p10.log 2>&1 || rc0=1
# --control corrupts the reference by one ulp and asks only whether the memcmp can see it. It is a
# separate question with its own exit code: 0 = every comparison differed = the memcmp is live.
env -u DSV4_HADAMARD_STAGE ./build/gate_hadamard_alias --repeats 4 --control >> $EV/gate_hadamard_alias_1p10.log 2>&1
crc=$?; [ "$crc" = "0" ] || { say "FATAL: gate_hadamard_alias --control returned $crc; the memcmp is not live."; exit 1; }
say "gate_hadamard_alias --control PASS (the memcmp sees a one-ulp change)"
cat $EV/gate_hadamard_alias_1p10.log
[ "$rc0" = "0" ] || { say "FATAL: gate_hadamard_alias failed. 1.10 is not correct; nothing else runs."; exit 1; }

# 0b. THE ARM-INVARIANT GATES. gate_units carries the hadamard GOLDEN, so it is the one that says
# the staged kernel still computes the transform; the rest say 1.10 broke nothing else.
for arm in stage flat; do
  if [ "$arm" = flat ]; then export DSV4_HADAMARD_STAGE=0; else unset DSV4_HADAMARD_STAGE; fi
  for g in gate_units gate_index_score gate_indexer_decode gate_compressed_decode \
           gate_compressed_graph gate_indexer_graph gate_compressor_emit gate_prefill_len \
           gate_idx_pack gate_scratch_init; do
    [ -x build/$g ] || { say "SKIP $g (no binary)"; continue; }
    if [ "$g" = gate_units ]; then ./build/$g ref/goldens > $EV/${g}_1p10_$arm.log 2>&1
    else                           ./build/$g            > $EV/${g}_1p10_$arm.log 2>&1; fi
    r=$?; say "arm=$arm $g rc=$r"; grep -E 'FAIL|PASS' $EV/${g}_1p10_$arm.log | tail -3
    [ "$r" = "0" ] || rc0=1
  done
done
unset DSV4_HADAMARD_STAGE
[ "$rc0" = "0" ] || { say "FATAL: an in-situ gate failed. STOPPING before spending any loads."; exit 1; }
fi

# ============ PHASE 1 -- build/decode, CONTROL arm (pre-1.10 flat kernel) =====================
# The gate prompt prefills 6 ids, far below the 160-position threshold 1.9 measured, so the
# token-id comparison IS a valid gate here -- which is the whole reason it is run at this length.
if phase 1; then
say "=== PHASE 1: build/decode, CONTROL (flat kernel, DSV4_HADAMARD_STAGE=0) ==="
wait_mem || exit 1
DSV4_HADAMARD_STAGE=0 DSV4_DPROF=1 \
  bash scripts/run_model.sh $EV/decode_1p10_off.log ./build/decode "$CKPT" "0,671,6102,294,8760,344" 8 "" 64
wait_decode || exit 1
[ -s "$EV/decode_1p10_off.log" ] || { say "FATAL: control decode produced no log."; exit 1; }
grep -E 'GATE|tok/verify|tok/s|LOSSLESS|generated:|mem ' $EV/decode_1p10_off.log | tail -30
fi

# ============ PHASE 2 -- build/decode, STAGED arm ============================================
if phase 2; then
say "=== PHASE 2: build/decode, STAGED (1.10) ==="
wait_mem || exit 1
DSV4_DPROF=1 \
  bash scripts/run_model.sh $EV/decode_1p10_on.log ./build/decode "$CKPT" "0,671,6102,294,8760,344" 8 "" 64
wait_decode || exit 1
[ -s "$EV/decode_1p10_on.log" ] || { say "FATAL: staged decode produced no log."; exit 1; }
grep -E 'GATE|tok/verify|tok/s|LOSSLESS|generated:|mem ' $EV/decode_1p10_on.log | tail -30
say "--- generated ids, both arms at a 6-id prompt (must be IDENTICAL) ---"
a=$(grep -m1 'generated:' $EV/decode_1p10_off.log); b=$(grep -m1 'generated:' $EV/decode_1p10_on.log)
echo "off: $a"; echo " on: $b"
[ "$a" = "$b" ] && say "TOKEN-ID GATE: PASS (identical)" || say "TOKEN-ID GATE: FAIL (arms diverged)"
fi

# ============ PHASE 3 -- tok/s sweep, CONTROL arm ============================================
if phase 3; then
say "=== PHASE 3: tok/s sweep, CONTROL (flat kernel) ==="
export DSV4_HADAMARD_STAGE=0
: > $EV/server_1p10_off.log
start_server $EV/server_1p10_off.log off || exit 1
python3 tools/decode_fit_probe.py --outdir "$OFF_OUT" --targets "$TARGETS" --reps "$REPS" --ckpt "$CKPT"
rc=$?; say "control probe rc=$rc"; server_down
[ "$rc" = "0" ] || { say "FATAL: control sweep failed"; exit 1; }
unset DSV4_HADAMARD_STAGE
fi

# ============ PHASE 4 -- tok/s sweep, STAGED arm =============================================
if phase 4; then
say "=== PHASE 4: tok/s sweep, STAGED (1.10) ==="
unset DSV4_HADAMARD_STAGE
: > $EV/server_1p10_on.log
start_server $EV/server_1p10_on.log on || exit 1
python3 tools/decode_fit_probe.py --outdir "$ON_OUT" --targets "$TARGETS" --reps "$REPS" --ckpt "$CKPT"
rc=$?; say "staged probe rc=$rc"; server_down
[ "$rc" = "0" ] || { say "FATAL: staged sweep failed"; exit 1; }
fi

# ============ PHASE 5 -- the SAME PAIR AGAIN, ARM ORDER REVERSED =============================
# 1.11's headline was that one arm order called a real effect a null: pairing removes between-LEG
# variance and does NOTHING to a between-LOAD offset, which traps §19 measures at 0.6 % = ~0.8
# ms/forward -- larger than anything this item can plausibly move. So the pair is run twice with the
# order reversed and the two paired means averaged: drift enters with opposite sign and cancels.
if phase 5; then
say "=== PHASE 5: tok/s sweep, STAGED FIRST (reversed pair) ==="
unset DSV4_HADAMARD_STAGE
: > $EV/server_1p10_on2.log
start_server $EV/server_1p10_on2.log on2 || exit 1
python3 tools/decode_fit_probe.py --outdir "$ON2_OUT" --targets "$TARGETS" --reps "$REPS" --ckpt "$CKPT"
rc=$?; say "staged(2) probe rc=$rc"; server_down
[ "$rc" = "0" ] || { say "FATAL: staged(2) sweep failed"; exit 1; }
fi

if phase 6; then
say "=== PHASE 6: tok/s sweep, CONTROL SECOND (reversed pair) ==="
export DSV4_HADAMARD_STAGE=0
: > $EV/server_1p10_off2.log
start_server $EV/server_1p10_off2.log off2 || exit 1
python3 tools/decode_fit_probe.py --outdir "$OFF2_OUT" --targets "$TARGETS" --reps "$REPS" --ckpt "$CKPT"
rc=$?; say "control(2) probe rc=$rc"; server_down
unset DSV4_HADAMARD_STAGE
[ "$rc" = "0" ] || { say "FATAL: control(2) sweep failed"; exit 1; }
fi

# ============ PHASE 7 -- the paired band, and tau ============================================
if phase 7; then
say "=== PHASE 7: reports ==="
say "--- pair 1 (control first, so drift favours the control) ---"
python3 tools/paired_band.py "$OFF_OUT" "$ON_OUT" --label-before flat --label-after staged \
  2>&1 | tee $EV/pair_1p10_a.txt
say "--- pair 2 (staged first, so drift favours the flat kernel) ---"
python3 tools/paired_band.py "$OFF2_OUT" "$ON2_OUT" --label-before flat --label-after staged \
  2>&1 | tee $EV/pair_1p10_b.txt
say "--- drift-free estimate: the two paired means averaged ---"
python3 tools/paired_band.py "$OFF_OUT" "$ON_OUT" --reversed "$OFF2_OUT" "$ON2_OUT" \
  --label-before flat --label-after staged 2>&1 | tee $EV/pair_1p10.txt
say "--- tau, both arms, every leg ---"
for d in "$OFF_OUT" "$ON_OUT" "$OFF2_OUT" "$ON2_OUT"; do
  echo "### $d"
  python3 - "$d" <<'PYEOF'
import json,os,sys,statistics as st
d=sys.argv[1]
for name in ('postfix.sweep.jsonl','control.fresh.jsonl'):
    p=os.path.join(d,name)
    if not os.path.exists(p): continue
    rows=[json.loads(l) for l in open(p) if l.strip()]
    for r in rows:
        print(f"  {name:22s} leg={r.get('leg','?'):24s} ctx={r.get('ctx','?'):>7} "
              f"tau={r.get('tau','?')} tok/s={r.get('tok_s','?')}")
    taus=[r['tau'] for r in rows if isinstance(r.get('tau'),(int,float))]
    if taus: print(f"  {name:22s} tau mean {st.mean(taus):.4f} over {len(taus)} legs")
PYEOF
done | tee $EV/tau_1p10.txt
fi
say "done."
