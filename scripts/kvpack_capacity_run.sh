#!/usr/bin/env bash
# kvpack_capacity_run.sh — DECODE_LADDER 1b.2, the CAPACITY leg that kvpack_ab_run.sh deliberately
# left out: the FP32 arm at the seqmax the item claims does not fit.
#
#   nohup setsid bash scripts/kvpack_capacity_run.sh > evidence/decode_loop/kvpack_capacity.log 2>&1 </dev/null &
#
# WHY THIS RUNS AT ALL. KV_PRECISION_FINDINGS.md §3 says seqmax 32768 "does not fit today" and
# DECODE_LADDER 1b.2 inherited that as the item's whole justification. The arithmetic says
# otherwise: at seqmax 16384 the engine's own resident set is 112.0/122.8 GiB, and going to 32768
# adds 1.61 GiB of KV plus 0.25 GiB of main_x, i.e. ~113.9 GiB with ~8.9 GiB still free. A premise
# that load-bearing should be MEASURED, not inherited, and this is the load that measures it.
#
# WHY IT IS SAFE AND 65536 IS NOT. 8.9 GiB of predicted headroom is a normal load on this box, and
# memguard is armed. 65536 FP32 would predict ~117.7/122.8 -- 5 GiB -- and this box does not OOM
# gracefully (two whole-machine takedowns, neither with an oom-kill line in dmesg). So 32768 is
# measured and 65536 is left as arithmetic.
set -u
cd "$(dirname "$0")/.."
export PATH=/usr/local/cuda/bin:$PATH
EV=evidence/decode_loop
BIGSEQ="${BIGSEQ:-32768}"
say(){ echo "[1b.2-cap] $(date -Is) $*"; }
server_down(){ pkill -f 'build/dsv4-server --ckpt' 2>/dev/null || true
  for _ in $(seq 1 60); do pgrep -f 'build/dsv4-server --ckpt' >/dev/null || { sleep 5; return 0; }; sleep 5; done; return 1; }
wait_mem(){ for _ in $(seq 1 120); do
    a=$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo); [ "$a" -ge 105 ] && return 0
    say "waiting for page cache: ${a} GiB"; sleep 15; done; return 1; }

for arm in 0 1; do
  wait_mem || exit 1
  LOG=$EV/server_1b2_cap_p$arm.log; : > "$LOG"
  say "=== seqmax=$BIGSEQ DSV4_KV_PACK=$arm ==="
  export DSV4_KV_PACK=$arm
  SEQMAX="$BIGSEQ" LOG="$LOG" bash scripts/run_server.sh || { say "run_server refused"; exit 1; }
  setsid nohup env PAT=dsv4-server bash scripts/memguard.sh > "${LOG%.log}.memguard.log" 2>&1 </dev/null &
  ok=0
  for _ in $(seq 1 480); do
    curl -s -m 10 -o /dev/null http://localhost:8080/health 2>/dev/null && { ok=1; break; }
    pgrep -f 'build/dsv4-server --ckpt' >/dev/null || { say "server died while loading"; break; }
    sleep 5
  done
  if [ "$ok" = 1 ]; then
    say "healthy at seqmax $BIGSEQ pack=$arm"
    grep -E '^\[engine\] ready' "$LOG"
    curl -s -m 300 http://localhost:8080/v1/completions -H 'Content-Type: application/json' \
      -d '{"model":"dsv4","prompt":"The capital of France is","max_tokens":8,"temperature":0}'; echo
  else
    say "DID NOT COME UP at seqmax $BIGSEQ pack=$arm -- recorded, not hidden"; tail -25 "$LOG"
  fi
  server_down
  unset DSV4_KV_PACK
done
say "--- both arms, resident set at seqmax $BIGSEQ ---"
for arm in 0 1; do echo "### pack=$arm"; grep -E '^\[engine\] ready' $EV/server_1b2_cap_p$arm.log || echo "(never became ready)"; done \
  | tee $EV/kvpack_1b2_capacity.txt
say done.
