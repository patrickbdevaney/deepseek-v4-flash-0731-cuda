#!/usr/bin/env bash
# dprof_ctx_run.sh — ladder item 0.4: one server load, DSV4_DPROF=1, four context depths,
# and a per-verify attribution table for every one of them.
#
# WHY A SCRIPT AND NOT A SEQUENCE OF COMMANDS. This is ~35 minutes of load-plus-sweep that needs no
# steering, so CLAUDE.md says it must be detached before it starts, not babysat by the session that
# happened to launch it. It also has to run three things in a fixed order with a health gate between
# them; a half-run that starts the probe against a still-loading engine writes nothing and wastes the
# whole load.
#
#   nohup setsid bash scripts/dprof_ctx_run.sh > evidence/decode_loop/dprof_ctx_run.log 2>&1 </dev/null &
#
# It leaves the server UP on purpose. Killing it would make the tables unrepeatable, and the caller
# may want a second sweep against the same resident weights; stop it with
# `pkill -f 'build/dsv4-server --ckpt'` when done.
set -u
cd "$(dirname "$0")/.."

LOG=evidence/decode_loop/server_0p4.log
OUT=evidence/decode_loop/dprof
SEQMAX="${SEQMAX:-16384}"
TARGETS="${TARGETS:-12288,6144,3072,768}"
REPS="${REPS:-2}"
MAXTOK="${MAXTOK:-128}"
# One table every EVERY-th verify. 1 would emit ~75 tables per leg (~3,400 lines); 4 still gives
# ~19 samples per leg, which is far more than the two points 0.2 tried to fit a slope through.
export DSV4_DPROF=1
export DSV4_DPROF_EVERY="${DSV4_DPROF_EVERY:-4}"

say(){ echo "[0.4] $(date -Is) $*"; }

# ---- 1. the binary must be the one we are about to reason about ------------------------------
if [ ! -x build/dsv4-server ]; then say "build/dsv4-server missing"; exit 1; fi
if [ src/engine.cu -nt build/dsv4-server ] || [ kernels/dprof.cu -nt build/dsv4-server ] \
   || [ include/dprof.h -nt build/dsv4-server ]; then
  say "REFUSING: build/dsv4-server is older than the sources it must describe. Run scripts/build_server.sh."
  exit 1
fi
say "binary $(date -Is -r build/dsv4-server), sources ok"

# ---- 2. server, detached, guarded ------------------------------------------------------------
if curl -s -m 10 -o /dev/null http://localhost:8080/health 2>/dev/null; then
  say "a server is already healthy on :8080 — reusing it (NOT starting a second: see with_model_lock.sh)"
else
  say "starting server seqmax=$SEQMAX with DSV4_DPROF=1 DSV4_DPROF_EVERY=$DSV4_DPROF_EVERY"
  SEQMAX="$SEQMAX" LOG="$LOG" bash scripts/run_server.sh || { say "run_server refused"; exit 1; }
  # run_server.sh detaches the server but arms no memory guard, and this box does not OOM
  # gracefully -- seqmax 16384 sat at 120.0/122.8 GiB on the 0.3 run, i.e. 2.8 GiB of headroom.
  setsid nohup env PAT=dsv4-server bash scripts/memguard.sh \
      > evidence/decode_loop/memguard_0p4.log 2>&1 < /dev/null &
  say "memguard armed -> evidence/decode_loop/memguard_0p4.log"
fi

say "waiting for health (up to 30 min: ~101 GiB to load)"
ok=0
for _ in $(seq 1 360); do
  if curl -s -m 10 -o /dev/null http://localhost:8080/health 2>/dev/null; then ok=1; break; fi
  # A DEAD LOADER MUST NOT LOOK LIKE A SLOW ONE. CLAUDE.md: a stage that "completes" against a dead
  # engine is worse than one that dies.
  if ! pgrep -f 'build/dsv4-server --ckpt' > /dev/null; then
    say "FATAL: server process is gone while we waited. Tail of $LOG:"; tail -30 "$LOG"; exit 1
  fi
  sleep 5
done
[ "$ok" = "1" ] || { say "FATAL: server never became healthy"; tail -30 "$LOG"; exit 1; }
say "healthy: $(curl -s -m 10 http://localhost:8080/health)"

# ---- 3. the sweep ----------------------------------------------------------------------------
mkdir -p "$OUT"
say "probe targets=$TARGETS reps=$REPS max_tokens=$MAXTOK -> $OUT"
python3 tools/decode_fit_probe.py --outdir "$OUT" --targets "$TARGETS" --reps "$REPS" \
        --max-tokens "$MAXTOK" --no-control
rc=$?
say "probe rc=$rc"
[ "$rc" = "0" ] || exit "$rc"

# ---- 4. the attribution ----------------------------------------------------------------------
say "attribution:"
python3 tools/dprof_ctx.py "$LOG" | tee evidence/decode_loop/dprof_ctx_0p4.txt
say "done. server left up; stop with: pkill -f 'build/dsv4-server --ckpt'"
