#!/usr/bin/env bash
# perf_refresh.sh — keep the inference-performance corpus current for as long as work is running.
#
# The battery, the extension pass and every later benchmark all append to the same jsonl files, so
# the corpus is a moving target. Rebuilding it by hand means the analysis is always describing
# whatever the evidence looked like the last time someone remembered — which is how PERF.md would
# end up disagreeing with the data it claims to summarise.
#
# The rebuild is a pure function of the evidence and takes under a second on ~1000 rows, so there is
# nothing to be gained by making it incremental and something to lose: an incremental warehouse
# keeps rows that eval_drop_record.py has since removed.
#
# TOUCHES NO ENGINE. perf_ingest.py reads files; perf_report.py reads SQLite. The only process in
# this system that opens a socket is perf_sample.py, and it calls GET /metrics, which is an snprintf
# over atomics in the HTTP thread pool -- no engine lock, no KV, no kernel. That distinction is the
# whole reason this can run while a benchmark is being scored, and it must not be eroded: nothing
# here may ever call a generation endpoint.
#
#   nohup setsid bash scripts/perf_refresh.sh > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
EVERY="${EVERY:-1800}"
LOG="${LOG:-evidence/perf/refresh.log}"
mkdir -p evidence/perf

while true; do
  {
    echo "=== perf refresh $(date -Is) ==="
    python3 tools/perf_ingest.py 2>&1
    python3 tools/perf_report.py --out PERF.md 2>&1
  } >> "$LOG" 2>&1

  # Stop when there is no longer any source of new requests: no battery, no extension pass. The
  # sampler is left running deliberately -- it is the only record of what the engine did while
  # idle, and an idle engine is still evidence.
  if ! pgrep -f "bash scripts/eval_supervise.sh" > /dev/null \
     && ! pgrep -f "bash scripts/eval_extend_all.sh" > /dev/null; then
    echo "[perf-refresh $(date -Is)] no battery and no extension pass, exiting" >> "$LOG"
    exit 0
  fi
  sleep "$EVERY"
done
