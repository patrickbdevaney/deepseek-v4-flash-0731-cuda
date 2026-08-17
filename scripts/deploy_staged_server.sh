#!/usr/bin/env bash
# deploy_staged_server.sh — swap in build/dsv4-server.staged during a window where nothing is running.
#
# WHY STAGED AND NOT BUILT IN PLACE. `scripts/build_server.sh` links straight to
# `build/dsv4-server`, which is the path the live server is executing from. Linking over a running
# binary either fails with ETXTBSY or, if the linker unlinks first, leaves a half-written file where
# the restart script expects an executable. Neither is acceptable while a six-day battery is
# scoring, so the new binary is built to `.staged` and swapped here, deliberately, in a window.
#
# THE WINDOW IS NOT OPTIONAL. Bringing the engine down means a ~10-15 minute reload of a 101 GiB
# checkpoint, and taking it down mid-programme is documented in this repo as the single most
# expensive class of mistake. This refuses to run while the battery, the extension pass or the
# multi-turn run hold the engine, and it will not start unless the staged binary passes the CPU
# gates first.
#
# WHAT THE NEW BINARY ADDS: per-request `spec_profile` -- the joint (realised verify width,
# accepted prefix length) histogram, from which the conditional accept hazard h(j) is recoverable.
# `tokens_per_verify` alone cannot give that, and h(j) is the quantity draft-head training actually
# moves (see accept_profile.py). Two integer increments per verify, no sync, no device memory.
#
#   bash scripts/deploy_staged_server.sh
set -eu
cd "$(dirname "$0")/.."
STAGED=build/dsv4-server.staged
LIVE=build/dsv4-server

[ -x "$STAGED" ] || { echo "no $STAGED — build it first (see the nvcc line in build_server.sh)"; exit 1; }

for p in "bash scripts/eval_supervise.sh" "bash scripts/run_evals.sh" \
         "bash scripts/eval_extend_all.sh" "bash scripts/eval_bfcl_mt_run.sh" \
         "tools/eval_suite.py" "tools/eval_extend.py" "tools/eval_bfcl_mt.py" \
         "tools/eval_force.py"; do
  if pgrep -f "$p" > /dev/null; then
    echo "REFUSING: '$p' is running. A restart now would cost it whatever is in flight."
    exit 1
  fi
done

echo "== CPU gates on the staged build =="
for g in build/gate_tokenizer build/gate_stream build/gate_encoding build/gate_api; do
  [ -x "$g" ] && { "$g" > /dev/null && echo "  ok  $g"; } || echo "  skip $g (not built)"
done

echo "== swapping =="
cp -a "$LIVE" "${LIVE}.prev.$(date +%Y%m%d-%H%M%S)"   # keep a rollback that is known to have served
pkill -f "build/dsv4-server --ckpt" || true
sleep 5
mv "$STAGED" "$LIVE"
echo "swapped. previous binary kept as ${LIVE}.prev.*"

echo "== restarting =="
nohup setsid scripts/with_model_lock.sh env SEQMAX=32768 EXT_CHUNK=64 bash scripts/serve.sh \
      > evidence/eval_server.log 2>&1 &
echo "loading (~10-15 min for 101 GiB). Watch: tail -f evidence/eval_server.log"
echo "Then confirm the new field is live:"
echo "  curl -s localhost:8080/v1/completions -H 'Content-Type: application/json' \\"
echo "       -d '{\"prompt\":\"2+2=\",\"max_tokens\":8}' | python3 -m json.tool | grep -A4 spec_profile"
