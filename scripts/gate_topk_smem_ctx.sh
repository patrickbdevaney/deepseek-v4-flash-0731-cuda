#!/usr/bin/env bash
# gate_topk_smem_ctx.sh — DECODE_LADDER item 1.4's engine-path leg above the dynamic-shared-memory
# ceiling, as a before/after over ONE binary.
#
# Session-bound on purpose (CLAUDE.md's converse clause): it is ~2 minutes, it needs no checkpoint
# and no GPU tenancy beyond ~1 GiB, and every leg's result decides whether the next one runs. It is
# not a battery.
#
# Each leg is its own PROCESS because DSV4_TOPK_RADIX / DSV4_TOPK_GATE / DSV4_TOPK_SMEM_OPTIN are
# read once into function-local statics; setting them between calls inside one process would silently
# have no effect, which is the sort of thing that turns an A/B into two identical arms.
set -u
cd "$(dirname "$0")/.."

BIN=build/gate_topk_smem_ctx
BELOW="${BELOW:-8192}"        # under the 49,140 ceiling: proves the harness agrees with itself
ABOVE="${ABOVE:-49208}"       # just over it: T=12,301, 49,216 B > the 49,152 B default
DEEP="${DEEP:-200004}"        # T=50,000, 200,016 B -- deep inside the opt-in range
OVER="${OVER:-240004}"        # T=60,000, 240,016 B > the 232,448 B opt-in maximum: must abort

[ -x "$BIN" ] || { echo "FATAL: $BIN missing. Build it with the command in tests/gate_topk_smem_ctx.cu"; exit 1; }
for f in tests/gate_topk_smem_ctx.cu include/indexer.h kernels/compressed_decode.cu kernels/indexer.cu; do
  [ "$f" -nt "$BIN" ] && { echo "FATAL: $f is newer than $BIN. Rebuild before trusting this."; exit 1; }
done

fail=0
# One leg = one process. Its whole output is printed, indented, and the hash is read back from the
# same text the reader sees -- not from a second, invisible invocation. `tee /dev/stderr` inside a
# command substitution interleaves the two streams and produced an unreadable log the first time.
LEGLOG=$(mktemp)
trap 'rm -f "$LEGLOG"' EXIT
leg(){ # leg <tag> <ctx> [env...]  -> echoes the hash, prints the leg
  local tag="$1" ctx="$2"; shift 2
  env "$@" "$BIN" "$ctx" "$tag" > "$LEGLOG" 2>&1
  sed 's/^/    /' "$LEGLOG" >&2
  grep -o 'hash=[0-9a-f]*' "$LEGLOG" | tail -1
}

echo "== 1. below the ceiling: all three arms must agree =="
b_rx=$(leg below-radix "$BELOW" DSV4_TOPK_RADIX=1)
b_sc=$(leg below-scan  "$BELOW" DSV4_TOPK_RADIX=0)
b_gt=$(leg below-gate  "$BELOW" DSV4_TOPK_RADIX=1 DSV4_TOPK_GATE=1)
if [ "$b_rx" = "$b_sc" ] && [ "$b_rx" = "$b_gt" ]; then echo "  OK  all three arms $b_rx"
else echo "  FAIL  radix=$b_rx scan=$b_sc gate=$b_gt"; fail=1; fi

echo
echo "== 2. ABOVE the ceiling, opt-in ON (the shipped state): the arms must still agree =="
a_rx=$(leg above-radix "$ABOVE" DSV4_TOPK_RADIX=1)
a_sc=$(leg above-scan  "$ABOVE" DSV4_TOPK_RADIX=0)
a_gt=$(leg above-gate  "$ABOVE" DSV4_TOPK_RADIX=1 DSV4_TOPK_GATE=1)
if [ "$a_rx" = "$a_sc" ] && [ "$a_rx" = "$a_gt" ]; then echo "  OK  all three arms $a_rx"
else echo "  FAIL  radix=$a_rx scan=$a_sc gate=$a_gt"; fail=1; fi

echo
echo "== 3. ABOVE the ceiling, opt-in OFF (pre-1.4 restored): the defect must REPRODUCE =="
p_sc=$(leg pre1p4-scan "$ABOVE" DSV4_TOPK_SMEM_OPTIN=0 DSV4_TOPK_RADIX=0)
if [ -n "$p_sc" ] && [ "$p_sc" != "$a_rx" ]; then
  echo "  OK  pre-1.4 returns $p_sc where the correct answer is $a_rx -- wrong, and it exited 0"
else
  echo "  FAIL  the before-arm did not reproduce the defect (got '$p_sc', correct '$a_rx')."
  echo "        Either the ceiling moved or DSV4_TOPK_SMEM_OPTIN=0 is not reaching the launch."
  fail=1
fi
echo "  (the in-situ reference under the same switch, which must report a FALSE FAIL:)"
env DSV4_TOPK_SMEM_OPTIN=0 DSV4_TOPK_RADIX=1 DSV4_TOPK_GATE=1 "$BIN" "$ABOVE" pre1p4-gate 2>&1 | sed 's/^/    /'

echo
echo "== 4. deep inside the opt-in range: T=50,000 =="
d_rx=$(leg deep-radix "$DEEP" DSV4_TOPK_RADIX=1)
d_sc=$(leg deep-scan  "$DEEP" DSV4_TOPK_RADIX=0)
if [ "$d_rx" = "$d_sc" ]; then echo "  OK  both arms $d_rx"; else echo "  FAIL  radix=$d_rx scan=$d_sc"; fail=1; fi

echo
echo "== 5. above the opt-in maximum: must ABORT, not return =="
env DSV4_TOPK_RADIX=0 "$BIN" "$OVER" over-optin > /tmp/smem_over.log 2>&1
rc=$?
sed 's/^/    /' /tmp/smem_over.log
if [ "$rc" = "134" ]; then echo "  OK  SIGABRT (rc 134)"; else echo "  FAIL  rc=$rc, expected 134 (SIGABRT)"; fail=1; fi

echo
if [ "$fail" = "0" ]; then echo "GATE: PASS — the engine's decode step is correct above context 49,140, and the defect it"
                           echo "        replaces is reproducible in the same binary with DSV4_TOPK_SMEM_OPTIN=0."
else echo "GATE: FAIL"; fi
exit "$fail"
