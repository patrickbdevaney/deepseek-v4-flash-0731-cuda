#!/usr/bin/env bash
# baseline_tau.sh — measure the DEPLOYED head's suite tau at the width the engine now serves.
#
# WHY THIS HAS TO EXIST. Ladder 2.1 shipped the draft block 6 -> 5. `tau` is tokens committed per
# target forward and its CEILING IS THE WIDTH, so a block-5 tau cannot be compared with the
# block-6 tau in every row of HEAD_REGISTRY.md. Promoting a P2 head against an archived block-6
# number would compare a 5-ceilinged measurement with a 6-ceilinged one and call the difference a
# regression. promote_head.py --incumbent-tau takes a same-width value; this produces it.
#
# It measures `config/live_ckpt`, i.e. whatever is actually deployed, rather than a name -- the
# incumbent is what a user would get today, not what the registry says was best.
set -uo pipefail
cd "$(dirname "$0")/.."
BLK="${S5_BLOCK:-5}"
# DEFAULTS to the deployed head, which is the incumbent a promotion must beat. BASELINE_CKPT
# overrides it for the one other measurement this protocol needs: RUN-0, the stock head, which the
# release rule in HEAD_REGISTRY.md compares the reconstructive categories against. Run-0 had only
# ever been measured at block 6, so after ladder 2.1 the release rule could not be evaluated at all
# -- its floors were 6-ceilinged and every candidate is 5-ceilinged. The stock head IS the base
# checkpoint's own, so measuring it needs no staging and never touches config/live_ckpt.
CKPT="${BASELINE_CKPT:-$(cat config/live_ckpt)}"
OUT="${1:-evidence/baseline_tau_blk${BLK}.log}"
echo "[baseline] measuring $CKPT at block $BLK, frozen 8-prompt suite, NGEN0=200"
SUITE=$(cat protocol/suite_prompts.txt)
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; sleep 2
DSV4_PROMPTS="$SUITE" \
DSV4_BLKSWEEP="$(python3 -c "print(','.join(f'$BLK:1:1.5:{i}' for i in range(0,9)))")" \
    scripts/run_model.sh "$OUT" ./build/decode "$CKPT" "0,671,6102,294,8760,344" 8 "" 200
while pgrep -x decode >/dev/null; do sleep 30; done
python3 - "$OUT" <<'PY'
import sys, os
sys.path.insert(0, "tools")
from promote_head import parse_eval
ev = parse_eval(sys.argv[1])
print(f"[baseline] blocks={ev['blocks']}  n_suite={ev['n_suite']}  "
      f"suite_tau={ev['suite_tau']}  suite_tok_s={ev['suite_tok_s']}  base_ar={ev['base_ar_tok_s']}")
if ev["suite_tau"] is None or ev["n_suite"] < 8:
    sys.exit("[baseline] FAILED to parse a complete suite -- not a usable incumbent")
# Only the DEPLOYED head's number is the incumbent bar. A run-0 measurement must never overwrite
# it -- that would silently lower the promotion bar to the stock head's tau.
if os.environ.get("BASELINE_CKPT"):
    print("[baseline] BASELINE_CKPT set -> this is NOT the incumbent; baseline_tau.value untouched")
else:
    open("evidence/baseline_tau.value", "w").write(str(ev["suite_tau"]))
    print("[baseline] wrote evidence/baseline_tau.value")
PY
