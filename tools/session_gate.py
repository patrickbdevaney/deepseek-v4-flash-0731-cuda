#!/usr/bin/env python3
"""session_gate.py — apply the S5 session decision rule to an eval log.

This is NOT the promotion rule. Promotion asks "is this head better than the incumbent, everywhere"
(`promote_head.py`, >3.5% suite-mean gain). The SESSION gate asks a different and earlier question:
**did training move the thing session 1 was designed to test at all, and is it worth paying for the
next session?** A head can fail promotion and still say GO -- that is the expected outcome of a
narrow single-domain proof, and conflating the two rules would end the programme on its first
deliberately-narrow run.

THE THRESHOLDS ARE COPIED FROM `S5_PROGRESSION.md` §2, WHICH WAS WRITTEN BEFORE THE DATA EXISTS.
That is the entire value of this file: the rule cannot be adjusted after seeing the number, because
the number does not exist yet and the rule is already in git.

  reasoning tau >= 2.6   -> GO          the defect is real and repairable
  reasoning tau 2.1-2.6  -> GO_REPRICE  real but small; the full run is worth ~half the papers imply
  reasoning tau <  2.1   -> STOP        within noise of the 1.85 baseline; training does not move it
  suite mean DROPS       -> STOP        overfitting to one domain; the corpus must be mixed
  LOSSLESS gate fails    -> STOP        we perturbed something the verify depends on

Exit codes: 0 GO, 2 GO_REPRICE, 3 STOP. The orchestrator chains on 0 only.

  python3 tools/session_gate.py --eval <log> [--baseline-reasoning 1.85] [--baseline-suite 3.5362]
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from promote_head import parse_eval                                     # noqa: E402

# Index -> category for `protocol/suite_prompts.txt`. Established by DECODING the frozen prompt file
# with the checkpoint's own tokenizer, not by matching token counts against a table sorted by tau --
# two prompts are 15 ids long and a count-match would have had to guess between them.
CATEGORY = {1: "agentic_format", 2: "code_edit", 3: "code_gen", 4: "explanation",
            5: "long_context", 6: "multi_turn", 7: "reasoning", 8: "short_factual"}

GO, REPRICE, STOP = 2.6, 2.1, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--baseline-reasoning", type=float, default=1.85, dest="base_r",
                    help="F96 reasoning tau on the untrained head")
    ap.add_argument("--baseline-suite", type=float, default=3.5362, dest="base_s",
                    help="F96 suite-mean tau on the untrained head")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    e = parse_eval(a.eval)
    by_cat = {}
    for r in e["points"]:
        if r["prompt"] in CATEGORY:
            by_cat[CATEGORY[r["prompt"]]] = r

    print(f"eval: {a.eval}")
    print(f"  lossless={e['lossless']}  gate={e['gate']}  blocks={e['blocks']}  "
          f"instruments={e['instruments_present'] or 'none'}")
    for c in sorted(by_cat, key=lambda k: -by_cat[k]["tau"]):
        r = by_cat[c]
        print(f"    {c:16s} tau {r['tau']:5.2f}   {r['tok_s']:6.2f} tok/s")
    print(f"  suite mean tau {e['suite_tau']}  ({e['n_suite']} prompts, prompt 0 excluded)")

    reasons, verdict = [], "GO"
    if not e["lossless"]:
        verdict, _ = "STOP", reasons.append("LOSSLESS gate did not pass")
    if e["gate_fail"]:
        verdict, _ = "STOP", reasons.append("first-token GATE FAIL")
    if e["instruments_present"]:
        verdict, _ = "STOP", reasons.append(f"profiling instruments in the binary: {e['instruments_present']}")

    r_tau = by_cat.get("reasoning", {}).get("tau")
    if r_tau is None:
        verdict, _ = "STOP", reasons.append("no reasoning-category point in the log")
    elif verdict != "STOP":
        if r_tau >= GO:
            reasons.append(f"reasoning tau {r_tau:.2f} >= {GO} (baseline {a.base_r})")
        elif r_tau >= REPRICE:
            verdict = "GO_REPRICE"
            reasons.append(f"reasoning tau {r_tau:.2f} in [{REPRICE}, {GO}) -- real but small; "
                           f"re-price the full run at roughly half")
        else:
            verdict = "STOP"
            reasons.append(f"reasoning tau {r_tau:.2f} < {REPRICE}, within noise of the "
                           f"{a.base_r} baseline -- training does not move this head")

    if e["suite_tau"] is not None and e["suite_tau"] < a.base_s and verdict != "STOP":
        verdict = "STOP"
        reasons.append(f"suite mean tau {e['suite_tau']} DROPPED below the {a.base_s} baseline -- "
                       f"single-domain overfit; the corpus must be mixed from the start")

    print(f"\nVERDICT: {verdict}")
    for r in reasons:
        print(f"  - {r}")
    if a.json_out:
        json.dump({"verdict": verdict, "reasons": reasons, "reasoning_tau": r_tau,
                   "suite_tau": e["suite_tau"], "suite_tok_s": e["suite_tok_s"],
                   "base_ar_tok_s": e["base_ar_tok_s"],
                   "by_category": {c: {"tau": v["tau"], "tok_s": v["tok_s"]} for c, v in by_cat.items()},
                   "eval_log": os.path.abspath(a.eval)}, open(a.json_out, "w"), indent=1)
        print(f"  -> {a.json_out}")
    sys.exit({"GO": 0, "GO_REPRICE": 2, "STOP": 3}[verdict])


if __name__ == "__main__":
    main()
