#!/usr/bin/env python3
"""release_rule.py — evaluate HEAD_REGISTRY.md's release rule on a candidate's eval log.

PROMOTION AND RELEASE ARE DIFFERENT GATES, and 2026-08-23 is the case that separates them:
`s3recap-p25-b0.1` beat the incumbent's suite mean by 4.13 %, was correctly promoted, is correctly
deployed -- and fails 3 of the 6 per-category floors. A head can win the mean by hollowing out the
reconstructive categories, which are exactly the ones an agentic coding harness lives in.

THE RULE (HEAD_REGISTRY.md, written 2026-08-20 before the P2 candidates existed):

    reconstructive  long_context, agentic_format, code_edit   >= run-0 minus 0.2
    constructive    explanation, code_gen, reasoning          >= the incumbent's value

WHY THIS TOOL EXISTS RATHER THAN A PARAGRAPH. The rule was **unevaluable from ladder 2.1 until
2026-08-23** and nobody noticed, because it is consulted only at release time and nothing was
released in between. `tau`'s ceiling IS the draft width, so run-0's block-6 floors were being
compared against block-5 candidates -- 6-ceilinged numbers gating 5-ceilinged measurements. A rule
that is checked by hand, once, at the end, is a rule that silently stops applying.

All three sets of numbers must come from the SAME block width. Produce run-0 with:

    BASELINE_CKPT=~/models/DeepSeek-V4-Flash-0731-REAP bash scripts/baseline_tau.sh evidence/run0_tau_blk5.log

CATEGORY ORDER IS ALPHABETICAL and prompt 0 is the control, excluded. That mapping is not asserted
from documentation -- it is confirmed by run-0's own shape: the stock head's constructive categories
(code_gen, explanation, reasoning) read 1.86/1.70/1.82 and its reconstructive ones read 4.08-5.00,
the 2.7x split this project has measured repeatedly.

    python3 tools/release_rule.py evidence/<candidate>_eval.log \\
        --run0 evidence/run0_tau_blk5.log --incumbent evidence/baseline_tau_blk5.log
"""
import argparse, re, sys

CATS = ["agentic_format", "code_edit", "code_gen", "explanation",
        "long_context", "multi_turn", "reasoning", "short_factual"]
RECON = ["long_context", "agentic_format", "code_edit"]
CONSTR = ["explanation", "code_gen", "reasoning"]
RECON_SLACK = 0.2


def taus(path):
    """prompt index -> tau, from the blksweep table. Prompt 0 is the control and is dropped."""
    out, blocks = {}, set()
    for line in open(path):
        m = re.search(r"\[blksweep\]\s+(\d+)\s+\d+\s+[\d.]+\s+[\d.]+\s+(\d+)\s*\|\s*([\d.]+)", line)
        if m:
            blocks.add(int(m.group(1)))
            out[int(m.group(2))] = float(m.group(3))
    if len(blocks) != 1:
        sys.exit(f"{path}: expected exactly one block width, found {sorted(blocks) or 'none'} -- "
                 "tau is not comparable across widths and this rule must not be applied across them")
    cats = {CATS[i - 1]: v for i, v in out.items() if 1 <= i <= 8}
    if len(cats) != 8:
        sys.exit(f"{path}: parsed {len(cats)}/8 categories -- an incomplete suite cannot be graded")
    return cats, blocks.pop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("--run0", required=True, help="the STOCK head at the served width")
    ap.add_argument("--incumbent", required=True, help="the deployed head at the served width")
    a = ap.parse_args()

    cand, bc = taus(a.candidate)
    run0, b0 = taus(a.run0)
    inc, bi = taus(a.incumbent)
    if not (bc == b0 == bi):
        sys.exit(f"block widths differ: candidate {bc}, run-0 {b0}, incumbent {bi}. "
                 "tau's ceiling IS the width; these are not comparable.")

    print(f"release rule at block {bc}\n")
    print("%-16s %7s %7s %7s | %s" % ("category", "run-0", "incumb", "cand", "floor"))
    fails = []
    for c in CATS:
        if c in RECON:
            fl, kind = run0[c] - RECON_SLACK, "recon"
        elif c in CONSTR:
            fl, kind = inc[c], "constr"
        else:
            print("%-16s %7.2f %7.2f %7.2f | not floored" % (c, run0[c], inc[c], cand[c]))
            continue
        ok = cand[c] >= fl
        if not ok:
            fails.append((c, cand[c], fl))
        print("%-16s %7.2f %7.2f %7.2f | %-6s >= %.2f  %s"
              % (c, run0[c], inc[c], cand[c], kind, fl, "PASS" if ok else "FAIL"))

    mean = lambda d: sum(d.values()) / len(d)
    print("\n%-16s %7.4f %7.4f %7.4f  (8 categories; prompt 0 is the control and is excluded)"
          % ("suite mean", mean(run0), mean(inc), mean(cand)))

    print("\n" + "=" * 74)
    if fails:
        print("RELEASE: REFUSED -- %d floor(s) failed" % len(fails))
        for c, v, f in fails:
            print("   %-16s %.2f < %.2f   short by %.2f" % (c, v, f, f - v))
        print("\n  The head is archived and recorded, NOT released, however good its mean.")
        print("  This does not undo a promotion: promotion asks 'does it beat the incumbent on")
        print("  average', release asks 'does it beat it WITHOUT hollowing out the categories an")
        print("  agentic workload actually runs in'. A head can pass the first and fail this.")
        return 1
    print("RELEASE: PASS -- every floor cleared.")
    print("  Repoint ~/model-backups/releases/CURRENT_BEST at this head's bundle.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
