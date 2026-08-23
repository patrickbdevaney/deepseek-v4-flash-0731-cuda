#!/usr/bin/env python3
"""analyze_ck.py — does the best draft width differ by task shape?

That single question decides ROADMAP.md lever 1. tok/s at width k already equals
E[A(T_k)] / C(k), so the per-prompt argmax over k IS the choice an adaptive engine would make --
measured, not modelled. The gain column is the honest upper bound on the engine change: what you
would get if the engine picked each prompt's own best width instead of one width for all.
"""
import re, sys
from collections import defaultdict

CATS = ["control", "agentic_format", "code_edit", "code_gen", "explanation",
        "long_context", "multi_turn", "reasoning", "short_factual"]

rows = defaultdict(dict)          # prompt -> width -> (tau, tok_s)
for line in open(sys.argv[1]):
    m = re.search(r"\[blksweep\]\s+(\d+)\s+\d+\s+[\d.]+\s+[\d.]+\s+(\d+)\s*\|\s*([\d.]+)\s*\|"
                  r"\s*[\d.]+\s*\|\s*([\d.]+)", line)
    if m:
        k, p, tau, toks = int(m.group(1)), int(m.group(2)), float(m.group(3)), float(m.group(4))
        rows[p][k] = (tau, toks)

if not rows:
    sys.exit("no blksweep rows parsed")

widths = sorted({k for d in rows.values() for k in d})
print("tok/s by prompt x width\n")
print("%-16s %s" % ("prompt", "".join("%8d" % k for k in widths)))
for p in sorted(rows):
    name = CATS[p] if p < len(CATS) else str(p)
    print("%-16s %s" % (name, "".join("%8.2f" % rows[p][k][1] if k in rows[p] else "       -"
                                      for k in widths)))

print("\nper-prompt optimum vs the served width of 5\n")
print("%-16s %6s %9s %9s %8s" % ("prompt", "k*", "tok/s@k*", "tok/s@5", "gain"))
tot5 = totstar = 0.0
kstars = []
for p in sorted(rows):
    name = CATS[p] if p < len(CATS) else str(p)
    kstar = max(rows[p], key=lambda k: rows[p][k][1])
    best = rows[p][kstar][1]
    at5 = rows[p].get(5, (None, None))[1]
    kstars.append(kstar)
    if at5:
        tot5 += at5; totstar += best
        print("%-16s %6d %9.2f %9.2f %+7.1f%%" % (name, kstar, best, at5, 100*(best-at5)/at5))

print("\n" + "=" * 62)
uniq = sorted(set(kstars))
if len(uniq) == 1:
    print("VERDICT: k* = %d for EVERY prompt." % uniq[0])
    print("  Adaptive block width has nothing to exploit -- one width is optimal across a 2.7x")
    print("  spread in acceptance. ROADMAP.md lever 1 is REFUTED and must be re-priced to ~0")
    print("  before any CUDA is written. This is the cheap outcome and it is worth knowing.")
else:
    gain = 100 * (totstar - tot5) / tot5
    print("VERDICT: k* varies -- %s across the suite." % uniq)
    print("  An engine that picked each prompt's own width would gain %+.1f%% on the suite mean." % gain)
    print("  That is the MEASURED upper bound for lever 1, replacing the +20-25%% estimate.")
    print("  It is an upper bound because a real engine must PREDICT k* per position from the")
    print("  confidence head (AUC 0.88), not read it off a table computed with hindsight.")
