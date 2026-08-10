#!/usr/bin/env python3
"""ablation_report.py — collect the CE/TV arms into one table against the same references.

Every arm is trained on the SAME captured sequences and measured on the SAME frozen suite at the
SAME adaptK, so the only variable is the loss weighting. The columns that decide anything are the
suite pooled rate (what the registry promotes on) and the margin sharpness (the mechanism this
ablation is testing): if shifting weight from TV to CE sharpens the draft, margins rise, the adaptK
gate extends more often, and the rate should follow. If margins move and rate does not, the
mechanism is wrong and the hypothesis dies with it.

  python3 tools/ablation_report.py --session s2
"""
import argparse, glob, json, os, re, statistics as st, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from holdout_rate import rate as pooled_rate                            # noqa: E402
from accept_profile import parse as parse_verifies, hazard              # noqa: E402

CAT = {1: "agentic_format", 2: "code_edit", 3: "code_gen", 4: "explanation",
       5: "long_context", 6: "multi_turn", 7: "reasoning", 8: "short_factual"}
MP = re.compile(r"\s*\[margins\]((?:\s+[-\d.]+)+)\s+K=(\d+)")


def suite_tau(path):
    per = {}
    for l in open(path, errors="ignore"):
        if l.startswith("[blksweep]") and "|" in l and "prompt" not in l:
            f = [x.strip() for x in l.split("|")]
            try:
                per[int(f[0].split()[-1])] = float(f[1])
            except (ValueError, IndexError):
                pass
    ks = [k for k in CAT if k in per]
    return (st.mean(per[k] for k in ks) if ks else None), per


def margins(path, block=6):
    """median margin at the GATING positions 1..block-1 -- what adaptK actually reads."""
    cols = [[] for _ in range(block)]
    for l in open(path, errors="ignore"):
        m = MP.match(l)
        if not m:
            continue
        v = [float(x) for x in m.group(1).split()]
        if len(v) >= block:
            for i in range(block):
                cols[i].append(v[i])
    if not cols[0]:
        return None, None
    gating = [x for i in range(1, block) for x in cols[i]]
    return st.median(gating), st.median(cols[0])


def row(label, path):
    if not os.path.exists(path):
        return None
    r = pooled_rate(path)
    tau, _ = suite_tau(path)
    gm, m0 = margins(path)
    rows = parse_verifies(path)
    h = hazard(rows, 6) if rows else []
    h0 = h[0][0] if h and h[0] else None
    fullw = (100 * sum(1 for x in rows if x["K"] == 7) / len(rows)) if rows else None
    return {"label": label, "rate": r["tok_s_pooled"] if r else None, "tau": tau,
            "gate_margin": gm, "m0": m0, "h0": h0, "fullw": fullw}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="s2")
    ap.add_argument("--evidence", default="/home/patrickd/deepseek-v4-flash-0731-cuda/evidence")
    a = ap.parse_args()

    rows = [row("untrained (run 0)", os.path.join(a.evidence, "baseline_blk6_suite.log")),
            row("s1 reasoning-only", os.path.join(a.evidence, "s1_eval.log")),
            row(f"{a.session} balanced 0.1/0.9", os.path.join(a.evidence, f"{a.session}_eval.log"))]
    for f in sorted(glob.glob(os.path.join(a.evidence, f"{a.session}_abl_*_eval.log"))):
        m = re.search(r"_abl_ce([\d.]+)_tv([\d.]+)_eval", f)
        rows.append(row(f"{a.session} balanced {m.group(1)}/{m.group(2)}" if m else os.path.basename(f), f))
    rows = [r for r in rows if r]

    print(f"\nCE/TV ABLATION -- same capture, same frozen suite, same adaptK 1.5\n")
    print(f"  {'arm':<28} {'tok/s':>7} {'suite tau':>10} {'h(0)':>7} {'K=7':>7} "
          f"{'gate margin':>12} {'m(0)':>7}")
    for r in rows:
        f = lambda v, s: (s % v) if v is not None else "     --"
        print(f"  {r['label']:<28} {f(r['rate'],'%7.2f')} {f(r['tau'],'%10.4f')} "
              f"{f(100*r['h0'] if r['h0'] is not None else None,'%6.1f%%')} "
              f"{f(r['fullw'],'%6.1f%%')} {f(r['gate_margin'],'%12.2f')} {f(r['m0'],'%7.2f')}")

    best = max((r for r in rows if r["rate"]), key=lambda r: r["rate"], default=None)
    if best:
        print(f"\n  best pooled rate: {best['label']} at {best['rate']:.2f} tok/s")
        print(f"  NOTE: the registry promotes on a >3.5% run-to-run margin over the incumbent, so a")
        print(f"  win inside that band is a tie and ties go to the incumbent.")
    print("\n  gate margin = median top1-top2 over gating positions 1-5, which is what adaptK reads.")
    print("  The hypothesis under test: weight toward CE -> sharper draft -> larger margins ->")
    print("  the gate extends more often -> higher rate. Margins moving without rate moving")
    print("  falsifies the mechanism, not just the setting.")


if __name__ == "__main__":
    main()
