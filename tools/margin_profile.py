#!/usr/bin/env python3
"""margin_profile.py — the per-position draft-margin profile, and the diff between two runs.

F109-F111 reduced the whole question of "can this reach 30 tok/s" to one measurable quantity: the
draft's logit margin at gating positions 1-5. +1.0 added to every gating margin reaches ~27.9 tok/s,
+1.5 reaches ~31.2. So the single most informative thing to measure about a trained head is **how
far its margins moved**, per position, against the same prompts.

This dumps that profile from any spec-decode log, and diffs two profiles. Run it on pass 1 to fix
the untrained baseline, and on the trained head's hold-out eval to read the shift directly, rather
than inferring it from tau.

WHY PER-POSITION AND NOT AN AVERAGE. The gate is a conjunction over positions 1-5: a verify reaches
full width only if *all* of them clear. A mean margin can rise while the weakest position does not
move, and the weakest position is the one that caps K. The p10/p25 columns are therefore more
decision-relevant than the median.

  python3 tools/margin_profile.py --log <log> [--json-out f]
  python3 tools/margin_profile.py --log <after> --vs <before.json>
"""
import argparse, json, re, statistics as st, sys

MP = re.compile(r"\s*\[margins\]((?:\s+[-\d.]+)+)\s+K=(\d+)")


def profile(path, gate=1.5, block=6):
    cols = [[] for _ in range(block)]
    Ks = []
    for line in open(path, errors="ignore"):
        m = MP.match(line)
        if not m:
            continue
        v = [float(x) for x in m.group(1).split()]
        if len(v) < block:
            continue
        for i in range(block):
            cols[i].append(v[i])
        Ks.append(int(m.group(2)))
    if not Ks:
        sys.exit(f"no [margins] lines in {path}")

    def q(v, p):
        s = sorted(v)
        return s[min(len(s) - 1, int(p * len(s)))]

    prof = []
    for i, c in enumerate(cols):
        prof.append({"pos": i, "n": len(c), "p10": round(q(c, .10), 3), "p25": round(q(c, .25), 3),
                     "median": round(st.median(c), 3), "p75": round(q(c, .75), 3),
                     "share_ge_gate": round(100 * sum(1 for x in c if x >= gate) / len(c), 2)})
    # a verify reaches full width only if gating positions 1..block-1 all clear
    full = sum(1 for j in range(len(Ks)) if all(cols[i][j] >= gate for i in range(1, block)))
    return {"log": path, "gate": gate, "block": block, "n_verifies": len(Ks),
            "positions": prof, "full_width_share": round(100 * full / len(Ks), 2),
            "measured_K7_share": round(100 * sum(1 for k in Ks if k == block + 1) / len(Ks), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--vs", default=None, help="a previously written profile JSON to diff against")
    ap.add_argument("--gate", type=float, default=1.5)
    ap.add_argument("--block", type=int, default=6)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    p = profile(a.log, a.gate, a.block)
    print(f"{p['log']}: {p['n_verifies']} verifies")
    print(f"{'pos':>4} {'p10':>8} {'p25':>8} {'median':>8} {'p75':>8} {'>=gate':>8}   gating?")
    for r in p["positions"]:
        g = "yes" if 1 <= r["pos"] <= a.block - 1 else "no (acceptance only)"
        print(f"{r['pos']:>4} {r['p10']:>8.2f} {r['p25']:>8.2f} {r['median']:>8.2f} "
              f"{r['p75']:>8.2f} {r['share_ge_gate']:>7.1f}%   {g}")
    print(f"  full-width share: {p['full_width_share']:.1f}% predicted, "
          f"{p['measured_K7_share']:.1f}% measured")

    if a.vs:
        b = json.load(open(a.vs))
        print(f"\nDIFF vs {b['log']} ({b['n_verifies']} verifies)")
        print(f"{'pos':>4} {'d p10':>8} {'d p25':>8} {'d median':>9} {'d >=gate':>9}")
        for r, s in zip(p["positions"], b["positions"]):
            print(f"{r['pos']:>4} {r['p10']-s['p10']:>+8.2f} {r['p25']-s['p25']:>+8.2f} "
                  f"{r['median']-s['median']:>+9.2f} {r['share_ge_gate']-s['share_ge_gate']:>+8.1f}%")
        d = p["full_width_share"] - b["full_width_share"]
        print(f"  full-width share {b['full_width_share']:.1f}% -> {p['full_width_share']:.1f}% "
              f"({d:+.1f} points)")
        # the gating positions are what F109 priced; report their weakest movement
        gm = [r["p25"] - s["p25"] for r, s in zip(p["positions"], b["positions"])
              if 1 <= r["pos"] <= a.block - 1]
        print(f"  WEAKEST gating-position p25 shift: {min(gm):+.2f}  (F109: +1.0 -> ~27.9 tok/s, "
              f"+1.5 -> ~31.2; the gate is a conjunction, so the weakest one is what binds)")

    if a.json_out:
        json.dump(p, open(a.json_out, "w"), indent=1)
        print(f"\n-> {a.json_out}")


if __name__ == "__main__":
    main()
