#!/usr/bin/env python3
"""head_ab.py -- paired A/B of two draft heads measured on the SAME frozen 8-prompt protocol.

WHY PAIRED. The suite prompts differ enormously from each other -- s3's own promotion log spans
18.8 to 35.1 tok/s across the eight -- so an unpaired difference of suite means is dominated by
which prompts happened to be in the suite, not by the head. Both arms run the identical prompt list
in the identical order, so every prompt is its own control and the statistic is the mean of the
per-prompt DIFFERENCES, with a band from their standard error. That is the same discipline the
kernel items on the ladder use, and for the same reason: this project's run-to-run spread is 3.5%,
which swamps a point estimate.

Suite means are reported too, excluding prompt 0, because that is the number HEAD_REGISTRY.md
records and comparability with the registry is the whole point of a frozen protocol (F96).

AND PAIRING BY PROMPT DOES NOT REMOVE THE BETWEEN-LOAD SPREAD. Two heads cannot be measured in one
process -- `decode` takes a single head dir -- so the arms are separate checkpoint loads, and the
between-load spread on this box is 5.7 % against 0.6 % within a load (measurement-and-traps.md
§19). That drift is COMMON to every prompt in an arm, so per-prompt pairing cannot cancel it. Two
statistics are reported that can:

  * `tau` -- an exact draft/target token comparison. It is a property of the head and the prompt,
    not of how fast the engine ran, so it does not move with clocks, thermals or page cache. It is
    the primary evidence here.
  * `tok/s / base AR tok/s` -- the speculative speedup over this arm's OWN base-AR measurement.
    Base AR never touches the draft head, so it is a within-arm control for exactly the drift that
    pairing leaves behind.

Raw tok/s is reported as well and is the weakest of the three; when it disagrees with the other
two, believe them.

  python3 tools/head_ab.py --a <before.log> --b <after.log> [--label-a X --label-b Y]
"""
import argparse, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from promote_head import parse_eval          # same parser the registry uses -- one definition


def band(xs):
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m, sd / math.sqrt(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--label-a", default="A"); ap.add_argument("--label-b", default="B")
    g = ap.parse_args()
    ea, eb = parse_eval(g.a), parse_eval(g.b)

    for tag, e, p in ((g.label_a, ea, g.a), (g.label_b, eb, g.b)):
        print(f"{tag:>12}: {p}")
        print(f"{'':>12}  LOSSLESS {'PASS' if e['lossless'] else '*** ABSENT ***'}   "
              f"first-token gate {'PASS' if e['gate'] and not e['gate_fail'] else '*** FAIL ***'}   "
              f"base AR {e['base_ar_tok_s']} tok/s   clean={not e['instruments_present']}")
    if not (ea["lossless"] and eb["lossless"]):
        print("\n*** at least one arm has no passing LOSSLESS gate -- the comparison is void ***")

    pa = {r["prompt"]: r for r in ea["points"]}
    pb = {r["prompt"]: r for r in eb["points"]}
    common = sorted(set(pa) & set(pb))
    if set(pa) != set(pb):
        print(f"\nWARNING: arms cover different prompts ({sorted(pa)} vs {sorted(pb)}); "
              f"pairing on the {len(common)} in common")

    ra, rb = ea["base_ar_tok_s"], eb["base_ar_tok_s"]
    if not ra or not rb:
        print("\nWARNING: an arm has no 'WARM decode' base-AR line; the drift control is unavailable")
        ra = ra or 1.0; rb = rb or 1.0
    print(f"\nbase AR: {g.label_a} {ra} tok/s   {g.label_b} {rb} tok/s   "
          f"({100*(rb/ra-1):+.2f} % -- head-independent, so this IS the between-load drift)")

    print(f"\n| prompt | {g.label_a} tau | {g.label_b} tau | d tau | {g.label_a} tok/s | "
          f"{g.label_b} tok/s | d tok/s | d % | {g.label_a} x base | {g.label_b} x base | d x base |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    dt, ds, dp, dx = [], [], [], []
    for i in common:
        a, b = pa[i], pb[i]
        d_tau = b["tau"] - a["tau"]
        d_ts = b["tok_s"] - a["tok_s"]
        pct = 100.0 * d_ts / a["tok_s"]
        xa, xb = a["tok_s"] / ra, b["tok_s"] / rb
        note = "  *(control, excluded)*" if i == 0 else ""
        print(f"| {i}{note} | {a['tau']:.2f} | {b['tau']:.2f} | {d_tau:+.2f} | {a['tok_s']:.2f} | "
              f"{b['tok_s']:.2f} | {d_ts:+.2f} | {pct:+.2f} % | {xa:.3f} | {xb:.3f} | {xb-xa:+.3f} |")
        if i != 0:
            dt.append(d_tau); ds.append(d_ts); dp.append(pct); dx.append(100.0 * (xb / xa - 1))

    mt, et = band(dt); ms, es = band(ds); mp, ep = band(dp); mx, ex = band(dx)
    print(f"\nPAIRED over the {len(dt)} suite prompts (prompt 0 excluded, F96):")
    print(f"  d tau        = {mt:+.4f} +/- {et:.4f}   ({sum(1 for x in dt if x > 0)}/{len(dt)} legs positive)")
    print(f"  d tok/s      = {ms:+.4f} +/- {es:.4f}   ({sum(1 for x in ds if x > 0)}/{len(ds)} legs positive)")
    print(f"  d %          = {mp:+.3f} +/- {ep:.3f} %")
    print(f"  d % x base   = {mx:+.3f} +/- {ex:.3f} %   "
          f"({sum(1 for x in dx if x > 0)}/{len(dx)} legs positive)   <- drift-controlled")
    print(f"\nSUITE MEANS (the registry number):")
    print(f"  {g.label_a}: tau {ea['suite_tau']}  {ea['suite_tok_s']} tok/s   (n={ea['n_suite']})")
    print(f"  {g.label_b}: tau {eb['suite_tau']}  {eb['suite_tok_s']} tok/s   (n={eb['n_suite']})")
    if ea["suite_tok_s"] and eb["suite_tok_s"]:
        r = eb["suite_tok_s"] / ea["suite_tok_s"]
        print(f"  ratio {r:.4f}x  ({100*(r-1):+.2f} %)   run-to-run spread on this box is 3.5 %")
    return 0


sys.exit(main())
