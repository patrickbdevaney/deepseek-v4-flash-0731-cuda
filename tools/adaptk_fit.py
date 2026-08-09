#!/usr/bin/env python3
"""adaptk_fit.py — fit the adaptive-K threshold from measured margins instead of sweeping it blind.

`src/decode.cu` chooses the verify width from the draft's own logit margins:

    VK = 2; while (VK < VKCAP && hmarg[VK-1] >= adaptK) ++VK;

so a SINGLE scalar `adaptK` gates every extension, from "verify one more cheap token" to "verify the
seventh". The code comment next to it says the threshold should be fitted offline from a margin-vs-
acceptance calibration rather than swept across 15-minute runs. This is that fit.

THE ECONOMICS, which is why one scalar cannot be right. Extending the verify by one position costs
`dms(K)` -- measured, and it is NOT constant: the step from K=2 is ~14 ms while the step to K=7 is
~25 ms. It yields one token with probability P(accept at that position). So extending pays iff

    P(accept) > r * dms(K)          r = the rate you would otherwise run at, in tokens/ms

The right-hand side rises with K, so the threshold must rise with K too. A scalar 1.5 that is
correct for the first extension is too permissive for the last.

CENSORING, stated because it bounds what this can conclude. Positions past the chosen K were never
verified, so acceptance below the operating threshold is unobserved for those positions. This fit
can therefore say "raise the threshold" with evidence and can only say "lower it" by extrapolation.
Position 0 is the exception -- `VK` starts at 2, so it is always verified and its calibration is
uncensored across the whole margin range.

  python3 tools/adaptk_fit.py --log <spec-decode log> [--rate 0]
"""
import argparse, re, statistics as st, sys

MP = re.compile(r"\s*\[margins\]((?:\s+[-\d.]+)+)\s+K=(\d+)")
VP = re.compile(r"\s*verify \d+: accepted (\d+)/(\d+) \(K=(\d+)\) \+ correction -> \+(\d+) tokens \(([\d.]+) ms\)")


def parse(path):
    rows, pend = [], None
    for l in open(path, errors="ignore"):
        m = MP.match(l)
        if m:
            pend = [float(x) for x in m.group(1).split()]
            continue
        v = VP.match(l)
        if v and pend is not None:
            rows.append({"marg": pend, "acc": int(v.group(1)), "K": int(v.group(3)),
                         "tok": int(v.group(4)), "ms": float(v.group(5))})
            pend = None
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--rate", type=float, default=0.0,
                    help="opportunity cost in tokens/ms; 0 = use the log's own measured rate")
    ap.add_argument("--grid", default="1.0,1.5,2.0,3.0,4.0,5.0,6.0,8.0,10.0")
    a = ap.parse_args()

    rows = parse(a.log)
    if not rows:
        sys.exit(f"no paired margin/verify records in {a.log}")

    tot_tok = sum(r["tok"] for r in rows)
    tot_ms = sum(r["ms"] for r in rows)
    rate = a.rate or tot_tok / tot_ms
    print(f"{len(rows)} verifies | measured {1000*rate:.2f} tok/s ({rate:.5f} tok/ms)\n")

    # measured cost of each extension, from the verify times by width
    ms_by_K = {}
    for r in rows:
        ms_by_K.setdefault(r["K"], []).append(r["ms"])
    mk = {K: st.mean(v) for K, v in ms_by_K.items() if len(v) >= 30}
    dms = {K: mk[K] - mk[K - 1] for K in sorted(mk) if K - 1 in mk}
    print("cost of extending the verify by one position (measured):")
    for K in sorted(dms):
        print(f"  K={K-1} -> {K}: {dms[K]:>5.1f} ms   break-even P(accept) = "
              f"{rate*dms[K]:>5.1%}")

    # ---- calibration: P(accept at the gating position | its margin) -----------------------------
    # BUCKET conditionals, not cumulative P(accept | margin >= t). The decision at threshold t is
    # marginal -- "for a position whose margin sits just above t, does extending pay?" -- and the
    # cumulative version answers a different question, averaging in every high-margin position and
    # so overstating P at every t. Using it produced a schedule of a flat 1.0, which is also the
    # smallest grid value: the tell that the quantity was wrong.
    grid = [float(x) for x in a.grid.split(",")]
    edges = list(zip(grid, grid[1:] + [float("inf")]))
    print(f"\nP(accept) in each margin BUCKET, for the position that bucket gates:")
    print(f"{'margin':>14} {'n':>7} {'P(accept)':>10}")
    cal = []
    for lo, hi in edges:
        sel = [r for r in rows if r["K"] >= 3 and lo <= r["marg"][r["K"] - 2] < hi]
        if len(sel) < 30:
            continue
        p = sum(1 for r in sel if r["acc"] >= r["K"] - 1) / len(sel)
        cal.append((lo, hi, len(sel), p))
        print(f"{lo:>6.1f}-{hi if hi < 1e9 else 99:<7.1f} {len(sel):>7} {100*p:>9.1f}%")

    floor = min(c[0] for c in cal) if cal else None
    print(f"\nCENSORING FLOOR: {floor:g}. This log ran at adaptK={floor:g}, so no gating position "
          f"below it was ever verified. The fit can justify RAISING the threshold and cannot "
          f"evaluate lowering it.")

    # ---- the per-position threshold this implies ------------------------------------------------
    print(f"\nIMPLIED PER-POSITION THRESHOLD (lowest bucket whose marginal P clears that K's cost):")
    sched = {}
    for K in sorted(dms):
        need = rate * dms[K]
        pick = next((lo for lo, hi, n, p in cal if p > need), None)
        sched[K] = pick
        note = "" if pick is None else (" (at the censoring floor: cannot be justified lower)"
                                        if pick == floor else "")
        print(f"  extending to K={K}: needs P > {need:>5.1%} -> threshold "
              f"{pick if pick is not None else '> the observed range'}{note}")
    cur = [v for v in sched.values() if v is not None]
    if cur:
        print(f"\nThe shipped value is a SCALAR 1.5 on every extension. This fit says it should "
              f"RISE with K, over {min(cur):g}..{max(cur):g} -- because the cost of an extension "
              f"rises with K ({min(dms.values()):.1f} -> {max(dms.values()):.1f} ms) while a single "
              f"threshold holds the required acceptance constant.")
    # ---- counterfactual replay ------------------------------------------------------------------
    # A STRICTER policy is exactly evaluable offline. If the new schedule picks K' <= K, then every
    # position it verifies was already verified in this log, so its acceptance is observed -- no
    # counterfactual guessing. Tokens = min(acc, K'-1) + 1 (the correction always lands); time =
    # the measured mean verify cost at width K'. A policy that was ever LOOSER than the log's would
    # need acceptance data that does not exist, and this replay refuses rather than extrapolating.
    def replay(sched_fn, label):
        tok = ms = 0.0
        widths = {}
        for r in rows:
            K2 = 2
            while K2 < r["K"] + (0 if r["K"] >= 7 else 0) and K2 < 7:
                if r["marg"][K2 - 1] >= sched_fn(K2 + 1):
                    K2 += 1
                else:
                    break
            K2 = min(K2, r["K"])                       # never looser than what was observed
            widths[K2] = widths.get(K2, 0) + 1
            tok += min(r["acc"], K2 - 1) + 1
            ms += mk.get(K2, r["ms"])
        rt = 1000 * tok / ms
        mix = " ".join(f"K{k}:{100*v/len(rows):.0f}%" for k, v in sorted(widths.items()))
        print(f"  {label:<34} {rt:>6.2f} tok/s   {tok/len(rows):>5.3f} tok/verify   {mix}")
        return rt

    print(f"\nCOUNTERFACTUAL REPLAY over the same {len(rows)} verifies "
          f"(stricter-only, so every verified position's acceptance is observed):")
    base = replay(lambda K: 1.5, "shipped: flat 1.5")
    fit = replay(lambda K: sched.get(K) or 1.5, f"fitted: {[sched.get(k) for k in sorted(dms)]}")
    for flat in (2.0, 3.0, 4.0, 5.0):
        replay(lambda K, f=flat: f, f"flat {flat:g} (control)")
    print(f"\n  fitted vs shipped: {100*(fit-base)/base:+.2f}%")
    print("  The flat controls matter: if a flat threshold matches the fitted schedule, the "
          "per-position structure is not what is doing the work and the simpler policy wins.")

    print("\nThis is a CANDIDATE POLICY, not a result. It is fitted on one workload with the "
          "censoring noted in the docstring, and the opportunity-cost rate `r` moves once the "
          "policy changes it -- so it must be A/B'd on the frozen suite before it is believed.")


if __name__ == "__main__":
    main()
