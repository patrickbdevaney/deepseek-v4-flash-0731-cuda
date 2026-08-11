#!/usr/bin/env python3
"""sweep_analyze.py — separate a threshold effect from the box's run-order drift.

F120: this machine's decode rate drifts UPWARD with run order — on a policy that never changed,
successive runs measured 23.999 -> 24.487 -> 24.638, and across five identical-policy replicates the
spread was 6.0 % (sd 2.1 %). A sweep that walks its parameter monotonically therefore cannot tell a
real effect from the drift, and the first version of the adaptK sweep did exactly that.

Shuffling the run order converts that bias into noise. Recording the run index lets us do better and
subtract it: fit

    tok_s ~ mean + effect(threshold) + beta * run_index

by ordinary least squares, report the threshold effects with the drift removed, and report beta
itself so the drift is visible rather than assumed. Effects are quoted against the 1.5 reference
(the shipped threshold) with a standard error, because the decision is "does anything beat 1.5 by
more than measurement error", not "which cell of the table is largest".

  python3 tools/sweep_analyze.py --csv evidence/suiteK_<tag>_measurements.csv
"""
import argparse, csv, math, statistics as st, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--ref", default="1.5", help="reference threshold to compare against")
    a = ap.parse_args()

    rows = []
    with open(a.csv) as f:
        for r in csv.DictReader(f):
            try:
                tok = float(r["tok_s"])
            except (ValueError, KeyError):
                continue
            if tok > 0:
                rows.append({"i": int(r["run_index"]), "thr": r["threshold"], "tok": tok})
    if not rows:
        sys.exit(f"no usable measurements in {a.csv}")

    # factor levels are numeric for a threshold sweep and head NAMES for a head-off; sort
    # numerically when we can so the table reads in order, otherwise alphabetically
    def _key(x):
        try:
            return (0, float(x), "")
        except ValueError:
            return (1, 0.0, x)
    thrs = sorted({r["thr"] for r in rows}, key=_key)
    print(f"\n{len(rows)} measurements over {len(thrs)} thresholds\n")
    print(f"  {'adaptK':>7} {'n':>3} {'mean':>8} {'sd':>7} {'min':>7} {'max':>7}")
    raw = {}
    for t in thrs:
        v = [r["tok"] for r in rows if r["thr"] == t]
        raw[t] = st.mean(v)
        sd = st.stdev(v) if len(v) > 1 else float("nan")
        print(f"  {t:>7} {len(v):>3} {st.mean(v):>8.2f} {sd:>7.2f} {min(v):>7.2f} {max(v):>7.2f}")

    # ---- OLS with a run-index covariate ---------------------------------------------------------
    # design: intercept + one dummy per non-reference threshold + run_index
    if a.ref not in thrs:
        a.ref = thrs[len(thrs) // 2]
    others = [t for t in thrs if t != a.ref]
    cols = ["const"] + [f"thr={t}" for t in others] + ["run_index"]
    X, y = [], []
    for r in rows:
        row = [1.0] + [1.0 if r["thr"] == t else 0.0 for t in others] + [float(r["i"])]
        X.append(row)
        y.append(r["tok"])

    try:
        import numpy as np
    except ImportError:
        print("\n  (numpy unavailable: raw means only, drift not subtracted)")
        return
    Xm, yv = np.array(X), np.array(y)
    beta, *_ = np.linalg.lstsq(Xm, yv, rcond=None)
    resid = yv - Xm @ beta
    dof = len(rows) - Xm.shape[1]
    if dof <= 0:
        print("\n  too few measurements to fit a drift term")
        return
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.pinv(Xm.T @ Xm)
    se = np.sqrt(np.diag(cov))

    print(f"\nOLS  tok_s ~ threshold + run_index      (reference threshold {a.ref}, "
          f"residual sd {math.sqrt(s2):.3f} tok/s)")
    print(f"  {'term':>14} {'estimate':>10} {'se':>8} {'t':>7}")
    for name, b, e in zip(cols, beta, se):
        t = b / e if e > 0 else float("nan")
        print(f"  {name:>14} {b:>10.3f} {e:>8.3f} {t:>7.2f}")

    drift = beta[-1]
    print(f"\n  DRIFT: {drift:+.4f} tok/s per run"
          f"  ({drift*len(rows):+.2f} tok/s across the whole batch of {len(rows)})")
    print(f"  This is the term the first sweep confounded with the threshold. It is estimated here")
    print(f"  rather than assumed away, and the threshold effects below have it removed.")

    print(f"\n  THRESHOLD EFFECTS vs {a.ref}, drift-adjusted:")
    sig = []
    for name, b, e in list(zip(cols, beta, se))[1:-1]:
        t = b / e if e > 0 else 0.0
        verdict = "DISTINGUISHABLE" if abs(t) >= 2 else "not distinguishable from the reference"
        if abs(t) >= 2:
            sig.append((name, b))
        print(f"    {name:<12} {b:+6.3f} tok/s  (t={t:+.2f})  {verdict}")
    print(f"\n  |t| >= 2 is the bar. Anything below it is a number, not a result -- this box's")
    print(f"  identical-policy replicates span 6 % (F120), which is larger than every effect the")
    print(f"  first sweep appeared to find.")
    if not sig:
        print(f"\n  CONCLUSION: no threshold is distinguishable from {a.ref}. The shipped value stands.")
    else:
        print(f"\n  CONCLUSION: {', '.join(n for n, _ in sig)} differ from {a.ref} beyond measurement")
        print(f"  error. Shipping any of them still requires re-baselining the incumbent at that")
        print(f"  threshold -- the registry number is defined at adaptK 1.50.")


if __name__ == "__main__":
    main()
