#!/usr/bin/env python3
"""holdout_tau.py — score the F108 secondary gate: median tau over a reasoning hold-out.

WHY THIS EXISTS. The pre-registered session-1 gate reads tau off **one** suite prompt. F108 measured
63 real reasoning prompts and found the untrained head at median tau 3.483 at the protocol's own
NGEN0=200, against that prompt's 1.85 -- the suite prompt sits below the minimum of its own
category. One prompt is a benchmark, not an estimate.

So this scores the same quantity over a 32-prompt hold-out that the head has never trained on. It
reads the engine's per-verify records rather than the summary line, which lets it truncate every
sequence at a common token budget: tau grows with generation length (F92, reproduced in F108 --
2.67 at 32 tokens rising to 4.10 at 512), so sequences must be compared at matched length or the
comparison measures how long each one happened to run.

  python3 tools/holdout_tau.py --log <eval log> [--budget 200] [--baseline 3.483] [--json-out f]
"""
import argparse, json, re, statistics as st, sys


def parse_runs(path):
    """-> list of per-sequence lists of accepted-token counts, one entry per verify."""
    seqs, cur = [], []
    pat = re.compile(r"\s*verify \d+: accepted \d+/\d+ \(K=\d+\) \+ correction -> \+(\d+) tokens")
    for line in open(path, errors="ignore"):
        m = pat.match(line)
        if m:
            cur.append(int(m.group(1)))
            continue
        if "mean tokens/verify" in line and cur:
            seqs.append(cur)
            cur = []
    if cur:
        seqs.append(cur)
    return seqs


def tau_at(runs, budget):
    tot = nv = 0
    for r in runs:
        tot += r
        nv += 1
        if tot >= budget:
            break
    return (tot / nv) if nv else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--budget", type=int, default=200, help="tokens; the protocol's NGEN0")
    ap.add_argument("--baseline", type=float, default=3.483,
                    help="untrained median tau on this hold-out (F108, n=63)")
    ap.add_argument("--go-delta", type=float, default=0.35, dest="go_delta",
                    help="tau gain that clears the bar; +0.35 is +2.7 tok/s = +11.8%%")
    ap.add_argument("--saturated-delta", type=float, default=0.15, dest="sat_delta")
    ap.add_argument("--only-last", type=int, default=0, dest="only_last",
                    help="score only the last N sequences in the log. Use this to compute the "
                         "UNTRAINED baseline from pass 1 over exactly the hold-out sequences: the "
                         "trained and untrained numbers are then the same prompts, paired, and a "
                         "partial-sample baseline cannot drift under the gate (3.483 at n=63 "
                         "became 3.458 at n=69 during development).")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    seqs = parse_runs(a.log)
    if not seqs:
        sys.exit(f"no per-verify records in {a.log} -- was this a spec-decode run?")
    if a.only_last:
        if len(seqs) < a.only_last:
            sys.exit(f"log has {len(seqs)} sequences, --only-last {a.only_last} requested")
        seqs = seqs[-a.only_last:]
    v = [x for x in (tau_at(s, a.budget) for s in seqs) if x]
    if not v:
        sys.exit("no sequence reached the budget")
    med, mean = st.median(v), st.mean(v)
    # Report the spread, not just the point. n=32 medians carry real uncertainty and a gate read off
    # a single number would be quoting a point estimate as if it were exact.
    sv = sorted(v)
    p25, p75 = sv[len(sv) // 4], sv[3 * len(sv) // 4]
    d = med - a.baseline

    print(f"hold-out tau at budget {a.budget} tokens: n={len(v)}")
    print(f"  median {med:.3f}   mean {mean:.3f}   p25 {p25:.3f}   p75 {p75:.3f}   "
          f"min {sv[0]:.3f}   max {sv[-1]:.3f}")
    print(f"  baseline (untrained) {a.baseline:.3f}   delta {d:+.3f} tau  "
          f"({7.664*d:+.2f} tok/s, {100*7.664*d/22.655:+.1f}%)")

    if d >= a.go_delta:
        verdict = "GO"
    elif d >= a.sat_delta:
        verdict = "GO_REPRICE"
    else:
        verdict = "SATURATED"
    print(f"  VERDICT (secondary): {verdict}   "
          f"[GO >= +{a.go_delta}, saturated < +{a.sat_delta}]")
    if a.json_out:
        json.dump({"log": a.log, "budget": a.budget, "n": len(v), "median": round(med, 4),
                   "mean": round(mean, 4), "p25": round(p25, 4), "p75": round(p75, 4),
                   "baseline": a.baseline, "delta": round(d, 4), "verdict": verdict},
                  open(a.json_out, "w"), indent=1)
    return 0 if verdict == "GO" else (2 if verdict == "GO_REPRICE" else 3)


if __name__ == "__main__":
    sys.exit(main())
