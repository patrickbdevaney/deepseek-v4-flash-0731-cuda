#!/usr/bin/env python3
"""holdout_rate.py — pooled decode rate from a blksweep log, the quantity adaptK should be fitted on.

WHY THIS EXISTS. The session re-fitted adaptK by maximising hold-out median tau, and on s1 that
selected 0.5 -- which measured the WORST decode rate of the five thresholds swept:

    adaptK   tok/verify   tok/s
      0.0        4.894    26.00
      0.5        5.043    25.12   <- max tau, min rate
      1.0        4.919    26.18
      1.5        4.879    26.78   <- max rate
      2.0        4.726    26.66

tau is tokens per verify; the objective is tokens per second, and rate = tau / (ms per verify).
Lowering the threshold widens the verify, which buys tau at a more-than-proportional cost in ms, so
the two objectives anti-correlate across this range. Fitting on tau does not merely fail to find the
optimum -- it walks away from it.

POOLED, not the mean of per-prompt rates. A mean of ratios weights an 8-token prompt the same as a
200-token one; pooling totals answers "how fast did this policy decode the hold-out", which is the
question. Both are printed because when they disagree the per-prompt spread is the story.

  python3 tools/holdout_rate.py --log <log> [--json-out f]
"""
import argparse, json, sys

def parse(path):
    rows = []
    for l in open(path, errors="ignore"):
        if not l.startswith("[blksweep]") or "|" not in l or "prompt" in l:
            continue
        p = [x.strip() for x in l.split("|")]
        try:
            rows.append((float(p[1]), float(p[2]), float(p[3])))   # tok/verify, ms/tok, tok/s
        except (ValueError, IndexError):
            continue
    return rows


def rate(path):
    rows = parse(path)
    if not rows:
        return None
    tot_tok = sum(r[0] for r in rows)
    tot_ms = sum(r[0] * r[1] for r in rows)
    return {"log": path, "n_prompts": len(rows),
            "tok_per_verify": round(tot_tok / len(rows), 4),
            "ms_per_tok": round(tot_ms / tot_tok, 3),
            "tok_s_pooled": round(1000 * tot_tok / tot_ms, 3),
            "tok_s_mean": round(sum(r[2] for r in rows) / len(rows), 3),
            "tok_s_min": round(min(r[2] for r in rows), 3),
            "tok_s_max": round(max(r[2] for r in rows), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--quiet", action="store_true", help="print only the pooled rate")
    a = ap.parse_args()
    r = rate(a.log)
    if r is None:
        sys.exit(f"no blksweep rows in {a.log}")
    if a.quiet:
        print(r["tok_s_pooled"])
    else:
        print(f"{r['log']}: {r['n_prompts']} prompts")
        print(f"  tok/verify {r['tok_per_verify']:.3f}   ms/tok {r['ms_per_tok']:.1f}")
        print(f"  tok/s pooled {r['tok_s_pooled']:.2f}   mean {r['tok_s_mean']:.2f}   "
              f"range {r['tok_s_min']:.2f}-{r['tok_s_max']:.2f}")
    if a.json_out:
        json.dump(r, open(a.json_out, "w"), indent=1)


if __name__ == "__main__":
    main()
