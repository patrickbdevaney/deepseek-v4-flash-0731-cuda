#!/usr/bin/env python3
"""accept_profile.py — per-position acceptance hazard, the quantity tau and tok/s both hide.

tau is one number summarising a chain of dependent events. A verify accepts a PREFIX: position j is
only reached if positions 0..j-1 were all accepted, so the observable

    h(j) = P(draft[j] == target[j] | positions 0..j-1 all accepted)

is what training actually moves, and tau is its cumulative consequence:

    E[accepted] = sum_j prod_{i<=j} h(i)

Two heads with the same tau can have very different h: one that accepts position 0 nearly always and
dies at position 2, versus one that is mediocre everywhere. They respond to more training, and to a
different adaptK, in opposite ways -- so "did acceptance rise" is a question about h, not tau.

CONDITIONING IS NOT OPTIONAL. A raw "how often was position 3 accepted" divides by all verifies,
including those that never reached position 3, and so conflates a drafter that fails EARLY with one
that fails AT position 3. Denominators here are verifies that actually reached the position, which
is also why the counts fall off with j and the deepest positions are the noisiest.

  python3 tools/accept_profile.py --log <log> [--vs <baseline log>] [--by-prompt]
"""
import argparse, re, sys

VP = re.compile(r"\s*verify \d+: accepted (\d+)/(\d+) \(K=(\d+)\) \+ correction -> \+(\d+) tokens \(([\d.]+) ms\)")
PP = re.compile(r"\s*\[blksweep\].*?\|.*?\|")
PROMPT = re.compile(r"prompt (\d+)")


def parse(path):
    """Verify records, tagged with the prompt index they belong to when the log marks it."""
    rows, cur = [], None
    for l in open(path, errors="ignore"):
        m = re.search(r"^\[decode\] PREFILL:", l)
        if m:
            cur = (cur + 1) if cur is not None else 0
        v = VP.match(l)
        if v:
            rows.append({"acc": int(v.group(1)), "K": int(v.group(3)),
                         "tok": int(v.group(4)), "ms": float(v.group(5)), "prompt": cur})
    return rows


def hazard(rows, maxpos=6):
    """h(j) with denominators restricted to verifies that actually reached position j."""
    out = []
    for j in range(maxpos):
        # a verify offers position j only if its width K covers it (K-1 draft positions verified)
        reached = [r for r in rows if r["K"] - 1 > j and r["acc"] >= j]
        if not reached:
            out.append(None)
            continue
        acc = sum(1 for r in reached if r["acc"] > j)
        out.append((acc / len(reached), len(reached)))
    return out


def summarise(path, maxpos=6):
    rows = parse(path)
    if not rows:
        sys.exit(f"no verify records in {path}")
    h = hazard(rows, maxpos)
    tot_tok = sum(r["tok"] for r in rows)
    tot_ms = sum(r["ms"] for r in rows)
    return {"path": path, "n": len(rows), "h": h,
            "tau": tot_tok / len(rows), "tok_s": 1000 * tot_tok / tot_ms,
            "mean_K": sum(r["K"] for r in rows) / len(rows),
            "mean_acc": sum(r["acc"] for r in rows) / len(rows), "rows": rows}


def show(s, label):
    print(f"\n{label}: {s['n']} verifies   tau {s['tau']:.3f}   {s['tok_s']:.2f} tok/s   "
          f"mean K {s['mean_K']:.2f}   mean accepted {s['mean_acc']:.3f}")
    print(f"  {'pos':>4} {'h(j)':>9} {'n reached':>11}")
    for j, e in enumerate(s["h"]):
        if e is None:
            print(f"  {j:>4} {'--':>9} {0:>11}")
        else:
            print(f"  {j:>4} {100*e[0]:>8.1f}% {e[1]:>11}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--vs", default=None, help="baseline log to diff against")
    ap.add_argument("--maxpos", type=int, default=6)
    ap.add_argument("--by-prompt", action="store_true")
    a = ap.parse_args()

    s = summarise(a.log, a.maxpos)
    show(s, "TRAINED" if a.vs else a.log)

    if a.vs:
        b = summarise(a.vs, a.maxpos)
        show(b, "BASELINE")
        print(f"\nDIFF (trained - baseline)")
        print(f"  {'pos':>4} {'d h(j)':>9}   survival contribution")
        surv_t = surv_b = 1.0
        for j, (et, eb) in enumerate(zip(s["h"], b["h"])):
            if et is None or eb is None:
                continue
            surv_t *= et[0]
            surv_b *= eb[0]
            print(f"  {j:>4} {100*(et[0]-eb[0]):>+8.1f}%   P(reach {j+1}) "
                  f"{100*surv_b:>5.1f}% -> {100*surv_t:>5.1f}%")
        print(f"\n  tau  {b['tau']:.3f} -> {s['tau']:.3f}  ({100*(s['tau']-b['tau'])/b['tau']:+.1f}%)")
        print(f"  rate {b['tok_s']:.2f} -> {s['tok_s']:.2f} tok/s "
              f"({100*(s['tok_s']-b['tok_s'])/b['tok_s']:+.1f}%)")

    if a.by_prompt:
        print(f"\nBY PROMPT")
        print(f"  {'prompt':>6} {'n':>6} {'tau':>7} {'h(0)':>8} {'h(1)':>8} {'h(2)':>8}")
        for p in sorted({r["prompt"] for r in s["rows"] if r["prompt"] is not None}):
            pr = [r for r in s["rows"] if r["prompt"] == p]
            hp = hazard(pr, 3)
            f = lambda e: f"{100*e[0]:>7.1f}%" if e else f"{'--':>8}"
            print(f"  {p:>6} {len(pr):>6} {sum(r['tok'] for r in pr)/len(pr):>7.3f} "
                  f"{f(hp[0])} {f(hp[1])} {f(hp[2])}")


if __name__ == "__main__":
    main()
