#!/usr/bin/env python3
"""propose_arms.py — pick the next draft-head arms to run, from what has already been measured.

WHY A SEARCH AND NOT A LIST. Every arm so far was chosen by hand, one at a time, by reading the
previous result. That works while a human is in the loop and stops the moment they are not -- the
box then idles, which is the expensive failure on hardware that costs nothing per hour to run but
cannot be got back. This turns the same reasoning into a rule that runs unattended.

THE SEARCH. Coordinate descent around the best-scoring configuration: for each axis, propose the
untried neighbours of the incumbent's value, nearest-first. One axis moves at a time, which is the
same discipline the ladder applies to kernels -- one change per measurement -- and it means every
result is attributable.

TWO OBJECTIVES, NOT ONE. `tau` decides promotion; the per-category floors decide release, and
2026-08-23 produced a head that wins the first by 4.13 % and fails half of the second. So the
proposer ranks by tau but PREFERS a release-passing arm at equal tau, and reports both. A search
that optimised the mean alone would walk straight back into the head this project just refused.

EXHAUSTION. When every neighbour of the incumbent on every axis has been measured and none beat it,
the space around the optimum is characterised and the search stops. The caller then moves on --
it does not keep proposing arms to look busy.

  python3 tools/propose_arms.py --state evidence/autopilot/arms.jsonl --max 3
"""
import argparse, json, os, sys

# Axis -> ordered candidate values. The incumbent's value must appear in its axis.
AXES = {
    "beta":        [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5],
    "anchor_pow":  [1.0, 1.5, 2.0, 3.0],
    "deficit_clamp": [1.5, 2.0, 3.0, 5.0],
    "a_ce":        [0.1, 0.3, 0.5, 1.0],
    "pos_per_seq": [8, 16, 24, 32],
}
# a_tv is pinned to 1 - a_ce: the two were always swept as a simplex and an arm that breaks that
# is not comparable with any row in the registry.
NOISE = 0.035


def load(path):
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def key(cfg):
    return tuple(sorted((k, float(v)) for k, v in cfg.items() if k in AXES))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="evidence/autopilot/arms.jsonl")
    ap.add_argument("--max", type=int, default=3, help="arms per generation")
    a = ap.parse_args()

    done = load(a.state)
    scored = [d for d in done if d.get("tau") is not None]
    if not scored:
        print("EXHAUSTED: no scored arms in state -- nothing to search around", file=sys.stderr)
        return 2

    # Rank by tau; break ties toward an arm that also passes the release floors.
    best = max(scored, key=lambda d: (round(d["tau"], 4), bool(d.get("release_pass"))))
    seen = {key(d["cfg"]) for d in done}
    print("# incumbent %s  tau %.4f  release %s"
          % (best["name"], best["tau"], "PASS" if best.get("release_pass") else "FAIL"),
          file=sys.stderr)

    props = []
    for axis, values in AXES.items():
        cur = best["cfg"].get(axis)
        if cur is None or float(cur) not in [float(v) for v in values]:
            continue
        i = [float(v) for v in values].index(float(cur))
        # nearest neighbours first: one step down, one step up
        for j in (i - 1, i + 1):
            if not (0 <= j < len(values)):
                continue
            cfg = dict(best["cfg"])
            cfg[axis] = values[j]
            if axis == "a_ce":
                cfg["a_tv"] = round(1.0 - float(values[j]), 3)
            if key(cfg) in seen:
                continue
            seen.add(key(cfg))
            props.append((axis, values[j], cfg))

    if not props:
        print("EXHAUSTED: every neighbour of the incumbent on every axis has been measured",
              file=sys.stderr)
        return 1

    # One axis at a time, but spread the generation across DIFFERENT axes rather than spending it
    # all on one -- a generation that only moves beta cannot discover that pos_per_seq mattered.
    props.sort(key=lambda t: t[0])
    picked, used = [], set()
    for axis, val, cfg in props:
        if axis in used and len(picked) < a.max:
            continue
        if len(picked) >= a.max:
            break
        used.add(axis)
        picked.append((axis, val, cfg))
    for axis, val, cfg in props:                      # backfill if axes ran out
        if len(picked) >= a.max:
            break
        if (axis, val, cfg) not in picked:
            picked.append((axis, val, cfg))

    for axis, val, cfg in picked[:a.max]:
        name = "auto-%s%s" % (axis.replace("_", ""), str(val).replace(".", "p"))
        print("\t".join([name, str(cfg.get("a_ce", 0.1)), str(cfg.get("a_tv", 0.9)),
                         str(int(cfg.get("deficit", 1))), str(cfg.get("beta", 0.1)),
                         str(cfg.get("anchor_pow", 1.0)), str(cfg.get("deficit_clamp", 3.0)),
                         str(int(cfg.get("pos_per_seq", 16)))]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
