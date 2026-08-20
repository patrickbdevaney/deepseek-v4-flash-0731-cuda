#!/usr/bin/env python3
"""lhash_compare.py — ladder 1.9. Compare two runs' PER-LAYER PREFILL hashes and report, per sweep
point, the FIRST LAYER whose hidden state differs.

`DSV4_HASH=2` (src/decode.cu) prints one line per prefill layer:

    [lhash] point <p> layer <L> ratio <r> : <fnv64 of the hidden state after that layer>

The prefill is a straight chain, so the first differing layer is the first layer that computed
something different; every layer after it is only inheriting. `ratio` is that layer's compression
ratio -- 0 is a pure sliding layer, non-zero is a compressed layer -- so the table also says which
KIND of layer broke first, which is the thing that names a subsystem.

  python3 tools/lhash_compare.py <A.log> <B.log> [PSp1,PSp2,...]
  python3 tools/lhash_compare.py --within <log> [PSp1,PSp2,...]

`--within` compares a single run's sweep points AGAINST EACH OTHER rather than against a second
run. Point it at a sweep whose entries are the same prompt repeated, and it answers the question a
two-process comparison cannot: is the nondeterminism a property of the process (addresses, whatever
a fresh cudaMalloc held) or of the computation? Four identical points that disagree inside ONE
process are a race.

The optional third argument labels each sweep point with its prefill length, because the whole
question 1.9 asks is at WHICH LENGTH reproducibility stops. Exit 0 if every layer of every point
matches, 1 otherwise.
"""
import re, sys

PAT = re.compile(r"\[lhash\] point (\d+) layer\s+(\d+) ratio\s+(\d+) : (\w+)")

def load(path):
    d = {}
    for ln in open(path, errors="replace"):
        m = PAT.match(ln)
        if m:
            d[(int(m.group(1)), int(m.group(2)))] = (int(m.group(3)), m.group(4))
    return d

def within(path, psp):
    d = load(path)
    if not d:
        print("MALFORMED: no [lhash] lines (was DSV4_HASH=2 set?)"); return 2
    pts = sorted({k[0] for k in d})
    groups = {}
    for pt in pts:
        groups.setdefault(psp[pt] if pt < len(psp) else pt, []).append(pt)
    bad = False
    for lab, grp in groups.items():
        if len(grp) < 2:
            print(f"  PSp {lab}: only one point, nothing to compare within"); continue
        ref = grp[0]
        n = sum(1 for k in d if k[0] == ref)
        for g in grp[1:]:
            first = next((L for L in range(n) if d[(ref, L)][1] != d.get((g, L), (0, None))[1]), None)
            nd = sum(1 for L in range(n) if d[(ref, L)][1] != d.get((g, L), (0, None))[1])
            if nd:
                bad = True
            print(f"  PSp {lab}: point {ref} vs point {g}: first DIFF layer "
                  f"{first if first is not None else 'none'}  ({nd}/{n} differ)")
    print("VERDICT: " + ("identical sweep points inside ONE process DISAGREE -> a race, not "
                         "per-process state" if bad else
                         "every repeat of every point is byte-identical within this process"))
    return 1 if bad else 0

def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--within":
        psp = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else []
        return within(sys.argv[2], psp)
    if len(sys.argv) < 3:
        print(__doc__); return 2
    a, b = load(sys.argv[1]), load(sys.argv[2])
    psp = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else []
    if not a or not b:
        print("MALFORMED: no [lhash] lines on one side (was DSV4_HASH=2 set?)"); return 2
    pts = sorted({k[0] for k in a} & {k[0] for k in b})
    print(f"{'pt':>3} {'PSp':>6} {'layers':>7} {'first DIFF layer':>17} {'its ratio':>10} {'#layers DIFF':>13}")
    bad = False
    for pt in pts:
        n = sum(1 for k in a if k[0] == pt)
        first, nd = None, 0
        for L in range(n):
            if (pt, L) not in b:
                continue
            if a[(pt, L)][1] != b[(pt, L)][1]:
                nd += 1
                if first is None:
                    first = L
        if nd:
            bad = True
        lab = str(psp[pt]) if pt < len(psp) else "?"
        rt = a[(pt, first)][0] if first is not None else None
        print(f"{pt:>3} {lab:>6} {n:>7} {(str(first) if first is not None else 'none'):>17} "
              f"{(str(rt) if rt is not None else '-'):>10} {nd:>6}/{n}")
    print("VERDICT: " + ("the prefill does NOT reproduce itself at every point above"
                         if bad else "every prefill layer of every point is byte-identical"))
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main())
