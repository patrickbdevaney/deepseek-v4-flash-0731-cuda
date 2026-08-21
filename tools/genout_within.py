#!/usr/bin/env python3
"""genout_within.py — ladder 1.10. Does ONE process reproduce ITSELF on a repeated sweep point?

`genout_compare.py` compares two DSV4_GENOUT files, which means two processes and therefore two
100.40 GiB checkpoint loads. On this box that is the expensive and the FRAGILE way to ask the
question -- 1.10's first two attempts at it lost an arm to the memguard at MemAvailable 1.0 GB,
because 100.40 GiB of weights in a 122.8 GiB pool leaves the loader ~7 GiB of headroom and the
memguard also fires on the fall RATE. It is also the WEAKER question: two loads differ in their
allocator state, so "the two files differ" has an answer that is not about the computation.

Point the sweep at the SAME prompt several times in ONE process and every pair is exactly the same
input on exactly the same state. 1.0's original observation is in this form -- "identical points
give different token sequences" -- so this is the instrument that observation deserved.

  python3 tools/genout_within.py <genout.txt> <run.log>

GROUPING IS TAKEN FROM THE RUN LOG, NOT GUESSED. `build/decode` prints
`[genout] wrote <p> prompt + <g> generated ids` once per point, in order, so the exact prompt length
of each line is known and two points are repeats iff their PROMPTS are equal. Grouping by a fixed
prefix instead would merge two different prompts that share an opening (the winladder suite shares
its first eight ids across every entry), and grouping by total length would SPLIT a genuine pair
whose divergence ended generation early -- a false clean, which is the one failure mode a
determinism gate must not have. Exit 0 iff every pair inside every group is identical at every
position.
"""
import re, sys, collections, itertools

GPAT = re.compile(r"\[genout\] wrote (\d+) prompt \+ (\d+) generated ids")


def main(argv):
    path = argv[0]
    lines = [l.strip() for l in open(path) if l.strip()]
    pts = [[int(x) for x in l.split(",")] for l in lines]
    if not pts:
        print(f"MALFORMED: {path} has no points"); return 2

    plens = []
    if len(argv) > 1:
        for ln in open(argv[1], errors="replace"):
            m = GPAT.search(ln)
            if m: plens.append(int(m.group(1)))
    if len(plens) != len(pts):
        print(f"MALFORMED: {len(pts)} genout point(s) but {len(plens)} '[genout] wrote' line(s) in "
              f"{argv[1] if len(argv) > 1 else '<no log given>'} -- the grouping would be a guess, "
              f"so nothing is reported."); return 2

    groups = collections.defaultdict(list)
    for i, p in enumerate(pts):
        groups[tuple(p[:plens[i]])].append(i)

    print(f"{path}: {len(pts)} point(s) in {len(groups)} repeat group(s)")
    bad = 0
    for k, idxs in groups.items():
        n = len(idxs)
        if n < 2:
            print(f"  prompt of {len(k)} ids: only one point, nothing to compare within"); continue
        npairs = 0; ndiff = 0; firsts = []
        for a, b in itertools.combinations(idxs, 2):
            npairs += 1
            pa, pb = pts[a], pts[b]
            m = min(len(pa), len(pb))
            first = next((i for i in range(m) if pa[i] != pb[i]), None)
            if first is None and len(pa) != len(pb): first = m
            if first is not None:
                ndiff += 1; firsts.append(first)
        bad += ndiff
        lens = {len(pts[i]) - len(k) for i in idxs}
        note = (f"first divergence at generated position(s) "
                f"{sorted({f - len(k) for f in firsts})}") if firsts else "identical at every position"
        print(f"  prompt of {len(k)} ids, {n} repeats, {sorted(lens)} generated: "
              f"{ndiff}/{npairs} pairs differ -- {note}")
    print("VERDICT: " + ("the process DOES NOT reproduce itself" if bad else
                         "every repeat of every point is byte-identical"))
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) < 3: print(__doc__); sys.exit(2)
    sys.exit(main(sys.argv[1:]))
