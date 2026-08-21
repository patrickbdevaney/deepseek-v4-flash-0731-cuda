#!/usr/bin/env python3
"""lhash_pairs.py — ladder 1.10. ALL-PAIRS version of lhash_compare's `--within`.

1.9 established that `build/decode`'s prefill stops reproducing itself at >= 192 positions and that
the residue is inside `compressed_attn_forward`. 1.10 has to NAME the kernel, and the instrument for
that is an ablation sweep: run the same 8-point repeat sweep under one env flag per candidate kernel
and see which flag makes the divergence go away. That needs a sharper read-out than `--within` gave:

  * `--within` compares grp[0] against every other point ONLY. Point 0 is not interchangeable with
    the rest -- the R logs show `scratch0=...d2000000` at point 0 and `...d2200000` at points 1..7,
    i.e. the FIRST point of a process allocates at a different address than every later one. Every
    number 1.9 reported therefore had an address change riding along with the repeat. ALL PAIRS
    excludes that: pairs drawn from points 1,2,3 share an allocator state exactly.
  * A bisection needs a RATE, not a verdict. "12 of 12 pairs diverge" and "1 of 12" are different
    findings and `--within`'s one-line VERDICT collapses them. An ablation that merely perturbs
    timing moves the rate; one that removes the racing kernel takes it to zero.
  * The FIRST-DIFF LAYER's compression ratio is the discriminator this item turns on. Layers
    alternate ratio 4 / ratio 128 from layer 2, and only ratio-4 layers run `indexer_forward` and
    the overlapping compressor at all -- so a histogram over the ratio of the first differing layer
    separates "the indexer races" from "something every compressed layer runs races".

  python3 tools/lhash_pairs.py <label=log> [<label=log> ...]

Points are grouped by their PSp (parsed from the run's own `[hash] point <p> prompt <q> PSp=<n>`
line, not supplied by hand as `--within` required). Pairs inside one log are WITHIN-process; pairs
across two logs of the same arm are CROSS-process. Exit 0 iff no pair differs.
"""
import re, sys, collections, itertools

LPAT = re.compile(r"\[lhash\] point (\d+) layer\s+(\d+) ratio\s+(\d+) : (\w+)")
HPAT = re.compile(r"\[hash\] point (\d+) prompt (\d+) PSp=(\d+)")

def load(path):
    h, psp = {}, {}
    for ln in open(path, errors="replace"):
        m = LPAT.match(ln)
        if m:
            h[(int(m.group(1)), int(m.group(2)))] = (int(m.group(3)), m.group(4)); continue
        m = HPAT.match(ln)
        if m:
            psp[int(m.group(1))] = int(m.group(3))
    return h, psp

def main(argv):
    logs = []
    for a in argv:
        lab, _, path = a.partition("=")
        if not path: lab, path = path or a, a
        h, psp = load(path)
        if not h:
            print(f"MALFORMED: {path} has no [lhash] lines (was DSV4_HASH=2 set?)"); return 2
        logs.append((lab or path, h, psp))

    # (label, point) -> (psp, {layer: hash}, {layer: ratio})
    inst = {}
    for lab, h, psp in logs:
        for (p, L), (r, v) in h.items():
            inst.setdefault((lab, p), [psp.get(p, -1), {}, {}])
            inst[(lab, p)][1][L] = v
            inst[(lab, p)][2][L] = r

    groups = collections.defaultdict(list)
    for k, v in inst.items(): groups[v[0]].append(k)

    anybad = False
    for g in sorted(groups):
        keys = sorted(groups[g])
        nlay = max(len(inst[k][1]) for k in keys)
        cats = {"within": [0, 0], "cross": [0, 0]}      # [divergent, total]
        firsts, ratios = collections.Counter(), collections.Counter()
        for a, b in itertools.combinations(keys, 2):
            cat = "within" if a[0] == b[0] else "cross"
            ha, hb = inst[a][1], inst[b][1]
            diff = [L for L in range(nlay) if ha.get(L) != hb.get(L)]
            cats[cat][1] += 1
            if diff:
                cats[cat][0] += 1; anybad = True
                firsts[diff[0]] += 1
                ratios[inst[a][2].get(diff[0], -1)] += 1
        print(f"  PSp {g}: within-process {cats['within'][0]}/{cats['within'][1]} diverge   "
              f"cross-process {cats['cross'][0]}/{cats['cross'][1]} diverge   (of {nlay} layers)")
        if firsts:
            print(f"      first-diff layer: {dict(sorted(firsts.items()))}")
            print(f"      its ratio:        {dict(sorted(ratios.items()))}"
                  f"   <- ratio 4 runs indexer_forward + overlap compressor; ratio 128 runs neither")
    print("VERDICT: " + ("DIVERGES" if anybad else "CLEAN -- every pair byte-identical at every layer"))
    return 1 if anybad else 0

if __name__ == "__main__":
    if len(sys.argv) < 2: print(__doc__); sys.exit(2)
    sys.exit(main(sys.argv[1:]))
