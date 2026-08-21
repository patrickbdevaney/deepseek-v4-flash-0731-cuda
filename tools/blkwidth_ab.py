#!/usr/bin/env python3
"""blkwidth_ab.py -- DECODE_LADDER 2.1. Read a palindromic block-width sweep out of ONE load.

WHY NOT `promote_head.parse_eval`. That parser reads the final `[blksweep]` table, which the engine
prints only after every point has finished, and it keys on (block, prompt) with no notion of a
repeat. This sweep runs each (prompt, width) TWICE by design and must survive a partial run, so it
reads the per-point lines the engine emits as it goes and keeps them IN ORDER. The run index is the
whole point: within one load this box drifts several percent with run order, and the palindrome is
what cancels it.

THE STATISTIC. For each (prompt, width) the two mirrored occurrences are averaged -- their run
indices sum to a constant, so any drift that is linear in run index cancels exactly. Widths are then
compared to the baseline width PAIRED BY PROMPT, because the suite prompts span 18-35 tok/s and an
unpaired difference of means is dominated by which prompts are in the suite. Bands are 2 SE of the
per-prompt differences.

`tau` is computed from the engine's own integer counts (tokens / verifies), not from the 2-decimal
summary field: it is an exact draft/target comparison and deserves to be reported exactly.

  python3 tools/blkwidth_ab.py --log <sweep log> [--genout <file>] [--baseline 6]
"""
import argparse, math, re, sys
from collections import defaultdict

RE_START = re.compile(r"^\[spec\] decoding (\d+) tokens \(block=(\d+), draft passes=(\d+), "
                      r"adaptK=([\d.]+), prompt=(\d+) s=(\d+)\)")
RE_GEN   = re.compile(r"^\[spec\] generated (\d+) tokens over (\d+) verifies: "
                      r"mean tokens/verify = ([\d.]+) \(block=(\d+), max (\d+)\)")
RE_RATE  = re.compile(r"^\[spec\] SPEC-DECODE: ([\d.]+) ms/tok = ([\d.]+) tok/s")
RE_LOSS  = re.compile(r"^\[spec\] LOSSLESS GATE: (.*)$")
RE_VER   = re.compile(r"^\s*verify (\d+): accepted \d+/\d+ \(K=(\d+)\) \+ correction -> "
                      r"\+(\d+) tokens \(([\d.]+) ms\)\s+cpos=(\d+)")
RE_WARM  = re.compile(r"^\[decode\] WARM decode: ([\d.]+) ms/tok = ([\d.]+) tok/s")


def band(xs):
    n = len(xs)
    if not n:
        return 0.0, 0.0
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m, sd / math.sqrt(n)


def parse(path):
    pts, cur, base_ar, gates = [], None, None, []
    for line in open(path, errors="ignore"):
        m = RE_WARM.match(line)
        if m and base_ar is None:
            base_ar = float(m.group(2))
            continue
        m = RE_START.match(line)
        if m:
            cur = {"idx": len(pts), "blk": int(m.group(2)), "prompt": int(m.group(5)),
                   "adaptK": float(m.group(4)), "lossless": None}
            continue
        if cur is None:
            continue
        m = RE_VER.match(line)
        if m:
            cur.setdefault("rounds", []).append(
                {"n": int(m.group(1)), "K": int(m.group(2)), "acc": int(m.group(3)),
                 "ms": float(m.group(4)), "cpos": int(m.group(5))})
            continue
        m = RE_GEN.match(line)
        if m:
            cur["ntok"], cur["nver"] = int(m.group(1)), int(m.group(2))
            cur["tau"] = cur["ntok"] / cur["nver"]
            cur["vkcap"] = int(m.group(5))
            continue
        m = RE_LOSS.match(line)
        if m:
            cur["lossless"] = "PASS" if "-> PASS" in m.group(1) else "FAIL"
            gates.append((cur["blk"], cur["prompt"], cur["lossless"]))
            continue
        m = RE_RATE.match(line)
        if m:
            cur["ms_tok"], cur["tok_s"] = float(m.group(1)), float(m.group(2))
            pts.append(cur)
            cur = None
    return pts, base_ar, gates


def parse_genout(path):
    """-> list of id-lists, one per sweep point, in sweep order."""
    try:
        return [[int(v) for v in ln.split(",") if v.strip()] for ln in open(path) if ln.strip()]
    except OSError:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--genout", default=None)
    ap.add_argument("--baseline", type=int, default=6, help="the shipped width, i.e. the before-arm")
    ap.add_argument("--drop-first", type=int, default=1, dest="drop_first",
                    help="warm-up points to discard; the first run of a batch is always the slowest")
    a = ap.parse_args()

    pts, base_ar, gates = parse(a.log)
    if not pts:
        print("no completed sweep points in %s -- NOT evidence" % a.log)
        return 1
    warm, pts = pts[:a.drop_first], pts[a.drop_first:]

    print("blkwidth sweep -- DECODE_LADDER 2.1")
    print("  log            : %s" % a.log)
    print("  points          : %d completed (+%d warm-up discarded)" % (len(pts), len(warm)))
    print("  base AR (M=1)   : %s tok/s" % (("%.2f" % base_ar) if base_ar else "?"))
    nfail = sum(1 for _, _, v in gates if v != "PASS")
    print("  LOSSLESS gate   : %d/%d PASS%s" % (len(gates) - nfail, len(gates),
                                                "" if not nfail else "   <<< FAIL"))
    if nfail:
        for b, p, v in gates:
            if v != "PASS":
                print("      block %d prompt %d -> %s" % (b, p, v))

    # --- mirrored average -------------------------------------------------
    reps = defaultdict(list)          # (prompt, blk) -> [point, ...] in run order
    for r in pts:
        reps[(r["prompt"], r["blk"])].append(r)
    widths = sorted({r["blk"] for r in pts})
    prompts = sorted({r["prompt"] for r in pts})
    suite = [p for p in prompts if p != 0]     # F96: prompt 0 is the control, not the suite

    def avg(prompt, blk, key):
        rs = reps.get((prompt, blk)) or []
        return (sum(r[key] for r in rs) / len(rs)) if rs else None

    # within-load drift, measured rather than assumed: second occurrence minus first
    d2 = [rs[1]["tok_s"] - rs[0]["tok_s"] for rs in reps.values() if len(rs) >= 2]
    if d2:
        m, se = band(d2)
        print("  within-window drift (2nd occurrence - 1st, n=%d): %+.3f +/- %.3f tok/s "
              "-- cancelled by averaging the pair" % (len(d2), m, 2 * se))
    incomplete = [k for k, v in reps.items() if len(v) < 2]
    if incomplete:
        print("  NOTE: %d (prompt,width) cells have only one occurrence and are NOT drift-cancelled: %s"
              % (len(incomplete), sorted(incomplete)))

    # --- per-width table --------------------------------------------------
    print("\n| BLK | ceiling | suite tau | suite tok/s | vs BLK=%d | p0 tau | p0 tok/s | tau/ceiling |"
          % a.baseline)
    print("|---|---|---|---|---|---|---|---|")
    ref = {p: avg(p, a.baseline, "tok_s") for p in prompts}
    rows = {}
    for b in widths:
        st = [avg(p, b, "tau") for p in suite if avg(p, b, "tau") is not None]
        sr = [avg(p, b, "tok_s") for p in suite if avg(p, b, "tok_s") is not None]
        if not sr:
            continue
        mt, mr = sum(st) / len(st), sum(sr) / len(sr)
        rows[b] = (mt, mr)
        base = sum(v for p, v in ref.items() if p in suite and v is not None)
        base /= max(1, len([p for p in suite if ref.get(p) is not None]))
        print("| %d | %d | %.4f | %.4f | %+.2f %% | %s | %s | %.3f |"
              % (b, b + 1, mt, mr, 100.0 * (mr / base - 1.0) if base else 0.0,
                 ("%.4f" % avg(0, b, "tau")) if avg(0, b, "tau") is not None else "-",
                 ("%.4f" % avg(0, b, "tok_s")) if avg(0, b, "tok_s") is not None else "-",
                 mt / (b + 1)))

    # --- paired against the shipped width ---------------------------------
    print("\nPAIRED vs BLK=%d, over the %d suite prompts (prompt 0 excluded, F96);"
          " bands are 2 SE of the per-prompt differences" % (a.baseline, len(suite)))
    print("| BLK | d tok/s | d % | legs + | d tau |")
    print("|---|---|---|---|---|")
    for b in widths:
        if b == a.baseline:
            continue
        dr, dp, dt = [], [], []
        for p in suite:
            x, y = avg(p, a.baseline, "tok_s"), avg(p, b, "tok_s")
            xa, ya = avg(p, a.baseline, "tau"), avg(p, b, "tau")
            if None in (x, y, xa, ya):
                continue
            dr.append(y - x); dp.append(100.0 * (y / x - 1.0)); dt.append(ya - xa)
        if not dr:
            continue
        mr, sr = band(dr); mp, sp = band(dp); mt, stau = band(dt)
        print("| %d | %+.3f +/- %.3f | %+.2f +/- %.2f %% | %d/%d | %+.4f +/- %.4f |"
              % (b, mr, 2 * sr, mp, 2 * sp, sum(1 for v in dr if v > 0), len(dr), mt, 2 * stau))

    # --- per-prompt optimum ----------------------------------------------
    print("\nPer-prompt optimum (mirrored-averaged tok/s):")
    print("| prompt | best BLK | best tok/s | BLK=%d tok/s | gain | tau at best | tau at %d |"
          % (a.baseline, a.baseline))
    print("|---|---|---|---|---|---|---|")
    for p in prompts:
        cand = [(avg(p, b, "tok_s"), b) for b in widths if avg(p, b, "tok_s") is not None]
        if not cand:
            continue
        bv, bb = max(cand)
        rv = avg(p, a.baseline, "tok_s")
        print("| %d%s | %d | %.2f | %s | %s | %.3f | %s |"
              % (p, " *(control)*" if p == 0 else "", bb, bv,
                 ("%.2f" % rv) if rv else "-",
                 ("%+.2f %%" % (100.0 * (bv / rv - 1.0))) if rv else "-",
                 avg(p, bb, "tau"),
                 ("%.3f" % avg(p, a.baseline, "tau")) if avg(p, a.baseline, "tau") is not None else "-"))

    # --- the cost model behind whatever optimum this finds -----------------
    # WHY THIS IS HERE AND NOT IN A SEPARATE INSTRUMENT. The engine already prints one line per
    # verify round with its realised K and its wall time, so the two prices that decide the optimum
    # are in the log this sweep already wrote. `d ms / d BLK` is what a wider block costs on the
    # DRAFT side (the MTP block runs at M=BLK whether or not the extra proposals are verified);
    # `d ms / d K` is what one more VERIFIED position costs -- the quantity
    # DECODE_ZENITH_FINDINGS 3.2 said was ~free because `k_topk_verify<<<K,32>>>` ran one active
    # thread per block, and the quantity ladder 1.1/1.2 changed. Fitted per prompt so each prompt is
    # its own control, then banded across prompts.
    def ols(rows):
        n = len(rows)
        X = [[1.0, float(r[0]), float(r[1])] for r in rows]
        y = [float(r[2]) for r in rows]
        A = [[sum(X[i][a] * X[i][b] for i in range(n)) for b in range(3)] for a in range(3)]
        v = [sum(X[i][a] * y[i] for i in range(n)) for a in range(3)]
        for c in range(3):                                   # gaussian elimination, 3x3
            pv = max(range(c, 3), key=lambda r2: abs(A[r2][c]))
            if abs(A[pv][c]) < 1e-12:
                return None
            A[c], A[pv] = A[pv], A[c]; v[c], v[pv] = v[pv], v[c]
            for r2 in range(3):
                if r2 == c:
                    continue
                f = A[r2][c] / A[c][c]
                for c2 in range(3):
                    A[r2][c2] -= f * A[c][c2]
                v[r2] -= f * v[c]
        return [v[c] / A[c][c] for c in range(3)]

    SKIP = 2      # the first rounds of a point carry the re-prefill's tail
    cb, ck, nr = [], [], 0
    for p in prompts:
        rows = []
        for r in pts:
            if r["prompt"] != p:
                continue
            for rd in r.get("rounds", [])[SKIP:]:
                rows.append((r["blk"], rd["K"], rd["ms"]))
        nr += len(rows)
        if len(rows) < 20:
            continue
        c = ols(rows)
        if c:
            cb.append(c[1]); ck.append(c[2])
    if cb:
        mb, sb = band(cb); mk, sk = band(ck)
        print("\nROUND COST MODEL  ms_round ~ c0 + cB*BLK + cK*K   (%d rounds, %d prompts fitted"
              " separately, first %d rounds of each point dropped)" % (nr, len(cb), SKIP))
        print("  cB  (one more DRAFTED position, verified or not) : %+.4f +/- %.4f ms" % (mb, 2 * sb))
        print("  cK  (one more VERIFIED position)                 : %+.4f +/- %.4f ms" % (mk, 2 * sk))
        print("  break-even: a wider block pays only if the extra acceptance it buys exceeds its own")
        print("  draft cost -- at cB above, one extra block position costs %.4f ms/round." % mb)

    # --- emitted-id divergence -------------------------------------------
    if a.genout:
        seqs = parse_genout(a.genout)
        print("\nEmitted ids across widths (DSV4_GENOUT, %d sequences).  Changing BLK changes M in "
              "the verify\nforward and therefore MoE atomic reduction order, so identity is NOT "
              "guaranteed by construction --\nthis MEASURES the divergence instead of assuming it." % len(seqs))
        if len(seqs) < len(pts) + len(warm):
            print("  NOTE: %d sequences for %d points -- truncated run." % (len(seqs), len(pts) + len(warm)))
        # sequences are appended in sweep order, warm-up included
        allpts = warm + pts
        n = min(len(seqs), len(allpts))
        bykey = defaultdict(list)
        for i in range(n):
            bykey[allpts[i]["prompt"]].append((allpts[i]["blk"], seqs[i]))
        print("| prompt | sequences | identical to the BLK=%d reference | first divergence |" % a.baseline)
        print("|---|---|---|---|")
        for p in sorted(bykey):
            group = bykey[p]
            refseq = next((s for b, s in group if b == a.baseline), group[0][1])
            same, firsts = 0, []
            for b, s in group:
                k = next((i for i in range(min(len(s), len(refseq))) if s[i] != refseq[i]), None)
                if k is None and len(s) == len(refseq):
                    same += 1
                else:
                    firsts.append((b, k if k is not None else min(len(s), len(refseq))))
            print("| %d | %d | %d | %s |"
                  % (p, len(group), same,
                     "-" if not firsts else ", ".join("BLK %d@%d" % t for t in sorted(firsts))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
