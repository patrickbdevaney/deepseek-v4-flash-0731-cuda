#!/usr/bin/env python3
"""qproj_bimodal.py -- DECODE_LADDER item 1.8.

Tests ONE falsifiable mechanism for the `cattn:q_proj` / `q:wq_a` bimodality 0.4 found, against
per-verify dprof tables in a server log.

THE MECHANISM.  `compressed_verify_step_indexer` (kernels/compressed_decode.cu) forks the two
`compressor_emit_group` calls onto the side stream `g_side` BEFORE it issues `build_qKV` on the
main stream (Finding 55/56, ATTN_SPLIT).  The emits fire only for groups that COMPLETE inside the
verify block:  `for j in [pos, pos+K): if (j+1) % ratio == 0`.  So on a step where a group boundary
falls in the block, the 21 ratio-4 indexer layers each run two emits CONCURRENTLY with build_qKV --
and `cattn:q_proj`/`q:wq_a` are timed by events on the main stream, so they absorb that contention.
On a step where no boundary falls in the block the side stream is empty and the same GEMM is timed
alone.

THE PREDICTION, which is arithmetic and not a fit:  a step is "slow" IFF
    exists j in [ctx, ctx+VB) with (j+1) % 4 == 0
and the excess is LINEAR in the NUMBER of such j (a block wide enough to straddle two boundaries
must cost twice the excess).  `ctx` in the dprof tag is `cpos`, the block's first position
(src/engine.cu), and VB is the realised verify width, so both come straight out of the tag.

Usage:  python3 tools/qproj_bimodal.py <server.log> [more.log ...]
"""
import re, sys, statistics as st

RATIO = 4   # the DSA indexer layers; the 20 ratio-128 layers use the strided path (no side stream)

def parse(path):
    rows, cur = [], None
    for line in open(path, errors="replace"):
        m = re.match(r"\[dprof\] ctx=(\d+) VB=(\d+) verify=(\d+)", line)
        if m:
            cur = {"ctx": int(m.group(1)), "vb": int(m.group(2)), "verify": int(m.group(3)), "src": path}
            rows.append(cur); continue
        if cur is None: continue
        m = re.match(r"\[dprof\] (.*?)\s{2,}([0-9.]+)\s+(-?[0-9.]+)%\s+(\d+)\s*$", line)
        if m: cur[m.group(1).strip()] = float(m.group(2)); continue
        m = re.match(r"\[dprof\] (TOTAL)\s+([0-9.]+)", line)
        if m: cur["TOTAL"] = float(m.group(2))
    return rows

def ngroups(r):
    return sum(1 for j in range(r["ctx"], r["ctx"] + r["vb"]) if (j + 1) % RATIO == 0)

def band(v):
    if len(v) < 2: return f"{v[0]:.2f}" if v else "-"
    return f"{st.median(v):.2f} [{min(v):.2f}, {max(v):.2f}]"

def main(paths):
    rows = [r for p in paths for r in parse(p)]
    for r in rows: r["g"] = ngroups(r)
    print(f"{len(rows)} per-verify dprof tables from {len(paths)} log(s)")
    MARKS = ["q:wq_a", "cattn:compress", "cattn:q_proj", "TOTAL"]
    have = [m for m in MARKS if any(m in r for r in rows)]

    print("\nBY PREDICTED NUMBER OF COMPLETING GROUPS  g = #{ j in [ctx, ctx+VB) : (j+1) %% %d == 0 }" % RATIO)
    print(f"{'g':>2} {'n':>4} " + " ".join(f"{m:>26}" for m in have))
    for g in sorted({r["g"] for r in rows}):
        sub = [r for r in rows if r["g"] == g]
        print(f"{g:>2} {len(sub):>4} " + " ".join(f"{band([r[m] for r in sub if m in r]):>26}" for m in have))

    # SEPARATION: is g a perfect classifier of the bimodality?  Threshold midway between the two
    # medians of the g=0 and g>=1 populations; report every misclassified sample explicitly.
    if "q:wq_a" in have:
        lo = [r["q:wq_a"] for r in rows if r["g"] == 0]
        hi = [r["q:wq_a"] for r in rows if r["g"] >= 1]
        if lo and hi:
            thr = 0.5 * (max(lo) + min(hi))
            bad = [r for r in rows if (r["q:wq_a"] > thr) != (r["g"] >= 1)]
            print(f"\nq:wq_a  g=0: max {max(lo):.2f} (n={len(lo)})   g>=1: min {min(hi):.2f} (n={len(hi)})")
            print(f"threshold {thr:.2f} -> {len(rows)-len(bad)}/{len(rows)} classified correctly, "
                  f"{len(bad)} misclassified" + (" -- SEPARATION IS PERFECT" if not bad else ""))
            for r in bad:
                print(f"  MISS ctx={r['ctx']} VB={r['vb']} g={r['g']} q:wq_a={r['q:wq_a']:.2f}")
            if not bad and max(lo) < min(hi):
                print(f"gap between the two populations: {min(hi)-max(lo):.2f} ms "
                      f"({min(hi)/max(lo):.2f}x), no overlap at all")

    # LINEARITY IN g: the excess must scale with the number of emits, not just its presence.
    print("\nEXCESS OVER THE g=0 MEDIAN, ms  (must be ~linear in g if the emits are the cause)")
    base = {m: st.median([r[m] for r in rows if r["g"] == 0 and m in r]) for m in have}
    print(f"{'g':>2} " + " ".join(f"{m:>16}" for m in have) + "   q_proj+compress")
    for g in sorted({r["g"] for r in rows}):
        sub = [r for r in rows if r["g"] == g]
        ex = {m: st.median([r[m] for r in sub if m in r]) - base[m] for m in have}
        s = ex.get("cattn:q_proj", 0) + ex.get("cattn:compress", 0)
        print(f"{g:>2} " + " ".join(f"{ex[m]:>16.2f}" for m in have) + f"   {s:>15.2f}")

    # THE SECOND RATIO, as an independent check of the same mechanism. The 20 strided layers emit at
    # (j+1)%128==0 through `compressed_verify_step_strided`, which does NOT use a side stream at all
    # -- so a ratio-128 boundary in the block must show up as a large `cattn:compress` and NOT as a
    # `q:wq_a` swing, in BOTH arms.
    if "cattn:compress" in have:
        out = [r for r in rows if r.get("cattn:compress", 0) > 20]
        pred = [r for r in out if any((j + 1) % 128 == 0 for j in range(r["ctx"], r["ctx"] + r["vb"]))]
        print(f"\nRATIO-128 CHECK: {len(out)} sample(s) with cattn:compress > 20 ms; "
              f"{len(pred)}/{len(out)} have a ratio-128 group boundary in the block"
              + (" -- ALL of them" if out and len(pred) == len(out) else ""))
        for r in out:
            hits = [j for j in range(r["ctx"], r["ctx"] + r["vb"]) if (j + 1) % 128 == 0]
            print(f"  ctx={r['ctx']} VB={r['vb']} compress={r['cattn:compress']:.1f} ms  "
                  f"ratio-128 boundaries in block: {hits}  q:wq_a={r.get('q:wq_a', float('nan')):.2f}")

    print("\nBY CONTEXT POINT (the swing must be present at every context if it is not a context effect)")
    print(f"{'ctx~':>6} {'n':>4} {'g=0 q:wq_a':>18} {'g>=1 q:wq_a':>18} {'swing':>8}")
    for pt in sorted({round(r["ctx"], -3) for r in rows}):
        sub = [r for r in rows if round(r["ctx"], -3) == pt and "q:wq_a" in r]
        lo = [r["q:wq_a"] for r in sub if r["g"] == 0]; hi = [r["q:wq_a"] for r in sub if r["g"] >= 1]
        sw = f"{st.median(hi)/st.median(lo):.2f}x" if lo and hi else "-"
        print(f"{pt:>6} {len(sub):>4} {band(lo):>18} {band(hi):>18} {sw:>8}")

if __name__ == "__main__":
    if len(sys.argv) < 2: print(__doc__); sys.exit(2)
    main(sys.argv[1:])
