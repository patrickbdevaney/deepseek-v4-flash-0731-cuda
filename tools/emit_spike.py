#!/usr/bin/env python3
"""emit_spike.py -- DECODE_LADDER item 1.12.

THE CLAIM 1.12 IS MAKING, and the only instrument that can see it.

`compressor_emit_group` fires for the groups that COMPLETE inside a verify block:
    for j in [ctx, ctx+VB): if (j+1) % ratio == 0
The 20 odd layers are ratio-128, so that is true on roughly VB/128 of steps -- 4 % of them at
VB=5. On those steps the 20 strided layers each run two [128,512]x[512,4096] GEMMs through
`compressed_verify_step_strided`, which uses NO side stream, so every millisecond lands on the
critical path and 1.8 measured the mark at 43-50 ms against a ~1 ms baseline.

AMORTISED THAT IS BELOW THE NOISE FLOOR, WHICH IS WHY THIS TOOL EXISTS. 4 % of a 13 ms saving is
~0.5 ms/forward against a 3.5 % run-to-run spread on a ~150 ms forward. A tok/s number cannot
resolve it and a paired band can barely; the SPIKE itself is a 40 ms effect on an identified
population of steps, and it is measured by taking the conditional distribution rather than the mean.

So: classify every per-verify dprof table by
    g = #{ j in [ctx, ctx+VB) : (j+1) % ratio == 0 }
and report `cattn:compress` for g == 0 and g >= 1 SEPARATELY, per arm. The g == 0 population is the
control built into the same run -- it shares the load, the clocks and the thermal state with the
spike population, which is what makes the before/after comparison of the g >= 1 population mean
something even though the two arms are different checkpoint loads.

Usage:  python3 tools/emit_spike.py --before <log ...> --after <log ...> [--ratio 128] [--mark cattn:compress]
"""
import re, sys, statistics as st, argparse

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

def band(v):
    if not v: return "-", None
    if len(v) == 1: return f"{v[0]:.2f} (n=1)", v[0]
    return f"{st.median(v):.2f} [{min(v):.2f}, {max(v):.2f}]", st.median(v)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", nargs="+", required=True)
    ap.add_argument("--after",  nargs="+", required=True)
    ap.add_argument("--ratio", type=int, default=128)
    ap.add_argument("--marks", default="cattn:compress,TOTAL,cattn:q_proj,cattn:sparse")
    a = ap.parse_args()
    marks = [m.strip() for m in a.marks.split(",")]
    arms = {}
    for name, paths in (("before", a.before), ("after", a.after)):
        rows = [r for p in paths for r in parse(p)]
        for r in rows:
            r["g"] = sum(1 for j in range(r["ctx"], r["ctx"]+r["vb"]) if (j+1) % a.ratio == 0)
        arms[name] = rows
        print(f"{name:>6}: {len(rows)} per-verify dprof tables from {len(paths)} log(s); "
              f"{sum(1 for r in rows if r['g']>=1)} carry a ratio-{a.ratio} boundary")
    print()
    have = [m for m in marks if any(m in r for rows in arms.values() for r in rows)]
    med = {}
    for g_sel, gname in ((lambda g: g == 0, f"g==0  (no ratio-{a.ratio} emit)"),
                         (lambda g: g >= 1, f"g>=1  (THE SPIKE: {a.ratio}-emit steps)")):
        print(f"-- {gname} --")
        print(f"{'mark':>16} {'n':>4} {'before  median [min, max]':>30} {'after   median [min, max]':>30} {'delta':>10}")
        for m in have:
            out = []
            for name in ("before","after"):
                v = [r[m] for r in arms[name] if g_sel(r["g"]) and m in r]
                s, mm = band(v); out.append((s, mm, len(v)))
            d = (out[1][1]-out[0][1]) if (out[0][1] is not None and out[1][1] is not None) else None
            med[(m, gname[:4])] = (out[0][1], out[1][1])
            print(f"{m:>16} {out[0][2]:>4} {out[0][0]:>30} {out[1][0]:>30} "
                  f"{('%+.2f'%d) if d is not None else '-':>10}")
        print()
    # The headline: the excess a ratio-N emit step carries over a non-emit step, per arm. This is
    # differenced WITHIN an arm first, so a between-load offset cancels before the arms are compared.
    print(f"-- EXCESS of a g>=1 step over the g==0 median, WITHIN each arm (a between-load offset cancels here) --")
    for m in have:
        b0, a0 = med.get((m, "g==0"), (None,None)); b1, a1 = med.get((m, "g>=1"), (None,None))
        if None in (b0,a0,b1,a1): continue
        eb, ea = b1-b0, a1-a0
        rel = f"{eb/ea:.2f}x" if ea else "-"
        print(f"{m:>16}   before {eb:+8.2f} ms    after {ea:+8.2f} ms    delta {ea-eb:+8.2f} ms   {rel}")

main()
