#!/usr/bin/env python3
"""paired_band.py — the per-leg paired band that DECODE_LADDER's ratchet is stated in.

`mainkv_ab_compare.py` prints medians per (kind, target). That is the right table for a change
whose size is comparable to the run-to-run spread of a POINT, but it throws away the pairing:
`decode_fit_probe.py` runs the same corpus with `seed=1000+rep`, so leg `sweep-t6144-r3` in the
before arm and leg `sweep-t6144-r3` in the after arm are the SAME PROMPT AND THE SAME SAMPLE, and
their difference has far less variance than either has on its own. 1.8 resolved 0.81 ms/forward out
of a 3.5 % point-to-point spread exactly this way, and wrote the table out by hand; this is that
table, once, for every item that needs it.

  python3 tools/paired_band.py <before_dir> <after_dir> [--label-before X --label-after Y]

Reports every leg, the paired mean, sd, and the 2 SE band, plus the SIGN COUNT -- how many legs
moved the same way -- because a mean that is inside its own band and a mean carried by one leg are
different results and the mean alone cannot tell them apart. It also splits the band into a FLAT
term and a PER-1000-CONTEXT term by least squares on the same per-leg deltas, which is what says
whether a saving is Term A or Term B.

`control` legs are reported SEPARATELY. In `decode_fit_probe.py` they are NOT legs the change
cannot touch -- they are fresh-BUILD legs from a second document, the control for the prefix-cache
shortcut, and they run the same kernels as the sweep. They are split out because they are a
different corpus taken at the END of the run, so pooling them silently mixes a corpus change and a
drift position into the band.

DRIFT IS NOT REMOVED BY PAIRING WHEN THE TWO ARMS ARE TWO CHECKPOINT LOADS. Pairing removes the
between-LEG variance; it does not remove a constant offset between the arms, and
measurement-and-traps §19 measures exactly that offset at 0.6 % between loads = ~0.8 ms/forward
here. When the band is the same size as that offset, one ordering cannot decide the sign. Run the
pair AGAIN with the arm order REVERSED and average the two paired means: drift enters the two runs
with opposite sign and cancels, the effect does not. `--reversed` takes the second pair and reports
the drift-free estimate.
"""
import json, math, os, statistics as st, sys


def load(d):
    out = {}
    for name in ('postfix.sweep.jsonl', 'control.fresh.jsonl'):
        p = os.path.join(d, name)
        if not os.path.exists(p):
            continue
        for line in open(p):
            line = line.strip()
            if line:
                r = json.loads(line)
                out[r['id']] = r
    return out


def fwd(r):
    t, u = r['timings'], r['usage']
    return t['decode_ms'] / u['completion_tokens'] * t['tokens_per_verify']


def ctx(r):
    return r['usage']['prompt_tokens'] + r['usage']['completion_tokens'] / 2


def band(deltas):
    n = len(deltas)
    m = st.mean(deltas)
    sd = st.stdev(deltas) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    return n, m, sd, (m - 2 * se, m + 2 * se)


def report(rows, title, lb, la):
    if not rows:
        return
    print(f'\n--- {title} ---')
    print(f'{"leg":>20}{"ctx":>8}{lb:>10}{la:>10}{"delta":>9}{"tau b":>8}{"tau a":>8}')
    for leg, c, b, a, tb, ta in rows:
        print(f'{leg:>20}{c:8.0f}{b:10.2f}{a:10.2f}{a-b:+9.2f}{tb:8.3f}{ta:8.3f}')
    deltas = [a - b for _, _, b, a, _, _ in rows]
    n, m, sd, (lo, hi) = band(deltas)
    neg = sum(1 for d in deltas if d < 0)
    taueq = sum(1 for _, _, _, _, tb, ta in rows if abs(tb - ta) < 5e-4)
    print(f'\nn={n}  paired mean {m:+.3f} ms/forward   sd {sd:.3f}   2 SE band [{lo:+.3f}, {hi:+.3f}]')
    print(f'legs faster after: {neg}/{n}   tau equal to 3 d.p.: {taueq}/{n}')
    print('VERDICT: ' + ('the band EXCLUDES zero — a real paired saving.' if hi < 0 else
                         'the band EXCLUDES zero — a real paired COST.' if lo > 0 else
                         'the band COVERS ZERO — this is a NULL, not a win.'))
    # flat vs per-1000-context split, least squares on the per-leg deltas
    if n > 2 and len({round(c) for _, c, _, _, _, _ in rows}) > 1:
        xs = [c / 1000.0 for _, c, _, _, _, _ in rows]
        xm, ym = st.mean(xs), st.mean(deltas)
        sxx = sum((x - xm) ** 2 for x in xs)
        slope = sum((x - xm) * (y - ym) for x, y in zip(xs, deltas)) / sxx
        icpt = ym - slope * xm
        res = [y - (icpt + slope * x) for x, y in zip(xs, deltas)]
        sse = sum(r * r for r in res)
        sst = sum((y - ym) ** 2 for y in deltas)
        sres = math.sqrt(sse / (n - 2)) if n > 2 else 0.0
        se_s = sres / math.sqrt(sxx) if sxx else float('inf')
        se_i = sres * math.sqrt(1.0 / n + xm * xm / sxx) if sxx else float('inf')
        print(f'split:   flat {icpt:+.3f} +/- {2*se_i:.3f} ms/forward   '
              f'context {slope:+.4f} +/- {2*se_s:.4f} ms per 1000 ctx   '
              f'R^2 {1 - sse/sst if sst else float("nan"):.3f}')
        print('         (a saving that is FLAT is Term A; one that scales with ctx is Term B)')


def deltas_of(bdir, adir, kind='sweep'):
    """Per-leg (id, ctx, delta) for one ordered pair of arms."""
    A, B = load(bdir), load(adir)
    shared = sorted(set(A) & set(B))
    return [(k, ctx(A[k]), fwd(B[k]) - fwd(A[k])) for k in shared if A[k]['kind'] == kind]


def pooled(d1, d2):
    """Drift-free estimate from two runs of the SAME pair with the arm order reversed.

    Run 1 measured `effect + drift`, run 2 measured `effect - drift` (drift = whatever the second
    load of a session costs relative to the first, measured at 0.6 % in traps §19). The mean of the
    two per-leg deltas cancels it leg by leg; half their difference IS the drift, and printing it is
    what turns "the band covers zero" into a statement about which of the two it was.
    """
    m1, m2 = dict((k, d) for k, _, d in d1), dict((k, d) for k, _, d in d2)
    cx = dict((k, c) for k, c, _ in d1)
    ks = sorted(set(m1) & set(m2))
    return [(k, cx[k], 0.5 * (m1[k] + m2[k]), 0.5 * (m1[k] - m2[k])) for k in ks]


def main(argv):
    lb = la = None
    rev = None
    args, skip = [], 0
    for i, a in enumerate(argv[1:], 1):
        if skip: skip -= 1; continue
        if a == '--label-before': lb = argv[i + 1]; skip = 1
        elif a == '--label-after': la = argv[i + 1]; skip = 1
        elif a == '--reversed': rev = (argv[i + 1], argv[i + 2]); skip = 2
        elif a.startswith('--'): pass
        else: args.append(a)
    if len(args) != 2:
        print(__doc__); return 2
    if rev:
        d1, d2 = deltas_of(args[0], args[1]), deltas_of(rev[0], rev[1])
        rows = pooled(d1, d2)
        if not rows:
            print('no legs in common between the two runs'); return 2
        print('POOLED OVER TWO ARM ORDERS — the drift-free estimate.')
        print('run 1: control arm first (drift penalises the change).')
        print('run 2: change arm first (drift penalises the control).\n')
        print(f'{"leg":>20}{"ctx":>8}{"effect":>10}{"drift":>10}')
        for k, c, e, dr in rows:
            print(f'{k:>20}{c:8.0f}{e:+10.2f}{dr:+10.2f}')
        eff = [e for _, _, e, _ in rows]
        drf = [dr for _, _, _, dr in rows]
        n, m, sd, (lo, hi) = band(eff)
        neg = sum(1 for e in eff if e < 0)
        print(f'\nn={n}  DRIFT-FREE paired mean {m:+.3f} ms/forward   sd {sd:.3f}   '
              f'2 SE band [{lo:+.3f}, {hi:+.3f}]')
        print(f'legs faster after: {neg}/{n}')
        _, dm, dsd, (dlo, dhi) = band(drf)
        print(f'drift term (half the difference between the two orders): {dm:+.3f} ms/forward   '
              f'sd {dsd:.3f}   2 SE [{dlo:+.3f}, {dhi:+.3f}]')
        print('VERDICT: ' + ('the drift-free band EXCLUDES zero — a real paired saving.' if hi < 0 else
                             'the drift-free band EXCLUDES zero — a real paired COST.' if lo > 0 else
                             'the drift-free band COVERS ZERO — this is a NULL, not a win.'))
        if n > 2 and len({round(c) for _, c, _, _ in rows}) > 1:
            xs = [c / 1000.0 for _, c, _, _ in rows]
            xm, ym = st.mean(xs), st.mean(eff)
            sxx = sum((x - xm) ** 2 for x in xs)
            slope = sum((x - xm) * (y - ym) for x, y in zip(xs, eff)) / sxx
            icpt = ym - slope * xm
            res = [y - (icpt + slope * x) for x, y in zip(xs, eff)]
            sse = sum(r * r for r in res); sst = sum((y - ym) ** 2 for y in eff)
            sres = math.sqrt(sse / (n - 2))
            print(f'split:   flat {icpt:+.3f} +/- {2*sres*math.sqrt(1.0/n + xm*xm/sxx):.3f} ms/forward   '
                  f'context {slope:+.4f} +/- {2*sres/math.sqrt(sxx):.4f} ms per 1000 ctx   '
                  f'R^2 {1 - sse/sst if sst else float("nan"):.3f}')
        return 0
    A, B = load(args[0]), load(args[1])
    lb, la = (lb or 'before')[:9], (la or 'after')[:9]
    shared = sorted(set(A) & set(B))
    if not shared:
        print('no legs in common'); return 2
    if {A[k].get('corpus_sha256') for k in shared} != {B[k].get('corpus_sha256') for k in shared}:
        print('FAIL: the two arms did not read the same corpus; the A/B is void.'); return 1
    nohash = [k for k in shared if not A[k].get('text_sha256') or not B[k].get('text_sha256')]
    miss = [k for k in shared if k not in nohash and A[k]['text_sha256'] != B[k]['text_sha256']]
    print(f'token identity: {len(shared)-len(miss)-len(nohash)}/{len(shared)} legs byte-identical, '
          f'{len(miss)} differ, {len(nohash)} unproven')

    def rows_for(kind):
        return sorted(((k, ctx(A[k]), fwd(A[k]), fwd(B[k]),
                        A[k]['timings']['tokens_per_verify'], B[k]['timings']['tokens_per_verify'])
                       for k in shared if A[k]['kind'] == kind), key=lambda r: r[0])

    report(rows_for('sweep'), 'SWEEP legs (the band)', lb, la)
    report(rows_for('control'), 'CONTROL legs (reported separately, never pooled)', lb, la)
    return 1 if (miss or nohash) else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
