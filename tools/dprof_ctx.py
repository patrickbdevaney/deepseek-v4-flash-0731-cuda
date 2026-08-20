#!/usr/bin/env python3
"""dprof_ctx.py — attribute a decode step's CONTEXT SLOPE to individual sub-ops.

WHAT THIS IS FOR (ladder item 0.4). `tools/decode_model.py` fits the whole forward as
`a + b x context` and 0.3 measured `b = 7.362 +/- 0.370 ms per 1000 context` over context
249..12,410 — 41 % of a forward at the top of that range. `b` is a single number and says nothing
about WHICH kernel grows. This reads the per-verify `[dprof]` tables that `src/engine.cu` now emits
under `DSV4_DPROF=1`, each tagged with the context it was taken at, and fits the SAME linear model
per mark. The per-mark slopes are the attribution; they must sum, across the top-level rows, to
roughly the slope of the step as a whole, and this prints that check rather than assuming it.

WHY NOT JUST DIFF TWO RUNS, WHICH IS WHAT 0.2 DID. Because the run-to-run spread here is 3.5 % and
the effect at ctx 3000 is ~20 ms of a 150 ms forward: two points cannot separate them. This fits
many verifies across four context depths from ONE server load, so load-to-load variance is not in
the residual at all, and it reports a standard error on every slope so a mark that is flat is
distinguishable from a mark that is merely noisy.

  DSV4_DPROF=1 server ...                                    # engine emits one table per verify
  python3 tools/dprof_ctx.py evidence/decode_loop/server_0p4.log
"""
import re, sys, statistics as st

HDR = re.compile(r'^\[dprof\] ctx=(\d+) VB=(\d+) verify=(\d+)')
ROW = re.compile(r'^\[dprof\] (.+?)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)%\s+(\d+)\s*$')
# Leading spaces in a mark name are NOT decoration: dprof.cu encodes nesting depth in them, and
# `  cattn:indexer` is a child of `ATTENTION` while `cattn:indexer` would be a sibling. Stripping
# them makes a nested mark look top-level and double-counts it into the slope budget.
TOT = re.compile(r'^\[dprof\] TOTAL\s+(\d+\.\d+)\s*$')
DRAFT = ('draft:main_kv', 'draft:block', 'draft:head')


def parse(paths):
    """-> list of samples, each dict(ctx, vb, marks={name: ms}). One sample per verify step."""
    out, cur = [], None
    for p in paths:
        for line in open(p, errors='replace'):
            line = line.rstrip('\n')
            m = HDR.match(line)
            if m:
                if cur:
                    out.append(cur)
                cur = dict(ctx=int(m.group(1)), vb=int(m.group(2)),
                           verify=int(m.group(3)), marks={})
                continue
            if cur is None:
                continue
            if 'INVALID' in line:
                # dprof_report's own guard: marks recorded outside the reported window. Such a
                # sample is not a profile of anything, so drop it rather than fit it.
                cur = None
                continue
            m = TOT.match(line)
            if m:
                cur['marks']['TOTAL'] = float(m.group(1))
                out.append(cur)
                cur = None
                continue
            if line.startswith('[dprof] phase'):
                continue
            m = ROW.match(line)
            if m:
                cur['marks'][m.group(1).rstrip()] = float(m.group(2))
    if cur:
        out.append(cur)
    # A step total the wall-clock fit can actually be compared against: the verify stack (TOTAL)
    # plus the draft, which is not in TOTAL by construction (see include/dprof.h).
    for s in out:
        if 'TOTAL' in s['marks']:
            s['marks']['STEP (tot+draft)'] = s['marks']['TOTAL'] + sum(
                s['marks'].get(d, 0.0) for d in DRAFT)
    return out


def ols(y, X):
    """least squares with standard errors; Gauss-Jordan for solution and inverse at once."""
    p, n = len(X[0]), len(y)
    if n <= p:
        return [0.0] * p, [float('inf')] * p, 0.0
    A = [[sum(X[i][a] * X[i][b] for i in range(n)) for b in range(p)] for a in range(p)]
    B = [sum(X[i][a] * y[i] for i in range(n)) for a in range(p)]
    M = [A[i][:] + [B[i]] + [1.0 if i == j else 0.0 for j in range(p)] for i in range(p)]
    for c in range(p):
        piv = max(range(c, p), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        d = M[c][c]
        if d == 0:
            return [0.0] * p, [float('inf')] * p, 0.0
        for j in range(len(M[c])):
            M[c][j] /= d
        for r in range(p):
            if r != c:
                f = M[r][c]
                for j in range(len(M[r])):
                    M[r][j] -= f * M[c][j]
    beta = [M[i][p] for i in range(p)]
    res = [y[i] - sum(beta[a] * X[i][a] for a in range(p)) for i in range(n)]
    s2 = sum(r * r for r in res) / (n - p)
    se = [(s2 * M[i][p + 1 + i]) ** 0.5 for i in range(p)]
    ybar = sum(y) / n
    ss = sum((v - ybar) ** 2 for v in y)
    return beta, se, (1 - sum(r * r for r in res) / ss) if ss else 0.0


def main(paths):
    keep_warmup = '--keep-warmup' in paths
    paths = [p for p in paths if not p.startswith('--')]
    S = parse(paths)
    # DROP verify=0. It is not a steady-state step: it is the first verify of a leg, it always runs
    # at the full VB=7 width (the adaptive-verify controller has no margins yet), and on every leg
    # but the first it directly follows a `rewind_to`. Median STEP is 219.5 ms against ~160 for the
    # rest, and its MoE is 88.6 ms against 35 -- an expert-union effect of the width, not of context.
    # Leaving it in was worth 0.9 ms/1000 of spurious NEGATIVE slope on MoE.
    if not keep_warmup:
        n0 = len(S)
        S = [s for s in S if s.get('verify', 1) != 0]
        print(f'dropped {n0 - len(S)} warmup samples (verify=0); --keep-warmup to retain them')
    if not S:
        sys.exit('no [dprof] ctx= tables found — was the server run with DSV4_DPROF=1 and this build?')
    depths = sorted({s['ctx'] for s in S})
    # bucket by context point (the probe visits a handful of depths; each grows by a few tokens
    # per verify within a leg, so round to the nearest 512 for the display table only)
    # Snap to the probe's own target depths: a fixed grid put ctx 760 in a "512" bucket that no
    # leg ever visited, which read as a point rather than as the bottom of the 768 leg.
    pts = [768, 1536, 3072, 6144, 9216, 12288]
    def bucket(c):
        return min(pts, key=lambda p: abs(p - c))
    buckets = sorted({bucket(s['ctx']) for s in S})
    print(f'{len(S)} verify samples, context {min(depths)}..{max(depths)}, '
          f'{len(buckets)} points: {buckets}')
    print(f'mean verify width VB {st.mean(s["vb"] for s in S):.2f} '
          f'(min {min(s["vb"] for s in S)}, max {max(s["vb"] for s in S)})\n')

    names = [n for n in dict.fromkeys(k for s in S for k in s['marks'])]
    rows = []
    for n in names:
        g = [s for s in S if n in s['marks']]
        if len(g) < 8:
            continue
        y = [s['marks'][n] for s in g]
        b, se, r2 = ols(y, [[1.0, s['ctx'] / 1000.0] for s in g])
        b2, se2, _ = ols(y, [[1.0, s['ctx'] / 1000.0, float(s['vb'])] for s in g])
        rows.append(dict(name=n, a=b[0], b=b[1], se=se[1], r2=r2, bw=b2[1], sew=se2[1],
                         n=len(g), med=st.median(y),
                         per={bk: st.median([s['marks'][n] for s in g if bucket(s['ctx']) == bk])
                              for bk in buckets if any(bucket(s['ctx']) == bk for s in g)}))

    step = next((r for r in rows if r['name'] == 'STEP (tot+draft)'), None)
    # A mark's share is of the WHOLE STEP's slope, and only marks that partition the step get one:
    # the un-indented verify phases (which sum to TOTAL) plus the three draft rows. Indented rows
    # are children of one of those and would double-count.
    part = {r['name'] for r in rows
            if (not r['name'].startswith(' ') and r['name'] not in ('TOTAL', 'STEP (tot+draft)'))}
    slope_sum = sum(r['b'] for r in rows if r['name'] in part)
    denom = step['b'] if step and abs(step['b']) > 1e-9 else (slope_sum or 1.0)

    print('SLOPE PER MARK, ms per 1000 context.  "+VB" holds the realised verify width fixed,')
    print('because width sets bytes-per-forward and correlates with context (0.3).')
    print(f'{"mark":18}{"ms@med":>9}{"b":>9}{"+/-":>7}{"b|VB":>9}{"+/-":>7}{"R^2":>7}{"share":>8}  n')
    for r in sorted(rows, key=lambda r: -r['b']):
        share = f'{100*r["b"]/denom:6.1f}%' if r['name'] in part else '      -'
        print(f'{r["name"]:18}{r["med"]:9.2f}{r["b"]:9.3f}{r["se"]:7.3f}'
              f'{r["bw"]:9.3f}{r["sew"]:7.3f}{r["r2"]:7.3f}{share:>8}  {r["n"]}')

    if step:
        print(f'\nsum of partitioning slopes = {slope_sum:.3f} ms/1000 vs measured STEP slope '
              f'{step["b"]:.3f} +/- {step["se"]:.3f} ms/1000  '
              f'(they should agree; a gap is time inside the step that no mark covers)')
    print('\nPER-POINT MEDIANS (ms), the marks that carry the slope:')
    hdr = ''.join(f'{bk:>9}' for bk in buckets)
    print(f'{"mark":18}{hdr}')
    for r in sorted(rows, key=lambda r: -r['b'])[:14]:
        print(f'{r["name"]:18}' + ''.join(
            f'{r["per"][bk]:9.2f}' if bk in r['per'] else f'{"-":>9}' for bk in buckets))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:] or ['evidence/decode_loop/server_0p4.log']))
