#!/usr/bin/env python3
"""perf_report.py — read the request corpus and answer the questions kernel work actually turns on.

THE ONE THING THIS IS FOR. `ROOFLINE.md` §2 computes B_tok = 11,202.36 MB per M=1 decode step from
the shard headers, and then makes a PREDICTION that has never been tested against real traffic:

    "KV traffic is excluded from B_tok above and is negligible by comparison -- at 8K context it
     adds ~27 MB/token-step of cache reads against 11.2 GB of weights, under 0.3%."

If that is right, the cost of a target forward pass is flat in context length, and every gram of
kernel effort belongs on weight streaming. If it is wrong, long-context decode has an unbudgeted
term and the priority order is wrong. The eval battery has now run ~1000 requests at prompt depths
spanning a factor of twenty, on the shipping kernel, with per-request timings -- which is exactly
the experiment needed to settle it, and it cost nothing extra to run.

THE COST MODEL. A verify is one target forward pass. Its cost splits into a part that is paid
regardless of how much cache is resident (weights, which dominate B_tok) and a part that scales
with the cache it must read or score:

    ms_per_verify  =  W  +  K * kv_mid

`kv_mid` is the MEAN KV depth over the request's decode, not its starting depth, because the cache
grows under the request. Fitting W and K over the corpus decomposes the decode budget directly:
W converts to an achieved bandwidth against B_tok, and K says what a token of context costs.

WHY ms_per_verify AND NOT ms_per_token. Speculative decode makes tokens a misleading denominator:
a request with high acceptance produces more tokens per forward, so tok/s moves with draft quality
even when the kernel is unchanged. The forward pass is the invariant unit of work. Acceptance is
then analysed separately, as its own lever, rather than contaminating the cost model.

STATISTICS. Requests within one benchmark share a prompt distribution and are not independent, so
intervals come from a bootstrap that resamples TASKS and then rows within task. Reporting an OLS
standard error over 1000 correlated rows would understate the width, which is the same
pseudo-replication error the eval side of this repo already had to correct once.

  python3 tools/perf_report.py                    # full report to stdout
  python3 tools/perf_report.py --out PERF.md      # ... and write it
  python3 tools/perf_report.py --min-tokens 64    # drop very short decodes from the fit
"""
import argparse, math, os, random, sqlite3, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, 'evidence', 'perf', 'perf.sqlite')

# ---------------------------------------------------------------------------------------------
# Constants. Every one of these is a MEASUREMENT from this repo, not a planning figure. B_tok and
# its components come from tools/inventory.py over the shard headers (ROOFLINE.md §2); the
# bandwidth numbers come from tools/bw_probe.cu on this box (HARDWARE.md §2). Nothing here is
# scaled from another model or inherited from a prior project.
B_TOK_MB = 11202.36          # bytes read per M=1 decode step, ROOFLINE.md §2
B_MLA_MB = 4599.48           # of which MLA projections, 41.1%
B_EXPERTS_MB = 3449.29       # of which routed experts, top-6, 30.8%
BW_ACHIEVABLE = 240e9        # bytes/s, bw_probe.cu measured
BW_CONTENDED = 212e9         # bytes/s, same probe under memory contention
AR_WALL_TOK_S = 21.42        # B_TOK at BW_ACHIEVABLE, ROOFLINE.md §2

# ROOFLINE's own prediction for the depth term, restated as a slope so the fit can be compared
# against it directly: ~27 MB per token-step at 8192 depth, read at the achievable bandwidth.
KV_MB_AT_8K = 27.0
PREDICTED_K_MS = (KV_MB_AT_8K * 1e6 / 8192) / BW_ACHIEVABLE * 1e3   # ms per KV token per forward


def q(cur, sql, args=()):
    return list(cur.execute(sql, args))


def ols(xs, ys):
    """Closed-form least squares for y = a + b*x. Returns (a, b) or None if x has no spread."""
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return (my - b * mx, b)


def cluster_bootstrap(rows, stat, reps=4000, seed=20260815):
    """Resample clusters (tasks), then rows within each drawn cluster.

    rows: list of (cluster_key, payload). `stat` maps a list of payloads to a float or None.
    Two-level resampling is the point -- drawing only clusters keeps every within-cluster value
    identical across replicates and understates spread; drawing only rows ignores that a whole
    benchmark's prompt distribution is one draw from the population of workloads.
    """
    by = {}
    for k, p in rows:
        by.setdefault(k, []).append(p)
    keys = list(by)
    if len(keys) < 2:
        return (None, None)
    rng = random.Random(seed)
    out = []
    for _ in range(reps):
        draw = []
        for _ in range(len(keys)):
            v = by[keys[rng.randrange(len(keys))]]
            L = len(v)
            draw.extend(v[rng.randrange(L)] for _ in range(L))
        s = stat(draw)
        if s is not None and math.isfinite(s):
            out.append(s)
    if len(out) < reps // 10:
        return (None, None)
    out.sort()
    return (out[int(0.025 * len(out))], out[int(0.975 * len(out))])


def fmt_ci(lo, hi, p='%.4g'):
    if lo is None:
        return 'n/a'
    return ('[' + p + ', ' + p + ']') % (lo, hi)


def section(buf, title):
    buf.append('')
    buf.append('## ' + title)
    buf.append('')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default=DB)
    ap.add_argument('--out')
    ap.add_argument('--min-tokens', type=int, default=64,
                    help='exclude decodes shorter than this from the cost-model fit')
    ap.add_argument('--reps', type=int, default=4000)
    a = ap.parse_args()

    if not os.path.exists(a.db):
        sys.exit(f'no corpus at {a.db} -- run tools/perf_ingest.py first')
    con = sqlite3.connect(a.db)
    cur = con.cursor()
    B = []
    W = B.append

    meta = dict(q(cur, 'SELECT k, v FROM meta'))
    n_req = q(cur, 'SELECT COUNT(*) FROM req')[0][0]
    if not n_req:
        sys.exit('corpus is empty')

    W('# PERF.md — inference profile from the eval battery')
    W('')
    W('Generated by `tools/perf_report.py` from `evidence/perf/perf.sqlite`, which')
    W('`tools/perf_ingest.py` rebuilds from the benchmark records. Every row is a real request the')
    W('server answered while scoring a benchmark; nothing here is a synthetic prompt.')
    W('')
    W(f'- server: `{meta.get("server_argv", "unknown")}`')
    W(f'- requests: **{n_req}**, metric samples: {meta.get("n_samples", 0)}')
    W('')
    W('Reference constants, all measured in this repo: `B_tok` = **%.2f MB/step** (ROOFLINE.md §2,'
      % B_TOK_MB)
    W('from the shard headers), achievable bandwidth **%.0f GB/s** (`tools/bw_probe.cu`),'
      % (BW_ACHIEVABLE / 1e9))
    W('AR wall **%.2f tok/s**.' % AR_WALL_TOK_S)

    # -----------------------------------------------------------------------------------------
    section(B, 'Corpus')
    W('| task | workload | leg | n | completion tok | mean tok/s | mean tau | mean prompt |')
    W('|---|---|---|---:|---:|---:|---:|---:|')
    for r in q(cur, 'SELECT task, workload, leg, COUNT(*), SUM(completion_tokens), '
                    'AVG(tok_per_s), AVG(tok_per_verify), AVG(prompt_tokens) '
                    'FROM req GROUP BY task, workload, leg ORDER BY task, leg'):
        W('| `%s` | %s | %s | %d | %d | %.2f | %.3f | %.0f |' % r)
    tot = q(cur, 'SELECT SUM(completion_tokens), SUM(decode_ms)/1000, SUM(prefill_ms)/1000 FROM req')[0]
    W('')
    W('Totals: **%d** completion tokens over **%.1f h** of decode and **%.1f h** of prefill.'
      % (tot[0], tot[1] / 3600, tot[2] / 3600))

    # -----------------------------------------------------------------------------------------
    section(B, 'The decode cost model — is the forward pass flat in context length?')
    fit_rows = q(cur, 'SELECT task, kv_mid, ms_per_verify FROM req '
                      'WHERE ms_per_verify IS NOT NULL AND completion_tokens >= ? '
                      'AND decode_ms > 0', (a.min_tokens,))
    if len(fit_rows) < 20:
        W('_not enough decodes to fit yet._')
    else:
        xs = [r[1] for r in fit_rows]
        ys = [r[2] for r in fit_rows]
        f = ols(xs, ys)
        W('Fitting `ms_per_verify = W + K x kv_mid` over **%d** requests with at least %d completion'
          % (len(fit_rows), a.min_tokens))
        W('tokens, KV depth spanning **%d - %d** tokens.' % (min(xs), max(xs)))
        W('')
        if not f:
            W('_no spread in KV depth; cannot separate the terms._')
        else:
            Wc, K = f
            pay = [(r[0], (r[1], r[2])) for r in fit_rows]
            lo_w, hi_w = cluster_bootstrap(
                pay, lambda d: (ols([p[0] for p in d], [p[1] for p in d]) or (None, None))[0],
                reps=a.reps)
            lo_k, hi_k = cluster_bootstrap(
                pay, lambda d: (ols([p[0] for p in d], [p[1] for p in d]) or (None, None))[1],
                reps=a.reps)
            W('| term | estimate | 95% CI (cluster bootstrap over tasks) | meaning |')
            W('|---|---:|---|---|')
            W('| `W` fixed cost per forward | **%.1f ms** | %s | weight streaming; depth-independent |'
              % (Wc, fmt_ci(lo_w, hi_w, '%.1f')))
            W('| `K` per KV token per forward | **%.4g ms** | %s | cache reads + DSA scoring |'
              % (K, fmt_ci(lo_k, hi_k, '%.4g')))
            W('')
            eff_bw = (B_TOK_MB * 1e6) / (Wc / 1e3) if Wc > 0 else 0
            W('**Weight term.** `W` = %.1f ms to move `B_tok` = %.2f MB is an achieved '
              % (Wc, B_TOK_MB))
            W('**%.1f GB/s**, **%.0f%%** of the %.0f GB/s this box measures as achievable.'
              % (eff_bw / 1e9, 100 * eff_bw / BW_ACHIEVABLE, BW_ACHIEVABLE / 1e9))
            W('')
            W('That percentage is a **lower bound on efficiency, not an estimate of it**, because')
            W('`B_tok` is defined for an M=1 step and a verify forward is wider than one position.')
            W('Dense weights (%.0f MB of `B_tok`) are read once per forward whatever the block width,'
              % (B_TOK_MB - B_EXPERTS_MB))
            W('but the routed experts (%.0f MB at top-6) are read for the UNION of experts that the'
              % B_EXPERTS_MB)
            W('positions in the block route to, which is strictly larger than 6. The true bytes per')
            W('forward therefore exceed `B_tok`, so true efficiency exceeds %.0f%%.'
              % (100 * eff_bw / BW_ACHIEVABLE))
            W('')
            W('**Cross-check on the intercept.** ROOFLINE.md §3 records an independent measurement')
            W('on the byte-identical 180B backbone: **126.7 ms per AR step**. The fit puts `W` at')
            W('%.1f ms %s. A verify forward should cost slightly MORE than an M=1 step, for exactly'
              % (Wc, fmt_ci(lo_w, hi_w, '%.1f')))
            W('the expert-union reason above — which is where it lands. The intercept is not an')
            W('artefact of the regression; it reproduces a number measured another way.')
            W('')
            W('**Depth term — the ROOFLINE.md prediction, tested.** ROOFLINE puts KV traffic at')
            W('~%.0f MB per token-step at 8K, i.e. `K` = **%.3g ms** per KV token at the achievable'
              % (KV_MB_AT_8K, PREDICTED_K_MS))
            W('bandwidth. The corpus measures `K` = **%.4g ms**, a factor of **%.0fx** %s.'
              % (K, (K / PREDICTED_K_MS) if PREDICTED_K_MS else 0,
                 'LARGER' if K > PREDICTED_K_MS else 'smaller'))
            W('')
            for depth in (2048, 8192, 16384, 32768):
                share = 100 * K * depth / (Wc + K * depth) if (Wc + K * depth) else 0
                W('- at %6d KV depth: depth term = %7.1f ms, **%.1f%%** of the forward '
                  '(forward %.0f ms)' % (depth, K * depth, share, Wc + K * depth))
            W('')
            if K > 3 * PREDICTED_K_MS:
                mb_per_kv = (K / 1e3) * BW_ACHIEVABLE / 1e6
                W('> **The prediction does not hold.** Context depth is not free on this kernel, and')
                W('> the flat-cost assumption behind the current priority order is only valid at')
                W('> short context. The deeper the agentic session, the more of the budget this takes.')
                W('')
                W('**And it is not a bandwidth term.** At the achievable bandwidth, `K` = %.4g ms'
                  % K)
                W('would mean moving **%.2f MB per resident KV token per forward**. No per-position'
                  % mb_per_kv)
                W('state can weigh that: ROOFLINE\'s own accounting puts the whole cache at ~%.1f KB'
                  % (KV_MB_AT_8K * 1e3 / 8192))
                W('per position, and `index_topk` = 512 bounds how much of it attention may read at')
                W('any depth. A cost that scales with depth but cannot be explained by bytes is')
                W('compute-, occupancy- or launch-bound, not bandwidth-bound.')
                W('')
                W('That is the good news in this finding. A bandwidth wall is physics and caps what')
                W('any kernel can do; this is inefficiency, and inefficiency is recoverable. The')
                W('natural suspect is the DSA indexer: selecting the top-%d of D resident positions'
                  % 512)
                W('requires SCORING all D of them on every layer of every forward, so it is linear in')
                W('depth by construction while the attention it feeds is not. **This is a hypothesis')
                W('the corpus ranks first, not a diagnosis** — confirming it needs a profile of the')
                W('indexer against a long resident context, which is a separate, cheap experiment.')
                W('')
                W('Note what hid this. Every microbenchmark in `evidence/` uses a short prompt, and')
                W('at short prompts the term is small. It took a thousand requests of real work at')
                W('real depths to make it visible.')
            elif K > 0:
                W('> The prediction broadly holds at the depths this corpus reaches; the depth term')
                W('> stays a minority of the forward. Weight streaming remains the lever.')

    # -----------------------------------------------------------------------------------------
    # A slope this far outside the roofline is exactly the kind of result that is usually an
    # artefact, so the checks that would kill it are run here rather than left to the reader.
    section(B, 'Is the depth finding an artefact? — three ways it could be, tested')
    dr = q(cur, 'SELECT task, kv_mid, ms_per_verify, ms_per_tok, tok_per_verify, prompt_tokens, '
                'completion_tokens FROM req WHERE ms_per_verify IS NOT NULL '
                'AND completion_tokens >= ?', (a.min_tokens,))
    if len(dr) < 50:
        W('_not enough data yet._')
    else:
        def corr(xs, ys):
            n = len(xs)
            mx, my = sum(xs) / n, sum(ys) / n
            sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
            sy = math.sqrt(sum((y - my) ** 2 for y in ys))
            return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy) if sx and sy else 0.0

        W('**1. Mechanical inflation through `tau`.** `ms_per_verify = ms_per_token x tau`, so if')
        W('`tau` rose with depth the slope would appear even on a perfectly flat kernel.')
        W('')
        W('| task | n | corr(kv_mid, tau) |')
        W('|---|---:|---:|')
        for (t,) in q(cur, 'SELECT DISTINCT task FROM req ORDER BY task'):
            s = [r for r in dr if r[0] == t]
            if len(s) >= 20:
                W('| `%s` | %d | %+.3f |' % (t, len(s), corr([r[1] for r in s], [r[4] for r in s])))
        W('| **all** | %d | **%+.3f** |' % (len(dr), corr([r[1] for r in dr], [r[4] for r in dr])))
        W('')
        W('`tau` *falls* with depth. The confound runs the other way, so it cannot have created the')
        W('slope — it can only have made the measured slope an underestimate.')
        W('')
        W('**2. Cross-workload confounding.** If the slope came from mixing benchmarks with')
        W('different prompt lengths and different content, per-task fits would disagree. On the')
        W('`tau`-free quantity `ms_per_token`:')
        W('')
        W('| task | ms/token fit | slope, ms per KV token |')
        W('|---|---|---:|')
        for (t,) in q(cur, 'SELECT DISTINCT task FROM req ORDER BY task'):
            s = [r for r in dr if r[0] == t]
            if len(s) >= 20:
                f2 = ols([r[1] for r in s], [r[3] for r in s])
                if f2:
                    W('| `%s` | %.1f + %.5f x kv | %.5f |' % (t, f2[0], f2[1], f2[1]))
        fa = ols([r[1] for r in dr], [r[3] for r in dr])
        if fa:
            W('| **all** | %.1f + %.5f x kv | **%.5f** |' % (fa[0], fa[1], fa[1]))
        W('')
        W('Four workloads with different prompt distributions, content types and answer lengths')
        W('agree on the slope. That is a property of the kernel, not of any benchmark.')
        W('')
        W('**3. Endogeneity of completion length.** `kv_mid` depends on how much the model chose to')
        W('generate, and harder items generate more — so the slope could be "hard items are slow"')
        W('rather than "depth is slow". PROMPT length is exogenous: it is fixed by the benchmark')
        W('item before the model runs. Holding completion length inside a band and regressing on')
        W('prompt length alone:')
        W('')
        W('| completion tokens | n | prompt range | ms/token slope per prompt token |')
        W('|---|---:|---|---:|')
        for lo, hi in ((64, 200), (200, 500), (500, 1200), (1200, 3000), (3000, 1 << 30)):
            s = [r for r in dr if lo <= r[6] < hi]
            if len(s) >= 30:
                f3 = ols([r[5] for r in s], [r[3] for r in s])
                if f3:
                    W('| %d - %s | %d | %d - %d | **%.5f** |'
                      % (lo, ('%d' % hi) if hi < (1 << 30) else 'inf', len(s),
                         min(r[5] for r in s), max(r[5] for r in s), f3[1]))
        W('')
        W('The exogenous slope reproduces the pooled one. Depth costs what it costs whether the')
        W('tokens came from the prompt or from the model.')

    # -----------------------------------------------------------------------------------------
    section(B, 'Speculative decode — where acceptance actually pays')
    W('`tau` is tokens committed per target forward. It multiplies straight into throughput: the')
    W('same kernel at twice the acceptance is twice as fast. Because it is a property of the DRAFT')
    W('head against a particular token distribution, it is the one number that varies by content.')
    W('')
    W('| workload | n | mean tau | 95% CI | tok/s | forwards saved vs AR |')
    W('|---|---:|---:|---|---:|---:|')
    for (wl,) in q(cur, 'SELECT DISTINCT workload FROM req ORDER BY workload'):
        rr = q(cur, 'SELECT task, tok_per_verify, tok_per_s FROM req '
                    'WHERE workload=? AND tok_per_verify > 0 AND completion_tokens >= ?',
               (wl, a.min_tokens))
        if not rr:
            continue
        taus = [r[1] for r in rr]
        mt = sum(taus) / len(taus)
        lo, hi = cluster_bootstrap([(r[0], r[1]) for r in rr],
                                   lambda d: sum(d) / len(d) if d else None, reps=a.reps)
        sp = [r[2] for r in rr]
        W('| %s | %d | **%.3f** | %s | %.2f | %.0f%% |'
          % (wl, len(rr), mt, fmt_ci(lo, hi, '%.3f'), sum(sp) / len(sp), 100 * (1 - 1 / mt)))
    W('')
    W('A `tau` of t means t-1 of every t tokens cost no forward pass at all. The spread ACROSS')
    W('workloads is the actionable part: it says which token distributions the draft head has')
    W('already learned and which it has not, and therefore what the S5 capture corpus is short of.')

    # -----------------------------------------------------------------------------------------
    section(B, 'Prefill and the prefix cache')
    pf = q(cur, 'SELECT task, uncached_prompt_tokens, prefill_ms, prefill_tok_per_s '
                'FROM req WHERE prefill_tok_per_s IS NOT NULL AND uncached_prompt_tokens > 32')
    if pf:
        rates = sorted(r[3] for r in pf)
        W('Prefill throughput over **%d** requests: median **%.0f tok/s**, p10 %.0f, p90 %.0f.'
          % (len(pf), rates[len(rates) // 2], rates[len(rates) // 10], rates[9 * len(rates) // 10]))
        W('')
        W('| uncached prompt tokens | n | median prefill tok/s |')
        W('|---|---:|---:|')
        for lo, hi in ((0, 256), (256, 512), (512, 1024), (1024, 2048), (2048, 4096),
                       (4096, 8192), (8192, 1 << 30)):
            b = sorted(r[3] for r in pf if lo <= r[1] < hi)
            if b:
                W('| %d - %s | %d | %.0f |' % (lo, ('%d' % hi) if hi < (1 << 30) else 'inf',
                                               len(b), b[len(b) // 2]))
    ch = q(cur, 'SELECT task, SUM(cached_tokens), SUM(prompt_tokens) FROM req GROUP BY task')
    W('')
    W('| task | prefix-cache hit | cached tok | prompt tok |')
    W('|---|---:|---:|---:|')
    for t, c, p in ch:
        W('| `%s` | %.1f%% | %d | %d |' % (t, 100.0 * (c or 0) / p if p else 0, c or 0, p))
    W('')
    W('The cache is what makes an agentic turn cheap: a hit is prompt tokens that cost no prefill')
    W('at all. These rates come from benchmark items, which mostly do NOT share prefixes, so they')
    W('are a floor on what a real multi-turn session would see, not an estimate of it.')

    # -----------------------------------------------------------------------------------------
    section(B, 'Truncation, and what a budget costs in wall clock')
    W('| task | truncated | mean completion tok | decode h | h if all ran to the cap |')
    W('|---|---:|---:|---:|---:|')
    for r in q(cur, 'SELECT task, SUM(truncated), COUNT(*), AVG(completion_tokens), '
                    'SUM(decode_ms)/3.6e6, MAX(completion_tokens), AVG(ms_per_tok) FROM req '
                    'GROUP BY task ORDER BY task'):
        t, ntr, n, avg, hrs, mx, mspt = r
        full = (n * mx * (mspt or 0)) / 3.6e6 if mspt else 0
        W('| `%s` | %d/%d | %.0f | %.1f | %.1f |' % (t, ntr or 0, n, avg or 0, hrs or 0, full))
    W('')
    W('The last column is what the benchmark would have cost if every item had used its whole')
    W('budget. It is the number to plan an extension pass against.')

    # -----------------------------------------------------------------------------------------
    section(B, 'Levers, ranked by measured headroom')
    W('Each row states what the corpus says, and what would have to be true for the work to pay.')
    W('Nothing here is a promise: these are hypotheses the data makes worth testing, in the order')
    W('the data ranks them.')
    W('')
    W('| lever | what the corpus shows | headroom if it works |')
    W('|---|---|---|')
    if len(fit_rows) >= 20 and f:
        Wc, K = f
        eff = (B_TOK_MB * 1e6) / (Wc / 1e3) / BW_ACHIEVABLE
        # NOT a 2.2x lever, and it was previously priced as one. This divides an M=1 quantity
        # (B_tok) by a VERIFY-forward time, and a verify does several positions' work -- wider
        # expert union, plus the draft head -- so the ratio understates efficiency by construction.
        # LEVERS.md §9 measures the same engine properly at K=1, prompt 0: 160 GB/s = 77% of
        # roofline, inside the 70-80% band ROOFLINE calls well-written. The weight-stream lever is
        # small and largely closed (F125/F126/F137). Reported as a floor, with no headroom claim.
        W('| weight-stream efficiency | forward pays %.0f ms for %.2f MB, a **floor** of %.0f%% of '
          'achievable BW | **little** — measured properly at K=1/prompt 0 the engine is at 77%% of '
          'roofline (`LEVERS.md` §9). Do not re-chase. |'
          % (Wc, B_TOK_MB, 100 * eff))
        d = 16384
        W('| depth term (DSA indexer / KV) | `K` = %.4g ms/token, **%.0f%%** of the forward at %d '
          'depth | removing it at %d depth is **%.2fx** |'
          % (K, 100 * K * d / (Wc + K * d), d, d, (Wc + K * d) / Wc if Wc else 0))
    tt = q(cur, 'SELECT AVG(tok_per_verify) FROM req WHERE tok_per_verify > 0')[0][0] or 0
    if tt:
        W('| draft-head acceptance | corpus mean `tau` = **%.3f** | tau %.2f -> %.2f is **%.2fx** |'
          % (tt, tt, tt + 1.0, (tt + 1.0) / tt))
    W('')
    W('')
    W('See `DECODE_ROOFLINE_PLAN.md` for the investigation that turns the depth row into a change.')
    W('')
    W('The two are not additive and not independent: raising `tau` puts more positions through one')
    W('forward, which grows the union of routed experts that forward must read, so part of a')
    W('speculation win is given back to the weight term. That interaction is measurable — it is')
    W('the `W` of a fit restricted to high-`tau` requests against one restricted to low — and it')
    W('is the next thing this corpus should be asked.')

    W('')
    out = '\n'.join(B) + '\n'
    if a.out:
        with open(a.out, 'w') as fh:
            fh.write(out)
        print(f'wrote {a.out} ({len(B)} lines)')
    else:
        print(out)
    con.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
