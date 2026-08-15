#!/usr/bin/env python3
"""perf_ingest.py — turn the eval battery's exhaust into a queryable inference-performance corpus.

WHY THIS EXISTS. Every benchmark request the server answers carries a full timing record —
prefill_ms, decode_ms, tokens_per_second, tokens_per_verify — alongside prompt/cached/completion
token counts. The battery has therefore been running, as a side effect of measuring capability, the
largest real-workload inference profile this project has: ~1000 requests, five workloads, prompt
lengths spanning a factor of twenty, all on the shipping kernel at a fixed server configuration.
That is a better feedstock for kernel work than any microbenchmark in `evidence/`, because it is
the actual distribution of work the server sees rather than a synthetic prompt chosen to be
convenient.

Until now that data was write-only. It sat in per-benchmark jsonl files whose reader
(`eval_suite.py --report`) cares only about `correct`. This builds the other view: one row per
request, timings normalised and derived quantities precomputed, in a SQLite file that
`perf_report.py` and ad-hoc queries can both read.

FULL REBUILD, NOT INCREMENTAL. The corpus is ~1000 rows and takes well under a second to rebuild,
so there is no reason to carry the bug surface of incremental upsert — no key collisions to reason
about when a benchmark is re-run, no stale rows when a record is dropped by eval_drop_record.py.
Rebuilding from the jsonl every time makes the warehouse a pure function of the evidence, which is
the same property `runs_index.py` relies on: a row that vanished from the evidence vanishes here.

READ-ONLY WITH RESPECT TO THE ENGINE. This never opens a socket to the server. It reads files.
Running it while the battery is scoring is safe, which is the whole point — the alternative is
waiting six days to look at the data.

  python3 tools/perf_ingest.py                 # rebuild evidence/perf/perf.sqlite
  python3 tools/perf_ingest.py --summary       # ... and print what landed
"""
import argparse, glob, json, os, sqlite3, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVALS = os.path.join(ROOT, 'evidence', 'evals')
PERF = os.path.join(ROOT, 'evidence', 'perf')
DB = os.path.join(PERF, 'perf.sqlite')

# Workload class per benchmark. The point of the corpus is that decode behaviour is not uniform
# across content -- a chain-of-thought proof, a Python function and a JSON tool call have different
# token distributions and therefore different draft-head acceptance. Grouping by task alone would
# mix scicode and humaneval (both code) with gpqa (prose), so carry the class explicitly.
WORKLOAD = {
    'gpqa_diamond': 'reasoning-prose',
    'mmlu_pro':     'reasoning-prose',
    'aime24':       'math',
    'aime25':       'math',
    'math500':      'math',
    'humaneval':    'code',
    'lcb':          'code',
    'scicode':      'code',
    'bfcl':         'tool-call',
    'bfcl_live':    'tool-call',
}

# ONE source of truth for the request table. The column list and the DDL are generated from this,
# because deriving one from the other by parsing SQL text silently mis-binds values the moment a
# line carries two columns -- which it did, and which produced a table of 1050 rows with a NULL
# task before anyone looked at the output.
COLS = [
    ('src', 'TEXT'), ('task', 'TEXT'), ('effort', 'TEXT'), ('workload', 'TEXT'), ('leg', 'TEXT'),
    ('item_id', 'TEXT'), ('rep', 'INTEGER'), ('prompt_sha256', 'TEXT'),
    ('correct', 'INTEGER'), ('truncated', 'INTEGER'), ('finish_reason', 'TEXT'),
    ('category', 'TEXT'), ('subject', 'TEXT'), ('level', 'TEXT'), ('budget', 'INTEGER'),

    ('prompt_tokens', 'INTEGER'), ('cached_tokens', 'INTEGER'), ('completion_tokens', 'INTEGER'),
    ('uncached_prompt_tokens', 'INTEGER'),

    ('prefill_ms', 'REAL'), ('decode_ms', 'REAL'), ('tok_per_s', 'REAL'),
    ('tok_per_verify', 'REAL'),

    # derived
    ('verifies', 'REAL'),           # completion_tokens / tok_per_verify; count of target forwards
    ('ms_per_tok', 'REAL'),         # decode_ms / completion_tokens
    ('ms_per_verify', 'REAL'),      # decode_ms / verifies; the cost of ONE target forward pass
    ('kv_start', 'INTEGER'),        # KV depth when decode begins
    ('kv_end', 'INTEGER'),          # KV depth when decode ends
    ('kv_mid', 'REAL'),             # mean KV depth over the decode; the regressor for depth
    ('prefill_tok_per_s', 'REAL'),  # uncached prompt tokens / prefill seconds
    ('cache_hit_frac', 'REAL'),
    ('reasoning_chars', 'INTEGER'), ('content_chars', 'INTEGER'), ('reasoning_frac', 'REAL'),
]

SCHEMA = """
DROP TABLE IF EXISTS req;
CREATE TABLE req (%s);
CREATE INDEX req_task ON req (task, leg);
CREATE INDEX req_depth ON req (kv_mid);
DROP TABLE IF EXISTS meta;
CREATE TABLE meta (k TEXT PRIMARY KEY, v TEXT);
DROP TABLE IF EXISTS sample;
CREATE TABLE sample (
    ts REAL, requests INTEGER, errors INTEGER,
    prompt_tok INTEGER, cached_tok INTEGER, completion_tok INTEGER, verifies INTEGER,
    decode_s REAL, prefill_s REAL, queue_depth INTEGER
);
""" % (', '.join(f'{n} {t}' for n, t in COLS))


def server_config():
    """Recover the running server's geometry from /proc rather than assuming defaults.

    blk / adaptK / ext_chunk change what a 'verify' costs, so a corpus that does not record them
    cannot be compared against a later one. If the server is not running we record nothing rather
    than writing down the header defaults as if they had been observed.
    """
    try:
        out = subprocess.run(['ps', '-eo', 'args'], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return {}
    for line in out.splitlines():
        if 'dsv4-server' in line and '--ckpt' in line:
            tok = line.split()
            cfg = {'argv': line.strip()}
            for i, t in enumerate(tok):
                if t.startswith('--') and i + 1 < len(tok):
                    cfg[t[2:]] = tok[i + 1]
            return cfg
    return {}


def rows_from(path):
    """One row per RECORD. An extension record is emitted as two legs, not one.

    eval_extend.py continues a truncated trace from its stored prefix, so the merged record
    describes two physically different generations: a base leg that decoded from a short prompt,
    and a continuation leg that decoded from a prompt of base_prompt+base_completion tokens -- the
    only requests in the whole programme that reach 8k-24k KV depth. Averaging them into one row
    would throw away the deepest and most interesting measurement we have, so they are split.
    """
    src = os.path.basename(path)
    stem = src[:-6] if src.endswith('.jsonl') else src        # gpqa_diamond.low24k
    task, _, effort = stem.partition('.')
    seen = {}
    for line in open(path, errors='replace'):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue                                           # a half-written trailing record
        u = r.get('usage') or {}
        t = r.get('timings') or {}
        if not u or not t:
            continue
        iid = r.get('id') or ''
        rep = seen.get(iid, 0)
        seen[iid] = rep + 1

        ext = r.get('extension') or {}
        base = dict(
            src=src, task=task, effort=effort, workload=WORKLOAD.get(task, 'unknown'),
            item_id=iid, rep=rep, prompt_sha256=r.get('prompt_sha256'),
            correct=int(bool(r.get('correct'))), truncated=int(bool(r.get('truncated'))),
            finish_reason=r.get('finish_reason'), category=r.get('category'),
            subject=r.get('subject'), level=str(r.get('level')) if r.get('level') is not None else None,
            budget=r.get('budget'),
            reasoning_chars=r.get('reasoning_chars') or 0,
            content_chars=len(r.get('content') or ''),
        )

        if ext:
            # The merged record's `usage` is a SUM across both legs and its `timings` describe only
            # the base leg (eval_extend.py copies them forward). Reconstruct each leg from the
            # extension block, and emit the continuation only when its own timings were recorded --
            # see the `ext_timings` field written by eval_extend.py. Older extension records
            # predate that field; for those we emit the base leg alone rather than inventing a
            # decode rate for the continuation.
            b_ct = ext.get('base_completion_tokens') or 0
            e_ct = ext.get('ext_completion_tokens') or 0
            b_pt = (u.get('prompt_tokens') or 0)
            yield _derive(dict(base, leg='base',
                               prompt_tokens=b_pt, cached_tokens=(u.get('prompt_tokens_details') or {}).get('cached_tokens') or 0,
                               completion_tokens=b_ct, prefill_ms=t.get('prefill_ms') or 0.0,
                               decode_ms=t.get('decode_ms') or 0.0, tok_per_s=t.get('tokens_per_second') or 0.0,
                               tok_per_verify=t.get('tokens_per_verify') or 0.0))
            et = ext.get('ext_timings') or {}
            if et:
                yield _derive(dict(base, leg='ext',
                                   prompt_tokens=ext.get('continuation_prompt_tokens') or (b_pt + b_ct),
                                   cached_tokens=et.get('cached_tokens') or 0,
                                   completion_tokens=e_ct,
                                   prefill_ms=et.get('prefill_ms') or 0.0,
                                   decode_ms=et.get('decode_ms') or 0.0,
                                   tok_per_s=et.get('tokens_per_second') or 0.0,
                                   tok_per_verify=et.get('tokens_per_verify') or 0.0))
        else:
            yield _derive(dict(base, leg='base',
                               prompt_tokens=u.get('prompt_tokens') or 0,
                               cached_tokens=(u.get('prompt_tokens_details') or {}).get('cached_tokens') or 0,
                               completion_tokens=u.get('completion_tokens') or 0,
                               prefill_ms=t.get('prefill_ms') or 0.0,
                               decode_ms=t.get('decode_ms') or 0.0,
                               tok_per_s=t.get('tokens_per_second') or 0.0,
                               tok_per_verify=t.get('tokens_per_verify') or 0.0))


def _derive(d):
    pt, ct = d['prompt_tokens'], d['completion_tokens']
    cached = d['cached_tokens'] or 0
    d['uncached_prompt_tokens'] = max(0, pt - cached)
    tpv = d['tok_per_verify'] or 0.0
    # `verifies` is the count of TARGET forward passes, which is what a per-forward cost model is
    # about. The server reports the ratio rather than the count, so recover it; a request that
    # generated nothing has no forwards to speak of.
    d['verifies'] = (ct / tpv) if (tpv > 0 and ct > 0) else None
    d['ms_per_tok'] = (d['decode_ms'] / ct) if ct else None
    d['ms_per_verify'] = (d['decode_ms'] / d['verifies']) if d['verifies'] else None
    d['kv_start'] = pt
    d['kv_end'] = pt + ct
    # Mean KV depth over the decode. The attention cost of a forward grows with the cache it reads,
    # so the regressor for a request is the AVERAGE depth across its steps, not the starting depth.
    d['kv_mid'] = pt + ct / 2.0
    d['prefill_tok_per_s'] = (d['uncached_prompt_tokens'] / (d['prefill_ms'] / 1000.0)) \
        if d['prefill_ms'] > 0 else None
    d['cache_hit_frac'] = (cached / pt) if pt else None
    tot = d['reasoning_chars'] + d['content_chars']
    d['reasoning_frac'] = (d['reasoning_chars'] / tot) if tot else None
    return d


def load_samples(cur):
    """Fold in the /metrics time series if perf_sample.py has been running."""
    path = os.path.join(PERF, 'metrics.jsonl')
    if not os.path.exists(path):
        return 0
    n = 0
    for line in open(path, errors='replace'):
        line = line.strip()
        if not line:
            continue
        try:
            s = json.loads(line)
        except Exception:
            continue
        cur.execute('INSERT INTO sample VALUES (?,?,?,?,?,?,?,?,?,?)', (
            s.get('ts'), s.get('requests_total'), s.get('errors_total'),
            s.get('prompt_tokens_total'), s.get('cached_prompt_tokens_total'),
            s.get('completion_tokens_total'), s.get('verifies_total'),
            s.get('decode_seconds_total'), s.get('prefill_seconds_total'),
            s.get('queue_depth')))
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--summary', action='store_true')
    a = ap.parse_args()

    os.makedirs(PERF, exist_ok=True)
    con = sqlite3.connect(DB)
    cur = con.cursor()
    cur.executescript(SCHEMA)

    cols = [n for n, _ in COLS]
    ph = ','.join('?' * len(cols))

    total = 0
    for path in sorted(glob.glob(os.path.join(EVALS, '*.jsonl'))):
        k = 0
        for d in rows_from(path):
            cur.execute(f'INSERT INTO req ({",".join(cols)}) VALUES ({ph})',
                        [d.get(c) for c in cols])
            k += 1
        total += k
        if a.summary and k:
            print(f'  {os.path.basename(path):32s} {k:5d} requests')

    ns = load_samples(cur)
    cfg = server_config()
    for k, v in [('server_argv', cfg.get('argv', '')), ('seqmax', cfg.get('seqmax', '')),
                 ('ext_chunk', cfg.get('ext-chunk', '')), ('blk', cfg.get('blk', '')),
                 ('adaptk', cfg.get('adaptk', '')), ('ckpt', cfg.get('ckpt', '')),
                 ('n_requests', str(total)), ('n_samples', str(ns))]:
        cur.execute('INSERT OR REPLACE INTO meta VALUES (?,?)', (k, v))
    con.commit()

    print(f'perf warehouse: {total} requests, {ns} metric samples -> {os.path.relpath(DB, ROOT)}')
    if not cfg:
        print('  note: server not running, geometry (blk/adaptk/ext_chunk) not recorded this pass')
    if a.summary:
        for row in cur.execute(
                'SELECT task, leg, COUNT(*), SUM(completion_tokens), ROUND(AVG(tok_per_s),2), '
                'ROUND(AVG(tok_per_verify),3) FROM req GROUP BY task, leg ORDER BY task, leg'):
            t, leg, n, tok, sp, tau = row
            print('  %-16s %-5s n=%-5d tok=%-9d %6.2f tok/s  tau=%.3f'
                  % (t, leg, n, tok or 0, sp or 0.0, tau or 0.0))
    con.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
