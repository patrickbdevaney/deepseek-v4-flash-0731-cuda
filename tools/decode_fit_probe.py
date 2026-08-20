#!/usr/bin/env python3
"""decode_fit_probe.py — generate the corpus `tools/decode_model.py` fits, under CONTROLLED context.

WHY THIS EXISTS. `decode_model.py` was written to fit `ms_per_forward = a + b x context` from the
evaluation battery's own 2,156 generation legs, and that is the right instrument for "what does the
engine cost on real work". It is the wrong instrument for "did the kernel change move b", for three
reasons:

  1. The battery cannot be re-run. Its units are disarmed on purpose (CLAUDE.md), a second client on
     the engine turns a scored item into a banked wrong answer, and a full re-run is ~20 hours.
  2. Its context spread is an accident of the tasks. Prompts top out at 3,492 tokens (p50 282) and
     almost all of the x-axis range comes from GENERATED tokens, so prompt length carries R^2 0.082
     and the fit is conditioned on one narrow axis.
  3. Task identity is confounded with context. Long generations are the hard tasks, hard tasks have
     lower tau, and tau is the divisor that turns ms/token into ms/forward.

This probe holds everything else fixed and moves only context: one document, one continuation task,
one completion budget, N repeats per point. What comes out is the same record schema the battery
writes, so `decode_model.py --dir` fits it unchanged.

HOW IT IS CHEAP. The engine keeps a longest-common-prefix KV cache (`src/engine.cu:596`). Every
prompt here is a token-prefix of the SAME document, and the points are visited in DESCENDING order,
so exactly one full prefill happens -- the deepest one -- and every later point is a `rewind_to`.
A 12k-token sweep that would cost ~40 minutes of prefill at the measured ~60 tok/s costs ~3.

AND WHY THAT IS CHECKED RATHER THAN ASSUMED. If `rewind_to` left the compressed or index cache in a
different state from a freshly built one, every number here would be measuring the cheap path and
silently not the shipped one. So the run ends with FRESH-BUILD CONTROLS: the same context depths
taken from a DIFFERENT document, which shares one token of prefix with the sweep and therefore has
its whole context built by `extend` rather than rewound. If decode cost agrees between the two, the
shortcut is sound; if it does not, the sweep is void and this says so. (The very first leg of the
run takes `prefill_full`, since the engine starts empty, so all three paths are exercised.)

TRANSPORT FAILURE IS FATAL, AND NOTHING IS WRITTEN FOR AN ITEM THAT GENERATED NOTHING. CLAUDE.md,
after `eval_bfcl_mt.py` "scored" 400 items against a dead engine and published two rows of zeros.

  python3 tools/decode_fit_probe.py --host localhost:8080 --outdir evidence/decode_loop/fit
  python3 tools/decode_fit_probe.py --report            # re-derive the tables from the records
"""
import argparse, hashlib, json, os, sys, time, urllib.error, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# One document, so the only thing that varies across points is how much of it is resident. Content
# cannot affect the cost anyway -- the DSA scan is a fixed-size sweep over the whole cache and does
# not branch on values -- but holding it fixed removes the question.
CORPUS = os.path.join(ROOT, 'LOOP_LOG.md')
# A DIFFERENT document for the controls: it must share no prefix with CORPUS, or the control would
# hit the very cache it exists to bypass.
CONTROL_CORPUS = os.path.join(ROOT, 'COMPRESSION_PLAYBOOK.md')

TARGETS = [12288, 9216, 6144, 3072, 1536, 768, 384, 128]   # prompt tokens, DESCENDING (see above)
CONTROL_TARGETS = [6144, 1536]
REPS = 6
MAX_TOKENS = 256          # >= 64, which is decode_model.py's floor for a usable leg
TEMP, TOP_P = 1.0, 0.95   # the model card's sampling, and the battery's -- tau depends on it


def load_tokenizer(ckpt):
    """The checkpoint's own tokenizer.json, never a reimplementation (tools/encode_prompt.py)."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(ckpt, 'tokenizer.json'))
    got = tok.encode('The capital of France is', add_special_tokens=False).ids
    if got != [671, 6102, 294, 8760, 344]:
        sys.exit(f'tokenizer gate FAILED: {got} != canonical ids; prompts would not be trustworthy')
    return tok


def prefixes(tok, path, targets):
    """text prefixes of `path` at each token target, cut on a line boundary.

    Cut on a boundary the BPE cannot straddle, then VERIFY by re-encoding that each shorter prefix
    is a true token-prefix of the longest. A near-prefix would still run; it would just quietly cost
    a full prefill, and the run would take an hour instead of half of one for no stated reason.
    """
    doc = open(path, encoding='utf-8').read()
    ids = tok.encode(doc, add_special_tokens=False).ids
    out = {}
    for t in targets:
        if t > len(ids):
            sys.exit(f'{path} is only {len(ids)} tokens, need {t}')
        # walk back from the token cut to the nearest newline in the decoded text
        txt = tok.decode(ids[:t])
        nl = txt.rfind('\n')
        out[t] = txt[:nl + 1] if nl > 0 else txt
    longest = tok.encode(out[max(targets)], add_special_tokens=False).ids
    for t in targets:
        e = tok.encode(out[t], add_special_tokens=False).ids
        if e != longest[:len(e)]:
            print(f'[probe] WARNING target {t}: not a token-prefix of the longest prompt; '
                  f'this point will pay a full prefill', flush=True)
    return out


def post(host, path, payload, timeout):
    req = urllib.request.Request(f'http://{host}{path}',
                                 data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def run_point(host, args, fh, seen, tag, prompt, target, rep, corpus_sha):
    rid = f'{tag}-t{target}-r{rep}'
    if rid in seen:
        print(f'[probe] {rid} already on disk, skipping', flush=True)
        return None
    t0 = time.time()
    try:
        # A raw completion, not a chat turn: no template, no thinking block, so `prompt_tokens` is
        # exactly what this script chose and nothing is inserted between the points. This is also
        # the endpoint ladder item 0.1 instrumented, so the sweep re-proves that fix in passing.
        r = post(host, '/v1/completions',
                 dict(prompt=prompt, max_tokens=args.max_tokens, temperature=TEMP, top_p=TOP_P,
                      seed=1000 + rep),
                 timeout=args.timeout)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
        # TRANSPORT FAILURE IS FATAL. A stage that "completes" against a dead engine is worse than
        # one that dies (CLAUDE.md).
        sys.exit(f'[probe] FATAL transport failure on {rid} after {time.time()-t0:.0f}s: {e}')
    u, tm = r.get('usage') or {}, r.get('timings') or {}
    ct = u.get('completion_tokens', 0)
    if not tm.get('decode_ms') or not tm.get('tokens_per_verify') or ct < 64:
        # DO NOT BANK A RECORD FOR AN ITEM THAT GENERATED NOTHING USABLE.
        print(f'[probe] {rid}: unusable (ct={ct} timings={tm}); not recorded', flush=True)
        return None
    rec = dict(id=rid, kind=tag, target_tokens=target, rep=rep, corpus_sha256=corpus_sha,
               usage=u, timings=tm, spec_profile=r.get('spec_profile'),
               wall_s=round(time.time() - t0, 2))
    fh.write(json.dumps(rec) + '\n')
    fh.flush()
    os.fsync(fh.fileno())
    fpw = tm['decode_ms'] / ct * tm['tokens_per_verify']
    print(f'[probe] {rid}: pt={u.get("prompt_tokens")} cached={(u.get("prompt_tokens_details") or {}).get("cached_tokens")} '
          f'ct={ct} prefill={tm["prefill_ms"]:.0f}ms decode={tm["decode_ms"]:.0f}ms '
          f'tau={tm["tokens_per_verify"]:.3f} -> {fpw:.2f} ms/forward  ({time.time()-t0:.0f}s)', flush=True)
    return rec


def existing(path):
    if not os.path.exists(path):
        return set()
    ids = set()
    for line in open(path):
        line = line.strip()
        if line:
            try:
                ids.add(json.loads(line)['id'])
            except Exception:
                pass
    return ids


def ols(y, X):
    """Least squares with standard errors. Two predictors at most; normal equations are fine."""
    p, n = len(X[0]), len(y)
    A = [[sum(X[i][a] * X[i][b] for i in range(n)) for b in range(p)] for a in range(p)]
    B = [sum(X[i][a] * y[i] for i in range(n)) for a in range(p)]
    M = [A[i][:] + [B[i]] + [1.0 if i == j else 0.0 for j in range(p)] for i in range(p)]
    for c in range(p):                                  # Gauss-Jordan, solution and inverse at once
        piv = max(range(c, p), key=lambda r: abs(M[r][c])); M[c], M[piv] = M[piv], M[c]
        d = M[c][c]
        for j in range(len(M[c])): M[c][j] /= d
        for r in range(p):
            if r != c:
                f = M[r][c]
                for j in range(len(M[r])): M[r][j] -= f * M[c][j]
    beta = [M[i][p] for i in range(p)]
    res = [y[i] - sum(beta[a] * X[i][a] for a in range(p)) for i in range(n)]
    s2 = sum(r * r for r in res) / (n - p)
    se = [(s2 * M[i][p + 1 + i]) ** 0.5 for i in range(p)]
    ybar = sum(y) / n
    r2 = 1 - sum(r * r for r in res) / sum((v - ybar) ** 2 for v in y)
    return beta, se, r2


def report(outdir):
    """Per-point medians and the width-controlled regression, from the records on disk.

    WHY THE SECOND REGRESSION IS NOT OPTIONAL. `ms per forward` is `ms/token x tau`, and tau moves
    with the realised verify width, which is what sets how large an expert union each forward reads.
    In this sweep width correlates +0.35 with context, so a slope fitted on context alone could be
    an acceptance effect wearing a context costume. Fitting both says how much of `b` survives with
    width held fixed -- and that residual is the number the ladder's stop condition is about.
    """
    import statistics as st
    rows = []
    for name, kind in (('postfix.sweep.jsonl', 'sweep'), ('control.fresh.jsonl', 'control')):
        path = os.path.join(outdir, name)
        if not os.path.exists(path):
            continue
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line); t, u = r['timings'], r['usage']; sp = r.get('spec_profile') or {}
            ct = u['completion_tokens']
            rows.append(dict(kind=kind, target=r['target_tokens'], ctx=u['prompt_tokens'] + ct / 2,
                             fwd=t['decode_ms'] / ct * t['tokens_per_verify'],
                             tau=t['tokens_per_verify'], width=sp.get('mean_width'),
                             tps=t['tokens_per_second']))
    if not rows:
        sys.exit(f'no records under {outdir}')
    print(f'{"kind":8}{"ctx":>7}{"n":>3}{"ms/fwd med":>12}{"spread":>8}{"tau":>8}{"width":>8}{"tok/s":>8}')
    for k in sorted({(r['kind'], r['target']) for r in rows}, key=lambda x: (x[0], -x[1])):
        g = [r for r in rows if (r['kind'], r['target']) == k]
        f = [x['fwd'] for x in g]
        print(f'{k[0]:8}{st.median(x["ctx"] for x in g):7.0f}{len(g):3d}{st.median(f):12.2f}'
              f'{(max(f)-min(f))/st.median(f)*100:7.1f}%{st.median(x["tau"] for x in g):8.3f}'
              f'{st.median(x["width"] for x in g):8.3f}{st.median(x["tps"] for x in g):8.2f}')

    S = [r for r in rows if r['kind'] == 'sweep' and r['width']]
    y = [r['fwd'] for r in S]
    b1, se1, r21 = ols(y, [[1.0, r['ctx']] for r in S])
    b2, se2, r22 = ols(y, [[1.0, r['ctx'], r['width']] for r in S])
    print(f'\n  context only   fwd = {b1[0]:.2f} + {b1[1]*1000:.3f} x (ctx/1000)'
          f'                    R^2 {r21:.3f}  SE(b) {se1[1]*1000:.3f}')
    print(f'  + verify width fwd = {b2[0]:.2f} + {b2[1]*1000:.3f} x (ctx/1000) + {b2[2]:.2f} x width'
          f'   R^2 {r22:.3f}  SE(b) {se2[1]*1000:.3f}  SE(c) {se2[2]:.2f}')

    def corr(f):
        mx = sum(r['ctx'] for r in S) / len(S); mv = sum(f(r) for r in S) / len(S)
        cov = sum((f(r) - mv) * (r['ctx'] - mx) for r in S)
        return cov / ((sum((f(r)-mv)**2 for r in S) * sum((r['ctx']-mx)**2 for r in S)) ** 0.5)
    print(f'\n  corr(mean_width, ctx) = {corr(lambda r: r["width"]):+.3f}'
          f'   corr(tau, ctx) = {corr(lambda r: r["tau"]):+.3f}')
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='localhost:8080')
    ap.add_argument('--outdir', default=os.path.join(ROOT, 'evidence', 'decode_loop', 'fit'))
    ap.add_argument('--ckpt', default=os.path.expanduser('~/models/DeepSeek-V4-Flash-0731-REAP'))
    ap.add_argument('--reps', type=int, default=REPS)
    ap.add_argument('--max-tokens', type=int, default=MAX_TOKENS)
    ap.add_argument('--timeout', type=int, default=1800)
    ap.add_argument('--report', action='store_true',
                    help='no server needed: re-derive the per-point table and both regressions')
    a = ap.parse_args()

    if a.report:
        return report(a.outdir)

    os.makedirs(a.outdir, exist_ok=True)
    sweep_path = os.path.join(a.outdir, 'postfix.sweep.jsonl')
    ctrl_path = os.path.join(a.outdir, 'control.fresh.jsonl')

    try:
        h = json.loads(urllib.request.urlopen(f'http://{a.host}/health', timeout=10).read().decode())
        print(f'[probe] server healthy: {h}', flush=True)
    except Exception as e:
        sys.exit(f'[probe] FATAL: no healthy server on {a.host}: {e}')

    tok = load_tokenizer(a.ckpt)
    sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]
    sweep = prefixes(tok, CORPUS, TARGETS)
    ctrl = prefixes(tok, CONTROL_CORPUS, CONTROL_TARGETS)
    print(f'[probe] corpus {os.path.basename(CORPUS)} {sha(CORPUS)}   '
          f'control {os.path.basename(CONTROL_CORPUS)} {sha(CONTROL_CORPUS)}', flush=True)

    seen = existing(sweep_path) | existing(ctrl_path)
    n = 0
    with open(sweep_path, 'a') as fh:
        for target in TARGETS:                       # descending: one full prefill for the whole run
            for rep in range(a.reps):
                if run_point(a.host, a, fh, seen, 'sweep', sweep[target], target, rep, sha(CORPUS)):
                    n += 1

    # The controls run LAST and DESCENDING among themselves, from a document that shares no prefix
    # with the sweep, so the first one takes `prefill_full`. If these disagree with the sweep at the
    # same depth, the prefix-cache shortcut is not equivalent and the sweep must be thrown away.
    with open(ctrl_path, 'a') as fh:
        for target in CONTROL_TARGETS:
            for rep in range(2):
                if run_point(a.host, a, fh, seen, 'control', ctrl[target], target, rep,
                             sha(CONTROL_CORPUS)):
                    n += 1

    print(f'[probe] wrote {n} usable records -> {sweep_path} and {ctrl_path}', flush=True)
    return 0 if n else 1


if __name__ == '__main__':
    sys.exit(main())
