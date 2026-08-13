#!/usr/bin/env python3
"""eval_suite.py — publishable capability numbers for the REAP checkpoint, with provenance.

WHAT THIS IS FOR. `0xSero/DeepSeek-V4-Flash-0731-REAP` removed 37.5 % of the experts (256 -> 160)
and stores them in MXFP4. Every "frontier capability" statement in this repo is a claim about the
*unpruned* `deepseek-ai/DeepSeek-V4-Flash-0731`, whose published scores are its own. This measures
what WE actually serve, on this box, through the CUDA engine that ships.

DESIGN RULES, each of which exists because the alternative would let the number drift:

1. **Exactly scorable only.** Every task here has a machine-checkable gold: an integer, a letter, a
   normalised expression, or a unit test that passes or does not. No LLM judge, no partial credit,
   no "contains the answer somewhere" matching. That rules out the benchmarks DeepSeek actually
   leads on -- Terminal Bench, SWE, Toolathlon are agent rollouts -- and it is the right trade: a
   defensible small number beats an indefensible big one. `EVALS.md` records what was excluded and
   why, so the omissions are part of the report rather than a silence.
2. **Pinned snapshots, never a download at eval time.** Rows come from a commit sha recorded in
   `evidence/evals/datasets.json` by `tools/eval_fetch.py`. A benchmark that fetches while it scores
   can quietly grade a different set of rows on a re-run, and then the published number cannot be
   reproduced. Fetching is a separate, deliberate step.
3. **Through the server**, i.e. the same binary, tokenizer, chat template and sampler a user gets.
   Not a special eval path.
4. **Every generation is kept.** `*.jsonl` holds the full response, the extracted answer, the gold,
   and the engine's own timing counters for each item. A number in the report can always be traced
   back to the text that produced it.
5. **Resumable, and honest about what did not finish.** A 500-item run at ~20 tok/s is many hours
   and this box has rebooted mid-work; completed ids are skipped on restart. The report always
   prints n actually scored against n in the benchmark, so a partial run can never be mistaken for
   a complete one.

CONTEXT. The server runs at seqmax=4096 because that is what fits in 122.8 GiB alongside 100.4 GiB
of weights -- see EVALS.md. Reasoning traces are therefore capped well below what the model would
emit unbounded, and `truncated` in the report counts the items that hit the ceiling. Those items are
scored as attempted and wrong rather than dropped, which makes every number here a floor.

  python3 tools/eval_suite.py --task gsm8k --n 250
  python3 tools/eval_suite.py --task aime24
  python3 tools/eval_suite.py --task gpqa_diamond
  python3 tools/eval_suite.py --task mmlu_pro --n 500
  python3 tools/eval_suite.py --task math500 --n 200
  python3 tools/eval_suite.py --task humaneval
  python3 tools/eval_suite.py --report            # summarise everything already run
"""
import argparse, glob, json, math, os, random, re, subprocess, sys, tempfile, time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'evidence', 'evals')
HF = os.path.expanduser('~/.cache/huggingface/datasets')
PINS = os.path.join(OUT, 'datasets.json')


# ----------------------------------------------------------------------------- datasets (pinned)
def _arrow(pattern):
    """Read every arrow shard matching `pattern` under the HF datasets cache."""
    import pyarrow as pa
    files = sorted(glob.glob(os.path.join(HF, pattern), recursive=True))
    if not files:
        sys.exit(f"no local snapshot for {pattern} — this harness never downloads (rule 2)")
    rows, snaps = [], set()
    for f in files:
        # .../<dataset>/<config>/0.0.0/<snapshot-hash>/<file>.arrow
        snaps.add(os.path.basename(os.path.dirname(f)))
        with pa.memory_map(f, 'rb') as src:
            rows.extend(pa.ipc.open_stream(src).read_all().to_pylist())
    return rows, ','.join(sorted(snaps))


def _pinned(repo, suffix):
    """Read the pinned snapshot of `repo`, returning (rows, sha). Provenance comes from the pin
    file, not from whatever happens to be in the cache, so the sha in the report is the sha read."""
    if not os.path.exists(PINS):
        sys.exit('evidence/evals/datasets.json missing — run tools/eval_fetch.py first')
    pins = json.load(open(PINS))
    if repo not in pins:
        sys.exit(f'{repo} not pinned — run tools/eval_fetch.py')
    d = pins[repo]
    files = sorted(glob.glob(os.path.join(d['local'], '**', '*' + suffix), recursive=True))
    if not files:
        sys.exit(f'{repo} pinned at {d["sha"][:12]} but no *{suffix} under {d["local"]}')
    rows = []
    if suffix == '.parquet':
        import pyarrow.parquet as pq
        for f in files:
            rows.extend(pq.read_table(f).to_pylist())
    else:
        for f in files:
            rows.extend(json.loads(l) for l in open(f) if l.strip())
    return rows, d['sha']


def _sample(rows, n, seed=1234):
    """Deterministic subsample, returned as (source_index, row) pairs.

    Two properties matter and both bite if you drop them. The subset is a fixed random draw rather
    than a prefix, because MMLU-Pro is ordered by category and a prefix would score one subject and
    call it the benchmark. And the SOURCE index travels with the row, because ids are built from it:
    if an id were the position within the sample, then `--n 3` and `--n 200` would mint the same id
    for different questions and the resume logic would score one and skip the other.
    """
    if not n or n >= len(rows):
        return list(enumerate(rows))
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    return [(i, rows[i]) for i in sorted(idx[:n])]


BOXED = "Put your final answer in \\boxed{}."


def task_gsm8k(n):
    rows, snap = _arrow('openai___gsm8k/**/gsm8k-test.arrow')
    items = [dict(id=f'gsm8k-{i:04d}', prompt=r['question'] + '\n\n' + BOXED,
                  gold=re.search(r'####\s*(-?[\d,]+)', r['answer']).group(1).replace(',', ''))
             for i, r in _sample(rows, n)]
    return items, snap, 'openai/gsm8k (main, test, 1319)', 'integer'


def task_aime(year):
    if year == 2024:
        rows, snap = _arrow('Maxwell-Jia___aime_2024/**/*.arrow')
        get = lambda r: (r['ID'], r['Problem'], r['Answer'])
        src = 'Maxwell-Jia/AIME_2024 (30)'
    else:
        rows, snap = _arrow('opencompass___aime2025/**/*.arrow')
        get = lambda r: (None, r['question'], r['answer'])
        src = 'opencompass/AIME2025 (I+II, 30)'
    items = []
    for i, r in enumerate(rows):
        rid, prob, ans = get(r)
        items.append(dict(id=f'aime{year}-{rid or i:04}', prompt=prob + '\n\n' + BOXED,
                          gold=str(ans).strip()))
    return items, snap, src, 'integer'


def task_gpqa():
    rows, snap = _arrow('fingertap___gpqa-diamond/**/*.arrow')
    items = [dict(id=f'gpqa-{i:04d}', gold=r['answer'].strip().upper(),
                  prompt=r['question'] + '\n\nThink step by step, then end with "Answer: X" where '
                                         'X is the letter of the correct option.')
             for i, r in enumerate(rows)]
    return items, snap, 'fingertap/GPQA-Diamond (test, 198)', 'letter'


LETTERS = 'ABCDEFGHIJ'


def task_mmlu_pro(n):
    """MMLU-Pro: 10 options, so the random baseline is 10 % and the letter set runs A-J. Options are
    a list on the row and are rendered here, because the row's `question` alone is unanswerable."""
    rows, snap = _pinned('TIGER-Lab/MMLU-Pro', '.parquet')
    items = []
    for _, r in _sample(rows, n):
        opts = '\n'.join(f'{LETTERS[i]}. {o}' for i, o in enumerate(r['options']))
        items.append(dict(id=f'mmlupro-{r["question_id"]}', gold=r['answer'].strip().upper(),
                          category=r['category'],
                          prompt=f'{r["question"]}\n\n{opts}\n\nThink step by step, then end with '
                                 f'"Answer: X" where X is the letter of the correct option.'))
    return items, snap, 'TIGER-Lab/MMLU-Pro (test, 12032)', 'letter'


def task_math500(n):
    rows, snap = _pinned('HuggingFaceH4/MATH-500', '.jsonl')
    items = [dict(id='math500-' + r['unique_id'].strip('/').replace('/', '_').replace('.json', ''),
                  gold=str(r['answer']), subject=r['subject'], level=r['level'],
                  prompt=r['problem'] + '\n\n' + BOXED)
             for _, r in _sample(rows, n)]
    return items, snap, 'HuggingFaceH4/MATH-500 (500)', 'math'


def task_humaneval(n):
    rows, snap = _pinned('openai/openai_humaneval', '.parquet')
    items = [dict(id=r['task_id'].replace('/', '-'), gold='pass',
                  test=r['test'], entry_point=r['entry_point'], stub=r['prompt'],
                  prompt='Complete this Python function. Reply with the full function in a single '
                         '```python code block, no explanation.\n\n```python\n' + r['prompt'] + '```')
             for _, r in _sample(rows, n)]
    return items, snap, 'openai/openai_humaneval (164)', 'code'


TASKS = {
    'gsm8k':        task_gsm8k,
    'aime24':       lambda n: task_aime(2024),
    'aime25':       lambda n: task_aime(2025),
    'gpqa_diamond': lambda n: task_gpqa(),
    'mmlu_pro':     task_mmlu_pro,
    'math500':      task_math500,
    'humaneval':    task_humaneval,
}

# Per-task completion ceiling. The server's context is 4096 total, so these are budgets rather than
# preferences: prompt + completion has to fit. Maths gets the most because that is where the trace
# runs long; multiple-choice and code need far less.
MAXTOK = dict(gsm8k=1600, aime24=3400, aime25=3400, gpqa_diamond=3000,
              mmlu_pro=2600, math500=3000, humaneval=1600)


# ----------------------------------------------------------------------------- answer extraction
def _last_int(s):
    m = re.findall(r'-?\d[\d,]*', s)
    return m[-1].replace(',', '') if m else None


def _boxed(text):
    """Last \\boxed{...}, brace-balanced -- a regex on [^{}]* loses \\boxed{\\frac{1}{2}}, which is
    most of MATH-500's harder answers."""
    out, i = None, 0
    while True:
        j = text.find('\\boxed{', i)
        if j < 0:
            return out
        k, depth = j + 7, 1
        while k < len(text) and depth:
            depth += (text[k] == '{') - (text[k] == '}')
            k += 1
        out, i = text[j + 7:k - 1], k


def norm_math(s):
    """Normalise a LaTeX answer to the form MATH-500's gold is written in. Only formatting is
    stripped -- nothing here can turn a wrong expression into a right one."""
    if s is None:
        return None
    s = s.strip().strip('$').strip()
    s = re.sub(r'\\(?:left|right|!|,|;|:)', '', s)
    s = re.sub(r'\\text\{([^{}]*)\}', r'\1', s)
    s = re.sub(r'\\mbox\{([^{}]*)\}', r'\1', s)
    s = s.replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac')
    s = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'\1/\2', s)
    s = re.sub(r'\\frac(\d)(\d)', r'\1/\2', s)
    s = re.sub(r'\\sqrt\{([^{}]+)\}', r'sqrt(\1)', s)
    s = s.replace('^{\\circ}', '').replace('^\\circ', '').replace('\\%', '').replace('%', '')
    s = s.replace('\\$', '').replace('$', '').replace(' ', '').rstrip('.')
    if re.fullmatch(r'-?\d+\.0+', s):
        s = s.split('.')[0]
    if re.fullmatch(r'-?[\d,]+', s):
        s = s.replace(',', '')
    return s


def extract(kind, text):
    """Prefer the requested format; fall back so a correct answer stated plainly still counts.

    Every fallback can only ever make the score HIGHER, which is the direction that matters: a low
    number here cannot then be dismissed as an artefact of strict parsing.
    """
    if not text:
        return None
    if kind in ('integer', 'math'):
        b = _boxed(text)
        if b is not None:
            return (_last_int(b) or norm_math(b)) if kind == 'integer' else norm_math(b)
        tail = text[-500:]
        m = re.search(r'(?:final answer|answer)\s*(?:is)?\s*[:=]?\s*\$?(-?\d[\d,]*)', tail, re.I)
        if m:
            return m.group(1).replace(',', '')
        return _last_int(tail) if kind == 'integer' else None
    if kind == 'letter':
        for pat in (r'\\boxed\{\s*([A-J])\s*\}',
                    r'\banswer\s*(?:is)?\s*[:=]?\s*\(?\*{0,2}([A-J])\b'):
            ms = re.findall(pat, text, re.I)
            if ms:
                return ms[-1].upper()
        m = re.findall(r'\b([A-J])\b', text[-200:])
        return m[-1].upper() if m else None
    if kind == 'code':
        blocks = re.findall(r'```(?:python|py)?\s*\n(.*?)```', text, re.S)
        return blocks[-1] if blocks else text
    return None


def run_code(code, item, timeout=20):
    """HumanEval pass@1: the extracted block plus the benchmark's own test, in a fresh interpreter.

    The stub is prepended so a model that emitted only a body -- or that used a name from the stub's
    imports without repeating them -- is not failed for formatting. `check(entry_point)` raising, or
    the process timing out, is a fail: exactly the benchmark's own criterion.
    """
    if not code:
        return False
    src = f"{item['stub']}\n    pass\n\n{code}\n\n{item['test']}\ncheck({item['entry_point']})\n"
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False,
                                     dir=os.environ.get('TMPDIR', '/tmp')) as f:
        f.write(src)
        p = f.name
    try:
        return subprocess.run([sys.executable, p], capture_output=True, timeout=timeout).returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(p)


def correct(kind, got, gold, item=None):
    if got is None:
        return False
    if kind == 'code':
        return run_code(got, item)
    if kind == 'letter':
        return got == gold
    if kind == 'math':
        a, b = norm_math(got), norm_math(gold)
        if a == b:
            return True
        try:
            return abs(float(a) - float(b)) < 1e-9
        except (TypeError, ValueError):
            return False
    try:
        return abs(float(got) - float(gold)) < 1e-9
    except ValueError:
        return got == gold


def wilson(k, n, z=1.96):
    """Wilson score interval. Reported instead of k/n +- 1.96*sqrt(p(1-p)/n) because several of
    these tasks are 30 items, where the normal approximation is simply wrong near the ends."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (100*max(0, c-h), 100*min(1, c+h))


# ----------------------------------------------------------------------------- the server
def ask(host, prompt, effort, temp, top_p, max_tokens, timeout):
    body = json.dumps(dict(model='dsv4', messages=[dict(role='user', content=prompt)],
                           thinking_mode='thinking', reasoning_effort=effort,
                           temperature=temp, top_p=top_p, max_tokens=max_tokens)).encode()
    req = urllib.request.Request(f'http://{host}/v1/chat/completions', data=body,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=list(TASKS))
    ap.add_argument('--n', type=int, default=0, help='subsample size (deterministic); 0 = all')
    ap.add_argument('--host', default='localhost:8080')
    ap.add_argument('--effort', default='high', choices=['low', 'high', 'max'])
    ap.add_argument('--temp', type=float, default=1.0)      # model card: temperature 1.0
    ap.add_argument('--top-p', type=float, default=0.95)    # model card: top_p 0.95
    ap.add_argument('--max-tokens', type=int, default=0, help='0 = the per-task budget in MAXTOK')
    ap.add_argument('--timeout', type=int, default=1200)
    ap.add_argument('--report', action='store_true')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    if a.report:
        return report()
    if not a.task:
        sys.exit('--task or --report')

    items, snap, src, kind = TASKS[a.task](a.n)
    maxtok = a.max_tokens or MAXTOK[a.task]
    path = os.path.join(OUT, f'{a.task}.jsonl')
    done = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                done.add(json.loads(line)['id'])
            except Exception:
                pass
    todo = [it for it in items if it['id'] not in done]
    print(f'[{a.task}] {len(items)} items from {src} (snapshot {snap[:12]}), '
          f'{len(done)} already done, {len(todo)} to go, max_tokens={maxtok}', flush=True)

    meta = dict(task=a.task, source=src, snapshot=snap, n_items=len(items), scoring=kind,
                effort=a.effort, temperature=a.temp, top_p=a.top_p, max_tokens=maxtok,
                started=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    with open(os.path.join(OUT, f'{a.task}.meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    t0, nok = time.time(), 0
    for i, it in enumerate(todo):
        try:
            r = ask(a.host, it['prompt'], a.effort, a.temp, a.top_p, maxtok, a.timeout)
        except Exception as e:
            print(f'  [{it["id"]}] REQUEST FAILED: {e} — stopping, rerun to resume', flush=True)
            break
        ch = r['choices'][0]
        content = ch['message'].get('content') or ''
        reasoning = ch['message'].get('reasoning_content') or ''
        got = extract(kind, content) or extract(kind, reasoning)   # fall back into the CoT
        ok = bool(correct(kind, got, it['gold'], it))
        nok += ok
        rec = dict(id=it['id'], gold=it['gold'], got=(got if kind != 'code' else None),
                   correct=ok, finish_reason=ch.get('finish_reason'),
                   category=it.get('category'), subject=it.get('subject'), level=it.get('level'),
                   usage=r.get('usage'), timings=r.get('timings'),
                   reasoning_chars=len(reasoning), content=content, reasoning=reasoning)
        with open(path, 'a') as f:
            f.write(json.dumps(rec) + '\n')
        el = time.time() - t0
        shown = ('pass' if ok else 'fail') if kind == 'code' else str(got)[:24]
        print(f'  [{i+1}/{len(todo)}] {it["id"]} gold={str(it["gold"])[:20]} got={shown} '
              f'{"OK " if ok else "   "} run={nok}/{i+1} {ch.get("finish_reason")} '
              f'{(r.get("timings") or {}).get("tokens_per_second", 0):.1f} tok/s '
              f'({el/60:.1f} min)', flush=True)
    report()


def report():
    lines = []
    for task in TASKS:
        path = os.path.join(OUT, f'{task}.jsonl')
        if not os.path.exists(path):
            continue
        recs = {}
        for l in open(path):
            if l.strip():
                r = json.loads(l)
                recs[r['id']] = r       # last write wins, so a re-scored item cannot double count
        recs = list(recs.values())
        if not recs:
            continue
        mpath = os.path.join(OUT, f'{task}.meta.json')
        meta = json.load(open(mpath)) if os.path.exists(mpath) else {}
        k, n = sum(r['correct'] for r in recs), len(recs)
        lo, hi = wilson(k, n)
        trunc = sum(1 for r in recs if r.get('finish_reason') == 'length')
        toks = [(r.get('usage') or {}).get('completion_tokens', 0) for r in recs]
        tps = [t for t in ((r.get('timings') or {}).get('tokens_per_second', 0) for r in recs) if t]
        lines.append(dict(task=task, n=n, n_total=meta.get('n_items'), correct=k,
                          acc=round(100*k/n, 1), ci=[round(lo, 1), round(hi, 1)],
                          truncated=trunc, mean_completion_tokens=round(sum(toks)/n),
                          mean_tok_s=round(sum(tps)/len(tps), 2) if tps else None,
                          effort=meta.get('effort'), temperature=meta.get('temperature'),
                          top_p=meta.get('top_p'), max_tokens=meta.get('max_tokens'),
                          source=meta.get('source'), snapshot=meta.get('snapshot')))
    with open(os.path.join(OUT, 'summary.json'), 'w') as f:
        json.dump(lines, f, indent=2)
    hdr = f'{"task":<14}{"scored":>12}{"acc %":>8}{"95% CI":>16}{"trunc":>7}{"mean tok":>10}{"tok/s":>8}'
    print('\n' + hdr)
    for l in lines:
        ci = '[%.1f, %.1f]' % (l['ci'][0], l['ci'][1])
        scored = '%d/%s' % (l['n'], l['n_total'] or '?')
        print(f'{l["task"]:<14}{scored:>12}{l["acc"]:>8.1f}{ci:>16}{l["truncated"]:>7}'
              f'{l["mean_completion_tokens"]:>10}{(l["mean_tok_s"] or 0):>8.1f}')
    return lines


if __name__ == '__main__':
    main()
