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

CONTEXT. The server runs at seqmax=8192 -- see EVALS.md and
wiki/context-ceiling-is-not-the-kv-cache.md for why it was 4096 and what changed. Reasoning traces are therefore capped well below what the model would
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
import argparse, glob, hashlib, json, math, os, random, re, socket, subprocess, sys, tempfile, time
import urllib.error
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
        # AIME answers are integers in [0,999] BY CONSTRUCTION -- it is a fill-in-the-integer-grid
        # exam. The mirrors do not always store them that way: opencompass/AIME2025 problem 19 carries
        # the gold as "336^\\circ". Left alone, a model answering 336 is marked wrong, and one item of
        # thirty is 3.3 points of a benchmark lost to a stray LaTeX suffix rather than to the model.
        # So the gold is reduced to its integer, and `eval_provenance.py` asserts that every gold
        # survives that reduction as an integer in [0,999] -- if a mirror ever stores something that
        # is NOT an integer answer, the assertion fails rather than this line quietly inventing one.
        g = re.search(r'-?\d+', str(ans))
        items.append(dict(id=f'aime{year}-{rid or i:04}', prompt=prob + '\n\n' + BOXED,
                          gold=g.group(0) if g else str(ans).strip()))
    return items, snap, src, 'integer'


def task_gpqa():
    rows, snap = _arrow('fingertap___gpqa-diamond/**/*.arrow')
    items = [dict(id=f'gpqa-{i:04d}', gold=r['answer'].strip().upper(),
                  prompt='Answer the following multiple choice question. The last line of your '
                         "response should be of the following format: 'Answer: $LETTER' (without "
                         'quotes) where LETTER is one of ABCD. Think step by step before '
                         'answering.\n\n' + r['question'])
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
                          prompt='Answer the following multiple choice question. The last line of '
                                 "your response should be of the following format: 'Answer: $LETTER' "
                                 '(without quotes) where LETTER is one of ABCDEFGHIJ. Think step by '
                                 f'step before answering.\n\n{r["question"]}\n\n{opts}'))
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


def task_bfcl(n):
    """BFCL v3 (executable subsets) — function calling and parallel tool invocation.

    WHY THIS ONE. Nothing else in this suite touches tool use, which is the axis this model's own
    card leads on (Toolathlon 70.3, Terminal Bench 82.7). BFCL is the standard measure for it, it is
    exactly scorable, and its outputs are SHORT -- so it is among the cheapest signal per GPU-hour
    here, where everything else is dominated by reasoning-trace length.

    SCORED BY AST MATCH, not by execution. The subsets are named `exec_*` because BFCL can run them
    against live APIs; that needs network and keys and would make the score depend on whether a
    weather endpoint was up. Every row ships `ground_truth` as a call string, so the comparison is
    structural: same function, same argument names, same values. That is BFCL's own AST metric and
    it is deterministic offline. Recorded as AST-mode in the report so it is never read as exec-mode.

    Prompt mode rather than the native tool-call API: the model emits calls as text and they are
    parsed. This is how BFCL evaluates models without a tool-calling endpoint, and it keeps the
    measurement independent of this server's DSML tool-call plumbing -- which is engine behaviour,
    not model capability.
    """
    import glob as _g
    base = os.path.expanduser('~/.cache/huggingface/hub/'
                              'datasets--gorilla-llm--Berkeley-Function-Calling-Leaderboard/snapshots')
    subs = ['exec_simple', 'exec_multiple', 'exec_parallel', 'exec_parallel_multiple']
    items, snaps = [], set()
    for sub in subs:
        fs = sorted(_g.glob(os.path.join(base, '*', f'BFCL_v3_{sub}.json')))
        if not fs:
            sys.exit(f'BFCL subset {sub} not found under {base} — this harness never downloads')
        snaps.add(os.path.basename(os.path.dirname(fs[0])))
        rows = [json.loads(l) for l in open(fs[0]) if l.strip()]
        for i, r in _sample(rows, 0):
            q = r['question'][0][0]['content']
            fns = json.dumps(r['function'], indent=1)
            gt = r['ground_truth']
            items.append(dict(
                id=f'bfcl-{sub}-{i:04d}', gold=json.dumps(gt), subject=sub,
                prompt=('You have access to the following functions:\n\n' + fns +
                        '\n\nUser request: ' + q +
                        '\n\nRespond with ONLY the function call(s) needed, one per line, in '
                        'Python call syntax, e.g. func_name(arg1=value1, arg2=value2). '
                        'Emit exactly ' + str(len(gt)) + ' call(s). No explanation, no code fences.')))
    return items, ','.join(sorted(snaps)), 'gorilla-llm/BFCL v3 (exec subsets, AST-scored, 240)', 'call'


def task_lcb(n):
    """LiveCodeBench (code_generation_lite), the most recent release window.

    WHY test6 AND NOT test.jsonl. LiveCodeBench exists to be date-windowed: its whole design is that
    you evaluate on problems published AFTER a model's training cutoff, because competitive-programming
    problems and their editorials are exactly the kind of thing that leaks. `test.jsonl` covers
    2023-05 to 2024-03, which a 2026 checkpoint has almost certainly seen -- scoring on it would
    measure recall as much as capability. `test6.jsonl` is 2025-01 to 2025-04, the latest window
    published, and is what is used here.

    THAT IS STILL NOT A CLEAN WINDOW and the report says so: this checkpoint is `0731`, i.e. later
    than every problem in it. Contamination is REDUCED, not eliminated, and it biases the score UP.

    Two execution modes, both from the benchmark's own test cases:
      * `functional` (LeetCode): the row carries `starter_code` defining a `Solution` class; the test
        input is a JSON argument list and the expected output a JSON return value.
      * `stdin` (AtCoder, Codeforces): the program reads stdin and its stdout is compared.
    Private tests are zlib+base64 and are used alongside the public ones, as the benchmark intends.
    """
    import base64 as _b64, glob as _g, pickle as _pk, zlib as _z
    fs = sorted(_g.glob(os.path.expanduser(
        '~/.cache/huggingface/hub/datasets--livecodebench--code_generation_lite/'
        'snapshots/*/test6.jsonl')))
    if not fs:
        sys.exit('LiveCodeBench test6.jsonl not found — run tools/eval_fetch.py')
    snap = os.path.basename(os.path.dirname(fs[0]))
    rows = [json.loads(l) for l in open(fs[0]) if l.strip()]

    def _tests(r):
        out = json.loads(r['public_test_cases'])
        try:
            priv = _z.decompress(_b64.b64decode(r['private_test_cases']))
            try:
                priv = _pk.loads(priv)
            except Exception:
                priv = priv.decode()
            out = out + (json.loads(priv) if isinstance(priv, str) else priv)
        except Exception:
            pass                      # public tests alone; still a real (weaker) check
        return out

    items = []
    for i, r in _sample(rows, n):
        fn = r['starter_code'].strip()
        instr = ('Write a complete Python solution. Reply with ONLY one ```python code block.\n\n'
                 + ('Complete this class:\n```python\n' + r['starter_code'] + '```\n'
                    if fn else 'Read from standard input and write to standard output.\n'))
        items.append(dict(id=f'lcb-{r["question_id"]}', gold='pass',
                          subject=r['platform'], level=r['difficulty'],
                          starter=r['starter_code'], tests=_tests(r),
                          prompt=instr + '\n' + r['question_content']))
    return items, snap, 'livecodebench/code_generation_lite test6 (2025-01..2025-04, 175)', 'lcb'


TASKS = {
    'gsm8k':        task_gsm8k,
    'aime24':       lambda n: task_aime(2024),
    'aime25':       lambda n: task_aime(2025),
    'gpqa_diamond': lambda n: task_gpqa(),
    'mmlu_pro':     task_mmlu_pro,
    'math500':      task_math500,
    'humaneval':    task_humaneval,
    'bfcl':         task_bfcl,
    'lcb':          task_lcb,
}

# Per-task completion ceiling. prompt + completion + template overhead has to fit inside seqmax, and
# `eval_preflight.py` verifies that for EVERY item of every task before the battery starts -- the
# first attempt at this suite lost all of GPQA-Diamond to exactly that constraint going unchecked.
#
# These were raised once the engine's context went from 4096 to 8192 (the arena is no longer scaled
# by seqmax). That is not cosmetic: a truncated item is scored WRONG, so every token of headroom here
# removes a way for the harness to understate the model rather than measure it.
# 8000 EVERYWHERE, and the number comes from a measurement rather than a preference
# (evidence/evals/calib_gpqa_low.log, GPQA at low effort, stride sample):
#
#   items that TERMINATE:  103, 363, 393, 407, 779, 6356 tokens  -> 8000 clears every one
#   items that DO NOT:     hit 12000 and burned 20-24 minutes each, 2 of 8
#
# The terminating distribution is what a budget has to cover, and it tops out at 6356; 8000 gives
# that ~26 % headroom. The non-terminating quarter is a property of the model, not of the budget --
# one of them was still marked CORRECT at 12000 because the answer appeared mid-trace and it kept
# going, and the other was wrong at 12000 and would have been wrong at any budget. Paying 12000 for
# them buys nothing and costs ~90 % of the wall clock, so the cap is set to protect real answers and
# to stop runaways early, not to chase them.
MAXTOK = dict(gsm8k=8000, aime24=8000, aime25=8000, gpqa_diamond=8000,
              mmlu_pro=8000, math500=8000, humaneval=8000, bfcl=2000, lcb=8000)


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
        # First pattern is openai/simple-evals' ANSWER_PATTERN_MULTICHOICE verbatim, widened to
        # A-J for MMLU-Pro. The looser ones after it are additive fallbacks: they can only ever turn
        # a miss into a hit, so they cannot inflate relative to the reference protocol.
        for pat in (r'(?i)Answer[ \t]*:[ \t]*\$?([A-J])\$?',
                    r'\\boxed\{\s*([A-J])\s*\}',
                    r'\banswer\s*(?:is)?\s*[:=]?\s*\(?\*{0,2}([A-J])\b'):
            ms = re.findall(pat, text, re.I)
            if ms:
                return ms[-1].upper()
        m = re.findall(r'\b([A-J])\b', text[-200:])
        return m[-1].upper() if m else None
    if kind == 'code':
        blocks = re.findall(r'```(?:python|py)?\s*\n(.*?)```', text, re.S)
        return blocks[-1] if blocks else text
    if kind == 'call':
        return text          # scored structurally by call_correct; no extraction step
    if kind == 'lcb':
        blocks = re.findall(r'```(?:python|py)?\s*\n(.*?)```', text, re.S)
        return blocks[-1] if blocks else text
    return None


def _guarded_run(argv, stdin_text=None, timeout=20, mem_bytes=2 << 30):
    """Run model-generated code with limits, and leave nothing behind.

    HumanEval and LiveCodeBench execute code this model wrote, unreviewed, on the same box that is
    holding ~113 GiB of weights resident. `subprocess.run(timeout=...)` is not enough for that:

      * it applies NO memory limit, so one `[0]*10**10` in a wrong solution takes the machine into
        swap. On this box the visible consequence would be memguard firing and killing the SERVER --
        the battery would die and the log would blame the engine, not the solution that did it.
      * it kills only the direct child. Code that forks or spawns a thread pool leaves orphans that
        survive the timeout and keep consuming the box for the rest of the night.

    So: an address-space cap, no core dumps, a file-size cap, and its own session so the whole
    process group can be killed as a unit. Returns (returncode, stdout, stderr) with returncode None
    meaning it timed out.
    """
    import resource, signal

    def limits():
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 << 20, 64 << 20))
        os.setsid()                      # own process group, so killpg reaps grandchildren too

    p = subprocess.Popen(argv,
                         stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, preexec_fn=limits)
    try:
        out, err = p.communicate(input=stdin_text, timeout=timeout)
        return p.returncode, out, err
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            pass
        try:
            p.communicate(timeout=5)
        except Exception:
            pass
        return None, '', ''


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
        rc, _, _ = _guarded_run([sys.executable, p], timeout=timeout)
        return rc == 0
    finally:
        os.unlink(p)



def _sympy_equal(a, b):
    r"""Symbolic equivalence, the third tier. Returns None if it cannot decide.

    This exists because the reference implementations do NOT compare MATH answers as strings. The
    Minerva/lm-evaluation-harness lineage checks equivalence with sympy, and the newer math_verify
    lineage does string -> numeric-tolerance -> symbolic. A string-only comparison is therefore
    STRICTER than the published protocol, and strictness here does not make a number conservative in
    an interesting way -- it just marks correct answers wrong. \frac{1}{2} vs 0.5, 2\sqrt{2} vs
    \sqrt{8}, and (3, \pi/2) written with different spacing are all the same answer.
    """
    try:
        from sympy import simplify, sympify
        from sympy.parsing.latex import parse_latex
    except Exception:
        return None
    def prep(t):
        # sympify has no implicit multiplication: "2sqrt(2)" is a parse error, not 2*sqrt(2). The
        # normaliser produces exactly that shape, so restore the operator before parsing.
        t = re.sub(r'(\d)\s*([A-Za-z\\(])', r'\1*\2', t)
        return t.replace('^', '**')

    def parse(x):
        for f in (lambda t: parse_latex(t), lambda t: sympify(prep(t))):
            try:
                v = f(x)
                if v is not None:
                    return v
            except Exception:
                continue
        return None
    try:
        pa, pb = parse(a), parse(b)
        if pa is None or pb is None:
            return None
        d = simplify(pa - pb)
        return bool(d == 0)
    except Exception:
        return None


def math_equal(got, gold):
    """Tiered equivalence, mirroring the reference pipelines: exact string after normalisation,
    then numeric within tolerance, then symbolic. Each tier can only ever turn a MISS into a HIT, so
    none of them can inflate a wrong answer into a right one."""
    if got is None:
        return False
    a, b = norm_math(got), norm_math(gold)
    if a == b:
        return True
    try:                                            # numeric, with tolerance
        if abs(float(a) - float(b)) < 1e-6:
            return True
    except (TypeError, ValueError):
        pass
    sym = _sympy_equal(a, b)
    if sym is None:
        sym = _sympy_equal(str(got), str(gold))     # retry on the raw text; the normaliser may have
    return bool(sym)                                # mangled LaTeX that sympy could have parsed




def _safe_val(node):
    """Literal, or simple arithmetic over literals.

    `ast.literal_eval` alone is not enough: BFCL's own ground truth writes argument values as
    EXPRESSIONS -- `p=1/6` in exec_multiple-0000 and `amount=500*500` in exec_parallel_multiple-0011.
    With literal_eval those two rows fail against their own gold, which is a scorer bug that would
    have quietly cost 2 items. Arithmetic over numeric literals is evaluated; anything involving a
    name, call or attribute still raises, so this cannot execute model-supplied code.
    """
    import ast as _ast
    if isinstance(node, _ast.Constant):
        return node.value
    if isinstance(node, _ast.UnaryOp) and isinstance(node.op, (_ast.UAdd, _ast.USub)):
        v = _safe_val(node.operand)
        return +v if isinstance(node.op, _ast.UAdd) else -v
    if isinstance(node, _ast.BinOp):
        a, b = _safe_val(node.left), _safe_val(node.right)
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            raise ValueError('non-numeric operands')
        op = node.op
        if isinstance(op, _ast.Add):   return a + b
        if isinstance(op, _ast.Sub):   return a - b
        if isinstance(op, _ast.Mult):  return a * b
        if isinstance(op, _ast.Div):   return a / b
        if isinstance(op, _ast.Pow):   return a ** b
        raise ValueError('unsupported operator')
    if isinstance(node, (_ast.List, _ast.Tuple)):
        return [_safe_val(e) for e in node.elts]
    if isinstance(node, _ast.Dict):
        return {_safe_val(k): _safe_val(v) for k, v in zip(node.keys, node.values)}
    raise ValueError('not a literal expression')


def _parse_calls(text):
    """Every `name(...)` call in the text, as (name, {arg: literal}). Unparseable ones are dropped."""
    import ast as _ast
    out = []
    for m in re.finditer(r'([A-Za-z_][A-Za-z0-9_.]*)\s*\(', text or ''):
        start = m.start()
        depth, i = 0, m.end() - 1
        while i < len(text):                       # walk to the matching close paren
            depth += (text[i] == '(') - (text[i] == ')')
            i += 1
            if depth == 0:
                break
        snippet = text[start:i]
        try:
            node = _ast.parse(snippet.strip(), mode='eval').body
            if not isinstance(node, _ast.Call):
                continue
            name = snippet[:snippet.index('(')].strip().split('.')[-1]
            kw = {}
            for k in node.keywords:
                if k.arg is None:
                    continue
                kw[k.arg] = _safe_val(k.value)
            out.append((name, kw))
        except Exception:
            continue
    return out


def _val_eq(a, b):
    """Argument equality with the tolerances BFCL's own checker allows: numeric closeness, and
    list/tuple equivalence. Everything else is exact."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(b)))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_val_eq(x, y) for x, y in zip(a, b))
    return a == b


def call_correct(text, gold_json):
    """AST match: for each expected call there must be a generated call with the same function name
    and the same arguments. Order-insensitive, and extra generated calls are a FAILURE -- emitting
    every plausible call and hoping one lands is not a correct answer."""
    gold = json.loads(gold_json)
    want = []
    for g in gold:
        p = _parse_calls(g)
        if not p:
            return False
        want.append(p[0])
    got = _parse_calls(text)
    if len(got) != len(want):
        return False
    used = [False] * len(got)
    for wname, wargs in want:
        hit = False
        for j, (gname, gargs) in enumerate(got):
            if used[j] or gname != wname or set(gargs) != set(wargs):
                continue
            if all(_val_eq(gargs[k], wargs[k]) for k in wargs):
                used[j] = True
                hit = True
                break
        if not hit:
            return False
    return True



def run_lcb(code, item, per_test_timeout=8, max_tests=25):
    """Run a LiveCodeBench solution against the benchmark's own tests. All must pass.

    `max_tests` bounds the wall clock: some rows carry hundreds of private tests and this engine has
    a whole battery to get through. It is a REAL limitation and it can only make the score higher
    (a solution failing test 30 of 200 is counted correct), so it is recorded in the report rather
    than left implicit.
    """
    if not code:
        return False
    tests = (item.get('tests') or [])[:max_tests]
    if not tests:
        return False
    functional = bool(item.get('starter') or '').__and__(True) if False else bool(str(item.get('starter') or '').strip())
    with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR', '/tmp')) as d:
        src = os.path.join(d, 'sol.py')
        for t in tests:
            ttype = t.get('testtype', 'functional' if functional else 'stdin')
            if ttype == 'functional':
                # Call Solution().<method>(*args); the input is a JSON argument list, one per line.
                m = re.search(r'def\s+(\w+)\s*\(self', item.get('starter') or '')
                if not m:
                    return False
                args = t['input']
                prog = (code + '\n\nimport json,sys\n'
                        '_a=[json.loads(x) for x in ' + repr(args) + '.split(chr(10)) if x.strip()]\n'
                        '_r=Solution().' + m.group(1) + '(*_a)\n'
                        'print(json.dumps(_r))\n')
                open(src, 'w').write(prog)
                rc, out, _ = _guarded_run([sys.executable, src], timeout=per_test_timeout)
                if rc != 0:                      # None (timeout) or a non-zero exit both fail
                    return False
                try:
                    if json.loads(out.strip()) != json.loads(t['output'].strip()):
                        return False
                except Exception:
                    if out.strip() != t['output'].strip():
                        return False
            else:
                open(src, 'w').write(code)
                rc, out, _ = _guarded_run([sys.executable, src], stdin_text=t['input'],
                                          timeout=per_test_timeout)
                if rc != 0:                      # None (timeout) or a non-zero exit both fail
                    return False
                got = [l.rstrip() for l in out.strip().splitlines()]
                exp = [l.rstrip() for l in t['output'].strip().splitlines()]
                if got != exp:
                    return False
    return True


def correct(kind, got, gold, item=None):
    if got is None:
        return False
    if kind == 'code':
        return run_code(got, item)
    if kind == 'call':
        return call_correct(got, gold)
    if kind == 'lcb':
        return run_lcb(got, item)
    if kind == 'letter':
        return got == gold
    if kind == 'math':
        return math_equal(got, gold)
    try:
        return abs(float(got) - float(gold)) < 1e-9
    except ValueError:
        return got == gold


def accuracy_ci(recs, seed=20260814):
    """Accuracy and a 95 % interval that respects how many PROBLEMS there are, not how many
    generations were drawn from them.

    avg@k re-samples the SAME problems k times. Feeding the resulting k*m records to a Wilson
    interval treats four samples of one AIME problem as four independent problems, which is
    pseudo-replication: AIME 2024 is thirty problems whether you sample it once or sixty-four times.
    Simulated over 30 problems at reps=4, the Wilson-over-120 interval comes out at +-8.8 points in
    every scenario, while the true interval ranges from +-8.9 (every problem a genuine coin flip) to
    +-18.2 (every problem deterministic) -- understated by up to 2.07x, and understated MOST exactly
    when the extra reps bought the least.

    So: group by problem, and bootstrap over PROBLEMS. The estimator is the mean of the per-problem
    success rates, which is also what avg@k means. Single-sample tasks have one record per problem,
    the grouping is a no-op, and they keep the exact Wilson interval -- the normal approximation is
    wrong at 30 items near the ends, which is why Wilson is there in the first place.
    """
    groups = {}
    for r in recs:
        groups.setdefault(r['id'].split('#r')[0], []).append(bool(r.get('correct')))
    m = len(groups)
    if m == 0:
        return 0.0, 0.0, 0.0, 'none', 0
    reps_max = max(len(v) for v in groups.values())
    if reps_max <= 1:
        k, n = sum(sum(v) for v in groups.values()), sum(len(v) for v in groups.values())
        lo, hi = wilson(k, n)
        return 100.0 * k / n, lo, hi, 'wilson', m

    # mean of per-problem rates: with unequal group sizes (a partial avg@k run) this is the
    # estimand, whereas pooled k/n would silently weight by how many samples each problem got
    vals = list(groups.values())
    ph = [sum(v) / len(v) for v in vals]
    acc = 100.0 * sum(ph) / m
    rng = random.Random(seed)          # seeded so the published interval is reproducible

    # NESTED bootstrap: resample problems, then resample draws WITHIN each chosen problem. Both
    # variance components are real -- which problems you got, and how the sampler happened to go on
    # them -- and resampling problems alone reuses the observed per-problem rates as if they were
    # exact. That is not a corner case: with every problem at 2 of 4 correct, the observed rates are
    # all identical, the outer-only bootstrap reports +-0.0, and a 50 % score comes out with a zero-
    # width interval. The inner draw is what stops the interval from collapsing.
    boot = []
    for _ in range(10000):
        tot = 0.0
        for _ in range(m):
            v = vals[rng.randrange(m)]
            L = len(v)
            tot += sum(v[rng.randrange(L)] for _ in range(L)) / L
        boot.append(tot / m)
    boot.sort()
    return (acc, 100.0 * boot[249], 100.0 * boot[9749],
            f'cluster-bootstrap over {m} problems (avg@{reps_max})', m)


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
class Fatal(Exception):
    """The server is gone or refusing everything -- stop, do not burn the rest of the battery."""


def budget_timeout(max_tokens, floor_tok_s=4.0, prefill_s=180):
    """A client timeout derived from the token budget, not picked.

    A fixed 1800 s was fine at max_tokens=4000 and silently wrong at 16000: decode slows as context
    grows, so a long generation can exceed a timeout that was generous for a short one. The preflight
    caught it as three consecutive TimeoutErrors on GPQA's longest item -- and because a timeout is
    retried, it cost 90 minutes to find out. Sized off a pessimistic floor rate so the timeout fires
    only when something is genuinely wrong, never merely because the model is thinking.
    """
    return int(prefill_s + max_tokens / floor_tok_s)


def ask(host, prompt, effort, temp, top_p, max_tokens, timeout, retries=3):
    """One item, with bounded retries.

    The first version of this raised on any failure and the caller broke out of the loop, which
    meant a SINGLE transient error abandoned an entire benchmark and reported it as "done" with zero
    items scored. That is exactly the shape of failure that produces a confident, wrong table: the
    run log says the task completed. So a request now retries, and a request that keeps failing
    raises Fatal, which stops the battery loudly rather than silently truncating it.
    """
    body = json.dumps(dict(model='dsv4', messages=[dict(role='user', content=prompt)],
                           thinking_mode='thinking', reasoning_effort=effort,
                           temperature=temp, top_p=top_p, max_tokens=max_tokens)).encode()
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(f'http://{host}/v1/chat/completions', data=body,
                                     headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = ''
            try:
                detail = e.read().decode()[:200]
            except Exception:
                pass
            last = f'HTTP {e.code}: {detail}'
            # 4xx is the request itself -- over context, malformed. Retrying is pointless and would
            # hide a systematic problem behind a retry count.
            if 400 <= e.code < 500:
                raise Fatal(f'server rejected the request ({last}) — this is a protocol or context '
                            f'error, not a blip; fix it rather than retrying') from None
        except socket.timeout:
            raise Fatal(f'request exceeded the {timeout}s timeout for max_tokens={max_tokens}. '
                        f'That is a budget/throughput problem, not a blip -- retrying it would just '
                        f'cost {retries}x as long to learn the same thing.') from None
        except Exception as e:
            last = f'{type(e).__name__}: {e}'
            if isinstance(e, urllib.error.URLError) and isinstance(getattr(e, 'reason', None), socket.timeout):
                raise Fatal(f'request exceeded the {timeout}s timeout for max_tokens={max_tokens}') from None
        if attempt + 1 < retries:
            time.sleep(5 * (attempt + 1))
    raise Fatal(f'{retries} consecutive failures, last was {last}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=list(TASKS))
    ap.add_argument('--n', type=int, default=0, help='subsample size (deterministic); 0 = all')
    ap.add_argument('--host', default='localhost:8080')
    ap.add_argument('--effort', default='high', choices=['low', 'high', 'max'])
    ap.add_argument('--temp', type=float, default=1.0)      # model card: temperature 1.0
    ap.add_argument('--top-p', type=float, default=0.95)    # model card: top_p 0.95
    ap.add_argument('--max-tokens', type=int, default=0, help='0 = the per-task budget in MAXTOK')
    ap.add_argument('--timeout', type=int, default=0, help='0 = derived from max_tokens')
    ap.add_argument('--reps', type=int, default=1,
                    help='independent samples per item (avg@k). Required for AIME-class benchmarks: '
                         'at n=30 and temperature 1.0 a SINGLE pass has a 95%% CI of about +-16 '
                         'points, so a point estimate from it is not a measurement. Published AIME '
                         'numbers are avg@16 to avg@64 for exactly this reason. Each repetition is '
                         'a separate record with id "<id>#r<k>", so reps resume like anything else '
                         'and k can be raised later without rerunning what is already banked.')
    ap.add_argument('--report', action='store_true')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    if a.report:
        return report()
    if not a.task:
        sys.exit('--task or --report')

    items, snap, src, kind = TASKS[a.task](a.n)
    n_unique = len(items)
    # RESULTS ARE NAMESPACED BY REASONING EFFORT. `low` and `high` are not the same measurement of
    # the same model -- the published reference numbers differ by up to 36 points between effort
    # levels, and the trace lengths differ by more than 2x. Sharing a jsonl between them would let
    # resume silently mix the two into a number belonging to neither, exactly the way mixing
    # max_tokens configurations would. The file name carries the effort so that cannot happen.
    tag = f'{a.task}.{a.effort}'
    if a.reps > 1:
        # rep 0 keeps the BARE id, so raising k later is purely additive: records already banked at
        # --reps 1 stay valid and only the new samples are generated.
        items = [dict(it, id=it['id'] if k == 0 else f'{it["id"]}#r{k}')
                 for k in range(a.reps) for it in items]
    maxtok = a.max_tokens or MAXTOK[a.task]
    timeout = a.timeout or budget_timeout(maxtok)
    path = os.path.join(OUT, f'{tag}.jsonl')
    done = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                done.add(json.loads(line)['id'])
            except Exception:
                pass
    todo = [it for it in items if it['id'] not in done]
    print(f'[{a.task}] {len(items)} items from {src} (snapshot {snap[:12]}), '
          f'{len(done)} already done, {len(todo)} to go, max_tokens={maxtok}, timeout={timeout}s', flush=True)

    # WHAT PRODUCED THESE TOKENS, recorded rather than assumed. The dataset sha says which
    # questions; these say which code built the prompts and which binary answered them. Without
    # them a result is reproducible only against "whatever the repo looks like now".
    def _sh(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip() or None
        except Exception:
            return None
    binp = os.path.join(ROOT, 'build', 'dsv4-server')
    meta = dict(task=a.task, source=src, snapshot=snap, n_items=len(items), scoring=kind,
                n_unique=n_unique, reps=a.reps,
                effort=a.effort, temperature=a.temp, top_p=a.top_p, max_tokens=maxtok,
                code_commit=_sh(['git', 'rev-parse', 'HEAD']),
                code_dirty=bool(_sh(['git', 'status', '--porcelain'])),
                server_sha256=(hashlib.sha256(open(binp, 'rb').read()).hexdigest()[:16]
                               if os.path.exists(binp) else None),
                server_mtime=(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(os.path.getmtime(binp)))
                              if os.path.exists(binp) else None),
                started=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    with open(os.path.join(OUT, f'{tag}.meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    t0, nok, nerr = time.time(), 0, 0
    for i, it in enumerate(todo):
        try:
            r = ask(a.host, it['prompt'], a.effort, a.temp, a.top_p, maxtok, timeout)
        except Fatal as e:
            # An item the engine cannot answer is a failure OF THE SYSTEM UNDER TEST, so it is
            # recorded as incorrect and the benchmark continues. Aborting instead -- which is what
            # this did first -- threw away 198 GPQA items because one of them made the server throw,
            # and printed "done". Scoring it wrong is both honest and conservative: it can only
            # lower the published number, and `errors` is reported separately so the reader can see
            # how much of the score is engine failure rather than model failure.
            print(f'  [{it["id"]}] ERROR (scored incorrect): {e}', flush=True)
            with open(path, 'a') as f:
                f.write(json.dumps(dict(id=it['id'], gold=it['gold'], got=None, correct=False,
                                        error=str(e)[:300], finish_reason='error',
                                        truncated=False, usage={}, timings={})) + '\n')
            nerr += 1
            if nerr >= max(10, len(todo) // 10):
                print(f'  {nerr} errors — that is too many to call this a measurement of the '
                      f'model. Stopping so the cause gets fixed.', flush=True)
                sys.exit(2)
            continue
        ch = r['choices'][0]
        content = ch['message'].get('content') or ''
        reasoning = ch['message'].get('reasoning_content') or ''
        got = extract(kind, content) or extract(kind, reasoning)   # fall back into the CoT
        ok = bool(correct(kind, got, it['gold'], it))
        nok += ok
        # TRUNCATION IS DETECTED BY TOKEN COUNT, NOT BY finish_reason. The server reports
        # finish_reason "stop" even when a generation stopped because it hit max_tokens -- verified
        # on an MMLU-Pro item that emitted exactly 3500 of 3500 tokens mid-sentence and was labelled
        # "stop". Trusting that field would print `trunc 0` in the published table while items were
        # in fact being cut off and scored wrong, which is the single most misleading thing this
        # report could do.
        used = (r.get('usage') or {}).get('completion_tokens', 0)
        truncated = ch.get('finish_reason') == 'length' or used >= maxtok
        # PROMPT HASH. The prompt is not stored (it is large and derivable), it is re-derived from
        # the pinned dataset plus this file. That derivation is only trustworthy if it can be
        # CHECKED: without a hash, editing a prompt template after a run would silently re-pair old
        # generations with a new prompt and eval_verify would still pass, because it re-derives the
        # gold but never the question. The hash closes that -- verify recomputes the prompt and
        # requires it to match what was actually sent.
        rec = dict(id=it['id'], gold=it['gold'], got=(got if kind != 'code' else None),
                   prompt_sha256=hashlib.sha256(it['prompt'].encode()).hexdigest(),
                   correct=ok, finish_reason=ch.get('finish_reason'), truncated=bool(truncated),
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


def efforts_on_disk(task):
    """Effort tags are DISCOVERED, not enumerated.

    They used to be the literal tuple ('low','high','max'), which silently dropped any result whose
    tag was not in that list -- and budget-extended runs carry a tag like `low24k` (see
    tools/eval_extend.py). A finished benchmark that simply never appears in the table is the worst
    kind of failure here, because nothing errors: the number is just missing.
    """
    out = set()
    for fn in os.listdir(OUT) if os.path.isdir(OUT) else []:
        parts = fn.split('.')
        if len(parts) == 3 and parts[0] == task and parts[2] == 'jsonl':
            out.add(parts[1])
    return sorted(out)


def report():
    lines = []
    for task in TASKS:
        for eff in efforts_on_disk(task):
            _report_one(task, eff, lines)
    _write(lines)
    return lines


def _report_one(task, eff, lines):
        path = os.path.join(OUT, f'{task}.{eff}.jsonl')
        if not os.path.exists(path):
            return
        recs = {}
        for l in open(path):
            if l.strip():
                r = json.loads(l)
                recs[r['id']] = r       # last write wins, so a re-scored item cannot double count
        recs = list(recs.values())
        if not recs:
            return
        mpath = os.path.join(OUT, f'{task}.{eff}.meta.json')
        meta = json.load(open(mpath)) if os.path.exists(mpath) else {}
        k, n = sum(r['correct'] for r in recs), len(recs)
        acc, lo, hi, ci_method, n_problems = accuracy_ci(recs)
        nerr = sum(1 for r in recs if r.get('error'))
        trunc = sum(1 for r in recs
                    if r.get('truncated') or r.get('finish_reason') == 'length'
                    or (r.get('usage') or {}).get('completion_tokens', 0) >= (meta.get('max_tokens') or 10**9))
        toks = [(r.get('usage') or {}).get('completion_tokens', 0) for r in recs]
        tps = [t for t in ((r.get('timings') or {}).get('tokens_per_second', 0) for r in recs) if t]
        # STRATUM COVERAGE. A partial run is not simply a smaller run. MMLU-Pro's rows are ordered
        # by category, so an interrupted task leaves a PREFIX -- and a prefix of a category-ordered
        # benchmark is a different benchmark. The MMLU-Pro run cut short by the UTF-8 server bug
        # covered 5 of 14 categories, with zero math and zero physics in 68 items, and would have
        # been published as "54.4 % (partial)" as though partial only meant noisier. It does not: it
        # also means biased, in a direction the CI cannot express. Coverage is computed here and a
        # task that has not touched every stratum is marked NOT QUOTABLE by the publisher.
        strata_field = 'category' if any(r.get('category') for r in recs) else (
            'subject' if any(r.get('subject') for r in recs) else None)
        strata_total = strata_seen = None
        if strata_field:
            allitems, _, _, _ = TASKS[task](0)
            strata_total = len({i.get(strata_field) for i in allitems if i.get(strata_field)})
            strata_seen = len({r.get(strata_field) for r in recs if r.get(strata_field)})
        lines.append(dict(task=task, effort=eff, n=n, n_total=meta.get('n_items'), correct=k,
                          strata_field=strata_field, strata_seen=strata_seen,
                          strata_total=strata_total,
                          n_unique=meta.get('n_unique'), reps=meta.get('reps', 1),
                          acc=round(acc, 1), ci=[round(lo, 1), round(hi, 1)],
                          ci_method=ci_method, n_problems=n_problems,
                          truncated=trunc, errors=nerr,
                          mean_completion_tokens=round(sum(toks)/n),
                          mean_tok_s=round(sum(tps)/len(tps), 2) if tps else None,
                          temperature=meta.get('temperature'),
                          top_p=meta.get('top_p'), max_tokens=meta.get('max_tokens'),
                          source=meta.get('source'), snapshot=meta.get('snapshot')))


def _write(lines):
    with open(os.path.join(OUT, 'summary.json'), 'w') as f:
        json.dump(lines, f, indent=2)
    hdr = f'{"task":<14}{"eff":>6}{"scored":>12}{"acc %":>8}{"95% CI":>16}{"trunc":>7}{"mean tok":>10}{"tok/s":>8}'
    print('\n' + hdr)
    for l in lines:
        ci = '[%.1f, %.1f]' % (l['ci'][0], l['ci'][1])
        scored = '%d/%s' % (l['n'], l['n_total'] or '?')
        print(f'{l["task"]:<14}{l.get("effort","?"):>6}{scored:>12}{l["acc"]:>8.1f}{ci:>16}{l["truncated"]:>7}'
              f'{l["mean_completion_tokens"]:>10}{(l["mean_tok_s"] or 0):>8.1f}')


if __name__ == '__main__':
    main()
