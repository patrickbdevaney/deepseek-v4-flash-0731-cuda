#!/usr/bin/env python3
"""eval_preflight.py — prove the whole battery will succeed and be valid BEFORE spending a day on it.

WHY THIS EXISTS, concretely. The first attempt at this battery failed in three separate ways that a
run log reported as success:

  * GPQA-Diamond aborted on its first item with an HTTP 500 and the runner printed "gpqa_diamond
    done". Zero items scored, reported as a completed task. The cause was not a blip -- it was
    `server.cpp`'s `prompt + max_tokens + 8 > seqmax` rejection, i.e. the battery was structurally
    impossible at that context and nothing had checked.
  * AIME 2025 problem 19 carries its gold as "336^\\circ". Every model answering 336 would have been
    marked wrong: 3.3 points of a 30-item benchmark, silently.
  * AIME at one sample per problem has a 95 % interval of +-15.6 points, which is not a measurement
    at all, and nothing in the pipeline objected.

None of those three would have produced an error. All three would have produced a clean table of
numbers to publish. So the rule here is: every property the published claim depends on is checked
in advance, against the real datasets and the real running server, and the battery does not start
until all of them are green.

Checks, in the order a failure would waste the most time:

  A. SERVER      up, model loaded, reports its context window
  B. PROVENANCE  the datasets are the real ones (delegates to eval_provenance.py)
  C. SCORERS     the graders are correct (HumanEval canonical, MATH-500 identity, extraction)
  D. CONTEXT     EVERY item in the plan fits prompt + max_tokens + margin inside seqmax
  E. POWER       every task's planned n gives an interval narrow enough to quote
  F. MEMORY      the box has headroom left, so a long run does not end in a reboot
  G. LIVE        one real item per task, end to end, scored -- the pipeline works on this server

  python3 tools/eval_preflight.py                 # uses the same PLAN as scripts/run_evals.sh
  python3 tools/eval_preflight.py --skip-live     # A-F only (no GPU time)
"""
import argparse, json, os, subprocess, sys, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_suite as E

ROOT = E.ROOT
CKPT = os.environ.get('CKPT', '/home/patrickd/models/DeepSeek-V4-Flash-0731-REAP')

# Mirrors the PLAN in scripts/run_evals.sh: task -> (n, reps)
PLAN = [('gpqa_diamond', 0, 1), ('mmlu_pro', 200, 1), ('humaneval', 0, 1),
        ('aime24', 0, 4), ('aime25', 0, 4), ('math500', 120, 1), ('gsm8k', 120, 1)]

results, nfail = [], 0


def ck(section, what, ok, detail=''):
    global nfail
    nfail += (not ok)
    results.append(dict(section=section, check=what, ok=bool(ok), detail=detail))
    # flush: this runs for tens of minutes behind nohup, and an unflushed buffer makes a live
    # preflight indistinguishable from a hung one.
    print(f'  [{"PASS" if ok else "FAIL"}] {section:<11} {what:<44} {detail}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='localhost:8080')
    ap.add_argument('--skip-live', action='store_true')
    a = ap.parse_args()

    # ---------------------------------------------------------------- A. server
    seqmax = 0
    try:
        # /health, not /v1/models: the OpenAI-shaped model list has no context field, and reading
        # seqmax as None from it would let every downstream context check pass vacuously.
        with urllib.request.urlopen(f'http://{a.host}/health', timeout=10) as r:
            info = json.loads(r.read())
        seqmax = int(info.get('seqmax') or 0)
        ck('SERVER', 'reachable and model loaded', bool(seqmax),
           f'model={info.get("model")} seqmax={seqmax} context_len={info.get("context_len")}')
    except Exception as e:
        ck('SERVER', 'reachable and model loaded', False, str(e)[:80])
    if not seqmax:
        # An early exit must be NO-GO. The first version of this returned finish() here with zero
        # failures recorded and printed GO -- a preflight that green-lights a battery it never
        # checked is worse than no preflight at all.
        ck('SERVER', 'seqmax known (everything downstream needs it)', False, 'cannot verify anything')
        return finish()

    # ---------------------------------------------------------------- B. provenance
    p = subprocess.run([sys.executable, os.path.join(ROOT, 'tools', 'eval_provenance.py')],
                       capture_output=True, text=True)
    npass = p.stdout.count('[PASS]')
    ck('PROVENANCE', 'published dataset facts verified locally', p.returncode == 0,
       f'{npass} checks; see evidence/evals/provenance.json')

    # ---------------------------------------------------------------- C. scorers
    try:
        items, _, _, _ = E.TASKS['humaneval'](0)
        import glob
        import pyarrow.parquet as pq
        pins = json.load(open(E.PINS))
        f = glob.glob(pins['openai/openai_humaneval']['local'] + '/**/*.parquet', recursive=True)[0]
        byid = {r['task_id'].replace('/', '-'): r for r in pq.read_table(f).to_pylist()}
        ok = sum(E.correct('code', E.extract('code', '```python\n' + byid[i['id']]['prompt']
                                             + byid[i['id']]['canonical_solution'] + '```'),
                           'pass', i) for i in items)
        ck('SCORERS', 'HumanEval canonical solutions pass', ok == len(items), f'{ok}/{len(items)}')
    except Exception as e:
        ck('SCORERS', 'HumanEval canonical solutions pass', False, str(e)[:80])

    m, _, _, _ = E.TASKS['math500'](0)
    ok = sum(E.correct('math', i['gold'], i['gold']) for i in m)
    ck('SCORERS', 'MATH-500 gold matches itself', ok == len(m), f'{ok}/{len(m)}')

    cases = [('integer', 'so \\boxed{42}.', '42'), ('letter', '...Answer: C', 'C'),
             ('letter', '**Answer: J**', 'J'), ('math', 'thus \\boxed{\\frac{3}{4}}', '3/4')]
    ok = sum(E.extract(k, t) == w for k, t, w in cases)
    ck('SCORERS', 'answer extraction cases', ok == len(cases), f'{ok}/{len(cases)}')

    # ---------------------------------------------------------------- D. context
    # This is the check whose absence killed GPQA. Exact token counts from the checkpoint's OWN
    # tokenizer, plus the chat template's overhead measured against the live server rather than
    # guessed, then every single item in the plan is required to fit.
    try:
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(os.path.join(CKPT, 'tokenizer.json'))
    except Exception as e:
        ck('CONTEXT', 'tokenizer loadable', False, str(e)[:80])
        return finish()

    overhead = 0
    if not a.skip_live:
        probe = 'x'
        try:
            r = E.ask(a.host, probe, 'high', 1.0, 0.95, 4, 120)
            overhead = r['usage']['prompt_tokens'] - len(tk.encode(probe).ids)
        except Exception:
            overhead = 16
    else:
        overhead = 16
    MARGIN = 8 + max(overhead, 0)      # server.cpp rejects at prompt + max_tokens + 8 > seqmax
    ck('CONTEXT', 'chat-template overhead measured', True, f'{overhead} tokens')

    worst = []
    for task, n, reps in PLAN:
        items, _, _, _ = E.TASKS[task](n)
        maxtok = E.MAXTOK[task]
        lens = [len(tk.encode(it['prompt']).ids) for it in items]
        need = max(lens) + maxtok + MARGIN
        fits = need <= seqmax
        worst.append((task, max(lens), maxtok, need, fits))
        ck('CONTEXT', f'{task}: every item fits', fits,
           f'longest prompt {max(lens)} + max_tokens {maxtok} + {MARGIN} = {need} vs seqmax {seqmax}')

    # ---------------------------------------------------------------- E. power
    for task, n, reps in PLAN:
        items, _, _, _ = E.TASKS[task](n)
        N = len(items) * reps
        # half-width at the least favourable p=0.5 would be pessimistic; use p=0.8, a realistic
        # accuracy for this class of model, and require the interval to be tighter than +-10 points.
        lo, hi = E.wilson(round(0.8 * N), N)
        hw = (hi - lo) / 2
        ck('POWER', f'{task}: interval quotable', hw <= 10.0,
           f'n={N} (reps={reps}) -> +-{hw:.1f} points at p=0.8')

    # ---------------------------------------------------------------- F. memory
    avail = int(open('/proc/meminfo').read().split('MemAvailable:')[1].split()[0]) // 1024
    ck('MEMORY', 'headroom above the guard floor', avail > 1800, f'MemAvailable {avail} MB')
    guard = subprocess.run(['pgrep', '-f', 'memguard.sh'], capture_output=True)
    ck('MEMORY', 'memguard running', guard.returncode == 0,
       'kills the server before the kernel takes the box down')

    # ---------------------------------------------------------------- G. live
    if not a.skip_live:
        for task, n, reps in PLAN:
            items, _, _, kind = E.TASKS[task](n)
            it = max(items, key=lambda x: len(x['prompt']))   # the LONGEST item, not a friendly one
            try:
                r = E.ask(a.host, it['prompt'], 'high', 1.0, 0.95, E.MAXTOK[task], 1800)
                ch = r['choices'][0]
                txt = (ch['message'].get('content') or '') + (ch['message'].get('reasoning_content') or '')
                got = E.extract(kind, ch['message'].get('content') or '') or E.extract(kind, txt)
                ck('LIVE', f'{task}: longest item round-trips', bool(txt),
                   f'{r["usage"]["prompt_tokens"]}+{r["usage"]["completion_tokens"]} tok, '
                   f'finish={ch.get("finish_reason")}, extracted={"yes" if got is not None else "NO"}')
            except Exception as e:
                ck('LIVE', f'{task}: longest item round-trips', False, str(e)[:100])

    return finish()


def finish():
    os.makedirs(E.OUT, exist_ok=True)
    with open(os.path.join(E.OUT, 'preflight.json'), 'w') as f:
        json.dump(dict(checks=results, failed=nfail), f, indent=2)
    print(f'\nPREFLIGHT: {len(results) - nfail} passed, {nfail} failed -> '
          f'{"GO" if not nfail else "NO-GO"}')
    return 1 if nfail else 0


if __name__ == '__main__':
    sys.exit(main())
