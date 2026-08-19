#!/usr/bin/env python3
"""eval_reference.py — run THIS harness against the UNPRUNED model, so the REAP delta is measured.

WHY THIS EXISTS. The battery measures one checkpoint under one protocol. It cannot, on its own,
say what pruning cost: that is a DIFFERENCE, and a difference needs both terms. Comparing our rows
against DeepSeek's published numbers for DeepSeek-V4-Flash-0731 would produce a delta dominated by
harness differences -- different reasoning effort, different token budgets, different answer
extraction, different scoring, different sampling -- not by the removal of experts. "REAP costs
about a point" is exactly the claim this repo exists to test, so it must not be assumed en route.

WHAT MAKES THE COMPARISON VALID. Every axis except the weights is held fixed, and each is held by
construction rather than by care:

  * the same items, from the same pinned snapshot, via eval_suite.TASKS
  * the same prompt string -- and the same sha256 is stored, so eval_verify catches any drift
  * the same answer extraction and the same scoring, via eval_suite.extract / eval_suite.correct
  * the same token budget (MAXTOK, or --budget for an extended leg) and the same effort
  * the same record schema, so eval_land / eval_verify / eval_publish read these files unchanged

What is NOT held fixed is the serving stack: this reaches an API, ours is a CUDA server on Thor.
That is a real caveat and it is written into the meta rather than left for a reader to infer --
sampling implementation, tokenizer edge cases and speculative decoding all differ, so a small delta
is not automatically attributable to pruning. It bounds the claim; it does not license it.

THE MODEL ID IS REQUIRED, NEVER GUESSED. Naming the wrong model produces a beautifully formatted
comparison against something else entirely, and nothing downstream could detect it. Pass --model
explicitly and record it.

  export DEEPSEEK_API_KEY=...
  python3 tools/eval_reference.py --task gpqa_diamond --model <id> --dry-run     # cost, no calls
  python3 tools/eval_reference.py --task gpqa_diamond --model <id> --n 0
  python3 tools/eval_reference.py --task gpqa_diamond --model <id> --n 0 --budget 24000

It is RESUMPTIVE (skips ids already on disk) and never touches the local engine, so it can run
alongside the battery.
"""
import argparse, hashlib, json, os, random, sys, time, urllib.error, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_suite as E

OUT = E.OUT


def chat(host, key, model, prompt, effort, maxtok, temp, top_p, timeout, retries=5):
    """One completion, with backoff. A 429 or a 5xx is a transport fact, not a wrong answer, so it
    is retried; a 4xx is a request we built wrong and must stop on rather than paper over."""
    body = dict(model=model, messages=[dict(role='user', content=prompt)],
                max_tokens=maxtok, temperature=temp, top_p=top_p)
    if effort:
        body['reasoning_effort'] = effort
    data = json.dumps(body).encode()
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(f'https://{host}/chat/completions', data=data,
                                     headers={'Content-Type': 'application/json',
                                              'Authorization': f'Bearer {key}'})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = ''
            try:
                detail = e.read().decode()[:300]
            except Exception:
                pass
            last = f'HTTP {e.code}: {detail}'
            if e.code in (400, 401, 403, 404):
                raise E.Fatal(f'the request itself is wrong, not the connection: {last}')
        except Exception as e:
            last = f'{type(e).__name__}: {e}'
        if attempt + 1 < retries:
            time.sleep(min(60, 4 * 2 ** attempt) * (0.5 + random.random()))
    raise E.Fatal(f'reference API failed {retries}x: {last}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True, choices=sorted(E.TASKS))
    ap.add_argument('--model', required=True,
                    help='EXACT api model id for the unpruned model. Never guessed, always recorded.')
    ap.add_argument('--host', default='api.deepseek.com/v1')
    ap.add_argument('--n', type=int, default=0, help='0 = every item')
    ap.add_argument('--reps', type=int, default=1)
    ap.add_argument('--effort', default='low')
    ap.add_argument('--budget', type=int, default=0, help='override MAXTOK, to match an extended leg')
    ap.add_argument('--temp', type=float, default=1.0)
    ap.add_argument('--top-p', type=float, default=0.95)
    ap.add_argument('--tag', default='', help='record-file tag; defaults to ref<effort>[<budget>k]')
    ap.add_argument('--dry-run', action='store_true', help='report what it would send and stop')
    a = ap.parse_args()

    key = os.environ.get('DEEPSEEK_API_KEY', '')
    if not key and not a.dry_run:
        sys.exit('set DEEPSEEK_API_KEY (it is never read from a file or a flag, so it cannot be '
                 'committed by accident)')

    items, snap, src, kind = E.TASKS[a.task](a.n)
    maxtok = a.budget or E.MAXTOK[a.task]
    # The tag carries the budget whenever it is not the default, because a reference run at a
    # different budget is a different measurement and must never share a file with one at the
    # default. Same rule as eval_extend's low -> low24k.
    tag = a.tag or (f'ref{a.effort}' + (f'{a.budget // 1000}k' if a.budget else ''))
    path = os.path.join(OUT, f'{a.task}.{tag}.jsonl')

    done = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                done.add(json.loads(line)['id'])
            except Exception:
                pass

    todo = []
    for rep in range(a.reps):
        for it in items:
            rid = it['id'] if rep == 0 else f'{it["id"]}#r{rep}'
            if rid not in done:
                todo.append((rid, it))

    print(f'[{a.task}@{tag}] {len(items)} items x{a.reps} reps against {a.model} '
          f'(snapshot {snap}), {len(done)} already done, {len(todo)} to go, max_tokens={maxtok}')
    if a.dry_run:
        est = sum(len(it['prompt']) for _, it in todo) / 4.0
        print(f'dry run: ~{est/1e3:.0f}k prompt tokens in, up to {len(todo)*maxtok/1e6:.2f}M '
              f'completion tokens out. Nothing sent.')
        print(f'first prompt sha256: {hashlib.sha256(todo[0][1]["prompt"].encode()).hexdigest()}'
              if todo else 'nothing to do')
        return 0

    nok = nerr = 0
    t0 = time.time()
    for i, (rid, it) in enumerate(todo):
        try:
            r = chat(a.host, key, a.model, it['prompt'], a.effort, maxtok, a.temp, a.top_p,
                     E.budget_timeout(maxtok))
        except E.Fatal as e:
            print(f'  [{i+1}/{len(todo)}] {rid}: {e}', flush=True)
            nerr += 1
            # SAME RULE AS THE LOCAL BATTERY: a run that is mostly transport failures is not a
            # measurement of anything, and banking those rows as wrong answers would poison the
            # comparison in the direction that flatters the REAP.
            if nerr >= max(10, len(todo) // 10):
                sys.exit(f'{nerr} errors — too many to call this a measurement. Stopping.')
            continue
        ch = (r.get('choices') or [{}])[0]
        msg = ch.get('message') or {}
        content = msg.get('content') or ''
        reasoning = msg.get('reasoning_content') or ''
        got = E.extract(kind, content) or E.extract(kind, reasoning)
        ok = bool(E.correct(kind, got, it['gold'], it))
        nok += ok
        used = (r.get('usage') or {}).get('completion_tokens', 0)
        # Truncation by TOKEN COUNT, not by finish_reason -- the same lesson the local runner
        # learned. A row whose truncation is under-reported is a row that looks quotable and is not.
        truncated = ch.get('finish_reason') == 'length' or used >= maxtok
        rec = dict(id=rid, gold=it['gold'], got=(got if kind != 'code' else None),
                   prompt_sha256=hashlib.sha256(it['prompt'].encode()).hexdigest(),
                   correct=ok, finish_reason=ch.get('finish_reason'), truncated=bool(truncated),
                   category=it.get('category'), subject=it.get('subject'), level=it.get('level'),
                   usage=r.get('usage'), timings={},
                   reference=dict(model=a.model, host=a.host, served_by='api'),
                   reasoning_chars=len(reasoning), content=content, reasoning=reasoning)
        with open(path, 'a') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        el = (time.time() - t0) / 60
        print(f'  [{i+1}/{len(todo)}] {rid} gold={str(it["gold"])[:20]} '
              f'got={("pass" if ok else "fail") if kind == "code" else str(got)[:24]} '
              f'{"OK " if ok else "   "} run={nok}/{i+1} {ch.get("finish_reason")} '
              f'{used} tok ({el:.1f} min)', flush=True)

    meta = dict(task=a.task, effort=tag, max_tokens=maxtok, temperature=a.temp, top_p=a.top_p,
                reasoning_effort=a.effort, snapshot=snap, source=src, n_items=len(items),
                reps=a.reps, reference_model=a.model, reference_host=a.host,
                role='unpruned reference for the REAP delta',
                caveat=('same items, prompts, budget, extraction and scoring as the local battery; '
                        'the SERVING STACK differs -- this is an API, the battery is a CUDA server '
                        'on Jetson Thor -- so sampling implementation, tokenizer edge cases and '
                        'speculative decoding are not held fixed. A small delta is not '
                        'automatically attributable to expert pruning.'))
    with open(os.path.join(OUT, f'{a.task}.{tag}.meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'\n{a.task}: {nok}/{len(todo)} correct this pass, {nerr} errors — wrote {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
