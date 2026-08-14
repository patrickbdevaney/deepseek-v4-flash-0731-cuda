#!/usr/bin/env python3
"""eval_calibrate.py — measure the completion-length tail, then size the battery from it.

The last battery was invalidated by a token budget chosen by guess: 48 % of GPQA items truncated,
and truncated items score at chance, so the published number was a coin flip weighted by max_tokens.
Raising the budget to "as much as possible" is not the fix either -- at 16000 tokens and the ~8 tok/s
this engine sustains at long context, one item can take 33 minutes and a 198-item benchmark becomes
a 25-hour task. Budget and runtime trade directly against each other, and neither can be picked
without knowing where the distribution actually ends.

So: run a small deterministic sample at a DELIBERATELY GENEROUS budget, and report where the mass
is. That answers the only question that matters -- what budget makes truncation rare -- and it costs
an hour instead of discovering it 25 hours in.

Output is a per-task recommendation: the budget at which the measured truncation rate falls below
the target, and the wall-clock the full task would then cost.

  python3 tools/eval_calibrate.py --task gpqa_diamond --n 20 --budget 16000
"""
import argparse, json, os, statistics as st, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_suite as E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', default='gpqa_diamond')
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--budget', type=int, default=16000)
    ap.add_argument('--host', default='localhost:8080')
    ap.add_argument('--target-trunc', type=float, default=0.05, help='acceptable truncation rate')
    ap.add_argument('--effort', default='high', choices=['low','high','max'])
    a = ap.parse_args()

    items, snap, src, kind = E.TASKS[a.task](0)
    # A fixed stride across the whole benchmark rather than the first n: the point is the shape of
    # the distribution, and the first n items of a category-ordered set are not a sample of it.
    step = max(1, len(items) // a.n)
    sample = items[::step][:a.n]
    print(f'[calibrate] {a.task} @ effort={a.effort}: {len(sample)} of {len(items)} items, budget {a.budget}, '
          f'timeout {E.budget_timeout(a.budget)}s', flush=True)

    toks, secs, done, trunc = [], [], 0, 0
    t0 = time.time()
    for i, it in enumerate(sample):
        t = time.time()
        try:
            r = E.ask(a.host, it['prompt'], a.effort, 1.0, 0.95, a.budget, E.budget_timeout(a.budget))
        except Exception as e:
            print(f'  [{i+1}/{len(sample)}] {it["id"]} FAILED {str(e)[:80]}', flush=True)
            continue
        el = time.time() - t
        n = r['usage']['completion_tokens']
        cut = n >= a.budget
        toks.append(n); secs.append(el); done += 1; trunc += cut
        ch = r['choices'][0]
        got = (E.extract(kind, ch['message'].get('content') or '')
               or E.extract(kind, ch['message'].get('reasoning_content') or ''))
        print(f'  [{i+1}/{len(sample)}] {it["id"]:<16} {n:>6} tok {el:>6.0f}s '
              f'{"TRUNC" if cut else "     "} got={got} gold={it["gold"]} '
              f'({(time.time()-t0)/60:.0f} min)', flush=True)

    if not toks:
        sys.exit('no successful items')
    toks_s = sorted(toks)
    def pct(p): return toks_s[min(len(toks_s) - 1, int(p * len(toks_s)))]
    rate = sum(toks) / sum(secs)
    out = dict(task=a.task, effort=a.effort, sampled=done, budget=a.budget,
               truncated=trunc, trunc_rate=round(trunc / done, 3),
               median=st.median(toks_s), p75=pct(.75), p90=pct(.90), p95=pct(.95), max=toks_s[-1],
               mean_tok_s=round(rate, 2), mean_s_per_item=round(sum(secs) / done, 1),
               full_task_hours=round(len(items) * (sum(secs) / done) / 3600, 1))
    print(f'\n  truncated {trunc}/{done} = {100*trunc/done:.0f} %   (target < {100*a.target_trunc:.0f} %)')
    print(f'  completion tokens: median {out["median"]}  p75 {out["p75"]}  p90 {out["p90"]}  '
          f'p95 {out["p95"]}  max {out["max"]}')
    print(f'  throughput {out["mean_tok_s"]} tok/s, {out["mean_s_per_item"]}s per item')
    print(f'  -> the FULL {len(items)}-item task at this budget: ~{out["full_task_hours"]} hours')
    if trunc / done <= a.target_trunc:
        print(f'  -> budget {a.budget} MEETS the truncation target; p95 is {out["p95"]}, so '
              f'{int(out["p95"]*1.2)} would too and would be cheaper.')
    else:
        print(f'  -> budget {a.budget} does NOT meet it. The distribution is censored here, so the '
              f'honest options are a larger budget, or publishing at a lower reasoning effort '
              f'whose traces fit, with the truncation rate disclosed.')
    p = os.path.join(E.OUT, f'calibration_{a.task}.{a.effort}.json')
    os.makedirs(E.OUT, exist_ok=True)
    json.dump(out, open(p, 'w'), indent=2)
    print(f'  wrote {p}')


if __name__ == '__main__':
    main()
