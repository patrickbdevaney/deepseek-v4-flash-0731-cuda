#!/usr/bin/env python3
"""eval_budget.py — will the battery finish, and what has to be cut if it will not.

A benchmark plan is a claim about wall clock, and on this box wall clock is the binding constraint
on everything: the engine sustains ~10.6 tok/s at long context, so the whole suite has a hard
ceiling of roughly 0.9M completion tokens per day. A plan that needs 6 weeks is not a plan, and
finding that out on day 12 is the expensive way to learn it.

So the cost of every queued task is estimated from MEASURED per-item token counts where a run or a
calibration exists, and from an explicit stated assumption where it does not -- with the source of
each number printed, so an estimate resting on a guess is never mistaken for one resting on data.

  python3 tools/eval_budget.py                       # the current plan
  python3 tools/eval_budget.py --days 7              # what fits in a week
"""
import argparse, glob, json, os, statistics, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_suite as E

OUT = E.OUT

# Fallback mean completion tokens for tasks with neither a run nor a calibration on disk. Stated,
# not hidden: each is flagged ASSUMED in the output.
ASSUMED = dict(mmlu_pro=4000, humaneval=2500, lcb=6000, math500=5000,
               aime24=7000, aime25=7000, scicode=3500, bfcl_live=300, bfcl=250,
               gpqa_diamond=4200)


def measured(task, effort):
    """Mean completion tokens per item, preferring real records, then a calibration log."""
    p = os.path.join(OUT, f'{task}.{effort}.jsonl')
    if os.path.exists(p):
        toks = [(json.loads(l).get('usage') or {}).get('completion_tokens', 0)
                for l in open(p) if l.strip()]
        toks = [t for t in toks if t]
        if len(toks) >= 8:
            return statistics.mean(toks), f'{len(toks)} scored records'
    c = os.path.join(OUT, f'calib_{task}_{effort}.log')
    if os.path.exists(c):
        toks = []
        for line in open(c, errors='ignore'):
            parts = line.split()
            for i, w in enumerate(parts):
                if w == 'tok' and i and parts[i - 1].isdigit():
                    toks.append(int(parts[i - 1]))
        if len(toks) >= 3:
            return statistics.mean(toks), f'calibration, n={len(toks)}'
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--effort', default='low')
    ap.add_argument('--days', type=float, default=7.0, help='wall-clock ceiling to fit')
    ap.add_argument('--tok-s', type=float, default=0.0, help='0 = read from the live server')
    a = ap.parse_args()

    tps = a.tok_s
    src_tps = 'given'
    if not tps:
        try:
            import urllib.request
            for line in urllib.request.urlopen('http://localhost:8080/metrics', timeout=10) \
                    .read().decode().splitlines():
                if line.startswith('dsv4_decode_tokens_per_second'):
                    tps = float(line.split()[1]); src_tps = 'live server, cumulative'
        except Exception:
            pass
    if not tps:
        tps, src_tps = 10.6, 'default'
    cap = a.days * 86400 * tps

    print(f'sustained {tps:.2f} tok/s ({src_tps})  ->  {tps*86400/1e6:.2f}M tokens/day, '
          f'{cap/1e6:.2f}M in {a.days:g} days\n')

    plan = [t.split(':') for t in os.environ.get(
        'PLAN', 'gpqa_diamond:0:1 bfcl:0:1 mmlu_pro:150:1 humaneval:0:1 lcb:0:1 math500:100:1 '
        'scicode:0:1 bfcl_live:0:1 aime24:0:4 aime25:0:4').split()]

    print(f'{"task":<14}{"items":>7}{"reps":>5}{"tok/item":>10}{"source":>22}'
          f'{"Mtok":>8}{"days":>7}{"cum":>7}')
    total = 0.0
    rows = []
    for task, n, reps in plan:
        n, reps = int(n), int(reps)
        items, _, _, _ = E.TASKS[task](n)
        # what is LEFT, not what the task contains -- a half-finished benchmark is half the cost
        p = os.path.join(OUT, f'{task}.{a.effort}.jsonl')
        already = sum(1 for _ in open(p)) if os.path.exists(p) else 0
        todo = max(0, len(items) * reps - already)
        m, src = measured(task, a.effort)
        if m is None:
            m, src = ASSUMED.get(task, 4000), 'ASSUMED'
        cost = todo * m
        total += cost
        rows.append((task, todo, cost))
        print(f'{task:<14}{todo:>7}{reps:>5}{m:>10.0f}{src:>22}'
              f'{cost/1e6:>8.2f}{cost/(tps*86400):>7.2f}{total/(tps*86400):>7.2f}')

    # The GPQA extension is not in the plan list but is committed work.
    ext = 0
    gp = os.path.join(OUT, f'gpqa_diamond.{a.effort}.jsonl')
    if os.path.exists(gp):
        recs = [json.loads(l) for l in open(gp) if l.strip()]
        rate = sum(1 for r in recs if r.get('truncated')) / max(1, len(recs))
        n_items = len(E.TASKS['gpqa_diamond'](0)[0])
        ext = rate * n_items * 11000            # ~11k extra tokens per continued trace
        total += ext
        print(f'{"gpqa 24k ext":<14}{int(rate*n_items):>7}{1:>5}{11000:>10.0f}'
              f'{"projected from " + format(rate, ".0%"):>22}'
              f'{ext/1e6:>8.2f}{ext/(tps*86400):>7.2f}{total/(tps*86400):>7.2f}')

    days = total / (tps * 86400)
    print(f'\nTOTAL {total/1e6:.2f}M completion tokens = {days:.1f} days at {tps:.1f} tok/s')
    if days <= a.days:
        print(f'FITS in the {a.days:g}-day ceiling with {a.days-days:.1f} days spare.')
    else:
        over = total - cap
        print(f'OVER the {a.days:g}-day ceiling by {days-a.days:.1f} days '
              f'({over/1e6:.2f}M tokens). Cheapest cuts, largest first:')
        for task, todo, cost in sorted(rows, key=lambda r: -r[2])[:4]:
            print(f'  - {task}: {cost/1e6:.2f}M ({cost/(tps*86400):.1f} days)')
        print('  AIME reps 4 -> 2 halves both AIME rows and is defensible on its own: the nested '
              'bootstrap shows reps buy width only when the model is genuinely stochastic.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
