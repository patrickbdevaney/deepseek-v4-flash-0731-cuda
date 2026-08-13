#!/usr/bin/env python3
"""eval_verify.py — re-derive every published number from the raw generations, independently.

A results table is a claim. This is the audit that makes it checkable by someone who does not
trust the process that produced it, and it uses NOTHING from that process except the raw text the
model emitted:

  1. Re-read every stored generation from `evidence/evals/<task>.jsonl`.
  2. Re-extract the answer from that text with the scorer, ignoring the `got` field on the record.
  3. Re-derive the gold from the PINNED DATASET by item id -- not from the `gold` field on the
     record -- so a corrupted or hand-edited gold cannot survive.
  4. Re-score, and require the recomputed accuracy to equal what `summary.json` publishes.

It also checks the things that would let a table be right for the wrong reason: duplicate ids, ids
that do not exist in the benchmark, records with no generated text and no recorded error, and
`correct` flags that disagree with a fresh scoring of the same text.

Anyone can run this on a clone of the repo with no GPU and no model:

  python3 tools/eval_verify.py                # every task that has results
  python3 tools/eval_verify.py --task gpqa_diamond
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_suite as E

OUT = E.OUT
nfail = 0


def ck(task, what, ok, detail=''):
    global nfail
    nfail += (not ok)
    print(f'  [{"PASS" if ok else "FAIL"}] {task:<14} {what:<40} {detail}', flush=True)


def verify(task, published):
    path = os.path.join(OUT, f'{task}.jsonl')
    if not os.path.exists(path):
        return None
    recs = [json.loads(l) for l in open(path) if l.strip()]

    # The benchmark itself, read fresh from the pinned snapshot. `--n 0` so every id in the file can
    # be looked up regardless of what subset the run used.
    items, snap, src, kind = E.TASKS[task](0)
    by_id = {it['id']: it for it in items}

    # ids: unique, and every one really belongs to this benchmark (strip the avg@k "#rN" suffix).
    ids = [r['id'] for r in recs]
    base = [i.split('#r')[0] for i in ids]
    ck(task, 'no duplicate records', len(set(ids)) == len(ids),
       f'{len(ids)} records, {len(set(ids))} unique')
    unknown = [b for b in base if b not in by_id]
    ck(task, 'every id exists in the benchmark', not unknown,
       f'{len(unknown)} unknown' + (f' e.g. {unknown[:2]}' if unknown else ''))

    # gold on the record must equal gold re-derived from the dataset
    badgold = [r['id'] for r in recs
               if r['id'].split('#r')[0] in by_id
               and str(r.get('gold')) != str(by_id[r['id'].split('#r')[0]]['gold'])]
    ck(task, 'gold matches the pinned dataset', not badgold,
       f'{len(badgold)} mismatched' + (f' e.g. {badgold[:2]}' if badgold else ''))

    # every record either has generated text or a recorded engine error
    empty = [r['id'] for r in recs
             if not (r.get('content') or r.get('reasoning')) and not r.get('error')]
    ck(task, 'no silent empty generations', not empty, f'{len(empty)} empty')

    # RE-SCORE from the raw text. `got` and `correct` on the record are ignored entirely.
    redone = 0
    disagree = []
    for r in recs:
        b = r['id'].split('#r')[0]
        if b not in by_id:
            continue
        it = by_id[b]
        if r.get('error'):
            ok = False
        else:
            got = E.extract(kind, r.get('content') or '') or E.extract(kind, r.get('reasoning') or '')
            ok = bool(E.correct(kind, got, it['gold'], it))
        redone += ok
        if ok != bool(r.get('correct')):
            disagree.append(r['id'])
    ck(task, 'stored correct flags reproduce', not disagree,
       f'{len(disagree)} disagree' + (f' e.g. {disagree[:2]}' if disagree else ''))

    acc = 100.0 * redone / len(recs) if recs else 0.0
    if published is not None:
        ck(task, 'published accuracy is reproducible', abs(acc - published['acc']) < 0.05,
           f'recomputed {acc:.1f} % vs published {published["acc"]:.1f} % over {len(recs)} records')
        ck(task, 'snapshot matches the published one',
           (published.get('snapshot') or '').startswith(snap[:12]) or snap.startswith(
               (published.get('snapshot') or '')[:12]),
           f'{snap[:12]} vs {(published.get("snapshot") or "?")[:12]}')
    else:
        print(f'  [ .. ] {task:<14} recomputed {acc:.1f} % over {len(recs)} records (not yet published)')
    return dict(task=task, records=len(recs), recomputed_acc=round(acc, 2),
                published_acc=(published or {}).get('acc'), snapshot=snap, source=src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', default=None)
    a = ap.parse_args()
    spath = os.path.join(OUT, 'summary.json')
    pub = {r['task']: r for r in json.load(open(spath))} if os.path.exists(spath) else {}

    rows = []
    for task in E.TASKS:
        if a.task and task != a.task:
            continue
        r = verify(task, pub.get(task))
        if r:
            rows.append(r)
    with open(os.path.join(OUT, 'verification.json'), 'w') as f:
        json.dump(dict(tasks=rows, failed=nfail), f, indent=2)
    print(f'\nVERIFY: {nfail} failed -> {"PASS" if not nfail else "FAIL"}')
    return 1 if nfail else 0


if __name__ == '__main__':
    sys.exit(main())
