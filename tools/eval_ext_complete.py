#!/usr/bin/env python3
"""eval_ext_complete.py — is a task's merged (extended) file FINISHED?

ONE DEFINITION, TWO CALLERS. eval_force_all.sh must know whether to force the extended row or fall
back to the base one; eval_extend_retry.sh must know whether to run the extension again. Both were
about to answer it with their own inline shell, and two definitions of "finished" that drift apart
is how a chain reports success over an amputated file. So the answer lives here and both ask it.

WHY `[ -s ]` IS NOT THE ANSWER. eval_extend.py copies the traces that TERMINATED across first and
continues the truncated ones one at a time, so a run that dies part way leaves a NON-EMPTY file
holding only the easy half -- 0 % truncated by construction. GPQA-Diamond left exactly such a file
on 2026-08-19 (147 of 198 records) and every downstream `[ -s ]` test would have called it good.

FINISHED means both of:
  * at least as many records as the base run, and
  * the meta file exists, which eval_extend.py writes as its very last act and therefore only when
    it reached the end.

  python3 tools/eval_ext_complete.py gpqa_diamond low low24k          # exit 0 = complete
  python3 tools/eval_ext_complete.py gpqa_diamond low low24k --needs-retry 5
"""
import argparse, json, os, sys

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'evidence', 'evals')


def nlines(path):
    if not os.path.exists(path):
        return 0
    with open(path, 'rb') as f:
        return sum(1 for line in f if line.strip())


def state(task, eff, tag):
    nbase = nlines(os.path.join(OUT, f'{task}.{eff}.jsonl'))
    next_ = nlines(os.path.join(OUT, f'{task}.{tag}.jsonl'))
    meta = os.path.exists(os.path.join(OUT, f'{task}.{tag}.meta.json'))
    ok = bool(nbase) and next_ >= nbase and meta
    return ok, f'{next_}/{nbase} records, meta={"yes" if meta else "no"}'


def trunc_pct(task, eff):
    """Truncation of the BASE row, from the report. A row at or under the gate never needed
    extending, so its absent merged file is correct rather than incomplete."""
    try:
        rows = json.load(open(os.path.join(OUT, 'summary.json')))
    except Exception:
        return None
    r = next((x for x in rows if x['task'] == task and x.get('effort') == eff), None)
    n = (r or {}).get('n') or 0
    return 100.0 * (r.get('truncated') or 0) / n if r and n else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('task')
    ap.add_argument('effort')
    ap.add_argument('tag')
    ap.add_argument('--needs-retry', type=float, metavar='GATE',
                    help='invert: exit 0 when the row is over GATE%% truncated AND incomplete')
    a = ap.parse_args()

    ok, desc = state(a.task, a.effort, a.tag)
    if a.needs_retry is None:
        print(desc)
        return 0 if ok else 1

    pct = trunc_pct(a.task, a.effort)
    if pct is None or pct <= a.needs_retry:
        print(f'{desc}, base truncation {"unknown" if pct is None else f"{pct:.1f}%"} '
              f'— no extension was ever required')
        return 1
    print(desc)
    return 1 if ok else 0


if __name__ == '__main__':
    sys.exit(main())
