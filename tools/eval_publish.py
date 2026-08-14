#!/usr/bin/env python3
"""eval_publish.py — rewrite the RESULTS block of EVALS.md from evidence/evals/summary.json.

The table is generated rather than written so that the published number and the scored jsonl cannot
drift apart. Hand-transcribing a benchmark table is exactly the kind of step that puts a wrong digit
in front of a reader who has no way to check it.

  python3 tools/eval_suite.py --report && python3 tools/eval_publish.py
"""
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY = os.path.join(ROOT, 'evidence', 'evals', 'summary.json')
DOC = os.path.join(ROOT, 'EVALS.md')

# Reference numbers for the UNPRUNED deepseek-ai/DeepSeek-V4-Flash-0731, PER REASONING EFFORT.
# Third-party aggregation, unattributed -- see EVALS.md, "The comparison problem".
#
# Effort-keyed on purpose: the aggregator's own table moves GPQA by 16 points and LiveCodeBench by
# 36 between `standard` and `high`, so holding a low-effort result against a high-effort number is
# not a conservative comparison, it is the wrong one. A task absent here has no published number at
# that effort, which is worth showing rather than hiding.
REF = {
    'low':  {'gpqa_diamond': 71.20, 'mmlu_pro': 83.00},   # aggregator's "standard" column
    'high': {'gpqa_diamond': 87.40, 'mmlu_pro': 86.40},
    'max':  {'gpqa_diamond': 88.10, 'mmlu_pro': 86.20},
}

LABEL = {'gpqa_diamond': 'GPQA-Diamond', 'mmlu_pro': 'MMLU-Pro', 'aime24': 'AIME 2024',
         'aime25': 'AIME 2025', 'math500': 'MATH-500', 'humaneval': 'HumanEval',
         'gsm8k': 'GSM8K'}


def main():
    if not os.path.exists(SUMMARY):
        sys.exit('no summary.json — run tools/eval_suite.py --report first')
    rows = json.load(open(SUMMARY))
    if not rows:
        sys.exit('summary.json is empty')

    out = ['| benchmark | effort | scored | **acc %** | 95 % CI (Wilson) | trunc | err | '
           'mean out tok | tok/s | unpruned 0731 @ same effort |',
           '|---|---|---:|---:|---|---:|---:|---:|---:|---:|']
    for r in rows:
        eff = r.get('effort', 'high')
        ref = REF.get(eff, {}).get(r['task'])
        complete = r['n_total'] and r['n'] >= r['n_total']
        # A partial run that has not covered every stratum is not merely noisier, it is BIASED --
        # see the coverage note in eval_suite.report(). Say so in the table instead of letting
        # "(partial)" imply the only cost was sample size.
        st, sn = r.get('strata_total'), r.get('strata_seen')
        if not complete and st and sn is not None and sn < st:
            complete = False
            r = dict(r, _flag=f' **— NOT QUOTABLE, {sn} of {st} {r.get("strata_field","strata")} strata**')
        out.append('| {lbl}{star} | **{eff}** | {n}/{tot} | **{acc:.1f}** | [{lo:.1f}, {hi:.1f}] | '
                   '{tr} | {er} | {mt} | {tps:.1f} | {ref} |'.format(
                       eff=eff,
                       lbl=LABEL.get(r['task'], r['task']),
                       star=('' if complete else ' *(partial)*') + r.get('_flag', ''),
                       n=r['n'], tot=r['n_total'] or '?', acc=r['acc'],
                       lo=r['ci'][0], hi=r['ci'][1], tr=r['truncated'], er=r.get('errors', 0),
                       mt=r['mean_completion_tokens'], tps=r['mean_tok_s'] or 0,
                       ref=f'{ref:.2f}' if ref else '— *(none published)*'))

    tot_items = sum(r['n'] for r in rows)
    tot_tok = sum(r['n'] * r['mean_completion_tokens'] for r in rows)
    out += ['',
            f'{tot_items} items scored, {tot_tok:,} completion tokens generated. Sampling held at '
            '`temperature = 1.0`, `top_p = 0.95`; the reference column is the aggregator number at '
            'the SAME reasoning effort as the row.',
            '',
            'Per-task provenance (dataset and pinned snapshot):', '',
            '| benchmark | dataset | snapshot | max_tokens |', '|---|---|---|---:|']
    for r in rows:
        out.append(f'| {LABEL.get(r["task"], r["task"])} | {r["source"]} | '
                   f'`{(r["snapshot"] or "")[:12]}` | {r["max_tokens"]} |')

    doc = open(DOC).read()
    a, b = doc.index('<!-- RESULTS -->'), doc.index('<!-- /RESULTS -->')
    doc = doc[:a] + '<!-- RESULTS -->\n' + '\n'.join(out) + '\n' + doc[b:]
    open(DOC, 'w').write(doc)
    print('\n'.join(out))
    print(f'\nwrote {DOC}')


if __name__ == '__main__':
    main()
