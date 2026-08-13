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

# Reference numbers for the UNPRUNED deepseek-ai/DeepSeek-V4-Flash-0731, at the same reasoning
# effort this harness uses (`high`). Third-party aggregation, unattributed -- see EVALS.md, "The
# comparison problem". A task absent here has no published number to compare against, which is
# itself worth showing rather than hiding.
REF_HIGH = {'gpqa_diamond': 87.40, 'mmlu_pro': 86.40}

LABEL = {'gpqa_diamond': 'GPQA-Diamond', 'mmlu_pro': 'MMLU-Pro', 'aime24': 'AIME 2024',
         'aime25': 'AIME 2025', 'math500': 'MATH-500', 'humaneval': 'HumanEval',
         'gsm8k': 'GSM8K'}


def main():
    if not os.path.exists(SUMMARY):
        sys.exit('no summary.json — run tools/eval_suite.py --report first')
    rows = json.load(open(SUMMARY))
    if not rows:
        sys.exit('summary.json is empty')

    out = ['| benchmark | scored | **acc %** | 95 % CI (Wilson) | trunc | mean out tok | tok/s | '
           'unpruned 0731 @ high |',
           '|---|---:|---:|---|---:|---:|---:|---:|']
    for r in rows:
        ref = REF_HIGH.get(r['task'])
        complete = r['n_total'] and r['n'] >= r['n_total']
        out.append('| {lbl}{star} | {n}/{tot} | **{acc:.1f}** | [{lo:.1f}, {hi:.1f}] | {tr} | '
                   '{mt} | {tps:.1f} | {ref} |'.format(
                       lbl=LABEL.get(r['task'], r['task']), star='' if complete else ' *(partial)*',
                       n=r['n'], tot=r['n_total'] or '?', acc=r['acc'],
                       lo=r['ci'][0], hi=r['ci'][1], tr=r['truncated'],
                       mt=r['mean_completion_tokens'], tps=r['mean_tok_s'] or 0,
                       ref=f'{ref:.2f}' if ref else '— *(none published)*'))

    tot_items = sum(r['n'] for r in rows)
    tot_tok = sum(r['n'] * r['mean_completion_tokens'] for r in rows)
    out += ['',
            f'{tot_items} items scored, {tot_tok:,} completion tokens generated. Sampling held at '
            '`temperature = 1.0`, `top_p = 0.95`, `reasoning_effort = high` throughout.',
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
