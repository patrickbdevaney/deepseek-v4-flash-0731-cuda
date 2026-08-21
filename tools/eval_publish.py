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

# Labels carry the SUBSET in the name where the row is not the whole benchmark. A cell reading
# "BFCL v3" next to a leaderboard number would be read as the leaderboard's quantity; it is not, and
# the qualifier has to travel with the number rather than live in a footnote the reader may skip.
LABEL = {'gpqa_diamond': 'GPQA-Diamond', 'mmlu_pro': 'MMLU-Pro', 'aime24': 'AIME 2024',
         'aime25': 'AIME 2025', 'math500': 'MATH-500', 'humaneval': 'HumanEval',
         'gsm8k': 'GSM8K',
         'bfcl': 'BFCL v3 — 4 `exec_*` categories, AST †',
         'bfcl_live': 'BFCL v3 — `live_*` categories, AST †',
         'lcb': 'LiveCodeBench — `test6` window †',
         'scicode': 'SciCode — subproblem pass rate',
         'bfcl_mt_base': 'BFCL v4 — `multi_turn_base`, state-match ‡',
         'bfcl_mt_miss_func': 'BFCL v4 — `multi_turn_miss_func`, state-match ‡'}

# The full protocol, per benchmark. LiveCodeBench states the comparability condition for all of
# these: scores are comparable only when the release, date window, scenario, metric, sampling count,
# temperature and execution policy match. Harness IDENTITY is not on that list -- protocol is -- so
# the protocol is published in full rather than named by reference to a harness.
#
# (split, scenario, extraction, scoring, execution)
PROTOCOL = {
    'bfcl_mt_base': ('BFCL v4 `multi_turn_base`, all 200', 'stateful multi-turn, prompting mode',
                     "BFCL `default_decode_execute_prompting`",
                     'BFCL `multi_turn_checker` — final backend STATE vs the ground-truth '
                     'sequence, not call-string matching',
                     "calls executed against BFCL's own stateful backends, AST-validated first"),
    'bfcl_mt_miss_func': ('BFCL v4 `multi_turn_miss_func`, all 200',
                     'stateful multi-turn; a required function is withheld and injected at a '
                     'named turn',
                     "BFCL `default_decode_execute_prompting`",
                     'BFCL `multi_turn_checker` — final backend STATE',
                     "calls executed against BFCL's own stateful backends, AST-validated first"),
    'gpqa_diamond': ('test, all 198', '0-shot generative CoT', 'final letter A–D',
                     'exact letter match', 'none'),
    'mmlu_pro':     ('test, seeded random subset of 12 032', '0-shot generative CoT',
                     'final letter A–J', 'exact letter match', 'none'),
    'math500':      ('test, seeded random subset of 500', '0-shot generative CoT',
                     'last brace-balanced \\boxed{}', 'string → numeric → sympy equivalence', 'none'),
    'aime24':       ('all 30', '0-shot generative CoT', 'final integer 0–999',
                     'exact integer match', 'none'),
    'aime25':       ('all 30', '0-shot generative CoT', 'final integer 0–999',
                     'exact integer match', 'none'),
    'humaneval':    ('all 164', '0-shot', 'first ```python block',
                     "pass@1 — the benchmark's own check(entry_point)",
                     'sandboxed subprocess, 20 s, 2 GiB address space'),
    'bfcl':         ('4 of BFCL v3\'s `exec_*` categories, 240 items',
                     'prompt mode — calls emitted as text, not via a tool-call API',
                     'Python call expressions, one per line',
                     "BFCL's own AST metric: same function, argument names and values",
                     'none — AST comparison, not execution'),
    'scicode':      ('test split, 65 research problems → 291 subproblems, 16 science subfields',
                     '0-shot per subproblem; the model\'s OWN earlier steps are in scope, no gold '
                     'solutions exist', 'first ```python block',
                     "the benchmark's own numeric tests, all must pass",
                     'sandboxed subprocess, 90 s, 4 GiB address space'),
    'bfcl_live':    ('6 `live_*` categories, seeded cap of 150 per category',
                     'prompt mode — calls emitted as text',
                     'Python call expressions, or the literal NO_CALL',
                     "BFCL possible-answer AST match; relevance categories scored on whether a call "
                     'was emitted at all', 'none — AST comparison, not execution'),
    'lcb':          ('`code_generation_lite` `test6`, window 2025-01 → 2025-04',
                     '0-shot, complete program', 'first ```python block',
                     'all tests pass (public + private)',
                     'sandboxed subprocess, 8 s per test, 2 GiB address space'),
}

# Deviations from the canonical protocol, stated rather than discovered by a reader. Each says which
# DIRECTION it biases the score, because a limitation whose sign is unknown is not disclosed.
DEVIATIONS = {
    'bfcl_mt_base': ['PROMPTING mode, not native function-calling: function docs are placed in the '
                     'system prompt as text and the model is asked for `[func(a=1), ...]`. This is '
                     "BFCL's `prompt` column, not its `FC` column, and it matches how `exec_*` and "
                     '`live_*` were run here, so the three rows are comparable to each other. '
                     'Direction of bias: unknown — a model may do better or worse than with its '
                     'native tool tokens.',
                     "execution results are returned in the CHECKPOINT'S native `<tool_result>` "
                     'wrapper (CHAT_FORMAT.md) rather than BFCL\'s `role: tool` message, which '
                     "relies on each model's chat template. Direction of bias: favours this model "
                     'slightly, since results arrive in the form it was trained on.',
                     'BFCL v4 data, while the `exec_*`/`live_*` rows above are v3. v3 and v4 share '
                     'all 200 ids but only ~92 are byte-identical. v4 is used because it matches '
                     'the version of the checker that scores it; both self-gate at 200/200.'],
    'bfcl_mt_miss_func': ['same three deviations as `bfcl_mt_base`.'],
    'bfcl': ['runs 4 `exec_*` categories (240 items), **not** the BFCL v3 aggregate, which also '
             'spans multi-turn, relevance/irrelevance detection and non-Python languages — '
             '**this row is not the BFCL leaderboard quantity**',
             'scored by AST match rather than live execution, so a call that is structurally right '
             'but fails against a real API counts correct (biases **up**)'],
    'lcb': ['at most 25 tests per problem, so a solution failing test 30 of 200 counts correct '
            '(biases **up**)',
            'single sample per problem, not the official multi-sample pass@1 average',
            'the `0731` checkpoint postdates every problem in the window, so contamination is '
            'reduced but **not** eliminated (biases **up**)'],
    'bfcl_live': ['a seeded cap of 150 items per category, because the `live_*` sets run from 1052 '
                  'rows (`live_multiple`) to 16 (`live_parallel`) and taking all of them would let '
                  'two categories set the headline',
                  'category-balanced rather than BFCL\'s own weighting, so this is **not** the BFCL '
                  'leaderboard aggregate'],
    'scicode': ['subproblem pass rate is the headline; main-problem accuracy (every subproblem of a '
                'problem correct) is the harder published metric and is derived from the same '
                'records rather than reported in its place',
                'later steps run against the model\'s OWN earlier code, since no gold solutions '
                'ship — a wrong early step propagates, which is the intended behaviour and biases '
                '**down** relative to a gold-context setting'],
    'mmlu_pro': ['a seeded random subset, not the full split — unbiased in expectation, but wider '
                 'than the published interval on the full set'],
    'math500': ['a seeded random subset, not the full 500'],
}

# BUDGET FORCING IS A PROPERTY OF THE RUN, NOT OF THE BENCHMARK, so it cannot live in DEVIATIONS
# above -- that dict is keyed by task, and attaching this to `gpqa_diamond` would libel the
# unforced `low` row of the same task. It is keyed off the effort tag instead (tools/eval_force.py
# emits `<effort><cap>kforced`).
#
# WHY THIS EXISTS AT ALL. A forced row reports ~0 % truncation and therefore sails through the
# NOT-QUOTABLE gate below -- which is correct, it genuinely is not budget-limited, and also exactly
# how a protocol deviation would come to be published invisibly. The gate that catches truncation
# cannot catch this, so this does.
FORCED_SUFFIX = 'forced'
FORCED_DEVIATION = [
    'BUDGET FORCING. Traces that hit the thinking cap were **not** given more thinking room. The '
    'end-of-thinking delimiter was appended to the exact stored prefix and the model answered from '
    'the reasoning it had already produced. Truncation is therefore ~0 % **by construction**, not '
    'because the traces finished. Precedent: s1, Muennighoff et al. 2025 (arXiv:2501.19393).',
    'This row is **not comparable** with an unforced published reference number, and the reference '
    'column is suppressed for it. It answers "what would the model say if made to stop here", '
    'which is a different question from "what does the model say".',
    'Direction of bias: **unknown, and not assumed**. Forcing rescues items that were going to '
    'score at chance, which biases up; it also cuts off reasoning that had not yet reached its '
    'answer, which biases down. `tools/eval_force.py --pair-with` measures the net on the same '
    'prefixes with an exact McNemar test rather than arguing about it.',
]


def is_forced(r):
    return str(r.get('effort', '')).endswith(FORCED_SUFFIX)



def engine_provenance_block():
    """The configurations that produced these rows, and what may be pooled across them.

    The protocol block above answers LiveCodeBench's comparability condition -- release, window,
    scenario, metric, sampling count, temperature, execution policy. None of those name the
    SPECULATOR, the engine revision or the draft width, so a battery suspended under one
    configuration and finished under another publishes as one homogeneous run unless this says
    otherwise. evidence/evals/RUN_PROVENANCE.jsonl is stamped whenever the battery is (re)started.
    """
    p = 'evidence/evals/RUN_PROVENANCE.jsonl'
    if not os.path.exists(p):
        return []
    recs = []
    for line in open(p):
        line = line.strip()
        if line:
            try:
                recs.append(json.loads(line))
            except ValueError:
                pass
    if not recs:
        return []
    out = ['', '### Engine configuration', '',
           'Accuracy columns pool across these configurations and throughput columns do not. The '
           'engine\'s speculative acceptance rule draws every emitted token from the **target** '
           'model\'s distribution, so which draft head is loaded does not bias the output -- items '
           'scored under different heads are independent draws from the same distribution. `tok/s` '
           'is a joint property of engine, head and draft width: on one frozen protocol, across '
           'five decode-kernel rewrites, suite `tau` reproduced to four decimal places while suite '
           'tok/s moved -2.3 % and -5.0 % and base AR moved -17.4 %.', '',
           '| stamped | draft head | block | engine rev |', '|---|---|---|---|']
    for r in recs:
        out.append('| %s | `%s` | %s | `%s`%s |'
                   % (str(r.get('stamped', '?'))[:19], r.get('head', '?'), r.get('block', '?'),
                      r.get('engine_rev', '?'), ' (dirty)' if r.get('engine_dirty') else ''))
    if len(recs) > 1:
        out += ['', 'More than one row above means the battery spanned a configuration change. '
                    'Rows completed before a stamp were produced by the configuration above it.']
    return out


def main():
    if not os.path.exists(SUMMARY):
        sys.exit('no summary.json — run tools/eval_suite.py --report first')
    rows = json.load(open(SUMMARY))
    if not rows:
        sys.exit('summary.json is empty')

    out = ['| benchmark | effort | scored | **acc %** | 95 % CI | trunc | err | '
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
        # A TRUNCATED ITEM IS SCORED WRONG, so past a few percent the row stops being a measurement
        # of the model and becomes a measurement of max_tokens. GPQA at an 8000-token cap truncated
        # 31.7 % and scored 65.8 % pooled against 91.5 % on the traces that terminated -- a 26-point
        # gap that no reader could infer from a "trunc" count sitting in a column. Say it in the row.
        rate = (r.get('truncated', 0) / r['n']) if r.get('n') else 0.0
        if rate > 0.05:
            complete = False
            r = dict(r, _flag=r.get('_flag', '') +
                     f' **— NOT QUOTABLE, {rate:.0%} truncated at max_tokens={r.get("max_tokens")}; '
                     f'this measures the budget, not the model. Extend with tools/eval_extend.py**')
        # A forced row is quotable -- but only as a forced row. Mark it in the cell so the label
        # travels with the number if someone copies a single line out of the table.
        if is_forced(r):
            r = dict(r, _flag=r.get('_flag', '') + ' ‡ **budget-forced**')
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
            'Interval method per row: ' + ', '.join(
                sorted({f'{LABEL.get(r["task"], r["task"])} — {r.get("ci_method", "wilson")}'
                        for r in rows})) + '. Single-sample tasks use a Wilson score interval; '
            'avg@k tasks use a nested bootstrap over PROBLEMS, because k samples of one problem are '
            'not k problems and pooling them into a Wilson interval understates the width by up to '
            '2x — most severely when the extra samples bought the least.',
            '',
            'Per-task provenance (dataset and pinned snapshot):', '',
            '| benchmark | dataset | snapshot | max_tokens |', '|---|---|---|---:|']
    for r in rows:
        out.append(f'| {LABEL.get(r["task"], r["task"])} | {r["source"]} | '
                   f'`{(r["snapshot"] or "")[:12]}` | {r["max_tokens"]} |')
    if any(r['task'] in DEVIATIONS for r in rows):
        out += ['', '† This row is **not** the full benchmark — see the protocol block below.']
    if any(is_forced(r) for r in rows):
        out += ['', '‡ This row was **budget-forced**: its ~0 % truncation is by construction, not '
                'because the traces finished. It is not comparable with an unforced reference — '
                'see the protocol block below.']

    proto = protocol_block(rows) + engine_provenance_block()

    doc = open(DOC).read()
    a, b = doc.index('<!-- RESULTS -->'), doc.index('<!-- /RESULTS -->')
    doc = doc[:a] + '<!-- RESULTS -->\n' + '\n'.join(out) + '\n' + doc[b:]
    a, b = doc.index('<!-- PROTOCOL -->'), doc.index('<!-- /PROTOCOL -->')
    doc = doc[:a] + '<!-- PROTOCOL -->\n' + '\n'.join(proto) + '\n' + doc[b:]
    open(DOC, 'w').write(doc)
    print('\n'.join(out))
    print(f'\nwrote {DOC} (results + protocol)')


def protocol_block(rows):
    """The full evaluation protocol, emitted so comparability can be judged rather than assumed.

    LiveCodeBench states the condition plainly: scores are comparable only when the release, date
    window, scenario, metric, sampling count, temperature and execution policy all match. None of
    those are implied by naming a harness -- and naming one would not settle it anyway, since
    lm-evaluation-harness itself ships GPQA in both a loglikelihood and a generative-CoT variant
    that disagree on the same model. So the protocol is published in full, and every deviation is
    listed WITH THE DIRECTION IT BIASES THE SCORE, because a limitation whose sign is unknown has
    not really been disclosed.
    """
    out = ['| benchmark | split / subset | scenario | answer extraction | scoring | execution | '
           'budget | samples |',
           '|---|---|---|---|---|---|---:|---:|']
    for r in rows:
        p = PROTOCOL.get(r['task'])
        if not p:
            continue
        out.append(f'| {LABEL.get(r["task"], r["task"])} | ' + ' | '.join(p) +
                   f' | {r.get("max_tokens")} | {r.get("reps", 1)} |')

    # A ROW WITH NO RECORDED SAMPLING PARAMS CANNOT CONTRIBUTE A CLAIM ABOUT SAMPLING, AND MUST NOT
    # BRING THE PUBLISH DOWN. An extension that dies before writing its meta leaves a row whose
    # temperature and top_p are None; sorting None against a float raises TypeError, and on
    # 2026-08-19 that killed the landing of aime24 -- a task that had extended perfectly, 60 of 60 --
    # because two OTHER tasks had partial merged files. A row that cannot describe its own decoding
    # is dropped from this sentence and stays out of it until its meta exists.
    samp = sorted({(r.get('temperature'), r.get('top_p'), r.get('effort', '?')) for r in rows
                   if r.get('temperature') is not None and r.get('top_p') is not None},
                  key=lambda t: (float(t[0]), float(t[1]), str(t[2])))
    out += ['', '**Decoding.** ' + ('; '.join(
        f'`{e}` reasoning effort at temperature {t}, top_p {tp}' for t, tp, e in samp)
        or 'not recorded for any row in this table') +
        '. Sampling is stochastic rather than greedy, which is why interval width and avg@k are '
        'reported rather than a bare point estimate. Per-benchmark token budget and sample count '
        'are in the table above; each budget was set from an uncensored length calibration, not '
        'chosen.']

    out += ['', '**Intervals.** Single-sample tasks use a Wilson score interval. avg@k tasks use a '
            'nested bootstrap over problems: k samples of one problem are not k independent '
            'problems, and pooling them into a Wilson interval understates the width by up to 2x '
            'here — most severely when the extra samples bought the least. This is the clustering '
            'correction of Miller, *Adding Error Bars to Evals* (arXiv:2411.00640), which reports '
            'cluster-adjusted standard errors up to 3x the naive ones.']

    dev = [(t, DEVIATIONS[t]) for t in dict.fromkeys(r['task'] for r in rows) if t in DEVIATIONS]
    if dev:
        out += ['', '**Deviations from the canonical protocol.** Each is stated with the direction '
                'it moves the score.', '']
        for t, ds in dev:
            out.append(f'- **{LABEL.get(t, t)}**')
            out += [f'  - {d}' for d in ds]

    forced = [r for r in rows if is_forced(r)]
    if forced:
        names = ', '.join(dict.fromkeys(
            f'{LABEL.get(r["task"], r["task"])} (`{r["effort"]}`)' for r in forced))
        out += ['', f'**Budget forcing ‡** — applies to: {names}.', '']
        out += [f'- {d}' for d in FORCED_DEVIATION]

    out += ['', '**What this protocol does and does not license.** Every number here is '
            're-derivable from the stored generations by `tools/eval_verify.py` with no GPU and no '
            'model, against a sha-pinned dataset. That makes the numbers *reproducible*. It does '
            'not make them *leaderboard-rankable*: the reference column is third-party aggregation '
            'whose harness, token budget and sampling count are not ours and are not stated, so a '
            'gap of a few points between a row and its reference is not interpretable in either '
            'direction. The comparison this evidence genuinely supports is a paired one — REAP '
            'against unpruned, same harness, same decoding, only the weights different — and that '
            'baseline has not been run here.']
    return out


if __name__ == '__main__':
    main()
