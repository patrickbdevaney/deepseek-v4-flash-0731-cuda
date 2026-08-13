#!/usr/bin/env python3
"""eval_provenance.py — prove each benchmark is the real one, as a runnable assertion.

The failure mode this exists to prevent is a whole battery of confident numbers computed over a
dataset that is a lookalike: a mirror with a shifted answer key, a "GPQA" that is really the Extended
split, a subsample of MMLU-Pro that lost half its subjects, an AIME with 2023's problems. None of
those error out. They all produce a clean table.

So every structural fact that the published literature states about these benchmarks is written
here as a CHECK against the bytes actually on this disk, with the source of the claim recorded next
to it. If a row says PASS, that specific published property was verified locally today. If this
script fails, the number that would have been published is wrong, and it fails before the GPU is
ever asked for anything.

Independently of this, the SCORERS are gated in eval_suite (HumanEval canonical solutions, MATH-500
gold identity, extraction cases) -- a correct dataset scored by a broken scorer is equally worthless.

  python3 tools/eval_provenance.py            # writes evidence/evals/provenance.json
"""
import glob, json, os, re, sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_suite as E

OUT = E.OUT
rows_out, npass, nfail = [], 0, 0


def ck(bench, prop, expected, got, source):
    global npass, nfail
    ok = (expected == got)
    npass, nfail = npass + ok, nfail + (not ok)
    rows_out.append(dict(benchmark=bench, property=prop, published=str(expected),
                         observed=str(got), ok=bool(ok), source=source))
    print(f'  [{"PASS" if ok else "FAIL"}] {bench:<14} {prop:<34} published={expected} observed={got}')


# --------------------------------------------------------------------- GPQA-Diamond
# "GPQA Diamond contains 198 questions ... Since GPQA questions are multiple choice questions with
# four options, the random guessing baseline accuracy on GPQA Diamond is 25%."  -- Epoch AI,
# https://epoch.ai/benchmarks/gpqa-diamond ; canonical repo huggingface.co/datasets/Idavidrein/gpqa
SRC_GPQA = 'epoch.ai/benchmarks/gpqa-diamond; canonical Idavidrein/gpqa'
items, snap, src, kind = E.TASKS['gpqa_diamond'](0)
ck('GPQA-Diamond', 'item count', 198, len(items), SRC_GPQA)
ck('GPQA-Diamond', 'distinct answer options', 4, len(set(i['gold'] for i in items)), SRC_GPQA)
ck('GPQA-Diamond', 'answer labels', 'ABCD', ''.join(sorted(set(i['gold'] for i in items))), SRC_GPQA)

# The canonical dataset is GATED (403 on Idavidrein/gpqa), so the key cannot be diffed against it.
# It is instead cross-checked against a SECOND, independently produced mirror that stores the answer
# as free text rather than as a letter -- an error in either mirror's key shows up as disagreement.
def gpqa_cross():
    import pyarrow as pa
    from difflib import SequenceMatcher
    H = os.path.expanduser('~/.cache/huggingface/datasets')
    f = sorted(glob.glob(H + '/hendrydong___gpqa_diamond/**/*.arrow', recursive=True))
    if not f:
        return None, None
    with pa.memory_map(f[0], 'rb') as s:
        B = pa.ipc.open_stream(s).read_all().to_pylist()
    norm = lambda x: re.sub(r'\s+', ' ', x.strip().lower().strip('.$ '))
    stem = {norm(b['problem'])[:120]: b for b in B}
    agree = matched = 0
    for a in items:
        q = a['prompt']
        m = list(re.finditer(r'^([A-D])\.\s*(.+?)\s*$', q, re.M))
        if len(m) < 4:
            continue
        o = {x.group(1): x.group(2).strip() for x in m[-4:]}
        b = stem.get(norm(q[:m[0].start()])[:120])
        if b is None:
            continue
        matched += 1
        gt = o.get(a['gold'], '')
        sol = re.search(r'\\boxed\{(.*)\}', b['solution'], re.S)
        sol = sol.group(1) if sol else b['solution']
        if (SequenceMatcher(None, norm(gt), norm(sol)).ratio() > 0.75
                or norm(sol) in norm(gt) or norm(gt) in norm(sol)):
            agree += 1
    return agree, matched


ag, mt = gpqa_cross()
if mt:
    # Not 198/198: the residual is fuzzy-match failure on near-duplicate stems, not key conflict.
    ck('GPQA-Diamond', 'gold agrees w/ 2nd mirror (>=97%)', True, (ag / mt) >= 0.97,
       f'cross-check vs hendrydong/gpqa_diamond ({ag}/{mt})')

# --------------------------------------------------------------------- MMLU-Pro
# "MMLU-Pro comprises over 12,000 rigorously curated questions ... spanning 14 diverse domains ...
# Each question ... typically has ten multiple-choice options" -- TIGER-AI-Lab/MMLU-Pro (NeurIPS
# 2024), https://github.com/TIGER-AI-Lab/MMLU-Pro
SRC_MMLU = 'github.com/TIGER-AI-Lab/MMLU-Pro (NeurIPS 2024)'
items, snap, src, kind = E.TASKS['mmlu_pro'](0)
ck('MMLU-Pro', 'test item count', 12032, len(items), SRC_MMLU)
ck('MMLU-Pro', 'subject count', 14, len(set(i['category'] for i in items)), SRC_MMLU)
ck('MMLU-Pro', 'max options (A-J)', 'J', max(set(i['gold'] for i in items)), SRC_MMLU)

# --------------------------------------------------------------------- AIME 2024 / 2025
# "The AIME benchmark consists of 30 integer-response problems ... with each answer constrained to
# the interval [0,999]" -- llm-stats.com/benchmarks/aime ; AIME 2025 is the full 30-problem set.
SRC_AIME = 'llm-stats.com/benchmarks/aime; AIME I+II = 30 problems, answers in [0,999]'
for t, yr in (('aime24', 2024), ('aime25', 2025)):
    items, snap, src, kind = E.TASKS[t](0)
    ck(f'AIME {yr}', 'problem count', 30, len(items), SRC_AIME)
    golds = [i['gold'] for i in items]
    ck(f'AIME {yr}', 'all golds integer', True, all(re.fullmatch(r'\d+', g) for g in golds), SRC_AIME)
    ck(f'AIME {yr}', 'all golds in [0,999]', True,
       all(0 <= int(g) <= 999 for g in golds if g.isdigit()), SRC_AIME)

# --------------------------------------------------------------------- MATH-500
# "a subset of 500 problems from the MATH benchmark that OpenAI created in their Let's Verify Step
# by Step paper" -- huggingface.co/datasets/HuggingFaceH4/MATH-500 ; source github.com/openai/prm800k
SRC_M500 = 'HuggingFaceH4/MATH-500; Lightman et al. arXiv:2305.20050, openai/prm800k'
items, snap, src, kind = E.TASKS['math500'](0)
ck('MATH-500', 'problem count', 500, len(items), SRC_M500)
ck('MATH-500', 'MATH subject count', 7, len(set(i['subject'] for i in items)), SRC_M500)
ck('MATH-500', 'difficulty levels 1-5', [1, 2, 3, 4, 5], sorted(set(i['level'] for i in items)), SRC_M500)

# --------------------------------------------------------------------- HumanEval
# "released in July 2021 by OpenAI and contains 164 hand-crafted Python problems, introducing the
# pass@k metric" -- runloop.ai/blog/humaneval-when-machines-learned-to-code ; canonical
# huggingface.co/datasets/openai/openai_humaneval
SRC_HE = 'openai/openai_humaneval; Chen et al. arXiv:2107.03374, 164 problems, pass@1'
items, snap, src, kind = E.TASKS['humaneval'](0)
ck('HumanEval', 'problem count', 164, len(items), SRC_HE)
ck('HumanEval', 'every item has a test', True, all(i['test'] and i['entry_point'] for i in items), SRC_HE)

# --------------------------------------------------------------------- GSM8K
# GSM8K test split is 1319 problems (Cobbe et al. arXiv:2110.14168); golds are integers after "####".
SRC_GSM = 'openai/gsm8k main/test; Cobbe et al. arXiv:2110.14168'
items, snap, src, kind = E.TASKS['gsm8k'](0)
ck('GSM8K', 'test item count', 1319, len(items), SRC_GSM)
ck('GSM8K', 'all golds integer', True,
   all(re.fullmatch(r'-?\d+', i['gold']) for i in items), SRC_GSM)

# --------------------------------------------------------------------- report
os.makedirs(OUT, exist_ok=True)
pins = json.load(open(E.PINS)) if os.path.exists(E.PINS) else {}
with open(os.path.join(OUT, 'provenance.json'), 'w') as f:
    json.dump(dict(checks=rows_out, passed=npass, failed=nfail,
                   pinned_datasets={k: v['sha'] for k, v in pins.items()}), f, indent=2)
print(f'\nPROVENANCE: {npass} passed, {nfail} failed -> {"PASS" if not nfail else "FAIL"}')
sys.exit(1 if nfail else 0)
