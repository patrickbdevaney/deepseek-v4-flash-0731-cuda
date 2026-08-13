#!/usr/bin/env python3
"""eval_fetch.py — pin the benchmark datasets, once, with their commit sha recorded.

`eval_suite.py` never downloads (its rule 2): a benchmark that fetches at eval time can silently
score a different set of rows on a re-run, and then the published number is unreproducible. So the
fetching lives here, is run deliberately, and writes `evidence/evals/datasets.json` mapping each
repo to the exact commit sha the eval read. That file is the provenance record — with it, anyone can
`snapshot_download(..., revision=sha)` and get byte-identical rows.

Downloads parquet only (`allow_patterns`), so this pulls single-digit MB per benchmark rather than
whatever loose files the repo happens to carry.

  python3 tools/eval_fetch.py
"""
import json, os, sys
from huggingface_hub import snapshot_download, HfApi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'evidence', 'evals')

# repo -> patterns. Chosen because each has a machine-checkable gold AND a widely published number
# for frontier models, so the REAP checkpoint can be put next to them without a judge in the loop.
REPOS = [
    ('TIGER-Lab/MMLU-Pro',              ['data/test-*.parquet']),                 # 86.40 published
    ('HuggingFaceH4/MATH-500',          ['*.jsonl', '**/*.parquet']),
    ('google/IFEval',                   ['**/*.parquet', '*.jsonl']),
    ('openai/openai_humaneval',         ['**/*.parquet']),
    ('livecodebench/code_generation_lite', ['**/*.jsonl*']),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, 'datasets.json')
    rec = json.load(open(path)) if os.path.exists(path) else {}
    api = HfApi()
    for repo, pats in REPOS:
        try:
            sha = api.dataset_info(repo).sha
            local = snapshot_download(repo, repo_type='dataset', revision=sha,
                                      allow_patterns=pats)
            n = sum(len(f) for _, _, f in os.walk(local))
            rec[repo] = dict(sha=sha, local=local, files=n)
            print(f'ok   {repo:<40} {sha[:12]}  {n} files')
        except Exception as e:
            print(f'SKIP {repo:<40} {type(e).__name__}: {e}', file=sys.stderr)
    with open(path, 'w') as f:
        json.dump(rec, f, indent=2)
    print(f'\nwrote {path}')


if __name__ == '__main__':
    main()
