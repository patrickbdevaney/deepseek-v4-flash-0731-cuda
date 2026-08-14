#!/usr/bin/env python3
"""eval_pin_prompts.py — pin the prompt set a run actually used, for runs that predate per-record hashes.

WHY. Prompts are not stored per item; they are re-derived from the pinned dataset plus this repo's
harness. That is sound only if the derivation can be CHECKED, and for runs written before
`prompt_sha256` existed it could not be: a later edit to a prompt template would silently re-pair old
generations with a different question, and every other check in `eval_verify` would still pass,
because it re-derives the GOLD and never the QUESTION.

It is still closeable after the fact, because the prompt set is a pure function of
(pinned dataset snapshot, prompt-construction code). So: rebuild the prompts from the harness AS OF
THE COMMIT THE RUN STARTED AT, rebuild them from HEAD, and require them to be byte-identical. If
they are, the prompt HEAD re-derives is provably the prompt that was sent, and the hash over the
whole set is recorded so any future edit breaks the check loudly.

This does NOT fabricate per-record hashes. It records a set-level proof with the commit it was
proven against, which is a weaker but honest claim, and `eval_verify` reports it as exactly that.

  python3 tools/eval_pin_prompts.py --task gpqa_diamond --effort low --since <run-start-commit>
"""
import argparse, hashlib, importlib.util, json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import eval_suite as E

PIN = os.path.join(E.OUT, 'prompt_provenance.json')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def prompts_of(mod, task):
    items = mod.TASKS[task](0)[0]
    return {i['id']: i['prompt'] for i in items}, {i['id']: str(i['gold']) for i in items}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True)
    ap.add_argument('--effort', required=True)
    ap.add_argument('--since', required=True, help='commit the run started at')
    a = ap.parse_args()

    tmp = os.path.join('/tmp', f'eval_suite_at_{a.since[:12]}.py')
    src = subprocess.run(['git', 'show', f'{a.since}:tools/eval_suite.py'],
                         cwd=ROOT, capture_output=True, text=True)
    if src.returncode:
        sys.exit(f'cannot read tools/eval_suite.py at {a.since}')
    open(tmp, 'w').write(src.stdout)

    old_p, old_g = prompts_of(load(tmp, 'ev_at_run'), a.task)
    new_p, new_g = prompts_of(E, a.task)

    identical = (old_p == new_p) and (old_g == new_g)
    h = hashlib.sha256(''.join(new_p[k] for k in sorted(new_p)).encode()).hexdigest()
    print(f'{a.task}: {len(old_p)} prompts at {a.since[:12]}, {len(new_p)} at HEAD, '
          f'identical={identical}')
    if not identical:
        diff = [k for k in new_p if old_p.get(k) != new_p.get(k)]
        sys.exit(f'PROMPTS CHANGED since the run ({len(diff)} differ, e.g. {diff[:3]}). '
                 f'The stored generations cannot be paired with HEAD\'s prompts. Do not pin.')

    rec = json.load(open(PIN)) if os.path.exists(PIN) else {}
    binp = os.path.join(ROOT, 'build', 'dsv4-server')
    rec[f'{a.task}.{a.effort}'] = dict(
        run_start_commit=a.since,
        prompts_sha256=h,
        n_prompts=len(new_p),
        proven_identical_to=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                           capture_output=True, text=True).stdout.strip(),
        server_sha256=(hashlib.sha256(open(binp, 'rb').read()).hexdigest()[:16]
                       if os.path.exists(binp) else None),
        note=('Run predates per-record prompt_sha256. Prompts and golds rebuilt from the harness at '
              'the run-start commit are byte-identical to HEAD, so HEAD re-derives exactly what was '
              'sent. This is a SET-level proof, weaker than a per-record hash, and is reported as '
              'such.'))
    os.makedirs(E.OUT, exist_ok=True)
    json.dump(rec, open(PIN, 'w'), indent=2)
    print(f'prompts sha256 {h[:32]}  -> pinned in {PIN}')


if __name__ == '__main__':
    main()
