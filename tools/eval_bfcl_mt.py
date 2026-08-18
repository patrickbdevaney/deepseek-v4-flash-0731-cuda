#!/usr/bin/env python3
"""eval_bfcl_mt.py — BFCL multi-turn: stateful, multi-hop, tool-executing trajectories.

WHY THIS AND NOT MORE OF WHAT WE HAVE. `exec_*` and `live_*` are single-shot: one prompt, one set
of calls, scored by matching the call against a ground-truth call. Nothing about them can observe
the failure mode that matters most for an agentic coding model and that a REAP most plausibly
damages -- a small error early in a trajectory that the model then builds on. Multi-turn is the
only category family that can: the model's calls are EXECUTED against live state, the results come
back, and the next turn depends on what the previous turn actually did to the world. Scoring
compares the FINAL STATE of the backends against the state the ground-truth call sequence produces,
so a trajectory that ends in the right place by a different route passes, and one that drifts
early fails no matter how well-formed its later calls are.

WE USE BFCL'S OWN CODE, NOT A REIMPLEMENTATION. The eight stateful backends (GorillaFileSystem,
TradingBot, TravelAPI, VehicleControlAPI, TwitterAPI, TicketAPI, MessageAPI, MathAPI), the state
comparison, the system-prompt assembly and the response decoder all come from the published
`bfcl_eval` package. Reimplementing a file system simulator from its docstrings would produce a
number that is not comparable to anything, and the divergences would be silent. The package lives
in `.venv-bfcl` because its full dependency set (faiss, sentence-transformers, a pinned numpy)
must not touch the environment the rest of this repo runs in; only the light deps are installed.

SELF-GATE. Replaying each item's ground truth as if it were the model's output scores
**400/400 on both v3 and v4 data, zero mismatches, zero exceptions**. An integration that cannot
score the ground truth perfectly is measuring itself, so this gate runs before any model does
(`--self-gate`), and it is the reason the v3-vs-v4 question was decided by measurement rather than
argument: both pass identically, so v4 is used because it matches the checker's version and is the
current leaderboard.

TWO PROTOCOL DEVIATIONS, BOTH DELIBERATE, BOTH REPORTED:
  1. PROMPTING MODE, not native function-calling. Function docs go into the system prompt as text
     and the model is asked for `[func(a=1), ...]`, which is BFCL's "prompt" column rather than its
     "FC" column. This matches how `exec_*` and `live_*` were already run here, so the three rows
     are comparable to each other.
  2. Execution results are returned in the CHECKPOINT'S native `<tool_result>` wrapper
     (CHAT_FORMAT.md), not BFCL's `role: tool` message. BFCL relies on each model's chat template
     to render that role; ours is applied here, and this is what this checkpoint documents.

SAFETY. BFCL executes model output with `eval()`. Its guard is a blocklist of bare names
(`kill`, `exit`, `remove`, ...) which `__import__('os').system(...)` walks straight past. Our own
model is not an adversary, but a malformed generation should not be able to reach the filesystem,
so every decoded call is AST-validated here before it is allowed near `eval`: it must parse to a
single Call whose func is a plain Name and whose arguments are literals. Anything else is recorded
as a malformed call -- which is also the correct SCORING outcome, since BFCL expects well-formed
calls.

  .venv-bfcl/bin/python tools/eval_bfcl_mt.py --self-gate
  .venv-bfcl/bin/python tools/eval_bfcl_mt.py --category base --n 0
"""
import argparse, ast, contextlib, hashlib, io, json, os, re, sys, time
import urllib.error, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'evidence', 'evals')

from bfcl_eval.constants.default_prompts import (
    DEFAULT_SYSTEM_PROMPT_FORMAT, MAXIMUM_STEP_LIMIT,
    DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING)
from bfcl_eval.constants.executable_backend_config import MULTI_TURN_FUNC_DOC_FILE_MAPPING
from bfcl_eval.model_handler.utils import (
    formulate_system_prompt, default_decode_execute_prompting)
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    execute_multi_turn_func_call, is_empty_execute_response)
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker
import bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils as MTU

PKG = os.path.dirname(os.path.abspath(__import__('bfcl_eval').__file__))
DATA = os.path.join(PKG, 'data')
DOCS = os.path.join(DATA, 'multi_turn_func_doc')

# The serving chat format. Same tokens eval_suite.py and eval_extend.py use; a trace built any
# other way is not a trace of what this server does.
BOS = '<｜begin▁of▁sentence｜>'
USER_SP, ASSISTANT_SP, EOS = '<｜User｜>', '<｜Assistant｜>', '<｜end▁of▁sentence｜>'
MODEL_NAME = 'dsv4-flash-0731-reap'


def load_category(cat):
    items = [json.loads(l) for l in open(os.path.join(DATA, f'BFCL_v4_multi_turn_{cat}.json'))
             if l.strip()]
    gts = {g['id']: g['ground_truth'] for g in
           (json.loads(l) for l in
            open(os.path.join(DATA, 'possible_answer', f'BFCL_v4_multi_turn_{cat}.json'))
            if l.strip())}
    for it in items:
        # Multi-turn items ship without function docs; they are assembled from involved_classes.
        # For miss_func the held-out docs are REMOVED here and injected at their turn, which is
        # the whole point of that category: the model must first recognise it cannot comply.
        funcs = []
        for cls in it['involved_classes']:
            # The func-doc files are JSONL, one function per line -- not a JSON array.
            with open(os.path.join(DOCS, MULTI_TURN_FUNC_DOC_FILE_MAPPING[cls])) as fh:
                funcs.extend(json.loads(l) for l in fh if l.strip())
        holdout = it.get('missed_function') or {}
        held = {}
        for tidx, names in holdout.items():
            held[tidx] = []
            for nm in names:
                for i, fd in enumerate(funcs):
                    if fd['name'] == nm:
                        held[tidx].append(funcs.pop(i))
                        break
        it['function'] = funcs
        it['_holdout'] = held
    return items, gts


SAFE_NODES = (ast.Constant, ast.List, ast.Tuple, ast.Dict, ast.Set, ast.UnaryOp, ast.USub, ast.UAdd)

# The shape check alone is not enough: `exec("...")` is a single Call with a Name func and a
# literal argument, so it passes on shape and would reach the builtin. BFCL's own blocklist misses
# it too (it covers kill/exit/quit/remove/unlink/popen/Popen/run). Unknown names are deliberately
# still EXECUTED -- a call to a function that does not exist raises inside eval and the model gets
# that error back as feedback, which is the behaviour BFCL scores. Only these are refused.
DANGEROUS = {
    'exec', 'eval', 'compile', '__import__', 'open', 'input', 'breakpoint', 'exit', 'quit',
    'globals', 'locals', 'vars', 'getattr', 'setattr', 'delattr', 'setenv', 'system',
    'kill', 'remove', 'unlink', 'rmtree', 'popen', 'Popen', 'run', 'spawn', 'fork', 'chdir',
}


def call_is_safe(call_str):
    """A decoded call may reach eval() only if it is literally a call with literal arguments.

    BFCL's own guard checks a blocklist of bare names, which `__import__('os').system(...)` does
    not trip. This is the complement: an allowlist on SHAPE. Attribute access, nested calls,
    comprehensions and names as arguments are all rejected, which leaves nothing that can reach
    the interpreter's namespace.
    """
    try:
        tree = ast.parse(call_str.strip(), mode='eval')
    except Exception:
        return False
    node = tree.body
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        return False
    if node.func.id in DANGEROUS:
        return False
    for arg in list(node.args) + [kw.value for kw in node.keywords]:
        for sub in ast.walk(arg):
            if not isinstance(sub, SAFE_NODES + (ast.Load,)):
                return False
    return True


def clear_state(test_entry_id):
    """Drop the backend instances BFCL parks in module globals, keyed by test id.

    execute_multi_turn_func_call caches instances in globals() under
    `{model}_{test_entry_id}_{Class}_instance` so that state survives across turns. That is exactly
    right within an item and exactly wrong across a re-run of the same item -- a resumed or
    repeated entry would inherit the previous attempt's world. Cleared between items.
    """
    safe = re.sub(r'[-./:]', '_', test_entry_id)
    g = vars(MTU)                       # `globals()[name] = instance` inside that module
    for k in [k for k in list(g) if safe in k and k.endswith('_instance')]:
        g.pop(k, None)


def complete(host, prompt, max_tokens, timeout, temp=0.0):
    body = json.dumps(dict(prompt=prompt, temperature=temp, top_p=1.0,
                           max_tokens=max_tokens)).encode()
    req = urllib.request.Request(f'http://{host}/v1/completions', data=body,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def strip_think(text):
    e = text.find('</think>')
    if e == -1:
        return '', text
    r = text[:e]
    if r.startswith('<think>'):
        r = r[len('<think>'):]
    return r, text[e + len('</think>'):]


def run_item(it, host, max_tokens, timeout, log):
    """Drive one item through its turns. Returns (decoded_per_turn, stats)."""
    sys_prompt = formulate_system_prompt(DEFAULT_SYSTEM_PROMPT_FORMAT, it['function'])
    convo = BOS + sys_prompt
    turns = it['question']
    holdout = it['_holdout']
    per_turn, stats = [], dict(prompt_tokens=0, completion_tokens=0, steps=0,
                               prefill_ms=0.0, decode_ms=0.0, malformed=0, forced_quit=False,
                               truncated=False)
    for tidx, msgs in enumerate(turns):
        if str(tidx) in holdout:
            # A holdout turn carries no user message; the withheld docs arrive instead.
            msgs = [dict(role='user',
                         content=DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING.format(
                             functions=holdout[str(tidx)]))]
        user_text = '\n'.join(m['content'] for m in msgs if m.get('role') == 'user')
        convo += USER_SP + user_text
        steps = []
        for step in range(MAXIMUM_STEP_LIMIT + 1):
            if step > MAXIMUM_STEP_LIMIT:
                stats['forced_quit'] = True
                break
            convo += ASSISTANT_SP
            try:
                resp = complete(host, convo, max_tokens, timeout)
            except Exception as e:
                log(f'      step {step}: request failed: {str(e)[:90]}')
                stats['forced_quit'] = True
                break
            ch = (resp.get('choices') or [{}])[0]
            text = ch.get('text') or ''
            u, tm = resp.get('usage') or {}, resp.get('timings') or {}
            stats['prompt_tokens'] += u.get('prompt_tokens', 0)
            stats['completion_tokens'] += u.get('completion_tokens', 0)
            stats['prefill_ms'] += tm.get('prefill_ms', 0.0)
            stats['decode_ms'] += tm.get('decode_ms', 0.0)
            stats['steps'] += 1
            if ch.get('finish_reason') == 'length':
                stats['truncated'] = True
            if text.endswith(EOS):
                text = text[:-len(EOS)]
            _, body = strip_think(text)
            convo += text + EOS

            try:
                decoded = default_decode_execute_prompting(body, has_tool_call_tag=False)
            except Exception:
                break                                   # undecodable ⇒ turn is over, per BFCL
            if is_empty_execute_response(decoded):
                break                                   # no calls ⇒ turn is over, per BFCL

            safe = [c for c in decoded if call_is_safe(c)]
            stats['malformed'] += len(decoded) - len(safe)
            steps.append(decoded)                        # score what the model SAID...
            if not safe:
                break
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                results, _ = execute_multi_turn_func_call(   # ...execute only what is safe
                    safe, it['initial_config'], it['involved_classes'],
                    MODEL_NAME, it['id'],
                    long_context=('long_context' in it['id'] or 'composite' in it['id']),
                    is_evaL_run=False)
            convo += USER_SP + '<tool_result>' + json.dumps(results)[:8000] + '</tool_result>'
        per_turn.append(steps)
    return per_turn, stats


def score(per_turn, gt, it, cat):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        r = multi_turn_checker(per_turn, gt, it, f'multi_turn_{cat}', MODEL_NAME)
    return bool(r.get('valid')), r


def self_gate(cat, n=0):
    """Replay ground truth as the model's output. Anything short of 100% means WE are broken."""
    items, gts = load_category(cat)
    if n:
        items = items[:n]
    ok = bad = err = 0
    for it in items:
        g = gts.get(it['id'])
        if g is None:
            continue
        clear_state(it['id'])
        try:
            good, r = score([[t] for t in g], g, it, cat)
            ok, bad = (ok + 1, bad) if good else (ok, bad + 1)
            if not good:
                print(f'  MISMATCH {it["id"]}: {str(r.get("error"))[:110]}')
        except Exception as e:
            err += 1
            print(f'  EXCEPTION {it["id"]}: {str(e)[:110]}')
    print(f'self-gate multi_turn_{cat}: ok={ok} mismatch={bad} exception={err}')
    return bad == 0 and err == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--category', default='base', choices=('base', 'miss_func', 'miss_param',
                                                           'long_context'))
    ap.add_argument('--n', type=int, default=0, help='0 = all')
    ap.add_argument('--host', default='localhost:8080')
    ap.add_argument('--effort', default='low')
    ap.add_argument('--max-tokens', type=int, default=2000)
    ap.add_argument('--timeout', type=float, default=1800)
    ap.add_argument('--self-gate', action='store_true')
    a = ap.parse_args()

    if a.self_gate:
        return 0 if self_gate(a.category, a.n) else 1

    if not self_gate(a.category, 12):
        sys.exit('self-gate failed on a sample — refusing to score a model with a broken checker')

    items, gts = load_category(a.category)
    if a.n:
        items = items[:a.n]
    task = f'bfcl_mt_{a.category}'
    path = os.path.join(OUT, f'{task}.{a.effort}.jsonl')
    done = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                done.add(json.loads(line)['id'])
            except Exception:
                pass
    todo = [it for it in items if it['id'] not in done]
    print(f'{task}: {len(todo)} to run ({len(done)} already on disk)')

    nok = len(done) and sum(1 for line in open(path) if json.loads(line).get('correct'))
    t0 = time.time()
    nrun = ntok = ndead = 0
    with open(path, 'a') as fo:
        for i, it in enumerate(todo):
            clear_state(it['id'])
            g = gts.get(it['id'])
            if g is None:
                continue
            per_turn, st = run_item(it, a.host, a.max_tokens, a.timeout,
                                    lambda m: print(m, flush=True))
            nrun += 1
            ntok += st['completion_tokens']
            # AN ITEM THAT GENERATED NOTHING DID NOT MEASURE THE MODEL. Zero steps and zero
            # completion tokens means every request for this item failed at the transport --
            # the engine was unreachable, not the model wrong. Writing it would bank a wrong
            # answer AND make the resume skip it, because resume keys off ids already on disk.
            if st['steps'] == 0 and st['completion_tokens'] == 0:
                ndead += 1
                print(f'  [{i+1}/{len(todo)}] {it["id"]:28s} DEAD  no tokens generated — '
                      f'not scored, not written, will retry on resume', flush=True)
                continue
            try:
                good, r = score(per_turn, g, it, a.category)
            except Exception as e:
                good, r = False, {'error': f'checker exception: {e}'}
            nok += good
            rec = dict(
                id=it['id'], gold='state-match',
                got=('valid' if good else str(r.get('error'))[:200]),
                prompt_sha256=hashlib.sha256(
                    json.dumps(it['question'], sort_keys=True).encode()).hexdigest(),
                correct=good, finish_reason='length' if st['truncated'] else 'stop',
                truncated=st['truncated'], category=a.category,
                subject=','.join(it['involved_classes']), level=None,
                usage=dict(prompt_tokens=st['prompt_tokens'],
                           completion_tokens=st['completion_tokens'],
                           total_tokens=st['prompt_tokens'] + st['completion_tokens'],
                           prompt_tokens_details=dict(cached_tokens=0)),
                timings=dict(prefill_ms=st['prefill_ms'], decode_ms=st['decode_ms'],
                             tokens_per_second=(st['completion_tokens'] /
                                                (st['decode_ms'] / 1000)) if st['decode_ms'] else 0,
                             tokens_per_verify=0.0),
                reasoning_chars=0, content=json.dumps(per_turn)[:20000], reasoning='',
                multi_turn=dict(turns=len(it['question']), steps=st['steps'],
                                malformed_calls=st['malformed'], forced_quit=st['forced_quit']))
            fo.write(json.dumps(rec, ensure_ascii=False) + '\n')
            fo.flush()
            el = (time.time() - t0) / 60
            print(f'  [{i+1}/{len(todo)}] {it["id"]:28s} {"OK " if good else "   "} '
                  f'turns={len(it["question"])} steps={st["steps"]:2d} '
                  f'tok={st["completion_tokens"]:5d} run={nok} ({el:.1f} min)', flush=True)
    print(f'{task}: {nok}/{len(done) + len(todo)} correct '
          f'({ndead} dead of {nrun} attempted)')

    # A DEAD ENGINE MUST NOT LOOK LIKE A SCORE OF ZERO. On 2026-08-17 an SSH drop killed the
    # server; every request here failed with ECONNREFUSED, every item "scored" 0 with 0 tokens,
    # and this returned 0. So the smoke gate in eval_bfcl_mt_run.sh passed, the 400-item "full"
    # run finished in two minutes, and two rows of pure transport failure were committed and
    # pushed as published results. The smoke gate can only protect the run if a broken run is
    # actually a non-zero exit.
    if nrun and not ntok:
        print(f'{task}: ABORT — {nrun} items attempted and the engine produced ZERO tokens. '
              f'That is a transport failure, not a score.', file=sys.stderr)
        return 2
    if nrun and ndead / nrun > 0.05:
        print(f'{task}: ABORT — {ndead}/{nrun} items ({ndead/nrun:.0%}) generated no tokens at '
              f'all. Refusing to report a run this damaged as a score.', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
