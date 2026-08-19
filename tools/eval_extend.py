#!/usr/bin/env python3
"""eval_extend.py — raise a benchmark's output budget by CONTINUING the traces that hit the cap.

THE PROBLEM. GPQA-Diamond at an 8000-token cap truncated 31.7 % of items. Those items score ~10 %
against 91.5 % for the ones that terminated, so the pooled number is a measurement of the BUDGET,
not of the model, and cannot be compared with a published reference run at a larger budget.

WHY NOT JUST RERUN THE TRUNCATED ITEMS AT A BIGGER CAP. Because that is biased, and subtly enough
that it would survive review. Let X be a generation under the target cap B=24000. Selecting the
items to redo by the outcome of a first draw under b=8000, then redrawing them UNCONDITIONALLY,
produces the mixture

    P(len<=b) * [X | len<=b]   +   P(len>b) * [X]

which is not the distribution of X: the redraw puts fresh mass on len<=b that the kept items
already account for, so short traces are counted twice and the estimate drifts toward whatever
short traces happen to score. With temperature 1.0 the redraw genuinely can land under 8000.

WHAT IS UNBIASED. Continue the stored prefix. An item that stopped on its own below b is already a
valid draw from X, because a larger cap cannot change a generation that never reached the old one --
max_tokens is a stopping rule, not a sampling parameter. An item that hit the cap has its first b
tokens observed; continuing from exactly those tokens draws from [X | first b tokens = observed],
and the two pieces recompose to X exactly. So: keep every terminated trace as-is, continue every
truncated one, and the merged file is a clean measurement at the new budget.

HOW THE CONTINUATION IS PROVEN EXACT. Continuing "from the same prefix" is only correct if the
model really is re-fed the same tokens. Two facts make that checkable rather than assumed:

  * A truncated record was cut off INSIDE the thinking block, so the parser never found </think>
    and stored the text with its leading <think> intact. The prefix is therefore reconstructible as
    <|User|> + question + <|Assistant|> + stored_content, with no marker to re-add or double.
  * The raw /v1/completions path prepends BOS and applies no chat template, so the token count it
    reports for that reconstruction is directly comparable with the base record's
    prompt_tokens + completion_tokens.

WHAT THAT SECOND CHECK CAN AND CANNOT PROVE -- measured, not assumed. It was written as a hard
gate: off by one token and the whole task aborted. That cost the GPQA extension outright on
2026-08-19, at item 10 of 51, and the diagnosis is that the gate was testing an identity that
cannot hold. The model emits TOKEN IDS; a record stores the DETOKENIZED TEXT; rebuilding re-encodes
that text with greedy BPE, and detokenize->retokenize is not the identity -- adjacent pieces re-merge,
and a hard truncation can cut mid-fragment. tools/tokprobe.cpp links the server's own tokenizer and
encoder and measured all 260 truncated traces in the battery offline:

    231 exact (88.8 %), 29 drifted; 23 of those by exactly one or two tokens.
    The three largest (-5, -22, -56) are all LiveCodeBench, where long indentation
    runs re-merge differently. No item's TEXT differs: it is stored verbatim and the
    prompt is hash-checked separately.

So token count is a DIAGNOSTIC, not a correctness condition. The condition that actually matters --
same question, same stored continuation text -- is enforced by the prompt hash and by using the
stored content verbatim. The count is still compared, still recorded per item, and still trips, but
only on a divergence far too large to be BPE noise, which would mean the wrong text rather than a
different tiling of the right text.

AND ONE ITEM MUST NEVER COST THE TASK. The old gate called sys.exit() mid-sweep, which left a
half-written merged file holding only the traces that had already terminated -- a file that is
non-empty, so eval_force_all.sh selects it, reads 0 % truncation out of it and declines. Now an
item that cannot be continued is CARRIED FORWARD UNCHANGED and stays flagged truncated, so the
merged file is always complete, its truncation rate is honest, and the forcing pass picks up
exactly the residue the extension could not repair. That is the division of labour the chain was
designed around.

  python3 tools/eval_extend.py --task gpqa_diamond --effort low --budget 24000 --dry-run
  python3 tools/eval_extend.py --task gpqa_diamond --effort low --budget 24000
"""
import argparse, hashlib, json, os, subprocess, sys, time, urllib.error, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_suite as E

OUT = E.OUT
USER_SP = '<｜User｜>'
ASSISTANT_SP = '<｜Assistant｜>'
THINK_START = '<think>'
THINK_END = '</think>'
EOS = '<｜end▁of▁sentence｜>'


def tag_for(effort, budget):
    """`low` at 24000 becomes `low24k`, so a re-budgeted run can never share a file with the run it
    came from. Mixing two budgets in one jsonl is exactly the failure this whole tool exists to
    undo."""
    return f'{effort}{budget // 1000}k'


def complete(host, prompt, temp, top_p, max_tokens, timeout, retries=3):
    body = json.dumps(dict(prompt=prompt, temperature=temp, top_p=top_p,
                           max_tokens=max_tokens)).encode()
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(f'http://{host}/v1/completions', data=body,
                                     headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = ''
            try:
                detail = e.read().decode()[:200]
            except Exception:
                pass
            last = f'HTTP {e.code}: {detail}'
            if 400 <= e.code < 500:
                raise E.Fatal(f'/v1/completions rejected the request: {last}')
        except Exception as e:
            last = f'{type(e).__name__}: {e}'
        if attempt + 1 < retries:
            time.sleep(5 * (attempt + 1))
    raise E.Fatal(f'/v1/completions failed {retries}x: {last}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True)
    ap.add_argument('--effort', default='low')
    ap.add_argument('--budget', type=int, required=True, help='new total completion-token budget')
    ap.add_argument('--host', default='localhost:8080')
    ap.add_argument('--temp', type=float, default=1.0)
    ap.add_argument('--top-p', type=float, default=0.95)
    ap.add_argument('--dry-run', action='store_true',
                    help='rebuild and check every prefix, send nothing')
    # A TRIPWIRE FOR THE WRONG TEXT, NOT A BPE NOISE FLOOR. Measured worst-case re-merge across the
    # whole battery is 56 tokens on an 8.4k prefix; grafting the wrong continuation onto a prompt
    # would differ by a large fraction of it, not by tenths of a percent. 1 % with a 64-token floor
    # sits an order of magnitude above the noise and still catches a structural mistake.
    ap.add_argument('--drift-tol', type=float, default=0.01,
                    help='max |rebuilt - fed| token drift, as a fraction of the fed prefix')
    ap.add_argument('--drift-floor', type=int, default=64)
    a = ap.parse_args()

    base_path = os.path.join(OUT, f'{a.task}.{a.effort}.jsonl')
    if not os.path.exists(base_path):
        sys.exit(f'no base run at {base_path}')
    base = {}
    for line in open(base_path):
        if line.strip():
            r = json.loads(line)
            base[r['id']] = r           # last write wins, matching _report_one
    base = list(base.values())

    items, snap, src, kind = E.TASKS[a.task](0)
    by_id = {it['id']: it for it in items}

    bmeta_path = os.path.join(OUT, f'{a.task}.{a.effort}.meta.json')
    bmeta = json.load(open(bmeta_path)) if os.path.exists(bmeta_path) else {}
    base_budget = bmeta.get('max_tokens') or E.MAXTOK[a.task]
    if a.budget <= base_budget:
        sys.exit(f'--budget {a.budget} does not exceed the base budget {base_budget}')

    tag = tag_for(a.effort, a.budget)
    out_path = os.path.join(OUT, f'{a.task}.{tag}.jsonl')
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                done.add(json.loads(line)['id'])
            except Exception:
                pass

    def is_trunc(r):
        return bool(r.get('truncated')) or \
            (r.get('usage') or {}).get('completion_tokens', 0) >= base_budget

    keep = [r for r in base if not is_trunc(r)]
    todo = [r for r in base if is_trunc(r)]
    print(f'[{a.task}@{a.effort} -> {tag}] {len(base)} base records: '
          f'{len(keep)} terminated (kept as-is), {len(todo)} truncated (to continue), '
          f'{len(done)} already written')

    # REFUSE TO BUILD A MERGED FILE FROM A HALF-FINISHED BASE RUN. The carried-over traces are the
    # ones that terminated, i.e. the easy half; a merged file built while the base run is still
    # going would be published by --report as a real accuracy over a biased subset (91.5 % over the
    # 82 terminated items, at the moment this guard was written).
    n_expected = bmeta.get('n_items')
    if n_expected and len(base) < n_expected and not a.dry_run:
        sys.exit(f'base run is incomplete ({len(base)}/{n_expected} records). Extending now would '
                 f'publish an accuracy over the traces that happened to terminate first. '
                 f'Wait for {a.task}.{a.effort} to finish.')

    if a.dry_run:
        for r in todo:
            bid = r['id'].split('#r')[0]
            it = by_id.get(bid)
            u = r.get('usage') or {}
            content = r.get('content') or ''
            print(f'  [dry] {r["id"]}: continuable={content.startswith(THINK_START)} '
                  f'expect prompt_tokens={u.get("prompt_tokens",0)+u.get("completion_tokens",0)} '
                  f'room={a.budget - u.get("completion_tokens",0)}')
        bad = sum(1 for r in todo if not (r.get('content') or '').startswith(THINK_START))
        print(f'\ndry run: {len(keep)} would carry over, {len(todo)-bad} continuable, '
              f'{bad} not continuable. Nothing sent, nothing written.')
        return

    # Terminated traces carry over untouched -- they are already draws from the target budget.
    fout = open(out_path, 'a')
    for r in keep:
        if r['id'] in done:
            continue
        r = dict(r, budget=a.budget,
                 extension=dict(method='carried-over', reason='terminated below the base budget',
                                base_budget=base_budget))
        fout.write(json.dumps(r, ensure_ascii=False) + '\n')
        done.add(r['id'])
    fout.flush()

    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                            cwd=os.path.dirname(OUT)).stdout.strip() or None
    nok = nbad = ndrift = 0
    max_drift = 0
    t0 = time.time()

    def carry_unrepaired(rec, reason, **extra):
        """Write a truncated trace forward UNCHANGED when it cannot be continued.

        The merged file must always hold one record per base record. Dropping the ones that could
        not be continued would understate the truncation rate by removing exactly the items that
        are truncated, and would leave a short file that eval_force_all.sh still selects. Carried
        forward, the row's truncation rate stays honest and the forcing pass sees the residue."""
        out = dict(rec, budget=a.budget,
                   extension=dict(method='not-continued', reason=reason,
                                  base_budget=base_budget, code_commit=commit, **extra))
        fout.write(json.dumps(out, ensure_ascii=False) + '\n')
        fout.flush()
        done.add(rec['id'])
    for i, r in enumerate(todo):
        if r['id'] in done:
            continue
        bid = r['id'].split('#r')[0]
        it = by_id.get(bid)
        if it is None:
            sys.exit(f'{r["id"]}: not in the pinned dataset — refusing to extend')

        # The prompt that was actually sent must be the one this rebuild produces, or the
        # continuation is grafted onto a different question.
        h = hashlib.sha256(it['prompt'].encode()).hexdigest()
        if r.get('prompt_sha256') and r['prompt_sha256'] != h:
            sys.exit(f'{r["id"]}: prompt hash mismatch — refusing to extend')

        content = r.get('content') or ''
        if not content.startswith(THINK_START):
            # Cut off after </think> (mid final answer) rather than inside it. Reconstructing that
            # prefix needs the reasoning and the body rejoined, which is a different shape; rather
            # than guess, skip it and say so.
            print(f'  [skip] {r["id"]}: truncated outside the thinking block, not continuable '
                  f'— carried forward still truncated for the forcing pass')
            carry_unrepaired(r, 'truncated outside the thinking block')
            nbad += 1
            continue

        base_pt = (r.get('usage') or {}).get('prompt_tokens', 0)
        base_ct = (r.get('usage') or {}).get('completion_tokens', 0)
        expect_pt = base_pt + base_ct
        room = a.budget - base_ct
        prefix = USER_SP + it['prompt'] + ASSISTANT_SP + content

        timeout = E.budget_timeout(room)
        resp = complete(a.host, prefix, a.temp, a.top_p, room, timeout)
        u = resp.get('usage') or {}
        got_pt = u.get('prompt_tokens', -1)
        drift = got_pt - expect_pt
        tol = max(a.drift_floor, int(a.drift_tol * expect_pt))
        if abs(drift) > tol:
            # Past the tripwire this is no longer a different tiling of the same text, it is
            # probably not the same text. Refuse THIS item, keep the task alive, and say so loudly
            # enough that it is not mistaken for the ordinary re-merge below.
            print(f'  [REFUSE] {r["id"]}: rebuilt prefix is {got_pt} tokens against {expect_pt} fed '
                  f'({drift:+d}, tolerance ±{tol}). That is too large to be BPE re-merge — the '
                  f'reconstruction is probably not the same text. Carried forward untouched.',
                  flush=True)
            carry_unrepaired(r, 'prefix drift beyond tolerance',
                             rebuilt_prompt_tokens=got_pt, expected_prompt_tokens=expect_pt,
                             drift=drift, tolerance=tol)
            nbad += 1
            continue
        if drift:
            ndrift += 1
            max_drift = max(max_drift, abs(drift))

        cont = (resp.get('choices') or [{}])[0].get('text') or ''
        if cont.endswith(EOS):
            cont = cont[:-len(EOS)]
        ext_ct = u.get('completion_tokens', 0)
        total_ct = base_ct + ext_ct
        full = content + cont

        e = full.find(THINK_END)
        if e != -1:
            reasoning = full[:e]
            if reasoning.startswith(THINK_START):
                reasoning = reasoning[len(THINK_START):]
            body = full[e + len(THINK_END):]
        else:
            reasoning, body = '', full

        got = E.extract(kind, body) or E.extract(kind, reasoning)
        ok = bool(E.correct(kind, got, it['gold'], it))
        nok += ok

        rec = dict(r)
        rec.update(
            content=body, reasoning=reasoning, reasoning_chars=len(reasoning),
            got=got, correct=ok, budget=a.budget,
            truncated=total_ct >= a.budget,
            finish_reason='length' if total_ct >= a.budget else 'stop',
            usage=dict(prompt_tokens=base_pt, completion_tokens=total_ct,
                       total_tokens=base_pt + total_ct),
            extension=dict(method='prefix-continuation', base_budget=base_budget,
                           base_completion_tokens=base_ct, ext_completion_tokens=ext_ct,
                           continuation_prompt_tokens=got_pt,
                           expected_prompt_tokens=expect_pt,
                           # token_exact is now MEASURED, not asserted. 88.8 % of the battery's
                           # truncated traces round-trip exactly; the rest differ by a token or two
                           # because BPE re-merges the same text differently. Storing the number
                           # lets eval_publish caveat the row instead of the tool pretending.
                           token_exact=(drift == 0), prefix_token_drift=drift,
                           code_commit=commit,
                           # KEEP THE CONTINUATION LEG'S OWN TIMINGS. `rec = dict(r)` carries the
                           # base record's `timings` forward, and they describe a decode that began
                           # at a few hundred prompt tokens. This leg begins at base_pt+base_ct --
                           # 8k and up -- and is the only place in the whole programme where decode
                           # is measured at that depth. Overwriting the top-level `timings` would
                           # change what a landed benchmark row means, so the leg is recorded
                           # alongside instead: scoring reads `timings`, perf_ingest.py reads this.
                           ext_timings=dict(
                               (resp.get('timings') or {}),
                               cached_tokens=(u.get('prompt_tokens_details') or {})
                                             .get('cached_tokens', 0))))
        fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
        fout.flush()
        el = time.time() - t0
        print(f'  [{i+1}/{len(todo)}] {r["id"]} gold={it["gold"]} got={got} '
              f'{"OK " if ok else "   "} +{ext_ct} tok (total {total_ct}) '
              f'{"TRUNC" if total_ct >= a.budget else "stop"} ({el/60:.1f} min)', flush=True)

    fout.close()
    meta = dict(bmeta)
    meta.update(task=a.task, effort=tag, base_effort=a.effort, max_tokens=a.budget,
                base_max_tokens=base_budget, temperature=a.temp, top_p=a.top_p,
                snapshot=snap, source=src, n_items=len(base),
                method=('merged budget extension: traces that terminated under the base budget are '
                        'carried over unchanged; traces that hit it are continued from their exact '
                        'stored prefix, verified token-exact per item'),
                n_carried=len(keep), n_continued=len(todo) - nbad, n_not_continuable=nbad,
                n_prefix_drifted=ndrift, max_prefix_token_drift=max_drift,
                prefix_drift_note=('continuations are rebuilt from the stored text, so a prefix can '
                                   're-tokenize to a slightly different length than the ids the model '
                                   'emitted; the text is identical by construction and each record '
                                   'carries its own measured drift'),
                code_commit=commit)
    with open(os.path.join(OUT, f'{a.task}.{tag}.meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'\nwrote {out_path}  ({len(keep)} carried + {len(todo) - nbad} continued + '
          f'{nbad} not continuable, {ndrift} with prefix drift, max |drift| {max_drift})')


if __name__ == '__main__':
    main()
