#!/usr/bin/env python3
"""eval_force.py — budget forcing: close the thinking block at the cap and make the model answer.

WHAT PROBLEM THIS SOLVES, AND WHAT IT DOES NOT. A truncated item is scored WRONG -- not "answered
badly", but *absent*: the model never left the thinking block, so there is nothing to extract. The
4k analysis measured the cost directly (evidence/evals/archive_budget4k/truncation_analysis.txt):
truncated items scored 25.0 % on GPQA against 95.2 % for terminated ones, i.e. AT CHANCE. So the
published number is a weighted average of the model's accuracy and a coin flip, with the weight set
by max_tokens.

`eval_extend.py` fixes this the honest way -- give the trace more room and let it finish. That works
where the length distribution has a reachable tail. Fitted log-normal survival on the censored 8k
data says it works for mmlu_pro (12.0 % -> ~2.4 %) and scicode (18.6 % -> ~3.5 %), and does NOT work
for gpqa_diamond (25.9 % -> ~9.1 %) or lcb (55.9 % -> ~37.8 %). Those two have sigma 1.67 and 2.94;
their tails do not end anywhere we can afford, and the server's --seqmax 32768 is a hard ceiling on
trying (raising it costs fp32 KV memory we do not have).

Budget forcing is the other lever. At the cap, append the end-of-thinking delimiter to the stored
prefix and let the model answer from the reasoning it has already done. Truncation goes to zero BY
CONSTRUCTION, because the answer span is generated under its own budget after the thinking block is
closed. Precedent: s1 (Muennighoff et al. 2025) uses exactly this to control test-time compute.

  ** THIS IS A PROTOCOL DEVIATION AND MUST BE PUBLISHED AS ONE. **

A forced row and an unforced row are not the same measurement. Forcing asks "what would the model
answer if made to stop here", which is a real and interesting question, but it is not the question
the reference harnesses ask. The output tag is therefore `<effort><cap>kforced` -- it can never
share a file, a row, or a column with an unforced run, and tools/eval_publish.py carries an explicit
DEVIATIONS entry for it.

WHY IT IS ALSO CHEAP, WHICH IS NOT THE POINT BUT MATTERS. Extending GPQA's 51 truncated items to 24k
costs up to 51 x 16000 = 816k tokens. Forcing them costs 51 x ~512. On this box that is the
difference between ~20 hours and ~20 minutes.

WHY IT IS A PAIRED EXPERIMENT FOR FREE. Forcing and extension continue THE SAME stored 8k prefix.
Every forced item therefore has a natural partner in the extended run -- same question, same
reasoning, two different endings. `--pair-with` scores that partnership with an exact McNemar test,
which is what turns "we deviated" into "we deviated and here is what the deviation cost".

TOKEN EXACTNESS. Same hard gate as eval_extend.py, plus one more unknown. The forced prefix is
`base_prompt + base_completion + SUFFIX`, so the identity to check is
`prompt_tokens == base_pt + base_ct + T` where T is the suffix's token length. T is MEASURED at
startup by a two-request calibration against a real item, never assumed -- byte-level BPE can merge
at a boundary, and a suffix that silently costs 2 tokens instead of 1 would turn the per-item gate
into a rubber stamp. The calibration's first leg also re-proves eval_extend's base identity.

ORDER OF OPERATIONS: EXTEND FIRST, THEN FORCE THE RESIDUE. Forcing is applied to the run that has
already had every cheap repair, which after the battery is the EXTENDED file (`low24k`), not the
base (`low`). Two reasons, and the first is the one that matters:

  * Forcing `low` throws away the extension's work. An item that would have finished on its own at
    19k gets its thinking amputated at 8k instead, and the row is needlessly worse.
  * A row forced at 24k of thinking is a stronger measurement than one forced at 8k, so the
    deviation buys more.

`--only-if-over 0.05` makes this safe to run across every task without deciding in advance which
ones still need it: rows the extension already rescued exit having sent nothing.

  # after the battery
  bash scripts/eval_extend_all.sh                                   # 8000 -> 24000, unbiased
  python3 tools/eval_force.py --task gpqa_diamond --effort low24k --only-if-over 0.05 --dry-run
  python3 tools/eval_force.py --task gpqa_diamond --effort low24k --only-if-over 0.05
  python3 tools/eval_force.py --task gpqa_diamond --effort low24k --pair-with low24k --report-only
"""
import argparse, hashlib, json, math, os, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_suite as E
import eval_extend as X

OUT = E.OUT
USER_SP, ASSISTANT_SP = X.USER_SP, X.ASSISTANT_SP
THINK_START, THINK_END, EOS = X.THINK_START, X.THINK_END, X.EOS

# How much room the ANSWER span gets, once thinking is closed. This is not a thinking budget -- it
# is the length of a final answer, so it is sized by output format, not by problem difficulty.
# A letter answer needs a sentence; a code answer needs a full program. If the answer span itself
# hits this cap the item is still truncated and is reported as such rather than counted as fixed.
ANSWER_TOKENS = dict(letter=512, integer=512, math=1024, call=512,
                     code=2048, lcb=2048, scicode=2048)
ANSWER_DEFAULT = 1024


# TIMEOUTS MUST BE SIZED OFF PREFIX DEPTH HERE, WHICH IS WHY E.budget_timeout IS NOT USED.
# That helper takes a flat 180 s prefill allowance and scales the rest off max_tokens, which is
# right for eval_suite (few-hundred-token prompt, big generation) and right for eval_extend (8k
# prompt, 16000 tokens of room -> a 4180 s timeout that dwarfs everything). Forcing is the exact
# inverse: an 8k-24k prefix and a ~512-token answer. `budget_timeout(512)` is 308 s, and prefilling
# 24k tokens alone takes ~490 s -- the request would time out, retry 3x, and die Fatal, having
# burned an hour of engine time proving nothing.
#
# The floors are measured, from evidence/perf/perf.sqlite over 1688 requests: prefill settles at
# 44-57 tok/s for prompts of 1000-4000 tokens (the sub-1000 band's 96 tok/s mean is inflated by
# prefix-cache hits, and the 55469 max is entirely cache). 40 tok/s is below every uncached
# observation. NOTE the corpus contains ZERO requests above 4000 prompt tokens -- nothing in the
# battery has ever prefilled this deep -- so the deep end is an extrapolation of a rate that is
# flat across the range we can see, and the floor is set pessimistically because of it.
PREFILL_FLOOR_TOK_S = 40.0
DECODE_FLOOR_TOK_S = 4.0


def force_timeout(prompt_tokens, max_tokens, fixed_s=60):
    """Client timeout covering BOTH legs of a deep-prefix, short-answer request."""
    return int(fixed_s + prompt_tokens / PREFILL_FLOOR_TOK_S + max_tokens / DECODE_FLOOR_TOK_S)


def tag_for(effort, cap):
    """`low` forced at an 8000-token thinking cap becomes `low8kforced`.

    Distinct from eval_extend's `low24k` by construction: a reader scanning EVALS.md cannot confuse
    a forced row with a budget-extended one, and efforts_on_disk() discovers it without a code
    change (it splits the filename on '.', so the tag must simply contain no dot).
    """
    return f'{effort}{cap // 1000}kforced'


def calibrate_suffix_tokens(host, prefix_no_suffix, suffix, expect_pt, temp, top_p):
    """Measure how many tokens SUFFIX costs when appended to a real prefix. Do not assume 1.

    `</think>` is a special token and should be atomic, and the text before it should tokenize
    identically because eval_extend already proved `base_prompt + base_completion` reproduces
    base_pt + base_ct exactly. Should is not measured. Byte-level BPE merges across boundaries, and
    if the suffix cost 2 tokens rather than 1 every per-item assertion below would pass for the
    wrong reason. Two requests, two tokens of generation, and the unknown becomes a constant.

    The first leg is also a free re-proof of eval_extend's identity on this task: if the prefix
    WITHOUT the suffix does not tokenize to base_pt + base_ct, the stored record and the live
    tokenizer disagree and nothing downstream is trustworthy.
    """
    tmo = force_timeout(expect_pt, 1)
    a = X.complete(host, prefix_no_suffix, temp, top_p, 1, tmo)
    got_a = (a.get('usage') or {}).get('prompt_tokens', -1)
    if got_a != expect_pt:
        raise E.Fatal(
            f'calibration leg 1: prefix WITHOUT suffix tokenized to {got_a}, expected {expect_pt} '
            f'(base prompt_tokens + completion_tokens). The stored trace and the live tokenizer '
            f'disagree — refusing to force anything.')
    b = X.complete(host, prefix_no_suffix + suffix, temp, top_p, 1, tmo)
    got_b = (b.get('usage') or {}).get('prompt_tokens', -1)
    t = got_b - got_a
    if t < 1:
        raise E.Fatal(f'calibration leg 2: suffix {suffix!r} measured {t} tokens, which is '
                      f'impossible. Got prompt_tokens {got_b} vs {got_a}.')
    return t


def mcnemar_exact(b, c):
    """Two-sided exact McNemar on discordant pairs. b = forced right & free wrong, c = the reverse.

    The concordant pairs carry no information about whether forcing changes the answer, which is
    exactly why an unpaired chi-square over the two accuracies would be the wrong test here: it
    would treat two views of the SAME 51 traces as two independent samples.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def report_paired(task, forced_tag, other_tag):
    """Score forcing against the free continuation of the same prefixes."""
    def load(tag):
        p = os.path.join(OUT, f'{task}.{tag}.jsonl')
        if not os.path.exists(p):
            sys.exit(f'no run at {p}')
        d = {}
        for l in open(p):
            if l.strip():
                r = json.loads(l)
                d[r['id']] = r
        return d

    F, O = load(forced_tag), load(other_tag)
    # Only items that were actually FORCED are informative. The carried-over terminated traces are
    # byte-identical in both files and would dilute every count toward agreement.
    ids = [i for i in F if i in O and (F[i].get('forcing') or {}).get('method') == 'budget-forced']
    if not ids:
        sys.exit(f'no forced items in common between {forced_tag} and {other_tag}')
    b = sum(1 for i in ids if F[i]['correct'] and not O[i]['correct'])
    c = sum(1 for i in ids if not F[i]['correct'] and O[i]['correct'])
    both = sum(1 for i in ids if F[i]['correct'] and O[i]['correct'])
    neither = len(ids) - b - c - both
    p = mcnemar_exact(b, c)
    ftok = [(F[i].get('usage') or {}).get('completion_tokens', 0) for i in ids]
    otok = [(O[i].get('usage') or {}).get('completion_tokens', 0) for i in ids]
    print(f'\nPAIRED: {task}  {forced_tag} vs {other_tag}   n={len(ids)} forced items')
    print(f'  both correct            {both}')
    print(f'  forced only  (b)        {b}')
    print(f'  {other_tag} only  (c)   {c}')
    print(f'  neither                 {neither}')
    print(f'  forced acc              {100*(both+b)/len(ids):.1f} %')
    print(f'  {other_tag} acc         {100*(both+c)/len(ids):.1f} %')
    print(f'  exact McNemar p         {p:.4f}'
          f'{"  (no detectable cost to forcing)" if p > 0.05 else "  (forcing CHANGES the score)"}')
    print(f'  mean tokens             forced {sum(ftok)/len(ftok):.0f} vs '
          f'{other_tag} {sum(otok)/len(otok):.0f}')
    return dict(n=len(ids), b=b, c=c, both=both, neither=neither, p=p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True)
    ap.add_argument('--effort', default='low',
                    help='tag of the run to force, e.g. `low` or `low24k`')
    ap.add_argument('--host', default='localhost:8080')
    ap.add_argument('--temp', type=float, default=1.0)
    ap.add_argument('--top-p', type=float, default=0.95)
    ap.add_argument('--suffix', default=THINK_END,
                    help='what to append at the cap. Default is the bare end-of-thinking '
                         'delimiter — the MINIMAL intervention. Anything more (e.g. '
                         '"</think>\\n\\nAnswer:") is a larger deviation and is recorded verbatim '
                         'in the meta so a reader can see exactly what the model was handed.')
    ap.add_argument('--answer-tokens', type=int, default=0,
                    help='budget for the answer span (0 = per-kind default)')
    ap.add_argument('--only-if-over', type=float, default=None, metavar='RATE',
                    help='do nothing unless the source row is more than RATE truncated (e.g. 0.05, '
                         "eval_publish's quotability gate). Forcing is a protocol deviation, so a "
                         'row that the budget extension already rescued must not pay for one. Use '
                         'this rather than remembering which rows needed it.')
    ap.add_argument('--dry-run', action='store_true',
                    help='rebuild and check every prefix, send nothing')
    ap.add_argument('--pair-with', default=None,
                    help='score against another tag of the same task, e.g. low24k')
    ap.add_argument('--report-only', action='store_true',
                    help='with --pair-with, only report; generate nothing')
    a = ap.parse_args()

    items, snap, src, kind = E.TASKS[a.task](0)
    by_id = {it['id']: it for it in items}
    ans_tok = a.answer_tokens or ANSWER_TOKENS.get(kind, ANSWER_DEFAULT)

    base_path = os.path.join(OUT, f'{a.task}.{a.effort}.jsonl')
    if not os.path.exists(base_path):
        sys.exit(f'no base run at {base_path}')
    base = {}
    for line in open(base_path):
        if line.strip():
            r = json.loads(line)
            base[r['id']] = r           # last write wins, matching _report_one
    base = list(base.values())

    bmeta_path = os.path.join(OUT, f'{a.task}.{a.effort}.meta.json')
    bmeta = json.load(open(bmeta_path)) if os.path.exists(bmeta_path) else {}
    cap = bmeta.get('max_tokens') or E.MAXTOK[a.task]
    tag = tag_for(bmeta.get('base_effort') or a.effort, cap)

    if a.pair_with and a.report_only:
        report_paired(a.task, tag, a.pair_with)
        return

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
            (r.get('usage') or {}).get('completion_tokens', 0) >= cap

    keep = [r for r in base if not is_trunc(r)]
    todo = [r for r in base if is_trunc(r)]
    print(f'[{a.task}@{a.effort} -> {tag}] {len(base)} base records: '
          f'{len(keep)} terminated (kept as-is), {len(todo)} at the {cap}-token cap (to force), '
          f'{len(done)} already written')
    print(f'  suffix={a.suffix!r}  answer budget={ans_tok} tokens  kind={kind}')

    # SAME GUARD AS eval_extend, AND IT MUST COME BEFORE --only-if-over: the truncation rate of a
    # half-finished run is not the row's truncation rate, so gating on it could wave a row through
    # as "does not need forcing" on the strength of whichever items happened to finish first.
    n_expected = bmeta.get('n_items')
    if n_expected and len(base) < n_expected and not a.dry_run:
        sys.exit(f'base run is incomplete ({len(base)}/{n_expected} records). Forcing now would '
                 f'publish an accuracy over the traces that happened to terminate first. '
                 f'Wait for {a.task}.{a.effort} to finish.')

    # THE GATE THAT KEEPS THE DEVIATION PROPORTIONATE. Every forced row costs a paragraph of
    # published caveat, so a row the budget extension already brought under the quotability gate
    # must not be forced merely because forcing is available and cheap.
    if a.only_if_over is not None:
        rate = len(todo) / len(base) if base else 0.0
        if rate <= a.only_if_over:
            print(f'  {rate:.1%} truncated is at or under the {a.only_if_over:.0%} gate — '
                  f'this row does not need forcing. Doing nothing.')
            return

    if a.dry_run:
        bad = 0
        for r in todo:
            bid = r['id'].split('#r')[0]
            it = by_id.get(bid)
            u = r.get('usage') or {}
            content = r.get('content') or ''
            cont = content.startswith(THINK_START)
            bad += not cont
            h = hashlib.sha256(it['prompt'].encode()).hexdigest() if it else None
            print(f'  [dry] {r["id"]}: forceable={cont} '
                  f'prompt_ok={(not r.get("prompt_sha256")) or r["prompt_sha256"] == h} '
                  f'expect prompt_tokens={u.get("prompt_tokens",0)+u.get("completion_tokens",0)}+T')
        print(f'\ndry run: {len(keep)} would carry over, {len(todo)-bad} forceable, '
              f'{bad} truncated outside the thinking block. Nothing sent, nothing written.')
        return

    if not todo:
        print('nothing at the cap — this row does not need forcing.')
        return

    # Terminated traces carry over untouched. They were never forced and must not be labelled as
    # though they were; `forcing.method` distinguishes them, and report_paired() keys off it.
    fout = open(out_path, 'a')
    for r in keep:
        if r['id'] in done:
            continue
        r = dict(r, forcing=dict(method='carried-over', reason='terminated below the cap',
                                 cap=cap))
        fout.write(json.dumps(r, ensure_ascii=False) + '\n')
        done.add(r['id'])
    fout.flush()

    # Calibrate T against the first forceable item, before committing to anything.
    cal = next((r for r in todo if (r.get('content') or '').startswith(THINK_START)), None)
    if cal is None:
        sys.exit('every truncated trace stopped outside the thinking block — nothing to force')
    cal_it = by_id[cal['id'].split('#r')[0]]
    cal_u = cal.get('usage') or {}
    T = calibrate_suffix_tokens(
        a.host, USER_SP + cal_it['prompt'] + ASSISTANT_SP + (cal.get('content') or ''),
        a.suffix, cal_u.get('prompt_tokens', 0) + cal_u.get('completion_tokens', 0),
        a.temp, a.top_p)
    print(f'  calibrated: suffix {a.suffix!r} costs {T} token(s); base identity re-proved on '
          f'{cal["id"]}')

    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                            cwd=os.path.dirname(OUT)).stdout.strip() or None
    nok = nbad = nstill = 0
    t0 = time.time()
    for i, r in enumerate(todo):
        if r['id'] in done:
            continue
        bid = r['id'].split('#r')[0]
        it = by_id.get(bid)
        if it is None:
            sys.exit(f'{r["id"]}: not in the pinned dataset — refusing to force')

        h = hashlib.sha256(it['prompt'].encode()).hexdigest()
        if r.get('prompt_sha256') and r['prompt_sha256'] != h:
            sys.exit(f'{r["id"]}: prompt hash mismatch — refusing to force')

        content = r.get('content') or ''
        if not content.startswith(THINK_START):
            # Stopped after </think>, mid final answer. Appending another </think> would produce a
            # doubled delimiter and a prefix the model never saw. Skip rather than guess.
            print(f'  [skip] {r["id"]}: truncated outside the thinking block, not forceable')
            nbad += 1
            continue

        base_pt = (r.get('usage') or {}).get('prompt_tokens', 0)
        base_ct = (r.get('usage') or {}).get('completion_tokens', 0)
        expect_pt = base_pt + base_ct + T
        prefix = USER_SP + it['prompt'] + ASSISTANT_SP + content + a.suffix

        resp = X.complete(a.host, prefix, a.temp, a.top_p, ans_tok,
                          force_timeout(expect_pt, ans_tok))
        u = resp.get('usage') or {}
        got_pt = u.get('prompt_tokens', -1)
        if got_pt != expect_pt:
            sys.exit(
                f'\n{r["id"]}: FORCED PREFIX IS NOT TOKEN-EXACT — refusing to continue.\n'
                f'  the base run fed {base_pt} prompt + {base_ct} completion tokens\n'
                f'  the suffix {a.suffix!r} calibrated to {T} token(s)\n'
                f'  so the rebuilt prefix must tokenize to {expect_pt}; it tokenized to {got_pt}\n'
                f'  forcing from a different token sequence would measure a different thing.')

        body = (resp.get('choices') or [{}])[0].get('text') or ''
        if body.endswith(EOS):
            body = body[:-len(EOS)]
        ans_ct = u.get('completion_tokens', 0)
        # The ANSWER span can itself hit its cap. That item is still truncated and is counted here
        # rather than silently presented as repaired -- the whole purpose of this tool is to make
        # the truncation column mean something, so it must not become a place where truncation
        # hides.
        still = ans_ct >= ans_tok
        nstill += still

        # The reasoning is whatever the model had produced when the cap hit, with its opening
        # <think> stripped. There is no </think> in it by construction -- we supplied that.
        reasoning = content[len(THINK_START):]
        got = E.extract(kind, body) or E.extract(kind, reasoning)
        ok = bool(E.correct(kind, got, it['gold'], it))
        nok += ok

        rec = dict(r)
        rec.update(
            content=body, reasoning=reasoning, reasoning_chars=len(reasoning),
            got=got, correct=ok,
            truncated=bool(still),
            finish_reason='length' if still else 'stop',
            usage=dict(prompt_tokens=base_pt, completion_tokens=base_ct + ans_ct,
                       total_tokens=base_pt + base_ct + ans_ct),
            forcing=dict(method='budget-forced', cap=cap, suffix=a.suffix, suffix_tokens=T,
                         thinking_tokens=base_ct, answer_tokens=ans_ct,
                         answer_budget=ans_tok, answer_truncated=bool(still),
                         forced_prompt_tokens=got_pt, expected_prompt_tokens=expect_pt,
                         token_exact=True, code_commit=commit,
                         # Same reasoning as eval_extend's ext_timings: this leg decodes at a
                         # prompt depth of base_pt+base_ct, not at the base record's few hundred,
                         # so its timings describe a different regime and must not overwrite the
                         # top-level `timings` that a landed benchmark row was measured with.
                         forced_timings=dict(
                             (resp.get('timings') or {}),
                             cached_tokens=(u.get('prompt_tokens_details') or {})
                                           .get('cached_tokens', 0))))
        fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
        fout.flush()
        el = time.time() - t0
        print(f'  [{i+1}/{len(todo)}] {r["id"]} gold={it["gold"]} got={got} '
              f'{"OK " if ok else "   "} think={base_ct} +answer={ans_ct} '
              f'{"STILL-TRUNC" if still else "stop"} ({el/60:.1f} min)', flush=True)

    fout.close()
    meta = dict(bmeta)
    meta.update(
        task=a.task, effort=tag, base_effort=a.effort, max_tokens=cap,
        thinking_cap=cap, answer_budget=ans_tok, force_suffix=a.suffix, suffix_tokens=T,
        temperature=a.temp, top_p=a.top_p, snapshot=snap, source=src, n_items=len(base),
        protocol_deviation=(
            'BUDGET FORCING. Traces that hit the thinking cap were not given more thinking room; '
            'the end-of-thinking delimiter was appended to their exact stored prefix and the model '
            'answered from the reasoning it had. This is not the reference protocol and this row '
            'is not comparable with an unforced published number. Precedent: s1, Muennighoff et '
            'al. 2025.'),
        method=('forced budget: traces that terminated under the cap are carried over unchanged; '
                'traces that hit it get the suffix appended and an answer span of '
                f'{ans_tok} tokens, verified token-exact per item against a calibrated suffix '
                'length'),
        n_carried=len(keep), n_forced=len(todo) - nbad, n_not_forceable=nbad,
        n_answer_truncated=nstill, code_commit=commit)
    with open(os.path.join(OUT, f'{a.task}.{tag}.meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'\nwrote {out_path}  ({len(keep)} carried + {len(todo) - nbad} forced, '
          f'{nbad} not forceable, {nstill} still truncated in the answer span)')
    if nbad or nstill:
        print(f'  NOTE: {nbad + nstill} item(s) remain truncated — this row is '
              f'{100*(nbad+nstill)/len(base):.1f} % truncated, not 0 %.')

    if a.pair_with:
        report_paired(a.task, tag, a.pair_with)


if __name__ == '__main__':
    main()
