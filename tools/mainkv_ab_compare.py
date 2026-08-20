#!/usr/bin/env python3
"""mainkv_ab_compare.py — the token-identity half of a decode A/B, and the paired timing table.

WHY THIS EXISTS. DECODE_LADDER.md's first invariant is that a kernel change either produces
byte-identical generated token ids or ships behind an explicit gate. For a change on the DRAFT side
that is not a formality: the draft only proposes, so a wrong draft does not change what the target
emits -- it changes how much of the draft is accepted, i.e. tau. A run can therefore be "correct"
and 40 % slower at the same time, and a throughput number alone cannot tell the two apart.

`decode_fit_probe.py` samples with `seed=1000+rep` and the engine reseeds per request, so a sweep is
deterministic given the same weights and kernels. Two arms that produce the same completion hash at
every (target, rep) have produced byte-identical token sequences at every context in the sweep. That
is the invariant, checked on the same records that carry the timings, at no extra runtime.

  python3 tools/mainkv_ab_compare.py <before_dir> <after_dir>

Exit status is 1 if any leg present in both arms disagrees, so a caller can gate on it.
"""
import json, os, statistics as st, sys


def load(d):
    out = {}
    for name in ('postfix.sweep.jsonl', 'control.fresh.jsonl'):
        p = os.path.join(d, name)
        if not os.path.exists(p):
            continue
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[r['id']] = r
    return out


def fwd(r):
    t, u = r['timings'], r['usage']
    return t['decode_ms'] / u['completion_tokens'] * t['tokens_per_verify']


def ctx(r):
    return r['usage']['prompt_tokens'] + r['usage']['completion_tokens'] / 2


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    A, B = load(argv[1]), load(argv[2])
    shared = sorted(set(A) & set(B))
    if not shared:
        print(f'no legs in common between {argv[1]} and {argv[2]}')
        return 2

    # --- corpus identity first. A prompt that differs makes every other comparison meaningless. ---
    csa = {A[k].get('corpus_sha256') for k in shared}
    csb = {B[k].get('corpus_sha256') for k in shared}
    print(f'corpus sha256: before {sorted(csa)}  after {sorted(csb)}')
    if csa != csb:
        print('FAIL: the two arms did not read the same corpus; the A/B is void.')
        return 1

    # --- token identity ---
    # A LEG WITH NO HASH IS NOT A PASSING LEG. Records written before this field existed compare
    # None == None, which would report "byte-identical" for a comparison that was never made.
    nohash = [k for k in shared if not A[k].get('text_sha256') or not B[k].get('text_sha256')]
    miss = [k for k in shared if k not in nohash
            and A[k]['text_sha256'] != B[k]['text_sha256']]
    print(f'\ntoken identity: {len(shared)-len(miss)-len(nohash)}/{len(shared)} legs proven '
          f'byte-identical, {len(miss)} differ, {len(nohash)} unproven (no hash on record)')
    for k in miss[:10]:
        print(f'  DIFFER {k}: {A[k].get("text_sha256","-")[:16]} vs {B[k].get("text_sha256","-")[:16]}'
              f'  (len {A[k].get("text_len")} vs {B[k].get("text_len")})')

    # --- paired timing table, per point, medians over reps ---
    pts = sorted({(A[k]['kind'], A[k]['target_tokens']) for k in shared}, key=lambda x: (x[0], -x[1]))
    print(f'\n{"kind":8}{"target":>8}{"n":>3}{"ctx":>7}'
          f'{"before":>10}{"after":>10}{"delta":>9}{"speedup":>9}'
          f'{"tau b":>8}{"tau a":>8}{"w b":>7}{"w a":>7}{"tok/s b":>9}{"tok/s a":>9}')
    for kind, target in pts:
        ks = [k for k in shared if A[k]['kind'] == kind and A[k]['target_tokens'] == target]
        fb, fa = st.median(fwd(A[k]) for k in ks), st.median(fwd(B[k]) for k in ks)
        tb, ta = (st.median(A[k]['timings']['tokens_per_verify'] for k in ks),
                  st.median(B[k]['timings']['tokens_per_verify'] for k in ks))
        wb = st.median((A[k].get('spec_profile') or {}).get('mean_width') or 0 for k in ks)
        wa = st.median((B[k].get('spec_profile') or {}).get('mean_width') or 0 for k in ks)
        sb = st.median(A[k]['timings']['tokens_per_second'] for k in ks)
        sa = st.median(B[k]['timings']['tokens_per_second'] for k in ks)
        print(f'{kind:8}{target:8d}{len(ks):3d}{st.median(ctx(A[k]) for k in ks):7.0f}'
              f'{fb:10.2f}{fa:10.2f}{fa-fb:+9.2f}{fb/fa:8.3f}x'
              f'{tb:8.3f}{ta:8.3f}{wb:7.2f}{wa:7.2f}{sb:9.2f}{sa:9.2f}')

    print('\n"before"/"after" are ms per FORWARD (decode_ms/token x tau), the quantity '
          'tools/decode_model.py fits. tok/s is the user-visible number.')
    return 1 if (miss or nohash) else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
