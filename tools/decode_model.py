#!/usr/bin/env python3
"""decode_model.py — fit decode cost as `a + b x context` from the battery's own records.

WHY A SINGLE tok/s NUMBER HID THE BOTTLENECK FOR THE WHOLE PROJECT. Every optimisation pass on
this repo measured decode on short prompts and reported one throughput figure. Benchmarks said
24-25 tok/s; the evaluation battery sustained 8-15 and fell to ~10 on the long rows. Both were
correct measurements of different points on a line nobody had fitted, and the term that dominates
real work -- everything that scales with context -- was invisible at the contexts being tested.

    ms per target forward = a + b x context

`a` is the context-independent cost: weights read per forward, comparable directly against the
bandwidth wall. `b` is everything that grows with resident context: the DSA index path, the
compressed-cache reads, the top-k selection. They are different problems with different fixes,
and only the split says which one to work on.

PER LEG, NOT PER RECORD. An extended record carries the BASE leg's `timings` while its `usage`
has been updated to the TOTAL completion tokens -- eval_extend.py keeps them apart on purpose, and
dividing one by the other silently understates cost on exactly the longest generations. The
continuation leg's own timings are empty because /v1/completions returns no `timings` object at
all, so decode above the base budget is currently UNMEASURED. Fix that in the server before
trusting any number at high context.

PER FORWARD, NOT PER TOKEN. Speculation commits tau tokens per target forward (tau ~2.9 here), so
tok/s is not comparable with a roofline computed from bytes per forward. Records carry
`timings.tokens_per_verify`; multiplying ms/token by it removes speculation from the comparison
entirely. Without that correction the constant term looks ~3x better than it is.

  python3 tools/decode_model.py                    # every row on disk
  python3 tools/decode_model.py --task lcb         # one task
  python3 tools/decode_model.py --per-token        # tok/s view instead of the roofline view

Report BOTH coefficients after every optimisation, never a single number at one context.
"""
import argparse, glob, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'evidence', 'evals')
BW = 240.0               # GB/s achievable, measured by tools/bw_probe.cu
B_TOK = 11202.36         # MB read by an M=1 AR step (MODEL_INVENTORY.md)
B_ROUTED = 3449.29       # MB of that which is the top-6 routed experts
B_DRAFT = 1892.0         # MB floor for the 3 MTP blocks + draft lm_head

# THE WALL FOR A TARGET FORWARD IS NOT THE WALL FOR A TOKEN, AND CONFLATING THEM COST THIS PROJECT
# A HEADLINE. B_tok is defined for ONE position. A speculative target forward verifies K positions
# and reads the UNION of the experts they route to -- measured at 17.53 of a possible 30 at K=5
# (LOOP_LOG, DSV4_MOEUNION; the instrument self-validates at K=1 -> exactly 6.00) -- plus the whole
# draft side. Dividing B_tok by a per-forward wall time therefore understates efficiency by ~1.76x.
# PERF.md says so in as many words ("a lower bound on efficiency, not an estimate of it"); the 34%
# that fell out of it was then quoted as an estimate anyway, here and in the research prompt.
UNION_K5 = 17.53         # measured expert union at K=5
UNION_FIT = 22.0         # the same quantity implied by the K-sweep fit -- an honest upper end


def forward_wall_ms(union):
    """ms a target forward would take at BW, given an expert-union assumption."""
    mb = (B_TOK - B_ROUTED) + union / 6.0 * B_ROUTED + B_DRAFT
    return mb / (BW * 1000.0) * 1000.0, mb


def collect(pattern, per_token):
    pts = []
    for p in glob.glob(os.path.join(OUT, pattern)):
        if '.meta.' in p:
            continue
        for line in open(p):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            t, u = r.get('timings') or {}, r.get('usage') or {}
            e = r.get('extension') or {}
            dm, pt_ = t.get('decode_ms'), u.get('prompt_tokens', 0)
            tv = t.get('tokens_per_verify')
            # `timings` always describes the BASE leg. For a continued record `usage` describes the
            # MERGED total, so the divisor must come from the extension block or the record reads
            # ~3x faster than it ran.
            ct = (e.get('base_completion_tokens') if e.get('method') == 'prefix-continuation'
                  else u.get('completion_tokens', 0))
            if not dm or not ct or ct < 64:
                continue
            if not per_token and not tv:
                continue                       # cannot divide out speculation for this record
            # Mean resident context DURING decode, not the final length: the cost is incurred all
            # the way up, so charging the whole generation at its end length overstates the slope.
            ctx = pt_ + ct / 2.0
            y = dm / ct * (1.0 if per_token else tv)
            pts.append((ctx, y, tv or 1.0))
    return pts


def fit(pts):
    n = len(pts)
    mx = sum(x for x, _, _ in pts) / n
    my = sum(y for _, y, _ in pts) / n
    sxx = sum((x - mx) ** 2 for x, _, _ in pts)
    b = sum((x - mx) * (y - my) for x, y, _ in pts) / sxx
    a = my - b * mx
    # R^2, so a fit over a task with almost no context spread announces itself as untrustworthy
    ss_res = sum((y - (a + b * x)) ** 2 for x, y, _ in pts)
    ss_tot = sum((y - my) ** 2 for _, y, _ in pts)
    return a, b, (1 - ss_res / ss_tot) if ss_tot else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', default='*')
    ap.add_argument('--per-token', action='store_true',
                    help='fit ms/token (throughput view) instead of ms/forward (roofline view)')
    a = ap.parse_args()

    pts = collect(f'{a.task}.*.jsonl', a.per_token)
    if len(pts) < 40:
        sys.exit(f'only {len(pts)} usable generations — need ~40 with a real spread of context')
    A, B, r2 = fit(pts)
    taus = sorted(t for _, _, t in pts)
    unit = 'token' if a.per_token else 'target forward'

    print(f'n = {len(pts)} generations   tau p50 = {taus[len(taus)//2]:.2f}   R^2 = {r2:.3f}')
    print(f'\n  ms per {unit} = {A:.2f} + {B*1000:.3f} x (context / 1000)\n')
    # NEVER PRINT A ROW THE DATA DOES NOT REACH. The first version of this tool extrapolated the
    # fit to 24000 and reported "78 % of a forward is context" as though it had been measured; the
    # records stop near 6000, because the only generations that go deeper are continuation legs and
    # those carry no timings. An extrapolation printed in the same table as a measurement is
    # indistinguishable from one, which is the exact failure this instrument exists to end.
    ctx_max = max(x for x, _, _ in pts)
    print(f'  measured context range: {min(x for x,_,_ in pts):.0f} to {ctx_max:.0f}\n')
    print(f'  {"context":>8}{"ms":>10}{"tok/s":>9}{"ctx share":>11}')
    for c in (0, 2000, 4000, 8000, 16000, 24000):
        if c > ctx_max:
            print(f'  {c:>8}{"— beyond the measured range; instrument /v1/completions to reach it":>41}')
            continue
        ms = A + B * c
        tps = 1000.0 / ms * (1.0 if a.per_token else taus[len(taus)//2])
        print(f'  {c:>8}{ms:>10.2f}{tps:>9.2f}{B*c/ms:>10.0%}')

    if not a.per_token:
        # Report a BAND, not a point. The expert union is the uncertain input and it moves the
        # answer by ~8 points, so quoting one number here is what created the problem above.
        print()
        for name, u in (('measured union 17.53', UNION_K5), ('fit-implied union 22.0', UNION_FIT)):
            wall, mb = forward_wall_ms(u)
            print(f'  {name:<24} wall {wall:6.2f} ms ({mb:6.0f} MB)   constant term is '
                  f'{wall/A*100:4.1f}% of achievable, headroom {A/wall:.2f}x')
        lo, _ = forward_wall_ms(UNION_K5)
        hi, _ = forward_wall_ms(UNION_FIT)
        print(f'\n  Term A headroom {A/hi:.2f}-{A/lo:.2f}x. Term B, at the deepest measured context,')
        print(f'  is {B*ctx_max:.0f} ms against a byte floor of well under 1 ms -- two orders of')
        print(f'  magnitude, and the only place a large factor is available.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
