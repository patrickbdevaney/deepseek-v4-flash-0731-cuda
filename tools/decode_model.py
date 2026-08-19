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
WALL_MS = 46.70          # ms per target forward at 240 GB/s achievable -- see ROOFLINE.md
BW = 240.0


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
            dm, ct, pt_ = t.get('decode_ms'), u.get('completion_tokens', 0), u.get('prompt_tokens', 0)
            tv = t.get('tokens_per_verify')
            if not dm or ct < 64:
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
    print(f'  {"context":>8}{"ms":>10}{"tok/s":>9}{"ctx share":>11}')
    for c in (0, 2000, 4000, 8000, 16000, 24000):
        ms = A + B * c
        tps = 1000.0 / ms * (1.0 if a.per_token else taus[len(taus)//2])
        print(f'  {c:>8}{ms:>10.2f}{tps:>9.2f}{B*c/ms:>10.0%}')

    if not a.per_token:
        # The whole point of the split: name the headroom in each term separately, because they
        # are different engineering problems and the bigger one changes with context.
        print(f'\n  constant term      {A:7.2f} ms   vs {WALL_MS:.2f} ms wall @ {BW:.0f} GB/s'
              f'   -> {WALL_MS/A*100:.0f}% of achievable, headroom {A/WALL_MS:.2f}x')
        for c in (8000, 24000):
            tot = A + B * c
            print(f'  at ctx {c:>5}: fixing the constant alone -> {tot/(WALL_MS + B*c):.2f}x   '
                  f'fixing the context term alone -> {tot/A:.2f}x   both -> {tot/WALL_MS:.2f}x')
    return 0


if __name__ == '__main__':
    sys.exit(main())
