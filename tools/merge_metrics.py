#!/usr/bin/env python3
"""merge_metrics.py — stitch per-chunk train_metrics.json into one session-level record.

A chunked session is one training run split for disk reasons, so its metrics should read as one
run: a single loss history ordered by global step, and diagnostics pooled over every position seen
rather than taken from whichever chunk happened to be last.

  python3 tools/merge_metrics.py <chunk metrics...> --out <session metrics>
"""
import json, sys


def main():
    args = sys.argv[1:]
    out = None
    if "--out" in args:
        i = args.index("--out")
        out = args[i + 1]
        args = args[:i] + args[i + 2:]
    if not args:
        sys.exit("no input metrics")

    parts = []
    for p in args:
        try:
            parts.append(json.load(open(p)))
        except Exception as e:
            print(f"  skipping {p}: {e}")
    if not parts:
        sys.exit("no readable metrics")

    hist = sorted((h for p in parts for h in p.get("history", [])), key=lambda h: h["step"])
    tail = hist[-max(1, len(hist) // 10):] if hist else []

    # Pool the diagnostics by TOKEN COUNT, not by averaging the per-chunk percentages: chunks can
    # differ in size and an unweighted mean of percentages would over-weight a small chunk.
    def pooled(key):
        num = den = 0.0
        for p in parts:
            v, n = p.get(key), p.get("n_sequences") or 0
            if v is not None and n:
                num += v * n; den += n
        return round(num / den, 4) if den else None

    m = dict(parts[-1])                                   # carry the last chunk's config fields
    m.update({
        "chunks": len(parts),
        "chunk_sources": args,
        "n_sequences": sum(p.get("n_sequences", 0) for p in parts),
        "steps": len(hist),
        "loss_first": hist[0]["loss"] if hist else None,
        "loss_last": hist[-1]["loss"] if hist else None,
        "loss_tail_mean": (sum(h["loss"] for h in tail) / len(tail)) if tail else None,
        "diag_draft_eq_target_argmax_pct": pooled("diag_draft_eq_target_argmax_pct"),
        "diag_draft_eq_ground_truth_pct": pooled("diag_draft_eq_ground_truth_pct"),
        "history": hist,
    })
    if out:
        json.dump(m, open(out, "w"), indent=1)
    print(f"merged {len(parts)} chunk(s), {m['n_sequences']} sequences, {m['steps']} steps: "
          f"loss {m['loss_first']} -> {m['loss_last']} (tail mean {m['loss_tail_mean']})"
          + (f" -> {out}" if out else ""))


if __name__ == "__main__":
    main()
