#!/usr/bin/env python3
"""target_margin.py — the ceiling on what fine-tuning the drafter can buy, measured on the TARGET.

F110 closed every route through the K-selection controller at +1.8 % and concluded that the whole
remaining path to 30 tok/s runs through the draft's logit **margins** -- specifically, that gating
positions 1-5 need roughly +1.0 to +1.5 added to their margins.

That conclusion has an unexamined premise: **that such margins are available at all.** The drafter
is trained toward the target's distribution (TV carries 0.9 of the loss weight), so the best a
perfectly-trained drafter can do at a position is reproduce the *target's own* top-2 logit gap. If
the target is itself uncertain there, no amount of training produces a confident draft, the gate
stays shut, and 30 tok/s is unreachable by fine-tuning at any corpus size.

So this measures the target's margin distribution from the captured `lm_head` input:

    target_logits = lm_head @ lm_in[t]      ->      margin = top1 - top2

and reports the K=7 share a perfectly-matched drafter would reach -- the ceiling the fine-tune is
working toward, as opposed to the 33.0 % it starts from.

WHAT THIS IS NOT. A perfectly-matched drafter is not achievable; the drafter conditions on three
backbone taps and its own drafted prefix, not on the full forward pass. This is an upper bound. Its
value is asymmetric and that is the point: if the bound is comfortably above the required share the
premise survives, and if it is below, S5 cannot reach 30 no matter how well it trains.

  python3 tools/target_margin.py --capture <dir> --ckpt <ckpt> [--limit 10] [--gate 1.5]
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from read_capture import load_shard                                   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--limit", type=int, default=10, help="sequences to read")
    ap.add_argument("--gate", type=float, default=1.5, help="the engine's adaptK threshold")
    ap.add_argument("--block", type=int, default=6)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--per-k", dest="per_k", default=None,
                    help="measured K:tokens:ms triples from a spec-decode log, comma separated")
    ap.add_argument("--accept", type=float, default=0.95,
                    help="per-position acceptance assumed for a margin-matched drafter")
    a = ap.parse_args()

    import torch
    from safetensors.torch import load_file

    idx = json.load(open(os.path.join(a.ckpt, "model.safetensors.index.json")))
    hk = next(k for k in idx["weight_map"] if k in ("lm_head.weight", "head.weight"))
    W = load_file(os.path.join(a.ckpt, idx["weight_map"][hk]))[hk].to(a.device)
    print(f"lm_head {hk}: {tuple(W.shape)} {W.dtype}")

    rows = [json.loads(l) for l in open(os.path.join(a.capture, "manifest.jsonl")) if l.strip()][:a.limit]
    allm = []
    for r in rows:
        sh = load_shard(os.path.join(a.capture, r["file"]), as_float32=True)
        if sh.get("lm_in") is None:
            sys.exit("capture is v1 (no lm_head input); re-capture with the current engine")
        x = torch.from_numpy(sh["lm_in"].copy()).to(a.device, W.dtype)
        lg = (x @ W.T).float()
        top2 = lg.topk(2, dim=-1).values
        allm.append((top2[:, 0] - top2[:, 1]))
        print(f"  {r['file']}: {lg.shape[0]} positions")
    m = torch.cat(allm)
    n = m.numel()
    q = torch.quantile(m, torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90]))
    print(f"\nTARGET top1-top2 logit margin over {n} positions:")
    print(f"  p10 {q[0]:.2f}   p25 {q[1]:.2f}   median {q[2]:.2f}   p75 {q[3]:.2f}   p90 {q[4]:.2f}")
    share = float((m >= a.gate).float().mean())
    print(f"  share >= gate {a.gate}: {100*share:.1f}%")

    # A verify reaches K=7 only if the gating positions 1..block-1 ALL clear. Under perfect
    # matching those are consecutive target positions, so estimate the conjunction empirically by
    # sliding a window rather than assuming independence -- adjacent positions are correlated
    # (a confident region stays confident) and independence would understate the share badly.
    ng = a.block - 1
    ok = (m >= a.gate).float()
    win = ok.unfold(0, ng, 1).prod(dim=-1) if n > ng else ok[:0]
    print(f"\n  CEILING: {100*float(win.mean()):.1f}% of positions start a run of {ng} consecutive "
          f"gating margins above {a.gate}")
    ind = share ** ng
    print(f"  (independence would have said {100*ind:.1f}% -- adjacency correlation is worth "
          f"{100*(float(win.mean())-ind):+.1f} points, which is why this is measured, not assumed)")
    print(f"\n  engine today: 33.0 % of verifies reach K=7 (F109/F110, untrained drafter)")
    print(f"  30 tok/s needs ~85-100 %.")

    # ---- the K-mix and tok/s a perfectly-matched drafter would reach ----------------------------
    # Replicate src/decode.cu's loop against the TARGET's margins. A verify starting at position t
    # gates on the margins at t+1..t+block-1, so this is the same 5-position run measured above,
    # now resolved into the full width distribution instead of just its top bin.
    if a.per_k:
        import statistics as st
        pk = {}
        for tok_ms in a.per_k.split(","):
            K, tk, ms = tok_ms.split(":")
            pk[int(K)] = (float(tk), float(ms))
        # ACCEPTANCE AND MARGINS MUST BE MODELLED CONSISTENTLY. Pairing the target's margins with
        # pass 1's measured acceptance is incoherent: a drafter whose margins equal the target's
        # has essentially the target's distribution, so under greedy decode its argmax IS the
        # target's and acceptance approaches 1. `--accept` makes the assumption explicit and
        # sweepable rather than smuggling pass 1's numbers into a hypothetical drafter.
        ml = m.tolist()
        print(f"\n  threshold sweep, drafter margins == TARGET margins, acceptance {a.accept:.2f}:")
        print(f"  {'gate':>6} {'K=7 share':>10} {'tok/verify':>11} {'tok/s':>8}")
        for thr in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0):
            mix = {}
            for t in range(len(ml) - a.block):
                VK = 2
                while VK < a.block + 1 and ml[t + VK - 1] >= thr:
                    VK += 1
                mix[VK] = mix.get(VK, 0) + 1
            n2 = sum(mix.values())
            # expected tokens at width K with per-position acceptance p: 1 + sum_{j=1..K-1} p^j
            def et(K, p=a.accept):
                return 1.0 + sum(p ** j for j in range(1, K))
            tk = sum(mix[K] * et(K) for K in mix)
            ms = sum(mix[K] * pk[K][1] for K in mix)
            mark = "  <- shipped" if abs(thr - 1.5) < 1e-9 else ""
            print(f"  {thr:>6.2f} {100*mix.get(7,0)/n2:>9.1f}% {tk/n2:>11.3f} "
                  f"{1000*tk/ms:>8.2f}{mark}")
        print(f"\n  THE COUPLING THIS EXPOSES: the gate exists because the drafter is unreliable, so"
              f"\n  the optimal threshold FALLS as the drafter improves. F109/F110 showed the "
              f"threshold is\n  not a lever *for today's drafter*; this shows that conclusion does "
              f"not survive training.\n  Re-fitting adaptK on the trained head is therefore a "
              f"required post-training step, not an\n  optional one -- and it is the step that "
              f"'the threshold does not matter' would have skipped.")


if __name__ == "__main__":
    main()
