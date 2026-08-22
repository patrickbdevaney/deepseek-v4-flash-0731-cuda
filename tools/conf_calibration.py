#!/usr/bin/env python3
"""conf_calibration.py — the gate that actually decides ladder 2.3.

WHY THE USUAL GATE CANNOT DECIDE THIS ONE. `promote_head.py` grades a head on suite mean `tau`
measured on the frozen 8-prompt protocol, which holds the draft width **fixed at 5**. The confidence
head exists so the ENGINE can VARY that width -- spend 5 drafts where acceptance is likely and 1-2
where it is not, per `argmax_k E[A(T_k)] / C(k)`. A fixed-width instrument cannot see a
variable-width win by construction, so the protocol's refusal of the 2.3 arms is expected and
carries no information about the lever. Grading 2.3 on it would be exactly the failure
`wiki/measurement-and-traps.md` keeps recording: using the ruler that happens to be lying around.

Worse, 2.3 is forced to carry HASS -- the `accepted` label is only meaningful when the draft is
free-running (F100; train_head.py:800) -- and P2.6 measured HASS at **-0.046 tau**. So these arms
are expected to score BELOW the incumbent for a reason that has nothing to do with whether the
confidence head works.

THE QUESTION THAT DOES DECIDE IT, and it needs no engine change: **does the trained confidence head
predict acceptance?** If it does not, the verify-time CUDA work is worthless and must not be
written -- 2.3 then closes as a negative for the price of two GPU sessions instead of a rewrite.

TWO-STAGE, CHEAP-NEGATIVE-FIRST. This tool scores AUC on capture the head has already trained on,
which INFLATES the arm's number through memorisation. That is deliberate and it is safe in one
direction only:

    arm AUC ~ 0.5 even on data it trained on   -> DECISIVE NEGATIVE. Stop. No held-out run needed.
    arm AUC >> control AUC                     -> PROMISING, NOT PROVEN. The gap is real (the
                                                  control's confidence head is untrained at
                                                  a_conf=0 and cannot memorise either, so the two
                                                  are inflated differently) but the level is not.
                                                  Confirm on held-out capture before writing CUDA.

The control is `s3recap-hass1-p25`: identical recipe, identical data, `a_conf = 0`, so its
confidence head received no gradient. It is the paired null.

USAGE

    python3 tools/conf_calibration.py evidence/calib_s3recap-conf1.0.jsonl \\
                                      --control evidence/calib_s3recap-hass1-p25.jsonl

Produce a dump with `train/train_head.py --calib-out <path>`; at `--lr 0 --resume <trained-dir>`
that is a pure inference pass over the capture and moves no weights.
"""
import argparse, json, sys
from collections import defaultdict


def auc(pairs):
    """Rank AUC of score vs binary label, ties averaged. Returns (auc, n_pos, n_neg)."""
    pos = sum(1 for _, a in pairs if a)
    neg = len(pairs) - pos
    if pos == 0 or neg == 0:
        return None, pos, neg
    ranked = sorted(pairs, key=lambda t: t[0])
    ranks, i = {}, 0
    while i < len(ranked):
        j = i
        while j + 1 < len(ranked) and ranked[j + 1][0] == ranked[i][0]:
            j += 1
        r = (i + j) / 2.0 + 1.0                      # 1-indexed, ties share the mean rank
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    s = sum(ranks[k] for k, (_, a) in enumerate(ranked) if a)
    return (s - pos * (pos + 1) / 2.0) / (pos * neg), pos, neg


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                rows.append((d["k"], d["c"], d["a"]))
    return rows


def report(name, rows):
    overall, p, n = auc([(c, a) for _, c, a in rows])
    print(f"\n{name}")
    print(f"  {len(rows):,} positions, {p:,} accepted ({100*p/max(len(rows),1):.1f} %)")
    if overall is None:
        print("  AUC: undefined -- one class is empty, so this dump cannot answer anything")
        return None
    print(f"  AUC overall: {overall:.4f}")
    byk = defaultdict(list)
    for k, c, a in rows:
        byk[k].append((c, a))
    print("  by draft position k (acceptance decays with k; a head that only learned the PRIOR")
    print("  will score ~0.5 within each k while looking informative pooled):")
    for k in sorted(byk):
        akv, kp, kn = auc(byk[k])
        rate = kp / max(kp + kn, 1)
        print(f"    k={k}  n={kp+kn:>7,}  accept {rate:6.1%}  "
              + (f"AUC {akv:.4f}" if akv is not None else "AUC undefined (one class empty)"))
    return overall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", help="JSONL from --calib-out for the a_conf > 0 arm")
    ap.add_argument("--control", help="JSONL for the a_conf = 0 paired null (s3recap-hass1-p25)")
    ap.add_argument("--floor", type=float, default=0.55,
                    help="AUC below which the arm is a decisive negative (default 0.55)")
    a = ap.parse_args()

    arm = report(f"ARM      {a.arm}", load(a.arm))
    ctl = report(f"CONTROL  {a.control}", load(a.control)) if a.control else None

    print("\n" + "=" * 78)
    if arm is None:
        print("VERDICT: INDETERMINATE -- the arm dump has only one class.")
        return 2
    if arm < a.floor:
        print(f"VERDICT: NEGATIVE. Arm AUC {arm:.4f} < {a.floor:.2f} ON DATA IT TRAINED ON.")
        print("  A head that cannot separate accepted from rejected positions even with")
        print("  memorisation available will not do it at serve time. Ladder 2.3's verify-time")
        print("  engine change is NOT justified -- do not write it. Close 2.3 as a negative and")
        print("  record it in wiki/negative-results.md.")
        return 1
    if ctl is not None and arm - ctl < 0.02:
        print(f"VERDICT: NEGATIVE. Arm {arm:.4f} vs untrained control {ctl:.4f} "
              f"(gap {arm-ctl:+.4f}).")
        print("  Whatever separation exists is present WITHOUT training the confidence head, so")
        print("  a_conf bought nothing. The signal is in the architecture, not in the arm.")
        return 1
    print(f"VERDICT: PROMISING, NOT PROVEN. Arm AUC {arm:.4f}"
          + (f" vs control {ctl:.4f} (gap {arm-ctl:+.4f})." if ctl is not None else "."))
    print("  The GAP is trustworthy -- the control's confidence head is untrained and cannot")
    print("  memorise either, so both numbers are inflated by the same mechanism. The LEVEL is")
    print("  not: this is capture the arm trained on. Before writing any CUDA:")
    print("    1. capture the 64 held-out sequences' taps and re-run this on them;")
    print("    2. only then implement verify-time adaptive width and re-measure tau with the")
    print("       width VARYING, against the fixed-width incumbent 3.8413.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
