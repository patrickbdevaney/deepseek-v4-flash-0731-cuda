#!/usr/bin/env python3
"""stephash_compare.py — ladder 1.9. Find the first verify step where two runs of build/decode
disagree, and name the FIRST FIELD that disagrees there.

WHY A FIELD AND NOT A STEP. Generation is autoregressive: one flipped token permanently
de-synchronises everything after it, so a token-id diff (tools/genout_compare.py) reports the step
at which the two runs came apart and can say nothing at all about which computation produced the
flip -- every downstream quantity differs by then, including the ones that were only following
orders. `DSV4_STEPHASH` writes the whole causal chain of one verify, in dataflow order:

    mkv  hash of mkv[0] rows [0,ctx)   the draft's persistent main-KV cache
    mx   hash of main_x rows [0,ctx)   the draft's persistent hidden state
    din  hash of the draft block input after embed+hc_expand
    draft hash of the draft's proposed ids AND the margins the width controller reads
    lg   hash of the target's verify logits, VB x VOCAB
    acc/corr the accept decision

At the FIRST differing step every field before the first differing one is still equal, so the
leftmost differing field names the link that broke rather than its consequences. That is the whole
point of the instrument.

  python3 tools/stephash_compare.py <A.txt> <B.txt>

Exit 0 if every step matches in both files, 1 if they diverge, 2 on a malformed comparison.
"""
import sys, re

ORDER = ["cpos", "ctx", "cur", "VK", "mkv", "mx", "din", "draft", "lg", "acc", "corr"]
# What each field being the FIRST to differ implies. Printed with the verdict so the reader does
# not have to hold the dataflow in their head.
MEANS = {
    "cpos":  "the two runs are not even at the same position -- the comparison is misaligned, not a divergence",
    "ctx":   "same as cpos: misaligned, void",
    "cur":   "the committed token entering this step already differs, so the divergence is EARLIER "
             "than this step and this file's first differing step is not the first event",
    "VK":    "the adaptive verify width differs while the margins that set it (inside `draft`) do not "
             "-- that would be a host-side bug in the width controller",
    "mkv":   "the draft's PERSISTENT main-KV cache differs before the draft reads it: dspark_main_kv "
             "(or its incremental form) is the source",
    "mx":    "main_x -- the draft's persistent hidden state, written by dspark_main_x from the "
             "verify taps -- differs before the draft reads it",
    "din":   "the draft's block input differs although mkv/main_x match: k_embed / k_hc_expand, "
             "i.e. a pure function of ids, which would be very strange",
    "draft": "mkv, main_x and the draft input all MATCH and the draft's output does not -- the "
             "DSpark draft chain (3 blocks + head) is the nondeterministic component",
    "lg":    "the whole draft side matches and the TARGET's verify logits differ -- the 43-layer "
             "verify forward is the nondeterministic component",
    "acc":   "the logits match and the accept decision does not -- host-side argmax/accept logic",
    "corr":  "same as acc: host-side, downstream of identical logits",
}

def parse(path):
    rows = []
    pat = re.compile(r"(\w+)=([0-9a-fx\-]+)")
    for ln in open(path):
        ln = ln.strip()
        if not ln.startswith("p"):
            continue
        head = ln.split()
        pt, vf = head[0], head[1]
        d = {k: v for k, v in pat.findall(ln)}
        d["_key"] = (pt, vf)
        rows.append(d)
    return rows

def main():
    if len(sys.argv) != 3:
        print(__doc__); return 2
    A, B = parse(sys.argv[1]), parse(sys.argv[2])
    print(f"{sys.argv[1]}: {len(A)} steps   {sys.argv[2]}: {len(B)} steps")
    if not A or not B:
        print("MALFORMED: one side has no steps"); return 2

    # A FIELD THAT WAS NEVER MEASURED COMPARES EQUAL, AND EQUAL READS AS EXONERATED.
    # DSV4_STEPHASH_LVL=1 skips the two expensive device hashes and writes them as zero. On the
    # first run of the H arms that made `mkv`/`mx` compare equal at a step where `draft` differed,
    # and this tool duly reported "mkv and main_x MATCH and the draft's output does not -- the
    # DSpark draft chain is the nondeterministic component". That conclusion was manufactured
    # entirely by the level flag. Drop all-zero fields from the ordering and SAY SO, so a level-1
    # file can never be read as evidence about a quantity level 1 does not compute.
    unmeasured = [f for f in ORDER
                  if all(r.get(f) in (None, "0000000000000000") for r in A)
                  and all(r.get(f) in (None, "0000000000000000") for r in B)]
    order = [f for f in ORDER if f not in unmeasured]
    if unmeasured:
        print(f"  NOT MEASURED in these files (zero on both sides, excluded): {','.join(unmeasured)}")
        print(f"  -> re-run with DSV4_STEPHASH_LVL=2 before drawing any conclusion about them")

    # Group by sweep point so each point gets its own verdict; a point is only comparable up to the
    # shorter of the two, and a length difference is itself reported (a run that diverged emits a
    # different number of verifies for the same NGEN).
    pts = sorted({r["_key"][0] for r in A} | {r["_key"][0] for r in B},
                 key=lambda p: int(p[1:]))
    diverged = False
    for pt in pts:
        a = [r for r in A if r["_key"][0] == pt]
        b = [r for r in B if r["_key"][0] == pt]
        n = min(len(a), len(b))
        if not n:
            print(f"  {pt}: MISSING on one side ({len(a)} vs {len(b)} steps)"); diverged = True; continue
        first = None
        for i in range(n):
            fields = [f for f in order if a[i].get(f) != b[i].get(f)]
            if fields:
                first = (i, fields); break
        if first is None:
            extra = "" if len(a) == len(b) else f"  (lengths {len(a)} vs {len(b)} -- compared {n})"
            print(f"  {pt}: IDENTICAL over {n} verifies, every field{extra}")
            continue
        diverged = True
        i, fields = first
        lead = fields[0]
        print(f"  {pt}: DIVERGES at verify {a[i].get('_key')[1]} (index {i} of {n}), "
              f"cpos={a[i].get('cpos')} ctx={a[i].get('ctx')}")
        print(f"        first differing field: {lead}   (all differing here: {','.join(fields)})")
        print(f"        -> {MEANS.get(lead, 'unknown field')}")
        if unmeasured:
            print(f"        !! that reading ASSUMES {','.join(unmeasured)} were equal; they were NOT "
                  f"COMPUTED in these files. The claim is only valid at DSV4_STEPHASH_LVL=2.")
        for f in order:
            if f in fields:
                print(f"          {f:6s} A={a[i].get(f)}  B={b[i].get(f)}   <-- differs")
        # The step BEFORE is the last agreement; print it so the reader can see the state the
        # divergence grew out of rather than taking the claim on trust.
        if i > 0:
            print(f"        last agreeing verify: index {i-1}, cpos={a[i-1].get('cpos')}, "
                  f"acc={a[i-1].get('acc')} corr={a[i-1].get('corr')}")
    print("VERDICT: " + ("the two runs DIVERGE" if diverged
                         else "the two runs are IDENTICAL at every field of every verify"))
    return 1 if diverged else 0

if __name__ == "__main__":
    sys.exit(main())
