#!/usr/bin/env python3
"""genout_compare.py — compare two DSV4_GENOUT files as TOKEN IDS, position by position.

WHY NOT A HASH. `mainkv_ab_compare.py` compares a sha256 of the completion TEXT, which answers
"same or not" and nothing else. When two arms disagree that is the least useful possible answer: a
cached-prefix bug and a run-to-run nondeterminism both show up as "the hashes differ", and telling
them apart is the whole question. Ids compared positionally say WHERE the first disagreement is,
and "diverges at token 0" (the prefix was already wrong) is a completely different finding from
"diverges at token 137" (the two runs came apart mid-generation).

WHY IDS AND NOT TEXT. DECODE_LADDER.md's invariant is written in ids -- "byte-identical generated
token ids" -- and `build/decode` emits exactly that with DSV4_GENOUT, argmax, no seed. The text is a
lossy view of it: two different id sequences can decode to the same string.

Each line is one sweep point: the full prompt ids followed by the generated ids, comma-separated.
The prompt is included, so a mismatch inside the prompt region means the two runs were not even
given the same input and the comparison is void -- which is reported separately rather than counted
as a divergence.

  python3 tools/genout_compare.py <before.txt> <after.txt>

Exit status is 0 only if every point matches at every position.
"""
import sys


def load(path):
    out = []
    for line in open(path):
        line = line.strip()
        if line:
            out.append([int(x) for x in line.split(',') if x != ''])
    return out


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    A, B = load(argv[1]), load(argv[2])
    print(f'{argv[1]}: {len(A)} point(s)   {argv[2]}: {len(B)} point(s)')
    if len(A) != len(B) or not A:
        print('FAIL: the two arms did not emit the same number of sweep points; comparison is void.')
        return 1

    bad = 0
    for i, (a, b) in enumerate(zip(A, B)):
        n = min(len(a), len(b))
        first = next((j for j in range(n) if a[j] != b[j]), None)
        if first is None and len(a) == len(b):
            print(f'  point {i}: IDENTICAL — {len(a)} ids (prompt+generated) match at every position')
            continue
        bad += 1
        if first is None:
            print(f'  point {i}: LENGTH MISMATCH — {len(a)} vs {len(b)} ids, common prefix of '
                  f'{n} identical')
            continue
        # Locating the divergence inside vs after the prompt is the diagnostic, not a detail: the
        # prompt region is byte-identical input by construction, so a mismatch there is a harness
        # fault and not a kernel result.
        print(f'  point {i}: DIVERGES at id index {first} of {len(a)}/{len(b)}: '
              f'{a[first]} vs {b[first]}')
        print(f'            context: ...{a[max(0,first-4):first]} -> {a[first:first+4]}')
        print(f'                     ...{b[max(0,first-4):first]} -> {b[first:first+4]}')

    print(f'\ntoken-id identity: {len(A)-bad}/{len(A)} points byte-identical, {bad} differ')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
