#!/usr/bin/env python3
"""encode_prompt.py — turn text into the token-id list `build/decode` takes as argv[2].

Why this exists, and why it is Python: R1 (re-fit the adaptive-verify threshold across more than
one prompt) needs ids for prompts other than the canonical one, and INVENTING ids is exactly the
class of mistake this project has a rule against. So the ids must come from the checkpoint's own
tokenizer.

Cycle 1 wrote `tools/encode_prompt.cpp` against the in-tree `include/tokenizer.h`. Cycle 2 built it
and its self-gate FAILED (see LOOP_LOG Finding 51 for the exact output): `include/tokenizer.h` is a
gemma-4 sentencepiece tokenizer — it normalizes ' ' to U+2581 and byte-falls-back unknown chars —
while this checkpoint's tokenizer.json is a ByteLevel BPE with 'G-with-dot' for space and a
three-stage Split-regex pre-tokenizer. Reimplementing that pre-tokenizer in C++ correctly is a
project of its own and every bug in it silently produces plausible-looking wrong ids. The
`tokenizers` library (0.23.1 on this box) executes the checkpoint's own tokenizer.json spec
verbatim, pre-tokenizer included, so it cannot drift from the checkpoint.

This tool GATES ITSELF exactly as the C++ one did: if `The capital of France is` does not come back
as 671,6102,294,8760,344 it prints nothing usable and exits non-zero.

  run:  python3 tools/encode_prompt.py <ckpt>/tokenizer.json ["text" ...]
        (with no texts it emits the built-in R1 prompt suite)
"""
import sys
from tokenizers import Tokenizer

# The R1 suite. Text only — the ids are derived below, never typed. Chosen to vary the two things
# the adaptive-verify threshold is plausibly sensitive to: how predictable the continuation is
# (a memorized fact vs. free prose) and what kind of token stream it is (English, list, code).
SUITE = [
    "The capital of France is",                                  # canonical: memorized, highly predictable
    "The three primary colors are red, blue, and",                # list continuation
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",   # code, rigid syntax
    "In 1969, the first humans to walk on the Moon were",         # factual recall, longer context
    "She opened the door and, to her complete surprise,",         # open-ended prose, low predictability
    "The derivative of x squared with respect to x is",           # math, short and determinate
]

CANON_TEXT = "The capital of France is"
CANON_IDS = [671, 6102, 294, 8760, 344]
BOS = 0  # '<...begin_of_sentence...>' — id 0 in this checkpoint; the canonical argv starts with it


def main():
    if len(sys.argv) < 2:
        print("usage: %s <tokenizer.json> [text...]" % sys.argv[0], file=sys.stderr)
        return 2
    tok = Tokenizer.from_file(sys.argv[1])
    print("[encode] vocab %d" % tok.get_vocab_size())

    got = tok.encode(CANON_TEXT, add_special_tokens=False).ids
    ok = got == CANON_IDS
    print("[encode] GATE canonical: got %s   want %s  -> %s"
          % (got, CANON_IDS, "GATE PASS" if ok else "GATE FAIL"))
    if not ok:
        print("[encode] encoder does not reproduce the canonical prompt on this checkpoint;\n"
              "[encode] its ids are NOT usable. Refusing to print prompts.", file=sys.stderr)
        return 1

    # Round-trip gate: decode(encode(t)) == t for every prompt we are about to emit. A BPE that
    # encodes but does not round-trip is silently dropping or mangling characters, and the model
    # would then be prefilled on a prompt nobody wrote.
    texts = sys.argv[2:] if len(sys.argv) > 2 else SUITE
    rows = []
    for t in texts:
        ids = tok.encode(t, add_special_tokens=False).ids
        rt = tok.decode(ids)
        if rt != t:
            print("[encode] GATE roundtrip FAIL for %r -> %r" % (t, rt), file=sys.stderr)
            return 1
        rows.append((t, [BOS] + ids))
    print("[encode] GATE roundtrip: %d/%d prompts decode back exactly -> GATE PASS"
          % (len(rows), len(rows)))

    for t, ids in rows:
        print("%s      # (%d ids) %r" % (",".join(str(i) for i in ids), len(ids), t))
    # DSV4_PROMPTS form: prompts separated by ';', ids by ','.
    print("\n[encode] DSV4_PROMPTS=%s" % ";".join(",".join(str(i) for i in ids) for _, ids in rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
