#!/usr/bin/env python3
"""make_corpus.py — build a CATEGORY-BALANCED prompt corpus for an S5 session.

WHY. Session 1 trained on reasoning prompts only (GSM8K-style math). Measured against the frozen
suite, the result was rank-ordered and unambiguous:

    IMPROVED   explanation +0.53, code_gen +0.51, reasoning +0.71, multi_turn +0.77
    REGRESSED  long_context -1.01, code_edit -0.54, agentic_format -0.45, short_factual -0.20

The three weakest categories improved most and the strongest regressed hardest. Net suite gain was
+5.5 % tok/s, but roughly half of what training bought on weak categories it gave back on strong
ones. A single-domain corpus does not just under-serve the other categories -- it actively trades
them away. So the corpus must be mixed, and balanced, from the start.

SOURCES. Real prompts, never invented, from datasets already on this box or fetched to
`corpus_src/`. Each category is drawn from a source that genuinely exhibits it:

    short_factual    dolly closed_qa/open_qa, short answers
    explanation      dolly general_qa / brainstorming
    reasoning        GSM8K train (the session-1 domain, now one eighth instead of all)
    code_gen         MBPP task descriptions
    code_edit        MBPP solutions with a single mutation applied + "fix it" framing
    multi_turn       UltraChat, truncated to a 2-turn exchange
    agentic_format   Berkeley Function-Calling Leaderboard questions + their tool schema
    long_context     dolly entries with a long `context` field (the needle is really in there)

TOKENISATION goes through the checkpoint's own tokenizer.json, and the module refuses to emit
anything if the canonical probe does not round-trip -- the same self-gate tools/encode_prompt.py
uses. Inventing ids is the one thing this project never does.

LAYOUT. scripts/s5_session.sh reserves the LAST `--holdout` lines as the never-trained hold-out, so
this writes the training prompts first and the hold-out last, each block balanced across all eight
categories. Session 1's hold-out was reasoning-only, which turned out to be a weak instrument: the
adaptK gate barely binds on it (mean K 5.26 at threshold 0.0 vs 5.27 at 1.5), so it could not see
the constraint the suite sees. A balanced hold-out can.

  python3 tools/make_corpus.py --out <file> --n 536 --holdout 64 [--src corpus_src]
"""
import argparse, hashlib, json, os, random, re, sys

CATS = ["short_factual", "explanation", "reasoning", "code_gen",
        "code_edit", "multi_turn", "agentic_format", "long_context"]
CANON_TEXT, CANON_IDS, BOS = "The capital of France is", [671, 6102, 294, 8760, 344], 0


def _clean(s, limit=1200):
    s = re.sub(r"\s+\n", "\n", (s or "").strip())
    return s[:limit]


def load_dolly(path):
    by = {}
    if not os.path.exists(path):
        return by
    for line in open(path, errors="ignore"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        by.setdefault(r.get("category", "?"), []).append(r)
    return by


def load_parquet(path, cols=None):
    if not os.path.exists(path):
        return []
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=cols)
    return t.to_pylist()


def load_gsm8k():
    """GSM8K from the local HF arrow cache -- no network, and it is the session-1 domain."""
    import glob
    import pyarrow as pa
    base = os.path.expanduser("~/.cache/huggingface/datasets/openai___gsm8k")
    out = []
    for f in sorted(glob.glob(os.path.join(base, "**", "*train*.arrow"), recursive=True)):
        try:
            with pa.memory_map(f, "rb") as src:
                for b in pa.ipc.open_stream(src):
                    for r in b.to_pylist():
                        if r.get("question"):
                            out.append(r["question"])
        except (pa.ArrowInvalid, OSError):
            continue
    return out


def load_bfcl():
    """BFCL lives as content-addressed blobs; find the ones that parse as its question jsonl."""
    import glob
    out = []
    base = os.path.expanduser("~/.cache/huggingface/hub/"
                              "datasets--gorilla-llm--Berkeley-Function-Calling-Leaderboard/blobs")
    for f in glob.glob(os.path.join(base, "*")):
        try:
            with open(f, errors="ignore") as fh:
                head = fh.readline()
                if '"question"' not in head:
                    continue
                fh.seek(0)
                for line in fh:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    q, fn = r.get("question"), r.get("function")
                    while isinstance(q, list) and q:
                        q = q[0]
                    if isinstance(q, dict):
                        q = q.get("content")
                    if not q or not fn:
                        continue
                    fns = fn if isinstance(fn, list) else [fn]
                    names = ", ".join(str(x.get("name")) for x in fns[:3] if isinstance(x, dict))
                    if not names:
                        continue
                    out.append(f"You have these tools available: {names}.\n\n{q}\n\n"
                               f"State exactly which tool you would call and with what arguments.")
        except OSError:
            continue
    return out


MUTATIONS = [("+", "-"), ("-", "+"), ("<=", "<"), (">=", ">"), ("==", "!="),
             ("range(1,", "range(0,"), ("len(", "len(")]


def make_code_edit(rows, rnd):
    """A real MBPP solution with ONE operator flipped, framed as a bug report.

    The mutation is applied to the actual shipped solution rather than to invented code, and the
    prompt says what the function should do, so the task is well posed. Entries where no mutation
    applies are skipped rather than emitted unmutated -- an unmutated 'fix this bug' prompt is a
    trap that teaches the model the wrong thing.
    """
    out = []
    for r in rows:
        code, text = r.get("code"), r.get("text")
        if not code or not text:
            continue
        rnd.shuffle(MUTATIONS)
        for a, b in MUTATIONS:
            if a in code and a != b:
                out.append(f"This function is supposed to {text[0].lower() + text[1:]}\n"
                           f"It has a bug:\n\n{code.replace(a, b, 1)}\n\nFix it. Return only the "
                           f"corrected function.")
                break
    return out


def make_multi_turn(rows, rnd):
    out = []
    for r in rows:
        msgs = r.get("messages") or []
        if len(msgs) < 3:
            continue
        u1, a1, u2 = msgs[0].get("content"), msgs[1].get("content"), msgs[2].get("content")
        if not (u1 and a1 and u2):
            continue
        out.append(f"User: {_clean(u1, 400)}\nAssistant: {_clean(a1, 400)}\nUser: {_clean(u2, 300)}\n"
                   f"Assistant:")
    return out


def build(src, rnd):
    """-> {category: [prompt text, ...]}, each already shuffled."""
    pools = {c: [] for c in CATS}
    dolly = load_dolly(os.path.join(src, "dolly.jsonl"))

    for c in ("closed_qa", "open_qa"):
        for r in dolly.get(c, []):
            q = _clean(r.get("instruction"), 300)
            if q and len(q) > 15:
                pools["short_factual"].append(q)
    for c in ("general_qa", "brainstorming", "creative_writing"):
        for r in dolly.get(c, []):
            q = _clean(r.get("instruction"), 300)
            if q and len(q) > 25:
                pools["explanation"].append(q)
    # long_context: entries whose `context` is genuinely long, question asked AFTER the passage
    for c in ("closed_qa", "information_extraction", "summarization"):
        for r in dolly.get(c, []):
            ctx, q = _clean(r.get("context"), 6000), _clean(r.get("instruction"), 300)
            if ctx and q and len(ctx) > 1500:
                pools["long_context"].append(f"{ctx}\n\nBased on the passage above: {q}")

    pools["reasoning"] = [_clean(q, 600) for q in load_gsm8k()]
    mbpp = load_parquet(os.path.join(src, "mbpp.parquet"))
    pools["code_gen"] = [f"{_clean(r['text'], 400)} Write the Python function. Code only."
                         for r in mbpp if r.get("text")]
    pools["code_edit"] = make_code_edit(mbpp, rnd)
    uc = load_parquet(os.path.join(src, "ultrachat.parquet"), cols=["messages"])
    pools["multi_turn"] = make_multi_turn(uc, rnd)
    pools["agentic_format"] = load_bfcl()

    for c in pools:
        # de-duplicate before shuffling: repeats inside one category silently reweight the mix
        seen, uniq = set(), []
        for p in pools[c]:
            k = hashlib.sha1(p.encode()).hexdigest()
            if k not in seen:
                seen.add(k)
                uniq.append(p)
        rnd.shuffle(uniq)
        pools[c] = uniq
    return pools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=536, help="total prompts (train + hold-out)")
    ap.add_argument("--holdout", type=int, default=64)
    ap.add_argument("--src", default="/home/patrickd/s5-capture/corpus_src")
    ap.add_argument("--tokenizer",
                    default="/home/patrickd/models/DeepSeek-V4-Flash-0731-REAP/tokenizer.json")
    ap.add_argument("--max-tokens", type=int, default=160,
                    help="cap for ordinary prompts; long_context is allowed 8x this")
    ap.add_argument("--min-tokens", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260810)
    a = ap.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer)
    got = tok.encode(CANON_TEXT, add_special_tokens=False).ids
    if got != CANON_IDS:
        sys.exit(f"[corpus] TOKENIZER GATE FAILED: {CANON_TEXT!r} -> {got}, expected {CANON_IDS}. "
                 f"Refusing to emit ids that do not come from the checkpoint's own tokenizer.")
    print(f"[corpus] tokenizer gate OK (vocab {tok.get_vocab_size()})")

    rnd = random.Random(a.seed)
    pools = build(a.src, rnd)
    print("[corpus] available per category:")
    for c in CATS:
        print(f"    {c:16} {len(pools[c]):>6}")
    empty = [c for c in CATS if not pools[c]]
    if empty:
        sys.exit(f"[corpus] no prompts for {empty} -- fix the source before building a corpus that "
                 f"silently omits a category the frozen suite measures")

    per_hold = a.holdout // len(CATS)
    per_train = (a.n - a.holdout) // len(CATS)
    need = per_train + per_hold
    short = [c for c in CATS if len(pools[c]) < need]
    if short:
        sys.exit(f"[corpus] need {need}/category, short on {[(c, len(pools[c])) for c in short]}")

    def enc(text, cat):
        cap = a.max_tokens * 8 if cat == "long_context" else a.max_tokens
        ids = tok.encode(text, add_special_tokens=False).ids[:cap]
        return [BOS] + ids if len(ids) >= a.min_tokens else None

    train, hold, counts = [], [], {c: 0 for c in CATS}
    for c in CATS:
        taken = []
        for text in pools[c]:
            e = enc(text, c)
            if e:
                taken.append(e)
            if len(taken) >= need:
                break
        if len(taken) < need:
            sys.exit(f"[corpus] {c}: only {len(taken)}/{need} survived length filtering")
        train += taken[:per_train]
        hold += taken[per_train:need]
        counts[c] = len(taken[:need])

    rnd.shuffle(train)
    rnd.shuffle(hold)          # hold-out balanced by construction, order randomised
    allp = train + hold        # s5_session.sh reserves the LAST --holdout lines

    with open(a.out, "w") as f:
        for ids in allp:
            f.write(",".join(str(i) for i in ids) + "\n")
    lens = [len(x) for x in allp]
    print(f"[corpus] wrote {len(allp)} prompts -> {a.out}")
    print(f"[corpus]   train {len(train)} ({per_train}/category), hold-out {len(hold)} "
          f"({per_hold}/category, balanced -- session 1's was reasoning-only)")
    print(f"[corpus]   prompt tokens: min {min(lens)} median {sorted(lens)[len(lens)//2]} "
          f"max {max(lens)}")


if __name__ == "__main__":
    main()
