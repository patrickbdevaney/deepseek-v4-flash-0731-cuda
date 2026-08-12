#!/usr/bin/env python3
"""capability_eval.py — task #11: measure capability on the REAP checkpoint, not on its reputation.

NORTH_STAR.md §6 says it plainly: *"The ~50 Artificial-Analysis-class capability figure is a claim
about the BASE checkpoint."* REAP removed 37.5 % of the experts (256 -> 160). Whether the pruned
checkpoint keeps that capability has never been measured here, and every "frontier + fast" framing
in this project rests on it. This is the smallest honest test that can move that number from
inherited to measured.

GSM8K, because it is the one benchmark on this box whose answers are **exactly scorable**: the gold
answer is the integer after `####`, so there is no judge, no partial credit and no room to grade
generously. It measures multi-step arithmetic reasoning specifically -- it is NOT an
Artificial-Analysis composite, and reporting it as one would repeat exactly the substitution this
task exists to end.

    prompts  -> tokenised with the CHECKPOINT'S OWN tokenizer (never invented ids)
    generate -> the CUDA engine, greedy, the same binary that serves
    score    -> last integer in the completion == the integer after ####

  python3 tools/capability_eval.py --prep --n 60          # write prompts + gold
  python3 tools/capability_eval.py --score <gen.txt>      # after the engine has run
"""
import argparse, glob, json, os, re, sys

CK = '/home/patrickd/models/DeepSeek-V4-Flash-0731-REAP'
OUT = '/home/patrickd/s5-capture/capability'


def load_gsm8k_test(n):
    import pyarrow as pa
    base = os.path.expanduser('~/.cache/huggingface/datasets/openai___gsm8k')
    rows = []
    for f in sorted(glob.glob(os.path.join(base, '**', '*test*.arrow'), recursive=True)):
        with pa.memory_map(f, 'rb') as src:
            for b in pa.ipc.open_stream(src):
                for r in b.to_pylist():
                    if r.get('question') and r.get('answer'):
                        rows.append(r)
    return rows[:n]


GOLD = re.compile(r'####\s*(-?[\d,]+)')
NUM = re.compile(r'-?\d[\d,]*\.?\d*')


def gold_of(ans):
    m = GOLD.search(ans)
    return None if not m else m.group(1).replace(',', '')


def pred_of(text):
    """Last number in the completion. GSM8K convention; deliberately not a lenient parser."""
    xs = NUM.findall(text)
    if not xs:
        return None
    v = xs[-1].replace(',', '').rstrip('.')
    if v.endswith('.0'):
        v = v[:-2]
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prep', action='store_true')
    ap.add_argument('--score', default=None)
    ap.add_argument('--n', type=int, default=60)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    if a.prep:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(os.path.join(CK, 'tokenizer.json'))
        got = tok.encode('The capital of France is', add_special_tokens=False).ids
        if got != [671, 6102, 294, 8760, 344]:
            sys.exit(f'[cap] TOKENIZER GATE FAILED: {got}')
        rows = load_gsm8k_test(a.n)
        gold, lines = [], []
        for r in rows:
            g = gold_of(r['answer'])
            if g is None:
                continue
            # zero-shot with an explicit answer format: the model must commit to a final number,
            # and the scorer takes the LAST number, so a wandering completion scores wrong rather
            # than being rescued by a generous parser.
            p = (f"{r['question'].strip()}\n\n"
                 f"Solve this step by step, then give the final numeric answer on the last line "
                 f"as: #### <number>\n")
            ids = tok.encode(p, add_special_tokens=False).ids
            lines.append(','.join(str(x) for x in [0] + ids))
            gold.append(g)
        open(os.path.join(OUT, 'prompts.txt'), 'w').write('\n'.join(lines) + '\n')
        json.dump(gold, open(os.path.join(OUT, 'gold.json'), 'w'))
        print(f'[cap] tokenizer gate OK; wrote {len(lines)} GSM8K test prompts -> {OUT}/prompts.txt')
        print(f'[cap] median prompt {sorted(len(l.split(","))for l in lines)[len(lines)//2]} tokens')
        return

    if a.score:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(os.path.join(CK, 'tokenizer.json'))
        gold = json.load(open(os.path.join(OUT, 'gold.json')))
        prompts = [l.strip() for l in open(os.path.join(OUT, 'prompts.txt')) if l.strip()]
        gens = [l.strip() for l in open(a.score) if l.strip()]
        n_ok = n_tot = 0
        wrong = []
        for i, g in enumerate(gens):
            if i >= len(gold):
                break
            ids = [int(x) for x in g.split(',') if x]
            plen = len(prompts[i].split(','))
            comp = tok.decode(ids[plen:])           # only the COMPLETION is scored
            p = pred_of(comp)
            n_tot += 1
            if p is not None and p == gold[i]:
                n_ok += 1
            elif len(wrong) < 3:
                wrong.append((i, gold[i], p, comp[-90:].replace('\n', ' ')))
        acc = 100.0 * n_ok / max(n_tot, 1)
        # Wilson interval: n=60 is a small sample and a bare percentage would overstate precision
        import math
        z = 1.96
        ph = n_ok / max(n_tot, 1)
        den = 1 + z*z/n_tot
        c = (ph + z*z/(2*n_tot)) / den
        hw = z*math.sqrt(ph*(1-ph)/n_tot + z*z/(4*n_tot*n_tot))/den
        print(f'\n  GSM8K (test split, zero-shot, greedy, exact match)')
        print(f'  {n_ok}/{n_tot} = {acc:.1f} %   95 % CI [{100*(c-hw):.1f}, {100*(c+hw):.1f}]')
        for i, g, p, t in wrong:
            print(f'    miss #{i}: gold {g}, pred {p} | ...{t}')
        print(f'\n  This is GSM8K only -- multi-step arithmetic. It is NOT an Artificial-Analysis')
        print(f'  composite, and the inherited ~50 figure is not comparable to it. What it can')
        print(f'  settle is whether REAP pruning left the checkpoint able to reason at all.')
        return
    ap.print_help()


if __name__ == '__main__':
    main()
