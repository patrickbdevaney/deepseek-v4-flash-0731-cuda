#!/usr/bin/env python3
"""make_bundle.py — build the one self-contained directory to publish.

The product of this project is two things: the CUDA engine (in git, therefore safe) and the
speculator weights (not in git, therefore at risk). A user who has the GitHub repo still needs the
weights, and a user who has the weights needs to know which head is which and how to load either.
This builds the single directory that answers both.

WHAT IT CONTAINS AND WHY.

    default-s3/    the promoted head -- 25.53 tok/s suite, +12.7 % over stock
    base-stock/    the checkpoint's ORIGINAL MTP tensors, same 72 names, so the fine-tune is
                   reversible and comparable. Without this a user cannot A/B what we changed, and
                   cannot get back to the shipped behaviour except by re-downloading 108 GB.
    tools/         build_trained_head.py, so the bundle is self-contained rather than a pointer

Both heads ship as the ~1 GB of MTP tensors, not as 7 GB materialised heads. Every build log says
`copied 2905 untouched (experts byte-for-byte)`: ~6 GB of a head is the base checkpoint's own
weights copied verbatim, and redistributing those inside an artifact that adds nothing to them is
the wrong shape. `build_trained_head.py` reconstitutes either head deterministically in ~1 min.

  python3 tools/make_bundle.py --out <dir>
"""
import argparse, hashlib, json, os, shutil, struct, subprocess, sys

CK = '/home/patrickd/models/DeepSeek-V4-Flash-0731-REAP'
STORE = os.path.expanduser('~/model-backups/heads')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha256(p, buf=1 << 20):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(buf), b''):
            h.update(b)
    return h.hexdigest()


def st_header(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def extract_base_mtp(names, out_path):
    """Write the checkpoint's ORIGINAL tensors for `names` into one safetensors file.

    Raw bytes and original dtypes are preserved, so this is a byte-exact record of what the
    fine-tune replaced -- not a re-quantised approximation of it.
    """
    idx = json.load(open(os.path.join(CK, 'model.safetensors.index.json')))['weight_map']
    hdrs, blobs, off = {}, [], 0
    for n in sorted(names):
        if n not in idx:
            raise SystemExit(f'base tensor missing from checkpoint: {n}')
        p = os.path.join(CK, idx[n])
        h, base = st_header(p)
        m = h[n]
        o0, o1 = m['data_offsets']
        with open(p, 'rb') as f:
            f.seek(base + o0)
            raw = f.read(o1 - o0)
        hdrs[n] = {'dtype': m['dtype'], 'shape': m['shape'], 'data_offsets': [off, off + len(raw)]}
        off += len(raw)
        blobs.append(raw)
    hdrs['__metadata__'] = {
        'source': 'DeepSeek-V4-Flash-0731-REAP, as shipped',
        'what': 'the ORIGINAL MTP draft-head tensors this project fine-tuned, for reversibility',
        'tensors': str(len(names))}
    hj = json.dumps(hdrs).encode()
    with open(out_path, 'wb') as f:
        f.write(struct.pack('<Q', len(hj)))
        f.write(hj)
        for b in blobs:
            f.write(b)
    return off


README = """---
base_model: 0xSero/DeepSeek-V4-Flash-0731-REAP
tags: [speculative-decoding, draft-model, mtp, jetson]
library_name: safetensors
---

# DSpark MTP draft heads for `DeepSeek-V4-Flash-0731-REAP`

**Not a standalone model.** These are **MTP draft-head tensors** — the speculator for self-speculative
decoding against `0xSero/DeepSeek-V4-Flash-0731-REAP`. They do nothing without that checkpoint.

Engine: **https://github.com/{repo_slug}** — a from-scratch pure-CUDA inference server for this
checkpoint, hand-tuned for Jetson AGX Thor (`sm_110a`).

## What is in here

| directory | head | suite decode |
|---|---|---|
| `default-s3/` | **fine-tuned, the one to use** | **{s3_tps} tok/s** |
| `base-stock/` | the checkpoint's original MTP, byte-exact | {base_tps} tok/s |

`base-stock/` exists so the fine-tune is **reversible and comparable** — you can A/B what changed
without re-downloading 108 GB.

## Why it is lossless

Verification is **greedy and exact**: the draft proposes, the base model verifies in one pass, and
the longest correct prefix is accepted. **The emitted token sequence is identical to what the base
model would have produced alone.** This is latency, not a quality trade. Every head here passed
`LOSSLESS GATE: first 8 tokens match base AR` before it was measured; a head that changes the output
is rejected at any speed.

## Use

```bash
git clone https://github.com/{repo_slug} && cd {repo_dir}
python3 tools/build_trained_head.py \\
    --base /path/to/DeepSeek-V4-Flash-0731-REAP \\
    --trained /path/to/default-s3/mtp_trained.safetensors \\
    --out /path/to/head
bash scripts/build_decode.sh
scripts/run_model.sh out.log ./build/decode /path/to/DeepSeek-V4-Flash-0731-REAP \\
    "0,671,6102,294,8760,344" 8 /path/to/head 200
```

The build substitutes these tensors into the checkpoint and re-quantises them to the formats it uses
(fp8 e4m3 with F8_E8M0 128x128 block scales; bf16 and fp32 for the rest). It is deterministic,
recomputes each block scale from the trained values, snaps it to an exact power of two *before*
quantising (E8M0 stores a bare exponent), and **refuses any tensor whose round-trip error exceeds
0.10**. Worst observed for `default-s3`: **0.0014**.

**To run the stock speculator instead, pass no head at all** — the engine then uses the checkpoint's
own MTP blocks:

```bash
scripts/run_model.sh out.log ./build/decode /path/to/DeepSeek-V4-Flash-0731-REAP \\
    "0,671,6102,294,8760,344" 8 "" 200
```

That is the exact command every paired control in this project was measured with. `base-stock/` is a
**byte-exact record** of the original tensors (weights *and* their E8M0 block scales) so you can diff
what the fine-tune changed — it is not needed to revert, because reverting means using the checkpoint
unmodified.

## Measured

Jetson AGX Thor (`sm_110a`, 20 SMs, 122.8 GiB unified LPDDR5X, CUDA 13.0), block 6, adaptK 1.50,
8-prompt suite at NGEN0 >= 200, clocks pinned, page cache dropped, no profiling instruments.

| | tok/s |
|---|---|
| base autoregressive (no speculation) | 13.8 |
| stock MTP head | {base_tps} |
| **`default-s3`** | **{s3_tps}** |

Acceptance tau (tokens per verify, suite mean, max 7): stock 3.5362 → **`default-s3` {s3_tau}**.

Trained on activations captured from this checkpoint over **1 472 prompt continuations the base
model generated itself**, balanced across 8 task categories, in 3 chunks with weights and AdamW
moments carried across (equivalent to one continuous run).

## Limitations, measured rather than assumed

1. **It regresses on held-out continuation drafting.** Against a true paired control — the stock head
   over the same prompts, budget and threshold — the fine-tuned heads are *slower* at drafting the
   model's own long continuations. For `s3`: tau 6.00 → 4.62. It wins the mixed suite and loses this.
   Both are real; which matters depends on your workload.
2. **Training helps weak categories and hurts strong ones.** Rank-ordered across the suite, the
   categories the stock head was worst at improved most.
3. **Tuned to one engine and one checkpoint.** The head taps specific backbone layers and was
   fine-tuned against activations from this exact checkpoint. Do not expect it to transfer to another
   quantisation, another REAP revision, or another serving stack.

Full history — including the rejected candidates and the corrections — is in `LOOP_LOG.md`,
`S5_PROGRESSION.md` and `HEAD_REGISTRY.md` in the engine repo.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.expanduser('~/model-backups/releases/dspark-mtp-0731reap-bundle'))
    ap.add_argument('--head', default='s3')
    ap.add_argument('--repo-slug', default='<your-org>/deepseek-v4-flash-0731-cuda')
    a = ap.parse_args()

    src = os.path.join(STORE, a.head)
    tr = os.path.join(src, 'mtp_trained.safetensors')
    if not os.path.exists(tr):
        sys.exit(f'{src} has no mtp_trained.safetensors')
    os.makedirs(os.path.join(a.out, 'default-' + a.head), exist_ok=True)
    os.makedirs(os.path.join(a.out, 'base-stock'), exist_ok=True)
    os.makedirs(os.path.join(a.out, 'tools'), exist_ok=True)

    for fn in ('mtp_trained.safetensors', 'head_card.json', 'eval.log', 'train_metrics.json'):
        p = os.path.join(src, fn)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(a.out, 'default-' + a.head, fn))
    print(f'[bundle] default-{a.head}: copied the promoted head')

    hdr, _ = st_header(tr)
    names = [k for k in hdr if k != '__metadata__']
    # INCLUDE THE BLOCK SCALES. The trained file carries 72 weight tensors and no scales, because the
    # trainer's parameter list holds parameters and a scale is a buffer. For the trained head that is
    # correct -- build_trained_head.py RECOMPUTES each scale from the trained values. For a byte-exact
    # record of the original it is not: an fp8 weight without its E8M0 block scale is a table of
    # codes, not a tensor, and anything reconstructed from it alone is wrong by whatever the scale
    # was. First cut of this bundle shipped exactly that, with a README telling people to rebuild
    # from it.
    idx_all = json.load(open(os.path.join(CK, 'model.safetensors.index.json')))['weight_map']
    scales = [n.rsplit('.', 1)[0] + '.scale' for n in names]
    names_full = names + [s for s in scales if s in idx_all]
    nb = extract_base_mtp(names_full, os.path.join(a.out, 'base-stock', 'mtp_base.safetensors'))
    print(f'[bundle] base-stock: {len(names)} original tensors + '
          f'{len(names_full)-len(names)} block scales, {nb/1e6:.0f} MB, byte-exact')
    bl = os.path.join(ROOT, 'evidence', 'baseline_blk6_suite.log')
    if os.path.exists(bl):
        shutil.copy2(bl, os.path.join(a.out, 'base-stock', 'eval.log'))

    shutil.copy2(os.path.join(ROOT, 'tools', 'build_trained_head.py'),
                 os.path.join(a.out, 'tools', 'build_trained_head.py'))

    card = json.load(open(os.path.join(src, 'head_card.json')))
    m = card.get('measurement') or {}
    tps = m.get('suite_tok_s') or m.get('suite_mean_tok_s')
    tau = m.get('suite_tau') or m.get('suite_mean_tau')
    rev = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                         cwd=ROOT).stdout.strip()
    open(os.path.join(a.out, 'README.md'), 'w').write(README.format(
        repo_slug=a.repo_slug, repo_dir=a.repo_slug.split('/')[-1],
        s3_tps=f'{tps:.2f}' if tps else '25.53', base_tps='22.66',
        s3_tau=f'{tau:.4f}' if tau else '3.8438'))

    sums = []
    for root, _, files in os.walk(a.out):
        for fn in sorted(files):
            if fn == 'SHA256SUMS':
                continue
            p = os.path.join(root, fn)
            sums.append(f'{sha256(p)}  {os.path.relpath(p, a.out)}')
    open(os.path.join(a.out, 'SHA256SUMS'), 'w').write('\n'.join(sums) + '\n')

    json.dump({'bundle': os.path.basename(a.out), 'default_head': a.head,
               'base_model': '0xSero/DeepSeek-V4-Flash-0731-REAP',
               'engine_git_rev': rev, 'measurement': m,
               'contents': sorted(os.path.relpath(os.path.join(r, f), a.out)
                                  for r, _, fs in os.walk(a.out) for f in fs)},
              open(os.path.join(a.out, 'provenance.json'), 'w'), indent=2)

    total = sum(os.path.getsize(os.path.join(r, f))
                for r, _, fs in os.walk(a.out) for f in fs)
    print(f'\n[bundle] {a.out}')
    print(f'[bundle] {len(sums)+1} files, {total/1e6:.0f} MB total')
    print(f'[bundle] verify with:  cd {a.out} && sha256sum -c SHA256SUMS')


if __name__ == '__main__':
    main()
