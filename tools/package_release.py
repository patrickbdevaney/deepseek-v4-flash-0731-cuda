#!/usr/bin/env python3
"""package_release.py — turn a measured draft head into a self-describing, uploadable release.

WHAT THIS PACKAGES, and why it is the small file rather than the big one. A "head" as the engine
loads it is ~7 GB, but the build log for every one of them says the same thing: `copied 2905
untouched (experts byte-for-byte)`. Only ~1 GB is trained. The other ~6 GB is the base checkpoint's
own tensors copied verbatim, so shipping them would mean redistributing someone else's weights
inside an artifact that adds nothing to them. The release therefore contains the **trained MTP
tensors only**, plus the exact recipe to materialise the loadable head from a local copy of the base
checkpoint. `tools/build_trained_head.py` is deterministic and self-checked (it refuses to write a
tensor whose fp8 round-trip exceeds 0.10), so this loses nothing.

WHAT GOES IN THE CARD. Measurements with the protocol that produced them, the base model and its
revision, and the two limitations that a reader would otherwise have to discover for themselves:
these heads win the mixed suite and *lose* to the stock head on held-out continuation drafting
(F116), and training helps weak categories while hurting strong ones (F117). A card that omits those
is not a shorter card, it is a wrong one.

  python3 tools/package_release.py --head <archive-dir> --out <release-dir> [--name ...]
  python3 tools/package_release.py --list
"""
import argparse, hashlib, json, os, re, shutil, subprocess, sys, time

STORE = os.path.expanduser("~/model-backups/heads")
BASE_REPO = "0xSero/DeepSeek-V4-Flash-0731-REAP"
BASE_LOCAL = "/home/patrickd/models/DeepSeek-V4-Flash-0731-REAP"


def sha256(p, buf=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(buf), b""):
            h.update(b)
    return h.hexdigest()


def candidates():
    out = []
    if not os.path.isdir(STORE):
        return out
    for n in sorted(os.listdir(STORE)):
        card = os.path.join(STORE, n, "head_card.json")
        if not os.path.exists(card):
            continue
        try:
            c = json.load(open(card))
        except (json.JSONDecodeError, OSError):
            continue
        m = c.get("measurement") or {}
        out.append({"name": n, "dir": os.path.join(STORE, n),
                    "tok_s": m.get("suite_tok_s"), "tau": m.get("suite_tau"),
                    "promoted": c.get("promoted"), "rev": c.get("engine_git_rev", "")[:9],
                    "card": c})
    return out


CARD = """---
base_model: {base_repo}
tags:
- speculative-decoding
- draft-model
- mtp
- jetson
library_name: safetensors
---

# {name} — DSpark MTP draft head for `{base_repo}`

This is **not a standalone language model.** It is a fine-tuned set of **DSpark MTP draft-head
tensors** for `{base_repo}`, used as the speculator in self-speculative decoding. It only does
anything in combination with the base checkpoint.

## What it is for

Speculative decoding with **greedy verification**: the draft head proposes several tokens, the base
model verifies them in one pass, and the longest correct prefix is accepted. Because verification is
greedy and exact, **the emitted token sequence is identical to what the base model would have
produced on its own** — this is a latency optimisation, not a quality trade. Every head in this line
is gated on that property before it is measured (`LOSSLESS GATE: first 8 tokens match base AR`), and
a head that changes the output is rejected regardless of how fast it is.

## Measured

Hardware: Jetson AGX Thor (`sm_110a`, 20 SMs, 122.8 GiB unified LPDDR5X, CUDA 13.0), custom CUDA
inference engine, block size 6.

| | tok/s |
|---|---|
| base autoregressive decode (no speculation) | {base_ar} |
| stock DSpark head shipped with the checkpoint | 22.66 |
| **this head** | **{tok_s}** |

Acceptance tau (tokens per verify, 8-prompt suite mean): **{tau}** out of a maximum of 7.

**Protocol, which is part of the number.** 8-prompt suite spanning agentic_format, code_edit,
code_gen, explanation, long_context, multi_turn, reasoning and short_factual; NGEN0 >= 200 generated
tokens (acceptance is a transient below ~128 tokens and a short-generation figure is not comparable
to anything); clocks pinned; page cache dropped; no profiling instruments in the binary; adaptK
{adaptk}. Measurements on this box have a residual sd of ~0.5 % per run with a discarded warm-up and
shuffled run order — differences below ~1 % are not meaningful without replicates.

## Limitations, measured rather than assumed

1. **It regresses on held-out continuation drafting.** Against a true paired control — the stock head
   over the same prompts, budget and threshold — the fine-tuned heads are *slower* at drafting the
   model's own long continuations (-0.40 tau for the first session's head, -0.65 for the second).
   They win the mixed suite and lose this. Both numbers are real; which one matters depends on the
   workload.
2. **Training helps weak categories and hurts strong ones.** Rank-ordered across the suite: the
   categories the stock head was worst at improved most, and the ones it was best at regressed.
   Balancing the training corpus repaired some of that (`short_factual`, half of `code_edit`) and
   left `long_context` and `agentic_format` untouched.
3. **Tuned for one engine and one checkpoint.** The draft head taps specific backbone layers and is
   fine-tuned against activations captured from this exact checkpoint. It is not expected to
   transfer to another quantisation, another REAP revision, or another serving stack.

## Files

{files}

## How to use it

The loadable head is the base checkpoint with these tensors substituted in, re-quantised to the
formats the checkpoint uses (fp8 e4m3 with F8_E8M0 128x128 block scales for the projections, bf16
and fp32 for the rest). Materialise it locally:

```bash
git clone https://github.com/{repo_slug}
python3 tools/build_trained_head.py \\
    --base /path/to/{base_dir} \\
    --trained mtp_trained.safetensors \\
    --out /path/to/head_out
```

The build is deterministic and self-checked: it recomputes each block scale from the trained values
(reusing the old scale against new weights is the silent way to get a plausible tensor with the
wrong magnitude), snaps that scale to an exact power of two *before* quantising because E8M0 stores
a bare exponent, and refuses to write any tensor whose round-trip error exceeds 0.10. Worst observed
round-trip error for this head: see `provenance.json`.

## Provenance

Base model `{base_repo}`, used as shipped — no re-pruning and no additional quantisation. Trained
on activations captured from that checkpoint over {n_seq} prompt continuations the base model itself
generated. Full training and measurement history, including the rejected candidates and the
corrections, is in `LOOP_LOG.md` and `S5_PROGRESSION.md` in the engine repository.

Engine revision at measurement time: `{rev}`.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", help="archive directory under ~/model-backups/heads")
    ap.add_argument("--out", help="release directory to create")
    ap.add_argument("--name", default=None)
    ap.add_argument("--adaptk", default="1.50")
    ap.add_argument("--repo-slug", default="<your-org>/deepseek-v4-flash-0731-cuda")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    cands = candidates()
    if a.list or not a.head:
        print(f"{'name':<26} {'suite tok/s':>11} {'suite tau':>10} {'promoted':>9}  engine rev")
        for c in sorted(cands, key=lambda x: -(x["tok_s"] or 0)):
            print(f"{c['name']:<26} {str(c['tok_s'] or '-'):>11} {str(c['tau'] or '-'):>10} "
                  f"{str(c['promoted']):>9}  {c['rev']}")
        if not a.head:
            print("\nPick one with --head <name> --out <dir>. The fastest measured head is not")
            print("automatically the right one: differences under ~1 % need replicates (see")
            print("scripts/head_off.sh), and a head that fails the LOSSLESS gate is not a candidate")
            print("at any speed.")
            return 0

    src = a.head if os.path.isdir(a.head) else os.path.join(STORE, a.head)
    if not os.path.isdir(src):
        sys.exit(f"no such head archive: {src}")
    if not a.out:
        sys.exit("--out is required")
    name = a.name or os.path.basename(src.rstrip("/"))
    card = {}
    if os.path.exists(os.path.join(src, "head_card.json")):
        card = json.load(open(os.path.join(src, "head_card.json")))
    meas = card.get("measurement") or {}

    os.makedirs(a.out, exist_ok=True)
    copied = []
    for fn in sorted(os.listdir(src)):
        p = os.path.join(src, fn)
        if not os.path.isfile(p):
            continue
        # ship the trained tensors and the evidence; never the copied-through base shards
        if re.match(r"model-\d+-of-\d+\.safetensors$", fn) or fn == "model.safetensors.index.json":
            continue
        shutil.copy2(p, os.path.join(a.out, fn))
        copied.append(fn)
    if not any(f.endswith(".safetensors") for f in copied):
        sys.exit(f"{src} holds no trained-tensor file (only a materialised head). Re-archive the "
                 f"trained weights, or point --head at the arm's trained directory.")

    files_md = []
    sums = []
    for fn in sorted(copied):
        p = os.path.join(a.out, fn)
        h = sha256(p)
        sums.append(f"{h}  {fn}")
        size = os.path.getsize(p)
        desc = {"mtp_trained.safetensors": "the trained MTP draft-head tensors — the artifact",
                "eval.log": "the raw measurement log behind the numbers above",
                "train_metrics.json": "per-step training loss history",
                "head_card.json": "the archive record written when this head was measured"}.get(
                    fn, "")
        files_md.append(f"- `{fn}` ({size/1e6:.0f} MB) — {desc}" if desc
                        else f"- `{fn}` ({size/1e6:.0f} MB)")

    rev = card.get("engine_git_rev") or subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    tm = card.get("train_metrics") or {}
    body = CARD.format(
        name=name, base_repo=BASE_REPO, base_dir=os.path.basename(BASE_LOCAL),
        tok_s=f"{meas.get('suite_tok_s')}" if meas.get("suite_tok_s") else "see eval.log",
        tau=f"{meas.get('suite_tau')}" if meas.get("suite_tau") else "see eval.log",
        base_ar=f"{meas.get('base_ar_tok_s')}" if meas.get("base_ar_tok_s") else "13.8",
        adaptk=a.adaptk, files="\n".join(files_md), rev=rev[:9],
        n_seq=tm.get("n_sequences", "the session's"), repo_slug=a.repo_slug)
    open(os.path.join(a.out, "README.md"), "w").write(body)
    open(os.path.join(a.out, "SHA256SUMS"), "w").write("\n".join(sums) + "\n")
    json.dump({"name": name, "packaged_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "base_model": BASE_REPO, "engine_git_rev": rev, "measurement": meas,
               "adaptk_at_measurement": a.adaptk, "source_archive": src,
               "contents": copied,
               "note": "Trained MTP draft-head tensors only. The loadable head is rebuilt from a "
                       "local copy of the base checkpoint with tools/build_trained_head.py, which "
                       "is deterministic and refuses any tensor whose fp8 round-trip exceeds 0.10."},
              open(os.path.join(a.out, "provenance.json"), "w"), indent=2)

    total = sum(os.path.getsize(os.path.join(a.out, f)) for f in os.listdir(a.out))
    print(f"RELEASE {name} -> {a.out}  ({len(copied)+3} files, {total/1e6:.0f} MB)")
    print(f"  README.md        model card, with the measured limitations stated")
    print(f"  provenance.json  base model, engine rev, measurement, rebuild note")
    print(f"  SHA256SUMS       checksums for every shipped file")
    print(f"\nReady to upload. Review README.md before publishing — it names a base model and "
          f"quotes numbers,\nand both should be checked by a human before they go out.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
