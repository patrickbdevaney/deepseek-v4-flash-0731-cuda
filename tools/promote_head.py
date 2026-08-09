#!/usr/bin/env python3
"""promote_head.py — turn a trained draft head into a PRESERVED, PUBLISHABLE artifact.

WHY THIS EXISTS. The product of this project is two things: the CUDA engine, and the speculator
weights. The engine is in git and therefore safe. **A trained head is not.** Without this tool a
winning head is an untracked `.safetensors` in a scratch directory, with its measurement in a log
that will be overwritten and its lineage in nobody's head. That is how you lose the actual product
and keep only the thing that was easy to keep.

So: a head is only PROMOTED if it can prove it deserves to be, and promotion writes down everything
needed to reproduce, verify and publish it.

THE SELECTION RULE IS FIXED HERE, BEFORE ANY HEAD EXISTS, so it cannot be fitted to a result:

  1. `LOSSLESS GATE ... PASS` must appear in the eval log. A head that changes the emitted
     sequence is disqualified no matter what tau it reaches. (F68 is why: a +28% speedup that
     passed every other gate and was decoding a different sequence.)
  2. `first decoded token argmax = 11111 -> GATE PASS` must appear.
  3. The eval must be the FROZEN protocol: 8-prompt suite, NGEN0 >= 200, block 6, adaptK 1.50,
     clean run. F92 measured tau at 1.39 over the first 32 generated tokens rising to ~3.2 after
     ~128, so a short-generation number is a transient and is not admissible.
  4. Suite-mean tau must EXCEED the incumbent's by more than the measured run-to-run spread
     (3.5%, F94). Ties go to the incumbent -- a head that is not clearly better is not better.

Everything else is bookkeeping, and the bookkeeping is the point.

  promote  : python3 tools/promote_head.py promote --head <dir> --eval <log> --name <tag> [--notes ...]
  list     : python3 tools/promote_head.py list
"""
import argparse, hashlib, json, os, re, shutil, subprocess, sys, time

REGISTRY = "HEAD_REGISTRY.md"
STORE    = os.path.expanduser("~/model-backups/heads")
NOISE    = 0.035          # measured cross-run spread on identical config (F94)


def sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                return h.hexdigest()
            h.update(b)


def parse_eval(log):
    """Pull the admissible numbers out of an eval log. Returns None if the log is not admissible."""
    t = open(log, errors="ignore").read()
    out = {"lossless": "LOSSLESS GATE" in t and "LOSSLESS GATE: first 8 tokens match base AR -> PASS" in t,
           "gate": "-> GATE PASS" in t, "gate_fail": "GATE FAIL" in t}
    rows = re.findall(r"^\[blksweep\]\s+(\d+)\s+\d+\s+[\d.]+\s+[\d.]+\s+(\d+)\s+\|\s+([\d.]+)\s+\|"
                      r"\s+([\d.]+)\s+\|\s+([\d.]+)", t, re.M)
    out["points"] = [{"block": int(b), "prompt": int(p), "tau": float(ta),
                      "ms_tok": float(ms), "tok_s": float(ts)} for b, p, ta, ms, ts in rows]
    # prompt 0 is the canonical CONTROL and is excluded from the mean (F96: it is the worst case
    # and it regresses at block 6; quoting it as the headline is how this project misread itself).
    real = [r for r in out["points"] if r["prompt"] != 0]
    out["n_suite"] = len(real)
    out["suite_tau"]   = round(sum(r["tau"]   for r in real) / len(real), 4) if real else None
    out["suite_tok_s"] = round(sum(r["tok_s"] for r in real) / len(real), 4) if real else None
    m = re.search(r"WARM decode: [\d.]+ ms/tok = ([\d.]+) tok/s", t)
    out["base_ar_tok_s"] = float(m.group(1)) if m else None
    out["blocks"] = sorted({r["block"] for r in out["points"]})
    inst = [k for k in ("[dprof]", "[specprof]", "[ksweep]", "[moebytes]") if k in t]
    out["instruments_present"] = inst
    return out


def incumbent():
    if not os.path.exists(REGISTRY):
        return None
    best = None
    for line in open(REGISTRY):
        m = re.match(r"\|\s*`([^`]+)`\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|.*\|\s*(PROMOTED|baseline)\s*\|", line)
        if m and (best is None or float(m.group(3)) > best["suite_tok_s"]):
            best = {"name": m.group(1), "suite_tau": float(m.group(2)), "suite_tok_s": float(m.group(3))}
    return best



def archive(a):
    """UNCONDITIONAL save. Runs BEFORE the promotion gate and regardless of its verdict.

    Promotion is a judgement; archiving is preservation, and conflating them loses data. A head that
    misses the bar still carries information -- it is a measured point on the acceptance-vs-corpus
    curve, and session 3's size is chosen from exactly those points. Losing a failed head means
    re-running its session to recover a number we already paid for.
    """
    dst = os.path.join(STORE, a.name)
    os.makedirs(dst, exist_ok=True)
    files = []
    for fn in sorted(os.listdir(a.head)):
        src = os.path.join(a.head, fn)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(dst, fn))
            files.append({"file": fn, "bytes": os.path.getsize(src), "sha256": sha256(src)})
    ev = parse_eval(a.eval) if a.eval and os.path.exists(a.eval) else {}
    if a.eval and os.path.exists(a.eval):
        shutil.copy2(a.eval, os.path.join(dst, "eval.log"))
    rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    card = {"name": a.name, "archived_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "engine_git_rev": rev, "promoted": False, "measurement": ev,
            "train_metrics_file": a.metrics, "notes": a.notes or "", "files": files}
    if a.metrics and os.path.exists(a.metrics):
        shutil.copy2(a.metrics, os.path.join(dst, "train_metrics.json"))
        try: card["train_metrics"] = json.load(open(a.metrics))
        except Exception: pass
    json.dump(card, open(os.path.join(dst, "head_card.json"), "w"), indent=2)
    print(f"ARCHIVED {a.name} -> {dst} ({len(files)} files, sha256 recorded)")
    if ev.get("suite_tok_s"):
        print(f"  suite mean: tau {ev['suite_tau']}  {ev['suite_tok_s']} tok/s")
    return 0

def promote(a):
    ev = parse_eval(a.eval)
    inc = incumbent()
    fails = []
    if not ev["lossless"]:            fails.append("no passing LOSSLESS gate in the eval log")
    if ev["gate_fail"]:               fails.append("eval log contains GATE FAIL")
    if not ev["gate"]:                fails.append("no 'GATE PASS' first-token check")
    if ev["instruments_present"]:     fails.append(f"eval was NOT clean: {ev['instruments_present']}")
    if ev["n_suite"] < 8:             fails.append(f"suite has {ev['n_suite']} real prompts, need 8")
    if ev["blocks"] != [6]:           fails.append(f"eval blocks {ev['blocks']}, protocol is block 6")
    if ev["suite_tok_s"] is None:     fails.append("no suite points parsed")
    if inc and ev["suite_tok_s"] is not None:
        need = inc["suite_tok_s"] * (1 + NOISE)
        if ev["suite_tok_s"] <= need:
            fails.append(f"suite {ev['suite_tok_s']:.2f} tok/s does not beat incumbent "
                         f"{inc['suite_tok_s']:.2f} by the {100*NOISE:.1f}% run-to-run spread "
                         f"(needs > {need:.2f}); ties go to the incumbent")
    if fails:
        print("REFUSED to promote:")
        for f in fails:
            print(f"  - {f}")
        # Record the refusal in the registry too. A candidate that was measured and rejected is a
        # data point; leaving it out of the ledger makes the record look like only winners existed.
        if os.path.exists(REGISTRY) and ev.get("suite_tok_s"):
            rev = subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True).stdout.strip()
            with open(REGISTRY, "a") as f:
                f.write(f"| `{a.name}` | {ev['suite_tau']} | {ev['suite_tok_s']} | "
                        f"{ev['base_ar_tok_s']} | `{rev[:9]}` | not promoted: {fails[0][:60]} |\n")
        return 1

    dst = os.path.join(STORE, a.name)
    os.makedirs(dst, exist_ok=True)
    files = []
    for fn in sorted(os.listdir(a.head)):
        src = os.path.join(a.head, fn)
        if not os.path.isfile(src):
            continue
        shutil.copy2(src, os.path.join(dst, fn))
        files.append({"file": fn, "bytes": os.path.getsize(src), "sha256": sha256(src)})
    shutil.copy2(a.eval, os.path.join(dst, "eval.log"))

    rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                text=True).stdout.strip())
    card = {
        "name": a.name, "promoted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "engine_git_rev": rev, "engine_tree_dirty": dirty,
        "base_model": "0xSero/DeepSeek-V4-Flash-0731-REAP",
        "base_source_model": "deepseek-ai/DeepSeek-V4-Flash-0731",
        "base_source_revision": "9e165c30e2704aec5d9d593cce3eebd58bbef1cb",
        "reap": {"original_experts_per_layer": 256, "kept_experts_per_layer": 160},
        "measurement": {
            "protocol": "8-prompt suite (scripts/prompt_suite.json, one per category), NGEN0>=200 "
                        "so tau is past the drafter's 128-token sliding window (F92); block 6, "
                        "adaptK 1.50; clean run; canonical 6-token prompt is a CONTROL, excluded "
                        "from the mean (F96).",
            "suite_mean_tau": ev["suite_tau"], "suite_mean_tok_s": ev["suite_tok_s"],
            "base_ar_tok_s": ev["base_ar_tok_s"], "n_suite_prompts": ev["n_suite"],
            "per_prompt": ev["points"], "lossless_gate": "PASS",
        },
        "incumbent_at_promotion": inc, "notes": a.notes or "",
        "files": files,
    }
    json.dump(card, open(os.path.join(dst, "head_card.json"), "w"), indent=2)

    new = not os.path.exists(REGISTRY)
    with open(REGISTRY, "a") as f:
        if new:
            f.write("# Draft-head registry\n\nEvery candidate head, its measurement, and whether it "
                    "was promoted. Written by `tools/promote_head.py`, which refuses to promote a "
                    "head that cannot pass the gates in its docstring.\n\n"
                    "| name | suite tau | suite tok/s | base AR | engine rev | status |\n"
                    "|---|---|---|---|---|---|\n")
        f.write(f"| `{a.name}` | {ev['suite_tau']} | {ev['suite_tok_s']} | "
                f"{ev['base_ar_tok_s']} | `{rev[:9]}` | PROMOTED |\n")
    print(f"PROMOTED {a.name}")
    print(f"  suite mean: tau {ev['suite_tau']}  {ev['suite_tok_s']} tok/s  (base AR {ev['base_ar_tok_s']})")
    if inc:
        print(f"  beats incumbent {inc['name']}: {inc['suite_tok_s']:.2f} -> {ev['suite_tok_s']:.2f} tok/s "
              f"({100*(ev['suite_tok_s']-inc['suite_tok_s'])/inc['suite_tok_s']:+.1f}%)")
    print(f"  artifact: {dst}  ({len(files)} files, sha256 recorded)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("promote"); p.add_argument("--head", required=True)
    p.add_argument("--eval", required=True); p.add_argument("--name", required=True)
    p.add_argument("--notes", default=""); p.add_argument("--metrics", default=None)
    q = sub.add_parser("archive"); q.add_argument("--head", required=True)
    q.add_argument("--eval", default=None); q.add_argument("--name", required=True)
    q.add_argument("--notes", default=""); q.add_argument("--metrics", default=None)
    sub.add_parser("list")
    a = ap.parse_args()
    if a.cmd == "archive":
        sys.exit(archive(a))
    if a.cmd == "list":
        print(open(REGISTRY).read() if os.path.exists(REGISTRY) else "(no registry yet)")
        sys.exit(0)
    sys.exit(promote(a))
