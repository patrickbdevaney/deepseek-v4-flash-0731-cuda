#!/usr/bin/env python3
"""verify_head_archive.py — prove the draft-head archive is intact and complete.

WHY THIS EXISTS. The product of this project is two things: the CUDA engine, which is in git and
therefore safe, and the speculator weights, which are NOT in git and therefore at risk.
`tools/verify_staged_ckpt.py` proves the LIVE farm is a faithful one-head swap of the base
checkpoint. Nothing proved the ARCHIVE those symlinks point into is still there and still correct --
and the live farm links straight into `model-backups/heads/<name>/`, so a damaged archive is a
damaged server, discovered at the next load instead of now.

It also answers the question the registry cannot: **is every head named in HEAD_REGISTRY.md actually
backed up?** A row with no directory is a head that exists only as a number in a table.

WHAT IT CHECKS

  heads/<name>/     against head_card.json's `files` list      (size, and sha256 under --full)
  releases/<name>/  against SHA256SUMS                          (size, and sha256 under --full)
  HEAD_REGISTRY.md  every named head has a directory            (completeness)
  reverse           every directory is named in the registry    (orphans)

A head archived as `mtp_trained.safetensors` ONLY is complete by design, not deficient: the loadable
shards are regenerated from it by `tools/build_trained_head.py`, deterministically and self-checked,
and skipping them saves ~7 GB per head. That case is reported as `SOURCE-ONLY`, not as a failure.

DEFAULT IS --quick (stat only). The full pass hashes ~30 GB and will contend with any benchmark
running on this box; hashing during a measurement is how you turn a clean A/B into a mystery.
Run --full only when the decode loop is idle.

  python3 tools/verify_head_archive.py                 # size + presence, seconds
  python3 tools/verify_head_archive.py --full          # + sha256, minutes, loop must be idle
  python3 tools/verify_head_archive.py --json out.json # machine-readable

Exit 0 iff every check passed.
"""
import argparse, hashlib, json, os, re, sys

HEADS = os.path.expanduser("~/model-backups/heads")
RELEASES = os.path.expanduser("~/model-backups/releases")
REGISTRY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "HEAD_REGISTRY.md")
# The source of truth a head can be rebuilt from. Its presence is what makes a directory a real
# backup rather than a copy of derived artifacts.
SOURCE = "mtp_trained.safetensors"


def sha256(path, buf=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def check_file(path, want_bytes, want_sha, full):
    """Return (ok, note). want_bytes/want_sha may be None when the manifest omits them."""
    if not os.path.exists(path):
        return False, "MISSING"
    got = os.path.getsize(path)
    if want_bytes is not None and got != want_bytes:
        return False, "SIZE %d != %d" % (got, want_bytes)
    if full and want_sha:
        if sha256(path) != want_sha:
            return False, "SHA256 MISMATCH"
    return True, "ok"


def check_head(d, full):
    """A head backup, verified against its own head_card.json."""
    name = os.path.basename(d)
    r = {"name": name, "kind": "head", "dir": d, "files": [], "problems": []}
    card_p = os.path.join(d, "head_card.json")
    if not os.path.exists(card_p):
        r["problems"].append("no head_card.json — nothing states what this directory should contain")
        return r
    try:
        card = json.load(open(card_p))
    except Exception as e:
        r["problems"].append("head_card.json unreadable: %s" % e)
        return r

    r["promoted"] = card.get("promoted")
    files = card.get("files") or []
    if not files:
        # Ablation heads are archived source-only and their cards carry no file list. That is a
        # deliberate 7 GB saving, so verify what IS there rather than inventing a requirement.
        files = [{"file": f} for f in sorted(os.listdir(d)) if f != "head_card.json"]
        r["manifest"] = "implicit (card has no files list)"
    else:
        r["manifest"] = "head_card.json"

    for f in files:
        fn = f["file"]
        ok, note = check_file(os.path.join(d, fn), f.get("bytes"), f.get("sha256"), full)
        r["files"].append({"file": fn, "ok": ok, "note": note})
        if not ok:
            r["problems"].append("%s: %s" % (fn, note))

    have = set(os.listdir(d))
    if SOURCE in have:
        r["rebuildable"] = True
        r["shape"] = "FULL" if any(x.startswith("model-000") for x in have) else "SOURCE-ONLY"
    else:
        r["rebuildable"] = False
        r["shape"] = "DERIVED-ONLY"
        # The stock head is the base checkpoint's own and was never trained, so it has no source
        # tensors and needs none: the base checkpoint is its source.
        if not card.get("provenance"):
            r["problems"].append(
                "no %s — this head cannot be rebuilt if its shards are damaged" % SOURCE)
    return r


def check_release(d, full):
    """A release bundle, verified against its SHA256SUMS."""
    name = os.path.basename(d)
    r = {"name": name, "kind": "release", "dir": d, "files": [], "problems": []}
    sums = os.path.join(d, "SHA256SUMS")
    if not os.path.exists(sums):
        r["problems"].append("no SHA256SUMS — an unverifiable bundle must not be published")
        return r
    for line in open(sums):
        line = line.strip()
        if not line:
            continue
        m = re.match(r"([0-9a-f]{64})\s+(.+)$", line)
        if not m:
            r["problems"].append("unparseable SHA256SUMS line: %s" % line[:60])
            continue
        want, fn = m.group(1), m.group(2)
        ok, note = check_file(os.path.join(d, fn), None, want, full)
        r["files"].append({"file": fn, "ok": ok, "note": note})
        if not ok:
            r["problems"].append("%s: %s" % (fn, note))
    for req in ("README.md", "provenance.json"):
        if not os.path.exists(os.path.join(d, req)):
            r["problems"].append("no %s — required for an uploadable bundle" % req)
    return r


def registry_names():
    """Head names the registry claims exist, from the first column of its RESULTS table.

    NOT every backticked first cell. `HEAD_REGISTRY.md` also carries prose tables -- the corpus
    inventory (`s3/gen.txt`, `s1_gen.txt`) and the release pointer (`CURRENT_BEST`) -- whose first
    cells look identical to a head row. Reporting those as missing directories produced three
    permanent false alarms, and a checker that always cries wolf is a checker nobody reads.

    A results row is identified structurally, by the shape only it has: >= 7 pipe-delimited cells
    with a parseable `tau` in the second. That survives new columns being appended (the `blk`
    column was added on 2026-08-21) without needing to be told about them.
    """
    if not os.path.exists(REGISTRY):
        return None
    names = []
    for line in open(REGISTRY):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 7:
            continue
        m = re.match(r"^`([^`]+)`$", cells[0])
        if not m:
            continue
        try:
            float(cells[1])
        except ValueError:
            continue
        names.append(m.group(1))
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="hash every file (~30 GB); run only when the decode loop is idle")
    ap.add_argument("--json", help="write the full result here")
    a = ap.parse_args()

    results = []
    for root, fn in ((HEADS, check_head), (RELEASES, check_release)):
        if not os.path.isdir(root):
            print("MISSING ROOT: %s" % root)
            return 2
        for name in sorted(os.listdir(root)):
            d = os.path.join(root, name)
            if os.path.isdir(d):
                results.append(fn(d, a.full))

    mode = "sha256 + size" if a.full else "size + presence (use --full to hash)"
    print("verify_head_archive — %s\n" % mode)

    w = max(len(r["name"]) for r in results) + 2
    for r in results:
        state = "FAIL" if r["problems"] else "PASS"
        extra = r.get("shape", "")
        if r["kind"] == "head" and r.get("promoted"):
            extra += " PROMOTED"
        print("  %-4s %-*s %2d files  %s" % (state, w, r["name"], len(r["files"]), extra))
        for p in r["problems"]:
            print("         ! %s" % p)

    # Completeness: the registry and the archive must agree in both directions.
    reg = registry_names()
    gaps = []
    if reg is not None:
        have = {r["name"] for r in results if r["kind"] == "head"}
        for n in reg:
            if n not in have:
                gaps.append("registry names `%s` but %s/%s does not exist" % (n, HEADS, n))
        for n in sorted(have - set(reg)):
            gaps.append("%s/%s exists but no registry row names it" % (HEADS, n))
    else:
        gaps.append("HEAD_REGISTRY.md not found — completeness not checked")

    print()
    if gaps:
        print("COMPLETENESS:")
        for g in gaps:
            print("  ! %s" % g)
    else:
        print("COMPLETENESS: every registry row has a directory, and every directory has a row.")

    bad = [r for r in results if r["problems"]]
    nfile = sum(len(r["files"]) for r in results)
    print("\n%d directories, %d files checked, %d directories with problems, %d completeness gaps"
          % (len(results), nfile, len(bad), len(gaps)))

    if a.json:
        json.dump({"mode": mode, "results": results, "completeness_gaps": gaps},
                  open(a.json, "w"), indent=2)
        print("wrote %s" % a.json)

    return 1 if (bad or gaps) else 0


if __name__ == "__main__":
    sys.exit(main())
