#!/usr/bin/env python3
"""ingest_arms.py — fold every measured block-5 head into the autopilot's search state.

WHY. The search reasons from `evidence/autopilot/arms.jsonl`. Arms run by a hand-launched chain --
which is how every arm before the autopilot existed, and how the P2.5b anchor-shape arms are running
right now -- land in HEAD_REGISTRY.md and nowhere else. A search that cannot see them will re-propose
configurations already measured, and worse, will search around a stale incumbent.

Configuration comes from the head's own `train_metrics.json`, which records the full loss config as
of 2026-08-23. Arms trained before that record only `a_conf`, so their config cannot be recovered
from the artifact and they are skipped with a note rather than guessed at -- a fabricated cfg would
poison the dedup set and silently prevent a real arm from ever being proposed.
"""
import json, os, re, sys

STATE = "evidence/autopilot/arms.jsonl"
HEADS = os.path.expanduser("~/model-backups/heads")
AXES = ("a_ce", "a_tv", "deficit", "beta", "anchor_pow", "deficit_clamp", "pos_per_seq")


def registry_block5():
    out = {}
    for line in open("HEAD_REGISTRY.md"):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 7:
            continue
        m = re.match(r"^`([^`]+)`$", cells[0])
        if not m:
            continue
        try:
            tau = float(cells[1]); blk = int(cells[2])
        except ValueError:
            continue
        if blk == 5:
            out[m.group(1)] = tau
    return out


def main():
    have = set()
    if os.path.exists(STATE):
        for line in open(STATE):
            if line.strip():
                have.add(json.loads(line)["name"])
    added = skipped = 0
    with open(STATE, "a") as f:
        for name, tau in registry_block5().items():
            if name in have:
                continue
            mp = os.path.join(HEADS, name, "train_metrics.json")
            if not os.path.exists(mp):
                print("skip %s: no train_metrics.json" % name); skipped += 1; continue
            try:
                m = json.load(open(mp))
            except Exception as e:
                print("skip %s: unreadable metrics (%s)" % (name, e)); skipped += 1; continue
            if any(k not in m for k in ("beta", "anchor_pow")):
                print("skip %s: metrics predate the full-config record; cfg not recoverable" % name)
                skipped += 1; continue
            cfg = {k: m.get(k) for k in AXES}
            cfg["deficit"] = int(bool(m.get("deficit")))
            rel = os.path.join("evidence", "autopilot", "release_%s.log" % name)
            passed = os.path.exists(rel) and "RELEASE: PASS" in open(rel).read()
            f.write(json.dumps({"name": name, "tau": tau, "release_pass": passed,
                                "cfg": cfg, "note": "ingested from HEAD_REGISTRY.md"}) + "\n")
            print("ingest %s: tau %.4f cfg %s" % (name, tau, cfg)); added += 1
    print("\n%d ingested, %d skipped" % (added, skipped))


if __name__ == "__main__":
    sys.exit(main())
