#!/usr/bin/env python3
"""verify_staged_ckpt.py -- prove a staged checkpoint is the base checkpoint with exactly one
head swapped in, and nothing else.

WHY THIS EXISTS. `stage_head.sh` builds a directory of symlinks. A symlink farm is cheap and
reversible, which is exactly why it is easy to get subtly wrong: one stale link, one shard whose
tensor list drifted, one dtype that changed between the base and the trained head, and the engine
loads a checkpoint that is neither the base nor the head. That failure does not throw -- it
produces wrong logits, and a draft head producing wrong logits shows up as a tau change, which is
precisely the number the deployment is trying to measure. So the staging is checked BEFORE any
model load, on the CPU, in seconds:

  1. every tensor in the staged index resolves in the file it is mapped to;
  2. every tensor's dtype, shape and byte length equal the base checkpoint's -- drop-in or nothing;
  3. the head tensors come from DIFFERENT bytes than the base (else we staged a no-op and would
     have "measured" the shipped head twice and called it a deployment);
  4. every non-head file is the SAME INODE as the base -- no copies, no drift, and the 100 GiB is
     not duplicated;
  5. the head shards' sha256 match the ones head_card.json recorded at promotion time, so what is
     served is provably the artifact that was measured and archived.

  python3 tools/verify_staged_ckpt.py --staged DIR --base DIR [--head-store DIR] [--prefix mtp.]
"""
import argparse, hashlib, json, os, struct, sys

def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))

def sha256(path, buf=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                return h.hexdigest()
            h.update(b)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staged", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--head-store", default=None, help="archive dir holding head_card.json")
    ap.add_argument("--prefix", default="mtp.", help="tensor-name prefix that identifies the head")
    a = ap.parse_args()

    fails, notes = [], []
    idx = os.path.join(a.staged, "model.safetensors.index.json")
    if not os.path.exists(idx):
        print(f"FAIL: no index at {idx}"); return 1
    wm = json.load(open(idx))["weight_map"]
    base_wm = json.load(open(os.path.join(a.base, "model.safetensors.index.json")))["weight_map"]

    if set(wm) != set(base_wm):
        fails.append(f"staged index has {len(wm)} tensors, base has {len(base_wm)}; sets differ")

    # --- (4) inode identity per file, and which files are swapped
    files = sorted(set(wm.values()))
    swapped, shared, missing = [], [], []
    for fn in files:
        sp, bp = os.path.join(a.staged, fn), os.path.join(a.base, fn)
        if not os.path.exists(sp):
            missing.append(fn); continue
        if not os.path.exists(bp):
            fails.append(f"{fn} not present in base"); continue
        (shared if os.stat(sp).st_ino == os.stat(bp).st_ino else swapped).append(fn)
    if missing:
        fails.append(f"{len(missing)} shard(s) missing from the staged dir: {missing[:3]}")

    # --- (1)(2) header parity, tensor by tensor
    hdr_s, hdr_b, n_ok, n_head = {}, {}, 0, 0
    for name, fn in sorted(wm.items()):
        if fn in missing:
            continue
        if fn not in hdr_s:
            hdr_s[fn] = header(os.path.join(a.staged, fn))
            hdr_b[fn] = header(os.path.join(a.base, base_wm.get(name, fn)))
        ts, tb = hdr_s[fn].get(name), hdr_b[fn].get(name)
        if ts is None:
            fails.append(f"tensor {name} mapped to {fn} but not present there"); continue
        if tb is None:
            fails.append(f"tensor {name} absent from the base shard"); continue
        ls = ts["data_offsets"][1] - ts["data_offsets"][0]
        lb = tb["data_offsets"][1] - tb["data_offsets"][0]
        if ts["dtype"] != tb["dtype"] or ts["shape"] != tb["shape"] or ls != lb:
            fails.append(f"{name}: {ts['dtype']}{ts['shape']}/{ls}B vs base {tb['dtype']}{tb['shape']}/{lb}B")
        else:
            n_ok += 1
        if name.startswith(a.prefix):
            n_head += 1

    # --- (3) the swap is not a no-op
    head_files = sorted({fn for name, fn in wm.items() if name.startswith(a.prefix)})
    non_head_swapped = [f for f in swapped if f not in head_files]
    head_not_swapped = [f for f in head_files if f in shared]
    if head_not_swapped:
        fails.append(f"head shard(s) are the SAME bytes as base -- nothing was staged: {head_not_swapped}")
    if non_head_swapped:
        fails.append(f"non-head shard(s) diverge from base: {non_head_swapped[:3]}")

    # --- (5) provenance
    if a.head_store:
        card_p = os.path.join(a.head_store, "head_card.json")
        if not os.path.exists(card_p):
            fails.append(f"no head_card.json in {a.head_store}")
        else:
            card = json.load(open(card_p))
            rec = {f["file"]: f["sha256"] for f in card.get("files", [])}
            for fn in head_files:
                if fn not in rec:
                    fails.append(f"{fn} has no recorded sha256 in head_card.json"); continue
                got = sha256(os.path.join(a.staged, fn))
                if got != rec[fn]:
                    fails.append(f"{fn} sha256 {got[:12]} != archived {rec[fn][:12]}")
                else:
                    notes.append(f"sha256 ok {fn} = {got[:12]}...")
            m = card.get("measurement", {})
            notes.append(f"head '{card.get('name')}' promoted {card.get('promoted_utc')}, "
                         f"archived tau {m.get('suite_mean_tau')} / {m.get('suite_mean_tok_s')} tok/s")

    print(f"staged   : {a.staged}")
    print(f"base     : {a.base}")
    print(f"tensors  : {n_ok}/{len(wm)} drop-in identical (dtype, shape, byte length)")
    print(f"head     : {n_head} '{a.prefix}*' tensors in {len(head_files)} swapped shard(s): {head_files}")
    print(f"shared   : {len(shared)} file(s) are the same inode as base (no copy, no drift)")
    for n in notes:
        print(f"  note   : {n}")
    if fails:
        print(f"\nSTAGE VERIFY: FAIL ({len(fails)})")
        for f in fails[:20]:
            print(f"  - {f}")
        return 1
    print("\nSTAGE VERIFY: PASS")
    return 0

sys.exit(main())
