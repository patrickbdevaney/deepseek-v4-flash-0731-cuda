#!/usr/bin/env python3
"""make_head_dir.py — build a standalone DSpark head directory the engine can load as argv[4].

The engine already has a weight-override path: `argv[4]` opens a SEPARATE WeightStore filtered to
the `mtp.` prefix (src/decode.cu, `separate_head`). It was kept "for backward compatibility with an
external head" and is exactly what S5 needs to load a fine-tuned head without ever writing into the
read-only checkpoint directory.

What it needs is a directory whose `model.safetensors.index.json` mentions ONLY `mtp.*` tensors and
only the shards that hold them. The checkpoint's own index lists all 48 shards, so handing the
engine the backup directory unfiltered would have it look for 45 files that are not there.

  identity (Stage 0 control):  python3 tools/make_head_dir.py <ckpt> <outdir>
  fine-tuned:                  same, then overwrite the tensors in <outdir> from training

The identity build is the control that matters: loading the SAME weights through the override path
must reproduce the embedded-head run exactly. If it does not, every later comparison against a
fine-tuned head is measuring the loader, not the training.
"""
import json, os, shutil, sys

def main(ckpt, out):
    os.makedirs(out, exist_ok=True)
    idx = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))
    wm = idx["weight_map"]
    mtp = {k: v for k, v in wm.items() if k.startswith("mtp.")}
    if not mtp:
        print("FAIL: no mtp.* tensors in the index"); return 1
    shards = sorted(set(mtp.values()))
    print(f"{len(mtp)} mtp tensors across {len(shards)} shard(s): {', '.join(shards)}")

    for sh in shards:
        src, dst = os.path.join(ckpt, sh), os.path.join(out, sh)
        if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
            print(f"  {sh}: already present, skipping copy")
            continue
        print(f"  copying {sh} ...")
        shutil.copy2(src, dst)

    # Filtered index. total_size must describe THIS directory, not the checkpoint, or a loader that
    # sanity-checks it will reject the head for a size mismatch it cannot otherwise explain.
    total = sum(os.path.getsize(os.path.join(out, sh)) for sh in shards)
    json.dump({"metadata": {"total_size": total}, "weight_map": mtp},
              open(os.path.join(out, "model.safetensors.index.json"), "w"), indent=1)
    for aux in ("config.json",):
        s = os.path.join(ckpt, aux)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(out, aux))
    print(f"wrote {out}/model.safetensors.index.json  ({len(mtp)} tensors, {total/1e9:.2f} GB)")
    return 0

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
