#!/usr/bin/env python3
"""Harvest the safetensors header of every shard via HTTP range requests.

The full checkpoint is 100.4 GiB; the headers are ~9 MB. This lets the roofline and
inventory be computed and reviewed before (or independently of) the download, and gives
the repo a committed, auditable copy of every tensor's dtype and shape.

Writes docs/hdrs/model-000NN-of-00048.safetensors.json (one JSON per shard) plus the
small metadata files (config.json, reap_plan.json, REAP_MANIFEST.json, the index).

Usage: python3 tools/fetch_headers.py
"""
import concurrent.futures as cf
import json
import os
import struct
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ID = "0xSero/DeepSeek-V4-Flash-0731-REAP"
BASE = f"https://huggingface.co/{REPO_ID}/resolve/main/"
N_SHARDS = 48
OUT = os.path.join(REPO, "docs", "hdrs")

SMALL = [
    "config.json",
    "reap_plan.json",
    "REAP_MANIFEST.json",
    "model.safetensors.index.json",
    "generation_config.json",
    "validation/structural-validation.json",
    "validation/runtime-smoke.json",
]


def curl(url, rng=None):
    cmd = ["curl", "-sSL", "--fail"]
    if rng:
        cmd += ["-r", rng]
    cmd.append(url)
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"curl failed for {url}: {r.stderr.decode()[:200]}")
    return r.stdout


def shard_header(i):
    name = f"model-{i:05d}-of-{N_SHARDS:05d}.safetensors"
    dest = os.path.join(OUT, name + ".json")
    if os.path.exists(dest) and os.path.getsize(dest) > 100:
        return name, "cached"
    n = struct.unpack("<Q", curl(BASE + name, "0-7")[:8])[0]
    hdr = json.loads(curl(BASE + name, f"8-{8 + n - 1}")[:n])
    json.dump(hdr, open(dest, "w"))
    return name, f"{len(hdr)} tensors"


def main():
    os.makedirs(OUT, exist_ok=True)
    for f in SMALL:
        dest = os.path.join(REPO, "docs", os.path.basename(f))
        open(dest, "wb").write(curl(BASE + f))
        print(f"  {f} -> docs/{os.path.basename(f)}")
    with cf.ThreadPoolExecutor(12) as ex:
        for name, note in ex.map(shard_header, range(1, N_SHARDS + 1)):
            print(f"  {name}: {note}")
    print(f"done -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
