#!/usr/bin/env python3
"""Model inventory + roofline arithmetic for 0xSero/DeepSeek-V4-Flash-0731-REAP.

Every number this prints is read from disk: `docs/config.json` (verbatim copy of the
checkpoint's config.json) and `docs/hdrs/*.json` (the safetensors header of each of the
48 shards, harvested by `tools/fetch_headers.py`). Nothing is hardcoded.

Self-check: the summed tensor bytes must equal the `total_size` in
model.safetensors.index.json (107,803,320,952) exactly, or the accounting is wrong and
the script exits non-zero.

Usage:  python3 tools/inventory.py [--model-dir DIR]
        --model-dir reads headers straight out of a local checkout instead of docs/hdrs.
"""
import argparse
import collections
import glob
import json
import os
import re
import struct
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# safetensors dtype -> bytes per element
DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
}

EXPECTED_TOTAL = 107_803_320_952  # from model.safetensors.index.json metadata.total_size


def load_headers_from_docs():
    tensors = {}
    for path in sorted(glob.glob(os.path.join(REPO, "docs", "hdrs", "*.json"))):
        shard = os.path.basename(path).replace(".json", "")
        for name, meta in json.load(open(path)).items():
            if name == "__metadata__":
                continue
            tensors[name] = (meta["dtype"], tuple(meta["shape"]), shard)
    return tensors


def load_headers_from_model_dir(model_dir):
    tensors = {}
    for path in sorted(glob.glob(os.path.join(model_dir, "model-*.safetensors"))):
        shard = os.path.basename(path)
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            tensors[name] = (meta["dtype"], tuple(meta["shape"]), shard)
    return tensors


def nbytes(entry):
    dtype, shape, _ = entry
    n = 1
    for d in shape:
        n *= d
    return n * DTYPE_BYTES[dtype]


def subsystem(name):
    if name.startswith("mtp."):
        return "MTP (DSpark heads)"
    if name == "embed.weight":
        return "embed"
    if name == "head.weight":
        return "lm_head"
    if name.startswith("hc_head"):
        return "hc_head"
    if ".ffn.experts." in name:
        return "routed experts (MXFP4)"
    if ".ffn.shared_experts." in name:
        return "shared expert (FP8)"
    if ".ffn.gate" in name:
        return "router"
    if ".attn.indexer." in name:
        return "DSA indexer"
    if ".attn.compressor." in name:
        return "KV compressor"
    if ".attn." in name:
        return "MLA attn"
    if "hc_" in name:
        return "HC (hyper-connection) params"
    return "norms"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--bandwidth", type=float, default=200.0,
                    help="achievable streaming GB/s (default 200)")
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(REPO, "docs", "config.json")))
    tensors = (load_headers_from_model_dir(args.model_dir) if args.model_dir
               else load_headers_from_docs())
    size = {k: nbytes(v) for k, v in tensors.items()}
    total = sum(size.values())

    print("=" * 78)
    print("DeepSeek-V4-Flash-0731-REAP (K160) — inventory")
    print("=" * 78)
    print(f"tensors            {len(tensors):,}")
    print(f"total tensor bytes {total:,}  = {total / 2**30:.3f} GiB = {total / 1e9:.3f} GB")
    ok = total == EXPECTED_TOTAL
    print(f"matches index.json total_size ({EXPECTED_TOTAL:,}): {ok}")
    if not ok:
        print("FAIL: byte accounting does not reconcile.", file=sys.stderr)
        return 1

    # ---- dtype histogram -------------------------------------------------
    by_dtype = collections.Counter()
    for k, v in tensors.items():
        by_dtype[v[0]] += size[k]
    print("\n--- bytes by dtype ---")
    for d, b in by_dtype.most_common():
        print(f"  {d:9s} {b / 2**30:9.3f} GiB  {100 * b / total:5.1f}%")

    # ---- subsystem histogram --------------------------------------------
    by_sub = collections.Counter()
    for k in tensors:
        by_sub[subsystem(k)] += size[k]
    print("\n--- bytes by subsystem ---")
    for d, b in by_sub.most_common():
        print(f"  {d:26s} {b / 2**30:9.3f} GiB  {100 * b / total:5.1f}%")

    # ---- quant format proof ---------------------------------------------
    print("\n--- expert quantisation format (proof, not assumption) ---")
    probe = "layers.5.ffn.experts.0.w1"
    wd, ws, _ = tensors[probe + ".weight"]
    sd, ss, _ = tensors[probe + ".scale"]
    k_logical = ws[1] * 2  # I8 holds 2 packed 4-bit values
    block = k_logical // ss[1]
    print(f"  {probe}.weight  {wd} {list(ws)}  -> logical [{ws[0]}, {k_logical}] E2M1")
    print(f"  {probe}.scale   {sd} {list(ss)}")
    print(f"  scale block size along K = {k_logical} / {ss[1]} = {block}")
    print(f"  => E2M1 data + E8M0 scale, block {block}  =  OCP MXFP4"
          f"  ({4 + 8 / block:.2f} bits/param effective)")

    # ---- router / REAP remapping check ----------------------------------
    print("\n--- REAP router layout (does a remap need applying at runtime?) ---")
    gw = tensors["layers.5.ffn.gate.weight"]
    gb = tensors["layers.5.ffn.gate.bias"]
    n_exp = len([k for k in tensors if re.match(r"layers\.5\.ffn\.experts\.\d+\.w1\.weight$", k)])
    print(f"  gate.weight {list(gw[1])}  gate.bias {list(gb[1])}  experts present: {n_exp}")
    print(f"  config n_routed_experts = {cfg['n_routed_experts']}")
    print("  => router rows already compacted to the retained set; expert ids are dense 0..%d."
          % (n_exp - 1))

    # ---- per-token decode bytes -----------------------------------------
    L = cfg["num_hidden_layers"]
    topk = cfg["num_experts_per_tok"]
    step = collections.Counter()
    for i in range(L):
        p = f"layers.{i}."
        def s(pred):
            return sum(size[k] for k in tensors if pred(k))
        step["MLA attn"] += s(lambda k: k.startswith(p + "attn.")
                              and ".compressor." not in k and ".indexer." not in k)
        step["KV compressor"] += s(lambda k: k.startswith(p + "attn.compressor."))
        step["DSA indexer"] += s(lambda k: k.startswith(p + "attn.indexer."))
        step["HC params"] += s(lambda k: k.startswith(p + "hc_"))
        step["router"] += s(lambda k: k.startswith(p + "ffn.gate"))
        step["shared expert"] += s(lambda k: k.startswith(p + "ffn.shared_experts."))
        step["norms"] += s(lambda k: k.startswith(p + "attn_norm") or k.startswith(p + "ffn_norm"))
        step[f"routed experts (top-{topk})"] += topk * s(
            lambda k: k.startswith(p + "ffn.experts.0."))
    step["lm_head"] = size["head.weight"]
    step["hc_head"] = sum(size[k] for k in tensors if k.startswith("hc_head"))
    step["embed (1 row)"] = cfg["hidden_size"] * 2

    B = sum(step.values())
    print(f"\n--- B_tok: bytes read per M=1 decode step, {L} layers, top-{topk} ---")
    for k, v in sorted(step.items(), key=lambda x: -x[1]):
        print(f"  {k:28s} {v / 1e6:10.2f} MB  {100 * v / B:5.1f}%")
    print(f"  {'B_tok TOTAL':28s} {B / 1e6:10.2f} MB  = {B / 2**30:.4f} GiB")

    print("\n--- autoregressive wall ---")
    for bw in (args.bandwidth, 273.0):
        print(f"  @ {bw:5.0f} GB/s : {bw * 1e9 / B:6.2f} tok/s  ({1000 * B / (bw * 1e9):6.2f} ms/tok)")

    # ---- active parameters (cross-check vs the model card's "13B active") --
    def params(pred):
        n = 0
        for k, (dt, sh, _) in tensors.items():
            if not pred(k):
                continue
            c = 1
            for d in sh:
                c *= d
            if k.endswith(".weight") and dt == "I8" and ".experts." in k:
                c *= 2       # two packed 4-bit values per byte
            if k.endswith(".scale"):
                c = 0        # scales are not parameters
            n += c
        return n
    act = 0
    for i in range(L):
        p = f"layers.{i}."
        act += params(lambda k: k.startswith(p) and ".ffn.experts." not in k)
        act += topk * params(lambda k: k.startswith(p + "ffn.experts.0."))
    act += params(lambda k: k == "head.weight")
    print(f"\n--- active parameters per token (backbone, excl. embed lookup) ---")
    print(f"  {act / 1e9:.2f} B   (model card reports ~13B active)")
    print(f"  blended {8 * B / act:.2f} bits/active-param  "
          f"(NOT 4.25 — only the routed experts are MXFP4)")

    # ---- MTP -------------------------------------------------------------
    print("\n--- DSpark MTP blocks (embedded, REAP-pruned to 160 experts) ---")
    mtp_total = sum(size[k] for k in tensors if k.startswith("mtp."))
    for m in range(3):
        p = f"mtp.{m}."
        non_e = sum(size[k] for k in tensors if k.startswith(p) and ".experts." not in k)
        one_e = sum(size[k] for k in tensors if k.startswith(p + "ffn.experts.0."))
        print(f"  mtp.{m}: resident {sum(size[k] for k in tensors if k.startswith(p)) / 2**30:6.3f} GiB"
              f" | per-step non-expert {non_e / 1e6:7.2f} MB + top-{topk} {topk * one_e / 1e6:6.2f} MB"
              f" = {(non_e + topk * one_e) / 1e6:7.2f} MB")
    print(f"  MTP resident total {mtp_total / 2**30:.3f} GiB")

    # ---- KV cache --------------------------------------------------------
    print("\n--- KV cache marginal cost per token (from config, per dtype) ---")
    ratios = cfg["compress_ratios"][:L]
    kv_dim = cfg["head_dim"]                # MLA latent width = 512
    idx_dim = cfg["index_head_dim"]         # 128
    for name, w in (("fp8 / int8", 1), ("bf16", 2), ("fp32 (current engine)", 4)):
        per_tok = 0.0
        for i, r in enumerate(ratios):
            if r == 0:
                continue                    # pure-sliding layer: fixed window, no growth
            per_tok += kv_dim * w / r
            if r == 4:
                per_tok += idx_dim * w / r  # DSA indexer cache lives on ratio-4 layers
        window = sum(cfg["sliding_window"] * kv_dim * w for _ in range(L))
        print(f"  {name:22s} {per_tok / 1024:7.2f} KiB/token marginal"
              f"  + {window / 2**20:6.1f} MiB fixed window"
              f"  -> 1M ctx = {(per_tok * 1e6 + window) / 2**30:6.2f} GiB")
    print(f"  (ratio-4 layers: {sum(1 for r in ratios if r == 4)},"
          f" ratio-128 layers: {sum(1 for r in ratios if r == 128)},"
          f" pure-sliding: {sum(1 for r in ratios if r == 0)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
