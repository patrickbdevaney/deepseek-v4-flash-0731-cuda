#!/usr/bin/env python3
"""read_capture.py — reader and validator for S5 capture shards (.dspc).

The training script imports `load_shard`; run this file directly to validate a capture directory.

FORMAT (little-endian). Deliberately dumb: a header, the token ids, then the taps. No compression
and no framework dependency, because the producer is a CUDA binary and the consumer is PyTorch in a
container, and anything cleverer becomes a second thing that can disagree with itself.

  u32 magic   = 0x43505344 ('DSPC' little-endian)
  u32 version = 1
  u32 n_tok         positions with taps  (== PSp, the prefill length)
  u32 n_taps  = 3   backbone layers 40/41/42
  u32 d       = 4096
  u32 dtype   = 1   (bf16)
  u32 n_ids         full prompt length (n_tok + 1: the last id is the target for position n_tok-1)
  u32 reserved
  i32 ids[n_ids]
  bf16 taps[n_tok][n_taps][d]        layout matches k_tap_pool: mh[t*(3*d) + slot*d + j]

WHY THE TAPS AND NOT THE PROJECTED FEATURE: `main_proj` is trainable, so capturing after it would
bake in weights the fine-tune changes. WHY NO TARGET DISTRIBUTION: 129280 floats/token is
258 KB/token; it is recomputed in training from these taps through the frozen lm_head.
"""
import json, os, struct, sys
import numpy as np

MAGIC = 0x43505344

def load_shard(path, as_float32=False):
    """-> dict(ids: int32[n_ids], taps: (n_tok, n_taps, d) bf16-as-uint16 or float32)"""
    with open(path, "rb") as f:
        hdr = struct.unpack("<8I", f.read(32))
        magic, ver, n_tok, n_taps, d, dtype, n_ids, _ = hdr
        if magic != MAGIC:
            raise ValueError(f"{path}: bad magic {magic:#x} (expected {MAGIC:#x})")
        if ver != 1:
            raise ValueError(f"{path}: unsupported version {ver}")
        if dtype != 1:
            raise ValueError(f"{path}: unsupported dtype {dtype} (only bf16=1)")
        ids = np.frombuffer(f.read(4 * n_ids), dtype="<i4")
        raw = np.frombuffer(f.read(2 * n_tok * n_taps * d), dtype="<u2")
        if raw.size != n_tok * n_taps * d:
            raise ValueError(f"{path}: truncated: got {raw.size} of {n_tok*n_taps*d} tap values")
    taps = raw.reshape(n_tok, n_taps, d)
    if as_float32:
        # bf16 -> fp32 is an exact widening: the bf16 bits ARE the top half of the fp32 word.
        taps = (taps.astype(np.uint32) << 16).view(np.float32)
    return {"ids": ids, "taps": taps, "n_tok": n_tok, "n_taps": n_taps, "d": d}

def validate(capdir):
    man = os.path.join(capdir, "manifest.jsonl")
    if not os.path.exists(man):
        print(f"FAIL: no manifest.jsonl in {capdir}"); return 1
    rows, bad = [], 0
    for ln, line in enumerate(open(man), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A killed run can leave a half-written final line. That is expected and recoverable:
            # report it and ignore it, rather than refusing to read the whole capture.
            print(f"  note: manifest line {ln} is incomplete (killed mid-write); ignoring")
    print(f"manifest: {len(rows)} complete sample(s)")
    tot_tok = tot_bytes = 0
    for r in rows:
        p = os.path.join(capdir, r["file"])
        if not os.path.exists(p):
            print(f"FAIL: {r['file']} in manifest but missing on disk"); bad += 1; continue
        try:
            s = load_shard(p, as_float32=True)
        except Exception as e:
            print(f"FAIL: {r['file']}: {e}"); bad += 1; continue
        t = s["taps"]
        finite = np.isfinite(t).all()
        allzero = not t.any()
        # Each tap slot should have its own distribution; identical slots would mean the tap-pool
        # slot index never varied, which is a real failure mode of the capture call site.
        slot_std = [float(t[:, k, :].std()) for k in range(s["n_taps"])]
        ok = finite and not allzero and min(slot_std) > 0
        # ids must be n_tok+1: the extra id is the target for the last captured position.
        ids_ok = (len(s["ids"]) == s["n_tok"] + 1)
        print(f"  {r['file']}: n_tok={s['n_tok']} taps={s['n_taps']} d={s['d']} "
              f"ids={len(s['ids'])}{'' if ids_ok else ' <-- EXPECTED n_tok+1'} "
              f"slot_std={[round(x,4) for x in slot_std]} "
              f"finite={finite} -> {'OK' if (ok and ids_ok) else 'BAD'}")
        if not (ok and ids_ok):
            bad += 1
        tot_tok += s["n_tok"]; tot_bytes += r["bytes"]
    print(f"total: {tot_tok} tokens, {tot_bytes/1048576:.1f} MB "
          f"({tot_bytes/max(tot_tok,1)/1024:.1f} KB/token)")
    print("VALIDATE:", "FAIL" if bad else "PASS")
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(validate(sys.argv[1] if len(sys.argv) > 1 else "."))
