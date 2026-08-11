#!/usr/bin/env python3
"""activation_sparsity.py — is the published MoE activation-sparsity lever alive on THIS checkpoint?

arXiv:2605.08575 reports that pre-trained MoE experts are internally dormant: up to ~90% of a
selected expert's intermediate neurons can be zeroed with little accuracy loss, for a claimed 2.5x
MoE-layer and 1.2x end-to-end speedup. On this engine the MoE term is large enough that such a lever
would be worth tens of percent.

WHY IT IS WORTH RE-TESTING HERE. The sibling Laguna project measured this on its own weights and
KILLED it (`~/laguna-s1-cuda-server/ACTIVATION_SPARSITY.md`): at the paper's 87% operating point,
zeroing cost 43.7% relative L2 error unstructured and 83% block-structured. Its stated reason was a
size argument -- every published measurement is on models with `moe_intermediate` >= 1408, and
Laguna's is **1024**. **This checkpoint's `moe_intermediate` is 2048**, above the published range.
So the one lever that failed there on width grounds is the one that might survive here, and it is
the only item in that comparison where the information flows the other way.

METHOD, following Laguna's so the two are comparable. For real routed (expert, token) pairs compute
the true intermediate activation

    h = SiLU(w1 @ x) * (w3 @ x)          (2048 neurons)

and then the mean relative L2 error of h when only the top-s fraction of neurons is kept, under
three keep-patterns: unstructured, block-32 (the native MXFP4 scale block, and the only unit a
kernel could actually skip), and block-32 after permuting neurons by mean |h| (the paper's proposed
bit-exact fix, which concentrates chronically-dormant neurons into shared blocks).

APPROXIMATION, stated plainly. `x` is a real captured hidden state at this depth passed through the
layer's FFN norm; it omits that layer's own attention contribution to the residual, because the
capture stores layer taps rather than MoE inputs. It is the model's own activation distribution at
depth, not a bit-exact MoE input. Dormancy is a property of the weights against realistic input
statistics, so this is adequate to decide the lever -- but it is not a bit-exact reproduction and a
positive result here would need an in-engine dump before anything is built on it.

  python3 tools/activation_sparsity.py [--layer 41] [--tokens 256] [--experts 6]
"""
import argparse, json, os, struct
import numpy as np

CK = '/home/patrickd/models/DeepSeek-V4-Flash-0731-REAP'
E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)   # engine LUT, kernels/cutlass_moe.cu:113


def _hdr(p):
    with open(p, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def load(name, idx):
    """Read one tensor out of the shard it lives in, without loading the shard."""
    path = os.path.join(CK, idx[name])
    h, base = _hdr(path)
    meta = h[name]
    o0, o1 = meta['data_offsets']
    dt = {'I8': np.uint8, 'F8_E8M0': np.uint8, 'BF16': np.uint16,
          'F32': np.float32, 'F8_E4M3': np.uint8}[meta['dtype']]
    with open(path, 'rb') as f:
        f.seek(base + o0)
        raw = np.frombuffer(f.read(o1 - o0), dtype=dt)
    return raw.reshape(meta['shape']), meta['dtype']


def bf16(u):
    return (u.astype(np.uint32) << 16).view(np.float32)


def dequant_mxfp4(w_packed, scale_e8m0):
    """[R, K/2] uint8 packed E2M1 + [R, K/32] E8M0 -> [R, K] float32.

    Low nibble is the lower-index value (the order `cvt.f16x2.e2m1x2` consumes).
    """
    R, Kh = w_packed.shape
    lo = w_packed & 0xF
    hi = w_packed >> 4
    codes = np.empty((R, Kh * 2), dtype=np.uint8)
    codes[:, 0::2] = lo
    codes[:, 1::2] = hi
    vals = E2M1[codes & 7] * np.where(codes & 8, -1.0, 1.0).astype(np.float32)
    sc = np.exp2(scale_e8m0.astype(np.float32) - 127.0)            # E8M0
    K = vals.shape[1]
    return (vals.reshape(R, K // 32, 32) * sc[:, :, None]).reshape(R, K)


def curves(H, block=32):
    """H: (rows, n_neurons). -> {sparsity: (unstructured, block, block-sorted)} mean rel L2 error."""
    n = H.shape[1]
    energy = H.astype(np.float64) ** 2
    denom = np.sqrt(energy.sum(1))
    denom[denom == 0] = 1.0
    order_global = np.argsort(-np.abs(H).mean(0))                  # permute by MEAN |h| (offline, bit-exact)
    Hs = H[:, order_global]
    out = {}
    for s in (0.25, 0.50, 0.75, 0.87, 0.90):
        keep = max(1, int(round(n * (1 - s))))
        # (1) unstructured: keep the largest `keep` per row
        idx = np.argpartition(-np.abs(H), keep - 1, axis=1)[:, :keep]
        m = np.zeros_like(H, dtype=bool)
        np.put_along_axis(m, idx, True, axis=1)
        e1 = np.sqrt((energy * ~m).sum(1)) / denom
        # (2) block-32, chosen per row by block energy
        nb = n // block
        be = energy.reshape(-1, nb, block).sum(2)
        kb = max(1, int(round(nb * (1 - s))))
        bidx = np.argpartition(-be, kb - 1, axis=1)[:, :kb]
        bm = np.zeros((H.shape[0], nb), dtype=bool)
        np.put_along_axis(bm, bidx, True, axis=1)
        e2 = np.sqrt((energy.reshape(-1, nb, block) * ~bm[:, :, None]).sum((1, 2))) / denom
        # (3) block-32 after the global permutation -- blocks are now shared across rows
        es = Hs.astype(np.float64) ** 2
        bes = es.reshape(-1, nb, block).sum(2)
        bsi = np.argpartition(-bes, kb - 1, axis=1)[:, :kb]
        bsm = np.zeros((H.shape[0], nb), dtype=bool)
        np.put_along_axis(bsm, bsi, True, axis=1)
        e3 = np.sqrt((es.reshape(-1, nb, block) * ~bsm[:, :, None]).sum((1, 2))) / denom
        out[s] = (e1.mean(), e2.mean(), e3.mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layer', type=int, default=41)
    ap.add_argument('--tokens', type=int, default=256)
    ap.add_argument('--tap', type=int, default=0, help='which captured tap to use as the residual')
    ap.add_argument('--capture', default=None)
    a = ap.parse_args()

    idx = json.load(open(os.path.join(CK, 'model.safetensors.index.json')))['weight_map']
    L = a.layer

    # real hidden states from an S5 capture
    cap = a.capture
    if cap is None:
        import glob
        c = sorted(glob.glob('/home/patrickd/s5-capture/*/c*/cap/manifest.jsonl'))
        if not c:
            raise SystemExit('no capture found; pass --capture <dir>')
        cap = os.path.dirname(c[-1])
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from read_capture import load_shard
    rows = [json.loads(l) for l in open(os.path.join(cap, 'manifest.jsonl')) if l.strip()]
    sh = load_shard(os.path.join(cap, rows[0]['file']), as_float32=True)
    taps = sh['taps']
    X = np.asarray(taps[:a.tokens, a.tap, :], dtype=np.float32)
    print(f"[act] capture {cap}")
    print(f"[act] {X.shape[0]} real hidden states, d={X.shape[1]}, layer {L}")

    # FFN norm (RMSNorm) then router
    gw, _ = load(f'layers.{L}.ffn_norm.weight', idx)
    g = bf16(gw).astype(np.float32)
    Xn = X / np.sqrt((X ** 2).mean(1, keepdims=True) + 1e-6) * g

    rw, _ = load(f'layers.{L}.ffn.gate.weight', idx)
    rb, _ = load(f'layers.{L}.ffn.gate.bias', idx)
    R = bf16(rw).astype(np.float32)
    scores = Xn @ R.T + np.asarray(rb, dtype=np.float32)
    topk = np.argsort(-scores, axis=1)[:, :6]                       # top-6 of 160
    sel, cnt = np.unique(topk, return_counts=True)
    order = np.argsort(-cnt)
    print(f"[act] routed to {len(sel)} distinct experts; busiest: "
          + ', '.join(f'e{sel[i]}({cnt[i]})' for i in order[:5]))

    # compute h for the busiest experts against the tokens that actually routed to them
    H = []
    for e in sel[order[:8]]:
        toks = np.where((topk == e).any(1))[0]
        if len(toks) < 4:
            continue
        w1, _ = load(f'layers.{L}.ffn.experts.{e}.w1.weight', idx)
        s1, _ = load(f'layers.{L}.ffn.experts.{e}.w1.scale', idx)
        w3, _ = load(f'layers.{L}.ffn.experts.{e}.w3.weight', idx)
        s3, _ = load(f'layers.{L}.ffn.experts.{e}.w3.scale', idx)
        W1 = dequant_mxfp4(w1, s1)
        W3 = dequant_mxfp4(w3, s3)
        xt = Xn[toks]
        gate = xt @ W1.T
        up = xt @ W3.T
        h = (gate / (1.0 + np.exp(-gate))) * up                     # SiLU(gate) * up
        H.append(h.astype(np.float32))
    H = np.concatenate(H, 0)
    print(f"[act] {H.shape[0]} expert-token rows of {H.shape[1]} neurons "
          f"(moe_intermediate={H.shape[1]})\n")

    c = curves(H)
    print(f"  {'sparsity':>9} {'(1) unstructured':>18} {'(2) block-32':>14} {'(3) block-32 sorted':>21}")
    for s, (e1, e2, e3) in c.items():
        mark = '   <- paper operating point' if abs(s - 0.87) < 1e-9 else ''
        print(f"  {s:>9.2f} {e1:>18.4f} {e2:>14.4f} {e3:>21.4f}{mark}")
    print(f"\n  Laguna, moe_intermediate 1024, same method, at 0.87: 0.4374 / 0.8314 / 0.8220")
    e1, e2, e3 = c[0.87]
    print(f"  ours,   moe_intermediate 2048,             at 0.87: {e1:.4f} / {e2:.4f} / {e3:.4f}")
    print()
    verdict = ("SURVIVES -- worth an in-engine dump to confirm bit-exactly"
               if e2 < 0.10 else
               "DEAD, same as Laguna -- block-32 is the only unit a kernel can skip, and it is "
               "nowhere near lossless")
    print(f"  VERDICT: {verdict}")
    print("  Column (2) is the one that decides it: unstructured sparsity is unimplementable on a")
    print("  GEMV that reads 32-value MXFP4 blocks, so column (1) is informational only.")


if __name__ == '__main__':
    main()
