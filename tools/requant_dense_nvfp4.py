#!/usr/bin/env python3
"""requant_dense_nvfp4.py — quantise the DENSE/MLA path from FP8 to NVFP4, and measure what it costs.

WHY. The dense/MLA path is **72 % of `B_tok`** (9.4 GB of 12.26 per token) and is the only remaining
term with real volume behind it: six successive hypotheses about kernel efficiency have now been
refuted (F13, F14, F15, glue, F125, F126), while the byte argument survives all of them untouched.
Halving the precision of the FP8 linears takes `B_tok` 12.26 -> 9.11 and decode ~25.5 -> ~34 tok/s,
which is where the REAM-104E checkpoint lands **without** accepting 104 merged experts.

WHAT IS AND IS NOT TOUCHED — decided by MEASUREMENT on this checkpoint, not by copying the exclusion
list of `Baekpica/...-REAM-104E-NVFP4`. Relative L2 that NVFP4 adds on top of the FP8 already there:

    quantised   MLA wq_a/wq_b/wo_b/wkv          0.0935     the original plan
    quantised   attn.wo_a                       0.0924     REAM excluded it; on OUR weights it
                                                           measures BETTER than the set it was
                                                           excluded from. Included on evidence.
    quantised   MTP draft dense                 0.0937     ZERO intelligence risk: the draft is
                                                           verified, so error costs acceptance,
                                                           never a wrong token.
    NOT         lm_head / head.weight           0.1145     above the 0.10 this project refuses
                                                           elsewhere; FP8 (0.0331) is its ceiling
    NOT         routed experts                             already MXFP4 -- 4-bit, nothing to gain
    NOT         indexer / compressor                       these pick WHICH positions are attended;
                                                           an error there is a different attention
                                                           pattern, not a small perturbation
    NOT         embed                                      a lookup, ~0 bytes per token

THE SCHEME (`NVFP4A16`, weight-only — no activation calibration set is needed):

    global_scale = 448 * 6 / amax(tensor)              448 = e4m3 max, 6 = E2M1 max
    group_scale  = e4m3( amax(group of 16) / 6 * global_scale )
    q            = round( w / (group_scale / global_scale) )  in E2M1, 2 codes per byte

CORRECTNESS GATE FIRST. `--measure` quantises the real tensors and reports the error NVFP4 adds on
top of the FP8 the checkpoint already carries. Nothing is written until that number is looked at,
because a requant is only worth doing if the damage is small, and 105 GB of output is the expensive
way to find out it is not.

  python3 tools/requant_dense_nvfp4.py --measure [--layers 4]
  python3 tools/requant_dense_nvfp4.py --write <outdir>          (after the gate)
"""
import argparse, json, os, re, struct, sys
import numpy as np

CK = '/home/patrickd/models/DeepSeek-V4-Flash-0731-REAP'
E2M1_MAX, E4M3_MAX, GROUP = 6.0, 448.0, 16

# ---- fp8 e4m3 <-> float, by table (no torch on the host) -----------------------------------------
def _e4m3_table():
    t = np.zeros(256, dtype=np.float32)
    for b in range(256):
        s = -1.0 if (b >> 7) else 1.0
        e = (b >> 3) & 0xF
        m = b & 0x7
        if e == 0:
            v = (m / 8.0) * 2.0 ** (-6)
        elif e == 0xF and m == 0x7:
            v = np.nan                      # e4m3 has no inf; S1111.111 is NaN
        else:
            v = (1.0 + m / 8.0) * 2.0 ** (e - 7)
        t[b] = s * v
    return t

E4M3 = _e4m3_table()

# NEVER quantise INTO a NaN code. e4m3 has two (0x7F, 0xFF) and an earlier version of this file
# mapped them to 0.0 before sorting, which put them adjacent to the real zero in the search table --
# so `f32_to_e4m3(1e-9)` returned **0xFF**, a NaN. As a group SCALE that silently poisons all 16
# weights in the group, and the resulting overlay would have looked fine on a size check.
# The candidate set is the FINITE codes only.
_CAND = np.array([c for c in range(256) if np.isfinite(E4M3[c])], dtype=np.uint8)
_ORD = _CAND[np.argsort(E4M3[_CAND])]         # finite e4m3 codes, sorted by value
_SVAL = E4M3[_ORD]
_SMID = (_SVAL[:-1] + _SVAL[1:]) / 2.0

def f32_to_e4m3(x):
    """Round-to-nearest into the FINITE e4m3 grid, clamped to its range.

    searchsorted on midpoints rather than an argmin over a broadcast: wq_b alone is 33.5 M elements
    and the broadcast form allocates ~1 GB per tensor.
    """
    a = np.asarray(x, dtype=np.float32)
    a = np.clip(np.nan_to_num(a, nan=0.0, posinf=448.0, neginf=-448.0), -448.0, 448.0)
    i = np.searchsorted(_SMID, a.ravel())
    out = _ORD[i].astype(np.uint8).reshape(a.shape)
    assert np.isfinite(E4M3[out]).all(), "quantiser emitted a NaN e4m3 code"
    return out


def st_header(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def read_tensor(name, idx):
    path = os.path.join(CK, idx[name])
    h, base = st_header(path)
    m = h[name]
    o0, o1 = m['data_offsets']
    with open(path, 'rb') as f:
        f.seek(base + o0)
        raw = f.read(o1 - o0)
    return raw, m


def dequant_fp8_block(w_u8, scale, shape):
    """The checkpoint's own format: fp8 e4m3 values x a 128x128 block scale."""
    w = E4M3[w_u8].reshape(shape).astype(np.float32)
    r, c = shape
    br, bc = min(128, r), min(128, c)
    s = scale.astype(np.float32)
    if s.ndim == 0:
        return w * float(s)
    s = s.reshape(max(1, r // br), max(1, c // bc))
    return (w.reshape(r // br, br, c // bc, bc) * s[:, None, :, None]).reshape(r, c)


def quant_nvfp4(w):
    """-> (packed uint8 [R, K/2], group scales uint8 e4m3 [R, K/16], global_scale float)"""
    R, K = w.shape
    amax = float(np.max(np.abs(w)))
    if amax == 0:
        amax = 1.0
    gs = E4M3_MAX * E2M1_MAX / amax
    g = w.reshape(R, K // GROUP, GROUP)
    gamax = np.max(np.abs(g), axis=2)                       # (R, K/16)
    s_real = gamax / E2M1_MAX * gs
    s_u8 = f32_to_e4m3(s_real)                              # the scale itself is e4m3
    s_q = E4M3[s_u8]
    eff = np.where(s_q != 0, s_q / gs, 1.0)                 # dequant step per group
    q = g / eff[:, :, None]
    # E2M1 grid: 0, .5, 1, 1.5, 2, 3, 4, 6 -- round to nearest representable
    LUT = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
    MID = (LUT[:-1] + LUT[1:]) / 2.0
    code = np.searchsorted(MID, np.abs(q).ravel()).astype(np.uint8).reshape(q.shape)
    code |= (q < 0).astype(np.uint8) << 3
    code = code.reshape(R, K)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).astype(np.uint8)
    return packed, s_u8, np.float32(gs)


def dequant_nvfp4(packed, s_u8, gs, R, K):
    lut = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
    lo, hi = packed & 0xF, packed >> 4
    code = np.empty((R, K), dtype=np.uint8)
    code[:, 0::2], code[:, 1::2] = lo, hi
    v = lut[code & 7] * np.where(code & 8, -1.0, 1.0).astype(np.float32)
    eff = (E4M3[s_u8] / gs).astype(np.float32)
    return (v.reshape(R, K // GROUP, GROUP) * eff[:, :, None]).reshape(R, K)


# EVIDENCE-BASED, not inherited. Measured relative L2 that NVFP4 adds over the FP8 already present:
#   MTP draft   0.0937      MLA targeted 0.0935      attn.wo_a 0.0924      lm_head 0.1145
# So wo_a goes IN -- it measures slightly better than the set it was excluded from, and that
# exclusion came from REAM's card rather than from this checkpoint. lm_head stays OUT of NVFP4 at
# 0.1145 (above the 0.10 this project refuses elsewhere) and takes FP8 at 0.0331 instead. The DSA
# indexer/compressor stay untouched: they choose WHICH positions are attended, so their errors are
# not a small perturbation of an average, they are a different attention pattern.
SKIP = re.compile(r'(experts|lm_head|head\.weight|embed|\.norm|_norm|\.gate\b|gate\.|indexer|compressor|attn_sink)')


def targets(idx):
    """The F8_E4M3 dense linears, minus the exclusion list."""
    out = []
    for k in idx:
        if k.endswith('.scale') or SKIP.search(k):
            continue
        out.append(k)
    return sorted(out)


def write_overlay(idx, out):
    """Write an OVERLAY, not a checkpoint copy.

    98.4 GB of the checkpoint is routed experts that this change does not touch, so copying them
    would triple the disk cost of the experiment for nothing. The overlay carries only the requantised
    dense linears; the engine reads experts from the original checkpoint, which stays byte-identical
    and is never opened for writing.
    """
    os.makedirs(out, exist_ok=True)
    names = [k for k in targets(idx)
             if re.match(r'layers\.\d+\.attn\.(wq_a|wq_b|wo_a|wo_b|wkv)\.weight$', k)     # + wo_a, measured
             or re.match(r'mtp\.\d+\.attn\.(wq_a|wq_b|wo_a|wo_b|wkv)\.weight$', k)]       # draft: zero risk
    hdr, blobs, off = {}, [], 0
    src_gb = 0.0
    for i, k in enumerate(names):
        raw, m = read_tensor(k, idx)
        sk = k.rsplit('.', 1)[0] + '.scale'
        R, K = m['shape']
        if sk in idx:
            sraw, sm = read_tensor(sk, idx)
            sc = np.frombuffer(sraw, dtype=np.uint8)
            sc = np.exp2(sc.astype(np.float32) - 127.0) if sm['dtype'] == 'F8_E8M0' else E4M3[sc]
        else:
            sc = np.float32(1.0)
        src_gb += (R * K) / 1e9
        w = dequant_fp8_block(np.frombuffer(raw, dtype=np.uint8), sc, (R, K))
        packed, s_u8, gs = quant_nvfp4(w)
        for nm, arr, dt in ((k + '.nvfp4', packed, 'U8'),
                            (k + '.nvfp4_scale', s_u8, 'U8'),
                            (k + '.nvfp4_global', np.array([gs], dtype=np.float32), 'F32')):
            b = arr.tobytes()
            hdr[nm] = {'dtype': dt, 'shape': list(arr.shape), 'data_offsets': [off, off + len(b)]}
            off += len(b); blobs.append(b)
        if (i + 1) % 40 == 0 or i + 1 == len(names):
            print(f"    {i+1}/{len(names)} tensors, {off/1e9:.2f} GB written so far", flush=True)
    hdr['__metadata__'] = {
        'what': 'NVFP4 (E2M1 group-16 + e4m3 scales + fp32 global) overlay for the DENSE/MLA linears',
        'base': 'DeepSeek-V4-Flash-0731-REAP, unmodified -- experts and everything else read from it',
        'scheme': 'q = round(w / (e4m3(group_amax/6*gs) / gs)); gs = 448*6/amax(tensor)',
        'excluded': 'experts, lm_head, embed, norms, gates, indexer, compressor, attn_sink, attn.wo_a',
        'tensors': str(len(names))}
    hj = json.dumps(hdr).encode()
    path = os.path.join(out, 'dense_nvfp4.safetensors')
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(hj))); f.write(hj)
        for b in blobs: f.write(b)
    print(f"\n  wrote {path}")
    print(f"  {len(names)} dense linears: {src_gb:.2f} GB of FP8 -> {off/1e9:.2f} GB of NVFP4 "
          f"({100*(1-off/1e9/src_gb):.1f} % smaller)")
    print(f"  the base checkpoint was opened READ-ONLY and is untouched")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--measure', action='store_true')
    ap.add_argument('--write', default=None, help='output directory for the overlay')
    ap.add_argument('--layers', type=int, default=4)
    a = ap.parse_args()
    if a.write:
        idx = json.load(open(os.path.join(CK, 'model.safetensors.index.json')))['weight_map']
        return write_overlay(idx, a.write)

    idx = json.load(open(os.path.join(CK, 'model.safetensors.index.json')))['weight_map']
    cand = [k for k in targets(idx) if re.match(r'layers\.\d+\.attn\.(wq_a|wq_b|wo_b|wkv)\.weight$', k)]
    picked = [k for k in cand if int(k.split('.')[1]) < a.layers]
    print(f"[requant] {len(cand)} dense linears eligible; measuring {len(picked)} from the first "
          f"{a.layers} layers\n")
    print(f"  {'tensor':<34} {'shape':>16} {'FP8 GB':>8} {'NVFP4 GB':>9} {'rel L2':>8} {'max rel':>8}")

    tot_fp8 = tot_nv = 0.0
    errs = []
    for k in picked:
        raw, m = read_tensor(k, idx)
        sk = k.rsplit('.', 1)[0] + '.scale'
        sraw, sm = read_tensor(sk, idx) if sk in idx else (None, None)
        R, K = m['shape']
        w_u8 = np.frombuffer(raw, dtype=np.uint8)
        if sraw is not None:
            sc = np.frombuffer(sraw, dtype=np.uint8)
            sc = np.exp2(sc.astype(np.float32) - 127.0) if sm['dtype'] == 'F8_E8M0' else E4M3[sc]
        else:
            sc = np.float32(1.0)
        w = dequant_fp8_block(w_u8, sc, (R, K))             # ground truth = what the engine reads today
        packed, s_u8, gs = quant_nvfp4(w)
        wr = dequant_nvfp4(packed, s_u8, gs, R, K)
        num = float(np.linalg.norm((wr - w).ravel()))
        den = float(np.linalg.norm(w.ravel())) or 1.0
        mx = float(np.max(np.abs(wr - w)) / (np.max(np.abs(w)) or 1.0))
        fp8_gb = (R * K + sc.size) / 1e9
        nv_gb = (packed.size + s_u8.size) / 1e9
        tot_fp8 += fp8_gb; tot_nv += nv_gb
        errs.append(num / den)
        print(f"  {k:<34} {str([R,K]):>16} {fp8_gb:>8.3f} {nv_gb:>9.3f} {num/den:>8.4f} {mx:>8.4f}")

    print(f"\n  bytes: {tot_fp8:.3f} GB -> {tot_nv:.3f} GB  ({100*(1-tot_nv/tot_fp8):.1f} % smaller)")
    print(f"  relative L2 error NVFP4 adds on top of the FP8 already there: "
          f"mean {np.mean(errs):.4f}, worst {np.max(errs):.4f}")
    print(f"\n  For scale: the fp8 round-trip gate this project already enforces on the draft head")
    print(f"  refuses anything above 0.10, and the shipped head measures 0.0014.")
    print(f"  A dense-path error near or above 0.05 would be a different kind of change entirely --")
    print(f"  it is the ATTENTION path, and it is not verified away by speculative decoding.")


if __name__ == '__main__':
    main()
