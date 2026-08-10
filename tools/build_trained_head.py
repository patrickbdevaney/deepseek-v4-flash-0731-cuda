#!/usr/bin/env python3
"""build_trained_head.py — turn trained bf16 tensors into a head directory the ENGINE can load.

THE PROBLEM THIS SOLVES, which is easy to miss until the eval fails: the trainer dequantises every
`mtp.*` tensor to bf16 and saves the trained ones as bf16. **The engine does not read bf16 there.**
It reads `mtp.0.attn.wq_a.weight` as FP8 e4m3 plus a separate `.scale`, exactly as the checkpoint
stores it. Handing it a bf16 tensor of the same name produces a plausible load and garbage weights.

So a trained head must be written back in **each tensor's ORIGINAL format**:

  * was FP8 e4m3 + F8_E8M0 128x128 block scale  -> re-quantise to that, rewriting the scale
  * was BF16 (norms, markov, confidence proj)   -> store bf16 unchanged
  * was MXFP4 (routed experts)                  -> NEVER TOUCHED; copied byte-for-byte

On the no-additional-quantisation rule: this does not violate it. The rule protects the SHIPPED
CHECKPOINT from being re-quantised to a lower precision. Here the experts are copied verbatim and
never decoded, and the trained non-expert tensors are written back at the precision they already
had. Storing them at a *higher* precision than the format would need a loader change and double the
head's footprint; storing them lower would be the thing the rule forbids.

  python3 tools/build_trained_head.py --base <ckpt> --trained <mtp_trained.safetensors> --out <dir>
"""
import argparse, json, os, shutil, sys
import numpy as np


FP8_MAX = 448.0


def _quant_fp8_e8m0(t, scale_dtype):
    """(bf16/f32 tensor) -> (fp8_e4m3 weights, E8M0 128x128 block scales) that DEQUANTISE BACK.

    The invariant that matters, and the one `--selftest` checks: `q.float() * scale` must reproduce
    `t` to fp8 precision. Any scheme where the divisor and the stored scale differ fails it.
    """
    import torch
    r, c = t.shape
    br, bc = min(128, r), min(128, c)
    tb = t.reshape(r // br, br, c // bc, bc)
    amax = tb.abs().amax(dim=(1, 3), keepdim=True)
    # exponent per block; blocks that are entirely zero get 2^0 rather than -inf
    e = torch.where(amax > 0, torch.ceil(torch.log2(amax / FP8_MAX)), torch.zeros_like(amax))
    e = e.clamp(-127, 127)
    scale = torch.pow(2.0, e)                                   # exactly representable in E8M0
    q = (tb / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).reshape(r, c)
    s = scale.reshape(r // br, c // bc)
    if scale_dtype == torch.uint8:
        s_out = (e.reshape(r // br, c // bc) + 127.0).to(torch.uint8)
    else:
        s_out = s.to(scale_dtype)                               # float8_e8m0fnu: exact for 2^k
    return q, s_out


def _engine_scale(t):
    """Decode an E8M0 scale EXACTLY as the engine and `train/train_head.py:dequant_mtp` do.

    Not `t.float()`. The question the selftest has to answer is not "does torch round-trip its own
    dtype" but "does the BYTE we wrote mean, to the reader, the number we intended" -- and the
    reader is `exp2(byte - 127)` over the raw bits.
    """
    import torch
    if t.dtype == torch.uint8:
        return torch.exp2(t.float() - 127.0)
    return torch.exp2(t.view(torch.uint8).float() - 127.0)


def selftest():
    import torch
    torch.manual_seed(0)
    sdt = torch.float8_e8m0fnu if hasattr(torch, "float8_e8m0fnu") else torch.uint8
    worst = 0.0
    for (r, c) in ((128, 128), (256, 512), (1024, 4096)):
        for mag in (1e-3, 1.0, 60.0):
            t = torch.randn(r, c) * mag
            t[0, 0] = 0.0
            q, s = _quant_fp8_e8m0(t, sdt)
            sf = _engine_scale(s)
            rec = q.float().reshape(r // min(128, r), min(128, r), c // min(128, c), min(128, c)) \
                   * sf.reshape(r // min(128, r), 1, c // min(128, c), 1)
            rec = rec.reshape(r, c)
            rel = ((rec - t).abs().max() / t.abs().max()).item()
            worst = max(worst, rel)
            assert rel < 0.07, f"{r}x{c} mag {mag}: rel err {rel:.4f} -- scale/divisor disagree"
    # a block whose max is exactly a power of two is the case ceil-vs-round disagrees on
    t = torch.full((128, 128), 448.0); t[5, 5] = -448.0
    q, s = _quant_fp8_e8m0(t, sdt)
    sf = (s.float() if sdt != torch.uint8 else torch.pow(2.0, s.float() - 127.0))
    assert (q.float() * sf).abs().max().item() <= 448.0 + 1e-3, "clamped a representable weight"
    print(f"selftest OK (worst relative error {worst:.4f}, fp8 e4m3 has ~2^-3 mantissa)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=("--selftest" not in sys.argv), help="checkpoint dir (source of format + untouched tensors)")
    ap.add_argument("--trained", required=("--selftest" not in sys.argv), help="mtp_trained.safetensors from train_head.py")
    ap.add_argument("--out", required=("--selftest" not in sys.argv))
    ap.add_argument("--selftest", action="store_true",
                    help="check the fp8+E8M0 round-trip and exit; requires no checkpoint")
    a, _ = ap.parse_known_args()
    if "--selftest" in sys.argv:
        selftest(); return

    import torch
    from safetensors.torch import load_file, save_file

    idx = json.load(open(os.path.join(a.base, "model.safetensors.index.json")))
    wm = {k: v for k, v in idx["weight_map"].items() if k.startswith("mtp.")}
    shards = sorted(set(wm.values()))
    trained = load_file(a.trained)
    print(f"trained tensors: {len(trained)}")

    os.makedirs(a.out, exist_ok=True)
    n_fp8 = n_bf16 = n_copy = 0
    worst_rt = 0.0

    for sh in shards:
        src = load_file(os.path.join(a.base, sh))
        out = {}
        for k, v in src.items():
            if k not in trained:
                out[k] = v                                  # untouched: experts, scales we rewrite below
                n_copy += 1
                continue
            t = trained[k].float()
            sk = k.rsplit(".", 1)[0] + ".scale"
            if sk in src and v.dtype == torch.float8_e4m3fn:
                # FP8 e4m3 + 128x128 block scale. Recompute the scale from the trained values so the
                # pair stays consistent -- reusing the old scale against new weights is the silent
                # way to get a plausible tensor with the wrong magnitude.
                #
                # THE SCALE IS **F8_E8M0**: a bare exponent, so the only representable values are
                # exact powers of two. Quantising against a continuous amax/448 and *then* rounding
                # the exponent for storage would leave the engine dequantising with a different
                # number than the one the weights were divided by -- a silent magnitude error of up
                # to sqrt(2) per block, on every weight, with no gate that would catch it. So snap
                # the scale to a power of two FIRST and quantise against that exact value.
                # ceil, not round: rounding down can push |t|/scale past 448 and clamp real weights.
                out[k], out[sk] = _quant_fp8_e8m0(t, src[sk].dtype)
                # Per-tensor round-trip on the REAL data, not just synthetic. Decode what we just
                # wrote using the engine's formula and compare to the trained values. A format
                # error shows up here as a number, at build time, instead of as a mysteriously
                # worse tau three hours later.
                r, c = t.shape
                br, bc = min(128, r), min(128, c)
                rec = (out[k].float().reshape(r // br, br, c // bc, bc)
                       * _engine_scale(out[sk]).reshape(r // br, 1, c // bc, 1)).reshape(r, c)
                rel = float((rec - t).abs().max() / t.abs().max().clamp_min(1e-30))
                worst_rt = max(worst_rt, rel)
                n_fp8 += 1
            elif v.dtype == torch.bfloat16:
                out[k] = t.to(torch.bfloat16)
                n_bf16 += 1
            elif v.dtype in (torch.float32, torch.float16):
                # Unquantised small tensors -- attn_sink is fp32 in this checkpoint. The trainer
                # saves in bf16, so writing back to fp32 is an exact upcast and to fp16 is a widening
                # of the exponent range: no quantisation decision to get wrong, hence no round-trip
                # check needed here. Kept as an explicit branch rather than folded into the bf16 case
                # so the shipped dtype always matches what the loader expects to read.
                out[k] = t.to(v.dtype)
                n_bf16 += 1
            else:
                print(f"  REFUSING {k}: trained but stored as {v.dtype}; no safe write-back path")
                sys.exit(2)
        save_file(out, os.path.join(a.out, sh), metadata={"format": "pt"})
        print(f"  wrote {sh}")

    total = sum(os.path.getsize(os.path.join(a.out, s)) for s in shards)
    json.dump({"metadata": {"total_size": total}, "weight_map": wm},
              open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=1)
    for aux in ("config.json",):
        p = os.path.join(a.base, aux)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(a.out, aux))
    print(f"re-quantised {n_fp8} fp8 (+scales), {n_bf16} bf16, copied {n_copy} untouched "
          f"(experts byte-for-byte) -> {a.out}  ({total/1e9:.2f} GB)")
    print(f"fp8 round-trip worst relative error: {worst_rt:.4f}  "
          f"(e4m3 half-step is ~0.0625; anything near 1.0 means the scale and the divisor disagree)")
    if worst_rt > 0.10:
        sys.exit(f"REFUSING to ship: round-trip error {worst_rt:.4f} exceeds fp8 e4m3 precision. "
                 f"The written scale does not decode to the value the weights were divided by.")


if __name__ == "__main__":
    main()
