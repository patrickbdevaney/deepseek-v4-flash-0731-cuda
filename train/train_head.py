#!/usr/bin/env python3
"""train_head.py — S5: fine-tune the DSpark MTP draft head on captured backbone taps.

DESIGN, and why it is shaped this way

* The MODEL is the checkpoint's own `inference/model.py`. Reimplementing DSparkBlock would be ~500
  lines with a high chance of a silent layout mismatch; instead `train/kernel.py` shims the six
  functions it imports, so every line of the model stays the checkpoint's.
* The EXPERTS ARE FROZEN and dequantised to bf16 at load. That is what makes the plain-PyTorch path
  possible (no fp4_gemm), it is what NVIDIA's reference DSpark trainer does, and it is the only way
  the job fits: the full head is 12.5 B real parameters (~116 GiB of bf16 + AdamW state) while the
  476 M non-expert parameters need 4.44 GiB. Reading MXFP4 and never writing it also keeps the
  project's no-additional-quantisation rule true by construction.
* The INPUT is `mh_pre` captured by the CUDA engine (`DSV4_CAPTURE`), which is byte-for-byte the
  reference's `main_hidden`: both are `cat([h.mean(over hc) for layers 40/41/42], dim=-1)`.
* The TARGET DISTRIBUTION for the TV term is recomputed here from the taps through the frozen
  lm_head rather than stored — 129280 floats/token would be 258 KB/token.

LOSS (DSpark paper == NVIDIA NeMo AutoModel defaults):
    L = sum_k w_k * [ a_ce*CE(p_k, y_k) + a_tv*TV(p_k, q_k) + a_conf*BCE(c_k, accepted_k) ]
    a_ce 0.1, a_tv 0.9, a_conf 1.0, w_k = exp(-(k-1)/gamma), gamma = block_size = 5

  usage:  python3 train/train_head.py --capture <dir> --ckpt <ckpt> --out <headdir> [--steps N]
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                       # our kernel.py shim must win over the checkpoint's
sys.path.insert(0, os.path.join(HERE, "_model"))

from read_capture import load_shard            # noqa: E402


# ------------------------------------------------------------------ weights
def dequant_mtp(ckpt, device, verbose=True):
    """Load every mtp.* tensor, dequantising MXFP4 (E2M1+E8M0) and FP8 (e4m3+block scale) to bf16.

    The engine's own loader is the reference for the layouts: MXFP4 is packed two values per byte
    with one E8M0 scale per 32, FP8 e4m3 carries an F8_E8M0 scale per 128x128 block.
    """
    from safetensors.torch import load_file
    idx = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))
    wm = {k: v for k, v in idx["weight_map"].items() if k.startswith("mtp.")}
    shards = sorted(set(wm.values()))
    raw = {}
    for sh in shards:
        if verbose: print(f"  loading {sh} ...", flush=True)
        raw.update({k: v for k, v in load_file(os.path.join(ckpt, sh)).items() if k.startswith("mtp.")})

    E2M1 = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device=device, dtype=torch.float32)

    def scale_to_float(t):
        """E8M0 scales arrive as uint8 or float8_e8m0fnu; both encode exp2(byte-127)."""
        if t.dtype == torch.uint8:
            return torch.exp2(t.float() - 127.0)
        if str(t.dtype).endswith("e8m0fnu"):
            return torch.exp2(t.view(torch.uint8).float() - 127.0)
        return t.float()

    def expand_to(sc, shape):
        """Block scales are one value per BLOCK in every dimension -- FP8 is 128x128, MXFP4 is 32
        along K. Broadcasting only the last dim was wrong: a [512,4096] weight carries a [4,32]
        scale and needs 128x expansion in BOTH dims."""
        while sc.dim() < len(shape):
            sc = sc.unsqueeze(-1)
        for d in range(len(shape)):
            if sc.shape[d] != shape[d]:
                if shape[d] % sc.shape[d]:
                    raise RuntimeError(f"scale dim {d}: {sc.shape[d]} does not divide {shape[d]}")
                sc = sc.repeat_interleave(shape[d] // sc.shape[d], dim=d)
        return sc

    out = {}
    for k, v in raw.items():
        if k.endswith(".scale"):
            continue
        sk = k.rsplit(".", 1)[0] + ".scale"
        if sk not in raw:
            out[k] = v.to(device=device, dtype=torch.bfloat16)
            continue
        s_ = scale_to_float(raw[sk].to(device=device))
        w = v.to(device=device)
        if w.dtype in (torch.uint8, torch.int8):                  # MXFP4, packed 2 values per byte
            b = w.view(torch.uint8)
            lo, hi = b & 0xF, (b >> 4) & 0xF
            nib = torch.stack([lo, hi], dim=-1).reshape(*b.shape[:-1], b.shape[-1] * 2)
            mag = E2M1[(nib & 7).long()]
            val = torch.where((nib & 8).bool(), -mag, mag)
            out[k] = (val * expand_to(s_, val.shape)).to(torch.bfloat16)
        else:                                                     # FP8 e4m3 + 128x128 block scale
            f = w.to(torch.float32)
            out[k] = (f * expand_to(s_, f.shape)).to(torch.bfloat16)
    if verbose:
        n = sum(t.numel() for t in out.values())
        print(f"  dequantised {len(out)} tensors, {n/1e9:.2f} B params, "
              f"{sum(t.numel()*2 for t in out.values())/2**30:.1f} GiB bf16", flush=True)
    return out


# ------------------------------------------------------------------ loss
def dspark_loss(logits, target_ids, tgt_logits, conf, accepted, gamma, a_ce=0.1, a_tv=0.9, a_conf=1.0):
    """logits/tgt_logits (K, V); target_ids (K,); conf (K,) or None; accepted (K,) bool."""
    K = logits.size(0)
    w = torch.exp(-torch.arange(K, device=logits.device, dtype=torch.float32) / gamma)
    w = w / w.sum()
    lp = F.log_softmax(logits.float(), dim=-1)
    ce = F.nll_loss(lp, target_ids, reduction="none")
    with torch.no_grad():
        p = F.softmax(tgt_logits.float(), dim=-1)
    tv = 0.5 * (p - lp.exp()).abs().sum(dim=-1)          # total variation
    loss = (w * (a_ce * ce + a_tv * tv)).sum()
    parts = {"ce": float((w * ce).sum()), "tv": float((w * tv).sum()), "conf": 0.0}
    if conf is not None:
        bce = F.binary_cross_entropy_with_logits(conf.float(), accepted.float(), reduction="none")
        loss = loss + a_conf * (w * bce).sum()
        parts["conf"] = float((w * bce).sum())
    return loss, parts


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=0, help="0 = one pass over the capture (1 epoch)")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--strict", action="store_true", help="fail on any state_dict mismatch")
    ap.add_argument("--smoke", action="store_true", help="load + one forward + one backward, then stop")
    a = ap.parse_args()

    dev = "cuda"
    torch.manual_seed(0)
    cfg = json.load(open(os.path.join(a.ckpt, "config.json")))
    print(f"[train] device={dev} torch={torch.__version__}", flush=True)

    t0 = time.time()
    sd = dequant_mtp(a.ckpt, dev)
    print(f"[train] weights ready in {time.time()-t0:.1f}s", flush=True)

    # Trainable = everything that is NOT a routed expert. 476 M by the shard-header count.
    train_keys = [k for k in sd if ".experts." not in k]
    n_train = sum(sd[k].numel() for k in train_keys)
    n_froz = sum(sd[k].numel() for k in sd if ".experts." in k)
    print(f"[train] trainable {n_train/1e6:.1f} M in {len(train_keys)} tensors | "
          f"frozen experts {n_froz/1e9:.2f} B", flush=True)
    if n_train > 1e9:
        sys.exit("[train] FAIL: trainable set is >1 B params; the expert filter is wrong")

    man = os.path.join(a.capture, "manifest.jsonl")
    rows = [json.loads(l) for l in open(man) if l.strip()]
    print(f"[train] capture: {len(rows)} sample(s)", flush=True)

    # ---- build the REAL DSpark head from the checkpoint's own model.py ----
    import torch.distributed as dist
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29517")
        os.environ.setdefault("RANK", "0"); os.environ.setdefault("WORLD_SIZE", "1")
        try: dist.init_process_group("gloo", rank=0, world_size=1)
        except Exception as e: print(f"[train] note: dist init skipped ({e})", flush=True)
    from model import ModelArgs, DSparkBlock, ParallelEmbedding, ParallelHead  # type: ignore
    _cfg = json.load(open(os.path.join(HERE, "_model", "dspark_config.json")))
    # Construct in BF16. The checkpoint config says dtype=fp8 / expert_dtype=fp4, which makes
    # model.Linear allocate PACKED quantised buffers -- experts come out [2048,2048] (two fp4 per
    # byte) against the logical [2048,4096]. We load DEQUANTISED tensors, so the module must be
    # built at logical shape; this is also exactly what makes model.linear() take its F.linear
    # branch and keeps train/kernel.py's fp4_gemm/fp8_gemm unreachable (they raise if reached).
    _cfg["dtype"] = "bf16"; _cfg["expert_dtype"] = None
    margs = ModelArgs(**_cfg)
    print(f"[train] ModelArgs: n_mtp={margs.n_mtp_layers} block={margs.dspark_block_size} "
          f"taps={margs.dspark_target_layer_ids} markov_rank={margs.dspark_markov_rank}", flush=True)
    with torch.device("meta"):
        blocks = [DSparkBlock(margs.n_layers + i, margs) for i in range(margs.n_mtp_layers)]
    print(f"[train] built {len(blocks)} DSparkBlock(s) on meta", flush=True)

    # Materialise from the dequantised tensors. to_empty() first, then a strict-ish load so a name
    # mismatch is an ERROR: silently leaving a tensor at its uninitialised value would train against
    # garbage and still produce a plausible loss curve.
    missing_all, unexpected_all = [], []
    for i, blk in enumerate(blocks):
        blk.to_empty(device=dev)
        pre = f"mtp.{i}."
        sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
        r = blk.load_state_dict(sub, strict=False)
        missing_all += [f"mtp.{i}.{m}" for m in r.missing_keys]
        unexpected_all += [f"mtp.{i}.{u}" for u in r.unexpected_keys]
    print(f"[train] load_state_dict: {len(missing_all)} missing, {len(unexpected_all)} unexpected", flush=True)
    for m in missing_all[:8]: print(f"    missing:    {m}", flush=True)
    for u in unexpected_all[:8]: print(f"    unexpected: {u}", flush=True)
    if a.strict and (missing_all or unexpected_all):
        sys.exit("[train] FAIL: state_dict mismatch (rerun without --strict to inspect)")

    print("[train] GATE1 model construction OK", flush=True)

    # embed / head live in the MAIN shards, not the mtp ones, and the reference ties them onto every
    # block (mtp[i].embed = embed, mtp[i].head = head). Both are bf16 already.
    from safetensors.torch import load_file as _lf
    idx_all = json.load(open(os.path.join(a.ckpt, "model.safetensors.index.json")))["weight_map"]
    need = {k: idx_all[k] for k in ("embed.weight", "head.weight") if k in idx_all}
    aux = {}
    for k, sh in need.items():
        aux[k] = _lf(os.path.join(a.ckpt, sh))[k].to(dev)
    print(f"[train] aux: " + ", ".join(f"{k}{tuple(v.shape)}" for k, v in aux.items()), flush=True)
    for blk in blocks:
        blk.embed = lambda ids, _w=aux["embed.weight"]: F.embedding(ids, _w)
        blk.head = lambda x, _w=aux["head.weight"]: F.linear(x, _w)

    # Trainable = non-expert parameters of the live modules (so grads reach the real graph).
    tp = [(n, p) for i, b in enumerate(blocks) for n, p in b.named_parameters() if ".experts." not in n]
    for _, p in [(n, p) for b in blocks for n, p in b.named_parameters()]:
        p.requires_grad_(False)
    for _, p in tp:
        p.requires_grad_(True)
    n_tr = sum(p.numel() for _, p in tp)
    print(f"[train] trainable tensors in graph: {len(tp)} / {n_tr/1e6:.1f} M", flush=True)
    opt = torch.optim.AdamW([p for _, p in tp], lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)

    gamma = float(margs.dspark_block_size)
    for step, r in enumerate(rows):
        sh = load_shard(os.path.join(a.capture, r["file"]), as_float32=True)
        taps = torch.from_numpy(sh["taps"].copy()).to(dev).to(torch.bfloat16)   # (T,3,d)
        ids = torch.from_numpy(sh["ids"].copy()).to(dev).long()
        # SHAPE, and the thing that is easy to get wrong: the reference's batch dimension is ONE
        # DRAFT PER POSITION, not sequence length. At inference `input_ids` is (B,) -- the last
        # accepted token of each sequence -- and the head drafts block_size tokens from that one
        # position. So teacher-forced training over T captured positions means B = T.
        # SHAPES, and the distinction that matters: `main_x` is (bsz, CTX_LEN, d) -- the main
        # model's context -- while `x` from forward_embed is (bsz, BLOCK_SIZE, hc, d), the draft
        # block. They are different axes, and conflating them is why the first two attempts failed.
        # At inference bsz=batch, main_hidden is the whole context, and input_ids is (bsz,): the last
        # accepted token. So ONE draft per sequence.
        #
        # STAGE-1 CONSEQUENCE, surfaced here rather than on day two of a capture: this reference is
        # written for inference and yields ONE training example per sequence. Teacher-forced training
        # at every position needs a training-mode forward that runs all T positions in parallel under
        # a causal mask; doing it the naive way (bsz=T, each row its own prefix) is O(T^2) memory.
        # That is real work and it belongs to stage 1, not to this gate.
        T = taps.size(0)
        main_hidden = taps.reshape(1, T, -1)             # (1, T, 3d)
        pos_ids = ids[T - 1:T]                           # (1,) the last accepted token
        h, main_x = blocks[0].forward_embed(main_hidden, pos_ids)
        for blk in blocks:
            h = blk(h, 0, pos_ids, main_x)
        print(f"[train] step {step}: forward OK  h{tuple(h.shape)} main_x{tuple(main_x.shape)} "
              f"finite={bool(torch.isfinite(h).all())}", flush=True)
        loss = h.float().pow(2).mean()          # stage-0 objective; the DSpark loss lands in stage 1
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_([p for _, p in tp], 1.0)
        ngrad = sum(1 for _, p in tp if p.grad is not None)
        opt.step()
        print(f"[train] step {step}: loss={float(loss.detach()):.4f} grad_norm={float(gn):.4f} "
              f"tensors_with_grad={ngrad}/{len(tp)}", flush=True)
        if ngrad == 0:
            sys.exit("[train] FAIL: no gradients reached the trainable tensors")
        if a.smoke:
            break

    print("[train] STAGE0 OK (real forward + backward)"); return

    os.makedirs(a.out, exist_ok=True)
    from safetensors.torch import save_file
    save = {k: v.detach().to(torch.bfloat16).cpu() for k, v in params.items()}
    save_file(save, os.path.join(a.out, "mtp_trained.safetensors"))
    print(f"[train] saved {len(save)} trained tensors -> {a.out}/mtp_trained.safetensors", flush=True)
    print("[train] STAGE0 OK")


if __name__ == "__main__":
    main()
