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
import argparse, json, math, os, sys, time
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
    # a_conf DEFAULTS TO 0, deliberately, and the DSpark paper's value is 1.0.
    #
    # Wired and measured: ce=10.43, tv=0.93, conf=10034. The confidence term is ~1000x the other
    # two and would swamp the objective. Two reasons, and neither is a coding error:
    #   * the head takes the UN-normalised x (pre self.norm), whose scale is large -- the captured
    #     taps have std ~190 -- so its raw logit is large;
    #   * it was trained to predict acceptance under FREE-RUNNING drafting, and teacher forcing
    #     makes almost every position accepted, so it is confidently wrong on this data by
    #     construction. That is the HASS mismatch showing up in the confidence signal rather than
    #     in the draft input.
    # Fixing it needs free-running draft labels (session 2's HASS work), not a scale factor. Until
    # then the term is OFF and sessions train on ce+tv, which are correct. Shipping a loss term
    # that is 1000x the others without understanding it is how you get a plausible wrong answer.
    ap.add_argument("--a-conf", type=float, default=0.0, dest="a_conf",
                    help="confidence-BCE weight; 0 until free-running labels exist (see comment)")
    ap.add_argument("--pos-per-seq", type=int, default=8, help="training positions sampled per sequence")
    ap.add_argument("--strict", action="store_true", help="fail on any state_dict mismatch")
    ap.add_argument("--block", type=int, default=6, help="draft block; MUST match the engine (default 6)")
    ap.add_argument("--bisect", default=None, help="engine intermediates.bin to diff against")
    ap.add_argument("--gate", action="store_true", help="replay engine probes, report agreement")
    ap.add_argument("--smoke", action="store_true", help="load + one forward + one backward, then stop")
    ap.add_argument("--metrics-out", default=None, dest="metrics_out",
                    help="write per-step loss history + diagnostics as JSON (archived with the head)")
    a = ap.parse_args()

    dev = "cuda"
    torch.manual_seed(0)
    if os.environ.get("ANOMALY"): torch.autograd.set_detect_anomaly(True)
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
    from model import (ModelArgs, DSparkBlock, ParallelEmbedding, ParallelHead,  # type: ignore
                       get_dspark_topk_idxs)
    _cfg = json.load(open(os.path.join(HERE, "_model", "dspark_config.json")))
    # Construct in BF16. The checkpoint config says dtype=fp8 / expert_dtype=fp4, which makes
    # model.Linear allocate PACKED quantised buffers -- experts come out [2048,2048] (two fp4 per
    # byte) against the logical [2048,4096]. We load DEQUANTISED tensors, so the module must be
    # built at logical shape; this is also exactly what makes model.linear() take its F.linear
    # branch and keeps train/kernel.py's fp4_gemm/fp8_gemm unreachable (they raise if reached).
    _cfg["dtype"] = "bf16"; _cfg["expert_dtype"] = None
    # BLOCK SIZE MUST MATCH THE ENGINE, and it does not by default. inference/config.json says
    # dspark_block_size: 5 -- the width the head was TRAINED at -- while the engine's serving
    # default is 6 since F94/F96. A 5-wide port against a 6-wide engine drafts a different number
    # of noise tokens through a different attention length: not a subtle numerics gap, a different
    # computation. This is the same trained-width vs serving-width distinction that produced F93's
    # defect, showing up a third time.
    _cfg["dspark_block_size"] = a.block
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
    # RE-MATERIALISE NON-PERSISTENT BUFFERS. `freqs_cis` is registered with persistent=False, so it
    # is excluded from state_dict BY DESIGN -- load_state_dict never fills it -- while to_empty()
    # has already replaced it with uninitialised memory. The port was therefore doing every rotary
    # embedding against GARBAGE, which is why q and the block-KV both diverged from an input that
    # was exact, and why main_x and hc_pre (the only stages with no rope) were fine.
    #
    # to_empty() silently invalidates every non-persistent buffer. load_state_dict cannot warn about
    # it, because from its point of view nothing is missing.
    from model import precompute_freqs_cis  # type: ignore
    _osl = _cfg.get("original_seq_len", 65536)
    nbuf = 0
    for blk in blocks:
        at = blk.attn
        fc = precompute_freqs_cis(at.rope_head_dim, margs.max_seq_len, _osl,
                                  margs.rope_theta, margs.rope_factor,
                                  margs.beta_fast, margs.beta_slow).to(dev)
        at.freqs_cis = fc; nbuf += 1
    print(f"[train] re-materialised freqs_cis on {nbuf} attention module(s) "
          f"(persistent=False buffers are NOT in state_dict and to_empty() zapped them)", flush=True)
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
        # ParallelHead is called as head(x, full_logits=True); with world_size 1 the full logits
        # ARE the local logits, so the flag is a no-op here -- but it must be accepted.
        blk.head = lambda x, full_logits=False, _w=aux["head.weight"]: F.linear(x, _w.to(x.dtype))

    # Trainable = non-expert parameters of the live modules (so grads reach the real graph).
    tp = [(n, p) for i, b in enumerate(blocks) for n, p in b.named_parameters() if ".experts." not in n]
    for _, p in [(n, p) for b in blocks for n, p in b.named_parameters()]:
        p.requires_grad_(False)
    for _, p in tp:
        p.requires_grad_(True)
    n_tr = sum(p.numel() for _, p in tp)
    print(f"[train] trainable tensors in graph: {len(tp)} / {n_tr/1e6:.1f} M", flush=True)
    opt = torch.optim.AdamW([p for _, p in tp], lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)

    _dbg = {"agree":0,"hit":0,"n":0,"tgt_top1_p":0.0,"drf_top1_p":0.0,"m":0}
    _hist = []   # per-step loss record, written out by --metrics-out
    gamma = float(margs.dspark_block_size)
    BS = margs.dspark_block_size

    def forward_head_tf(blk, x, seed, target_ids):
        """TEACHER-FORCED forward_head. The checkpoint's forward_head is an inference loop:

            output_ids[:, 0] = input_ids
            for i in range(block_size):
                bias, emb = markov_head(output_ids[:, i])   # embedding SAVES this view for backward
                logits[:, i].add_(bias)                     # in-place
                output_ids[:, i+1] = sample(logits[:, i])   # then MUTATES the saved tensor

        which is both non-differentiable (it samples) and illegal under autograd (the mutation bumps
        the version of a tensor the embedding still needs) -- that is the LongTensor[1] version
        error anomaly detection pointed at.

        Training feeds the GROUND TRUTH into the Markov head instead of the model's own samples.
        That removes the sampling, removes the in-place mutation, and is what the DSpark and FastMTP
        recipes prescribe. It is also, by construction, the train/inference mismatch HASS addresses
        -- worth revisiting in stage 1, but teacher forcing is the correct starting point.
        """
        x = blk.hc_head(x, blk.hc_head_fn, blk.hc_head_scale, blk.hc_head_base)
        logits = blk.head(blk.norm(x), full_logits=True)          # (B, BS, V)
        ids_in = torch.cat([seed, target_ids[:-1]], dim=0).view(1, -1)   # ground-truth prefix
        out_logits, embeds = [], []
        for i in range(blk.block_size):
            bias, emb = blk.markov_head(ids_in[:, i])
            out_logits.append(logits[:, i] + bias)                # NOT add_
            embeds.append(emb)
        conf = blk.confidence_head(x, torch.stack(embeds, dim=1))
        return torch.stack(out_logits, dim=1), conf



    # ================= BISECT (--bisect <intermediates.bin>) =================
    # Reads the engine's intermediate tensors at one probe position and reports, stage by stage,
    # where the port first diverges. F102: the port scores 0/288 and four guesses at the cause were
    # all wrong (STE, seed clone, retain_graph, block width). This replaces guessing with the one
    # question that matters -- which tensor is the FIRST to disagree.
    if a.bisect:
        import struct as _st
        with open(a.bisect, "rb") as f:
            t_probe, cur_probe, blk_probe = _st.unpack("<3I", f.read(12))
            eng = {}
            while True:
                hdr = f.read(4)
                if len(hdr) < 4: break
                L = _st.unpack("<I", hdr)[0]
                tag = f.read(L).decode()
                n = _st.unpack("<I", f.read(4))[0]
                eng[tag] = np.frombuffer(f.read(4 * n), dtype="<f4").copy()
        print(f"[bisect] engine probe: t={t_probe} seed={cur_probe} block={blk_probe}")
        print(f"[bisect] tensors: {[(k, v.size) for k, v in eng.items()]}")
        if blk_probe != BS:
            print(f"[bisect] WARNING: engine block {blk_probe} != port block {BS}; pass --block {blk_probe}")

        sh = load_shard(os.path.join(a.capture, rows[0]["file"]), as_float32=True)
        taps = torch.from_numpy(sh["taps"].copy()).to(dev).to(torch.bfloat16)
        ids  = torch.from_numpy(sh["ids"].copy()).to(dev).long()
        T = taps.size(0); main_hidden = taps.reshape(1, T, -1)

        def cmp(tag, port_t):
            if tag not in eng:
                return
            e = torch.from_numpy(eng[tag]).to(dev).float().flatten()
            p_ = port_t.detach().float().flatten()
            if e.numel() != p_.numel():
                print(f"  {tag:<16} SHAPE MISMATCH engine {e.numel()} vs port {p_.numel()}"); return
            cos = float(torch.nn.functional.cosine_similarity(e.unsqueeze(0), p_.unsqueeze(0)))
            rel = float((e - p_).norm() / (e.norm() + 1e-9))
            flag = "OK" if cos > 0.999 else ("CLOSE" if cos > 0.9 else "**DIVERGED**")
            print(f"  {tag:<16} cosine={cos:+.6f} rel_err={rel:.4f}  {flag}")

        with torch.no_grad():
            # PREFILL ONLY UP TO THE PROBE POSITION. The kv_cache is a RING indexed by
            # absolute_pos % win: a full-length prefill wraps it, so after T=194 with win=128 slot 0
            # holds absolute position 128, not 0. get_dspark_topk_idxs(start_pos=8) then returns raw
            # slots arange(9) = 0..8, which contain absolute 128..136 -- the WRONG KEYS, while the
            # engine reads main_kv[wstart..t] = absolute 0..8.
            #
            # The reference is not wrong: at inference start_pos >= win, so it reads the WHOLE ring
            # and the rotation is irrelevant (attention is permutation-invariant over keys once RoPE
            # has baked position in). It was MY harness driving it at start_pos=8 against a
            # full-length prefill -- a state serving never produces. Prefilling to exactly t+1 makes
            # seqlen <= win take the non-wrapping branch, so slot i == absolute i and the port reads
            # what the engine reads.
            ctx = main_hidden[:, :t_probe + 1]
            h0, mx0 = blocks[0].forward_embed(ctx, ids[:1].clone())
            hh = h0
            for blk in blocks:
                hh = blk(hh, 0, ids[:1].clone(), mx0)
            for _b in blocks:
                for _n, _bu in _b.named_buffers(recurse=True):
                    if _bu is not None and _bu.is_floating_point(): _bu.detach_()
            seed = ids[t_probe:t_probe + 1].clone()
            hb, mx = blocks[0].forward_embed(main_hidden[:, t_probe:t_probe + 1], seed)
            print(f"[bisect] port seed={int(seed.item())} (engine {cur_probe}) "
                  f"{'MATCH' if int(seed.item())==cur_probe else '** MISMATCH **'}")
            cmp("main_x_t", mx)
            cmp("after_embed", hb)
            # ---- diff BLOCK 0's sub-stages, using the MODEL's own hc_pre/hc_post ----
            # Mirrors kernels/dspark_attn.cu dspark_block_forward:
            #   hc_pre(attn) -> rmsnorm -> attn -> hc_post -> hc_pre(ffn) -> rmsnorm -> moe -> hc_post
            # Calling the reference methods (not a second reimplementation) keeps this a diff of the
            # port as it actually runs.
            b0 = blocks[0]
            x1, post_g, comb_g = b0.hc_pre(hb, b0.hc_attn_fn, b0.hc_attn_scale, b0.hc_attn_base)
            cmp("b_hcpre_attn_x1", x1)
            cmp("b_hcpre_attn_post", post_g)
            cmp("b_hcpre_attn_comb", comb_g)
            x1n = b0.attn_norm(x1)
            cmp("b_rmsnorm_attn", x1n)
            # ---- replicate DSparkAttention's q and assembled kv, and diff them ----
            at = b0.attn
            _rd = at.rope_head_dim; _win = at.window_size
            _seqlen = mx.size(1)
            _fc = at.freqs_cis[int(t_probe) + _seqlen: int(t_probe) + _seqlen + BS]
            _q = at.q_norm(at.wq_a(x1n))
            _q = at.wq_b(_q).unflatten(-1, (at.n_local_heads, at.head_dim))
            _q = _q * torch.rsqrt(_q.square().mean(-1, keepdim=True) + at.eps)
            from model import rope_tail as _rt
            _q = _rt(_q, _rd, _fc)
            cmp("a_q", _q)
            _bkv = at.kv_norm(at.wkv(x1n)); _bkv = _rt(_bkv, _rd, _fc)
            cmp("a_bkv", _bkv)
            _ti = get_dspark_topk_idxs(_win, 1, BS, int(t_probe))
            _kvfull = torch.cat([at.kv_cache[:1], _bkv], dim=1)
            _sel = _kvfull[0][_ti[0, 0].long()]          # the 15 keys actually attended
            cmp("a_kv_all", _sel)
            # continue exactly as Block.forward does, comparing each remaining stage
            ao = b0.attn(x1n, int(t_probe), mx)
            cmp("b_attn_out", ao)
            hp = b0.hc_post(ao, hb, post_g, comb_g)
            cmp("b_hcpost_attn", hp)
            x2, post2, comb2 = b0.hc_pre(hp, b0.hc_ffn_fn, b0.hc_ffn_scale, b0.hc_ffn_base)
            x2n = b0.ffn_norm(x2)
            cmp("b_rmsnorm_ffn", x2n)
            mo = b0.ffn(x2n, seed)
            cmp("b_moe_out", mo)

            for i, blk in enumerate(blocks):
                hb = blk(hb, int(t_probe), seed, mx)
                cmp(f"after_block{i}", hb)
        print("[bisect] the FIRST **DIVERGED** line above is the bug")
        return

    # ================= EQUIVALENCE GATE (--gate) =================
    # F101 halted session 1 because the port agreed with the engine on 1.5% of draft positions.
    # This replays the ENGINE's own probe positions and reports agreement, so the port can be
    # bisected against per-position ground truth instead of guessed at.
    #
    # It must use the engine's GREEDY AUTOREGRESSIVE head, not the teacher-forced one: the engine
    # feeds its OWN proposal at position i into the markov head for position i+1 (F27, device-side
    # greedy AR over the block). Teacher forcing is correct for TRAINING and wrong for this
    # comparison -- using it here would measure a different function than the server runs.
    def forward_head_greedy(blk, x, seed):
        x = blk.hc_head(x, blk.hc_head_fn, blk.hc_head_scale, blk.hc_head_base)
        logits = blk.head(blk.norm(x), full_logits=True)          # (1, BS, V)
        out = [int(seed.item())]
        for i in range(blk.block_size):
            bias, _ = blk.markov_head(torch.tensor([out[i]], device=x.device))
            out.append(int((logits[:, i] + bias).argmax(dim=-1).item()))
        return out[1:]

    if a.gate:
        import glob as _glob
        tot = agree = 0
        for prow in rows:
            gpath = os.path.join(a.capture, prow["file"].replace("shard_", "drafts_").replace(".dspc", ".txt"))
            if not os.path.exists(gpath):
                continue
            sh = load_shard(os.path.join(a.capture, prow["file"]), as_float32=True)
            taps = torch.from_numpy(sh["taps"].copy()).to(dev).to(torch.bfloat16)
            ids  = torch.from_numpy(sh["ids"].copy()).to(dev).long()
            T = taps.size(0); main_hidden = taps.reshape(1, T, -1)
            for line in open(gpath):
                v = line.split(); t = int(v[0]); eng = [int(x) for x in v[2:2 + BS]]
                # Re-prefill to exactly t+1 per probe: the kv_cache is a ring and a full-length
                # prefill wraps it, so slot i != absolute i. See the --bisect note.
                with torch.no_grad():
                    h0, mx0 = blocks[0].forward_embed(main_hidden[:, :t + 1], ids[:1].clone())
                    hh = h0
                    for blk in blocks:
                        hh = blk(hh, 0, ids[:1].clone(), mx0)
                for _b in blocks:
                    for _n, _bu in _b.named_buffers(recurse=True):
                        if _bu is not None and _bu.is_floating_point():
                            _bu.detach_()
                with torch.no_grad():
                    seed = ids[t:t + 1].clone()
                    hb, mx = blocks[0].forward_embed(main_hidden[:, t:t + 1], seed)
                    for blk in blocks:
                        hb = blk(hb, int(t), seed, mx)
                    port = forward_head_greedy(blocks[-1], hb, seed)
                for k in range(BS):
                    tot += 1; agree += int(port[k] == eng[k])
            print(f"[gate] {os.path.basename(gpath)}: running agreement {100*agree/max(tot,1):.1f}%", flush=True)
        print(f"[gate] PORT vs ENGINE draft agreement: {agree}/{tot} = {100*agree/max(tot,1):.1f}%  "
              f"(need >= 90% before any session runs)")
        return

    # ---- TRAINING-MODE FORWARD ----------------------------------------------------------------
    # The reference is two-phase and the probe found it the hard way: at start_pos == 0 the block
    # ONLY populates its KV cache and returns without attending, so h came back detached from
    # main_x and no gradient could reach anything. forward_spec says the same thing plainly --
    # `if start_pos == 0: return`.
    #
    # That two-phase shape is what makes teacher-forced training tractable, and it is much better
    # than the O(T^2) formulation I feared:
    #   phase A, ONCE per sequence: start_pos=0 over the full context -> fills the cache, O(T)
    #   phase B, per training position t: start_pos=t drafts BS tokens against that shared cache,
    #            costing only BS queries. So T examples cost O(T) cache + O(T * BS * ctx), not
    #            O(T^2) memory, and one captured sequence yields ~T training examples rather than 1.
    def draft_at(positions, ids_1d, main_hidden_1T):
        """-> logits (P, BS, V) for each start position in `positions`."""
        h0, main_x = blocks[0].forward_embed(main_hidden_1T, ids_1d[:1])
        hh = h0
        for blk in blocks:                                   # phase A: fill the cache
            hh = blk(hh, 0, ids_1d[:1], main_x)
        outs = []
        for t in positions:                                  # phase B: draft from each position
            seed = ids_1d[t:t + 1]
            # At start_pos>0 the block is in INCREMENTAL mode: it appends ONE context position to
            # the sliding-window cache (`kv_cache[:bsz, start_pos % win] = main_kv.squeeze(1)`), so
            # main_x must be (bsz, 1, d) -- the single new hidden state -- not the whole prefix.
            # Phase A already put positions 0..T-1 in the cache; this rewrites slot t%win with the
            # same value it already holds.
            hb, mx = blocks[0].forward_embed(main_hidden_1T[:, t:t + 1], seed)
            for blk in blocks:
                hb = blk(hb, int(t), seed, mx)
            _, logits, conf = blocks[-1].forward_head(hb, seed)
            outs.append((logits, conf))
        return outs

    # LR SCHEDULE. --warmup existed as a flag and did nothing: every step ran at the flat --lr.
    # Fine-tuning a *pretrained* head at a flat 5e-5 from step 0 is the standard way to lose the
    # thing you are trying to keep, and a flag that silently has no effect is the same failure
    # class as the persistent=False buffers (F105) -- no error, plausible output, wrong function.
    _nsteps = a.steps if a.steps > 0 else len(rows)
    _nwarm = max(1, int(a.warmup * _nsteps))
    def _lr_at(s):
        if s < _nwarm: return a.lr * (s + 1) / _nwarm
        prog = (s - _nwarm) / max(1, _nsteps - _nwarm)
        return a.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
    print(f"[train] lr schedule: linear warmup {_nwarm} step(s) -> cosine over {_nsteps}, "
          f"peak {a.lr:g}", flush=True)

    for step, r in enumerate(rows):
        lr_now = _lr_at(step)
        for g in opt.param_groups: g["lr"] = lr_now
        sh = load_shard(os.path.join(a.capture, r["file"]), as_float32=True)
        taps = torch.from_numpy(sh["taps"].copy()).to(dev).to(torch.bfloat16)
        ids = torch.from_numpy(sh["ids"].copy()).to(dev).long()
        lm_in = (torch.from_numpy(sh["lm_in"].copy()).to(dev).to(torch.bfloat16)
                 if sh.get("lm_in") is not None else None)
        head_w = aux["head.weight"]
        T = taps.size(0)
        main_hidden = taps.reshape(1, T, -1)
        # A position is trainable only if all BS targets exist after it.
        usable = [t for t in range(1, T) if t + BS < ids.numel()]
        if not usable:
            print(f"[train] step {step}: sequence too short for block={BS}, skipping"); continue
        sel = usable[:: max(1, len(usable) // max(1, min(len(usable), a.pos_per_seq)))][:a.pos_per_seq]
        # BACKWARD PER POSITION, not once over the sum. Every position writes the SAME sliding
        # kv_cache buffer, so a deferred backward hits
        #   "one of the variables needed for gradient computation has been modified by an inplace
        #    operation ... is at version 6; expected version 5"
        # -- earlier positions' cached values are gone by the time the graph is walked. Stepping the
        # backward immediately after each position's forward keeps each graph valid, and accumulates
        # into .grad exactly as a summed loss would.
        opt.zero_grad(set_to_none=True)
        # PHASE A UNDER no_grad, and this is a design decision, not a workaround.
        #
        # The cache fill writes blk.kv_cache IN PLACE, so its autograd history persists in a module
        # buffer across shards -- shard n+1's fill then traces back into shard n's freed graph and
        # raises "backward through the graph a second time". `retain_graph` does not fix that; it
        # only defers it, because the offending reference is the BUFFER, not the loop.
        #
        # Building the graph is also not what we want. Gradient reaches main_proj through phase B's
        # OWN forward_embed at each training position; the cache is the model's CONTEXT, and
        # treating context as given is the standard formulation. Dropping the phase-A path costs a
        # second-order term and buys independent per-position graphs, no retain_graph, less memory
        # and a faster step.
        with torch.no_grad():
            h0, main_x = blocks[0].forward_embed(main_hidden, ids[:1].clone())
            hh = h0
            for blk in blocks:                               # phase A: fill the cache, O(T)
                hh = blk(hh, 0, ids[:1].clone(), main_x)
        tot_loss, nparts, nb = 0.0, {"ce": 0.0, "tv": 0.0, "conf": 0.0}, 0
        for t in sel:                                        # phase B: draft from each position
            # SEVER THE CACHE HISTORY BEFORE EACH POSITION. Phase B also writes kv_cache
            # (`kv_cache[:bsz, start_pos % win] = main_kv.squeeze(1)`), so position k+1's forward
            # traces back into position k's already-freed graph through the module BUFFER. Running
            # phase A under no_grad was necessary but not sufficient -- the writes inside phase B
            # are the other half. detach_() in place leaves the VALUES (the context the attention
            # needs) and drops only the history.
            for _b in blocks:
                for _n, _buf in _b.named_buffers(recurse=True):
                    if _buf is not None and _buf.is_floating_point():
                        _buf.detach_()
            # clone: `seed` was a VIEW of `ids`, and forward_embed writes through it
            # (`draft_input_ids[:, 0] = input_ids`), so successive positions bumped the
            # version counter of a tensor an earlier graph still referenced.
            seed = ids[t:t + 1].clone()
            hb, mx = blocks[0].forward_embed(main_hidden[:, t:t + 1], seed)
            for blk in blocks:
                hb = blk(hb, int(t), seed, mx)
            tgt = ids[t + 1:t + 1 + BS]
            logits, conf = forward_head_tf(blocks[-1], hb, seed, tgt)
            lg = logits[0]                                   # (BS, V)
            # TARGET DISTRIBUTION for the TV term, from the CAPTURED lm_head input.
            # It is NOT reconstructible from the taps: those are h.mean(dim=hc) while this is the
            # learned Sinkhorn hc_head combination. Passing lg.detach() (what stage 0 did) makes
            # TV(p,q) identically zero and silently drops 90% of the loss weight -- measured as
            # tv=0.0000. The target for draft position k is the base model's distribution at the
            # position that produced token tgt[k], i.e. lm_in[t+k].
            if lm_in is None:
                sys.exit("[train] FAIL: capture is v1 (no lm_head input). Re-capture with the "
                         "current engine; the TV term cannot be computed without it.")
            with torch.no_grad():
                rows = lm_in[t:t + BS].to(dev)               # (BS, d)
                tgt_lg = F.linear(rows.to(head_w.dtype), head_w).float()
            # CONFIDENCE LABEL: did this draft position match the target's own next token? That is
            # exactly what the confidence head is trained to predict, and it needs no capture field
            # -- it is computable from the draft's own argmax against the ground truth.
            with torch.no_grad():
                accepted = (lg.argmax(dim=-1) == tgt)
                # DIAGNOSTIC. TV near 1.0 with acceptance ~30% is inconsistent for peaked
                # distributions, so distinguish the two explanations before trusting either:
                #   draft-argmax == target-argmax  -> alignment is right, the draft is just broad
                #   near zero                      -> tgt_lg is misaligned and TV is meaningless
                _dbg["agree"] += int((lg.argmax(-1) == tgt_lg.argmax(-1)).sum())
                _dbg["hit"]   += int(accepted.sum())
                _dbg["n"]     += tgt.numel()
                _dbg["tgt_top1_p"] += float(F.softmax(tgt_lg, -1).max(-1).values.mean())
                _dbg["drf_top1_p"] += float(F.softmax(lg.float(), -1).max(-1).values.mean())
                _dbg["m"] += 1
            cvec = conf.reshape(-1)[:BS] if conf is not None else None
            l, parts = dspark_loss(lg, tgt, tgt_lg, cvec, accepted, gamma, a_conf=a.a_conf)
            # PHASE A IS SHARED. The KV-cache fill runs once per sequence, outside this loop, and
            # every position's graph traces back through it -- so the first backward frees it and
            # the second raises "Trying to backward through the graph a second time". retain_graph
            # keeps phase A alive across the positions that share it; only the last may free it.
            (l / len(sel)).backward()      # graphs are independent now; see phase A note
            tot_loss += float(l.detach()); nb += 1
            for k2 in nparts: nparts[k2] += parts[k2]
        gn = torch.nn.utils.clip_grad_norm_([p for _, p in tp], 1.0)
        ngrad = sum(1 for _, p in tp if p.grad is not None)
        opt.step()
        loss_sum = torch.tensor(tot_loss)
        nparts = {k2: v / max(nb, 1) for k2, v in nparts.items()}
        print(f"[train] step {step}: T={T} positions={len(sel)} "
              f"loss={tot_loss/max(nb,1):.4f} ce={nparts['ce']:.4f} tv={nparts['tv']:.4f} "
              f"conf={nparts['conf']:.4f} "
              f"grad_norm={float(gn):.4f} tensors_with_grad={ngrad}/{len(tp)}", flush=True)
        _hist.append({"step": step, "T": T, "positions": len(sel),
                      "loss": tot_loss / max(nb, 1), "ce": nparts["ce"], "tv": nparts["tv"],
                      "conf": nparts["conf"], "grad_norm": float(gn), "lr": lr_now})
        if ngrad == 0:
            sys.exit("[train] FAIL: no gradients reached the trainable tensors")
        if a.smoke:
            break

    if _dbg["n"]:
        print(f"[diag] draft-argmax == TARGET-argmax : {100*_dbg['agree']/_dbg['n']:.1f}%  "
              f"| == ground-truth token: {100*_dbg['hit']/_dbg['n']:.1f}%")
        print(f"[diag] mean top-1 prob  target {_dbg['tgt_top1_p']/_dbg['m']:.3f}  "
              f"draft {_dbg['drf_top1_p']/_dbg['m']:.3f}   "
              f"(TV near 1 with a PEAKED draft => misalignment; with a FLAT draft => genuine)")
    # The save used to sit behind an unconditional `return` left over from stage 0, when the only
    # question was whether a backward pass ran at all. Reaching a real session with it still there
    # would have burned the capture and the training run and produced no weights -- and the log
    # would have said "STAGE0 OK".
    if a.smoke:
        print("[train] smoke run: not saving"); return

    os.makedirs(a.out, exist_ok=True)
    from safetensors.torch import save_file
    save = {k: v.detach().to(torch.bfloat16).cpu() for k, v in params.items()}
    save_file(save, os.path.join(a.out, "mtp_trained.safetensors"))
    print(f"[train] saved {len(save)} trained tensors -> {a.out}/mtp_trained.safetensors", flush=True)

    if a.metrics_out:
        first = _hist[0] if _hist else {}
        last = _hist[-1] if _hist else {}
        tail = _hist[-max(1, len(_hist) // 10):] if _hist else []
        met = {
            "capture": a.capture, "n_sequences": len(rows), "steps": len(_hist),
            "block": a.block, "lr_peak": a.lr, "warmup_steps": _nwarm,
            "pos_per_seq": a.pos_per_seq, "a_conf": a.a_conf,
            "trainable_params": int(n_train), "n_tensors": len(save),
            "loss_first": first.get("loss"), "loss_last": last.get("loss"),
            "loss_tail_mean": (sum(h["loss"] for h in tail) / len(tail)) if tail else None,
            "ce_last": last.get("ce"), "tv_last": last.get("tv"), "conf_last": last.get("conf"),
            "diag_draft_eq_target_argmax_pct": (100.0 * _dbg["agree"] / _dbg["n"]) if _dbg["n"] else None,
            "diag_draft_eq_ground_truth_pct": (100.0 * _dbg["hit"] / _dbg["n"]) if _dbg["n"] else None,
            "diag_target_top1_p": (_dbg["tgt_top1_p"] / _dbg["m"]) if _dbg["m"] else None,
            "diag_draft_top1_p": (_dbg["drf_top1_p"] / _dbg["m"]) if _dbg["m"] else None,
            "history": _hist,
        }
        os.makedirs(os.path.dirname(os.path.abspath(a.metrics_out)) or ".", exist_ok=True)
        json.dump(met, open(a.metrics_out, "w"), indent=1)
        print(f"[train] metrics -> {a.metrics_out}  "
              f"(loss {met['loss_first']:.4f} -> {met['loss_last']:.4f}, "
              f"tail mean {met['loss_tail_mean']:.4f})", flush=True)
    print("[train] TRAIN OK")


if __name__ == "__main__":
    main()
