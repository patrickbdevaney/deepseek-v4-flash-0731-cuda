"""kernel.py — plain-PyTorch shim for the checkpoint's `inference/kernel.py`.

WHY THIS EXISTS. The checkpoint ships an authoritative PyTorch model (`inference/model.py`) with
`DSparkBlock`, `DSparkAttention`, `DSparkMarkovHead` and `DSparkConfidenceHead` — the exact modules
S5 fine-tunes. Reimplementing them would be ~500 lines of careful code with a high chance of a
silent layout mismatch against the checkpoint. But `model.py` imports `kernel.py`, which needs
`tilelang==0.1.8` and `fast_hadamard_transform`, and the Thor container has torch 2.10.0 + CUDA and
NEITHER. Installing a TileLang compiler stack on sm_110a is an unbounded dependency that could
consume the whole Stage-0 budget and teach us nothing about the recipe.

So: keep the authoritative model, replace the six functions it imports. Six functions is far less
code than one model, and every line of the model stays the checkpoint's own.

WHAT IS DELIBERATELY DIFFERENT FROM THE REAL KERNELS
  * Everything is differentiable. The real kernels are inference-only; training needs autograd
    through the frozen expert path as well as the trainable one.
  * `fp8_gemm`/`fp4_gemm` are never reached in this configuration: the loader dequantises all
    weights to bf16, so `model.linear()` takes its `F.linear` branch. They are implemented anyway
    (as a plain matmul against an already-dequantised weight) so that a partially-quantised load
    fails loudly rather than silently taking a wrong path.
  * `act_quant(..., inplace=True)` is a REAL simulated quant-dequant, not a no-op. The engine
    applies `act_quant_fp8sim`/`fp4sim` to the KV path, and F89 measured that this engine's
    acceptance is extremely sensitive to exactly this kind of numeric detail — a 28% acceptance
    swing from an accumulation-order change alone. Skipping the simulation would train the head on
    features the server never produces.
"""
import torch
import torch.nn.functional as F

FP8_MAX = 448.0          # e4m3 finite max
FP4_LEVELS = None        # built lazily on first use, on the right device


# ---------------------------------------------------------------- act_quant
def _blockwise_absmax(x, block_size):
    N = x.size(-1)
    assert N % block_size == 0, f"{N} % {block_size} != 0"
    xb = x.reshape(*x.shape[:-1], N // block_size, block_size)
    return xb, xb.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    """Block-wise FP8 (e4m3) quantisation. inplace=True does fused quant+dequant back in place."""
    xb, amax = _blockwise_absmax(x, block_size)
    scale = amax / FP8_MAX
    if scale_fmt is not None:                       # MXFP: scales rounded to a power of two
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    q = (xb / scale).clamp(-FP8_MAX, FP8_MAX)
    # Round to the e4m3 grid by a real cast; this is the part a naive shim gets wrong by rounding
    # to integers instead, which is a different grid entirely.
    q = q.to(torch.float8_e4m3fn).to(xb.dtype)
    if inplace:
        deq = (q * scale).reshape(x.shape).to(x.dtype)
        # STRAIGHT-THROUGH ESTIMATOR. The cast to float8_e4m3fn is NOT differentiable: it returns a
        # tensor with no grad_fn, so `x.copy_(deq)` wrote a detached value into the graph and every
        # gradient downstream vanished -- which is exactly the "element 0 does not require grad"
        # that blocked stage 0. Forward value stays the quantised one; gradient passes through as
        # identity, which is the standard quantisation-aware-training treatment and the only way a
        # simulated-quant step can sit inside a trained graph at all.
        # `.data.copy_` NOT `.copy_`: the caller ignores our return value and keeps using its own
        # tensor, so we must mutate in place -- but a tracked in-place write bumps the version
        # counter of a tensor autograd still needs, which is the
        #   "[CUDABFloat16Type [1,5,64,512]] ... is at version 2; expected version 0"
        # failure. Writing through .data changes the VALUE without adding a graph node, so the
        # forward sees quantised activations and the backward sees identity. That is the straight-
        # through estimator, done in the only way this call convention allows.
        x.data.copy_(deq)
        return x
    return q.reshape(x.shape).to(torch.float8_e4m3fn), scale.squeeze(-1).to(scale_dtype)


def fp4_act_quant(x, block_size=32, inplace=False):
    """Block-wise OCP MXFP4 (E2M1) quantisation with an E8M0 (power-of-two) scale."""
    global FP4_LEVELS
    if FP4_LEVELS is None or FP4_LEVELS.device != x.device:
        FP4_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                                  device=x.device, dtype=torch.float32)
    xb, amax = _blockwise_absmax(x, block_size)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 6.0)))     # E8M0: power of two only
    q = (xb / scale)
    mag = q.abs().unsqueeze(-1)                       # (..., 1) broadcasts against FP4_LEVELS (8,)
    idx = (mag - FP4_LEVELS).abs().argmin(dim=-1)     # nearest E2M1 level
    deq = torch.sign(q) * FP4_LEVELS[idx] * scale
    deq = deq.reshape(x.shape).to(x.dtype)
    if inplace:
        x.data.copy_(deq)                    # STE via .data, same reason as act_quant above
        return x
    return deq


# ---------------------------------------------------------------- gemms
def _dequant_guard(weight):
    if weight.dtype in (torch.float8_e4m3fn, torch.uint8, torch.int8):
        raise RuntimeError(
            "fp8_gemm/fp4_gemm reached with a STILL-QUANTISED weight. The training loader is "
            "supposed to dequantise every mtp.* tensor to bf16 so model.linear() takes its "
            "F.linear branch. Reaching here means some tensor was missed, and silently computing "
            "on raw quantised bytes would produce a plausible wrong answer.")


def fp8_gemm(x, s, weight, weight_scale=None, scale_dtype=torch.float32):
    _dequant_guard(weight)
    return F.linear(x.to(weight.dtype) if x.dtype != weight.dtype else x, weight)


def fp4_gemm(x, s, weight, weight_scale=None, scale_dtype=torch.float32):
    _dequant_guard(weight)
    return F.linear(x.to(weight.dtype) if x.dtype != weight.dtype else x, weight)


# ---------------------------------------------------------------- sparse_attn
def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """q (b,s,h,d); kv (b,n,d) — ONE kv head shared across all h (num_key_value_heads == 1);
    topk_idxs (b,s,t) with -1 marking a masked slot; attn_sink (h,) adds to the DENOMINATOR only."""
    b, s, h, d = q.shape
    t = topk_idxs.size(-1)
    # get_dspark_topk_idxs builds its index on the CPU (the real kernel takes a host pointer);
    # torch.gather requires index and source on the same device.
    if topk_idxs.device != kv.device:
        topk_idxs = topk_idxs.to(kv.device)
    idx = topk_idxs.clamp_min(0)                                   # gather needs a valid index
    # kv can arrive fp32 (the cache buffer) while q is bf16; einsum will not mix them.
    kv = kv.to(q.dtype)
    gathered = torch.gather(kv.unsqueeze(1).expand(b, s, kv.size(1), d), 2,
                            idx.unsqueeze(-1).expand(b, s, t, d))  # (b,s,t,d)
    scores = torch.einsum("bshd,bstd->bsht", q, gathered) * softmax_scale
    scores = scores.masked_fill((topk_idxs < 0).unsqueeze(2), float("-inf"))
    # The sink contributes to the denominator only, so fold it in as an extra logit whose value is
    # attn_sink[h] and whose "value vector" is zero — algebraically identical to the kernel's
    # run_sum += exp(sink - max), and numerically stable through the same softmax.
    sink = attn_sink.to(q.dtype).view(1, 1, h, 1).expand(b, s, h, 1)
    full = torch.cat([scores, sink], dim=-1)
    p = torch.softmax(full.float(), dim=-1).to(q.dtype)
    p = p[..., :t]                                                 # drop the sink's weight
    return torch.einsum("bsht,bstd->bshd", p, gathered)


# ---------------------------------------------------------------- hc_split_sinkhorn
def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    """Transcribed from the checkpoint's hc_split_sinkhorn_kernel, NOT inferred from the name.

    My first attempt guessed this and was wrong in four ways, every one of which would have trained
    the head against a different function than the server runs:
      * hc_scale is a 3-VECTOR (one per sub-block), not a scalar;
      * `pre` adds eps after the sigmoid;
      * `post` carries a factor of 2;
      * `comb` starts with a ROW SOFTMAX and a column normalise, and only then runs
        (sinkhorn_iters - 1) further row/column passes.

    Layout of `mixes` (last dim = (2+hc)*hc):
        [0:hc]        -> pre gate
        [hc:2hc]      -> post gate
        [2hc:]        -> comb, row-major comb[j,k] at 2hc + j*hc + k
    """
    b, s_, _ = mixes.shape
    hc = hc_mult
    pre  = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = mixes[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]
    comb = comb.view(b, s_, hc, hc).float()
    comb = torch.softmax(comb, dim=-1) + eps                      # row softmax
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)          # column normalise
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb.to(mixes.dtype)
