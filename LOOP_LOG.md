# LOOP_LOG.md

One entry per gate / iteration: **finding → root cause → fix → why**. Explainability is
permanent (Constitution Art VI·5). Speed A/Bs go in `OPTIMIZATION_LOG.md`.

---

## 2026-08-06 — Phase 0/1: Gates H1, P0, R1

**Gate H1 — PASS.** Thor `sm_110a`, CUDA 13.0 (nvcc V13.0.48), driver 580.00, L4T R38.4,
aarch64 / 14 cores, 122 GiB unified / **117 GiB available**, 230 GiB disk free. `HARDWARE.md`.
Note `nvidia-smi` reports `Memory-Usage: Not Supported` on Thor — `free` is the authoritative
memory instrument.

**Method note.** All Phase-0/1 analysis was done from **safetensors headers fetched by HTTP range
request** (~9 MB) rather than waiting on the 100.4 GiB download. `tools/fetch_headers.py` reads
the 8-byte header length then range-reads the JSON header of each of the 48 shards. This made
Gate R1 completable in parallel with the download, and leaves the repo with a committed,
auditable copy of every tensor's dtype and shape.

**Self-check that validates the accounting.** `tools/inventory.py` refuses to report unless the
summed tensor bytes equal `index.json`'s `metadata.total_size` exactly (107,803,320,952). It
does. Independently, the derived activated-parameter count comes out at **13.26 B** against the
model card's "~13B active" — computed from tensor shapes with 2 params per packed I8 byte and
scales excluded. Two independent reconciliations; the byte model is right.

### Finding 1 — the directive's `B_tok` estimate is ~1.6× low, and the AR wall is ~2.6× lower than its anchor

- **Finding.** `B_tok = 11.202 GB/token`, giving an AR wall of 17.85 tok/s at 200 GB/s. The
  directive's §4.2 anchor projected ~46 tok/s.
- **Root cause.** §4.2 blends "13B active params at MXFP4 (~4.3 bits)". Only the **routed
  experts** are MXFP4. They are 85.3% of stored bytes but only **30.8%** of per-token bytes.
  Everything else — MLA, shared expert, compressor, indexer, router, `lm_head` — is FP8/BF16.
  The true blend is **6.76 bits/active-param**.
- **Consequence.** 46 tok/s is above the memory wall and unreachable at any kernel quality. The
  secondary Qwen-3.5-122B active-param-scaling anchor is discarded entirely.
- **Why it matters.** Every downstream target in the directive was priced off that number.

### Finding 2 — the anchor problem solves itself: the prior checkpoint has an identical byte profile

- **Finding.** Running the same accounting over the local `~/models/DeepSeek-V4-Flash-180B`
  shards yields `B_tok = 11.202 GB` — identical component by component (MLA 4599.48 MB, experts
  3449.29, shared 1082.20, lm_head 1059.06, …).
- **Root cause.** 47 of 48 config keys are identical; the 43-layer backbone geometry is the same.
  The checkpoints differ in post-training and in the 3 embedded MTP blocks, neither of which
  changes bandwidth.
- **Consequence.** The prior project's **measured** 126.7 ms/tok (7.89 tok/s) transfers with no
  scaling assumption. Implied efficiency **88.4 GB/s = 32% of peak** — not the 25% the prior
  repo's own docs claim, because they used an undercounted 8.77 GB/tok.
- **Why it matters.** Directive §13.3 called the projection "weakly anchored". It is the
  opposite: same geometry, same box, same toolchain, already measured. Confidence is HIGH.

### Finding 3 — MXFP4 is not new work; the prior checkpoint was already MXFP4

- **Finding.** `layers.5.ffn.experts.0.w1.weight` is `I8 [2048, 2048]` (logical `[2048, 4096]`
  E2M1) with `.scale` `F8_E8M0 [2048, 128]` → **block 32 along K = OCP MXFP4**, 4.25 bits/param.
  The 180B checkpoint's tensors are **byte-identical in layout**.
- **Root cause of the confusion.** DeepSeek's `config.json` says only `expert_dtype: fp4`; the
  prior project's notes wrote that down as "NVFP4". The tensor shapes say otherwise.
- **Consequence.** `~/dspark-cuda-reap-finetune/kernels/moe.cu:25` already walks `KBw = K / 32`
  with e8m0 scales — it *is* an MXFP4 kernel, already gated. **G1 is effectively pre-passed.**
- **Why it matters.** The directive lists this as a new dequant kernel and a `ARCH_DELTA` risk row.

### Finding 4 — there is no REAP expert-ID remapping to replicate

- **Finding.** `reap_plan.json` and `REAP_MANIFEST.json` contain **no retained-expert ID list**.
- **Root cause.** The REAP build used "transferred ranking with router identity alignment":
  `identity_nearest_all = true`, `hash_tid2eid_equal_fraction = [1.0, 1.0, 1.0]`. `gate.weight`
  ships `[160, 4096]`, `gate.bias` `[160]`, expert tensors dense `0..159`, `tid2eid` pre-remapped.
- **Consequence.** G6 loses its remap step and becomes a plain index-exact top-6 check. The
  loader's "bake the compacted expert layout into the offline repack" instruction (§6) is a no-op
  — the checkpoint is already compacted.

### Finding 5 — MLA, not MoE, is the dominant per-token cost

- **Finding.** MLA attention is **41.1% of `B_tok`** (4599 MB) — more than all six routed experts
  combined (3449 MB, 30.8%).
- **Root cause.** Per layer, `wq_b [32768,1024]`, `wo_a [8192,4096]` and `wo_b [4096,8192]` are
  33.5 MB each in FP8, read in full every M=1 step. The *cache* is compressed; the *projections*
  are not.
- **Consequence.** The directive's Phase-7 priority order (DSA first) is inverted: MLA projection
  GEMV bandwidth is #1, MXFP4 MoE GEMV with hardware `cvt.f16x2.e2m1x2` unpack is #2, DSA drops
  to #6 on weight bandwidth. DSA stays on the list because its *latency* term is unmeasured.
- **Why it matters.** This is precisely the trap §4.1 of the directive warned about; the warning
  was correct and the magnitude is larger than it implied.

### Finding 6 — KV is a non-issue, and the inherited prior is ~11× too pessimistic

- **Finding.** From `compress_ratios` (2 pure-sliding, 21 × ratio-4, 20 × ratio-128), `head_dim`
  512 and `index_head_dim` 128: **3.36 KiB/token at fp8** → 3.21 GiB at 1M context, ~105 MiB at 32K.
- **Root cause of the discrepancy.** The directive's 37.7 KB/token prior came from the 180B
  project's vLLM-era measurement, most likely from a runtime that did not implement the
  ratio-4/ratio-128 compressor and therefore paid full per-token KV.
- **Secondary finding.** The prior CUDA engine allocates KV in **fp32**
  (`src/decode.cu:105-108,243`). Moving to fp8 is a 4× win on both traffic and footprint and is
  memory-*negative*, which the hard memory constraint actively rewards.

### Finding 7 — the correctness-oracle risk (directive §13.5) is already retired

- **Finding.** SGLang does not exist on this box; the reference `inference/` needs `tilelang` +
  `fast_hadamard_transform` (neither builds on `sm_110a`) and `transformers>=5.0`.
- **Resolution, already built.** `~/dspark-cuda-reap-finetune/ref/` reimplements all five tilelang
  kernels plus Walsh-Hadamard in pure torch, so the **verbatim** reference `model.py` runs CPU-only
  with no unbuildable dependency. That is the oracle. No ARM64 runtime port is required.
- **Carry-over caveat.** `--runtime nvidia` Docker containers wedge the device on this box
  (D-state `runc`, `docker rm -f` hangs). Oracle work runs CPU-only in a plain container.

### Gate R1 — PASS

`ROOFLINE.md` complete: measured `B_tok`, AR wall, both anchors with confidence explicitly
weighted, an `E_frac(k)` / `c_v(k)` / `S(k)` table computed by `tools/verify_cost.py`, and the
DSA-verify-cost unknown flagged for empirical resolution in Phase 5.

**Speculation read-out worth recording now:** 69.2% of `B_tok` is **k-invariant** (read once per
verify pass), so `c_v` grows sub-linearly — but the `S(k)` table still puts **k\* at 2–3, not 7**,
and caps speculation-alone at ~1.5–1.9× for plausible α. The vLLM recipe's
`num_speculative_tokens: 7` is a bad prior for this engine.

**Open, carried into Phase 2+:** achievable streaming BW unmeasured · `ncu` blocked by
`ERR_NVGPUCTRPERM` (needs sudo) · DSA verify cost unmodelled · the inherited `c_v(5)=2.6×` vs
2.12× discrepancy · 0731 acceptance rate α.
