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

---

## 2026-08-06 (later) — instrumentation unblocked: bandwidth measured, `ncu` unblocked

User granted sudo and push access. Both Gate-R1 open items #1 and #2 resolved.

### Finding 8 — achievable bandwidth is 240 GB/s, not the inherited ~200

- **Finding.** `tools/bw_probe.cu` (grid-stride `float4` streaming read, 320 blocks × 256
  threads, buffer ≫ L2, reduced so nothing is DCE'd) measures **240.7 / 241.5 / 242.8 GB/s** at
  1 and 4 GiB, and **212.3 GB/s** at 8 GiB. That is **88–89% of the 273 GB/s spec peak**.
- **Root cause of the old figure.** ~200 GB/s was a planning number carried from prior projects,
  never measured on this build. It was ~17% pessimistic.
- **Consequence.** The AR wall moves **17.85 → 21.42 tok/s**. Today's measured 7.89 tok/s is
  **37% of achievable** (not 32% of spec peak), and the 70–80% target band becomes
  **15.0–17.1 tok/s** against a 21.4 wall. Previously the target and the wall coincided; now
  there is real headroom above the band. DSpark band moves to **22–36 tok/s, centred ~28**.
- **Caveat.** The sweep ran while the 100 GiB checkpoint download was writing. The 8 GiB point is
  contended; re-run idle before treating 240 as final. If anything this underestimates.

### Finding 9 — `ncu` is unblocked, and its memory metric is a trap on this chip

- **Finding.** `sudo /usr/local/cuda-13.0/bin/ncu` collects counters successfully. (Plain
  `sudo ncu` fails `command not found` — not on root's `PATH`.) Persisted
  `NVreg_RestrictProfilingToAdminUsers=0` to `/etc/modprobe.d/nvidia-profiler.conf` for
  unprivileged profiling **after the next reboot**; not needed to proceed.
- **The trap.** Profiling `stream_read` — independently measured at **244 GB/s ≈ 89% of spec
  peak** — `ncu` reports `Memory Throughput 30.26%`, exactly equal to `L2 Cache Throughput
  30.26%`, with `Average MC Channel Active Cycles = (!) nan` and `dram__cycles_active` missing.
- **Root cause.** Thor has no discrete DRAM and exposes no memory-controller counters, so
  SpeedOfLight's "Memory Throughput" degenerates to L2 throughput. `ncu` then emits the advisory
  "memory bandwidth below 60% of peak typically indicates latency issues" — wrong here.
- **Consequence, and why this mattered to catch now.** The prior project's planned first step was
  *"unblock ncu; confirm Memory% vs Compute% per kernel to prove the software-dequant
  compute-bound hypothesis"*. Run naively that would have reported ~30% memory throughput on
  **every** kernel including already-optimal ones, and sent the optimisation loop chasing
  phantom latency problems. **Bandwidth utilisation stays an analytical quantity on Thor**
  (byte model ÷ wall-clock). `ncu` remains the right tool for `Compute (SM) Throughput`,
  occupancy, warp-stall reasons, and cache hit rates — which is what the compute-bound-dequant
  hypothesis actually needs to be tested against.
- **Recorded as a standing rule** in `README.md`, `STATUS.md` and `HARDWARE.md` §3.

### Checkpoint acquisition — identity and integrity verified (directive §13.6)

- All 48 shards + metadata downloaded, no `.incomplete` files, 101 GiB on disk (129 GiB free).
- **`sha256sum -c SHA256SUMS`: 79/79 OK, 0 failures.**
- **Cross-check against the remotely-harvested headers**: 45,821 tensors on both sides, keys
  identical, **0 dtype/shape mismatches**, byte total reconciles to 107,803,320,952. So the
  Phase-1 roofline — computed before the download finished — is validated against the real
  artifact, not merely against repo metadata.
- Provenance re-confirmed from the shipped manifest: `source_model:
  deepseek-ai/DeepSeek-V4-Flash-0731`, `source_revision: 9e165c30…`, structural validation
  `status: pass`. This is the 0731 lineage, not the pre-0731 preview and not an NVFP4 requant.

**Process note (small, but it cost a confusing minute):** an `until ! pgrep -f "sha256sum -c"`
wait loop **never terminates**, because the loop's own shell command line contains the pattern
and `pgrep` matches itself. Verify completion from the output file, or use `pgrep -f` with a
pattern that cannot match the waiter.

### Finding 10 — `reasoning_effort` and `thinking_mode` are orthogonal, and one of them wrecks the prefix cache

- **Finding.** Directive §9 frames the model as using "`low`/`high`/`max` reasoning_effort, **not**
  a binary thinking toggle". `encoding_dsv4.py` has **both**: `thinking_mode ∈ {chat, thinking}`
  (binary) *and* `reasoning_effort ∈ {low, high, max}` — and `reasoning_effort` has **no effect**
  in chat mode.
- **Consequence for the server.** Both must be exposed per-request; neither substitutes for the
  other. Details in `CHAT_FORMAT.md`.
- **Design consequence, worth knowing before Phase 6.** `reasoning_effort` is realised *purely as
  a text prefix prepended before the system message* (default `low` = no prefix at all). Because
  it sits at the very front of the prompt, **changing `reasoning_effort` invalidates the entire
  prefix cache**, while changing `thinking_mode` only perturbs the tail. Prefix caching is called
  out in the directive as a high-leverage Phase-6 feature; this interacts with it directly.
- **Secondary.** Tool calls are **DSML**, not JSON, and each parameter carries
  `string="true|false"` — the only signal separating the literal string `"5"` from the number `5`.
  Neither gemma's nor Laguna's parser has this shape; §9's "do not assume it matches gemma's or
  Laguna's" is correct and load-bearing. Four golden encode vectors ship in `encoding/tests/`,
  which gives Gate S1 a byte-exact acceptance test.

---

## 2026-08-06 (later still) — Gate K PASS, Gate L1 PASS on the 0731 checkpoint

### Gate K — PASS (20/20 units)

Sources copied from `~/dspark-cuda-reap-finetune` (left read-only). Goldens regenerated from
scratch in a CPU-torch container (`vllm-dflash-thor:ddtree`, plain `docker run`, never
`--runtime nvidia`). All pass: `fp8_block_gemm` (2e-5), TC fp8 mma (cosine 1.000000, 18.09x over
the scalar path), `hc_sinkhorn` (1e-7), `sparse_attn` (3.7e-3, bf16 rounding), `rope` (2.4e-7),
`rmsnorm` (7e-7), `act_quant`, `ogroup_gemm`, **`fp4_gemm` (MXFP4) 1e-5**, `router`
(idx_mismatch=0), `moe` — per-token, batched, device-route and batched+TC all cosine
**1.0000000** — `hc`, `compressor` (+overlap, +full, +rotate), `hadamard`, `index_score`,
`act_quant_fp4`, `indexer`, `yarn`.

**The MXFP4 path passing unchanged is the concrete confirmation of Finding 3.** No new dequant
kernel was needed or written.

**Fix required to build at all:** the prior repo's `scripts/build_gate.sh` omitted
`kernels/dscratch.cu` and no longer linked (`undefined reference to g_arena*`). It had bit-rotted
after the arena was introduced. Repaired here.

### Gate P0/A1 addendum — structural validation on the real checkpoint

`tools/inspect_weights.cpp` (pure C++/mmap, no CUDA) first reported **16 MISSING** — all in MTP.
Root cause: its expectations still encoded the **180B**'s plain nextn head
(`enorm`/`hnorm`/`e_proj`/`h_proj`/`norm`/`hc_head_fn` on every stage), which does not exist in
0731. Rewrote them for the real DSpark chain. Now:

```
shards=48  tensors=45821
TOTAL 107.803 GB = 100.400 GiB   (matches index total_size)
structural validation: checked=1443  missing=0  shape_mismatch=0   RESULT: PASS
```

Every inferred DSpark shape was confirmed by the checkpoint:
`mtp.0.main_proj [4096, 12288]` — exactly `3 × DIM`, i.e. the three tapped backbone layers
(40/41/42) concatenated, not summed or pooled before projection — and
`mtp.2.confidence_head.proj [1, 4352]` = `DIM + HEAD_DIM/2`.

### Gate L1 — PASS

```
integrated=1 hostRegisterSupported=1 canUseHostPtrForRegMem=1 canMapHostMem=1
loaded 100.400 GiB, 45821 tensors -> device pointers
GPU read of layers.1.attn.wq_a.weight[:32]: MATCH file bytes
Full-model weight->device load (single-copy pread, 100.4 GiB): PASS
```

- Loaded byte count equals the index's `total_size` exactly.
- **Zero-copy**: Thor reports `integrated=1` and the weights are host-registered and mapped, so
  there is no second 100 GiB device allocation. This is what makes the model fit at all.
- Peak system memory ~102 GiB used of 122, bottoming out at **20 GiB available** — consistent
  with `ROOFLINE.md`'s predicted 16.6 GiB headroom plus the process's own overhead.
- Ran from a **cold page cache** (`drop_caches` first) so the timing is worst-case, and detached
  per the hard rule.

**Note on the "cached second start under 60 s" half of the directive's L1 criterion:** the loader
deliberately does single-copy `pread` + `posix_fadvise(DONTNEED)` so it does *not* leave the
weights in page cache — that is the whole reason peak stays near 100 GiB instead of doubling.
A warm restart therefore re-reads from disk by design. Making startup fast is a *separate*
optimisation (and would trade against the memory constraint), not something the current loader
regresses on. Recorded rather than silently treated as met.

---

## 2026-08-06 — Gate G8 PASS: the full model runs correctly on 0731, and the anchor holds

### The run

```
[decode] loaded 100.40 GiB, 45821 tensors
[decode] structs built. mem 111.3/122.8 GiB
[decode] prefill 5 positions... done.
  step 0 pos 5 -> token 11111  (125.6 ms warmup)
  step 1 ... step 7                (124.0 - 135.8 ms)
[decode] generated: 11111 16 455 6102 294 16603 344 29168
[decode] WARM decode: 128.1 ms/tok = 7.81 tok/s  (M=1 steady state, 7-step avg)
[spec-verify] M=5 verify in ONE forward: 336.1 ms  -> 1.91x if all accepted
[spec-verify] MATCH 5/5 -> PASS   (M=K verify == K sequential decodes)
[decode] mem 111.9/122.8 GiB
```

Detokenised: prompt `"The capital of France is"` →

> **"The capital of France is Paris. The capital of Spain is Madrid"**

**The full 43-layer stack — MLA + KV compressor + DSA indexer + hyper-connections/Sinkhorn +
MXFP4 160-expert MoE + hash routing + shared expert — is numerically correct end-to-end on the
0731 checkpoint.** Gate G8 PASS.

**Stale-gate correction.** The run printed `GATE FAIL` while being correct. The binary hardcoded
`argmax == 270`, which was the 180B project's expectation for a *different* probe prompt
("The capital of France is the powerhouse of" → `" the"`). Our prompt is
`"The capital of France is"`, whose correct continuation is `" Paris"` = **11111** — exactly what
the model produced. Replaced the hardcoded constant with a `DSV4_EXPECT` env var (default 11111)
so the assertion travels with the prompt instead of silently rotting. Left as a caution: a gate
that encodes a *prompt-specific* answer as a global constant will eventually report a false
failure on correct code, which is as harmful as a false pass.

### Finding 11 — the ROOFLINE §3 anchor transferred to within 1.1%

| | ms/tok | tok/s | effective BW | % of 240 achievable |
|---|---|---|---|---|
| 180B, prior project | 126.7 | 7.89 | 88.4 GB/s | 36.8% |
| **0731, measured here** | **128.1** | **7.81** | **87.5 GB/s** | **36.4%** |

The prediction was that the identical `B_tok` (11.202 GB, matched component-by-component) makes
the prior measurement a *direct* anchor rather than a scaled one. It came in **1.1% off**. This
retires directive risk §13.3 ("the speed projection is weakly anchored") empirically rather than
by argument, and it confirms the whole ported stack behaves as the byte model says it should.

**Base AR is therefore 7.81 tok/s at 36.4% of achievable bandwidth, against a 21.4 tok/s wall.**
The 15–19 tok/s band in `ROOFLINE.md` stands, and the work to reach it is the Phase-7 ladder.

### Finding 12 — the `c_v(5) = 2.6x` anomaly REPRODUCES, and now it is ours to fix

`ROOFLINE.md` §5 flagged an unexplained inherited discrepancy: the prior project measured an M=5
verify at 2.6x an M=1 decode where the byte model predicts 2.120x. **It reproduces exactly here:
336.1 ms / 128.1 ms = 2.62x.**

Quantified against our own measured bandwidth:

```
ideal-dedup B_verify(5)          = 23,754 MB  -> 271.8 ms at the measured 87.5 GB/s
actual                                          336.1 ms
implied bytes moved              = 29,375 MB
EXCESS                           =  5,622 MB   (+23.7%)
```

So the verify pass moves ~5.6 GB more than an expert-deduplicated verify should. That is close to
(though not exactly) one extra full pass over the top-6 routed experts (3,449 MB) plus change,
which is consistent with the standing hypothesis — **the M=K verify is re-reading expert weights
per (token, expert) instead of once per activated expert.** The prior project's research doc
named this and left an explicit unresolved action: *audit whether `k_grouped_w4a8` dedups by
expert*. That audit is now the top open item, and unlike before it is measurable end-to-end here.

Worth noting what this is **not**: it is not a correctness problem. `MATCH 5/5` — the M=K verify
produces bit-identical tokens to 5 sequential decodes. It is purely wasted bandwidth.

If the dedup lands, `c_v(5)` goes 2.62 → ~2.12 and, at unchanged acceptance, the
`S(k)` table's k=2–3 optimum sharpens. Note the run already reports **1.91x if all 5 accepted**,
which is the ceiling, not the expectation — realised speed-up depends on the acceptance rate α,
still unmeasured.

### Finding 13 — the inherited "MoE expert-union dilation" hypothesis is REFUTED by code inspection

`ROOFLINE.md` §5 carried forward the prior project's leading explanation for `c_v(5) = 2.6x`,
together with its explicit unresolved action: *"audit whether `k_grouped_w4a8` dedups by expert"*.

**Audit done. It does.** `kernels/tc_moe_gemm.cu`:

- `k_build_tiles` (:213) walks the counting-sort offsets and, for each expert `e`, emits
  `ceil(rows_e / 16)` tiles carrying `(tile_e = e, tile_row0)`.
- `k_grouped_w4a8_e8m0_kernel` (:319) reads `wptr[e]` **once per tile**, and partitions that
  expert's weight across `gridDim.x = N/8` n-blocks by byte range (`wb = wprE + n_block*kg8*512`),
  so every weight byte is read exactly once per tile.
- At M=5 no expert receives more than 5 gathered rows, so `ceil(rows_e/16) == 1` — **exactly one
  weight read per activated expert.**
- The `g_moe_gemv` alternative (a scalar-nibble GEMV) is **off by default** (`src/decode.cu:96`,
  needs `MOE_GEMV=1`), so M=1 and M=K take the *same* deduplicated tile path.

Expert reads therefore go 6 (M=1) → `|union|` ≈ 28 (M=5), which is precisely the
`E[|union|](5) = 27.83` the model assumes. **The MoE is behaving exactly as `ROOFLINE.md` §5
prices it. It is not the source of the excess.**

**So where does the ~64 ms / ~5.6 GB-equivalent excess come from?** The remaining candidate is the
term the model does *not* price and that this project flagged from the start as its one real blind
spot: **work that scales with M while weight traffic does not** — the DSA lightning indexer and
sparse attention, which run a top-512 selection and an irregular gather **per query position**, so
5 positions cost ~5x the *compute* while re-reading the same weights once.

`ROOFLINE.md` §5 caveat 2 guessed this would cut in our favour (better arithmetic intensity at
M=k). **That guess looks wrong**, and in hindsight it was the optimistic reading: at 36% of
achievable bandwidth we are substantially compute/latency-bound already, so a 5x compute term is
not hidden under the memory stream — it is exposed.

**Consequence for the priority order.** `ROOFLINE.md` §6 demoted DSA to #6 *on weight bandwidth*
while explicitly keeping it on the list because "its latency contribution is unmeasured and its
verify-step behaviour is the project's one real blind spot". That caveat is now carrying the
weight: **for the speculative path specifically, DSA is the prime suspect, not the MoE.** Base-AR
priorities (MLA #1, MXFP4 MoE GEMV #2) are unaffected — this is a verify-step finding.

**Next measurement, and it is now cheap to get right.** Per-kernel wall-clock at M=1 vs M=5,
plus `ncu`'s `Compute (SM) Throughput` on the indexer/sparse-attention kernels — which is the
metric that IS honest on Thor (Finding 9). If indexer+sparse-attn time scales ~5x while the MoE
and MLA GEMMs stay flat, the diagnosis is confirmed and the fix is a batched indexer that shares
selection work across verify positions.

**Minor, found in passing:** `kernels/moe.cu:231` calls `tc_ensure_repacked` for all
`nr = 160` experts × 3 matrices on *every* MoE invocation. It is idempotent (a host-side
`std::set<void*>` probe, `tc_moe_gemm.cu:193`), so it does no device work after warm-up — but it
is 480 host-side set lookups per layer, ~20,640 per decode step, on the critical path. Order ~1 ms
of the 128 ms step. Worth hoisting, not urgent, and logged so it is not rediscovered.

---

## 2026-08-06 — Phase 5 / Gate D1: DSpark runs on the embedded heads; the K-sweep refutes my own hypothesis

### Gate D1 — DSpark spec-decode WORKS, and lands at parity

```
[spec] using EMBEDDED DSpark heads from the main checkpoint (no extra memory)
[spec] NSTAGE=3 head-experts=160 BLK=5
[spec] head built. mem 113.8/122.8 GiB
  verify 1..8: accepted 1,2,0,1,4,1,4,4 of 4  (+correction each)
[spec] generated 25 tokens over 8 verifies: mean tokens/verify = 3.12 (block=5, max 5)
[spec] SPEC-DECODE: 129.1 ms/tok = 7.75 tok/s  (vs base 128.2 ms/tok -> 0.99x)
```

**The memory-neutral rewire worked**: `NSTAGE=3`, `head-experts=160`, discovered from the main
checkpoint, **no second WeightStore** — total 113.8 GiB vs 111.2 GiB for base decode, i.e. the
DSpark path costs +2.6 GiB of activations rather than the +6.5 GiB a duplicate store would have
added on top. That would not have fit alongside the KV and arena.

**Acceptance is not the problem.** 3.12 tokens per verify out of a max of 5 is a *good* rate for an
un-fine-tuned head (α ≈ 0.7). **Cost is the problem** — see Finding 15.

### Finding 14 — the K-sweep REFUTES the DSA hypothesis. It was mine, and it was wrong.

Finding 13 refuted the prior project's MoE explanation and nominated DSA (top-512 select +
irregular gather **per query position**) as the successor. The experiment was designed to be
decisive: time the verify per layer flavour, since pure-sliding layers have neither compressor nor
indexer and are the control.

```
per-flavour growth K=1 -> K=5
  slide (no compressor, no indexer) : 2.754x   <- control
  r128  (compressor, NO indexer)    : 2.573x
  r4    (compressor + DSA indexer)  : 2.488x
```

**The DSA layers grow the SLOWEST.** The prediction was that `r4` would grow materially faster than
`slide`; it grows *slower*, and the three flavours are within 10% of each other. DSA is not the
cause of the verify excess. Recorded plainly because a hypothesis that survives on reputation is
worse than no hypothesis: two successive explanations for `c_v` have now been wrong, and both were
plausible-sounding. The layer-flavour split is what settled it, not argument.

### Finding 15 — the real mechanism: a step-function M>=2 penalty, because the M=1 fast paths vanish

Fitting the sweep against the layer-only byte model (`B_fixed_layers + |union|(K)/6 * B_expert`,
priced at the *measured* 87.5 GB/s):

| K | measured | byte model | excess |
|---|---:|---:|---:|
| 1 | 115.94 ms | **115.92 ms** | **+0.02** |
| 2 | 225.60 | 153.87 | +71.73 |
| 3 | 255.45 | 190.38 | +65.07 |
| 4 | 272.82 | 225.53 | +47.29 |
| 5 | 293.96 | 259.37 | +34.59 |

**At K=1 the byte model is exact to 0.02 ms.** The entire anomaly is a *step* at K≥2 — step deltas
are `+109.7, +29.8, +17.4, +21.1`, i.e. K=1→2 nearly doubles the cost while K=2→5 adds little.
That is the signature of a fixed penalty, not a per-position cost.

**Root cause, confirmed in code.** Two dense paths have M=1-only fast kernels:

- `kernels/fp8_block_gemm.cu:99` — `if (g_tc_fp8 && M == 1 …) fp8_gemv_m1_kernel`; at M≥2 it falls
  through to `tc_fp8_gemm`, an **m16-tile mma**.
- `kernels/mla_attn.cu:268` — `if(bs==1 …) ogroup_gemv_fp8_kernel`; at bs>1 it falls through to
  `tc_ogroup_fp8_kernel`, also m16, **plus** an extra `k_f2h` conversion pass and a `dmalloc`/`dfree`.

At M=1 those GEMVs are bandwidth-bound and optimal (hence the exact model fit). At M=2–5 the m16
tile computes 16 rows of mma to use 2–5 — 3–8x wasted mma throughput, and it is mma-latency bound,
not bandwidth bound. The penalty is therefore *flat* across M=2..16, which is exactly the measured
shape.

**Why the excess shrinks with K:** the model assumes independent routing. Real routing is
correlated, so the true expert union is smaller than `|union|(K)`. Solving at K=5 implies a real
union of ~22 experts against the modelled 27.8 — 74% of slots distinct, entirely plausible for
adjacent tokens. So two effects overlap: a constant ~72 ms M≥2 penalty, partly masked by the model
over-charging for experts at larger K. **`ROOFLINE.md` §5's `E_frac` is conservative** — the real
expert union is cheaper than modelled, which is a point in speculation's favour.

**Do NOT "fix" this with an M=K GEMV.** The prior project already A/B'd exactly that and it was
slower (334 → 362 ms); the code carries the negative result at `fp8_block_gemm.cu:101-105`
(`GEMV_MK=1`, default off). Reason: a GEMV does M scalar dots per weight read, while the mma reads
the weight once and does M×N. The fix is a **small-M tile shape** (fewer wasted rows per mma), not
a different algorithm class.

### Finding 16 — the flat M≥2 penalty INVERTS the k* guidance: bigger K is better, not smaller

`ROOFLINE.md` §5 predicted `k* = 2–3` from the pure byte model. The measured curve says otherwise,
precisely *because* of the step penalty — once you have paid it at K=2, additional verify positions
are cheap:

| K | verify+lm_head | per token if all accepted | speed-up |
|---|---:|---:|---:|
| 1 | 128.2 ms | 128.2 | 1.00x |
| 2 | 237.9 | 119.0 | 1.08x |
| 3 | 267.8 | 89.2 | 1.44x |
| 4 | 285.1 | 71.3 | 1.80x |
| 5 | 306.3 | **61.3** | **2.09x** |

**k\* is at or above `dspark_block_size = 5`, not 2–3.** The `S(k)` table in `ROOFLINE.md` §5 is
superseded by measurement for this engine. Note the *reason* is a defect: if Finding 15's penalty
is fixed, small K gets cheaper and the optimum moves back down. Both facts should be carried
together — **k\* depends on whether the small-M GEMM is fixed**, and the sweep must be re-run after
any change there.

### Finding 17 — the DSpark draft head is ~6x off its own roofline, and that is what eats the win

Decomposing the measured 425 ms spec step: K=5 verify (294 ms) + lm_head (~12 ms) leaves
**~119 ms for the draft head**. Its byte model — 3 DSpark blocks (MLA + top-6 of 160 experts each),
the shared `lm_head`, and the rank-256 markov head — is ~1.75 GB → **~20 ms** at 87.5 GB/s.
**The draft is ~6x off roofline**, and it costs almost as much as a whole M=1 decode step.

That is decisive for the economics:

| scenario | ms/verify | ms/token at a=3.12 | vs base |
|---|---:|---:|---:|
| **measured** | 425 | 136.2 | **0.94x** (parity) |
| draft cost halved | 366 | 117.2 | 1.09x |
| draft at its roofline | 314 | 100.7 | **1.27x** |

**Why it is slow is not mysterious:** `kernels/dspark_real.cu` and `kernels/dspark_attn.cu` were
written for the prior project's *correctness* milestone (Gate 2-real) and **never went through the
optimization loop** — every one of the 9 banked decode optimizations targeted the main path. The
draft also runs at M=BLK=5, so it pays Finding 15's penalty three times over, once per stage.

**Gate D1 verdict: DSpark is correct, integrated, memory-neutral, and at parity (0.99x).** It is
not yet a win, and the reason is entirely cost — acceptance at 3.12/5 is already adequate. Two
levers, in order: (1) small-M dense GEMM (Finding 15) — helps verify *and* draft; (2) put the
DSpark kernels through the optimization loop (Finding 17).
