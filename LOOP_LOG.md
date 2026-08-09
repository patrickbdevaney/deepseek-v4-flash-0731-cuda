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

---

## 2026-08-06 — Phase 7 opens with a fast harness, and measurement overturns the priority order

### tools/gemm_bench.cu — the loop is now seconds, not 10 minutes

Every measurement so far cost a ~10-minute cold load of a 100 GiB checkpoint, which is fatal to
"one change per measurement". These kernels do not care what the weight bytes *contain*, only
their shapes — so `tools/gemm_bench.cu` allocates the real per-layer shapes (read from
`docs/hdrs`, not invented), runs the real kernels, and reports achieved bandwidth against the
measured 240 GB/s roof. Runs in seconds and is `ncu`-friendly.

### Finding 18 — ROOFLINE §6's priority order is INVERTED by measurement. MLA is not the problem.

| kernel (per layer, M=1) | w_MB | GB/s | % of 240 achievable |
|---|---:|---:|---:|
| MLA `wo_b` [4096, 8192] | 33.55 | **225.5** | **94%** |
| MLA `wq_b` [32768, 1024] | 33.55 | **214.6** | **89%** |
| MLA `wkv` [512, 4096] | 2.10 | 129.7 | 54% |
| MLA `wq_a` [1024, 4096] | 4.19 | 45.1 | **19%** |
| **MXFP4 grouped MoE** (6 experts) | 25.17 | **92.0** | **38%** |

`ROOFLINE.md` §6 ranked **MLA #1** and MXFP4 MoE GEMV #2, reasoning purely from byte share
(41.1% vs 30.8%). **The two MLA matrices that carry ~94% of MLA's bytes are already at 89–94% of
achievable bandwidth — there is essentially nothing to win there.** The MoE, at 38%, is the real
gap, and it is flat at ~92 GB/s across M=1..8 for *both* the mma and GEMV paths.

This is the clearest lesson of the project so far: **byte share ranks what is worth optimising only
if efficiency is uniform. It is not.** The correct ranking is `bytes × (1 − efficiency)`.

### Finding 19 — the prior project's nominated "⭐ top lever" is already spent

`DECODE_GAP_RESEARCH.md` T1.1 was: *"rebuild the FP4 MoE GEMV with HARDWARE x2 unpack —
our rejected fp4 GEMV used SCALAR nibble decode, so the rejection doesn't cover this"*, rated a
step-change.

`kernels/tc_moe_gemm.cu:14`:
```cpp
__device__ __forceinline__ __half2 tcm_fp4x2(unsigned char b){
    __half2_raw r=__nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)b,__NV_E2M1); ...
```
**The hardware unpack is already in the kernel**, and that kernel is the one measuring 92 GB/s.
The lever was banked at some point without the research doc being updated. Carrying it forward
would have burned a cycle re-implementing what exists.

### Finding 20 — the small-N GEMVs lose ~50% to WAVE QUANTISATION, and `ncu` says so directly

`ncu` on `fp8_gemv_m1_kernel` at the `wq_a` shape (the 19% case):

```
Grid Size        128        Block Size 256      Waves Per SM  1.07
Compute (SM) Throughput  41.45%      Memory Throughput 14.20% (= L2; see HARDWARE.md §3)
"1 full wave and a partial wave of 8 thread blocks ... up to 50.0% of the total runtime"
```

The grid is `(N*32+255)/256` blocks — **128 for N=1024**. Against 20 SMs that is 1.07 waves: a
full wave plus a tail of 8 blocks that runs almost the whole kernel duration again for 6% of the
work. The pattern predicts the rest of the table exactly:

| shape | N | grid blocks | waves | measured |
|---|---:|---:|---:|---:|
| `wkv` | 512 | 64 | **0.53** (under one wave!) | 54% |
| `wq_a` | 1024 | 128 | **1.07** | 19% |
| `wq_b` | 32768 | 4096 | ~34 (tail amortised) | 89% |
| `wo_b` | 4096 | 512 | ~4.3 | 94% |

Efficiency tracks wave count, not shape or bandwidth. **Fix: a grid-stride/persistent launch sized
to an exact multiple of the SM count**, so there is no tail. Cheap, local, and gated by
`tests/gate_fp8_gemv`.

### Finding 21 — the MoE kernel runs ONE WARP PER BLOCK, capping occupancy at 50%

`ncu` on `k_grouped_w4a8_e8m0_kernel`:

```
Grid Size 1,536    Block Size 32     <- one warp per block
Registers/Thread 40                  Block Limit (registers) 48
Theoretical Occupancy 50%            Achieved Occupancy 46.47%
Compute (SM) Throughput 24.74%       Memory Throughput 37.25%
Waves Per SM 3.20
```

Neither compute- nor bandwidth-saturated: **latency-bound with half the warp slots unusable**,
because a 32-thread block consumes a whole block slot to hold a single warp. The kernel is
`<<<grid, 32>>>` throughout (`tc_moe_gemm.cu:362`). Packing several warps per block — each taking a
different n-block or tile — should lift theoretical occupancy toward 100% without touching the
math, so it can be gated cosine-1.0 against the current output.

**This is now the #1 lever**: MoE is 30.8% of `B_tok` at 38% efficiency, so `bytes × (1 − eff)` is
the largest single term in the whole model.

### Correction to Finding 15 — the code-level attribution was wrong; the mechanism is still open

Finding 15 attributed the ~72 ms M≥2 verify penalty to the `M==1`/`bs==1` GEMV fast paths in
`fp8_block_gemm.cu` and `mla_attn.cu` falling through to m16-tile mma. The microbenchmark now
prices that directly:

```
MLA dense sum, M=1: 0.4112 ms/layer      M=2: 0.4418 ms/layer      (+7.4%)
-> x43 layers = +1.3 ms   ... against a 72 ms penalty to explain
```

**The dense-GEMM fallback accounts for under 2% of it.** The step is real and reproducible, but its
cause is not what I said. The remaining candidates, none yet measured:
- **`ogroup` (`wo_a`, 33.5 MB/layer)** — its M≥2 fallback additionally allocates, runs a separate
  `k_f2h` conversion pass, and frees, per layer per step (`mla_attn.cu:271-273`). Not benched.
- **HC + 20-iteration Sinkhorn**, per token per layer — plausibly latency-bound and linear in M.
- **sparse attention / compressor / indexer** at M=K.

The layer-flavour split (Finding 14) already showed the growth is spread evenly across flavours,
which points at something every layer does — HC/Sinkhorn and `ogroup` both qualify; the
compressor and indexer do not. **Next: extend `gemm_bench` to `ogroup_gemm_fp8` and `hc_*` at
M=1..5.** Logged as open rather than patched over: this is the third mechanism proposed for the
verify excess, and the first two were wrong.

### Standing priority order, now measurement-backed

1. **MoE grouped GEMM occupancy** (Finding 21) — 30.8% of `B_tok` at 38% efficiency.
2. **Wave quantisation in the small-N GEMVs** (Finding 20) — `wq_a`, `wkv`, and every other
   small-N call; cheap and local.
3. **DSpark draft head** (Finding 17) — ~6x off roofline, gates the whole speculative win.
4. **The M≥2 verify penalty** (Finding 15, mechanism open) — bench `ogroup` and HC first.
5. ~~MLA projection GEMV~~ — **retired**, already at 89–94%.
6. ~~MXFP4 hardware unpack~~ — **retired**, already implemented (Finding 19).

### Optimization #1 — MoE grouped GEMM: warps-per-block 1 -> 4. GATED, measured on the bench.

Applied Finding 21. `k_grouped_w4a8_e8m0_kernel` and its float-scale twin now derive
`n_block = blockIdx.x*(blockDim.x>>5) + (threadIdx.x>>5)` and launch `<<<grid, 32*wpb>>>`;
`MOE_WPB` (default 4) restores the old behaviour at 1 for a reproducible A/B. The warps were
already independent (no `__shared__`, no `__syncthreads`), so this is **pure launch geometry** —
the per-warp math is byte-identical.

**Correctness first:** Gate K re-run — `moe batched`, `moe device_route`, `moe batched+TC` all
**cosine 1.0000000**, `fp4_gemm` unchanged. `Gate K (units): PASS`.

**Occupancy (ncu), the mechanism:**

| | before (wpb=1) | after (wpb=4) |
|---|---|---|
| Block size | 32 (1 warp) | 128 (4 warps) |
| Theoretical occupancy | 50% | **100%** |
| Achieved occupancy | 46.47% | **85.34%** |
| Compute (SM) throughput | 24.74% | 45.62% |
| Duration (isolated launch) | 511.74 us | **279.68 us** |

**Throughput (gemm_bench, same process, A/B by env):**

| M | wpb=1 | wpb=4 | speed-up |
|---:|---:|---:|---:|
| 1 | 111.9 GB/s | 121.6 | 1.09x |
| 3 | 113.3 | 128.3 | 1.13x |
| 5 | 116.7 | 130.5 | 1.12x |
| 8 | 118.8 | 131.5 | 1.11x |

`wpb=2` matches `wpb=4`; `wpb=8` is marginally worse. Default set to 4.

**The two measurements disagree and I do not yet know which transfers.** `ncu`'s isolated launch
says 1.83x; the bench's hot A/B loop says 1.09-1.13x. The likely reason is that the bench re-reads
the same 25 MB of expert weights 30 times, so the baseline enjoys cache reuse the real decode never
gets — in the full model each step streams ~1.1 GB of expert weights across 43 layers, far beyond
any cache, which is closer to `ncu`'s cold case. **The full-model run is the arbiter**; recorded as
a genuine open question rather than quoting the flattering number.

### Finding 22 — I violated the single-tenant rule and nearly hard-hung the box. Now enforced in code.

A launch chained as `build … | head -3 && sync && … && setsid nohup ./build/decode … &` appeared to
fail — `~/opt1.log` did not exist when checked, because `head -3` had SIGPIPE'd and the build was
still running behind the `&`. I relaunched. **Both launches then succeeded**, and two processes each
mapping 100.4 GiB drove the machine to `available: 0`, `free: 565 MB`, swap in use, load average 9.4.
This is precisely the condition that forced a physical power-cycle on the prior project.

Recovered by `kill -9` on both PIDs; memory returned to 117 GiB available with no reboot.

**Three lessons, the third being the one that matters:**
1. Do not chain a launch behind `| head` — SIGPIPE makes a *successful* pipeline look failed.
2. Never compile a heavy TU while a full-model load is in flight; the `g++` run competing for the
   last GiB is what turned "tight" into "zero".
3. **A hard rule written only in a document is not a control.** The single-tenant rule has been in
   `HARDWARE.md` §4 and `STATUS.md` from the start, and I still broke it — under exactly the
   circumstance (an ambiguous failure) where care is hardest. So it is now mechanical:
   `scripts/run_model.sh` takes an `flock` on `/tmp/dsv4-fullmodel.lock`, refuses a second launch
   with the offending PID printed, refuses to start below 105 GiB available, and always
   `setsid nohup`s. **All full-model launches go through it from here.**

---

## 2026-08-06 — Phase 6: the chat encoder is byte-exact against DeepSeek's own goldens

`include/encoding_dsv4.h` ports `encoding/encoding_dsv4.py` to C++ (no Python on the hot path).
`tests/gate_encoding.cpp` runs it against the four golden vectors shipped in the checkpoint:

```
[vector 1, thinking]   2390 bytes byte-exact -> PASS     (tools + DSML tool-call block)
[vector 2, thinking]    342 bytes byte-exact -> PASS     (multi-turn, drop_thinking)
[vector 3, thinking]   3313 bytes byte-exact -> PASS     (Chinese, developer role, tools)
[vector 4, chat    ]   2552 bytes byte-exact -> PASS     (Chinese, chat mode, tables)
[roundtrip] BOS + <think> present -> PASS
[effort] low+prefix==high, prefix present, absent in chat -> PASS
Gate ENCODING: 6 passed, 0 failed -> PASS
```

**Two non-obvious things were required to get byte-exactness**, both in JSON serialisation of the
embedded tool schemas — the schemas go into the prompt *verbatim*, so their formatting is
load-bearing:

1. **`nlohmann::json` sorts object keys** (it is backed by `std::map`), while Python's `json.dumps`
   preserves insertion order. Switched to `nlohmann::ordered_json`. Symptom was
   `{"description":...,"name":...}` against the golden's `{"name": ..., "description": ...}`.
2. **Python's `json.dumps` default separators are `", "` and `": "` — with spaces.** nlohmann emits
   `,` and `:`. Wrote a small recursive `to_json` that reproduces Python's spacing and delegates
   scalars to `dump()` (which already escapes correctly and emits raw UTF-8, i.e.
   `ensure_ascii=False`).

Neither would have been caught by a "looks right" review; both were caught instantly by a
byte-exact gate against vendor-supplied goldens. Vectors 3 and 4 also confirm UTF-8 is handled
end-to-end.

The `[effort]` assertion additionally pins the property `CHAT_FORMAT.md` §2.2 flags as a
prefix-cache hazard: `high` output equals `low` output plus the prefix, at the *front*, and has no
effect at all in chat mode.

### Optimization #1 — FULL-MODEL A/B: CONFIRMED, 7.80 -> 8.63 tok/s (1.107x)

Run through the new `scripts/run_model.sh` guard (single tenant, cold cache, detached).

| | before | after |
|---|---:|---:|
| **Warm M=1 decode** | 128.2 ms/tok = **7.80 tok/s** | 115.8 ms/tok = **8.63 tok/s** |
| Effective bandwidth | 87.4 GB/s = 36.4% of achievable | **96.7 GB/s = 40.3%** |
| 43-layer verify, K=1 | 115.94 ms | 103.17 ms |
| M=5 verify | 337.0 ms | 300.5 ms |
| Spec-decode | 129.1 ms/tok | 118.8 ms/tok |
| Correctness | argmax 11111 | argmax 11111 — **GATE PASS** |

**Speed-up 1.107x, and it resolves the open question I logged.** The bench had predicted
1.09–1.13x and `ncu`'s isolated-launch duration had suggested 1.83x; I recorded that I did not know
which would transfer. **The bench transferred; `ncu`'s isolated duration did not.** The reason is
now clear in hindsight — `ncu` serialises and cold-starts a single launch, which exaggerates a
latency-bound kernel's improvement, whereas in the real decode the MoE launches are pipelined with
everything else. **Rule for this project: use `ncu` to find the *mechanism*, use `gemm_bench` to
predict the *magnitude*, and use the full model to confirm.**

Spec-decode is still 0.97x of base — but only because the base got faster too; in absolute terms it
improved 129.1 → 118.8 ms/tok. Findings 16/17 are unchanged: the draft head remains the blocker.

The K-sweep reconfirms Finding 14 on the faster kernels (`slide` 2.739x, `r128` 2.461x,
`r4` 2.510x — DSA still not the driver) and the K≥2 step penalty persists at +0.700 at K=2, so
Finding 15's mechanism remains open and is unaffected by this change.

### Optimization #2 — wave quantisation in the small-N GEMVs: NEGATIVE RESULT, reverted

Applied Finding 20's fix: made `fp8_gemv_m1_kernel` grid-stride and capped the launch so it is a
whole number of waves. Two variants, neither adopted.

| variant | wq_a (N=1024) | wq_b (N=32768) | wkv (N=512) | wo_b (N=4096) |
|---|---:|---:|---:|---:|
| original (one warp per row) | 45.1 GB/s | **214.6** | 129.7 | **225.5** |
| clamp to one full wave | 41.8 | **186.3** | 124.9 | **194.7** |
| round down to whole waves | 41.6 / 156.8 (!) | 192.4 / 187.7 | 127.3 / 127.1 | 212.3 / 207.0 |

**Clamping to a single wave is clearly worse** on the two large shapes — it strips memory-level
parallelism from `wq_b` and `wo_b`, which were already at 89–94%. Rounding down to whole waves
avoids most of that harm but shows **no demonstrable gain**: back-to-back runs of the same binary
give `wq_a` as 41.6 and 156.8 GB/s, a 3.8x swing. **Reverted to the original launch.** The
grid-stride loop is retained because it is a no-op when `blocks == want`, so the mechanism stays
available without changing behaviour.

**Two lessons, and the second is the more important:**

1. **The bench is not trustworthy for the small shapes.** A 4.19 MB weight is small enough that
   allocation/first-touch effects dominate a 40-rep loop. It needs a longer warm-up and repeat
   trials before it can adjudicate anything at that size. Its verdict on the *large* shapes (and on
   Opt #1, which the full model then confirmed at 1.107x) has held up.

2. **Finding 20 was right about the mechanism and wrong about the priority.** I ranked it #2 on the
   *efficiency deficit* alone, having written one paragraph earlier that the correct ranking is
   `bytes x (1 - efficiency)` — and then not applied it. The arithmetic I should have done first:

   | shape | share of `B_tok` | efficiency | recoverable |
   |---|---:|---:|---:|
   | `wq_a` | 1.6% | 19% | ~1.3% |
   | `wkv`  | 0.8% | 54% | ~0.4% |
   | **total** | **2.4%** | | **~1.7%** |

   **The entire lever is worth under 2% of the decode step**, so it could never have paid for the
   risk of perturbing `wq_b`/`wo_b`, which together carry ~26% of `B_tok` at 89–94%. Wave
   quantisation is real and `ncu` diagnosed it correctly; it simply is not worth fixing here.
   Demoted from #2 to last.

The revised standing order is in `OPTIMIZATION_LOG.md`: the DSpark draft head (Finding 17, ~6x off
roofline and gating the whole speculative win) and the M>=2 step penalty (Finding 15, mechanism
still open) are both worth far more than anything left in the dense GEMV path.

### Finding 15, third pass — both remaining suspects MEASURED. Neither explains it. But HC is a new lever.

Extended `gemm_bench` to `ogroup_gemm_fp8` and `hc_pre`+Sinkhorn, with a better method (median of 5
trials, 3-call warm-up each) after Opt #2 showed the old mean-of-N was not robust at small shapes.

```
                          M=1      M=2      M=3      M=5      M=8      per-layer x43
ogroup_gemm_fp8        0.0257   0.0626   0.0650   0.0700   0.0780     1.1 -> 3.0 ms
hc_pre + sinkhorn      0.1272   0.1272   0.1291   0.1294   0.1310    10.9 -> 11.1 ms (x2/layer)
```

**`ogroup` has exactly the right shape** — 2.4x jump at M=2, then flat to M=8, which is the
step-function signature Finding 15 describes, and its M≥2 fallback does allocate + run a separate
`k_f2h` pass + free. **But the magnitude is wrong by a factor of ~45**: the step is
`(0.0626-0.0257) x 43 = 1.6 ms` against a ~72 ms penalty, i.e. **2.2% of it**.

**HC + Sinkhorn is flat across M** (0.1272 → 0.1310, +3%). Not the mechanism at all.

**So Finding 15's mechanism is still open after three hypotheses** — MoE union (refuted by code),
DSA (refuted by the flavour split), and now the dense/ogroup/HC path (refuted by direct
measurement, all three contributing <2% each). What is left inside a layer and unmeasured: the
sparse attention itself, the compressor/indexer forwards, and the MoE *dispatch* around the grouped
GEMM (gather, act-quant, swiglu, scatter, the deterministic combine). Recording the elimination
rather than proposing a fourth guess: the honest state is that ~70 ms of the M≥2 verify step is
still unattributed, and the next step is instrumentation inside the layer, not another hypothesis.

Worth noting the sweep also shows the excess **shrank** with Opt #1 — at K=5 it went from +34.6 ms
to ~+23 ms — so part of the "penalty" was always MoE inefficiency scaling with `|union|`, not a
fixed M≥2 cost.

### Finding 23 (new lever, found while eliminating) — HC + Sinkhorn is ~8x off its roofline

The elimination measurement turned up something more useful than the thing it was looking for:

```
HC cost   0.1272 ms/call x 2 calls/layer (attn + ffn) x 43 layers = 10.94 ms
          = 9.4% of the 115.8 ms decode step
HC bytes  135.28 MB -> 1.40 ms at the measured 96.7 GB/s
          => ~7.8x off roofline, and completely M-invariant
```

Hyper-connections are only 1.2% of `B_tok` but **9.4% of wall-clock**. By the
`bytes x (1 - efficiency)` rule this is now the **second** largest recoverable term after the MoE —
worth ~9 ms/step, i.e. roughly another 1.09x on its own. The cause is almost certainly the
**20 sequential Sinkhorn iterations** (`hc_sinkhorn_iters: 20` from config), each a tiny
normalisation over a `[bs, 4, 4]` matrix — 20 dependent round-trips of trivial work, which is
latency, not bandwidth. A single fused kernel keeping the 4x4 in registers across all 20 iterations
is the obvious fix and it is entirely local.

**This is the third time the measured ranking has disagreed with the byte-share ranking**
(MLA over-ranked, small-N GEMVs over-ranked, HC under-ranked). The byte model predicts the *floor*
extremely well — it nailed K=1 to 0.02 ms — but says nothing about which kernels reach it.

### Finding 24 + Optimization #3 — the Sinkhorn kernel ran ONE SCALAR THREAD with a spilled array

Chasing Finding 23 into `hc_pre` split the cost four ways, and it was not the part I expected:

```
                        M=1      M=2      M=3      M=5      M=8
hc_pre + sinkhorn    0.1137   0.1162   0.1188   0.1249   0.1295
  of which sinkhorn  0.0862   0.0862   0.0866   0.0862   0.0863   <- 76% of hc_pre, and flat in M
```

`kernels/hc_sinkhorn.cu:14` reads `if (i >= n || threadIdx.x != 0) return;` — **the kernel launches
32 threads and immediately retires 31 of them.** One scalar thread then runs all 20 iterations over
`float c[HCMAX*HCMAX]`, indexed with runtime `(j,k)`. nvcc cannot keep a dynamically-indexed array
in registers, so `c` lives in **local memory, which is DRAM-backed**. Roughly 640 dependent local
round-trips per token — which is why normalising a **4x4 matrix** cost 86 us and 7.4 ms of the
decode step.

**Rewrote it warp-parallel in registers** (`hc_sinkhorn_warp_kernel`): one warp per token, lane `L`
holds `comb[j,k]` with `j = L/hc`, `k = L%hc`. Row sums reduce over the low `log2(hc)` lane bits and
column sums over the next `log2(hc)`, via `__shfl_xor_sync`. Those XOR masks never cross a group
boundary, so the idle lanes (`L >= hc*hc`) cannot contaminate a reduction. Guarded to `hc` a power
of two with `hc*hc <= 32` (config `hc_mult: 4` -> 16 lanes); anything else keeps the scalar path, and
`HC_SCALAR=1` forces it for A/B.

Also fixed alongside (**one measurement each**): `k_mixes` was one *warp* per output, i.e. 24 warps
total to stream the 1.57 MB `hc_attn_fn` — now one block per output with a two-stage reduction and
`float4` loads.

**Correctness:** Gate K PASS, `[hc_sinkhorn]` `max_abs` pre 5.96e-08 / post 1.19e-07 / comb
**8.94e-08** (was 1.19e-07 — marginally *better*, since the register path avoids a round-trip).

| | scalar (old) | warp (new) | |
|---|---:|---:|---|
| `hc_sinkhorn` | 0.0862 ms | **0.0165 ms** | **5.2x** |
| `hc_pre` total | 0.1137 | **0.0436** | **2.6x** |
| per decode step (x2/layer x43) | 9.8 ms | **3.7 ms** | **-6.1 ms** |

Predicted full-model effect: 115.8 -> ~108.6 ms/tok, i.e. ~9.2 tok/s. Full-model run is the arbiter.

**Why this was worth finding.** The byte model rates HC at 1.2% of `B_tok` and would never have
flagged it; it was 9.4% of wall-clock. And the kernel had been *correct* since the prior project's
Gate K — it passed every correctness gate it was ever given, because "one thread does all the work"
is a performance defect that no correctness oracle can see. It took profiling the composite, then
splitting the composite, to reach it.

### Finding 25 — the `float4` fast path faulted on real weights, and no unit gate could have caught it

The first full-model run of Optimization #3 **crashed**:

```
[decode] structs built. mem 110.2/122.8 GiB
[decode] prefill 5 positions...
cuda kernels/moe.cu:222 misaligned address
```

(The report site is misleading — CUDA errors are sticky and surface at the next sync, so `moe.cu`
was merely the first thing to check after the faulting launch.)

**Cause:** the `float4` loads I added to `k_mixes` need **16-byte** alignment on both operands.
`x` comes from the arena and is fine, but **`hc_fn` is a weight tensor mapped straight out of the
safetensors shard**, and safetensors only guarantees **8-byte** tensor offsets. A `float4` load on a
merely-8-byte-aligned pointer faults.

**Fixed** by testing both pointers at runtime and falling back to scalar loads
(`vec4 = (hcd%4==0) && !((uintptr_t)xr & 15) && !((uintptr_t)fr & 15)`).

**Why this matters beyond the bug.** Gate K passed the change — twice — because
`ref/gen_units.py` writes goldens that the harness loads into its own `cudaMalloc`'d buffers,
which are 256-byte aligned. **The unit gate structurally cannot exercise real weight alignment.**
This codebase already knew the hazard: the MoE GEMM carries `off_b = (uintptr_t)wprE & 15` plus a
funnel shift for exactly this reason. I added a vectorised load without checking whether the
pointer it reads is a mapped weight.

Standing rule added: **any new vectorised load on a tensor that comes from `WeightStore` must
either check alignment at runtime or use the funnel-shift pattern.** Unit-gate PASS is not evidence
on this axis; only a full-model run is.

### Optimization #3 — FULL-MODEL A/B: CONFIRMED, 8.63 -> 9.26 tok/s (1.072x)

| | before Opt #3 | after |
|---|---:|---:|
| **Warm M=1 decode** | 115.8 ms/tok = **8.63 tok/s** | 108.0 ms/tok = **9.26 tok/s** |
| Effective bandwidth | 96.7 GB/s = 40.3% | **103.7 GB/s = 43.2%** |
| 43-layer verify, K=1 | 103.17 ms | 95.38 ms |
| M=5 verify | 300.5 ms | 295.0 ms |
| Correctness | argmax 11111 | argmax 11111 — **GATE PASS** |

Predicted 108.6 ms from the bench (-6.1 ms x 2 calls/layer); measured **108.0**. The bench has now
predicted two adopted optimizations to within ~1%, which settles the instrument policy: **`ncu` for
mechanism, `gemm_bench` for magnitude, full model to confirm.**

**Cumulative: 128.2 -> 108.0 ms/tok, 7.80 -> 9.26 tok/s = 1.187x**, at 43.2% of achievable
bandwidth against a 21.4 tok/s wall.

The K-sweep reconfirms Finding 14 a third time on faster kernels (`slide` 2.879x, `r128` 2.660x,
`r4` 2.604x — DSA still grows *slowest*), and the K>=2 step penalty is undiminished (+0.519 at K=5),
consistent with Finding 15's mechanism living somewhere neither optimization touched.

Spec-decode is 117.0 ms/tok (0.92x of base) — it keeps losing ground as the base improves, exactly
as Finding 17 predicts: the draft head is a fixed ~119 ms cost that the base-path optimizations do
not touch, so every base win makes speculation look worse. **The draft head is now unambiguously the
top lever.**

---

## 2026-08-06 — Profiling the draft head, and finding the answer is somewhere else

### Finding 26a — the spec step, attributed. The draft is 19.7%; the VERIFY is 80.3%.

`DSV4_SPECPROF=1` adds CUDA-event timers around each phase of a verify round. Mean of 7 rounds:

```
draft: main_kv      0.55 ms  ( 0.1%)
draft: 3 blocks    21.17 ms  ( 5.5%)
draft: fwd_head    54.01 ms  (14.0%)   <- host AR loop over 5 block positions
verify 43 layers  308.70 ms  (80.3%)
TOTAL             384.43 ms
```

**This corrects Finding 17.** I had estimated the draft at ~119 ms by subtracting a standalone M=5
verify (294 ms) from the spec round (425 ms). The verify *inside* the spec loop actually costs
308.7 ms (it also rebuilds/rolls back KV), so the draft is **75.7 ms, not 119** — and it is 19.7%
of the round, not 28%.

What that means for the economics, at the measured acceptance `a = 3.12`:

| scenario | ms/round | S |
|---|---:|---:|
| measured | 384.4 | **0.88x** (reported 0.93x) |
| `fwd_head` halved | 357.4 | 0.95x |
| **draft made entirely FREE** | 308.7 | **1.10x** |
| verify at its byte-model `c_v = 2.12` | 305.5 | 1.11x |
| both | 229.8 | **1.47x** |

**Optimising the draft head to zero would only reach 1.10x.** The lever that matters is the verify —
i.e. Finding 15, still unattributed after three refuted hypotheses. Recording this before doing the
work, because the instinct to optimise the thing you were just looking at is exactly what the
`bytes x (1 - efficiency)` rule keeps catching.

### Finding 26b — but the draft's biggest term is a defect that ALSO costs the base path

`dspark_forward_head` is a **host-driven** autoregressive loop over the 5 block positions. Per
position it: copies a token id H2D, runs the markov GEMV, adds the bias, **syncs**, copies the full
129,280-float logits row **D2H**, and does the **argmax on the CPU**. Five syncs, 2.6 MB of D2H, and
five 129k-element host scans per draft.

Underneath that, a bigger and simpler problem, found by reading `decode.cu:127`:

```cpp
const float *head_w = L.bf16("head.weight");     // BF16 [129280,4096] -> materialised as F32
```

`Loader::bf16` **dequantises to f32**. So `lm_head` occupies and is read as **2118 MB instead of
1059 MB — every single decode step**, and the markov tables likewise (132 MB instead of 66 MB, and
`markov_w2` is re-read once per block position, so 5x). On top of that `gemm_fp32` launches
`<<<dim3(N,M), 32>>>` — **646,400 one-warp blocks** for `lm_head`, the same 50%-occupancy defect as
Finding 21.

Consequences, at the measured 113.5 GB/s:

```
lm_head  2118 MB -> 18.7 ms of a 108.4 ms decode step (17%)
BF16-native would save ~9.3 ms AND free 2.1 GiB of headroom
B_tok was modelled at 11,202 MB; the engine actually moves 12,261 MB
  -> real efficiency is 47.3% of achievable, not 43.2%
```

**`ROOFLINE.md`'s `B_tok` measured the checkpoint, not the engine.** The byte model is right about
what is *stored*; it silently assumed the engine reads weights in their stored dtype. It does not.

**Fix (`gemm_bf16w`)**: read BF16 natively with `__nv_bfloat162` loads, four warps per block, and a
runtime alignment check (Finding 25's lesson — this reads a mapped safetensors tensor). Wired into
all three `lm_head` call sites (decode, verify, draft) and both markov tables.

**Gated before measuring** (`tests/gate_bf16w.cu`) against the old path — the same weights
dequantised to f32 and fed to `gemm_fp32`, which is exactly what the engine did before:

```
[lm_head M=1 ] cosine=1.000000000 max_abs=6.56e-07 rel=2.82e-07 argmax MATCH -> PASS
[lm_head M=5 ] cosine=1.000000000 max_abs=7.15e-07 rel=2.90e-07 argmax MATCH -> PASS
[markov  M=1 ] cosine=1.000000000 max_abs=7.45e-08 rel=1.01e-07 argmax MATCH -> PASS
[markov  M=5 ] cosine=1.000000000 max_abs=8.94e-08 rel=1.26e-07 argmax MATCH -> PASS
```

BF16 -> F32 is lossless, so the only admissible difference is fp32 accumulation reassociation (the
bf16x2 path pairs elements); `argmax MATCH` is the property that actually matters for greedy decode.

### Optimization #5 — device-side draft AR loop. Draft 75.7 -> 54.7 ms; speculation reaches parity.

`dspark_forward_head`'s greedy loop over the 5 block positions ran on the **host**: per position it
synced, copied the whole 129,280-float logits row D2H, and scanned it on the CPU. Five syncs,
2.6 MB of D2H and five 129k host scans per draft, stalling an otherwise fully asynchronous kernel
chain. The dependency (position `i+1` needs the argmax of position `i`) is real — but it is a
**device** dependency; nothing needs to reach the host until the block is done.

Moved entirely on-device: `k_seed_first` / `k_pick` / `k_argmax_row` (block-reduction argmax) /
`k_store`, and removed the `cudaStreamSynchronize` that `dspark_markov` was doing internally.

**Correctness — the strongest available check:** the draft tokens are the thing that must not move,
because a different draft changes acceptance and makes the comparison meaningless. The generated
sequence is **byte-identical** to the host-loop run:

```
11111 16 455 6102 294 16603 344 29168 16 455 6102 294 29585 344 76405 16 455 6102 294 14251 344 16235 16 455 6102
```
and `mean tokens/verify = 3.12` is unchanged.

| phase | before Opt #4 | after #4 | after #5 |
|---|---:|---:|---:|
| draft `main_kv` | 0.55 | 0.51 | 0.56 |
| draft 3 blocks | 21.17 | 20.83 | 21.71 |
| **draft `fwd_head`** | **54.01** | 39.18 | **32.38** |
| verify 43 layers | 308.70 | 293.03 | 291.80 |
| **round total** | **384.43** | 353.54 | **346.45** |
| spec-decode vs base | 0.93x | 0.98x | **1.00x** |

**Draft head: 75.7 -> 54.7 ms (1.39x); `fwd_head` alone 54.0 -> 32.4 (1.67x).**
Speculation is now at exact parity (105.4 vs 105.2 ms/tok) instead of losing 7%.

**Where the draft stands now.** `fwd_head` at 32.4 ms against a ~11.9 ms byte model (lm_head at M=5
plus 5x `markov_w2`) is still ~2.7x off — the five markov GEMV -> add-bias -> argmax chains are
latency-linked and each is far too small to fill the device. But **the verify is now 84.2% of the
round**, so further draft work has very little leverage: even reducing `fwd_head` to zero would move
speculation from 1.00x to only ~1.10x.

**The draft head is no longer the blocker. Finding 15 is** — the ~70 ms of unattributed M>=2 verify
cost is worth 1.00x -> ~1.4x on speculation and is 84% of the spec round. That is where the next
work belongs, and it needs instrumentation inside the layer rather than a fourth hypothesis.

---

## 2026-08-06 — FINDING 15 IS CLOSED. The M>=2 verify penalty is the ATTENTION GLUE.

Three hypotheses had been refuted (MoE expert-union by code inspection, DSA by the layer-flavour
split, ogroup/HC by direct bench) and I had explicitly refused to offer a fourth guess, saying the
next step was instrumentation. This is that instrumentation.

`include/dprof.h` + `kernels/dprof.cu`: named-phase GPU timing. Events are recorded into a
preallocated pool and **not synchronised until the report**, so the instrumented path stays
asynchronous — a sync per phase would have manufactured exactly the stalls being hunted, which is
the trap `dspark_forward_head` had fallen into. Enabled with `DSV4_DPROF=1`, compiled to a branch
otherwise.

### Level 1 — the verify step by sub-op, summed over all 43 layers

| phase | K=1 | K=2 | K=5 | K1→K2 | K1→K5 |
|---|---:|---:|---:|---:|---:|
| hc_pre (attn) | 2.55 | 2.77 | 3.76 | 1.09x | 1.47x |
| rmsnorm (attn) | 0.75 | 0.74 | 0.79 | 0.99x | 1.05x |
| **ATTENTION** | **44.24** | **114.78** | **128.38** | **2.59x** | **2.90x** |
| hc_post (attn) | 0.28 | 0.30 | 0.44 | 1.07x | 1.57x |
| hc_pre (ffn) | 3.70 | 3.78 | 5.59 | 1.02x | 1.51x |
| rmsnorm (ffn) | 0.76 | 0.77 | 0.76 | 1.01x | 1.00x |
| MoE | 43.42 | 73.16 | 110.44 | 1.68x | 2.54x |
| hc_post (ffn) | 0.25 | 0.30 | 0.44 | 1.20x | 1.76x |
| kv xin copy | 0.19 | 0.18 | 0.19 | 0.95x | 1.00x |
| TOTAL | 96.13 | 196.77 | 250.79 | 2.05x | 2.61x |

**ATTENTION jumps 44.24 -> 114.78 ms at K=2 — +70.5 ms — then is nearly flat to K=5 (128.38).**
That is the entire missing penalty, and it is a step, not a per-position cost. Everything else
behaves: MoE grows 2.54x against a ~3.7x byte model (correlated routing makes the real expert union
*cheaper* than modelled, as suspected), and the HC/norm/copy phases are flat.

Attention weight bytes are **K-invariant** — the same 4599 MB are read once whether verifying 1
token or 5. So the +70.5 ms is not bandwidth. It is work that scales with M.

### Level 2 — inside attention (the 2 pure-sliding layers, where `mla_verify_step` runs)

| sub-op | K=1 | K=2 | K=5 | K1→K2 |
|---|---:|---:|---:|---:|
| `q_proj` (act_quant, wq_a, rmsnorm, act_quant, wq_b, rmsnorm, rope) | 0.55 | 1.53 | 1.54 | **2.78x** |
| `kv_write` (wkv, rmsnorm, rope, act_quant) | 0.08 | 0.21 | 0.21 | **2.62x** |
| **`sparse_attn`** | **0.05** | **0.05** | **0.06** | **1.00x** |
| `ogroup` (rope⁻¹, ogroup_gemm, wo_b) | 1.03 | 3.11 | 3.13 | **3.02x** |
| SUM | 1.71 | 4.90 | | 2.87x |

```
step per layer at K=2 = (4.90 - 1.71)/2 = 1.595 ms
scaled to 43 layers   = 68.6 ms      vs the ~70 ms to explain   -> ACCOUNTED
```

**The penalty is the dense-projection + elementwise chains — `act_quant`, `rmsnorm`, `rope`,
`kv-write` — around the GEMMs. And `sparse_attn` is 0.05 ms and completely FLAT (1.00x).**

### What this corrects

- **DSA is exonerated a fourth time, now directly rather than by inference.** The sparse attention
  is 0.05 ms of a 96 ms step and does not grow with K at all. The directive's §4.3/§7 framing of
  DSA verify cost as the project's central unknown was wrong for this engine, and I spent a
  hypothesis on it too.
- **My original Finding 15 attribution was half right.** I blamed the `M==1`-only GEMV fast paths
  falling through to m16-tile mma, then retracted it when the bench priced `fp8_block_gemm`'s
  M=1→M=2 step at only +7.4% (~1.3 ms over 43 layers). Both readings were incomplete: the GEMMs
  alone really are nearly flat, but the **glue around them is not**, and I had never benched the
  glue. The bench measured what I thought to measure, which is a different thing from what mattered.
- The prior project's research doc named this exact lever — **T1.2, "fuse the attention/indexer/
  compressor glue (RoPE + norm + quant + KV-write)", citing vLLM at 2-20x** — and ranked it second.
  It was never taken. It is now empirically the single largest remaining item, and it is the same
  class of defect as Findings 21/24/26b: correctness-first kernels launched at shapes that were
  fine at M=1 and fall off a cliff at M>=2.

### What it is worth

`c_v(5)` is currently 2.61. Removing the 68.6 ms step takes the K=5 verify from 250.8 to ~182 ms,
i.e. `c_v` -> ~1.9. At the measured acceptance `a = 3.12` that moves speculation from **1.00x to
~1.6x**, and it also cuts the base decode's own attention glue. **This is now the top lever in the
project**, ahead of the remaining MoE efficiency.

---

## 2026-08-06 — Finding 28: the glue was NOT the cause either. It is COLD-WEIGHT COALESCING.

Finding 15 closed with "the M>=2 penalty is the attention glue". **That was still wrong**, and the
bench said so in seconds — which is why the fast harness was worth building.

### The glue is innocent

| op (real per-layer shape) | M=1 | M=2 | M1→M2 |
|---|---:|---:|---:|
| `act_quant_fp8 [M,4096]` | 0.0042 | 0.0042 | 1.00x |
| `act_quant_fp8 [M,1024]` | 0.0041 | 0.0041 | 1.00x |
| `rmsnorm [M,1024]` | 0.0062 | 0.0062 | 1.00x |
| `rmsnorm [M*64,512]` | 0.0043 | 0.0062 | 1.45x |
| `rope [M*64,64]` | 0.0044 | 0.0041 | 0.94x |
| `act_quant_fp8sim` | 0.0041 | 0.0041 | 1.00x |
| **glue step, summed** | | | **0.0017 ms/layer -> 0.1 ms over 43** |

**0.1 ms of a ~70 ms penalty.** The elementwise chain does nothing.

### The real cause: the isolated bench was HOT, and in situ is always COLD

`q_proj`/`ogroup` are GEMM + glue, and the glue is flat, so it had to be the GEMMs — but the bench
had priced `fp8_block_gemm`'s M=1→M=2 step at +7.4%. The bench looped 40x on **one** 33 MB weight.
In situ every layer's weights come from a 100 GiB working set and are **always cold**. Rotating over
12 copies (402 MB, far beyond any cache):

| `wq_b [32768,1024]` | M=1 | M=2 | M=5 | M1→M2 |
|---|---:|---:|---:|---:|
| **HOT** (one weight reused) | 0.1744 | 0.1230 | 0.1603 | **0.71x** |
| **COLD** (12 rotating) | 0.1986 | **0.6199** | 0.6042 | **3.12x** |

**3.12x cold, 0.71x hot — from the same code.** That matches the in-situ `q_proj` 2.78x and
`ogroup` 3.02x. The measurement, not the mechanism, had been wrong all along.

### Why cold hurts the m16 tile: B access coalescing

```
fp8_gemv_m1 : lane L reads B[n*K + kb*128 + L*4]   -> 32 lanes x 4B = 128 CONTIGUOUS bytes
tc_fp8 m16  : lane L reads B[(n0 + L/4)*K + ...]   -> 8 rows STRIDED BY K, 32B from each
```
The GEMV issues one fully-coalesced 128 B request per warp; the m16 tile issues eight 32 B requests
scattered 1024 B apart. Hot, L2 absorbs it. Cold, it is ~3x the DRAM transactions for the same bytes.

**This also explains, retroactively, why Opt #2 (wave quantisation) measured as noise**: at those
shapes I was measuring a hot loop too.

### Fix applied, and its honest limit

Templated `fp8_gemv_mkT_kernel<M>` — the M=K GEMV with `acc[M]` sized at compile time instead of
`acc[GEMV_MK_MAXM]`=16, which is why the prior project's A/B rejected it (registers capped
occupancy). Same coalesced B access as the M=1 GEMV. Gate K PASS.

Cold A/B against the m16 tile:

| M | GEMV | m16 | |
|---:|---:|---:|---|
| 2 | 0.2680 | 0.5354 | **2.00x faster** |
| 3 | 0.3589 | 0.5331 | **1.49x** |
| 5 | 0.5176 | 0.5384 | 1.04x — a wash |
| 8 | 0.7721 | 0.5522 | **0.72x — slower** |

Adopted for **M = 2..4 only**, where it clearly wins (`TC_MK=1` forces the old path). The GEMV does
M scalar dots per weight read, so it goes ALU-bound as M grows; by M=8 the tile's compute density
wins back what its access pattern loses.

**This does not move our operating point.** Base decode is M=1 (untouched by construction — the
dispatch is `M >= 2`), and the verify runs at K=5 where the two paths are within 4%. It is banked
because it is strictly better for M=2..4 and it pins the mechanism.

**The correct fix is to repack B for the fp8 m16 path**, exactly as the FP4 grouped MoE path already
does (`tc_ensure_repacked` -> `(N/8)*(K/128)*512` mma-order layout, which is why that kernel's B
reads *are* contiguous). That would give the tile's compute density AND the GEMV's coalescing, and
should recover most of the ~70 ms at K=5. It is the single largest remaining item and it is a
known-good pattern already in this repo — but it is a real kernel rewrite plus a repack pass, not a
threshold change, so it is scoped rather than started.

### Where the decode number now stands

| optimization | base decode |
|---|---|
| ported as-is | 7.80 tok/s |
| #1 MoE occupancy | 8.63 |
| #3 HC/Sinkhorn warp-parallel | 9.26 |
| #4 BF16-native lm_head/markov | **9.51** |
| #2 wave quantisation | reverted (negative) |
| #5 device draft AR loop | no base effect (draft only) |
| #6 small-M coalesced GEMV | **no base effect by construction (M>=2 only)** |

**The base number has stopped moving** for the levers found so far: at M=1 the large MLA GEMMs are
already at 89-94% of achievable and the glue is flat. The two remaining measured gaps are the MoE
(~55% of achievable after Opt #1) and the m16 B-repack (M>=2 only, so it buys verify/draft, not
base). Further base gains require attacking the MoE beyond occupancy — a different kernel, not
another launch-geometry tweak.

---

## 2026-08-06 — Finding 29: tcgen05 and TMA are AVAILABLE on Thor. The inherited blocker was a flag error.

Before dispatching the deep-research effort I re-tested the one inherited "fact" that constrained the
most: that `sm_110a` lacks 5th-gen tensor cores. **It does not.**

```
compile (compute_110a)                 runtime (real Thor silicon)
  tcgen05.fence/alloc/mma/ld  COMPILES   tcgen05 alloc + relinquish + dealloc : OK
  cp.async.bulk (TMA)         COMPILES   TMA bulk copy w/ mbarrier completion : OK
  mma m16n8k32 e4m3           COMPILES
  mma e2m1 (FP4)              BLOCKED  <- "mma with with FP6/FP4 floating point type"
  mma .kind::mxf4 block_scale BLOCKED  <- "mma with block scale"
```

**Root cause of the false blocker — one letter.**
```
-arch=sm_110   -> compute_110   -> tcgen05 BLOCKED
-arch=sm_110a  -> compute_110a  -> tcgen05 COMPILES
```
The prior project's own recorded error message says `.target 'sm_110'` — not `sm_110a`. The `a` suffix
enables architecture-specific features, and tcgen05 is behind it. That conclusion propagated into
`DECODE_GAP_RESEARCH.md` ("DeepGEMM's SM100 fp8×fp4 kernels are a REWRITE, not a port"), then into this
project's `HARDWARE.md` §5 and `ROOFLINE.md`, and I carried it forward unquestioned through six
optimisation rounds. It shaped which levers were even considered.

A second, subtler trap: building an **executable** with `-arch=sm_110a` still emits `compute_110` PTX
alongside the SASS, and *that* fails. The correct form is
`-gencode arch=compute_110a,code=sm_110a`. My own first runtime attempt hit exactly this and briefly
looked like a confirmation of the old claim.

Captured permanently as `tools/arch_probe.cu` + `tools/arch_probe_runtime.cu` + `scripts/arch_probe.sh`,
which test **compile and runtime**, so this cannot rot again.

### What changes

- **TMA is the textbook fix for Finding 28.** The M≥2 penalty is B-operand coalescing loss (8 rows
  strided by K, 32 B each). `cp.async.bulk.tensor` does hardware-managed, swizzled, fully-coalesced
  global→shared staging — exactly the missing piece, and it needs no weight repack.
- **tcgen05 reopens the CUTLASS/DeepGEMM SM100 kernel family** as a port rather than a rewrite.
- **The user's FP4 premise is half right, and the half that fails is now nailed down:** Thor has FP4
  *storage* and hardware FP4→half *conversion* (both already used), but **no FP4 tensor-core matmul and
  no block-scaled mma** — confirmed on the arch-specific target, with ptxas naming the feature. So MXFP4
  weights must still be converted before an FP8/FP16 `mma`. That is a real ceiling, not a flag error.

### Discipline note

Every earlier finding in this log was validated by measurement. This one was inherited as a *negative*
capability claim and never re-tested, because "the compiler rejects it" felt like proof. Negative
capability claims deserve the same re-verification as performance claims — more, because they silently
remove options rather than producing a wrong number. `scripts/arch_probe.sh` should be re-run after
every CUDA toolkit update.

---

## 2026-08-06 — Finding 30: Thor DOES have FP4 tensor-core compute. I tested the wrong instruction family.

The user pushed back on my "Thor has no FP4 tensor cores" conclusion. They were right and I was wrong.

I had probed `mma.sync` — the Ampere-lineage family — on `sm_110a` and `sm_110f`, seen
`Instruction 'mma with block scale' not supported`, seen it *compile* on `sm_121a`, and concluded
Thor lacks FP4 while GB10 has it. **I never probed the `tcgen05` family, even though I had just
established in Finding 29 that tcgen05 compiles and runs on this box.**

```
tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X   -> COMPILES on sm_110a
SASS: UTCOMMA.4X gdesc[UR8], gdesc[UR8], tmem[UR6], tmem[UR4], idesc[UR5], tmem[UR6], !UPT
```
Plus `.kind::mxf8f6f4.block_scale` and `.kind::f8f6f4` → `UTCQMMA`, and `.kind::f16` → `UTCHMMA`.

### The two chips are mirror images

| | `mma.sync` FP4 | `tcgen05` FP4 |
|---|---|---|
| **Thor `sm_110a`** | BLOCKED | **AVAILABLE** (`UTCOMMA.4X`) |
| **GB10 `sm_121a`** | **AVAILABLE** | BLOCKED — `sm_121` has no `tcgen05` at all |

Thor is **datacenter-lineage** Blackwell (tensor memory + `tcgen05`, SM100-like); GB10 is
**consumer-lineage** (SM120-like). Both have hardware MXFP4/NVFP4 block-scaled matmul, reached
through opposite instruction families. Probing only the family the *other* chip uses yields a
confident and completely wrong "no FP4 here".

### Why I got this wrong twice in a row

Finding 29 corrected an inherited negative claim (`sm_110` vs `sm_110a`) and I wrote at the time
that "negative capability claims deserve more re-verification than performance claims". Then I
immediately produced a *new* negative capability claim from a single-family probe and called it
"definitive" and "settled". The failure mode is identical to the one I had just documented: I
proved absence within the search space I happened to look at, and reported absence in general.

The correct discipline for a negative capability claim is to **enumerate the families that could
provide the capability before concluding any of them lacks it.** For tensor-core matmul on Blackwell
that means `mma.sync`, `wgmma`, and `tcgen05` — three families, and I checked one.

### What it changes

- **MXFP4 experts (28% of `B_tok`) can go into a hardware block-scaled MMA** instead of
  `cvt` → FP16 → `mma.sync`, removing the conversion work entirely from the largest 4-bit surface.
- The "permanent structural disadvantage vs DGX Spark" I wrote one commit earlier **does not exist.**
  That paragraph was wrong and has been replaced.
- `tcgen05` + TMA + FP4 together are exactly the CUTLASS/DeepGEMM SM100 kernel family. The entire
  "arch-blocked, needs a rewrite" framing inherited from the prior project is void.

### The gate that still stands

**SASS emission proves ISA and assembler support, not silicon correctness.** `tcgen05` alloc/dealloc
and TMA bulk copy are runtime-verified (Finding 29); a *complete* `mxf4nvf4` MMA is not — it needs
tensor-memory allocation, matrix descriptors, and the scale-vector layout wired up before it will
execute. **That runtime verification is the next step and it is the one that decides whether this is
a real lever or a paper one.** Until it passes, this finding is "the instruction exists", not
"the hardware does the math".

`tools/tcgen05_probe.cu` + the extended `scripts/arch_probe.sh` now cover both families.

---

## 2026-08-06 — Tier-1 #1: MoE output-column blocking BN=2. Real, gated, and NOT ENOUGH.

Implemented the top item from `IMPLEMENTATION_PLAN.md`: the MoE GEMV used **one warp per output
column**, so the activation `uint4` pair was re-loaded for every output even though all columns at
the same `k` read the same activation. Now BN=2 columns per warp, activation loaded once, grid
halved, 128 threads/block (the measured optimum).

**Gated:** `gate_fp4_gemv` **cosine 1.0000000**, `maxabs 6.10e-05`; Gate K all MoE paths
**cosine 1.0000000**. Per-column accumulation order over `kb` unchanged.

**Measured:** GEMV **91.0 → 108.4 GB/s (+19%)**. A real improvement to that kernel.

**But it delivers zero end-to-end**, because the GEMV still loses to the m16 mma path
(**121.6 GB/s at M=1**) and is therefore off by default. So the change is banked, not adopted.

### Why we do not reach the 242 GB/s the research measured

The agent's kernel and ours differ in the inner loop, and the gap is instruction count, not memory:

```
ours,  per 16 B weight block (32 fp4 weights):
  32x gv_fp4()   __constant__ LUT lookup + sign branch      SCALAR
  32x gv_e4m3()  __nv_cvt_fp8_to_halfraw + __half2float     SCALAR
  32x scalar FMA
  => ~96 scalar ops per 16 B

theirs (measured 242-249 GB/s):
  __nv_cvt_fp4x2_to_halfraw2   2 weights per instruction
  __hfma2                      2 MACs per instruction
  => ~32 ops per 16 B          = 3x fewer instructions
```

Issue-rate budget: 20 SMs x 4 schedulers x 1.575 GHz = **126 G warp-inst/s**. At 240 GB/s a warp
must consume 512 B in ~270 warp-instructions; **our GEMV needs roughly 3x that.**

**The measured signature confirms it:** our GEMV is **flat at 108-109 GB/s across M = 1, 2, 3, 5, 8**
while the mma path *rises* 121 → 131 over the same sweep. Flat-in-M is the compute-bound signature;
BN=2 removed the activation-reload overhead and simply exposed the next wall.

**So the remaining work on this lever is a dequant rewrite, not a blocking change:** replace the
scalar `GEMV_E2M1` LUT + per-nibble float math with `__nv_cvt_fp4x2_to_halfraw2` + `__hfma2`, i.e.
the same hardware path the grouped mma kernel already uses (`tcm_fp4x2`). That changes accumulation
from f32 to half2 and therefore **cannot** hold the current cosine-1.0 gate — it needs a tolerance
gate, which is a deliberate change of gate class and the reason it is scoped rather than done here.

**Recorded rather than quietly dropped**, because "BN=2 didn't help" is misleading: BN=2 worked
exactly as the research predicted (+19%), and what it revealed is that our GEMV was never
bandwidth-bound in the first place. The research number was achievable — with a different inner loop.

---

## 2026-08-06 — Finding 31: half2 dequant. 2.6x on the MoE kernel, +1.5% end-to-end, and it costs DSpark acceptance.

User authorised the gate-class change. Rewrote the MoE GEMV inner loop from scalar to half2:
`cvt.rn.f16x2.e2m1x2` (2 fp4 -> half2) + `cvt.rn.f16x2.e4m3x2` (2 fp8 -> half2) + `__hfma2`,
replacing 32 `__constant__` LUT lookups + 32 fp8->float converts + 32 scalar FMAs per 16 B.
~96 ops -> ~32. Accumulate in half2 **within** a 32-element block, widen to f32 across blocks, so
precision loss is bounded to 32 accumulations rather than 4096.

### The tolerance gate

The first run printed `cosine=0.9999999 maxabs=3.63e-01 -> FAIL` against the old absolute
threshold. That threshold was the wrong criterion, and this project had already established why
(prior repo, ratio-4/ratio-128 attention gates): per-element absolute/relative error is
**pathological** for deep fp8/fp4 compositions — it grows with K on near-zero outputs while the
result stays correct. The established metric for this class is
**cosine > 0.9999 AND rms_rel < 1e-2 AND max_abs/|o|max < 5e-3**. Re-gated on those:

```
[fp4_gemv] cosine=0.9999999  rms_rel=4.04e-04  max_abs/|o|max=5.01e-04  (|o|max=725.131)  PASS
```
The alarming `3.63e-01` was 5e-4 *relative* — `|o|max` is 725. **These thresholds were adopted with
evidence in earlier work, not invented to admit this change.**

### Speed

| M | mma (GB/s) | **half2 GEMV** | |
|---:|---:|---:|---|
| 1 | 121.5 | **314.6** | **2.59x** |
| 2 | 125.7 | 232.2 | 1.85x |
| 5 | 130.4 | 179.6 | 1.38x |
| 8 | 130.9 | 230.3 | 1.76x |

(The >240 figure is hot-cache inflation; the *relative* comparison is what holds.) The GEMV had
been off by default since the prior project A/B'd it as slower — **it was slower only because of
the scalar inner loop, exactly as the flat-in-M signature predicted.**

**Full model: 101.7 -> 100.2 ms/tok, 9.83 -> 9.98 tok/s.** GATE PASS, and the generated token
sequence is **byte-identical** (`11111 16 455 6102 294 16603 344 29168`) despite the numerics
change — the perturbation is far below the argmax margin at every position.

### The cost, stated plainly

**DSpark acceptance collapsed 3.12 -> 1.00 tokens/verify.** Acceptance is an *exact token*
comparison between draft and target; half2 perturbs the target's logits just enough to break
agreement, even though the greedy output is unchanged. Speculation was already at parity so this
costs nothing measurable today, but it **blocks the speculative path** until resolved.

### Why it cannot simply be scoped to M=1

I tried gating the GEMV to `bs == 1` to keep the base win and leave verify on the mma path. It
produced **argmax 260 instead of 11111**. Root cause: **`tc_ensure_repacked` rewrites each expert
weight IN PLACE into mma-fragment order, and is skipped when the GEMV is active**; the GEMV reads
the ORIGINAL packed fp4. The two paths need mutually exclusive layouts of the same bytes. With the
global flag set but the mma selected at M=5, prefill read unrepacked weights through the mma tile.

Selecting per-M would require either a second copy of the expert weights (**~86 GiB — impossible
here**) or a non-mutating repack (extra bandwidth on every access, which defeats the point).

**So this is a genuine either/or, and it is the right thing to hand over rather than decide
silently:**
- **all-GEMV (current):** 9.98 tok/s base, DSpark acceptance 1.00
- **all-mma:** 9.83 tok/s base, DSpark acceptance 3.12

Kept on GEMV because base decode is the headline number and speculation is at parity either way.
`MOE_MMA=1` restores the mma path in one env var, with no rebuild.

**The resolution that gets both** is to make the *draft* numerically consistent with the target
again — i.e. run the MTP blocks through the same half2 kernel (they already do) *and* re-derive
acceptance, or fine-tune the head against the half2 target. The REAP-repair fine-tune already on
the plan would subsume this, since it retrains the head against whatever the target actually
computes.

---

## 2026-08-06 — Finding 32: BF16-native compressor. Fewer bytes, SLOWER. The axis only pays when the kernel is bandwidth-bound.

The compressor `wkv`/`wgate` (and the indexer's) ship as BF16 and were being expanded to f32 by
`Loader::bf16` — 526 MB/step read as 1052. Exactly the defect the `lm_head` fix (Opt #4) cured, with
the same template. `research/MLA_DECODE.md` ranked it **#1 at ~+5%**, and my own byte model agreed.

Implemented: `gemm_bf16w` path in `compressor_forward` / `compressor_emit_group`, native BF16
pointers in `decode.cu`.

**Correct**: gate PASS, generated tokens byte-identical, resident memory **108.1 → 107.6 GiB**.
**And slower: 100.2 → 103.5 ms/tok (9.98 → 9.66 tok/s).**

### Why the byte model was wrong here

The same reason the MoE GEMV was slow before half2: **`gemm_fp32` is a warp-per-output-column scalar
dot product, and it is compute-bound, not bandwidth-bound.** Going to BF16 halves the bytes but adds
a `__bfloat1622float2` per pair of elements on top of the existing FMA — more ALU per byte, on a
kernel whose limit was already ALU.

**The general rule this establishes, and it retro-explains three earlier results:**

> Reducing bytes only helps a kernel that is bandwidth-bound. On a compute/issue-bound kernel it is
> neutral at best and negative when the narrower format needs conversion.

- `lm_head` BF16 (Opt #4, **+2.7%**) — that GEMM *was* bandwidth-bound (2118 MB in one call, the
  largest single read in the step), so the bytes mattered.
- MoE GEMV pre-half2 — flat at 108 GB/s across M=1..8, compute-bound; BN=2 blocking gave +19% on the
  kernel and **0 end-to-end** until the inner loop was fixed.
- This compressor — compute-bound, so the byte saving is a loss.

**Corrected ordering for every remaining byte lever:** measure whether the consuming kernel is
bandwidth- or compute-bound *first*. If compute-bound, fix the inner loop (half2/vectorised dequant)
**before** narrowing the format — then the byte reduction pays on top. The compressor is now a
strong candidate for a half2 rewrite of `gemm_fp32` itself, after which this change should be
re-tried and should win.

Left in the tree behind `COMP_BF16=1` (default **off**) so the re-try after a `gemm_fp32` rewrite is
one env var away, and the −0.5 GiB memory saving is available if headroom ever binds.

---

## 2026-08-06 — Finding 33: the BF16 compressor left a type-punned pointer behind. Two paths, one weight.

Reverting Finding 32 by flipping `g_compressor_bf16` to default-off did **not** revert it. The
`decode.cu` wiring still handed `W.get(...).dev` — raw BF16 — to `mc_wkv`/`idx_c_wkv`, while the
now-default `else` branch fed those pointers to `gemm_fp32`, which reads them as f32.

Worse, `emit_group_dp` (`compressed_decode.cu:363`) reaches the same pointers through
`gemm_fp32_cond`, which has **no bf16 variant at all** — so on the device-pos decode path the
weights were misread *in both flag states*. The binary I had rebuilt, written up and pushed was
never measured; I reported 9.98 tok/s for it, which was the number from the *previous* build.

Two lessons, and the second is the one that generalises:

1. **A flag that selects a kernel must also select the data.** The dispatch was on
   `g_compressor_bf16`, but the pointer's *type* was decided unconditionally at load. A boolean
   guarding half a decision is worse than no boolean, because it reads as if it guards all of it.
2. **Never write up a config that has not been run.** The rebuild after flipping the default was
   one command; measuring it was one more. I skipped the second and documented the first.

Fixed by making the pointer f32 unconditionally (`L.bf16(...)`). The BF16 experiment can be redone
properly once `gemm_fp32_cond` has a bf16 twin — until then there is one pointer type in the tree.

---

## 2026-08-06 — Finding 34 (Opt #10): `gemm_fp32` was the ILP=1 / one-warp-per-block defect, for the third time.

`gemm_fp32` launched `<<<dim3(N,M), 32>>>` with `acc += a[k]*b[k]`. Both halves of the defect that
Finding 21 fixed in the MoE and Finding 26 fixed in `gemm_bf16w` — one warp per block caps occupancy
at 50%, and a single dependent load→FMA chain leaves memory-level parallelism unused. This call site
was simply never revisited after those two.

float4 + 4 accumulator chains + 4 warps/block, with the Finding 25 runtime alignment check. Same
treatment applied to `gemm_fp32_cond` and to `gemm_bf16w` (float4 = 8 bf16 per load).

**100.2 → 99.9 ms/tok (9.98 → 10.01 tok/s).** Gate PASS, tokens byte-identical.

+0.3% is small, and that is exactly what Finding 32's rule predicts: the compressor is ~4% of
`B_tok`, so even a large efficiency win on it is a small end-to-end win. Adopted anyway because it
is free, and because it is the *precondition* for re-trying the BF16 compressor — the reason that
experiment lost was this kernel's inner loop.

**The pattern is now established well enough to search for rather than stumble on:** every
warp-per-output reduction in this codebase was written correctness-first and inherited both defects.
The remaining ones were found by grep, not by profiling.

---

## 2026-08-06 — Finding 35 (Opt #11): the ogroup GEMV read ONE BYTE per lane.

```
for(int dd=kb*128+lane; dd<(kb+1)*128; dd+=32) acc += og[dd]*ogm_e4m3(wr[dd])*ws;
```

32 lanes × 1 byte = a **32-byte request** where the memory system wants 128. This is the identical
mechanism Finding 15 traced under the M≥2 verify penalty — there the m16 tile issued 8×32B against
the GEMV's 1×128B — and it had been sitting in the M=1 path the whole time. `wo_a` is ~33 MB/layer,
~1.44 GB/token.

One `uint32_t` per lane (128 contiguous bytes per warp) + `float4` activations + 4 accumulator
chains. `ws` is a power of two, so folding it into the weight rather than the product is exact.

**And it had no unit gate.** `tests/gate_ogroup.cu` covers `ogroup_gemm` — f32 weights, bs=8, the
tensor-core tile path. The decode kernel is `ogroup_gemv_fp8_kernel`, reached only via
`ogroup_gemm_fp8` at bs==1: different kernel, different format, different M. The source comment
claimed "Gated cosine vs ogroup_gemm oracle (tests/gate_ogroup_gemv.cu)" — **a file that did not
exist**. The most expensive kernel in attention was the least tested one, and the comment said the
opposite. Wrote `tests/gate_ogroup_gemv.cu` (vec4 vs scalar vs f32 oracle, cosine 1.000000000) and
registered it in `build_gate.sh`.

---

## 2026-08-06 — Finding 36 (Opt #12): `k_rsqrt` was one block, and k_mixes had already loaded its input.

`hc_pre` ran `k_rsqrt<<<bs,256>>>` — at decode bs=1, i.e. **one block on one SM** to reduce 64 KB —
and every `k_mixes` block then waited on its result. But `k_mixes` streams the whole of `xr`
already, so it can accumulate Σx² from the same registers for one extra FMA per element and zero
extra loads. The redundancy across the 24 blocks is free; they all re-read `xr` from L2 regardless.

Together with Opt #11: **99.9 → 96.9 ms/tok (10.01 → 10.32 tok/s), +3.1%.** Gate PASS, tokens
byte-identical.

---

## 2026-08-06 — Finding 39 (S1): draft acceptance was 0/4 because the GEMV read f32 scales as e8m0 BYTES.

Finding 31 recorded that half2 "costs DSpark acceptance (3.12 → 1.00)" and framed it as a precision
trade: half2 perturbs the target's logits just below the argmax margin, and acceptance is an exact
token match. I handed that to the user as a real either-or. **It was a bug, and the write-up was
wrong.**

Two things gave it away. First, **0/4 accepted on 24 of 24 verifies**. Margin noise gives occasional
acceptance; a hard zero is structural. Second, `git show` on the half2 commit: it did not only
rewrite the kernel, it flipped the default —

```
-  g_moe_gemv = (getenv("MOE_GEMV") != nullptr)   // default OFF -> mma path
+  g_moe_gemv = (getenv("MOE_MMA")  == nullptr)   // default ON  -> GEMV path
```

Every 3.12 run was on the mma path and every 1.00 run on the GEMV path. The variable that changed
was **which kernel runs**, not its precision — and I attributed the effect to the change I had been
thinking about rather than the one I had made.

**Root cause.** `moe_forward` selects the GEMV on `use_gemv` alone and casts the scale tables:
`tc_fp4_grouped_gemv_e8m0(..., (const uint8_t* const*)s1d, ...)`. The GEMV reads them as e8m0
exponent bytes (`exp2f(Sn[kb]-127)`) and has no f32-scale variant. `fill_moe` sets
`e8m0_scales=true` for the main model — but the DSpark draft blocks (`decode.cu:458`) filled theirs
with `LH.scale(...)`, a **dequant to f32**, and left `e8m0_scales` false. The mma path branches on
that flag correctly; the GEMV silently reinterpreted the low byte of each float as an exponent. The
draft's MoE had been computing garbage for eleven runs, while the main path — which *does* set the
flag — kept passing its argmax gate.

**Fixed** by giving the draft native e8m0 scale bytes (also faster: it now uses the GEMV too), and
by making the precondition part of the predicate so the two can never disagree again:
`const bool use_gemv = g_moe_gemv && w.e8m0_scales;`

**Acceptance 1.00 → 3.00 on the default path.** Gate PASS.

**The lesson is about the cast.** A C-style cast on a `void*`-shaped pointer table is a silent
type pun: it made an f32/e8m0 configuration error unrepresentable in the type system and
undetectable at runtime. The unit gates could not see it either, because they construct their own
weights and set the flag consistently.

---

## 2026-08-06 — Finding 40 (S2): acceptance was never the blocker. `c_v` is.

With acceptance restored to 3.00, speculation still measured **0.92× of base** (9.45 vs 10.27).
The M=5 verify costs 279 ms against a 97 ms M=1 step: **c_v = 2.88**. The K=1 → K=5 sub-phase
profile shows exactly where, and separates the legitimate from the wasted:

| phase | K=1 | K=5 | ratio | expected |
|---|---|---|---|---|
| `moe:w1w3` / `moe:w2` | 12.40 / 5.79 | 52.67 / 24.45 | 4.25× | **~4.2 ✓** expert union — unavoidable |
| `cattn:ogroup` | 16.63 | 64.34 | **3.87×** | ~1.0 — identical weights for all 5 tokens |
| `cattn:q_proj` | 11.35 | 34.32 | **3.02×** | ~1.0 |
| `moe:shared` | 7.55 | 23.15 | **3.07×** | ~1.0 |

~85 ms of the 235 ms verify is re-reading weights that should be read once. Mechanism is Finding
15's: the m16 mma tile wastes 11 of 16 rows at M=5 *and* issues 8×32B requests per fragment.

**Fix applied to ogroup** — a templated M=K GEMV (`ogroup_gemv_mk_kernel<M>`, M=2..8) that loads
each weight row once and dots it against all M activation rows. The stale comment at that call site
claimed "M=K GEMV A/B'd SLOWER: acc[bs] array kills occupancy"; that was the *untemplated* version,
whose `acc[]` was sized to the maximum M — precisely the defect Finding 28 fixed for the dense fp8
GEMV. Templated, `acc[]` is M registers.

**M=5 verify 279.4 → 253.7 ms, c_v 2.88 → 2.60, speculation 0.92× → 0.98×.** Verify argmax matches
the AR tokens exactly. Base decode unchanged.

**Still open, same treatment, ~38 ms:** `cattn:q_proj` and `moe:shared` both go through
`fp8_block_gemm` at M=5 and scale ~3×. `fp8_gemv_mkT_kernel<M>` already exists (Finding 28) — it
needs to be reached at M=5 for these two call sites.

---

## Finding 41 — the m16 tile was fetching every weight sector for half its payload

`tc_fp8_gemm` is reached for every dense FP8 GEMM at M>=2, which is the whole spec-decode verify.
Its cost looked like an mma-tiling problem for seven rounds. It was not. On wq_b [32768,1024] with
12 rotating copies (so nothing is served from cache):

|          | M=1  | M=2  | M=3  | M=5  | M=8  |
|----------|------|------|------|------|------|
| HOT      | .145 | .248 | .353 | .153 | .203 |
| COLD     | .157 | .499 | .505 | .488 | .503 |

**A kernel 3.2x slower cold than hot is neither compute- nor bandwidth-bound — it is fetching bytes
it does not use.** The B operand was `*(const unsigned*)(B + nn*K + k0 + tid4*4)`: four lanes cover
16 contiguous bytes of one row, eight rows per warp, strided by K. Sixteen bytes is half a 32-byte
sector and the other half arrives on a different instruction, so cold, every sector is fetched twice.

Fix: stage a [8W x 128] B tile through shared memory with eight consecutive lanes walking 128
contiguous bytes of one row, then read mma fragments from smem. The tile is not reused — this buys
the global access pattern and nothing else. Row stride padded 128 -> 144 B so the eight lane groups
land on banks 4g..4g+3 (a permutation of all 32) instead of colliding 8 ways on bank 0. Tile height
is adaptive: at 8 warps, N=512 gives 8 blocks for 20 SMs, so W shrinks until the grid is >= 64.

Cold, on the six shapes decode issues, at M=5: **wo_b 2.9x, sw2 2.5x, wq_b 2.1x, wq_a/wkv 1.7x,
sw1/3 1.6x** — and flat in M to 16, which is what makes a larger speculation block thinkable.

Three things this cost, all worth keeping:

1. **The comparison that set the old cutoff was never run.** `fp8_block_gemm` capped the small-M
   GEMV at M<=4 citing "a wash at M=5 (1.04x)" from gemm_bench's COLD+GEMV_MK row. That row called
   `setenv("GEMV_MK",...)`, and nothing has read `GEMV_MK` since the dispatch was rewritten — it had
   been measuring the default path against itself. A cached `getenv` cannot be re-read inside one
   process, so both knobs are now setters (`fp8_set_gemv_mk_maxm`, `tc_fp8_set_smem`).
2. **Against the fixed tile the GEMV loses everywhere at M>=2**, by 1.5-2.3x even at M=2..4 where it
   was the default. Cutoff moved to M=1. It is exactly linear in M (0.267/0.356/0.517/0.770 at
   M=2/3/5/8) because it decodes the same activation bytes once per warp per m per k.
3. **A shared-A variant of that GEMV was SLOWER at every M** — retired with the measurement. It
   trades L1 traffic for 4x the smem traffic (float4 per m per k where the fp8 read was 4 bytes).

**And the gate could not have caught the bug that shipped.** The staged uint4 load crashed prefill
immediately (`misaligned address` -> wo_b): every weight is a pointer into a mapped safetensors file
and is only 4-byte aligned, while a gate that allocates B with `cudaMalloc` always gets 256. The
kernel is now templated on alignment (1x uint4, or 4x unsigned covering the same 16 bytes and the
same cache lines), and the gate runs every shape and M at B+0 *and* B+4. Unaligned costs 10-20% and
still wins 1.5-2.4x. General rule: **a gate that allocates its own inputs cannot test alignment.**

## Finding 42 — the lm_head was read once per token in the block

`gemm_bf16w` is one warp per OUTPUT ELEMENT with `grid(N/wpb, M)`, so at M>1 it reads all of B M
times. For every other caller B is a few MB. For the lm_head B is [129280, 4096] bf16 = **1.06 GB**,
and the verify calls it at M=BLK: 5.3 GB of DRAM per verify to produce five rows — more traffic than
any single MoE phase. Same fix as ogroup_gemv_mk (Finding 40): one warp per n, B row read once,
dotted against all MM activation rows. A is re-read per warp but is 80 KB, i.e. L2-resident.

Templated on M=2..8 *and* on B alignment, for the Finding 41 reason: falling back to the M=1 kernel
when the base is only 4-byte aligned would silently restore the M-times-B-traffic behaviour, which
is the one outcome worse than either fast path.

## Measured together, full model, canonical prompt

| | before | after |
|---|---|---|
| base AR (M=1, control — neither change touches it) | 97.6 ms/tok | **97.4 ms/tok** |
| M=5 verify, one forward | 253.7 ms | **200.6 ms** |
| c_v | 2.60 | **2.06** |
| per-verify cycle (draft + verify) | 310 ms | **250 ms** |
| speculation vs base | 0.98x | **1.18x** |
| **spec-decode** | 10.04 tok/s | **12.12 tok/s** |

GATE PASS, verify argmax == AR tokens 5/5, generated token stream byte-identical.
Acceptance unchanged at 3.00/5 — this is entirely `c_v`, which was the correct target.

## Finding 43 — two more "read B once" fixes, and the block-size question closed

**Block size is not a lever.** `DSV4_BLKSWEEP` runs the spec loop once per block size in one process
(each point re-prefills, so they are independent), which turns a four-point sweep of the most
consequential speculation parameter from four 10-minute checkpoint loads into one:

| BLK | mean tok/verify | ms/tok | tok/s | vs base |
|---|---|---|---|---|
| 5  | 3.00 | 69.7  | 14.35 | **1.39x** |
| 8  | 3.00 | 94.1  | 10.63 | 1.03x |
| 12 | 3.00 | 130.2 | 7.68  | 0.74x |
| 16 | 2.89 | 178.9 | 5.59  | 0.54x |

The per-verify accept sequence at BLK=8 is *identical* to BLK=5 — 1,1,1,1,4,1,4,1,4 — so the fifth
through eighth proposals are always wrong. The draft saturates at four correct tokens in its best
case and one in its common case, and extra block width buys only verify cost. With three chained MTP
stages that is a property of the head, not of the search. **Acceptance, not block width, is what is
left on the speculation side.** Retired with a measurement; do not re-queue block-size sweeps.

**Finding 42's defect twice more.** `ogroup_gemv_mk` reads the weight once per M rows but M*512
bytes of f32 `o` per 128 weight bytes — 4M times the weight traffic, 20x at M=5, ~27 GB of L2 per
verify. `o` depends only on the group and gr = g*R + r, so NR consecutive rows share it exactly:
give each warp NR=4 rows and the ratio falls to 4M/NR. And `gemm_fp32` is warp-per-output-element,
so `compressor_emit_group` read its two 16.8 MB f32 weights eight times each (M = 2*ratio = 8) per
group emit; chunked in 8s because ratio is 128 for twenty layers and a 128-row template would spill.

Both gate **bit-identical** to the kernels they replace (rms = 0.0 at every M, including the chunk
tails at M=13 and the sixteen-chunk M=128), because both keep the operation order exactly.

| phase (K=5, ms) | before | after |
|---|---|---|
| cattn:ogroup | 28.02 | 24.28 |
| cattn:compress | 10.11 | 8.46 |
| M=5 verify | 179.4 | **173.5** |
| spec-decode | 14.36 tok/s | **14.76 tok/s** |

**Both under-delivered against prediction by more than 2x** (predicted ~10 and ~7 ms, got 3.7 and
1.7), which by the flywheel's own rule means the ranking model is wrong, not the levers. It is: both
phases are chains of small kernels — a group emit is ~8 launches per layer across 21 layers — and
the byte model attributes all of a phase's time to its GEMMs when much of it is launch-bound glue
moving almost nothing. **The next ranking pass must separate byte-carrying time from glue time.**
Doing that on the K=1 profile splits base AR cleanly: ~59 ms carrying 8.67 GB (147 GB/s) and ~37 ms
of glue carrying almost nothing — and the 37 ms is the larger of the two gaps to the 240 GB/s roof.

## Finding 44 — the roofline the whole log was measured against is not the one the weights live on

Every byte-carrying phase in the engine sat in a narrow band well under 240 GB/s, and the band was
the same for kernels with nothing else in common (K=1, in situ): `cattn:ogroup` 170, `moe w13+w2`
168, `lm_head` 183, `cattn:q_proj` 142. **A uniform ~70% across unrelated kernels is not a kernel
property.** The one thing they share is where the weights live.

`WeightStore` preads each shard into `cudaHostAlloc(cudaHostAllocMapped)` and hands kernels the
`cudaHostGetDevicePointer` alias — zero-copy mapped host memory, chosen because the model is 100.4
GiB in a 122 GiB unified pool and a device copy would double it. `ROOFLINE.md`'s 240 GB/s came from
`bw_probe`, which measures a **`cudaMalloc`** buffer. `tools/alloc_probe` runs the same streaming
kernel on the same bytes out of four allocators:

| allocator | 4 GiB stream / strided | 16 GiB stream / strided |
|---|---|---|
| `cudaMalloc` (device) | 233.8 / 235.0 | 227.9 / 234.1 |
| `cudaHostAlloc` Mapped — **where the weights were** | 180.2 / 194.3 | 184.8 / 215.7 |
| `cudaMallocManaged` | 181.8 / 234.8 | 177.9 / 232.2 |
| `cudaMallocManaged` + PreferredLocation + prefetch | 231.3 / 235.1 | 200.4 / 232.9 |

Moved the weights to managed memory with the preferred location set to the device and the range
prefetched (`DSV4_WEIGHTS=mapped` restores the old allocator; a failed managed allocation falls back
automatically rather than dying after a ten-minute load).

| | mapped | managed |
|---|---|---|
| base AR | 96.5 ms/tok | **92.8** |
| base AR + CUDA graph | 85.4 | **81.3 (12.30 tok/s)** |
| M=5 verify | 173.5 | **168.8** |
| spec-decode | 14.76 tok/s | **15.49 tok/s (1.44x)** |

**Predicted −11 ms, measured −3.7.** The 4 GiB probe says 1.21x on strided reads; the 16 GiB probe
says 1.08x, and the engine's access pattern is strided, not streaming. Sizing a probe to the working
set matters as much for the allocator as it did for the weights (Finding 41's HOT/COLD rows).

**Two corrections to the log, both of which change future rankings:**

1. **The achievable figure for this engine is ~233 GB/s strided out of managed memory, and was ~216
   out of mapped.** Every "% of achievable" recorded against 240 before this was measured against a
   ceiling the weights could not reach. The kernels were never as bad as the column implied — but
   they are not at the roof either: 158-183 GB/s against 233 is 68-78%, so real headroom remains.
2. **`B_tok`/BW puts the base-AR ceiling at 12.26 GB / 233 = 52.6 ms = 19.0 tok/s.** Graph-captured
   base AR is 81.3 ms = 12.30 tok/s, i.e. 65% of it.

## CUDA graph re-gate (G9)

The full-step 43-layer capture behind `GRAPH=1` had not been re-gated since any of this work.
**92.8 -> 81.3 ms/tok, 1.14x, GATE PASS, tokens identical.** That is ~11.5 ms of launch overhead in
a step with ~600 launches, and it is the first direct confirmation of the Finding 43 correction that
glue — not GEMM bandwidth — is the larger of the two gaps to the roof. The verify path is a
comparable kernel chain and is not yet captured.

## Session cumulative

| | start | now |
|---|---|---|
| base AR | 9.98 tok/s | 10.78 (12.30 with graph) |
| spec-decode | 10.04 tok/s | **15.49 tok/s** |
| speculation vs base | 0.98x | 1.44x |
| M=5 verify | 279.4 ms | 168.8 ms |
| c_v | 2.88 | 1.82 |

## Finding 45 — draft refinement makes acceptance WORSE, and the ceiling arithmetic that follows

The draft's three MTP blocks see `DSPARK_NOISE_TID` at block positions 1..BLK-1, so they condition
the entire block on a token carrying no information; only the markov head at the output does any
sequencing. The obvious fix is a second pass with the first pass's own proposals in those slots.
It is free of correctness risk by construction — the draft is a proposal, the verify is unchanged —
so the only question is whether acceptance rises by more than the extra block chain costs.

| draft passes | mean tokens/verify | ms/tok | tok/s |
|---|---|---|---|
| 1 | **3.00** | 64.6 | **15.49** |
| 2 | 2.08 | 105.8 | 9.46 |

**It does not cost acceptance a little; it destroys it (3.00 -> 2.08).** The heads were trained with
the noise token as the placeholder, so a plausible token in that slot is *off-distribution* — the
opposite of the intuition. Retired with a measurement. `DSV4_BLKSWEEP="5:1,5:2"` reproduces it in
one process. (3 passes also overflowed the arena, since each pass re-dmallocs the chain; fixed with
an `arena_reset()` per pass so the knob stays usable, not because refinement is worth revisiting.)

### The ceiling, from measured bytes

With acceptance fixed at 3.00 by the shipped heads and flat in block size (Finding 43), the only
remaining question is how close the cycle can get to its byte floor. At the corrected 233 GB/s
(Finding 44), per-expert 12.58 MB fp4, dense remainder 9.01 GB:

| | bytes | floor | measured |
|---|---|---|---|
| base AR | 12.26 GB | 52.6 ms (19.0 tok/s) | 92.8 (81.3 w/ graph) |
| verify K=5 (union ~25) | 22.54 GB | 96.7 ms | 168.8 |
| draft (3 MTP blocks + lm_head) | 2.3 GB | 9.9 ms | 27.3 |
| **cycle** | | **106.6 ms** | **201.8** |

**Measured `c_v` is 1.82 against a byte-model floor of 1.84 — the verify's K-scaling is already
optimal.** The whole remaining 95 ms is a *uniform* 1.89x inefficiency present equally at K=1 and
K=5, the same factor that makes base AR 92.8 instead of 52.6. That is the useful conclusion:
**there is no speculation-specific work left; every ms of base-AR efficiency now converts
proportionally into the spec number.**

At the floor, with acceptance 3.00: **28.1 tok/s**. The target band needs:

| target | tokens/verify required |
|---|---|
| 28 tok/s | 3.00 (measured) — reachable by kernel work alone |
| 35 tok/s | 3.70 |
| 50 tok/s | 5.30 — **exceeds BLK=5's maximum of 5; not reachable at any kernel speed** |

Raising block size does not help (Finding 43: the accept sequence is identical at BLK=5 and 8).
So 35+ requires either a REAP-repair fine-tune of the MTP heads (S3 — a training task, outside a
pure-CUDA inference server) or fewer bytes, i.e. quantising MLA (41% of `B_tok`), which the
project's non-negotiables forbid. **Both are decisions for the user, not defaults to adopt.**

## Finding 46 — the indexer was rebuilding `qr` from scratch, and the verify has no launch gap to graph

**Redundant recompute.** `compressed_verify_step_indexer` needs `act_quant(rmsnorm(wq_a @ x))` for the
indexer's query projection. `build_qKV` had just computed exactly that and freed it. The verify
rebuilt it from `x`: `act_quant(x)` + the whole `wq_a` GEMM + `rmsnorm` + `act_quant` — four kernels
and **4.19 MB of redundant weight traffic per layer across the 21 indexer layers**, under a comment
calling the recompute "cheap". The two M=1 decode paths had the same defect in a smaller form: a
second `act_quant_fp8(qr, Q_LORA, 128)` producing bytes identical to the `qrq/qrs` computed twenty
lines earlier on an unmodified `qr`. Hand the buffers out of `build_qKV` instead.

| phase (K=5, ms) | before | after |
|---|---|---|
| `cattn:indexer` | 10.74 | **9.27** |
| `ATTENTION` | 65.17 | **62.43** |
| dprof TOTAL | 175.42 | **170.59** |
| base AR | 92.9 | 92.1 (graph 80.4 = 12.44 tok/s) |
| spec-decode | 15.32 | 15.27 — **inside the 15.3-15.5 band** |

Real but ~0.7% of the cycle, i.e. below the noise floor of a single end-to-end run.

**And the profile retires verify-path graph capture (G2) before it was written.** The K=5 dprof
TOTAL is 170.59 against a measured verify of 168.5 — **the marks account for ~101% of the step, so
there is essentially no time between phases in the verify.** Compare base AR at K=1: TOTAL ~83
against 92.1 measured, a ~9 ms between-phase gap, and the CUDA graph recovered 11.7 ms. The graph's
win on base AR was almost entirely that between-layer host work (arena reset, per-layer setup); the
verify does not have it. Scaling the residual ~2.7 ms of within-phase recovery by the verify's
larger size puts a verify graph at **~2-3% for a device-pos rewrite of every verify kernel.** Not
worth it at this position in the queue. Requeue only if the between-phase gap reappears.

**Phase A stopping rule reached.** The last three levers measured +0.4% (ogroup NR), ~0% (fp32 M=K,
inside noise) and +0.7% (this one). Glue fusion is exhausted as a percentage lever: per-call glue
costs 4-12 us against phases of 8-25 ms, and the remaining fusions (rmsnorm+act_quant, rmsnorm+rope)
would remove ~4 launches/layer for an estimated 0.5-0.8 ms in the graph-captured path. What is left
above 1% is kernel bandwidth — the MoE at 177 GB/s and `cattn:ogroup` at 121 — not launch count.

## Finding 47 — ncu: both kernels are latency-bound, not bandwidth-bound; and the bench overstates every kernel win

`tools/ncu_target.cu` + `scripts/ncu_probe.sh` profile exactly the two kernels the queue said were
the whole remaining gap, one launch each at the real shapes. (First attempt profiled the MoE *mma*
kernel — the engine's default is the GEMV, selected inside `moe_forward` by `g_moe_gemv`. Same class
of mistake as the stale `GEMV_MK` bench row in Finding 41: profiling a path decode does not take.)

| metric | MoE GEMV, M=5 | ogroup GEMV, M=5 NR=8 |
|---|---|---|
| Memory throughput | 25.4% | 14.3% |
| Compute (SM) throughput | 37.2% | 37.6% |
| Mem pipes busy | 8.8% | 9.4% |
| Executed IPC | 1.35 of 4 | 1.60 of 4 |
| Registers/thread | 52 | **128** |
| Theoretical / achieved occupancy | 75% / 64.5% | **33.3% / 33.2%** |
| **`long_scoreboard` stall** | **20.44 of 24.35 warp-cycles (84%)** | 6.62 of 9.97 (66%) |

**Neither kernel is bandwidth-bound.** Nothing is above 40% of peak and the dominant stall in both
is `long_scoreboard` — waiting on global loads. They are latency-bound with too few requests in
flight, which is Little's Law again (Finding 20 / Opt #7) in two more places.

Two different causes, so two different fixes:

- **ogroup** materialised `d[NR][4]` — the dequantised weights for all NR rows — before touching the
  activations. At NR=8, M=5 that is 40 accumulators plus 32 weight floats, and the kernel compiled
  to 128 registers, capping it at 2 blocks/SM. Restructured to issue the NR raw weight loads and M
  activation float4s up front (the MLP), then dequantise one row at a time into 4 reused registers,
  plus `__launch_bounds__(256, 4)` to force the register cap. `long_scoreboard` 6.62 -> 4.32.
- **MoE GEMV** issued and consumed each weight pair inside the same `u` iteration. Hoisted all BN
  pairs and their scales ahead of the compute. Registers 52 -> 48, achieved occupancy 70% -> 76%.

Both gate **bit-identical** (rms 0.0 at every M).

### The result, and the pattern it completes

| | bench (cold, 400 MB rotation) | in situ (100 GiB working set) |
|---|---|---|
| ogroup wo_a M=5 | 0.272 -> **0.200-0.228 ms** (−20 to −26%) | `cattn:ogroup` 22.39 -> **21.62 ms** (−3.4%) |
| MoE GEMV M=5 | 0.5402 -> 0.5398 (cache-served, invalid) | `moe:w2` 24.19 -> 22.36, `w1w3` 52.27 -> 52.95 |
| | | verify 168.5 -> **167.5**, spec **15.29 tok/s** (flat) |

**This is the fourth consecutive time a kernel win measured in `gemm_bench` has arrived 2-4x smaller
in situ** (ogroup NR, gemm_fp32 M=K, indexer dedup, and now this). The cause is now clear enough to
write down: **the bench launches the same kernel repeatedly on rotating weights, so consecutive
launches overlap — the tail of one drains while the next ramps. In the engine every kernel is
serialised by a data dependency on the previous one, so its tail wave is fully exposed.** ncu says
ogroup runs 3.2 waves, i.e. a partial wave worth up to 25% of its runtime, and the bench hides
exactly that. `gemm_bench` is a valid instrument for *which of two kernels is faster* and an invalid
one for *how much a kernel is worth end-to-end*.

**Phase A is over.** Levers are now landing at under 1% each while the measurement noise band on a
full-model run is +/-1%. The remaining gap is not a kernel that can be tuned; it is the serialised
tail of ~600 dependent launches, which is a scheduling problem, not a bandwidth or occupancy one.

## Finding 48 — both scoped structural items retired with measurements

The queue's two remaining structural items were verify-path CUDA-graph capture and a persistent
megakernel. Both are now measured or costed rather than estimated, and neither survives.

### Verify graph: 1.05x, measured without writing it

A reusable verify graph needs device-pos variants of every verify kernel (~400 lines, mirroring the
decode path's `_dp` family) because `Tf`, `ntot`, `topk` and `wmax` all move with `pos` and `T`.
But the *question* needs none of that: capture ONE verify at a fixed position and replay it. The
replay recomputes the same position so its outputs are meaningless — the **time** is exactly what a
device-pos verify graph would be worth. Forty lines to de-risk four hundred (`VERIFYGRAPH=1`).

Prerequisite: `cudaMemcpyAsync` from PAGEABLE host memory is not capturable, and the host-built
`comb` arrays were the only uncapturable thing left in the verify (the indexer variant, 21 of 43
layers, already builds its comb on device via `k_comb_verify`). One grow-only pinned staging buffer
fixed all three sites.

```
[vgraph] M=5 verify, 2788 graph nodes: ungraphed 163.9 ms -> graph 156.3 ms (1.05x)
```

**7.6 ms, against base AR's 1.17x / 13 ms on a comparable node count.** The reason is not subtle:
at M=5 every kernel does ~4x the work of its M=1 counterpart, so the CPU stays ahead of the GPU and
launch overhead is hidden. **Base AR is launch-bound; the verify is not.** 3.8% of the cycle for a
400-line device-pos rewrite of every verify kernel — retired. Requeue only if the verify's kernels
get much faster (which would re-expose the launch cost).

### Megakernel: negative expected value, on this project's own measurements

`src/megakernel.cu` exists but is a dead port artifact: 160 lines fusing `input_rmsnorm + Q/K/V
projection` for a plain-QKV FP4 architecture with per-tensor group scales. DSV4 has MLA
(`wq_a -> q_norm -> wq_b`, separate `wkv`), hyper-connections, a compressor and a DSA indexer. It is
not wired into `decode.cu` and none of it is reusable. A DSV4 megakernel is a from-scratch project,
and three measurements already on the board say it would lose:

1. **Register pressure.** A fused kernel compiles to the max of all its stages. ncu (Finding 47) has
   `ogroup` at 128 registers / 33% occupancy and the MoE GEMV at 48 / 75%. Fusing them runs *every*
   stage at 33% — and ncu says both stages are latency-bound and want *more* warps, not fewer. The
   megakernel makes the diagnosed problem worse by construction.
2. **Barriers.** ~25 stages per layer x 43 layers is ~1075 grid-wide syncs per token. At a few us
   each on 20 SMs that is 2-5 ms — a large fraction of the 13 ms of launch overhead it would remove.
3. **The prize is already measured and it is small.** The launch overhead a megakernel targets is
   13 ms in base AR (which CUDA graphs already recover, at zero occupancy cost) and 7.6 ms in the
   verify. There is no third pot.

Retired before implementation, with the arithmetic recorded.

### And the remaining gap is not allocation fragmentation

The engine reads 45,821 tensors across ~20 managed shard allocations while `alloc_probe` reads one
contiguous buffer, so page-table locality was the obvious next suspect. Same bytes, same kernel,
split N ways and walked in sequence: **192.1 / 182.9 / 181.4 / 183.3 / 183.6 GB/s at N = 1 / 4 / 16
/ 64 / 256.** A 5% effect, not the 1.5x being hunted. Ruled out.

What is left is what ncu already said: latency, with memory throughput at 14-25% and
`long_scoreboard` at 66-84%. Base AR moves 12.26 GB in 79.3 ms = 155 GB/s of weight traffic against
a 234 GB/s strided ceiling. Every structural fix for that has now been tried and measured.

## Finding 49 — Phase D, run 1: the loop had been treating a decision variable as a constant

**Trigger.** Three consecutive levers under 0.5% and two Phase-B re-rankings returning the same top
entry. Per the revised entry condition (DECODE_FLYWHEEL §Phase D), a queue that keeps re-deriving
the same answer signals an exhausted *model*, not exhausted levers.

**Method.** `RESEARCH_PROMPT_v2.md` — written against the measured residual, not the topic. v1 asked
"how do we move more bytes"; that question was answered by Findings 44/47 and it was the wrong one.
v2 asks six questions, each naming the measurement that would falsify its answer, and §Q6 asks what
the project is *not* asking. Queried the arXiv API directly (`abs:"speculative decoding" AND
abs:"mixture of experts"`, sorted by date) rather than web search: complete, dated, not SEO-shaped.

**What came back.** The residual the loop had accepted as physics —

> *"the verify at K=5 reads the union of ~25 of 160 experts per layer instead of 6, so the MoE (52%
> of the step) gets almost no weight-sharing benefit from batching; measured `c_v` 1.82 against a
> byte-model floor of 1.84, so the K-scaling is already optimal"*

— is an actively-worked 2026 problem with a name, **expert scattering**, and a family of solutions,
several of them training-free *and* lossless:

| paper | arXiv | what it does | lossless? | reported |
|---|---|---|---|---|
| **EVICT** | 2605.00342 | truncates the draft **before** verification, keeping only the cost-effective prefix, from drafter signals + offline-profiled verify cost | **yes** | 2.35x over AR, 1.21x over EAGLE-3 |
| **EcoSpec** | 2607.12696 | adds predicted **marginal expert activation cost** to draft selection; favours paths reusing already-covered experts | **yes** (target verification rule unmodified) | up to 1.62x on DeepSeek-V3.1 |
| MoE-Spec | 2602.16052 | verification-time expert budgeting, loads only high-contribution experts | no (approximates) | 10-30% over EAGLE-3 |
| EdgeXpert | 2608.05303 | prompt-wise expert reuse + depth-aware coalescing, **edge devices** | — | 56.3% latency reduction |
| AcceptMoE | 2608.02989 | verifier-side expert selector from target-router + commitment scores | no (-0.27 pp) | 2.06x under offloading |
| SP-MoE | 2510.10302 | SD-aware expert prefetching from draft/target correspondence | yes | 1.07-3.5x TPOT |

**The correction to our own model.** Findings 43/45 concluded "acceptance is fixed at 3.00 and `c_v`
is at its byte floor, therefore speculation is finished". Both halves are true and the conclusion
still does not follow: **the floor itself is a function of how many positions we choose to verify,
and we were choosing a constant 5.** The union is a decision variable. Seven rounds treated it as a
constant of nature.

**The lever this yields here.** Our draft is a linear chain, not a tree, so EcoSpec's path selection
has nothing to choose between — but EVICT's truncation applies directly as **adaptive verify width**.
Priced from our own ksweep, with the draft head's top1-top2 logit margin as the drafter signal
(computed for free inside the argmax it already does):

```
cycle(K)          106.9  137.9  161.3  176.6  198.5 ms   for K=1..5
marginal cost            31.0   23.3   15.3   21.9 ms
break-even P(accept)     0.47   0.35   0.23   0.33        (marginal_ms / 66.2 ms-per-token)
measured marginal acceptance rate: 0.33
```

**The measured acceptance rate straddles its own break-even thresholds** — precisely the regime
where a per-verify signal beats any fixed width. Oracle bound on the observed accept pattern (six
verifies accepting 1, three accepting 4): **18.97 vs 15.12 tok/s, +25%.**

Truncation is **lossless by construction**: verifying fewer proposals cannot change what the target
emits, only how many tokens one verify commits. No accuracy argument is needed, which is what
separates this from S5/AcceptMoE.

**Implementation.** `k_argmax_row` gained an optional top1-minus-top2 output (one extra shared array
and reduction over an already-loaded row; the vocab=129,280 scan is unchanged). `DSV4_BLKSWEEP`
entries became `BLK[:passes[:adaptK]]` so thresholds sweep in one checkpoint load. `adaptK=0` is
byte-identical to the previous behaviour, so the sweep carries its own control.

Two bugs found by printing the signal instead of trusting it:
- **Off-by-one in the gate.** `VK=k` verifies `[cur, draft[0..k-2]]`, so extending to `k+1` adds
  `draft[k-1]` and must gate on `hmarg[VK-1]`. The first version gated on `hmarg[VK-2]` — a token
  already inside the block, which always passes for a confident `draft[0]`.
- **The signal is not perfectly calibrated.** Verify 3 had margins `5.15 5.09 7.23 0.74 2.69` and
  still accepted only 1: the drafter was confident about a token the target did not choose. So the
  threshold is fitted from a printed calibration set (every verify's margins against what was
  actually accepted) rather than swept blind across 15-minute runs.

**Next lever if the margin proves too weak:** the draft's own MoE routers pick experts at every
block position over the same 160 experts. The union of *those* picks is a direct on-device estimate
of the union the verify would activate — EcoSpec's "lightweight expert predictor" without a separate
model. Signal already computed, currently discarded.

### Finding 49, measured: +2.1%, lossless, and the mechanism explains why it is not more

Within-run A/B (same prompt, same state, four thresholds in one checkpoint load), 61 tokens over
18 verifies at every setting:

| adaptK | total ms | tokens | ms/tok | tok/s |
|---|---|---|---|---|
| 0.00 (fixed width — control) | 3694 | 61 | 60.5 | 16.52 |
| 0.50 | 3666 | 61 | 60.1 | 16.65 |
| **1.50** | **3616** | **61** | **59.3** | **16.86** |
| 4.00 | 3635 | 61 | 59.6 | 16.78 |

**The generated token sequence is byte-identical at all four thresholds** — losslessness confirmed
empirically, not just argued. +2.1%, and because it is a within-run A/B rather than four separate
15-minute runs, it sits above the +/-1% cross-run noise band.

**The drafter's margin is genuinely predictive** — the calibration set (every verify's margins
against what the target actually accepted) gives:

| position | reached | accepted (mean margin) | rejected (mean margin) | AUC |
|---|---|---|---|---|
| draft[0] | 18x | 18, 4.96 | 0 | — |
| **draft[1]** | 18x | 9, 3.67 | 9, 1.25 | **0.78** |
| draft[2] | 9x | 8, 4.60 | 1, 0.37 | 1.00 |
| draft[3] | 8x | 8, 3.81 | 0 | — |

So the signal is not the limit. **The cost structure is**, and the per-K table says why:

```
K   mean ms/verify   mean tokens   ms/token
2       142.4            2.00         71.2     <- truncating all the way is the WORST rate
3       165.1            2.50-3.00    55-66
4       180-185          3.50-4.00    46-53
5       203-205          3.39-3.83    53-61
```

Two facts kill the large win my own arithmetic predicted:

1. **The fixed part of a verify is 54% of the widest one** (`cycle(1)` 106.9 vs `cycle(5)` 198.5).
   Truncation can only ever recover the other 46%, and only on the verifies it truncates correctly.
2. **Our expert union at K=5 is ~25 of 160 = 15.6%.** EVICT and EcoSpec report 1.2-1.6x in a regime
   where a draft **tree** of 30+ nodes scatters across a far larger union. **A linear chain of 5 has
   very little to cut.** The mechanism is right; our instance of it is small.

This is the honest shape of the result: the frame correction in Finding 49 was real and the loop was
wrong to treat the union as a constant — but for *this* draft topology the decision variable has a
narrow range. **The technique's value here is bounded by the draft being a chain, not a tree**, and
building a tree draft is gated by the same thing everything else is: acceptance comes from three
frozen MTP heads that saturate at four correct proposals.

Adopted at `adaptK=1.5` (`NO_ADAPTK=1` restores fixed width). Caveat recorded: the threshold is
fitted on 18 verifies of one prompt. `scripts/prompt_suite.json` exists and this should be
re-fitted across it before the number is trusted beyond this benchmark.

### Baseline correction

The control point in this run measures **16.52-16.86 tok/s**, not the 15.3 carried since Finding 47.
The difference is the generation length: `NGEN0=60` gives 18 verifies and mean 3.39 tokens/verify,
where the 24-token runs gave 9 verifies and 3.00. **The short run was dominated by the early,
less-repetitive part of the sequence and understated steady-state speculation by ~10%.** All
speculative numbers before this one are low by roughly that much; base AR is unaffected.

---

## Finding 50 — cycle 1 measured nothing: the executor session cannot execute

**HALT. No number in this entry, because no number could be produced.** The autonomous executor
runs under `--permission-mode acceptEdits` (`scripts/flywheel.sh:131`). That mode auto-approves
*file edits*. It does not auto-approve **Bash**. Read-only inspection (`ls`, `grep`, `sed`, `git
status`, `git log`) runs sandboxed and is allowed; everything that writes or executes is denied:

| attempted | result |
|---|---|
| `g++ -O2 -std=c++17 -I include tools/encode_prompt.cpp -o build/encode_prompt` | denied |
| `g++ --version` | denied |
| `./build/inspect_weights` | denied |
| `git add -A` | denied |

So this cycle could not build, could not run a unit gate, could not run `scripts/run_model.sh`, and
**could not commit**. The artifacts below are written to the working tree and are UNCOMMITTED; a
human must commit them, or re-run the loop with Bash permission and let cycle 2 do it.

The design consequence is worth stating plainly, because it makes the loop's central promise void:
**an executor that can edit files but not run them can still write findings — which is the single
most dangerous failure mode this project has** (Finding 33: a config written up that had never been
executed). The only safe behaviour available is to produce no numbers and halt, which is what this
entry does. The fix is a permission decision, not a code change: give the executor an allowlist for
`g++`/`nvcc`/`./build/*`/`scripts/*.sh`/`git`, or launch it with permissions skipped.

### What could still be settled without executing: two queue dispositions, from code evidence

Neither of these is a measurement, and neither is written as one. Both are *existence* questions,
which source inspection can answer with the same authority as a run.

**A6 — CUTLASS `reg_reconfig.h` missing `1100` clause — RETIRED, expected end-to-end value zero.**
The lever is real upstream (`setmaxnreg` compiles out on Thor → register spills → 1.74x on FMHA,
issue #3056 / PR #3308). It cannot pay here because **no CUTLASS kernel is ever launched by this
engine.** Evidence:

- `grep -rln cutlass --include=*.cu --include=*.h --include=*.cuh .` (excluding `ref/`) matches
  exactly two files: `include/cutlass_moe.h` and `kernels/cutlass_moe.cu`.
- The only callers of `cutlass_nvfp4_gemm` are inside `kernels/cutlass_moe.cu` itself (lines 136,
  177 — its own self-tests). No engine translation unit includes `cutlass_moe.h`.
- `scripts/build.sh` links `build/cutlass_moe.o` and its comment says "for the TC verify path", but
  the verify path calls `block_verify_step` / `cblock_verify_step`, not the CUTLASS wrapper. The
  object is linked and dead.
- The only `setmaxnreg` in the tree is `tools/cap_probe.cu:104`, a capability probe.

`reg_reconfig.h` is a header consumed only when CUTLASS templates are compiled, and only
`cutlass_moe.cu` compiles them. Patching it changes one object file that never runs. **Retired for
the decode path; it becomes live again only if the engine ever routes a GEMM through CUTLASS.**

**A1 — intra-expert activation sparsity — REMOVED from the Phase-A queue, re-scoped to Phase C and
to the user.** Three separate reasons, none of which needs a run:

1. **It is not lossless.** Skipping neurons changes the logits. The queue entry's "runtime-only, no
   artifact change" is true about the *weights* and false about the *outputs*. Every other lever
   adopted this cycle-set was defended by exact-output equality (Finding 49 verified byte-identical
   sequences). There is no accuracy harness in this repo to replace that argument with, and no
   published sparsity-vs-accuracy curve for this checkpoint. That makes it a quality-cost decision,
   which invariant 10 assigns to the user, not to the executor.
2. **The MXFP4 layout bounds the realizable saving to ~1/3 of expert bytes, and our own notes
   already said so.** `research/MOE_DECODE.md:98` lists it under "levers that are DEAD for us:
   MXFP4's 32-element blocks make sub-block skipping impossible without breaking the scale layout."
   Concretely: neuron *j* is a contiguous **row** of w1/w3 (skippable) but a **column** of w2, i.e.
   an index inside w2's 32-element K-blocks (not skippable without unaligned scales). And the gate
   that decides which neurons are inactive is w1's own output, so w1 must be read in full.
   Only **w3** is skippable — one of the three expert matrices. `research/BYTE_REDUCTION.md`'s
   +11–17% is priced against all 4,531 MB of expert bytes and does not carry that derate.
3. **The ranking model says byte reduction is the wrong axis for this phase.** §2 rule 1: byte
   reduction only pays on a bandwidth-bound kernel. Finding 47 measured the MoE phase
   latency-bound — 25% memory throughput, 84% of stall cycles on `long_scoreboard`. A1 is a
   pure byte-reduction lever aimed at the phase we have already measured as not byte-limited. This
   is the exact shape of Finding 32 (BF16 compressor: ranked #1 at +5%, measured −3%).

### The artifact this cycle did produce: `tools/encode_prompt.cpp` (UNBUILT, UNGATED)

R1 — re-fit the adaptive-verify threshold across `scripts/prompt_suite.json` — is the queue entry
Finding 49 itself asked for, and it is blocked on something mundane: `build/decode` takes token ids
on argv, `prompt_suite.json` holds chat text for a *different* project (`gemma4-cuda-server`), and
**inventing token ids is precisely the class of mistake the invariants forbid.** So the ids have to
come from the checkpoint's own tokenizer.

`tools/encode_prompt.cpp` is 45 lines that load `<ckpt>/tokenizer.json` via the in-tree
`include/tokenizer.h` and print an argv-ready id list. It **gates itself first**: it encodes
`The capital of France is` and refuses to print anything unless that returns
`671,6102,294,8760,344`, the canonical prompt recorded in `scripts/run_model.sh`. That gate matters
more than usual here, because `include/tokenizer.h` is a gemma-4 BPE tokenizer carried over from
this codebase's ancestor and its hard-coded special ids (`bos_id=2`, `turn_start=105`) do **not**
obviously match a DeepSeek checkpoint whose canonical prompt starts with id 0.

**It has never been compiled and never been run. Its gate has never fired in either direction.**
Cycle 2's first action should be to build it and run the gate — and if the gate FAILS, that is a
result, not a setback: it would mean the in-tree tokenizer does not match this checkpoint, and R1
needs a different id source before it can be attempted at all.

R1 also needs a second, larger piece that this cycle did **not** write, deliberately: `build/decode`
prefills exactly one prompt (`argv[2]`), so a multi-prompt within-run A/B needs the `DSV4_BLKSWEEP`
entry syntax extended with a prompt index and the `bsi` loop re-prefilling per prompt (`PS`, `s`,
`ids` become per-point; `mh_pre` must be sized at the longest prompt, it is currently `PS*3*d*4`).
That is a ~20-line change to `src/decode.cu`, and **committing an uncompilable edit to the
measurement harness would have been worse than committing nothing.** It is scoped, not done.

---

## Finding 51 — R1 is BLOCKED, and the two defects that block it are the cycle's result

**Cycle 2. One full-model run (`~/cycle2.log`, 17 within-run points, base-AR gate PASS: first decoded
token argmax 11111). No adaptK re-fit number is quoted from it, because the instrument that produced
it is unsound in two independent ways, both demonstrated below.** End-to-end movement on the shipped
configuration: **0%** — nothing in the engine changed; the canonical prompt reproduces
**16.77–16.95 tok/s** against the 16.86 baseline, inside the ±1% band.

### First: the id source. The in-tree tokenizer FAILS this checkpoint; the reference one passes.

Cycle 1 left `tools/encode_prompt.cpp` unbuilt with its gate never fired. Fired this cycle:

```
[encode] GATE canonical: got 671 464 388 367 79666 464 388 367 2154 464 388 367 51725 464 388 367 278
                        want 671 6102 294 8760 344   -> GATE FAIL
```

`include/tokenizer.h` is a gemma-4 BPE carried over from this codebase's ancestor. It reads
DeepSeek's `tokenizer.json` (vocab 128000 by its count, 129280 by the reference) and gets the first
token right, then byte-falls-back on every word boundary — it does not implement the GPT-2
byte-level space mapping the merges are written in (`Ġcapital`, not `▁capital`). **The gate did its
job: the tool refused to print.** The same gate against `tokenizers` 0.23.1 reading the checkpoint's
own `tokenizer.json`:

```
[encode] GATE canonical: got [671, 6102, 294, 8760, 344]  want [671, 6102, 294, 8760, 344] -> GATE PASS
```

`tools/encode_prompt.cpp` is therefore **deleted**. `tools/encode_prompt.py` (already at HEAD from a
quarantined cycle, kept verbatim) supersedes it: same canonical self-gate, plus a round-trip gate
over every prompt it emits, and it prints a ready `DSV4_PROMPTS=` line. Both its gates pass —
canonical exact, **6/6 round-trip**.

The four prompts this run actually used, passed on argv rather than taken from the tool's built-in
suite (record them here; the run is not reproducible from the tool alone):

| p | text | ids |
|---|---|---|
| 0 | `The capital of France is` | `0,671,6102,294,8760,344` |
| 1 | `def fibonacci(n):\n    if n <= 1:\n        return n\n    return` | `0,3465,55155,3913,3395,361,855,313,8593,223,19,1137,528,1354,313,201,361,1354` |
| 2 | `Q: What is the largest planet in the solar system?\nA:` | `0,51,28,1999,344,270,9152,13540,295,270,11250,1487,2755,35,28` |
| 3 | `In 1969, humans first walked on the surface of the` | `0,1124,223,2722,27,14,11212,1257,13577,377,270,4433,294,270` |

### Defect D1 — `adaptK=0` in a sweep entry does not mean fixed width

`src/decode.cu`: `adaptK = adaptSweep[bsi] > 0.f ? adaptSweep[bsi] : (getenv("NO_ADAPTK") ? 0.f : 1.5f)`.
A sweep entry of `0` is indistinguishable from *unset*, so it falls through to the **1.5 default**
unless `NO_ADAPTK=1` is also in the environment. The `[blksweep]` table printed the *requested*
value. So the four fixed-width control points in this run — one per prompt — silently ran at 1.5,
and were printed as `0.00`. **This run therefore contains no fixed-width control at all.**

`~/adaptk3.log` (Finding 49) escaped this only because it was launched with `NO_ADAPTK=1` in the
environment, which the per-point `[spec] decoding ... adaptK=0.00` line confirms. Finding 49's
numbers stand. But the field's meaning depends on an env var, which is the exact shape of a
silently-mislabelled table. Mitigated this cycle by printing both requested and effective adaptK
(display only, compile- and parse-gated, **not** re-measured); the semantics still need fixing.

### Defect D2 — the multi-prompt harness leaks state, so non-first points are not trustworthy

The run was designed with the canonical prompt as both the **first** and the **last** point at the
same setting, so that cross-prompt leakage would show up as a divergence. On the canonical prompt it
**passed**: points 0 and 16 are byte-identical — same 61 tokens over 21 verifies, mean 2.90, and the
same printed token sequence — with twelve points on three other prompts of different lengths in
between. That gate would have been the whole basis for trusting the instrument. It is not enough:

- **Prompt 2, two points at identical effective settings, emitted DIFFERENT token sequences.**
  Point 8 (first p2 point): `539 51 28 1999 344 270 6102 294 8760 2755 35 28 11111 ...` — a plausible
  continuation of its own prompt. Points 9, 10, 11: `455 6102 294 29585 344 76405 16 455 6102 294 ...`
  — the *canonical* prompt's "capital of X is Y" pattern. Points 9/10/11 agree exactly with each
  other and disagree with 8, so this is deterministic-but-contaminated, not noise.
- **Prompt 3, two points at identical effective settings (12 and 14), agree on every emitted token
  but disagree on the draft from verify 1 onward:** margins `1.22 0.20 5.42 0.03 1.79 K=2` vs
  `0.44 0.79 0.48 2.06 0.44 K=2`, and 22 vs 20 verifies for the same 64 tokens.

Replicate spread in tok/s at one identical setting, same prompt, same process:

| prompt | replicates at effective adaptK=1.5 | spread |
|---|---|---|
| 0 (canonical) | 16.77, 16.94, 16.95 | **1.1%** |
| 1 (code) | 13.05, 13.85 | 6.1% |
| 2 (QA) | 19.96, 18.39 | 8.5% |
| 3 (prose) | 16.53, 19.11 | **15.6%** |

**Every adaptK difference this run could have reported is smaller than the replicate spread of the
setting itself.** The one prompt whose replicates are stable is the one every number in this project
was measured on, so no prior finding is impugned — but R1's premise, that a multi-prompt within-run
A/B is worth more than separate runs, is **false as implemented**. `KV[L].T=0` plus a re-prefill is
not a full reset when the prompt *length* changes; something in the draft/main_x/window path
survives it. Not root-caused this cycle, deliberately: that is the next lever, not this one.

### Disposition

**R1 stays at the head of the queue, blocked, behind a new lever I1: make one process able to run
the same point twice and get the same answer.** Its gate is already written and already failed —
the same prompt at the same setting twice in a row, on a *non-canonical* prompt, must emit an
identical sequence and an identical verify split. Until that passes, no multi-prompt number is
admissible, and the adaptive-verify threshold shipped at 1.5 remains fitted on 18 verifies of one
prompt exactly as Finding 49 warned.

---

## Finding 52 — D2 root-caused and fixed: `run_layer` prefilled every sweep point at the ARGV prompt's length

**Cycle 3. One full-model run (`~/cycle3.log`, base-AR gate PASS: first decoded token argmax 11111).
End-to-end movement on the shipped configuration: 0.0%** — canonical prompt 16.83 / 16.90 tok/s
against the 16.86 baseline, base AR 10.33 tok/s against 10.31 (`~/adaptk3.log`) and 10.27
(`~/cycle2.log`). Nothing on the shipped path changed and the numbers confirm it.

### The bug, found by inspection before any run

`src/decode.cu:265`, `run_layer`, is a `[&]` lambda whose prefill branch passed `PS` to
`block_prefill_cache` / `cblock_prefill_cache`. Name lookup inside a lambda body is **lexical**, and
resolved at the lambda's definition point — where the only `PS` in scope is line 96's,
`argv[2].size()-1`. The multi-prompt sweep loop then declared its own

```
const std::vector<int>& ids = prompts[promptSweep[bsi]];
const int s = (int)ids.size(), PS = s-1;          // shadows line 96
```

and called `run_layer(Lyr,true,0,h,h2,d_ids)`. The shadow is invisible to the callee. So on every
sweep point:

| what | length used |
|---|---|
| `k_embed` / `k_hc_expand` | the point's prompt (correct) |
| **`block_prefill_cache` / `cblock_prefill_cache` — the KV caches** | **the argv prompt (WRONG)** |
| `dspark_tap_pool` / `dspark_main_x` / `cpos` / the whole spec loop | the point's prompt (correct) |

With the canonical prompt on argv, every point prefilled **5** positions. A sweep point on an
18-token prompt therefore decoded from `cpos=17` over KV caches holding 5 rows; positions 5..16 were
whatever the *previous* points had left in `win_kv`/`xin`/`h`/`h2`. That is D2, exactly, and it
predicts cycle 2's signature with no free parameters: prompt 0 (the argv prompt, `PS`=5=`PSp`) had
**all five** of its points byte-identical, while prompts 1/2/3 each had their FIRST point disagree
with the rest. Cycle 2's `[spec] decoding ... s=%d` header printed the *shadowed* `s`, so the run
looked correctly labelled while the caches were not.

### The fix, and the gate that would have caught it

1. The prefill length is now an **explicit parameter** `npre` on `run_layer` (8 call sites). A
   parameter cannot be shadowed out from under the callee; a capture can.
2. The sweep loop no longer shadows at all: `pids` / `ps` / `PSp`.
3. **New in-run gate, every point, before any measurement:** a compressed layer emits exactly
   `floor(PSp/ratio)` rows during prefill, so `KV[L].T` is a direct readout of the length the
   prefill *actually ran at*. `KV[L].T != PSp/ratio` for any of the 21 compressed layers prints
   `[spec] GATE FAIL` and exits 3. Under the old code, `PSp=17` against an argv `PS` of 5 gives
   `T=1` where 4 is required. **This gate PASSED at all 11 points that ran.**

### The measurement: the I1 replicate gate now passes on 3 of 4 prompts, and fails on the 4th

Nine gate points, prompts interleaved so a replicate pair is separated by a different-length prompt.
Hashes are over the emitted token list and over the full per-verify decision trace
(margins, K, accepted/VB, cpos) with timings stripped:

| pair | prompt | s | result |
|---|---|---|---|
| PT0 / PT8 | 0 canonical | 6 | **identical** — 61 tokens, 21 verifies, 2.90/verify, 16.83 / 16.90 tok/s |
| PT3 / PT5 | 1 colors | 11 | **identical** — 61 tokens, 23 verifies, 2.65/verify, 15.45 / 15.45 tok/s |
| PT6 / PT7 | 3 moon | 15 | **identical** — 60 tokens, 34 verifies, 1.76/verify, 11.10 / 11.11 tok/s |
| PT1 / PT2 / PT4 | 2 fibonacci | **18** | **THREE different sequences**: 11.05 / 14.62 / 16.36 tok/s, 32 / 23 / 21 verifies, first-verify margins `7.01 5.85 0.70 3.56 1.80` vs `8.20 6.39 1.80 4.09 3.55` vs `7.74 6.87 0.11 2.53 2.30` |

Cycle 2 had **0 of 3** non-canonical prompts reproducing. Cycle 3 has **2 of 3** (plus canonical).
PT3/PT5 and PT6/PT7 are byte-identical in both the sequence and the decision trace — not merely
close in tok/s. **I1 is therefore partially, not fully, discharged.**

### The run also died, on the same prompt

```
cuda kernels/indexer.cu:91 an illegal memory access was encountered
```

after PT10 and before PT11 printed its header — i.e. inside **PT11's prefill**, which is prompt 2
again (s=18). PT1/PT2/PT4 had already prefilled that same prompt successfully, so the fault is
state- or history-dependent, not a fixed out-of-bounds. `indexer.cu:91` is the
`cudaStreamSynchronize` at the end of `indexer_forward`, which only *reports* an earlier async
fault; it does not locate it.

**Both surviving symptoms are on prompt 2 and only prompt 2, which is the only prompt whose length
equals `SMAX`.** The one buffer in the spec path sized with zero slack is
`mh_pre = (SMAX-1)*3*d*4` (`src/decode.cu:615`), exactly `PSp_max` rows — every shorter prompt
overruns *into* the allocation, the SMAX prompt overruns *out of* it. That is a lead, **not a
demonstrated cause**, and it is not the only candidate (`indexer_forward` also does raw
`cudaMalloc`/`cudaFree` per layer per call with 11 GiB of headroom in a 122.8 GiB managed pool).

**The next iteration's falsification test costs no extra run and is decisive:** append a prompt
LONGER than 18 to `DSV4_PROMPTS` so `SMAX > 18`, and re-run the same 9 gate points. If prompt 2 then
reproduces, the defect is "the prompt of length SMAX", i.e. a zero-slack buffer. If it still fails,
the defect is specific to that prompt's length/content and `SMAX` is a coincidence.

### Bonus, and it is not a small one: adaptive verify width is LOSSLESS, demonstrated

PT3 / PT9 / PT10 are prompt 1 at adaptK **1.50 / 1.00 / 2.50**. All three emit the **identical
61-token sequence** (same hash) with different verify counts — 23 / 23 / 26 — and different K
choices per verify. Finding 49 argued losslessness from the structure of the accept rule
("verifying fewer proposals cannot change what the target emits"). This is the first direct
measurement of it: the threshold moves the *cost* of a verify and nothing else.

Within-run, one replicate each (PT3 and PT5 are exact replicates of the 1.50 point, both 15.45):

| adaptK on prompt 1 | verifies | tok/s |
|---|---|---|
| 1.00 | 23 | 14.76 |
| **1.50 (shipped)** | 23 | **15.45** |
| 2.50 | 26 | 14.51 |

**This is not adopted and R1 stays blocked.** It is one prompt, and defect D1 (a sweep entry of `0`
falls through to the 1.5 default) means the run still contains no fixed-width control. It does say
the shipped 1.5 is not obviously mis-set.

### Disposition — HALT

Invariant 7: a gate failed and a run faulted, so this stops here rather than being built on.
`halt=true`. The queue head is now **I2 — the SMAX-length prompt does not reproduce and its 12th
prefill faults** — with the single-run falsification test above as its first action.

---

## Finding 53 — the fault on `indexer.cu:91` is a real, reproducible shared-memory over-read in the top-k scan, found without a model run

**Cycle 4. NO full-model run — lever I2 was at stage `candidate`, so this cycle builds and gates it and
the next one measures it. Everything below was measured on unit gates and `compute-sanitizer`.**

Cycle 3 halted on `cuda kernels/indexer.cu:91 an illegal memory access` during the prefill of the one
prompt whose length equals `SMAX`, and named a lead: `mh_pre = (SMAX-1)*3*d*4`, the zero-slack buffer.
That lead is **wrong, and dischargeable by inspection**: `k_tap_pool` (`kernels/dspark_real.cu:44`)
writes `mh[t*(n_taps*d) + slot*d + j]` for `t < PSp`, i.e. exactly `PSp*3*d` floats, and the maximum
`PSp` is `SMAX-1` because `PSp = len(prompt)-1`. Zero slack, but an exact fit. Not the cause.

### The gate that found the real one

`tests/gate_prefill_len.cu`, new. Nothing in this suite had ever varied the prompt LENGTH — every
gate ran at one fixed `s` — which is precisely how a length-dependent defect survives.

The invariant it asserts: every prefill path here is causal and every prompt is shorter than the
sliding `WINDOW`, so output row `i` is a function of `x[0..i]` alone. There is no cross-row reduction
anywhere in the chain (every GEMM is per-output-row, `rmsnorm`/`act_quant`/`rope` are per row, the
compressor pools per group, `sparse_attn` is per query), so for two lengths `s < S` over the same `x`
the first `s` rows must agree **bit-exactly**. It checks the KV caches too — `win_kv`, `comp_kv`,
`idx_ckv` — not just the block output, because those are what a wrong-length prefill leaves behind.
Coverage: the sliding layer (`hc_pre` → `rmsnorm` → `mla_cache_kv` → `mla_forward` → `hc_post`), the
ratio-4 indexer layer and the ratio-128 strided layer, at `s = 1..20`, at weight alignment `+0` **and**
`+4` (standing caveat 2). It also drains and names `cudaGetLastError` after every stage.

**Result 1, a negative one and the more useful half: prefix-invariance HOLDS, bit-exactly, at all 20
lengths in both alignments, on all 8 tracked outputs.** The prefill attention chain is not
length-dependent. `PSp=17` is not special. That kills the whole "the prefill runs differently at
SMAX" hypothesis, which is where cycle 3 pointed.

**Result 2, the defect.** Under `compute-sanitizer --tool memcheck`:

```
Invalid __shared__ read of size 12 bytes
    at k_topk_offset(...)+0x1280 in indexer.cu:60
    Access to 0x400 is out of bounds
...
cuda kernels/indexer.cu:91 unspecified launch failure
```

`k_topk_offset` declares `extern __shared__ float sh[]`, fills `sh[0..T-1]` and scans it with
`for(t=0;t<T;++t)`, and was launched with exactly `T*sizeof(float)`. **nvcc widens that scan into a
12-byte vectorised shared load, so the request is too small whenever `T < 3`.** Measured directly, by
sweeping `s` through `tests/gate_indexer_decode` under memcheck:

| s | T = s/4 | smem requested | memcheck |
|---|---|---|---|
| 4, 5 | 1 | 4 B | **5 / 6 errors, launch killed, reported at `indexer.cu:91`** |
| 8 | 2 | 8 B | **9 errors, launch killed, reported at `indexer.cu:91`** |
| 12 | 3 | 12 B | 0 errors |
| 16, 17, 18 | 4 | 16 B | 0 errors |

The same undersizing is in three sibling kernels — `k_topk_decode`, `k_topk_verify` and
`k_topk_masked` in `compressed_decode.cu`, all `extern __shared__ float sh[]` launched at `n*4`. They
do not trip at today's context lengths, but they are the same defect and are fixed together;
`include/indexer.h` now carries one `topk_scan_smem(n)` helper that rounds `n+3` up to 4 floats.

**Honest scope.** The prompt that crashed cycle 3 had `PSp=17` → `T=4`, which does **not** over-read.
Two of that run's four prompts did (`PSp=5` → `T=1`, `PSp=10` → `T=2`), on all 21 ratio-4 layers of
every prefill, and the canonical prompt is one of them — this has been happening on every run this
project has ever made. So: a real memory-safety defect, reproducible, on exactly the line cycle 3's
fault named, now fixed and gated. **Whether it is sufficient to explain the cycle-3 fault is NOT
established, and the next cycle's run is the test.**

### Result 3 — the engine cannot see a kernel that failed to launch

Chasing the attribution turned up a second defect. Measured on this box (`/tmp/zg.cu`, 6 lines):

```
gridDim0 -> 1 invalid argument       # the launch fails
after sync -> 0 no error             # cudaDeviceSynchronize returns SUCCESS
```

A launch failure is reported **only** through the thread's last-error slot, and this engine never
called `cudaGetLastError` anywhere. So a kernel that never ran was indistinguishable from one that
did, and the stale code sat in the slot until some unrelated `CU()` happened to pick it up.

And it was firing constantly: `compressor_forward` computes `groups = s/ratio`, which is **0 for every
ratio-128 layer at every prompt this project has run** (longest 18 tokens, ratio 128), and fell
through to five launches with `gridDim = (0*d+255)/256 = 0`. Twenty layers × every prefill. The same
hole existed one level up in `indexer_forward` for `T = s/ratio == 0` (prompt length ≤ ratio). Both
now return early — which is what the code always meant, since there is no complete group to emit.

`dsync()` — the one call every sub-function already ends with — now drains the slot and names the
file:line (`dsync_at`, `include/dscratch.h`). The **sync** stays a no-op under the decode arena; only
a TLS read is added. It reports by default and aborts only under `DSV4_STRICT_LAUNCH=1`, because a
stale code must never be able to kill a 15-minute model run on its own, and must never be silent.

### Verification

| check | result |
|---|---|
| `gate_prefill_len`, 20 lengths × 2 alignments, plain | **GATE PASS** — 0 prefix mismatches, 0 stages leaving a CUDA error |
| `gate_prefill_len`, same, under `memcheck` (2m45s) | **GATE PASS, ERROR SUMMARY: 0 errors** |
| `gate_indexer_decode` at s = 4, 5, 8, 12, 16, 17, 18 under `memcheck` | all **PASS**, 0 errors (was 5/6/9 errors at s = 4/5/8) |
| `gate_units` (goldens), `gate_encoding`, `gate_bf16w`, `gate_api`, `gate_ogroup_gemv`, `gate_tc_fp8_smem` | all exit 0 |
| `gate_compressed_decode`, `gate_compressor_emit`, `gate_indexer_graph`, `gate_compressed_graph` | all **PASS**, cosine 1.00000000, rms 0.00e+00 |
| `scripts/build_decode.sh` | builds clean |

Note the equivalence gates are bit-exact (`rms=0.00e+00`) after the change, which is the point: a
larger shared-memory request cannot alter a result, so this is a memory-safety fix with provably zero
numerical consequence.

### Disposition

Halt stays cleared. I2 advances `candidate` → `implemented`. Three changes land on the shipped path
this cycle and the next measurement carries all three: the `topk_scan_smem` rounding (four launches),
the two `groups==0`/`T==0` early returns, and the `dsync` last-error drain. None is a performance
lever and none can change a number — but per invariant 1 they are named here so the next cycle's
result is not attributed to them.

**Next cycle (I2 at `implemented`) is one full-model run** with `NGEN0=60` and a prompt appended that
is LONGER than 18 ids, so `SMAX > 18`. That run answers three things at once: does the s=18 prompt now
reproduce byte-identically, does the fault recur, and — from the `SMAX > 18` change — was "the prompt
of length SMAX" ever the right frame. Watch stderr for `[launch] file:line pending CUDA error`, which
is now the engine's own voice for anything that fails to launch.

---

## Finding 54 — the top-k fix bought reproducibility and did not buy the crash: the fault follows SMAX but is not caused by length

**Cycle 5. ONE full-model run, `evidence/cycle5.log`, 16 sweep points planned, 12 completed, NGEN0=60.**
Lever I2 was at stage `implemented`, so this cycle is its single measurement. The run answers all three
questions cycle 4 posed, and two of the three answers are not the expected ones.

The launch, recorded in full because the run is not reproducible without it:

```
DSV4_PROMPTS="<p1 colors, 11 ids>;<p2 fibonacci, 18>;<p3 moon, 15>;<p4 apollo, 28>"
DSV4_BLKSWEEP="5:1:1.5:0,5:1:1.5:2,5:1:1.5:2,5:1:1.5:1,5:1:1.5:2,5:1:1.5:1,5:1:1.5:3,5:1:1.5:3,
               5:1:1.5:0,5:1:1.0:1,5:1:2.5:1,5:1:1.5:2,5:1:1.5:4,5:1:1.5:4,5:1:1.5:4,5:1:1.5:0"
scripts/run_model.sh ~/cycle5.log ./build/decode <ckpt> "0,671,6102,294,8760,344" 8 "" 60
```

Points 0–10 are cycle 3's eleven points in cycle 3's order, unchanged, so the comparison is controlled.
Prompt 4 is new and is 28 ids, so `SMAX` moves 18 → 28 and prompt 2 stops being the longest prompt.
Its ids come from `tools/encode_prompt.py` (canonical gate PASS, round-trip PASS); the sweep table was
gated with `DSV4_PARSE_ONLY=1` before the load (`SMAX=28 BLKMAX=5 seqmax=101`).

### (a) Reproducibility: fixed at the token level, not at the decision level

| prompt | points | tokens generated | verifies | tok/s | distinct token sequences |
|---|---|---|---|---|---|
| 0 canonical, s=6 | PT0, PT8 | 61, 61 | 21, 21 | **16.96, 17.04** | 1 |
| 1 colors, s=11 | PT3, PT5 (aK 1.5) | 61, 61 | 23, 23 | 15.55, 15.55 | 1 |
| 2 fibonacci, s=18 | PT1, PT2, PT4, PT11 | 60 ×4 | **18, 19, 18, 18** | **18.61, 17.81, 18.50, 18.55** | **1** |
| 3 moon, s=15 | PT6, PT7 | 60, 60 | 34, 34 | 11.22, 11.26 | 1 |

**Twelve points, four prompts, exactly four distinct token sequences** (`[spec] tokens:` lines, first 40
tokens — that is all the engine prints). Cycle 3 had **three different sequences on prompt 2 alone**,
at 11.05 / 14.62 / 16.36 tok/s. The spread on prompt 2 collapses from **±20 % to ±2.2 %**.

The residual is one level down. Hashing each point's full `[margins]` trace:

| prompt-2 point | margin-trace hash | verifies |
|---|---|---|
| PT1 | `db8a0284…` | 18 |
| PT2 | `a40691b4…` | 19 |
| PT4 | `dce5b92f…` | 18 |
| PT11 | `dce5b92f…` | 18 |

Three distinct decision traces for one identical output. First verify, same context, same weights:
PT1 `9.05 6.45 1.12 1.42 2.11`, PT4/PT11 `8.64 5.37 0.06 2.81 1.88`, PT2 `7.72 7.09 2.19 3.31 1.69`.
Every other prompt's replicate pair is trace-identical (`PT0≡PT8`, `PT3≡PT5`, `PT6≡PT7`). So the draft
head's logit gaps are still not bit-stable **on prompt 2 only**, and on PT2 one gap crossed the 1.5
threshold and bought a 19th verify. Per Finding 49 that is a *cost* difference and not a correctness
one — and this run is the third independent confirmation of that, because PT3 (aK 1.50), PT9 (1.00) and
PT10 (2.50) emit the **same 61 tokens** over 23 / 23 / 26 verifies at 15.55 / 14.85 / 14.63 tok/s.

**I1 is therefore much closer to discharged than it was, and is not discharged.** 4/4 prompts reproduce
their output; 3/4 reproduce their decisions.

### (b) The fault recurs

```
[spec] SPEC-DECODE: 53.9 ms/tok = 18.55 tok/s        <- point 11, prompt 2, completes normally
cuda kernels/indexer.cu:96 an illegal memory access was encountered
```

Line 96 is the same `cudaStreamSynchronize` cycle 3 hit at line 91 (the file grew five lines). It killed
the run entering **point 12**, i.e. inside the prefill of prompt 4 — the run never printed point 12's
header. Points 12–15 did not run. **No `[launch] … pending CUDA error` line appears anywhere in the 644
lines**, so cycle 4's last-error drain was live and found nothing: this is a genuine asynchronous illegal
access, not a launch that failed to start.

### (c) `SMAX > 18` moved the fault — and that still does not make it a length defect

Cycle 3 crashed in the prefill of prompt 2 when prompt 2 was the `SMAX` prompt. Cycle 5 crashed in the
prefill of prompt 4 when prompt 4 became the `SMAX` prompt, and prompt 2 — same ids, same position in
the sweep — prefilled four times without incident. The fault follows `SMAX`. That is the answer cycle 4
asked for, and taken alone it says "zero-slack buffer sized on the longest prompt".

Taken with the rest of the evidence it cannot be the whole story:

1. `build/gate_prefill_len "5,10,14,17,21,24,27"` — the exact prefill lengths this run used, `PSp=27`
   being prompt 4's — **GATE PASS, 0 prefix mismatches, 0 stages leaving a CUDA error, both weight
   alignments** (`+0` and `+4`). Run this cycle. The prefill attention chain at the crashing length is
   clean in isolation.
2. In cycle 3 the `SMAX` prompt prefilled **successfully three times** and faulted on the fourth. Being
   the longest prompt is not sufficient.
3. Here the `SMAX` prompt faulted on its **first** prefill — but at sweep point 12, after 21 ratio-4
   layers × 12 prefills of `cudaMalloc`/`cudaFree` churn.

The frame that fits all three is **accumulation across sweep points, tripped at the largest working
set**: something grows or fragments over points, and the longest prompt is simply the first allocation
big enough to fall off the end of it. `indexer_forward` (`kernels/indexer.cu:78-83`) and its caller
(`kernels/compressed_attn.cu:62`) do eight raw `cudaMalloc`/`cudaFree` per indexer layer per call inside
a 122.8 GiB managed pool holding 111.5 GiB of resident weights — ~11 GiB of headroom, churned 21×
per prefill. Sizes are correct by inspection (`s*T*4`, `s*itopk*4`, all `s`- and `T`-derived), so this is
a lead about *allocator behaviour*, not about a wrong size, and it is a lead and not a cause.

Note also that cycle 5 got **one point further than cycle 3** (12 completed vs 11). Whether cycle 4's
removal of ~20 failing gridDim-0 launches per prefill moved that boundary is not established and must
not be written as if it were.

### The numbers, for the record

| | cycle 3 (`evidence/cycle3.log`) | cycle 5 (`evidence/cycle5.log`) |
|---|---|---|
| base AR, M=1 warm | 96.8 ms/tok = 10.33 tok/s | 95.6 ms/tok = **10.46** tok/s |
| canonical spec, both bookends | 16.83 / 16.90 | **16.96 / 17.04** |
| M=K verify equivalence gate | PASS | **MATCH 5/5 → PASS** |
| first-token gate | PASS (11111) | **PASS (11111)** |

**The baseline does not move.** +0.8 % on the canonical prompt is inside the ±1 % cross-run band
(invariant 6), and the three changes this run carried — a *larger* shared-memory request, two skipped
no-op launches, one TLS read per `dsync` — cannot produce a speedup by construction. Recording it as a
gain would be the Finding 33 failure mode. `spec_tok_s` stays at 16.86.

### Disposition — HALT

Invariant 7: the run faulted, so this stops here. `halt=true`.

I2 advances `implemented` → **`measured` and is adopted on its own criterion**: it converted three
divergent token sequences into one, it is bit-exact by construction, and it fixed a real out-of-bounds
that had been firing on every prefill this project ever ran. It did **not** discharge the crash, and the
crash was never claimed as its target — cycle 4's write-up said so explicitly.

The crash becomes a new queue head, **I3**, with a frame the loop has not tried before: not "which
length", but "what accumulates". Its first action needs **no model run**, which matters because the loop
should not spend a 15-minute checkpoint load on a hypothesis a unit gate can kill: instrument
`cudaMemGetInfo` at every sweep-point boundary and around `indexer_forward`, and write a standalone
gate that replays N cycles of the indexer's malloc/free pattern at ascending `s` in a pool sized to
leave ~11 GiB free. If free memory is flat across points, the accumulation frame dies and the next
suspect is `arena_reset` / the per-point head rebuild.

---

## Finding 55 — a decode-sized kernel cannot saturate this memory system, and the engine only ever runs one

This closes the "uniform ~1.9x" that four cycles wrote down as unexplained.

### The gap, re-measured on the current baseline

Every per-op number in this log came from `evidence/dprof5.log`, which was taken at **14.36 tok/s** —
before managed weights, the graph re-gate and adaptive verify width. So the whole priority model was
running on numbers ~17 % stale. Re-measured today (`~/dprof6.log`, 17.05 tok/s spec / 10.30 base, first
token 11111 → GATE PASS), the K=5 verify is **163.95 ms** and splits into two populations that behave
completely differently:

| group | ms | bytes | GB/s | % of 233 achievable |
|---|---|---|---|---|
| routed MoE (`w1w3` 49.03 + `w2` 22.35) | **71.4** | 17.18 GB | **241** | **~100 % — at the roofline** |
| everything else | **92.6** | 8.81 GB | **95** | 41 % |

The routed MoE is *done*. It is one launch with ~60k warps of work and it saturates DRAM. Nothing in
the second group does: `cattn:ogroup` 21.78, `cattn:q_proj` 17.33, `moe:shared` 10.37, `cattn:indexer`
9.36, `cattn:compress` 8.40, `lm_head` 6.47, plus ~14 ms of glue that moves almost no bytes at all.

### Two explanations eliminated first

- **Working-set size** — `tools/footprint_probe.cu` (new). Read a *fixed* 1 GiB out of regions from
  0.5 GiB to 64 GiB, both allocators, engine-shaped strided access: **230–246 GB/s everywhere**.
  Streaming out of a 111 GiB managed pool costs nothing, and the roofline is real and reachable at
  scale. A whole class of TLB/SMMU/page-locality hypotheses is dead.
- **Launch overhead** — already dead: the `VERIFYGRAPH` experiment, 2788 graph nodes, **1.05x**.

And the mechanism this log had written down for the bench-vs-in-situ gap is **wrong**: `gemm_bench`'s
`timeit` launches its reps back-to-back on **stream 0**, which is stream-ordered, so the bench does not
overlap anything either. Finding 47's "the bench relaunches on rotating weights so launches overlap"
should be read as "the bench's weights are partly L2-resident" — its own HOT-vs-COLD row is already
1.73x at M=5.

### The actual cause, measured

`tools/overlap_probe.cu` (new). Same kernel, same shapes, same total bytes, ≥1 GB of rotating cold
weights per lane; the only variable is whether W independent copies share one stream or get one each:

| shape, M=5 | 1 stream | 4 streams | |
|---|---|---|---|
| `wkv  [512,4096]`   | 47.8 GB/s | **150.4** | **3.15x** |
| `wq_a [1024,4096]`  | 86.4 | **194.2** | **2.25x** |
| `wo_b [4096,8192]`  | 144.2 | 196.1 | 1.36x |
| `wq_b [32768,1024]` | 165.0 | 173.5 | 1.04x — already saturated |

**A decode-sized kernel does not have enough concurrent misses in flight to cover DRAM latency.** The
smaller the output, the worse it is, and the effect vanishes exactly where the kernel gets big enough
(`wq_b`, 32768 rows). The engine runs one kernel at a time down a serialised layer chain, so it leaves
1.4–3.2x of the memory system idle for most of every layer. That is the uniform gap, it is not a
property of any kernel, and no amount of per-kernel tuning can reach it.

**The lever this creates: where a layer holds two genuinely independent chains, run them on two
streams.** Note the asymmetry that decides which pairs are worth it — overlap buys nothing against a
kernel that is *already* saturated, and everything against one that is not.

### Two things tried and NOT adopted, both retired with a measurement

- **`__ldcs` on the ogroup weight stream** (the policy the roofline-achieving MXFP4 GEMV uses). ncu
  said the kernel was latency-bound, `long_scoreboard` 7.33/issue, `lts__t_sector_hit_rate` 65.6 %
  where activation-only reuse predicts ~84 % — i.e. the weight stream looked like it was evicting the
  0.7 MB `o`. Marking it evict-first **cut the stall ratio to 3.46 and made the kernel slower**:
  242.8 → 280.4 µs (ncu), 0.1968–0.2041 → 0.2565 ms (`gemm_bench`, M=5/NR=4, band from 3 runs). The hit
  rate *fell* to 61.8, so the weight line was not the evictor. Do not re-queue a cache-hint change here
  without a hit-rate measurement showing L2 is the binding resource.
- **Shared-memory staging of `o` in the ogroup M=K GEMV** (`ogroup_gemv_mk_smem_kernel`, kept behind
  `OG_SMEM=1`). All 8 warps of a block read *identical* activation float4s, so the traffic is 8x
  redundant within the block; staging removes that, and it is **bit-exact at M=2,3,5,8,12,16** (the
  gate checks exact equality, not cosine). It still does not win: NR=2 improves 0.264 → 0.200 but the
  engine runs NR=4, where it is a 40 % *regression* (0.199 → 0.280), and its best point after doubling
  the chunk size to halve the barrier count lands at 0.1996–0.2012 against the incumbent's
  0.1968–0.2041. **A wash.** The `__syncthreads` pair forces the 8 warps into lockstep and destroys
  exactly the skew that was hiding latency — which is the same lesson as above, from the other side:
  traffic was not the binding resource, overlap was.

### Standing corrections this finding forces

1. `evidence/dprof5.log` is **stale** (14.36 tok/s era). Use `~/dprof6.log`. Any ranking derived from
   the old per-op table should be re-derived.
2. The routed MoE is **at the roofline**. It cannot be made faster by kernel work; only by moving
   fewer expert bytes, which is an algorithmic change (verify width), not a CUDA one.
3. `gemm_bench` ranks kernels **against each other on one stream**. It cannot see the concurrency
   effect at all, so it systematically under-values anything that makes a kernel bigger or lets two
   run together. Do not use it to price this class of change.

---

## Finding 56 — the first exploitation of Finding 55: the shared expert was never dependent on the routed ones

**ADOPTED.** `moe:shared` 10.37 → **0.27 ms**; K=5 verify 157.98 → **155.22 ms**; spec **17.05 → 17.32
tok/s**, base AR **10.30 → 10.50**. Token sequence byte-identical, `MATCH 5/5` verify-equivalence gate
PASS, first-token gate PASS (11111).

### What it is

The shared expert reads only `x` and writes only its own buffers. It never depended on the routed
experts — it was merely *issued after them on the same stream*, and so paid the full standalone price
of its bytes: 574 MB in 10.37 ms = **55 GB/s**, next to a routed grouped GEMV moving 17.18 GB at
**241 GB/s**, i.e. at the roofline. Finding 55 says a kernel that slow is latency-starved, not
bandwidth-starved, so its traffic should be nearly free if folded into a stream that is already
saturated. It forks onto `g_side` at the top of `moe_forward` and joins at the combine.

Two preconditions, both of which are why this had not been done before:

- **Its own buffers.** The shared path used to reuse the routed path's `Xeq/Xes/Gb/Ub/Hb/Hqb/Hsb/OEb`.
  Running both on those concurrently is a race that still returns plausible numbers. The new buffers
  are `bs` rows, not `maxm` — ~236 KB at bs=5 against a 512 MB arena.
- **Graph capture.** The base-AR path captures all 43 layers into one graph (worth 1.26x).
  `tests/gate_forkjoin_graph.cu` (new, in the suite and the selftest) reproduces exactly this pattern
  — 43 record/wait pairs on ONE event pair — with trivial kernels, and checks the *result*, not just
  that capture succeeded: 172 nodes, eager == graph == expected. Two seconds instead of a 15-minute
  load to find out. The events must be `cudaEventDisableTiming` and both stream and events are created
  in `arena_init`, which is guaranteed to run before any capture.

### The bug in the first attempt, kept because it is the general trap

The first version recorded `g_side_fork` immediately *before* the shared expert's code — which is
*after* the entire routed path is already enqueued on `stream`. The side stream dutifully waited for
all of it and overlapped nothing. Measured: `moe:shared` 10.37 → **10.48**, TOTAL 163.95 → **166.51**.
Exactly no effect, and it would have read as "Finding 55 does not transfer to the engine".

**An event records a POINT IN THE STREAM, and that point has to precede the work you want to overlap
with.** Fork placement is the whole change; the code that runs on the side stream is identical either
way. Any future fork/join in this engine should be checked against this first.

### What it cost, and why the gain is 1.7 % and not 5 %

The pure byte model says folding 574 MB into a 241 GB/s stream costs +2.3 ms, so the saving should be
10.37 − 2.3 ≈ 8 ms. Measured, the routed GEMVs went **+6.68 ms** (`w1w3` 49.03 → 54.79, `w2` 22.35 →
23.27) and the net saving was **2.78 ms**. The shared expert is FP8 and its GEMVs are small; run
concurrently they compete for SMs and L2, not only for DRAM bytes. **Overlap is not free against a
saturated kernel — it is only cheaper than serialising.** Price the next pair with that, not with the
byte model, which over-predicted by ~3x here.

Corroborated three independent ways, all moving together: dprof TOTAL −1.7 %, ksweep K=5 −1.7 %,
end-to-end spec +1.6 % and base AR +1.9 %.

---

## Finding 57 — C1: the kv chain forked off the q chain, and the pricing model needed refitting

**ADOPTED, small.** `cattn:q_proj` **23.85 → 19.99 ms**; K=5 verify **153.48 → 152.26 ms** (−0.8 %);
spec **17.48 → 17.61 tok/s** (+0.7 %). Token sequence byte-identical; all gates PASS.

The kv chain (`wkv` GEMM → rmsnorm → rope → `act_quant_fp8sim`) consumes `xq/xs` and nothing else, so
it is independent of the whole `wq_a → rmsnorm → act_quant → wq_b` chain that follows it in
`build_qKV`. `wkv [512,4096]` is the **worst shape on `overlap_probe`** — 47.8 GB/s standalone at M=5
against 150.4 on four streams — because 512 rows is 32 m16 tiles, and 32 blocks cannot cover DRAM
latency on 20 SMs.

### The nesting hazard, and why there are now two side streams

`compressed_verify_step_indexer` forks the compressor emits onto `g_side` and then calls `build_qKV`
**inside** that region. A second fork there reusing `g_side_fork`/`g_side_join` would re-record an
event whose first pair is still outstanding — which does not fail, does not warn, and silently
rewires the dependency graph. So `dscratch` now carries a second stream and a second event pair, and
the rule is written at the declaration: **one fork site owns exactly one pair; add a third, never
share one.**

### The pricing model was wrong low, and is refit

Finding 56 said "expect ~20-25 % of the hidden op's standalone cost". This predicted ~0.4-0.5 ms and
delivered **1.22 ms** — 2.5x the prediction, which is exactly the falsification condition the queue
entry named. Refit across all three pairs, using the measured drop in the region that *contained* the
hidden work as the denominator:

| pair | hidden | partner's state | recovered |
|---|---|---|---|
| shared expert ∥ routed experts | 10.37 ms | routed at roofline (241 GB/s) | 2.78 = **27 %** |
| compressor emits ∥ `build_qKV` | 8.56 | q chain, mixed (contains the saturated `wq_b`) | 1.74 = **20 %** |
| kv chain ∥ q chain | 3.86 | q chain, and `wkv` is the most starved shape in the engine | 1.22 = **32 %** |

**Recovery is 20-32 %, and it rises the more starved BOTH sides are.** The byte model (which predicts
~80 %) remains wrong by 3-4x and should not be used. The mechanism for the shortfall is visible in the
numbers every time: the hidden op's cost does not vanish, it reappears in the partner region, because
two concurrent kernels contend for SMs and L2 as well as for DRAM.

### Honest scale

This is a +0.7 % end-to-end change sitting right at the ±1 % cross-run band, and it would not be
adoptable on the end-to-end number alone. It is adopted on the two tighter within-run measurements
that agree with it and with each other — `cattn:q_proj` −3.86 ms and ksweep K=5 −1.22 ms are GPU-event
sums over 41 and 43 layers, not single-run wall clock — plus a byte-identical token sequence. The
concurrency lever's large wins are now taken; what remains in this class is sub-1 % per pair.

---

## Finding 58 — I3 is refuted by direct measurement, and the line number was never a location

Three cycles chased `an illegal memory access at kernels/indexer.cu:91`, then `:96`. **Both are
`CUI(cudaStreamSynchronize(stream))` at the end of `indexer_forward`** — and that is the *first real
sync in the whole layer's attention path*. Every launch from `compressed_attn.cu:36` onward (the q
chain, the kv chain, `compressor_forward`, the indexer's own GEMMs, ~20 launches) is still in flight
when it runs, because the sub-functions' own `dsync()` calls are no-ops while the arena is on. An
asynchronous fault from any of them surfaces there.

So the line number was never evidence about *where*, and every hypothesis built on it — "the fault is
in the top-k", "the fault is length-dependent", "something accumulates" — was reading a sync point as
a stack frame. That is the finding, independent of the fault itself.

### Instruments added (both permanent)

- **`DSV4_SYNCPROBE=1`** → `dprobe()`, a real checked sync placed after 21 individual launches across
  `compressed_attn.cu` and `indexer.cu`. It stops at the FIRST fault with the exact file:line of the
  launch that caused it. Off by default, one predicted branch; decode is untouched because nothing on
  the decode path calls it.
- **`DSV4_MEMTRACE=1`** → `cudaMemGetInfo` at every sweep-point boundary, which is precisely the
  measurement I3's falsification criterion asked for and which nothing had ever printed.
- **`DSV4_BALLAST_GB=n`** → reserves device memory, so the headroom the faulting run had is settable
  instead of hoped for.

### The accumulation frame is dead

I3 predicted free memory would decay across sweep points, with the longest prompt being the first
allocation big enough to fall off the end. Measured, 17 points, at the faulting run's own headroom:

```
point  0 (prompt 0): free 10.360 GiB      point  9 (prompt 1): free 11.241
point  4 (prompt 2): free 10.732          point 12 (prompt 4): free 11.407
point  8 (prompt 0): free 11.180          point 16 (prompt 0): free 11.608
```

Free memory does not decay — it **rises monotonically**, 10.360 → 11.608 GiB. The criterion said "if
FLAT, the accumulation frame is dead"; it is not even flat. Nothing leaks; the allocator consolidates
as the run proceeds. **Do not re-queue an allocator-accumulation hypothesis for this fault.**

### The fault does not reproduce on the current tree

Three full 17-point runs of a faithful reconstruction of cycle 5's sweep — same point order, same
prompt lengths, the long prompt first appearing at point 12 where cycle 5 died, `NGEN0=60`:

| run | condition | result |
|---|---|---|
| `crashprobe.log` | `DSV4_SYNCPROBE=1` (prefill fully serialised + checked) | **17/17 clean** |
| `crashctl.log` | control, no probe | **17/17 clean** |
| `crashballast.log` | `DSV4_BALLAST_GB=3` → 111.6/122.8 GiB, free 11.27 (cycle 5 sat at 111.5) | **17/17 clean** |

What is NOT controlled: cycle 5's exact prompt token ids are not recoverable from the transcripts
(the recovered `DSV4_PROMPTS` is cycle 3's set, whose SMAX is 18, not 28), so a data-dependent fault
in the top-k → `combined` → `sparse_attn` gather cannot be excluded. **Disposition: not reproducible,
not closed.** The honest state is that the instrument now exists and the next recurrence gets
attributed to a launch in one run instead of costing a cycle to re-frame. Do not spend another cycle
hypothesising about this fault without a reproduction.

---

## Finding 59 — adaptive verify width is worth 7x what the log says, because it was measured on the one prompt where it does nothing

D1 is fixed (a negative `adaptK` in a sweep entry now means fixed width), which produced the **first
fixed-width control since Finding 49**. Adaptive width has been ON by default since then on the
strength of **+2.1 % measured on the canonical prompt**. Across five prompts:

| prompt | adaptive (1.5) | fixed | adaptive gain |
|---|---|---|---|
| 0 — canonical | 17.42 / 17.65 | 17.65 / 17.71 | **−0.5 % (a wash)** |
| 1 (11 ids) | 16.01 | 14.62 | **+9.5 %** |
| 2 (18 ids) | 19.03 | 14.87 | **+28.0 %** |
| 3 (15 ids) | 11.52 | 10.68 | **+7.9 %** |
| 4 (29 ids) | 15.41 | 13.74 | **+12.2 %** |

**Finding 49's +2.1 % was measured on the single prompt where the lever is worth nothing.** On four
prompts it had never seen it is worth +8 % to +28 %.

### Losslessness is now actually tested

Every prior check compared adaptive against adaptive at a different threshold (1.5 vs 1.0 vs 2.5) —
which cannot detect a lever that is lossy in the same way at every threshold. Fixed width was not
expressible until this cycle, so the comparison that matters had never been run. It has now:
**all five prompts emit byte-identical 40-token sequences under adaptive and fixed width.** The
losslessness claim survives its first real test.

### The methodology consequence, which is the bigger half

Every baseline number this project has ever quoted comes from prompt 0. Across these five prompts the
same engine runs **11.5 to 19.0 tok/s** — the canonical prompt is not representative, and it is
specifically *insensitive* to verify-width adaptation, i.e. to the whole class of levers that trade
verify width against acceptance. A lever measured only there will be systematically under-valued by
this class of error, and Finding 49 is a worked example: 7x.

**Quote the canonical number as the canonical number, and gate any acceptance/width lever on the
multi-prompt sweep.** `17.6 tok/s` is prompt 0; the engine's range on varied prompts is 11.5-19.0.

---

## Finding 60 — the engine is nondeterministic on identical input, it always was, and it costs 2.2x

The single most consequential measurement this cycle, found by accident while looking for a
warm-up ramp. **36 sweep points that are byte-identical in every parameter** (same prompt, same
adaptK, same block size, same process, back to back):

```
tok/s      10.04 ... 22.52     mean 18.09     sd 3.12 (17%)
acceptance 1.40 ... 3.33       20 distinct values in 36 points
tokens     19 distinct sequences in 36 points, diverging as early as index 2
margins    36 distinct verify-1 margin vectors in 36 points
```

**The FIRST verify of every point already differs**, so this is not accumulated drift — the draft
forward produces different numbers on identical inputs and identical state.

### What it is NOT (each eliminated with a measurement, not an argument)

| hypothesis | test | result |
|---|---|---|
| the new intra-layer concurrency | 36-point control with `NO_MOESPLIT=1 NO_ATTN_SPLIT=1 NO_KV_SPLIT=1` | **identical pathology**: 19 sequences, 20 acceptances, same 1.4-3.33 range. Pre-existing. |
| MoE atomics | grep | the only `atomicAdd` scatter is in the **non-batched** path, which decode does not take (`w.batched && w.device_route`) |
| stale window read in the draft | `dspark_attn.cu:29` | `nwin = min(t+1, win)` clamps correctly |
| uninitialised arena scratch | new `DSV4_ARENA_ZERO=1` (zeroes to the high-water mark on every `arena_reset`) | **still nondeterministic**: 8 points → 8 distinct verify-1 margins. Arena exonerated. |

Remaining suspects, in order: the **prefill's raw `cudaMalloc` scratch** (`compressed_attn_forward`
allocates 17 buffers per layer per call and never zeroes them — the arena analogue that this cycle
did NOT test), the persistent `cudaMalloc` buffers read before written (`mh_pre` is written only for
`PSp` of `SMAX-1` rows), and genuine FP non-associativity from scheduling-dependent reduction order.

### Why it matters more than any remaining kernel lever

Acceptance swinging 1.40-3.33 on identical input **is** the tok/s swinging 10.0-22.5. The engine's
mean is 18.1 while its own good draws are 21-22. Nothing in the kernel work left on the table is
worth anything close to that.

It also invalidates a measurement practice this project has used throughout: **a single sweep point
is a ±17 % measurement on a sensitive prompt.** Every single-point A/B in this log carries that error
bar unless the two arms were adjacent. The `blksweep` replicate machinery added this cycle is the
fix — use ≥3 replicates per cell and compare means.

### Two things that are NOT the cause but are worth having

- **Clock pinning (`jetson_clocks`) is a real, free gain.** The GPU idles at 315 MHz of 1386 and the
  **memory controller at 2750 MHz of 4266** under `nvhost_podgov`/`bpmp-bwmgr`. Paired at matched
  sweep positions: **rep1 +6.4 %, rep2 +3.5 %, rep3 +3.0 %**, and the cold base-AR window (measured
  right after load, before the governor ramps) goes **10.48 → 12.65 tok/s, +20.7 %**. Streaming
  roofline is unchanged (235.6 GB/s) because a long probe ramps the governor by itself — which is
  exactly why this was invisible to every roofline measurement ever taken here.
- **The "ramp" is mostly not DVFS.** Pinning only moves rep3/rep1 from 1.143 to 1.107. The residual
  is the nondeterminism above, not a warm-up: with 36 identical points the sequence is noisy, not
  monotone.

### Best estimate of this cycle's concurrency work, from the paired 36-point data

Splits ON mean **18.09** vs splits OFF **17.54** over 36 matched points = **+3.1 %**. That supersedes
the +2.5 % single-point figure, and it is the number to quote.

---

## Finding 61 — N1 localised to one function call, and it is NOT what Finding 60 said it was

Finding 60 called this "nondeterminism". It is not. **The engine produces the SAME sequence of prefill
states on every run** — hashes `d75b8bc5… / 1515183c… / 33a81742… / 876ecdcc… / a78519f8…` reproduce
byte for byte across six separate runs, with concurrency on and off, with the arena zeroed and not,
with the caches zeroed and not. It is a **deterministic 5-cycle**: point 0 == point 5, 1 == 6, 2 == 7.

### Where it is

`DSV4_HASH=1` hashes `main_x`/`mh_pre` at every sweep point; `DSV4_HASH=2` hashes the hidden state
after every one of the 43 layers; and an input hash covers `h0` (post-embed) and `h` (post-hc_expand).

- **The input to layer 0 is bit-identical at every point** (`h0=de51597d…`, `h=ca0416b3…`, always).
- **Layer 0's output already differs** — all 43 layers differ, the first one included. Layer 0 is
  `compress_ratio==0`, a pure sliding layer: `hc_pre → rmsnorm → mla_cache_kv → mla_forward →
  hc_post → moe_forward`.
- **The weights are intact** (1 MB of layer 0's `wq_a` hashes constant at every point), so this is
  not an out-of-bounds write corrupting the model.
- **Scratch addresses are constant** (`scratch0=0x32f9d0000` at every point), so it is not the
  allocator handing out different alignments.
- `h` alternates between two buffers with **period 2** (the 43 odd swaps), and the cycle is **5**, so
  the swap is not it either.

**Identical inputs, identical weights, identical addresses → different output, on a fixed 5-cycle.**

### The clue that should be pulled next

**It does not happen at `NGEN0=20`. It happens at `NGEN0=60`.** At 20, four consecutive identical
points give one hash. So the effect requires the PREVIOUS point's decode to have run far enough, and
whatever it leaves behind is not any buffer that has been zeroed. `seqmax` also differs between the
two (51 vs 91), so buffer *sizes* are a live variable and not only their contents.

### Eliminated, each with a measurement

| hypothesis | test | verdict |
|---|---|---|
| the intra-layer concurrency (Findings 55-57) | prefill hash sequence with all three splits OFF | **identical, byte for byte.** Fully exonerated at the sharpest available resolution. |
| uninitialised prefill scratch | `tests/gate_scratch_init` (new): same weights/input, scratch poisoned 0x00 vs 0xFF vs 0x3C, arena included, lengths 1..29 | bitwise identical everywhere — **no uninitialised read** |
| the arena | `DSV4_ARENA_ZERO=1` + hash | identical hash sequence |
| persistent KV / `xin` / `main_x` / `mh_pre` | `DSV4_ZERO_CACHES=1` + hash | identical hash sequence |
| MoE | `gate_units`: 32 repeats of the decode configuration on one input | 0 differ, bit-exact |
| allocator addresses | `scratch0` traced per point | constant |
| weight corruption from an OOB decode write | 1 MB weight hash per point | constant |
| `tc_ensure_repacked` rewriting expert weights lazily | code | guarded by `if(!g_moe_gemv)`; the engine takes the GEMV path, which reads original fp4 |

### Retraction

Finding 60 recorded "zeroing the prefill scratch moves 8/8 distinct margin vectors to 5/8, so it is a
REAL contributor". **That is withdrawn.** The unit gate shows the prefill chain is poison-independent,
and the engine emits the same hash sequence zeroed and unzeroed — 8/8 → 5/8 was a 5-cycle sampled 8
times, not an effect of zeroing. `NO_ZERO_SCRATCH` stays default-on only because it is free (decode
never touches those buffers) and the comment in both files now says it fixes nothing.

The measurement lesson is the same one Finding 60 raised, applied to my own result: **a count of
distinct values over 8 samples is not evidence about a mechanism.** The hash sequence is, because it
is exact and reproducible.

---

## Finding 62 — FOUND AND FIXED: `tc_ogroup_fp8_kernel` never wrote rows 16+, so every prompt of 18+ tokens had garbage in its prefill

**This was never a performance bug. The engine was producing incorrect output.**

### The defect

`tc_ogroup_fp8_kernel` (and its f32 twin `tc_ogroup_kernel`) is a **single m16 mma tile**: `gid=lane>>2`
covers rows 0-7 and `gid+8` covers 8-15. There was **no loop over row tiles**. For `bs > 16` every row
from 16 up was never computed and never stored, leaving the caller's `og` buffer uninitialised there —
and `mla_forward` then feeds `og` straight into `act_quant_fp8` and `wo_b`.

The row masks (`m0`/`m8`) made the *reads* safe, so nothing ever faulted and no gate ever fired. The
only symptom was that the prefill output for positions ≥16 in the two pure-sliding layers was whatever
the allocator last left at that address.

`ogroup_gemm_fp8` dispatches the M=K GEMV only for `bs<=16`; at `bs=17` it falls through to this tile.
**Prefill runs at `bs = PSp = len(prompt)-1`, so any prompt of 18 tokens or more hit it.** Decode
(bs=1) and the verify (bs≤5) never did, which is why the decode gates stayed green.

### Why it hid for the whole project

The canonical gate prompt is **6 tokens**. `PSp=5`. It never reaches the tile. Every gate, every
baseline and every `GATE PASS` in this log was measured on a prompt that cannot trigger the bug, while
the multi-prompt sweeps that *did* trigger it reported it as "nondeterminism" for three cycles.

### The chain that found it

1. `DSV4_HASH=1/2` — hash `main_x`/`mh_pre` per sweep point, and the hidden state after every layer.
   Showed the prefill was a **deterministic 5-cycle**, layer 0 already differing, input bit-identical.
2. **`seqmax` was the discriminator, not decode length**: holding `NGEN0=20` and raising `seqmax`
   51 → 91 (via `NDEC`) turned the effect on. That says "reads another allocation's contents", which
   is what a layout-dependent uninitialised read looks like.
3. `compute-sanitizer --tool initcheck` on a short-decode repro: **65,536 uninitialised reads, one
   site** — `act_quant_fp8_kernel` at `mla_attn.cu:159`, via `mla_forward` ← `block_prefill_cache`.
4. The reported block index decoded to **row 16 of bs=17** — the first row past the tile.

### The fix, and the gate that now holds it

`blockIdx.z` walks the rows in steps of 16 in both kernels; grids become `(R+7)/8, G, (bs+15)/16`.
`tests/gate_ogroup_gemv` now covers **M=17, 24, 33**, and it fails without the fix and passes with it:

| M | before | after |
|---|---|---|
| 17 | cosine **0.9419** | 0.999999983 |
| 24 | cosine **0.6702** | 0.999999983 |
| 33 | cosine **0.6976** | 0.999999983 |

### In situ, the whole of Findings 60/61 collapses

Four byte-identical sweep points on the 18-token prompt:

| | before | after |
|---|---|---|
| distinct prefill states | 5-cycle | **1 of 4** |
| distinct token sequences | 19 of 36 | **1 of 4** |
| distinct verify-1 margins | 36 of 36 | **1 of 4** |

**The engine is deterministic.** Findings 60 and 61 described the symptom correctly and named the
cause wrongly; this is the cause. The canonical baseline is unchanged (18.13 vs 18.19, prompt 0 never
triggered it), all gates pass, `MATCH 5/5` and first-token 11111 hold.

### The lesson worth keeping

A masked read is not a safe read. `m0`/`m8` silenced the symptom of a missing loop, and every
correctness gate in this project ran at a prompt length that could not reach it. **Gate the shape
range the engine actually spans, not the one the canonical prompt happens to use** — the sweep
prompts (11, 15, 18, 29 tokens) had been running through this code for cycles.

---

## Finding 63 — adaptive verify width, re-measured on a correct prefill: +9-11% where it engages, and my own 7x claim was inflated

Finding 59 measured adaptive-vs-fixed verify width across five prompts and reported +9.5 / +28.0 /
+7.9 / +12.2 %, concluding the lever was worth "7x what the log says". That sweep ran on a **partly
garbage prefill** (Finding 62: every prompt of 18+ tokens had uninitialised rows in its sliding-layer
prefill output). Re-run post-fix, two replicates, arms at adjacent sweep positions
(`evidence/adaptfix2.log`):

| prompt | adaptive | fixed | gain (rep1 / rep2) | acceptance a/f |
|---|---|---|---|---|
| 0 — canonical, 6 ids | 17.89 / 17.97 | 18.02 / 18.00 | −0.7 % / −0.2 % | 2.90 / 3.39 |
| 1 — 11 ids | 16.34 / 16.40 | 14.93 / 14.98 | **+9.4 % / +9.5 %** | 2.65 / 2.90 |
| 2 — 18 ids | 19.32 / 19.38 | 17.02 / 19.26 | +13.5 % / +0.6 % | **3.33 / 3.33** |
| 3 — 15 ids | 11.81 / 13.65 | 10.87 / 12.32 | **+8.6 % / +10.8 %** | 1.76 / 2.07 |
| 4 — 29 ids | 12.09 / 13.97 | 11.10 / 12.61 | **+8.9 % / +10.8 %** | 1.82 / 2.14 |

**Corrected verdict: +9 to +11 % on the three prompts where the lever engages, a wash on the two where
it does not.** Not the +28 % that prompt 2 showed pre-fix.

Read prompt 2 carefully, because it is the clean case: **acceptance is 3.33 under BOTH modes**, i.e.
the adaptive threshold never narrowed a single verify there, so the two arms did *identical work* and
any tok/s difference between them is measurement noise by construction. Rep 1 says +13.5 %, rep 2 says
+0.6 %. That is a 13 % timing spread on two runs that are provably the same computation — a direct
measurement of trap #5 (sweep position), and the reason the pre-fix +28 % should never have been
quoted from one replicate.

So Finding 59's *direction* holds and two of its four magnitudes hold (p1 +9.5 → +9.4, p3 +7.9 →
+8.6/+10.8). Its headline number does not. **Against the originally recorded +2.1 %, adaptive width is
worth roughly 4-5x, not 7x**, and the honest way to state it is per-prompt, not as a single ratio.

### The determinism fix, confirmed at the behavioural level

Every cell's two replicates have **exactly** matching acceptance — 2.90/2.90, 2.65/2.65, 3.33/3.33,
1.76/1.76, 1.82/1.82, across five prompts and both modes. Before Finding 62, 36 identical points gave
20 distinct acceptance values. The engine now does the same work on the same input every time.

**Timing still varies with sweep position** (p3 adaptive: 11.81 in replicate 1, 13.65 in replicate 2,
same acceptance). Output determinism and timing determinism are different properties; F62 fixed the
first. Keep putting A/B arms at adjacent positions.

---

## Finding 64 — the MoE was never at the roofline: it re-read every expert once per row, and the byte model that said otherwise was measuring the wrong thing

**ADOPTED. 18.13 → 19.22 tok/s spec (+6.0 %); K=5 verify 152.1 → 141.5 ms (−7.0 %); `moe:w1w3`
55.66 → 44.54 ms (−20 %). Output byte-identical, all gates PASS.**

### The assumption that was wrong

`LEVERS.md` opened with "the routed MoE is 50 % of the verify and it is at the memory roofline —
kernel work cannot touch it". That rested on `ROOFLINE.md`'s `c_v` byte model, which puts the K=5
expert union at **29.9 of a possible 30**, i.e. the five block positions share almost no experts. It
was never measured. `DSV4_MOEUNION=1` (new: counts `e` with `off[e+1]>off[e]`, which *is* the union)
measures it, verify-only:

| K | modelled | **measured** |
|---|---|---|
| 1 | 6 | **6.00** (exactly top-6 — the instrument validates itself) |
| 2 | ~11.7 | 9.67 |
| 3 | ~17 | 12.58 |
| 4 | ~22 | 15.16 |
| 5 | **29.9** | **17.53** |

Consecutive tokens share most of their experts. The model assumed they barely do.

### What the measurement then exposed

With the real union, the MoE appeared to be at **127 GB/s = 55 % of roofline** — 36 ms of headroom in
the single largest block of the verify. That reading was also wrong, and the kernel says why:

```
for(int r=0;r<me;++r){            // rows this expert serves
    for(int kb=lane; kb<nb32; kb+=32){
        WAv[u]=__ldcs(...);       // <-- the weight load is INSIDE the row loop
```

**The weight loads sat inside the row loop**, so an expert serving `me` rows had its entire weight
matrix re-read `me` times. Real traffic is `rows x expert_bytes` = 30 × 13.37 MB × 43 = **17.25 GB**,
not `union x expert_bytes` = **10.08 GB**. So the kernel was at ~217 GB/s — genuinely near roofline —
**for bytes it had no need to move**. Both the "at the roofline" claim and the "55 % of roofline"
correction were artifacts of counting bytes the wrong way; only the kernel settles it.

The `29.9` in the byte model happens to be close to `rows` (30), which is why the model matched the
measured time for the whole project and never looked wrong.

### The fix

This is the **same "read B once, dot it against all M rows" transformation Findings 40/42/43 applied
to `ogroup`, `lm_head` and `gemm_fp32`** — the MoE grouped GEMV never got it. Weights and scales are
hoisted out of the row loop; `RB` rows accumulate against one load. Per-(row,n) accumulation order
over `kb` is unchanged and the inner half2 block math is untouched, so it is **bit-exact**, and the
model confirms it: base-AR tokens and verify argmax are identical to the previous kernel.

`RB` is a template parameter, and it must **follow the batch**, which cost a measurement to learn:

| | baseline (RB=1) | RB=8 always | **RB by batch** |
|---|---|---|---|
| spec tok/s | 17.81 | 18.48 | **19.01** |
| base AR tok/s | 12.39 | **12.19** | **12.64** |
| ksweep K=5 | 152.14 | 143.42 | **141.53** |

At M=1 decode every expert serves exactly one row, so there is nothing to amortise and a large `RB`
only holds 16 accumulators live — a **2.9 % base-AR regression**. `RB = smallest power of two ≥ bs`
gives base AR the original kernel byte-for-byte and the verify one weight load per expert. `MOE_RB=n`
overrides for A/B; `MOE_RB=1` is the old kernel exactly.

`moe:w2` did **not** improve (23.63 → 23.88). Its K is `inter`=2048, so `nb32`=64 and each lane runs
only two `kb` iterations — at that shape the kernel is issue/latency bound, not bandwidth bound, and
removing bytes buys nothing. Amortisation only pays where the weight stream is the constraint.

### The lesson

**A byte model is a hypothesis, not a measurement.** This one was load-bearing for the entire
priority order — it is what marked half the verify "closed" — and it was never checked against the
engine. The check cost one counter and one run. Anywhere else this project reasons from modelled bytes
rather than measured ones deserves the same treatment.

---

## Finding 65 — Finding 64 shipped the wrong RB: rows-per-expert is not `bs`, and the histogram is skewed to 1

**ADOPTED. Spec 19.22 → 19.31 tok/s; K=5 verify 141.53 → 138.04 ms; cumulative over Finding 64's
starting point: verify 152.14 → 138.04 (−9.3 %), spec 17.81 → 19.13 (+7.4 %) on the DPROF/KSWEEP
control, 19.31 clean. Output byte-identical to the pre-F64 kernel; all gates PASS.**

Finding 64 chose `RB = smallest power of two ≥ bs`, reasoning that an expert can serve at most `bs`
rows. True but useless: it is an upper bound, not the distribution. Measured (`DSV4_MOEUNION=1`, now
also reporting the rows-per-expert histogram) at K=5:

```
union 17.53 experts over 30 rows;  max rows/expert = 5
me=1: ~70 %   me<=2: ~88 %   me<=3: ~95 %   me=4,5: the tail
```

**Roughly seven in ten experts serve exactly one row**, so `RB=8` allocated `acc[8][BN]` — 16
accumulators live — to hold a single row's worth of work. That is precisely the occupancy trap
Finding 28 recorded on the fp8 GEMV, and I cited it in the F64 comment while walking into it.

`ncu` on the measured grouping (`tools/ncu_target.cu` now builds 18 experts over 30 rows instead of 30
one-row experts, which is why it could not see this before):

| RB | time | registers | occupancy |
|---|---|---|---|
| 1 | 441.9 µs | 55 | 71.4 % |
| **2** | **423.8 µs** | 62 | 63.1 % |
| 4 | 434.4 µs | 68 | 55.3 % |
| 8 | 497.3 µs | 85 | **39.4 %** ← what F64 shipped, *worse than RB=1* |

In situ, `RB=1 at bs==1 else 2`:

| config | spec | base AR | ksweep K=5 | `moe:w1w3` | `moe:w2` |
|---|---|---|---|---|---|
| pre-F64 | 17.81 | 12.39 | 152.14 | 54.46 | 23.60 |
| F64 (RB by bs) | 19.01 | 12.64 | 141.53 | 44.54 | 23.88 |
| **F65 (RB=1/2)** | **19.13** | **12.68** | **138.04** | **43.24** | **21.36** |

Note `moe:w2` finally moves here (23.88 → 21.36, −10.6 %) where F64 left it flat: its problem was
never bytes, it was that the RB=8 register pressure cost more than the amortisation returned at
`K=inter=2048`.

**The transferable point: an upper bound is not a distribution.** `bs` bounds rows-per-expert, and
sizing a register array from it was wrong by 4x on the common case. Any future kernel parameter that
scales a live register array should be fitted to the measured histogram, not to the worst case — and
the histogram is now printed by the same instrument that measures the union.

A second-order lesson for the probes: `ncu_target` had been modelling 30 experts of one row each,
which for a row-amortised kernel is the one shape where amortisation cannot show up at all. **A probe
that models the worst case measures a kernel the engine never runs.**

---

## Finding 66 — B6 retired twice: the expert pointers are never aligned, and shuffling the funnel partner is worse than loading it

Two negative results on the same lever, both cheap, neither costing a model run.

**B6 as stated is impossible.** `k_grouped_fp4_gemv_e8m0` loads both `wa` and `wa+16` and funnel-shifts
because the weight pointer is misaligned; the obvious fast path is "skip it when aligned". Scanning
`docs/hdrs` (all 48 shard headers, 45,821 tensors) says that never happens:

```
ALL tensors,    data_offsets[0] mod 16:  {0: 296, 4: 2, 8: 44400, 12: 1123}
EXPERT tensors, data_offsets[0] mod 16:  {8: 43470, 12: 966}      <- none at 0
```

Every expert tensor is misaligned, ~98 % of them by exactly 8 bytes. The fast path would never fire.
Note this also updates Finding 41's standing caveat: the misalignment is not 4-byte-arbitrary, it is a
**constant 8**, because every tensor in this checkpoint has a size that is a multiple of 16 and the
data blob starts 8 bytes off.

**B6' — supply the partner by warp shuffle — is worse than the load it removes.** Lane L's second load
is bit-for-bit lane L+1's first load, so a `__shfl_down` should give it for free on a kernel that is
latency-bound on exactly those loads. Measured on the corrected `ncu_target` grouping at RB=2:

| | time | registers | occupancy | global ld requests | long_scoreboard |
|---|---|---|---|---|---|
| two loads (shipped) | **423.8 µs** | 62 | 63.1 % | 1,058,816 | 3.90 |
| shuffled partner | 650.6 µs | 67 | 55.7 % | 1,058,816 | 9.81 |

**+54 % slower, and the request count did not move at all.** Lane 31 has no partner in the warp so it
still needs a real load, and that load is *predicated*, not branched away — the warp executes both the
shuffle chain and the load path. Four `__shfl_down` per `uint4` per operand is 8 extra instructions per
`kb` buying nothing, and they consume the same issue slots the loads were waiting on.

**The transferable point: "these two values are identical" does not imply "the second one is free".**
On a warp-uniform load the coalescer had already merged lane L's and lane L+1's requests into the same
sectors, so the second load was never costing DRAM traffic — only an instruction that the hardware was
already pipelining. Replacing a coalesced load with a shuffle chain trades a free thing for a costly
one. Check whether the redundancy actually reaches memory before removing it.

---

## Finding 67 — `wq_a` runs at 26 GB/s because the m16 tile's warp count is N/8, and three fixes for it all failed

No adoption. Four measurements, three retired levers, and one diagnosis worth keeping.

### The diagnosis

Third-level dprof marks (new: `q:aq(x)`, `q:wq_a`, `q:rms+aq`, `q:wq_b`, `q:tail`, `q:kv_join`,
`o:rope`, `o:wo_a`, `o:wo_b`) split the two worst-efficiency regions:

| mark | ms | bytes | GB/s | % roofline |
|---|---|---|---|---|
| `q:aq(x)` | 0.41 | — | — | 10 µs/layer |
| **`q:wq_a`** | **6.58** | 0.17 GB | **26** | **11 %** |
| `q:rms+aq` | 0.64 | — | — | 8 µs/kernel |
| `q:wq_b` | 11.67 | 1.38 GB | 117 | 50 % |
| `o:wo_a` | 11.14 | 1.38 GB | 124 | 53 % |
| `o:wo_b` | 9.24 | 1.38 GB | 149 | 64 % |
| `q:kv_join` | **0.05** | — | — | C1's fork is fully hidden |

**`tc_fp8_gemm` launches `grid.x = N/(8W)` blocks of `W` warps, so its total warp count is `N/8`
regardless of `W`** — the adaptive-`W` heuristic moves warps between blocks and never creates any. At
N=1024 that is **128 warps** on a device with ~1000 slots. `wq_b` (N=32768 → 4096 warps) reaches 117
GB/s; `wq_a` reaches 26. The efficiency of every fp8 GEMM here is predicted by `N/8`.

This also kills the fusion idea before it was built: `q:rms+aq` is **0.64 ms for two kernels over 41
layers**, i.e. ~8 µs each. Fusing rmsnorm into act_quant is worth at most 0.4 %, not the 4 % a
"tiny kernels are the problem" story would have implied. One extra mark decided that.

### Three fixes, all retired with measurements

1. **Align the weights so the `uint4` fast path fires.** Every fp8 GEMM selects `uint4` vs 4×`unsigned`
   staging on `((uintptr_t)B & 15)==0`, and Finding 66 showed that is **never** true in the engine
   (offsets are 8 or 12 mod 16) while **always** true in `gemm_bench` (cudaMalloc). A per-shard pad
   makes 99.06 % of tensors 16-byte aligned — implemented, and it bought **nothing** (`q:wq_a` 6.58 →
   6.52, spec 19.13 → 18.59). The bench had already answered this and I misread it: its `m16+smem B+4`
   column is 0.0518 vs 0.0484 aligned. **Alignment is worth 7 %, not 3.3x.**
2. **Route everything through the M=K GEMV** (`GEMV_MK_MAXM=5`), which puts one warp per output *row*.
   `q:wq_a` 6.58 → 5.85, but `q:wq_b` 11.67 → **21.26** and `o:wo_b` 9.24 → **20.56**. TOTAL 144.94 →
   176.15. It is a crossover in N, not a better kernel.
3. **Route only small N through the GEMV.** At `N<=2048` the win was swamped: TOTAL → 151.39, driven by
   **`moe:w1w3` +4.80 ms** — a mark that does not use `fp8_block_gemm` at all. The threshold had caught
   the *shared expert* (N=2048), which runs on a side stream concurrently with the routed MoE, so
   slowing the hidden op extended the kernel it hides behind. At `N<=1024` the MoE recovered (43.85)
   and `q:wq_a` still gained 0.87 ms — but `q:rms+aq` rose 0.64 → 2.41 and TOTAL was still worse
   (147.23, ksweep 141.29). **Retired.**

**The transferable rule from (3): anything that changes a kernel running on a side stream must be
priced against the MAIN stream, not against itself.** The Finding 56 table said overlap returns 20-32 %
of a hidden op's cost; it says equally that *slowing* a hidden op costs the partner, and here that
cost was 4x the gain being chased.

### What survives

`wq_a`'s 26 GB/s is real and the mechanism is understood. The remaining untried fix is **split-K** — a
K-split multiplies the warp count directly rather than trading warps between blocks — and it is the
only one of the four that attacks `N/8` itself. It is not bit-exact (the reduction order changes), so
it needs a tolerance gate rather than an equality gate. Expected: `q:wq_a` 6.58 → ~1.7 ms if it reaches
even 100 GB/s, i.e. ~3 % of the verify.

---

## Finding 68 — split-K retired, and it produced a FAKE 28 % speedup that every gate passed

**Not adopted.** But the way it failed is the finding, and it left a gate behind.

### The lever

Finding 67 established that `tc_fp8_gemm`'s total warp count is `N/8` regardless of its adaptive-`W`
heuristic, so `wq_a` (N=1024, 128 warps) runs at 26 GB/s. Split-K is the only fix that *creates* warps
rather than moving them: `blockIdx.z` takes a contiguous slice of the K-blocks, each writes its own
partial, and a **fixed-order** second pass sums them (an `atomicAdd` epilogue would have been one
kernel instead of two and would have reintroduced exactly the nondeterminism Finding 62 removed).

It works, in the narrow sense: `q:wq_a` **6.58 → 5.69 ms** across three variants.

| variant | `q:wq_a` | `cattn:ogroup` | ksweep K=5 | spec |
|---|---|---|---|---|
| baseline | 6.58 | 20.79 | 137.92 | **19.13** |
| auto SK (warps<1024) | 5.89 | 22.18 | 139.70 | 18.26 |
| SK only where starved (warps<256) | 5.73 | 22.07 | 138.62 | 17.25 |
| + persistent workspace | 5.69 | 21.83 | 138.72 | **24.44** ?! |

Two things went wrong, and the second is the important one.

**Arena churn.** The partial buffer came from `dmalloc`, which bumps the arena offset for every
allocation after it in the same layer — so buffers the split never touched moved. `cattn:ogroup` rose
20.79 → 22.07 while `wo_b` was not even being split. Moving the workspace to one grow-only
`cudaMalloc` fixed that specific effect. **A scratch allocator with a shared bump pointer makes every
allocation a global variable.**

### The fake speedup

The last row reports **24.44 tok/s, +28 %** — with `ksweep` unchanged at 138.7. A verify that did not
get faster cannot make the cycle 28 % faster. The tokens say what happened:

```
baseline:  11111 16 455 6102 294 16603 344 29168 16 455 6102 294 14251 ...
split-K:   11111 16 455 6102 294  8760 344 11111 16 455 6102 294  8760 ...   <- a cycle
```

Split-K's reduction-order change perturbed the logits enough to drop the model into a **degenerate
repeating loop** from token 6. A repetitive sequence is trivially predictable, so acceptance rose
**2.90 → 3.86** and tok/s rose with it. The speedup is entirely a quality collapse.

**Every gate passed.** `first decoded token argmax = 11111 -> GATE PASS` (position 0 only).
`MATCH 5/5 -> PASS` (the M=K verify against K sequential decodes, at one position, before the drift
compounds). `gate_tc_fp8_smem` PASS (cosine against the oracle on synthetic weights, where a tolerance
that is fine for one GEMM says nothing about 43 layers of compounding).

**This is the worst failure mode this engine has: a change that degrades output and is rewarded by the
headline metric for doing so.** Any acceptance-based number is corrupted by a quality regression,
because acceptance measures agreement between the draft and the verify — and both moved.

### The gate it left behind

Speculative decoding is supposed to be **lossless**: the verify corrects every draft, so the emitted
sequence must equal what base AR emits from the same prompt. The engine already generates both in the
same process and never compared them. It does now:

```
[spec] LOSSLESS GATE: first 8 tokens match base AR -> PASS
```

One comparison, no extra work, and it fails at token 6 on the split-K build. **Cosine-against-an-oracle
on one GEMM is not a substitute for end-to-end sequence equality**, and this is the second time this
project has found a defect that unit gates waved through (Finding 62 was the first).

### Disposition

Split-K is **retired**. Not for being slow — `q:wq_a` genuinely improved 14 % — but because the
reduction-order change is not numerically safe at this depth, and the only way to get the warps is to
change that order. `wq_a`'s 26 GB/s stands as understood-and-unfixed. Do not re-queue split-K, or any
other non-bit-exact kernel change, without the lossless gate in the run.

---

## Finding 69 — the MoE GEMV's `BN` sweep was measuring dead-code elimination, and the store was a latent wrong-answer bug

`BN` (output columns per warp) carried a comment recording a measurement — "BN=1 at 155-160 GB/s and
BN=2 at 242-249" — and BN=4 had never been tried. On the corrected `ncu_target` grouping BN=4 came out
at **258 µs against BN=2's 426**, a 1.65x win with *identical* registers (61 vs 62) and *identical*
occupancy (62.9 % vs 63.2 %). A 1.65x speedup that costs no registers and no occupancy is not a
speedup, and it was not:

```
if(lane==0){
    out[... + nbase]   = a0v;
    if(nact>1) out[... + nbase+1] = a1v;      // <- exactly two columns, whatever BN says
}
```

**The store was hardcoded to two columns.** At BN=4 the compiler sees accumulators 2..3 are never
read, deletes them, and with them **half the weight loads**. The kernel "ran" 1.65x faster while
writing half its outputs. Any BN sweep against that store was measuring work deletion.

Generalised the store to `BN` columns and re-swept honestly:

| BN | time | registers | occupancy |
|---|---|---|---|
| **2** | **431.8 µs** | 64 | 63.2 % |
| 3 | 459.6 | 78 | 47.4 % |
| 4 | 492.2 | 87 | 39.6 % |
| 6 | 555.2 | 114 | 31.9 % |

**BN=2 is optimal and the original choice was right.** Monotonic: more columns per warp means more
live accumulators and weight registers, and this kernel is occupancy-limited. BN=2 with the fixed
store costs nothing (431.8 vs 425.6, same band), and the fix removes a hazard that would have produced
silently wrong results the first time anyone changed `BN` — which is exactly what a "tunable" invites.

**The rule: a parameter that array sizes depend on must be honoured by every loop that touches those
arrays, or it is not a parameter, it is a comment.** And more generally — a speedup with no cost in
any resource is a measurement error until proven otherwise. That heuristic caught this in one look.

---

## Finding 70 — the RB choice was decided by the probe's grouping, twice, and the real histogram says RB=4

**Adopted, small.** `moe:w1w3` 42.68 → 42.14 ms (−1.3 %), ksweep K=5 138.21 → 137.93, spec 19.11 →
19.20. All gates PASS including the new lossless gate.

Finding 65 picked RB=2 from an `ncu_target` grouping whose `take` was clamped at 2 — so **no tile ever
needed chunking**, which is the one thing RB exists to control. Profiling a distribution the engine
does not have is the same error F65 itself was written to fix, one level down.

The clean verify-only histogram (`DSV4_MOEUNION=1`, 754 experts at K=5):

| rows/expert | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| share | 61.7 % | 20.7 % | 7.8 % | 4.5 % | 5.3 % |

Weight reads per expert, by RB: **RB=2 → 1.229**, **RB=4 → 1.053**, RB=8 → 1.000. And the ranking
inverts once the probe models it:

| | RB=1 | RB=2 | RB=4 | RB=8 |
|---|---|---|---|---|
| probe clamped at me≤2 (F65) | 441.9 | **423.8** | 434.4 | 497.3 |
| probe on the real histogram | 559.1 | 534.7 | **499.8** | 530.6 |

RB=8 loses in both because 16 live accumulators cost more occupancy than the last few re-reads are
worth; RB=4 is the point where chunking is nearly eliminated (1.053 reads) before the register cliff.

### Honest scale

The probe predicted −6.5 % and in situ delivered **−1.3 %** on `moe:w1w3`, with ksweep and spec both
inside their band. The gap is the kernel not being purely bandwidth-bound — it sits at ~76 % of
roofline, so a 14 % cut in bytes does not buy 14 % of time. **Adopted on the tightest mark plus a
mechanism, not on the end-to-end number**, and recorded as ~1 % rather than the 6.5 % the probe
implied. A probe predicts ranking; it does not predict magnitude.

### The standing lesson, now three times over

F65 (probe modelled 1 row per expert), F69 (store hardcoded to 2 columns so a BN sweep measured dead
code), and F70 (probe clamped rows at 2) are the same failure: **the measurement apparatus quietly
encoded an assumption, and the sweep then confirmed it.** Before sweeping a parameter, check that the
harness can actually express the regime the parameter governs.

---

## Finding 71 — `index_score` was one thread per output on a single block: 6.05 ms for 97k MACs

**ADOPTED. `i:score` 6.05 → 0.75 ms (−87 %); `cattn:indexer` 9.14 → 3.96 (−57 %); spec 19.31 → 20.16
tok/s (+4.4 %); base AR 12.58 → 13.47 (+7.1 %).** Full 40-token spec sequence **identical** to the
previous build; first-token, MATCH 5/5 and the lossless gate all PASS.

### What it was

Third-level marks inside the indexer (new: `i:qidx`, `i:iw`, `i:score`, `i:topk`) put **6.05 of the
indexer's 9.14 ms in one kernel** — 4.2 % of the entire K=5 verify:

| mark | ms (21 layers) |
|---|---|
| `i:qidx` (`idx_wq_b` + rope + hadamard + fp4sim) | 2.24 |
| `i:iw` | 0.62 |
| **`i:score`** | **6.05** |
| `i:topk` | 0.12 |

`index_score_kernel` assigns **one thread per (query, compressed-row)** and at decode there are only
`S*T ≈ 95` of them. That is a **single block, three warps, one SM**, each thread walking `H*d = 1024`
MACs serially with a stride-1 read no other lane shares — 288 µs per layer for 97k MACs.

The whole engine had been tuned around bandwidth; this one was three warps of a 20-SM device.

### The fix

One **warp** per (query, row): 32× the threads, the `d` loop lane-strided so consecutive lanes read
consecutive floats of *both* operands, and the per-head dot closing in a shuffle tree. Same head
order, same relu, same weights.

`ksweep` K=5 moves only 137.01 → 136.33, but **spec gains 4.4 % and base AR 7.1 %** — because
`index_score` is on the M=1 decode path and the prefill as well, and `ksweep` measures only the K=5
verify. A lever whose mark is inside the verify can still pay most of its rent outside it.

### On shipping a non-bit-exact change one finding after F68 retired one

The dot over `d` changes from serial to tree order, so this is exactly the class F68 caught producing
a *fake* 28 % speedup. The difference is what was checked:

- the **lossless gate** F68 left behind: `first 8 tokens match base AR -> PASS`;
- the **full 40-token spec sequence is byte-identical** to the pre-change build;
- `MATCH 5/5`, first-token argmax, and every unit gate pass.

A tree reduction over 128 elements is also *more* accurate than the serial one it replaces, which is
the opposite of split-K's situation — there the split changed the summation grouping across a 4096-long
K and the perturbation compounded through 43 layers into a repeating loop. **"Not bit-exact" is not
one category.** The question is whether the perturbation is smaller than the one already present and
whether the emitted sequence survives, and both are now cheap to answer.

---

## Finding 72 — the funnel was buying alignment the weights already had: two `uint2` loads instead of two `uint4` plus a shift

**ADOPTED. `moe:w1w3` 43.41 → 40.36 ms (−7.0 %); `MoE` 73.14 → 69.56 (−4.9 %); ksweep K=5 136.33 →
132.53 (−2.8 %); spec 20.16 → 20.44 tok/s; base AR 13.47 → 13.50.** Spec sequence identical, lossless
gate PASS.

The MoE GEMV loads **two `uint4`** per (lane, kb, operand) and funnel-shifts them, because the expert
weight pointer is not 16-byte aligned. Finding 66 measured *how* misaligned: 43,470 of 44,436 expert
tensors sit at `data_offset % 16 == 8`, 966 at 12, none at 0. Residue 8 is not 16-byte aligned — but
it **is 8-byte aligned**, and two `uint2` loads fetch exactly the sixteen bytes wanted:

| | instructions | bytes requested | funnel ALU | weight registers |
|---|---|---|---|---|
| two `uint4` + shift | 2 | 32 | yes | 8 per BN |
| two `uint2` | 2 | **16** | **none** | **4 per BN** |

Same instruction count, half the traffic, no shift. `ncu`, clean A/B on the measured grouping:

| | funnel | uint2 |
|---|---|---|
| RB=2 | 506.98 µs, 64 reg | **486.50, 61 reg** |
| RB=4 | 525.95 µs, 69 reg | **481.44, 70 reg** |

F66 had retired "skip the funnel when aligned" as impossible because the pointers are *never*
16-aligned. That was true and it was the wrong question. **The right one is what alignment they
actually have** — and a constant 8 is a fact you can build on, not an obstacle.

### Two bugs on the way in, both mine, both in the guard rather than the kernel

1. **A `static` cache answered for a struct it never examined.** The flag was computed once from
   whichever `MoEWeights` called first — the 43 main layers, all 8-aligned — and then applied to the
   **DSpark MTP draft blocks**, a different struct with different tensors. A `uint2` load at a 4-byte
   aligned address: `cuda kernels/dspark_attn.cu:85 misaligned address`, which per Finding 58 is the
   first real sync after the draft's MoE, not the fault site. Now computed per weights struct.
2. **Per-call was correct and cost 3.05 ms.** `moe:group` went 2.74 → 5.79 — a mark containing none of
   the changed code. 480 host dereferences of `w.w1p[]` sit between `dprof_begin` and the first kernel
   launch, so the GPU idles through them and the region absorbs the stall. **A host-side cost can
   appear inside a GPU mark**; if a region moves and contains nothing that changed, look at what the
   host is doing between the marks. Fixed with a per-struct cache keyed on the expert-pointer table —
   ~46 entries, computed once each, and unable to go stale the way the `static` did.

The rule that covers both: **a cached property must be keyed on the thing it describes.** Process-wide
was too coarse and per-call was too expensive; the struct was the right granularity, and it was the
granularity the property actually belongs to.

---

## Finding 73 — the MoE grouping had two `<<<1,1>>>` scans over nr=160; parallelising them took `moe:group` down 20 %, and the end-to-end number that came with it is NOT the lever

**ADOPTED on the tight mark. `moe:group` 2.66 → 2.12 ms (−20.3 %), bit-identical output.** The
26-token spec sequence is byte-identical to the control, acceptance 2.89 = 2.89, LOSSLESS GATE PASS,
MATCH 5/5, first-token argmax 11111. All 17 unit gates pass. Clocks pinned (`sudo jetson_clocks`);
caches dropped before the run. Log: `evidence/moescan.log`, control `evidence/uint2c.log`.

### The lever

Lever B0 — audit launch geometry at DECODE shapes, the class that produced Finding 71. Two of
`moe:group`'s six launches were one thread doing the whole problem:

| kernel | geometry | work |
|---|---|---|
| `k_moe_prefix` | `<<<1,1>>>` | exclusive scan of `counts[160]` |
| `k_build_tiles` | `<<<1,1>>>` | scan of `ceil(me/16)` over 160 experts, emitting tile descriptors |

Both run once per layer, 43 layers per verify step and again per token in base AR. `moe:group` moves
~0.9 MB (gather+quant of 30 rows), so its 2.66 ms was never bytes — it was latency, and nothing had
checked the geometry. Replaced with one-block 256-thread Hillis-Steele scans, chunked so `nr>256`
still works. Emission order is unchanged, so the outputs are **bit-identical integers**, not
approximately equal. `DSV4_SERIAL_SCAN=1` restores the `<<<1,1>>>` kernels.

### The attribution, which is the part worth keeping

Four new fourth-level marks inside `moe:group` (`mg:count`, `mg:prefix`, `mg:scatter`, `mg:tiles`):

| mark | K=5 | K=1 | geometry |
|---|---|---|---|
| `mg:count` | 0.19 | 0.20 | already parallel — **this is the launch floor** |
| `mg:prefix` | 0.25 | 0.24 | was `<<<1,1>>>` |
| `mg:scatter` | 0.24 | 0.24 | already parallel |
| `mg:tiles` | 0.24 | 0.23 | was `<<<1,1>>>` |
| unmarked remainder | 1.20 | — | `k_gather_x` + `act_quant_fp8` |

Two things fall out. First, **the fixed scans now sit within 30 % of `mg:count`, a trivially parallel
kernel over 30 elements — i.e. at the launch-latency floor of this box (~4.5–5.8 µs per launch).
There is nothing left in them, and B0's MoE-grouping branch is closed.** Second, every mark is
**identical at K=1 and K=5**: this cost is O(nr) and completely independent of batch size. That
matters twice over — it is why the saving pays per token in base AR as well as per verify, and it is
the internal control that killed the end-to-end claim below.

By subtraction the two serial scans cost ~1.03 ms before (≈12 µs/layer each) against ~0.49 ms now.
That is **half** what a naive model predicts: 160 dependent global loads at ~300 cycles would be
~35 µs/layer. The loads of `counts[e]` are *independent* — only the additions chain — so the compiler
pipelines them. **A single-thread loop over N elements is not N serial memory latencies unless the
addresses depend on the previous iteration.** The win is real but it is a launch-geometry win of
about 2x, not the 6x the latency model implied.

### The end-to-end number, and why it is not being claimed

| | `uint2c` (control) | `moescan` (this) | delta |
|---|---|---|---|
| ksweep K=1 | 64.73 | 64.84 | **+0.2 %** |
| ksweep K=5 | 132.53 | 128.26 | −3.2 % |
| base AR tok/s | 13.17 | 13.75 | +4.4 % |
| spec tok/s | 19.93 | 20.43 | +2.5 % |

Every one of those is in the right direction and every one is too big. The mechanism saves ~0.54 ms
per 43-layer step. That is 0.4 % of the K=5 verify and ~0.7 % of a base-AR token — an order of
magnitude under what the table shows.

**The `mg:*` marks are the falsification.** The change is provably flat in K, so it must move K=1 and
K=5 by the *same absolute* amount — ~0.5 ms, which is 0.8 % of K=1. K=1 moved **+0.11 ms, the wrong
way**. A change that is flat in K cannot produce a K-dependent delta, so the −4.27 ms at K=5 is
run-to-run variation, not the lever. `base` at 13.75 also *exceeds* the non-dprof `final6.log` run
(13.50), which is impossible if dprof instrumentation costs anything at all — the same conclusion
from the other side.

So: adopted on `moe:group` −20 % plus a mechanism confirmed by four new marks, and **recorded as
~0.4 % of the verify / ~0.7 % of a base-AR token, not as the +2.5 % the run printed.** This is
Finding 70's rule pointed the other way — there a probe over-promised and in situ under-delivered;
here the in-situ end-to-end over-delivered against the mark. Both times the mark plus the mechanism
was the number to keep.

**Only one valid control existed.** `uint2.log` (6.41 ms `moe:group`) and `uint2b.log` (5.79) are
Finding 72's *buggy* per-call-align8 intermediates, not the shipped configuration; only `uint2c.log`
(2.66) is. A cycle that had averaged all three would have invented a 3 % "control spread" out of a
known bug. Check what a control was actually measuring before you use its variance.

### B0's other named candidates, closed with numbers

- `k_topk_verify` `<<<K,32>>>` with an early return leaving 5 working threads: `i:topk` = **0.12 ms**
  at K=5, below B0's own 0.5 ms falsification bar. Leave it.
- `k_topk_decode` `<<<1,32>>>`: same kernel family, bounded by the same mark.
- `k_dg`, `k_advance_T`, `k_incr` `<<<1,1>>>`: genuinely one scalar's worth of work. Nothing to
  parallelise.

B0's named list is now exhausted: two fixed, the rest measured under the bar.

### A gate for a bit-exactness claim, not an assertion

`tests/gate_moe_scan.cu` (new, in `build_gate.sh`) diffs both scans against the `<<<1,1>>>` kernels
for **equality**, over `nr` ∈ {1,2,15,16,17,160,255,256,257,400} — below, at and above the 256-thread
block width so the chunk carry is actually exercised — crossed with dense / all-empty / one-expert /
exactly-16-each row distributions, with the outputs poisoned to 0xEE first so a kernel that writes
nothing cannot pass by inheriting the other's bytes. 41/41 exact. Per trap 9, a harness that cannot
express the regime confirms itself; the `nr=257` and `nr=400` cases exist for exactly that reason.

### Caveat on this cycle's evidence

The one sanctioned full-model run was spent with `DSV4_DPROF=1`, so this build has **no clean
non-dprof end-to-end number**. The 20.44 / 13.50 baseline in `FLYWHEEL_STATE.json` stands and was not
re-measured; do not compare it against 20.43 / 13.75 from this log, which carries dprof overhead.

---

## Finding 74 — the fp8 tile staged one 128-K-block per barrier pair; staging KC of them took the four-mark GEMM block down 13.5 %, bit-identical, and spec to 21.68 tok/s

**ADOPTED. `q:wq_b` 11.18 → 7.98 ms (−28.6 %), `q:wq_a` 6.71 → 5.80 (−13.6 %), `o:wo_b` 9.76 → 8.67
(−11.2 %); K=5 verify TOTAL 134.77 → 127.18 ms (−5.6 %); spec 20.43 → 21.68 tok/s (+6.1 %).** The
26-token spec sequence is byte-identical to the control, acceptance 2.89 = 2.89, LOSSLESS GATE PASS,
MATCH 5/5, first-token argmax 11111. All 17 gate binaries pass plus a new one. Clocks pinned
(`sudo jetson_clocks`); caches dropped before the run. Log: `evidence/kchunk.log`, control
`evidence/moescan.log` (same binary config: `DSV4_DPROF=1 DSV4_KSWEEP=1`, NDEC=8, prompt 0).

### The lever, and how the profile named it without a new instrument

Lever B8 — the four fp8 GEMM marks, 39.1 ms of the 134.8 ms verify (29 %), the largest untouched
region. LEVERS.md set the falsification as "count this kernel's bytes the way F64 did for the MoE; if
it is already at roofline, retire it in one cycle without a build." The bytes are structural, and the
count needs no new code: 41 compressed layers × the weight matrix, read once per step by both paths.

The answer was already sitting in the K=1 column of `evidence/moescan.log`. At K=1 every dense fp8
GEMM dispatches to `fp8_gemv_m1_kernel`; at K≥2 the *same bytes* go through the smem-staged m16 tile:

| mark | weight bytes/step | K=1 (M=1 GEMV) | K=5 (m16 tile) |
|---|---|---|---|
| `q:wq_a` [1024,4096] | 0.172 GB | 1.49 ms = **115 GB/s** | 6.71 ms = **26 GB/s** |
| `q:wq_b` [32768,1024] | 1.376 GB | 7.04 ms = **195 GB/s** | 11.18 ms = **123 GB/s** |
| `o:wo_b` [4096,8192] | 1.376 GB | 7.43 ms = **185 GB/s** | 9.76 ms = **141 GB/s** |
| `o:wo_a` [8×1024,4096] | 1.376 GB | 8.21 ms = **168 GB/s** | 11.46 ms = **120 GB/s** |

**Identical bytes, 1.3–4.5× the time, and the M=1 kernel proves 185–195 GB/s is reachable on these
exact rows.** So B8 was never a roofline; it was memory-level parallelism, and the M=1 column is the
control that says so. The block sat at 110 GB/s = 47 % of the 233 GB/s achievable.

### The mechanism

`tc_fp8_smemB_kernel` staged **one** 128-byte K-block per row per barrier pair: `__syncthreads`, two
`uint4` per thread, `__syncthreads`, four `mma`, repeat. Two things make that fatal at these shapes.
The kernel's total warp count is structurally **N/8** — wq_a at N=1024 is 128 warps for the whole
device, 6.4 per SM — and each of those warps has only 32 bytes outstanding at a time. Little's Law
needs ~280 KB in flight device-wide to hold 233 GB/s; wq_a had ~131 KB, halved again by the barrier,
so it ran one DRAM round trip per K-block with nothing to overlap it with.

Fix: stage **KC** consecutive K-blocks per barrier pair. Per-thread staging is `32*KC` bytes and is
independent of WARPS (8W rows × KC·128 B over 32W threads), so KC alone sets bytes-in-flight while
the barrier count falls by KC. The loads go into a register array **first** and to smem after — a
load→store→load loop reverts to one outstanding miss no matter how big KC is, which is the whole
point. Row stride becomes `KC*128+16` = `32KC+4` words, so row g still starts at bank 4g and the
eight fragment groups are still a permutation of all 32 banks, for every KC.

`acc` still takes each 128-block's `cb` in kblk order 0,1,2,…, so this is **bit-identical**, not
close. `TCB_KC=1` restores the old kernel exactly.

### Picking KC, and why it is a lookup

`gemm_bench` COLD, real shapes, M=5, the `m16+smem B+4` column (4-byte-aligned weights = what decode
actually passes), ms(GB/s) — `evidence/kchunk_bench.log`:

| shape | W | KC=1 | KC=2 | KC=4 | KC=8 |
|---|---|---|---|---|---|
| wq_a [1024,4096] | 2 | .0501(84) | **.0402(104)** | .0521(81) | .0438(96) |
| wq_b [4096,1024] | 8 | .0339(124) | **.0309(136)** | .0332(126) | .0328(128) |
| wkv [512,4096] | 1 | .0434(48) | .0329(64) | .0361(58) | **.0266(79)** |
| wo_b [4096,4096] | 8 | .1131(148) | **.1054(159)** | .1072(156) | .1074(156) |
| sw1/3 [2048,4096] | 8 | .0676(124) | **.0588(143)** | .0671(125) | .0606(139) |
| sw2 [4096,2048] | 8 | .0616(136) | **.0571(147)** | .0584(144) | .0597(140) |

KC=4 is worse than **both** 2 and 8 on five of six shapes. Non-monotone, so the rule is a lookup —
KC=2, except KC=8 at W=1 — not a formula, and it is fitted to one bench (trap 3: `gemm_bench` ranks,
it does not predict). W=8 is hard-capped at KC≤4 because 64 rows × 8 K-blocks is a 66 KB static
`__shared__`, which does not compile; that instantiation must not exist even on a forced path.

### The in-situ result, and the two controls that make it believable

| | control `moescan` | this `kchunk` | delta |
|---|---|---|---|
| `q:wq_b` K=5 | 11.18 | 7.98 | **−28.6 %** |
| `q:wq_a` K=5 | 6.71 | 5.80 | −13.6 % |
| `o:wo_b` K=5 | 9.76 | 8.67 | −11.2 % |
| `o:wo_a` K=5 | 11.46 | 11.37 | **−0.8 % — untouched kernel** |
| dprof TOTAL K=5 | 134.77 | 127.18 | −5.6 % |
| ksweep K=5 | 128.26 | 121.11 | −5.6 % |
| ksweep K=1 | 64.84 | 64.74 | **−0.15 %** |
| base AR tok/s | 13.75 | 13.78 | **+0.2 %** |
| spec tok/s | 20.43 | 21.68 | **+6.1 %** |

Per trap 12, before believing a cross-run delta, find a mark inside the same run the change must
**not** have moved. There are two, and they are structural rather than lucky:

1. **`o:wo_a` is a different kernel.** It is `ogroup_gemv_mk_kernel` in `mla_attn.cu`, not the fp8
   tile. It moved −0.8 % while its two neighbours in the same region moved −11 % and −29 %.
2. **K=1 and base AR never enter this kernel at all.** At M=1 `fp8_block_gemm` dispatches to
   `fp8_gemv_m1_kernel`, so a change confined to the M≥2 tile must leave them flat. ksweep K=1 moved
   −0.15 % and base AR +0.2 %. That also bounds this run pair's noise floor at ~±2 % (the K=1 dprof
   marks scatter 1.49→1.50, 7.04→6.96, 7.43→7.50, 13.57→13.83).

This is the mirror image of Finding 73, where the end-to-end delta was *larger* than the mechanism
could explain and was therefore not claimed. Here the mechanism predicts a K-dependent saving that is
zero at K=1, and that is exactly the shape measured.

**The spec number is a paired comparison, not two run means.** All nine verifies ran at the same K
with the same accept counts and the same drafted tokens, so they pair one-to-one:

| verify | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | Σ |
|---|---|---|---|---|---|---|---|---|---|---|
| control ms | 121.4 | 128.0 | 149.8 | 127.0 | 154.8 | 137.0 | 163.3 | 163.6 | 151.4 | 1296.3 |
| this ms | 112.5 | 117.8 | 141.8 | 119.5 | 144.3 | 131.4 | 154.7 | 154.9 | 142.5 | 1219.4 |
| delta | −7.3 % | −8.0 % | −5.3 % | −5.9 % | −6.8 % | −4.1 % | −5.3 % | −5.3 % | −5.9 % | **−5.9 %** |

Nine of nine negative. Both runs carry dprof overhead; the 20.44/13.50 numbers in
`FLYWHEEL_STATE.json` are non-dprof and were not re-measured, so **21.68 is not comparable to 20.44**
— the claim is +6.1 % against the like-for-like control.

### What is NOT claimed

`moe:w1w3` 40.35 → 39.46 (−2.2 %) and `moe:w2` 20.20 → 19.21 (−4.9 %) also moved, in kernels
(`tc_moe_gemm.cu`) this change does not touch. There is a plausible mechanism — the shared expert
runs on a side stream concurrent with the routed GEMM and *does* go through this tile (sw1/3 and sw2
improved 13 % and 7 % in the bench), so the F56/F57 pricing model run backwards predicts the partner
gets some of it back. But −2.2 % is inside the ±2 % noise this run pair demonstrates, and one run
cannot separate the two. Recorded, not claimed. The attributable saving is the 5.20 ms across
`q:wq_a`+`q:wq_b`+`o:wo_b`; the verify moved 7.59 ms.

### A gate for the bit-exactness claim, not an assertion

`tests/gate_tc_fp8_kc.cu` (new, in `build_gate.sh`) diffs KC ∈ {2,4,8,auto} against KC=1 with
`memcmp` — equality of float bit patterns, because `gate_tc_fp8_smem` next door is a **cosine** gate
and would have passed a reduction-order change. Finding 68 is why that distinction gets its own
binary. 504/504 exact over 7 shapes × M ∈ {1,2,3,5,8,13,16,17,20} × B offset {0,4} × 4 KC values,
outputs poisoned to 0xEE first so a kernel that writes nothing cannot inherit a pass. The shape list
spans every W the dispatcher picks (32768→W=8, 1024→W=2, 512→W=1, 520→N%64 tail) and KB from 8 to 64,
per trap 9.

### Where B8 stands now

The three tile marks are 27.65 → 22.45 ms. Against the M=1 path's own demonstrated rates on the same
bytes (115/195/185 GB/s → 16.0 ms) there is still ~6.5 ms in them, and `o:wo_a` — a *different*
kernel, 11.37 ms at 120 GB/s against 168 at M=1 — has not been touched at all. **B8 stays open with
~10 ms (7 % of the verify) left and a sharpened falsification: the M=1 GEMV's achieved GB/s on the
same weight bytes is the target, not the 233 GB/s roofline.**

### Two process notes

- `scripts/build_gate.sh` builds 10 of the 17 gate binaries; `gate_compressed_decode` and
  `gate_indexer_decode` were stale from 2026-08-07 and had to be rebuilt by hand from the build line
  in their own headers before they could say anything about this change. Both PASS rebuilt.
- Running the gate binaries in a `for` loop with a shared `ref/goldens` argument makes
  `gate_prefill_len` parse it as a sweep length, sweep s=0, and report **GATE FAIL** with eight
  "invalid argument" launches. It passes with no argument. Only `gate_units` and `gate_encoding` take
  a path. A gate harness that passes the wrong argv reports a failure that is about the harness.

## Finding 75 — prefill is 48 tok/s, only 3.4x decode, and the arena was a prompt-length ceiling

Measured to size a draft-head fine-tune (S5), and it answered a different question on the way.

`evidence/prefill_sweep.log` + `evidence/prefill_long.log`, real ids from the checkpoint's own
tokenizer (`tools/encode_prompt.py`'s gate reproduced 671,6102,294,8760,344 before anything ran):

| PS | 5 | 255 | 511 | 1023 | 2047 |
|---|---|---|---|---|---|
| tok/s | 28.0 | 52.6 | 50.3 | 47.7 | 43.8 |

**Prefill is ~48 tok/s against 14.20 tok/s for M=1 decode in the same run — a factor of 3.4.** A
batched forward over 255 positions amortises the 12.26 GB weight read 255 ways and should be
compute-bound; that it is not means the whole path is running on kernels fitted to M=1. Nobody has
ever profiled it. New lever B9, explicitly NOT counted toward the decode-lever queue.

**The arena was a silent prompt-length ceiling.** PS=1023 asked for 538451712 bytes against a
hardcoded 512 MiB and died with `arena overflow 538451712>536870912`, taking the run with it. The
arena holds per-position intermediates, so its high-water mark scales with the longest prompt; 512 MB
was sized for the 6-token gate prompt. Now sized from SMAX. This is the same shape as Finding 62 — a
constant fitted to the canonical short prompt, wrong for every longer one, invisible because the gate
prompt never reaches it.

Two instrumentation points added, both prefill sites (the initial one and the sweep's re-prefill), so
one checkpoint load yields a whole length curve instead of one point per 15-minute load.

## Finding 76 — the ogroup GEMV spills 44 bytes and it is LATENCY-bound: a free, bit-identical, gemm_bench-verified −7..−15 % on that kernel is worth **+0.1 %** in situ. NEGATIVE.

The lever was B8's own named next move: `o:wo_a` is `ogroup_gemv_mk_kernel`, 11.37 ms = 8.9 % of the
K=5 verify, running at 120 GB/s where the same weight bytes go through the M=1 GEMV at 168. F74 fixed
the memory-level parallelism of the *other* three marks in the block; this kernel had had no
equivalent treatment.

### What ncu said, which is not what the lever assumed

`build/ncu_target 4`, one launch of `ogroup_gemv_mk_kernel<5,4>` on the real shape [8×1024, 4096]:

| | |
|---|---|
| Registers / thread | **64** (`__launch_bounds__(256,4)` → 65536/(256·4)) |
| **Local memory spilling requests** | **806,912**, spill overhead 100 % |
| Warp cycles per issued instruction | 12.98, of which **8.1 (62.3 %) = L1TEX scoreboard** |
| Memory Throughput / Compute (SM) | 41.3 % / 49.5 % |
| Theoretical / achieved occupancy | 66.7 % (register-limited) / 56.1 % |
| Executed instructions | 14,800,896 = **226 warp-instructions per warp per k-block**, of which ~100 are FP32 |

So the kernel is neither bandwidth-bound nor at the roofline: it is **spilling**, and 126 of the 226
instructions per warp-k-block are not arithmetic. That reads as an issue-rate problem with an obvious
cure, and the cheapest register to give back is free.

### The transform, and why it is bit-identical rather than merely close

`ws[r]` is indexed `(rr/128, kb)` with `rr = gr0+r`, `gr0` a multiple of NR, and NR ∈ {1,2,4,8} so
NR | 128. Every row a warp owns therefore lands in the **same** 128-row scale block: `ws[0..NR-1]`
are the same float, always. The compiler cannot see it because `rr` goes through the tail ternary, so
it emitted **NR scale loads and NR `exp2f` per k-block** where one would do. WS1 computes it once —
value equality, not a reassociation, so `memcmp` is the right instrument and not a cosine (trap 6).
`tests/gate_og_ws1.cu`, new: **72/72 configurations bit-identical** over M ∈ {2,3,4,5,6,7,8,13,16} ×
NR ∈ {1,2,4,8} × two shapes, outputs poisoned to 0xEE first.

**A hoisted pointer cost more than the loads it saved.** The first version hoisted
`const uint8_t* scrow = wsc + (gr0/128)*scw` out of the k-loop. `ptxas -v` went **44 → 60 bytes of
spill** at `<5,4>` and gemm_bench lost **4.4 %**: a 64-bit pointer live across the whole loop
occupies two permanently-allocated registers in a kernel that is already 8 registers over its cap,
which is worse than three short-lived ones. Re-expressed as a 32-bit `int scoff`, spill went back to
44 bytes — identical to the incumbent. *In a register-starved kernel, "hoist the loop-invariant" can
be a pessimisation, and `ptxas -v` costs one recompile to find out.*

### gemm_bench said this was a real win. It was not.

Real shape, 12 rotating copies, 3 replicates per arm, ms (lower better):

| M (=K) | NR the dispatch picks | WS1 on | WS1 off | bench delta | **in-situ `o:wo_a` delta** |
|---|---|---|---|---|---|
| 2 | 2 | 0.1658 | 0.1794 | **−7.6 %** | −2.5 % |
| 3 | 2 | 0.1902 | 0.1954 | −2.7 % | −2.4 % |
| 4 | 2 | — | — | — | −4.2 % |
| 5 | 4 | 0.1978 | 0.1976 | **+0.1 %** | +1.5 % |
| 1 | *(kernel not used)* | 0.1620 | 0.1631 | ±0 | +1.7 % |

Off-dispatch points were larger still — M=3/NR=4 **−14.8 %**, M=8/NR=4 **−14.9 %**, M=5/NR=2
**−12.7 %** — and WS1 was never slower at any (M,NR). The exception is exactly the shipped
`<5,4>`, and `ptxas -v` says why: it is the one instantiation whose spill is **unchanged** at 44
bytes, so the freed registers never became occupancy.

### The measurement that closed it

`evidence/ogws1.log` vs `evidence/kchunk.log`, both `DSV4_DPROF=1 DSV4_KSWEEP=1`, clocks pinned
(`jetson_clocks`), caches dropped.

| | control | this | delta |
|---|---|---|---|
| spec tok/s | 21.68 | 21.67 | **−0.05 %** |
| base AR tok/s | 13.78 | 13.72 | −0.4 % |
| ksweep K=5 | 121.11 | 120.52 | −0.5 % |
| ksweep K=1 / K=2 / K=3 / K=4 | 64.74 / 80.93 / 98.05 / 108.67 | 65.09 / 80.71 / 98.20 / 108.39 | +0.5 / −0.3 / +0.2 / −0.3 % |
| dprof TOTAL K=5 | 127.18 | 127.21 | +0.0 % |

**The tightest instrument available says +0.1 %.** All nine spec verifies pair 1:1 at identical K
and identical accept counts, so they compare directly rather than as two run means:

| verify | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | Σ |
|---|---|---|---|---|---|---|---|---|---|---|
| K, accepted | 2, 1/1 | 2, 1/1 | 4, 1/3 | 2, 1/1 | 4, 3/3 | 3, 2/2 | 5, 4/4 | 5, 1/4 | 4, 3/3 | |
| control ms | 112.5 | 117.8 | 141.8 | 119.5 | 144.3 | 131.4 | 154.7 | 154.9 | 142.5 | 1219.4 |
| this ms | 113.4 | 117.5 | 141.4 | 119.2 | 144.2 | 131.7 | 154.8 | 155.4 | 143.2 | 1220.8 |
| delta | +0.8 % | −0.3 % | −0.3 % | −0.3 % | −0.1 % | +0.2 % | +0.1 % | +0.3 % | +0.5 % | **+0.1 %** |

That instrument's own spread is ±0.8 %; F74 moved it −5.9 % on the same nine points. The emitted
26-token spec sequence and the 8-token base-AR sequence are both byte-identical to the control,
acceptance 2.89 = 2.89, **LOSSLESS GATE PASS**, first token 11111.

**Why the four negative `o:wo_a` marks are not the result.** `o:wo_a` at **K=1 moved +1.7 %, and it
cannot have moved** — at M=1 `ogroup_gemm_fp8` dispatches to `ogroup_gemv_fp8_kernel`, which this
change does not touch. That is trap 12's within-run control, and it bounds this mark's noise at
~±2 %. An independent control pair (`kchunk` vs `moescan`, two runs of *different* builds whose change
did not touch this kernel either) scatters `o:wo_a` from **−1.8 % to +2.4 %** across the same five K.
Against a ±2.4 % floor, K=2 (−2.5 %) and K=3 (−2.4 %) are the floor, K=5 is +1.5 %, and one mark of
five outside the band is not a signal. The verify totals, which average 41 launches, all sit inside
±0.5 %.

### The mechanism, and the family it closes

ncu already said it and the lever ignored it: **8.1 of the 13.0 cycles between issues are L1TEX
scoreboard stalls.** The kernel is *latency*-bound, not issue-bound. Deleting instructions from a
warp that is parked waiting on memory returns the memory latency, which is zero — and the freed
registers only pay if they cross a spill or occupancy threshold, which at `<5,4>` they provably do
not. gemm_bench sees a win because a standalone launch with nothing to contend with is issue-limited;
in situ, 41 launches per verify overlap other work and are not.

**This closes the family, not just the instance.** The other instruction-count cures for this kernel
have exactly the same shape and need not be built: pairing the four e4m3 decodes into two
`__nv_cvt_fp8x2_to_halfraw2` (saves 8 of 226), and replacing `exp2f` with `__int_as_float(e<<23)`
(saves a MUFU, and is *not* bit-identical at `e==0`, which is 2^-127 subnormal, not zero). Anything
that does not change this kernel's **memory** behaviour is priced at zero here.

The register cap is also confirmed exhausted a second time, now with WS1 on: at M=5/NR=4,
`OGMK_BLOCKS_PER_SM` = 4 → **0.1978**, 3 → 0.2010, 2 → 0.1997. B7' said BPS=4 with the spilling code;
it is still BPS=4 without three of the spilled registers.

### Disposition: kept, defaulted OFF

`OG_WS1=1` opts in; the default is the pre-change kernel, which is the arm every baseline in
`FLYWHEEL_STATE.json` was measured under. Nothing unmeasured ships. The code and the gate stay
because the value-equality argument and its memcmp harness are the expensive part, and a future
variant that changes the kernel's *memory* behaviour — the smem-staged variant reading `o4[m]`
lazily per m from shared memory instead of holding M float4s live, which is the only idea left that
removes ~16 registers rather than 3 — would want both.

### Three process results

- **`gate_fp4_gemv` had not linked, and therefore had not tested anything, since F65/F72.** Its local
  `extern` declaration of `tc_fp4_grouped_gemv_e8m0` was two parameters behind the engine
  (`rows_hint`, `align8`), so it failed at *link* time — which looks like a build problem, not like a
  gate that stopped existing — while a binary from 2026-08-06 sat in `build/` looking like a pass.
  Declaration repaired to the engine signature and the call now passes the values `moe_forward` uses.
- **`build_gate.sh` now builds all 19 gate binaries, not 10.** F74 flagged that the other 7 go stale
  silently; that is how the item above survived. Sources are listed in the script, so there is one
  place that can drift instead of nineteen.
- **Trap 16 again, this time in the harness I wrote to check trap 16.** Running the suite with a
  shared `ref/goldens` argument makes `gate_encoding` — which takes its *own* optional directory —
  exit 2 and look red. It passes with no argument. 19/19 PASS.
- `tools/dprof_diff.sh` added: pairs two dprof+ksweep logs by (K, sub-op). Every cycle since F70 has
  read these tables by eye, and F73's lesson was that the decisive number is the control the change
  must *not* have moved. It is the tool that produced the K=1 row above.

## Finding 77 — the clean re-baseline F73 and F74 both skipped: 21.76 / 13.64

Cycles 10 and 11 each spent their one sanctioned run on `DSV4_DPROF=1`, so two adopted findings landed
against a baseline of 20.44/13.50 that predated both. `evidence/clean_post_f76.log`, no dprof, no
ksweep, matching `final6.log`'s config exactly:

```
first decoded token argmax = 11111  -> GATE PASS
WARM decode: 73.3 ms/tok = 13.64 tok/s
generated 26 tokens over 9 verifies: mean tokens/verify = 2.89 (block=5, max 5)
LOSSLESS GATE: first 8 tokens match base AR -> PASS
SPEC-DECODE: 46.0 ms/tok = 21.76 tok/s  (-> 1.60x)
```

**21.76 / 13.64, up from 20.44 / 13.50: +6.5 % spec, +1.0 % base.**

Two things this settles beyond the headline:

- **Acceptance is unchanged at 2.89** (final6: 2.89). F73 and F74 claimed to be bit-identical; a pure
  kernel win must leave drafting untouched, and it does. If acceptance had moved, the "bit-identical"
  claim would have been wrong regardless of what the gates said.
- **dprof overhead is ~0.4 %** — 21.76 clean against 21.68 with dprof, inside the run-to-run band.
  Finding 73 reached the same conclusion from the other side when its dprof run read `base` 13.75
  against a clean 13.50. So the profiler is not distorting the marks it reports.

Session arc: **16.86 -> 21.76 spec (+29.1 %)**, **10.30 -> 13.64 base (+32.4 %)**. Against the measured
ceilings that is 70.6 % of the 30.8 tok/s spec cycle floor at acceptance 2.89, and 71.8 % of the 19.0
tok/s base roofline.

The process fix that came with it is in `scripts/flywheel.sh`: the harness now classifies each cycle's
run and maintains `baseline.dprof_runs_since_clean`, and at >= 2 step 6 makes the next run a mandatory
clean re-baseline. The executor had already *noticed* this and written an honest note refusing to
overwrite a clean baseline with a contaminated one -- but noticing is not a mechanism, and it went two
cycles anyway.

---

## Finding 78 — double-buffering the fp8 tile's staged chunk: bit-identical, +0.28 % in situ. B8's last named move is dead, and the reason is a FOUR-register occupancy step

**NEGATIVE. Kept behind `TCB_DB=1`, default OFF.** Nine paired spec verifies **1219.4 → 1222.8 ms
(+0.28 %)**, spec **21.68 → 21.62 tok/s (−0.3 %)**, base AR 13.78 → 13.69 (−0.7 %), ksweep K=5
121.11 → 121.19 (+0.1 %). The mark the lever was aimed at, `o:wo_b`, moved **the wrong way at every
K≥2** (+4.6/+5.0/+3.7/**+2.8 %**). 19/19 gates PASS, `gate_tc_fp8_kc` now **1134/1134 bit-exact**
(it sweeps DB as well as KC), the 26-token spec sequence and 8-token base-AR sequence are
byte-identical to the control, acceptance 2.89 = 2.89, **LOSSLESS GATE PASS**, first token 11111.
Clocks pinned (`sudo jetson_clocks`), caches dropped. Log `evidence/dbuf.log`, control
`evidence/kchunk.log` (same config: `DSV4_DPROF=1 DSV4_KSWEEP=1`, NDEC=8, 24 spec tokens, block=5,
passes=1, adaptK=1.50, prompt 0).

### The lever

LEVERS.md B8 named exactly one remaining move on the three fp8 tile marks (`q:wq_b` 8.0 + `o:wo_b`
8.7 + `q:wq_a` 5.8 = 22.5 ms): **double-buffer the staged K-chunk so round n+1's global loads issue
before round n's mma** — the F74 mechanism taken one step further. F74 raised the bytes in flight per
round (KC K-blocks per barrier pair instead of 1) but left the round itself serial:

```
DB=0:  [sync] issue NH loads .. WAIT .. store [sync] mma        [sync] issue .. WAIT .. store [sync] mma
DB=1:  .. store [sync] issue(n+1) mma(n)  [sync] store(n+1) [sync] issue(n+2) mma(n+1) ..
```

so DRAM latency is exposed once per round no matter how large KC is. `DB=true` issues the next
round's loads into a second register array immediately after the store barrier, giving them the whole
mma phase to land. Arithmetic is untouched — same bytes, same smem, same `kblk` order into `acc` —
so the claim is bit-equality, and `gate_tc_fp8_kc` was extended to sweep `DB ∈ {0,1}` against the
KC=1/DB=0 reference (504 → 1134 cases, all exact).

### What actually happened: a 25 % occupancy loss bought with FOUR registers

`ptxas -v` first, per trap 19. Unbounded, DB takes the shipped `<8,2>` instantiation from **64 to 68
registers** — which looks like nothing, and is not: at 256 threads/block, 65536/(256·64) = **4
blocks/SM** and 65536/(256·72, the 8-register allocation granularity) = **3**. The first bench
(`evidence/db_bench.log`, COLD, `m16+smem B+4`, M=5, 3 alternating replicates each arm) priced that
step:

| shape (W) | wo_b (8) | sw2 (8) | wq_b (8) | wq_a (2) | sw1/3 (4) | wkv (1) |
|---|---|---|---|---|---|---|
| DB unbounded | **+38.5 %** | +20.7 % | +13.5 % | +1.2 % | −8.4 % | −3.7 % |
| DB, `__launch_bounds__` back to 4 blocks/SM | +3.1 % | −1.0 % | +2.4 % | −0.8 % | −7.2 % | −3.2 % |

(`evidence/db_bench2.log` for the second row.) **Restoring the occupancy recovered 35 points on
`wo_b`.** Every shape that regressed is a W=8 shape sitting exactly on the 4→3 step; the two that
gain — sw1/3 at W=4 and wkv at W=1 — were never near it. This is the cleanest measurement in the
project of how expensive a register is *near a step* and how nearly free it is away from one.

Getting the bound on was itself a trap worth recording: putting `__launch_bounds__(32*WARPS, 1)` on
the shared template took the **DB=false** `<8,2>` instantiation from 64 to **84** registers —
`minBlocks=1` tells ptxas one resident block is enough, so it stops economising — which would have
silently changed the *control arm of this very A/B*. The fix is that the body is now a `__device__`
function and the DB=false entry carries no attribute at all, verified back to 64/64/64/128 registers
with zero spill across `<8,2>/<4,2>/<2,2>/<1,8>`.

### The measurement that closed it

With occupancy restored the bench said "wash, slightly negative on the W=8 shapes", and in situ
agrees to within its own noise:

| | control | this | delta |
|---|---|---|---|
| spec tok/s | 21.68 | 21.62 | −0.3 % |
| base AR tok/s | 13.78 | 13.69 | −0.7 % |
| ksweep K=5 | 121.11 | 121.19 | +0.1 % |
| dprof TOTAL K=5 | 127.18 | 127.58 | +0.3 % |
| `o:wo_b` K=5 | 8.67 | 8.91 | **+2.8 %** |
| `q:wq_b` K=5 | 7.98 | 7.85 | −1.6 % |
| `q:wq_a` K=5 | 5.80 | 5.70 | −1.7 % |

The nine spec verifies pair 1:1 at identical K and identical accept counts, so they compare directly:

| verify | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | Σ |
|---|---|---|---|---|---|---|---|---|---|---|
| K, accepted | 2, 1/1 | 2, 1/1 | 4, 1/3 | 2, 1/1 | 4, 3/3 | 3, 2/2 | 5, 4/4 | 5, 1/4 | 4, 3/3 | |
| control ms | 112.5 | 117.8 | 141.8 | 119.5 | 144.3 | 131.4 | 154.7 | 154.9 | 142.5 | 1219.4 |
| this ms | 112.7 | 117.8 | 141.4 | 120.2 | 145.9 | 130.0 | 155.4 | 155.4 | 144.0 | 1222.8 |
| delta | +0.2 % | 0.0 % | −0.3 % | +0.6 % | +1.1 % | −1.1 % | +0.5 % | +0.3 % | +1.1 % | **+0.28 %** |

**`o:wo_b` is the control that decides it.** It is the one mark whose kernel this change was supposed
to speed up, the bench predicted +3.1 % for it after the occupancy fix, and it measured **+2.8 %** —
sign and magnitude agreeing across two independent instruments is not noise. `q:wq_b` and `q:wq_a`
move −1.6 %/−1.7 % against a same-run scatter that F76 bounded at ±2.4 % on this class of mark, and
`moe:w1w3` (−0.5 %) and `i:score` (−2.6 %), which this change cannot touch, sit in the same band.

### The mechanism, and what it says about the tile

The prefetch works — the loads do get the mma phase — but **the mma phase is too short to hide a DRAM
round trip at these shapes.** One round at KC=2 is 8 `mma.m16n8k32` plus 32 A-fragment loads that hit
L1/L2; that is a few hundred cycles against a ~600-cycle miss. So the change converts "wait the full
latency at the barrier" into "wait most of the latency at the barrier", and pays for the remainder
with a second live register array. At W=8 that array is on the wrong side of an occupancy step and
the trade is negative; at W=4/W=1 it is positive in the bench but those shapes are the shared expert
and `wkv`, both already **forked to a side stream and hidden** (F55), which is why nothing shows up
in situ. The deeper pipeline this kernel actually wants is `cp.async` staging global→shared with no
register array at all — a different change, not a tuning of this one, and one that must clear the
same occupancy bar it just failed.

### Disposition and what is left of B8

`TCB_DB=1` opts in; the default is the pre-change kernel, which is the arm every baseline in
`FLYWHEEL_STATE.json` was measured under. **B8's tile half is now closed**: F67 killed both reroutes,
F68 killed split-K on numerics, F74 took the win that was there, and this closes double-buffering.
What remains of B8 is the single untried idea on `o:wo_a` — the `OG_SMEM=1` variant reading `o4[m]`
lazily from shared memory — and this cycle sharpens its prior in *both* directions: `ogroup_gemv_mk`
spills 44 bytes at `<5,4>`, so it is genuinely register-starved and freeing ~16 could cross a real
step (unlike the 4 registers here, which crossed one by accident); but F78 is also the second cycle
running to show that the *bench* overstates what a register buys in situ.

---

## Finding 79 — B8' dies at `ptxas -v`, `o:wo_a` runs out of ideas, and a clean-vs-clean control says the cross-run floor is 1.5 %

**NEGATIVE, and it closes B8 entirely. Kept behind `OG_SMEM_LAZY=1`, default OFF.** The lever moved
`ogroup_gemv_mk_smem_kernel<5,4>` from **396 → 368 bytes of spill at an unchanged 64 registers**
(−7 %, where ~16 registers were predicted) and priced at **−1.1 %** against the `OG_SMEM` arm it
modifies while remaining **+37.8 %** against the kernel that actually ships. 19/19 gates PASS,
bit-identical at all 9 M in `gate_ogroup_gemv` (extended to sweep it). Clocks pinned
(`sudo jetson_clocks`), caches dropped. Bench `evidence/oglazy_bench.log`, gate
`evidence/oglazy_gate.log`, run `evidence/clean_post_f79.log`.

### The lever and its stated falsification

LEVERS.md B8' was the single untried idea left on `o:wo_a` (`ogroup_gemv_mk_kernel`, 11.4 ms at K=5,
the block's worst mark). The argument: `float4 o4[M]` in the smem-staged variant is 4M = **20 live
registers at M=5**, in a kernel `ptxas` shows 8 registers over its 64-register `__launch_bounds__`
cap; in the *non*-staged kernel that array is load-bearing (global loads, and hoisting them is the
MLP the phase needs), but under `OG_SMEM` they are smem reads at ~30 cycles with no MLP argument, so
they can be re-read per (r, m) instead of held. F55 measured the staged variant −40 % at NR=4 *with*
the 20 registers still in it, so the register argument had never been tested. The queue entry named
the falsification: **`nvcc -Xptxas -v` BEFORE building — if the spill does not fall, stop.**

### ptxas answered, and the answer was no

| `<5,4>` instantiation | registers | spill |
|---|---|---|
| `ogroup_gemv_mk_kernel` (SHIPPED, untouched control) | 64 | 44 B |
| `ogroup_gemv_mk_smem_kernel` LAZY=0 (F55) | 64 | 396 B |
| `ogroup_gemv_mk_smem_kernel` LAZY=1 (this) | 64 | **368 B** |

A 7 % move, not 16 registers. **The mechanism is that you cannot free a register by not writing it
down**: `sh` is never written inside that loop, so hoisting the `ld.shared` back to exactly where
`o4[M]` had been is both legal and the lower-latency schedule, and ptxas does it. That is trap 19
from the other direction — there the source *forced* two registers live and ptxas could not undo it;
here the source merely *suggests* fewer and ptxas ignores the suggestion. New trap 23.

The control arm is clean: the shipped kernel is byte-for-byte 64 registers / 44 bytes across
`<5,4>`, and the LAZY=0 instantiation is unchanged at 396 — no trap-22 contamination from adding the
template parameter.

### The bench confirmed it, on the arm it modifies

`gemm_bench` "ogroup wo_a COLD", 12 rotating copies, **3 alternating replicates per arm** (trap 5),
means in ms:

| M | dispatch NR | shipped | `OG_SMEM` | `+LAZY` | LAZY vs shipped |
|---|---|---|---|---|---|
| 2 | 2 | 0.1717 | 0.1533 | 0.1532 | −10.8 % |
| 3 | 2 | 0.1962 | 0.1729 | 0.1732 | −11.7 % |
| **5** | **4** | **0.1958** | **0.2729** | **0.2697** | **+37.8 %** |
| 8 | 2 | 0.3610 | 0.2810 | 0.2714 | −24.8 % |

Across every (M, NR) cell the **lazy-vs-smem** delta is inside ±3.4 %. The 20 registers were not the
binding cost of the staged kernel, so giving them back changes nothing — which retires B8', and with
it the whole of B8: F67 killed both reroutes, F68 killed split-K on numerics, F74 took the win that
was there, F78 closed double-buffering, and this closes `o:wo_a`.

### The side result: F55's 40 % kill was M=5-only, and the sign flips below M=4

Look at that table again along **M** rather than along NR. F55 swept NR at **M=5 only** and the
`OG_SMEM` note has read "40 % regression" ever since — and it reproduces exactly here (+39.4 % at
M=5/NR=4). But at M=2 and M=3 the staged kernel beats the shipped one by **10.8 % and 11.7 %**
against the NR the dispatch actually picks, and at M=8 by 24.8 %. Nothing about F55 was wrong; it was
one column of a two-axis table. New trap 24: **before re-deriving a retired lever's mechanism, check
which axes its killing measurement actually varied.**

Priced honestly it is still small, and it is logged as **B8'' at ~0.4 %, SUB-1 %, not counted toward
the pivot queue**: `o:wo_a` is 10.74 ms of an 85.9 ms K=2 verify and 10.26 of 104.7 at K=3
(`evidence/kchunk.log`), and only 4 of the 9 verifies in a canonical run are K≤3 → 4.68 ms of
1194.7 = 0.4 %, before the trap-3 discount. M=4 sits on the crossover and the bench skips it.

### The mandated clean re-baseline, which turned into the control this project lacked

`baseline.dprof_runs_since_clean` was 2, so step 6 forced a CLEAN run: **spec 22.07 tok/s, base AR
13.78 tok/s (72.6 ms/tok), acceptance 2.89, 1.60x, GATE PASS, LOSSLESS GATE PASS**, first token
11111, generated sequence identical to `clean_post_f76.log`.

**This is not a +1.4 % gain.** The lever shipped default OFF, `OG_SMEM` is default OFF, and ptxas
confirms the shipped kernel unchanged — so this run and `clean_post_f76.log` measure the *same
default arm*, and the nine verifies pair 1:1 at identical K and identical accept counts:

| verify | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | Σ |
|---|---|---|---|---|---|---|---|---|---|---|
| K, accepted | 2, 1/1 | 2, 1/1 | 4, 1/3 | 2, 1/1 | 4, 3/3 | 3, 2/2 | 5, 4/4 | 5, 1/4 | 4, 3/3 | |
| f76 ms | 109.2 | 115.5 | 141.3 | 117.1 | 141.8 | 127.5 | 155.6 | 155.8 | 148.6 | 1212.4 |
| f79 ms | 107.4 | 114.1 | 137.8 | 115.6 | 140.8 | 127.0 | 154.9 | 154.2 | 142.9 | 1194.7 |
| delta | −1.6 % | −1.2 % | −2.5 % | −1.3 % | −0.7 % | −0.4 % | −0.4 % | −1.0 % | −3.8 % | **−1.5 %** |

**Nine out of nine in the same direction, on an unchanged binary.** The "±0.8 % own spread" F76 and
F78 quoted for this instrument is the spread *within* a matched pair; the floor for the **cross-run**
comparison those findings actually performed is **~1.5 %, and it is not zero-mean**. F76 (+0.1 %) and
F78 (+0.28 %) were called null and stay null — this only strengthens both. F74's +6.1 % clears it 4x.
But no cross-run claim under ~1.5 % is supportable here, and that is the size of every remaining §4
item, including the B8'' found above. New trap 25.

### Disposition

`OG_SMEM_LAZY=1` opts in and requires `OG_SMEM=1`; the default is unchanged. **B8 and B8' are
retired**; B8'' is logged sub-1 %. `pivot_criterion.open_nontraining_levers` falls **2 → 1** (S6
alone: `B8-cpasync` stays uncounted as a different kernel, B1 and B8'' are sub-1 %, S5/S7 need
training, B5 needs the quantisation constraint relaxed, B9 is prefill). `consecutive_sub_half_pct`
goes **2 → 3**, which **fires the pivot criterion** — three consecutive cycles (F76 +0.1 %, F78
+0.28 %, F79 negative) adopting nothing ≥1 %. That is the criterion working, not failing: the kernel
queue is one item deep, its own measuring instrument cannot resolve what is left in it, and S5 — the
draft-head fine-tune, +24 % at acceptance 3.6, lossless by construction — is the lever that is
actually left.

---

## Finding 80 — S6 dies at its own falsification: the suffix drafter's ORACLE ceiling is exactly +0.0 %, because speculation hands a retrieval drafter its worst possible query

**NEGATIVE, and it retires the last non-training lever in the project.** The first RESEARCH phase this
loop has ever run (6 queries, axes A–F) promoted one lever and folded one refinement; the cycle's
measurement then priced **S6** — suffix-automaton / prompt-lookup drafting ahead of the MTP — from one
checkpoint load, **without building the cascade**, and killed it: over 21 verifies the suffix drafter
proposed **23 accepted-token-equivalents against the MTP's 61 (−62.3 %)**, beat the MTP in **0 of 21**
verifies, and the **oracle `max(MTP, suffix)` is 61 tokens = +0.0 %**. There is no gating rule that
recovers anything, because there is nothing to gate. 20/20 gates PASS, GATE PASS, MATCH 5/5, LOSSLESS
GATE PASS, clocks pinned (`sudo jetson_clocks`), caches dropped. Run `evidence/suffixprobe_f80.log`.

### Why this was measurable at all

F79 left the loop with an instrument whose cross-run floor is ~1.5 % (trap 25) and a queue in which
every remaining item was priced below it. S6's own LEVERS.md entry named the way out: *"the draft is
only ~13 ms of a 151 ms cycle, so the win is NOT the skipped head — it is any acceptance gained at the
same K. Price it as acceptance, not as draft time saved."* **Acceptance is a counted integer**, so it
has no variance floor. `DSV4_SUFFIXPROBE=1` computes, at every verify, what a suffix drafter would have
proposed from the committed sequence and how much of it the target would have accepted — beside what
the MTP actually got. Finding 67 killed a lever this way before it was built; this is the same move.

The probe is **read-only**: it touches no device buffer and no engine state. The control is exact —
the emitted sequence's **first 26 tokens are byte-identical to `clean_post_f79.log`**, and base AR is
**13.73 vs 13.78 tok/s (−0.36 %, inside the floor)**.

**Exactness, stated rather than assumed.** `tam[i]` is the target's argmax *given the MTP's* draft
prefix, so it is valid ground truth for a different draft only while that draft agrees — i.e. for
`i <= acc`. The probe therefore reports a **sound lower bound** and an optimistic estimate. Here they
**coincide at 23 tokens**, because the suffix drafter never once reached `acc`, so the ambiguity never
arises and no number below is an extrapolation.

### The result

| arm | tokens / 21 verifies | tok/verify | vs MTP |
|---|---|---|---|
| **MTP draft (shipped)** | **61** | **2.905** | — |
| suffix only (sound lower bound) | 23 | 1.095 | **−62.3 %** |
| suffix only (optimistic) | 23 | 1.095 | −62.3 % |
| **ORACLE `max(MTP, suffix)`** | **61** | **2.905** | **+0.0 %  ← S6 CEILING** |
| cascade, suffix if mlen≥1 | 52 | 2.476 | −14.8 % |
| cascade, suffix if mlen≥2 / ≥3 / ≥4 | 56 | 2.667 | −8.2 % |
| cascade, suffix if mlen≥6 | 61 | 2.905 | +0.0 % ← *never fires* |

**The best cascade is the one that never uses the suffix draft.** A match existed in only 8 of 21
verifies; the match-length histogram is `mlen = 0 ×13, 1 ×2, 3 ×1, 4 ×4, 5 ×1` and never reached the
block size, so the mlen≥6 row is "use the MTP always" wearing a threshold.

### The mechanism, and it is structural, not prompt-specific

**Speculative decoding hands a retrieval drafter its worst possible query.** The anchor at every
drafting point is `cur` — and `cur` is always the **correction** token, the one token the MTP just
mispredicted. Whenever the sequence is locally predictable the MTP accepts straight through it, so the
next draft never starts *inside* a predictable span; it starts on the surprise. The anchor is therefore
**selected to be the least repetitive token in the sequence**, and that is exactly the query a suffix
automaton is worst at. It shows up directly: in **13 of 21 verifies `mlen = 0`** — not even the current
token had ever occurred before, so there was no proposal to make at all.

The eight verifies where a match *did* exist show the other half of it. This model's generation on the
gate prompt is a period-8 degenerate loop, `16 455 6102 294 X 344 Y`, in which `X` and `Y` take a
**fresh value every cycle** (`16603/14251/29585/25062/10322` and `29168/16235/76405/40977/17575`). The
deterministic part of the repeat — `16 455 6102 294` — is precisely what the MTP already accepts (four
of the `acc_mtp=4` verifies are exactly that run). So a long match only ever puts the suffix drafter at
the varying slot, proposing the *previous* cycle's value:

| mlen | suffix proposed | target emitted | acc_sfx | acc_mtp |
|---|---|---|---|---|
| 4 | 16603 344 | **14251** 344 | 0 | 2 |
| 4 | 29585 344 | **25062** 344 | 0 | 0 |
| 4 | 10322 | **270** | 0 | 1 |
| 5 | 3702 14460 344 6693 | **17705** 4106 344 8703 | 0 | 0 |
| 1 | 6102 294 270 17705 | **87966** 16 455 6102 | 0 | 4 |

The suffix drafter has **strictly less information than the MTP** and is aimed at the one position the
surface repeat cannot predict. That is why the ceiling is not merely small but exactly zero.

**And this was the most favourable input available.** A period-8 repeating decode is the best case a
prompt-lookup drafter can be handed; on novel prose it can only do worse. An oracle ceiling of +0.0 %
with 0 wins in 21 on the best case is not a variance result. New trap 27.

### What would reopen it — and why that is not this engine

SuffixDecoding's 1.7×-over-Prompt-Lookup is reported on **SWE-Bench-style agentic** traces, where whole
spans (file contents, tool output) are copied verbatim and long enough that the anchor lands *inside* a
copied span instead of on a correction. That is a real regime and this measurement does not refute it —
it refutes S6 **for the workload this engine is measured on**, which is every number in LEVERS.md.
Reopening requires new evidence of the specific kind: a long-repeated-context prompt (the shape of
`scripts/prompt_suite.json`'s `longctx_001`, `filler_repeats: 400`) on which `mlen` routinely reaches
the block size. Its ids would have to come from `tools/encode_prompt.py`, and no such prompt exists in
the harness today. **S6 is retired until one does.**

### The research phase, executed for the first time

6 queries, one per axis, all appended to `RESEARCH_LOG.md` §2 with outcomes including the four that
found nothing. **Promoted: `B8-cpasync` → open at ~1–3 %** — not on a new argument but on new evidence
against F78's specific measurement: the consensus design is a *decoupled producer warp* filling a
**≥4-stage** smem ring, and the sources name F78's failure mode outright ("with only 2 stages, a small
delay in TMA completion or barrier arrival can stall tensor-core issue almost immediately"), which a
deeper *register* buffer in the same warp cannot fix. **Folded: UniSpec**'s n-gram confidence estimator
into S6 as its gating rule — now moot. **Rejected with reasons recorded:** the whole expert-prefetch
family (assumes an offload transfer we do not have), relaxed/"loosely" speculative decoding (violates
lossless — F68), MLA kernel work (this model is not classic MLA), Stream-K (not bit-exact — F68/B2),
persistent megakernel (bounded at 1.05× by F46).

### Numbers, and one that is deliberately not claimed

Gates: **20/20 PASS** (19 prior + the new `gate_suffix_draft`, 8/8 host checks). Four gates first read
red and all four were the harness passing `ref/goldens` to binaries that take argv[1] as a shape —
`gate_encoding`, `gate_compressed_decode`, `gate_indexer_decode`, `gate_prefill_len` all PASS on their
own defaults. That is the third recurrence of this artefact; **new trap 26**.

Run: base AR **13.73 tok/s (72.8 ms/tok)**, spec **21.69 tok/s**, acceptance **2.90**, 1.58×. **The
spec number is NOT comparable to the 22.07 baseline and is not written to it**: this run used
`NGEN0=60` (21 verifies) against the baseline's `NGEN0=24` (9 verifies), so it averages over a longer
and growing context — verify 20 costs 162.1 ms where the baseline's verify 9 cost 142.9 ms. The
comparable half is base AR, which is a clean control at −0.36 %. `baseline.dprof_runs_since_clean`
stays 0.

### Disposition

**S6 is RETIRED** (§3, speculation). `pivot_criterion.open_nontraining_levers` stays **1**, but the
occupant has changed: S6 is out and this cycle's research promotion `B8-cpasync` is in.
`consecutive_sub_half_pct` goes **3 → 4**. The pivot criterion fired at cycle 14 on the
three-cycles-without-a-1 %-adoption arm and **this cycle supplies the stronger, second reason**: the
pivot analysis named S6 as "the only qualifying item" left beside the training lever, and S6 now has a
**measured** ceiling of exactly zero rather than a projected one. S5 — the draft-head fine-tune,
lossless by construction, +24 % at acceptance 3.6 — is not merely the largest lever left; after F80 it
is the only one that has never been measured against.

---

## Finding 81 — B8-cpasync built and RETIRED: the cp.async ring hands back 16 registers and RAISES occupancy 4 → 5 blocks/SM, and is 15–53 % SLOWER. Depth is negative, and cp-size 16 does not save it

**NEGATIVE, and it empties the kernel queue.** The lever the cycle-15 research phase promoted — a
`cp.async` staging ring for the fp8 tile, ≥4 stages, replacing F78's 2-stage *register* double buffer —
was implemented, is **bit-exact** (`gate_tc_fp8_kc` **1512/1512**, +378 new cases), **passed both of its
cheap falsification steps**, and then lost the bench by 15–53 %. It ships **default OFF** behind
`TCB_CPA=<stages>` and the default arm is proven inert: this cycle's clean run emits a spec token
sequence **byte-identical** to `clean_post_f79.log`. 20/20 gates PASS, GATE PASS, MATCH 5/5, LOSSLESS
GATE PASS, clocks pinned (`sudo jetson_clocks`), caches dropped.

### The two cheap steps both said GO, which is why this was worth building

The falsification order in LEVERS.md said: stop at the first no. Steps (1) and (2) cost no checkpoint
load and neither was a no.

**(2) The ISA is there.** `cp.async.bulk.tensor.2d`, `cp.async.bulk`, `mbarrier.init`,
`mbarrier.try_wait.parity` and `cp.async.ca.shared.global` **all assemble at `-arch=sm_110a`**.

**(1) `ptxas -v`, and it is the opposite of F78's problem.** F78 failed on occupancy: its second live
`uint4` array took `<8,2>` from 64 to 68 registers, 4 blocks/SM to 3. The ring removes the staging
array outright, and the device has smem to spare — **233 472 B/SM**, of which the shipped kernel uses
17408 x 4 = 69 632, i.e. 30 %. Measured:

| kernel (the arm the engine runs) | regs | smem | blocks/SM |
|---|---|---|---|
| `smemB_kernel<8,2,false>` — **shipped** | 64 | 17408 | **4** |
| `cpa_kernel<8,4,false>` — 4-stage ring | **48** | 36864 | **5** |

So 16 registers back and **+25 % occupancy**. That took one extra recompile to find, and the knob is
worth recording because it inverts the result: **`cp.async` is fire-and-forget, so a ROLLED issue loop
still has every copy in flight, but an UNROLLED one keeps H addresses live** — trap 19, H times over.

| `TCB_CPA_UF` | regs | blocks/SM |
|---|---|---|
| 8 (full unroll) | 78 | **3** — F78's failure mode by a different door |
| 4 | 56 | 4 |
| 2 | 48 | 5 |
| **1 (default)** | **48** | **5** |

### And then it lost, at every shape

`gemm_bench` COLD, cp.async arms placed at sweep positions **adjacent** to the arm they must beat
(trap 5), 24 reps over ≥400 MB of rotating weight copies. M=5, ms, vs `m16+smem B+4`
(`evidence/cpasync_bench_f81.log`):

| shape | m16+smem B+4 | cpa NS=2 | cpa NS=4 | cpa NS=8 | NS=4 vs shipped |
|---|---|---|---|---|---|
| wq_a [1024,4096] | 0.0402 | 0.0503 | 0.0498 | 0.0490 | **+23.9 %** |
| wq_b [4096,1024] | 0.0307 | 0.0373 | 0.0425 | 0.0414 | **+38.4 %** |
| wkv [512,4096] | 0.0269 | 0.0434 | 0.0411 | 0.0449 | **+52.8 %** |
| wo_b [4096,4096] | 0.1066 | 0.1322 | 0.1491 | 0.1537 | **+39.9 %** |
| sw1/3 [2048,4096] | 0.0570 | 0.0654 | 0.0655 | 0.0722 | **+14.9 %** |
| sw2 [4096,2048] | 0.0562 | 0.0670 | 0.0766 | 0.0738 | **+36.3 %** |

The M=1 rows are a **within-bench control**: at M=1 the dispatcher uses the m1 GEMV, so the cpa columns
must equal the columns beside them, and they do (wq_a 0.0301 vs 0.0301). The comparison is sound.

**NS=2 beats NS=4 on 4 of 6 shapes.** Depth is *negative*, which is the direct refutation of the
promoted claim — the literature's "≥4 stages, because 2 stalls tensor-core issue immediately" is the
one thing this measurement says is backwards here.

### The disambiguation is the half that makes this permanent

`cp.async` requires cp-size == natural alignment, and F66 counted this engine's fp8 weights at
`data_offset % 16 == 8` (43,470 of 44,436) or 12 (966), **never 0** — the same fact that forced the
`AL16` template on the LDG path. So the ring above ran at **cp-size 4**: eight 4-byte DMAs where it
wanted two 16-byte ones. That is a confound, and leaving it would have left the next cycle re-arguing
"it only failed because of alignment". One more bench, no checkpoint
(`evidence/cpasync_bench_align_f81.log`), M=5, cpa NS=4 at **B+0** so `AL16=true` and cp-size 16, now
against `m16+smem` at **B+0** as well:

| shape | m16+smem (B+0) | cpa NS4 **cp-size 16** | cpa NS4 cp-size 4 | cp16 vs shipped |
|---|---|---|---|---|
| wq_a | 0.0369 | 0.0390 | 0.0518 | **+5.7 %** |
| wq_b | 0.0282 | 0.0315 | 0.0405 | **+11.7 %** |
| wkv | 0.0280 | 0.0311 | 0.0422 | **+11.1 %** |
| wo_b | 0.0844 | 0.1067 | 0.1428 | **+26.4 %** |
| sw1/3 | 0.0501 | 0.0486 | 0.0654 | **−3.0 %** ← the only win |
| sw2 | 0.0460 | 0.0550 | 0.0759 | **+19.6 %** |

**cp-size 4 costs ~25 %** (wo_b 0.1428 → 0.1067) — a real number, and new trap 31, since any future
`cp.async`/TMA idea here inherits that tax and must be priced at B+4. But **cp-size 16 still loses at
5 of 6 shapes**, so **no alignment work rescues the lever**. F67 had already measured the shard pad as
buying nothing; now it would not even buy this. That is what turns "blocked on alignment" into a close.

### Mechanism: it moved the one thing F74 moved, in the wrong direction

The ring stages **one** K-block per stage, so `cp.async.wait_group` + `__syncthreads` runs **once per
K-block** where the shipped KC=2 pays it once per two. **F74's adopted win was exactly the opposite
trade** — "stage KC consecutive K-blocks per barrier pair … the barrier count drops by KC", worth
−28.6 % on `q:wq_b`. A stage is 64 rows x 128 B = 8 KB spread over 256 threads; there is not enough
work per stage to amortise a barrier pair, so a deeper ring buys bytes-in-flight the LDG path already
had (F74's `issue`-then-store gets NH loads in flight per thread) at the cost of the thing that
actually paid. That is **new trap 29**, and it also explains why deeper is worse rather than better.

**And the occupancy win was real and did not help.** 48 registers vs 64, 5 blocks/SM vs 4, +25 % — the
precise bar F78 failed — and still 15–53 % slower. The shipped kernel at 4 blocks/SM already runs
`wo_b` at 199 GB/s against a 233–240 GB/s achievable; the 5th block adds contention, not bandwidth.
**New trap 30: an occupancy win still has to be measured as time.**

### Not run in situ, and that is the stated protocol

The falsification order said stop at the first no, and trap 3's discount has never turned a large bench
negative into a win — F76 went bench −7.6 % → in situ +0.1 %, F78 bench +3.1 % → in situ +2.8 %, i.e.
the bench understates *magnitude* but has never flipped *sign* on this kernel. A checkpoint load to
confirm a 15–53 % regression is the one thing the ledger exists to prevent.

### The run went to a better question, and it re-prices trap 25

The change is default OFF, so the cycle's ONE run (`baseline.dprof_runs_since_clean` was 0, so a clean
run was permitted rather than mandated) was spent on the control: **is the default arm still the
baseline arm, and what is the cross-run floor really?** `evidence/clean_post_f81.log`, CLEAN — no
`DSV4_DPROF`, no `DSV4_KSWEEP`, no `TCB_CPA`.

**Default-OFF is proven, not asserted.** The spec token sequence is **byte-identical** to
`clean_post_f79.log`; acceptance **2.89** unchanged; first token 11111; GATE PASS, MATCH 5/5, LOSSLESS
GATE PASS. `ptxas -v` independently shows the shipped `smemB_kernel<8,2,false>` still at **64 registers
/ 0 spill / 17408 B** — the new template is separate, so trap 22 (a shared template's other
instantiation moving) does not apply here.

**And this is the SECOND pair of identical-arm clean runs, which trap 25 badly needed.** Trap 25 set
the cross-run floor at ~1.5 % from **one** pair. The nine verifies pair 1:1 again at identical K and
identical accept counts:

| pair | paired-verify total | spec tok/s | verifies in one direction |
|---|---|---|---|
| f76 → f79 (trap 25's pair) | 1212.4 → 1194.7 = **−1.5 %** | 21.76 → 22.07 (+1.42 %) | **9/9** |
| **f79 → f81 (this pair)** | **1194.7 → 1196.7 = +0.17 %** | 22.07 → 22.06 (**−0.05 %**) | 2/9 |

Per-verify: +1.21, −0.09, −0.15, +0.17, +0.28, +0.16, +0.00, +0.13, +0.00 %. Base AR 13.78 → 13.83
(+0.36 %). **So the instrument is capable of ~0.2 % reproducibility, and the 1.5 % in trap 25 was an
occasional systematic shift, not a per-run noise band.** The practical rule sharpens rather than
loosens: a *single* cross-run comparison is still unsafe below ~1.5 % because you cannot tell which
kind of pair you drew — but a sub-1 % effect is resolvable with **two** pairs, where before the ledger
read as though nothing under 1.5 % could ever be measured. That matters, because everything left in §4
is sub-1 %.

### Numbers

Gates **20/20 PASS** (`evidence/gates_f81.log`). `gate_encoding` first read red and was trap 26 for the
**fourth** recurrence — it wants **no** argument; it PASSES 6/0 on its own default. `gate_tc_fp8_kc`
1512/1512 exact, extended this cycle with the `TCB_CPA ∈ {2,4,8}` arms at every M and both B offsets,
including a depth **larger than KB** so the empty-`commit_group` padding that keeps `wait_group`'s
depth a compile-time constant is exercised — without it the mma reads a stage that has not landed, and
that is a race a shape-poor sweep would miss (trap 9).

Run: spec **22.06 tok/s** (45.3 ms/tok), base AR **13.83 tok/s** (72.3 ms/tok), acceptance **2.89**,
**1.60x**, 26 tokens over 9 verifies.

### Disposition — the kernel queue is EMPTY

**B8-cpasync is RETIRED** (LEVERS.md §3, kernels; §4 row struck). It was the **only** open ≥1 %
non-training lever, so `pivot_criterion.open_nontraining_levers` goes **1 → 0**. B8'' (~0.4 %) and B1
(0.3–0.5 %) are explicitly sub-1 %; B5 needs the no-additional-quantisation constraint relaxed; S5 and
S7 need training; B9 is prefill; S3 is priced and rejected; S2 and S4 have no number. `ptxas -v` and
one bench closed the last named idea on the fp8 GEMM block, which F78 and F79 had already closed
elsewhere — **B8 is now closed in every direction**.

`consecutive_sub_half_pct` goes **4 → 5**. The three-cycles-without-a-1 %-adoption arm of the pivot
criterion has been fired since cycle 14; this is the fifth. The queue-empty arm requires 0 on **two
consecutive** audits and this is the first, so on the letter of the criterion the second arm is one
audit from firing — but the first arm fired two cycles ago and F80 and F81 have since measured the two
items the cycle-14 pivot analysis named as remaining (S6: oracle ceiling **+0.0 %**; B8-cpasync:
**+15–53 %** in the bench). **S5 — the draft-head fine-tune, lossless by construction, +24 % at
acceptance 3.6 — is now the only lever in the project with an unmeasured upside.**

---

## Finding 82 — the DRAFT half never got the arena the verify runs on: **10.19 ms/round, 7.4 % of the spec cycle, is host time inside raw `cudaMalloc`/`cudaFree` on an already-drained device**

**Cycle 17. POSITIVE — a lever, priced, not yet built.** The kernel queue was **0** at ORIENT, so
`RESEARCH_LOG.md` §1 made the research phase mandatory. It ran (seven queries, axes A–F, all written
back) and **promoted nothing** — but two independent 2026 sources name *host/launch-side overhead* as
the leak in batch-1 MoE decode, this project has measured that claim on its **verify** half (F46,
`VERIFYGRAPH` = 1.05x) and **never on its draft half**, and the draft half's only attribution in the
whole project is `evidence/f47.log`, from **before every verify-side adoption**. Reading the draft
path for that reason found this.

### What the code says, before any measurement

`include/dscratch.h`'s own opening line has said it since F44: *"At M=1 the per-call
cudaMalloc/cudaFree/cudaStreamSynchronize in every sub-function dominate."* The whole verify path is
on that arena. The DSpark **draft** path is not, and never was:

| draft function | file | raw allocs | raw sync |
|---|---|---|---|
| `dspark_main_kv` x3 stages | `dspark_attn.cu:13` | 2 malloc + 2 free each | 1 each |
| `dspark_attn_forward` x3 | `dspark_attn.cu:24` | 14 malloc + 14 free each | 1 each |
| `dspark_block_forward` x3 | `dspark_attn.cu:70` | 5 malloc + 5 free each | 1 each |
| `dspark_forward_head` x1 | `dspark_real.cu:105` | 7 malloc + 7 free | — |
| `dspark_main_x` x1 (commit) | `dspark_real.cu:12` | 2 malloc + 2 free | 1 |

Predicted ~140 malloc/free and 9–10 syncs per verify round. **Measured 134 and 9.** Every one of
these functions calls `cudaStreamSynchronize` and *then* frees, so the frees run on a device that has
already been drained — which is what makes their cost attributable rather than arguable.

### The instrument (this cycle's one change, default-invisible)

`rmalloc`/`rfree`/`rsync` in `dscratch.h`: drop-in wrappers that accumulate **host** time inside the
driver call, two `steady_clock` reads (~25 ns) against a call that costs microseconds. Applied to the
three draft files only. Counters print **only** under `DSV4_SPECPROF`, so the shipped path is
unchanged in behaviour; `g_raw_n` is a **counted integer** and immune to trap 25. The stale
`fwd_head` label ("host AR loop") was also corrected — that loop went device-side at F27.

### The numbers (`evidence/specprof_f82.log`, clocks pinned, caches dropped, ONE run)

```
[specprof] per verify round, mean of 8 (ms):
[specprof]   draft: main_kv     0.25  ( 0.2%)
[specprof]   draft: 3 blocks   13.47  ( 9.8%)
[specprof]   draft: fwd_head   10.13  ( 7.4%)  <- DEVICE-side AR over 5 positions (F27)
[specprof]   verify 43 layer  113.50  (82.6%)
[specprof]   TOTAL            137.35
[specprof]   cudaMalloc+Free   10.19  ( 7.4% of round)   134.0 calls/round
[specprof]   cudaStreamSync    12.99  ( 9.5% of round)     9.0 calls/round
[specprof]   draft raw TOTAL   23.18  (16.9% of round, 97.2% of the draft half)
[specprof]   rest-of-round      0.03 malloc/free + 0.38 sync  (4.0 + 1.0 calls)
```

**The draft half is 23.85 ms of a 137.35 ms round = 17.4 %**, not the 14 % LEVERS.md §1 has carried
since F47. It grew as a *share* because the verify shrank under F64/F65/F70/F71/F72/F74 while the
draft was never touched.

**10.19 ms — 7.4 % of the entire spec cycle — is host time inside `cudaMalloc`/`cudaFree`**, 134 calls
at a mean of **76 µs each**. 127 of those 134 run after their function's own
`cudaStreamSynchronize`, i.e. **on a drained, idle GPU**; only `dspark_forward_head`'s 7 frees (5 %)
could be absorbing device wait. So **≥ 7.0 % of the cycle is provably GPU-idle allocator time.**

**What is NOT claimed:** the 12.99 ms of `cudaStreamSynchronize` is 9 calls at 1.44 ms and is mostly
the host *waiting for real draft GPU work*. It is not waste and this finding does not price it as
waste. What it does show is that the draft runs as **ten serialised drain-points**, where the verify
runs as one asynchronous chain — and those syncs exist **because** the raw allocator requires a drain
before free. They go away with the arena as a consequence, not as a separate claim.

### Control — the default arm is untouched, and the instrument's own cost is measured

Spec token sequence and base-AR sequence are both **byte-identical** to `clean_post_f81.log`;
acceptance **2.89**; first token 11111; GATE PASS, MATCH, LOSSLESS GATE PASS. The nine verifies pair
1:1 at identical K: **1196.7 → 1208.8 ms = +1.01 %, 9/9 in one direction** (+1.29/+0.79/+1.53/+0.35/
+0.78/+0.16/+0.39/+0.39/+3.36 %), spec 22.06 → **21.84** (−1.0 %), base 13.83 → **13.72** (−0.8 %).
That is **the `DSV4_SPECPROF` instrument's own cost** — five `cudaEventRecord` on the null stream per
round — with a known cause and a consistent sign, so **this run is a PROFILING run and the baseline
is NOT updated from it.** `baseline.dprof_runs_since_clean` → **1**.

**New trap 32:** `scripts/flywheel.sh` classifies a run clean/profiling by grepping `^\[dprof\]`. A
`DSV4_SPECPROF` run prints `[specprof]` and is invisible to that grep, so the harness would score
this contaminated run as a clean re-baseline. The counter is set by hand this cycle and the gap is
recorded so the next cycle does not inherit a 21.84 baseline that no clean run ever produced.

### Gates

**20/20 PASS** (`evidence/gates_f82.log`), every one with a verdict line — including `gate_encoding`,
run with **no argument**, which is trap 26 for the fifth time and this time it was avoided rather than
diagnosed. `gate_tc_fp8_kc` 1512/1512 exact.

### Disposition — the queue is NOT empty

**New lever B10** (LEVERS.md §4): *put the DSpark draft path on the existing arena.* It is the same
change that took base AR 92.5 → 79.3 ms/tok at F44, applied to the one region that never got it, and
the target is measured rather than modelled: **7.4 % of the cycle in the allocator, plus whatever
share of the ten forced drains is launch gap rather than work.** Expected **+5 to +7 %**, which is
larger than anything the project has had open since F74. `pivot_criterion.open_nontraining_levers`
goes **0 → 1**, so the queue-empty arm of the pivot does **not** fire a second time.

**Two rules this cycle earns.** (1) *A region's SHARE goes stale even when its absolute cost does
not.* The draft was 14 % at F47 and is 17.4 % now with nobody having touched it; §1's table was a
verify table calling itself a cycle table. (2) *An empty research phase is not a wasted one if it
tells you where to read.* The seven queries promoted nothing — but "batch-1 decode leaks on the host
side" pointed at the one half of our cycle that had never been checked for it, and the answer was a
7.4 % lever sitting in five functions that predate the arena.

---

## Finding 83 — B10 built: the draft path is on the arena, **134 `cudaMalloc`/`cudaFree` and 10 syncs per round are gone, the output is bit-identical — and it bought 0.50 %, not 7.4 %**

**Cycle 18. The lever SHIPS and its expected value is RETIRED.** F82 priced B10 at +5 to +7 % from a
measured 10.19 ms/round of host time inside the DSpark draft path's raw allocator. It was built
exactly as the falsification order specified. **The predicted saving was 10.19 ms/round; the measured
saving is 0.67 ms/round — 6.5 % of the target, a 15x miss.**

### What was built

`dkmalloc` / `dkfree` / `dksync` in `include/dscratch.h` — a switchable seam over the F44 arena.
Default = arena (`dmalloc` bumps, `dfree` is a no-op, `dksync` degrades to `dsync`'s Finding-53
last-error read). `DSV4_DRAFT_RAW=1` restores the pre-F83 raw path **exactly**, including F82's
`rmalloc`/`rfree`/`rsync` instrument, so the A/B is one env var. Applied to all five draft functions:
`dspark_main_kv`, `dspark_attn_forward`, `dspark_block_forward` (`kernels/dspark_attn.cu`),
`dspark_main_x`, `dspark_forward_head` (`kernels/dspark_real.cu`).

The lifetime argument, which is the way this change could have been silently wrong (falsification
step 1): **every `dkmalloc` is function-local scratch consumed inside one call, and no `arena_reset()`
runs inside any of these functions.** The buffers that cross a reset — `mkv[st]`, `xa`/`xb`/`xemb`/
`dout`/`dmarg`, `main_x` — are all raw `cudaMalloc` at `decode.cu:646` and stay that way. Dropping
the ten per-function syncs is legal because `decode.cu`'s `for(pass)` loop does a blocking
`cudaMemcpy` **and** an explicit `cudaDeviceSynchronize` after its `arena_reset()` and before the
first `dkmalloc` of the pass, so reset memory is drained before it is re-issued; the chain's ordering
never depended on those syncs, it comes from being one stream. `dspark_attn_forward`'s H2D of the
stack-local `hidx` stays correct without its sync because `cudaMemcpyAsync` out of **pageable** host
memory is specified to copy into the staging buffer before returning. When `arena_init` was never
called (the `forward` gate-2-real binary, unit gates) `g_arena_on` is false and `dmalloc`/`dfree`/
`dsync` fall back to raw — zero behaviour change, which is why no gate needed touching.

### Capacity (falsification step 2) — the estimate was 40x low and it still fits

```
[arena] first draft: draft footprint 210.72 MB, global hwm 210.72 MB / cap 512.00 MB (41.2%), draft path = ARENA
```

B10 predicted "~5 MB/round". The real number is **210.72 MB**: with `dfree` a no-op and only one
`arena_reset()` per pass, the draft accumulates the *whole* 3-stage chain — and each stage's MoE,
`hc_pre`/`hc_post` and attention scratch were already `dmalloc` (they are shared with the verify
path), so they now stack 3x instead of being recycled. It fits at 41.2 % of a 512 MB arena, and the
**draft is now the arena's global high-water mark** — it exceeds the 43-layer prefill at PSp=5. New
capacity fact for anyone adding a 4th MTP stage or widening BLK.

### The result (`evidence/clean_post_f83.log`, CLEAN — no DPROF, no KSWEEP, no SPECPROF; clocks pinned with `jetson_clocks`, caches dropped; ONE run)

**Bit-identical, checked rather than assumed.** `diff` over the spec token sequence, the base-AR
token sequence and all 45 drafter margins against `clean_post_f81.log` is **empty**. Every verify
pairs 1:1 at identical K and identical accept counts. Acceptance **2.89**, first token 11111,
GATE PASS, MATCH 5/5, LOSSLESS GATE PASS.

| | F81 (raw allocator) | F83 (arena) | Δ |
|---|---|---|---|
| spec | 45.3 ms/tok = **22.06** tok/s | 45.1 = **22.15** | **+0.41 %** |
| base AR (no draft path) | 72.3 ms/tok = **13.83** | 72.6 = **13.78** | −0.36 % |
| 9 paired verifies, sum | **1196.7 ms** | **1190.7 ms** | **−0.50 %, 9/9 one direction** |
| acceptance | 2.89 | 2.89 | 0 |

per-verify: **−1.29 / −0.09 / −0.15 / −0.26 / −0.42 / −0.94 / −0.84 / −0.32 / −0.28 %**.

**The three draft-free control marks all moved the OTHER way** — cold prefill 206.1 → 206.9 ms
(+0.4 %), base AR 72.3 → 72.6 (+0.4 %), warm prefill 137.1 → 137.9 (+0.6 %). So the pair drawn was a
marginally *slow* one and the nine draft-containing marks still went 9/9 faster. The direction is
real; the magnitude is 0.50 % and, per trap 25, a single cross-run pair cannot resolve that on its
own. **Which does not matter, because the kill number needs no timing resolution at all:**

> **F82 priced 10.19 ms/round. All of it was removed. The round got 0.67 ms/round faster.
> 93.5 % of the priced time was never on the device timeline.**

### Mechanism — trap 34

F82's measurement was correct and its inference was not. It bracketed each driver call with
`steady_clock` and, seeing that 127 of 134 calls are issued after their own function's
`cudaStreamSynchronize`, concluded the device was idle for their duration. But "the device was
drained when this call *started*" does not imply "the device was idle *throughout*": each function
issues its whole malloc batch at the **top**, and the kernels of the previous function — and, inside
`dspark_block_forward`, of `hc_pre` and `rmsnorm` before the nested `dspark_attn_forward`'s 13
mallocs — are still draining underneath. Only the frees genuinely follow a drain, and they are at
most half the 134. **Host time inside a driver call is not device-timeline time. The instrument that
would have answered this before the build is the gap between two device events, not a host bracket.**
F82 declined to price its 12.99 ms of `cudaStreamSynchronize` as waste for exactly this reason; the
error was applying that instinct to only half the data.

**Left open, one profiling run and a two-line instrument change:** `g_raw_ms` lumps malloc and free.
Splitting it says which half carried the 76 µs mean, and whether any of the free time was absorbing
side-stream (`g_side`) MoE work — `cudaStreamSynchronize(0)` does not wait for a non-blocking side
stream but `cudaFree`'s implicit device sync does. That is a question about F82's number, not a lever.

### Disposition

**KEPT, default ON.** It is bit-identical, never worse (9/9), deletes 134 driver calls and 10 device
drains per round, and makes the draft path structurally match the verify path; `DSV4_DRAFT_RAW=1`
keeps the old arm one env var away. **But it is a sub-1 % item, not a 5–7 % one**, so
`pivot_criterion.open_nontraining_levers` goes **1 → 0**: B8'' (~0.4 %) and B1 (0.3–0.5 %) are
explicitly sub-1 %, B5 needs the no-additional-quantisation constraint relaxed, S5/S7 need training,
B9 is prefill. Cycle 17 reset the queue-empty counter, so this is the **first** of the two consecutive
audits at 0. The other arm — three consecutive cycles adopting nothing ≥1 % — has been fired since
cycle 14 and this cycle does not clear it.

### Gates

**20/20 PASS** (`evidence/gates_f83.log`), every one with a verdict line, `gate_encoding` run with no
argument. `gate_tc_fp8_kc` 1512/1512 exact. Note honestly what they do and do not cover: **no gate
exercises the draft path under an arena** — the only binary that calls `dspark_*` outside `decode` is
`forward`'s gate-2-real, which never calls `arena_init`, so it takes the raw fallback. The gate that
actually validated B10 is the byte-identical spec sequence in the model run, which is what the
falsification order said it would be.

---

## Finding 84-HALT — HALT. The operator set the kill switch at 12:29:25 while this cycle was already inside its 20-minute preflight, and the pivot the loop was about to declare is suppressed by that same file

**Cycle 19 (the cron script's "cycle 18"). NO lever was picked, NO code was changed, NO model run
was taken.** `DERIVED-ONLY:` this finding contains no performance number of its own — every number in
it is a filesystem birth time, a process start time or a cron-log line, all read this cycle. Nothing
was measured because nothing should have been.

### The kill switch

```
STOP birth : 2026-08-08 12:29:25.761773288 -0400   (stat: Birth == Modify == Change, size 0, untracked)
cron tick  : Sat Aug  8 12:17:00 2026              (ps -o lstart= -p 1634197, still alive)
cycle start: [flywheel 2026-08-08T12:37:16-04:00] cycle 18 starting; 114 GiB free; HEAD 635fefe
```

`scripts/flywheel.sh:61` — `[ -f "$ROOT/FLYWHEEL_STOP" ] && exit 0` — is the *first* preflight check,
and it ran at ~12:17:00, **twelve minutes before the file existed**. Between that check and the agent
launch the script does `drop_caches`, the free-memory reclaim and `flywheel_selftest.sh`; that took
20 minutes, and the operator hit stop in the middle of it. **The kill switch was not ignored, it was
outrun.** The `Birth` timestamp rules out a `touch -d`/`-r` artifact, `git ls-tree HEAD/origin/main`
shows it untracked, the reflog shows no checkout or reset that could have restored it, and
`grep -rn FLYWHEEL_STOP scripts/` finds only the three readers (`flywheel.sh:61`,
`flywheel_audit.sh:25`, `flywheel_observe.sh:18`) and zero writers. An external actor created it.

It was created **4 minutes 28 seconds after `2ece9f9`** — the byte-floor commit whose last line is
*"Two unrelated methods now agree the kernel path is finished, and the remaining 1.4x lives entirely
in acceptance."* Stop-after-that-sentence is a coherent instruction, and it is the instruction the
pivot criterion was written to earn.

### The thing worth recording: the stop file also suppresses the PIVOT

`open_nontraining_levers` is **0** in `FLYWHEEL_STATE.json` and `.flywheel_openprev` is **0** (written
by the 12:31:27 audit, committed as `635fefe`). That is exactly `OPEN=0 && PREV=0 && ! -f
FLYWHEEL_PIVOT` — the condition at `flywheel_audit.sh:42`. **The next auditor run is the one that
declares the PIVOT.** It will never reach it: the `FLYWHEEL_STOP` early-exit is at line **25**, the
pivot block at line **42**. `FLYWHEEL_PIVOT` is absent and `FLYWHEEL_AUDIT.md` has no banner.

So the loop's own exhaustion declaration — two consecutive audits at an empty non-training queue,
the outcome the criterion calls "the correct outcome, not a failure" — became due in the same minute
it was made unreachable. **This is not fixed here.** Reordering the auditor is a harness change, and
a stopped loop is the wrong place to make one; a cycle that "fixes" its way past a kill switch is the
failure mode the switch exists to prevent. It is written down so the operator can decide, and the fix
if wanted is one line: move the pivot block above the STOP check, or run it as
`bash scripts/flywheel_audit.sh` once with `FLYWHEEL_STOP` moved aside.

### Queue count (phase 6b), counted honestly and unchanged at 0

`B8''` ~0.4 %, `B1` 0.3–1 %, `B3` ~0.5 % — all explicitly sub-1 %. `B7`/`B7'` exhausted in both
knobs. `B0`, `B2`, `B4`, `B8`, `B8'`, `B8-cpasync`, `S1`, `S6` retired with numbers. `B10` demoted to
≤0.5 % by its own measurement (F83). `B5` needs the *no additional quantisation* constraint relaxed.
`B9` is prefill, not decode. `S3` is priced and rejected on arithmetic; `S2` has no idea attached and
`S4` is bounded small by the measured 17.53 union. `S5` and `S7` are training jobs. **Zero open,
≥1 %, non-training, no constraint relaxed.** No item was inflated to keep the loop alive.

### What was deliberately NOT done

- **No research phase.** `RESEARCH_LOG.md §1` makes one mandatory at queue 0, and it would have been
  the second consecutive empty phase that defines "done". Running eight searches against a set kill
  switch is continuing the loop, not orienting inside it.
- **`halt: true` was NOT set in `FLYWHEEL_STATE.json`.** The operator chose the file switch, which
  `flywheel.sh:61` already honours ahead of the `halt` check at line 63. A second latch adds nothing
  while STOP is present and silently blocks the resume when it is removed. The halt is recorded here
  and in `halt_considered`; the switch stays the operator's.
- **No model run, no gates, no build.** Clocks were not pinned and caches were not dropped, because
  nothing was measured — stated explicitly so no number in this cycle can be mistaken for one.

**The disposition of the loop: stopped at `spec 22.15 tok/s / base 13.78` (`evidence/clean_post_f83.log`),
queue 0, pivot due and suppressed, remaining lever S5 — the draft-head fine-tune, lossless by
construction, and not a kernel change.**

> **Numbering note.** Cycle 19 wrote its own halt record as "Finding 84" while my kill switch was
> landing, and I then numbered this one 84 as well without checking. The halt record is a
> `DERIVED-ONLY:` entry with no measurement in it and is referenced nowhere; these operator findings
> 84-87 are cited by four commit messages and by `LEVERS.md`. So the halt record is relabelled
> **84-HALT** and the performance numbering is left alone. Renumbering the cited ones to fix a
> cosmetic collision would have broken the references that make the log worth keeping.

## Finding 84 — prefill attributed to 99.98 %: it is TWO equal halves, and the MoE moves **3.26x the bytes it needs to** — measured, not modelled

**B9, operator-run (the flywheel is stopped).** F75 measured prefill end-to-end at 48 tok/s and stopped.
Three runs later it is fully attributed. `evidence/prefill_dprof_b9d.log`, PS=1022, clocks pinned,
caches dropped, prompt = 1023 real ids from the checkpoint's own tokenizer (`inference/model.py`
text, self-gated on `The capital of France is` -> 671,6102,294,8760,344).

**The instrument is free.** 21351.6 ms with full marks + the MoE byte counter, against 21362.4 ms
clean = **-0.05 %**. So the composition below is trustworthy, which is not a given: `DSV4_MOEUNION`
adds a `cudaStreamSynchronize` + D2H per MoE call and drains the pipeline 43 times.

### Where the 21.35 s goes

| region | ms | % |
|---|---|---|
| **MoE** | 9100.0 | **42.6** |
| **ATTENTION** (`compressed_attn_forward`) | 9086.0 | **42.6** |
| **KV cache population** (`compressed_attn_cache_r4`) | 2233.5 | **10.5** |
| `hc_pre` (attn+ffn) | 805.9 | 3.8 |
| everything else | 121.5 | 0.6 |
| **SUM** | **21346.9** | **99.98** |

| sub-op | ms | % |
|---|---|---|
| `moe:w1w3` | 5503.5 | 25.8 |
| `cattn:sparse` | 2926.8 | 13.7 |
| `moe:w2` | 2867.8 | 13.4 |
| `cattn:ogroup` | 2847.4 | 13.3 |
| `cattn:indexer` | 2083.0 | 9.8 |
| `cattn:q_proj` | 801.5 | 3.8 |

### The measurement that matters: 3.26x byte redundancy

`tc_build_tiles` emits `tile_row0 = r0 + 16*j` — a tile is **16 rows**, and every tile re-reads its
expert's entire weight matrix. At bs=1 that is invisible (one row, one tile, redundancy 1.0), which
is exactly why seventeen cycles of decode optimisation never surfaced it.

```
[moebytes] PS=1022  calls 43  union/call 141.2  rows/call 6132.0  tiles/call 460.7
[moebytes] expert weight traffic: ACTUAL 264.8 GB vs IDEAL 81.2 GB  -> redundancy 3.26x
```

`rows/call 6132.0` = 1022 x top-6 exactly, so the counter is reading the real routing. 460.7 tiles
against 141.2 experts touched is the redundancy, and it is **measured, not inferred**.

**264.8 GB in 9.100 s = 29.1 GB/s = 12.5 % of the 233 GB/s roofline.** So the MoE is doing BOTH
things wrong at once: moving 3.26x the necessary bytes, *and* moving them at an eighth of roofline.
Fixing only the redundancy, at today's poor rate, is 9.10 s -> 2.79 s.

### The attention half is not bandwidth at all — it is compute, and it is idle

- `cattn:sparse`  2.10 TFLOP / 2.927 s = **0.72 TFLOPS**
- `cattn:ogroup`  5.62 TFLOP / 2.847 s = **1.98 TFLOPS**

At 1022 positions we are 5-10x past the compute/bandwidth crossover, so these should be compute-bound
and near peak. They are running at single-digit-percent of any plausible peak. **Caveat stated
because it is load-bearing: this project has NEVER measured this box's compute peak** — every
roofline in the ledger is a bandwidth roofline from `tools/bw_probe.cu`. "0.72 TFLOPS is bad" rests
on a spec-sheet-class assumption, and closing that with a `gemm_bench` at prefill shapes should
precede any target-setting.

### Three process notes

1. **The first B9 run was wasted and the reason generalises.** `dprof_init()` was called only inside
   the `DSV4_KSWEEP` branch, which runs *after* the prefill, so `g_dprof_on` was false throughout and
   every mark was silently dropped — a 20-minute checkpoint load producing a log with no `[dprof]`
   line. That is the cheap failure. The expensive one is a report that looks complete because the
   marks it is missing never announce themselves.
2. **The second run's byte count was contaminated and I nearly published it.** The union/tile
   counters are cumulative and were never reset, so they swept up the PS=5 prefill, the first-token
   gate and 344 warm-decode calls at bs=1 — all redundancy 1.0 by construction, all diluting the
   average. The printed 2.27x was a lower bound, not the measurement. It was caught by arithmetic,
   not by the instrument: backing out the prefill-only union gave 173 experts against a hard maximum
   of 160. **An impossible intermediate is worth more than a plausible one.**
3. **The report's `*** INVALID: attn children > ATTENTION` is an artifact of MY id choice, not an
   inconsistency.** `DP_C_COMPRESS` was assigned to the KV cache population, which lives *outside*
   `DP_ATTN` in `cblock_prefill_cache`; the checker assumes every `cattn:*` is a child of ATTENTION.
   The 99.98 % accounting above is what settles it. Left as-is rather than silenced — a self-check
   that cries wolf is still the reason the first run's missing marks were noticed.

### What this makes B9 worth

Two named, independent fixes: (a) larger MoE tiles so an expert's weights are read once across all
its rows, (b) prefill-shaped attention kernels instead of the M=1-shaped ones inherited from decode.
Conservatively 21.35 s -> ~5 s is **4x**, which turns the S5 capture arithmetic from 4.9 days at 20K
samples into roughly one. **Prefill was never a decode lever and still is not; it is a multiplier on
the fine-tune's cost, which is why it goes first.**

## Finding 85 — the prefill MoE moves **11.26x** the bytes it needs, the tile count said 3.26x and was wrong, and `MOE_MMA=1` buys **+16.7 % prefill** for **-16 % decode**

**B9, operator-run.** `evidence/prefill_moemma_b9e.log`, two identical PS=1022 sweep points so the
in-place repack lands outside the measurement; the points agree to **0.02 %**.

### The correction: tiles are not the traffic

F84 reported 3.26x redundancy from a tile count and called the MoE latency-bound at 12.5 % of
roofline. **Both were wrong, and for the same reason.** In `k_grouped_fp4_gemv_e8m0` the weight load
sits INSIDE the `rb` loop, so one 16-row tile costs `ceil(me/RB)` full weight reads, not one.
Traffic is a function of **RB**, and the tile row cap does not enter it — splitting 43 rows into
16-row or 64-row tiles gives identical bytes at fixed RB. **The fix F84 recommended ("larger tiles")
would have moved nothing.**

Counter rewritten to count RB-chunks, measured on the same run:

```
tiles 19779   RB-chunks 68332   -> redundancy 11.26x   (tiles-only would say 3.26x and be wrong)
ACTUAL 913.6 GB vs IDEAL 81.2 GB
```

Hand-computed 892.9 GB against measured 913.6 GB — 2.3 %. At `RB=4` (what any batch >2 rows gets,
`rows_hint<=2 ? 2 : 4`) an expert serving ~43 rows costs **11 weight reads**. So the prefill MoE is
**bandwidth-bound at ~42 % of roofline moving 11x the necessary bytes** — not latency-bound.

### Why RB is not the lever either

`acc[RB][BN]` is live regardless of real row count (the F28 occupancy trap). F64/F70 measured the
wall: RB=8 costs 85 registers / 39.4 % occupancy and LOSES in situ. Amortising 43 rows needs RB~43,
which this kernel cannot hold. **The structural answer is the mma path**, which tiles through shared
memory instead of holding accumulators per row.

### Measured A/B

| | GEMV (default) | `MOE_MMA=1` | |
|---|---|---|---|
| prefill | 21351.6 ms / 47.9 tok/s | **18300 ms / 55.9 tok/s** | **+16.7 %** |
| MoE | 9100.0 ms | **6078.8 ms** | **-33.2 %** |
| ATTENTION | 9086.0 ms | 9063.6 ms | unchanged (other path) |
| base AR decode | 13.78 tok/s | 11.56 tok/s | **-16.1 %** |
| spec decode | 22.15 tok/s | 17.94 tok/s | **-19 %** |

**The prediction was too optimistic and the miss is informative.** I predicted MoE 9.10 -> 1-3 s; it
landed at 6.08 s. At 31 TFLOP of expert work that is **5.1 TFLOPS**, and against the 81.2 GB ideal
traffic only 13.4 GB/s — so under mma the MoE has stopped being a byte problem and become a
compute/occupancy one. Real headroom remains; it is a different lever.

### Deployment: two workloads, two flags, no code

`tc_ensure_repacked` mutates weights **in place** and the GEMV requires the ORIGINAL layout, so the
two paths cannot coexist in one process — this is a process-wide choice, not a per-call one. That
maps exactly onto what we need: **capture is teacher-forced prefill, so `MOE_MMA=1` is nearly free
there** (the decode regression does not apply), while the serving engine keeps GEMV. The deeper fix
— a GEMV variant reading the repacked layout so one process can do both — is real work and belongs
behind the attention half.

**Caveat on the printed bytes:** the counter mirrors the GEMV RB rule, so under `MOE_MMA=1` the
913.6 GB figure is the GEMV *counterfactual*, not this run's traffic. It is the clean measurement of
GEMV redundancy; it does not describe mma.

### Two observations for later

1. **ATTENTION is now 56.4 % of prefill at 9.06 s** and is untouched by any of this. Next target.
2. **Acceptance on the 1023-token prompt was 3.57-4.00 vs 2.89 on the canonical 6-token prompt.**
   Different prompt, highly predictable content (source code), so NOT like-for-like — but if long
   real prompts genuinely accept better it shifts the S5 expectation, and it is cheap to test.

## Finding 86 — `sparse_attn` was carrying 42 dead registers and one warp per block: **-31 % on the mark, +5.4 % prefill, and decode got faster too**

**B9, operator-run.** `evidence/prefill_sparse_b9f.log`, `MOE_MMA=1` held constant so the delta
isolates the attention change; two PS=1022 points agreeing to 0.02 %.

### Two defects, both bit-exact to fix, both invisible at m=1

1. **42 dead registers.** `qreg[32]`/`acc[32]` were sized for the kernel's stated contract `d<=1024`.
   All **ten** call sites in this engine pass `d = HEAD_DIM = 512`, so `per = 16` and half of both
   arrays never held anything. `ptxas -v`: **128 -> 86 registers, 0 spills either way** — occupancy
   ceiling 16 -> ~23 warps/SM. The generality was real but unused, and at m=1 decode occupancy is not
   the binding constraint, so it cost nothing visible until prefill pushed 65,408 warps through it.
2. **One warp per block.** `num_key_value_heads == 1`, so `kv` has NO head dimension and all h=64
   heads of a query read the IDENTICAL key vectors — as 64 separate 32-thread blocks, free to land on
   64 different SMs, each re-reading the same 2 KB from L2. HPB heads of one query in ONE block puts
   them on ONE SM, where reads 2..HPB hit L1.

Per-(query,head) accumulation order over `t` is untouched, so this is **bit-exact**:
`gate_compressed_decode` prefills at s=256 (total 16384 -> the HPB=8 path) and returns
`rms=0.00e+00`, and `gate_prefill_len` reports 0 prefix mismatches.

### HPB must follow the batch — the same trap RB fell into twice

At m=1 decode there are only `b*m*h = 64` warps in the whole launch, so a constant HPB=8 would put 8
blocks on 20 SMs and starve the machine, converting a prefill win into a decode regression. HPB is
sized to keep >=4 blocks/SM: **1 at decode** (original launch shape), 4 at a K=5 verify, **8 at a
1022-token prefill**. `DSV4_SPARSE_HPB=1` restores the old launch for the A/B.

### Measured

| | before | after | |
|---|---|---|---|
| prefill | 18300 ms / 55.9 tok/s | **17364 ms / 58.9 tok/s** | **+5.4 %** |
| `cattn:sparse` | 2926.9 ms | **2016.9 ms** | **-31.1 %** |
| ATTENTION | 9063.6 ms | 8109.3 ms | -10.5 % |
| base AR decode | 11.56 tok/s | **11.90 tok/s** | **+2.9 %** |

Predicted 1800-2300 ms for the mark; landed 2016.9. **Decode improved as well** — at m=1 the
launcher picks HPB=1, the original launch shape, but still gets the PER=16 register win. A
prefill-motivated change that helps decode is rare enough to record.

### Cumulative B9 so far

| stage | prefill | tok/s |
|---|---|---|
| baseline (F84) | 21351.6 ms | 47.9 |
| `MOE_MMA=1` (F85) | 18300 ms | 55.9 |
| + sparse PER/HPB (F86) | **17364 ms** | **58.9** |

**+23.0 % cumulative.** 20K-sample capture: 4.9 d -> **4.0 d**. Real, not yet decisive.

### The pattern, now three for three

| | tuned for | wrong for prefill because |
|---|---|---|
| MoE `RB=4` | 1-5 rows/expert at decode | 43 rows/expert -> 11 weight reads |
| MoE GEMV vs mma | M=1: 350 vs 121 GB/s | M=1022 inverts it |
| `sparse_attn` regs + 1 warp/block | generic d<=1024, 64 warps total | d=512, 65,408 warps |

None are bugs. Each is a correct decision for the batch size it was made at, applied unchanged to a
batch 1000x larger — exactly what seventeen cycles of decode tuning would leave behind in functions
prefill shares.

### What is left, and the honest ceiling

Remaining attention marks: `cattn:ogroup` 2786, `cattn:compress` 2239, `cattn:indexer` 2078,
`cattn:sparse` 2017. Four items of similar size; ~30 % each would take prefill to ~14.6 s.

**But the structural number says that is not where this ends.** 1022 tokens x ~12.5 B active params
= 25.6 TFLOP in 17.36 s = **1.47 TFLOPS**. `sparse_attn` computes fp32 dot products with warp
shuffles and **no tensor cores at all** — 5 shuffles plus a broadcast to reduce 16 FMAs of useful
work. The remaining order of magnitude is a flash-attention-style tensor-core kernel for the
score path (`sparse` + `indexer` = 4.1 s together), not more tuning of this one.

## Finding 87 — the compute peak, measured at last: **FP32 5.45 / BF16 53.1 / FP8 92.2 TFLOPS** — and my "1 % of peak" claims in F84-F86 were wrong by an order of magnitude

**Operator-run, `tools/flops_probe.cu`, clocks pinned.** Every roofline in this project has been a
BANDWIDTH roofline (`bw_probe`, 233 GB/s). Compute was never measured, and F84/F85/F86 each leaned
on "that TFLOPS number is low" — an assumption, not a measurement, that decides whether the
remaining prefill work is a tensor-core rewrite or fine tuning.

```
device: NVIDIA Thor  SMs=20  clock=1.049 GHz  cc=11.0
FP32 FFMA                  5.45 TFLOPS      (reference for 128 cores/SM: 5.37 -> core count confirmed)
FP16 mma m16n8k16         53.07 TFLOPS
BF16 mma m16n8k16         53.08 TFLOPS
FP8 e4m3 mma m16n8k32     92.18 TFLOPS
```

`NACC=32` independent FFMA chains, so this is throughput- not latency-bound — one dependent chain
would have measured FFMA latency and reported ~1/4 of peak, the classic way this benchmark lies.

### The correction I owe

I wrote that prefill at 1.47 TFLOPS was "~1-2 % of plausible peak" and that `cattn:sparse` at
0.72 TFLOPS was catastrophic. **Both assumed a ~100 TFLOPS peak that exists only in FP8 tensor
cores.** Against the real FP32 ceiling:

| | measured | % of **5.45 TFLOPS** FP32 peak |
|---|---|---|
| whole prefill | 1.47 TFLOPS | **27 %** |
| `cattn:sparse` (post-F86) | 1.04 TFLOPS | **19 %** |
| `cattn:ogroup` | 1.98 TFLOPS | **36 %** |

The fp32 attention path is running at a *reasonable* fraction of what fp32 on this chip can do. It
was never 1 % of anything. **Thor's compute lives almost entirely in the tensor cores — bf16 is
9.7x fp32, fp8 is 16.9x — and not one kernel in our attention path touches them.**

### What this settles

- **fp32 tuning is nearly exhausted.** `cattn:sparse` at 19 % of the fp32 peak has ~5x left in
  theory and realistically ~2x. The four remaining attention marks are each worth tenths of a
  second, not seconds. **F86's own suggestion — "three or four more changes of the kind just made" —
  is worth far less than I priced it at.**
- **The tensor-core rewrite is confirmed as the order-of-magnitude lever, with a real number behind
  it now.** A bf16 flash-attention score path at a conservative 30-50 % of the 53.1 TFLOPS peak is
  **15-25x** on `cattn:sparse` alone.

Sizing the two rewrites against measured peaks:

| step | saves | prefill | tok/s |
|---|---|---|---|
| today (post-F86) | — | 17.36 s | 58.9 |
| `sparse`+`indexer` -> bf16 TC @40 % of peak | 3.90 s | 13.47 s | 75.9 |
| + `ogroup` -> fp8 TC @30 % of peak | 2.58 s | 10.88 s | **93.9** |

**20K-sample capture: 4.0 days -> 2.5 days.**

### Caveats that belong with these numbers

1. **30-50 % of mma peak is an assumption**, and a friendly one. It is what well-written
   flash-attention kernels reach on datacenter parts; our sparse path gathers keys through an index
   list, which is harder. Treat 93.9 tok/s as the optimistic end.
2. **Bit-exactness is off the table.** Every prefill change so far (F85, F86) was bit-exact and
   gated on `rms=0.00e+00`. A bf16 tensor-core score path is a numerics change and needs a
   cosine-tolerance gate instead — a different, weaker guarantee, and the LOSSLESS spec gate becomes
   the thing that must hold.
3. **MXFP4 tensor cores are NOT covered by this probe.** `HARDWARE.md` records that `mma.sync` FP4
   is blocked on `sm_110a` and Thor reaches FP4 only via `tcgen05`, which needs tensor-memory
   allocation and matrix descriptors. The MoE at 5.1 TFLOPS against a 92 TFLOPS fp8 peak looks like
   5 % — but the honest statement is that its ceiling has not been measured.

## Finding 88 — the ogroup tensor-core path re-dequantised every weight row **64 times**: -33 % on the mark, +6.0 % prefill

**B9, operator-run.** `evidence/prefill_ogmt_b9g.log`, `MOE_MMA=1` held constant, two PS=1022 points
agreeing to 0.1 %.

Prefill at bs>16 already *reached* tensor cores — `tc_ogroup_fp8_kernel`, which F62 wrote to fix a
correctness bug (rows 16+ were never written) and which nobody has looked at for speed since. Three
defects, all the now-familiar shape:

1. `<<<grid,32>>>` — **one warp per block**, 65,536 blocks at bs=1022.
2. The weight row is loaded and **dequantised once per m-tile** — two `exp2f`, four scalar byte reads
   and four fp8->half converts per k-step, repeated for all **64** m-tiles, for bytes that never change.
3. Scalar byte weight loads. F35 fixed exactly this in the M=1 GEMV; the TC path never got it.

Fixing (2) subsumes (3): hoist the load+dequant out of an m-tile loop so MT tiles share one dequant —
the F64 row-amortisation transformation applied to the ogroup TC path. **Bit-exact**: each output
(r,n) still accumulates over k0 through the same mma sequence; only the order in which independent
outputs are produced changes. `gate_compressed_decode` runs at s=256 = 16 m-tiles = the MT=8 path and
returns `rms=0.00e+00`; `gate_ogroup_gemv` PASS.

MT follows the m-tile count, so the verify (bs<=16, never reaches here) and any single-tile shape get
MT=1, which is the original kernel. `OG_TC_MT=1` forces it for the A/B.

| | before | after | |
|---|---|---|---|
| prefill | 17364 ms / 58.9 tok/s | **16373 ms / 62.4 tok/s** | **+6.0 %** |
| `cattn:ogroup` | 2786.1 ms | **1866.1 ms** | **-33.0 %** |

### Cumulative B9

| stage | prefill | tok/s | |
|---|---|---|---|
| baseline (F84) | 21351.6 ms | 47.9 | |
| `MOE_MMA=1` (F85) | 18300 ms | 55.9 | +16.7 % |
| sparse PER/HPB (F86) | 17364 ms | 58.9 | +5.4 % |
| ogroup m-tile (F88) | **16373 ms** | **62.4** | **+6.0 %** |

**+30.3 % cumulative, all bit-exact.** 20K-sample capture: 4.9 d -> **3.8 d**.

**The pattern is now four for four.** Every one of these was a correct decision for the batch size it
was made at, applied unchanged to a batch 1000x larger: MoE `RB=4` (1-5 rows/expert at decode), MoE
GEMV-vs-mma (M=1 inverts at M=1022), `sparse_attn` registers + 1 warp/block (generic `d<=1024`, 64
warps total), and now ogroup m-tile dequant (bs<=16 has 1 m-tile, bs=1022 has 64). **None is a bug.**
Seventeen cycles optimised decode inside functions prefill shares, and prefill was never measured.

## Finding 89 — the compressor is numerics-FRAGILE: changing its GEMM kernel at all costs **28 % of acceptance**, and a 54x smaller error buys nothing

**B9, operator-run.** Five runs, `evidence/prefill_{ogmt_b9g,bf16tc_b9h,split2_b9i,compbf16_only_b9j}.log`.

### What was built

A bf16 tensor-core `gemm_bf16w` (64x64 block tile, 4 warps 2x2, BK=32, fragment layout copied from
the already-gated `ogm_mma`). `gemm_bf16w` at M=1022 was warp-per-output-ELEMENT — 261,632 blocks
running a dense `[1022x4096]x[4096x1024]` at **0.32 TFLOPS** against a measured 53.08 TFLOPS bf16
peak. New gate `tests/gate_bf16w_tc.cu`, because **`gate_bf16w` only tests M=1 and NO gate in the
project could reach M>=64.**

### The measurements, in the order they were taken

| config | acceptance | prefill |
|---|---|---|
| exact (b9g) | **4.00 / 4.00** | 62.4 tok/s |
| `COMP_BF16=1` **only**, no TC (b9j) | **2.80 / 2.89** | 65.4 |
| `COMP_BF16=1` + TC, NS=1 (b9h) | 2.18 / 2.78 | 69.5 |
| `COMP_BF16=1` + TC, split NS=2 (b9i) | 2.89 / 2.40 | 69.0 |

Four earlier runs on this prompt sat in a **3.57-4.00** band, so the drop is real, not drift.

### Three results, and the third is the one that matters

1. **The tensor-core kernel is exonerated.** `COMP_BF16=1` alone — the scalar bf16 kernel, no mma —
   already costs the acceptance. The TC path adds speed on top without further damage.
2. **Split-bf16 works and does not help.** `gate_bf16w_tc`: rms_rel **5.46e-4 at NS=1 -> 1.02e-5 at
   NS=2**, a 54x improvement, for acceptance 2.78 -> 2.89. **Error magnitude is not the mechanism.**
   NS=3 measured identical to NS=2, i.e. two terms already sit at the accumulation-order floor
   between a tiled kernel and a scalar one.
3. **`gemm_fp32` and `gemm_bf16w` receive NUMERICALLY IDENTICAL WEIGHTS.** `wkv`/`wgate` ship bf16
   and `Loader::bf16` expands them to fp32 exactly. The two kernels differ only in accumulation
   order. **So changing the compressor's GEMM kernel at all, with identical inputs, costs 28 % of
   acceptance.**

### The mechanism this implies

Not top-k near-ties as I first argued — that predicts damage proportional to perturbation size, and
a 54x reduction bought nothing. The likely amplifier is **`act_quant_fp8sim` / `act_quant_fp4sim`**,
which compute a block scale from the **max over a 64- or 32-wide block**. A change in one element
that moves the block max rescales **every** value in that block, turning a last-bit difference into
a whole-block shift — which then feeds the indexer's top-512 selection. That is a threshold effect
with no gradient, which is exactly the shape the data shows.

### Consequence for S5 — the part that binds

**Capture must use the same numerics as serving.** We store `mh_pre`, which depends on the entire
forward including the indexer's key selection. Capturing under one GEMM kernel and serving under
another trains the head on features it will never see — the precise train/inference mismatch the
whole recipe exists to prevent, and we now have a 28 % measurement of how much this engine cares.

**So the lever is retired and prefill stands at 62.4 tok/s with exact numerics.** The +11.4 % was
real and is not worth a train/serve numerics split, when the entire point of S5 is to raise the
acceptance this would spend.

The one coherent alternative, recorded but NOT taken: standardise on `COMP_BF16=1`+TC for **both**
capture and serving and let the fine-tune re-align the head to those numerics. It is principled —
the head is being retrained anyway — but it bets the whole acceptance gain on recovering 28 % that
we would have caused ourselves, to save 10 % of capture wall time.

### Process note

**I violated one-change-per-measurement and it cost two runs.** `COMP_BF16` and `TC_BF16W` went on
together; when acceptance fell I blamed the tensor cores and built a 54x-more-accurate kernel to fix
a problem they were not causing. The isolation run that should have been first was fourth.

Also, the stale-binary trap fired **three times** in this sequence, and each time the tell was a
number that was too *consistent*, never an error message: five gates reporting `rms=0.00e+00` on a
bf16 change; three NS values reporting byte-identical accuracy after a failed compile. **Before
believing a comparison, check that its arms differ where they must.**

## Finding 90 — S5 Stage 0: capture and the weight-override path both work, and the reference PyTorch model validates the capture design

**Operator-run.** `evidence/stage0_capture.log`, exact numerics (no `MOE_MMA`), external head via `argv[4]`.

### The override path already existed

`src/decode.cu` opens a SEPARATE `WeightStore` filtered to the `mtp.` prefix when `argv[4]` names a
directory other than the checkpoint — written "for backward compatibility with an external head" and
unused since. That is exactly what S5 needs: load a fine-tuned head without writing into the
read-only checkpoint. It needed only a directory whose index mentions `mtp.*` alone
(`tools/make_head_dir.py`; symlinks the sha-verified backup shards rather than copying 6.6 GB, and
rewrites `total_size` to describe the new directory).

**Identity control PASSES:**
```
[spec] loading EXTERNAL DSpark head .../dspark-head-identity ...
[spec] head loaded 6.53 GiB, 2977 mtp tensors (SEPARATE store)
[decode] first decoded token argmax = 11111 -> GATE PASS
[spec] mean tokens/verify = 2.89
```
2.89 is the recorded baseline exactly, so later comparisons measure training, not the loader.
**Cost: the separate store adds 6.53 GiB** (109.2 -> ~115.7 of 122.8). Fine for evaluation, not
headroom we have twice.

### Capture

`DSV4_CAPTURE=<dir>` dumps `mh_pre` (3 taps x 4096 per position, bf16) + token ids, one file per
sweep point so `DSV4_PROMPTS` + `DSV4_BLKSWEEP` amortise the ~10-minute checkpoint load across N
samples. Append-only `manifest.jsonl` so a killed run resumes from the last complete line.

```
shard_p00_s000005.dspc: n_tok=5    ids=6     slot_std=[2408.6, 2391.6, 2254.2]  OK
shard_p01_s001022.dspc: n_tok=1022 ids=1023  slot_std=[189.0, 192.5, 201.3]     OK
total: 1027 tokens, 24.1 MB (24.0 KB/token)   VALIDATE: PASS
```

The **per-slot standard deviations** are the check that matters: they differ across the three taps,
proving the `slot` index varied and we captured layers 40/41/42 rather than one layer three times.
A byte-count check would have passed on that failure.

### The reference PyTorch model validates the capture design

`inference/model.py` ships `DSparkBlock`/`DSparkAttention`/`DSparkMarkovHead`/`DSparkConfidenceHead`,
and its `main_hidden` is built as `torch.cat([h.mean(dim=2) for target layers], dim=-1)`. Our
`k_tap_pool` computes the same mean into `mh[t*(3*d)+slot*d+j]`. **The captured `mh_pre` IS the
reference's `main_hidden`.** And `forward_embed` applies `main_norm(main_proj(main_hidden))`
afterwards, confirming that capturing UPSTREAM of `main_proj` — the trainable feature projection —
was correct.

### The Stage-0 gate that remains, and the decision it forces

The reference model imports `kernel.py`, which requires **`tilelang==0.1.8`** and
`fast_hadamard_transform`. Measured in `ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor`:

```
torch 2.10.0 cuda True
tilelang MISSING
fast_hadamard_transform MISSING
```

So using the reference as-is means installing a TileLang compiler stack on `sm_110a` — uncertain,
and capable of consuming the entire Stage-0 budget on its own.

**Decision: do not take the dependency.** Use the reference's STRUCTURE (which is readable and now
verified against our capture) with plain PyTorch ops, and **dequantise the frozen MXFP4 experts to
bf16 once** (~24-26 GB, comfortable with no engine resident). This removes `fp4_gemm`/`fp8_gemm`
entirely, makes everything differentiable by default, and does not violate no-additional-quantisation
— we read MXFP4 and never write it; only the 476 M non-expert parameters are ever saved.

The cost is writing plain-PyTorch `sparse_attn` and `hc_split_sinkhorn`. At 476 M trainable
parameters over ~1 K-token sequences that is fast enough, and it is bounded work with no external
compiler risk.

## Finding 91 — the checkpoint's reference model is INFERENCE-ONLY by construction, and making it trainable is a port, not a shim

**S5 Gate 1.** Continued from F90. Every remaining blocker traced to the same root cause, found by
anomaly detection rather than by guessing — after two guesses that were both wrong.

### What now works

* MXFP4/FP8 dequantisation: 12.60 B params, 23.5 GiB bf16; **0 missing / 0 unexpected** on
  `load_state_dict` across all 3 blocks.
* The **two-phase forward** — the reference's `start_pos == 0` only fills the KV cache and returns
  (`forward_spec` says `if start_pos == 0: return`), which is why `h` came back detached. Phase A
  fills the cache once per sequence at O(T); phase B drafts from each position against that shared
  cache at O(BS) each. **So one captured sequence yields ~T training examples, not 1, and without
  the O(T^2) memory I feared in F90.** That was the main open design risk and it is resolved.
* A **teacher-forced `forward_head`**. The reference version is an autoregressive SAMPLING loop that
  feeds its own samples forward and mutates `output_ids` after the Markov embedding has saved a view
  of it. Non-differentiable and illegal under autograd. Teacher forcing removes both and is what the
  DSpark/FastMTP recipes prescribe anyway.

### The root cause, stated plainly

`inference/model.py` is written to run under `torch.inference_mode()`, where in-place mutation of
tensors that autograd would need is free. On the DSpark path alone:

```
10x apply_rotary_emb(x[..., -rd:], ...)   # rotates a SLICE in place
 5x act_quant(..., True) / fp4_act_quant(..., True)
 1x logits[:, i].add_(bias)
```

Each one raises `... modified by an inplace operation ... is at version N; expected version M` the
moment a gradient needs the pre-mutation value. I fixed my own two (`act_quant` via `.data.copy_`,
a genuine straight-through estimator) and the next site is the model's own `apply_rotary_emb`.

**This is not a shim problem and no amount of kernel substitution fixes it.** Making the reference
trainable means rewriting ~16 in-place sites out-of-place inside the checkpoint's file — bounded and
mechanical, but a PORT, and every edit is a chance to diverge from the weights it was written for.

### What this does to the Gate 2 estimate

| | before | now |
|---|---|---|
| engineering before capture | "half a day" (F90) | **1-2 days**: port ~16 sites, then re-validate the ported forward against the ENGINE's own draft output, because a silent divergence trains the head against the wrong function |
| capture | ~1.5 days | unchanged |

The re-validation is not optional. F89 measured that this engine loses **28 % of acceptance** from an
accumulation-order change alone; a ported forward that differs from the server by more than that is
worse than no fine-tune. The equivalence gate has to come before the capture, not after.

## Finding 92 — the three free measurements: the suite mean is **21.55 tok/s (not 27-30)**, but tau is **transient-suppressed and more than doubles over the drafter's 128-token window**

**Operator-run, `evidence/tau_suite_m1m2.log`.** 9 prompts (canonical control + one per category from
`scripts/prompt_suite.json`, ids from the checkpoint's own tokenizer, self-gated), NGEN0=200 — the
first time this engine has ever generated past the drafter's 128-token sliding window.

### M1 — the prompt-suite re-baseline REFUTES the workflow's headline

| category | tau | tok/s |
|---|---|---|
| long_context (83 tok) | **4.43** | **28.25** |
| agentic_format (35) | 3.96 | 26.52 |
| code_edit (42) | 3.85 | 25.67 |
| multi_turn (15) | 3.62 | 27.12 |
| canonical control (6) | 3.41 | 23.74 |
| short_factual (13) | 2.87 | 22.30 |
| code_gen (27) | 1.83 | 14.61 |
| reasoning (38) | 1.80 | 14.80 |
| explanation (15) | **1.68** | **13.12** |
| **MEAN of the 8 real prompts** | **3.00** | **21.55** |
| recorded baseline (canonical, NGEN0=24) | 2.89 | 22.15 |

**The deep-research synthesis claimed we were "already at 27.4-30.7 tok/s" on realistic prompts and
that most of the S5 budget was therefore spent before the first gradient step. That is wrong.** The
realistic-prompt mean is **21.55 tok/s — slightly BELOW our recorded 22.15 baseline.** The
1023-token code prompt that produced tau 3.57-4.00 was a favourable case, not a representative one,
and I extrapolated from it in exactly the way the report warned against.

**Our recorded baseline is representative.** That is the single most useful thing these measurements
established, and it cost one run.

The spread is real and matches the literature: **2.15x in tok/s (13.12 to 28.25)**, against a
published intrinsic domain spread of 2.1x for block/semi-AR drafters. It is **bimodal, and the split
is semantic**: constrained continuation (long context, tool format, code edit, multi-turn) runs
tau 3.6-4.4, while OPEN-ENDED GENERATION (explanation, reasoning, code_gen) collapses to tau
1.68-1.83. A drafter cannot guess what has not been constrained.

### M2 — CONFIRMED, and the effect is large

tau binned by generated-token index, pooled over all 9 prompts:

| generated tokens | n | tau |
|---|---|---|
| 0-31 | 59 | **1.39** |
| 32-63 | 125 | 1.83 |
| 64-95 | 89 | 2.96 |
| 96-127 | 90 | **3.23** |
| 128-159 | 101 | 2.84 |
| 160-191 | 92 | 3.13 |
| 192-223 | 86 | 2.97 |
| 224-255 | 25 | 3.36 |

**tau more than doubles — 1.39 to ~3.2 — and plateaus at 96-128 generated tokens, exactly the
drafter's 128-token sliding window (`sliding_window: 128`).** The predicted mechanism, confirmed:
the drafter's receptive field must FILL before it can draft well, and a short generation measures
mostly the transient.

Consequence: **every short-generation acceptance number in this project understates steady state.**
Our 2.89 was measured at NGEN0=24 — entirely inside the first two bins. The same prompt at NGEN0=200
gives **3.41 (+18 %)**, and the true steady state is ~3.1-3.2.

**Reporting rule earned: quote tau at steady state, past 128 generated tokens, and say the
generation length.** A tau without a generation length is not comparable to anything, including our
own earlier numbers.

### M3 — the bonus-token audit is clean, but it found a real defect

The convention is correct and comparable: `verify N: accepted acc/(VB-1) + correction -> +(acc+1)
tokens`, `tau = total/nverify` — **the bonus token is included**, same as SPEED-Bench, DSpark and
Nebius. There is no +1.00 shift, and the workflow's proposed correction is refuted.

But tracing it exposed this:

```cpp
int VK = BLK;                                   // caps at BLK = 5
const int VB = VK;
vtok[0] = cur;                                  // one slot spent on the KNOWN token
for(int i=1;i<VB;++i) vtok[i] = draft[i-1];     // so only VB-1 = 4 drafts are verified
```

`hmarg[]` carries **BLK = 5** margins, so the drafter produces five proposals — and `draft[4]` is
computed every round and **never verified**. Our ceiling is 5, not 6. That is exactly why the DSpark
paper reports **6.11** accepted length at gamma=5: it verifies gamma+1 positions.

**We pay to generate a token we discard, and our ceiling is one short of the architecture's.**
Raising the cap to `VK <= BLK+1` verifies all five drafts for one extra verify position; F64's union
measurements say that position costs ~14 % more expert bytes (K=4 union 15.16 -> K=5 17.53) for
20 % more ceiling, and `seqmax` already carries `+ BLKMAX + 8` of slack. **This is the cheapest
untested lever found in this session and it needs no training.**

### What this does to S5

The headroom is real but must be quoted correctly: steady-state suite tau is **~3.0-3.1 against a
ceiling of 5** (6 if the draft[4] defect is fixed). The three collapsed categories — explanation,
reasoning, code_gen at tau 1.68-1.83 — are open-ended generation, and are exactly where a fine-tune
on target-regenerated responses should help most. **Set the S5 success criterion as the 8-prompt
suite mean at NGEN0>=200, not as a single number on the canonical prompt.**

## Finding 93 — ADOPTED: the verify was checking 4 of the drafter's 5 proposals. Verifying all five is **+8.3 % tau / +3.4 % tok/s, lossless, no training**

**Operator-run.** `evidence/vkplus_suite.log` vs `evidence/tau_suite_m1m2.log` — identical 9 prompts,
identical NGEN0=200, one change.

### The defect

`hmarg[]` carries **BLK = 5** margins, so the drafter has always produced five proposals. But the
verify spends `vtok[0]` on the ALREADY-KNOWN token and fills `vtok[1..VB-1]` from `draft[]`, with
`VK` capped at `BLK`. So it checked `draft[0..3]` and **`draft[4]` was computed every round and
thrown away.** tau could never exceed 5 where the architecture allows 6 — which is exactly why the
DSpark paper reports **6.11** accepted length at gamma=5 while our ceiling read 5.

**Verify width and draft width are different numbers.** Verifying all BLK proposals needs
`VB = BLK + 1`.

### The part that would have been silent

`hv`, `hv2`, `collK`, `logK`, `mh_v` were all allocated `BLKMAX * ...`. At `VB=6` against
`BLKMAX=5` they overrun by exactly one position — and **that would not have crashed.** It would have
corrupted the last verified token, which is the highest-value token in the block because it gates
everything after it. Same shape as Finding 62. New `VBMAX = BLKMAX + 1` sizes every VERIFY-side
buffer; the DRAFT-side ones (`dbid`, `dmarg`, `xemb`, `xa`, `xb`) correctly stay at `BLKMAX`,
because the draft really is BLK wide.

### Measured

| prompt | tau | tok/s | |
|---|---|---|---|
| long_context | 4.43 -> **5.00** | 28.25 -> **29.89** | +5.8 % |
| agentic_format | 3.96 -> 4.41 | 26.52 -> 28.15 | +6.1 % |
| multi_turn | 3.62 -> 4.06 | 27.12 -> 28.66 | +5.7 % |
| short_factual | 2.87 -> 3.11 | 22.30 -> 22.95 | +2.9 % |
| canonical | 3.41 -> 3.61 | 23.74 -> 24.36 | +2.6 % |
| code_edit | 3.85 -> 4.08 | 25.67 -> 26.09 | +1.6 % |
| explanation | 1.68 -> 1.70 | 13.12 -> 13.18 | +0.5 % |
| code_gen | 1.83 -> 1.86 | 14.61 -> 14.63 | +0.1 % |
| reasoning | 1.80 -> 1.82 | 14.80 -> 14.75 | -0.3 % |
| **SUITE MEAN** | **3.00 -> 3.26 (+8.3 %)** | **21.55 -> 22.29 (+3.4 %)** | |

`GATE PASS`, `MATCH 5/5`, **`LOSSLESS GATE PASS`**. Base AR unchanged (13.80 -> 13.77).
`DSV4_NOVKPLUS=1` restores the old cap.

Predicted +3 to +8 % on the suite mean before the run; landed at +3.4 % tok/s. The DISTRIBUTION was
predicted too and matters more than the mean: the gain is concentrated in the four categories
already pressing the old ceiling, and is ~zero in the three collapsed ones (code_gen, reasoning,
explanation) which never reach VK=5 — for them a sixth position is dead weight the adaptive gate
correctly declines to spend.

### Two consequences

1. **`long_context` hit tau = 5.00 exactly, against the new ceiling of 6.** It is now pressing the
   new limit as hard as it pressed the old one. **Block size > 5 (F43, conditionally retired
   pending higher acceptance) is no longer conditional — it is the obvious next lever**, and this
   finding is the acceptance evidence F43 was waiting for.
2. **Every acceptance number this project has ever quoted was against a ceiling one token short of
   the architecture's.** The DSpark paper's 6.11 was never out of reach; we were measuring a
   different quantity.

## Finding 94 — block size: **6 wins (+3.3 %), 8 loses (-5.5 %)**, and the head generalises exactly ONE position past its training width

**Operator-run.** `evidence/blocksize_sweep.log`, 15 points: BLK in {5,6,8} x 5 prompts, NGEN0=200,
`adaptK=1.50` (the shipped setting, so this is the realistic operating point, not a forced-width
ceiling). **Control: prompt 0 at BLK=5 reproduces the F93 run to 0.08 %** (3.61/24.38 vs 3.61/24.36),
so the sweep measures block size and not drift.

| prompt | tau 5 -> 6 -> 8 | tok/s 5 -> 6 -> 8 | 6v5 | 8v6 |
|---|---|---|---|---|
| agentic | 4.41 -> 5.15 -> 5.10 | 28.18 -> **29.96** -> 28.83 | **+6.3 %** | -3.8 % |
| long_context | 5.00 -> 5.54 -> 5.43 | 30.95 -> **31.96** -> 28.72 | +3.3 % | -10.1 % |
| code_edit | 4.08 -> 4.43 -> 4.04 | 26.13 -> **26.82** -> 23.82 | +2.6 % | -11.2 % |
| multi_turn | 4.06 -> 4.46 -> 5.10 | 28.71 -> 28.99 -> **29.85** | +1.0 % | +3.0 % |
| canonical | 3.61 -> 3.61 -> 3.48 | **24.38** -> 22.91 -> 20.87 | -6.0 % | -8.9 % |
| **realistic mean** | | **28.49 -> 29.43 -> 27.80** | **+3.3 %** | **-5.5 %** |

`LOSSLESS GATE PASS`, `MATCH 5/5`.

### The mechanism is visible in tau, not in tok/s

**tau RISES 5->6 on all four realistic prompts. It FALLS 6->8 on three of four** (long_context
5.54 -> 5.43, code_edit 4.43 -> 4.04, agentic 5.15 -> 5.10). The drafter is not merely failing to
*pay* for positions 7-8 — it is producing *worse* drafts when asked for them, because the whole
block conditions on `DSPARK_NOISE_TID` at positions 1..BLK-1 and a wider noise field degrades the
proposals it was getting right.

**The head was trained at block 5 and generalises exactly one position past that, not three.** That
is a cleaner statement of the limit than any acceptance number, and it says the remaining width
headroom is not reachable by configuration — it needs training.

### Block size is a per-workload parameter, not a global one

The canonical prompt **regresses monotonically** (24.38 -> 22.91 -> 20.87) while its tau is flat
(3.61 -> 3.61 -> 3.48): a low-acceptance workload pays the wider verify for width it cannot use.
`adaptK` already adapts width *within* a block from the draft's own margins; **the block itself is
still a compile-time constant.** Making BLK follow measured acceptance the same way is the obvious
next lever, and F94 is its evidence.

### Honest limit on the magnitude

`long_context` measured **29.89** at BLK=5 in the F93 run and **30.95** here — the same
configuration, 3.5 % apart. So the +3.3 % headline sits inside single-pair noise and trap 25
applies. **What is solid is the SIGN, not the size:** tau rises 5->6 on 4 of 4 realistic prompts and
tok/s rises on 4 of 4, which a 3.5 % run-to-run spread cannot manufacture. Adopting BLK=6 for
realistic workloads is justified; quoting "+3.3 %" as a measured gain is not, until a second pair
confirms it.

## Finding 95 — adaptive block size: **NEGATIVE (+0.2 % vs fixed 5, -4.6 % vs fixed 6)**. It oscillates, and the reason is F92's transient

**Operator-run.** `evidence/adaptblk_suite.log`, all 9 prompts, NGEN0=200, start BLK=5, controller
free in [3,6], window 8 verifies, grow at `tau >= BLK-0.75`, shrink at `tau <= BLK-1.75`.
`LOSSLESS GATE PASS`, `MATCH 5/5` — it is correct, it is just not better.

| | suite mean (8 real) |
|---|---|
| fixed BLK=5 (F93) | 22.29 tok/s |
| **adaptive [3,6]** | **22.34 tok/s (+0.2 %)** |
| **fixed BLK=6** (F94, on its 4 prompts) | **29.43** |
| **adaptive** (same 4 prompts) | **28.09 (-4.6 %)** |

**A wash against fixed 5, and materially WORSE than simply fixing 6.**

### Why — and it connects two findings

The controller **thrashes**: 3-8 moves per prompt, e.g. prompt 3 goes `5->4->3->4->5->4->3->4`
and prompt 0 goes `5->4->5->4->5->6`. It almost never sits at a width long enough to profit from it,
because each move clears the history and re-enters a measurement transient.

The root cause is **Finding 92**. tau is **1.39 over the first 32 generated tokens** and only reaches
~3.2 after ~128, because the drafter's 128-token sliding window has to FILL. The controller starts
measuring immediately, sees the transient, and shrinks — every prompt's first move is downward
(`tau=1.75 -> BLK 4`, `tau=1.12 -> BLK 4`, `tau=1.00 -> BLK 3`). It then has to climb back through
the same hysteresis it just paid for. **It is not adapting to the workload; it is chasing the
window-fill curve.**

That is a real insight and it is the useful part of this finding: **any controller keyed on
short-horizon acceptance will fight the transient F92 measured.** A fix has to either suppress
adaptation until ~128 generated tokens, or use a window long enough that the transient is a small
fraction of it — at which point, on a 200-token generation, there is almost nothing left to adapt
over.

### The decision

**Ship fixed BLK=6 for realistic workloads; do not ship the controller.** F94 already established
6 > 5 on 4 of 4 realistic prompts, and the controller costs 4.6 % against that while adding a
mutable width, a device-buffer ceiling and an oscillation failure mode. `DSV4_ADAPTBLK` stays in the
tree, default OFF, as the A/B arm and as the starting point for a transient-aware version.

**Note the shape of the win it gave up**: the three collapsed categories DID improve (+1.4 to +1.8 %
on code_gen, explanation, reasoning) because shrinking below 5 genuinely helps a drafter that cannot
fill a block. The idea is right; the signal is too noisy at this horizon to act on.

### Second buffer-sizing near-miss in three findings

`dbid`, `dmarg`, `xemb`, `xa`, `xb` are all sized from `BLKMAX`, which a `BLK0=5` sweep sets to 5 —
so the controller stepping to 6 would have overrun all five, and like F93's verify-side buffers it
would **not have crashed**, it would have corrupted the last drafted proposal. Fixed at the sizing
site (`BLKMAX >= 6` whenever `DSV4_ADAPTBLK` is set), not at the controller.

**Pattern worth naming: any change to a width parameter must be paired with an audit of everything
sized from it.** Twice in three findings the code change was trivial and the buffer sizing was the
real work, and both times the failure mode was silent corruption rather than a crash.

## Finding 96 — block 6 shipped as the default: **+1.6 % suite mean, not the +3.3 % F94 quoted** — the subset excluded every prompt that regresses

**Operator-run, CLEAN.** `evidence/baseline_blk6_suite.log`. All 9 prompts, NGEN0=200, no profiling
instruments, clocks pinned, caches dropped. `GATE PASS`, `MATCH 5/5`, `LOSSLESS GATE PASS`, base AR
**13.76 unchanged**.

| prompt | tau 5 -> 6 | tok/s 5 -> 6 | |
|---|---|---|---|
| agentic | 4.41 -> 5.15 | 28.15 -> 29.98 | **+6.5 %** |
| long_context | 5.00 -> 5.54 | 29.89 -> 30.77 | +2.9 % |
| code_edit | 4.08 -> 4.43 | 26.09 -> 26.81 | +2.8 % |
| explanation | 1.70 -> 1.75 | 13.18 -> 13.38 | +1.5 % |
| multi_turn | 4.06 -> 4.46 | 28.66 -> 28.97 | +1.1 % |
| short_factual | 3.11 -> 3.27 | 22.95 -> 22.78 | -0.7 % |
| reasoning | 1.82 -> 1.85 | 14.75 -> 14.58 | -1.2 % |
| code_gen | 1.86 -> 1.84 | 14.63 -> 13.97 | **-4.5 %** |
| canonical (control) | 3.61 -> 3.61 | 24.36 -> 22.86 | **-6.2 %** |
| **SUITE MEAN (8)** | **3.25 -> 3.54 (+8.9 %)** | **22.29 -> 22.66** | **+1.6 %** |

### The correction I owe on my own number

F94 reported **+3.3 %**, measured on a 4-prompt subset — agentic, code_edit, long_context,
multi_turn. **I picked those four because they were the ones pressing the ceiling**, which is
exactly the set that benefits from a wider block. Re-measured here, that same subset still gives
+3.3 % (28.20 -> 29.13), so the number was not wrong — **the population was.** The full suite is
**+1.6 %**, because four prompts regress and the subset contained none of them.

**Selecting the measurement set by the mechanism you expect to help is a way to be precisely right
about the wrong thing.** F94's conclusion (ship 6) survives; its magnitude does not.

### The split is real and it is what F95 was reaching for

Block 6 helps high-acceptance workloads and **costs** low-acceptance ones: canonical -6.2 %,
code_gen -4.5 %. The spread between best (+6.5 %) and worst (-6.2 %) is ~13 points, so a controller
that could route each workload to its own width is worth several times the +1.6 % that a single
global constant buys. F95's controller failed by chasing F92's window-fill transient, not because
the opportunity is absent — **the opportunity is now measured at ~13 points of spread.**

### Shipped

`DSV4_BLK` defaults to 6; `DSV4_BLK=5` restores the previous behaviour. **`DSPARK_BLOCK` stays 5**
and is untouched: it is `config.json`'s `dspark_block_size`, the width the head was TRAINED at, and
a model constant. The number of proposals the ENGINE requests is a serving decision. Conflating
those two is what produced F93's defect in the first place.

### The new baseline, with its protocol attached

`FLYWHEEL_STATE.json` now records the baseline as an **8-prompt suite mean at NGEN0=200**, not a
single canonical number, with F92's rule written into it: **tau must be measured past the drafter's
128-token sliding window, and a tau without a generation length is not comparable to anything —
including this project's own earlier numbers.** The canonical 6-token prompt is retained as a
CONTROL only; it is the worst case and it regresses at block 6.

## Finding 97 — the DFlash drafters do not transfer, and the one idea worth taking from them (draft-vocab trim) is **NOT BUILDABLE YET** — measured, before building it

Operator supplied six HuggingFace DFlash/DSpark drafters for DeepSeek-V4-Flash. Checked their
configs and manifests rather than their names.

### One of them is our own head

`GaelicThunder/DSpark-DeepSeek-V4-Flash-0731-drafter-dflash-GGUF` is, per its own card, "the
official DSpark speculative drafter for DeepSeek-V4-Flash-0731 ... (`dflash` arch + DSpark
markov/confidence heads)", 6.5 GB against our 6.6 GB `mtp.*` shards. **Zero new capability**, but a
useful external reference: it reports **16.6 -> 30.5-31.5 t/s (1.84-1.90x), mean accepted 3.5/5** on
code generation. We measure **2.24x** and **tau 3.54** — on a *pruned* target. We are ahead.

### The real DFlash drafters are blocked on the TARGET, not the architecture

`RedHatAI/...speculator.dflash` and the `inference-optimization/*` family share one config:
5 dense llama-style layers, `head_dim 256`, `sliding_window 2048`, `block_size 8`
(`speculative_tokens: 7`), `draft_vocab_size 32000`, 2 B params — and
`aux_hidden_state_layer_ids: [3, 13, 23, 32, 42]`, five taps spread across the backbone, against our
three adjacent ones at [40,41,42].

Architecture and taps are both tractable engine work. **The verifier is not.** Every one of them
declares `verifier.name_or_path: "deepseek-ai/DeepSeek-V4-Flash"` — the unpruned base. Ours, from
`REAP_MANIFEST.json`: `source_model deepseek-ai/DeepSeek-V4-Flash-0731`, revision `9e165c30`,
`original_experts_per_layer 256`, **`kept_experts_per_layer 160` — 37.5 % of the experts removed per
layer.**

A drafter is trained to match one specific target's output distribution. **F89 measured that
changing accumulation order ALONE cost this engine 28 % of its acceptance.** Deleting 96 of 256
experts per layer is an incomparably larger perturbation. Building kernels first and discovering
that afterwards is the expensive order.

**On "DFlash performs best":** LMSYS establishes DFlash > EAGLE-3. But the DSpark paper's own
ablation says a 2-layer DSpark beats a 5-layer DFlash (+16.3 % accepted length), and DSpark is what
we run. The measured comparison is the honest one: RedHat's DFlash gets **~3.9 accepted at 7
positions with a 2 B drafter**; we get **3.54 at 6 with 12.5 B**. **DFlash's real advantage is a 6x
cheaper draft half, not higher acceptance** — and our draft is only 17.4 % of the cycle, so that
advantage is worth at most ~14 % here.

### The portable idea: `draft_vocab_size: 32000` — and why it is NOT buildable yet

FR-Spec vocabulary trimming, which the deep-research workflow independently ranked lever #4 at
+3.0-3.5 %. It needs no retraining and no architecture change. **I was about to build it. The
measurement says do not.**

First sample said 97.4 % of generated tokens fall below id 32000 — from **119 distinct tokens**
(the log prints 40 per run). Re-measured on a real 0.49 M-token corpus:

| K | id-order coverage | freq-ranked coverage | lm_head read |
|---|---|---|---|
| 16000 | 80.75 % | 100 % | 12.4 % |
| **32000** | **88.75 %** | 100 % | 24.8 % |
| 64000 | 95.40 % | 100 % | 49.5 % |

**My assumption that ByteLevel-BPE ids are roughly frequency-ordered is WRONG for this tokenizer.**
The most frequent tokens above id 32000 (`49551, 84293, 59275, ...`) carry **11.25 %** of all
tokens. Priced against the measured cycle:

```
K=32000  saves 4.74 ms = +3.5% of cycle | expected tau loss ~34%  -> NET -31%
K=64000  saves 3.18 ms = +2.3%          | expected tau loss ~15%  -> NET -13%
```

**Every id-ordered variant is net negative**, because the lever pays ~3 % and a coverage miss costs
an entire accepted run.

Frequency ranking fixes coverage (100 % at 16 k on this corpus) — but that corpus is 1.5 MB of *our
own markdown and CUDA source*, 13,366 distinct tokens of 129,280. A ranking fitted to it would
overfit to English technical prose and collapse on anything else. **FR-Spec's method requires a
corpus representative of serving traffic, and we do not have one.**

### The synergy worth recording

The corpus that makes this lever safe is **exactly the corpus S5's capture produces**:
Open-PerfectBlend prompts with responses regenerated by the target. That capture yields the token
frequency distribution of the target's own outputs as a by-product — which is precisely the FR-Spec
ranking, measured rather than assumed. **So the draft-vocab trim is not dead; it is DOWNSTREAM of
S5's capture, and free once that exists.**

## Finding 98 — the remaining kernel queue re-sized against the CURRENT engine: **three of four items are worse than advertised, two are unsafe, and the queue is done**

Fresh `DSV4_DPROF` at the current default (block 6, post-F93/F96), `evidence/dprof_blk6_current.log`.
The pots the deep-research workflow named are REAL and stable at the new operating point:

| pot | ms | % of a 69.5 ms step |
|---|---|---|
| `hc_pre` (attn 1.67 + ffn 2.78) | 4.45 | 6.4 % |
| `rmsnorm` (0.75 + 0.78) | 1.53 | 2.2 % |
| `hc_post` (0.25 + 0.26) | 0.51 | 0.8 % |
| **item 2 total** | **6.49** | **9.3 %** |
| `moe:router` + `moe:group` | 3.52 | 5.1 % |

**But the pots are not the gains.** Sized against mechanism rather than against the pot:

### Item 2 (mHC big-fusion) — ~0.9 %, not 2.2-4.1 %, and it cannot be bit-exact

`k_mixes` is a `[24 x 16384] x [16384]` matvec decomposed one block per output `m`, so **each of the
24 blocks re-reads the whole 64 KB `x[t]`** and recomputes the identical `sq` norm. Traffic is
3.15 MB/call where 1.64 MB would do — a 48 % cut, worth **0.56 ms/step (0.8 %)**, plus 172 fewer
launches at the measured 0.5 us node floor (0.09 ms, 0.1 %). **Total ~0.9 %.**

**And the fusion cannot preserve bit-exactness.** Reading `x` once requires one block computing all
24 `m`, which changes the thread->element mapping and therefore the accumulation order. **F89
measured a 28 % acceptance loss from exactly that class of change** in the compressor path. A 0.9 %
gain does not justify reopening that risk in the hyper-connection path, which feeds every layer.
**NOT BUILT.**

### Item 3 (MoE routing collapse) — highest risk in the list

Requires reproducing `noaux_tc` + `sqrtsoftplus` + group-limited routing bit-exactly. Routing
decides **which experts run** — that is not a numerics nudge, it is a different computation, and a
near-miss changes the expert union rather than the last mantissa bits. **NOT BUILT.**

### Item 8 (CUDA-graph the draft half) — blocked on a host-built index

`kernels/dspark_attn.cu:57` builds the sparse index on the HOST and H2D-copies it per round:
`cudaMemcpyAsync(idx, hidx.data(), ..., HostToDevice, stream)`. A graph would freeze those contents
at capture time, and `hidx` depends on `block`/`n`/`wstart`, which change every round. Capturing it
requires moving index construction on-device first (as the prefill path already does). **Not the
"low-medium" the workflow assumed.**

### What is left

Item 6 (Programmatic Dependent Launch) is the only remaining item that is bit-exact by construction
and not blocked — but it is ~245 mechanical call sites for a **+0.6 % floor / +2.0 % ceiling**, and
GPUOS's own limitations section puts the hard ceiling on ALL dispatch elimination at
**2788 nodes x 100 ns = 1.6 %, forever**.

**So the kernel queue is done, and this is the second independent time that conclusion has been
reached** — first at F83 when `open_nontraining_levers` hit 0, now at a finer grain with the
workflow's own ranked list re-sized against measurement. The difference is that this time the
remaining items were priced and rejected individually, with reasons, rather than exhausted.

**The binding constraint is no longer kernel efficiency. It is acceptance, and acceptance is a
training problem.**

## Finding 99 — GATE 1 CLOSED: the training loop runs end to end. The port was 6 rotary sites + one `*=`, and the fix that mattered was `retain_graph`

`train/_model/model.py` ported for autograd; `train/train_head.py` completes.

```
[train] load_state_dict: 0 missing, 0 unexpected
[train] step 1: T=1022 positions=4 loss=1.0521 ce=10.5213 grad_norm=2310.5471 tensors_with_grad=68/72
[train] STAGE0 OK (real forward + backward)
```

### What the port actually was

F91 counted ~16 in-place sites and called it a port of the whole file. **Only 8 are on the DSpark
path**, and the backbone's (lines 373-539) are never executed by us — this file is also the
inference reference, so leaving them alone is correct, not laziness.

* **6 rotary sites.** `apply_rotary_emb` ends in `y.copy_(x)`, an in-place write into a SLICE VIEW.
  Added an out-of-place twin + a `rope_tail(x, rd, ...)` helper that rebuilds the tensor with only
  its last `rd` columns rotated. **The arithmetic is copied verbatim — only the write differs — so
  the forward value is unchanged and this is NOT a numerics change**, which matters given F89.
  **Rotary could NOT take the `.data.copy_` straight-through trick used for `act_quant`**: simulated
  quantisation is a small perturbation whose identity gradient is the standard QAT approximation,
  while a rotation's identity gradient is simply wrong.
* **One `q *= rsqrt(...)`** where the multiplier depends on `q` — overwrites an operand that
  multiplication's own backward still needs.
* **`logits[:, i].add_()`** is in `forward_head`, already replaced by the teacher-forced variant.

### The last blocker was structural, not syntactic

After the port, `backward` raised *"Trying to backward through the graph a second time"*. Phase A
(the KV-cache fill) runs ONCE per sequence outside the position loop, so every position's graph
traces back through it and the first backward frees it. Fixed with
`backward(retain_graph=(t != sel[-1]))` — keep the shared phase alive, let only the last position
free it. **That is a property of the two-phase design (F91), not a bug**, and it is the reason the
cheap O(T) cache fill is shared at all.

### NOT DONE — two parts of the loss are inert, and the run above does not train anything meaningful

1. **`tv=0.0000`.** The TV term is **90 % of the loss weight** and is currently zero because the
   trainer passes `lg.detach()` as its own target. The real target distribution must be reconstructed
   from the captured taps through the FROZEN `lm_head` — tap[2] is layer 42, the last backbone layer,
   so `p ~ softmax(head(norm(hc_head(tap[2]))))`. Until that is wired, the run optimises CE alone.
2. **The confidence head is unused** (`conf=None`), so `a_conf = 1.0` of the objective is absent and
   4 of 72 tensors get no gradient — almost certainly its projection. Needs accepted/rejected labels,
   which the capture must record alongside the taps.

**So Gate 1 is mechanically closed and session 1 is NOT yet runnable.** Wiring those two is the last
work before the 500-sample narrow proof, and both are contained: one reconstruction from data we
already capture, one extra field in the capture record.

## Finding 100 — TV target wired (it was NOT reconstructible from the taps), confidence term measured and DELIBERATELY GATED OFF, and the head-preservation machinery built

### The TV target: F99's plan was wrong and the code says why

F99 recorded "recompute the target distribution from the taps through the frozen lm_head". **The
taps cannot do that.** `k_tap_pool` writes `h.mean(dim=hc)` — an UNWEIGHTED mean — while the real
lm_head input is `hc_head(h, hc_fn, hc_scale, hc_base)`, a LEARNED Sinkhorn-normalised combination
followed by `rmsnorm`. No arithmetic turns one into the other.

**Capture format v2** stores the real thing: the same `hc_head`+`rmsnorm` the `head_fwd` path runs,
over all prefill positions. The expensive part — the 129,280-column lm_head GEMM — stays in
training as one matmul against a frozen weight. Cost **+8 KB/token (24 -> 32.0 KB/tok, measured)**;
storage was never the constraint. The trainer now **hard-fails on a v1 capture** rather than falling
back, because a `tv=0.0000` that looks like a working run is exactly what hid this through Gate 1.

Measured with all terms live: **ce 10.43, tv 0.9254** (was 0.0000), 69/72 tensors with gradient.

### The confidence term is measured, understood, and OFF

`conf = 10034` against ce 10.43 and tv 0.93 — **~1000x the other two.** Two causes, neither a
coding error:

1. The head takes the **un-normalised** `x` (pre `self.norm`), whose scale is large — the captured
   taps have std ~190 — so its raw logit is large. `binary_cross_entropy_with_logits` is the correct
   function; the head genuinely returns a logit, not a probability.
2. It was trained to predict acceptance under **free-running** drafting, and teacher forcing makes
   almost every position accepted, so it is confidently wrong on this data **by construction**.
   **That is the HASS mismatch appearing in the confidence signal rather than in the draft input.**

Fixing it needs free-running draft labels — session 2's HASS work — not a scale factor. So
`--a-conf` defaults to **0.0** (the paper's value is 1.0) and sessions train on ce+tv, which are
correct. **Shipping a loss term 1000x the others without understanding it is how a plausible wrong
answer gets produced.**

### The product was half unprotected

The product of this project is two things: **the CUDA engine** (in git, therefore safe) and **the
speculator weights** (not in git, therefore at risk). A winning head would have been an untracked
`.safetensors` in a scratch directory, its measurement in a log due to be overwritten and its
lineage nowhere.

`tools/promote_head.py` + `HEAD_REGISTRY.md` fix that, and **the selection rule is fixed now, before
any candidate exists, so it cannot be fitted to a result**:

1. `LOSSLESS GATE ... PASS` must appear — a head that changes the emitted sequence is disqualified
   at any tau. **F68 is why.**
2. First-token `GATE PASS`.
3. The frozen protocol: 8-prompt suite, NGEN0>=200, block 6, adaptK 1.50, **clean run** (the tool
   refuses a log containing `[dprof]`/`[specprof]`/`[ksweep]`/`[moebytes]`).
4. Suite mean must beat the incumbent by more than the **measured 3.5 % run-to-run spread** (F94).
   **Ties go to the incumbent.**
5. Prompt 0 is a CONTROL and is excluded from the mean (F96).

Promotion copies the weights to `~/model-backups/heads/<name>/` with **sha256 per file**, the eval
log, and a `head_card.json` carrying the engine git rev, the base model and its REAP parameters
(256 -> 160 experts), the full per-prompt measurement, and the incumbent it beat.

Registry seeded with the shipped head as the incumbent from the F96 clean baseline:
**suite tau 3.5362, 22.6550 tok/s, base AR 13.76.** Every future head must beat 23.45 tok/s
(22.655 x 1.035) to be promoted.

## Finding 101 — SESSION 1 HALTED before capture: the ported forward reproduces the engine at **1.5 %** where the engine gets ~30 %. The equivalence gate fired, and it fired for free

The two-pass flow works. The trainer runs end to end on real reasoning data. **And the numbers say
do not use it.**

### What was built and verified

* `DSV4_GENOUT` (full prompt+generated id dump; the 40-token console print is unparseable) and
  `DSV4_PROMPTS_FILE` (500 prompts x ~200 ids exceeds the environment-variable limit).
* 500 reasoning prompts from **Open-PerfectBlend** — the corpus the DSpark paper used — via
  MetaMathQA / UltraInteract_sft / orca-math, tokenizer gate passing.
* Trial at 10 samples: **pass 1 round-trips 10/10 with exact seed prefixes**; **pass 2 captures
  10/10 at 32.0 KB/token** with distinct per-slot std; the trainer completes all 10 shards.
* Two more autograd defects fixed, both from in-place cache writes: phase A had to run under
  `no_grad` (its history lives in a module BUFFER and survives across shards, which `retain_graph`
  defers rather than fixes), and phase B's own cache writes need a per-position `detach_()`.

### The measurement that stops it

```
[diag] draft-argmax == TARGET-argmax : 1.5%   | == ground-truth token: 1.5%
[diag] mean top-1 prob  target 0.762  draft 0.309
```

The engine measures **tau 1.85 at block 6 on `reasoning`** — roughly **30 % per-position
acceptance**. The ported forward agrees **1.5 %** of the time. That is a **20x discrepancy**, and it
is not explained by a broad draft distribution: the draft's mean top-1 probability is 0.309, which
is peaked, not flat. **The PyTorch draft path is not computing what the engine computes.**

This is precisely the failure F91 named — *"a ported forward that differs from the server by more
than that is worse than no fine-tune"* — and F89's 28 % acceptance loss from an accumulation-order
change is the calibration for how little divergence this system tolerates.

**Training on this would align the head to a function the server does not run.** Session 1 is
halted before its 4.7 h of capture, not after.

### The TV number is therefore also not trustworthy

`tv = 0.95-1.0` looked like a striking result and I nearly acted on it: at a measured
`alpha = 1 - TV = 0.03`, Nebius's hybrid gives `lambda = exp(-3 x 0.03) = 0.91`, i.e. **91 % KL** —
which would invert F100's "do not adopt the LK hybrid" conclusion. **That inversion is now
suspended**, because a TV computed against a draft that is 20x off is measuring the port's bug, not
the objective. F100's arithmetic used `alpha = 0.692` inferred from tau; the honest position is that
**neither number is established** until the forward matches.

### What the gate needs, and why it must come before capture

The equivalence check requires the ENGINE's own draft proposals for the same input, which the
capture does not currently record. Concretely: add the draft's per-position argmax (and ideally its
top-1 margin, which `hmarg[]` already computes) to the capture record, then require
**draft-argmax agreement >= 90 %** between engine and port before any session runs.

**Cost of finding this here: about twenty minutes. Cost of finding it after session 1: the session,
plus a trained head that looked plausible and was aligned to nothing.** The 10-sample trial paid for
itself, and so did adding a diagnostic instead of trusting a suspicious number.

## Finding 102 — the equivalence gate is built and the port scores **0/288**. It is fundamentally wrong, not marginally, and guessing has been the wrong method

### The oracle exists now

`DSV4_CAPTURE` also writes `drafts_pNN_sNNNNNN.txt`: for 16 positions spread across the captured
range, the ENGINE's own `t`, seed token, `BLK` draft ids and `BLK` margins, produced by re-running
the identical draft chain (`dspark_main_kv` -> `k_embed` -> `k_hc_expand` -> 3x
`dspark_block_forward` -> `dspark_forward_head`) against the KV the prefill already built.

**It had to be a probe, not a log of the decode loop.** The capture covers prefill positions
`0..PSp-1` while spec decode visits positions AFTER `PSp`; logging the loop yields drafts at
positions we hold no taps for, and the two sets never intersect. Getting `anchor` right mattered
too — the spec loop uses `anchor = cpos-1, ctx = cpos`, so a probe at `pctx = t+1` needs
`panchor = t`. A gate calibrated against wrong drafts is worse than no gate.

**Sanity check on the oracle itself:** the engine's drafts match the actual next tokens
**48-59 %** of the time on self-generated text. That is the number the port has to reproduce.

### The verdict

```
[gate] PORT vs ENGINE draft agreement: 0/288 = 0.0%  (need >= 90% before any session runs)
```

**Zero.** Not the 1.5 % of F101 — that run compared a 5-wide port against a 6-wide engine, which was
itself a real defect (`inference/config.json` says `dspark_block_size: 5`, the width the head was
TRAINED at, while the engine's serving default has been 6 since F94/F96; the port now takes
`--block`, defaulting to 6). Fixing that did not help, which is itself information: **the block
width was not the cause.**

0/288 against a 129,280-token vocabulary is not an off-by-one. An indexing error would still agree
on the easy tokens. The port's draft path and the engine's share essentially nothing.

### What I am NOT going to do

Guess again. Four guesses this session were wrong — the STE, the seed clone, `retain_graph`, and now
the block width — and each cost a run. The candidates left (RoPE offset, `start_pos`/`anchor`
semantics, phase-A cache contents, `main_x` slicing) are indistinguishable from the outside.

**The next step is intermediate-tensor comparison**, and it needs the engine to dump `main_x`, each
block's output and the pre-bias logits at one probe position. Then the first tensor that diverges
names the bug in one run instead of four. That is bounded instrumentation, and it is the same move
that turned F101 from a mystery into a measurement.

### What this cost and what it bought

**Cost: no capture time at all.** The gate fired before session 1's 4.7 hours, which is exactly what
it was placed there to do. **Bought:** a permanent, cheap oracle that any future port change is
scored against automatically, plus the knowledge that the engine's own draft accuracy on
self-generated text is 48-59 % — a number nothing else in this project had.

## Finding 103 — BISECT SUCCEEDS: everything up to `after_embed` is EXACT, and the first DSpark block explodes by 18557x. Four guesses became one measurement

`DSV4_PROBE_DUMP` writes the engine's intermediate tensors at one probe position; `--bisect` diffs
the port against them stage by stage.

```
[bisect] engine probe: t=8 seed=39317 block=6
[bisect] port seed=39317 (engine 39317) MATCH
  main_x_t         cosine=+0.999671 rel_err=0.0257        OK
  after_embed      cosine=+1.000000 rel_err=0.0000        OK        <- EXACT
  after_block0     cosine=+0.193362 rel_err=18557.0156    **DIVERGED**
  after_block1     cosine=+0.110499 rel_err=8702.6494     **DIVERGED**
  after_block2     cosine=+0.045801 rel_err=2879.2866     **DIVERGED**
```

### What this eliminates, and it is most of the search space

* **The seed is right** (39317 both sides) — so `t`, the id indexing and the capture alignment are
  correct.
* **`main_x` is right** (cosine 0.9997; the 0.026 is bf16 rounding) — so `main_proj`, `main_norm`,
  the FP8 128x128 block-scale dequantisation and the tap layout are all correct. **The whole
  capture-v2 chain is validated by this line.**
* **`after_embed` is EXACT** (cosine 1.000000, rel_err 0.0000) — so `forward_embed`, the noise-token
  fill, the embedding and the hc expansion are correct, and the port's block width now matches.

**All four of my guesses from F102 are dead.** RoPE offset, `start_pos`/`anchor`, phase-A cache
contents and `main_x` slicing all sit upstream of or inside a stage that is provably exact.

### What it isolates

`dspark_block_forward` — one function. And `rel_err = 18557` with cosine 0.19 is a **magnitude
explosion**, not a rotation or an off-by-one: an indexing error preserves scale and loses direction,
while this loses both. The prime suspects are therefore the two things in that block that carry a
SCALE and that the port reimplements rather than inherits:

1. **`hc_split_sinkhorn` in `train/kernel.py`** — my transcription of the tilelang kernel. If `comb`
   is under-normalised the residual compounds across 4 streams x 3 blocks, which is exactly the
   shape of a 4-figure blow-up. I already got this function wrong once (F-port: hc_scale is a
   3-vector, `pre` adds eps, `post` carries a factor 2, and `comb` starts with a row softmax) and
   corrected it by transcription — but transcription was never verified against a number.
2. **The MXFP4 expert dequantisation** — though `main_x` being right narrows this, because the FP8
   path is proven and only the MXFP4 nibble/scale path is untested.

### Why this was worth building instead of guessing again

Four guesses cost four runs and were all wrong. **One dump plus one comparator cost two runs and
eliminated the entire upstream half of the pipeline**, including — as a bonus — an independent
validation that capture format v2 is correct, which nothing else had checked.

The next bisect goes one level deeper: dump inside `dspark_block_forward` (post-attention,
post-`hc_post`, post-MoE) and the same procedure names the sub-stage.

## Finding 104 — the bug is `DSparkAttention`, and the Sinkhorn shim I suspected is EXONERATED at cosine 1.000000

Sub-stage dumps inside `dspark_block_forward` (8 stages, `g_dspark_dump`, first block only) plus a
comparator that calls the reference's own `hc_pre`/`hc_post` rather than reimplementing them:

```
main_x_t           cosine=+0.999671   OK
after_embed        cosine=+1.000000   OK
b_hcpre_attn_x1    cosine=+0.999999   OK
b_hcpre_attn_post  cosine=+1.000000   OK
b_hcpre_attn_comb  cosine=+1.000000   OK      <- hc_split_sinkhorn is CORRECT
b_rmsnorm_attn     cosine=+0.999998   OK
b_attn_out         cosine=+0.203301   rel_err=30161   **DIVERGED**
```

### My prime suspect was wrong, and that is the point of measuring

F103 named `hc_split_sinkhorn` as the leading candidate: I had already got it wrong once, fixed it by
transcription, and never verified the transcription against a number. **It is exact** — `comb` at
cosine 1.000000, `post` at 1.000000. The transcription was right. Had I "fixed" it on suspicion I
would have broken a correct function and still had the bug.

**Five guesses this session, five wrong** (STE, seed clone, `retain_graph`, block width, Sinkhorn).
Every one of them was resolved by a measurement, and the measurements cost less than the guesses.

### What is isolated now

`DSparkAttention.forward` — and `rel_err 30161` says magnitude, not orientation. The two things in
that path the port supplies differently from the engine:

1. **`sparse_attn` in `train/kernel.py`** is hand-written (gather + einsum + sink folded as an extra
   zero-value logit). Checked by inspection against `mla_attn.cu`: the sink is applied UNSCALED
   while scores are scaled, which matches; masking by `-inf` matches `if (idx<0) continue`.
2. **The KV context.** The engine passes `main_kv` PRECOMPUTED by `dspark_main_kv` over the whole
   context; the reference computes it inside from `main_x` and accumulates into `self.kv_cache`.
   The gate feeds phase B a single position (`main_hidden[:, t:t+1]`), so if phase A did not leave
   the window populated the port attends to **one** key where the engine attends to up to
   **WINDOW=128** — which would change the output magnitude by orders of magnitude, exactly the
   symptom.

(2) is the stronger hypothesis precisely because it predicts a magnitude error rather than a
direction error, and `cosine 0.20 / rel_err 30161` is a magnitude error.

### Method note

Two rounds of bisect, four runs total, and the search went: whole pipeline -> one function -> one
sub-stage, with a correct component cleared along the way. The next round dumps inside
`dspark_attn_forward` (q, kv, the topk index, pre-softmax scores) and the same procedure will
separate "my sparse_attn is wrong" from "the port's attention sees the wrong context".

## Finding 105 — THE PORT IS FIXED. `freqs_cis` is a `persistent=False` buffer that `to_empty()` zapped: the port was doing every rotary embedding against **uninitialised memory**. 0/288 -> **265/288 (92.0 %)**

### The bug

```python
self.register_buffer("freqs_cis", freqs_cis, persistent=False)
```

`persistent=False` excludes a buffer from `state_dict` **by design**. The training loader calls
`to_empty(device=dev)` to materialise the meta-device modules, which replaces every buffer with
**uninitialised memory**, and then `load_state_dict` fills only what the checkpoint contains —
which, correctly, does not contain `freqs_cis` (it is computed, not stored; the checkpoint has no
`freqs*` tensor at all).

**So `load_state_dict` reported `0 missing, 0 unexpected` — truthfully — while the rotary tables were
garbage.** No error, no warning, no missing key. The port did every RoPE against noise.

### Why the bisect found it and five hypotheses did not

The signature was there once the sub-stages were compared:

```
b_rmsnorm_attn   cosine=+0.999998  OK     <- input to attention is EXACT
a_q              cosine=+0.400545  DIVERGED
a_bkv            cosine=+0.379183  DIVERGED
```

**Both `q` and the block-KV diverge from an exact input, and their only shared step is
`rope_tail(..., freqs_cis)`.** `main_x` and the whole `hc_pre` chain were fine precisely because
neither applies rope. That is a two-line deduction from the data, and it is not one I would have
reached by inspection — I had already cleared RoPE as a candidate in F102 on the grounds that an
indexing error preserves magnitude, and this was not an indexing error.

### Result

| stage | before | after |
|---|---|---|
| `a_q` | 0.400545 | **0.999574** |
| `a_bkv` | 0.379183 | **0.999782** |
| `a_kv_all` | 0.433926 | **0.999429** |
| `b_attn_out` | 0.193596 (rel_err 29435) | **0.998897** (rel_err 0.047) |
| `after_block2` | 0.046774 | **0.997544** |

```
[gate] PORT vs ENGINE draft agreement: 265/288 = 92.0%  (need >= 90%)
```

**PASSES.** The residual ~8 % is bf16 rounding compounding through three blocks plus greedy-argmax
ties — the per-stage `rel_err ~0.05` is exactly bf16, and no stage reads `**DIVERGED**`.

### The transferable lesson

**`to_empty()` silently invalidates every non-persistent buffer, and `load_state_dict` cannot warn
about it** — from its point of view nothing is missing. Any meta-device materialisation must
re-run whatever computed the non-persistent buffers. This is a general PyTorch hazard, not
something specific to this model.

It also belongs to the pattern this session has hit repeatedly: **the failure was silent and
plausible.** Six hypotheses (STE, seed clone, `retain_graph`, block width, Sinkhorn, ring-wrap) were
wrong; three rounds of bisect and four dumps found it. Two of those six — block width and ring-wrap —
were **real bugs that were not this bug**, and fixing them changed nothing, which is exactly why
guessing is expensive here: a plausible fix that changes nothing looks like evidence against the
next hypothesis.

**Session 1 is unblocked.**

---

## Finding 106 — the automation audit: three defects on the path between "training finished" and "a head the engine can load", all of which would have surfaced only *after* the 5-hour capture was paid for

The request was to make passing sessions chain automatically, save every meaningfully-produced
head, and record per-run metrics. Building that orchestrator meant reading the whole path end to
end for the first time — and the path had never been *run* end to end, because every session so far
stopped at "does a backward pass work". Three things were waiting in it.

### (a) The trainer could not save. A `return` from stage 0 was still in front of the save block

```python
    print("[train] STAGE0 OK (real forward + backward)"); return

    os.makedirs(a.out, exist_ok=True)          # <- unreachable
    save_file(save, os.path.join(a.out, "mtp_trained.safetensors"))
```

Left over from stage 0, when the only question was whether autograd reached the trainable tensors.
Session 1 would have run its full training pass, printed a falling loss curve, printed
**`[train] STAGE0 OK`**, and exited 0 with **no weights on disk**. The orchestrator's next step
would have failed on a missing file, hours after the capture that produced it.

The tell was never going to be an error. It was going to be a success message.

### (b) `--warmup` was a flag that did nothing

`--warmup 0.05` was parsed and never read; there was no scheduler, and every step ran at a flat
`--lr 5e-5`. Fine-tuning a *pretrained* head at a flat peak rate from step 0 is a standard way to
damage the thing you are trying to preserve. Implemented as linear warmup into a cosine decay over
the known step count (`len(rows)`, since one epoch = one pass over the capture).

Same failure class as F105: no error, plausible output, wrong function.

### (c) The re-quantiser wrote a scale the engine decodes to a different number

`build_trained_head.py` has to write trained bf16 back as FP8 e4m3 + block scale, because that is
what the engine reads. The scale dtype in the checkpoint is **`F8_E8M0`** — a bare exponent, so the
only representable values are exact powers of two:

```
mtp.0.attn.wq_a.weight  F8_E4M3  [1024, 4096]
mtp.0.attn.wq_a.scale   F8_E8M0  [8, 32]        <- 128x128 blocks, confirmed
```

The first version computed a continuous `scale = amax/448`, divided the weights by **that**, and
then rounded the exponent when storing. The engine dequantises with `exp2(byte - 127)` — the
*rounded* value. Divisor and stored scale therefore disagree by up to sqrt(2), per block, on every
FP8 weight in the head.

**No gate in this project would have caught it.** The head would load, run, produce tokens, pass
the LOSSLESS gate (verification is against the *target*, which is unaffected), and simply draft
worse. The result would have read as "fine-tuning didn't help" and could plausibly have ended S5.

Fixed by snapping the scale to a power of two **first** and quantising against that exact value
(`ceil`, not `round` — rounding down pushes `|t|/scale` past 448 and clamps real weights). Two
checks now exist:

| check | result |
|---|---|
| `--selftest`, synthetic, decoded with the engine's `exp2(byte-127)` over raw bits | worst rel err **0.0524** |
| per-tensor round-trip on the real trained data, at build time, refuses above 0.10 | wired |

0.0524 is e4m3's own half-step (~0.0625). That is the floor, and it is the checkpoint's native
precision for these tensors — not additional quantisation.

**Why the selftest decodes raw bytes rather than calling `.float()`:** the question is not whether
torch round-trips its own dtype. It is whether the byte we wrote means, *to the reader*, the number
we intended. The reader is the engine.

### What the orchestrator now enforces

`scripts/s5_session.sh` — one script, seven steps, each verifying its own output before the next:

1. **pass 1** generate — plus a round-trip check that every generated line *starts with its exact
   seed prompt*, which is what F101's truncated dump failed
2. **pass 2** capture — `read_capture.py` must print `VALIDATE: PASS`
3. **equivalence gate** — port-vs-engine agreement must be >= 90 % **before** training, not after
4. **train** — with the schedule from (b) and the save from (a), plus `--metrics-out`
5. **build** — the write-back from (c), with its round-trip gate
6. **eval** — the frozen protocol, `LOSSLESS GATE` mandatory
7. **archive always, promote on merit, chain only on GO**

### The chain rule is not the promotion rule, and conflating them would have ended S5 on session 1

`promote_head.py` asks "is this better than the incumbent, across the whole suite, by more than the
3.5 % run-to-run spread". `tools/session_gate.py` asks the earlier question: **did training move the
thing this session was built to test?** Session 1 is a deliberately narrow single-domain proof — it
can legitimately say GO while failing a suite-wide promotion bar, and gating the chain on promotion
would have discarded exactly the evidence the session was designed to produce.

Thresholds are transcribed from `S5_PROGRESSION.md` §2, written before the data exists:
`reasoning tau >= 2.6` GO, `2.1-2.6` GO but re-price, `< 2.1` STOP, suite mean drops STOP,
LOSSLESS fails STOP.

**The gate validates against itself.** Run on the *untrained* F96 baseline it reproduces every
category exactly — long_context 5.54, agentic_format 5.15, multi_turn 4.46, code_edit 4.43,
short_factual 3.27, reasoning 1.85, code_gen 1.84, explanation 1.75, mean 3.5362 — and returns
**STOP**, which is the correct verdict for a head that has not been trained. That agreement also
independently confirms the prompt-index -> category map, which was established by *decoding*
`protocol/suite_prompts.txt` with the checkpoint's own tokenizer rather than by matching token
counts (two prompts are 15 ids long; a count-match would have had to guess between `explanation`
and `multi_turn`).

The suite also moved out of a session-scoped temp directory into `protocol/suite_prompts.txt`. A
frozen measurement protocol that lives somewhere it can be garbage-collected is not frozen.

---

## Finding 107 (PRELIMINARY, n=47) — the draft-vocab trim is measured on the target's own outputs at last, and it is **net negative at every corpus size we could capture**. F97 said this lever was downstream of S5's capture; the capture is now producing the data, and the answer is no

F97 killed the id-ordered variants and named the one condition that would revive the lever: a
frequency ranking fitted to a corpus *representative of serving traffic*, which S5's capture
produces as a by-product. Session 1's pass 1 is generating exactly that corpus right now, so the
ranking is buildable from partial data — 47 sequences, 27.7 k of the target's own generated tokens.

`tools/vocab_rank.py`. The discipline it exists to enforce: **the ranking is fitted on a train split
and scored on held-out sequences it has never seen**, because a ranking scored on its own corpus
reports ~100 % by construction. F97 was already shown that shape once and correctly refused it.

```
      K   fitted %  HELD-OUT %     gap
   8000   100.000%     78.226%  21.774      <- fitted number is meaningless
  64000   100.000%     78.226%  21.774
```

**Held-out coverage is flat in K.** That is the whole story: the ranking has 1 922 entries, shorter
than any K tested, so the miss is not "tokens ranked too low" — it is **tokens never seen**. K is not
the variable. Corpus size is.

### The curve, and it does not get there

| train seq | distinct | held-out coverage | miss |
|---|---|---|---|
| 4 | 333 | 52.7 % | 47.3 % |
| 9 | 930 | 63.6 % | 36.4 % |
| 18 | 1 416 | 72.0 % | 28.0 % |
| 37 | 1 922 | **78.2 %** | **21.8 %** |

Miss multiplies by **0.784 per doubling** of corpus — a geometric decay against an exponential
requirement. Reaching the ~0.1 % miss this lever needs takes **~2.7 x 10^8 sequences.**

### Priced in tau, so the verdict is computed rather than asserted

Model, assumption stated: per-position acceptance independent with probability p, tau = 1 + sum
p^i over the block (the 1 is the bonus token, which this project's tau includes). p0 = 0.7578
reproduces the measured tau 3.5362 at block 6. A vocabulary miss ends a run exactly as a rejection
does, so it multiplies per-position acceptance by (1-m). Gain column uses **K=64000, the most
conservative trim tested**.

| sequences | miss | tau | tau cost | ms saved | **net** |
|---|---|---|---|---|---|
| **500** (session 1) | 9.01 % | 2.98 | -4.25 tok/s | +1.04 | **-14.2 %** |
| 2 000 (session 2) | 5.61 % | 3.18 | -2.76 | +1.04 | **-7.6 %** |
| 5 000 (session 3 max) | 4.11 % | 3.27 | -2.06 | +1.04 | **-4.5 %** |
| 20 000 | 2.56 % | 3.37 | -1.31 | +1.04 | **-1.2 %** |
| 100 000 | 1.48 % | 3.44 | -0.77 | +1.04 | **+1.2 %** |

**The lever does not cross zero until ~100 000 sequences, and it buys +1.2 % there.** At the
measured generate rate that capture is ~200x session 1 — over a month of wall time — for a gain
smaller than the 3.5 % run-to-run spread this project refuses to promote inside.

### Why the estimate is OPTIMISTIC, which matters more than its being preliminary

The held-out sequences come from the **same domain** as the training ones (reasoning, from
MetaMathQA / UltraInteract / orca-math). Real serving traffic spans the eight suite categories.
Cross-domain coverage can only be worse than within-domain coverage, so the true miss rate at any
corpus size is **above** these numbers and the true net is **below** them.

### Status

**PRELIMINARY: n=47 sequences of one domain.** F94 is this project's own precedent for a number
measured on a convenient subset and quoted as general — it reported +3.3 % on four prompts chosen
because they were ceiling-bound, against +1.6 % on the full suite. So this is recorded as a strong
signal and not a closure. `tools/vocab_rank.py --curve` re-runs in one command against the full 500
when pass 1 lands, and against session 2's 2 000 after that.

If the full-corpus curve confirms it, the entry in `LEVERS.md` moves from *blocked* to **closed,
twice measured** — and the useful part of that is knowing not to spend the capture on it.

---

## Finding 108 — the session-1 GO/NO-GO gate is calibrated on an **outlier prompt**. On 63 real reasoning prompts the *untrained* head already clears the bar, at matched generation length

Session 1 exists to test one premise: that `reasoning` is a collapsed category at tau 1.85 and that
training can repair it. The premise comes from **one prompt** — index 7 of the frozen suite, the
bat-and-ball question. Pass 1 is currently generating 500 real reasoning prompts
(MetaMathQA / UltraInteract_sft / orca-math) with the **untrained** head, which makes the premise
checkable for free, from data already on disk, before the session's answer exists.

### 63 sequences, untrained head, block 6

| | tau |
|---|---|
| min | 1.95 |
| p25 | 3.41 |
| **median** | **4.13** |
| p75 | 4.69 |
| max | 5.63 |

**Every one of the 63 is above the suite's reasoning point of 1.85.** 95.2 % are above the session-1
GO bar of 2.6 — with no training at all.

### Ruling out the obvious confounder, which is real but does not explain it

These ran at NGEN=512; the protocol measures at NGEN0=200, and F92 established that tau climbs as
the drafter's 128-token sliding window fills. So the comparison is only fair at matched length. The
per-verify records in the log give it exactly, by truncating each sequence at a token budget:

| budget | median tau | mean tau | share >= 2.6 |
|---|---|---|---|
| 32 | 2.667 | 2.742 | 57.1 % |
| 100 | 3.281 | 3.309 | 88.9 % |
| **200 (the protocol)** | **3.483** | **3.427** | **88.9 %** |
| 300 | 3.553 | 3.607 | 92.1 % |
| 512 | 4.095 | 4.001 | 95.2 % |

The length effect is real — 2.67 at 32 tokens to 4.10 at 512, which independently reproduces F92's
shape on new data. **But at the protocol's own 200 tokens the median is still 3.483 against the
suite prompt's 1.85.** The suite's reasoning prompt is below the minimum of 63 samples of its own
category.

### Both numbers are true; the question is which is representative

A bat-and-ball trick question is answered by deliberating, and deliberation is where a drafter has
least to work with. Math and multi-step solutions are structurally repetitive — LaTeX, enumerated
steps, restated quantities — and that is exactly what speculation feeds on. So 1.85 is a real
measurement of a hard prompt, not an error. What it is *not* is a measurement of the category, and
the session-1 plan reads it as one.

### What this changes, and what it deliberately does not

**It does not change the pre-registered gate.** Adjusting a threshold after seeing data is the exact
failure the pre-registration exists to prevent, and the rule in `S5_PROGRESSION.md` §2 stands as
written: suite reasoning prompt tau >= 2.6 -> GO. It will be reported.

**It does add a second, better-powered gate, declared now, before any trained head exists**, with
its baseline measured on the untrained head:

> **Secondary gate.** Median tau over a **32-prompt reasoning hold-out**, never trained on, at
> NGEN0=200, block 6, clean. **Untrained baseline: 3.483** (n=63, this finding).
> **GO at >= 3.83** (+0.35 tau, which at the measured 7.664 tok/s per unit tau is +2.7 tok/s or
> +11.8 % — a lift worth a session). Below +0.15 tau, the category is saturated and no amount of
> reasoning data will move it.

The hold-out must be **excluded from the training capture**, or it measures memorisation. The
orchestrator now reserves the last `S5_HOLDOUT` sequences of the corpus for this and never trains
on them.

### The wider consequence for how this project reads its own suite

Each of the three "collapsed" categories — reasoning 1.85, code_gen 1.84, explanation 1.75 — is
**one prompt**. This finding shows one of them is unrepresentative by a factor of ~1.9 at matched
length. The suite mean of 3.5362 is still the right *comparison* number, because it is the same
eight prompts every time and that is what makes runs comparable. But it is a fixed benchmark, not
an estimate of workload behaviour, and the two have been conflated whenever a category number was
read as "this is how the engine does on reasoning."

---

## Finding 109 — what "30 tok/s" actually requires, derived from measured verify economics. The adaptive-K controller is already near-optimal (+0.4 %), so the entire remaining path runs through **margins**, and the block-6 hard ceiling is **34.77**

Pass 1 emits, per verify, the draft's six logit margins, the chosen width K, the accepted count and
the wall time. 15 800 of them on the reasoning corpus with the untrained head is enough to replace
every estimate in this project's speed model with a measurement.

### 1. Where the time goes

| K | share of verifies | share of time | tokens / max | tok/s |
|---|---|---|---|---|
| 2 | 23.9 % | 17.3 % | 1.64/2 = 81.9 % | 14.57 |
| 3 | 17.0 % | 13.9 % | 2.24/3 = 74.6 % | 17.69 |
| 4 | 11.5 % | 10.5 % | 2.99/4 = 74.7 % | 21.14 |
| 5 | 8.2 % | 8.3 % | 3.81/5 = 76.1 % | 24.33 |
| 6 | 7.7 % | 8.8 % | 4.87/6 = 81.1 % | 27.67 |
| **7** | **31.7 %** | **41.2 %** | **6.23/7 = 89.1 %** | **30.97** |

Overall **24.41 tok/s**, mean 3.780 tokens/verify.

**The headroom is not inside K=7.** When the engine drafts at full width it already accepts 89.1 %
of the maximum. The gap is that **68 % of verifies never reach full width**, and a K=2 verify runs
at 14.57 tok/s.

### 2. The ceilings, exactly

| | tok/s |
|---|---|
| every verify at K=7, acceptance unchanged | **30.97** |
| every verify at K=7, perfect acceptance | **34.77** |
| 85 % of verifies at K=7 | 29.80 |
| 70 % | 28.50 |
| 50 % | 26.53 |
| today (31.7 %) | 24.41 |

**30 tok/s sits just inside the block-6 ceiling and needs ~85-100 % of verifies at full width.**
Anything past ~31 is not reachable at block 6 at all — which is why S5_PROGRESSION §3's block-8
test is worth more than its acceptance number.

### 3. The controller is already near-optimal — measured, and it is a negative result

`src/decode.cu` gates each extension on `hmarg[VK-1] >= adaptK`, a **scalar 1.5 applied to every
extension**. The extension cost is not constant (14.1 ms from K=2, 25.5 ms to K=7), so the required
acceptance is not constant either — 34.9 % for the first extension, 63.1 % for the last. The margin
is a well-calibrated predictor, so the schedule can be fitted rather than swept:

| gating margin | P(accept) |
|---|---|
| 1.5-2.0 | 36.5 % |
| 2.0-3.0 | 43.3 % |
| 3.0-4.0 | 53.5 % |
| 5.0-6.0 | 64.2 % |
| 8.0-10.0 | 89.6 % |
| 10+ | 95.5 % |

which implies a rising schedule `[1.5, 1.5, 2.0, 3.0, 5.0]` instead of a flat 1.5.

**Counterfactual replay over the same 15 630 verifies says it is worth +0.39 %.** A flat 2.0 gives
+0.53 %; flat 3, 4 and 5 are all *worse* than shipped. Every one of these is inside the 3.5 %
run-to-run spread, so **the retune is not a lever and the shipped 1.5 stands.**

The replay is exact rather than extrapolated: a stricter policy only verifies positions the log
already verified, so no acceptance has to be guessed. The tool refuses any policy looser than the
log's.

**Two methodological corrections were needed to get this right.** The first fit used the
*cumulative* P(accept | margin >= t) where the marginal decision needs the *bucket* conditional at
t; the cumulative version averages in every high-margin position, overstates P everywhere, and
returned a flat 1.0 — which was also the smallest grid value, the tell that the quantity was wrong.
The second is censoring: the log ran at adaptK=1.5, so no gating position below 1.5 was ever
verified. The fit can justify raising the threshold and **cannot** evaluate lowering it.

### 4. So the whole remaining path runs through margins, and here is the size of the shift

The engine computes all six margins on every verify regardless of the K it then picks, so this part
is uncensored.

| position | median margin | share >= 1.5 |
|---|---|---|
| 0 | 6.19 | 78.5 % |
| 1 | 5.16 | 76.5 % |
| 2 | 4.12 | 72.5 % |
| 3 | 3.23 | 68.3 % |
| 4 | 2.93 | 66.2 % |
| 5 | 2.24 | 60.9 % |

A verify reaches K=7 only if **all six** clear the gate: **31.1 %** predicted against ~32 % measured,
which validates the model.

> **CORRECTED, same day — the model above is wrong and the agreement that "validated" it was a
> coincidence.** The gate is `hmarg[VK-1]` with `VK` starting at 2, so it reads **`hmarg[1..5]`**;
> position 0's margin **never gates anything** (the code comment says so explicitly: "`hmarg[VK-2]`
> gates a token already in the block, which is the wrong one"). Replicating the loop exactly matches
> the engine's recorded K on **16 810/16 837 = 99.84 %** of verifies. The six-conjunct model above
> matches **41.17 %** — while reproducing the aggregate K=7 share to within a point.
>
> **That is the trap: the aggregate agreed while the per-verify assignment was wrong 59 % of the
> time.** An aggregate match is not model validation; comparing against the engine's own recorded
> decision is. The tell was a marginal value of *exactly* zero for position 5, which is not a number
> a real effect produces.
>
> Corrected numbers, from the validated model — baseline **24.76 tok/s at 33.0 % K=7**:
>
> | additive shift on gating positions 1-5 | K=7 share | tok/s |
> |---|---|---|
> | none | 33.0 % | 24.76 |
> | +0.5 | 40.7 % | 25.93 |
> | **+1.0** | 56.9 % | **27.85** |
> | **+1.5** | 100 % | **31.18** |
>
> So **~+1.2 to +1.3 reaches 30**, against the +1.4 the wrong model gave — the headline survives, the
> mechanism does not.
>
> Marginal value of lifting one gating position, everywhere: p1 **+1.07**, p2 +0.94, p3 +0.75,
> p4 +0.50, p5 +0.36 tok/s. Cumulatively **+6.42**, far more than the +3.62 sum: the gate is a
> conjunction, so fixing positions together is worth much more than fixing them separately.
>
> **This also settles a lever before it was built.** The DSpark loss weights positions
> 24.3/20.6/17.4/14.7/12.5/10.6 %, and the measured marginal-value profile decays with the same
> shape — so **reweighting the loss toward late positions is not indicated.** The one asymmetry
> worth knowing: position 0 carries the largest loss weight and has *zero* K-gating value; its
> entire contribution is through the acceptance chain.

| shift applied to every position | share reaching K=7 |
|---|---|
| none | 31.1 % |
| **+1 additive** | 54.0 % |
| **+2 additive** | 100 % |
| x2 multiplicative | 44.7 % |
| x5 | 65.9 % |
| x10 | 79.7 % |

**~+1.4 added to every margin reaches the ~85 % that 30 tok/s needs. Multiplicative sharpening does
not get there** — even x10 only reaches 79.7 %, because scaling barely moves a margin already near
zero, and it is the near-zero ones that cap K.

That distinction matters for what the fine-tune has to do: it is not enough to make the draft
*sharper* where it is already confident. It has to make it confident where it currently is not,
which is a distributional change rather than a temperature change.

### 5. The one structural reason to expect headroom

The shipped MTP head was trained by the model's authors against the **unpruned** model's output
distribution. REAP then removed **37.5 % of the routed experts**, which changes that distribution.
The drafter is therefore mismatched to its own target *by construction*, and repairing exactly that
mismatch is what S5 does. This is the strongest prior that the +1.4 shift is reachable — and it is
still a prior, not a measurement.

---

## Finding 110 — **no K-selection policy can be worth more than +1.8 %.** The shipped greedy rule is within 1.8 % of the theoretical optimum, so 100 % of the remaining path to 30 tok/s is the margins themselves

F109 established that tok/s is governed by the K-mix and that a fitted threshold schedule buys
+0.4 %. That leaves an obvious objection: the engine's rule is **greedy** — it stops at the first
margin below the gate, even when every later margin is large. A smarter rule might do much better.
This closes that question with an upper bound rather than another point estimate.

### First, the enabling measurement: margin is a position-independent sufficient statistic

P(accept | reached that position), by margin bucket, per draft position:

| margin | p0 | p1 | p2 | p3 | p4 | p5 |
|---|---|---|---|---|---|---|
| 1.5-2.0 | 58.7 % | 58.6 % | 61.5 % | 59.1 % | 60.3 % | 66.7 % |
| 2.0-3.0 | 68.4 % | 70.6 % | 69.6 % | 73.0 % | 73.6 % | 70.1 % |
| 3.0-5.0 | 82.4 % | 85.3 % | 85.0 % | 85.9 % | 89.8 % | 84.7 % |
| 5.0-8.0 | 93.5 % | 96.4 % | 97.0 % | 97.1 % | 98.0 % | 94.9 % |
| 8.0+ | 99.7 % | 99.8 % | 99.8 % | 99.9 % | 99.6 % | 99.5 % |

**The columns agree.** Conditional on reaching a position, its margin predicts acceptance the same
way regardless of which position it is. Position 0 is verified unconditionally, so it supplies the
sub-1.5 region the operating threshold censors everywhere else:

```
P(accept | margin):  <0.5 0.327 | 0.5-1 0.428 | 1-1.5 0.518 | 1.5-2 0.599
                     2-3  0.704 | 3-5   0.850 | 5-8   0.960 | 8+    0.998
```

This is what makes an optimal policy computable at all: expected tokens at any width follow from the
prefix products of the calibration, including widths this log never ran.

### The optimal policy, and it is barely better than the shipped one

Choosing K per verify to maximise `E[tokens] - r * cost(K)` — renewal-reward with `r` iterated to
its fixed point, which is the correct criterion rather than the myopic tokens-per-ms ratio:

| policy | tok/s | vs shipped |
|---|---|---|
| greedy, threshold 1.0 | 24.07 | **-1.5 %** |
| greedy, threshold 1.25 | 24.29 | -0.6 % |
| **greedy, threshold 1.5 (shipped)** | **24.44** | — |
| greedy, threshold 2.0 | 24.55 | +0.4 % |
| **optimal lookahead** | **24.89** | **+1.8 %** |

**+1.8 % is the ceiling on the whole avenue**, not the value of one candidate. It is what a policy
with perfect knowledge of the calibration and no restriction to a threshold rule achieves. The
shipped rule — one scalar, greedy, no lookahead — is within 1.8 % of it.

The +0.4 % for a flat 2.0 here independently reproduces the +0.53 % from F109's exact stricter-only
replay, by a completely different method (calibration model vs counterfactual replay of observed
acceptance). Two methods agreeing on a small number is worth more than either alone.

### What this settles

The K-mix is not a lever. Between F109's fitted schedule (+0.4 %) and this bound (+1.8 %), every
route through the controller is inside or barely outside the 3.5 % run-to-run spread.

**So the path to 30 tok/s is entirely the margins, and the size of the required shift is already
measured: +1.0 additive on gating positions 1-5 reaches 27.85 tok/s, +1.5 reaches 31.18.** There is
no controller change that substitutes for it, and now there is a number saying so rather than an
assumption.

That is a good outcome for S5's justification and a bad one for anyone hoping for a free win: the
fine-tune is not one lever among several, it is the only one left.

### Caveat, stated because it bounds the claim

The optimal-policy figure is **model-based**. A looser policy verifies positions this log never
verified, so expected tokens come from the calibration rather than from observed acceptance. The
calibration is measured and its position-independence is checked above, but the +1.8 % is an
estimate. It is used here as an *upper bound to close an avenue*, which is the one purpose an
estimate is safe for — a lever this small does not become worth building if the estimate is off by
half.
