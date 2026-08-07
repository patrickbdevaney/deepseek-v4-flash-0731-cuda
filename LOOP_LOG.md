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
