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
