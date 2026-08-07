# IMPLEMENTATION_PLAN.md — maximising physical decode, post-research

Synthesis of eight parallel research reports (`research/`) plus on-box verification. Written
2026-08-06. Supersedes `ROOFLINE.md` §6 and the earlier standing order in `OPTIMIZATION_LOG.md`.

---

## 0. What the research overturned

Four premises this project operated on for six optimisation rounds were wrong. All four were
**negative** claims — "the hardware can't", "the model can't" — which is the class of claim I now
treat as the least trustworthy, because it silently removes options instead of producing a visibly
wrong number.

| premise | reality | how established |
|---|---|---|
| "`tcgen05` not supported on Thor" | **Present and running.** Inherited from an error message naming `.target 'sm_110'` — the probe had never used `sm_110a`. | Finding 29, on-box compile + runtime |
| "Thor has no FP4 tensor cores" (mine, after probing only `mma.sync`) | **Present via `tcgen05.mma.kind::mxf4nvf4`**, SASS `UTCOMMA.4X`. cuBLASLt NVFP4 measured **571–657 TFLOP/s vs BF16 121–156 = 4.2–4.7×** — a dequant emulation cannot exceed the BF16 rate. | Finding 30 + `research/FP4_HARDWARE.md` |
| "the M≥2 penalty is B-operand coalescing; fix it with a repack" | **Repack buys nothing** — strided m16 measures **0.98×** vs contiguous at adequate grid size. The penalty is grid starvation and ILP, not layout. | `research/GB10_COMPARISON` measurements |
| "achievable bandwidth is 240 GB/s, so the wall is 21.4 tok/s" | Achieved BW is a strong function of **per-kernel working set**: 1 MB → 54 GB/s, 32 MB → 208, ≥512 MB → 240. Our kernels are 1.7–50 MB, so the realistic wall is **~17 tok/s**. | on-box sweep |

**And a build bug caused the first of these:** `nvcc -arch=sm_110a` runs two device passes; the
`compute_110` pass strips arch-specific features and fails, emitting an error that names `sm_110`.
**7 of our 8 build scripts used that form.** Fixed to `-gencode arch=compute_110a,code=sm_110a`.

---

## 1. Where we are

```
7.80 tok/s  ported as-is
9.83 tok/s  after Opts #1 #3 #4 #7   (1.261x, 120.6 GB/s, 50.2% of a 240 GB/s reference)
~17 tok/s   realistic wall given our kernel working-set sizes
21.4 tok/s  idealised roofline at 240 GB/s
```
Speculation sits at **1.00× of base** with acceptance **3.12 of 5** — better than vLLM's DSpark on
2×Spark (2.69 on prose). The blocker is verify cost (84% of the round), not acceptance.

**Comparison point, honestly decomposed.** The best single-Spark engine (antirez `ds4`, hand-written
CUDA) does **18.05 tok/s — on a ~2-bit GGUF**, `B_tok` ≈ 9.6 GB. Against our 12.26 GB and 9.51:
**1.28× (fewer bytes) × 1.48× (achieved BW) = 1.90×.** Half the gap is a *quantisation choice we
have deliberately not made*; half is engineering. No published single-Spark number exists for the
FP4/MXFP4 checkpoint at all.

---

## 2. The plan, ranked by expected gain ÷ cost

### Tier 1 — runtime-only, no checkpoint change, gateable bit-exact

| # | lever | expected | why it is credible |
|---|---|---|---|
| **1** | **MoE GEMM output-row blocking, BN≥2** | **+12.7% routed, +17.5% with shared expert** | An agent rebuilt *our exact shape*: BN=1 → 155 GB/s, **BN=2 → 242–249 (101% of achievable)**. Activation registers reuse across output rows halves instructions per weight byte. |
| **2** | **ILP audit across every remaining streaming loop** | up to 2× on any loop still at ILP=1 | Opt #7 proved the mechanism on the dense GEMV (+3.4% overall). The grouped MoE kernel, compressor, and indexer loops have not been audited. |
| **3** | **Kernel fusion to raise per-kernel working sets** | small kernels run at 23–59% of achievable; moving them to ~80% | Thor has **228 KB smem/SM — 2.28× GB10's** — which is exactly the resource this needs. Fuse router+HC+DSA+KV-compressor into layer-scope kernels. |
| **4** | **BF16-native compressor / indexer-compressor / router** | ~620 MB/step ≈ **+5%** | Same defect class as the `lm_head` fix already banked: `Loader::bf16` expands BF16→F32 at load. `decode.cu:140/179/185`. Proven template (`wo_a_native`). |
| **5** | **Intra-expert neuron activation sparsity** | **+11% (conservative) to +17%** | Training-free, 8 MoE models 1B–400B, *"no modification to activation function or model parameters"*. **The only >10% lever needing no artifact.** Derate for block-32 granularity; element-level sparsity overstates exploitable sparsity by up to 78 pp. |
| 6 | HC params F32 → FP16 | 68 MB, +0.6% | trivial |
| 7 | Skip the router on the 3 hash-routed layers | 3 routers + 3 sorts off the critical path | expert ids are a pure function of token id, known before the forward begins |

### Tier 2 — speculation (the verify is 84% of the round)

| # | lever | expected |
|---|---|---|
| 8 | **`tcgen05` MXFP4 at the M=5 verify** | tcgen05's minimum tile is **128×8×64** — awkward at M=1, well matched to M=5. This is where FP4 tensor cores actually pay. |
| 9 | Correct `E_frac`: real expert union is ~25 not 30 at K=5 | `c_v` floor 2.13 → **1.89**; makes the verify fix worth 1.29× not 1.18× |
| 10 | AcceptMoE-style commitment-weighted expert set at verify | **1.29× measured at batch 1 with resident weights**, −0.27 pp accuracy |
| 11 | REAP-repair fine-tune of the 3 MTP blocks + markov head | a 3.12 → **4.0–4.5** (not 4.8 — DSpark's own chat-domain lengths are 3.29–3.64) |

### Tier 3 — opt-in artifact (changes the weights; violates the current no-requant rule)

| lever | bytes | cost |
|---|---|---|
| MLA + shared expert → MXFP4 | 2,663 MB → ~28 tok/s wall | **+0.59% PPL**; reuses gated `fp4_gemm.cu`. Best bytes-per-damage in the report. **But**: every shipped DeepSeek recipe (NVIDIA ModelOpt, RedHat, DeepSeek's own V4-Pro) *excludes attention from quantisation entirely*. Unprecedented, not proven unsafe. |
| top-6 → top-4 | 1,150 MB, ~+10% | DeepSeekMoE Fig. 5 shows +0.06 nats at 2B; a third party reports top-4 is the **floor** for this family. |
| `lm_head` → 8-bit | 530 MB, +4.3% | what every GB10 competitor does |

### Explicitly rejected, with the measurement

- **m16 B-operand repack** — 0.98× measured; there is no penalty to recover.
- **Clock/EMC locking** — 106.8 vs 105.2 ms/tok. Our step is fully device-side; no host gaps.
- **MAXN power mode** — EMC is 4266 MHz in all four modes.
- **Expert prefetch / caching / sticky routing** — every published gain avoids a slow link we do not
  have. DeepSeekMoE has among the **lowest** routing consistency measured (36.9 SRP), and shared
  experts reduce it further.
- **Trees for speculation** — every node is a fresh 160-way router draw.
- **Draft/verify pipelining** — 20 SMs, one DRAM, both phases bandwidth-bound.
- **2:4 sparsity** — 25% saving at 4 bits, and no SM110 sparse kernels exist in CUTLASS.
- **MLA weight absorption** — already architectural in V4 (no `kv_b_proj` exists); every remaining
  fold is 2.0–3.6× byte-*increasing*.
- **FlashMLA port** — `setup.py` emits only `sm_90a` + `sm_100f`, and its 3000 GB/s is KV streaming
  at batch 128, which we do not have.

---

## 3. Two upstream patches worth carrying locally

1. **CUTLASS `arch/reg_reconfig.h` has no `1100` clause** → `setmaxnreg` silently compiled out on
   Thor → register spills. Upstream issue #3056 / PR #3308 measure **1.74× on FMHA**. Two lines.
2. **CuTeDSL 4.6.0** has no `(11,0)` entry → falls through to `sm_110` → block-scaled ops rejected.
   `export CUTE_DSL_ARCH=sm_110a`.

## 4. Standing measurement discipline

- **Bench COLD.** Warm-L2 microbenchmarks converted a request-count pathology into a non-event and
  cost this project two wrong diagnoses. Triton's `do_bench` zeroes a 256 MB buffer between
  iterations; ours must too.
- **`ncu`'s "Memory Throughput %" is L2 throughput on Thor** — a kernel at 89% of peak reports 30%.
- **Instrument policy:** `ncu` for mechanism, `gemm_bench` for magnitude, full model to confirm.
- **Never benchmark with other GPU work resident** — sibling processes halve bandwidth to 110–119
  GB/s, which is suspiciously close to a plausible engine number.
- **Probe every instruction family before concluding a capability is absent.** For Blackwell
  tensor-core matmul that is `mma.sync`, `wgmma`, and `tcgen05`.
