# Kernel optimisations — AR and speculative decode

Every optimisation below was **adopted**: built, gated, measured end-to-end on the full model, and
kept. Findings that were built and *rejected* are in [`negative-results.md`](negative-results.md),
which is the longer and more useful list.

Session arc: **base AR 10.30 → 13.83 tok/s (+34.3 %)**, **speculative 16.86 → 22.15 tok/s (+31.4 %)**.

---

## 1. The transformations that generalised

Three ideas account for most of the gain. Each was discovered once, then found again in three or
four other kernels — which is the main reason this page is organised by *mechanism* rather than by
kernel.

### 1.1 "Read B once" — amortise a weight read across every row that uses it

**Findings 40, 42, 43, 64.** The naive grouped-GEMV structure reads a weight matrix inside the row
loop, so an expert serving `m` rows re-reads its whole weight matrix `m` times. Hoisting the load
out of the row loop and accumulating `RB` rows against one load turns traffic from
`rows × weight_bytes` into `ceil(rows/RB) × weight_bytes`.

Applied to: `ogroup`, `lm_head`, `gemm_fp32` (F40/42/43), then the MoE grouped GEMV (F64), then the
ogroup **tensor-core** path (F88, prefill).

**The subtlety that cost two cycles: `RB` must follow rows-per-EXPERT, not batch size.**
`acc[RB][BN]` is live regardless of the real row count, so an oversized `RB` buys nothing and costs
occupancy. Measured histogram at K=5: ~70 % of experts serve exactly **one** row, ~88 % serve ≤2,
max 5. F65 shipped `RB` sized from `bs`; F70 corrected it to the measured histogram:

```
capped-at-2 probe:  RB=1 441.9  RB=2 423.8  RB=4 434.4  RB=8 497.3   -> picks RB=2
measured histogram: RB=1 559.1  RB=2 534.7  RB=4 499.8  RB=8 530.6   -> picks RB=4
```

The ranking **inverts** depending on which grouping the probe used. A microbenchmark whose input
distribution does not match production will confidently select the wrong parameter.

### 1.2 Thread-per-output is wrong at decode shapes

**Findings 71, 73.** Kernels written for large batches degrade into near-serial work at M=1.

- `index_score` was one thread per output on a **single block** — 6.05 ms to do 97 k MACs.
  Rewritten warp-per-output: **+4.4 % end-to-end**.
- `k_moe_prefix` and `k_build_tiles` were both `<<<1,1>>>` serial scans over `nr=160`.
  Parallelised: `moe:group` **−20.3 %**.

Both now sit at the **launch-latency floor** (0.24–0.25 ms across 43 layers, against 0.19 ms for a
trivially parallel neighbour), which is the signal that the class is exhausted. Re-opening it needs
a *new* kernel with bad geometry, not another pass over these.

### 1.3 Stage more per barrier

**Findings 74, 78.** The fp8 tile GEMM staged one 128-element K-block per `__syncthreads` pair.
Staging `KC` of them cut the barrier count proportionally: the four-mark GEMM block went **−13.5 %,
bit-identical**, worth **+6.1 % end-to-end** — the largest single adoption of the session.

Double-buffering that staged chunk (F78) was then worth only **+0.28 %**, and the reason is a
four-register occupancy step. That closed the family.

---

## 2. Structural wins

### 2.1 Full-step CUDA graph (F44)

Capture the entire decode step once, replay per token. **92.5 → 79.3 ms/tok** — the largest
structural win in the project's history, and the reason launch overhead is *not* on the open lever
list today.

Corollary from F46: a *verify* graph is worth only **1.05×** (2,788 nodes). The verify has no
launch gap to recover. Launch overhead was a decode problem, not a general one.

### 2.2 The arena (F44, F83)

Per-call `cudaMalloc`/`cudaFree`/`cudaStreamSynchronize` in every sub-function dominates at M=1.
`dscratch.h` replaces them with a bump allocator reset per layer.

**F83 is the cautionary half.** The DSpark *draft* path never got the arena; F82 measured
10.19 ms/round — 7.4 % of the spec cycle — inside raw `cudaMalloc`/`cudaFree` on a drained GPU and
priced the fix at +5–7 %. Built exactly as specified, it returned **+0.41 %**. The measured time was
**host** time overlapping **device** work; the GPU was the critical path throughout. Deleting 134
allocator calls and 10 syncs per round removed real work from the CPU and almost nothing from the
clock.

> **Rule earned:** host time is not recoverable time. Measure what is on the critical path, not what
> is expensive to execute.

### 2.3 Adaptive verify width (F49, F59, F63)

The speculation block length `K` is a decision variable, not a constant. Adapting it to the draft's
margin is worth **+9–11 % where it engages**.

This finding has an instructive history: F49 discovered it was being treated as a constant, F59
claimed 7×, and F63 — re-measured on a *correct* prefill after F62 — showed the 7× was inflated by
the same prefill bug. **A parameter fitted on broken data is fitted to the breakage.**

### 2.4 Fork/join the kv chain (F57)

The kv chain forked off the q chain, serialising two independent computations. Splitting them onto
separate streams recovered the overlap.

---

## 3. Precision and layout

| finding | change | result |
|---|---|---|
| F41 | m16 tile fetched every weight sector for half its payload | fixed; prerequisite for everything after |
| F66 | expert pointers are misaligned by a **constant 8 bytes** (43,470 of 44,436 tensors at `data_offset%16==8`) | the funnel shift was buying alignment that was already there |
| F72 | so: two `uint2` loads instead of two `uint4` + funnel shift | same instruction count, **half the bytes requested**, +1.4 % |
| F56 | the shared expert was never dependent on the routed ones | run them concurrently |

F66 → F72 is the cleanest example of the project's method: **measure the property you are
compensating for before writing the compensation.**

---

## 4. What every adopted change has in common

1. **A gate proved it before a benchmark did.** `gate_units`, `gate_grouped_moe`, `gate_ogroup_gemv`,
   `gate_prefill_len`, `gate_compressed_decode` and friends run against a PyTorch oracle. Several
   optimisations are *bit-identical* by construction and gated with `memcmp`.
2. **It was measured end-to-end on the full model**, not in `gemm_bench`. The bench overstates —
   F47 quantified this, and F76 is the extreme case: a free, bit-identical, `gemm_bench`-verified
   −7…−15 % on the kernel was **+0.1 %** in situ, because the kernel is latency-bound.
3. **The measurement was paired.** Nine spec verifies pairing 1:1 at identical `K` and identical
   accept counts is the instrument; end-to-end tok/s alone cannot resolve sub-1 % effects.

See [`measurement-and-traps.md`](measurement-and-traps.md) for why each of those is non-negotiable.
