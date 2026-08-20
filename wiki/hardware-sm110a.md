# Jetson AGX Thor / `sm_110a` — measured facts

Everything here is **measured on this box** with this toolkit (CUDA 13.0, JetPack R38.4.0), or is a
negative result that still binds. Re-deriving any of it is wasted time.

Full ISA table and provenance: `HARDWARE.md`.

---

## 1. The device

```
NVIDIA Thor   SMs = 20   cc = 11.0 -> build target sm_110a
unified LPDDR5X 122.8 GiB (shared with the OS)
```

**Unified memory is the defining constraint.** The engine sits at **109.2 of 122.8 GiB** with the
model resident, leaving **13.6 GiB**. That is the number every capacity question must be answered
against — not 122.8.

---

## 2. Bandwidth — measured (`tools/bw_probe.cu`)

| | |
|---|---|
| achievable streaming read | **233 GB/s** (235.6 pinned; a later probe reports 240) |
| flat to | 64 GiB |
| `B_tok` (weights read per token) | **12.26 GB** |

**The 19.0 tok/s "AR roofline" derived from these is a normalisation constant, not a target.** It
assumes every kernel moves bytes at full DRAM bandwidth *and* that the non-byte part of the step is
zero. `tools/byte_floor.py` measures both assumptions false:

- **22.3 ms of a 71.4 ms decode step is not bytes at all** — launch floors, Sinkhorn, activations,
  and at least one latency-bound kernel.
- Byte-moving marks average **191 GB/s**, not 233. The only mark above the probe is `moe:w2` at
  246 GB/s, which is **L2 reuse** of the expert set `w1w3` just touched — not a kernel beating DRAM,
  and excluded as a target rate.

| floor | ms/tok | tok/s |
|---|---|---|
| every byte mark at 233 GB/s (optimistic) | 62.6 | **15.98** |
| every byte mark at 198 GB/s (`q:wq_b` cold, realistic) | 69.8 | **14.33** |
| measured today | 71.4 | 13.83–14.01 |

**Remaining base-AR headroom: 2.2 % realistic, 12.3 % optimistic** — against the 37 % that 19.0
implies.

Per-kernel achievable rates (the honest bar, from the K=1 dprof column):
`q:wq_a` 115, `q:wq_b` 195, `o:wo_b` 185, `o:wo_a` 168 GB/s. A GEMV at M=1 has one row of reuse and
cannot saturate DRAM the way a GEMM does.

---

## 3. Compute — measured (`tools/flops_probe.cu`)

The project ran for its whole life without this number, and three findings leaned on assumptions
about it that were **wrong by an order of magnitude**.

```
FP32 FFMA                  5.45 TFLOPS   (reference for 128 cores/SM: 5.37 -> core count confirmed)
FP16 mma m16n8k16         53.07 TFLOPS
BF16 mma m16n8k16         53.08 TFLOPS
FP8 e4m3 mma m16n8k32     92.18 TFLOPS
```

Measured with 32 independent FFMA chains, so throughput- not latency-bound — one dependent chain
would measure FFMA latency and report ~1/4 of peak, the classic way this benchmark lies.

**Thor's compute lives almost entirely in the tensor cores: bf16 is 9.7× fp32, fp8 is 16.9×.**

Consequences, against real work in the engine:

| | measured | % of the 5.45 TFLOPS **fp32** peak |
|---|---|---|
| whole prefill | 1.47 TFLOPS | 27 % |
| `cattn:sparse` | 1.04 TFLOPS | 19 % |
| `cattn:ogroup` (fp32 arm) | 1.98 TFLOPS | 36 % |

So the fp32 attention path runs at a *reasonable* fraction of what fp32 on this chip can do. Earlier
claims that it was at "1 % of peak" assumed a ~100 TFLOPS ceiling that exists only in fp8 tensor
cores.

**FP4 is not covered by this probe** — see §4.

---

## 4. ISA facts that bind

| feature | status on `sm_110a` | consequence |
|---|---|---|
| `tcgen05.*` (5th-gen TC / tensor memory) | **SUPPORTED** — compiles and runs | SM100-class kernels are on the table |
| `cp.async.bulk` / `.tensor` (**TMA**) | **SUPPORTED** — runtime-verified | hardware-managed swizzled global→shared |
| `mma.sync` with **FP4/FP6** operands | **BLOCKED** | …but this is the wrong family for Thor |
| **`tcgen05.mma.kind::mxf4nvf4.block_scale`** | **AVAILABLE** — emits real SASS (`UTCOMMA.4X`) | Thor **has** hardware MXFP4 tensor-core matmul, reached via `tcgen05`, not `mma.sync` |
| `cp.async.cg.shared.global` | OK | async weight streaming available |

> **Thor and GB10 have FP4 compute through *opposite* instruction families.**
> Thor `sm_110a` is datacenter-lineage (tensor memory + `tcgen05`); GB10 `sm_121a` is
> consumer-lineage (`mma.sync` FP4, no `tcgen05` at all). Testing only the family the *other* chip
> uses produces a confident, wrong "this hardware has no FP4" — an error made twice here before it
> was pinned down. **Probe both: `bash scripts/arch_probe.sh`.**

**Not yet verified:** that a *complete* `mxf4nvf4` MMA executes correctly on silicon. `tcgen05`
alloc/dealloc and TMA are runtime-verified; a full FP4 MMA needs tensor-memory allocation, matrix
descriptors and the scale-vector layout wired up. **SASS emission proves ISA and assembler support,
not silicon correctness.** That runtime gate is the one that matters, and it is the reason the MoE's
apparent "5 % of the fp8 peak" is *not* a measured ceiling.

---

## 5. Operating rules

- **Pin clocks before any measurement** — `bash scripts/pin_clocks.sh pin`, which `run_model.sh`
  and `run_server.sh` now do for you (opt out with `DSV4_PIN_CLOCKS=0`) and which drops a
  `<log>.clocks` sidecar so the run states its own clock state. A roofline probe ramps the governor
  by itself, which is why no roofline number here ever saw an unpinned clock.
  **Corrected 2026-08-20 (ladder 3.1): pinning is worth +2.0-3.0 % on decode, not the +3.0-6.4 %
  this page and HARDWARE.md carried, and the reason is that a *decode* also ramps the governor by
  itself.** Sampled every 2 s, a governed box spends **97.7 % of its compute window at 1386 MHz and
  4266 MHz** — the pinned frequencies. The 315 MHz / 2750 MHz idle state is the ~90 s checkpoint
  load, not the run. Pinning buys the ~2 s ramp, and pinning matters chiefly because the base-AR
  window is measured *inside* that ramp ([`measurement-and-traps.md` §24](measurement-and-traps.md)
  — 88 ms/tok governed vs 72.8 pinned, a 21 % artefact that reproduces exactly).
- **`nvpmodel -m 0` (MAXN) is not worth taking.** It is the only way to reach the 1575 MHz core
  ceiling — `jetson_clocks` only raises a rail to its governor's ceiling and nvpmodel 1 caps GPU
  MAX_FREQ at 1386000000 — and at a verified 1575 MHz for 18/18 samples it measured
  **+1.68 +/- 5.83 % paired** against pinned-120W. A core-clock raise is the wrong lever for a
  bandwidth-bound engine. The sanctioned pin leaves the power mode alone.
- **Drop caches before each model run**: `sync; echo 3 | sudo tee /proc/sys/vm/drop_caches`.
- **Single tenant.** Two concurrent full-model processes exhaust the unified pool and thrash swap.
  `scripts/run_model.sh` enforces this with `flock` — on 2026-08-06 a failed-*looking* launch had in
  fact succeeded, the relaunch created a second loader, and the pair drove `available` to 0.
- **Batching is a short-context lever only.** Compressed KV is 86 KiB/token, so within the 13.6 GiB
  free: ~40 sequences at 4 K context, **5 at 32 K, 1 at 128 K**.

---

## 6. `cudaDeviceProp::clockRate` was removed in CUDA 13

Use `cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, 0)`. Reported base clock is 1.049 GHz;
measured FP32 slightly exceeds the reference computed from it, so the sustained clock is a little
higher.
