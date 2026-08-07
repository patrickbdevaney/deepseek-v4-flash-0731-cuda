# HARDWARE.md — Gate H1

**Status: PASS.** Measured on the target box, 2026-08-06. Nothing here is inherited from the
directive or from a prior project's notes — every line has a command next to it.

---

## 1. The box

| | value | how measured |
|---|---|---|
| Chip | NVIDIA Thor (Jetson AGX Thor) | `nvidia-smi --query-gpu=name` |
| Compute capability | **11.0** → build target `sm_110a` | `nvidia-smi --query-gpu=compute_cap` |
| SMs | 20 | prior probe, `laguna-s1-cuda-server/HARDWARE_PROBE.md` — re-probe at Gate B1 |
| CPU | 14 cores, aarch64 | `nproc`, `lscpu` |
| OS | Ubuntu 24.04.3 LTS, L4T R38 rev 4.0, kernel 6.8.12-tegra | `/etc/nv_tegra_release` |
| Driver / CUDA | 580.00 / CUDA 13.0 (nvcc V13.0.48) | `nvidia-smi`, `nvcc --version` |
| Memory (unified, total) | **122 GiB** | `free -g` |
| Memory (available at rest) | **117 GiB** | `free -g` |
| Swap | 31 GiB | `free -g` |
| Disk free (`/`, nvme0n1p1) | 230 GiB of 936 GiB at survey time | `df -h` |
| Peak LPDDR5X bandwidth | 273 GB/s (spec) | vendor |
| Achievable streaming BW | **240 GB/s measured** (212 under memory contention) | `tools/bw_probe.cu` — see §2 |

`nvidia-smi` reports `Memory-Usage: Not Supported` on Thor — device memory is the unified
pool, so **`free` is the authoritative memory instrument**, not `nvidia-smi`.

## 2. Achievable bandwidth — measured, not inherited

`tools/bw_probe.cu` is a grid-stride `float4` streaming read over a buffer far larger than L2,
reduced so nothing is optimised away. `nvcc -O3 -arch=sm_110a`, 320 blocks × 256 threads.

| buffer | GB/s | % of 273 spec |
|---|---:|---:|
| 1 GiB | 240.7 | 88% |
| 4 GiB | 241.5 / 242.8 | 89% |
| 8 GiB | 212.3 | 78% |

**~240 GB/s is the achievable streaming figure, ~212 GB/s under memory contention.** These were
taken while a 100 GiB checkpoint download was writing, so the 8 GiB point is contended and the
whole sweep is, if anything, a slight underestimate. **Re-run idle before treating 240 as final.**

This supersedes the ~200 GB/s planning figure carried from prior projects and **raises the AR
wall by ~20%** (`ROOFLINE.md` §2).

## 3. Profiling: `ncu` is unblocked, and it lies about memory on this chip

**Status: unblocked.** `sudo /usr/local/cuda-13.0/bin/ncu …` collects counters successfully.
(Plain `sudo ncu` fails with `command not found` — `ncu` is not on root's `PATH`; use the full
path.) Unprivileged `ncu` still returns `ERR_NVGPUCTRPERM`; `/etc/modprobe.d/nvidia-profiler.conf`
now carries `options nvidia NVreg_RestrictProfilingToAdminUsers=0`, which will lift that **at the
next reboot**. No reboot is needed to make progress — sudo works today.

> ### ⚠ `ncu`'s "Memory Throughput %" does NOT mean DRAM bandwidth utilisation on Thor
>
> Profiling `stream_read` — a kernel independently measured at **244 GB/s, i.e. ~89% of spec
> peak** — `ncu` reports:
>
> ```
> Memory Throughput           %   30.26      <- NOT bandwidth utilisation
> L2 Cache Throughput         %   30.26      <- identical: this is where the number comes from
> Compute (SM) Throughput     %    5.28
> Average MC Channel Active Cycles   (!) nan <- no DRAM counters on unified memory
> dram__cycles_active                missing
> ```
>
> Thor has no discrete DRAM and exposes no memory-controller counters, so SpeedOfLight's
> "Memory Throughput" degenerates to **L2 throughput**. A kernel running at 89% of achievable
> bandwidth reports 30%, and `ncu` then advises "memory bandwidth below 60% of peak typically
> indicates latency issues" — which is simply wrong here.
>
> **Consequence.** The prior project's planned step 0 — *"confirm Memory% vs Compute% per kernel
> to prove the software-dequant compute-bound hypothesis"* — would have been **actively misled**
> by this metric. On Thor:
> - **Bandwidth utilisation** must come from the analytical byte model ÷ wall-clock
>   (`tools/inventory.py` + timing). That stays the authoritative instrument.
> - **`ncu` is still valuable** for what it measures honestly: `Compute (SM) Throughput`,
>   occupancy, warp-stall reasons, L1/L2 hit rates, and instruction mix — which is exactly what
>   the compute-bound-dequant hypothesis actually needs.

## 4. The constraint that governs this project

Memory is unified and **shared with the host OS, the desktop session, and the agent tooling.**
At rest, ~5 GiB is already in use before we allocate anything.

- Checkpoint weights: **100.400 GiB** (see `ROOFLINE.md` §1).
- Available pool: **117 GiB**.
- **Headroom: ~16.6 GiB** for KV cache, activations, CUDA context, and everything the OS is
  doing while we run.

Two operational rules follow, both inherited from the prior 180B project where violating them
cost a physical power-cycle (`~/dspark-cuda-reap-finetune/GATE_LOG.md`):

1. **Memory-neutral optimisation only.** No optimisation may add persistent device memory.
   A +5.5 GiB dequant cache on the prior project starved the system and hard-hung the box.
   Prefer kernels that read quantised data in place over anything that materialises fp32.
2. **Always run full-model binaries detached to a file**, so an SSH drop or a wedged run never
   locks the user out of the machine:
   ```
   setsid nohup ./build/<binary> <args> > ~/run.log 2>&1 < /dev/null &
   ```
   Then poll the log. A full-model run monopolises the device for tens of seconds at ~90% of
   the pool — batch measurements, and warn before starting one.

## 5. Architecture facts that are already settled (empirical, prior project, CUDA 13.0)

These were tested on this exact box and toolkit. They are **negative results that still bind**,
and re-deriving them is wasted time. Source: `~/dspark-cuda-reap-finetune/DECODE_GAP_RESEARCH.md`,
`FP4_COMPUTE_NOTE.md`.

| Feature | Status on `sm_110a` | Consequence |
|---|---|---|
| `tcgen05.*` (5th-gen TC / tensor memory) | **SUPPORTED — compiles AND runs** (`alloc`/`dealloc`/`mma`/`ld` all pass ptxas on `compute_110a`; alloc+dealloc verified at runtime) | **The inherited "not supported" claim was a TARGET-FLAG ERROR** — see the box below. DeepGEMM/CUTLASS SM100-class kernels are back on the table. |
| `cp.async.bulk` / `.tensor` (**TMA**) | **SUPPORTED — compiles AND runs** (bulk copy with `mbarrier` completion verified at runtime) | Hardware-managed, swizzled, perfectly-coalesced global→shared transfer. This is the textbook fix for the B-operand coalescing loss in `LOOP_LOG` Finding 28. |
| `mma.sync` with **FP4/FP6 operands** or `.kind::mxf4` | **BLOCKED** on `sm_110a` and `sm_110f` | …but this is the WRONG FAMILY for Thor — see below. |
| **`tcgen05.mma.kind::mxf4nvf4.block_scale.scale_vec::4X`** | **AVAILABLE.** Emits real SASS: `UTCOMMA.4X gdesc[UR8], gdesc[UR8], tmem[UR6], tmem[UR4], idesc[UR5], tmem[UR6], !UPT` | **Thor HAS hardware MXFP4/NVFP4 block-scaled tensor-core matmul.** It is reached through the 5th-gen `tcgen05` path, not the Ampere-lineage `mma.sync` path. |
| `tcgen05.mma.kind::mxf8f6f4.block_scale`, `.kind::f8f6f4`, `.kind::f16` | **AVAILABLE** | The whole tcgen05 mixed-precision family is present. |

> ### Thor and GB10 have FP4 compute through OPPOSITE instruction families
>
> | | `mma.sync` FP4 | `tcgen05` FP4 |
> |---|---|---|
> | **Thor `sm_110a`** | BLOCKED | **AVAILABLE** (`UTCOMMA.4X`) |
> | **GB10 `sm_121a`** | **AVAILABLE** | BLOCKED — `sm_121` has no `tcgen05` at all |
>
> Thor is **datacenter-lineage** Blackwell (tensor memory + `tcgen05`, SM100-like). GB10 is
> **consumer-lineage** (SM120-like, `mma.sync` FP4). Testing only the family the *other* chip uses
> produces a confident, wrong "this hardware has no FP4" — which is exactly the error made here
> twice before this was pinned down. **Probe both families: `bash scripts/arch_probe.sh`.**
>
> **Not yet verified:** that a *complete* `mxf4nvf4` MMA executes correctly on the silicon. `tcgen05`
> alloc/dealloc and TMA are runtime-verified; a full FP4 MMA needs tensor-memory allocation, matrix
> descriptors and the scale-vector layout wired up. **SASS emission proves ISA + assembler support,
> not silicon correctness.** That runtime gate is the next step, and it is the gate that matters.
| `cp.async.cg.shared.global` | **OK** | Async weight streaming / software pipelining is available. |
| `__nv_cvt_fp4x2_to_halfraw2` (HW FP4×2 → half2 unpack) | **OK** | The fast MXFP4 dequant primitive works. This is the top unexploited lever (see `ROOFLINE.md` §6). |
| FP4 tensor-core **compute** | **BLOCKED** — re-confirmed on `compute_110a` with ptxas naming the feature explicitly | FP4 is a **storage/bandwidth** format only. FP8 `mma` is the compute ceiling. |

> ### ⚠ The target flag decides what exists. Get it wrong and you will conclude the hardware lacks features it has.
>
> ```
> -arch=sm_110    -> compute_110    -> tcgen05 BLOCKED      <-- where the inherited claim came from
> -arch=sm_110a   -> compute_110a   -> tcgen05 COMPILES     (but building an EXECUTABLE this way ALSO
>                                                            emits compute_110 PTX, which then fails)
> -gencode arch=compute_110a,code=sm_110a                   <-- correct for executables: SASS only
> ```
>
> The `a` suffix means "architecture-specific features enabled", and tcgen05 lives behind it. The prior
> project recorded `Instruction 'tcgen05.fence' not supported on .target 'sm_110'` — note the target in
> its own error message is `sm_110`, not `sm_110a`. That single missing letter propagated into
> `DECODE_GAP_RESEARCH.md` as "DeepGEMM is a rewrite, not a port", and into this project's `HARDWARE.md`
> and `ROOFLINE.md` as a hard architectural constraint. It was never true.
>
> **Reproduce with `bash scripts/arch_probe.sh`** — it tests compile *and* runtime, and it is now part of
> the repo precisely so this cannot rot again.

## 6. Remaining instrumentation gaps

- Achievable bandwidth is now measured (§2.1) but was taken under download contention; re-run
  `./build/bw_probe 4096 30` on an idle box to finalise it.
- `--runtime nvidia` Docker containers **wedge the device** on this box (stuck `runc` processes
  in uninterruptible D-state; `docker rm -f` hangs). Plain `docker run` works. All CUDA here
  compiles and runs on the host; only CPU-torch oracle work goes in a container.
