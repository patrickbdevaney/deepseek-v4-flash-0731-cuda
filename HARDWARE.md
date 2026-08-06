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
| Achievable streaming BW | ~200 GB/s (planning figure) | carried from prior projects; **unverified on this build — re-measure at Gate B1** |

`nvidia-smi` reports `Memory-Usage: Not Supported` on Thor — device memory is the unified
pool, so **`free` is the authoritative memory instrument**, not `nvidia-smi`.

## 2. The constraint that governs this project

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

## 3. Architecture facts that are already settled (empirical, prior project, CUDA 13.0)

These were tested on this exact box and toolkit. They are **negative results that still bind**,
and re-deriving them is wasted time. Source: `~/dspark-cuda-reap-finetune/DECODE_GAP_RESEARCH.md`,
`FP4_COMPUTE_NOTE.md`.

| Feature | Status on `sm_110a` | Consequence |
|---|---|---|
| `tcgen05.*` (5th-gen tensor core / UMMA) | **NOT supported** (`Instruction 'tcgen05.fence' not supported on .target 'sm_110'`) | DeepGEMM / SM100 fp8×fp4 kernels are a rewrite, not a port. Datacenter-Blackwell kernels do not transfer. |
| `cp.async.cg.shared.global` | **OK** | Async weight streaming / software pipelining is available. |
| `__nv_cvt_fp4x2_to_halfraw2` (HW FP4×2 → half2 unpack) | **OK** | The fast MXFP4 dequant primitive works. This is the top unexploited lever (see `ROOFLINE.md` §6). |
| FP4 tensor-core **compute** | **BLOCKED** on all paths (ptxas, cuBLASLt reports 0 algos, CUTLASS, SASS) | FP4 is a **storage/bandwidth** format only. FP8 `mma` is the compute ceiling. Re-test with `~/dspark-cuda-reap-finetune/tools/cublas_fp4_probe.cu` after every CUDA toolkit update. |

## 4. Known instrumentation gaps

- **`ncu` is blocked by `ERR_NVGPUCTRPERM`.** Nsight Compute's SpeedOfLight section needs
  elevated counter permissions. Unblocking it (`sudo ncu ...`, or setting
  `NVreg_RestrictProfilingToAdminUsers=0`) is a ~30-minute de-risk that converts the central
  "are our kernels compute-bound or memory-bound?" question from inference to measurement.
  **This is the first thing to do in Phase 7 and it needs the user's sudo.**
- Achievable streaming bandwidth (~200 GB/s) is a planning figure carried from prior work, not
  measured on this build. A standalone `memcpy`/stream microbenchmark should establish it before
  the roofline band in `ROOFLINE.md` is treated as final.
- `--runtime nvidia` Docker containers **wedge the device** on this box (stuck `runc` processes
  in uninterruptible D-state; `docker rm -f` hangs). Plain `docker run` works. All CUDA here
  compiles and runs on the host; only CPU-torch oracle work goes in a container.
