# research/FP4_HARDWARE.md — Q1 GROUND TRUTH (2026-08-06)

## Thor `sm_110a` HAS full hardware FP4 tensor-core compute. The old conclusion was wrong.

Proven four independent ways on this box: ptxas, SASS disassembly, a tensor-memory roundtrip on
silicon, and a **numerically exact 4096³ NVFP4 matmul through cuBLASLt**.

**One-line correction:** FP4 on Thor is not a `mma.sync` (warp-level) instruction. It is a
**`tcgen05.mma`** (5th-gen tensor core / UMMA) instruction.

| instruction | Thor `sm_110a` | GB10 / RTX 50 `sm_120a`/`121a` |
|---|---|---|
| `mma.sync…kind::mxf4.block_scale…m16n8k64` | ✗ | ✓ |
| `tcgen05.mma…kind::mxf4.block_scale.scale_vec::2X` (**MXFP4, ue8m0, blk32 — our format**) | **✓** | ✗ |
| `tcgen05.mma…kind::mxf4nvf4.block_scale.scale_vec::4X` (NVFP4, ue4m3, blk16) | **✓** | ✗ |

Complementary, not overlapping. Thor is SM100 lineage — it was literally `sm_101` before CUDA 13.0
renamed it (CUTLASS `generator.py:117`: *"From cuda 13.0, Thor SM is renumbered from 101 to 110"*).

## Measured on this box (cuBLASLt, CUDA 13.0), 4096³

| | TFLOP/s | ratio |
|---|---|---|
| **NVFP4** (E2M1 × ue4m3 blk16) | **571–657** | **4.2–4.7×** |
| FP8 E4M3 | 295–302 | ~2× |
| BF16 | 121–156 | 1× |

The clean 4:2:1 ratio is decisive: **a dequant-to-BF16 emulation can never exceed the BF16 rate**,
so 4.7× BF16 can only be a real FP4 datapath. All numbers are lower bounds (box was at 98% GPU from
concurrent agents). NVIDIA dense-FP4 peak is 1035 TFLOP/s; the marketed "2070 TFLOPS FP4" is
**sparse (2:4)**.

**Tensor Memory is full size**: `tcgen05.alloc` succeeds at 512 columns = **256 KB/SM, byte-identical
to GB200 SM100**. SMEM 232,448 B also identical. SM100-derived kernels need no resource retuning.

## ⚠ THE BUILD BUG — 7 of our 8 scripts were silently losing arch features

`nvcc -arch=sm_110a` runs **two** device passes:

| pass | `.target` | `__CUDA_ARCH_FEAT_SM110_ALL` | tcgen05 |
|---|---|---|---|
| `compute_110` | `sm_110` | **not defined** | **ptxas FAILS** |
| `compute_110a` | `sm_110a` | defined | compiles |

Unguarded `tcgen05` asm dies in pass 1, producing an error that **names `sm_110` even though you
passed `sm_110a`** — exactly the message the prior project recorded and that propagated for months.

**Fix: `-gencode arch=compute_110a,code=sm_110a`** (single pass). `scripts/build.sh:12` already used
the correct form for `cutlass_moe.cu`; **all seven other scripts used the broken form and have now
been corrected.**

**`sm_110f` is a separate trap** — it gets tcgen05 but ptxas rejects *every* `.scale_vec::`
block-scale variant. For FP4 there is exactly one correct target: `compute_110a`/`sm_110a`.

## Library support (measured)

| path | Thor |
|---|---|
| **NVFP4** via cuBLASLt (`VEC16_UE4M3`) | **works**, 5–7 algos, **including M=1** (M=2,4,6 return 0) |
| **MXFP4** via cuBLASLt (`VEC32_UE8M0`) — **our checkpoint's format** | **0 algos, `INVALID_VALUE`** — a *library* gap, not hardware |
| FP6 (E2M3/E3M2) via cuBLASLt | 0 algos — though `tcgen05.mma.kind::mxf8f6f4` and the `cvt`s assemble |
| FP8 blockwise `BLK128x128_32F` (DeepSeek layout) | not supported |
| **CUTLASS SM100 block-scaled NVFP4** | **works** — `cutlass_moe.cu` rebuilt and validated today, maxrel 0.0039 |

So MXFP4 is reachable **only via CUTLASS or hand-written `tcgen05`**, not cuBLASLt.
`tcgen05.cp…b8x16.b4x16_p64` gives hardware FP4 decompression on the SMEM→TMEM copy.

## Two upstream bugs worth patching locally

1. **CUTLASS `include/cutlass/arch/reg_reconfig.h` has no `1100` clause** (gate lists
   900/1000/1010/1030/1200/1210). `setmaxnreg` is silently compiled out on Thor → register spills.
   Upstream issue #3056 / PR #3308 measure **1.74× on FMHA** (1.33 ms → 762 µs; spill traffic
   529 MB → 0). Two-line patch, still unmerged.
2. **CuTeDSL 4.6.0** auto-detect has no `(11,0)` entry → falls through to `sm_110` → block-scaled ops
   rejected. Workaround: `export CUTE_DSL_ARCH=sm_110a`.

## The caveat that decides where this pays

FP4 tensor cores do **not** fix batch-1 decode. cuBLASLt's FP4 kernels reach only **~16–19% of
achievable bandwidth on cold weights at small M** — they are compute-tile kernels, not GEMVs. The
M=1 win from NVFP4 is that it *moves 4× fewer bytes*, not that the tensor core helps.

**`tcgen05`'s minimum tile is 128×8×64** (vs `mma.sync`'s 16×8×64), which is awkward at M=1 and is
the real constraint to design around. **This is why tcgen05 pays at the M=5 verify step, not at M=1
decode** — and the verify is 84% of the speculative round.

Also: **`sm_100f`/`sm_103f` cubins do NOT run on Thor** (`no kernel image available`). Thor is SM100
*lineage* but a separate CUDA *family*; SM100 binaries do not port, only source does.

## Institutional-memory note

`kernels/BUILD_cutlass.md` in this repo already documented a **working NVFP4 GEMM on 2026-07-01**.
That knowledge was lost, and `HARDWARE.md` was written asserting the opposite. NVIDIA staff also
state on the forums: *"SM101 supports the tcgen05 instructions. These can be used instead to do
MXFP4 or NVFP4 compute."* The marketing claim of "native FP4" was accurate; our notes were not.

PTX ISA: §9.7.17 (TensorCore 5th Gen), §9.7.17.10.9.1 (`tcgen05.mma`), §9.7.17.10.7 (block scaling),
§9.7.17.7.1 (alloc/dealloc), §9.7.17.9.2 (`tcgen05.cp`), §9.7.15.3 (`mma.sync` block scaling —
sm_120 only), §5.2.3, §9.7.9.22.
