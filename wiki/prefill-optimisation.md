# B9 — prefill optimisation

Prefill was never a decode lever and still is not. It matters because **capture is teacher-forced
prefill**, so prefill throughput is a direct multiplier on the draft-head fine-tune's wall time.

**47.9 → 62.4 tok/s (+30.3 %), all bit-exact.** 20 K-sample capture: 4.9 days → 3.8 days.

---

## 1. Why it was slow: one sentence

**Seventeen cycles optimised decode inside functions that prefill shares, and prefill was never
measured.** Every kernel on the prefill path had been tuned for M=1 and was being handed a
1022-token batch.

F75 measured prefill end-to-end at 48 tok/s — only **3.5× the M=1 decode rate for ~1000× the work** —
and stopped there. B9 is what happened when someone looked.

---

## 2. Attribution first

Nothing was optimised until the whole prefill was accounted for. Three instrumentation runs:

- `cblock_prefill_cache` and `block_prefill_cache` carried **no `dprof` marks at all** — only the MoE
  had them, because those live inside `moe_forward`.
- `compressed_attn_forward` — 6.9 s, **32 % of the prefill** — had no sub-marks at any level.
- The MoE tile counter had to be made **RB-aware** (see §4).

Result at PS=1022, **99.98 % accounted**:

| region | ms | % |
|---|---|---|
| MoE | 9100.0 | 42.6 |
| ATTENTION (`compressed_attn_forward`) | 9086.0 | 42.6 |
| KV cache population (`compressed_attn_cache_r4`) | 2233.5 | 10.5 |
| `hc_pre` (attn+ffn) | 805.9 | 3.8 |
| everything else | 121.5 | 0.6 |

The instrument is free: 21351.6 ms fully marked against 21362.4 ms clean = **−0.05 %**.

---

## 3. The four fixes

| # | finding | change | prefill | tok/s |
|---|---|---|---|---|
| 0 | F84 | baseline | 21351.6 ms | 47.9 |
| 1 | **F85** | `MOE_MMA=1` — tensor-core MoE instead of the M=1 GEMV | 18300 ms | 55.9 (+16.7 %) |
| 2 | **F86** | `sparse_attn`: size registers to real `d`, multiple heads per block | 17364 ms | 58.9 (+5.4 %) |
| 3 | **F88** | `tc_ogroup`: amortise weight dequant across m-tiles | **16373 ms** | **62.4 (+6.0 %)** |

### 3.1 MoE — GEMV vs mma (F85)

`src/decode.cu` makes the M=1 fp4 GEMV the **process-wide** default, which is right for decode
(350 GB/s vs 121 at M=1) and exactly wrong at 1022 tokens.

`tc_ensure_repacked` mutates weights **in place** and the GEMV needs the original layout, so the two
paths cannot coexist in one process — it is a process-wide choice. That maps cleanly onto the two
workloads: **capture is prefill → `MOE_MMA=1`; serving is decode → GEMV.** Costs −16 % decode, which
is irrelevant during capture.

### 3.2 `sparse_attn` (F86) — two bit-exact defects

- **42 dead registers.** `qreg[32]`/`acc[32]` were sized for the kernel's contract `d ≤ 1024`, but
  all ten call sites pass `HEAD_DIM = 512`, so `per = 16` and half of both arrays never held
  anything. `ptxas -v`: **128 → 86 registers**, 0 spills either way, occupancy ceiling 16 → ~23
  warps/SM.
- **One warp per block.** `num_key_value_heads == 1`, so `kv` has no head dimension and all 64 heads
  of a query read *identical* key vectors — as 64 separate 32-thread blocks free to land on 64
  different SMs. Putting `HPB` heads of one query in one block puts them on one SM, where reads
  2…HPB hit L1.

### 3.3 `tc_ogroup` (F88) — 64× redundant dequant

The bs>16 path already *reached* tensor cores, but re-loaded and re-**dequantised** each weight row
once per m-tile: two `exp2f`, four scalar byte reads and four fp8→half converts per k-step, repeated
for all 64 m-tiles, for bytes that never change. Hoisting the load+dequant out of an m-tile loop is
the F64 row-amortisation transformation applied to the tensor-core path. `cattn:ogroup`
**2786 → 1866 ms (−33 %)**.

---

## 4. The measurement that had to be corrected

F84 reported the prefill MoE at **3.26× byte redundancy, latency-bound at 12.5 % of roofline**, and
recommended "use larger tiles". **All three were wrong, from one mistake.**

The weight load in `k_grouped_fp4_gemv_e8m0` sits **inside** the `rb` loop, so a 16-row tile costs
`ceil(me/RB)` full weight reads, not one. Traffic is a function of **`RB`**, and the tile row cap
does not enter it at all — splitting 43 rows into 16-row or 64-row tiles gives identical bytes.
**The recommended fix would have moved nothing.**

Corrected instrument, measured on the same run:

```
tiles 19779   RB-chunks 68332   -> redundancy 11.26x   (tiles-only would say 3.26x and be wrong)
ACTUAL 913.6 GB vs IDEAL 81.2 GB
```

Hand-computed 892.9 GB against measured 913.6 GB — 2.3 %. So the prefill MoE was **bandwidth-bound
at ~42 % of roofline while moving 11× the bytes it needs**, not latency-bound at 12.5 %.

`RB` is not the lever either: `acc[RB][BN]` is live regardless of real row count, and F64/F70
measured `RB=8` at 85 registers / 39.4 % occupancy, losing in situ. Amortising 43 rows needs `RB≈43`,
which that kernel cannot hold. Hence the mma path.

---

## 5. The pattern, four for four

| kernel | tuned for | wrong for prefill because |
|---|---|---|
| MoE `RB=4` | 1–5 rows/expert at decode | 43 rows/expert → 11 weight reads |
| MoE GEMV vs mma | M=1: 350 vs 121 GB/s | M=1022 inverts it |
| `sparse_attn` regs + 1 warp/block | generic `d≤1024`, 64 warps total | `d=512`, 65,408 warps |
| `tc_ogroup` m-tile dequant | bs≤16 has 1 m-tile | bs=1022 has 64 |

**None of these is a bug.** Each is a correct decision for the batch size it was made at, applied
unchanged to a batch a thousand times larger.

---

## 6. What is left, priced against measured peaks

`tools/flops_probe.cu` finally measured compute (see [`hardware-sm110a.md`](hardware-sm110a.md)):
**FP32 5.45 / BF16 53.1 / FP8 92.2 TFLOPS.** That settles the remaining question:

- **fp32 tuning is nearly exhausted.** `cattn:sparse` runs at 19 % of the *fp32* peak. Perfect fp32
  would take it 2.02 s → 0.39 s, and realistically half that. The remaining marks are worth tenths
  of a second each.
- **Tensor cores are the order-of-magnitude lever.** bf16 is **9.7×** fp32; fp8 is **16.9×**. A bf16
  flash-attention score path at a conservative 30–50 % of peak is **15–25×** on `cattn:sparse`.

Sized end to end: `sparse`+`indexer` on bf16 TC @40 % and `ogroup` on fp8 TC @30 % would give
~10.9 s ≈ 94 tok/s, taking 20 K capture to ~2.5 days.

**Two caveats.** 30–50 % of mma peak is a friendly assumption for a path that gathers keys through a
sparse index list. And **bit-exactness is off the table** — every fix above was gated at
`rms=0.00e+00`; a bf16 score path is a numerics change needing a cosine-tolerance gate, with the
LOSSLESS spec gate as the real backstop.

Remaining marks: `cattn:compress` 2231, `cattn:indexer` 2081, `cattn:sparse` 2011, `cattn:ogroup`
1866 ms.

## 7. One of B9's four fixes had gone negative, and 1.7 took it back (2026-08-20)

B9's fix (2) was **key reuse in `sparse_attn`**: `num_key_value_heads == 1`, so all 64 heads of a
query read identical KV rows, and putting `HPB` heads in one block puts them on one SM where the
2nd..HPB'th reads hit L1. It shipped `HPB=8` at prefill, sized on the 1022-token shape.

**Re-measured 2026-08-20 at exactly that shape, `hpb=8` is 0.80x — a 20 % regression against the
kernel it replaced.**

```
m=1022 topk=1277 (1022-token prefill)      m=256 topk=320 (short prefill)
  hpb=1 smem=0   263.2 ms   1.00x            hpb=1 smem=0    15.78 ms  1.00x
  hpb=8 smem=0   328.5      0.80x  <-- shipped default
  hpb=4 smem=1   205.1      1.28x            hpb=4 smem=1    13.03     1.21x  <-- now default
```

Two things had to be true at once for this to survive. The reuse *mechanism* was never the binding
constraint — `hpb` on its own is a null at every shape, L1 was already catching it
([`negative-results.md` §4g](negative-results.md)) — so the only thing `HPB=8` was actually doing
at prefill was consolidating 65,408 warps onto fewer, wider blocks and losing occupancy. And
nothing re-ran the sweep after the shape moved ([`measurement-and-traps.md`
§28](measurement-and-traps.md)).

Ladder 1.7's default — `hpb=4`, with `smem=1` above 1024 warps — is **1.28x at the 1022-token
prefill and 1.21x at the short one**, bit-exact against the pre-B9 launch by memcmp, and
`gate_prefill_len` passes with 0 prefix mismatches. So prefill gets B9's claimed win for the first
time, by a different mechanism than B9 named: shared-memory staging of the gathered row with
`float4` loads, not L1 reuse. Mechanism in
[`kernel-optimisations.md` §2.9](kernel-optimisations.md).

**This is not yet an end-to-end prefill number.** The 62.4 tok/s in §4 was measured with the old
default and has not been re-run; what is measured here is the kernel, at the prefill shapes, in
`gate_sparse_hpb`. `cattn:sparse` is one mark of a prefill, so the end-to-end gain will be smaller
than 1.28x and is currently **unmeasured** — stated here rather than estimated, because §4 of this
page is about exactly that kind of correction.
