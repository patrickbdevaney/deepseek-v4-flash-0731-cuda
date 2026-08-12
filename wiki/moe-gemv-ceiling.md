# MoE GEMV — where the 33 % goes

> # ⛔ RETRACTED — F137, 2026-08-12. THE 155 GB/s IS AN ACCOUNTING ERROR.
>
> **Everything below is scored on a rate that divides the ROUTED expert bytes by a window in which
> the SHARED expert is concurrently streaming its own 1082.20 MB.** `moe.cu:436` forks the shared
> expert onto `g_side` before the routed gather and `moe.cu:516` joins it after `moe:combine`, which
> is why `moe:shared` prints 0.24 ms instead of its real cost — a fork this repo priced in F55 and
> then never fed back into the rate. Counting the concurrent bytes, on the same unchanged marks:
>
> | K | window | routed | +shared | GB/s | the number below |
> |---|---|---|---|---|---|
> | 1 | 21.08 ms | 3.368 GiB | 4.425 GiB | **215.0** | 163.6 |
> | 5 | 60.31 ms | 10.363 GiB | 11.420 GiB | **193.9** | 176.0 |
>
> Streaming reference, same box, same binary, clocks pinned: **202.5–214.9 GB/s**. **At M=1 the MoE
> GEMV is AT THE ROOFLINE.** `tools/moe_gemv_bench.cu` confirms it independently — the shipped kernel
> at the real M=1 grouping measures 200.7 GB/s against a 208.7 GB/s in-binary roofline.
>
> **So "the MoE GEMV is 67 % of achievable" is false, "MoE GEMV efficiency" is not a lever, and
> `LEVERS.md` §8 ranking it #1 was ranked off this page.** What is real is a ~10 % loss at the K=5
> *verify* grouping, which is the per-row inner loop (issue-bound, linear at 0.139 ms/row on a
> 0.244 ms intercept), not bandwidth — ceiling ~+4 % end-to-end and unreachable. See **F137** for the
> controls, the `ncu` attribution, the GEMV-vs-mma crossover at R=5.05, and the corrected base-AR
> budget table. The rest of this page is kept only as the record of how the number was arrived at.
>
> The alignment analysis below is unaffected and remains correct as description; it is just not worth
> what this page priced it at.

> **ANSWERED, and it redirects the lever.** Open question 3 below asked whether Laguna's 206 GB/s
> advantage is kernel quality or mix. It is **mix**. Laguna's own MoE kernel `k_moe_gateup_rp` runs
> at **177 GB/s = 70 % of its 254 ceiling**; ours runs at **155 GB/s = 67 % of our 233**. Those are
> the same kernel quality. The gap is entirely in the **dense/MLA GEMVs**: theirs measure
> **228-236 GB/s** (`o_proj` 228, `q_proj` 236), ours measure **115-195**. And dense is **72 % of our
> `B_tok`**. Priced: dense at 228 GB/s takes the AR step 79.1 → 65.0 ms, i.e. **12.6 → 15.4 tok/s**,
> which is essentially the whole gap the cross-model comparison found. **The MoE GEMV is not lever #1;
> the dense/MLA GEMV path is.** The alignment finding below still stands and still applies — the same
> `WeightStore` policy misaligns every tensor, not only the experts — but it should be aimed at the
> dense path first, where the bytes and the deficit both are.

**Status: research, not yet implemented.** Lever #1 in `LEVERS.md` §8. Everything below is either
measured (marked) or derived from measured numbers (marked). Nothing here has been built.

## The gap, priced two ways

| source | number |
|---|---|
| measured MoE GEMV rate (F67, at the optimal RB=2) | **155 GB/s = 67 % of roofline**, 62 registers, 63 % occupancy |
| this project's *optimistic* AR floor | 15.98 tok/s = 196 GB/s effective |
| **Laguna, same box, measured** | **206 GB/s effective at AR** — *above* our optimistic floor |

The third row is the one that matters: a peer pure-CUDA engine on this hardware sustains an
effective rate our own roofline calls a best case. Matching it puts base AR at **16.8 tok/s** from
today's 13.8, and lifts every speculative ceiling with it, because the MoE term sits in both the
intercept and the slope of `verify_ms(K) = 69.9 + 17.11K`.

## Root cause: the device pointers inherit the *file's* byte offsets

`include/weight_store.h` loads one buffer per shard and then resolves every tensor as

    d.dev = dev_base_[owner] + (t.data - host_base_[owner])      // shard base + FILE offset

So a tensor's device address carries whatever alignment the safetensors file gave it. And the file
gives a consistently bad one — measured across all 48 shard headers (F66):

> **43,470 of 44,436 expert tensors sit at `data_offset % 16 == 8`**, 966 at 12, **none at 0**.

Every expert weight pointer on the device is therefore misaligned by 8 bytes, and
`kernels/tc_moe_gemm.cu` compensates in the inner loop:

```
const uint8_t* wa = wb + (long)g*512 + lane*16 - off;
uint4 A=__ldcs((const uint4*)wa), B=__ldcs((const uint4*)(wa+16));
uint4 W=tcm_funnel16(A,B,k0f,shf);
```

Two `uint4` loads and a 4-way-branched funnel-shift to produce the 16 bytes this lane actually
wants. The kernel even notes the identity case — `off==0 (k0=0,sh=0) -> r=A` — which never fires,
because the loader never produces an aligned pointer.

## What the funnel actually costs — and what it does *not*

**It does not double DRAM traffic.** Lane `L` reads `[wa, wa+32)` and lane `L+1` reads
`[wa+16, wa+48)`, so consecutive lanes overlap by 16 B. A warp's footprint is 32×16 + 16 = **528 B
for 512 B of useful weight**, i.e. ~3 % extra, absorbed by L1/L2. Anyone pricing this as a 2x
bandwidth loss will over-promise the fix.

**What it costs is registers and issue slots**, which is exactly what the measurement says is
binding:

- 2 `__ldcs` per iteration instead of 1
- 4 `__funnelshift_r` per `uint4`, under a 4-way uniform branch on `k0`
- **`A` and `B` held live simultaneously — 8 registers where 4 would do**

F67 measured the result and named the cause: *"Even at the optimum RB=2 it runs 62 registers / 63 %
occupancy / 155 GB/s = 67 % of roofline. The register budget is dominated by the funnel pair (B6)
and `acc[RB][BN]`. B6 is the cheapest way in."*

At 63 % occupancy a memory-bound GEMV cannot cover DRAM latency — it is latency-bound, not
bandwidth-bound, which is precisely the signature of running below roofline while issuing enough
loads.

## Why B6 was closed, and why that closure was aimed at the wrong target

`LEVERS.md` B6 proposed *"skip the funnel when the weight pointer is 16 B-aligned"* and was closed
with **"it never is."** That is true and it is a statement about the *file*. B6' tried to supply the
funnel partner by warp shuffle and measured **worse** (423.8 → 650.6 µs): lane 31 still needs a real
load and it is predicated rather than branched away, so the warp pays both paths plus 8 shuffles.

Both attempts took the misalignment as a given. **It is not a given — it is a loader policy.** The
device layout is ours to choose; only the file layout is fixed.

## The proposed change

Place tensors at 16 B-aligned device offsets instead of shard-relative file offsets. The in-place
repack (`tc_moe_gemm.cu`, "REPACK-AT-LOAD", deliberately in-place to avoid an 82 GB doubling) stays
in place and simply becomes aligned. Then `off == 0`, the funnel collapses to its identity path, one
`uint4` load per iteration, and ~4-8 registers come back.

Cost, estimated:

- **memory**: < 16 B padding per tensor × ~47 k tensors ≈ **700 KB**. Negligible against 108 GB.
- **load time**: ~47 k `pread`s of ~2 MB instead of 48 of ~2.2 GB. Sequential, large; expected to be
  comparable, but it must be measured — L1 gate reports 44.6 s cold today.
- **risk**: contained. Alignment is a placement change; every kernel reads through `WeightStore`, and
  a tensor that is *more* aligned cannot become less correct. The `posix_fadvise` page-cache drop and
  the managed/pinned split both need rework, which is where the actual bugs would live.

## Expected gain, with honest bounds

MoE is **3.69 GB of the 12.26 GB step (30 %)**. At the measured 155 GB/s it costs ~23.8 ms of the
72.5 ms AR step.

| if occupancy takes the MoE GEMV to | MoE term | AR step | AR tok/s |
|---|---|---|---|
| 155 GB/s (today) | 23.8 ms | 72.5 | 13.8 |
| 180 GB/s | 20.5 | 69.2 | **14.5** |
| 200 GB/s | 18.5 | 67.2 | **14.9** |
| 233 GB/s (roofline, unreachable for a GEMV) | 15.8 | 64.5 | 15.5 |

**So alignment alone is worth roughly +5 to +8 %, not the +22 % the Laguna comparison implies.** It
is the cheapest and best-understood item, and it is not sufficient on its own. The rest of the gap
has to come from somewhere else, and the honest position is that we do not yet know where.

## Open questions, in priority order

1. **Is the M=1 path even the right kernel?** `tc_w4a8_pp_kernel` accumulates through
   `mma_m16n8k16` — a tensor-core tile — for a GEMV that has *one* row of A. `src/decode.cu:216`
   records that a dequant rewrite (`cvt.f16x2.e2m1x2` + `__hfma2`) *beats the mma path at M=1*. Which
   kernel actually runs at M=1 today, and what does each measure? This could be larger than the
   alignment item.
2. **What is a GEMV's real roofline here?** F67 quotes per-kernel achievable rates of 115-195 GB/s
   for the MLA GEMVs, because a GEMV has one row of reuse. If the MoE GEMV's honest bar is ~190 and
   not 233, then 155 → 190 is the whole prize and the Laguna comparison is telling us something about
   *their* mix rather than our headroom.
3. **Where does Laguna's 206 GB/s actually come from?** Its dense share is 60 % of `B_tok` against
   our 72 %, and dense GEMMs reach higher rates than MoE GEMVs. The 22 % gap may be partly a
   composition artefact rather than pure kernel quality — this needs its per-kernel breakdown, not
   its end-to-end number.

**Question 3 is the one to answer first**, because it decides whether the target is 16.8 tok/s or
something closer to 15. Everything else is sized against it.
