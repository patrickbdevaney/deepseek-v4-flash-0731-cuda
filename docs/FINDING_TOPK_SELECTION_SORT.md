# FINDING: the context-linear decode cost is a one-thread selection sort

**Confirmed 2026-08-19 by three independent measurements that agree.** Not yet fixed.

## The kernel

`kernels/compressed_decode.cu:550`, on the device-pos decode path, once per ratio-4 layer (21 of
them) per target forward:

```c
__global__ void k_topk_masked(int* out, const float* score, int Tmax, int topk, int winmax){
    if(threadIdx.x||blockIdx.x) return;                       // 1 thread of 32; 1 SM of 20
    extern __shared__ float sh[]; for(int t=0;t<Tmax;++t) sh[t]=score[t];
    for(int k=0;k<topk;++k){ float best=-1e29f; int bi=-1;
        for(int t=0;t<Tmax;++t) if(sh[t]>best){best=sh[t];bi=t;}   // O(topk x T) selection sort
        if(bi>=0){ sh[bi]=-1e30f; out[k]=winmax+bi; } else out[k]=-1; } }
```

Launched `<<<1,32,...>>>`: 31 lanes return immediately, 19 of 20 SMs idle, and the survivor runs
`topk x T` = 512 x (context/4) comparisons serially. `k_topk_decode`, `k_topk_verify` and
`k_topk_offset` are the same shape.

## Three measurements, one conclusion

**1. Regression over 2,156 real battery generations** (`tools/decode_model.py`):
`ms/forward = 136.44 + 30.053 x (context/1000)`, R^2 **0.965**, measured range 71-6592.
Cost tracks GENERATED tokens (R^2 0.871 alone), not prompt length (R^2 **0.082**) -- so the scan
is bounded by live context, and this is not a fixed-`Tmax` artefact.

**2. The kernel benchmarked standalone on this Thor**, compiled from this source, `k=512`:

| T (ctx) | shipped `<<<1,32>>>` | single-CTA radix select (exact) | speedup |
|---|---|---|---|
| 1024 (4k) | 9,672 us | 10.26 us | 943x |
| 2048 (8k) | 19,115 us | 23.83 us | 802x |
| 6000 (24k) | **55,745 us** | **26.13 us** | **2,134x** |
| 12288 (49k) | *launch fails* | 39.65 us | — |

Index sets verified bit-identical at every T. Slope of the shipped kernel: 9.3 us per unit T per
layer = 48.7 ms per 1000 tokens of context across 21 layers.

**3. Cycle arithmetic closes it.** Iterations per forward = 512 x (context/4) x 21 x tau(2.91):

| context | iters/forward | measured ms | **cycles/iter** |
|---|---|---|---|
| 2000 | 15.6M | 60.1 | **5.33** |
| 4000 | 31.3M | 120.2 | **5.33** |
| 6592 | 51.6M | 198.1 | **5.33** |

Constant at 5.33 cycles across a 3.3x context range, which is exactly what a shared-memory load,
compare, predicated update and loop cost. The inner loop alone accounts for the whole term.

## Why it was recorded as a null

`LOOP_LOG.md` Finding 71 profiled the sibling kernel at **T~19** -- context ~76 tokens -- where
`topk = min(512,19) = 19` and the sort is 361 comparisons. That produced `i:topk = 0.12 ms` and
`DECODE_MAX_REPORT.md:130`'s claim of being "2.6x faster than SGLang's optimised 15 us". SGLang's
15 us is K=512 over a **1M-token** context; ours was k=19 over 19 candidates. The two numbers were
never comparable, and `LEVERS.md` B0's "the class paid twice and is now dry" rests on that
comparison. **Every prior decode measurement in this repo was taken at short context, which is the
one regime where this kernel is free.**

## Two further defects found in the same path

**`index_score` is 2% of achievable bandwidth.** At T=6000, S=1, H=64, d=128: measured 592-658 us
against an 8.2 us pure-stream floor. Cost is exactly linear in H because each of the T warps
re-reads the entire 32 KB query tensor from global -- **196.6 MB of L1/L2 traffic per layer** --
plus a 6-deep dependent `__shfl` chain per head. It is a GEMM: `L[T,64] = Kc[T,128] x Q^T[128,64]`.
cuBLAS TF32 does it in 28.75 us; with a fused relu/head-weight epilogue, ~39 us total = **15.2x**.

**A hard context ceiling at ~49k.** `topk_scan_smem(n)` requests ~4n bytes of dynamic shared memory
with no `cudaFuncSetAttribute` opt-in, against a 48 KiB default limit (measured
`sharedMemPerBlock=49152`, `sharedMemPerBlockOptin=232448`). At T=12288 the launch silently fails
and returns garbage. The block-parallel replacement needs no dynamic shared memory, so the ceiling
disappears as a side effect.

## Status

Not fixed. `RESEARCH_PROMPT_DECODE_ZENITH.md` carries the full option space; three further research
fronts (weight path, speculation, KV precision) are still open at the time of writing.
