# Dense MLA GEMV — the real lever, and the invariant it collides with

**Status: analysis, nothing built.** This supersedes `moe-gemv-ceiling.md` as lever #1. The MoE GEMV
runs at 67 % of our roofline against Laguna's 70 % of theirs — the same kernel quality. The gap is
here, in the dense path, which is **72 % of `B_tok`**.

## The shapes, the parallelism, and the measured rates

Per layer, from the checkpoint headers, against F67's measured per-kernel rates:

| tensor | shape | MB | N | warps (N/8) | warps/SM | measured |
|---|---|---|---|---|---|---|
| `wq_a` | [1024, 4096] | 4.2 | 1024 | 128 | **6.4** | **115 GB/s** |
| `wkv` | [512, 4096] | 2.1 | 512 | 64 | **3.2** | (unmeasured, worst shape) |
| `wo_a` | [8192, 4096] | 33.6 | 8192 | 1024 | 51.2 | 168 GB/s |
| `wo_b` | [4096, 8192] | 33.6 | 4096 | 512 | 25.6 | 185 GB/s |
| `wq_b` | [32768, 1024] | 33.6 | 32768 | 4096 | 204.8 | 195 GB/s |

**107 MB per layer × 43 = 4.60 GB/token of MLA weight.** Thor has 20 SMs.

Two separate problems, and they need different fixes:

1. **`wq_a` and `wkv` are parallelism-starved** — 6.4 and 3.2 warps per SM. A GEMV parallelised over
   `N/8` warps simply cannot fill the device at N=1024 or 512, and 115 GB/s is what that looks like.
   Split-K is the textbook fix. **But they are 6.3 MB of 107 MB — 6 % of the bytes.** Fixing them
   completely is worth ~1-2 % end to end. Low value, do not start here.
2. **The big three carry 94 % of the bytes and run at 168-195 GB/s** against a streaming benchmark
   that sustains **224-237 GB/s** on this box. They have 25-205 warps/SM, so parallelism is *not*
   their problem. This is the +22 % the cross-model comparison found, and it is the whole job.

## Alignment is not the answer here (unlike the MoE)

Every one of these tensors is misaligned — measured across all 43 layers, **`wq_a`, `wq_b`, `wo_a`,
`wo_b`, `wkv` are 0 % aligned, 100 % at `data_offset % 16 == 8`** (only `attn_sink` and the
compressor tensors, 108 of 658, land at 0). So the loader policy described in
`moe-gemv-ceiling.md` hits the dense path just as hard.

**But this kernel does not care.** `fp8_gemv_m1_kernel` loads **4-byte** words (`unsigned`), not
`uint4`. A warp's 32 lanes × 4 B = 128 B is fully coalesced regardless of the 8-byte offset; the only
cost is one extra cache line per row (`ceil((K+8)/128)` vs `K/128`), i.e. **~3 % at K=4096**.

That is worth stating plainly because it would be easy to carry the alignment finding across from
the MoE work and over-promise it. **Align the tensors for the MoE kernel's sake, not for this one.**

## What is actually limiting the big three

The kernel already fixed the ILP problem once, and the comment records it:

> *"The original loop issued ONE 4-byte load per iteration and consumed it immediately — ILP = 1.
> Measured on this box: a streaming loop at ILP=1 sustains 110-132 GB/s; at ILP>=2 it sustains
> 224-237. Our whole engine measured 116.6 GB/s, which is squarely the ILP=1 number."*

It now unrolls by 4 and issues 8 loads before consuming any. Yet the big three sit at 168-195, not
224-237 — **72-82 % of the streaming rate**, with the loads already in flight.

The remaining suspect is the **accumulator dependency chain**:

```
acc += dot4(av0,bv0) * as[kb+0] * bsr[kb+0];
acc += dot4(av1,bv1) * as[kb+1] * bsr[kb+1];
acc += dot4(av2,bv2) * as[kb+2] * bsr[kb+2];
acc += dot4(av3,bv3) * as[kb+3] * bsr[kb+3];
```

A **single** `acc`, four strictly serial FP adds per iteration, each waiting on its `dot4` — and
`dot4` is itself a serial chain of 4 FP adds. The four `dot4`s are mutually independent and can
overlap; the four `acc +=` cannot. The textbook fix is four accumulators summed at the end.

## The invariant it collides with, which is why this is not a one-line patch

The kernel's own comment states the constraint:

> *"Accumulation order across kb is UNCHANGED (0,1,2,3,4,...), so this stays bit-exact."*

Four accumulators **reassociate** the sum. Floating-point addition is not associative, so the result
changes in the last ulp — and this project gates on correctness against a PyTorch oracle (G3/G4/G8)
and on `LOSSLESS: first 8 tokens match base AR` on every run. A last-ulp change can flip an argmax
and therefore a token.

So the honest framing of this lever is **not** "add four accumulators, get 20 %". It is:

1. Measure the ceiling first with a **throwaway** multi-accumulator variant, purely to find out
   whether the dependency chain is really what costs 168→228. If it is not, stop; the hypothesis is
   dead and nothing was risked.
2. Only if it is, decide what the project is willing to trade. Options, in increasing order of
   nerve: keep the order and find the throughput elsewhere (more outstanding loads, better scale
   handling, `__ldg`/`__ldcs` hints, wider vector loads now that alignment is on the table);
   accept reassociation for the *draft* path only, where a wrong token is rejected by verification
   anyway and costs nothing but a retry; or accept it globally and re-baseline every number in the
   registry.

**The draft-path-only option is the interesting one** and it is unique to a speculative engine: the
drafter's arithmetic does not have to be bit-exact, because the target verifies every token it
proposes. A faster, slightly-reassociated drafter that proposes the same tokens 99.9 % of the time
costs 0.1 % acceptance and breaks no invariant at all. That is worth measuring before touching the
target path.

## Order of work

1. **Throwaway multi-accumulator microbenchmark** on `wq_b`/`wo_a`/`wo_b` shapes. One question: does
   breaking the chain reach 224-237, or does it stall at 195 anyway? ~1 h, no engine change.
2. If yes → the draft-path-only variant, gated on acceptance rather than bit-exactness.
3. If no → the limiter is elsewhere; instrument for scale-load traffic and L2 behaviour on `A`,
   which at M=1 is one row reused across all N and should never leave L2.
4. Split-K for `wq_a`/`wkv` last. It is real and it is 6 % of the bytes.

Everything above is analysis of shapes, measured rates already in the log, and source. **No kernel
has been written and no number here is new measurement.**
