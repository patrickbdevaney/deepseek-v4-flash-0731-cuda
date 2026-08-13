# Why the needle will not move: AR and speculative decode against the measured roofline

This page exists because "we are at the roofline" is the kind of claim that is usually asserted and
almost never shown. Everything here is a measured number from this box with the run that produced it
named, and where a number was previously wrong it says so and says by how much.

**Short version.** Base autoregressive decode runs at **160 GB/s = 77 % of a measured 208.7 GB/s
streaming roofline**, and the largest single region inside it — the MoE — runs at **94 %**. The
remaining 23 % is not one addressable inefficiency; it is the sum of four regions, three of which
are within 82-94 % of achievable and the fourth of which is launch latency that does not scale with
grid size. Speculative decode multiplies through this by a factor that is bounded by acceptance, not
by bandwidth, and the acceptance ceiling is set by the draft head that ships. Every lever that could
plausibly have been worth more than ~4 % has now been measured and closed.

## What the roofline actually is on this box

| quantity | value | how |
|---|---|---|
| streaming read rate, clocks pinned | **202.5 - 214.9 GB/s** | `k_stream` in-binary, `tools/moe_gemv_bench.cu`, `tools/hc_glue_bench.cu` |
| the figure used throughout | **208.7 GB/s** | the in-binary reference measured alongside the kernel being scored |
| `HARDWARE.md` quoted | 240 idle / 212 contended | vendor/idle, not what a loaded engine sees |
| L2 | 33.6 MB | this is a trap, see below |
| empty-kernel launch floor | **2.07 µs**, and *flat in grid size* | measured at both `<<<1,32>>>` and `<<<24,256>>>` |
| SMs | 20 | |

Two measurement traps had to be defeated before any of the numbers below could be trusted, and both
had already produced a wrong published figure in this repo:

1. **L2 residency.** One M=1 MoE launch is 26.7 MB against a 33.6 MB L2. Benched over a small pool
   it reads **224 GB/s** — above the machine's DRAM rate — because it is not going to DRAM. Fixed by
   sizing the weight pool past L2 (pool=32); the anomaly vanished.
2. **Concurrency in the denominator.** See below. This one cost four optimisation cycles.

## Base AR, region by region

K=1, the 70.03 ms step from `evidence/kchunk.log`; bytes from `MODEL_INVENTORY.md`; scored against
208.7 GB/s.

| region | ms | % of step | bytes | GB/s | % of roofline |
|---|---|---|---|---|---|
| **MoE** (routed + shared, concurrent) | 23.20 | 33 % | 4531 MB | **195** | **94 %** |
| **Attention** (MLA + compressor + indexer) | 31.40 | 45 % | 5400 MB | 172 | 82 % |
| `lm_head` | 5.80 | 8 % | 1059 MB | 183 | 88 % |
| HC / rmsnorm / router / glue | 8.53 | 12 % | 211 MB | 25 | latency-bound |
| **whole step** | **70.03** | | **11.20 GB** | **160** | **77 %** |

Read the last column, not the third. The 23 % that separates 160 from 208.7 GB/s is **not** sitting
in one place waiting to be collected — it is 6 % lost in the MoE, 18 % in attention, 12 % in
`lm_head`, and one region that is not a bandwidth problem at all.

## The MoE: at the roofline, and the four cycles spent believing otherwise

For four optimisation cycles `LEVERS.md` ranked "MoE GEMV efficiency" as lever #1 on the strength of
**155 GB/s = 67 % of achievable**. That number was wrong, and the way it was wrong is worth keeping:

> it divided the **routed** expert bytes by a window in which the **shared** expert was concurrently
> streaming its own 1082.20 MB.

`moe.cu:436` forks the shared expert onto `g_side` before the routed gather and `moe.cu:516` joins
it after `moe:combine` — which is exactly why the `moe:shared` mark prints 0.24 ms rather than its
real cost. The fork was priced in F55 and then never fed back into the rate. Counting the concurrent
bytes, on unchanged marks:

| K | window (`w1w3`+`act`+`w2`) | routed | + shared | **GB/s** | previously published |
|---|---|---|---|---|---|
| 1 | 21.08 ms | 3.368 GiB | 4.425 GiB | **215.0** | 163.6 |
| 2 | 31.44 ms | 5.429 GiB | 6.486 GiB | **211.2** | 176.8 |
| 5 | 60.31 ms | 10.363 GiB | 11.420 GiB | **193.9** | 176.0 |

An **isolated** bench agrees independently — `tools/moe_gemv_bench.cu` runs the shipped entry point
at the real shapes and real groupings, pointers offset by 8 as the loader produces them, over a pool
past L2: **200.7 GB/s (w1w3) / 225.3 (w2) at M=1** against a 208.7 in-binary roofline.

**The MoE GEMV is at the roofline at M=1. It is the best-optimised region in the engine.** The
lesson generalises past this kernel: *any* rate computed over a window containing a forked stream is
an underestimate by exactly the forked bytes, and this engine forks in more than one place.

What survives is a ~10 % loss at the **K=5 verify** grouping, and it is not bandwidth. With DRAM
bytes held constant, time is linear in rows per expert — 0.400 / 0.510 / 0.645 / 0.794 / 0.951 ms at
R=1..5, i.e. **0.139 ms per row on a 0.244 ms intercept**. `ncu` attributes it: Compute throughput
30.2 % -> 47.8 %, warp cycles per issued instruction 29.8 -> 15.5, occupancy 74.8 % -> 56.9 %, Block
Limit Registers 9 -> 7. It is the per-row inner loop, issue-bound. Four candidate causes were
eliminated by control: tile count, early-exit `grid.y` blocks, `RB` row-blocking, and `BN`.

Ceiling if the row arithmetic vanished **entirely**, by two independent routes: **+4.0 %** (in situ,
K=5 window at the K=1 rate) and **+3.7 %** (bench, rows beyond the first free). It cannot vanish
entirely — every instruction cut costs registers against a 70-register kernel already at 7 blocks/SM,
and this engine has spent F76, F78, F79 and F81 on that exact trade in the sibling fp8 tile and won
none of them, with in-situ delivery running 1-6 % of the bench prediction.

### The tensor-core alternative, finally quantified

`LEVERS.md` §9 lever #3 (m16 B-repack, use the `mma` path) had been open for cycles without a
number. Measured head to head at the real shapes:

| path | cost model | at block 6 (R=1.67) | crossover |
|---|---|---|---|
| GEMV `k_grouped_fp4_gemv_e8m0` | 0.244 + 0.139 R ms | **0.476 ms** | |
| mma `tc_fp4_grouped_gemm_e8m0` | 0.504 + 0.087 R ms | 0.649 ms | **R = 5.05** |

Block 6 presents **1.67 rows per expert**. The tensor-core path does not break even until 5.05.
**Dead for decode at any block size this engine will run; alive for prefill**, where R is in the
hundreds.

## The glue: 12 % of the step that speculation has already mostly removed

The 8.53 ms region moves 211 MB — 25 GB/s — so it is latency, and the instinct is that latency is
attackable. Split at the decode shape by `tools/hc_glue_bench.cu` (no checkpoint load):

| launch | µs | share |
|---|---|---|
| `k_mixes` | 6.31 | 23.3 % |
| `hc_sinkhorn` | **16.57** | **61.2 %** |
| `k_combine` | 4.19 | 15.5 % |
| `hc_pre` total | **27.08** | sum of parts = 27.07 |

Sum of parts equals the whole, so the launches do not overlap — a serial chain. Against the 2.07 µs
launch floor: **6.2 µs is launch overhead, 14.5 µs is inside `hc_sinkhorn`**, which is a
doubly-stochastic normalisation of a **4x4** matrix over **20 dependent iterations** on one warp of
one SM. Its only real speed-up is replacing the IEEE divide with a reciprocal multiply, which is not
bit-exact and would break the LOSSLESS gate. And because the launch floor is *flat in grid size*
(2.07 µs at both `<<<1,32>>>` and `<<<24,256>>>`), no amount of shrinking these grids helps.

The measurement that settles it is that **the glue is near-flat in K** — five of its eight marks
move 0.95-1.01x across a 5x change in work, because it is paid per *step*, not per *token*:

| | K=1 | K=5 | ratio |
|---|---|---|---|
| glue total | 8.50 ms | 10.37 ms | 1.22x |
| step total | 70.03 ms | 127.18 ms | 1.82x |
| **glue as % of step** | **12.1 %** | **8.2 %** | |

At the shipping tau = 3.736 the 8.50 ms becomes **2.78 ms per emitted token — speculation has
already removed 67 % of this region** without anyone aiming at it. The one bit-exact lever left is
fusing `hc_pre`'s three launches into one cooperative kernel: 2 x 2.07 µs x 86 calls =
**~356 µs/step = 0.5 %**. Filed as untried; at 0.5 % it is below what a single mark can resolve.

## Speculative decode: bounded by acceptance, not by bandwidth

The spec cycle is `verify_ms(K) = 69.9 + 17.11 K` (F122). Two things follow directly and neither is
a bandwidth statement:

- The **intercept is 80 % of a K=1 step**. Speculation does not make the model cheaper; it amortises
  a step over tau tokens. The whole gain is acceptance.
- At block 6 the **perfect-acceptance ceiling is 40.6 tok/s**. Shipping is **24.73 tok/s** at
  tau = 3.736 on the frozen suite. The gap is entirely draft-head quality — i.e. task #10, a
  fine-tune — and not anything in the kernels.

This is why the MoE result matters more than its 4 % suggests: the MoE term appears in **both** the
intercept and the slope of `verify_ms`, so if it had really been sitting at 67 % of achievable it
would have lifted every speculative ceiling at once. It is not, so it does not.

Two structural things were also tried and are recorded as closed rather than pending: **typical
acceptance** (Medusa-style) is a win on a linear draft (+11 % tok/s at T=0.3), and **DDTree** is
correct but does not beat linear here because the regime is depth-dominated.

## What is actually left

Ordered by measured size, not by appeal:

1. **Draft-head acceptance** (task #10). 24.73 -> 40.6 tok/s is the only remaining lever worth more
   than 10 %, and it is a training problem, not a kernel problem.
2. **Attention at the MoE's rate**: 172 -> 195 GB/s would be ~5.5 ms of the 70.03 ms step. Already
   attacked twice (F125, F126); what remains is not identified.
3. **`hc_pre` cooperative fusion**: +0.5 %, bit-exact, untried, below single-mark resolution.
4. **MoE K=5 row loop**: +4 % ceiling, unreachable in practice; discounted by history to ~+2 %.

And what is closed: MoE GEMV efficiency (#13, at the roofline), m16 B-repack (#14, crossover R=5.05),
the glue region (F138), typical acceptance (won, shipped), DDTree (correct, not faster).

## Open, and recorded as open

`hc_pre (ffn)` costs **64.4 µs/layer** against `hc_pre (attn)`'s **37.0** at K=1 — same function,
same shapes, same 1.57 MB weight, **1.74x** — converging to 1.23x by K=5, while the standalone
kernel is 27.08 µs, *below both*. The likeliest explanation is that one mark charges for a
neighbour's tail across the `g_side` fork, which is the same mis-attribution class that produced the
155 GB/s error. It is worth at most the 0.5 % above even if fully explained, so it is a
correctness-of-marks question rather than a lever.
