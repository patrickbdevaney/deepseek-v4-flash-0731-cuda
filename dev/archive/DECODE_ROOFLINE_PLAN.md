# DECODE_ROOFLINE_PLAN.md — closing the gap between 77 % of roofline and what production sees

Referenced from `LEVERS.md`'s reopened-lever banner. `PERF.md` is the evidence; this is the plan.

---

## 0. The target, stated correctly — because the obvious phrasing is wrong

"Get compute and memory both to the roofline, nothing idle" is not achievable and should not be
attempted. At batch 1 every weight is read from DRAM, used for **one** multiply-accumulate against a
single activation vector, and discarded. Arithmetic intensity is a few FLOP/byte against a machine
balance in the hundreds. Measured live during the battery: `utilization.gpu` **96 %**,
`power.draw` **21.5 W**, ~0.4 TFLOP/s of arithmetic, ~61 GB/s of DRAM traffic.

**The idle SMs are correct.** A memory-bound kernel at the roofline still has idle compute; that is
what memory-bound means. Chasing SM utilisation here would optimise a number that should not move.

> **The target is: the memory system never starves, at working depth.**
> Success is DRAM bytes/second during a forward, not occupancy, not FLOPs, not watts.

The only ways to raise intensity are to change the workload — a wider speculative block, or batching
concurrent requests — and both are different projects with their own trade-offs (§5).

---

## 1. What is already closed — do not re-chase it

`LEVERS.md` §9's corrected base-AR budget, **at prompt 0**, K=1, 70.03 ms/step:

| region | ms | % | GB/s | % roofline |
|---|---:|---:|---:|---:|
| MoE (routed + shared, concurrent) | 23.20 | 33 % | 195 | **94 %** |
| Attention (MLA + compressor + indexer) | 31.40 | 45 % | 172 | **82 %** |
| `lm_head` | 5.80 | 8 % | 183 | **88 %** |
| HC / rmsnorm / router / glue | 8.53 | 12 % | 25 | latency-bound |
| **whole step** | **70.03** | | **160** | **77 %** |

**77 % of roofline is inside the 70–80 % band `ROOFLINE.md` calls well-written.** F125/F126/F137
closed the MoE twice. There is no large weight-streaming lever left at prompt 0, and any plan that
proposes one is re-deriving a closed result.

> **Correction to `PERF.md`'s lever table.** It reports "forward pays 137 ms for 11202 MB = 34 % of
> achievable BW" and prices 34 % → 75 % as a 2.20x lever. That comparison divides an **M=1**
> quantity (`B_tok`) by a **verify-forward** time, and a verify forward does several positions'
> work — wider expert union, plus the draft head. It was labelled a lower bound when written, but it
> should not be read as 2.2x of available headroom: measured properly at K=1 the engine is at 77 %,
> not 34 %. The weight-stream lever is small. **The depth term is the prize, and it is the whole
> prize.**

---

## 2. What is open — and it is one thing

`PERF.md`, fitted over 891 real requests spanning 151–6568 tokens of depth:

```
ms_per_verify = 136.8 ms  +  0.0307 ms x kv_mid
```

| KV depth | depth term | share of forward |
|---:|---:|---:|
| 0 | 0 ms | 0 % |
| 2 048 | 63 ms | 31 % |
| 8 192 | 251 ms | **65 %** |
| 16 384 | 503 ms | **79 %** |

So the engine that runs at **77 % of roofline at prompt 0** is running at roughly **25 %** at 8k of
resident context. Nothing about the weights changed. The entire collapse is depth.

And it is **not bytes**: at 240 GB/s the fitted slope would mean moving **7.36 MB per resident KV
token per forward**, against ~3.3 KB of actual per-position state, with `index_topk` = 512 bounding
what attention may read at any depth. A depth-linear cost that bytes cannot explain is compute-,
occupancy- or launch-bound — which is why it is recoverable.

---

## 3. Phase 1 — recompute the §1 budget table at depth. No new code.

This is the whole investigation in one experiment, and every instrument it needs already exists.

`DSV4_BLKSWEEP` entries are `BLK[:passes[:adaptK[:prompt]]]`, where `prompt` is an **index into
`DSV4_PROMPTS_FILE`** (one token-id list per line). Each sweep point re-prefills, so points are
independent — **one checkpoint load answers all four depths**, instead of four 15-minute loads.

1. Build `prompts_depth.txt` with four id lists at ~0, 2k, 8k, 16k tokens. Use
   `tools/encode_prompt.py`; ids must come from the checkpoint's own tokenizer, never invented.
2. Run once:
   ```
   DSV4_DPROF=1 DSV4_KSWEEP=1 DSV4_PROMPTS_FILE=prompts_depth.txt \
   DSV4_BLKSWEEP="6:1:1.5:0,6:1:1.5:1,6:1:1.5:2,6:1:1.5:3" ...
   ```
3. Pair the points with `tools/dprof_diff.sh`, which already diffs two DPROF logs by
   `(K, sub-op)`.

**The sub-op whose ms grows with the prompt index IS the depth term.** That is the answer, and it
costs one afternoon and no code.

Expected shape if the ranked hypothesis is right: the growth sits in the **Attention** row — the DSA
indexer must SCORE all D resident positions per layer per forward to select its top-512, so it is
linear in depth by construction while the attention it feeds is bounded at 512. If the growth is
somewhere else, the hypothesis is dead and the profile says where to look instead.

Watch the **glue** row too. It is already 12 % of the step at 25 GB/s and explicitly latency-bound;
if any of it touches the cache it will grow with depth as well, and it is the second-largest
non-roofline region in the engine.

---

## 4. Phase 2 — name the mechanism inside the winning sub-op

Only after Phase 1 names a region. Profiling everything first is how a cycle is wasted.

> ### ⚠ `ncu`'s "Memory Throughput %" is meaningless on this box
> `HARDWARE.md` §2 records the trap: profiling `stream_read`, independently measured at **244 GB/s
> (~89 % of spec peak)**, `ncu` reports **Memory Throughput 30.26 %** and then advises that
> "bandwidth below 60 % of peak typically indicates..." — advice that is exactly backwards here.
> **Use `dram__bytes.sum / gpu__time_duration`,** never the percentage.

Hypotheses to separate, in the order the evidence ranks them:

| # | hypothesis | what would confirm it | what would kill it |
|---|---|---|---|
| 1 | DSA top-512 selection is linear in D and poorly parallelised (full sort, or a single-block reduction over D) | indexer kernel duration grows linearly with D at low `dram__bytes` | duration flat, or bytes grow with D |
| 2 | KV layout: `compress_ratios` puts **21 layers at ratio 4** (the long caches) vs 20 at ratio 128 — a layout that strides badly costs most on exactly those layers | per-layer duration splits by ratio class | ratio classes cost the same |
| 3 | Launch count scales with depth | launches/forward grows with D | launch count flat (ROOFLINE §3 already reports a full 43-layer graph captured at parity, so this is unlikely) |
| 4 | Individual kernels cannot saturate DRAM alone and need a concurrent partner | `tools/overlap_probe.cu` — it exists for exactly this question | kernels saturate solo |

---

## 5. Success criterion, and the honest ceiling

**Criterion:** effective DRAM bandwidth during a forward **at 8k resident depth**, currently ~25 % of
240 GB/s. Target the same 70–80 % the engine already achieves at prompt 0.

If the depth term were removed entirely, the 8k forward goes 388 ms → 137 ms, about **2.8x**; at 16k,
639 ms → 137 ms, about **4.7x**. Those are ceilings on this lever alone, and they assume the term is
fully recoverable, which is exactly what Phase 1 and 2 are for.

**What this lever cannot do:** it cannot make batch-1 decode compute-bound, and it cannot raise the
prompt-0 number, which is already at 77 %. The AR wall at 240 GB/s stays **21.42 tok/s**
(`ROOFLINE.md` §2). Speculation is the only route past that, and it is priced separately.

**Raising arithmetic intensity is a different project.** A wider speculative block puts more
positions through one forward but grows the expert union it must read (the interaction the new
`spec_profile` telemetry is meant to price). Batching concurrent requests raises intensity properly,
but this server is single-lock by design and the eval programme depends on that. Neither belongs in
this lever.

---

## 6. Discipline — the traps this plan is standing in

- **Trap 25.** A single cross-run timing claim under ~1.5 % is unsupportable; a sub-1 % effect needs
  **two** matched pairs. The depth term clears this by more than an order of magnitude, but any
  *fix* for it will be measured against the same floor. Prefer **counted integers** where possible.
- **LOSSLESS GATE.** Any change here must leave output bit-identical. A faster kernel that changes
  a token has not optimised anything.
- **Do not re-baseline from a profiling run.** `DSV4_DPROF` costs ~0.4 %, `DSV4_SPECPROF` ~1.0 %
  (trap 32).
- **One restart window.** The engine is a ~10–15 min reload of 101 GiB and taking it down
  mid-programme is this repo's most expensive documented mistake. Phase 1 needs the engine alone —
  schedule it with the staged-binary deploy and the SUFFIXPROBE slice
  (`EVAL_NEXT_STEPS.md` §5), not separately.
