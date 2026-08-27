# The 4096 context ceiling is not the KV cache — MLA/DSA is doing its job, the engine is not

The obvious reading of "this server runs at seqmax=4096" is that a 43-layer model has an expensive
KV cache. For this architecture that reading is wrong, and it is wrong by a factor of 30.

DeepSeek-V4-Flash uses MLA (a single 512-wide latent KV per layer, `N_KV_HEADS = 1`) plus DSA
(compressed KV at ratio 4 or 128 per layer, and a 128-wide indexer on the ratio-4 layers). That is
supposed to make long context cheap in memory. **It does.** Counting every allocation in
`src/engine.cu` that is a function of `seqmax`, with the model's own constants
(`include/deepseek_v4.h`: `DIM = 4096`, `HEAD_DIM = 512`, `INDEX_HEAD_DIM = 128`, `HC_MULT = 4`,
43 layers, `compress_ratio(L) = 0` for L<2, else 4 on even and 128 on odd L):

| buffer | B/token | KiB/token | share |
|---|---:|---:|---:|
| **arena** — `(512 + 2 x seqmax) MiB` heuristic | 2,097,152 | 2048.0 | **67.9 %** |
| **`xin`** — attention-input history, 41 layers x `DIM` fp32 | 671,744 | 656.0 | **21.7 %** |
| `hbuf` + `hbuf2` — prefill HC activations, `hc x d` fp32 | 131,072 | 128.0 | 4.2 % |
| `win_kv` — **MLA latent**, all 43 layers x `HEAD_DIM` fp32 | 88,064 | 86.0 | 2.9 % |
| `mh_pre` — DSpark taps, `3 x d` fp32 | 49,152 | 48.0 | 1.6 % |
| `h0` + `main_x` | 32,768 | 32.0 | 1.1 % |
| `comp_kv` — **DSA compressed KV**, 41 layers | 11,072 | 10.8 | 0.4 % |
| `mkv[3]` — DSpark stage KV | 6,144 | 6.0 | 0.2 % |
| `idx_ckv` — **DSA indexer KV**, 21 layers | 2,688 | 2.6 | 0.1 % |
| **total that scales with `seqmax`** | **3,089,860** | **3017.4** | 100 % |

> **The real MLA + DSA KV cache is `win_kv` + `comp_kv` + `idx_ckv` = 101,824 B/token = 99.4 KiB
> per token — 3.3 % of what actually scales with `seqmax`.** The other 96.7 % is scratch.

What that means in the only units that matter:

| seqmax | KV cache alone | as this engine builds it |
|---:|---:|---:|
| 4,096 | 0.39 GiB | 11.79 GiB |
| 8,192 | 0.78 GiB | 23.57 GiB |
| 32,768 | 3.11 GiB | 94.30 GiB |
| 131,072 | **12.43 GiB** | 377.18 GiB |

**128K of context costs 12.4 GiB of KV on this architecture, and this box has ~22 GiB free after the
weights.** The context ceiling is an engine artefact, not an architectural one.

## The three offenders, in order

### 1. The arena, 2 MiB/token — 68 % of the problem, and it is a guess

```c
size_t arena_bytes = (size_t)512 << 20;
if (seqmax > 512) arena_bytes = (size_t)(512 + (size_t)seqmax * 2) << 20;
```

There is no model constant in that line. It is a heuristic that reserves 2 MiB of scratch per token
of *context*, when what the arena actually has to cover is the largest *prefill batch* — the GEMM
and activation scratch for M = however many tokens are pushed through at once. It scales with the
wrong quantity. `dscratch.cu` already tracks `g_arena_hwm`, a high-water mark, so the real
requirement is measurable rather than guessable; nobody has read it out and sized against it.

### 2. `xin`, 656 KiB/token — over-allocated by 16x to 512x, and this one is provable

`xin` is `[seqmax, DIM]` fp32 per compressed layer (41 of them) — the attention-input history the
compressor pools. `include/block_decode.h` justifies keeping it because "overlap groups span the
prefill/decode boundary, so prefill x1 must be retained". True, but the span is bounded, and
`kernels/compressor.cu:500` states the bound exactly:

```c
if(overlap){ tok0 = (g>=1) ? (g-1)*ratio : 0; ntok = (g>=1) ? 2*ratio : ratio; ... }
else       { tok0 = g*ratio;                  ntok = ratio;                    ... }
const float* xg = x + (size_t)tok0*dim;
```

**A group emit never reads further back than `2 x ratio` positions.** For the ratio-4 overlap layers
that is **8 tokens**; for the ratio-128 strided layers it is **128**. Against `seqmax = 4096` the
allocation is 512x what a ratio-4 layer can touch and 32x what a ratio-128 layer can. A ring buffer
of `2 x ratio` rows turns a 656 KiB/token term into a fixed ~164 MiB for the whole model. The cost
is index remapping in the verify path, which addresses `xin` by absolute position.

### 3. Prefill activation buffers, 208 KiB/token — sized to context, needed for a chunk

`hbuf`, `hbuf2`, `h0`, `main_x`, `mh_pre` are all `[seqmax, ...]`. They hold *prefill* activations.
If prefill were chunked they would size to the chunk, not the context — which is the same
observation as the arena, and the same fix.

## What this is worth

Fixing only the arena (size it from `g_arena_hwm` instead of from `seqmax`) drops the per-token cost
from 3017 to 969 KiB — **seqmax ~12,800 in the same memory**. Adding the `xin` ring buffer drops it
to 313 KiB/token — **seqmax ~40,000**. Both are bounded, local changes to allocation, not to
arithmetic, so both are bit-exact by construction.

**Caveat, stated because it has not been checked:** `win_kv` is allocated `[seqmax, HEAD_DIM]` while
`WINDOW = 128`, and `LayerKV` already carries a `winmax` field for the graph path. If the sliding
window is genuinely bounded at 128 then `win_kv` is over-allocated too, and the *true* KV cost is
lower still — `comp_kv` + `idx_ckv` + a fixed window is ~13 KiB/token. That has **not** been
verified against the attention kernels and is not counted in any number above; the 99.4 KiB/token
figure charges `win_kv` at full `seqmax` and is therefore the conservative one.

## The second ceiling, which is not memory at all: 49,140 tokens of shared memory

**Added 2026-08-20 by DECODE_LADDER item 1.4.** Everything above this line is about *allocation* —
how many GiB a context costs. There is a second, entirely independent context ceiling in this engine
that costs nothing and fails differently, and it is lower than most of the seqmax figures the page
argues for.

The four warp selection-sort top-k kernels (`k_topk_offset`, `k_topk_decode`, `k_topk_verify`,
`k_topk_masked`) stage the whole indexer score row in **dynamic shared memory**: `~4T` bytes, where
`T` is the compressed-row count, `context / ratio`. The per-block default on this device is 49,152 B:

| limit | bytes | T | context at ratio 4 |
|---|---:|---:|---:|
| `cudaDevAttrMaxSharedMemoryPerBlock` (default) | 49,152 | 12,285 | **49,140** |
| `cudaDevAttrMaxSharedMemoryPerBlockOptin` | 232,448 | 58,109 | **232,436** |

Above 49,140 the launch fails, `cudaDeviceSynchronize()` returns **success**, and the output index
array is left untouched — see [`measurement-and-traps.md` §17](measurement-and-traps.md). So the
"seqmax ~40,000 in the same memory" this page ends on was, before 1.4, the largest context that
would have worked *by luck*: the arena fix and the `xin` ring buffer would have bought memory
headroom and then walked into a silent wrong answer 9,000 tokens later.

Two things about it are worth carrying:

* **It is not the shipped path.** Item 1.2 replaced all four with a single-CTA radix select that uses
  ~7 KiB of *static* shared memory whatever `T` is. The engine's default path has no context ceiling
  from shared memory at all. What retained the defect was the `DSV4_TOPK_RADIX=0` A/B arm and the
  `DSV4_TOPK_GATE=1` in-situ reference.
* **The fix is an opt-in, and it is 4.7×, not infinite.** `topk_scan_smem_optin` in
  `include/indexer.h` raises those arms to 232,436 tokens of context and *aborts* above it.
  Bit-exactness against the radix select is gated at T = 12,286 / 16,381 / 24,570 / 58,045 on six
  distributions, under CUDA-graph capture as well as direct launch — and, since 2026-08-20, through
  the engine's own `compressed_decode_step_indexer` at contexts 49,207 and 200,003
  (`scripts/gate_topk_smem_ctx.sh`).
* **There was a third one, on a path nothing links.** `sdpa` (`kernels/attention.cu`) sizes its
  dynamic request `(head_dim + seq) * 4`, so it had the same defect at seq 12,224 — lower than any
  of the seqmax figures this page argues for. It is not linked into `build/decode` or
  `build/dsv4-server`; only `tests/test_attention.cu` calls it, and that test's golden dirs are not
  in this tree. Enumerating every `<<<g, b, smem, s>>>` in `kernels/` and `src/` is what found it:
  thirteen launches request dynamic shared memory, seven of them scale with a runtime quantity, and
  the other six are bounded by model or block constants at ≤ 4 KiB. **Grep for the launch
  configuration, not for the buffer.**

The transferable form: **a context ceiling need not be a memory ceiling.** Before raising `seqmax`,
grep for launch configurations computed from a runtime quantity — dynamic shared memory, grid size,
block size — because those fail at a fixed number that no allocation table will tell you about.

## Where the memory ceiling actually is now, and why the 49,140 leg had to be run outside the engine

**Measured from the allocation code, 2026-08-20, and this supersedes the table at the top of the
page for the current engine.** Both offenders that table names have since been fixed: the arena is
sized by `MAXB`, not `seqmax` (`src/engine.cu:469`, `(512 + MAXB*2) MiB` = 640 MiB flat), and `xin`
is a `2*ratio`-row ring (`xin_ring_alloc_rows`). What still scales with `seqmax` in `src/engine.cu`
is:

| buffer | B/token |
|---|---:|
| `win_kv` — 43 layers × `HEAD_DIM` fp32 | 88,064 |
| `comp_kv` — 41 layers × `HEAD_DIM` fp32 / ratio 4 | 20,992 |
| `main_x` — `DIM` fp32 | 16,384 |
| `mkv[3]` — DSpark stage KV | 6,144 |
| `idx_ckv` — 21 layers × `INDEX_HEAD_DIM` fp32 / ratio 4 | 2,688 |
| `d_ids` | 4 |
| **total** | **134,276** = 131 KiB/token |

That is **2.05 GiB at the seqmax 16,384 the engine runs today and 6.15 GiB at seqmax 49,152** — 4.10
GiB more, against the 3.7 GiB free that `[engine] ready. mem 119.1/122.8 GiB` reports at seqmax
16,384. **So the engine cannot be run past the 49,140 shared-memory ceiling on this box, and the
blocker is the 100.4 GiB of weights, not the KV.** 131 KiB/token is 3017 KiB/token's successor and
it is 23× better; the remaining 96.7 %-is-scratch claim in the table above no longer holds, but
`win_kv` at full `seqmax` (66 % of what is left, against a `WINDOW` of 128) is the same
over-allocation the caveat at the end of that section already flags, and it is still unverified.

## Why this matters beyond memory

Every number in `EVALS.md` is capped by the 4096-token context, and DeepSeek recommends 384K output
tokens at the `high` reasoning effort these evals use. Truncated items are scored wrong, so the
context ceiling is depressing the published capability of the checkpoint by an unknown amount. **The
cheapest way to raise those scores is not a better prompt or a bigger sample — it is deleting a
heuristic in `src/engine.cu:388`.**

---

## Resolved: 128k ships, and the page above is stale (2026-08-26)

Everything this page argued for has been built, and by the time it was re-read only one of its three
offenders was still open.

| offender | state |
|---|---|
| arena `(512 + 2 x seqmax) MiB` | **fixed** — `(512 + MAXB x 2) MiB`, sized by batch; 640 MiB in practice |
| `xin` 656 KiB/token | **fixed** — ring of `2 x ratio` rows, fixed ~124 MiB |
| prefill activation buffers | **fixed** — `h0`/`hbuf` are batch-width |
| top-k 49,140-token smem ceiling | **fixed** — shipped radix path requests zero dynamic smem |

What remained was the FP32 KV rows, and the fix for that already existed too (`DSV4_KV_PACK`,
720 B/row vs 2048) and was simply not being used. Measured on the server, weights 100.4 GiB of 122.8:

| seqmax | fp32 KV | packed KV |
|---:|---|---|
| 8,192 | ready, 119.6 / 122.8 | — |
| 32,768 | ready, 119.9 / 122.8 | — |
| **131,072** | **never reaches ready** | **ready, 117.0 / 122.8** |
| 262,144 | — | ready, 121.8 / 122.8 — **1.0 GiB headroom, below memguard's 1500 MB floor** |

**Packing is bit-exact.** Token streams identical over 400 generated tokens, and the reason is
structural rather than lucky: the FP32 path already runs `act_quant_fp8sim` on these rows and stores
the *dequantised* float in four bytes. Packing stores the e4m3 code instead. Same values.

It costs **~12 % of prefill** (107.9 -> 94.8 tok/s) because the consumer redoes the conversion on
read, so it is now **auto-enabled above seqmax 32768** (`kv_pack_init_seqmax`) rather than being on
unconditionally: 32,768 is the largest context FP32 is measured to survive, so below the threshold
nothing is paid. `DSV4_KV_PACK=0/1` still forces either way. The default `SEQMAX` moves 8192 ->
32768, which is 4x the context for free.

**256k is NOT supported.** It allocates and it cannot serve. Getting there needs `win_kv` (7.6 GiB
at 256k packed) rebased off absolute positions, and this page's speculation that it might be a cheap
128-row ring is **wrong**: `kv_row(win_kv, pos)` indexes by absolute position and the full history is
copied into `kv_all` every step, so it needs reader-side index remapping exactly like the `xin` fix.

**Trap for the next person.** The first 128k probe was run against `build/decode`, which prefills the
whole prompt as ONE batch and therefore must size scratch for M = 131,322 -- it cannot reach long
context by construction, no matter what is allocated. `dsv4-server` chunks prefill at `EXT_CHUNK`.
That is trap 45 (`size the probe against the binary you are launching`) recurring within one day.
