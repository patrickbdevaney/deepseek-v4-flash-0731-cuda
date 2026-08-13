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

## Why this matters beyond memory

Every number in `EVALS.md` is capped by the 4096-token context, and DeepSeek recommends 384K output
tokens at the `high` reasoning effort these evals use. Truncated items are scored wrong, so the
context ceiling is depressing the published capability of the checkpoint by an unknown amount. **The
cheapest way to raise those scores is not a better prompt or a bigger sample — it is deleting a
heuristic in `src/engine.cu:388`.**
