# deepseek-v4-flash-0731-cuda

A from-scratch pure-CUDA inference server for **`0xSero/DeepSeek-V4-Flash-0731-REAP`** (K160,
native MXFP4) with embedded **DSpark** self-speculative decoding, MLA + DSA attention, hand-tuned
for **Jetson AGX Thor, `sm_110a`**.

No Python on the hot path. Every kernel gated against a PyTorch oracle before it is trusted.

---

## Read these first

| file | what it holds |
|---|---|
| **`ROOFLINE.md`** | the arithmetic that governs the project — `B_tok`, the AR wall, the anchor, `E_frac(k)`, the corrected priority order |
| **`ARCH_DELTA.md`** | what ports from prior work, what is genuinely new, and the re-priced risk table |
| **`MODEL_INVENTORY.md`** | checkpoint identity + every architectural constant, each traceable to a file |
| **`HARDWARE.md`** | the box, the memory constraint, and the `sm_110a` facts already settled empirically |
| **`STATUS.md`** | gate ledger and what is in flight |

## The one-paragraph situation

The 0731-REAP checkpoint's 43-layer backbone is **geometrically identical** to the
`0xSero/DeepSeek-V4-Flash-180B` checkpoint that `~/dspark-cuda-reap-finetune` already runs
correctly on this box — same `B_tok` to the megabyte, same MXFP4 expert layout, same MLA/DSA/HC
structure, 47 of 48 config keys identical. That prior repo is a 7,870-line engine with MLA, the
KV compressor, the DSA indexer, hyper-connections + Sinkhorn, MXFP4 grouped MoE, FP8 block GEMM
and a bit-exact full-model CUDA-graph decode, all gated. **So this is a checkpoint migration plus
a decode-efficiency grind, not a greenfield kernel build.** What is genuinely new: the 3-stage
DSpark head is now *embedded and REAP-pruned in-family* (the prior project had to bolt on an
unpruned 256-expert head), the chat encoding has no Jinja template, and the server layer.

## The numbers that set expectations

```
weights            100.400 GiB  of a 117 GiB unified pool  -> 16.6 GiB headroom
B_tok               11.202 GB/token
achievable BW          240 GB/s   measured (tools/bw_probe.cu), not the ~200 inherited
AR wall              21.42 tok/s  @ 240 GB/s   (24.37 @ 273 spec)
measured today        7.89 tok/s  = 37% of achievable   <- direct, same B_tok, same box
base AR band         15-19 tok/s  after kernel work (70-80% of achievable)
DSpark band          22-36 tok/s  centred ~28, k* = 2-3 (NOT 7)
```

The largest per-token consumer is **MLA attention at 41.1%** — larger than all six routed
experts combined. The compressed KV cache is tiny; the projections that produce and consume the
latent are not.

## Reproducing the analysis without the weights

The full checkpoint is 100.4 GiB. The analysis needs only ~9 MB of safetensors headers, which are
committed under `docs/`:

```bash
python3 tools/fetch_headers.py     # harvest headers + metadata via HTTP range requests
python3 tools/inventory.py         # inventory, quant-format proof, B_tok, AR wall, KV model
python3 tools/verify_cost.py       # E_frac(k), c_v(k), the S(k) break-even table

nvcc -O3 -arch=sm_110a tools/bw_probe.cu -o build/bw_probe && ./build/bw_probe 4096 30
```

`tools/inventory.py` exits non-zero unless the summed tensor bytes reconcile exactly with
`index.json`'s `total_size`. Add `--model-dir DIR` to run against a local checkout instead
(pointing it at `~/models/DeepSeek-V4-Flash-180B` reproduces the identical-`B_tok` finding).

## Hard operating rules

- **Memory-neutral only.** The pool is unified and shared with the OS. A +5.5 GiB cache on the
  prior project hard-hung the box and forced a physical power-cycle.
- **Run full-model binaries detached**: `setsid nohup ./build/<bin> … > ~/run.log 2>&1 < /dev/null &`.
- **Single-tenant**: one process may hold the full model at a time.
- **Never loosen a gate.** Correctness gates precede speed gates, always.
- **Do not trust `ncu`'s "Memory Throughput %" on Thor** — it is L2 throughput, not bandwidth
  utilisation. Use the byte model ÷ wall-clock. See `HARDWARE.md` §3.
- **No invented model constants** — read from disk, cite the file.
- **No REAP pruning, no additional quantisation.** The checkpoint ships as it ships.

## Related repos (read-only)

- `~/dspark-cuda-reap-finetune` — the real baseline; MLA/DSA/DSpark kernels, gate log, decode-gap research
- `~/gemma-cuda-hybrid` — the CUDA constitution, optimisation methodology, and the server abstractions to inherit
- `~/laguna-s1-cuda-server` — the other MoE precedent; roofline and gate discipline

License of the target model: MIT (base and REAP derivative).
