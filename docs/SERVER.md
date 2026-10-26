# SERVER.md — Phase 6, the production server surface

The decode work is finished and banked (24.73 tok/s, `s3` head, FP8, adaptK 1.5). This phase turns
that engine into something an external agentic harness can call the way it would call llama.cpp or
vLLM: an OpenAI-compatible HTTP server, streaming, tools, thinking blocks, prefix caching, a web UI
and a terminal client.

**No Python on the request path.** One binary. `httplib.h` and `json.hpp` are header-only and
vendored; the chat encoder, the tokenizer, the streaming parser and the model are all C++/CUDA in
this repo. Python appears only in offline codegen (`tools/gen_unicode_tables.py`) and in the
reference-vector generators the gates compare against — never between a request and a token.

```
  bash scripts/build_server.sh      # gates + server + client
  bash scripts/serve.sh             # preflight the CPU gates, then listen on :8080
  build/dsv4-chat                   # terminal client
  open http://localhost:8080/       # web UI
```

---

## 1. What is here

| piece | file | gated by |
|---|---|---|
| Tokenizer (ByteLevel BPE) | `include/tokenizer_dsv4.h` | `tests/gate_tokenizer.cpp` |
| Chat encoder / DSML tools | `include/encoding_dsv4.h` | `tests/gate_encoding.cpp` |
| OpenAI request/response shaping | `include/openai_api.h` | `tests/gate_api.cpp` |
| Streaming split (think / content / tools) | `include/stream_parse.h` | `tests/gate_stream.cpp` |
| Persistent engine + prefix cache | `include/dsv4_engine.h`, `src/engine.cu` | `tests/gate_engine.cu` |
| HTTP server | `server/server.cpp` | the four CPU gates + `gate_engine` |
| Web UI | `include/webui_dsv4.h` | — |
| Terminal client | `server/chat.cpp` | — |

`include/tokenizer.h` and `include/webui.h` are gemma's, carried over with the rest of the port.
They are **not used** by this server and are kept only so the gemma reference stays readable
alongside; the DeepSeek versions are the `_dsv4` files.

---

## 2. The tokenizer was the critical path, and it is not gemma's

gemma-4 is SentencePiece-style: `' '` → `▁`, `byte_fallback` on for unknown characters, an
effectively no-op pre-tokenizer. DeepSeek-V4 is **ByteLevel BPE** (GPT-2 lineage): 128 000 vocab
plus 1 283 added tokens, `byte_fallback` off because it cannot be needed, and a four-stage
pre-tokenizer that splits the text before BPE ever runs:

```
Split(\p{N}{1,3},                      Isolated)   digits in groups of <=3
Split([一-龥぀-ヿ]+,   Isolated)   CJK / kana
Split(<the big alternation>,           Isolated)   the GPT-style word/punct/space split
ByteLevel(add_prefix_space=false, use_regex=false) byte -> private unicode alphabet
```

Adapting gemma's tokenizer by habit would have produced ids that are wrong but plausible — the worst
failure mode available, because nothing errors and the model simply answers a different question.

Two implementation notes worth keeping:

- **`\p{L}/\p{M}/\p{N}/\p{P}/\p{S}` are baked, not parsed.** `std::regex` has no unicode general
  categories. `tools/gen_unicode_tables.py` walks every codepoint once, offline, and emits 1 529
  sorted `[lo,hi]` ranges into `include/unicode_cat.h` (26 KB) which the tokenizer binary-searches.
  Codegen, not a runtime dependency.
- **The third stage's alternation is ordered, with real backtracking.** `\s+(?!\S)` needs a
  lookahead, `[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+` needs its optional first character to be tried
  greedily and then given back. Both are implemented literally in `stage3_match`.

### Gate result

`tests/gate_tokenizer.cpp` compares against HF `tokenizers` over a corpus chosen to break ByteLevel
BPE ports — digit-run boundaries, CJK adjacent to latin, whitespace before words vs at end of
string, CRLF and blank lines, combining marks, emoji and ZWJ sequences, added tokens at both ends
and adjacent, invalid-UTF-8 byte sequences, plus the repo's own prompt suite, the chat encoder's
four golden outputs, three source files and 400 GSM8K rows.

```
  encode:    2118/2118 cases exact (173 399 reference tokens)
  roundtrip: 2118/2118 cases lossless
  stream:    2118/2118 cases match one-shot decode
  speed:     4.1 MB/s (1.24 M tokens/s)
```

---

## 3. Streaming: thinking blocks and tool calls

Generation in thinking mode *starts inside* the reasoning block, because the prompt already ends
with `<think>`. The raw stream is therefore

```
{reasoning}</think>{content}<｜DSML｜tool_calls>…</｜DSML｜tool_calls>
```

`stream_parse.h` splits that live into `reasoning_content` deltas, `content` deltas and a buffered
tool block. Two hazards it exists to handle:

1. **Markers straddle chunks.** `"</thi"` + `"nk>"` must not reach the client as content. The
   splitter holds back the longest possible marker prefix and re-examines it on the next feed.
2. **Held-back boundaries land inside UTF-8 characters.** Every delta goes into a JSON string and
   nlohmann *throws* on invalid UTF-8 — which would turn a Chinese answer into a mid-stream 500. So
   emit boundaries are rounded down to a character boundary.

Tool calls are buffered and emitted in one final delta rather than streamed piecewise: each DSML
parameter's `string="true|false"` attribute is what decides its JSON type (CHAT_FORMAT.md §3), so
the block is not interpretable until it is complete. That is also what vLLM and SGLang do.

`tests/gate_stream.cpp` asserts the split is **independent of chunking**: the same completion fed
one byte at a time, in 22-byte chunks (exactly the marker length), and all at once must produce
identical deltas. 11/11 pass.

> A bug worth recording: the markers were first written as `"<\xEF\xBD\x9CDSML…"`. C hex escapes do
> not stop at two digits — `\x9CD` is one escape and overflows. Written as literal UTF-8 instead,
> which is what `encoding_dsv4.h` already did.

---

## 4. The engine, and why it is not `decode.cu`

`src/decode.cu` is the measurement harness every LOOP_LOG finding rests on. Its sweeps, probes and
env knobs *are* the instrument, and refactoring an instrument invalidates the readings taken with
it. So `src/engine.cu` is a **new translation unit over the same kernels**, carrying only the
shipping configuration — block 6, adaptK 1.5, VKPLUS on, the embedded `mtp.0/1/2` head — and none of
the sweep machinery. `tests/gate_engine.cu` is what holds the two to each other.

Shared loading helpers live in `include/dsv4_load.h`. `decode.cu` still has its own private copies:
de-duplicating it now would mean re-gating the frozen suite at ~10 minutes per point to prove a
copy-paste was faithful. The engine gate proves the property that actually matters instead.

### Sampling and speculation are compatible, exactly

The verify accepts `draft[i]` iff it equals the token the **target** produces at that position.
Replacing "argmax of the target logits" with "a sample from the target logits" preserves that: every
emitted token is still drawn from the target's own conditional, so the output distribution is the
model's. Speculation only decides how many positions are evaluated per forward, never which token is
emitted.

What it does cost is **acceptance** — matching a random draw is strictly harder than matching an
argmax — so tok/s falls as temperature rises. `temperature: 0` is the greedy path and is
bit-identical to the measured decode path.

---

## 5. Prefix caching

The engine keeps one KV context across requests. On a new request it finds the longest common
prefix with the resident context, rewinds to it, and computes only the remainder.

- **Rewind** is cheap and exact: sliding-window and `xin` caches are position-indexed, so they need
  nothing; only compressed layers carry a row *count*, and the number of compressed rows for
  positions `[0..n)` is `floor(n/ratio)` — the same accounting the accept path already does when it
  discards rows written by rejected drafts.
- **Extension** uses the M=K verify kernel at an offset, in chunks. That is the same
  batched-forward-at-a-position the accept path runs every round, so a cached turn and a fresh one
  go through identical arithmetic, only starting later.

`reasoning_effort` sits at position 0 of the prompt (CHAT_FORMAT.md §2.2), so changing it between
requests invalidates the entire cache while changing `thinking_mode` only perturbs the tail. The
server reports `usage.prompt_tokens_details.cached_tokens` so this is visible rather than folklore.

### What it is worth, and what it costs — measured

**Worth:** on a 40-token second turn sharing 32, prefill **712 ms → 144 ms, 4.9×**.

**Costs:** the honest answer needed the right instrument. Comparing generated *tokens* between a
cached and an uncached route can only say "they diverged at step k", which is equally consistent
with a broken KV and with a near-tie. So `gate_engine` compares the model **state** — the target's
logits for the next position — reached different ways:

```
cold reference: argmax 9544, top1-top2 margin 0.2122
shared K=9..20 (K%4 = 0,1,2,3 all covered): argmax 9544 for ALL TWELVE, top5 kept 4-5/5
```

Every shared-prefix length reproduces the cold decision, **including the ones that resume in the
middle of a compression group** — the case production actually hits, since a conversation shares
whatever length it shares.

**Token-identity, however, is not a property these kernels have, and the control proves it without
involving the cache at all.** Re-running the *same cached state* with only the extension batch shape
changed (`ext_chunk` 64 vs 7) moves the head of the logits by **1.40** and drops one of the top 5 —
against a decision margin of 0.21. Far too large for reassociation of a sum; the likely mechanism is
MoE expert *selection* flipping on near-tied routing scores, which swaps whole expert outputs rather
than perturbing one. Cold-vs-cached worst case was **2.47, 1.8× that floor**.

So cold and cached generate 20 identical tokens and then split at a near-tie. Requiring
"cached tokens == cold tokens" would demand something the model does not deliver with no cache in
play — the same thing `decode.cu` records when it attributes M=K-verify vs K-sequential-decode
differences to "MoE-atomic near-ties". The gate asserts the decidable property and *reports* the
divergence depth.

> **Open:** a 1.40 head-of-distribution swing from re-batching alone is larger than expected. The
> expert-selection-flip explanation fits every measurement here but has not been confirmed by
> instrumenting the router. Worth doing before anyone relies on bit-reproducibility.

---

## 6. API

`POST /v1/chat/completions` — streaming and non-streaming.

Beyond the OpenAI fields, and per CHAT_FORMAT.md, **two orthogonal controls**:

| field | values | default | note |
|---|---|---|---|
| `thinking_mode` | `chat`, `thinking` | `thinking` | also accepts `enable_thinking` and `chat_template_kwargs.thinking` |
| `reasoning_effort` | `low`, `high`, `max` | `low` | thinking mode only; realized as a text prefix at position 0 |

Sampling defaults follow DeepSeek's own recommendation and **differ from gemma's and Laguna's 0.7** —
temperature `1.0`, `top_p` `1.0`, or `0.95` when `tools` are present.

Responses carry `reasoning_content` as its own field (as vLLM and SGLang do), `tool_calls` parsed
back out of DSML with per-parameter types honoured, and a `timings` block with `tokens_per_second`,
`tokens_per_verify` and `prefill_ms`.

Also: `POST /v1/completions`, `GET /v1/models`, `GET /health`, `GET /metrics` (Prometheus text),
`GET /` (web UI).

### Concurrency is deliberately one-at-a-time

The engine owns a single KV cache, and on this box that is the right shape: a 122 GiB unified-memory
part running a 12.26 GB/token weight read has no headroom for a second context, and batching a
decode that is bandwidth-bound at M=1 buys nothing. Requests serialise; queue depth is reported in
`/metrics` rather than hidden.

---

## 7. Clients

- **Web UI** (`include/webui_dsv4.h`) — one self-contained page, no CDN. Collapsible thinking
  blocks, DSML tool calls rendered as structured calls, both thinking controls, live tok/s,
  tok/verify and cached-prompt-token counts.
- **Terminal client** (`server/chat.cpp`) — speaks the HTTP endpoint, so it exercises exactly what an
  external harness would. Streaming, slash commands (`/new /think /chat /effort /temp /sys /stats`),
  and `--once` for scripting.
