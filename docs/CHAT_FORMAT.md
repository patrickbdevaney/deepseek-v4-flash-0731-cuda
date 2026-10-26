# CHAT_FORMAT.md — the spec for Phase 6 (Gate S1)

Source of truth: `encoding/README.md` and `encoding/encoding_dsv4.py` in the checkpoint (verbatim
copies of `deepseek-ai/DeepSeek-V4-Flash-0731`'s), plus the four golden vectors in
`encoding/tests/test_input_{1..4}.json` → `test_output_{1..4}.txt`.

**There is no Jinja template.** The encoder is Python (`encode_messages` /
`parse_message_from_completion_text`); we port its *logic*, and gate the port against those four
golden vectors before wiring it to the server.

---

## 1. Special tokens

| token | purpose |
|---|---|
| `<｜begin▁of▁sentence｜>` | BOS, once at the very start |
| `<｜end▁of▁sentence｜>` | end of an assistant turn (EOS) |
| `<｜User｜>` / `<｜Assistant｜>` | turn prefixes |
| `<think>` / `</think>` | reasoning block delimiters |
| `<｜latest_reminder｜>` | date/locale reminder injection |
| `｜DSML｜` | tool-call markup namespace |

Roles: `system`, `user`, `assistant`, `tool`, `latest_reminder`, `developer`.
`developer` is internal to DeepSeek's search pipeline — **do not expose it**; the official API
rejects it.

## 2. Two orthogonal controls, not one

The directive (§9) says the model "uses `low`/`high`/`max` reasoning_effort, not a binary
thinking toggle". **It is both, and they are independent** — the server must expose both.

### 2.1 `thinking_mode` ∈ {`chat`, `thinking`} — binary

```
chat mode:      …<｜User｜>{msg}<｜Assistant｜></think>{response}<｜end▁of▁sentence｜>
thinking mode:  …<｜User｜>{msg}<｜Assistant｜><think>{reasoning}</think>{response}<｜end▁of▁sentence｜>
```
In chat mode `</think>` is emitted *immediately* after `<｜Assistant｜>`, closing the block so the
model goes straight to content. The prefill differs by exactly that token.

`drop_thinking` (default `true`): strips reasoning from assistant turns *before* the last user
message, keeping only the final turn's. **Automatically disabled when tools are present** on the
system or developer message — tool-calling conversations retain all reasoning.

### 2.2 `reasoning_effort` ∈ {`low`, `high`, `max`} — thinking mode only

Default is **`low`**. Critically, this is **realized purely as a text prefix prepended before the
system message** — no special token, no model-side mechanism, and the rest of the encoding is
byte-identical across levels:

| level | prefix |
|---|---|
| `low` (default) | **none at all** |
| `high` | `Reasoning Effort: Absolute maximum with no shortcuts permitted.` … |
| `max` | `Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.` … |

**`reasoning_effort` has no effect in chat mode** (there is no reasoning block to modulate).
Take the exact prefix strings from `encoding_dsv4.py:REASONING_EFFORT_PROMPTS` — do not retype
them; a paraphrase is a different prompt.

**Prefix-cache consequence.** Because the effort prefix sits at the very front of the prompt,
changing `reasoning_effort` between requests **invalidates the entire prefix cache**, while
changing `thinking_mode` only perturbs the tail. Worth knowing before the prefix-cache design
in Phase 6.

## 3. Tool calling — DSML, not JSON

Tools are declared on the `system`/`developer` message via an OpenAI-compatible `tools` field;
the encoder injects a `## Tools` block carrying `{tool_definitions_json}`. The model emits:

```xml
<｜DSML｜tool_calls>
<｜DSML｜invoke name="function_name">
<｜DSML｜parameter name="param" string="true">string_value</｜DSML｜parameter>
<｜DSML｜parameter name="count" string="false">5</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls><｜end▁of▁sentence｜>
```

- `string="true"` → the value is a **raw string**, passed through verbatim.
- `string="false"` → the value is **JSON** (number, boolean, array, object) and must be parsed.

That per-parameter `string` attribute is the whole ballgame for the parser: it is the only signal
distinguishing the literal string `"5"` from the number `5`. **Neither gemma's nor Laguna's
tool-call parser has this shape** — do not adapt one, write it against this spec.

Results go back as `<tool_result>` inside a **user** message:
```
<｜User｜><tool_result>{result_json}</tool_result><｜Assistant｜><think>…
```
Multiple results are ordered to match the `tool_calls` order in the preceding assistant message.

## 4. Sampling defaults

Per DeepSeek's own recommendation, and note this differs from Laguna's 0.7 — do not carry that
over by habit:

| | temperature | top_p |
|---|---|---|
| agentic / tool use | 1.0 | 0.95 |
| everything else | 1.0 | 1.0 |

## 5. Out of scope

`<｜action｜>`, `<｜title｜>`, `<｜query｜>`, `<｜authority｜>`, `<｜domain｜>`, `<｜extracted_url｜>`,
`<｜read_url｜>` drive DeepSeek's internal search-agent pipeline. Tokenize correctly, but do not
build server features around them.

## 6. Gate S1 acceptance

1. Port `encode_messages` / `parse_message_from_completion_text` to C++ (no Python on the hot path).
2. **Byte-exact** against all four `encoding/tests/test_input_N.json` → `test_output_N.txt` pairs.
3. Round-trip: parse → re-encode → identical prompt.
4. Multi-turn tool-calling session driven by a generic OpenAI-SDK agent loop, exercising all three
   `reasoning_effort` levels **and** both `thinking_mode` values.
