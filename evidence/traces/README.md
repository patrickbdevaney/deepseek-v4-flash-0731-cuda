# evidence/traces — generation traces from the eval battery

Rebuilt by `tools/trace_export.py`. **Gitignored, and that is deliberate.**

## What is here

One shard per benchmark per format. Each shard's first line is a header carrying the canary, the
redistribution policy, the pinned dataset snapshot and the code commit; every subsequent line is
one trace.

| format | shape | for |
|---|---|---|
| `text` | `{text, metadata}` — the exact served string, `<｜User｜>…<｜Assistant｜><think>…</think>…` | S5 tap capture, replay fixtures |
| `sft` | `{messages:[user, assistant], metadata}` with `reasoning_content` split out | supervised fine-tuning |
| `s5-prompts` | `{prompt, metadata}` | a capture corpus in `make_corpus.py` shape |
| `raw` | prompt + reasoning + content + usage + timings | anything else |

## Prompts are proven, not stored

The benchmark records hold `prompt_sha256`, not prompt text — a prompt is a pure function of
(pinned dataset snapshot, harness code). The exporter rebuilds each prompt through `eval_suite`'s
own task builder and **refuses to emit any trace whose rebuilt prompt does not match the recorded
hash**. Silently pairing a generation with the wrong question would corrupt every downstream use
and would be undetectable afterwards.

Each trace is labelled with how strongly its prompt is established:

- `per-record-sha256` — the run recorded a hash for this exact item and the rebuild matches.
- `set-level:<commit>` — the run predates per-record hashing; the whole prompt set is proven
  byte-identical between the run-start commit and HEAD (`evidence/evals/prompt_provenance.json`).
  Weaker, and reported as weaker rather than rounded up.

## Do not publish this

These prompts are **benchmark items**. Publishing them in plain text is how a benchmark dies: any
model later trained on the text is contaminated, and every score on that benchmark — including the
ones this repo publishes — stops meaning anything.

| task | policy | why |
|---|---|---|
| `gpqa_diamond` | **NEVER-PUBLISH** | ships a canary, gated on HF, authors ask it not be posted in the clear |
| `lcb` | **NEVER-PUBLISH** | LiveCodeBench is a contamination-controlled time window; plain text defeats it |
| everything else | CHECK-BEFORE-SHARING | still benchmark text; check each dataset card first |

`--format sft` refuses NEVER-PUBLISH tasks unless `--allow-redistribute` is passed. That flag does
not make anything publishable — it only stops the tool refusing. The files stay local either way.

Every shard carries the BIG-bench/GPQA canary GUID, so a corpus that ingests one of these by
accident is detectable after the fact. That is the only mitigation that survives a file escaping
this directory.

**The highest-value use of this data needs no redistribution at all.** `S5_RECIPE.md`'s mandatory
first step is regenerating responses with the target model; the battery has already produced 1.29M
such tokens, weighted toward the workloads where `PERF.md` measures the draft head weakest. That
consumer is entirely internal.
