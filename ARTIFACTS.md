# ARTIFACTS — where the draft heads live, and which one to upload

The engine is in git and therefore safe. **The weights are not** — they are the one product git
cannot hold. This file says exactly where they are.

## The one to upload

    /home/patrickd/model-backups/releases/dspark-mtp-draft-head-v1.0-s3/

**1035 MB, 8 files, checksum-verified** (`sha256sum -c SHA256SUMS` passes). This is the **`s3`
head — the promoted, shipped speculator**, packaged for HuggingFace.

| file | what it is |
|---|---|
| `mtp_trained.safetensors` | **the artifact** — 1035 MB of trained MTP draft-head tensors |
| `README.md` | HuggingFace model card, YAML frontmatter declares `base_model` |
| `provenance.json` | base model, engine git rev, measurement, rebuild recipe |
| `SHA256SUMS` | checksum per shipped file |
| `eval.log` | the raw measurement behind the numbers |
| `train_metrics.json` | per-step loss history, all 1472 steps |
| `head_card.json` | the archive record written at promotion |
| `config.json` | the checkpoint's architecture config |

**It ships the ~1 GB of trained tensors, not the 7 GB materialised head, deliberately.** Every build
log reports `copied 2905 untouched (experts byte-for-byte)` — ~6 GB of any head is the base
checkpoint's own weights copied verbatim, and redistributing those inside an artifact that adds
nothing to them is the wrong shape. `tools/build_trained_head.py` rebuilds the loadable head
deterministically in about a minute and refuses any tensor whose fp8 round-trip exceeds 0.10.

**Before uploading**, read `README.md`. It names `0xSero/DeepSeek-V4-Flash-0731-REAP` as the base
model and quotes measured numbers. The numbers are checked against the logs; whether that
attribution and the base model's license terms are presented the way you want is a human call, and
`--repo-slug` still says `<your-org>`.

## Every head ever built

`~/model-backups/heads/<name>/` — each with sha256 per file, its `eval.log`, `train_metrics.json`
and a `head_card.json` recording the engine revision it was measured at. Archiving is unconditional
and runs **before** the promotion gate, because promotion is a judgement and archiving is
preservation.

| archive | suite tok/s | status |
|---|---|---|
| `~/model-backups/heads/s3` | **25.53** | **PROMOTED — shipped** |
| `~/model-backups/heads/s2-abl-ce1.0_tv0.0` | 25.10 | reject (inside the 3.5 % band) |
| `~/model-backups/heads/s2-abl-ce0.9_tv0.1` | 24.90 | reject |
| `~/model-backups/heads/s2-abl-ce0.5_tv0.5` | 24.82 | reject |
| `~/model-backups/heads/s2` | 24.76 | reject |
| `~/model-backups/heads/s1` | 24.52 | superseded by s3 |
| (stock, in the checkpoint) | 22.66 | baseline |

`HEAD_REGISTRY.md` is the promotion record, rejects included. `RUNS.md` indexes every run and links
its evidence log.

## The working trees (not backups — these get deleted)

    /home/patrickd/s5-capture/s3/head/            materialised 7 GB head, rebuildable
    /home/patrickd/s5-capture/s3/c{0,1,2}/trained/ per-chunk weights + AdamW state
    /home/patrickd/s5-capture/s3/gen.txt          the 1536 pass-1 generations

Safe to delete once the archive is verified. The archive is the record; these are scratch.
