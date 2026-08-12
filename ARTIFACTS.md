# ARTIFACTS — where the draft heads live, and what to upload

The engine is in git and therefore safe. **The weights are not** — they are the one product git
cannot hold. This file says exactly where they are.

## THE DIRECTORY TO UPLOAD

    /home/patrickd/model-backups/releases/dspark-mtp-0731reap-bundle/

**1.6 GB, 9 files, `sha256sum -c SHA256SUMS` passes.** One self-contained bundle: the optimised head
as the default, the stock head preserved byte-exactly beside it, the rebuild tool, and a
HuggingFace-ready model card. Copy this one directory to Google Drive and on to HF; combined with the
engine repo on GitHub it is everything a user needs.

    dspark-mtp-0731reap-bundle/
      README.md                          HF model card (YAML frontmatter declares base_model)
      provenance.json                    base model, engine git rev, measurement, contents
      SHA256SUMS                         checksum for every file
      default-s3/
        mtp_trained.safetensors          THE ARTIFACT -- the promoted head, 25.53 tok/s
        head_card.json  eval.log  train_metrics.json
      base-stock/
        mtp_base.safetensors             the checkpoint's ORIGINAL MTP: 72 weights + 25 block
                                         scales, byte-exact, so the fine-tune is diffable
        eval.log                         the stock head's own measurement, 22.66 tok/s
      tools/
        build_trained_head.py            self-contained rebuild, not a pointer

**Both heads ship as ~1 GB of MTP tensors, not as 7 GB materialised heads.** Every build log reports
`copied 2905 untouched (experts byte-for-byte)` — ~6 GB of a head is the base checkpoint's own
weights copied verbatim, and redistributing those inside an artifact that adds nothing to them is the
wrong shape. `build_trained_head.py` reconstitutes the loadable head deterministically in ~1 min and
refuses any tensor whose fp8 round-trip exceeds 0.10 (worst observed for `s3`: **0.0014**).

**Reverting to stock does not need `base-stock/`** — pass no head argument and the engine uses the
checkpoint's own MTP blocks. That is how every paired control in this project was measured.
`base-stock/` is there so you can *diff* what the fine-tune changed. It carries the E8M0 block scales
alongside the fp8 weights: an fp8 weight without its scale is a table of codes, not a tensor.

**Before uploading**, read `README.md`. It names `0xSero/DeepSeek-V4-Flash-0731-REAP` as the base
model and quotes measured numbers. The numbers are checked against the logs; the attribution and the
base model's license terms are a human call, and `--repo-slug` still says `<your-org>`.

Rebuild the bundle any time with `python3 tools/make_bundle.py --out <dir> --repo-slug <org>/<repo>`.

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
