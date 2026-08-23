# BEST_SETUP.md — the winning configuration, stated so it can be rebuilt

What is served today, why, and the exact commands that reproduce it. Written 2026-08-23, after the
block-5 draft-head programme closed P2.5, P2.6 and 2.3's training half.

The narrative and the arms that lost are in [`wiki/draft-head-finetuning.md`](wiki/draft-head-finetuning.md)
§9–§10. This file is the operational one: **what to run to get this result back.**

## 1. What is served

| | |
|---|---|
| head | **`s3recap-p25-b0.1`** |
| suite mean `tau` | **3.8413 / 5** (77 % of the width ceiling) |
| suite mean tok/s | **28.3825** |
| base AR | 14.61 tok/s — speculation is **1.97×** |
| draft block width | **5** (`config.json`'s own `dspark_block_size`) |
| `config/live_ckpt` | `/home/patrickd/models/ckpt-head-s3recap-p25-b0.1` |
| engine rev at promotion | `a608d52d76f86d6325b472e34696b15d5b5fb6da` |
| source sha256 | recorded in `head_card.json`; `tools/verify_head_archive.py --full` checks it |

Against the stock head this project started from: **22.66 → 28.38 tok/s, +25.3 %**, every emitted
token still bit-identical to base AR.

## 2. The recipe, exactly

One epoch over 1,472 sequences (1,536 generated, 64 reserved as hold-out), three chunks of 491.

| knob | value | why this and not another |
|---|---|---|
| `--block` | **5** | resolved from `config.json`'s `dspark_block_size`. NOT a tuned value — ladder 2.1 found the served width had drifted to 6 and moving it back was worth +3.91 % at equal `tau`. |
| `--a-ce` / `--a-tv` | **0.1 / 0.9** | the control's weights. The ce/tv sweep (1.0/0.0 and 0.5/0.5) both lost; F119 had already falsified the CE/TV explanation of F117. |
| `--deficit` | **on** | weights each block's loss by `min((1-r)/mean(1-r), 3.0)`, `r` = that block's accepted fraction. Spends gradient on the head's worst positions, which is where a draft head's value lives (§8.3). |
| `--deficit-prior` | 0.4106 | the running mean's prior for the first 32 steps. |
| `--deficit-clamp` | 3.0 | |
| `--beta` | **0.1** | KL pull toward the head as it was before the first optimizer step, scaled by `r`. **Bracketed on both sides**: β=0 is short at +2.9 %, β=0.5 lands *below* the incumbent. |
| `--a-conf` | **0.0** | measured, not defaulted. See §4. |
| `--hass-from` | **0 (off)** | HASS is retired — `wiki/negative-results.md` §4l. |
| `--lr` | 5e-5 peak, 73 warmup steps | |
| `--pos-per-seq` | 16 | |
| trainable | 517,318,215 params across 72 tensors | |

Final chunk's metrics: loss 1.627 → 0.037, tail mean 0.739; ce 0.097, tv 0.029.

## 3. Reproducing it

```bash
# 1. the corpus already exists and is retained -- pass 1 (generation) is the expensive stage and
#    does NOT need to be re-run. /home/patrickd/s5-capture/s3/gen.txt, 1536 sequences.
# 2. run the arm. S5_* map onto the flags in the table above.
S5_GEN=/home/patrickd/s5-capture/s3/gen.txt S5_HOLDOUT=64 S5_CHUNK=491 S5_KEEP_CAP=1 \
S5_BLOCK=5 S5_ACE=0.1 S5_ATV=0.9 S5_DEFICIT=1 S5_BETA=0.1 \
  bash scripts/s5_session_p25.sh myhead /home/patrickd/s5-capture/mixed_prompts_s3.txt 1536 512
```

Re-capturing that corpus costs pass 2 + train only, ~0.4 h per 500 sequences. **Generation is what
dominates a session and it is already done.**

To deploy a head once it promotes:

```bash
bash scripts/stage_head.sh <name> --activate    # verifies the farm, then repoints config/live_ckpt
bash scripts/run_server.sh                      # picks it up on the next start
```

## 4. Two knobs that look like oversights and are measurements

**`--a-conf 0`.** F100 measured the confidence term at `conf` ≈ 10034 against `ce` 10.43 under
teacher forcing and pinned it at 0. Ladder 2.3 then measured the free-running magnitude at **O(1)**
and trained two arms at `a_conf` 1.0 and 0.1. Both are **worse** than this recipe, because getting
free-running labels requires HASS and HASS costs −0.046 `tau`. The confidence head gains **+0.018
AUC** from our fine-tuning, which does not pay for that. Keep it at 0.

**The confidence head is still good — it is just not ours.**
`mtp.2.confidence_head.proj.weight` ships in the base checkpoint, is byte-identical in every
`a_conf = 0` head including this one, and scores **AUC 0.88** predicting acceptance. That is what
justifies 2.3's serving-side work, and it needs no training arm at all.

**`--hass-from 0`.** Not untried: measured twice, monotone in the wrong direction. See §4l.

## 5. Durability — where this lives and how it is checked

| artifact | where | recoverable if the disk dies? |
|---|---|---|
| engine + every document | git, pushed to `origin/main` | **yes** |
| head weights (source) | `~/model-backups/heads/<name>/mtp_trained.safetensors`, 1 GB each | **only if uploaded** |
| head weights (loadable shards) | same directory, ~7 GB each | regenerate from the source |
| training corpus | `/home/patrickd/s5-capture/s3/gen.txt` | regenerable, but the generate pass is the expensive one |
| the recipe | **this file**, in git | yes |

    python3 tools/verify_head_archive.py          # presence + size + completeness, seconds
    python3 tools/verify_head_archive.py --full   # + sha256 of every file, needs an idle box

Every head's `mtp_trained.safetensors` now carries a recorded sha256 in its `head_card.json` (added
2026-08-23). Before that the archive could not detect corruption of the one file everything else
regenerates from, which made "rebuildable" an unfalsifiable claim.

**Nothing is ever deleted from `~/model-backups/heads/`.** Intermediate per-chunk resume state in
`s5-capture` (`c0`/`c1` weights and AdamW moments) IS deletable once the session is complete and
its final `c2` state is archived and hash-verified — that is how 63 GiB was reclaimed on
2026-08-23 — but the archive itself is append-only.
