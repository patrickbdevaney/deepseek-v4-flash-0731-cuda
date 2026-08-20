# Draft-head registry

Every candidate head, its measurement, and whether it was promoted.
Written by `tools/promote_head.py`, which REFUSES to promote a head that cannot pass the gates in
its docstring: passing LOSSLESS gate, clean run, the frozen 8-prompt protocol at NGEN0>=200 and
block 6, and a suite mean beating the incumbent by more than the measured 3.5% run-to-run spread.
**Ties go to the incumbent.**

The product of this project is two things: the CUDA engine (in git, therefore safe) and the
speculator weights (not in git, therefore at risk). This registry plus
`~/model-backups/heads/<name>/` is how the second one stops being at risk.

| name | suite tau | suite tok/s | base AR | engine rev | status |
|---|---|---|---|---|---|
| `shipped-dspark-0731reap` | 3.5362 | 22.6550 | 13.76 | `2632540` | baseline |
| `s1` | 3.5762 | 24.515 | 13.8 | `06762c117` | PROMOTED |
| `s2` | 3.6275 | 24.7575 | 13.83 | `d2a62fd1e` | not promoted: suite 24.76 tok/s does not beat incumbent 24.52 by the 3.5%  |
| `s2-abl-ce0.5_tv0.5` | 3.6412 | 24.8175 | 13.76 | `e2a6cb47a` | not promoted: suite 24.82 tok/s does not beat incumbent 24.52 by the 3.5%  |
| `s2-abl-ce0.9_tv0.1` | 3.64 | 24.8975 | 14.14 | `e2a6cb47a` | not promoted: suite 24.90 tok/s does not beat incumbent 24.52 by the 3.5%  |
| `s2-abl-ce1.0_tv0.0` | 3.6712 | 25.1038 | 13.7 | `e2a6cb47a` | not promoted: suite 25.10 tok/s does not beat incumbent 24.52 by the 3.5%  |
| `s3` | 3.8438 | 25.5312 | 13.8 | `85dbea6cf` | PROMOTED |

## What is actually being served

The registry records which heads were *promoted*. Promotion is not deployment: `promote_head.py`
archives and never writes the live checkpoint, and until 2026-08-20 nothing else could name a
different one either — so `s3` was promoted on 2026-08-12 and **no server loaded it for eight days**.

| live head | since | mechanism |
|---|---|---|
| `s3` | 2026-08-20 (ladder 2.2) | `config/live_ckpt` -> `~/models/ckpt-head-s3`, a symlink farm built by `scripts/stage_head.sh` |

`config/live_ckpt` is tracked in git, so which head is in production is reviewable in a diff.
`scripts/serve.sh` reads it, and every server launcher in the repo execs `serve.sh`. To change it:

    bash scripts/stage_head.sh <name> --activate     # verifies, then repoints; does NOT restart
    bash scripts/run_server.sh                       # picks it up on the next start

`scripts/stage_head.sh` refuses any head not marked `PROMOTED` or `baseline` above, and
`tools/verify_staged_ckpt.py` proves the farm is the base checkpoint with exactly one head swapped
in — all 45,821 tensors drop-in identical in dtype/shape/byte-length, the non-head shards the same
inode as base, and the head shards' sha256 equal to what `head_card.json` recorded at promotion.

**Re-measured on engine `93699e6` (2026-08-20), both arms back to back on the frozen protocol:**
shipped `tau` 3.5362 / 22.1425 tok/s, `s3` `tau` 3.8438 / 24.2512 tok/s — **+8.70 % `tau`,
+9.52 % tok/s**, base AR within 0.70 % between the loads. Both suite `tau` values reproduce the
rows above **exactly**, which is the evidence the re-measurement was faithful; the tok/s columns did
not, and [`wiki/measurement-and-traps.md` §22](wiki/measurement-and-traps.md) is why criterion 4
above should not be read off two different engine revisions.
