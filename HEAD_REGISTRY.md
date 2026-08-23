# Draft-head registry

Every candidate head, its measurement, and whether it was promoted.
Written by `tools/promote_head.py`, which REFUSES to promote a head that cannot pass the gates in
its docstring: passing LOSSLESS gate, clean run, the frozen 8-prompt protocol at NGEN0>=200 and
block 6, and a suite mean beating the incumbent by more than the measured 3.5% run-to-run spread.
**Ties go to the incumbent.**

The product of this project is two things: the CUDA engine (in git, therefore safe) and the
speculator weights (not in git, therefore at risk). This registry plus
`~/model-backups/heads/<name>/` is how the second one stops being at risk.

| name | suite tau | blk | suite tok/s | base AR | engine rev | status |
|---|---|---|---|---|---|---|
| `shipped-dspark-0731reap` | 3.5362 | 6 | 22.6550 | 13.76 | `2632540` | baseline |
| `s1` | 3.5762 | 6 | 24.515 | 13.8 | `06762c117` | PROMOTED |
| `s2` | 3.6275 | 6 | 24.7575 | 13.83 | `d2a62fd1e` | not promoted: suite 24.76 tok/s does not beat incumbent 24.52 by the 3.5%  |
| `s2-abl-ce0.5_tv0.5` | 3.6412 | 6 | 24.8175 | 13.76 | `e2a6cb47a` | not promoted: suite 24.82 tok/s does not beat incumbent 24.52 by the 3.5%  |
| `s2-abl-ce0.9_tv0.1` | 3.64 | 6 | 24.8975 | 14.14 | `e2a6cb47a` | not promoted: suite 24.90 tok/s does not beat incumbent 24.52 by the 3.5%  |
| `s2-abl-ce1.0_tv0.0` | 3.6712 | 6 | 25.1038 | 13.7 | `e2a6cb47a` | not promoted: suite 25.10 tok/s does not beat incumbent 24.52 by the 3.5%  |
| `s3` | 3.8438 | 6 | 25.5312 | 13.8 | `85dbea6cf` | PROMOTED |

## Two corrections that qualify every row above

**1. The ruler was wrong until 2026-08-21 (ladder 2.4).** `promote_head.py`'s docstring always said
the gate was suite-mean **`tau`**; the code compared **`suite_tok_s`**, against a row read out of
this file — i.e. a number recorded on whatever engine revision was current when *that* head was
measured. `tau` is a property of the HEAD and reproduces to four decimal places across 8 days and
five decode-kernel rewrites; tok/s is a property of the ENGINE and moved -2.3 % / -5.0 %, with base
AR moving **-17.4 %**. Fixed: the comparison and the incumbent search both use `tau`, and
`--incumbent-tau` accepts a value re-measured in the same session, printing which source was used.

Re-adjudicating the rows under the corrected rule changes exactly one verdict:

| head | `tau` | needed | corrected verdict | recorded verdict |
|---|---|---|---|---|
| `s1` | 3.5762 | > 3.6600 | **refuse** (+1.1 %, inside the 3.5 % spread) | PROMOTED |
| `s2` and the three ablations | 3.6275–3.6712 | > 3.7014 | refuse | refuse |
| `s3` | 3.8438 | > 3.7014 | **PROMOTE** | PROMOTED |

`s1` was promoted on a margin that the spread cannot resolve. It is moot for what is served — `s3`
superseded it — but the ledger should say so. Note the counterfactual: had `s1` correctly been
refused, the incumbent would have stayed at baseline 3.5362 and `s2-abl-ce1.0_tv0.0` (3.6712) would
have cleared 3.6600 and been promoted. The refusals were not all robust to the ruler.

**2. Every head above was trained on taps captured through a racing forward.** The capture stage of
`scripts/s5_session.sh` runs **`./build/decode`**, whose prefill was non-deterministic above ~192
positions until ladder 1.10 landed on 2026-08-21 (an aliased `hadamard`, three call sites passing
the same pointer twice; 56 of 56 pairs divergent before the fix, 56 of 56 byte-identical after).
The corpora are entirely inside that regime:

| corpus | sequences | min / median / max tokens | above 192 |
|---|---|---|---|
| `s3/gen.txt` | 1536 | 525 / 585 / 1799 | **1536 / 1536** |
| `s1_gen.txt` | 500 | 539 / 578 / 835 | **500 / 500** |

Whether this materially degraded the heads is **UNMEASURED**, and it should not be asserted either
way. It is cheap to measure: `s3/gen.txt` is retained, so re-capturing that exact corpus post-fix
costs pass 2 + train and needs **no vLLM generation**. That is the designated first session, and it
is a control against an archived number rather than a new experiment.

## Which head is the *best* one, and where to upload it from

Promotion (above) answers "does this beat the incumbent on the suite mean". It does not answer
"which head do we publish", and once P2 starts producing heads tuned on a pattern-balanced corpus
the mean alone **cannot** answer it: a head can win the mean by hollowing out the reconstructive
categories, which are exactly the ones an agentic coding harness lives in.

**The release rule, written 2026-08-20, before the P2 candidates exist.** The released head is the
one with the highest **suite mean `tau`** that also clears **both** floors:

| floor | categories | condition |
|---|---|---|
| reconstructive | `long_context`, `agentic_format`, `code_edit` | not below run-0 minus **0.2** |
| constructive | `explanation`, `code_gen`, `reasoning` | not below the **incumbent's** value |

Both arms measured **in the same session** as the incumbent — see ladder item 2.4 for why a
cross-session comparison is not admissible. A head that fails either floor is archived and recorded,
never released, however good its mean.

**The pointer.** `~/model-backups/releases/CURRENT_BEST` is a symlink to the release bundle that
currently satisfies the rule. It is the directory to upload; nothing else in `releases/` is a
publication candidate.

| | |
|---|---|
| `CURRENT_BEST` | -> `dspark-mtp-draft-head-v1.0-s3` |
| bundle contents | `README.md` (HF model-card frontmatter), `SHA256SUMS`, `provenance.json`, `head_card.json`, `mtp_trained.safetensors`, `train_metrics.json`, `eval.log`, `config.json` |
| next name | `dspark-mtp-draft-head-v2.0-<name>` — the directory name states the claim |

A P2 bundle's `README.md` must carry **the per-category `tau` table, not just the suite mean**. The
mean is what promotion is decided on; the table is what a downstream user needs to predict their own
workload, and it is the number this project learned the hard way is not interchangeable with it.


## The release rule, evaluated for the first time at block 5 (2026-08-23)

The rule above floors the reconstructive categories at **run-0 minus 0.2** and the constructive ones
at **the incumbent**. It had been **unevaluable since ladder 2.1**: run-0 existed only at block 6,
every candidate is block 5, and `tau` is not comparable across widths. `BASELINE_CKPT=<base ckpt>
scripts/baseline_tau.sh` now measures run-0 at the served width without staging anything —
**run-0 @ blk 5 = suite tau 3.2550, 23.2538 tok/s, base AR 13.86** (`evidence/run0_tau_blk5.log`).

| category | run-0 | `s3` (incumbent) | `s3recap-p25-b0.1` | floor | |
|---|---:|---:|---:|---:|---|
| long_context | **5.00** | 4.00 | 4.53 | >= 4.80 | **FAIL** −0.27 |
| agentic_format | **4.41** | 3.74 | 4.16 | >= 4.21 | **FAIL** −0.05 |
| code_edit | 4.08 | 3.85 | **4.18** | >= 3.88 | pass |
| reasoning | 1.82 | **3.70** | 3.57 | >= 3.70 | **FAIL** −0.13 |
| explanation | 1.70 | 2.94 | **3.00** | >= 2.94 | pass |
| code_gen | 1.86 | 2.56 | **2.59** | >= 2.56 | pass |
| multi_turn | 4.06 | **5.21** | 4.83 | — | not floored |
| short_factual | 3.11 | 3.51 | **3.87** | — | not floored |
| **suite mean** | 3.2550 | 3.6888 | **3.8413** | | |

**`s3recap-p25-b0.1` fails 3 of the 6 floors, so `CURRENT_BEST` is NOT repointed at it.** Promotion
and release are different gates and this is the case that separates them: the head wins the mean by
**+18.0 % over run-0** and is correctly promoted and correctly deployed, and it still gives back
ground on the two categories the stock head was best at. That is F117 stated in the released
artifact instead of in a wiki page.

`CURRENT_BEST` therefore still points at `dspark-mtp-draft-head-v1.0-s3`, which was released under
block-6 measurements and has not been re-adjudicated at block 5 either. **Neither pointer is
currently backed by a rule-passing block-5 measurement**, and saying so is more useful than moving
the symlink to whichever head is newest.

**What would pass.** The failures are small and one-directional: long_context −0.27 is the only one
outside noise. The deficit weighting that won P2.5 spends gradient on the head's *worst* positions,
which is why the constructive categories roughly doubled; the reconstructive give-back is the same
mechanism seen from the other side. A corpus reweighted toward long-context reconstruction, or a
deficit clamp that stops re-weighting once a category is already strong, is the obvious next arm —
and it is now measurable against a real run-0 rather than against a block-6 ghost.

**The published head states this itself.** `evidence/hf_model_card_s3recap-p25-b0.1.md` is the model
card as uploaded; it carries the per-category table and says in as many words that the head does not
pass the release rule and should be chosen by workload rather than by the mean.

## Keeping the archive real

The weights are not in git, so nothing about them is recoverable from a clone.
`tools/verify_head_archive.py` proves the archive is intact and complete — every file present at the
size and (under `--full`) the sha256 its `head_card.json` or `SHA256SUMS` claims, every registry row
backed by a directory, and every directory named by a registry row.

    python3 tools/verify_head_archive.py            # size + presence, seconds, safe any time
    python3 tools/verify_head_archive.py --full     # + sha256; reads ~30 GB, LOOP MUST BE IDLE

First run, 2026-08-20: **10 directories, 46 files, 0 problems, 0 completeness gaps**
(`evidence/head_archive_quick.json`). Heads archived as `mtp_trained.safetensors` only are complete
by design — the loadable shards regenerate from it deterministically via
`tools/build_trained_head.py`, and skipping them saves ~7 GB per head. That is reported
`SOURCE-ONLY`, not as a failure.

**Nothing is ever deleted from `~/model-backups/heads/`.** Every arm of every sweep is archived
whether or not it is promoted: a rejected head is still a measured point on the
acceptance-vs-corpus curve, and three of the rows above are refusals that ladder 2.4 exists to
re-adjudicate.

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
| `s3recap` | 3.625 | 5 | 26.8487 | 14.59 | `92c761d7e` | not promoted: suite tau 3.6250 does not beat incumbent 3.6888 by the 3.5%  |
| `s3recap-ce1.0` | 3.7325 | 5 | 27.6913 | 14.53 | `a608d52d7` | not promoted: suite tau 3.7325 does not beat incumbent 3.6888 by the 3.5%  |
| `s3recap-ce0.5` | 3.695 | 5 | 27.5113 | 14.61 | `a608d52d7` | not promoted: suite tau 3.6950 does not beat incumbent 3.6888 by the 3.5%  |
| `s3recap-deficit` | 3.7975 | 5 | 27.8712 | 14.59 | `a608d52d7` | not promoted: suite tau 3.7975 does not beat incumbent 3.6888 by the 3.5%  |
| `s3recap-p25-b0.1` | 3.8413 | 5 | 28.3825 | 14.61 | `a608d52d7` | PROMOTED |
| `s3recap-p25-b0.5` | 3.6738 | 5 | 27.0825 | 14.56 | `a608d52d7` | not promoted: suite tau 3.6738 does not beat incumbent 3.6888 by the 3.5%  |
| `s3recap-hass1` | 3.6225 | 5 | 26.7563 | 14.52 | `a608d52d7` | not promoted: suite tau 3.6225 does not beat incumbent 3.8413 by the 3.5%  |
| `s3recap-hass1-p25` | 3.795 | 5 | 27.8562 | 14.55 | `a38c6c85a` | not promoted: suite tau 3.7950 does not beat incumbent 3.8413 by the 3.5%  |
| `s3recap-conf1.0` | 3.8025 | 5 | 28.0187 | 14.57 | `b9c320838` | not promoted: suite tau 3.8025 does not beat incumbent 3.8413 by the 3.5%  |
| `s3recap-conf0.1` | 3.7925 | 5 | 28.0362 | 14.56 | `b9c320838` | not promoted: suite tau 3.7925 does not beat incumbent 3.8413 by the 3.5%  |
