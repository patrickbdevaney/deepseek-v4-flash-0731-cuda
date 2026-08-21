#!/usr/bin/env bash
# session_s3recap.sh — the first Thor-heavy, token-free session: re-capture s3's EXACT corpus with
# the fixed `hadamard`, and train on it.
#
# WHY THIS SESSION FIRST, AND WHY IT IS A CONTROL RATHER THAN AN EXPERIMENT. The capture stage of
# s5_session.sh runs ./build/decode, whose prefill was non-deterministic above ~192 positions until
# ladder 1.10 landed (aliased hadamard; 56/56 pairs divergent before, 56/56 byte-identical after).
# Every sequence in every corpus is inside that regime -- s3 1536/1536, min 525 tokens. So every
# head in HEAD_REGISTRY.md was trained on taps from a forward that did not reproduce itself.
# Whether that cost anything is UNMEASURED. Holding the corpus EXACTLY fixed and varying only the
# capture is the one comparison that answers it, and s3/gen.txt is retained, so pass 1 is skipped
# and no vLLM generation is needed. If the fix alone lifts tau, that gain applies to every head
# trained afterwards -- which is why it is worth knowing BEFORE the agentic corpus is built.
#
# It waits for the GPU rather than competing for it: ONE MODEL AT A TIME, 100.4 GiB in a 122 GiB
# pool. Started as a systemd --user unit so it survives the session that launched it.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG(){ printf '[s3recap %s] %s\n' "$(date -Is)" "$*"; }
DIE(){ LOG "HALT: $*"; exit 1; }

GEN=/home/patrickd/s5-capture/s3/gen.txt
PROMPTS=/home/patrickd/s5-capture/mixed_prompts_s3.txt
NAME=s3recap
N=1536

[ -s "$GEN" ]     || DIE "$GEN missing -- the whole point is to reuse s3's generation"
[ -s "$PROMPTS" ] || DIE "$PROMPTS missing -- pass 1's round-trip check needs the seed prompts"

# ---- gate 1: 2.1 must have closed, because the block width is baked into the CAPTURE ----
LOG "waiting for ladder 2.1 to close (block width is baked into DSV4_BLKSWEEP, not just --block)"
while ! grep -q '^- \[x\] \*\*2\.1' DECODE_LADDER.md; do
    pgrep -f 'decode_loop.sh' >/dev/null || { sleep 300; }      # loop gone: keep waiting, don't race
    sleep 120
done
LOG "2.1 is closed"

# ---- gate 2: the width we capture at must be the width the engine serves ----
# 2.1 shipped 6 -> 5 and this gate used to grep 2.1's write-up for "block 6". That was right for
# one day. The durable check reads the engine's own default and refuses to capture at anything
# else, because the width is baked into DSV4_BLKSWEEP at capture time, into --block at train time,
# and into tau's ceiling at eval time.
BLK="${S5_BLOCK:-5}"
ENGBLK=$(grep -oP 'int\s+blk\s*=\s*\K[0-9]+' include/dsv4_engine.h | head -1)
[ -n "$ENGBLK" ] || DIE "could not read the engine's default block from include/dsv4_engine.h"
[ "$BLK" = "$ENGBLK" ] || DIE "S5_BLOCK=$BLK but the engine serves $ENGBLK -- capturing at a width \
the engine does not run produces a head tuned for nothing"
LOG "capturing and training at block $BLK, which is what the engine serves"

# ---- gate 2b: a same-width incumbent, because tau's ceiling IS the width ----
# Every tau in HEAD_REGISTRY.md was taken at block 6. Comparing a block-5 candidate against one
# would be comparing a 5-ceilinged number with a 6-ceilinged one.
if [ ! -s evidence/baseline_tau.value ]; then
    LOG "no block-$BLK baseline yet; measuring the deployed head"
    bash scripts/baseline_tau.sh || DIE "baseline measurement failed"
fi
export S5_INCUMBENT_TAU=$(cat evidence/baseline_tau.value)
LOG "incumbent tau at block $BLK: $S5_INCUMBENT_TAU (deployed head, re-measured)"

# ---- gate 3: the GPU must be free ----
LOG "waiting for the GPU (one model at a time: 100.4 GiB of weights in a 122 GiB pool)"
while pgrep -x decode >/dev/null || pgrep -f 'dsv4-server --ckpt' >/dev/null \
      || pgrep -f 'decode_loop.sh' >/dev/null; do sleep 60; done
sleep 30
LOG "GPU is free; starting the session"

# 3 chunks, matching s3's own session shape (1536 corpus, 64 hold-out, 1472 training steps).
# S5_KEEP_CAP: 279 GB free against a 28 GB capture. Retaining it makes every later recipe arm
# cost training only. The capture is the artifact this session exists to produce correctly.
export S5_GEN="$GEN" S5_HOLDOUT=64 S5_CHUNK=491 S5_KEEP_CAP=1 S5_BLOCK="$BLK"
LOG "s5_session.sh $NAME $PROMPTS $N 512   (S5_GEN set: pass 1 is SKIPPED)"
bash scripts/s5_session.sh "$NAME" "$PROMPTS" "$N" 512
rc=$?
LOG "s5_session.sh exited rc=$rc"
exit $rc
