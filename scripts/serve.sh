#!/usr/bin/env bash
# Start the production server. Runs the CPU-only gates first (seconds, no GPU, no checkpoint) so a
# broken tokenizer or encoder is caught before a ~10-minute weight load, not after it.
set -e; cd "$(dirname "$0")/.."

# WHICH CHECKPOINT THE SERVER SERVES IS A TRACKED DECISION, NOT AN AMBIENT ONE.
# Every server start in this repo goes through this script (run_server.sh, eval_resume.sh,
# eval_supervise.sh and deploy_staged_server.sh all exec it), so this line is the single place the
# engine's weights are chosen -- and it used to hardcode the base checkpoint. That is why the
# promoted `s3` draft head sat archived and unserved for eight days: promote_head.py deliberately
# only archives, and nothing else could name a different checkpoint.
#
# `config/live_ckpt` is that name, and it is IN GIT, so which head is in production is reviewable
# in the same diff as everything else instead of living in one operator's shell history. It is
# written by scripts/stage_head.sh --activate. If it names a directory that no longer exists we
# fall back to the base checkpoint LOUDLY rather than refusing to start -- a missing symlink farm
# must not be able to take the server down.
CKPT_DEFAULT=/home/patrickd/models/DeepSeek-V4-Flash-0731-REAP
if [ -z "${CKPT:-}" ] && [ -s config/live_ckpt ]; then
  PINNED=$(sed -n '1{s/[[:space:]]*$//;p}' config/live_ckpt)
  if [ -d "$PINNED" ]; then
    CKPT="$PINNED"
    echo "  live checkpoint pinned by config/live_ckpt -> $CKPT"
  else
    echo "  WARNING: config/live_ckpt names '$PINNED', which is not a directory."
    echo "  WARNING: falling back to $CKPT_DEFAULT — the promoted head is NOT being served."
  fi
fi
CKPT="${CKPT:-$CKPT_DEFAULT}"
PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"
# CONTEXT DEFAULT (2026-08-26). Measured on this box, weights 100.4 GiB of 122.8:
#     seqmax   8,192  fp32 KV   ready 119.6/122.8
#     seqmax  32,768  fp32 KV   ready 119.9/122.8      <- default: 4x the old one, and free
#     seqmax 131,072  fp32 KV   NEVER REACHES READY
#     seqmax 131,072  packed    ready 117.0/122.8      <- SEQMAX=131072 just works; packing auto-ons
#     seqmax 262,144  packed    ready 121.8/122.8      <- 1.0 GiB headroom, BELOW memguard's 1500 MB
#                                                         floor. It allocates; it cannot serve. Do
#                                                         not use without freeing win_kv first.
# Packing is bit-exact (identical tokens over 400 generated) and costs ~12% prefill, so kv_pack_init
# turns it on only above 32768 -- see kernels/mla_attn.cu. 32768 therefore pays nothing.
SEQMAX="${SEQMAX:-32768}"
EXT_CHUNK="${EXT_CHUNK:-64}"

[ -x build/dsv4-server ] || { echo "build/dsv4-server missing — run scripts/build_server.sh"; exit 1; }

for g in gate_tokenizer gate_encoding gate_api gate_stream; do
  [ -x "build/$g" ] || { echo "build/$g missing — run scripts/build_server.sh"; exit 1; }
  if ! out=$(./build/$g 2>&1); then
    echo "$out"; echo "PREFLIGHT FAILED: $g"; exit 1
  fi
  echo "  preflight $g ok"
done

# MOE_MMA -- THE PREFILL/DECODE LAYOUT CHOICE, MADE EXPLICIT (2026-08-26).
# src/decode.cu sets g_moe_gemv = (getenv("MOE_MMA")==nullptr), so the tensor-core MoE is opt-in and
# this script had never opted in: production prefill ran the M=1 GEMV on a full-batch prefill, which
# is the exact mismatch F85 identified. It cannot just be turned on -- tc_ensure_repacked mutates the
# weights IN PLACE and the decode GEMV needs the original layout, so one process gets one layout.
#
# Measured on decode at PS=845, hpb=4/smem=1 both sides, token streams IDENTICAL (bit-exact):
#     prefill  61.7 -> 90.7 tok/s  (+47.0%)      spec decode  32.54 -> 27.85 tok/s  (-14.4%)
# The prefill figure includes MOE_NBLK=4 (default), which is worth +20% on its own at ZERO decode
# cost -- 27.81 vs 27.85 tok/s across the pair. Per request: prefill saves P * 5.182 ms/1000, decode
# costs R * 5.175 ms/1000.
#     BREAK-EVEN AT P/R = 1.00.  (It was 1.75 before NB=4 halved the prefill side of the trade.)
# So this wins whenever the prompt is at least as long as the response. Agentic coding runs P/R of
# 5-50; even chat at 1-3 is now on the winning side. Hence DEFAULT ON.
# DEFAULT ON as of 2026-08-26: this server exists for prompt-heavy agentic work, where P/R runs
# 5-50 against a break-even of 1.75. Set MOE_MMA=0 to serve a decode-dominated workload (P/R < 1.75)
# or to reproduce the pre-2026-08-26 decode headline.
if [ "${MOE_MMA:-1}" != "0" ]; then
    export MOE_MMA=1
    echo "  MOE_MMA=1 (default): tensor-core MoE. prefill +22.2%, decode -14.4%; break-even P/R 1.75."
else
    unset MOE_MMA
    echo "  MOE_MMA=0: M=1 GEMV MoE. Decode-optimal; prefill gives up 22.2%."
fi
exec ./build/dsv4-server --ckpt "$CKPT" --host "$HOST" --port "$PORT" --seqmax "$SEQMAX" \
     --ext-chunk "$EXT_CHUNK" "$@"
