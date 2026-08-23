#!/usr/bin/env bash
# autopilot.sh — keep the box working on the draft-head search without a human in the loop, and
# when the search is done, hand the GPU to the eval battery.
#
# WHAT THIS IS FOR. This hardware costs nothing per hour and cannot get an idle hour back. Every
# arm so far was chosen by hand from the previous result, which works while someone is watching and
# stops the moment they are not. This turns that judgement into a rule.
#
# WHAT IT WILL AND WILL NOT DECIDE
#
#   decides alone : which arm to run next, whether an arm promoted, whether it passes the release
#                   floors, when the search is exhausted, when to start the eval battery
#   decides alone : staging a head ONLY if it clears BOTH gates -- suite mean AND every per-category
#                   floor. s3recap-p25-b0.1 promoted on the mean and failed 3 of 6 floors, so
#                   "promoted" alone is not a deployment criterion and is not treated as one here.
#   never decides : anything needing new CUDA. Phase 2 (prefill) and phase 3 (prefix caching) are
#                   NOT in this loop, deliberately -- they need code written, not runs launched.
#                   The two phases that eat the most wall clock, arms and evals, are exactly the
#                   two that can run unattended, so those are what the overnight box does.
#
# STOP IT:  touch AUTOPILOT_STOP        (exits cleanly after the arm in flight)
# WATCH IT: tail -f evidence/autopilot/decisions.log
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
S5=/home/patrickd/s5-capture
STATE=evidence/autopilot/arms.jsonl
QUEUE=evidence/autopilot/queue.tsv
DEC=evidence/autopilot/decisions.log
MAXARMS="${AUTOPILOT_MAX_ARMS:-4}"         # 24 arms buys ~+1.5 tok/s at best; the corpus is the
                                           # binding constraint. See scripts/chain_corpus.sh.
MINFREE_GB=40
mkdir -p evidence/autopilot; touch "$QUEUE"
LOG(){ printf '[auto %s] %s\n' "$(date -Is)" "$*" | tee -a "$DEC"; }

gpu_busy(){ pgrep -x decode >/dev/null || pgrep -x dsv4-server >/dev/null; }
# dsv4-ck is in this list because the C(k) sweep PRICES the biggest remaining lever and must not
# queue behind four low-value arms -- and because this loop ends by starting the eval battery, so
# anything it does not wait for would contend with the battery instead.
other_chain(){ systemctl --user is-active --quiet dsv4-p25b dsv4-c23 dsv4-p26 dsv4-chain dsv4-corpus dsv4-ck dsv4-resume 2>/dev/null; }
free_gb(){ df --output=avail -BG / | tail -1 | tr -dc '0-9'; }

reclaim(){
    # c0/c1 resume state of any COMPLETED arm whose final weights are archived. Never touches the
    # archive, never touches c2 (which is what a session would resume from).
    local n=0
    for d in "$S5"/*/; do
        local a; a=$(basename "$d")
        [ -s "$HOME/model-backups/heads/$a/mtp_trained.safetensors" ] || continue
        for f in "$d"c[01]/trained/opt_state.pt "$d"c[01]/trained/mtp_trained.safetensors; do
            [ -e "$f" ] && { sudo rm -f "$f" && n=$((n+1)); }
        done
    done
    LOG "reclaim: removed $n intermediate files, $(free_gb) GB free"
}

record(){ # name a_ce a_tv deficit beta anchor_pow dclamp pos
    local name="$1"
    local tau rel
    tau=$(python3 - "$name" <<'PY'
import re,sys
name=sys.argv[1]
try:
    for line in open("HEAD_REGISTRY.md"):
        m=re.match(r"\|\s*`%s`\s*\|\s*([\d.]+)\s*\|" % re.escape(name), line)
        if m: print(m.group(1)); break
    else: print("")
except Exception: print("")
PY
)
    rel=fail
    if python3 tools/release_rule.py "evidence/${name}_eval.log" \
         --run0 evidence/run0_tau_blk5.log --incumbent evidence/baseline_tau_blk5.log \
         >> "evidence/autopilot/release_${name}.log" 2>&1; then rel=pass; fi
    python3 - "$name" "$tau" "$rel" "$2" "$3" "$4" "$5" "$6" "$7" "$8" <<'PY' >> "$STATE"
import json,sys
n,tau,rel,ace,atv,dfc,beta,ap,dcl,pos=sys.argv[1:11]
print(json.dumps({"name":n,"tau":float(tau) if tau else None,"release_pass":rel=="pass",
 "cfg":{"a_ce":float(ace),"a_tv":float(atv),"deficit":int(dfc),"beta":float(beta),
        "anchor_pow":float(ap),"deficit_clamp":float(dcl),"pos_per_seq":int(pos)}}))
PY
    LOG "RESULT $name: tau=${tau:-unparsed} release=$rel"
}

run_arm(){ # name a_ce a_tv deficit beta anchor_pow dclamp pos
    local name="$1"
    if [ -d "$HOME/model-backups/heads/$name" ]; then LOG "$name already archived, skipping"; return 0; fi
    LOG "ARM $name: a_ce=$2 a_tv=$3 deficit=$4 beta=$5 anchor_pow=$6 dclamp=$7 pos=$8"
    mkdir -p "$S5/$name"; ln -sfn ../s3/gen.txt "$S5/$name/gen.txt"
    for ci in 0 1 2; do
        mkdir -p "$S5/$name/c$ci"; ln -sfn ../../s3recap/c$ci/cap "$S5/$name/c$ci/cap"
        local lo=$(( ci*491+1 )) hi=$(( ci*491+491 )); [ "$hi" -gt 1472 ] && hi=1472
        sed -n "${lo},${hi}p" "$S5/s3/gen.txt" > "$S5/$name/c$ci/gen.txt"
    done
    S5_GEN="$S5/s3/gen.txt" S5_HOLDOUT=64 S5_CHUNK=491 S5_KEEP_CAP=1 S5_BLOCK=5 \
    S5_INCUMBENT_TAU="$(cat evidence/baseline_tau.value 2>/dev/null)" \
    S5_ACE="$2" S5_ATV="$3" S5_DEFICIT="$4" S5_BETA="$5" S5_ANCHOR_POW="$6" \
    S5_DCLAMP="$7" S5_POS="$8" \
        bash scripts/s5_session_auto.sh "$name" "$S5/mixed_prompts_s3.txt" 1536 512 \
        || LOG "arm $name FAILED (rc=$?) -- recording and continuing; a failed arm is a data point"
    record "$@"
}

finalize(){
    LOG "=========== SEARCH EXHAUSTED -- handing the GPU to the eval battery ==========="
    local best
    best=$(python3 - <<'PY'
import json
rows=[json.loads(l) for l in open("evidence/autopilot/arms.jsonl") if l.strip()]
ok=[r for r in rows if r.get("release_pass") and r.get("tau")]
print(max(ok,key=lambda r:r["tau"])["name"] if ok else "")
PY
)
    if [ -n "$best" ]; then
        LOG "best head clearing BOTH gates: $best -- staging it"
        bash scripts/stage_head.sh "$best" --activate 2>&1 | tail -5 | tee -a "$DEC"
    else
        LOG "NO arm cleared both gates. Live head unchanged ($(cat config/live_ckpt))."
        LOG "  The mean-only winner is deliberately NOT staged on that basis -- see HEAD_REGISTRY.md."
    fi
    LOG "starting the eval battery: extend -> retry -> force -> BFCL multi-turn"
    bash scripts/eval_resume.sh 2>&1 | tail -20 | tee -a "$DEC"
    LOG "eval battery launched; autopilot exiting. Watch: bash scripts/eval_watch.sh"
}

LOG "autopilot start: max $MAXARMS arms, then the eval battery"
n=0
while :; do
    [ -e AUTOPILOT_STOP ] && { LOG "AUTOPILOT_STOP present -- exiting cleanly"; exit 0; }
    if other_chain || gpu_busy; then LOG "GPU busy (another chain or a model resident); waiting"; sleep 300; continue; fi
    [ "$(free_gb)" -lt "$MINFREE_GB" ] && reclaim
    if [ "$(free_gb)" -lt "$MINFREE_GB" ]; then
        LOG "FATAL: only $(free_gb) GB free after reclaim -- refusing to start an arm that will die mid-chunk"
        exit 1
    fi
    # Fold in anything measured OUTSIDE this loop (a hand-launched chain, a previous autopilot run)
    # before reasoning about what to try next. Searching around a stale incumbent is the failure
    # mode this prevents.
    python3 tools/ingest_arms.py >> "$DEC" 2>&1
    if [ ! -s "$QUEUE" ]; then
        LOG "queue empty -- proposing the next generation"
        python3 tools/propose_arms.py --state "$STATE" --max 3 > "$QUEUE" 2>>"$DEC"
        rc=$?
        if [ "$rc" -ne 0 ] || [ ! -s "$QUEUE" ]; then finalize; exit 0; fi
        LOG "proposed: $(wc -l < "$QUEUE") arm(s)"
    fi
    read -r name ace atv dfc beta ap dcl pos < "$QUEUE"
    sed -i '1d' "$QUEUE"
    [ -z "${name:-}" ] && continue
    run_arm "$name" "$ace" "$atv" "$dfc" "$beta" "$ap" "$dcl" "$pos"
    n=$((n+1))
    [ "$n" -ge "$MAXARMS" ] && { LOG "hit the $MAXARMS-arm backstop"; finalize; exit 0; }
done
