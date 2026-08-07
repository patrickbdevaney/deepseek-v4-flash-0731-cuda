#!/usr/bin/env bash
# flywheel_selftest.sh — prove the environment works BEFORE spending agent turns in it.
#
# Cycle 1 burned an entire iteration discovering that Bash was not approved, and cycle 0 burned turns
# on `awk /proc/meminfo` being outside the allowed directories. Both are clerical failures, and the
# agent's turns are the most expensive thing in the loop. Everything that can fail for a reason
# unrelated to CUDA is checked here, once, cheaply, and reported as one actionable line.
#
# exit 0 = the environment is fit for a cycle. Anything else = do not start the agent.
set -uo pipefail
cd "$(dirname "$0")/.."
MODEL="${DSV4_MODEL:-$HOME/models/DeepSeek-V4-Flash-0731-REAP}"
fail=0
ok(){ printf '  ok   %s\n' "$*"; }
no(){ printf '  FAIL %s\n' "$*"; fail=1; }

echo "[selftest] toolchain"
command -v nvcc  >/dev/null && ok "nvcc $(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1)" || no "nvcc not on PATH"
command -v g++   >/dev/null && ok "g++ $(g++ -dumpversion)"   || no "g++ not on PATH"
command -v git   >/dev/null && ok "git"                        || no "git not on PATH"
command -v flock >/dev/null && ok "flock"                      || no "flock not on PATH"

echo "[selftest] gpu"
if nvidia-smi -L >/dev/null 2>&1 || [ -e /dev/nvgpu ] || [ -e /dev/nvhost-gpu ]; then ok "gpu present"
else no "no GPU device node"; fi
# A trivial compile+run proves the driver, the arch flag and the runtime all agree. If this fails,
# every kernel change the agent makes this cycle would fail for the same reason.
T=$(mktemp -d); cat > "$T/t.cu" <<'EOF'
#include <cstdio>
__global__ void k(int* o){ *o = 1234; }
int main(){ int *d,h=0; cudaMalloc(&d,4); k<<<1,1>>>(d);
            cudaMemcpy(&h,d,4,cudaMemcpyDeviceToHost);
            printf("%d\n", h); return h==1234?0:1; }
EOF
if nvcc -O0 -gencode arch=compute_110a,code=sm_110a "$T/t.cu" -o "$T/t" >/dev/null 2>&1 && [ "$("$T/t" 2>/dev/null)" = "1234" ]
then ok "sm_110a compile + launch"; else no "cannot compile/run a trivial sm_110a kernel"; fi
rm -rf "$T"

echo "[selftest] repo"
[ -w . ]                          && ok "repo writable"          || no "repo not writable"
git rev-parse --git-dir >/dev/null 2>&1 && ok "git repo"          || no "not a git repo"
git config user.email >/dev/null && git config user.name >/dev/null \
                                  && ok "git identity set"        || no "git user.name/user.email unset — the harness cannot commit"
for f in scripts/run_model.sh scripts/await_log.sh scripts/build_decode.sh scripts/build_gate.sh; do
  [ -x "$f" ] && ok "$f" || no "$f missing or not executable"
done

echo "[selftest] model + memory"
[ -d "$MODEL" ] && ok "checkpoint dir"                            || no "checkpoint dir missing: $MODEL"
[ -r "$MODEL/model.safetensors.index.json" ] && ok "checkpoint index readable" \
                                             || no "checkpoint index unreadable"
AVAIL=$(free -g | awk '/^Mem:/{print $7}')
[ "${AVAIL:-0}" -ge 105 ] && ok "${AVAIL} GiB available" || no "only ${AVAIL} GiB available; a load needs ~105"
DISK=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
[ "${DISK:-0}" -ge 5 ] && ok "${DISK} GiB disk free" || no "only ${DISK} GiB disk free"

echo "[selftest] gates build and pass"
if bash scripts/build_gate.sh >/dev/null 2>&1; then ok "gates build"
else no "scripts/build_gate.sh fails — the agent would inherit a broken tree"; fi
for g in gate_units gate_bf16w gate_ogroup_gemv gate_tc_fp8_smem; do
  [ -x "build/$g" ] || continue
  if ./build/$g 2>&1 | grep -qi FAIL; then no "build/$g FAILS before the cycle starts"
  else ok "build/$g"; fi
done

echo "[selftest] no competing work"
pgrep -f build/decode >/dev/null && no "a full-model process is already running" || ok "gpu idle"

[ "$fail" -eq 0 ] && echo "[selftest] PASS" || echo "[selftest] FAIL — not starting a cycle"
exit "$fail"
