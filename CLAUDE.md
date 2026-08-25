# Operating rules for this repo

## DETACHMENT: if it does not need steering, it must not be tied to a session

**Rule.** Any long-running programmatic workload that does not depend on Claude Code to steer,
code, engineer, or actively monitor it **must be detached from the session before it starts**.
That is every battery in this repo: the eval suite, the extension pass and its retry, the forcing
pass, BFCL multi-turn, profiling sweeps, fine-tuning runs, agentic trace generation.

**A new unattended stage is not finished until `detach_audit.sh` knows its name.** The audit works
from a pattern list, so a stage absent from `PATTERNS` reports as "all detached" by never being
looked at -- a green audit that proves nothing. Add the stage to that list in the same commit.

    nohup setsid <cmd> > <log> 2>&1 < /dev/null &      # minimum -- see the caveat below
    systemd --user unit + loginctl enable-linger        # better; survives every logout

**`nohup setsid` is NOT sufficient when the launcher is itself a systemd unit.** `setsid` changes
the session; it does not change the **cgroup**. A stage launched that way inherits its launcher's
cgroup, and under the default `KillMode=control-group` systemd kills everything left in that cgroup
the moment the unit's main process exits. Measured 2026-08-25: killed under `control-group`,
survives under `KillMode=process`. `autopilot.sh` launches the eval battery and then exits, so the
whole battery would have been killed seconds after starting. **Give each unattended stage its own
transient unit** (`scripts/eval_resume.sh`'s `spawn()`); then its lifetime is its own, and
`systemctl --user status dsv4-ev-<stage>` can see it.

Canonical entry point here is `scripts/eval_resume.sh`, which launches every stage detached and is
idempotent. `dsv4-evals.service` runs it at boot; `dsv4-evals-watchdog.timer` re-runs it every ten
minutes so a dead stage self-heals.

**The converse is part of the rule.** When Claude *is* steering — iterative CUDA kernel
edit/build/test loops, debugging, anything where the next step depends on reading the last result —
a session-bound process is correct. Do not detach work that needs a driver. Headless Claude Code is
the third option when work needs steering but must not be tied to an interactive terminal; the
operator and Claude decide together when that is worth it.

**The gap was structural, not an operator slip — and it is now closed.** `scripts/run_model.sh`
has said since it was written that the benchmark must be "DETACHED. setsid+nohup so an SSH drop or
a killed shell never leaves the GPU wedged". The server path never inherited that:
`grep -c 'setsid\|nohup' scripts/serve.sh scripts/with_model_lock.sh` returns **0 and 0**, so
whether the engine survived depended entirely on how the caller invoked it. Sanctioned launchers,
all of which detach and then *prove* it:

| use | launcher |
|---|---|
| the decode benchmark | `scripts/run_model.sh` |
| the server alone | `scripts/run_server.sh` |
| the whole eval programme | `scripts/eval_resume.sh` (idempotent; also starts every dependent stage) |

`serve.sh` is deliberately left foreground-capable for debugging. The rule is that anything
long-running goes through a launcher that detaches.

**Verify parentage, not intent.** Detachment is a property of the process tree, not of how the
launch command was written. `bash scripts/detach_audit.sh` checks it. PPID must be `1` or
`systemd --user`, TTY must be `?`, and the session id must not be the launching shell's.
When auditing whether the programme will survive unattended, audit **what is already running**, not
just the logic of the scripts.

**Prefer launching from a local session on the Thor** (console, rustdesk, tmux on the host) over SSH
from a laptop. SSH adds a network dependency that a fully local workload — local checkpoint, local
GPU, no internet — does not otherwise have.

**Why this rule exists.** On 2026-08-17T21:55 the battery lost ~16 hours. The engine and
`eval_supervise` were children of a Claude Code Bash tool invocation inside an SSH session; the
laptop lid closed, SSH dropped, SIGHUP killed both. Nothing noticed until 14:03 the next day. The
stages launched the same day with `nohup setsid` survived the identical event and shut themselves
down cleanly on their own guards — same box, same instant, opposite outcome, and the only
difference was parentage.

## A stage that "completes" against a dead engine is worse than one that dies

Same incident, second failure: `eval_bfcl_mt_run.sh`'s smoke gate passed against a dead engine
because `eval_bfcl_mt.py` ended in an unconditional `return 0`. It then "scored" 400 items in two
minutes with every request `ECONNREFUSED` and published two rows of zeros. Every unattended stage
must exit **non-zero** on transport failure, and must not write records for items that generated
nothing — writing them banks wrong answers and defeats the resume, which keys off ids already on
disk.
