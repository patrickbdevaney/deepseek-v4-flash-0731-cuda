#!/usr/bin/env bash
# detach_audit.sh — prove the unattended workloads are actually detached, rather than assuming it.
#
# A written rule does not survive contact with a hurried launch, and the failure mode is silent:
# a session-bound process looks completely healthy right up until the connection drops. On
# 2026-08-17 the engine and eval_supervise had been running happily for four days as children of a
# Claude Code Bash invocation inside an SSH session; the laptop lid closed and both took SIGHUP,
# costing ~16 hours. Nothing about their behaviour beforehand hinted at it.
#
# Detachment is a property of the PROCESS TREE, not of how the launch command was written, so this
# checks the tree: a safe process is reparented to init or to `systemd --user`, has no controlling
# terminal, and is not in the session of any interactive shell.
#
#   bash scripts/detach_audit.sh        # exits non-zero if anything is session-bound
set -u
cd "$(dirname "$0")/.."
# `run_model.sh` and `build/decode` ADDED 2026-08-20 by ladder item 1.4, and the omission is the
# exact failure CLAUDE.md names: the sanctioned launcher for every full-model benchmark, and the
# 100.4 GiB process it launches, were not in this list -- so an audit taken while one was running
# reported "all detached" having never looked at it. It printed one row (memguard) for a
# three-process tree. A green audit that proves nothing is worse than a red one.
PATTERNS="${PATTERNS:-dsv4-server --ckpt|eval_supervise.sh|eval_extend_all.sh|eval_extend_retry.sh|eval_extend.py --task|eval_force_all.sh|eval_bfcl_mt_run.sh|eval_watch.sh|run_evals.sh|eval_suite.py --task|memguard.sh|perf_sample.py|decode_fit_probe.py|dprof_ctx_run.sh|mainkv_ab_run.sh|mainkv_verify_run.sh|mainkv_decodegate_run.sh|mainkv_determinism_run.sh|topk_ab_run.sh|topk_early_ab_run.sh|ixgemm_ab_run.sh|clocks_ab_run.sh|run_model.sh|build/decode}"

# NEVER FLAG OUR OWN ANCESTRY. This script is itself run from a shell -- often a Claude Code Bash
# invocation, which IS session-bound and correctly so. Its command line contains the patterns we
# grep for, and truncating args for display is not a filter. Walk up from self and exclude the
# whole chain of ancestors, which is the only way to be sure we are not auditing ourselves.
SELF=""
_p=$$
while [ -n "$_p" ] && [ "$_p" != "1" ] && [ "$_p" != "0" ]; do
  SELF="$SELF $_p"
  _p=$(ps -o ppid= -p "$_p" 2>/dev/null | tr -d ' ')
done

bad=0; n=0
printf '%-8s %-8s %-9s %-6s %s\n' PID PPID PARENT TTY PROCESS
while read -r pid; do
  [ -n "$pid" ] || continue
  case " $SELF " in *" $pid "*) continue;; esac
  ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ') || continue
  [ -n "$ppid" ] || continue
  tty=$(ps -o tty= -p "$pid" 2>/dev/null | tr -d ' ')
  cmd=$(ps -o args= -p "$pid" 2>/dev/null | cut -c1-58)
  # The Claude Code Bash tool keeps a long-lived wrapper shell per session, and its command line
  # retains whatever it was last asked to run -- so a shell that once typed `build/decode` matches
  # this grep forever and reports as a SESSION-BOUND stage. It is not a stage; it is the session
  # doing the auditing, which is session-bound by definition. The SELF walk above only excludes
  # our own ancestors, not a sibling shell of the same session, so exclude the wrapper by its
  # unmistakable signature. A red audit that is wrong teaches people to ignore a red audit.
  case "$cmd" in *detach_audit*|*ps\ -o*|*shell-snapshots*) continue;; esac
  pcmd=$(ps -o comm= -p "$ppid" 2>/dev/null | tr -d ' ')
  n=$((n+1))

  # SAFE IS A PROPERTY OF THE WHOLE ANCESTRY, NOT THE IMMEDIATE PARENT. Walk up from the process:
  # it is detached if the chain reaches PID 1 or the user systemd manager without passing through
  # anything that dies with a login -- a controlling terminal, an sshd, or a Claude Code session.
  # Checking only the immediate parent was wrong and produced a false positive on
  # `eval_suite.py --task`, whose parent run_evals.sh is itself perfectly detached.
  ok=0
  anc="$pid"
  for _ in $(seq 1 24); do
    anc=$(ps -o ppid= -p "$anc" 2>/dev/null | tr -d ' ')
    [ -n "$anc" ] || break
    if [ "$anc" = "1" ]; then ok=1; break; fi
    acomm=$(ps -o comm= -p "$anc" 2>/dev/null | tr -d ' ')
    atty=$(ps  -o tty=  -p "$anc" 2>/dev/null | tr -d ' ')
    case "$acomm" in systemd) ok=1;; esac
    [ "$ok" = "1" ] && break
    # anything with a controlling terminal, or an ssh/login/claude session, is a death sentence
    case "$acomm" in sshd|login|claude|node) ok=0; break;; esac
    [ "$atty" = "?" ] || { ok=0; break; }
  done
  # A controlling terminal on the process ITSELF is disqualifying whatever the ancestry says.
  [ "$tty" = "?" ] || ok=0

  if [ "$ok" = "1" ]; then
    printf '%-8s %-8s %-9s %-6s %s\n' "$pid" "$ppid" "$pcmd" "$tty" "$cmd"
  else
    bad=$((bad+1))
    printf '%-8s %-8s %-9s %-6s %s   <-- SESSION-BOUND\n' "$pid" "$ppid" "$pcmd" "$tty" "$cmd"
  fi
done < <(pgrep -f "$PATTERNS" 2>/dev/null)

echo
if [ "$n" = "0" ]; then
  echo "detach_audit: nothing matching is running."
  exit 0
fi
if [ "$bad" = "0" ]; then
  echo "detach_audit: $n process(es), all detached. A dropped SSH session cannot touch them."
  exit 0
fi
echo "detach_audit: $bad of $n process(es) are SESSION-BOUND and will die with the terminal that"
echo "started them. Restart them through scripts/eval_resume.sh, which detaches every stage."
exit 1
