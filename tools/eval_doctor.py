#!/usr/bin/env python3
"""eval_doctor.py — find the structural faults that quietly ruin a long battery, and fix them.

The supervisor restarts the battery when it exits and memguard protects the box. Neither notices the
failures that leave everything apparently running while the RESULT is being destroyed:

  * the watcher dies, so finished benchmarks are never verified, committed or pushed -- and nothing
    restarts it until run_evals exits, which on a six-day battery may be days away
  * a process is killed mid-write and the jsonl ends in half a record, which breaks resume (the id
    is unreadable so the item is re-run and DUPLICATED) and fails verification later
  * two runners end up on the same task and interleave records into one file
  * the server answers /health while generation is wedged, so the log stops advancing and every
    watchdog stays green
  * the disk fills, and every subsequent write is lost silently

Each check states what is wrong, whether it was repaired, and what a human has to do if not.
Repairs are conservative: restart things that are safe to restart, truncate a partial trailing line
after taking a backup, and never delete evidence or touch the engine. Anything riskier is escalated
as ESCALATE, which the observer monitors for.

  python3 tools/eval_doctor.py            # check and repair once
  python3 tools/eval_doctor.py --check    # report only, change nothing
"""
import argparse, glob, json, os, re, shutil, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'evidence', 'evals')
RUN_LOG = os.path.join(OUT, 'run.log')
EFFORT = os.environ.get('EFFORT', 'low')
STALL_MIN = int(os.environ.get('STALL_MIN', '60'))
MIN_FREE_GB = float(os.environ.get('MIN_FREE_GB', '5'))

findings = []


def note(level, what, detail=''):
    findings.append((level, what, detail))
    print(f'[{level:8}] {what:<44} {detail}', flush=True)


def pgrep(pat):
    """Processes whose actual argv STARTS with `pat` — never a shell that merely mentions it.

    `pgrep -f` matches anywhere in the command line, so any wrapper shell running
    `bash -c '... eval_extend_all.sh ...'` matches too. That is not hypothetical: the first live
    test of this doctor killed the extension watcher, and the doctor then reported it ALIVE because
    it had matched the very shell that ran the check. A watchdog that reports a dead process as
    healthy is worse than no watchdog, so this reads /proc and compares the argv prefix, skipping
    itself and its own ancestors.
    """
    want = pat.split()
    me = os.getpid()
    mine = {me}
    try:                                        # walk up so the calling shell cannot match either
        pid = me
        for _ in range(6):
            ppid = int(open(f'/proc/{pid}/stat').read().rsplit(')', 1)[1].split()[1])
            if ppid <= 1:
                break
            mine.add(ppid)
            pid = ppid
    except Exception:
        pass

    out = []
    for d in os.listdir('/proc'):
        if not d.isdigit() or int(d) in mine:
            continue
        try:
            argv = open(f'/proc/{d}/cmdline', 'rb').read().split(b'\x00')
        except Exception:
            continue
        argv = [x.decode('utf-8', 'replace') for x in argv if x]
        if not argv:
            continue
        # argv[0] may be an absolute path (/bin/bash vs bash); compare on basename for the first
        head = [os.path.basename(argv[0])] + argv[1:]
        if head[:len(want)] == want:
            out.append(int(d))
    return out


def spawn(cmd, log):
    """Detached restart, output appended so a repair leaves a trail."""
    with open(log, 'a') as f:
        subprocess.Popen(['setsid'] + cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)


def read_tail(path, n=400_000):
    try:
        with open(path, 'rb') as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - n))
            return f.read().replace(b'\x00', b'').decode('utf-8', 'replace')
    except Exception:
        return ''


# --------------------------------------------------------------------------- integrity of records
def check_jsonl(fix):
    for path in sorted(glob.glob(os.path.join(OUT, f'*.{EFFORT}*.jsonl'))):
        name = os.path.basename(path)
        try:
            raw = open(path, 'rb').read()
        except Exception as e:
            note('ESCALATE', f'{name}: unreadable', str(e)[:60])
            continue
        if not raw:
            continue
        lines = raw.split(b'\n')
        trailing = lines[-1]
        # A file that does not end in a newline, whose last line will not parse, is a record that
        # was cut in half by a kill. Resume cannot read its id, so the item is silently re-run and
        # ends up in the file TWICE.
        broken = False
        if trailing.strip():
            try:
                json.loads(trailing.decode('utf-8', 'replace'))
            except Exception:
                broken = True
        if broken:
            age = time.time() - os.path.getmtime(path)
            if age < 60:
                note('WAIT', f'{name}: partial trailing record',
                     f'written {age:.0f}s ago — a live write, not damage; leaving alone')
            elif not fix:
                note('FAULT', f'{name}: partial trailing record', 'would truncate (--check mode)')
            else:
                shutil.copy2(path, path + '.bak')
                keep = raw[:raw.rfind(b'\n') + 1] if b'\n' in raw else b''
                with open(path, 'wb') as f:
                    f.write(keep)
                note('REPAIRED', f'{name}: truncated partial record',
                     f'backup at {os.path.basename(path)}.bak')

        # duplicate ids: resume is last-write-wins so this is not fatal, but it means an item was
        # scored twice and the n in the table is not the number of distinct items
        ids = []
        for l in raw.split(b'\n'):
            if l.strip():
                try:
                    ids.append(json.loads(l.decode('utf-8', 'replace'))['id'])
                except Exception:
                    pass
        dup = len(ids) - len(set(ids))
        if dup:
            note('FAULT', f'{name}: {dup} duplicate ids',
                 'report counts distinct ids, but a duplicate means an item ran twice')


# --------------------------------------------------------------------------------- live processes
def check_processes(fix):
    battery_live = bool(pgrep('bash scripts/eval_supervise.sh') or pgrep('bash scripts/run_evals.sh'))

    for label, pat, cmd, log in [
        ('memguard', 'bash scripts/memguard.sh', ['bash', 'scripts/memguard.sh'], '/dev/null'),
        ('watcher', 'bash scripts/eval_watch.sh', ['bash', 'scripts/eval_watch.sh'],
         os.path.join(OUT, 'watch.log')),
    ]:
        pids = pgrep(pat)
        if pids:
            note('OK', f'{label} alive', f'pid {pids[0]}')
            continue
        if label == 'watcher' and not battery_live:
            note('OK', 'watcher not running', 'no battery in flight; correct')
            continue
        if not fix:
            note('FAULT', f'{label} is not running', 'would restart (--check mode)')
        else:
            spawn(cmd, log)
            note('REPAIRED', f'{label} restarted',
                 'finished benchmarks would otherwise never land' if label == 'watcher'
                 else 'the box had no memory guard')

    ext = pgrep('bash scripts/eval_extend_all.sh')
    if battery_live and ext:
        note('OK', 'extension watcher alive', f'pid {ext[0]}')
    if battery_live and not ext:
        if not fix:
            note('FAULT', 'extension watcher not running', 'would restart (--check mode)')
        else:
            spawn(['bash', 'scripts/eval_extend_all.sh'], os.path.join(OUT, 'extend.log'))
            note('REPAIRED', 'extension watcher restarted',
                 'truncated rows would stay NOT QUOTABLE forever')

    if not battery_live:
        note('ESCALATE', 'no supervisor and no runner', 'the battery is NOT running')

    # two runners on one task interleave records into the same file
    tasks = {}
    for pid in pgrep('python3 tools/eval_suite.py') + pgrep('python3 -u tools/eval_suite.py'):
        try:
            argv = open(f'/proc/{pid}/cmdline', 'rb').read().decode('utf-8', 'replace').split('\x00')
        except Exception:
            continue
        if '--task' in argv:
            tasks.setdefault(argv[argv.index('--task') + 1], []).append(str(pid))
    for t, pids in tasks.items():
        if len(pids) > 1:
            note('ESCALATE', f'{len(pids)} runners on {t}', f'pids {",".join(pids)} — records will '
                 f'interleave; kill all but one BY PID')


# ------------------------------------------------------------------------------------ forward progress
def check_progress():
    if not os.path.exists(RUN_LOG):
        return
    idle = (time.time() - os.path.getmtime(RUN_LOG)) / 60.0
    scoring = bool(pgrep('python3 tools/eval_suite.py'))
    if scoring and idle > STALL_MIN:
        note('ESCALATE', 'log has not advanced', f'{idle:.0f} min with a runner alive — the engine '
             f'may be wedged while /health still answers')
    elif scoring:
        note('OK', 'battery is advancing', f'last write {idle:.0f} min ago')

    try:
        import urllib.request
        raw = urllib.request.urlopen('http://localhost:8080/metrics', timeout=5).read().decode()
        errs = next((float(l.split()[1]) for l in raw.splitlines()
                     if l.startswith('dsv4_errors_total')), 0)
        note('OK' if not errs else 'FAULT', 'engine error counter', f'{int(errs)} errors')
    except Exception:
        note('ESCALATE' if scoring else 'OK', 'server /metrics unreachable',
             'supervisor should restart it')


def check_disk():
    st = os.statvfs(OUT)
    free = st.f_bavail * st.f_frsize / 1e9
    note('OK' if free > MIN_FREE_GB else 'ESCALATE', 'disk free for evidence',
         f'{free:.1f} GB' + ('' if free > MIN_FREE_GB else f' — below {MIN_FREE_GB} GB, writes '
                             f'will be lost'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='report only, repair nothing')
    a = ap.parse_args()
    fix = not a.check
    print(f'=== doctor {time.strftime("%Y-%m-%dT%H:%M:%S%z")} '
          f'({"repair" if fix else "check-only"}) ===', flush=True)
    check_processes(fix)
    check_jsonl(fix)
    check_progress()
    check_disk()
    esc = sum(1 for l, _, _ in findings if l == 'ESCALATE')
    rep = sum(1 for l, _, _ in findings if l == 'REPAIRED')
    flt = sum(1 for l, _, _ in findings if l == 'FAULT')
    print(f'DOCTOR: {rep} repaired, {flt} faults, {esc} escalated', flush=True)
    return 2 if esc else 0


if __name__ == '__main__':
    sys.exit(main())
