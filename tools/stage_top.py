#!/usr/bin/env python3
"""stage_top.py — `top` for the post-battery chain: extension, forcing, multi-turn.

eval_top.py watches STAGE 1, the battery, and goes blank the moment it finishes. Everything that
turns a NOT QUOTABLE row into a publishable one happens after that, in three stages that run for
days behind nohup, self-order off each other's process names, and report only into three separate
logs nobody is watching. This is the window into those three.

WHAT IT IS FOR. The failure mode of the repair chain is not a crash, it is a QUIET WRONG ANSWER:
an extension that aborts half way and leaves a partial `low24k` file, a forcing sweep that then
selects that partial file, sees 0 % truncation in it and reports "declined by the gate" for a row
that is still 26 % truncated. Both stages look green in their own log. So this does not just tail
logs -- it cross-checks the logs against the record files on disk and shouts when they disagree.

STRICTLY READ-ONLY. Reads logs, counts records, polls /metrics. It never writes to evidence/,
never touches the engine, and attaching or detaching cannot disturb a run.

  python3 tools/stage_top.py             # live, refreshes every 2s
  python3 tools/stage_top.py --once      # one frame, for pasting into a report
  python3 tools/stage_top.py --interval 5

Keys: q or Ctrl-C to quit.
"""
import argparse, os, re, shutil, sys, termios, time, tty

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_top import (read_text, metrics, mem, bold, dim, green, red, yellow, cyan, mag, bar)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'evidence', 'evals')

RUN_LOG, EXT_LOG = os.path.join(OUT, 'run.log'), os.path.join(OUT, 'extend.log')
FRC_LOG, MT_LOG = os.path.join(OUT, 'force.log'), os.path.join(OUT, 'bfcl_mt.log')

# The order eval_extend_all.sh sweeps in. Matching it means the table reads as a queue.
EXT_TASKS = ['gpqa_diamond', 'mmlu_pro', 'aime24', 'aime25', 'lcb', 'math500', 'humaneval', 'scicode']
FRC_TASKS = ['gpqa_diamond', 'mmlu_pro', 'scicode', 'lcb', 'humaneval', 'math500', 'aime24', 'aime25']
MT_CATS = ['base', 'miss_func']
EFFORT = os.environ.get('EFFORT', 'low')
EXT_TAG = EFFORT + '24k'

RE_SAY = re.compile(r'^\[(?:extend|force|bfcl-mt) (\S+)\] (.*)$')
RE_EXT_HEAD = re.compile(r'^\[(\w+)@(\S+) -> (\S+)\] (\d+) base records: (\d+) terminated'
                         r'.*?(\d+) truncated.*?(\d+) already written')
RE_EXT_ITEM = re.compile(r'^\s*\[(\d+)/(\d+)\] (\S+) gold=(\S+) got=(\S+)\s+(OK)?\s*'
                         r'\+(\d+) tok \(total (\d+)\) (TRUNC|stop) \(([\d.]+) min\)')
RE_MT_ITEM = re.compile(r'^\s*\[(\d+)/(\d+)\] (\S+)\s+(OK)?\s*turns=(\d+) steps=\s*(\d+) '
                        r'tok=\s*(\d+) run=(\d+) \(([\d.]+) min\)')
RE_MT_DEAD = re.compile(r'^\s*\[(\d+)/(\d+)\] (\S+)\s+DEAD')


def nrec(name):
    """Line count of a record file without parsing it -- these reach 40 MB."""
    p = os.path.join(OUT, name)
    try:
        with open(p, 'rb') as f:
            return sum(1 for ln in f if ln.strip())
    except OSError:
        return 0


def procs():
    """Which stage scripts and workers are alive, read straight from /proc.

    Parentage is not checked here -- detach_audit.sh owns that question. This only answers
    'is it running', which is what decides whether a silent log means waiting or dead.
    """
    alive = set()
    for pid in os.listdir('/proc'):
        if not pid.isdigit():
            continue
        try:
            with open(f'/proc/{pid}/cmdline', 'rb') as f:
                cmd = f.read().replace(b'\x00', b' ').decode('utf-8', 'replace')
        except OSError:
            continue
        for key in ('eval_supervise.sh', 'run_evals.sh', 'eval_extend_all.sh', 'eval_force_all.sh',
                    'eval_bfcl_mt_run.sh', 'eval_extend.py', 'eval_force.py', 'eval_bfcl_mt.py',
                    'eval_suite.py'):
            if key in cmd:
                alive.add(key)
    return alive


def parse_extend(txt):
    """Per-task decisions and the live item feed for the extension pass."""
    st = dict(tasks={}, cur=None, head=None, items=[], done=False, stopped=None, started=None)
    for line in txt.splitlines():
        m = RE_EXT_HEAD.match(line)
        if m:
            st['cur'] = m.group(1)
            st['items'] = []
            st['head'] = dict(base=int(m.group(4)), term=int(m.group(5)),
                              trunc=int(m.group(6)), written=int(m.group(7)), tag=m.group(3))
            continue
        m = RE_EXT_ITEM.match(line)
        if m:
            st['items'].append(dict(i=int(m.group(1)), n=int(m.group(2)), id=m.group(3),
                                    gold=m.group(4), got=m.group(5), ok=bool(m.group(6)),
                                    add=int(m.group(7)), tot=int(m.group(8)),
                                    fin=m.group(9), mins=float(m.group(10))))
            continue
        m = RE_SAY.match(line)
        if not m:
            continue
        ts, msg = m.group(1), m.group(2)
        if 'ALL EXTENSIONS COMPLETE' in msg:
            st['done'], st['cur'] = True, None
            continue
        if msg.startswith('battery complete'):
            st['started'] = ts
            continue
        if 'stopping rather than extending' in msg:
            st['stopped'] = ts
            continue
        t = msg.split(':', 1)[0]
        if t not in EXT_TASKS:
            continue
        r = st['tasks'].setdefault(t, {})
        mm = re.search(r'([\d.]+)% truncated', msg)
        if mm:
            r['trunc'] = float(mm.group(1))
        mm = re.search(r'over (\d+) records', msg)
        if mm:
            r['n'] = int(mm.group(1))
        if 'continuing truncated traces' in msg:
            r['decision'], r['at'] = 'extend', ts
        elif 'no extension needed' in msg:
            r['decision'], r['result'], r['at'] = 'under gate', 'skipped', ts
        elif 'no records, skipping' in msg:
            r['decision'], r['result'] = 'no records', 'skipped'
        elif 'extension finished' in msg:
            r['result'], r['at'] = 'landed', ts
            st['cur'] = None
        elif 'EXTENSION FAILED' in msg:
            r['result'], r['at'] = 'FAILED', ts
            st['cur'] = None
        elif 'did not land' in msg:
            r['result'] = 'DID NOT LAND'
        elif 'not extending' in msg:
            r['result'] = 'incomplete'
        elif 'topped up to' in msg:
            r['topped'] = True
    return st


def parse_force(txt):
    st = dict(tasks={}, done=False, sweeping=False, refused=None, none_needed=False)
    for line in txt.splitlines():
        m = RE_SAY.match(line)
        if not m:
            continue
        ts, msg = m.group(1), m.group(2)
        if 'ALL FORCING COMPLETE' in msg:
            st['done'] = True
        elif msg.startswith('extension complete'):
            st['sweeping'] = True
        elif msg.startswith('Refusing.'):
            st['refused'] = ts
        elif 'no row needed forcing' in msg:
            st['none_needed'] = True
        t = msg.split(':', 1)[0]
        if t not in FRC_TASKS:
            continue
        r = st['tasks'].setdefault(t, {})
        mm = re.search(r'considering (\S+) -> (\S+)', msg)
        if mm:
            r['from'], r['tag'], r['state'], r['at'] = mm.group(1), mm.group(2), 'considering', ts
        elif 'forced, landing as' in msg:
            r['state'], r['at'] = 'forced', ts
        elif 'declined by the gate' in msg:
            r['state'], r['at'] = 'declined', ts
        elif 'FORCING FAILED' in msg:
            r['state'], r['at'] = 'FAILED', ts
        elif 'did not land' in msg:
            r['state'] = 'DID NOT LAND'
        elif 'skipping' in msg:
            r['state'] = 'skipped'
    return st


def parse_mt(txt):
    st = dict(cats={}, cur=None, items=[], done=False, failed=None, phase=None)
    for line in txt.splitlines():
        for rx, dead in ((RE_MT_ITEM, False), (RE_MT_DEAD, True)):
            m = rx.match(line)
            if m:
                it = dict(i=int(m.group(1)), n=int(m.group(2)), id=m.group(3), dead=dead)
                if not dead:
                    it.update(ok=bool(m.group(4)), turns=int(m.group(5)), steps=int(m.group(6)),
                              tok=int(m.group(7)), nok=int(m.group(8)), mins=float(m.group(9)))
                st['items'].append(it)
                break
        m = RE_SAY.match(line)
        if not m:
            continue
        ts, msg = m.group(1), m.group(2)
        if 'ALL MULTI-TURN COMPLETE' in msg:
            st['done'], st['cur'] = True, None
        elif 'SELF-GATE FAILED' in msg or 'SMOKE FAILED' in msg:
            st['failed'] = msg
        elif msg.startswith('engine is free'):
            st['phase'] = 'engine free'
        cat = msg.split(':', 1)[0]
        if cat not in MT_CATS:
            continue
        r = st['cats'].setdefault(cat, {})
        if 'smoke run' in msg:
            r['state'], st['cur'], st['items'] = 'smoke', cat, []
        elif 'full run' in msg:
            r['state'], st['cur'], st['items'], r['at'] = 'running', cat, [], ts
        elif 'exited non-zero' in msg:
            r['state'] = 'ABORTED'
        elif 'did not land' in msg:
            r['state'] = 'DID NOT LAND'
    return st


def faults(ext, frc, up, e_run, f_run):
    """Cross-check the logs against the files. This is the part a `tail` cannot do.

    THE ONE THAT MATTERS. eval_force_all.sh selects the extended file with a bare `[ -s ]` test, so
    an extension that ABORTED still leaves a non-empty low24k holding only the traces that had
    already terminated. Forcing then reads 0 % truncation out of it, declines, and prints "declined
    by the gate" -- and the row it was supposed to rescue stays NOT QUOTABLE with both logs green.
    """
    out = []
    if not up:
        out.append('the engine is DOWN — every stage below is either blocked or banking failures')
    for t in EXT_TASKS:
        # A HALF-WRITTEN FILE IS NOT A FAULT WHILE IT IS BEING WRITTEN. eval_extend.py copies the
        # terminated traces first and continues the truncated ones one at a time, so the task in
        # flight is partial by construction and missing its meta until the last line lands. Faulting
        # on that would cry wolf for hours of every extension and train the panel to be ignored.
        if t == ext['cur'] and e_run:
            continue
        base, e = nrec(f'{t}.{EFFORT}.jsonl'), nrec(f'{t}.{EXT_TAG}.jsonl')
        if not e:
            continue
        res = (ext['tasks'].get(t) or {}).get('result')
        if e < base and res != 'landed':
            out.append(f'{t}: {EXT_TAG} is PARTIAL ({e}/{base} records, extension {res or "abandoned"}). '
                       f'eval_force_all.sh selects it on a bare [ -s ] test, will read 0% truncation '
                       f'and DECLINE — the row stays NOT QUOTABLE and force.log says "declined by the gate".')
        elif not os.path.exists(os.path.join(OUT, f'{t}.{EXT_TAG}.meta.json')):
            out.append(f'{t}: {EXT_TAG}.jsonl exists with no .meta.json — eval_force.tag_for() will '
                       f'fall back to defaults and derive the wrong forced tag.')
    # ONLY IF IT IS STILL THE LAST WORD. Both of these are recorded permanently in a log that a
    # resume appends to, so a refusal from two days ago that eval_resume.sh already cleared must not
    # keep showing as a live fault -- a panel that is always red is a panel nobody reads.
    if ext['stopped'] and not (ext['started'] and ext['started'] > ext['stopped']):
        out.append(f'the extension stopped at {ext["stopped"]} rather than extend a partial battery')
    if frc['refused'] and not (frc['sweeping'] or f_run):
        out.append(f'forcing REFUSED at {frc["refused"]} — it saw no ALL EXTENSIONS COMPLETE marker')
    return out


def stage_line(name, state, detail):
    colour = dict(complete=green, running=cyan, waiting=dim, FAILED=red,
                  blocked=yellow, dead=red)[state]
    return f'  {colour(f"{state:<9}")} {bold(f"{name:<12}")} {dim(detail)}'


def frame():
    W = max(76, min(shutil.get_terminal_size((110, 44)).columns, 170))
    H = shutil.get_terminal_size((110, 44)).lines
    alive = procs()
    ext = parse_extend(read_text(EXT_LOG))
    frc = parse_force(read_text(FRC_LOG))
    mt = parse_mt(read_text(MT_LOG))
    mx = metrics()
    avail, _ = mem()
    up = mx.get('dsv4_requests_total') is not None
    battery_done = 'ALL TASKS COMPLETE' in read_text(RUN_LOG, 400_000)
    L = []

    L.append(bold(cyan('  DSV4-FLASH-0731-REAP  repair chain')) +
             dim(f'   extend → force → multi-turn   {time.strftime("%H:%M:%S")}'))
    L.append('  ' + (green('● server up') if up else red('● SERVER DOWN')) +
             dim(' │ ') + f'{mx.get("dsv4_decode_tokens_per_second", 0):5.2f} tok/s' +
             dim(' │ ') + f'queue {int(mx.get("dsv4_queue_depth", 0))}' +
             dim(' │ ') + (green('0 errors') if not int(mx.get('dsv4_errors_total', 0))
                           else red(f'{int(mx["dsv4_errors_total"])} ERRORS')) +
             dim(' │ ') + f'{mx.get("dsv4_completion_tokens_total", 0)/1e6:.2f}M tok' +
             dim(' │ ') + f'{avail:.1f} GiB free')

    # ---- the ladder ---------------------------------------------------------------------------
    L.append('')
    b_run = 'eval_supervise.sh' in alive or 'run_evals.sh' in alive
    L.append(stage_line('1 battery', 'complete' if battery_done else ('running' if b_run else 'dead'),
                        'ALL TASKS COMPLETE' if battery_done else 'see eval_top.py'))

    e_run = 'eval_extend_all.sh' in alive
    e_state = ('complete' if ext['done'] else 'FAILED' if ext['stopped'] and not e_run
               else 'dead' if not e_run else 'running' if ext['cur'] or ext['started'] else 'waiting')
    n_done = sum(1 for t in EXT_TASKS if (ext['tasks'].get(t) or {}).get('result'))
    L.append(stage_line('2 extend', e_state,
                        f'{n_done}/{len(EXT_TASKS)} tasks decided' +
                        (f'  ·  {ext["cur"]} in flight' if ext['cur'] else '')))

    f_run = 'eval_force_all.sh' in alive
    f_state = ('complete' if frc['done'] else 'FAILED' if frc['refused'] and not f_run
               else 'dead' if not f_run else 'running' if frc['sweeping'] else 'waiting')
    L.append(stage_line('3 force', f_state,
                        'sweeping rows still over the 5% gate' if frc['sweeping']
                        else 'blocked on the extension (correct — one client per engine)'))

    m_run = 'eval_bfcl_mt_run.sh' in alive
    m_state = ('complete' if mt['done'] else 'FAILED' if mt['failed']
               else 'dead' if not m_run else 'running' if mt['cur'] else 'waiting')
    L.append(stage_line('4 multiturn', m_state,
                        mt['failed'] or (f'{mt["cur"]} in flight' if mt['cur']
                                         else 'blocked on forcing')))

    # ---- faults -------------------------------------------------------------------------------
    fl = faults(ext, frc, up, e_run, f_run)
    if fl:
        L.append('')
        L.append('  ' + red(bold('── faults ')) + dim('─' * (W - 14)))
        for f in fl:
            words, line = f.split(), '   '
            for w in words:
                if len(line) + len(w) > W - 6:
                    L.append('  ' + red(line))
                    line = '     '
                line += w + ' '
            L.append('  ' + red(line))

    # ---- extension ----------------------------------------------------------------------------
    L.append('')
    L.append('  ' + dim('── extension  ') + dim(f'base → {EXT_TAG}  ' + '─' * (W - 30)))
    L.append('  ' + dim(f'  {"task":<15}{"base":>6}{"trunc":>8}  {"decision":<12}'
                        f'{"result":<14}{EXT_TAG:>7}'))
    for t in EXT_TASKS:
        r = ext['tasks'].get(t) or {}
        base, e = nrec(f'{t}.{EFFORT}.jsonl'), nrec(f'{t}.{EXT_TAG}.jsonl')
        res = r.get('result') or ('running' if t == ext['cur'] else '')
        col = (green if res == 'landed' else red if res in ('FAILED', 'DID NOT LAND')
               else cyan if res == 'running' else dim if res == 'skipped' else yellow)
        tr = f'{r["trunc"]:.1f}%' if 'trunc' in r else '-'
        mark = ' ' if e in (0, base) or res == 'landed' else red('!')
        L.append(f'  {mark} {t:<15}{base:>6}{tr:>8}  {r.get("decision", "-"):<12}'
                 + col(f'{res or "queued":<14}') + f'{e or "-":>7}')

    # ---- the live feed ------------------------------------------------------------------------
    if ext['cur'] and ext['items']:
        h = ext['head'] or {}
        it = ext['items'][-1]
        frac = it['i'] / it['n']
        eta = ''
        if len(ext['items']) >= 2:
            per = (it['mins'] - ext['items'][0]['mins']) / max(1, len(ext['items']) - 1)
            eta = dim(f'  eta {max(0.0, (it["n"] - it["i"]) * per) / 60:.1f}h ({per:.1f} min/item)')
        L.append('')
        L.append('  ' + bold(mag(ext['cur'])) + dim(f'   {h.get("term", "?")} terminated kept, '
                                                    f'{h.get("trunc", "?")} truncated to continue'))
        L.append('  ' + bar(frac, W - 40) + f'  {it["i"]}/{it["n"]}' + eta)
        room = max(3, H - len(L) - len(FRC_TASKS) - 12)
        for it in ext['items'][-room:]:
            mark = green('✔') if it['ok'] else red('✘')
            tr = red(' still TRUNC') if it['fin'] == 'TRUNC' else ''
            L.append(f'  {mark} {it["id"]:<20}' + dim('gold ') + f'{it["gold"]:<4}' +
                     dim('got ') + f'{it["got"]:<8}' + dim(f'+{it["add"]:>5} tok → {it["tot"]:>6}') +
                     tr + dim(f'   {it["mins"]:.0f}m'))

    # ---- forcing ------------------------------------------------------------------------------
    if frc['sweeping'] or frc['done']:
        L.append('')
        L.append('  ' + dim('── forcing  ') + dim('─' * (W - 16)))
        if frc['none_needed']:
            L.append('  ' + green('  no row needed forcing — the extension repaired everything'))
        for t in FRC_TASKS:
            r = frc['tasks'].get(t)
            if not r:
                continue
            s = r.get('state', '')
            col = (green if s == 'forced' else red if 'FAIL' in s or 'NOT LAND' in s
                   else dim if s == 'declined' else cyan)
            L.append(f'    {t:<15}' + col(f'{s:<14}') +
                     dim(f'{r.get("from", "")} → {r.get("tag", "")}'))

    # ---- multi-turn ---------------------------------------------------------------------------
    if mt['cur'] or mt['done'] or mt['cats']:
        L.append('')
        L.append('  ' + dim('── bfcl multi-turn  ') + dim('─' * (W - 24)))
        for cat in MT_CATS:
            r = mt['cats'].get(cat) or {}
            s = r.get('state', 'queued')
            col = green if mt['done'] else red if 'ABORT' in s or 'NOT LAND' in s else cyan
            det = ''
            if cat == mt['cur'] and mt['items']:
                it = mt['items'][-1]
                det = (f'{it["i"]}/{it["n"]}  ' +
                       (f'{it.get("nok", 0)} correct' if not it['dead'] else red('DEAD')))
            L.append(f'    {cat:<15}' + col(f'{s:<12}') + dim(det))
        if mt['cur'] and mt['items']:
            room = max(2, H - len(L) - 3)
            for it in mt['items'][-room:]:
                if it['dead']:
                    L.append(f'  {red("✘")} {it["id"]:<30}' + red('DEAD — no tokens generated'))
                else:
                    L.append(f'  {green("✔") if it["ok"] else red("✘")} {it["id"]:<30}' +
                             dim(f'turns {it["turns"]}  steps {it["steps"]:>2}  '
                                 f'{it["tok"]:>5} tok   {it["nok"]} correct   {it["mins"]:.0f}m'))

    for log, st in ((EXT_LOG, e_state), (FRC_LOG, f_state), (MT_LOG, m_state)):
        if st in ('waiting', 'running'):
            last = [l for l in read_text(log, 8000).splitlines() if l.strip()]
            if last:
                L.append('')
                L.append('  ' + dim(last[-1][:W - 4]))
            break
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--interval', type=float, default=2.0)
    ap.add_argument('--once', action='store_true')
    a = ap.parse_args()
    if a.once:
        print('\n'.join(frame()))
        return 0
    fd = sys.stdin.fileno() if sys.stdin.isatty() else None
    old = termios.tcgetattr(fd) if fd is not None else None
    try:
        if fd is not None:
            tty.setcbreak(fd)
        sys.stdout.write('\033[?25l')
        while True:
            sys.stdout.write('\033[H\033[J' + '\n'.join(frame()) + '\n')
            sys.stdout.flush()
            if fd is not None:
                import select
                if select.select([sys.stdin], [], [], a.interval)[0]:
                    if sys.stdin.read(1).lower() == 'q':
                        break
            else:
                time.sleep(a.interval)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write('\033[?25h\033[0m\n')
        if fd is not None and old:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return 0


if __name__ == '__main__':
    sys.exit(main())
