#!/usr/bin/env python3
"""eval_top.py — `top` for the eval battery. Watch every benchmark score itself, live.

The battery runs for days behind nohup and the only window into it was `tail` on a log that
contains NUL bytes. This is the window: what is running now, how each item scored the moment it
lands, whether the number being accumulated is publishable, and whether the whole plan still fits.

STRICTLY READ-ONLY. It tails logs, reads the record files, and polls /metrics. It never writes to
evidence/, never touches the battery, and taking it down or resizing the terminal cannot disturb a
run. Safe to attach and detach at will.

  python3 tools/eval_top.py               # live, refreshes every 2s
  python3 tools/eval_top.py --once        # one frame, for scripts and for pasting into a report
  python3 tools/eval_top.py --interval 5

Keys: q or Ctrl-C to quit.
"""
import argparse, json, math, os, re, shutil, sys, termios, time, tty, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'evidence', 'evals')
RUN_LOG = os.path.join(OUT, 'run.log')
WATCH_LOG = os.path.join(OUT, 'watch.log')
SUP_LOG = os.path.join(OUT, 'supervise.log')

PLAN = ['gpqa_diamond', 'bfcl', 'bfcl_live', 'scicode', 'mmlu_pro', 'humaneval',
        'math500', 'lcb', 'aime24', 'aime25']
TRUNC_LIMIT = 0.05           # matches eval_publish's NOT QUOTABLE gate

NC = os.environ.get('NO_COLOR') is not None


def c(s, code):
    return s if NC else f'\033[{code}m{s}\033[0m'


def dim(s):    return c(s, '2')
def bold(s):   return c(s, '1')
def green(s):  return c(s, '32')
def red(s):    return c(s, '31')
def yellow(s): return c(s, '33')
def cyan(s):   return c(s, '36')
def mag(s):    return c(s, '35')


def vislen(s):
    return len(re.sub(r'\033\[[0-9;]*m', '', s))


def wilson(k, n, z=1.96):
    if not n:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * max(0, ctr - h), 100 * min(1, ctr + h)


def read_text(path, tail_bytes=200_000):
    """Logs carry NUL bytes from interleaved writers; strip them rather than crash on them."""
    try:
        with open(path, 'rb') as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - tail_bytes))
            return f.read().replace(b'\x00', b'').decode('utf-8', 'replace')
    except Exception:
        return ''


def metrics():
    out = {}
    try:
        raw = urllib.request.urlopen('http://localhost:8080/metrics', timeout=2).read().decode()
        for line in raw.splitlines():
            if line.startswith('dsv4_') and ' ' in line:
                k, v = line.split(None, 1)
                try:
                    out[k] = float(v)
                except ValueError:
                    pass
    except Exception:
        pass
    return out


def mem():
    try:
        info = {}
        for line in open('/proc/meminfo'):
            k, v = line.split(':', 1)
            info[k] = int(v.split()[0]) / 1048576.0        # GiB
        return info.get('MemAvailable', 0), info.get('MemTotal', 0)
    except Exception:
        return 0, 0


_cache = {}


def records(task, effort):
    """Scored records for one task, cached on (size, mtime) so a 40 MB jsonl is not re-parsed at 2 Hz."""
    p = os.path.join(OUT, f'{task}.{effort}.jsonl')
    try:
        st = os.stat(p)
    except OSError:
        return []
    # Keyed PER FILE, not a single slot. The plan panel asks for all ten tasks every frame, so a
    # one-entry cache is a cache that always misses -- it would re-parse every jsonl in the battery
    # twice a second, and the files only grow.
    key = (st.st_size, st.st_mtime)
    hit = _cache.get(p)
    if hit and hit[0] == key:
        return hit[1]
    recs = []
    for line in read_text(p, 40_000_000).splitlines():
        line = line.strip()
        if line:
            try:
                recs.append(json.loads(line))
            except Exception:
                pass
    _cache[p] = (key, recs)
    return recs


def parse_run_log(txt):
    """Current task, its start banner, and the per-item result stream."""
    cur, todo, total, maxtok = None, None, None, None
    items = []
    for line in txt.splitlines():
        m = re.match(r'^===\s+\S+\s+(\w+)\s+n=', line)
        if m:
            # Only clear the stream when the TASK changes. A supervisor restart re-emits this banner
            # for the task already in flight, and clearing on the banner left the feed blank for the
            # ten-plus minutes until the next item finished -- which reads as "nothing is happening"
            # at exactly the moment someone is checking whether the restart worked.
            if m.group(1) != cur:
                items = []
            cur = m.group(1)
            continue
        m = re.match(r'^===\s+\S+\s+(\w+)\s+done', line)
        if m:
            cur = None
            continue
        m = re.match(r'^\[(\w+)\]\s+(\d+) items .*?(\d+) already done, (\d+) to go, '
                     r'max_tokens=(\d+)', line)
        if m:
            cur, total, todo, maxtok = m.group(1), int(m.group(2)), int(m.group(4)), int(m.group(5))
            continue
        m = re.match(r'^\s*\[(\d+)/(\d+)\]\s+(\S+)\s+gold=(\S+)\s+got=(\S+)\s+(OK)?\s*'
                     r'run=(\d+)/(\d+)\s+(\S+)\s+([\d.]+) tok/s\s+\(([\d.]+) min\)', line)
        if m:
            items.append(dict(i=int(m.group(1)), n=int(m.group(2)), id=m.group(3),
                              gold=m.group(4), got=m.group(5), ok=bool(m.group(6)),
                              nok=int(m.group(7)), seen=int(m.group(8)), finish=m.group(9),
                              tps=float(m.group(10)), mins=float(m.group(11))))
    return cur, todo, total, maxtok, items


def landed():
    """Which tasks the watcher has verified, committed and pushed."""
    out = {}
    for line in read_text(WATCH_LOG).splitlines():
        m = re.match(r'^\[land\] (\w+)@(\S+): (.+)$', line)
        if m:
            out[m.group(1)] = m.group(3)
    return out


def bar(frac, width, fill='━'):
    frac = max(0.0, min(1.0, frac))
    k = int(round(frac * width))
    return green(fill * k) + dim('┄' * (width - k))


def frame(effort):
    W = max(72, min(shutil.get_terminal_size((100, 40)).columns, 160))
    H = shutil.get_terminal_size((100, 40)).lines
    txt = read_text(RUN_LOG)
    cur, todo, total, maxtok, items = parse_run_log(txt)
    mx = metrics()
    avail, tot = mem()
    land = landed()
    L = []

    up = mx.get('dsv4_requests_total') is not None
    tps = mx.get('dsv4_decode_tokens_per_second', 0)
    qd = int(mx.get('dsv4_queue_depth', 0))
    errs = int(mx.get('dsv4_errors_total', 0))
    gen = mx.get('dsv4_completion_tokens_total', 0)

    L.append(bold(cyan('  DSV4-FLASH-0731-REAP  eval battery')) + dim(f'   {time.strftime("%H:%M:%S")}'))
    L.append('  ' + (green('● server up') if up else red('● SERVER DOWN')) +
             dim(' │ ') + f'{tps:5.2f} tok/s' +
             dim(' │ ') + f'queue {qd}' +
             dim(' │ ') + (green(f'{errs} errors') if not errs else red(f'{errs} ERRORS')) +
             dim(' │ ') + f'{gen/1e6:.2f}M tok generated' +
             dim(' │ ') + (f'{avail:.1f}/{tot:.0f} GiB free' if avail < 4 and False else
                           f'{avail:.1f} GiB free'))

    # ---- current task -------------------------------------------------------------------------
    L.append('')
    if cur:
        recs = records(cur, effort)
        n = len(recs)
        k = sum(1 for r in recs if r.get('correct'))
        tr = sum(1 for r in recs if r.get('truncated'))
        acc = 100.0 * k / n if n else 0.0
        lo, hi = wilson(k, n)
        rate = tr / n if n else 0.0
        done_frac = n / total if total else 0.0
        L.append('  ' + bold(mag(cur)) + dim(f'  @{effort}  max_tokens={maxtok}'))
        # ETA from the observed pace of THIS task's items, not from a nominal rate: the whole point
        # of the projection is that per-item cost varies by benchmark and by truncation.
        eta = ''
        if len(items) >= 2 and total:
            span = items[-1]['mins'] - items[0]['mins']
            per = span / max(1, len(items) - 1)
            left = (total - n) * per
            if left > 0:
                eta = dim(f'  eta {left/60:.1f}h' + (f' ({per:.1f} min/item)' if per else ''))
        L.append('  ' + bar(done_frac, W - 40) + f'  {n}/{total or "?"}' +
                 dim(f'  ({done_frac*100:.0f}%)') + eta)
        accs = f'{acc:5.1f}%'
        line = ('  ' + bold('acc ') + (green(accs) if rate <= TRUNC_LIMIT else yellow(accs)) +
                dim(f'  [{lo:.1f}, {hi:.1f}]') + dim(' │ ') +
                f'{k} correct')
        if tr:
            tag = f'{tr} truncated ({rate:.0%})'
            line += dim(' │ ') + (red(tag) if rate > TRUNC_LIMIT else yellow(tag))
        L.append(line)
        if rate > TRUNC_LIMIT:
            nt = [r for r in recs if not r.get('truncated')]
            ntacc = 100.0 * sum(1 for r in nt if r.get('correct')) / len(nt) if nt else 0
            L.append('  ' + red('NOT QUOTABLE') +
                     dim(f' — measures the budget, not the model. Terminating traces: '
                         f'{ntacc:.1f}% over {len(nt)}. Needs eval_extend.py'))
    else:
        L.append('  ' + dim('no task scoring right now (preflight, landing, or between tasks)'))

    # ---- the live stream ----------------------------------------------------------------------
    L.append('')
    L.append('  ' + dim('── live ' + '─' * (W - 12)))
    room = max(4, H - len(L) - len(PLAN) - 8)
    for it in items[-room:]:
        mark = green('✔') if it['ok'] else (dim('·') if it['got'] == 'None' else red('✘'))
        got = it['got'] if it['got'] != 'None' else dim('none')
        trunc = red(' TRUNC') if it['finish'] == 'length' else ''
        L.append(f'  {mark} ' + f'{it["id"]:<22}' +
                 dim('gold ') + f'{it["gold"]:<4}' + dim('got ') + f'{got:<10}' +
                 dim(f'{it["tps"]:5.1f} tok/s') + trunc +
                 dim(f'   {it["nok"]}/{it["seen"]}  {it["mins"]:.0f}m'))

    # ---- the plan -----------------------------------------------------------------------------
    L.append('')
    L.append('  ' + dim('── plan ' + '─' * (W - 12)))
    for t in PLAN:
        recs = records(t, effort)
        if t in land:
            state, detail = green('landed  '), dim(land[t])
        elif t == cur:
            state, detail = cyan('running '), dim(f'{len(recs)} scored')
        elif recs:
            state, detail = yellow('partial '), dim(f'{len(recs)} scored')
        else:
            state, detail = dim('queued  '), ''
        L.append(f'  {state} {t:<15}{detail}')

    sup = [l for l in read_text(SUP_LOG, 8000).splitlines() if l.strip()]
    if sup:
        L.append('')
        L.append('  ' + dim(sup[-1][:W - 4]))
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--interval', type=float, default=2.0)
    ap.add_argument('--effort', default=os.environ.get('EFFORT', 'low'))
    ap.add_argument('--once', action='store_true')
    a = ap.parse_args()

    if a.once:
        print('\n'.join(frame(a.effort)))
        return 0

    fd = sys.stdin.fileno() if sys.stdin.isatty() else None
    old = termios.tcgetattr(fd) if fd is not None else None
    try:
        if fd is not None:
            tty.setcbreak(fd)
        sys.stdout.write('\033[?25l')                      # hide cursor
        while True:
            lines = frame(a.effort)
            sys.stdout.write('\033[H\033[J' + '\n'.join(lines) + '\n')
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
        sys.stdout.write('\033[?25h\033[0m\n')             # restore cursor
        if fd is not None and old:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return 0


if __name__ == '__main__':
    sys.exit(main())
