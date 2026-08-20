#!/usr/bin/env python3
"""loop_top.py — watch the decode loop think, live.

`loop_stream.py` writes a compact one-line-per-event feed because it sits in a PIPE and must not
slow the loop down. This is the other half: it tails the RAW JSONL the loop already keeps and
renders it for a human -- reasoning wrapped and dimmed, tool calls coloured by what they touch,
tool RESULTS (which the compact feed drops entirely), and a running header with the ladder item,
elapsed time, tool count and cost.

It follows the NEWEST iteration file and re-latches when the loop moves to the next one, so it keeps
working across iteration boundaries without a restart.

STRICTLY READ-ONLY. It tails a file. Taking it down, resizing, or attaching mid-iteration cannot
disturb the loop.

  python3 tools/loop_top.py                # follow live
  python3 tools/loop_top.py --tail 200     # replay the last 200 events first, then follow
  python3 tools/loop_top.py --once         # dump what exists and exit
"""
import argparse, glob, json, os, re, shutil, sys, time

# LINE-BUFFER STDOUT. Python block-buffers when stdout is not a tty, so piping this into `tee`, or
# redirecting it to a file, or killing it with a signal, loses everything still in the buffer -- the
# first version printed NOTHING at all under `timeout ... > file`. A live viewer that only works on
# a bare terminal is not a live viewer.
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGDIR = os.path.join(ROOT, 'evidence', 'decode_loop')
DRIVER = os.path.join(LOGDIR, 'driver.log')
NC = os.environ.get('NO_COLOR') is not None

def c(s, code): return s if NC else f'\033[{code}m{s}\033[0m'
def dim(s): return c(s, '2')
def bold(s): return c(s, '1')
def red(s): return c(s, '31')
def grn(s): return c(s, '32')
def yel(s): return c(s, '33')
def blu(s): return c(s, '34')
def mag(s): return c(s, '35')
def cyn(s): return c(s, '36')

# Colour by what the tool DOES, so the shape of an iteration is readable at a glance: yellow is
# "ran something", magenta is "changed something", cyan is "looked at something".
TOOLCOL = {'Bash': yel, 'Edit': mag, 'Write': mag, 'NotebookEdit': mag,
           'Read': cyn, 'Glob': cyn, 'Grep': cyn, 'Task': blu, 'TodoWrite': dim}

def W(): return max(60, min(shutil.get_terminal_size((120, 40)).columns, 200))

def wrap(text, indent, width):
    out, line = [], ''
    for w in text.split():
        if len(line) + len(w) + 1 > width - indent:
            out.append(line); line = w
        else:
            line = (line + ' ' + w).strip()
    if line: out.append(line)
    return out

def worklogs():
    """Every log in the dir that is NOT the LLM transcript.

    When the loop launches a background A/B and then sits in a `sleep` waiting for it, the LLM emits
    one event every ten minutes and the terminal looks frozen -- while the actual work is pouring
    into mainkv_ab.log, a server log, or a probe log. A viewer that only follows the transcript shows
    an idle model and hides the running experiment. So follow both.
    """
    out = []
    for f in glob.glob(os.path.join(LOGDIR, '*.log')):
        b = os.path.basename(f)
        if b.startswith('iter') or b in ('live.log', 'driver.log'):
            continue
        out.append(f)
    return out


def newest():
    f = sorted(glob.glob(os.path.join(LOGDIR, 'iter*.log')), key=os.path.getmtime)
    return f[-1] if f else None

def cur_item():
    try:
        for ln in reversed(open(DRIVER, errors='replace').read().splitlines()):
            m = re.search(r'next item: \d+:- \[.\] (.+)', ln)
            if m: return re.sub(r'\*\*', '', m.group(1))[:90]
    except Exception:
        pass
    return '?'

class State:
    def __init__(s):
        s.ntool = 0; s.t0 = time.time(); s.cost = 0.0
        s.last = time.time(); s.beat = 0.0; s.pos = {}

def render(ev, st, width):
    """Returns True if anything was printed. The caller uses that -- NOT the fact that a line was
    read -- to decide whether the viewer is idle. `tool_progress` heartbeats arrive every ~30s from
    a long-running Bash and carry no content; counting them as activity kept the idle timer at zero
    and produced a screen that never printed and never admitted it was waiting."""
    ts = time.strftime('%H:%M:%S')
    t = ev.get('type')
    printed = False
    if t == 'assistant':
        for b in (ev.get('message') or {}).get('content') or []:
            if b.get('type') == 'text' and b.get('text', '').strip():
                for i, ln in enumerate(wrap(b['text'].strip(), 12, width)):
                    print(f'{dim(ts)}  {mag("think") if i==0 else "     "}   {dim(ln)}', flush=True); printed = True
            elif b.get('type') == 'tool_use':
                st.ntool += 1
                name, inp = b.get('name', '?'), b.get('input') or {}
                col = TOOLCOL.get(name, grn)
                arg = (inp.get('command') or inp.get('file_path') or inp.get('pattern')
                       or inp.get('path') or inp.get('description') or '')
                arg = ' '.join(str(arg).split())
                head = f'{dim(ts)}  {col(bold(name[:7].ljust(7)))} '
                lines = wrap(arg, 12, width) or ['']
                print(head + lines[0], flush=True); printed = True
                for ln in lines[1:4]:
                    print(f'{" "*10}{" "*8}{dim(ln)}', flush=True)
                if len(lines) > 4: print(f'{" "*18}{dim("…")}', flush=True)
    elif t == 'user':
        # Tool results. The compact feed drops these entirely, and they are where a failure first
        # becomes visible -- a non-zero exit, a gate line, a traceback.
        for b in (ev.get('message') or {}).get('content') or []:
            if b.get('type') != 'tool_result': continue
            body = b.get('content')
            if isinstance(body, list):
                body = ' '.join(x.get('text', '') for x in body if isinstance(x, dict))
            body = ' '.join(str(body or '').split())
            if not body: continue
            bad = b.get('is_error') or re.search(r'\b(error|Error|FAIL|Traceback|denied|fatal)\b', body)
            tag = red('  err  ') if bad else dim('   ->  ')
            print(f'{" "*10}{tag} {(red if bad else dim)(body[:width-20])}', flush=True); printed = True
    elif t == 'tool_progress':
        # A long-running tool saying it is still alive. This is the ONLY signal during a multi-minute
        # build or A/B, so it is the thing most worth showing -- rate-limited so a 30s heartbeat does
        # not become a wall of text.
        el = ev.get('elapsed_time_seconds') or 0
        if time.time() - st.beat < 20:
            return False
        st.beat = time.time()
        print(f'{dim(ts)}  {blu("running")} '
              f'{dim(str(ev.get("tool_name") or "?") + f" — {el/60:.1f} min elapsed")}', flush=True)
        return True
    elif t == 'result':
        st.cost = ev.get('total_cost_usd', 0) or 0
        ok = not ev.get('is_error')
        box = grn if ok else red
        printed = True
        print(box('─' * width), flush=True)
        print(box(bold(f'  {"DONE" if ok else "FAILED"}  ')) +
              f'  turns {ev.get("num_turns")}   tools {st.ntool}   '
              f'{ev.get("duration_ms",0)/1000:.0f}s   ${st.cost:.2f}   subtype={ev.get("subtype")}', flush=True)
        print(box('─' * width), flush=True)
        printed = True
    return printed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tail', type=int, default=40, help='replay this many trailing events first')
    ap.add_argument('--once', action='store_true')
    a = ap.parse_args()
    width = W()
    print(bold(cyn('  decode loop — live')) + dim(f'   {LOGDIR}'))
    print(dim(f'  item: {cur_item()}'))
    print(dim('─' * width))

    path, fh, st = None, None, State()
    while True:
        n = newest()
        if n != path:
            if fh: fh.close()
            path, st = n, State()
            if not path:
                if a.once: return 0
                time.sleep(2); continue
            fh = open(path, errors='replace')
            lines = fh.readlines()                       # replay tail, then follow from the end
            for ln in lines[-a.tail:]:
                try: render(json.loads(ln), st, width)
                except Exception: pass
            st.last = time.time(); st.beat = 0.0
            print(dim(f'  ── following {os.path.basename(path)} ' + '─' * max(0, width - 24)), flush=True)
        ln = fh.readline()
        if ln:
            try:
                if render(json.loads(ln), st, width):
                    st.last = time.time()
            except Exception:
                pass
            continue

        # Nothing from the model. Drain whatever the background work has written since last look,
        # so a long wait shows the experiment progressing instead of a frozen screen.
        moved = False
        for wl in worklogs():
            try:
                sz = os.path.getsize(wl)
            except OSError:
                continue
            pos = st.pos.get(wl)
            if pos is None:
                st.pos[wl] = sz            # start at the end; do not replay history
                continue
            if sz < pos:                   # truncated or rotated
                st.pos[wl] = 0
                pos = 0
            if sz > pos:
                with open(wl, errors='replace') as w:
                    w.seek(pos)
                    chunk = w.read(sz - pos)
                st.pos[wl] = sz
                tag = os.path.basename(wl).replace('.log', '')
                for row in chunk.splitlines():
                    row = row.strip()
                    if not row:
                        continue
                    bad = re.search(r'\b(FAIL|FATAL|error|Error|Traceback|refus)\b', row)
                    col = red if bad else dim
                    print(f'{dim(time.strftime("%H:%M:%S"))}  {blu(tag[:9].ljust(9))} {col(row[:width-22])}',
                          flush=True)
                    moved = True
                st.last = time.time()

        if a.once:
            return 0
        if not moved:
            idle = time.time() - st.last
            # A heartbeat, rate-limited, so silence is visibly ALIVE rather than ambiguous. Without
            # it a ten-minute wait is indistinguishable from a dead viewer.
            if idle > 45 and time.time() - st.beat > 45:
                st.beat = time.time()
                print(dim(f'{time.strftime("%H:%M:%S")}  ...       idle {idle/60:.1f} min '
                          f'— model waiting; {st.ntool} tools this iteration'), flush=True)
        time.sleep(0.4)

if __name__ == '__main__':
    try: sys.exit(main())
    except KeyboardInterrupt: print(); sys.exit(0)
