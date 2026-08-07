#!/usr/bin/env python3
"""flywheel_watch.py — render the headless agent's stream-json into readable, colourised dialogue.

`claude -p --output-format stream-json --verbose` emits one JSON object per line. The useful
structure is:

  {"type":"system","subtype":"init",...}         session start: model, tools, cwd
  {"type":"assistant","message":{...}}           content blocks: text | thinking | tool_use
  {"type":"user","message":{...}}                content blocks: tool_result (+ tool_use_result)
  {"type":"result","subtype":"success",...}      turns, duration, cost, token usage
  {"type":"stream_event","event":{...}}          incremental deltas (--include-partial-messages)

A caveat worth knowing before you go looking for it: **the CLI does not put thinking TEXT on the
stream.** `thinking_delta` events carry `{"thinking": "", "estimated_tokens": N}` and the completed
block carries a signature with empty content. So this renderer shows that the agent thought and
roughly how much, which is enough to see where it deliberated, but the reasoning itself is not
available to any consumer of stream-json. Do not read an empty thinking block as "it did not think".

Usage
  scripts/flywheel_watch.py                      follow the live cycle (tail -f semantics)
  scripts/flywheel_watch.py <file.jsonl>         replay a finished cycle
  scripts/flywheel_watch.py --list               list captured cycles
  scripts/flywheel_watch.py -q <file.jsonl>      quiet: tool calls and results only, no prose
  scripts/flywheel_watch.py --full <file.jsonl>  do not truncate tool output

Truncation is on by default because a single dprof or gemm_bench result is hundreds of lines and
the point of this view is the SHAPE of the iteration — what it decided, what it ran, what came
back — not the raw output, which is already in ~/ logs.
"""
import sys, os, json, time, glob, re, shutil

C = dict(rst="\033[0m", dim="\033[2m", b="\033[1m",
         red="\033[31m", grn="\033[32m", yel="\033[33m", blu="\033[34m",
         mag="\033[35m", cyn="\033[36m", gry="\033[90m")
if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
    C = {k: "" for k in C}
W = shutil.get_terminal_size((100, 40)).columns

def rule(ch="─"): return C["gry"] + ch * min(W, 100) + C["rst"]
def clip(s, n):
    s = str(s).replace("\n", "\\n")
    return s if len(s) <= n else s[:n - 1] + "…"

def body(text, colour, indent="    ", limit=None):
    """Indent a block, optionally trimming the middle — head and tail carry the information."""
    lines = text.rstrip("\n").split("\n")
    if limit and len(lines) > limit:
        head, tail = lines[: limit * 2 // 3], lines[-(limit // 3):]
        cut = len(lines) - len(head) - len(tail)
        lines = head + [f"{C['gry']}… {cut} lines elided ({len(text)} bytes total) …{C['rst']}"] + tail
    return "\n".join(indent + colour + l + C["rst"] for l in lines)

# Per-tool one-line summary of the input. The generic fallback is a truncated JSON dump.
def tool_line(name, inp):
    g = lambda k, d="": str(inp.get(k, d))
    if name == "Bash":       return clip(g("command"), W - 20)
    if name in ("Read", "Write"):  return g("file_path") + (f"  (+{len(g('content'))}B)" if name == "Write" else "")
    if name == "Edit":       return f"{g('file_path')}   {clip(g('old_string'), 40)} → {clip(g('new_string'), 40)}"
    if name == "Grep":       return f"/{g('pattern')}/  {g('path','.')}"
    if name == "Glob":       return g("pattern")
    if name == "WebFetch":   return g("url")
    if name == "WebSearch":  return g("query")
    if name == "Task":       return f"{g('subagent_type')}: {clip(g('description'), 60)}"
    return clip(json.dumps(inp), W - 20)

def render(fh, follow, quiet, full, since=0.0):
    tools, think_tok, n = {}, [0], [0]
    for line in fh:
        line = line.strip()
        if not line:
            if follow: time.sleep(0.25)
            continue
        if not line.startswith("{"):
            print(C["gry"] + clip(line, W) + C["rst"]); continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        try:
            _one(ev, tools, think_tok, n, quiet, full)
        except Exception as exc:            # a watcher that dies on one odd event is useless
            print(f"{C['red']}[watch] {type(exc).__name__}: {exc}{C['rst']}")
        if follow: sys.stdout.flush()


def _one(ev, tools, think_tok, n, quiet, full):
    t = ev.get("type")

    if t == "stream_event":
        d = (ev.get("event") or {}).get("delta") or {}
        if d.get("type") == "thinking_delta":
            # estimated_tokens is sometimes explicitly null, not absent
            think_tok[0] = max(think_tok[0], d.get("estimated_tokens") or 0)
        return

    if t == "system" and ev.get("subtype") == "init":
        print(rule("="))
        print(f"{C['b']}{C['cyn']}* cycle start{C['rst']}  model={ev.get('model')}  "
              f"cwd={ev.get('cwd')}  perm={ev.get('permissionMode')}")
        print(f"{C['gry']}  session {str(ev.get('session_id',''))[:8]}  "
              f"{len(ev.get('tools',[]))} tools{C['rst']}")
        print(rule("="))
        return

    if t == "assistant":
        for blk in (ev.get("message") or {}).get("content") or []:
            bt = blk.get("type")
            if bt == "thinking" and not quiet:
                tk = think_tok[0]; think_tok[0] = 0
                txt = blk.get("thinking") or ""
                if txt:
                    print(f"\n{C['mag']}* thinking{C['rst']}")
                    print(body(txt, C["gry"], limit=None if full else 24))
                else:
                    extra = f" (~{tk} tok, content not exposed on the stream)" if tk else ""
                    print(f"\n{C['mag']}* thinking{C['rst']}{C['gry']}{extra}{C['rst']}")
            elif bt == "text" and not quiet:
                txt = (blk.get("text") or "").strip()
                if txt:
                    print(f"\n{C['b']}> claude{C['rst']}")
                    print(body(txt, C["rst"], limit=None if full else 40))
            elif bt == "tool_use":
                n[0] += 1
                tools[blk.get("id")] = blk.get("name")
                print(f"\n{C['yel']}-> {blk.get('name')}{C['rst']}  {C['gry']}#{n[0]}{C['rst']}")
                print(f"    {C['cyn']}{tool_line(blk.get('name',''), blk.get('input') or {})}{C['rst']}")
        u = (ev.get("message") or {}).get("usage") or {}
        if (u.get("output_tokens") or 0) > 200 and not quiet:
            print(f"{C['gry']}    [out {u['output_tokens']} tok, "
                  f"cache read {u.get('cache_read_input_tokens',0)}]{C['rst']}")
        return

    if t == "user":
        content = (ev.get("message") or {}).get("content")
        for blk in content if isinstance(content, list) else []:
            if blk.get("type") != "tool_result": continue
            name = tools.get(blk.get("tool_use_id"), "?")
            c = blk.get("content")
            if isinstance(c, list):
                c = "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
            c = (c or "").rstrip()
            mark = f"{C['red']}x error{C['rst']}" if blk.get("is_error") else f"{C['grn']}ok{C['rst']}"
            print(f"  {mark} {C['gry']}{name} -> {c.count(chr(10))+1 if c else 0} lines, {len(c)}B{C['rst']}")
            if c: print(body(c, C["gry"], indent="    | ", limit=None if full else 14))
        return

    if t == "result":
        ok = not ev.get("is_error")
        print("\n" + rule("="))
        print(f"{C['b']}{C['grn'] if ok else C['red']}* cycle {'complete' if ok else 'FAILED'}"
              f"{C['rst']}  {ev.get('subtype','')}  stop={ev.get('stop_reason')}")
        u = ev.get("usage") or {}
        print(f"  turns={ev.get('num_turns')}  tools={n[0]}  "
              f"wall={(ev.get('duration_ms') or 0)/1000:.0f}s  api={(ev.get('duration_api_ms') or 0)/1000:.0f}s")
        print(f"  tokens: in {u.get('input_tokens',0)}  out {u.get('output_tokens',0)}  "
              f"cache_read {u.get('cache_read_input_tokens',0)}  "
              f"cache_write {u.get('cache_creation_input_tokens',0)}")
        if ev.get("total_cost_usd") is not None:
            print(f"  cost: ${ev['total_cost_usd']:.4f}")
        if ev.get("permission_denials"):
            print(f"  {C['red']}permission denials: {len(ev['permission_denials'])}{C['rst']}")
        print(rule("="))
        res = (ev.get("result") or "").strip()
        if res: print(body(res, C["rst"], indent="  ", limit=None if full else 30))
        return


def follow_file(path):
    """tail -f that survives the file not existing yet and being replaced between cycles."""
    print(f"{C['gry']}watching {path} — Ctrl-C to stop{C['rst']}")
    ino = None
    while True:
        try:
            st = os.stat(path)
            if st.st_ino != ino:
                ino = st.st_ino
                fh = open(path)
                render(iter(lambda: fh.readline(), None), follow=True,
                       quiet=False, full=False)
        except FileNotFoundError:
            pass
        time.sleep(1)

if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    args = [a for a in sys.argv[1:]]
    quiet = "-q" in args; full = "--full" in args
    args = [a for a in args if not a.startswith("-")]
    live = os.path.join(root, ".flywheel_last_run.jsonl")
    hist = sorted(glob.glob(os.path.join(root, ".flywheel_cycles", "*.jsonl")))

    if "--list" in sys.argv:
        for f in hist:
            try:
                last = [json.loads(l) for l in open(f) if l.startswith('{"type":"result')]
                r = last[-1] if last else {}
                print(f"{os.path.basename(f):28s} turns={r.get('num_turns','?'):>4} "
                      f"{r.get('duration_ms',0)/1000:6.0f}s ${r.get('total_cost_usd',0):.3f} "
                      f"{'ok' if not r.get('is_error') else 'ERR'}")
            except Exception as e:
                print(f"{os.path.basename(f):28s} (unreadable: {e})")
        sys.exit(0)

    if args:
        with open(args[0]) as fh: render(fh, False, quiet, full)
    elif os.path.exists(live) and not sys.stdin.isatty():
        with open(live) as fh: render(fh, False, quiet, full)
    else:
        try: follow_file(live)
        except KeyboardInterrupt: print()
