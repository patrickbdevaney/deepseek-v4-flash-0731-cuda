#!/usr/bin/env python3
"""loop_stream.py — turn `claude --output-format stream-json` into a watchable terminal feed.

WHY. The first loop iteration ran for half an hour with three lines in its transcript, because
`--output-format text` emits nothing until the run ends. An autonomous loop you cannot watch is one
you cannot stop early for the right reason -- and the first iterations tell you more about the loop
than about the work.

This reads the raw JSONL on stdin, writes it through UNCHANGED to `--raw` (so nothing is lost and
the transcript stays machine-readable), and prints one compact human line per event to stdout:

    14:02:11  think   Reading DECODE_LADDER.md to find the topmost unchecked item
    14:02:19  Read    tools/decode_model.py
    14:02:26  Bash    python3 tools/decode_model.py --dir evidence/...
    14:05:02  Edit    tools/decode_fit_probe.py
    14:31:40  DONE    rc=0  47 turns  $1.83

  claude -p ... --output-format stream-json --verbose | python3 tools/loop_stream.py --raw iterN.log
"""
import argparse, json, sys, time

def short(s, n):
    s = ' '.join(str(s).split())
    return s if len(s) <= n else s[:n-1] + '…'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', required=True, help='write the untouched JSONL here')
    a = ap.parse_args()
    raw = open(a.raw, 'a', buffering=1)
    ntool = 0
    for line in sys.stdin:
        raw.write(line)
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        ts = time.strftime('%H:%M:%S')
        t = ev.get('type')
        if t == 'assistant':
            for blk in (ev.get('message') or {}).get('content') or []:
                if blk.get('type') == 'text' and blk.get('text', '').strip():
                    print(f'{ts}  think   {short(blk["text"], 150)}', flush=True)
                elif blk.get('type') == 'tool_use':
                    ntool += 1
                    name, inp = blk.get('name', '?'), blk.get('input') or {}
                    arg = (inp.get('command') or inp.get('file_path') or inp.get('pattern')
                           or inp.get('path') or inp.get('description') or '')
                    print(f'{ts}  {name:<7} {short(arg, 150)}', flush=True)
        elif t == 'result':
            # The one line that says whether the iteration is worth anything.
            print(f'{ts}  DONE    subtype={ev.get("subtype")} turns={ev.get("num_turns")} '
                  f'tools={ntool} dur={ev.get("duration_ms",0)/1000:.0f}s '
                  f'cost=${ev.get("total_cost_usd",0):.2f} err={ev.get("is_error")}', flush=True)
    raw.close()
    return 0

if __name__ == '__main__':
    sys.exit(main())
