#!/usr/bin/env python3
"""eval_drop_record.py — remove a record that measures the HARNESS rather than the model, so it re-runs.

An item that failed for an infrastructure reason is banked as `correct: false`. Resume then skips
it, and a timeout or a transient engine fault becomes a permanent wrong answer that silently
understates the model. gpqa-0153 was the first: it exceeded its 2180 s client timeout only because a
calibration probe was holding the engine lock, and would otherwise have finished comfortably.

Deleting a line from a file the battery is APPENDING TO is the dangerous part. A naive rewrite loses
any record written between the read and the write. So this is a compare-and-swap: the file is read,
the replacement is staged, and the rename happens only if size and mtime are unchanged. If the
battery wrote in that window the tool refuses and changes nothing — run it again.

A backup is always left behind. This never edits a record, only removes whole ones, so nothing it
does can alter what the model actually produced.

  python3 tools/eval_drop_record.py --task gpqa_diamond --effort low --id gpqa-0153
  python3 tools/eval_drop_record.py --task gpqa_diamond --effort low --errors   # all error records
"""
import argparse, json, os, shutil, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'evidence', 'evals')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True)
    ap.add_argument('--effort', default='low')
    ap.add_argument('--id', action='append', default=[])
    ap.add_argument('--errors', action='store_true', help='drop every record carrying an error')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    path = os.path.join(OUT, f'{a.task}.{a.effort}.jsonl')
    if not os.path.exists(path):
        sys.exit(f'no such file: {path}')

    st = os.stat(path)
    before = (st.st_size, st.st_mtime)
    keep, dropped = [], []
    for line in open(path):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            dropped.append('<unparseable>')      # a half-written record is never worth keeping
            continue
        if r['id'] in a.id or (a.errors and r.get('error')):
            dropped.append(f'{r["id"]}: {str(r.get("error") or "requested")[:70]}')
        else:
            keep.append(line if line.endswith('\n') else line + '\n')

    if not dropped:
        print('nothing to drop')
        return 0
    print(f'dropping {len(dropped)} record(s) from {os.path.basename(path)}:')
    for d in dropped:
        print('  -', d)
    if a.dry_run:
        print('dry run, nothing written')
        return 0

    shutil.copy2(path, path + '.bak')
    fd, tmp = tempfile.mkstemp(dir=OUT, prefix='.drop-')
    with os.fdopen(fd, 'w') as f:
        f.writelines(keep)

    st2 = os.stat(path)
    if (st2.st_size, st2.st_mtime) != before:
        os.unlink(tmp)
        sys.exit('the battery appended while this ran — refusing to swap, nothing changed. '
                 'Run it again.')
    os.replace(tmp, path)
    print(f'{len(keep)} records kept, backup at {os.path.basename(path)}.bak')
    print('the dropped item(s) will re-run on the next pass over this task')
    return 0


if __name__ == '__main__':
    sys.exit(main())
