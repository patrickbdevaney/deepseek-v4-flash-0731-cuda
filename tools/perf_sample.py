#!/usr/bin/env python3
"""perf_sample.py — poll /metrics on a clock so the corpus has a time axis, not just a request axis.

WHAT THIS ADDS THAT THE PER-REQUEST RECORDS CANNOT. A benchmark record describes a request from the
inside: how long its own decode took. It cannot describe the gaps BETWEEN requests, and it cannot
report `verifies` as an integer (the server exposes only the ratio). Differencing the cumulative
counters across two samples recovers both -- exact verify counts per interval, and any wall-clock
that belongs to no request at all, which is where harness overhead, tokenisation and scoring hide.

It also gives the one thing the battery has no other way to notice: `queue_depth` over time. A
second client on the engine lock is what turned gpqa-0153 into a false negative. A record of the
gauge, sampled every 30 s for the length of the programme, makes that visible after the fact
instead of only when someone happens to be looking.

WHY THIS IS SAFE TO RUN AGAINST A SCORING ENGINE. GET /metrics is an snprintf over a handful of
std::atomic counters in the HTTP thread pool. It takes no engine lock, allocates no KV, and runs no
kernel. That is a different class of operation from the calibration probes that corrupted a GPQA
item by contending for the engine -- those called /v1/completions. This never touches a generation
endpoint, and there is no code path here that could.

The gauge is known to over-read: m_queued is incremented in a pre-routing handler and decremented
in set_post_routing_handler, which cpp-httplib does not run on every path, so it leaks. Treat it as
a CHANGE detector, not a level -- a step up of 1 that persists is a real concurrent client.

  nohup setsid python3 tools/perf_sample.py > /dev/null 2>&1 &
  python3 tools/perf_sample.py --every 30 --once      # single sample, for a smoke test
"""
import argparse, json, os, sys, time, urllib.error, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERF = os.path.join(ROOT, 'evidence', 'perf')
OUT = os.path.join(PERF, 'metrics.jsonl')

PREFIX = 'dsv4_'


def scrape(host, timeout):
    """Parse the Prometheus text body into {name: number}. Comment lines are dropped."""
    with urllib.request.urlopen(f'http://{host}/metrics', timeout=timeout) as r:
        body = r.read().decode('utf-8', 'replace')
    out = {}
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        name, _, val = line.partition(' ')
        if not name.startswith(PREFIX):
            continue
        try:
            f = float(val)
        except ValueError:
            continue
        out[name[len(PREFIX):]] = int(f) if f.is_integer() else f
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='localhost:8080')
    ap.add_argument('--every', type=float, default=30.0)
    ap.add_argument('--timeout', type=float, default=5.0)
    ap.add_argument('--once', action='store_true')
    a = ap.parse_args()

    os.makedirs(PERF, exist_ok=True)
    misses = 0
    while True:
        rec = {'ts': time.time()}
        try:
            rec.update(scrape(a.host, a.timeout))
            misses = 0
        except Exception as e:
            # A miss is data too: it dates a window in which the server was unreachable, which is
            # exactly the sort of thing that is invisible afterwards. Recorded, never retried in a
            # tight loop -- hammering an unresponsive server is how a slow engine becomes a dead one.
            rec['unreachable'] = str(e)[:120]
            misses += 1
        with open(OUT, 'a') as f:
            f.write(json.dumps(rec) + '\n')
        if a.once:
            print(json.dumps(rec, indent=2))
            return 0
        # Back off when the server is gone so a multi-hour outage does not write 100k dead rows.
        time.sleep(a.every * (min(misses, 10) or 1))


if __name__ == '__main__':
    sys.exit(main())
