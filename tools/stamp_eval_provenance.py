#!/usr/bin/env python3
"""stamp_eval_provenance.py — record WHICH configuration produced the eval rows that follow.

WHY. tools/eval_publish.py publishes the protocol in full -- split, scenario, extraction, scoring,
execution, temperature, top_p, effort -- because LiveCodeBench's comparability condition names those
and not the harness. It does NOT record the speculator head, the engine revision, or the draft
width. A battery that is suspended, resumed months later on a different head, and then published
therefore reads as one homogeneous run when it is not.

WHAT IS AND IS NOT COMPARABLE ACROSS A STAMP BOUNDARY:

  ACCURACY IS. The engine's acceptance rule draws every emitted token from the TARGET model's
  distribution (include/dsv4_engine.h), so which draft head is loaded does not bias the output --
  it changes only how many tokens are committed per forward. Items scored before and after a stamp
  are independent draws from the same distribution, so pooling them is unbiased.

  THROUGHPUT IS NOT. tok/s is a property of the engine, the head and the width together. Ladder 2.2
  measured the drift directly: across five decode-kernel rewrites suite tau reproduced to four
  decimal places while suite tok/s moved -2.3 % and -5.0 % and base AR moved -17.4 %. A tok/s
  column pooled across stamps is an average over configurations, not a measurement of one.
"""
import datetime, json, os, subprocess, sys

def main():
    if len(sys.argv) < 3:
        sys.exit("usage: stamp_eval_provenance.py <head> <block>")
    rev = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()[:9]
    dirty = bool(subprocess.run(["git", "status", "--porcelain"],
                                capture_output=True, text=True).stdout.strip())
    os.makedirs("evidence/evals", exist_ok=True)
    rec = {"stamped": datetime.datetime.now().astimezone().isoformat(),
           "head": sys.argv[1], "block": int(sys.argv[2]),
           "engine_rev": rev, "engine_dirty": dirty,
           "comparable": "accuracy pools across stamps (target-distribution sampling); "
                         "tok/s does NOT"}
    p = "evidence/evals/RUN_PROVENANCE.jsonl"
    with open(p, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print("[provenance] %s <- head=%s block=%s rev=%s%s"
          % (p, rec["head"], rec["block"], rev, " (dirty)" if dirty else ""))

if __name__ == "__main__":
    main()
