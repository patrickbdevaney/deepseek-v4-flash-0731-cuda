# Eval programme SUSPENDED — 2026-08-19T21:05:40-04:00

Suspended deliberately to free the GPU for decode-kernel work. **Resume with
`bash scripts/eval_resume.sh`**, which is idempotent and continues every stage from where
it stopped. Re-enable the units too:
```
systemctl --user enable --now dsv4-evals-watchdog.timer
systemctl --user enable dsv4-evals.service
```

## What was running at the moment of the stop
```
 143571  1-07:01:54 ./build/dsv4-server --ckpt /home/patrickd/models/DeepSeek-V4-Flash-0731-REAP --host 0.0.0.0 --port 8080 --seqmax 32768 --ext-chunk 64
 146963  1-06:59:31 bash scripts/eval_extend_all.sh
1336654    12:58:01 bash scripts/eval_force_all.sh
1336665    12:58:01 bash scripts/eval_bfcl_mt_run.sh
1742825    02:42:15 python3 tools/eval_extend.py --task aime25 --effort low --budget 24000
1768339    02:02:38 bash scripts/eval_extend_retry.sh
1946527       00:04 /bin/bash -c source /home/patrickd/.claude/shell-snapshots/snapshot-bash-1787076043289-xqqsrt.sh 2>/dev/null || true && shopt -u extglob 2>/dev/null || true && { \builtin unalias -- 'unsetenv'; \builtin unset -f -- 'unsetenv'; } >/dev/null 2>&1 || true && eval 'cd /home/patrickd/deepseek-v4-flash-0731-cuda; bash scripts/eval_suspend.sh 2>&1 | tail -30' < /dev/null && pwd -P >| /tmp/claude-b73f-cwd
1946529       00:04 bash scripts/eval_suspend.sh
```

## Stage markers
| run.log            | ALL TASKS COMPLETE |
| extend.log         |  |
| extend_retry.log   |  |
| force.log          | ALL EXTENSIONS COMPLETE |
| bfcl_mt.log        |  |

## Rows as of the stop
```

task             eff      scored   acc %          95% CI  trunc  mean tok   tok/s
aime24           low       60/60    85.0    [71.7, 96.7]      9      3735    15.6
aime24        low24k       60/60    91.7   [80.0, 100.0]      6      5809    15.6
aime25           low       60/60    68.3    [51.7, 83.3]     21      4742    14.9
aime25        low24k        45/?    96.2   [88.5, 100.0]      1      4695    16.3
gpqa_diamond     low     198/198    72.7    [66.1, 78.5]     51      3699    15.0
gpqa_diamond  low24k       147/?    93.9    [88.8, 96.7]      0      2207    16.7
mmlu_pro         low     150/150    73.3    [65.7, 79.8]     18      1948    17.3
mmlu_pro      low24k       141/?    76.6    [69.0, 82.8]      2      1892    17.8
math500          low     100/100    95.0    [88.8, 97.8]      1       940    21.6
humaneval        low     164/164    95.1    [90.7, 97.5]      6      1438    20.6
bfcl             low     240/240    86.2    [81.3, 90.0]      3       216    22.8
bfcl_live        low     508/508    78.7    [75.0, 82.1]      5       247    18.5
lcb              low     175/175    46.9    [39.6, 54.2]    104      5552    12.3
scicode          low     291/291    30.2    [24.1, 36.9]     54      3534    14.7
```

## Extension progress (the resumable part)
```
  [7/18] mmlupro-4549 gold=B got=J     +2509 tok (total 10509) stop (62.9 min)
  [8/18] mmlupro-5019 gold=D got=None     +16000 tok (total 24000) TRUNC (127.7 min)
  [9/18] mmlupro-5737 gold=E got=F     +16000 tok (total 24000) TRUNC (186.5 min)
[extend 2026-08-19T10:29:05-04:00] mmlu_pro: EXTENSION FAILED — the base row stays flagged NOT QUOTABLE, which is correct
[extend 2026-08-19T10:29:05-04:00] aime24: 15.0% truncated over 60 records — continuing truncated traces to 24000
[aime24@low -> low24k] 60 base records: 51 terminated (kept as-is), 9 truncated (to continue), 0 already written
  [1/9] aime2024-2024-I-8 gold=197 got=m+n     +16000 tok (total 24000) TRUNC (61.5 min)
  [2/9] aime2024-2024-I-12 gold=385 got=3     +16000 tok (total 24000) TRUNC (121.0 min)
  [3/9] aime2024-2024-I-11 gold=371 got=371 OK  +5086 tok (total 13086) stop (139.2 min)
  [4/9] aime2024-2024-II-15 gold=315 got=315 OK  +16000 tok (total 24000) TRUNC (203.1 min)
  [5/9] aime2024-2024-II-5 gold=80 got=80 OK  +11357 tok (total 19357) stop (240.2 min)
  [6/9] aime2024-2024-I-8#r1 gold=197 got=1     +16000 tok (total 24000) TRUNC (306.3 min)
  [7/9] aime2024-2024-I-12#r1 gold=385 got=1     +16000 tok (total 24000) TRUNC (368.9 min)
  [8/9] aime2024-2024-I-11#r1 gold=371 got=371 OK  +12020 tok (total 20020) stop (413.0 min)
  [9/9] aime2024-2024-II-15#r1 gold=315 got=165     +16000 tok (total 24000) TRUNC (474.3 min)
[extend 2026-08-19T18:23:20-04:00] aime24: extension finished, landing as aime24@low24k
[extend 2026-08-19T18:23:24-04:00] aime24: extension did not land (see above)
[extend 2026-08-19T18:23:24-04:00] aime25: 35.0% truncated over 60 records — continuing truncated traces to 24000
[aime25@low -> low24k] 60 base records: 39 terminated (kept as-is), 21 truncated (to continue), 0 already written
  [1/21] aime2025-0001 gold=588 got=588 OK  +2526 tok (total 10526) stop (8.9 min)
  [2/21] aime2025-0008 gold=62 got=62 OK  +7254 tok (total 15254) stop (29.4 min)
  [3/21] aime2025-0009 gold=81 got=81 OK  +4386 tok (total 12386) stop (44.3 min)
  [4/21] aime2025-0010 gold=259 got=259 OK  +3819 tok (total 11819) stop (57.3 min)
  [5/21] aime2025-0012 gold=204 got=204 OK  +12799 tok (total 20799) stop (104.9 min)
  [6/21] aime2025-0013 gold=60 got=24     +16000 tok (total 24000) TRUNC (161.7 min)
```
