# OOM and memory safety

Why this box freezes instead of OOM-killing, why the kernel cannot help, and what actually guards it.

## The measurement that explains every takedown

A 2048 MiB `cudaMalloc`, written to and synchronised, charges **42 MiB** to the calling process's
cgroup. Measured 2026-08-20 on `sm_110a` under a `systemd-run --user --scope -p MemoryAccounting=yes`
scope reading its own `memory.current` before and after.

```
cgroup memory.current  before=0.31 MiB  after=42.43 MiB  delta=42.12 MiB (allocated 2048 MiB)
```

Tegra unified memory is handed out by the driver's allocator, not the page allocator. The model's
~100.4 GiB is therefore **invisible to every kernel memory-accounting path**, and three things follow
that between them explain all four whole-machine takedowns this project has had (2026-08-12 x2,
2026-08-19, 2026-08-20):

| consequence | why |
|---|---|
| **no `oom-kill` line in dmesg, ever** | the OOM killer scores candidates by RSS/memcg. A process holding 100 GiB of unified memory scores as ~42 MiB. It is never selected, so the killer never runs — the absence of the line is not a missing log, it is the bug. |
| **the memory cannot be reclaimed** | driver-pinned: not page cache, not anonymous, not swappable. 32 GiB of swap sits at 256 KiB used while the box dies. |
| **`memory.max` cannot bound it** | a cgroup limit counts 42 MiB of a 2 GiB allocation, so the limit is never reached. cgroup-based protection here is false confidence. |

`systemd-oomd` is not an option either: **this kernel has no PSI**. `/proc/pressure/memory` does not
exist, and oomd is entirely PSI-driven.

**So polling `MemAvailable` is not the lazy defence, it is the only one.** Everything below follows
from that.

## Why the guard that existed could never have fired

A loaded server sits at `mem 119.6/122.8 GiB`. The measured healthy low-water across 13 runs is
**2389 MB**, against a kill floor of 1500 MB — **889 MB of separation**. The load ramp climbs ~100 GiB
in ~60 s = **~1710 MB/s**. So the entire window between "healthy plateau" and "dead" is

```
889 MB / 1710 MB/s = 0.52 seconds
```

The shipped guard used `POLL_S=2` x `BREACHES=3` and needed **6 seconds** to decide — about 10 GiB of
overshoot past its own floor. It was not slow, it was *arithmetically incapable* of ever firing
before the box died. Driven against a synthetic ramp it kills at **MemAvailable −5680 MB**: 5.7 GiB
past death, firing into a corpse. That is the negative control in `scripts/gate_memguard.sh`.

## The fix: slope, not just level

The floor cannot simply be raised. The healthy plateau is 2389 MB and this config's normal operating
point is under 3 GiB, so any floor high enough to catch the ramp early also kills servers that loaded
fine — which it did once already, costing a run.

The ramp and the plateau differ in a way a level test cannot see and a slope test cannot miss: a
server that has finished loading has slope ~0, while a runaway at the same `MemAvailable` is still
falling at ~1710 MB/s. Two independent triggers, plus a 10x faster poll:

| trigger | condition | default |
|---|---|---|
| LEVEL | `avail < FLOOR_MB` for `BREACHES` consecutive samples | 1500 MB x3 = 0.6 s |
| SLOPE | `avail < DANGER_MB` **and** falling faster than `RATE_MB_S` | below 2000 MB, > 400 MB/s, x2 = 0.4 s |

**`DANGER_MB` must sit below the healthy low-water, and getting this wrong kills good servers.** A
healthy ramp does not decelerate into its plateau — it falls at full rate right up to the moment
loading completes, then stops dead. So at any level *above* the plateau, "falling fast" describes the
healthy case exactly as well as the runaway and the slope test cannot separate them. It only
discriminates below where a healthy run ever reaches. Hence 2000 MB: 389 MB of margin against the
2389 MB low-water, and 1.2 s of warning on the fatal side.

Measured on the identical synthetic trace:

| settings | kills at |
|---|---|
| shipped until 2026-08-20 | **−5680 MB** (dead) |
| current | **+1502 MB**, 0.9 s of margin |

## The two guards

| | `scripts/memguard.sh` | `scripts/oomsentry.sh` |
|---|---|---|
| armed by | `run_model.sh`, per launch | systemd user unit, always |
| covers | anything launched the sanctioned way | a model started outside that path |
| floor / slope-arm | 1500 MB / 2000 MB | 1200 MB / 1800 MB |
| victim | the `comm` it was armed on | first match in a fixed allowlist |

The sentry's floors sit **below** memguard's so the armed guard always acts first and the sentry only
catches what it missed.

**The sentry can only ever kill a named model binary.** The victim must match `comm` exactly against
its `ALLOW` list. If memory is vanishing and nothing in `ALLOW` is running, it logs and does nothing —
because the alternative, picking the largest RSS, would pick a shell or an editor, and RSS is exactly
the number proved above not to reflect who is holding the memory. A process whose whole job is
`kill -9` gets the narrowest possible licence.

Both selectors use `comm`, never `pgrep -f`: Claude Code's bash wrapper embeds the text of the command
it is running into its own command line, so `pgrep -f decode` matches an interactive shell. A guard
that adopted one as its victim would eventually `kill -9` it. See
[`measurement-and-traps.md`](measurement-and-traps.md).

## Gates

| gate | proves |
|---|---|
| `scripts/gate_memguard.sh` | healthy ramp+plateau survives; runaway killed **while `MemAvailable` is still positive**; old settings kill at −5680 MB |
| sentry arms (same method) | allowlisted victim killed at +1160 MB; **non-allowlisted victim survives** |

Both drive the real polling loop — same file, same triggers, same timing — against a synthetic
`MemAvailable` trace injected via `$MEMINFO`. Testing a guard by reasoning about it is how the
6-second guard shipped.

## Not done, and why

**`vm.min_free_kbytes` is 45390 kB (44 MiB) on a 122 GiB box.** That is very little reclaim slack, and
raising it would plausibly reduce direct-reclaim livelock. It is deliberately **not** changed: raising
it lowers `MemAvailable` by the same amount, and `run_model.sh` refuses to launch below 105 GiB while
the box currently sits at ~104.6–110 GiB. The threshold is already marginal, so a speculative benefit
would be paid for with real, immediate launch failures. Fix the marginal threshold first.
