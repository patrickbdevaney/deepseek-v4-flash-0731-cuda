# Draft-head registry

Every candidate head, its measurement, and whether it was promoted.
Written by `tools/promote_head.py`, which REFUSES to promote a head that cannot pass the gates in
its docstring: passing LOSSLESS gate, clean run, the frozen 8-prompt protocol at NGEN0>=200 and
block 6, and a suite mean beating the incumbent by more than the measured 3.5% run-to-run spread.
**Ties go to the incumbent.**

The product of this project is two things: the CUDA engine (in git, therefore safe) and the
speculator weights (not in git, therefore at risk). This registry plus
`~/model-backups/heads/<name>/` is how the second one stops being at risk.

| name | suite tau | suite tok/s | base AR | engine rev | status |
|---|---|---|---|---|---|
| `shipped-dspark-0731reap` | 3.5362 | 22.6550 | 13.76 | `2632540` | baseline |
| `s1` | 3.5762 | 24.515 | 13.8 | `06762c117` | PROMOTED |
| `s2` | 3.6275 | 24.7575 | 13.83 | `d2a62fd1e` | not promoted: suite 24.76 tok/s does not beat incumbent 24.52 by the 3.5%  |
| `s2-abl-ce0.5_tv0.5` | 3.6412 | 24.8175 | 13.76 | `e2a6cb47a` | not promoted: suite 24.82 tok/s does not beat incumbent 24.52 by the 3.5%  |
| `s2-abl-ce0.9_tv0.1` | 3.64 | 24.8975 | 14.14 | `e2a6cb47a` | not promoted: suite 24.90 tok/s does not beat incumbent 24.52 by the 3.5%  |
| `s2-abl-ce1.0_tv0.0` | 3.6712 | 25.1038 | 13.7 | `e2a6cb47a` | not promoted: suite 25.10 tok/s does not beat incumbent 24.52 by the 3.5%  |
