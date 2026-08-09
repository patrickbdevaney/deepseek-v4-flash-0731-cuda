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
