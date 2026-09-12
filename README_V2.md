# Strict ADMM-DMPC v2 — environment and reproduction

This directory contains a frozen Python reference implementation of a convex
edge-splitting ADMM-DMPC formulation. It is independent of, and does not claim
executable equivalence to, the evolving Unity project that motivated the study.

## Environment

```powershell
python -m venv .venv-v2
& .\.venv-v2\Scripts\python.exe -m pip install -r requirements-lock.txt
```

`requirements-lock.txt` pins the dependency versions that produced the
committed data (see `REPRODUCIBILITY.md`).

## Correctness gate

```powershell
& .\.venv-v2\Scripts\python.exe -m pytest tests -q
```

The distributed solver must match the centralized OSQP solution before any
closed-loop experiment or result is accepted.

## Experiments

```powershell
& .\.venv-v2\Scripts\python.exe experiments_v2.py --profile pilot
& .\.venv-v2\Scripts\python.exe experiments_v2.py --profile full --output artifacts_v2_full
& .\.venv-v2\Scripts\python.exe run_e5_ablation.py
& .\.venv-v2\Scripts\python.exe export_ablation_table.py
& .\.venv-v2\Scripts\python.exe make_manifest.py
& .\.venv-v2\Scripts\python.exe run_extra_experiments.py
& .\.venv-v2\Scripts\python.exe run_scaling_noise_experiments.py
```

The pilot profile is a smoke study with three seeds. The full profile uses ten
seeds and is the only source for manuscript-level numerical claims. Runtime
measurements are machine dependent. The communication-denial branch is marked
as inexact and cannot report strict ADMM convergence.

`artifacts_v2_ablation/` is the canonical data directory cited by the paper
(the full run merged with the E5 fixed-weight ablation); its contents are
pinned by `artifacts_v2_ablation/manifest.json` (see `REPRODUCIBILITY.md`).
`artifacts_v3_extra/` holds the E6–E10 records (dual decomposition, decay-time
sweep, Lipschitz estimate, swarm-size scaling, and measurement noise).
