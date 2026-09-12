# Reproducibility

This repository is the reference implementation and data source for the
manuscript *Distributed Formation Model Predictive Control under Communication
Denial: An Edge-Splitting ADMM with Centralized-Equivalence Gating and
Confidence-Decayed Coupling*.

## What is frozen here

`artifacts_v2_ablation/` is the canonical data directory cited by the paper.
It contains:

- `raw/results.json` — the E0–E4 full runs (ten seeds) merged with the E5
  fixed-weight ablation;
- `raw/ablation.json` — the E5 ablation records;
- `tables/` — `summary.json`, `summary.md`, `results_values.tex`,
  `ablation_full.tex`, and the figures (PDF, PNG, and editable SVG for the two
  method figures).

Every committed file is pinned by `artifacts_v2_ablation/manifest.json`:

- `source_sha256` — the reproducibility surface (`strict_admm_dmpc/*.py`,
  `experiments_v2.py`, `run_e5_ablation.py`, `export_ablation_table.py`,
  `requirements.txt`, `requirements-lock.txt`), with the exact file list in
  `source_files`;
- `raw_sha256` — the raw records;
- `artifacts_sha256` — every processed table and figure.

`artifacts_v3_extra/` holds the E6–E10 records (dual decomposition, decay-time
sweep, Lipschitz estimate, swarm-size scaling, and measurement noise) under
`raw/` and their tables and figures under `tables/`.

## Reproduction path

```powershell
python -m venv .venv-v2
& .\.venv-v2\Scripts\python.exe -m pip install -r requirements-lock.txt
& .\.venv-v2\Scripts\python.exe experiments_v2.py --profile full --output artifacts_v2_full
& .\.venv-v2\Scripts\python.exe run_e5_ablation.py
& .\.venv-v2\Scripts\python.exe export_ablation_table.py
& .\.venv-v2\Scripts\python.exe make_manifest.py
& .\.venv-v2\Scripts\python.exe run_extra_experiments.py
& .\.venv-v2\Scripts\python.exe run_scaling_noise_experiments.py
& .\.venv-v2\Scripts\python.exe -m pytest tests -q
```

`make_manifest.py` is read-only over the data directory: it hashes the frozen
raw records and processed artifacts (and the current source) and writes only
`artifacts_v2_ablation/manifest.json`, so it can be re-run to re-pin the data
without re-running any experiment.

## Verified reproducibility

- The test suite passes (76 tests): centralized equivalence, topology
  variation, invalid inputs, warm-start validation, denial semantics,
  solver-workspace reuse, and synchronous closed-loop updates.
- `summarize_experiments`, figure rendering, and `export_ablation_table`
  reproduce byte-identical from the committed raw records.
- Headline numbers reproduce exactly from the current source: the E0 objective
  gap, the E1 `rho=1` mean iteration count, the E2 mean RMSE, and the E3/E4
  denial-window peak RMSE (4–5 significant figures).

A pre-v2 legacy-integrity check (a 77th test verifying that the pre-v2
manuscript and validation assets are byte-frozen) is part of the private working
directory only; it is intentionally not published here because those files are
out of scope for the code-and-data release.

## Honest limitations

- The committed source has drifted slightly from the snapshot that first produced
  `artifacts_v2_full` (the E4 acceleration experiment, the E5 ablation, and the
  figure redesign were added later). Re-running the current source reproduces
  every headline number; a minority of non-headline E1 configurations differ by
  off-by-one/off-by-two iteration counts, and the E3 *mean* RMSE (a secondary
  metric not reported in the paper) is sensitive to chaotic reconnection
  ordering.
- The paper treats the frozen `artifacts_v2_ablation` data as the canonical
  evidence. The re-run agreement above is the reproducibility check, not a
  second source of truth.

## Environment

`requirements-lock.txt` pins `numpy==2.2.3`, `scipy==1.16.1`, `osqp==1.1.1`,
`matplotlib==3.10.8`, `pandas==3.0.3`, `pytest==9.0.3`, `pytest-cov==7.1.0` on
Python 3.13. The `direct_dependency_versions` recorded in `manifest.json` match
this pin exactly.
