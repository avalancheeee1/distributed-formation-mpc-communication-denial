# Edge-Splitting ADMM for Distributed Formation MPC under Communication Denial

Reference implementation and data for the manuscript

> **Distributed Formation Model Predictive Control under Communication Denial:
> An Edge-Splitting ADMM with Centralized-Equivalence Gating and
> Confidence-Decayed Coupling** (under review)

This repository is the frozen Python reference implementation and data source
for the paper. It contains the convex edge-splitting ADMM-DMPC formulation for
multi-agent formation tracking, together with the raw experiment records and
the processed tables and figures that the paper cites.

## Contents

- `strict_admm_dmpc/` — the reference implementation (plant model, local and
  centralized QP solvers, the edge-splitting ADMM, dual decomposition, and the
  closed-loop simulation).
- `experiments_v2.py` — E0–E4 experiment driver (centralized equivalence,
  convergence, iteration budget, communication denial, acceleration).
- `run_e5_ablation.py` — E5 fixed-weight ablation (isolates confidence decay
  as the causal mechanism).
- `run_extra_experiments.py` — E6–E8 (dual-decomposition comparison, decay-time
  sweep, Lipschitz estimate).
- `run_scaling_noise_experiments.py` — E9–E10 (swarm-size scaling and
  measurement-noise robustness).
- `validate_lipschitz_analytic.py` — certifies, per seed, the analytic Lipschitz
  constant that replaces the E8 finite-difference estimate 0.94: the
  strong-convexity modulus read off the input cost, the unconditional bound, the
  sharp value solved on the free block, and the bias of the E8 chord protocol.
- `export_ablation_table.py`, `make_figures_v2.py` … `make_figures_v6.py`,
  `make_svg_method_figs.py` — table and figure generation.
- `make_manifest.py` — re-pins the SHA-256 manifest of the canonical data
  directory (read-only over the data; re-runs no experiment).
- `tests/` — the 76-test verification suite (centralized equivalence, topology
  variation, invalid inputs, warm-start validation, denial semantics,
  solver-workspace reuse, and synchronous closed-loop updates).
- `artifacts_v2_ablation/` — the canonical data directory cited by the paper
  (E0–E5), pinned by its `manifest.json`.
- `artifacts_v3_extra/` — the E6–E10 records (dual decomposition, decay-time
  sweep, Lipschitz, scaling, noise).
- `artifacts_v3_rev1/tables/lipschitz_analytic.json` — the certified analytic
  Lipschitz constant (the 90 per-seed records behind its summary). The rest of
  `artifacts_v3_rev1/` covers the stochastic-denial experiments and is not part
  of this release.

## Quick start

```powershell
python -m venv .venv-v2
& .\.venv-v2\Scripts\python.exe -m pip install -r requirements-lock.txt
& .\.venv-v2\Scripts\python.exe -m pytest tests -q
```

## Reproducibility

The canonical data is pinned by SHA-256 manifest; the raw records, processed
tables, and figures can be verified byte-identical from the committed source.
See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the frozen-data layout, the
reproduction path, and the honest limits of re-running. See
[README_V2.md](README_V2.md) for the environment setup and experiment commands.

## License

MIT — see [LICENSE](LICENSE).
