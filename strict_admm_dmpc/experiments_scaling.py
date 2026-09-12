"""Scaling and robustness experiments that close the two remaining gaps.

* E9 -- swarm-size scaling: run the centralized-equivalence correctness gate
  (``_gaps``) on 4 / 6 / 8 agents across the three topologies, showing that the
  edge-splitting ADMM preserves centralized equivalence and stays well-posed as
  the formation grows.
* E10 -- measurement noise: sweep the observed-position noise standard
  deviation and verify that the Theorem-3 saturation signature (the 60 s and
  30 s denial-window peaks coincide) survives localization noise, while the
  mismatched predictor's error keeps growing over the same window.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from strict_admm_dmpc.closed_loop import simulate_closed_loop
from strict_admm_dmpc.communication import DenialSchedule, PredictionMode
from strict_admm_dmpc.experiments import _gaps, build_scenario


# ---------------------------------------------------------------------------
# E9 -- swarm-size scaling
# ---------------------------------------------------------------------------
def run_e9_scaling(
    seeds: Iterable[int] = range(5),
    n_agents_list: Iterable[int] = (4, 6, 8),
    topologies: Iterable[str] = ("chain", "ring", "connected_random"),
    horizon: int = 10,
    rho: float = 1.0,
) -> list[dict]:
    """Correctness gate + iteration/runtime cost as the swarm grows.

    Reuses the same acceptance thresholds as E0 (relative objective and
    decision gap each below 2e-5), so the row ``passed`` is directly comparable
    with the published 4-agent gate.
    """
    n_agents_list = tuple(int(n) for n in n_agents_list)
    topologies = tuple(topologies)
    records: list[dict] = []
    for n_agents in n_agents_list:
        for topology in topologies:
            for seed in seeds:
                record = _gaps(
                    build_scenario(
                        seed, topology, horizon=horizon, rho=rho, n_agents=n_agents
                    )
                )
                records.append(
                    {"n_agents": n_agents, "topology": topology, "seed": seed, **record}
                )
    return records


# ---------------------------------------------------------------------------
# E10 -- measurement noise
# ---------------------------------------------------------------------------
def run_e10_noise(
    seeds: Iterable[int] = range(10),
    sigmas: Iterable[float] = (0.0, 0.05, 0.1, 0.2),
    durations: Iterable[float] = (30.0, 60.0),
    cap: int = 5,
) -> list[dict]:
    """Sweep observed-position noise; confirm saturation survives.

    Uses the zero-order-hold predictor in the nominal constant-velocity
    scenario -- the mismatched predictor whose error grows with outage -- so
    that a 60 s / 30 s peak ratio near unity is the fingerprint that the
    confidence-decayed coupling saturates, while ``prediction_error_peak``
    keeps growing over the same window.  ``noise_seed`` is pinned per scenario
    seed so the sweep is deterministic and reproducible.
    """
    sigmas = tuple(float(s) for s in sigmas)
    durations = tuple(float(d) for d in durations)
    records: list[dict] = []
    for seed in seeds:
        base = build_scenario(seed, "chain", horizon=10, rho=1.0)
        for sigma in sigmas:
            for duration in durations:
                denial = DenialSchedule(start=5.0, end=5.0 + duration)
                steps = int((max(durations) + 20.0) / base.dynamics.dt)
                result = simulate_closed_loop(
                    base,
                    steps=steps,
                    admm_cap=cap,
                    denial=denial,
                    prediction_mode=PredictionMode.ZERO_ORDER_HOLD,
                    confidence_decay=True,
                    measurement_noise_std=sigma,
                    noise_seed=seed + 100_000,
                )
                denied = (result.time >= denial.start) & (
                    result.time < denial.end
                )
                records.append(
                    {
                        "seed": seed,
                        "sigma": sigma,
                        "duration": duration,
                        "denial_peak_rmse": float(
                            np.max(result.formation_rmse[denied])
                        ),
                        "prediction_error_peak": float(
                            np.max(result.prediction_error[denied])
                        ),
                        "mean_rmse": float(np.mean(result.formation_rmse)),
                    }
                )
    return records
