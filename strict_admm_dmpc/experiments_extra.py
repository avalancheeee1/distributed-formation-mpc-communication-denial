"""Substantive-robustness experiments for the v2 manuscript.

Three experiments close the gaps flagged in review:

* E6 -- comparative distributed baseline: plain Lagrangian dual decomposition
  versus edge-splitting ADMM on the same QP/graph, giving the "why ADMM"
  iteration-to-accuracy counterfactual.
* E7 -- confidence time-constant sensitivity: sweep ``T_w`` over {6, 12, 24} s
  and verify the saturation signature of Theorem 3 holds across the range.
* E8 -- numerical Lipschitz constant: finite-difference the minimizer's
  sensitivity to the edge-weight magnitude, certifying the constant ``L`` of
  Theorem 3 that was previously left unidentified.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import numpy as np

from strict_admm_dmpc.admm import solve_strict_admm
from strict_admm_dmpc.centralized_qp import solve_centralized_qp
from strict_admm_dmpc.closed_loop import simulate_closed_loop
from strict_admm_dmpc.communication import DenialSchedule, PredictionMode
from strict_admm_dmpc.dual_decomposition import solve_dual_decomposition
from strict_admm_dmpc.experiments import build_scenario


def _decision_vector(states, controls) -> np.ndarray:
    return np.concatenate([states.ravel(), controls.ravel()])


def _objective_gap(objective: float, central: float) -> float:
    return abs(objective - central) / max(1.0, abs(central))


def _first_iteration_at_gap(
    objectives: list[float], central: float, threshold: float
) -> int:
    """First 1-based iteration whose relative objective gap is <= threshold."""
    for index, objective in enumerate(objectives, start=1):
        if _objective_gap(objective, central) <= threshold:
            return index
    return -1  # never reached within the recorded horizon


# ---------------------------------------------------------------------------
# E6 -- dual decomposition vs edge-splitting ADMM
# ---------------------------------------------------------------------------
def run_e6_dual_decomposition(
    seeds: Iterable[int] = range(10),
    topologies: Iterable[str] = ("chain", "ring", "connected_random"),
    horizon: int = 10,
    rho: float = 1.0,
    step_sizes: Iterable[float] = (0.75, 1.5, 3.0),
    max_iterations: int = 10_000,
    gap_threshold: float = 1e-4,
) -> list[dict]:
    """Compare ADMM and dual decomposition iteration-to-accuracy.

    Both solvers run on the identical scenario; ADMM stops on its residual
    tolerance while dual decomposition runs a diminishing-step schedule to a
    fixed budget.  For each solver we report the first iteration at which the
    relative objective gap drops to ``gap_threshold``, and the gap each solver
    leaves after the ADMM iteration count (the "same-budget" comparison).
    """
    topologies = tuple(topologies)
    step_sizes = tuple(float(step) for step in step_sizes)
    records: list[dict] = []
    for topology in topologies:
        for seed in seeds:
            scenario = build_scenario(seed, topology, horizon=horizon, rho=rho)
            central = solve_centralized_qp(scenario)

            admm = solve_strict_admm(scenario)
            admm_objectives = [record.objective for record in admm.history]
            admm_at_gap = _first_iteration_at_gap(
                admm_objectives, central.objective, gap_threshold
            )
            admm_budget = admm.iterations
            admm_final_gap = _objective_gap(admm.objective, central.objective)

            row: dict = {
                "topology": topology,
                "seed": seed,
                "admm_iterations": admm.iterations,
                "admm_iterations_to_gap": admm_at_gap,
                "admm_final_gap": float(admm_final_gap),
                "central_objective": float(central.objective),
            }
            for step_size in step_sizes:
                dd = solve_dual_decomposition(
                    scenario,
                    step_size=step_size,
                    schedule="diminishing",
                    max_iterations=max_iterations,
                    central_objective=central.objective,
                )
                dd_objectives = [it.objective for it in dd.history]
                dd_at_gap = _first_iteration_at_gap(
                    dd_objectives, central.objective, gap_threshold
                )
                # gap the dual decomposition leaves after the ADMM budget
                dd_after_admm = (
                    dd.history[admm_budget - 1].objective_gap
                    if 0 < admm_budget <= len(dd.history)
                    else dd.history[-1].objective_gap
                )
                key = str(step_size)
                row[f"dd_{key}_iterations_to_gap"] = dd_at_gap
                row[f"dd_{key}_gap_after_admm_budget"] = float(dd_after_admm)
                row[f"dd_{key}_final_gap"] = float(dd.history[-1].objective_gap)
            records.append(row)
    return records


def run_e6_convergence_trace(
    seed: int = 0,
    topology: str = "chain",
    horizon: int = 10,
    rho: float = 1.0,
    step_sizes: Iterable[float] = (0.75, 1.5, 3.0),
    max_iterations: int = 10_000,
) -> dict:
    """Return the full objective-gap traces of ADMM and dual decomposition."""
    scenario = build_scenario(seed, topology, horizon=horizon, rho=rho)
    central = solve_centralized_qp(scenario)
    admm = solve_strict_admm(scenario)
    trace: dict = {
        "admm": [
            {
                "iteration": record.iteration,
                "objective_gap": _objective_gap(
                    record.objective, central.objective
                ),
                "primal_residual": record.primal_residual,
            }
            for record in admm.history
        ],
        "admm_iterations": admm.iterations,
        "dual_decomposition": {},
        "central_objective": float(central.objective),
    }
    for step_size in step_sizes:
        dd = solve_dual_decomposition(
            scenario,
            step_size=float(step_size),
            schedule="diminishing",
            max_iterations=max_iterations,
            central_objective=central.objective,
        )
        trace["dual_decomposition"][str(step_size)] = [
            {
                "iteration": it.iteration,
                "objective_gap": it.objective_gap,
                "primal_residual": it.primal_residual,
            }
            for it in dd.history
        ]
    return trace


# ---------------------------------------------------------------------------
# E7 -- confidence time-constant sensitivity
# ---------------------------------------------------------------------------
def run_e7_tw_sweep(
    seeds: Iterable[int] = range(10),
    durations: Iterable[float] = (10.0, 30.0, 60.0),
    tws: Iterable[float] = (6.0, 12.0, 24.0),
    cap: int = 5,
) -> list[dict]:
    """Sweep the confidence time constant and verify saturation robustness.

    Uses the zero-order-hold predictor in the nominal constant-velocity
    scenario -- the mismatched predictor whose error grows with outage -- so
    saturation is the observable fingerprint of Theorem 3.  ``duration`` is
    the outage length; ``denial_peak_rmse`` is the denial-window peak formation
    error; ``prediction_error_peak`` is the peak measured prediction error over
    the same window (kept so the reader can confirm that the prediction error
    keeps growing while the formation error saturates).
    """
    durations = tuple(float(d) for d in durations)
    tws = tuple(float(t) for t in tws)
    records: list[dict] = []
    for seed in seeds:
        base = build_scenario(seed, "chain", horizon=10, rho=1.0)
        for tw in tws:
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
                    confidence_time_constant=tw,
                )
                denied = (result.time >= denial.start) & (
                    result.time < denial.end
                )
                records.append(
                    {
                        "seed": seed,
                        "tw": tw,
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


# ---------------------------------------------------------------------------
# E8 -- numerical Lipschitz constant L of Theorem 3
# ---------------------------------------------------------------------------
def run_e8_lipschitz(
    seeds: Iterable[int] = range(10),
    topologies: Iterable[str] = ("chain", "ring", "connected_random"),
    horizon: int = 10,
    scales: Iterable[float] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
) -> list[dict]:
    """Finite-difference the minimizer's sensitivity to the edge-weight scale.

    For a weight vector ``s * 1`` the Theorem 3 bound reads
    ``||x*(s) - x*_loc|| <= L * (s * |E|)``, so
    ``L >= ||x*(s) - x*_loc|| / (s * |E|)`` is a certified lower bound on the
    uniform Lipschitz constant.  We report the largest such ratio over the
    scalar path, which is the constant that governs the confidence-decay
    contraction (uniform scaling of every edge weight).
    """
    topologies = tuple(topologies)
    scales = tuple(float(s) for s in scales)
    records: list[dict] = []
    for topology in topologies:
        for seed in seeds:
            scenario = build_scenario(seed, topology, horizon=horizon, rho=1.0)
            n_edges = len(scenario.graph.edges)
            base_graph = scenario.graph
            # uncoupled optimum x*_loc = x*(w = 0)
            uncoupled = replace(
                scenario,
                graph=replace(base_graph, edge_weights=np.zeros(n_edges)),
            )
            x_loc = solve_centralized_qp(uncoupled)
            x_loc_vector = _decision_vector(x_loc.states, x_loc.controls)
            row: dict = {
                "topology": topology,
                "seed": seed,
                "n_edges": n_edges,
            }
            largest_ratio = 0.0
            for scale in scales:
                scaled = replace(
                    scenario,
                    graph=replace(
                        base_graph, edge_weights=scale * np.ones(n_edges)
                    ),
                )
                x_s = solve_centralized_qp(scaled)
                deviation = float(
                    np.linalg.norm(
                        _decision_vector(x_s.states, x_s.controls)
                        - x_loc_vector
                    )
                )
                row[f"deviation_s{scale:g}"] = deviation
                if scale > 0.0:
                    ratio = deviation / (scale * n_edges)
                    row[f"ratio_s{scale:g}"] = ratio
                    largest_ratio = max(largest_ratio, ratio)
            row["lipschitz_lower_bound"] = float(largest_ratio)
            records.append(row)
    return records
