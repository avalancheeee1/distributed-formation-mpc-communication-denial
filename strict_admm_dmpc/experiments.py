"""Deterministic paired experiment suites for the v2 manuscript."""

from __future__ import annotations

from dataclasses import replace
import time
from typing import Iterable

import numpy as np

from strict_admm_dmpc.admm import solve_strict_admm
from strict_admm_dmpc.centralized_qp import solve_centralized_qp
from strict_admm_dmpc.closed_loop import simulate_closed_loop, simulate_uncoupled_mpc
from strict_admm_dmpc.communication import DenialSchedule, PredictionMode
from strict_admm_dmpc.model import (
    FormationGraph,
    MPCConfig,
    MPCWeights,
    Scenario,
    build_double_integrator,
)


TOPOLOGY_EDGES: dict[str, tuple[tuple[int, int], ...]] = {
    "chain": ((0, 1), (1, 2), (2, 3)),
    "ring": ((0, 1), (1, 2), (2, 3), (0, 3)),
    "connected_random": ((0, 1), (1, 2), (2, 3), (0, 2)),
}


def topology_edges(
    topology: str, n_agents: int
) -> tuple[tuple[int, int], ...]:
    """Deterministic edge set for ``topology`` on ``n_agents`` nodes.

    The 4-agent edge sets are pinned by :data:`TOPOLOGY_EDGES` so the published
    E0/E1 results reproduce exactly; this generator extends the same three
    topologies to arbitrary swarm sizes for the scaling study (E9).
    """
    if n_agents < 2:
        raise ValueError("n_agents must be at least 2")
    chain = tuple((i, i + 1) for i in range(n_agents - 1))
    if topology == "chain":
        return chain
    if topology == "ring":
        return chain + ((n_agents - 1, 0),)
    if topology == "connected_random":
        # chain plus a single deterministic shortcut, mirroring the 4-agent
        # pattern ((0,1),(1,2),(2,3),(0,2)): a node-0 shortcut to node 2.
        return chain + ((0, 2),) if n_agents >= 4 else chain
    raise ValueError(f"unknown topology: {topology}")


def build_scenario(
    seed: int,
    topology: str = "chain",
    horizon: int = 10,
    rho: float = 1.0,
    dt: float = 0.2,
    gamma: float = 1.5,
    acceleration: float = 0.0,
    n_agents: int = 4,
) -> Scenario:
    """Build one frozen physical scenario; controller settings are explicit."""
    if n_agents == 4:
        try:
            edges = TOPOLOGY_EDGES[topology]
        except KeyError as exc:
            raise ValueError(f"unknown topology: {topology}") from exc
    else:
        edges = topology_edges(topology, n_agents)
    rng = np.random.default_rng(seed)
    offsets = np.column_stack([2.0 * np.arange(n_agents), np.zeros(n_agents)])
    initial = np.zeros((n_agents, 4))
    initial[:, :2] = offsets + rng.normal(0.0, 0.6, size=(n_agents, 2))
    initial[:, 2:] = np.array([0.4, 0.0]) + rng.normal(0.0, 0.15, size=(n_agents, 2))
    times = np.arange(horizon + 1) * dt
    reference = np.zeros((horizon + 1, 4))
    reference[:, 0] = 0.4 * times + 0.5 * acceleration * times**2
    reference[:, 2] = 0.4 + acceleration * times
    return Scenario(
        dynamics=build_double_integrator(dt),
        graph=FormationGraph(
            n_agents=n_agents,
            edges=edges,
            offsets=offsets,
            edge_weights=np.ones(len(edges)),
        ),
        weights=MPCWeights(
            q_position=2.0 * np.eye(2),
            q_velocity=0.5 * np.eye(2),
            r_input=0.15 * np.eye(2),
            p_terminal=np.diag([5.0, 5.0, 1.0, 1.0]),
            edge_metric=np.eye(2),
            gamma=gamma,
        ),
        config=MPCConfig(
            horizon=horizon,
            rho=rho,
            acceleration_lower=np.array([-2.0, -2.0]),
            acceleration_upper=np.array([2.0, 2.0]),
            absolute_tolerance=1e-6,
            relative_tolerance=1e-6,
            max_iterations=1_500,
        ),
        initial_states=initial,
        common_reference=reference,
        seed=seed,
        acceleration=np.array([acceleration, 0.0]),
    )


def _gaps(scenario: Scenario) -> dict[str, float | bool | int]:
    centralized = solve_centralized_qp(scenario)
    started = time.perf_counter()
    cpu_started = time.process_time()
    distributed = solve_strict_admm(scenario)
    cpu_elapsed = time.process_time() - cpu_started
    elapsed = time.perf_counter() - started
    objective_gap = abs(distributed.objective - centralized.objective) / max(
        1.0, abs(centralized.objective)
    )
    decision_gap = np.linalg.norm(
        np.concatenate(
            [
                (distributed.states - centralized.states).ravel(),
                (distributed.controls - centralized.controls).ravel(),
            ]
        )
    ) / max(
        1.0,
        np.linalg.norm(
            np.concatenate([centralized.states.ravel(), centralized.controls.ravel()])
        ),
    )
    return {
        "passed": bool(
            distributed.converged
            and objective_gap <= 2e-5
            and decision_gap <= 2e-5
        ),
        "iterations": distributed.iterations,
        "objective_gap": float(objective_gap),
        "decision_gap": float(decision_gap),
        "primal_residual": distributed.primal_residual,
        "dual_residual": distributed.dual_residual,
        "runtime_seconds": elapsed,
        "cpu_seconds": cpu_elapsed,
    }


def run_e0_correctness(seeds: Iterable[int] = range(10)) -> list[dict]:
    """Run the hard correctness gate on every supported topology."""
    records: list[dict] = []
    for topology in TOPOLOGY_EDGES:
        for seed in seeds:
            record = _gaps(build_scenario(seed, topology, horizon=5, rho=1.0))
            records.append({"topology": topology, "seed": seed, **record})
    return records


def run_e1_convergence(
    seeds: Iterable[int] = range(10),
    horizons: Iterable[int] = (5, 10, 20),
    rhos: Iterable[float] = (0.1, 1.0, 10.0),
) -> list[dict]:
    """Paired topology/horizon/penalty convergence sweep."""
    records: list[dict] = []
    for topology in TOPOLOGY_EDGES:
        for horizon in horizons:
            for rho in rhos:
                for seed in seeds:
                    record = _gaps(build_scenario(seed, topology, horizon, rho))
                    records.append(
                        {
                            "topology": topology,
                            "horizon": horizon,
                            "rho": rho,
                            "seed": seed,
                            **record,
                        }
                    )
    return records


def run_e2_budget(
    seeds: Iterable[int] = range(5),
    caps: Iterable[int] = (1, 2, 3, 5, 10, 20, 100),
    steps: int = 60,
) -> list[dict]:
    """Measure the tracking/runtime/communication Pareto surface."""
    caps = tuple(int(cap) for cap in caps)
    if not caps or any(cap < 1 for cap in caps):
        raise ValueError("caps must contain positive integers")
    records: list[dict] = []
    for seed in seeds:
        scenario = build_scenario(seed, "chain", horizon=10, rho=1.0)
        inactive_denial = DenialSchedule(start=10_000.0, end=10_001.0)
        ordered_caps = np.random.default_rng(seed + 10_000).permutation(caps)
        for cap_value in ordered_caps:
            cap = int(cap_value)
            started = time.perf_counter()
            cpu_started = time.process_time()
            result = simulate_closed_loop(
                scenario,
                steps=steps,
                admm_cap=cap,
                denial=inactive_denial,
                prediction_mode=PredictionMode.CONSTANT_VELOCITY,
            )
            elapsed = time.perf_counter() - started
            cpu_elapsed = time.process_time() - cpu_started
            records.append(
                {
                    "seed": seed,
                    "cap": cap,
                    "scenario_hash": result.scenario_hash,
                    "mean_rmse": float(np.mean(result.formation_rmse)),
                    "peak_rmse": float(np.max(result.formation_rmse)),
                    "terminal_rmse": float(result.formation_rmse[-1]),
                    "control_energy": float(
                        scenario.dynamics.dt * np.sum(result.controls**2)
                    ),
                    "runtime_seconds": elapsed,
                    "cpu_seconds": cpu_elapsed,
                    "communicated_scalars": float(
                        np.sum(result.communicated_scalars)
                    ),
                }
            )
    return records


def _positive_durations(durations: Iterable[float]) -> tuple[float, ...]:
    converted = tuple(float(duration) for duration in durations)
    if not converted or any(duration <= 0.0 for duration in converted):
        raise ValueError("durations must contain positive values")
    return converted


_DENIAL_SPECS: tuple[tuple[str, PredictionMode, bool, bool], ...] = (
    ("ideal", PredictionMode.CONSTANT_VELOCITY, False, True),
    ("local_mpc", PredictionMode.CONSTANT_VELOCITY, True, True),
    ("zoh", PredictionMode.ZERO_ORDER_HOLD, True, True),
    ("cv_prediction", PredictionMode.CONSTANT_VELOCITY, True, True),
)

_FIXED_SPECS: tuple[tuple[str, PredictionMode, bool, bool], ...] = (
    ("zoh_fixed", PredictionMode.ZERO_ORDER_HOLD, True, False),
    ("cv_fixed", PredictionMode.CONSTANT_VELOCITY, True, False),
)


def _run_denial_policies(
    base_builder,
    seeds: Iterable[int],
    durations: tuple[float, ...],
    cap: int,
    specs: tuple[tuple[str, PredictionMode, bool, bool], ...],
) -> list[dict]:
    """Run named denial policies; ``specs`` is (name, mode, comm, decay)."""
    records: list[dict] = []
    for seed in seeds:
        base = base_builder(seed)
        uncoupled = replace(base, weights=replace(base.weights, gamma=0.0))
        for duration in durations:
            denial = DenialSchedule(start=5.0, end=5.0 + duration)
            steps = int((max(durations) + 20.0) / base.dynamics.dt)
            for policy, mode, communication_enabled, confidence_decay in specs:
                scenario = uncoupled if policy == "local_mpc" else base
                started = time.perf_counter()
                cpu_started = time.process_time()
                if policy == "local_mpc":
                    result = simulate_uncoupled_mpc(
                        scenario,
                        steps=steps,
                        denial=denial,
                    )
                else:
                    result = simulate_closed_loop(
                        scenario,
                        steps=steps,
                        admm_cap=cap,
                        denial=denial,
                        prediction_mode=mode,
                        communication_enabled=communication_enabled,
                        confidence_decay=confidence_decay,
                    )
                elapsed = time.perf_counter() - started
                cpu_elapsed = time.process_time() - cpu_started
                denied = (result.time >= denial.start) & (result.time < denial.end)
                reconnected = (result.time >= denial.end) & (
                    result.time < denial.end + 10.0
                )
                records.append(
                    {
                        "seed": seed,
                        "duration": duration,
                        "policy": policy,
                        "scenario_hash": result.scenario_hash,
                        "denial_peak_rmse": float(
                            np.max(result.formation_rmse[denied])
                        ),
                        "reconnection_peak_rmse": float(
                            np.max(result.formation_rmse[reconnected])
                        ),
                        "mean_rmse": float(np.mean(result.formation_rmse)),
                        "denial_mean_rmse": float(
                            np.mean(result.formation_rmse[denied])
                        ),
                        "prediction_error_peak": float(
                            np.max(result.prediction_error[denied])
                        ),
                        "control_energy": float(
                            scenario.dynamics.dt * np.sum(result.controls**2)
                        ),
                        "runtime_seconds": elapsed,
                        "cpu_seconds": cpu_elapsed,
                        "communicated_scalars": float(
                            np.sum(result.communicated_scalars)
                        ),
                    }
                )
    return records


def run_e3_denial(
    seeds: Iterable[int] = range(5),
    durations: Iterable[float] = (10.0, 30.0, 60.0),
    cap: int = 5,
) -> list[dict]:
    """Denial policy comparison in the nominal constant-velocity scenario."""
    durations = _positive_durations(durations)
    return _run_denial_policies(
        lambda seed: build_scenario(seed, "chain", horizon=10, rho=1.0),
        seeds,
        durations,
        cap,
        _DENIAL_SPECS,
    )


def run_e4_acceleration(
    seeds: Iterable[int] = range(5),
    durations: Iterable[float] = (10.0, 30.0, 60.0),
    cap: int = 5,
) -> list[dict]:
    """Denial policy comparison under a uniformly accelerating reference."""
    durations = _positive_durations(durations)
    return _run_denial_policies(
        lambda seed: build_scenario(
            seed, "chain", horizon=10, rho=1.0, acceleration=0.02
        ),
        seeds,
        durations,
        cap,
        _DENIAL_SPECS,
    )


def run_e5_fixed_weight_ablation(
    seeds: Iterable[int] = range(10),
    durations: Iterable[float] = (10.0, 30.0, 60.0),
    cap: int = 5,
) -> dict[str, list[dict]]:
    """Fixed-weight (no confidence decay) ablation of the denial policies.

    Runs only the CV and ZOH variants with the edge weight held fixed
    (``confidence_decay=False``) in both the nominal and accelerating
    scenarios, so the comparison against the decaying variants isolates the
    confidence-decay mechanism from the predictor itself.
    """
    durations = _positive_durations(durations)
    e3_fixed = _run_denial_policies(
        lambda seed: build_scenario(seed, "chain", horizon=10, rho=1.0),
        seeds,
        durations,
        cap,
        _FIXED_SPECS,
    )
    e4_fixed = _run_denial_policies(
        lambda seed: build_scenario(
            seed, "chain", horizon=10, rho=1.0, acceleration=0.02
        ),
        seeds,
        durations,
        cap,
        _FIXED_SPECS,
    )
    return {"e3_fixed": e3_fixed, "e4_fixed": e4_fixed}
