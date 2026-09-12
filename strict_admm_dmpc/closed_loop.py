"""Fair, synchronous receding-horizon simulation for the strict solver."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib

import numpy as np

from strict_admm_dmpc.admm import EdgeState, shift_edge_state, solve_strict_admm
from strict_admm_dmpc.centralized_qp import solve_centralized_qp
from strict_admm_dmpc.communication import (
    DenialSchedule,
    MessageCache,
    PredictionMode,
)
from strict_admm_dmpc.metrics import formation_rmse
from strict_admm_dmpc.model import FloatArray, Scenario


@dataclass(frozen=True)
class ClosedLoopResult:
    time: FloatArray
    states: FloatArray
    controls: FloatArray
    reference: FloatArray
    formation_rmse: FloatArray
    communicated_scalars: FloatArray
    admm_iterations: FloatArray
    prediction_error: FloatArray
    scenario_hash: str

    def __post_init__(self) -> None:
        for name in (
            "time",
            "states",
            "controls",
            "reference",
            "formation_rmse",
            "communicated_scalars",
            "admm_iterations",
            "prediction_error",
        ):
            value = np.array(getattr(self, name), dtype=float, copy=True)
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain only finite values")
            value.flags.writeable = False
            object.__setattr__(self, name, value)


def _reference_for_time(scenario: Scenario, time: float) -> FloatArray:
    base = scenario.common_reference
    reference = base.copy()
    ny = scenario.dynamics.output_dimension
    acceleration = scenario.acceleration
    if not np.any(acceleration):
        reference[:, :ny] += time * base[0, ny : 2 * ny][None, :]
    else:
        horizon = scenario.config.horizon
        tt = time + np.arange(horizon + 1) * scenario.dynamics.dt
        reference[:, :ny] = (
            base[0, :ny][None, :]
            + base[0, ny : 2 * ny][None, :] * tt[:, None]
            + 0.5 * acceleration[None, :] * (tt**2)[:, None]
        )
        reference[:, ny : 2 * ny] = (
            base[0, ny : 2 * ny][None, :] + acceleration[None, :] * tt[:, None]
        )
    return reference


def _scenario_hash(
    scenario: Scenario,
    steps: int,
    denial: DenialSchedule,
) -> str:
    digest = hashlib.sha256()
    for array in (
        scenario.dynamics.A,
        scenario.dynamics.B,
        scenario.graph.offsets,
        scenario.graph.edge_weights,
        scenario.initial_states,
        scenario.common_reference,
        scenario.acceleration,
    ):
        digest.update(np.ascontiguousarray(array).tobytes())
    digest.update(repr((scenario.graph.edges, steps, denial)).encode())
    return digest.hexdigest()


def simulate_closed_loop(
    scenario: Scenario,
    steps: int,
    admm_cap: int,
    denial: DenialSchedule,
    prediction_mode: PredictionMode,
    agent_order: tuple[int, ...] | None = None,
    communication_enabled: bool = True,
    confidence_decay: bool = True,
    confidence_time_constant: float = 12.0,
    measurement_noise_std: float = 0.0,
    noise_seed: int | None = None,
) -> ClosedLoopResult:
    """Run a synchronous closed loop; controller cap is excluded from scenario hash.

    ``measurement_noise_std`` corrupts the observed *position* channels with
    i.i.d. Gaussian localization noise before they enter the message cache.  The
    plant state itself stays clean, so the noise only degrades the cached
    anchor from which denied predictions are extrapolated -- the channel through
    which Theorem 2's bounded-error hypothesis is stressed.
    """
    if steps < 1 or admm_cap < 1:
        raise ValueError("steps and admm_cap must be positive")
    if communication_enabled and denial.is_denied(0.0):
        raise ValueError("denial at the initial step requires a preloaded cache")
    if not np.isfinite(measurement_noise_std) or measurement_noise_std < 0.0:
        raise ValueError("measurement_noise_std must be finite and nonnegative")
    noise_rng = np.random.default_rng(
        scenario.seed if noise_seed is None else noise_seed
    )
    order = (
        tuple(range(scenario.graph.n_agents))
        if agent_order is None
        else agent_order
    )
    if tuple(sorted(order)) != tuple(range(scenario.graph.n_agents)):
        raise ValueError("agent_order must be a permutation of all agents")

    dt = scenario.dynamics.dt
    time = np.arange(steps, dtype=float) * dt
    nx = scenario.dynamics.state_dimension
    nu = scenario.dynamics.input_dimension
    ny = scenario.dynamics.output_dimension
    if nx < 2 * ny:
        raise ValueError("closed loop requires [output, output-rate, ...] state layout")
    state_history = np.empty((steps, scenario.graph.n_agents, nx))
    control_history = np.empty((steps, scenario.graph.n_agents, nu))
    reference_history = np.empty((steps, nx))
    communicated = np.zeros(steps)
    iterations = np.zeros(steps)
    prediction_error = np.zeros(steps)
    current_states = scenario.initial_states.copy()
    cache = MessageCache(
        scenario.graph.n_agents,
        denial,
        state_dimension=nx,
        output_dimension=ny,
        confidence_time_constant=confidence_time_constant,
    )
    warm_start: tuple[EdgeState, ...] | None = None
    cached_transmitted_duals = [
        (
            np.zeros(
                (
                    scenario.config.horizon + 1,
                    scenario.dynamics.output_dimension,
                )
            ),
            np.zeros(
                (
                    scenario.config.horizon + 1,
                    scenario.dynamics.output_dimension,
                )
            ),
        )
        for _ in scenario.graph.edges
    ]

    for time_index, current_time in enumerate(time):
        if time_index > 0:
            cached_transmitted_duals = [
                (
                    np.vstack([dual_i[1:], dual_i[-1]]),
                    np.vstack([dual_j[1:], dual_j[-1]]),
                )
                for dual_i, dual_j in cached_transmitted_duals
            ]
        state_history[time_index] = current_states
        observed_states = current_states
        if measurement_noise_std > 0.0:
            # corrupt position channels only (localization noise); the plant
            # state driving the dynamics is left untouched
            observed_states = observed_states.copy()
            observed_states[:, :ny] += noise_rng.normal(
                0.0, measurement_noise_std, size=(scenario.graph.n_agents, ny)
            )
        cache.observe(float(current_time), observed_states)
        reference = _reference_for_time(scenario, float(current_time))
        reference_history[time_index] = reference[0]
        remote_arguments = None
        predictions = None
        graph = scenario.graph
        denied = communication_enabled and denial.is_denied(float(current_time))
        if denied:
            predictions = [
                cache.predict_aligned(
                    agent,
                    float(current_time),
                    scenario.config.horizon,
                    dt,
                    graph.offsets[agent],
                    prediction_mode,
                )
                for agent in range(graph.n_agents)
            ]
            remote_arguments = {
                edge_id: (
                    predictions[agent_i].trajectory
                    + cached_transmitted_duals[edge_id][0],
                    predictions[agent_j].trajectory
                    + cached_transmitted_duals[edge_id][1],
                )
                for edge_id, (agent_i, agent_j) in enumerate(graph.edges)
            }
            if confidence_decay:
                edge_weights = np.array(
                    [
                        graph.edge_weights[edge_id]
                        * min(
                            predictions[agent_i].confidence,
                            predictions[agent_j].confidence,
                        )
                        for edge_id, (agent_i, agent_j) in enumerate(graph.edges)
                    ]
                )
                graph = replace(graph, edge_weights=edge_weights)

        config = replace(scenario.config, max_iterations=admm_cap)
        step_scenario = replace(
            scenario,
            graph=graph,
            config=config,
            initial_states=current_states,
            common_reference=reference,
        )
        shifted = None if warm_start is None else tuple(
            shift_edge_state(edge_state) for edge_state in warm_start
        )
        result = solve_strict_admm(
            step_scenario,
            warm_start=shifted,
            remote_arguments=remote_arguments,
            agent_order=order,
        )
        warm_start = result.edge_states
        if denied:
            # Full-horizon prediction error ‖ξ^k‖: the true remote output the
            # z-update would have used is the aligned output trajectory of the
            # actual solve; compare the cached extrapolation over the whole
            # N+1 horizon (not just the current step), which is the quantity
            # that perturbs the edge update in Theorem 2.
            true_aligned = (
                result.states @ scenario.dynamics.C.T
                - scenario.graph.offsets[:, None, :]
            )
            prediction_error[time_index] = max(
                float(
                    np.linalg.norm(
                        predictions[agent].trajectory - true_aligned[agent]
                    )
                )
                for agent in range(graph.n_agents)
            )
        if not denied:
            cached_transmitted_duals = [
                (edge_state.eta_i, edge_state.eta_j)
                for edge_state in result.edge_states
            ]
        control_history[time_index] = result.controls[:, 0]
        iterations[time_index] = result.iterations
        if not denied:
            communicated[time_index] = (
                result.iterations
                * len(graph.edges)
                * 2
                * (scenario.config.horizon + 1)
                * scenario.dynamics.output_dimension
                + len(graph.edges)
                * 2
                * (scenario.config.horizon + 1)
                * scenario.dynamics.output_dimension
                + scenario.graph.n_agents * scenario.dynamics.state_dimension
            )
        # The update is deliberately vectorized after every control is available.
        current_states = (
            current_states @ scenario.dynamics.A.T
            + control_history[time_index] @ scenario.dynamics.B.T
        )

    rmse = formation_rmse(state_history, reference_history, scenario.graph.offsets)
    return ClosedLoopResult(
        time=time,
        states=state_history,
        controls=control_history,
        reference=reference_history,
        formation_rmse=rmse,
        communicated_scalars=communicated,
        admm_iterations=iterations,
        prediction_error=prediction_error,
        scenario_hash=_scenario_hash(scenario, steps, denial),
    )


def simulate_uncoupled_mpc(
    scenario: Scenario,
    steps: int,
    denial: DenialSchedule,
) -> ClosedLoopResult:
    """Solve the gamma-zero separable MPC optimum without ADMM communication."""
    if steps < 1:
        raise ValueError("steps must be positive")
    uncoupled = replace(scenario, weights=replace(scenario.weights, gamma=0.0))
    n_agents = scenario.graph.n_agents
    nx = scenario.dynamics.state_dimension
    nu = scenario.dynamics.input_dimension
    dt = scenario.dynamics.dt
    time = np.arange(steps, dtype=float) * dt
    state_history = np.empty((steps, n_agents, nx))
    control_history = np.empty((steps, n_agents, nu))
    reference_history = np.empty((steps, nx))
    current_states = scenario.initial_states.copy()

    for time_index, current_time in enumerate(time):
        state_history[time_index] = current_states
        reference = _reference_for_time(scenario, float(current_time))
        reference_history[time_index] = reference[0]
        step_scenario = replace(
            uncoupled,
            initial_states=current_states,
            common_reference=reference,
        )
        optimum = solve_centralized_qp(step_scenario)
        control_history[time_index] = optimum.controls[:, 0]
        current_states = (
            current_states @ scenario.dynamics.A.T
            + control_history[time_index] @ scenario.dynamics.B.T
        )

    return ClosedLoopResult(
        time=time,
        states=state_history,
        controls=control_history,
        reference=reference_history,
        formation_rmse=formation_rmse(
            state_history, reference_history, scenario.graph.offsets
        ),
        communicated_scalars=np.zeros(steps),
        admm_iterations=np.zeros(steps),
        prediction_error=np.zeros(steps),
        scenario_hash=_scenario_hash(scenario, steps, denial),
    )
