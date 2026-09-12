"""Strict edge-splitting ADMM for the coupled formation MPC QP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from strict_admm_dmpc.centralized_qp import evaluate_global_objective
from strict_admm_dmpc.local_qp import LocalQPWorkspace
from strict_admm_dmpc.model import FloatArray, Scenario


@dataclass(frozen=True)
class EdgeState:
    z_i: FloatArray
    z_j: FloatArray
    eta_i: FloatArray
    eta_j: FloatArray

    def __post_init__(self) -> None:
        arrays = []
        for name in ("z_i", "z_j", "eta_i", "eta_j"):
            value = np.array(getattr(self, name), dtype=float, copy=True)
            if value.ndim != 2 or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be a finite two-dimensional array")
            value.flags.writeable = False
            object.__setattr__(self, name, value)
            arrays.append(value)
        if any(value.shape != arrays[0].shape for value in arrays[1:]):
            raise ValueError("all edge-state arrays must have matching shapes")


@dataclass(frozen=True)
class ADMMIterationRecord:
    iteration: int
    objective: float
    primal_residual: float
    dual_residual: float
    primal_tolerance: float
    dual_tolerance: float


@dataclass(frozen=True)
class ADMMResult:
    states: FloatArray
    controls: FloatArray
    edge_states: tuple[EdgeState, ...]
    objective: float
    primal_residual: float
    dual_residual: float
    residual_stopped: bool
    exact_communication: bool
    iterations: int
    history: tuple[ADMMIterationRecord, ...]

    def __post_init__(self) -> None:
        for name in ("states", "controls"):
            value = np.array(getattr(self, name), dtype=float, copy=True)
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain only finite values")
            value.flags.writeable = False
            object.__setattr__(self, name, value)

    @property
    def converged(self) -> bool:
        """True only for exact-message ADMM stopped by standard residuals."""
        return self.exact_communication and self.residual_stopped


def edge_prox_update(
    argument_i: FloatArray,
    argument_j: FloatArray,
    rho: float,
    coupling: float,
    metric: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Solve the edge proximal step by average/difference decomposition."""
    argument_i = np.asarray(argument_i, dtype=float)
    argument_j = np.asarray(argument_j, dtype=float)
    metric = np.asarray(metric, dtype=float)
    if not np.isfinite(rho) or rho <= 0.0:
        raise ValueError("rho must be positive")
    if not np.isfinite(coupling) or coupling < 0.0:
        raise ValueError("coupling must be nonnegative")
    if (
        argument_i.ndim != 2
        or argument_i.shape != argument_j.shape
        or not np.all(np.isfinite(argument_i))
        or not np.all(np.isfinite(argument_j))
    ):
        raise ValueError("edge endpoint arguments must have matching shapes")
    if (
        metric.shape != (argument_i.shape[1], argument_i.shape[1])
        or not np.all(np.isfinite(metric))
        or not np.allclose(metric, metric.T, atol=1e-12)
        or np.min(np.linalg.eigvalsh(metric)) < -1e-10
    ):
        raise ValueError("metric must be a finite symmetric PSD matrix")
    average = 0.5 * (argument_i + argument_j)
    system = rho * np.eye(metric.shape[0]) + 2.0 * coupling * metric
    difference = np.linalg.solve(
        system, (rho * (argument_i - argument_j)).T
    ).T
    return average + 0.5 * difference, average - 0.5 * difference


def shift_edge_state(state: EdgeState) -> EdgeState:
    """Shift every edge trajectory by one MPC step and repeat its terminal row."""

    def shift(values: FloatArray) -> FloatArray:
        return np.vstack([values[1:], values[-1]])

    return EdgeState(
        z_i=shift(state.z_i),
        z_j=shift(state.z_j),
        eta_i=shift(state.eta_i),
        eta_j=shift(state.eta_j),
    )


def _initial_edge_states(scenario: Scenario) -> tuple[EdgeState, ...]:
    shape = (scenario.config.horizon + 1, scenario.dynamics.output_dimension)
    return tuple(
        EdgeState(
            z_i=np.zeros(shape),
            z_j=np.zeros(shape),
            eta_i=np.zeros(shape),
            eta_j=np.zeros(shape),
        )
        for _ in scenario.graph.edges
    )


def solve_strict_admm(
    scenario: Scenario,
    warm_start: tuple[EdgeState, ...] | None = None,
    remote_arguments: dict[int, tuple[FloatArray, FloatArray]] | None = None,
    agent_order: tuple[int, ...] | None = None,
) -> ADMMResult:
    """Solve the coupled QP with strict synchronous edge-splitting ADMM."""
    edge_states = list(
        _initial_edge_states(scenario) if warm_start is None else warm_start
    )
    if len(edge_states) != len(scenario.graph.edges):
        raise ValueError("warm start must contain one state per graph edge")
    if remote_arguments is not None and any(
        not isinstance(edge_id, int)
        or isinstance(edge_id, bool)
        or not 0 <= edge_id < len(scenario.graph.edges)
        for edge_id in remote_arguments
    ):
        raise ValueError("remote argument edge id is out of range")
    expected_shape = (
        scenario.config.horizon + 1,
        scenario.dynamics.output_dimension,
    )
    if any(edge_state.z_i.shape != expected_shape for edge_state in edge_states):
        raise ValueError("warm-start edge state has incompatible shape")
    n_agents = scenario.graph.n_agents
    order = tuple(range(n_agents)) if agent_order is None else agent_order
    if tuple(sorted(order)) != tuple(range(n_agents)):
        raise ValueError("agent_order must be a permutation of all agents")
    horizon = scenario.config.horizon
    nx = scenario.dynamics.state_dimension
    nu = scenario.dynamics.input_dimension
    states = np.zeros((n_agents, horizon + 1, nx))
    controls = np.zeros((n_agents, horizon, nu))
    history: list[ADMMIterationRecord] = []
    residual_stopped = False
    local_workspaces = [
        LocalQPWorkspace(scenario, agent) for agent in range(n_agents)
    ]

    for iteration in range(1, scenario.config.max_iterations + 1):
        for agent in order:
            targets: list[FloatArray] = []
            for edge_id in scenario.graph.incident_edges(agent):
                edge = scenario.graph.edges[edge_id]
                edge_state = edge_states[edge_id]
                if agent == edge[0]:
                    targets.append(edge_state.z_i - edge_state.eta_i)
                else:
                    targets.append(edge_state.z_j - edge_state.eta_j)
            local = local_workspaces[agent].solve(scenario, tuple(targets))
            states[agent] = local.states
            controls[agent] = local.controls

        aligned = states @ scenario.dynamics.C.T - scenario.graph.offsets[:, None, :]
        previous = edge_states
        updated: list[EdgeState] = []
        primal_parts: list[FloatArray] = []
        y_parts: list[FloatArray] = []
        z_parts: list[FloatArray] = []
        dual_by_agent = np.zeros((n_agents, horizon + 1, nx))
        eta_by_agent = np.zeros((n_agents, horizon + 1, nx))
        for edge_id, (agent_i, agent_j) in enumerate(scenario.graph.edges):
            old = previous[edge_id]
            coupling = scenario.weights.gamma * scenario.graph.edge_weights[edge_id]
            argument_i = aligned[agent_i] + old.eta_i
            argument_j = aligned[agent_j] + old.eta_j
            if remote_arguments is None or edge_id not in remote_arguments:
                z_i, z_j = edge_prox_update(
                    argument_i,
                    argument_j,
                    scenario.config.rho,
                    coupling,
                    scenario.weights.edge_metric,
                )
            else:
                estimated_i, estimated_j = remote_arguments[edge_id]
                estimated_i = np.asarray(estimated_i, dtype=float)
                estimated_j = np.asarray(estimated_j, dtype=float)
                expected_remote_shape = (
                    horizon + 1,
                    scenario.dynamics.output_dimension,
                )
                if (
                    estimated_i.shape != expected_remote_shape
                    or estimated_j.shape != expected_remote_shape
                    or not np.all(np.isfinite(estimated_i))
                    or not np.all(np.isfinite(estimated_j))
                ):
                    raise ValueError("remote argument has incompatible shape")
                z_i, _ = edge_prox_update(
                    argument_i,
                    estimated_j,
                    scenario.config.rho,
                    coupling,
                    scenario.weights.edge_metric,
                )
                _, z_j = edge_prox_update(
                    estimated_i,
                    argument_j,
                    scenario.config.rho,
                    coupling,
                    scenario.weights.edge_metric,
                )
            residual_i = aligned[agent_i] - z_i
            residual_j = aligned[agent_j] - z_j
            eta_i = old.eta_i + residual_i
            eta_j = old.eta_j + residual_j
            updated.append(
                EdgeState(z_i=z_i, z_j=z_j, eta_i=eta_i, eta_j=eta_j)
            )
            primal_parts.extend([residual_i.ravel(), residual_j.ravel()])
            dual_by_agent[agent_i] += (z_i - old.z_i) @ scenario.dynamics.C
            dual_by_agent[agent_j] += (z_j - old.z_j) @ scenario.dynamics.C
            eta_by_agent[agent_i] += eta_i @ scenario.dynamics.C
            eta_by_agent[agent_j] += eta_j @ scenario.dynamics.C
            y_parts.extend([aligned[agent_i].ravel(), aligned[agent_j].ravel()])
            z_parts.extend([z_i.ravel(), z_j.ravel()])
        edge_states = updated
        primal_vector = np.concatenate(primal_parts) if primal_parts else np.zeros(0)
        y_vector = np.concatenate(y_parts) if y_parts else np.zeros(0)
        z_vector = np.concatenate(z_parts) if z_parts else np.zeros(0)
        primal_residual = float(np.linalg.norm(primal_vector))
        dual_residual = float(
            scenario.config.rho * np.linalg.norm(dual_by_agent.ravel())
        )
        dimension = max(1, primal_vector.size)
        primal_tolerance = float(
            np.sqrt(dimension) * scenario.config.absolute_tolerance
            + scenario.config.relative_tolerance
            * max(np.linalg.norm(y_vector), np.linalg.norm(z_vector))
        )
        dual_tolerance = float(
            np.sqrt(n_agents * ((horizon + 1) * nx + horizon * nu))
            * scenario.config.absolute_tolerance
            + scenario.config.relative_tolerance
            * scenario.config.rho
            * np.linalg.norm(eta_by_agent.ravel())
        )
        objective = evaluate_global_objective(scenario, states, controls)
        history.append(
            ADMMIterationRecord(
                iteration=iteration,
                objective=objective,
                primal_residual=primal_residual,
                dual_residual=dual_residual,
                primal_tolerance=primal_tolerance,
                dual_tolerance=dual_tolerance,
            )
        )
        if primal_residual <= primal_tolerance and dual_residual <= dual_tolerance:
            residual_stopped = True
            break

    last = history[-1]
    return ADMMResult(
        states=states,
        controls=controls,
        edge_states=tuple(edge_states),
        objective=last.objective,
        primal_residual=last.primal_residual,
        dual_residual=last.dual_residual,
        residual_stopped=residual_stopped,
        exact_communication=not remote_arguments,
        iterations=last.iteration,
        history=tuple(history),
    )
