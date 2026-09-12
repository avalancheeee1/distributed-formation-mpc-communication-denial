"""Centralized convex QP used as the numerical ground-truth baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import osqp
from numpy.typing import NDArray
from scipy import sparse

from strict_admm_dmpc.model import FloatArray, Scenario


@dataclass(frozen=True)
class CentralizedResult:
    status: str
    states: FloatArray
    controls: FloatArray
    objective: float
    max_dynamics_residual: float
    max_input_violation: float

    def __post_init__(self) -> None:
        for name in ("states", "controls"):
            value = np.array(getattr(self, name), dtype=float, copy=True)
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain only finite values")
            value.flags.writeable = False
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class _Layout:
    n_agents: int
    horizon: int
    nx: int
    nu: int

    @property
    def agent_size(self) -> int:
        return (self.horizon + 1) * self.nx + self.horizon * self.nu

    @property
    def size(self) -> int:
        return self.n_agents * self.agent_size

    def state(self, agent: int, step: int) -> slice:
        start = agent * self.agent_size + step * self.nx
        return slice(start, start + self.nx)

    def control(self, agent: int, step: int) -> slice:
        start = (
            agent * self.agent_size
            + (self.horizon + 1) * self.nx
            + step * self.nu
        )
        return slice(start, start + self.nu)


def _add_block(matrix: sparse.lil_matrix, row: slice, col: slice, value: FloatArray) -> None:
    matrix[row, col] = matrix[row, col] + value


def _assemble_problem(
    scenario: Scenario,
) -> tuple[sparse.csc_matrix, FloatArray, sparse.csc_matrix, FloatArray, FloatArray, _Layout]:
    dynamics = scenario.dynamics
    graph = scenario.graph
    config = scenario.config
    weights = scenario.weights
    layout = _Layout(
        n_agents=graph.n_agents,
        horizon=config.horizon,
        nx=dynamics.state_dimension,
        nu=dynamics.input_dimension,
    )
    p = sparse.lil_matrix((layout.size, layout.size), dtype=float)
    q = np.zeros(layout.size)
    q_stage = weights.stage_state

    for agent in range(graph.n_agents):
        reference = scenario.agent_reference(agent)
        for step in range(config.horizon + 1):
            state_slice = layout.state(agent, step)
            state_weight = weights.p_terminal if step == config.horizon else q_stage
            _add_block(p, state_slice, state_slice, state_weight)
            q[state_slice] -= state_weight @ reference[step]
        for step in range(config.horizon):
            control_slice = layout.control(agent, step)
            _add_block(p, control_slice, control_slice, weights.r_input)

    c = dynamics.C
    for edge_id, (agent_i, agent_j) in enumerate(graph.edges):
        coupling = weights.gamma * graph.edge_weights[edge_id]
        edge_hessian = coupling * (c.T @ weights.edge_metric @ c)
        offset_difference = graph.offsets[agent_i] - graph.offsets[agent_j]
        edge_linear = coupling * (c.T @ weights.edge_metric @ offset_difference)
        for step in range(config.horizon + 1):
            i_slice = layout.state(agent_i, step)
            j_slice = layout.state(agent_j, step)
            _add_block(p, i_slice, i_slice, edge_hessian)
            _add_block(p, j_slice, j_slice, edge_hessian)
            _add_block(p, i_slice, j_slice, -edge_hessian)
            _add_block(p, j_slice, i_slice, -edge_hessian)
            q[i_slice] -= edge_linear
            q[j_slice] += edge_linear

    equality_rows = graph.n_agents * (
        dynamics.state_dimension + config.horizon * dynamics.state_dimension
    )
    bound_rows = graph.n_agents * config.horizon * dynamics.input_dimension
    constraint = sparse.lil_matrix((equality_rows + bound_rows, layout.size))
    lower = np.zeros(equality_rows + bound_rows)
    upper = np.zeros(equality_rows + bound_rows)
    row = 0
    for agent in range(graph.n_agents):
        state_0 = layout.state(agent, 0)
        constraint[row : row + layout.nx, state_0] = np.eye(layout.nx)
        lower[row : row + layout.nx] = scenario.initial_states[agent]
        upper[row : row + layout.nx] = scenario.initial_states[agent]
        row += layout.nx
        for step in range(config.horizon):
            current = layout.state(agent, step)
            following = layout.state(agent, step + 1)
            control = layout.control(agent, step)
            constraint[row : row + layout.nx, following] = np.eye(layout.nx)
            constraint[row : row + layout.nx, current] = -dynamics.A
            constraint[row : row + layout.nx, control] = -dynamics.B
            row += layout.nx
    for agent in range(graph.n_agents):
        for step in range(config.horizon):
            control = layout.control(agent, step)
            constraint[row : row + layout.nu, control] = np.eye(layout.nu)
            lower[row : row + layout.nu] = config.acceleration_lower
            upper[row : row + layout.nu] = config.acceleration_upper
            row += layout.nu
    return (
        sparse.triu(p).tocsc(),
        q,
        constraint.tocsc(),
        lower,
        upper,
        layout,
    )


def solve_centralized_qp(scenario: Scenario) -> CentralizedResult:
    """Solve the complete coupled formation QP with OSQP."""
    p, q, constraint, lower, upper, layout = _assemble_problem(scenario)
    solver = osqp.OSQP()
    solver.setup(
        P=p,
        q=q,
        A=constraint,
        l=lower,
        u=upper,
        eps_abs=1e-10,
        eps_rel=1e-10,
        max_iter=100_000,
        polishing=True,
        verbose=False,
    )
    raw = solver.solve(raise_error=True)
    status = raw.info.status.lower()
    if status != "solved" or raw.x is None:
        raise RuntimeError(f"centralized QP failed: {raw.info.status}")
    states = np.empty(
        (layout.n_agents, layout.horizon + 1, layout.nx), dtype=float
    )
    controls = np.empty(
        (layout.n_agents, layout.horizon, layout.nu), dtype=float
    )
    for agent in range(layout.n_agents):
        for step in range(layout.horizon + 1):
            states[agent, step] = raw.x[layout.state(agent, step)]
        for step in range(layout.horizon):
            controls[agent, step] = raw.x[layout.control(agent, step)]
    dynamics_residual = states[:, 1:] - (
        np.einsum("ab,ntb->nta", scenario.dynamics.A, states[:, :-1])
        + np.einsum("ab,ntb->nta", scenario.dynamics.B, controls)
    )
    lower_violation = np.maximum(
        scenario.config.acceleration_lower - controls, 0.0
    )
    upper_violation = np.maximum(
        controls - scenario.config.acceleration_upper, 0.0
    )
    return CentralizedResult(
        status=status,
        states=states,
        controls=controls,
        objective=evaluate_global_objective(scenario, states, controls),
        max_dynamics_residual=float(np.max(np.abs(dynamics_residual))),
        max_input_violation=float(max(np.max(lower_violation), np.max(upper_violation))),
    )


def evaluate_global_objective(
    scenario: Scenario, states: FloatArray, controls: FloatArray
) -> float:
    """Evaluate the exact documented objective, including constant terms."""
    total = 0.0
    stage_weight = scenario.weights.stage_state
    for agent in range(scenario.graph.n_agents):
        reference = scenario.agent_reference(agent)
        for step in range(scenario.config.horizon):
            error = states[agent, step] - reference[step]
            total += 0.5 * float(error @ stage_weight @ error)
            control = controls[agent, step]
            total += 0.5 * float(control @ scenario.weights.r_input @ control)
        terminal_error = states[agent, -1] - reference[-1]
        total += 0.5 * float(
            terminal_error @ scenario.weights.p_terminal @ terminal_error
        )
    for edge_id, (agent_i, agent_j) in enumerate(scenario.graph.edges):
        aligned_i = (
            states[agent_i] @ scenario.dynamics.C.T
            - scenario.graph.offsets[agent_i]
        )
        aligned_j = (
            states[agent_j] @ scenario.dynamics.C.T
            - scenario.graph.offsets[agent_j]
        )
        difference = aligned_i - aligned_j
        coupling = scenario.weights.gamma * scenario.graph.edge_weights[edge_id]
        total += 0.5 * coupling * float(
            np.einsum(
                "ti,ij,tj->", difference, scenario.weights.edge_metric, difference
            )
        )
    return total
