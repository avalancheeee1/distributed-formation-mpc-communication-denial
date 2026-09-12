"""Plain Lagrangian dual decomposition (subgradient) baseline.

This is the canonical distributed alternative to the augmented-Lagrangian
edge-splitting ADMM of :mod:`strict_admm_dmpc.admm`: the same consensus
construction but with the penalty ``rho = 0``.  Each edge enforces consensus
only through a linear Lagrange multiplier that is advanced by a subgradient
(here a gradient, because every subproblem has a unique minimizer) step, so
the coupling enters each agent's local QP as a *linear* term rather than the
augmented quadratic term that gives ADMM its practical speed.

The comparison isolates exactly one ingredient: the augmented/proximal term.
Both methods solve the same quadratic program on the same graph; they differ
only in whether the consensus constraint is enforced by a quadratic penalty
(ADMM) or a linear multiplier advanced by a step-size schedule (dual
decomposition).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import osqp
from scipy import sparse

from strict_admm_dmpc.centralized_qp import evaluate_global_objective
from strict_admm_dmpc.model import FloatArray, Scenario


@dataclass(frozen=True)
class DualDecompositionIteration:
    iteration: int
    objective: float
    objective_gap: float
    primal_residual: float


@dataclass(frozen=True)
class DualDecompositionResult:
    states: FloatArray
    controls: FloatArray
    objective: float
    iterations: int
    primal_residual: float
    history: tuple[DualDecompositionIteration, ...]

    def __post_init__(self) -> None:
        for name in ("states", "controls"):
            value = np.array(getattr(self, name), dtype=float, copy=True)
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain only finite values")
            value.flags.writeable = False
            object.__setattr__(self, name, value)


class _AgentPrimalWorkspace:
    """Cached per-agent QP with a linear dual term and no augmented coupling.

    Mirrors :class:`strict_admm_dmpc.local_qp.LocalQPWorkspace` but drops the
    ``rho * C.T @ C`` state Hessian and the ``rho``-scaled linear target,
    replacing them with the linear term ``C.T @ mu_sum`` that the dual
    decomposition Lagrangian contributes to agent ``agent_id``.
    """

    def __init__(self, scenario: Scenario, agent_id: int) -> None:
        self.agent_id = agent_id
        self.horizon = scenario.config.horizon
        self.nx = scenario.dynamics.state_dimension
        self.nu = scenario.dynamics.input_dimension
        self.n_state = (self.horizon + 1) * self.nx
        self.size = self.n_state + self.horizon * self.nu
        p = self._assemble_hessian(scenario)
        constraint, lower, upper = self._assemble_constraints(scenario)
        self._solver = osqp.OSQP()
        self._solver.setup(
            P=sparse.triu(p).tocsc(),
            q=self._linear_term(scenario, np.zeros(self.size)),
            A=constraint.tocsc(),
            l=lower,
            u=upper,
            eps_abs=1e-10,
            eps_rel=1e-10,
            max_iter=100_000,
            polishing=True,
            verbose=False,
        )

    def _state_slice(self, step: int) -> slice:
        return slice(step * self.nx, (step + 1) * self.nx)

    def _control_slice(self, step: int) -> slice:
        start = self.n_state + step * self.nu
        return slice(start, start + self.nu)

    def _assemble_hessian(self, scenario: Scenario) -> sparse.lil_matrix:
        p = sparse.lil_matrix((self.size, self.size), dtype=float)
        for step in range(self.horizon + 1):
            state = self._state_slice(step)
            state_weight = (
                scenario.weights.p_terminal
                if step == self.horizon
                else scenario.weights.stage_state
            )
            p[state, state] = p[state, state] + state_weight
        for step in range(self.horizon):
            control = self._control_slice(step)
            p[control, control] = p[control, control] + scenario.weights.r_input
        return p

    def _assemble_constraints(
        self, scenario: Scenario
    ) -> tuple[sparse.lil_matrix, FloatArray, FloatArray]:
        equality_rows = self.nx + self.horizon * self.nx
        bound_rows = self.horizon * self.nu
        constraint = sparse.lil_matrix((equality_rows + bound_rows, self.size))
        lower = np.zeros(equality_rows + bound_rows)
        upper = np.zeros(equality_rows + bound_rows)
        row = 0
        constraint[row : row + self.nx, self._state_slice(0)] = np.eye(self.nx)
        lower[row : row + self.nx] = scenario.initial_states[self.agent_id]
        upper[row : row + self.nx] = scenario.initial_states[self.agent_id]
        row += self.nx
        for step in range(self.horizon):
            constraint[row : row + self.nx, self._state_slice(step + 1)] = np.eye(
                self.nx
            )
            constraint[row : row + self.nx, self._state_slice(step)] = (
                -scenario.dynamics.A
            )
            constraint[row : row + self.nx, self._control_slice(step)] = (
                -scenario.dynamics.B
            )
            row += self.nx
        for step in range(self.horizon):
            constraint[row : row + self.nu, self._control_slice(step)] = np.eye(
                self.nu
            )
            lower[row : row + self.nu] = scenario.config.acceleration_lower
            upper[row : row + self.nu] = scenario.config.acceleration_upper
            row += self.nu
        return constraint, lower, upper

    def _linear_term(
        self, scenario: Scenario, dual_state_term: FloatArray
    ) -> FloatArray:
        """Tracking term plus the linear dual contribution ``C.T @ mu_sum``."""
        q = np.zeros(self.size)
        reference = scenario.agent_reference(self.agent_id)
        for step in range(self.horizon + 1):
            state = self._state_slice(step)
            state_weight = (
                scenario.weights.p_terminal
                if step == self.horizon
                else scenario.weights.stage_state
            )
            q[state] -= state_weight @ reference[step]
        q += dual_state_term
        return q

    def solve(
        self, scenario: Scenario, dual_state_term: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        self._solver.update(q=self._linear_term(scenario, dual_state_term))
        raw = self._solver.solve(raise_error=True)
        if raw.info.status.lower() != "solved" or raw.x is None:
            raise RuntimeError(
                f"dual-decomposition primal QP failed for agent "
                f"{self.agent_id}: {raw.info.status}"
            )
        states = np.vstack(
            [raw.x[self._state_slice(step)] for step in range(self.horizon + 1)]
        )
        controls = np.vstack(
            [raw.x[self._control_slice(step)] for step in range(self.horizon)]
        )
        return states, controls


def _edge_sign(edge: tuple[int, int], agent_id: int) -> float:
    """+1 if ``agent_id`` is the first (canonical) endpoint, else -1."""
    return 1.0 if agent_id == edge[0] else -1.0


def solve_dual_decomposition(
    scenario: Scenario,
    step_size: float,
    schedule: str = "diminishing",
    max_iterations: int = 20_000,
    central_objective: float | None = None,
) -> DualDecompositionResult:
    """Solve the coupled QP by plain dual decomposition with a step schedule.

    The edge multiplier ``mu_e`` is conjugate to the single difference
    consensus ``s_{e,i} - s_{e,j} = delta_e`` (only the difference matters to
    the edge cost), so there is one multiplier per edge and no gauge freedom.
    Each iteration runs one per-agent primal solve and one closed-form
    ``delta``/dual update.  The multiplier starts at zero and is advanced by
    ``mu_e += step_k * ((s_i - s_j) - delta_e)``, the subgradient of the dual
    function.

    ``schedule`` selects the step-size sequence: ``"diminishing"`` uses
    ``step_k = step_size / sqrt(k)`` (square-summable but not summable, the
    standard guaranteed-convergent subgradient schedule), while ``"constant"``
    uses ``step_k = step_size`` for every iteration.
    """
    if not np.isfinite(step_size) or step_size <= 0.0:
        raise ValueError("step_size must be finite and positive")
    if schedule not in ("diminishing", "constant"):
        raise ValueError("schedule must be 'diminishing' or 'constant'")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")

    n_agents = scenario.graph.n_agents
    horizon = scenario.config.horizon
    ny = scenario.dynamics.output_dimension
    nx = scenario.dynamics.state_dimension
    shape = (horizon + 1, ny)
    offsets = scenario.graph.offsets
    c = scenario.dynamics.C
    gamma = scenario.weights.gamma

    workspaces = [
        _AgentPrimalWorkspace(scenario, agent) for agent in range(n_agents)
    ]
    # One multiplier per edge, stacked over the horizon.
    mu = [
        np.zeros(shape) for _ in scenario.graph.edges
    ]
    states = np.zeros((n_agents, horizon + 1, nx))
    controls = np.zeros((n_agents, horizon, scenario.dynamics.input_dimension))

    if central_objective is None:
        from strict_admm_dmpc.centralized_qp import solve_centralized_qp

        central_objective = solve_centralized_qp(scenario).objective

    history: list[DualDecompositionIteration] = []
    aligned = np.zeros((n_agents, horizon + 1, ny))
    best_objective = np.inf
    best_states = states.copy()
    best_controls = controls.copy()

    for iteration in range(1, max_iterations + 1):
        # --- per-agent primal solves -------------------------------------
        for agent in range(n_agents):
            # accumulate C^T @ mu over incident edges with the right sign;
            # the dual enters only the state coordinates of the local vector.
            mu_sum = np.zeros(shape)
            for edge_id in scenario.graph.incident_edges(agent):
                sign = _edge_sign(scenario.graph.edges[edge_id], agent)
                mu_sum = mu_sum + sign * mu[edge_id]
            term = np.zeros(workspaces[agent].size)
            cT_mu = c.T @ mu_sum.T  # shape (nx, horizon + 1)
            for step in range(horizon + 1):
                sl = slice(step * nx, (step + 1) * nx)
                term[sl] = cT_mu[:, step]
            states[agent], controls[agent] = workspaces[agent].solve(
                scenario, term
            )

        # --- aligned outputs --------------------------------------------
        aligned = states @ c.T - offsets[:, None, :]

        # --- delta + dual update ----------------------------------------
        step_k = (
            step_size / np.sqrt(iteration)
            if schedule == "diminishing"
            else step_size
        )
        primal_parts = []
        for edge_id, (agent_i, agent_j) in enumerate(scenario.graph.edges):
            coupling = gamma * scenario.graph.edge_weights[edge_id]
            # delta = W^{-1} mu / coupling; edge_metric == I in the study.
            metric = scenario.weights.edge_metric
            delta = np.linalg.solve(
                coupling * metric, mu[edge_id].T
            ).T
            residual = aligned[agent_i] - aligned[agent_j] - delta
            mu[edge_id] = mu[edge_id] + step_k * residual
            primal_parts.append(residual.ravel())

        objective = evaluate_global_objective(scenario, states, controls)
        primal_residual = float(
            np.linalg.norm(np.concatenate(primal_parts))
            if primal_parts
            else 0.0
        )
        if objective < best_objective:
            best_objective = objective
            best_states = states.copy()
            best_controls = controls.copy()
        history.append(
            DualDecompositionIteration(
                iteration=iteration,
                objective=objective,
                objective_gap=float(
                    (objective - central_objective)
                    / max(1.0, abs(central_objective))
                ),
                primal_residual=primal_residual,
            )
        )

    return DualDecompositionResult(
        states=best_states,
        controls=best_controls,
        objective=best_objective,
        iterations=max_iterations,
        primal_residual=history[-1].primal_residual,
        history=tuple(history),
    )
