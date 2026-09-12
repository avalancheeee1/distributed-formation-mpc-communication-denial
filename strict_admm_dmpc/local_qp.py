"""Cached per-agent convex QP for the strict ADMM primal update."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import osqp
from scipy import sparse

from strict_admm_dmpc.model import FloatArray, Scenario


@dataclass(frozen=True)
class LocalQPResult:
    states: FloatArray
    controls: FloatArray
    status: str

    def __post_init__(self) -> None:
        for name in ("states", "controls"):
            value = np.array(getattr(self, name), dtype=float, copy=True)
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain only finite values")
            value.flags.writeable = False
            object.__setattr__(self, name, value)


class LocalQPWorkspace:
    """Cache fixed OSQP matrices and update only vectors between iterations."""

    def __init__(self, scenario: Scenario, agent_id: int) -> None:
        self.agent_id = agent_id
        self.horizon = scenario.config.horizon
        self.nx = scenario.dynamics.state_dimension
        self.nu = scenario.dynamics.input_dimension
        self.ny = scenario.dynamics.output_dimension
        self.n_state = (self.horizon + 1) * self.nx
        self.size = self.n_state + self.horizon * self.nu
        self.edge_count = len(scenario.graph.incident_edges(agent_id))
        self._fixed_structure = self._structure_signature(scenario)
        p = self._assemble_hessian(scenario)
        constraint, lower, upper = self._assemble_constraints(scenario)
        zero_targets = tuple(
            np.zeros((self.horizon + 1, self.ny))
            for _ in range(self.edge_count)
        )
        self._solver = osqp.OSQP()
        self._solver.setup(
            P=sparse.triu(p).tocsc(),
            q=self._linear_term(scenario, zero_targets),
            A=constraint.tocsc(),
            l=lower,
            u=upper,
            eps_abs=1e-10,
            eps_rel=1e-10,
            max_iter=100_000,
            polishing=True,
            verbose=False,
        )

    def _structure_signature(self, scenario: Scenario) -> tuple[object, ...]:
        """Identify every value embedded in the cached Hessian/constraint matrix."""
        arrays = (
            scenario.dynamics.A,
            scenario.dynamics.B,
            scenario.dynamics.C,
            scenario.weights.stage_state,
            scenario.weights.r_input,
            scenario.weights.p_terminal,
        )
        return (
            scenario.config.horizon,
            scenario.config.rho,
            scenario.dynamics.state_dimension,
            scenario.dynamics.input_dimension,
            scenario.dynamics.output_dimension,
            len(scenario.graph.incident_edges(self.agent_id)),
            *(array.shape for array in arrays),
            *(array.tobytes() for array in arrays),
        )

    def _state_slice(self, step: int) -> slice:
        return slice(step * self.nx, (step + 1) * self.nx)

    def _control_slice(self, step: int) -> slice:
        start = self.n_state + step * self.nu
        return slice(start, start + self.nu)

    def _assemble_hessian(self, scenario: Scenario) -> sparse.lil_matrix:
        p = sparse.lil_matrix((self.size, self.size), dtype=float)
        edge_hessian = scenario.config.rho * (
            scenario.dynamics.C.T @ scenario.dynamics.C
        )
        for step in range(self.horizon + 1):
            state = self._state_slice(step)
            state_weight = (
                scenario.weights.p_terminal
                if step == self.horizon
                else scenario.weights.stage_state
            )
            p[state, state] = (
                p[state, state] + state_weight + self.edge_count * edge_hessian
            )
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
        self, scenario: Scenario, edge_targets: tuple[FloatArray, ...]
    ) -> FloatArray:
        if len(edge_targets) != self.edge_count:
            raise ValueError("edge target count does not match incident edges")
        for target in edge_targets:
            if target.shape != (self.horizon + 1, self.ny) or not np.all(
                np.isfinite(target)
            ):
                raise ValueError("edge target has incompatible dimensions")
        q = np.zeros(self.size)
        reference = scenario.agent_reference(self.agent_id)
        c = scenario.dynamics.C
        offset = scenario.graph.offsets[self.agent_id]
        for step in range(self.horizon + 1):
            state = self._state_slice(step)
            state_weight = (
                scenario.weights.p_terminal
                if step == self.horizon
                else scenario.weights.stage_state
            )
            q[state] -= state_weight @ reference[step]
            for target in edge_targets:
                q[state] -= scenario.config.rho * c.T @ (offset + target[step])
        return q

    def _bounds(self, scenario: Scenario) -> tuple[FloatArray, FloatArray]:
        _, lower, upper = self._assemble_constraints(scenario)
        return lower, upper

    def solve(
        self, scenario: Scenario, edge_targets: tuple[FloatArray, ...]
    ) -> LocalQPResult:
        """Update the changing vectors and use OSQP's retained warm start."""
        if self._structure_signature(scenario) != self._fixed_structure:
            raise ValueError("scenario fixed structure differs from QP workspace")
        lower, upper = self._bounds(scenario)
        self._solver.update(
            q=self._linear_term(scenario, edge_targets), l=lower, u=upper
        )
        raw = self._solver.solve(raise_error=True)
        status = raw.info.status.lower()
        if status != "solved" or raw.x is None:
            raise RuntimeError(
                f"local QP failed for agent {self.agent_id}: {raw.info.status}"
            )
        states = np.vstack(
            [raw.x[self._state_slice(step)] for step in range(self.horizon + 1)]
        )
        controls = np.vstack(
            [raw.x[self._control_slice(step)] for step in range(self.horizon)]
        )
        return LocalQPResult(states=states, controls=controls, status=status)


def solve_local_qp(
    scenario: Scenario,
    agent_id: int,
    edge_targets: tuple[FloatArray, ...],
) -> LocalQPResult:
    """Convenience wrapper for one standalone local-QP solve."""
    return LocalQPWorkspace(scenario, agent_id).solve(scenario, edge_targets)
