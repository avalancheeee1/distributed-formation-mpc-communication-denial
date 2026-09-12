"""Validated data model for the strict convex formation MPC problem."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _as_float_array(value: FloatArray, name: str) -> FloatArray:
    array = np.array(value, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.flags.writeable = False
    return array


def _validate_symmetric_psd(matrix: FloatArray, name: str) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square")
    if not np.allclose(matrix, matrix.T, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    if np.min(np.linalg.eigvalsh(matrix)) < -1e-10:
        raise ValueError(f"{name} must be positive semidefinite")


@dataclass(frozen=True)
class LinearDynamics:
    """Discrete linear dynamics and the formation output matrix."""

    A: FloatArray
    B: FloatArray
    C: FloatArray
    dt: float

    def __post_init__(self) -> None:
        a = _as_float_array(self.A, "A")
        b = _as_float_array(self.B, "B")
        c = _as_float_array(self.C, "C")
        if a.ndim != 2 or a.shape[0] != a.shape[1]:
            raise ValueError("A must be square")
        if b.ndim != 2 or b.shape[0] != a.shape[0]:
            raise ValueError("B has incompatible dimensions")
        if c.ndim != 2 or c.shape[1] != a.shape[0]:
            raise ValueError("C has incompatible dimensions")
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be positive")
        object.__setattr__(self, "A", a)
        object.__setattr__(self, "B", b)
        object.__setattr__(self, "C", c)

    @property
    def state_dimension(self) -> int:
        return self.A.shape[0]

    @property
    def input_dimension(self) -> int:
        return self.B.shape[1]

    @property
    def output_dimension(self) -> int:
        return self.C.shape[0]


@dataclass(frozen=True)
class FormationGraph:
    """Undirected formation graph with one canonical orientation per edge."""

    n_agents: int
    edges: tuple[tuple[int, int], ...]
    offsets: FloatArray
    edge_weights: FloatArray

    def __post_init__(self) -> None:
        if not isinstance(self.n_agents, int) or isinstance(self.n_agents, bool):
            raise ValueError("n_agents must be an integer")
        if self.n_agents < 1:
            raise ValueError("n_agents must be positive")
        canonical: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for raw_i, raw_j in self.edges:
            if any(
                not isinstance(endpoint, int) or isinstance(endpoint, bool)
                for endpoint in (raw_i, raw_j)
            ):
                raise ValueError("edge endpoints must be integers")
            if raw_i == raw_j:
                raise ValueError("self edges are not allowed")
            if not (0 <= raw_i < self.n_agents and 0 <= raw_j < self.n_agents):
                raise ValueError("edge endpoint is out of range")
            edge = (min(raw_i, raw_j), max(raw_i, raw_j))
            if edge in seen:
                raise ValueError(f"duplicate edge {edge}")
            seen.add(edge)
            canonical.append(edge)
        offsets = _as_float_array(self.offsets, "offsets")
        weights = _as_float_array(self.edge_weights, "edge_weights")
        if offsets.ndim != 2 or offsets.shape[0] != self.n_agents:
            raise ValueError("offsets must have one row per agent")
        if (
            weights.shape != (len(canonical),)
            or np.any(weights < 0.0)
            or np.any(weights > 1.0)
        ):
            raise ValueError("edge_weights must lie in [0, 1] with one value per edge")
        if self.n_agents > 1:
            adjacency = [set() for _ in range(self.n_agents)]
            for endpoint_i, endpoint_j in canonical:
                adjacency[endpoint_i].add(endpoint_j)
                adjacency[endpoint_j].add(endpoint_i)
            reached = {0}
            frontier = [0]
            while frontier:
                current = frontier.pop()
                for neighbor in adjacency[current] - reached:
                    reached.add(neighbor)
                    frontier.append(neighbor)
            if len(reached) != self.n_agents:
                raise ValueError("formation graph must be connected")
        object.__setattr__(self, "edges", tuple(canonical))
        object.__setattr__(self, "offsets", offsets)
        object.__setattr__(self, "edge_weights", weights)

    def incident_edges(self, agent_id: int) -> tuple[int, ...]:
        return tuple(
            edge_id
            for edge_id, edge in enumerate(self.edges)
            if agent_id in edge
        )


@dataclass(frozen=True)
class MPCWeights:
    """Quadratic weights for tracking, actuation, terminal, and edge costs."""

    q_position: FloatArray
    q_velocity: FloatArray
    r_input: FloatArray
    p_terminal: FloatArray
    edge_metric: FloatArray
    gamma: float

    def __post_init__(self) -> None:
        for name in (
            "q_position",
            "q_velocity",
            "r_input",
            "p_terminal",
            "edge_metric",
        ):
            matrix = _as_float_array(getattr(self, name), name)
            _validate_symmetric_psd(matrix, name)
            object.__setattr__(self, name, matrix)
        if np.min(np.linalg.eigvalsh(self.r_input)) <= 0.0:
            raise ValueError("r_input must be positive definite")
        if not np.isfinite(self.gamma) or self.gamma < 0.0:
            raise ValueError("gamma must be nonnegative")

    @property
    def stage_state(self) -> FloatArray:
        zeros = np.zeros((self.q_position.shape[0], self.q_velocity.shape[0]))
        return np.block(
            [[self.q_position, zeros], [zeros.T, self.q_velocity]]
        )


@dataclass(frozen=True)
class MPCConfig:
    """Horizon, ADMM, solver, and input-bound configuration."""

    horizon: int
    rho: float
    acceleration_lower: FloatArray
    acceleration_upper: FloatArray
    absolute_tolerance: float = 1e-6
    relative_tolerance: float = 1e-6
    max_iterations: int = 1_000

    def __post_init__(self) -> None:
        lower = _as_float_array(self.acceleration_lower, "acceleration_lower")
        upper = _as_float_array(self.acceleration_upper, "acceleration_upper")
        if not isinstance(self.horizon, int) or isinstance(self.horizon, bool):
            raise ValueError("horizon must be an integer")
        if not isinstance(self.max_iterations, int) or isinstance(
            self.max_iterations, bool
        ):
            raise ValueError("max_iterations must be an integer")
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        if not np.isfinite(self.rho) or self.rho <= 0.0:
            raise ValueError("rho must be positive")
        if lower.ndim != 1 or upper.shape != lower.shape or np.any(lower >= upper):
            raise ValueError("acceleration bounds are invalid")
        if not np.isfinite(self.absolute_tolerance) or not np.isfinite(
            self.relative_tolerance
        ):
            raise ValueError("ADMM tolerances must be finite and positive")
        if self.absolute_tolerance <= 0.0 or self.relative_tolerance <= 0.0:
            raise ValueError("ADMM tolerances must be positive")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        object.__setattr__(self, "acceleration_lower", lower)
        object.__setattr__(self, "acceleration_upper", upper)


@dataclass(frozen=True)
class Scenario:
    """A deterministic finite-horizon formation MPC instance."""

    dynamics: LinearDynamics
    graph: FormationGraph
    weights: MPCWeights
    config: MPCConfig
    initial_states: FloatArray
    common_reference: FloatArray
    seed: int
    acceleration: FloatArray = field(default_factory=lambda: np.zeros(2))

    def __post_init__(self) -> None:
        initial = _as_float_array(self.initial_states, "initial_states")
        reference = _as_float_array(self.common_reference, "common_reference")
        nx = self.dynamics.state_dimension
        nu = self.dynamics.input_dimension
        ny = self.dynamics.output_dimension
        if initial.shape != (self.graph.n_agents, nx):
            raise ValueError("initial_states has incompatible dimensions")
        if reference.shape != (self.config.horizon + 1, nx):
            raise ValueError("common_reference has incompatible dimensions")
        if self.graph.offsets.shape[1] != ny:
            raise ValueError("formation offsets must match output dimension")
        if self.config.acceleration_lower.shape != (nu,):
            raise ValueError("acceleration bounds must match input dimension")
        if self.weights.stage_state.shape != (nx, nx):
            raise ValueError("state weights must match state dimension")
        if self.weights.q_position.shape != (ny, ny):
            raise ValueError("position weights must match formation output dimension")
        if self.weights.q_velocity.shape != (nx - ny, nx - ny):
            raise ValueError("remaining-state weights must match non-output dimension")
        if self.weights.r_input.shape != (nu, nu):
            raise ValueError("input weights must match input dimension")
        if self.weights.p_terminal.shape != (nx, nx):
            raise ValueError("terminal weight must match state dimension")
        if self.weights.edge_metric.shape != (ny, ny):
            raise ValueError("edge metric must match output dimension")
        if nx < ny:
            raise ValueError("state dimension must contain the formation output")
        expected_c = np.hstack([np.eye(ny), np.zeros((ny, nx - ny))])
        if not np.allclose(self.dynamics.C, expected_c, atol=1e-12):
            raise ValueError(
                "C must select the leading formation output coordinates"
            )
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        acceleration = _as_float_array(self.acceleration, "acceleration")
        if acceleration.shape != (ny,):
            raise ValueError("acceleration must match output dimension")
        object.__setattr__(self, "initial_states", initial)
        object.__setattr__(self, "common_reference", reference)
        object.__setattr__(self, "acceleration", acceleration)

    def agent_reference(self, agent_id: int) -> FloatArray:
        reference = self.common_reference.copy()
        reference[:, : self.dynamics.output_dimension] += self.graph.offsets[agent_id]
        return reference


def build_double_integrator(dt: float) -> LinearDynamics:
    """Return the forward-Euler planar double-integrator used in the study."""
    identity = np.eye(2)
    zeros = np.zeros((2, 2))
    a = np.block([[identity, dt * identity], [zeros, identity]])
    b = np.vstack([zeros, dt * identity])
    c = np.hstack([identity, zeros])
    return LinearDynamics(A=a, B=b, C=c, dt=dt)
