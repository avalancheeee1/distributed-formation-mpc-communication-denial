from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from strict_admm_dmpc.model import (
    FormationGraph,
    LinearDynamics,
    MPCConfig,
    build_double_integrator,
)


def test_double_integrator_uses_control_in_position_after_two_steps() -> None:
    dynamics = build_double_integrator(0.1)
    state = np.array([0.0, 0.0, 0.0, 0.0])
    control = np.array([2.0, -1.0])

    state_1 = dynamics.A @ state + dynamics.B @ control
    state_2 = dynamics.A @ state_1 + dynamics.B @ control

    np.testing.assert_allclose(state_1[:2], [0.0, 0.0])
    np.testing.assert_allclose(state_2[:2], [0.02, -0.01])


def test_graph_canonicalizes_edges_and_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        FormationGraph(
            n_agents=2,
            edges=((0, 1), (1, 0)),
            offsets=np.zeros((2, 2)),
            edge_weights=np.ones(2),
        )


def test_graph_requires_positive_semidefinite_edge_weights() -> None:
    with pytest.raises(ValueError, match="edge_weights"):
        FormationGraph(
            n_agents=2,
            edges=((0, 1),),
            offsets=np.zeros((2, 2)),
            edge_weights=np.array([-0.1]),
        )


def test_graph_rejects_confidence_above_one() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        FormationGraph(
            n_agents=2,
            edges=((0, 1),),
            offsets=np.zeros((2, 2)),
            edge_weights=np.array([1.1]),
        )


def test_graph_requires_connectivity() -> None:
    with pytest.raises(ValueError, match="connected"):
        FormationGraph(
            n_agents=3,
            edges=((0, 1),),
            offsets=np.zeros((3, 2)),
            edge_weights=np.ones(1),
        )


def test_validated_arrays_are_immutable(two_agent_scenario) -> None:
    with pytest.raises(ValueError, match="read-only"):
        two_agent_scenario.initial_states[0, 0] = 999.0
    with pytest.raises(ValueError, match="read-only"):
        two_agent_scenario.graph.edge_weights[0] = 0.0


def test_scenario_rejects_output_matrix_incompatible_with_offset_layout(
    two_agent_scenario,
) -> None:
    dynamics = LinearDynamics(
        A=two_agent_scenario.dynamics.A,
        B=two_agent_scenario.dynamics.B,
        C=np.array([[1.0, 0.0, 0.2, 0.0], [0.0, 1.0, 0.0, 0.2]]),
        dt=two_agent_scenario.dynamics.dt,
    )

    with pytest.raises(ValueError, match="C must select the leading formation output"):
        replace(two_agent_scenario, dynamics=dynamics)


@pytest.mark.parametrize("field", ["absolute_tolerance", "relative_tolerance"])
def test_config_rejects_nonfinite_tolerances(field: str) -> None:
    kwargs = {
        "horizon": 3,
        "rho": 1.0,
        "acceleration_lower": np.array([-1.0, -1.0]),
        "acceleration_upper": np.array([1.0, 1.0]),
        field: float("nan"),
    }
    with pytest.raises(ValueError, match="tolerances"):
        MPCConfig(**kwargs)


@pytest.mark.parametrize("field,value", [("horizon", 2.5), ("max_iterations", True)])
def test_config_requires_integer_counts(field: str, value: object) -> None:
    kwargs = {
        "horizon": 3,
        "rho": 1.0,
        "acceleration_lower": np.array([-1.0, -1.0]),
        "acceleration_upper": np.array([1.0, 1.0]),
        "max_iterations": 10,
        field: value,
    }
    with pytest.raises(ValueError, match="integer"):
        MPCConfig(**kwargs)
