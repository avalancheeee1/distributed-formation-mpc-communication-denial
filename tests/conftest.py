from __future__ import annotations

import numpy as np
import pytest

from strict_admm_dmpc.model import (
    FormationGraph,
    MPCConfig,
    MPCWeights,
    Scenario,
    build_double_integrator,
)


@pytest.fixture
def two_agent_scenario() -> Scenario:
    dynamics = build_double_integrator(0.2)
    graph = FormationGraph(
        n_agents=2,
        edges=((0, 1),),
        offsets=np.array([[0.0, 0.0], [1.0, 0.0]]),
        edge_weights=np.array([0.8]),
    )
    weights = MPCWeights(
        q_position=np.diag([2.0, 2.0]),
        q_velocity=np.diag([0.5, 0.5]),
        r_input=np.diag([0.1, 0.1]),
        p_terminal=np.diag([4.0, 4.0, 1.0, 1.0]),
        edge_metric=np.eye(2),
        gamma=1.5,
    )
    config = MPCConfig(
        horizon=4,
        rho=1.0,
        acceleration_lower=np.array([-3.0, -3.0]),
        acceleration_upper=np.array([3.0, 3.0]),
        absolute_tolerance=1e-7,
        relative_tolerance=1e-7,
        max_iterations=2_000,
    )
    initial_states = np.array(
        [
            [0.8, -0.3, 0.2, 0.0],
            [2.0, 0.5, -0.1, 0.1],
        ]
    )
    common_reference = np.zeros((config.horizon + 1, 4))
    return Scenario(
        dynamics=dynamics,
        graph=graph,
        weights=weights,
        config=config,
        initial_states=initial_states,
        common_reference=common_reference,
        seed=7,
    )

