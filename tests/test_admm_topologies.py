from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from strict_admm_dmpc.admm import solve_strict_admm
from strict_admm_dmpc.centralized_qp import solve_centralized_qp
from strict_admm_dmpc.model import FormationGraph


TOPOLOGIES = {
    "chain": ((0, 1), (1, 2), (2, 3)),
    "ring": ((0, 1), (1, 2), (2, 3), (0, 3)),
    "connected_random": ((0, 1), (1, 2), (2, 3), (0, 2)),
}


@pytest.mark.parametrize("topology", tuple(TOPOLOGIES))
@pytest.mark.parametrize("seed", range(10))
def test_admm_matches_centralized_across_topologies(
    two_agent_scenario, topology: str, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    edges = TOPOLOGIES[topology]
    offsets = np.column_stack([np.arange(4, dtype=float), np.zeros(4)])
    graph = FormationGraph(
        n_agents=4,
        edges=edges,
        offsets=offsets,
        edge_weights=np.linspace(0.5, 1.0, len(edges)),
    )
    initial_states = np.zeros((4, 4))
    initial_states[:, :2] = offsets + rng.normal(0.0, 0.4, size=(4, 2))
    initial_states[:, 2:] = rng.normal(0.0, 0.15, size=(4, 2))
    config = replace(
        two_agent_scenario.config,
        horizon=3,
        absolute_tolerance=3e-7,
        relative_tolerance=3e-7,
    )
    scenario = replace(
        two_agent_scenario,
        graph=graph,
        initial_states=initial_states,
        common_reference=np.zeros((config.horizon + 1, 4)),
        config=config,
        seed=seed,
    )

    centralized = solve_centralized_qp(scenario)
    distributed = solve_strict_admm(scenario)
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

    assert distributed.converged
    assert distributed.primal_residual <= 1e-5
    assert objective_gap <= 2e-5
    assert decision_gap <= 2e-5

