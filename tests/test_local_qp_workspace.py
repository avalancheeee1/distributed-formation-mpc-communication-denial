from __future__ import annotations

from dataclasses import replace

import numpy as np
import osqp
import pytest

from strict_admm_dmpc.admm import solve_strict_admm
from strict_admm_dmpc.local_qp import LocalQPWorkspace


def test_local_osqp_setup_occurs_once_per_agent(monkeypatch, two_agent_scenario) -> None:
    setup_calls = 0
    original_setup = osqp.OSQP.setup

    def counted_setup(self, *args, **kwargs):
        nonlocal setup_calls
        setup_calls += 1
        return original_setup(self, *args, **kwargs)

    monkeypatch.setattr(osqp.OSQP, "setup", counted_setup)
    scenario = replace(
        two_agent_scenario,
        config=replace(two_agent_scenario.config, max_iterations=5),
    )

    solve_strict_admm(scenario)

    assert setup_calls == scenario.graph.n_agents


def test_workspace_rejects_changed_fixed_structure(two_agent_scenario) -> None:
    workspace = LocalQPWorkspace(two_agent_scenario, agent_id=0)
    changed = replace(
        two_agent_scenario,
        config=replace(two_agent_scenario.config, rho=5.0),
    )
    targets = tuple(
        np.zeros((changed.config.horizon + 1, changed.dynamics.output_dimension))
        for _ in changed.graph.incident_edges(0)
    )

    with pytest.raises(ValueError, match="fixed structure"):
        workspace.solve(changed, targets)
