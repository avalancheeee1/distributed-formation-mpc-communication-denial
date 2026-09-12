from __future__ import annotations

import numpy as np

from strict_admm_dmpc.admm import solve_strict_admm
from strict_admm_dmpc.centralized_qp import solve_centralized_qp


def test_admm_matches_centralized_solution(two_agent_scenario) -> None:
    centralized = solve_centralized_qp(two_agent_scenario)
    distributed = solve_strict_admm(two_agent_scenario)

    assert distributed.converged
    assert distributed.primal_residual <= 1e-6
    relative_objective_gap = abs(distributed.objective - centralized.objective) / max(
        1.0, abs(centralized.objective)
    )
    relative_state_error = np.linalg.norm(
        distributed.states - centralized.states
    ) / max(1.0, np.linalg.norm(centralized.states))
    relative_control_error = np.linalg.norm(
        distributed.controls - centralized.controls
    ) / max(1.0, np.linalg.norm(centralized.controls))

    assert relative_objective_gap <= 1e-5
    assert relative_state_error <= 1e-5
    assert relative_control_error <= 1e-5


def test_residual_history_contains_standard_stopping_terms(
    two_agent_scenario,
) -> None:
    distributed = solve_strict_admm(two_agent_scenario)
    last = distributed.history[-1]

    assert last.primal_residual >= 0.0
    assert last.dual_residual >= 0.0
    assert last.primal_tolerance > 0.0
    assert last.dual_tolerance > 0.0
    assert np.isfinite(last.objective)


def test_inexact_messages_never_claim_exact_convergence(two_agent_scenario) -> None:
    shape = (
        two_agent_scenario.config.horizon + 1,
        two_agent_scenario.dynamics.output_dimension,
    )
    remote = {0: (np.full(shape, 10.0), np.full(shape, -10.0))}

    result = solve_strict_admm(two_agent_scenario, remote_arguments=remote)

    assert not result.exact_communication
    assert not result.converged
    assert result.residual_stopped
