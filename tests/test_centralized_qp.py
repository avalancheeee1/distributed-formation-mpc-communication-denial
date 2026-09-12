from __future__ import annotations

from dataclasses import replace

import numpy as np

from strict_admm_dmpc.centralized_qp import (
    evaluate_global_objective,
    solve_centralized_qp,
)


def test_centralized_solution_is_feasible(two_agent_scenario) -> None:
    result = solve_centralized_qp(two_agent_scenario)

    assert result.status == "solved"
    assert result.max_dynamics_residual <= 1e-7
    assert result.max_input_violation <= 1e-9
    assert np.isfinite(result.objective)


def test_reported_objective_matches_direct_evaluation(two_agent_scenario) -> None:
    result = solve_centralized_qp(two_agent_scenario)

    direct = evaluate_global_objective(
        two_agent_scenario, result.states, result.controls
    )

    np.testing.assert_allclose(result.objective, direct, rtol=1e-7, atol=1e-8)


def test_zero_edge_weight_matches_uncoupled_problem(two_agent_scenario) -> None:
    graph = replace(
        two_agent_scenario.graph,
        edge_weights=np.zeros_like(two_agent_scenario.graph.edge_weights),
    )
    uncoupled = replace(two_agent_scenario, graph=graph)

    result = solve_centralized_qp(uncoupled)
    direct = evaluate_global_objective(uncoupled, result.states, result.controls)

    np.testing.assert_allclose(result.objective, direct, rtol=1e-7, atol=1e-8)


def test_translation_of_reference_and_states_preserves_controls(
    two_agent_scenario,
) -> None:
    base = solve_centralized_qp(two_agent_scenario)
    translation = np.array([4.0, -3.0])
    shifted_initial = two_agent_scenario.initial_states.copy()
    shifted_initial[:, :2] += translation
    shifted_reference = two_agent_scenario.common_reference.copy()
    shifted_reference[:, :2] += translation
    shifted = replace(
        two_agent_scenario,
        initial_states=shifted_initial,
        common_reference=shifted_reference,
    )

    translated = solve_centralized_qp(shifted)

    np.testing.assert_allclose(base.controls, translated.controls, atol=2e-6)

