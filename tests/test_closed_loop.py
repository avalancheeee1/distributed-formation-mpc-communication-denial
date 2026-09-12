from __future__ import annotations

from dataclasses import replace

import numpy as np

from strict_admm_dmpc.centralized_qp import solve_centralized_qp
from strict_admm_dmpc.closed_loop import (
    simulate_closed_loop,
    simulate_uncoupled_mpc,
)
from strict_admm_dmpc.communication import DenialSchedule, PredictionMode


def test_closed_loop_applies_all_controls_synchronously(two_agent_scenario) -> None:
    forward = simulate_closed_loop(
        two_agent_scenario,
        steps=3,
        admm_cap=4,
        denial=DenialSchedule(start=100.0, end=101.0),
        prediction_mode=PredictionMode.CONSTANT_VELOCITY,
        agent_order=(0, 1),
    )
    reverse = simulate_closed_loop(
        two_agent_scenario,
        steps=3,
        admm_cap=4,
        denial=DenialSchedule(start=100.0, end=101.0),
        prediction_mode=PredictionMode.CONSTANT_VELOCITY,
        agent_order=(1, 0),
    )

    np.testing.assert_allclose(forward.states, reverse.states, atol=1e-10)
    np.testing.assert_allclose(forward.controls, reverse.controls, atol=1e-10)


def test_denial_uses_zero_real_communication_and_common_reference(
    two_agent_scenario,
) -> None:
    denial = DenialSchedule(start=0.2, end=0.6)
    result = simulate_closed_loop(
        two_agent_scenario,
        steps=4,
        admm_cap=3,
        denial=denial,
        prediction_mode=PredictionMode.ZERO_ORDER_HOLD,
    )

    denied = (result.time >= denial.start) & (result.time < denial.end)
    assert np.all(result.communicated_scalars[denied] == 0)
    expected = formation_rmse_for_result(result, two_agent_scenario.graph.offsets)
    np.testing.assert_allclose(result.formation_rmse, expected)


def formation_rmse_for_result(result, offsets) -> np.ndarray:
    desired = result.reference[:, None, :2] + offsets[None, :, :]
    error = result.states[:, :, :2] - desired
    return np.sqrt(np.mean(np.sum(error**2, axis=2), axis=1))


def test_paired_run_changes_only_controller_cap(two_agent_scenario) -> None:
    denial = DenialSchedule(start=0.2, end=0.6)
    short = simulate_closed_loop(
        two_agent_scenario,
        steps=4,
        admm_cap=1,
        denial=denial,
        prediction_mode=PredictionMode.CONSTANT_VELOCITY,
    )
    long = simulate_closed_loop(
        two_agent_scenario,
        steps=4,
        admm_cap=5,
        denial=denial,
        prediction_mode=PredictionMode.CONSTANT_VELOCITY,
    )

    assert short.scenario_hash == long.scenario_hash
    np.testing.assert_allclose(short.reference, long.reference)
    np.testing.assert_allclose(short.states[0], long.states[0])


def test_uncoupled_baseline_solves_gamma_zero_optimum(two_agent_scenario) -> None:
    uncoupled = replace(
        two_agent_scenario,
        weights=replace(two_agent_scenario.weights, gamma=0.0),
    )
    expected = solve_centralized_qp(uncoupled)

    result = simulate_uncoupled_mpc(
        two_agent_scenario,
        steps=1,
        denial=DenialSchedule(start=0.2, end=0.6),
    )

    np.testing.assert_allclose(result.controls[0], expected.controls[:, 0])
    assert result.communicated_scalars[0] == 0
