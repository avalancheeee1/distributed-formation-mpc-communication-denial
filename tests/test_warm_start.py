from __future__ import annotations

import numpy as np
import pytest

from strict_admm_dmpc.admm import EdgeState, shift_edge_state, solve_strict_admm


def test_shift_edge_state_moves_horizon_and_repeats_terminal() -> None:
    values = np.arange(10, dtype=float).reshape(5, 2)
    state = EdgeState(
        z_i=values.copy(),
        z_j=(values + 10.0).copy(),
        eta_i=(values + 20.0).copy(),
        eta_j=(values + 30.0).copy(),
    )

    shifted = shift_edge_state(state)

    np.testing.assert_allclose(shifted.z_i[:-1], values[1:])
    np.testing.assert_allclose(shifted.z_i[-1], values[-1])
    np.testing.assert_allclose(shifted.eta_j[:-1], (values + 30.0)[1:])
    np.testing.assert_allclose(shifted.eta_j[-1], (values + 30.0)[-1])


def test_empty_warm_start_is_rejected_for_nonempty_graph(two_agent_scenario) -> None:
    with pytest.raises(ValueError, match="one state per graph edge"):
        solve_strict_admm(two_agent_scenario, warm_start=())


def test_malformed_edge_state_is_rejected(two_agent_scenario) -> None:
    wrong = np.zeros((2, 2))
    malformed = EdgeState(wrong, wrong, wrong, wrong)
    with pytest.raises(ValueError, match="incompatible shape"):
        solve_strict_admm(two_agent_scenario, warm_start=(malformed,))

