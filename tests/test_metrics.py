from __future__ import annotations

import numpy as np

from strict_admm_dmpc.metrics import formation_rmse, sustained_settling_time


def test_formation_rmse_uses_shared_ground_truth_reference() -> None:
    states = np.zeros((3, 2, 4))
    states[:, :, 0] = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    offsets = np.array([[0.0, 0.0], [1.0, 0.0]])
    reference = np.zeros((3, 4))
    reference[:, 0] = [0.0, 1.0, 2.0]

    rmse = formation_rmse(states, reference, offsets)

    np.testing.assert_allclose(rmse, 0.0)


def test_formation_rmse_supports_scalar_output() -> None:
    states = np.array([[[1.0, 0.0], [3.0, 0.0]]])
    reference = np.array([[1.0, 0.0]])
    offsets = np.array([[0.0], [2.0]])

    np.testing.assert_allclose(formation_rmse(states, reference, offsets), 0.0)


def test_settling_time_requires_sustained_residence() -> None:
    time = np.arange(8, dtype=float)
    error = np.array([2.0, 0.4, 0.3, 1.2, 0.4, 0.3, 0.2, 0.1])

    settling = sustained_settling_time(
        time, error, start_time=1.0, tolerance=0.5, residence_time=2.0
    )

    assert settling == 4.0
