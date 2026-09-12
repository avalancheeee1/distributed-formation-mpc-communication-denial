from __future__ import annotations

import numpy as np
import pytest

from strict_admm_dmpc.admm import edge_prox_update


def test_edge_update_matches_direct_linear_system() -> None:
    rng = np.random.default_rng(12)
    a = rng.normal(size=(5, 2))
    b = rng.normal(size=(5, 2))
    metric = np.array([[2.0, 0.3], [0.3, 1.0]])
    rho = 0.7
    coupling = 1.4

    z_i, z_j = edge_prox_update(a, b, rho, coupling, metric)

    block = np.block(
        [
            [rho * np.eye(2) + coupling * metric, -coupling * metric],
            [-coupling * metric, rho * np.eye(2) + coupling * metric],
        ]
    )
    for step in range(a.shape[0]):
        direct = np.linalg.solve(block, rho * np.concatenate([a[step], b[step]]))
        np.testing.assert_allclose(z_i[step], direct[:2], atol=1e-11)
        np.testing.assert_allclose(z_j[step], direct[2:], atol=1e-11)


def test_edge_update_is_endpoint_symmetric() -> None:
    a = np.array([[1.0, -2.0], [0.5, 0.2]])
    b = np.array([[-0.3, 0.8], [1.2, -0.4]])

    z_i, z_j = edge_prox_update(a, b, 1.0, 0.6, np.eye(2))
    swapped_j, swapped_i = edge_prox_update(b, a, 1.0, 0.6, np.eye(2))

    np.testing.assert_allclose(z_i, swapped_i)
    np.testing.assert_allclose(z_j, swapped_j)


def test_zero_coupling_returns_local_arguments() -> None:
    a = np.array([[1.0, 2.0]])
    b = np.array([[-1.0, 4.0]])

    z_i, z_j = edge_prox_update(a, b, 0.5, 0.0, np.eye(2))

    np.testing.assert_allclose(z_i, a)
    np.testing.assert_allclose(z_j, b)


@pytest.mark.parametrize(
    "metric",
    [
        np.array([[1.0, 2.0], [0.0, 1.0]]),
        np.array([[1.0, 0.0], [0.0, -0.1]]),
        np.array([[float("nan"), 0.0], [0.0, 1.0]]),
    ],
)
def test_edge_update_rejects_invalid_metric(metric: np.ndarray) -> None:
    with pytest.raises(ValueError, match="metric"):
        edge_prox_update(np.zeros((2, 2)), np.zeros((2, 2)), 1.0, 1.0, metric)
