from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from matrix_free_lanczos import (  # noqa: E402
    build_bttb_operator,
    lanczos_decomposition,
    sqrt_action,
)


def _covariance(h0, h1, h2):
    return np.exp(-np.sqrt(h0**2 + 0.7 * h1**2 + 1.3 * h2**2))


def _dense_covariance(shape):
    coordinates = np.stack(
        np.meshgrid(*[np.arange(size) for size in shape], indexing="ij"), axis=-1
    ).reshape(-1, 3)
    delta = coordinates[:, None, :] - coordinates[None, :, :]
    return _covariance(delta[..., 0], delta[..., 1], delta[..., 2])


def test_bttb_matvec_matches_dense_covariance():
    shape = (4, 5, 3)
    operator = build_bttb_operator(shape, (1.0, 1.0, 1.0), _covariance)
    dense = _dense_covariance(shape)
    vector = np.random.default_rng(4).standard_normal(dense.shape[0])
    np.testing.assert_allclose(operator.matvec(vector), dense @ vector, rtol=2e-13, atol=2e-13)


def test_full_lanczos_square_root_matches_dense_eigendecomposition():
    shape = (3, 3, 2)
    dense = _dense_covariance(shape)
    vector = np.random.default_rng(8).standard_normal(dense.shape[0])
    decomposition = lanczos_decomposition(
        lambda value: dense @ value,
        vector,
        max_dimension=dense.shape[0],
        breakdown_tolerance=1.0e-15,
    )
    actual, _ = sqrt_action(decomposition)
    eigenvalues, eigenvectors = np.linalg.eigh(dense)
    expected = eigenvectors @ (np.sqrt(np.maximum(eigenvalues, 0.0)) * (eigenvectors.T @ vector))
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-11)
