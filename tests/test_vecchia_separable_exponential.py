"""Numerical checks for the independent advected-separable Vecchia model."""

from __future__ import annotations

import numpy as np
import torch

from GEMS_TCO.vecchia.corridor_neighbors.corridor_lag643 import Lag643CorridorVecchia
from GEMS_TCO.vecchia.corridor_neighbors.separable_exponential import (
    AdvectedSeparableExponentialLag643CorridorVecchia,
    NoNuggetAdvectedSeparableExponentialLag643CorridorVecchia,
)


def _input_map() -> dict[str, torch.Tensor]:
    rows = torch.zeros((4, 11), dtype=torch.float64)
    rows[:, 0] = torch.tensor([0.0, 0.0, 0.2, 0.2])
    rows[:, 1] = torch.tensor([0.0, 0.3, 0.0, 0.3])
    rows[:, 3] = 0.0
    return {"t0": rows}


def _likelihood_input() -> tuple[dict[str, torch.Tensor], np.ndarray]:
    latitude = np.repeat(np.arange(8, dtype=np.float64) * 0.05, 8)
    longitude = np.tile(np.arange(8, dtype=np.float64) * 0.05, 8)
    grid = np.column_stack([latitude, longitude])
    result: dict[str, torch.Tensor] = {}
    for time_index in range(3):
        rows = np.zeros((64, 11), dtype=np.float64)
        rows[:, :2] = grid
        rows[:, 2] = np.sin(np.arange(64) * 0.17) + 0.03 * time_index
        rows[:, 3] = float(time_index)
        if time_index > 0:
            rows[:, time_index + 3] = 1.0
        result[str(time_index)] = torch.tensor(rows, dtype=torch.float64)
    return result, grid


def _raw(nugget: bool = True) -> torch.Tensor:
    physical = {
        "signal_variance": 2.3,
        "range_lat": 0.7,
        "range_lon": 0.9,
        "range_time": 1.4,
        "advec_lat": 0.06,
        "advec_lon": -0.12,
        "nugget": 0.17,
    }
    phi2 = 1.0 / physical["range_lon"]
    phi1 = physical["signal_variance"] * phi2
    phi3 = (physical["range_lon"] / physical["range_lat"]) ** 2
    phi4 = (physical["range_lon"] / physical["range_time"]) ** 2
    values = [
        np.log(phi1),
        np.log(phi2),
        np.log(phi3),
        np.log(phi4),
        physical["advec_lat"],
        physical["advec_lon"],
    ]
    if nugget:
        values.append(np.log(physical["nugget"]))
    return torch.tensor(values, dtype=torch.float64)


def _dense_formula(params: torch.Tensor, points: torch.Tensor, nugget: float) -> torch.Tensor:
    phi1, phi2, phi3, phi4 = torch.exp(params[:4])
    time = points[:, 2]
    latitude = points[:, 0] - params[4] * time
    longitude = points[:, 1] - params[5] * time
    dlat = latitude[:, None] - latitude[None, :]
    dlon = longitude[:, None] - longitude[None, :]
    dt = time[:, None] - time[None, :]
    spatial = torch.sqrt(phi3 * dlat.square() + dlon.square())
    temporal = torch.sqrt(phi4) * dt.abs()
    covariance = (phi1 / phi2) * torch.exp(-phi2 * (spatial + temporal))
    covariance = covariance + torch.eye(len(points), dtype=points.dtype) * nugget
    return covariance


def test_batched_and_point_covariance_match_dense_formula() -> None:
    model = AdvectedSeparableExponentialLag643CorridorVecchia(_input_map())
    params = _raw()
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [0.3, 0.1, 1.0], [-0.1, 0.5, 2.0]],
        dtype=torch.float64,
    )
    expected = _dense_formula(params, points, nugget=float(torch.exp(params[6])))
    actual_batch = model.batched_covariance(params, points[None])[0]
    assert torch.allclose(actual_batch, expected + torch.eye(3) * 1e-6, atol=1e-13)

    point_rows = torch.zeros((3, 4), dtype=torch.float64)
    point_rows[:, [0, 1, 3]] = points
    actual_point = model.point_covariance(params, point_rows, point_rows)
    assert torch.allclose(actual_point, expected + torch.eye(3) * 1e-8, atol=1e-13)


def test_no_nugget_variant_and_native_kernel_guard() -> None:
    model = NoNuggetAdvectedSeparableExponentialLag643CorridorVecchia(_input_map())
    params = _raw(nugget=False)
    points = torch.tensor([[[0.0, 0.0, 0.0], [0.2, 0.1, 1.0]]], dtype=torch.float64)
    expected = _dense_formula(params, points[0], nugget=0.0) + torch.eye(2) * 1e-6
    assert model.covariance_parameter_count == 6
    assert model.resolved_covariance_backend() == "torch"
    assert not model._supports_native_covariance()
    assert torch.allclose(model.batched_covariance(params, points)[0], expected, atol=1e-13)


def test_joint_matern_half_and_separable_have_identical_axis_margins() -> None:
    input_map = _input_map()
    joint = Lag643CorridorVecchia(smooth=0.5, input_map=input_map, covariance_backend="torch")
    separable = AdvectedSeparableExponentialLag643CorridorVecchia(input_map)
    params = _raw()
    # Pure spatial pairs have dt=0; pure comoving temporal pairs have zero
    # transformed spatial displacement.  The two models must agree there.
    spatial = torch.tensor([[[0.0, 0.0, 0.0], [0.2, 0.3, 0.0]]], dtype=torch.float64)
    temporal = torch.tensor(
        [[[0.0, 0.0, 0.0], [params[4].item(), params[5].item(), 1.0]]],
        dtype=torch.float64,
    )
    for coordinates in (spatial, temporal):
        joint_covariance = joint.batched_covariance(params, coordinates)
        separable_covariance = separable.batched_covariance(params, coordinates)
        assert torch.allclose(joint_covariance, separable_covariance, atol=1e-13)


def test_separable_covariance_has_finite_autograd() -> None:
    model = AdvectedSeparableExponentialLag643CorridorVecchia(_input_map())
    params = _raw().requires_grad_(True)
    points = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.2, 0.4, 1.0], [0.5, 0.1, 2.0]]],
        dtype=torch.float64,
    )
    loss = model.batched_covariance(params, points).square().sum()
    loss.backward()
    assert params.grad is not None
    assert torch.isfinite(params.grad).all()


def test_grouped_corridor_likelihood_and_gls_run_end_to_end() -> None:
    input_map, grid = _likelihood_input()
    model = NoNuggetAdvectedSeparableExponentialLag643CorridorVecchia(
        input_map=input_map,
        grid_coords=grid,
        target_chunk_size=16,
    )
    model.precompute_conditioning_sets()
    params = _raw(nugget=False).requires_grad_(True)
    loss = model.profiled_negative_log_likelihood(params)
    loss.backward()
    beta = model.estimate_gls_coefficients(params.detach())
    assert torch.isfinite(loss)
    assert params.grad is not None and torch.isfinite(params.grad).all()
    assert beta.shape == (9, 1)
    assert torch.isfinite(beta).all()


def test_point_covariance_rejects_malformed_public_inputs() -> None:
    model = AdvectedSeparableExponentialLag643CorridorVecchia(_input_map())
    params = _raw()
    valid = torch.zeros((2, 4), dtype=torch.float64)
    with np.testing.assert_raises_regex(ValueError, "at least four columns"):
        model.point_covariance(params, valid[:, :3], valid)
    nonfinite = valid.clone()
    nonfinite[0, 0] = float("nan")
    with np.testing.assert_raises_regex(ValueError, "must be finite"):
        model.point_covariance(params, nonfinite, valid)
    with np.testing.assert_raises_regex(TypeError, "same floating-point dtype"):
        model.point_covariance(params, valid.float(), valid.float())
