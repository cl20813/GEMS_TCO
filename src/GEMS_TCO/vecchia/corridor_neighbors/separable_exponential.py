r"""Advected separable exponential models for the lag-643 corridor engine.

The fitted parameterization and fixed conditioning geometry are identical to
the joint Matérn models.  Only the covariance construction changes.  For
space-time points ``(s, t)`` and ``(s', t')`` this module uses

.. math::

   C = \sigma^2 \exp\{-r_S(s-s'-v(t-t'))\}\exp\{-r_T(t-t')\},

where ``r_S`` and ``r_T`` use the spatial anisotropy, ranges, and advection
parameters reported by :meth:`GroupedVecchiaBase.interpretable_parameters`.
Thus the model is separable after the common advection coordinate transform,
while remaining nonseparable-looking in fixed geographic coordinates.
"""

from __future__ import annotations

import torch

from .._base import _stable_sqrt_distance
from .corridor_lag643 import REFERENCE_ADVEC_LON_ABS, Lag643CorridorVecchia


class _AdvectedSeparableExponentialMixin:
    """Covariance override shared by nugget and no-nugget variants."""

    def _batched_covariance(
        self,
        params: torch.Tensor,
        x_batch: torch.Tensor,
    ) -> torch.Tensor:
        phi1, phi2, phi3, phi4 = torch.exp(params[0:4])
        nugget = self._nugget_from_params(params)
        time = x_batch[:, :, 2]
        advected_latitude = x_batch[:, :, 0] - params[4] * time
        advected_longitude = x_batch[:, :, 1] - params[5] * time

        delta_latitude = advected_latitude.unsqueeze(2) - advected_latitude.unsqueeze(1)
        delta_longitude = advected_longitude.unsqueeze(2) - advected_longitude.unsqueeze(1)
        delta_time = time.unsqueeze(2) - time.unsqueeze(1)
        spatial_distance = _stable_sqrt_distance(
            phi3 * delta_latitude.square() + delta_longitude.square()
        )
        temporal_distance = delta_time.abs() * torch.sqrt(phi4)
        covariance = (phi1 / phi2) * torch.exp(-phi2 * (spatial_distance + temporal_distance))
        covariance.diagonal(dim1=-2, dim2=-1).add_(nugget + 1e-6)
        return covariance

    def point_covariance(
        self,
        params: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Return separable covariance between two point arrays.

        Rows follow the package convention
        ``[latitude, longitude, response, time, ...]``.  Exact matching
        space-time observations receive the modeled nugget; the numerical
        ``1e-8`` jitter is added only for the same ordered point set.
        """

        params = self._validated_params(params)
        if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise TypeError("x and y must be torch.Tensor point arrays")
        if x.ndim != 2 or y.ndim != 2 or x.shape[1] < 4 or y.shape[1] < 4:
            raise ValueError("x and y must be two-dimensional with at least four columns")
        if x.device != y.device or params.device != x.device:
            raise ValueError("params, x, and y must be on the same device")
        if x.dtype != y.dtype or x.dtype != params.dtype or not x.is_floating_point():
            raise TypeError("params, x, and y must use the same floating-point dtype")
        coordinate_columns = [0, 1, 3]
        if (
            not torch.isfinite(x[:, coordinate_columns]).all()
            or not torch.isfinite(y[:, coordinate_columns]).all()
        ):
            raise ValueError("point latitude, longitude, and time must be finite")
        phi1, phi2, phi3, phi4 = torch.exp(params[0:4])
        nugget = self._nugget_from_params(params)
        x_time = x[:, 3]
        y_time = y[:, 3]
        x_latitude = x[:, 0] - params[4] * x_time
        y_latitude = y[:, 0] - params[4] * y_time
        x_longitude = x[:, 1] - params[5] * x_time
        y_longitude = y[:, 1] - params[5] * y_time

        delta_latitude = x_latitude[:, None] - y_latitude[None, :]
        delta_longitude = x_longitude[:, None] - y_longitude[None, :]
        delta_time = x_time[:, None] - y_time[None, :]
        spatial_distance = _stable_sqrt_distance(
            phi3 * delta_latitude.square() + delta_longitude.square()
        )
        temporal_distance = delta_time.abs() * torch.sqrt(phi4)
        covariance = (phi1 / phi2) * torch.exp(-phi2 * (spatial_distance + temporal_distance))

        x_coordinates = x[:, [0, 1, 3]]
        y_coordinates = y[:, [0, 1, 3]]
        same_observation = torch.all(
            x_coordinates[:, None, :] == y_coordinates[None, :, :],
            dim=2,
        )
        covariance = covariance + same_observation.to(covariance.dtype) * nugget
        if self._same_ordered_points(x, y):
            covariance = (
                covariance
                + torch.eye(x.shape[0], device=covariance.device, dtype=covariance.dtype) * 1e-8
            )
        return covariance

    def _supports_native_covariance(self) -> bool:
        """The fused native kernel implements the joint, not separable, metric."""

        return False

    def _optimization_banner(self) -> str:
        return f"--- Starting advected-separable exponential L-BFGS ({self.device}) ---"


class _NoNuggetAdvectedSeparableExponentialMixin(_AdvectedSeparableExponentialMixin):
    covariance_parameter_count = 6

    def _nugget_from_params(self, params: torch.Tensor) -> torch.Tensor:
        return params.new_tensor(0.0)


class AdvectedSeparableExponentialLag643CorridorVecchia(
    _AdvectedSeparableExponentialMixin,
    Lag643CorridorVecchia,
):
    """Seven-parameter advected-separable exponential corridor model."""

    def __init__(
        self,
        input_map: dict,
        grid_coords=None,
        second_lag_stride: int = 2,
        reference_advec_lon_abs: float = REFERENCE_ADVEC_LON_ABS,
        target_chunk_size: int = 128,
        min_target_points: int = 1,
        max_neighbor_search=None,
    ):
        super().__init__(
            smooth=0.5,
            input_map=input_map,
            grid_coords=grid_coords,
            reference_advec_lon_abs=reference_advec_lon_abs,
            second_lag_stride=second_lag_stride,
            target_chunk_size=target_chunk_size,
            min_target_points=min_target_points,
            max_neighbor_search=max_neighbor_search,
            covariance_backend="torch",
        )


class NoNuggetAdvectedSeparableExponentialLag643CorridorVecchia(
    _NoNuggetAdvectedSeparableExponentialMixin,
    Lag643CorridorVecchia,
):
    """Six-parameter advected-separable exponential model with nugget zero."""

    def __init__(
        self,
        input_map: dict,
        grid_coords=None,
        second_lag_stride: int = 2,
        reference_advec_lon_abs: float = REFERENCE_ADVEC_LON_ABS,
        target_chunk_size: int = 128,
        min_target_points: int = 1,
        max_neighbor_search=None,
    ):
        super().__init__(
            smooth=0.5,
            input_map=input_map,
            grid_coords=grid_coords,
            reference_advec_lon_abs=reference_advec_lon_abs,
            second_lag_stride=second_lag_stride,
            target_chunk_size=target_chunk_size,
            min_target_points=min_target_points,
            max_neighbor_search=max_neighbor_search,
            covariance_backend="torch",
        )


__all__ = [
    "AdvectedSeparableExponentialLag643CorridorVecchia",
    "NoNuggetAdvectedSeparableExponentialLag643CorridorVecchia",
]
