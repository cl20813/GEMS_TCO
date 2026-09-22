"""Spline-Matérn variants of the 6/4/3 corridor model.

Why this file exists
--------------------
The base corridor engine is intentionally limited to a few closed-form
smoothness values such as 0.5 and 1.5. These variants support arbitrary
positive smoothness without changing the corridor geometry, regression
design, parameterization, or grouped-batch likelihood.

Parameterization
----------------
The inherited ST model uses

    sigmasq = phi1 / phi2
    scaled_distance = phi2 * d_ST

where ``d_ST`` is the anisotropic/advection-adjusted space-time distance from
the shared engine. ``scaled_distance`` is geometric distance divided by
the reported range, and the Matérn Bessel argument is
``sqrt(2 * smooth) * scaled_distance``.  This file changes only the Matérn
correlation shape:

    cov = sigmasq * Matern_corr(scaled_distance; smooth)

and keeps the shared optimizer, Vecchia conditioning, regression design, and
parameter interpretation.
"""

from __future__ import annotations

import numpy as np
import torch

from GEMS_TCO.spatial.matern_spline import _build_matern_spline_coeffs

from .corridor_lag643 import REFERENCE_ADVEC_LON_ABS, Lag643CorridorVecchia


class _STMaternSplineMixin:
    def _init_st_spline(self, smooth: float, n_points: int, r_max: float):
        smooth = float(smooth)
        if not np.isfinite(smooth) or smooth <= 0:
            raise ValueError(f"smooth must be finite and positive, got {smooth}")
        if int(n_points) < 2:
            raise ValueError("spline_n_points must be at least 2")
        if not np.isfinite(r_max) or float(r_max) <= 0:
            raise ValueError("spline_r_max must be finite and positive")
        self.smooth = smooth
        self._st_spline_n_points = int(n_points)
        self._st_spline_r_max = float(r_max)
        self._st_matern_spline_tensors = {}

    def _get_st_matern_spline_tensors(self):
        key = round(float(self.smooth), 8)
        if key in self._st_matern_spline_tensors:
            return self._st_matern_spline_tensors[key]
        coeffs = _build_matern_spline_coeffs(
            self.smooth,
            n_points=self._st_spline_n_points,
            r_max=self._st_spline_r_max,
        )
        tensors = {
            name: torch.tensor(arr, dtype=torch.float64, device=self.device)
            for name, arr in coeffs.items()
            if name != "r_max"
        }
        tensors["r_max"] = float(coeffs["r_max"])
        self._st_matern_spline_tensors[key] = tensors
        return tensors

    def _evaluate_matern_spline(self, r: torch.Tensor) -> torch.Tensor:
        sp = self._get_st_matern_spline_tensors()
        outside_table = r > self._st_spline_r_max
        r_c = r.clamp(0.0, sp["r_max"])
        orig_shape = r_c.shape
        r_flat = r_c.reshape(-1)
        idx = torch.searchsorted(sp["knots"], r_flat, right=True) - 1
        idx = idx.clamp(0, sp["knots"].numel() - 2)
        dx = r_flat - sp["knots"][idx]
        vals = sp["a"][idx] + dx * (sp["b"][idx] + dx * (sp["c"][idx] + dx * sp["d"][idx]))
        vals = vals.reshape(orig_shape).clamp(0.0, 1.0)
        # A Matérn correlation tends to zero.  Holding the final spline value
        # constant beyond the table would instead create an artificial
        # long-range correlation plateau.
        return vals.masked_fill(outside_table, 0.0)

    def _correlation(self, scaled_distance: torch.Tensor) -> torch.Tensor:
        if np.isclose(self.smooth, 0.5):
            return torch.exp(-scaled_distance)
        if np.isclose(self.smooth, 1.5):
            sqrt_three_distance = np.sqrt(3.0) * scaled_distance
            return (1.0 + sqrt_three_distance) * torch.exp(-sqrt_three_distance)
        return self._evaluate_matern_spline(scaled_distance)

    def _nugget_from_params(self, params):
        return torch.exp(params[6])


class _STNoNuggetSplineMixin(_STMaternSplineMixin):
    covariance_parameter_count = 6

    def _nugget_from_params(self, params):
        return params.new_tensor(0.0)


class SplineMaternLag643CorridorVecchia(_STMaternSplineMixin, Lag643CorridorVecchia):
    """Corridor-width 4x4 lag-643 ST cluster Vecchia with arbitrary smoothness."""

    def __init__(
        self,
        smooth: float,
        input_map: dict,
        grid_coords=None,
        second_lag_stride: int = 2,
        reference_advec_lon_abs: float = REFERENCE_ADVEC_LON_ABS,
        target_chunk_size: int = 128,
        min_target_points: int = 1,
        max_neighbor_search=None,
        spline_n_points: int = 1200,
        spline_r_max: float = 20.0,
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
        )
        self._init_st_spline(smooth, spline_n_points, spline_r_max)


class NoNuggetSplineMaternLag643CorridorVecchia(_STNoNuggetSplineMixin, Lag643CorridorVecchia):
    """Corridor-width 4x4 lag-643 ST cluster Vecchia with nugget fixed at 0."""

    def __init__(
        self,
        smooth: float,
        input_map: dict,
        grid_coords=None,
        second_lag_stride: int = 2,
        reference_advec_lon_abs: float = REFERENCE_ADVEC_LON_ABS,
        target_chunk_size: int = 128,
        min_target_points: int = 1,
        max_neighbor_search=None,
        spline_n_points: int = 1200,
        spline_r_max: float = 20.0,
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
        )
        self._init_st_spline(smooth, spline_n_points, spline_r_max)


__all__ = [
    "SplineMaternLag643CorridorVecchia",
    "NoNuggetSplineMaternLag643CorridorVecchia",
]
