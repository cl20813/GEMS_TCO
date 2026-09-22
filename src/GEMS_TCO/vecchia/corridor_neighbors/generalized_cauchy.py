"""Space-time corridor Vecchia models with generalized-Cauchy correlations.

The conditioning geometry, mean model, optimizer, and anisotropic/advection
distance are inherited from the 4x4 corridor model. Only the correlation shape
is changed to

    corr(r) = (1 + r^alpha)^(-beta / alpha),

where r is the same scaled space-time distance used by the Matérn models.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from .corridor_lag432 import Lag432CorridorVecchia
from .corridor_lag643 import REFERENCE_ADVEC_LON_ABS, Lag643CorridorVecchia


class _STGeneralizedCauchyMixin:
    def _init_st_cauchy(self, gc_alpha: float, gc_beta: float):
        self.gc_alpha = float(gc_alpha)
        self.gc_beta = float(gc_beta)
        if not np.isfinite(self.gc_alpha) or not 0 < self.gc_alpha <= 2:
            raise ValueError(f"gc_alpha must be finite and in (0, 2], got {gc_alpha}")
        if not np.isfinite(self.gc_beta) or self.gc_beta <= 0:
            raise ValueError(f"gc_beta must be finite and positive, got {gc_beta}")

    def _correlation(self, scaled_distance: torch.Tensor) -> torch.Tensor:
        alpha = scaled_distance.new_tensor(self.gc_alpha)
        beta = scaled_distance.new_tensor(self.gc_beta)
        positive = scaled_distance > 0
        scaled = scaled_distance.clamp_min(torch.finfo(scaled_distance.dtype).eps)
        correlation = torch.pow(1.0 + torch.pow(scaled, alpha), -beta / alpha)
        return torch.where(positive, correlation, torch.ones_like(correlation))

    def _nugget_from_params(self, params: torch.Tensor) -> torch.Tensor:
        return torch.exp(params[6])

    def interpretable_parameters(self, raw: List[float]) -> Dict[str, float]:
        """Add fixed generalized-Cauchy shape parameters to the fitted record."""

        converted = super().interpretable_parameters(raw)
        converted.update(gc_alpha=float(self.gc_alpha), gc_beta=float(self.gc_beta))
        return converted

    def _optimization_banner(self) -> str:
        return (
            "--- Starting ST generalized Cauchy L-BFGS "
            f"(alpha={self.gc_alpha:g}, beta={self.gc_beta:g}, "
            f"device={self.device}) ---"
        )


class _STNoNuggetGeneralizedCauchyMixin(_STGeneralizedCauchyMixin):
    covariance_parameter_count = 6

    def _nugget_from_params(self, params: torch.Tensor) -> torch.Tensor:
        return params.new_tensor(0.0)


class GeneralizedCauchyLag643CorridorVecchia(
    _STGeneralizedCauchyMixin,
    Lag643CorridorVecchia,
):
    """4x4 lag-643 corridor model with fixed generalized Cauchy alpha/beta."""

    def __init__(
        self,
        gc_alpha: float,
        gc_beta: float,
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
        )
        self._init_st_cauchy(gc_alpha=gc_alpha, gc_beta=gc_beta)


class NoNuggetGeneralizedCauchyLag643CorridorVecchia(
    _STNoNuggetGeneralizedCauchyMixin,
    Lag643CorridorVecchia,
):
    """4x4 lag-643 corridor generalized Cauchy model with nugget fixed at 0."""

    def __init__(
        self,
        gc_alpha: float,
        gc_beta: float,
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
        )
        self._init_st_cauchy(gc_alpha=gc_alpha, gc_beta=gc_beta)


class GeneralizedCauchyLag432CorridorVecchia(
    _STGeneralizedCauchyMixin,
    Lag432CorridorVecchia,
):
    """4x4 lag-432 corridor model with fixed generalized Cauchy alpha/beta."""

    def __init__(
        self,
        gc_alpha: float,
        gc_beta: float,
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
        )
        self._init_st_cauchy(gc_alpha=gc_alpha, gc_beta=gc_beta)


class NoNuggetGeneralizedCauchyLag432CorridorVecchia(
    _STNoNuggetGeneralizedCauchyMixin,
    Lag432CorridorVecchia,
):
    """4x4 lag-432 corridor generalized Cauchy model with nugget fixed at 0."""

    def __init__(
        self,
        gc_alpha: float,
        gc_beta: float,
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
        )
        self._init_st_cauchy(gc_alpha=gc_alpha, gc_beta=gc_beta)


__all__ = [
    "GeneralizedCauchyLag643CorridorVecchia",
    "NoNuggetGeneralizedCauchyLag643CorridorVecchia",
    "GeneralizedCauchyLag432CorridorVecchia",
    "NoNuggetGeneralizedCauchyLag432CorridorVecchia",
]
