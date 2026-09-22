"""Pure-space anisotropic Matérn cluster Vecchia model.

This module keeps the same cluster Vecchia geometry as
the isotropic cluster implementation:

  - fixed regular-grid clusters;
  - max-min ordering on cluster centroids;
  - target blocks are whole clusters;
  - conditioning blocks are previous same-time nearest cluster blocks.

Only the covariance parameterization changes from isotropic to anisotropic:

    params with nugget    = log(signal_variance), log(range_lat), log(range_lon), log(nugget)
    params without nugget = log(signal_variance), log(range_lat), log(range_lon)

and

    d(h)^2 = (delta_lat / range_lat)^2 + (delta_lon / range_lon)^2.

The model uses the standard Matérn range convention: the Bessel argument is
``sqrt(2 * smooth) * d(h)``.  At smoothness 0.5 the correlation is
``exp(-d)``, while at 1.5 it is
``(1 + sqrt(3)d) exp(-sqrt(3)d)``.  The same convention is used by the direct
Bessel full-likelihood implementation in :mod:`GEMS_TCO.spatial.matern_bessel`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .base import _MeanDesignMixin, _stable_sqrt_distance
from .isotropic import _ClusterSpatialVecchiaBase, _MaternCorrelationMixin


class _AnisoMaternCommonMixin(_MaternCorrelationMixin):
    def _cov_from_deltas(self, d_lat, d_lon, params: torch.Tensor):
        sigmasq, range_lat, range_lon, _ = self._raw_params(params)
        dist = _stable_sqrt_distance((d_lat / range_lat).pow(2) + (d_lon / range_lon).pow(2))
        return sigmasq * self._matern_corr(dist)


class _AnisoMaternSpaceMixin(_AnisoMaternCommonMixin):
    _n_covariance_parameters = 4

    def _raw_params(self, params: torch.Tensor):
        sigmasq = torch.exp(params[0])
        range_lat = torch.exp(params[1])
        range_lon = torch.exp(params[2])
        nugget = torch.exp(params[3])
        return sigmasq, range_lat, range_lon, nugget

    def _convert_params(self, raw: List[float]) -> Dict[str, float]:
        sigmasq, range_lat, range_lon, nugget = self._natural_values_from_raw_sequence(raw)
        return {
            "signal_variance": sigmasq,
            "range_lat": range_lat,
            "range_lon": range_lon,
            "nugget": nugget,
        }


class _AnisoNoNuggetMaternSpaceMixin(_AnisoMaternCommonMixin):
    _n_covariance_parameters = 3

    def _raw_params(self, params: torch.Tensor):
        sigmasq = torch.exp(params[0])
        range_lat = torch.exp(params[1])
        range_lon = torch.exp(params[2])
        nugget = params.new_tensor(0.0)
        return sigmasq, range_lat, range_lon, nugget

    def _convert_params(self, raw: List[float]) -> Dict[str, float]:
        sigmasq, range_lat, range_lon, nugget = self._natural_values_from_raw_sequence(raw)
        return {
            "signal_variance": sigmasq,
            "range_lat": range_lat,
            "range_lon": range_lon,
            "nugget": nugget,
        }


class AnisotropicMaternSpatialVecchia(
    _AnisoMaternSpaceMixin,
    _MeanDesignMixin,
    _ClusterSpatialVecchiaBase,
):
    """Block-target spatial Vecchia model with axis-aligned Matérn ranges.

    ``latlon_hour`` is the default 10-column mean; ``latlon`` is the standard
    three-column ``[1, lat, lon]`` design.
    """

    def __init__(
        self,
        smooth: float,
        input_map: Dict[str, torch.Tensor | np.ndarray],
        *,
        grid_coords: Optional[np.ndarray] = None,
        block_shape: Tuple[int, int] = (4, 4),
        n_neighbor_blocks: int = 6,
        target_chunk_size: int = 128,
        min_target_points: int = 1,
        max_neighbor_search: Optional[int] = None,
        lat_round_decimals: int = 10,
        lon_round_decimals: int = 10,
        mean_design: str = "latlon_hour",
    ):
        super().__init__(
            smooth=smooth,
            input_map=input_map,
            grid_coords=grid_coords,
            block_shape=block_shape,
            n_neighbor_blocks=n_neighbor_blocks,
            target_chunk_size=target_chunk_size,
            min_target_points=min_target_points,
            max_neighbor_search=max_neighbor_search,
            lat_round_decimals=lat_round_decimals,
            lon_round_decimals=lon_round_decimals,
        )
        self._init_matern_correlation(smooth)
        self._init_mean_design(mean_design)


class NoNuggetAnisotropicMaternSpatialVecchia(
    _AnisoNoNuggetMaternSpaceMixin,
    _MeanDesignMixin,
    _ClusterSpatialVecchiaBase,
):
    """Anisotropic Matérn spatial Vecchia model with nugget fixed to zero.

    ``latlon_hour`` is the default 10-column mean; ``latlon`` is the standard
    three-column ``[1, lat, lon]`` design.
    """

    def __init__(
        self,
        smooth: float,
        input_map: Dict[str, torch.Tensor | np.ndarray],
        *,
        grid_coords: Optional[np.ndarray] = None,
        block_shape: Tuple[int, int] = (4, 4),
        n_neighbor_blocks: int = 6,
        target_chunk_size: int = 128,
        min_target_points: int = 1,
        max_neighbor_search: Optional[int] = None,
        lat_round_decimals: int = 10,
        lon_round_decimals: int = 10,
        mean_design: str = "latlon_hour",
    ):
        super().__init__(
            smooth=smooth,
            input_map=input_map,
            grid_coords=grid_coords,
            block_shape=block_shape,
            n_neighbor_blocks=n_neighbor_blocks,
            target_chunk_size=target_chunk_size,
            min_target_points=min_target_points,
            max_neighbor_search=max_neighbor_search,
            lat_round_decimals=lat_round_decimals,
            lon_round_decimals=lon_round_decimals,
        )
        self._init_matern_correlation(smooth)
        self._init_mean_design(mean_design)


__all__ = [
    "AnisotropicMaternSpatialVecchia",
    "NoNuggetAnisotropicMaternSpatialVecchia",
]
