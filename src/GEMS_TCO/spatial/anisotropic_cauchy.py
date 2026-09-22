"""Pure-space anisotropic generalized Cauchy cluster Vecchia model.

The likelihood uses the package's pure-space block-Vecchia geometry:

  - fixed regular-grid clusters, usually 4x4 points;
  - max-min ordering on cluster centroids;
  - each target is a whole cluster block;
  - condition on previous same-time nearest cluster blocks.

Only the covariance and parameter interpretation differ.  The no-nugget model
uses the log-parameter vector

    params[0] = log phi1       phi1 = sigma^2 / range_lon
    params[1] = log phi2       phi2 = 1 / range_lon
    params[2] = log phi3       phi3 = (range_lon / range_lat)^2
    params[3] = log gc_beta

with fixed gc_alpha.  The covariance is

    C(h) = sigma^2 * (1 + d(h)^gc_alpha)^(-gc_beta / gc_alpha)

where

    d(h)^2 = (delta_lon / range_lon)^2 + (delta_lat / range_lat)^2.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .base import _MeanDesignMixin, _stable_sqrt_distance
from .isotropic import _ClusterSpatialVecchiaBase


class _AnisoGeneralizedCauchyNoNuggetMixin:
    """Generalized Cauchy covariance with phi reparameterization."""

    _n_covariance_parameters = 4

    def _initialize_cauchy(
        self,
        input_map: Dict[str, torch.Tensor | np.ndarray],
        grid_coords: Optional[np.ndarray],
        block_shape: Tuple[int, int],
        n_neighbor_blocks: int,
        target_chunk_size: int,
        min_target_points: int,
        max_neighbor_search: Optional[int],
        lat_round_decimals: int,
        lon_round_decimals: int,
        mean_design: str,
        gc_alpha: float,
    ) -> None:
        # ``smooth`` belongs to the shared geometry base but is not used by the
        # generalized Cauchy covariance.
        super().__init__(
            smooth=1.0,
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
        self.gc_alpha = float(gc_alpha)
        if not math.isfinite(self.gc_alpha) or not 0.0 < self.gc_alpha <= 2.0:
            raise ValueError(
                f"gc_alpha must be finite and in (0, 2] for a valid covariance, got {gc_alpha}"
            )
        self._init_mean_design(mean_design)

    def _raw_params(self, params: torch.Tensor):
        phi1 = torch.exp(params[0])
        phi2 = torch.exp(params[1])
        phi3 = torch.exp(params[2])
        gc_beta = torch.exp(params[3])
        sigmasq = phi1 / phi2
        range_lon = 1.0 / phi2
        range_lat = 1.0 / (phi2 * torch.sqrt(phi3))
        nugget = params.new_tensor(0.0)
        return sigmasq, range_lat, range_lon, nugget, phi1, phi2, phi3, gc_beta

    def _cov_from_deltas(self, d_lat, d_lon, params: torch.Tensor):
        sigmasq, _, _, _, _, phi2, phi3, gc_beta = self._raw_params(params)
        dist = _stable_sqrt_distance(d_lat.pow(2) * phi3 + d_lon.pow(2))
        scaled = dist * phi2
        alpha = scaled.new_tensor(self.gc_alpha)
        positive = scaled > 0
        scaled_power = torch.where(
            positive,
            torch.pow(scaled.clamp_min(torch.finfo(scaled.dtype).eps), alpha),
            torch.zeros_like(scaled),
        )
        corr = torch.pow(1.0 + scaled_power, -gc_beta / alpha)
        return sigmasq * corr

    def _convert_params(self, raw: List[float]) -> Dict[str, float]:
        (
            sigmasq,
            range_lat,
            range_lon,
            nugget,
            phi1,
            phi2,
            phi3,
            gc_beta,
        ) = self._natural_values_from_raw_sequence(raw)
        return {
            "signal_variance": sigmasq,
            "range_lon": range_lon,
            "range_lat": range_lat,
            "nugget": nugget,
            "phi1": phi1,
            "phi2": phi2,
            "phi3": phi3,
            "gc_alpha": float(self.gc_alpha),
            "gc_beta": gc_beta,
        }


class NoNuggetAnisotropicCauchySpatialVecchia(
    _AnisoGeneralizedCauchyNoNuggetMixin,
    _MeanDesignMixin,
    _ClusterSpatialVecchiaBase,
):
    """Cluster anisotropic generalized Cauchy model with nugget fixed to zero.

    ``latlon_hour`` is the default 10-column mean; ``latlon`` is the standard
    three-column ``[1, lat, lon]`` design.
    """

    def __init__(
        self,
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
        gc_alpha: float = 0.6,
    ):
        self._initialize_cauchy(
            input_map=input_map,
            grid_coords=grid_coords,
            block_shape=block_shape,
            n_neighbor_blocks=n_neighbor_blocks,
            target_chunk_size=target_chunk_size,
            min_target_points=min_target_points,
            max_neighbor_search=max_neighbor_search,
            lat_round_decimals=lat_round_decimals,
            lon_round_decimals=lon_round_decimals,
            mean_design=mean_design,
            gc_alpha=gc_alpha,
        )


class _AnisoGeneralizedCauchyFixedBetaNoNuggetMixin(_AnisoGeneralizedCauchyNoNuggetMixin):
    """Generalized Cauchy covariance with fixed beta and nugget fixed at 0."""

    _n_covariance_parameters = 3

    def _initialize_cauchy_fixed_beta(
        self,
        input_map: Dict[str, torch.Tensor | np.ndarray],
        grid_coords: Optional[np.ndarray],
        block_shape: Tuple[int, int],
        n_neighbor_blocks: int,
        target_chunk_size: int,
        min_target_points: int,
        max_neighbor_search: Optional[int],
        lat_round_decimals: int,
        lon_round_decimals: int,
        mean_design: str,
        gc_alpha: float,
        gc_beta: float,
    ) -> None:
        self.gc_beta_fixed = float(gc_beta)
        if not math.isfinite(self.gc_beta_fixed) or self.gc_beta_fixed <= 0:
            raise ValueError(f"gc_beta must be finite and positive, got {gc_beta}")
        self._initialize_cauchy(
            input_map=input_map,
            grid_coords=grid_coords,
            block_shape=block_shape,
            n_neighbor_blocks=n_neighbor_blocks,
            target_chunk_size=target_chunk_size,
            min_target_points=min_target_points,
            max_neighbor_search=max_neighbor_search,
            lat_round_decimals=lat_round_decimals,
            lon_round_decimals=lon_round_decimals,
            mean_design=mean_design,
            gc_alpha=gc_alpha,
        )

    def _raw_params(self, params: torch.Tensor):
        phi1 = torch.exp(params[0])
        phi2 = torch.exp(params[1])
        phi3 = torch.exp(params[2])
        gc_beta = params.new_tensor(self.gc_beta_fixed)
        sigmasq = phi1 / phi2
        range_lon = 1.0 / phi2
        range_lat = 1.0 / (phi2 * torch.sqrt(phi3))
        nugget = params.new_tensor(0.0)
        return sigmasq, range_lat, range_lon, nugget, phi1, phi2, phi3, gc_beta

    def _convert_params(self, raw: List[float]) -> Dict[str, float]:
        (
            sigmasq,
            range_lat,
            range_lon,
            nugget,
            phi1,
            phi2,
            phi3,
            _,
        ) = self._natural_values_from_raw_sequence(raw)
        return {
            "signal_variance": sigmasq,
            "range_lon": range_lon,
            "range_lat": range_lat,
            "nugget": nugget,
            "phi1": phi1,
            "phi2": phi2,
            "phi3": phi3,
            "gc_alpha": float(self.gc_alpha),
            "gc_beta": float(self.gc_beta_fixed),
        }


class NoNuggetFixedBetaAnisotropicCauchySpatialVecchia(
    _AnisoGeneralizedCauchyFixedBetaNoNuggetMixin,
    _MeanDesignMixin,
    _ClusterSpatialVecchiaBase,
):
    """No-nugget anisotropic generalized Cauchy model with fixed beta.

    ``latlon_hour`` is the default 10-column mean; ``latlon`` is the standard
    three-column ``[1, lat, lon]`` design.
    """

    def __init__(
        self,
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
        gc_alpha: float = 0.6,
        gc_beta: float = 1.0,
    ):
        self._initialize_cauchy_fixed_beta(
            input_map=input_map,
            grid_coords=grid_coords,
            block_shape=block_shape,
            n_neighbor_blocks=n_neighbor_blocks,
            target_chunk_size=target_chunk_size,
            min_target_points=min_target_points,
            max_neighbor_search=max_neighbor_search,
            lat_round_decimals=lat_round_decimals,
            lon_round_decimals=lon_round_decimals,
            mean_design=mean_design,
            gc_alpha=gc_alpha,
            gc_beta=gc_beta,
        )


class _AnisoGeneralizedCauchyFixedBetaNuggetMixin(_AnisoGeneralizedCauchyFixedBetaNoNuggetMixin):
    """Generalized Cauchy covariance with fixed beta and estimated nugget."""

    _n_covariance_parameters = 4

    def _raw_params(self, params: torch.Tensor):
        phi1 = torch.exp(params[0])
        phi2 = torch.exp(params[1])
        phi3 = torch.exp(params[2])
        nugget = torch.exp(params[3])
        gc_beta = params.new_tensor(self.gc_beta_fixed)
        sigmasq = phi1 / phi2
        range_lon = 1.0 / phi2
        range_lat = 1.0 / (phi2 * torch.sqrt(phi3))
        return sigmasq, range_lat, range_lon, nugget, phi1, phi2, phi3, gc_beta

    def _convert_params(self, raw: List[float]) -> Dict[str, float]:
        (
            sigmasq,
            range_lat,
            range_lon,
            nugget,
            phi1,
            phi2,
            phi3,
            _,
        ) = self._natural_values_from_raw_sequence(raw)
        return {
            "signal_variance": sigmasq,
            "range_lon": range_lon,
            "range_lat": range_lat,
            "nugget": nugget,
            "phi1": phi1,
            "phi2": phi2,
            "phi3": phi3,
            "gc_alpha": float(self.gc_alpha),
            "gc_beta": float(self.gc_beta_fixed),
        }


class FixedBetaAnisotropicCauchySpatialVecchia(
    _AnisoGeneralizedCauchyFixedBetaNuggetMixin,
    _MeanDesignMixin,
    _ClusterSpatialVecchiaBase,
):
    """Anisotropic generalized Cauchy model with fixed beta and fitted nugget.

    ``latlon_hour`` is the default 10-column mean; ``latlon`` is the standard
    three-column ``[1, lat, lon]`` design.
    """

    def __init__(
        self,
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
        gc_alpha: float = 0.6,
        gc_beta: float = 1.0,
    ):
        self._initialize_cauchy_fixed_beta(
            input_map=input_map,
            grid_coords=grid_coords,
            block_shape=block_shape,
            n_neighbor_blocks=n_neighbor_blocks,
            target_chunk_size=target_chunk_size,
            min_target_points=min_target_points,
            max_neighbor_search=max_neighbor_search,
            lat_round_decimals=lat_round_decimals,
            lon_round_decimals=lon_round_decimals,
            mean_design=mean_design,
            gc_alpha=gc_alpha,
            gc_beta=gc_beta,
        )


def cauchy_phi_init_from_natural(
    signal_variance: float,
    range_lat: float,
    range_lon: float,
    gc_beta: float,
) -> Dict[str, float]:
    """Convert positive natural parameters to the Cauchy phi parameterization."""
    values = {
        "signal_variance": float(signal_variance),
        "range_lat": float(range_lat),
        "range_lon": float(range_lon),
        "gc_beta": float(gc_beta),
    }
    invalid = [name for name, value in values.items() if not math.isfinite(value) or value <= 0]
    if invalid:
        raise ValueError(f"{', '.join(invalid)} must be finite and positive")
    signal_variance = values["signal_variance"]
    range_lat = values["range_lat"]
    range_lon = values["range_lon"]
    phi2 = 1.0 / range_lon
    phi3 = (range_lon / range_lat) ** 2
    phi1 = signal_variance * phi2
    derived = {"phi1": phi1, "phi2": phi2, "phi3": phi3}
    invalid_derived = [
        name for name, value in derived.items() if not math.isfinite(value) or value <= 0.0
    ]
    if invalid_derived:
        raise ValueError(
            "natural parameters produce non-finite or zero Cauchy "
            f"reparameterization values: {', '.join(invalid_derived)}"
        )
    return {
        "phi1": phi1,
        "phi2": phi2,
        "phi3": phi3,
        "gc_beta": values["gc_beta"],
    }


__all__ = [
    "NoNuggetAnisotropicCauchySpatialVecchia",
    "FixedBetaAnisotropicCauchySpatialVecchia",
    "NoNuggetFixedBetaAnisotropicCauchySpatialVecchia",
    "cauchy_phi_init_from_natural",
]
