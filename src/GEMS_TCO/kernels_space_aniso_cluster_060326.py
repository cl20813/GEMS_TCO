"""
kernels_space_aniso_cluster_060326.py

Pure-space anisotropic Matérn cluster Vecchia model.

This module keeps the same cluster Vecchia geometry as
``kernels_space_iso_cluster_052426``:

  - fixed regular-grid clusters;
  - max-min ordering on cluster centroids;
  - target blocks are whole clusters;
  - conditioning blocks are previous same-time nearest cluster blocks.

Only the covariance parameterization changes from isotropic to anisotropic:

    params with nugget    = log(sigmasq), log(range_lat), log(range_lon), log(nugget)
    params without nugget = log(sigmasq), log(range_lat), log(range_lon)

and

    d(h)^2 = (delta_lat / range_lat)^2 + (delta_lon / range_lon)^2.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from GEMS_TCO.kernels_space_base_engine_052126 import (
    _MeanDesignMixin,
    _build_matern_spline_coeffs,
)
from GEMS_TCO.kernels_space_iso_cluster_052426 import ClusterSpaceVecchiaFit


def _replace_smooth_arg(args, kwargs, fallback_smooth=0.5):
    args = tuple(args)
    if "smooth" in kwargs:
        original = float(kwargs["smooth"])
        kwargs = dict(kwargs)
        kwargs["smooth"] = float(fallback_smooth)
        return args, kwargs, original
    if args:
        original = float(args[0])
        return (float(fallback_smooth),) + args[1:], kwargs, original
    raise TypeError("smooth must be passed to the pure-space anisotropic cluster Vecchia class")


class _AnisoMaternCommonMixin:
    def _finish_init_with_any_smooth(self, args, kwargs, mean_design):
        args, kwargs, requested_smooth = _replace_smooth_arg(args, kwargs)
        super().__init__(*args, **kwargs)
        self.smooth = float(requested_smooth)
        if self.smooth <= 0:
            raise ValueError(f"smooth must be positive, got {self.smooth}")
        self._matern_spline_tensors = {}
        self._init_mean_design(mean_design)

    def _get_matern_spline_tensors(self, smooth: float):
        key = round(float(smooth), 8)
        if key in self._matern_spline_tensors:
            return self._matern_spline_tensors[key]
        coeffs = _build_matern_spline_coeffs(float(smooth))
        tensors = {
            name: torch.tensor(arr, dtype=torch.float64, device=self.device)
            for name, arr in coeffs.items()
            if name != "r_max"
        }
        tensors["r_max"] = float(coeffs["r_max"])
        self._matern_spline_tensors[key] = tensors
        return tensors

    def _matern_corr(self, dist: torch.Tensor) -> torch.Tensor:
        if self.smooth == 0.5:
            return torch.exp(-dist)
        if self.smooth == 1.5:
            return (1.0 + dist) * torch.exp(-dist)
        sp = self._get_matern_spline_tensors(self.smooth)
        r_c = dist.clamp(0.0, sp["r_max"])
        orig_shape = r_c.shape
        r_flat = r_c.reshape(-1)
        idx = torch.searchsorted(sp["knots"], r_flat, right=True) - 1
        idx = idx.clamp(0, sp["knots"].numel() - 2)
        dx = r_flat - sp["knots"][idx]
        vals = sp["a"][idx] + dx * (sp["b"][idx] + dx * (sp["c"][idx] + dx * sp["d"][idx]))
        return vals.reshape(orig_shape).clamp_min(0.0)

    def _cov_from_deltas(self, d_lat, d_lon, params: torch.Tensor):
        sigmasq, range_lat, range_lon, _ = self._raw_params(params)
        dist = torch.sqrt(
            d_lat.new_tensor(1e-8)
            + (d_lat / range_lat).pow(2)
            + (d_lon / range_lon).pow(2)
        )
        return sigmasq * self._matern_corr(dist)


class _AnisoMaternSpaceMixin(_AnisoMaternCommonMixin):
    def _raw_params(self, params: torch.Tensor):
        sigmasq = torch.exp(params[0])
        range_lat = torch.exp(params[1])
        range_lon = torch.exp(params[2])
        nugget = torch.exp(params[3])
        return sigmasq, range_lat, range_lon, nugget

    def _convert_params(self, raw: List[float]) -> Dict[str, float]:
        return {
            "sigmasq": float(np.exp(raw[0])),
            "range_lat": float(np.exp(raw[1])),
            "range_lon": float(np.exp(raw[2])),
            "range": float(np.exp(raw[2])),
            "nugget": float(np.exp(raw[3])),
        }


class _AnisoNoNuggetMaternSpaceMixin(_AnisoMaternCommonMixin):
    def _raw_params(self, params: torch.Tensor):
        sigmasq = torch.exp(params[0])
        range_lat = torch.exp(params[1])
        range_lon = torch.exp(params[2])
        nugget = params.new_tensor(0.0)
        return sigmasq, range_lat, range_lon, nugget

    def _convert_params(self, raw: List[float]) -> Dict[str, float]:
        return {
            "sigmasq": float(np.exp(raw[0])),
            "range_lat": float(np.exp(raw[1])),
            "range_lon": float(np.exp(raw[2])),
            "range": float(np.exp(raw[2])),
            "nugget": 0.0,
        }


class ClusterSpaceAnisoTrendVecchiaFit(_AnisoMaternSpaceMixin, _MeanDesignMixin, ClusterSpaceVecchiaFit):
    def __init__(self, *args, mean_design: str = "latlon", **kwargs):
        self._finish_init_with_any_smooth(args, kwargs, mean_design)


class ClusterSpaceAnisoNoNuggetTrendVecchiaFit(
    _AnisoNoNuggetMaternSpaceMixin,
    _MeanDesignMixin,
    ClusterSpaceVecchiaFit,
):
    def __init__(self, *args, mean_design: str = "latlon", **kwargs):
        self._finish_init_with_any_smooth(args, kwargs, mean_design)


__all__ = [
    "ClusterSpaceAnisoTrendVecchiaFit",
    "ClusterSpaceAnisoNoNuggetTrendVecchiaFit",
]
