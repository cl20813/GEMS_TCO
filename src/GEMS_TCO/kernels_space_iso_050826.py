"""
kernels_space_iso_050826.py

Pure-space isotropic Vecchia wrappers.

Estimated covariance parameters:
  log(sigmasq), log(range), log(nugget)

Nugget-free diagnostic variants estimate:
  log(sigmasq), log(range)

This reuses the same pure-space conditioning machinery as
kernels_space_050726.py, but constrains the spatial covariance to a single
isotropic range instead of separate latitude/longitude ranges.
"""

from __future__ import annotations

import numpy as np
import torch

from GEMS_TCO.kernels_space_050726 import ColumnSpaceVecchiaFit, HybridSpaceVecchiaFit
from GEMS_TCO.kernels_space_trend_050726 import _MeanDesignMixin
from GEMS_TCO.kernels_space_multiscale_050826 import _build_matern_spline_coeffs


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
    raise TypeError("smooth must be passed to the pure-space isotropic Vecchia class")


def _smooth_allowed_by_base(smooth):
    return float(smooth) in (0.5, 1.5)


class _IsoSpaceMixin:
    def _finish_init_with_any_smooth(self, args, kwargs, mean_design):
        args, kwargs, requested_smooth = _replace_smooth_arg(args, kwargs)
        super().__init__(*args, **kwargs)
        self.smooth = float(requested_smooth)
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

    def _raw_params(self, params: torch.Tensor):
        sigmasq = torch.exp(params[0])
        range_space = torch.exp(params[1])
        nugget = torch.exp(params[2])
        # The base pure-space likelihood expects four return values:
        # sigmasq, range_lat, range_lon, nugget.  Return range twice.
        return sigmasq, range_space, range_space, nugget

    def _cov_from_deltas(self, d_lat, d_lon, params: torch.Tensor):
        sigmasq, range_space, _, _ = self._raw_params(params)
        dist = torch.sqrt(d_lat.new_tensor(1e-8) + d_lat.pow(2) + d_lon.pow(2)) / range_space
        return sigmasq * self._matern_corr(dist)

    def _convert_params(self, raw):
        return {
            "sigmasq": float(np.exp(raw[0])),
            "range": float(np.exp(raw[1])),
            "nugget": float(np.exp(raw[2])),
        }


class _IsoNoNuggetSpaceMixin:
    _get_matern_spline_tensors = _IsoSpaceMixin._get_matern_spline_tensors
    _matern_corr = _IsoSpaceMixin._matern_corr

    def _finish_init_with_any_smooth(self, args, kwargs, mean_design):
        args, kwargs, requested_smooth = _replace_smooth_arg(args, kwargs)
        super().__init__(*args, **kwargs)
        self.smooth = float(requested_smooth)
        self._matern_spline_tensors = {}
        self._init_mean_design(mean_design)

    def _raw_params(self, params: torch.Tensor):
        sigmasq = torch.exp(params[0])
        range_space = torch.exp(params[1])
        nugget = params.new_tensor(0.0)
        return sigmasq, range_space, range_space, nugget

    def _cov_from_deltas(self, d_lat, d_lon, params: torch.Tensor):
        sigmasq, range_space, _, _ = self._raw_params(params)
        dist = torch.sqrt(d_lat.new_tensor(1e-8) + d_lat.pow(2) + d_lon.pow(2)) / range_space
        return sigmasq * self._matern_corr(dist)

    def _convert_params(self, raw):
        return {
            "sigmasq": float(np.exp(raw[0])),
            "range": float(np.exp(raw[1])),
            "nugget": 0.0,
        }


class ColumnSpaceIsoTrendVecchiaFit(_IsoSpaceMixin, _MeanDesignMixin, ColumnSpaceVecchiaFit):
    def __init__(self, *args, mean_design: str = "latlon", **kwargs):
        self._finish_init_with_any_smooth(args, kwargs, mean_design)


class HybridSpaceIsoTrendVecchiaFit(_IsoSpaceMixin, _MeanDesignMixin, HybridSpaceVecchiaFit):
    def __init__(self, *args, mean_design: str = "latlon", **kwargs):
        self._finish_init_with_any_smooth(args, kwargs, mean_design)


class ColumnSpaceIsoNoNuggetTrendVecchiaFit(_IsoNoNuggetSpaceMixin, _MeanDesignMixin, ColumnSpaceVecchiaFit):
    def __init__(self, *args, mean_design: str = "latlon", **kwargs):
        self._finish_init_with_any_smooth(args, kwargs, mean_design)


class HybridSpaceIsoNoNuggetTrendVecchiaFit(_IsoNoNuggetSpaceMixin, _MeanDesignMixin, HybridSpaceVecchiaFit):
    def __init__(self, *args, mean_design: str = "latlon", **kwargs):
        self._finish_init_with_any_smooth(args, kwargs, mean_design)


__all__ = [
    "ColumnSpaceIsoTrendVecchiaFit",
    "HybridSpaceIsoTrendVecchiaFit",
    "ColumnSpaceIsoNoNuggetTrendVecchiaFit",
    "HybridSpaceIsoNoNuggetTrendVecchiaFit",
]
