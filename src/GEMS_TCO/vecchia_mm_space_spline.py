"""
vecchia_mm_space_spline.py

Max-min pure-space isotropic Vecchia wrappers with spline Matérn correlation.

This module keeps the smoothness-grid experiments away from older kernel names.
The optimizer uses the microergodic-style parameterization

    phi2 = 1 / range
    phi1 = sigmasq * phi2
    sigmasq = phi1 / phi2

and evaluates Matérn correlations by cubic-spline interpolation for arbitrary
positive smoothness values such as 0.2, 0.25, ..., 0.45.
"""

from __future__ import annotations

import numpy as np
import torch

from GEMS_TCO.kernels_space_050726 import HybridSpaceVecchiaFit
from GEMS_TCO.kernels_space_multiscale_050826 import _build_matern_spline_coeffs
from GEMS_TCO.kernels_space_trend_050726 import _MeanDesignMixin


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
    raise TypeError("smooth must be passed to HybridMMSpaceSpline*")


class _MMSpaceSplineMixin:
    def _finish_init_with_any_smooth(self, args, kwargs, mean_design):
        args, kwargs, requested_smooth = _replace_smooth_arg(args, kwargs)
        if requested_smooth <= 0:
            raise ValueError(f"smooth must be positive, got {requested_smooth}")
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

    def _matern_spline_eval(self, dist: torch.Tensor, smooth: float) -> torch.Tensor:
        sp = self._get_matern_spline_tensors(smooth)
        r_c = dist.clamp(0.0, sp["r_max"])
        orig_shape = r_c.shape
        r_flat = r_c.reshape(-1)
        idx = torch.searchsorted(sp["knots"], r_flat, right=True) - 1
        idx = idx.clamp(0, sp["knots"].numel() - 2)
        dx = r_flat - sp["knots"][idx]
        vals = sp["a"][idx] + dx * (sp["b"][idx] + dx * (sp["c"][idx] + dx * sp["d"][idx]))
        return vals.reshape(orig_shape).clamp_min(0.0)

    def _matern_corr(self, dist: torch.Tensor) -> torch.Tensor:
        if self.smooth == 0.5:
            return torch.exp(-dist)
        if self.smooth == 1.5:
            return (1.0 + dist) * torch.exp(-dist)
        return self._matern_spline_eval(dist, self.smooth)

    def _cov_from_deltas(self, d_lat, d_lon, params: torch.Tensor):
        sigmasq, range_space, _, _ = self._raw_params(params)
        euclid = torch.sqrt(d_lat.new_tensor(1e-8) + d_lat.pow(2) + d_lon.pow(2))
        return sigmasq * self._matern_corr(euclid / range_space.clamp_min(1e-12))


class _MMSpaceSplineNuggetMixin(_MMSpaceSplineMixin):
    def _raw_params(self, params: torch.Tensor):
        phi1 = torch.exp(params[0])
        phi2 = torch.exp(params[1])
        sigmasq = phi1 / phi2
        range_space = 1.0 / phi2
        nugget = torch.exp(params[2])
        return sigmasq, range_space, range_space, nugget

    def _convert_params(self, raw):
        phi1 = float(np.exp(raw[0]))
        phi2 = float(np.exp(raw[1]))
        return {
            "sigmasq": phi1 / phi2,
            "range": 1.0 / phi2,
            "nugget": float(np.exp(raw[2])),
            "phi1": phi1,
            "phi2": phi2,
        }


class _MMSpaceSplineNoNuggetMixin(_MMSpaceSplineMixin):
    def _raw_params(self, params: torch.Tensor):
        phi1 = torch.exp(params[0])
        phi2 = torch.exp(params[1])
        sigmasq = phi1 / phi2
        range_space = 1.0 / phi2
        nugget = params.new_tensor(0.0)
        return sigmasq, range_space, range_space, nugget

    def _convert_params(self, raw):
        phi1 = float(np.exp(raw[0]))
        phi2 = float(np.exp(raw[1]))
        return {
            "sigmasq": phi1 / phi2,
            "range": 1.0 / phi2,
            "nugget": 0.0,
            "phi1": phi1,
            "phi2": phi2,
        }


class HybridMMSpaceSplineTrendVecchiaFit(
    _MMSpaceSplineNuggetMixin, _MeanDesignMixin, HybridSpaceVecchiaFit
):
    """Max-min hybrid pure-space Vecchia with free nugget."""

    def __init__(self, *args, mean_design: str = "lat", **kwargs):
        self._finish_init_with_any_smooth(args, kwargs, mean_design)


class HybridMMSpaceSplineNoNuggetTrendVecchiaFit(
    _MMSpaceSplineNoNuggetMixin, _MeanDesignMixin, HybridSpaceVecchiaFit
):
    """Max-min hybrid pure-space Vecchia with nugget fixed at zero."""

    def __init__(self, *args, mean_design: str = "lat", **kwargs):
        self._finish_init_with_any_smooth(args, kwargs, mean_design)


__all__ = [
    "HybridMMSpaceSplineTrendVecchiaFit",
    "HybridMMSpaceSplineNoNuggetTrendVecchiaFit",
]
