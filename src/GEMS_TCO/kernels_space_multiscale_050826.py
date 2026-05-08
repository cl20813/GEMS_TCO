"""
kernels_space_multiscale_050826.py

Pure-space two-scale isotropic Vecchia diagnostics.

The intended first use is a stable grid-search diagnostic:
  fixed range_short, fixed range_long, fixed nugget,
  estimate only log(sigmasq_short), log(sigmasq_long).

This avoids the identifiability problems of a fully free two-Matern model while
testing whether fine-resolution data increasingly prefer a short-range
component.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.interpolate import CubicSpline as _CubicSpline
from scipy.special import gamma as _scipy_gamma
from scipy.special import kv as _scipy_kv

from GEMS_TCO.kernels_space_050726 import ColumnSpaceVecchiaFit, HybridSpaceVecchiaFit
from GEMS_TCO.kernels_space_trend_050726 import _MeanDesignMixin


_MATERN_SPLINE_CACHE = {}


def _build_matern_spline_coeffs(smooth: float, n_points: int = 1200, r_max: float = 20.0):
    """Build natural-cubic spline coefficients for the standard Matérn correlation."""
    key = (round(float(smooth), 8), int(n_points), float(r_max))
    if key in _MATERN_SPLINE_CACHE:
        return _MATERN_SPLINE_CACHE[key]

    nu = float(smooth)
    r_arr = np.linspace(0.0, float(r_max), int(n_points), dtype=np.float64)
    f_arr = np.empty_like(r_arr)
    f_arr[0] = 1.0
    z = np.sqrt(2.0 * nu) * r_arr[1:]
    f_arr[1:] = (2.0 ** (1.0 - nu) / _scipy_gamma(nu)) * (z ** nu) * _scipy_kv(nu, z)
    f_arr = np.nan_to_num(f_arr, nan=0.0, posinf=1.0, neginf=0.0)
    f_arr = np.clip(f_arr, 0.0, 1.0)

    cs = _CubicSpline(r_arr, f_arr, bc_type="natural")
    coeffs = {
        "knots": r_arr,
        "a": cs.c[3].copy(),
        "b": cs.c[2].copy(),
        "c": cs.c[1].copy(),
        "d": cs.c[0].copy(),
        "r_max": float(r_max),
    }
    _MATERN_SPLINE_CACHE[key] = coeffs
    return coeffs


class _TwoScaleFixedRangeMixin:
    def _get_matern_spline_tensors(self, smooth: float):
        if not hasattr(self, "_matern_spline_tensors"):
            self._matern_spline_tensors = {}
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

    def _matern_corr(self, dist: torch.Tensor, smooth: float) -> torch.Tensor:
        smooth = float(smooth)
        if smooth == 0.5:
            return torch.exp(-dist)
        if smooth == 1.5:
            return (1.0 + dist) * torch.exp(-dist)
        if smooth <= 0.0:
            raise ValueError(f"smooth must be positive, got {smooth}")
        return self._matern_spline_eval(dist, smooth)

    def _raw_params(self, params: torch.Tensor):
        sigmasq_short = torch.exp(params[0])
        sigmasq_long = torch.exp(params[1])
        total_sigmasq = sigmasq_short + sigmasq_long
        range_short = params.new_tensor(float(self.range_short))
        range_long = params.new_tensor(float(self.range_long))
        nugget = params.new_tensor(float(self.nugget_fixed))
        return total_sigmasq, range_short, range_long, nugget

    def _cov_from_deltas(self, d_lat, d_lon, params: torch.Tensor):
        sigmasq_short = torch.exp(params[0])
        sigmasq_long = torch.exp(params[1])
        euclid = torch.sqrt(d_lat.new_tensor(1e-8) + d_lat.pow(2) + d_lon.pow(2))
        corr_short = self._matern_corr(euclid / float(self.range_short), self.smooth_short)
        corr_long = self._matern_corr(euclid / float(self.range_long), self.smooth_long)
        return sigmasq_short * corr_short + sigmasq_long * corr_long

    def _convert_params(self, raw):
        sigmasq_short = float(np.exp(raw[0]))
        sigmasq_long = float(np.exp(raw[1]))
        return {
            "sigmasq_short": sigmasq_short,
            "sigmasq_long": sigmasq_long,
            "sigmasq_total": sigmasq_short + sigmasq_long,
            "range_short": float(self.range_short),
            "range_long": float(self.range_long),
            "nugget": float(self.nugget_fixed),
        }


class ColumnSpaceTwoScaleFixedRangeTrendVecchiaFit(
    _TwoScaleFixedRangeMixin, _MeanDesignMixin, ColumnSpaceVecchiaFit
):
    def __init__(
        self,
        *args,
        range_short: float,
        range_long: float,
        nugget_fixed: float = 0.0,
        smooth_short: float | None = None,
        smooth_long: float | None = None,
        mean_design: str = "latlon",
        **kwargs,
    ):
        if not float(range_long) > float(range_short):
            raise ValueError("range_long must be larger than range_short")
        super().__init__(*args, **kwargs)
        self.range_short = float(range_short)
        self.range_long = float(range_long)
        self.nugget_fixed = float(nugget_fixed)
        self.smooth_short = float(self.smooth if smooth_short is None else smooth_short)
        self.smooth_long = float(self.smooth if smooth_long is None else smooth_long)
        self._init_mean_design(mean_design)


class HybridSpaceTwoScaleFixedRangeTrendVecchiaFit(
    _TwoScaleFixedRangeMixin, _MeanDesignMixin, HybridSpaceVecchiaFit
):
    def __init__(
        self,
        *args,
        range_short: float,
        range_long: float,
        nugget_fixed: float = 0.0,
        smooth_short: float | None = None,
        smooth_long: float | None = None,
        mean_design: str = "latlon",
        **kwargs,
    ):
        if not float(range_long) > float(range_short):
            raise ValueError("range_long must be larger than range_short")
        super().__init__(*args, **kwargs)
        self.range_short = float(range_short)
        self.range_long = float(range_long)
        self.nugget_fixed = float(nugget_fixed)
        self.smooth_short = float(self.smooth if smooth_short is None else smooth_short)
        self.smooth_long = float(self.smooth if smooth_long is None else smooth_long)
        self._init_mean_design(mean_design)


__all__ = [
    "ColumnSpaceTwoScaleFixedRangeTrendVecchiaFit",
    "HybridSpaceTwoScaleFixedRangeTrendVecchiaFit",
]
