"""
vecchia_st_spline.py

Spatio-temporal Vecchia wrappers with spline Matérn correlations.

Why this file exists
--------------------
The older ST Vecchia classes are intentionally limited to a few closed-form
smoothness values such as 0.5 and 1.5.  The July implied-spatial diagnostic
needs smoothness sweeps such as 0.2, 0.25, ..., 0.45.  This module keeps that
experiment in one direct place without changing the older kernels.

Parameterization
----------------
The inherited ST model uses

    sigmasq = phi1 / phi2
    scaled_distance = phi2 * d_ST

where ``d_ST`` is the anisotropic/advection-adjusted space-time distance from
the existing engine.  This file changes only the Matérn correlation shape:

    cov = sigmasq * Matern_corr(scaled_distance; smooth)

and keeps the existing optimizer, Vecchia conditioning, regression design, and
parameter interpretation.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.interpolate import CubicSpline
from scipy.special import gamma, kv

from GEMS_TCO.matern_vecchia_cluster_hybrid import ClusterHybridVecchiaFit
from GEMS_TCO.matern_vecchia_hybrid import HybridVecchiaFit


_MATERN_SPLINE_CACHE = {}


def _build_matern_spline_coeffs(smooth: float, n_points: int = 1200, r_max: float = 20.0):
    key = (round(float(smooth), 8), int(n_points), float(r_max))
    if key in _MATERN_SPLINE_CACHE:
        return _MATERN_SPLINE_CACHE[key]

    nu = float(smooth)
    if nu <= 0:
        raise ValueError(f"smooth must be positive, got {smooth}")

    r_arr = np.linspace(0.0, float(r_max), int(n_points), dtype=np.float64)
    f_arr = np.empty_like(r_arr)
    f_arr[0] = 1.0
    z = np.sqrt(2.0 * nu) * r_arr[1:]
    f_arr[1:] = (2.0 ** (1.0 - nu) / gamma(nu)) * (z ** nu) * kv(nu, z)
    f_arr = np.nan_to_num(f_arr, nan=0.0, posinf=1.0, neginf=0.0)
    f_arr = np.clip(f_arr, 0.0, 1.0)

    cs = CubicSpline(r_arr, f_arr, bc_type="natural")
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


class _STMaternSplineMixin:
    def _init_st_spline(self, smooth: float, n_points: int, r_max: float):
        smooth = float(smooth)
        if smooth <= 0:
            raise ValueError(f"smooth must be positive, got {smooth}")
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

    def _matern_spline_eval(self, r: torch.Tensor) -> torch.Tensor:
        sp = self._get_st_matern_spline_tensors()
        r_c = r.clamp(0.0, sp["r_max"])
        orig_shape = r_c.shape
        r_flat = r_c.reshape(-1)
        idx = torch.searchsorted(sp["knots"], r_flat, right=True) - 1
        idx = idx.clamp(0, sp["knots"].numel() - 2)
        dx = r_flat - sp["knots"][idx]
        vals = sp["a"][idx] + dx * (sp["b"][idx] + dx * (sp["c"][idx] + dx * sp["d"][idx]))
        return vals.reshape(orig_shape).clamp_min(0.0)

    def _matern_corr(self, scaled_distance: torch.Tensor) -> torch.Tensor:
        if np.isclose(self.smooth, 0.5):
            return torch.exp(-scaled_distance)
        if np.isclose(self.smooth, 1.5):
            return (1.0 + scaled_distance) * torch.exp(-scaled_distance)
        return self._matern_spline_eval(scaled_distance)

    def _nugget_from_params(self, params):
        return torch.exp(params[6])

    def matern_cov_batched(self, params, x_batch):
        phi1, phi2, phi3, phi4 = torch.exp(params[0:4])
        nugget = self._nugget_from_params(params)
        dist_params = torch.stack([phi3, phi4, params[4], params[5]])
        d = self.batched_manual_dist(dist_params, x_batch)
        cov = (phi1 / phi2) * self._matern_corr(d * phi2)

        B, N, _ = x_batch.shape
        eye = torch.eye(N, device=self.device, dtype=torch.float64).unsqueeze(0).expand(B, N, N)
        return cov + eye * (nugget + 1e-6)

    def matern_cov_aniso_STABLE_log_reparam(self, params, x, y):
        phi1, phi2, phi3, phi4 = torch.exp(params[0:4])
        nugget = self._nugget_from_params(params)
        sigmasq = phi1 / phi2

        dist_params = torch.stack([phi3, phi4, params[4], params[5]])
        distance = self.precompute_coords_aniso_STABLE(dist_params, x, y)
        cov = sigmasq * self._matern_corr(distance * phi2)

        if x.shape[0] == y.shape[0]:
            cov.diagonal().add_(nugget + 1e-8)
        return cov


class _STNoNuggetSplineMixin(_STMaternSplineMixin):
    def _nugget_from_params(self, params):
        return params.new_tensor(0.0)

    def _convert_params(self, raw):
        phi1, phi2, phi3, phi4 = np.exp(raw[0]), np.exp(raw[1]), np.exp(raw[2]), np.exp(raw[3])
        return {
            "sigma_sq": phi1 / phi2,
            "range_lon": 1.0 / phi2,
            "range_lat": 1.0 / (phi2 * np.sqrt(phi3)),
            "range_time": 1.0 / (phi2 * np.sqrt(phi4)),
            "advec_lat": raw[4],
            "advec_lon": raw[5],
            "nugget": 0.0,
        }


class HybridVecchiaSplineFit(_STMaternSplineMixin, HybridVecchiaFit):
    """Point-target hybrid ST Vecchia with arbitrary positive smoothness."""

    def __init__(
        self,
        smooth: float,
        input_map: dict,
        nns_map,
        mm_cond_number: int,
        nheads: int,
        limit_A: int = 20,
        limit_B_local: int = 16,
        limit_C_local: int = 12,
        daily_stride: int = 2,
        spatial_coords=None,
        lag1_lon_offset: float = 0.063,
        lag1_fresh_count: int = 2,
        lag2_fresh_count: int = 2,
        spline_n_points: int = 1200,
        spline_r_max: float = 20.0,
    ):
        super().__init__(
            smooth=0.5,
            input_map=input_map,
            nns_map=nns_map,
            mm_cond_number=mm_cond_number,
            nheads=nheads,
            limit_A=limit_A,
            limit_B_local=limit_B_local,
            limit_C_local=limit_C_local,
            daily_stride=daily_stride,
            spatial_coords=spatial_coords,
            lag1_lon_offset=lag1_lon_offset,
            lag1_fresh_count=lag1_fresh_count,
            lag2_fresh_count=lag2_fresh_count,
        )
        self._init_st_spline(smooth, spline_n_points, spline_r_max)


class HybridVecchiaNoNuggetSplineFit(_STNoNuggetSplineMixin, HybridVecchiaFit):
    """Point-target hybrid ST Vecchia with arbitrary smoothness and nugget fixed 0."""

    def __init__(
        self,
        smooth: float,
        input_map: dict,
        nns_map,
        mm_cond_number: int,
        nheads: int,
        limit_A: int = 20,
        limit_B_local: int = 16,
        limit_C_local: int = 12,
        daily_stride: int = 2,
        spatial_coords=None,
        lag1_lon_offset: float = 0.063,
        lag1_fresh_count: int = 2,
        lag2_fresh_count: int = 2,
        spline_n_points: int = 1200,
        spline_r_max: float = 20.0,
    ):
        super().__init__(
            smooth=0.5,
            input_map=input_map,
            nns_map=nns_map,
            mm_cond_number=mm_cond_number,
            nheads=nheads,
            limit_A=limit_A,
            limit_B_local=limit_B_local,
            limit_C_local=limit_C_local,
            daily_stride=daily_stride,
            spatial_coords=spatial_coords,
            lag1_lon_offset=lag1_lon_offset,
            lag1_fresh_count=lag1_fresh_count,
            lag2_fresh_count=lag2_fresh_count,
        )
        self._init_st_spline(smooth, spline_n_points, spline_r_max)


class ClusterHybridVecchiaSplineFit(_STMaternSplineMixin, ClusterHybridVecchiaFit):
    """Cluster-target hybrid ST Vecchia with arbitrary positive smoothness."""

    def __init__(
        self,
        smooth: float,
        input_map: dict,
        grid_coords=None,
        block_shape=(3, 3),
        n_neighbor_blocks_t: int = 6,
        lag1_same_block: bool = True,
        lag1_local_blocks: int = 3,
        lag1_shifted_blocks: int = 1,
        lag2_same_block: bool = True,
        lag2_local_blocks: int = 2,
        lag2_shifted_blocks: int = 1,
        daily_stride: int = 2,
        lag1_lon_offset: float = 0.063,
        lag2_lon_offset=None,
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
            block_shape=block_shape,
            n_neighbor_blocks_t=n_neighbor_blocks_t,
            lag1_same_block=lag1_same_block,
            lag1_local_blocks=lag1_local_blocks,
            lag1_shifted_blocks=lag1_shifted_blocks,
            lag2_same_block=lag2_same_block,
            lag2_local_blocks=lag2_local_blocks,
            lag2_shifted_blocks=lag2_shifted_blocks,
            daily_stride=daily_stride,
            lag1_lon_offset=lag1_lon_offset,
            lag2_lon_offset=lag2_lon_offset,
            target_chunk_size=target_chunk_size,
            min_target_points=min_target_points,
            max_neighbor_search=max_neighbor_search,
        )
        self._init_st_spline(smooth, spline_n_points, spline_r_max)


class ClusterHybridVecchiaNoNuggetSplineFit(_STNoNuggetSplineMixin, ClusterHybridVecchiaFit):
    """Cluster-target hybrid ST Vecchia with arbitrary smoothness and nugget fixed 0."""

    def __init__(
        self,
        smooth: float,
        input_map: dict,
        grid_coords=None,
        block_shape=(3, 3),
        n_neighbor_blocks_t: int = 6,
        lag1_same_block: bool = True,
        lag1_local_blocks: int = 3,
        lag1_shifted_blocks: int = 1,
        lag2_same_block: bool = True,
        lag2_local_blocks: int = 2,
        lag2_shifted_blocks: int = 1,
        daily_stride: int = 2,
        lag1_lon_offset: float = 0.063,
        lag2_lon_offset=None,
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
            block_shape=block_shape,
            n_neighbor_blocks_t=n_neighbor_blocks_t,
            lag1_same_block=lag1_same_block,
            lag1_local_blocks=lag1_local_blocks,
            lag1_shifted_blocks=lag1_shifted_blocks,
            lag2_same_block=lag2_same_block,
            lag2_local_blocks=lag2_local_blocks,
            lag2_shifted_blocks=lag2_shifted_blocks,
            daily_stride=daily_stride,
            lag1_lon_offset=lag1_lon_offset,
            lag2_lon_offset=lag2_lon_offset,
            target_chunk_size=target_chunk_size,
            min_target_points=min_target_points,
            max_neighbor_search=max_neighbor_search,
        )
        self._init_st_spline(smooth, spline_n_points, spline_r_max)


__all__ = [
    "HybridVecchiaSplineFit",
    "HybridVecchiaNoNuggetSplineFit",
    "ClusterHybridVecchiaSplineFit",
    "ClusterHybridVecchiaNoNuggetSplineFit",
]
