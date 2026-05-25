"""
kernels_space_iso_cluster_052426.py

Pure-space isotropic cluster Vecchia competitor.

Created 2026-05-24.

This module is the default isotropic pure-space Vecchia implementation.  The
covariance, GLS mean profiling, optimizer, smooth handling, and parameter
interpretation are the same isotropic pure-space Matérn setup used in the
earlier pointwise code.  The key likelihood geometry is cluster based:

  - build fixed regular-grid clusters, typically 4x4 grid cells;
  - order cluster centroids by max-min ordering;
  - each target is the whole cluster block;
  - condition only on previous same-time cluster blocks in that max-min order.

There is no t-1 or t-2 logic here.  This is deliberately the simplest cluster
pure-space analogue of the pointwise max-min Vecchia model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from GEMS_TCO import orderings
from GEMS_TCO.kernels_space_base_engine_052126 import (
    _MeanDesignMixin,
    _PureSpaceVecchiaBase,
    _build_matern_spline_coeffs,
)


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
    raise TypeError("smooth must be passed to the pure-space isotropic cluster Vecchia class")


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


@dataclass
class _PureSpaceClusterBatch:
    max_cond_points: int
    target_size: int
    rows: torch.Tensor
    coords: torch.Tensor
    X: torch.Tensor
    y: torch.Tensor
    is_dummy: torch.Tensor


class ClusterSpaceVecchiaFit(_PureSpaceVecchiaBase):
    """Block-target pure-space Vecchia with previous max-min cluster neighbors."""

    def __init__(
        self,
        smooth: float,
        input_map: Dict[str, torch.Tensor],
        grid_coords: Optional[np.ndarray] = None,
        block_shape: Tuple[int, int] = (4, 4),
        n_neighbor_blocks: int = 6,
        target_chunk_size: int = 128,
        min_target_points: int = 1,
        max_neighbor_search: Optional[int] = None,
        lat_round_decimals: int = 10,
        lon_round_decimals: int = 10,
    ):
        super().__init__(smooth=smooth, input_map=input_map, target_chunk_size=target_chunk_size)
        self.grid_coords = None if grid_coords is None else np.asarray(grid_coords, dtype=np.float64)
        self.block_shape = (int(block_shape[0]), int(block_shape[1]))
        self.n_neighbor_blocks = int(n_neighbor_blocks)
        self.min_target_points = int(min_target_points)
        self.max_neighbor_search = max_neighbor_search
        self.lat_round_decimals = int(lat_round_decimals)
        self.lon_round_decimals = int(lon_round_decimals)

        if self.block_shape[0] <= 0 or self.block_shape[1] <= 0:
            raise ValueError(f"block_shape must be positive, got {self.block_shape}")
        if self.n_neighbor_blocks <= 0:
            raise ValueError("n_neighbor_blocks must be positive")

        self.cluster_points: List[np.ndarray] = []
        self.cluster_centroids: Optional[np.ndarray] = None
        self.cluster_nns: Optional[np.ndarray] = None
        self.n_clusters = 0
        self.max_points_per_cluster = 0
        self.n_target_blocks = 0
        self.n_target_points = 0
        self._cluster_batches: List[_PureSpaceClusterBatch] = []

    def _grid_coords_np(self, n_points: int) -> np.ndarray:
        if self.grid_coords is not None:
            coords = np.asarray(self.grid_coords[:n_points], dtype=np.float64)
        else:
            first = next(iter(self.input_map.values()))
            if isinstance(first, torch.Tensor):
                coords = first[:n_points, :2].detach().cpu().numpy().astype(np.float64)
            else:
                coords = np.asarray(first[:n_points, :2], dtype=np.float64)
        if coords.shape != (n_points, 2):
            raise ValueError(f"grid_coords must have shape ({n_points}, 2), got {coords.shape}")
        if np.isnan(coords).any():
            raise ValueError("grid_coords contains NaN; cluster construction needs finite grid coordinates")
        return coords

    @staticmethod
    def _unique_inverse(values: np.ndarray, decimals: int) -> Tuple[np.ndarray, np.ndarray]:
        rounded = np.round(values.astype(np.float64), int(decimals))
        unique = np.unique(rounded)
        lookup = {v: i for i, v in enumerate(unique)}
        inverse = np.asarray([lookup[v] for v in rounded], dtype=np.int64)
        return unique, inverse

    @staticmethod
    def _as_zero_based_order(order: np.ndarray, n: int) -> np.ndarray:
        order = np.asarray(order, dtype=np.int64)
        if order.size != n:
            raise ValueError(f"max-min order length {order.size} != n_clusters {n}")
        if order.min() == 1 and order.max() == n:
            order = order - 1
        if order.min() < 0 or order.max() >= n:
            raise ValueError("max-min order has indices outside [0, n_clusters)")
        return order

    def _build_clusters(self, n_points: int) -> None:
        coords = self._grid_coords_np(n_points)
        _, row_idx = self._unique_inverse(coords[:, 0], self.lat_round_decimals)
        _, col_idx = self._unique_inverse(coords[:, 1], self.lon_round_decimals)

        br, bc = self.block_shape
        block_rows = np.floor_divide(row_idx, br)
        block_cols = np.floor_divide(col_idx, bc)
        raw: dict[tuple[int, int], list[int]] = {}
        for local_idx, key in enumerate(zip(block_rows.tolist(), block_cols.tolist())):
            raw.setdefault(key, []).append(local_idx)

        raw_keys = sorted(raw)
        raw_points = [np.asarray(raw[k], dtype=np.int64) for k in raw_keys]
        raw_centroids = np.vstack([coords[p].mean(axis=0) for p in raw_points])

        order = self._as_zero_based_order(orderings.maxmin_cpp(raw_centroids), len(raw_points))
        self.cluster_points = [raw_points[i] for i in order]
        self.cluster_centroids = raw_centroids[order]
        self.n_clusters = len(self.cluster_points)
        self.max_points_per_cluster = max((len(p) for p in self.cluster_points), default=0)

        max_blocks = self.max_neighbor_search
        if max_blocks is None:
            max_blocks = self.n_neighbor_blocks + 8
        self.cluster_nns = orderings.find_nns_l2(self.cluster_centroids, max_nn=int(max_blocks))

    @staticmethod
    def _append_previous_blocks(out: List[int], candidates: Sequence[int], block_idx: int, cap: int) -> None:
        seen = set(out)
        for cand in candidates:
            if len(out) >= cap:
                break
            cand = int(cand)
            if cand < 0 or cand >= int(block_idx) or cand in seen:
                continue
            out.append(cand)
            seen.add(cand)

    def _cov_full(self, coords: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)
        cov = self._cov_from_deltas(diff[..., 0], diff[..., 1], params)
        _, _, _, nugget = self._raw_params(params)
        n = coords.shape[1]
        eye = torch.eye(n, device=self.device, dtype=torch.float64).unsqueeze(0)
        return cov + eye * (nugget + 1e-6)

    def precompute_conditioning_sets(self):
        print(
            f"Pre-computing ClusterSpaceVecchia [block={self.block_shape[0]}x{self.block_shape[1]}, "
            f"B={self.n_neighbor_blocks}]...",
            end=" ",
        )
        t0 = time.time()
        max_cond_points = int(self.n_neighbor_blocks * max(1, self.block_shape[0] * self.block_shape[1]))
        all_data_list, full_data, n_real, num_cols, day_lengths, cumulative_len, valid_obs_np = self._make_full_data(
            max_cond_points
        )
        del all_data_list

        n_grid = day_lengths[0]
        self._build_clusters(n_grid)
        dummy_start = n_real
        max_cond_points = int(self.n_neighbor_blocks * self.max_points_per_cluster)

        batch_rows: dict[tuple[int, int], list[list[int]]] = {}
        m_sizes: list[int] = []
        target_sizes: list[int] = []
        self.n_target_blocks = 0
        self.n_target_points = 0

        if self.cluster_nns is None:
            raise RuntimeError("cluster_nns was not built")

        for time_idx, day_len in enumerate(day_lengths):
            offset = int(cumulative_len[time_idx])
            for block_idx, point_locals in enumerate(self.cluster_points):
                target = [
                    offset + int(p)
                    for p in point_locals
                    if valid_obs_np[offset + int(p)]
                ]
                if len(target) < self.min_target_points:
                    continue

                prev_blocks: list[int] = []
                self._append_previous_blocks(
                    prev_blocks,
                    self.cluster_nns[block_idx],
                    block_idx=block_idx,
                    cap=self.n_neighbor_blocks,
                )

                cond_points: list[int] = []
                seen_points = set()
                for cond_block in prev_blocks:
                    for p in self.cluster_points[cond_block]:
                        g = offset + int(p)
                        if g in seen_points or not valid_obs_np[g]:
                            continue
                        cond_points.append(g)
                        seen_points.add(g)

                m_sizes.append(len(cond_points))
                target_sizes.append(len(target))
                if len(cond_points) < max_cond_points:
                    padded = [dummy_start + k for k in range(max_cond_points - len(cond_points))] + cond_points
                else:
                    padded = cond_points[-max_cond_points:]

                batch_rows.setdefault((max_cond_points, len(target)), []).append(padded + target)
                self.n_target_blocks += 1
                self.n_target_points += len(target)

        self._cluster_batches = []
        for key in sorted(batch_rows, key=lambda x: (x[0], x[1])):
            m, target_size = key
            rows = torch.tensor(batch_rows[key], device=self.device, dtype=torch.long)
            gathered = full_data[rows].contiguous()
            is_dummy = (rows >= dummy_start).unsqueeze(-1)
            X = self._design_from_rows(gathered).masked_fill(is_dummy, 0.0).contiguous()
            y = gathered[..., 2:3].masked_fill(is_dummy, 0.0).contiguous()
            self._cluster_batches.append(
                _PureSpaceClusterBatch(
                    max_cond_points=int(m),
                    target_size=int(target_size),
                    rows=rows,
                    coords=gathered[..., [0, 1]].contiguous(),
                    X=X,
                    y=y,
                    is_dummy=is_dummy,
                )
            )

        self.Full_Data = full_data
        self.Heads_data = torch.empty((0, num_cols), device=self.device, dtype=torch.float64)
        self.Batched_Groups = []
        self.Grouped_Batches = []
        self.n_tails = int(self.n_target_points)
        self.is_precomputed = True

        m_arr = np.asarray(m_sizes, dtype=float) if m_sizes else np.asarray([0.0])
        t_arr = np.asarray(target_sizes, dtype=float) if target_sizes else np.asarray([0.0])
        print(
            f"Done in {time.time() - t0:.1f}s. clusters={self.n_clusters}, "
            f"max_points/block={self.max_points_per_cluster}, target_blocks={self.n_target_blocks}, "
            f"target_points={self.n_target_points}, m mean/med/max={m_arr.mean():.1f}/{np.median(m_arr):.0f}/{m_arr.max():.0f}, "
            f"target med/max={np.median(t_arr):.0f}/{t_arr.max():.0f}"
        )
        return self

    def _accumulate_gls_stats(self, params: torch.Tensor, include_y_quad: bool = True, catch_cholesky: bool = False):
        self._check_precomputed()
        XT_Sinv_X = torch.zeros((self.n_features, self.n_features), device=self.device, dtype=torch.float64)
        XT_Sinv_y = torch.zeros((self.n_features, 1), device=self.device, dtype=torch.float64)
        yT_Sinv_y = torch.tensor(0.0, device=self.device, dtype=torch.float64)
        log_det = torch.tensor(0.0, device=self.device, dtype=torch.float64)

        chunk_size = max(1, int(self.target_chunk_size))
        for batch in self._cluster_batches:
            if batch.coords.shape[0] == 0:
                continue
            target_slice = slice(batch.max_cond_points, batch.max_cond_points + batch.target_size)
            for start in range(0, batch.coords.shape[0], chunk_size):
                end = min(start + chunk_size, batch.coords.shape[0])
                try:
                    K = self._cov_full(batch.coords[start:end], params)
                    L = torch.linalg.cholesky(K)
                except torch.linalg.LinAlgError:
                    if catch_cholesky:
                        return None
                    raise

                z_X = torch.linalg.solve_triangular(L, batch.X[start:end], upper=False)
                z_y = torch.linalg.solve_triangular(L, batch.y[start:end], upper=False)
                u_X = z_X[:, target_slice, :].reshape(-1, self.n_features)
                u_y = z_y[:, target_slice, :].reshape(-1, 1)
                diag_L = torch.diagonal(L, dim1=1, dim2=2)[:, target_slice]

                if torch.any(diag_L <= 1e-12) or torch.isnan(diag_L).any():
                    if catch_cholesky:
                        return None
                    raise torch.linalg.LinAlgError("non-positive cluster conditional diagonal")

                log_det += 2.0 * torch.log(diag_L).sum()
                XT_Sinv_X += u_X.T @ u_X
                XT_Sinv_y += u_X.T @ u_y
                if include_y_quad:
                    yT_Sinv_y += (u_y.T @ u_y).squeeze()

        return XT_Sinv_X, XT_Sinv_y, yT_Sinv_y, log_det, int(self.n_target_points)

    def cluster_summary(self) -> dict[str, float]:
        if not self.is_precomputed:
            raise RuntimeError("Run precompute_conditioning_sets() first")
        m_vals = np.asarray([b.max_cond_points for b in self._cluster_batches], dtype=float)
        t_vals = np.asarray([b.target_size for b in self._cluster_batches], dtype=float)
        return {
            "n_clusters": int(self.n_clusters),
            "block_shape_lat": int(self.block_shape[0]),
            "block_shape_lon": int(self.block_shape[1]),
            "n_neighbor_blocks": int(self.n_neighbor_blocks),
            "max_points_per_cluster": int(self.max_points_per_cluster),
            "n_target_blocks": int(self.n_target_blocks),
            "n_target_points": int(self.n_target_points),
            "n_batches": int(len(self._cluster_batches)),
            "max_cond_points": int(m_vals.max()) if m_vals.size else 0,
            "median_cond_points": float(np.median(m_vals)) if m_vals.size else 0.0,
            "median_target_size": float(np.median(t_vals)) if t_vals.size else 0.0,
            "max_target_size": int(t_vals.max()) if t_vals.size else 0,
            "target_chunk_size": int(self.target_chunk_size),
        }


class ClusterSpaceIsoTrendVecchiaFit(_IsoSpaceMixin, _MeanDesignMixin, ClusterSpaceVecchiaFit):
    def __init__(self, *args, mean_design: str = "latlon", **kwargs):
        self._finish_init_with_any_smooth(args, kwargs, mean_design)


class ClusterSpaceIsoNoNuggetTrendVecchiaFit(_IsoNoNuggetSpaceMixin, _MeanDesignMixin, ClusterSpaceVecchiaFit):
    def __init__(self, *args, mean_design: str = "latlon", **kwargs):
        self._finish_init_with_any_smooth(args, kwargs, mean_design)


__all__ = [
    "ClusterSpaceVecchiaFit",
    "ClusterSpaceIsoTrendVecchiaFit",
    "ClusterSpaceIsoNoNuggetTrendVecchiaFit",
]
