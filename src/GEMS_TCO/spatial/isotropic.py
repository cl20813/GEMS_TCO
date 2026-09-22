"""Pure-space isotropic block-Vecchia model.

The covariance, GLS mean profiling, optimizer, smoothness handling, and
parameter interpretation form one block-target Matérn implementation. Its
likelihood geometry is:

  - build fixed regular-grid clusters, typically 4x4 grid cells;
  - order cluster centroids by max-min ordering;
  - each target is the whole cluster block;
  - condition only on previous same-time cluster blocks in that max-min order.

There is no temporal conditioning; time slots are independent spatial
replicates with shared covariance parameters.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from GEMS_TCO import orderings

from .base import _MeanDesignMixin, _PureSpaceVecchiaBase, _stable_sqrt_distance
from .matern_spline import _build_matern_spline_coeffs


class _MaternCorrelationMixin:
    """Matérn correlation using the standard geometric range convention.

    If ``dist`` is Euclidean distance divided by ``range``, the Bessel argument
    is ``sqrt(2 * nu) * dist``.  Thus the closed forms are ``exp(-dist)`` for
    nu=0.5 and ``(1 + sqrt(3) * dist) exp(-sqrt(3) * dist)`` for nu=1.5.
    Arbitrary smoothness uses the shared spline table on this same ``dist``
    scale, matching :func:`GEMS_TCO.spatial.matern_bessel.matern_corr_bessel`.
    """

    _spline_r_max = 20.0

    def _init_matern_correlation(self, smooth: float) -> None:
        self.smooth = float(smooth)
        if not math.isfinite(self.smooth) or self.smooth <= 0:
            raise ValueError(f"smooth must be finite and positive, got {smooth}")
        self._matern_spline_tensors = {}

    def _get_matern_spline_tensors(self, smooth: float):
        key = round(float(smooth), 8)
        if key in self._matern_spline_tensors:
            return self._matern_spline_tensors[key]
        coeffs = _build_matern_spline_coeffs(
            float(smooth),
            r_max=self._spline_r_max,
        )
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
            argument = math.sqrt(3.0) * dist
            return (1.0 + argument) * torch.exp(-argument)
        sp = self._get_matern_spline_tensors(self.smooth)
        outside_table = dist > self._spline_r_max
        r_c = dist.clamp(0.0, sp["r_max"])
        orig_shape = r_c.shape
        r_flat = r_c.reshape(-1)
        idx = torch.searchsorted(sp["knots"], r_flat, right=True) - 1
        idx = idx.clamp(0, sp["knots"].numel() - 2)
        dx = r_flat - sp["knots"][idx]
        vals = sp["a"][idx] + dx * (sp["b"][idx] + dx * (sp["c"][idx] + dx * sp["d"][idx]))
        vals = vals.reshape(orig_shape).clamp(0.0, 1.0)
        return vals.masked_fill(outside_table, 0.0)


class _IsoSpaceMixin(_MaternCorrelationMixin):

    _n_covariance_parameters = 3

    def _raw_params(self, params: torch.Tensor):
        sigmasq = torch.exp(params[0])
        range_space = torch.exp(params[1])
        nugget = torch.exp(params[2])
        return sigmasq, range_space, range_space, nugget

    def _cov_from_deltas(self, d_lat, d_lon, params: torch.Tensor):
        sigmasq, range_space, _, _ = self._raw_params(params)
        dist = _stable_sqrt_distance(d_lat.pow(2) + d_lon.pow(2)) / range_space
        return sigmasq * self._matern_corr(dist)

    def _convert_params(self, raw):
        sigmasq, range_space, _, nugget = self._natural_values_from_raw_sequence(raw)
        return {
            "signal_variance": sigmasq,
            "range": range_space,
            "nugget": nugget,
        }


class _IsoNoNuggetSpaceMixin(_MaternCorrelationMixin):
    _n_covariance_parameters = 2

    def _raw_params(self, params: torch.Tensor):
        sigmasq = torch.exp(params[0])
        range_space = torch.exp(params[1])
        nugget = params.new_tensor(0.0)
        return sigmasq, range_space, range_space, nugget

    def _cov_from_deltas(self, d_lat, d_lon, params: torch.Tensor):
        sigmasq, range_space, _, _ = self._raw_params(params)
        dist = _stable_sqrt_distance(d_lat.pow(2) + d_lon.pow(2)) / range_space
        return sigmasq * self._matern_corr(dist)

    def _convert_params(self, raw):
        sigmasq, range_space, _, nugget = self._natural_values_from_raw_sequence(raw)
        return {
            "signal_variance": sigmasq,
            "range": range_space,
            "nugget": nugget,
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


class _ClusterSpatialVecchiaBase(_PureSpaceVecchiaBase):
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
        try:
            block_shape_values = tuple(block_shape)
        except TypeError as exc:
            raise ValueError("block_shape must contain exactly two positive integers") from exc
        if len(block_shape_values) != 2:
            raise ValueError("block_shape must contain exactly two positive integers")
        if any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            for value in block_shape_values
        ):
            raise ValueError("block_shape must contain exactly two positive integers")
        self.grid_coords = (
            None if grid_coords is None else np.asarray(grid_coords, dtype=np.float64)
        )
        self.block_shape = (int(block_shape_values[0]), int(block_shape_values[1]))
        self.n_neighbor_blocks = int(n_neighbor_blocks)
        self.min_target_points = int(min_target_points)
        self.max_neighbor_search = max_neighbor_search
        self.lat_round_decimals = int(lat_round_decimals)
        self.lon_round_decimals = int(lon_round_decimals)

        if self.block_shape[0] <= 0 or self.block_shape[1] <= 0:
            raise ValueError(f"block_shape must be positive, got {self.block_shape}")
        if self.n_neighbor_blocks <= 0:
            raise ValueError("n_neighbor_blocks must be positive")
        if self.min_target_points <= 0:
            raise ValueError("min_target_points must be positive")
        if self.max_neighbor_search is not None and int(self.max_neighbor_search) <= 0:
            raise ValueError("max_neighbor_search must be positive when provided")

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
            coords = np.asarray(self.grid_coords, dtype=np.float64)
        else:
            first = next(iter(self.input_map.values()))
            if isinstance(first, torch.Tensor):
                coords = first[:n_points, :2].detach().cpu().numpy().astype(np.float64)
            else:
                coords = np.asarray(first[:n_points, :2], dtype=np.float64)
        if coords.shape != (n_points, 2):
            raise ValueError(f"grid_coords must have shape ({n_points}, 2), got {coords.shape}")
        if not np.isfinite(coords).all():
            raise ValueError("grid_coords must contain only finite coordinates")
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
        if not np.array_equal(np.sort(order), np.arange(n, dtype=np.int64)):
            raise ValueError("max-min order must be a permutation of all cluster indices")
        return order

    def _build_clusters(self, n_points: int) -> None:
        if int(n_points) <= 0:
            raise ValueError("the spatial grid must contain at least one point")
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

        order = self._as_zero_based_order(orderings.maxmin_order(raw_centroids), len(raw_points))
        self.cluster_points = [raw_points[i] for i in order]
        self.cluster_centroids = raw_centroids[order]
        self.n_clusters = len(self.cluster_points)
        self.max_points_per_cluster = max((len(p) for p in self.cluster_points), default=0)

        max_blocks = self.max_neighbor_search
        if max_blocks is None:
            max_blocks = self.n_neighbor_blocks + 8
        self.cluster_nns = orderings.predecessor_neighbors(
            self.cluster_centroids, max_neighbors=int(max_blocks)
        )

    @staticmethod
    def _append_previous_blocks(
        out: List[int], candidates: Sequence[int], block_idx: int, cap: int
    ) -> None:
        seen = set(out)
        for cand in candidates:
            if len(out) >= cap:
                break
            cand = int(cand)
            if cand < 0 or cand >= int(block_idx) or cand in seen:
                continue
            out.append(cand)
            seen.add(cand)

    def precompute_conditioning_sets(self, verbose: bool = False):
        """Build grouped pure-spatial conditionals for every replicate."""

        if verbose:
            print(
                f"Pre-computing IsotropicMaternSpatialVecchia "
                f"[block={self.block_shape[0]}x{self.block_shape[1]}, "
                f"B={self.n_neighbor_blocks}]...",
                end=" ",
            )
        t0 = time.time()
        first_data = next(iter(self.input_map.values()))
        n_grid_for_clusters = int(first_data.shape[0])
        self._build_clusters(n_grid_for_clusters)
        max_cond_points = int(self.n_neighbor_blocks * self.max_points_per_cluster)
        all_data_list, full_data, n_real, _, day_lengths, cumulative_len, valid_obs_np = (
            self._make_full_data(max_cond_points)
        )
        del all_data_list

        n_grid = day_lengths[0]
        if n_grid != n_grid_for_clusters:
            raise ValueError(
                f"Cluster grid length {n_grid_for_clusters} != data grid length {n_grid}"
            )
        dummy_start = n_real

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
                target = [offset + int(p) for p in point_locals if valid_obs_np[offset + int(p)]]
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
                    padded = [
                        dummy_start + k for k in range(max_cond_points - len(cond_points))
                    ] + cond_points
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

        self.is_precomputed = True

        m_arr = np.asarray(m_sizes, dtype=float) if m_sizes else np.asarray([0.0])
        t_arr = np.asarray(target_sizes, dtype=float) if target_sizes else np.asarray([0.0])
        if verbose:
            print(
                f"Done in {time.time() - t0:.1f}s. clusters={self.n_clusters}, "
                f"max_points/block={self.max_points_per_cluster}, "
                f"target_blocks={self.n_target_blocks}, "
                f"target_points={self.n_target_points}, "
                f"m mean/med/max={m_arr.mean():.1f}/{np.median(m_arr):.0f}/{m_arr.max():.0f}, "
                f"target med/max={np.median(t_arr):.0f}/{t_arr.max():.0f}"
            )
        return self

    def _accumulate_gls_stats(
        self, params: torch.Tensor, include_y_quad: bool = True, catch_cholesky: bool = False
    ):
        self._check_precomputed()
        XT_Sinv_X = torch.zeros(
            (self.n_features, self.n_features), device=self.device, dtype=torch.float64
        )
        XT_Sinv_y = torch.zeros((self.n_features, 1), device=self.device, dtype=torch.float64)
        yT_Sinv_y = torch.tensor(0.0, device=self.device, dtype=torch.float64)
        log_det = torch.tensor(0.0, device=self.device, dtype=torch.float64)
        if not torch.isfinite(params).all():
            if catch_cholesky:
                return None
            raise ValueError("covariance parameters must be finite")

        chunk_size = max(1, int(self.target_chunk_size))
        for batch in self._cluster_batches:
            if batch.coords.shape[0] == 0:
                continue
            target_slice = slice(batch.max_cond_points, batch.max_cond_points + batch.target_size)
            for start in range(0, batch.coords.shape[0], chunk_size):
                end = min(start + chunk_size, batch.coords.shape[0])
                try:
                    K = self._cov_full(batch.coords[start:end], params)
                    dummy = batch.is_dummy[start:end].squeeze(-1)
                    if torch.any(dummy):
                        real_pair = (~dummy).unsqueeze(2) & (~dummy).unsqueeze(1)
                        K = K.masked_fill(~real_pair, 0.0)
                        K = K + torch.diag_embed(dummy.to(dtype=K.dtype))
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

                if torch.any(diag_L <= 1e-12) or not torch.isfinite(diag_L).all():
                    if catch_cholesky:
                        return None
                    raise torch.linalg.LinAlgError("non-positive cluster conditional diagonal")

                log_det += 2.0 * torch.log(diag_L).sum()
                XT_Sinv_X += u_X.T @ u_X
                XT_Sinv_y += u_X.T @ u_y
                if include_y_quad:
                    yT_Sinv_y += (u_y.T @ u_y).squeeze()

        if self.n_target_points <= 0:
            if catch_cholesky:
                return None
            raise ValueError("the precomputed likelihood contains no target observations")
        return XT_Sinv_X, XT_Sinv_y, yT_Sinv_y, log_det, int(self.n_target_points)

    def cluster_summary(self) -> dict[str, float]:
        """Return geometry and batching counts after precomputation."""

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


class IsotropicMaternSpatialVecchia(
    _IsoSpaceMixin,
    _MeanDesignMixin,
    _ClusterSpatialVecchiaBase,
):
    """Block-target spatial Vecchia model with isotropic Matérn covariance.

    The default ``latlon_hour`` mean contains centered latitude/longitude and
    seven hour indicators; ``latlon`` is the three-column ``[1, lat, lon]``
    design shared with the direct full-likelihood implementation.
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


class NoNuggetIsotropicMaternSpatialVecchia(
    _IsoNoNuggetSpaceMixin,
    _MeanDesignMixin,
    _ClusterSpatialVecchiaBase,
):
    """Isotropic Matérn spatial Vecchia model with nugget fixed to zero.

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
    "IsotropicMaternSpatialVecchia",
    "NoNuggetIsotropicMaternSpatialVecchia",
]
