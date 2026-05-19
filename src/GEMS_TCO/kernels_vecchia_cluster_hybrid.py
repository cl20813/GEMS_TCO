"""
kernels_vecchia_cluster_hybrid.py

Cluster-target hybrid Vecchia prototype for real GEMS TCO experiments.

This module is intentionally independent of the point-target hybrid module.
It builds 3x3 (or user-specified) grid-cell clusters, orders the clusters by
max-min ordering of cluster centroids, and evaluates a block conditional
Vecchia likelihood.  Covariances are still computed from the coordinates in
``input_map``; pass ``grid_coords`` when ``input_map`` uses source coordinates
so clustering is based on the regular grid while covariance uses real
locations.

Default conditioning budget for a target cluster:
  A at t:     6 previous spatial neighbor clusters
  B at t-1:  same cluster + 3 local clusters + 1 shifted fresh cluster
  C at t-2:  same cluster + 2 local clusters + 1 shifted fresh cluster
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.neighbors import BallTree

from GEMS_TCO import kernels_vecchia
from GEMS_TCO import orderings


@dataclass
class _ClusterBatch:
    label: str
    max_cond_points: int
    target_size: int
    X: torch.Tensor
    Y: torch.Tensor
    Locs: torch.Tensor
    T: torch.Tensor
    is_dummy: torch.Tensor


class ClusterHybridVecchiaFit(kernels_vecchia.fit_vecchia_lbfgs):
    """Hybrid Vecchia model with cluster-valued targets.

    Parameters
    ----------
    input_map
        Dict of hourly tensors with columns
        [lat, lon, centered_o3, time, hour dummies...].
    grid_coords
        Optional ``(n_points, 2)`` array of regular-grid [lat, lon] coordinates
        matching the local point order of ``input_map``.  If ``input_map`` was
        built with ``keep_ori=True``, pass grid coordinates here so clusters are
        made on the grid but covariance uses source coordinates.
    block_shape
        Grid cells per target cluster, as ``(n_lat_cells, n_lon_cells)``.
    """

    def __init__(
        self,
        smooth: float,
        input_map: Dict[str, torch.Tensor],
        grid_coords: Optional[np.ndarray] = None,
        block_shape: Tuple[int, int] = (3, 3),
        n_neighbor_blocks_t: int = 6,
        lag1_same_block: bool = True,
        lag1_local_blocks: int = 3,
        lag1_shifted_blocks: int = 1,
        lag2_same_block: bool = True,
        lag2_local_blocks: int = 2,
        lag2_shifted_blocks: int = 1,
        daily_stride: int = 2,
        lag1_lon_offset: float = 0.063,
        lag2_lon_offset: Optional[float] = None,
        target_chunk_size: int = 128,
        min_target_points: int = 1,
        max_neighbor_search: Optional[int] = None,
    ):
        if smooth not in (0.5, 1.5):
            raise ValueError(f"smooth must be 0.5 or 1.5, got {smooth}")

        dummy_nns = [np.array([], dtype=np.int64)]
        super().__init__(
            smooth=smooth,
            input_map=input_map,
            nns_map=dummy_nns,
            mm_cond_number=0,
            nheads=0,
            limit_A=n_neighbor_blocks_t,
            limit_B=lag1_local_blocks,
            limit_C=lag2_local_blocks,
            daily_stride=daily_stride,
        )

        self.grid_coords = None if grid_coords is None else np.asarray(grid_coords, dtype=np.float64)
        self.block_shape = (int(block_shape[0]), int(block_shape[1]))
        self.n_neighbor_blocks_t = int(n_neighbor_blocks_t)
        self.lag1_same_block = bool(lag1_same_block)
        self.lag1_local_blocks = int(lag1_local_blocks)
        self.lag1_shifted_blocks = int(lag1_shifted_blocks)
        self.lag2_same_block = bool(lag2_same_block)
        self.lag2_local_blocks = int(lag2_local_blocks)
        self.lag2_shifted_blocks = int(lag2_shifted_blocks)
        self.daily_stride = int(daily_stride)
        self.lag1_lon_offset = float(abs(lag1_lon_offset))
        self.lag2_lon_offset = (
            float(abs(lag2_lon_offset))
            if lag2_lon_offset is not None
            else 2.0 * self.lag1_lon_offset
        )
        self.target_chunk_size = int(target_chunk_size)
        self.min_target_points = int(min_target_points)
        self.max_neighbor_search = max_neighbor_search

        self._cluster_batches: List[_ClusterBatch] = []
        self.cluster_points: List[np.ndarray] = []
        self.cluster_centroids: Optional[np.ndarray] = None
        self.cluster_nns: Optional[np.ndarray] = None
        self.shift_lookup_lag1: Optional[np.ndarray] = None
        self.shift_lookup_lag2: Optional[np.ndarray] = None
        self.n_clusters: int = 0
        self.max_points_per_cluster: int = 0
        self.n_target_blocks: int = 0
        self.n_target_points: int = 0

    # ------------------------------------------------------------------
    # Smooth-aware Matérn covariance, matching kernels_vecchia_hybrid.py
    # ------------------------------------------------------------------

    def matern_cov_batched(self, params, x_batch):
        phi1, phi2, phi3, phi4 = torch.exp(params[0:4])
        nugget = torch.exp(params[6])
        dist_params = torch.stack([phi3, phi4, params[4], params[5]])
        d = self.batched_manual_dist(dist_params, x_batch)
        scaled_d = d * phi2

        if self.smooth == 0.5:
            cov = (phi1 / phi2) * torch.exp(-scaled_d)
        else:
            cov = (phi1 / phi2) * (1.0 + scaled_d) * torch.exp(-scaled_d)

        B, N, _ = x_batch.shape
        eye = torch.eye(N, device=self.device, dtype=torch.float64).unsqueeze(0).expand(B, N, N)
        return cov + eye * (nugget + 1e-6)

    def matern_cov_aniso_STABLE_log_reparam(self, params, x, y):
        phi1, phi2, phi3, phi4 = torch.exp(params[0:4])
        nugget = torch.exp(params[6])
        advec_lat = params[4]
        advec_lon = params[5]
        sigmasq = phi1 / phi2

        dist_params = torch.stack([phi3, phi4, advec_lat, advec_lon])
        distance = self.precompute_coords_aniso_STABLE(dist_params, x, y)
        scaled_d = distance * phi2

        if self.smooth == 0.5:
            cov = sigmasq * torch.exp(-scaled_d)
        else:
            cov = sigmasq * (1.0 + scaled_d) * torch.exp(-scaled_d)

        if x.shape[0] == y.shape[0]:
            cov.diagonal().add_(nugget + 1e-8)
        return cov

    # ------------------------------------------------------------------
    # Cluster construction
    # ------------------------------------------------------------------

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
    def _unique_inverse(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rounded = np.round(values.astype(np.float64), 10)
        unique = np.unique(rounded)
        lookup = {v: i for i, v in enumerate(unique)}
        inverse = np.array([lookup[v] for v in rounded], dtype=np.int64)
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

    def _build_clusters(self, n_points: int):
        coords = self._grid_coords_np(n_points)
        _, row_idx = self._unique_inverse(coords[:, 0])
        _, col_idx = self._unique_inverse(coords[:, 1])

        br = self.block_shape[0]
        bc = self.block_shape[1]
        block_keys = list(zip((row_idx // br).tolist(), (col_idx // bc).tolist()))
        raw: Dict[Tuple[int, int], List[int]] = {}
        for local_idx, key in enumerate(block_keys):
            raw.setdefault(key, []).append(local_idx)

        raw_keys = sorted(raw)
        raw_points = [np.array(raw[k], dtype=np.int64) for k in raw_keys]
        raw_centroids = np.vstack([coords[p].mean(axis=0) for p in raw_points])

        order = self._as_zero_based_order(orderings.maxmin_cpp(raw_centroids), len(raw_points))
        self.cluster_points = [raw_points[i] for i in order]
        self.cluster_centroids = raw_centroids[order]
        self.n_clusters = len(self.cluster_points)
        self.max_points_per_cluster = max(len(p) for p in self.cluster_points)

        max_blocks = self.max_neighbor_search
        if max_blocks is None:
            max_blocks = max(
                self.n_neighbor_blocks_t,
                self.lag1_local_blocks,
                self.lag2_local_blocks,
                self.lag1_shifted_blocks,
                self.lag2_shifted_blocks,
                1,
            ) + 8
        self.cluster_nns = orderings.find_nns_l2(self.cluster_centroids, max_nn=int(max_blocks))
        self.shift_lookup_lag1 = self._build_shift_lookup(lon_offset=self.lag1_lon_offset)
        self.shift_lookup_lag2 = self._build_shift_lookup(lon_offset=self.lag2_lon_offset)

    def _build_shift_lookup(self, lon_offset: float) -> np.ndarray:
        if self.cluster_centroids is None:
            raise RuntimeError("Clusters must be built before shift lookup")
        coords = self.cluster_centroids
        tree = BallTree(np.radians(coords), metric="haversine")
        lats = coords[:, 0]
        lons = coords[:, 1]
        lon_min = float(np.nanmin(lons))
        lon_max = float(np.nanmax(lons))
        base_ids = np.arange(coords.shape[0], dtype=np.int64)
        target_lons = lons + float(lon_offset)
        outside = (target_lons < lon_min) | (target_lons > lon_max)
        query = np.column_stack([np.radians(lats), np.radians(target_lons)])
        _, idx = tree.query(query, k=1)
        lookup = idx.flatten().astype(np.int64)
        lookup[outside] = base_ids[outside]
        return lookup

    @staticmethod
    def _append_unique_blocks(out: List[int], candidates: Sequence[int], cap: int):
        for value in candidates:
            if len(out) >= cap:
                break
            value = int(value)
            if value < 0 or value in out:
                continue
            out.append(value)

    def _fresh_block_candidates(self, center_block: int, count: int) -> List[int]:
        if count <= 0:
            return []
        candidates = [int(center_block)]
        if self.cluster_nns is not None and 0 <= center_block < len(self.cluster_nns):
            candidates.extend(int(v) for v in self.cluster_nns[center_block] if int(v) >= 0)
        out: List[int] = []
        for cand in candidates:
            if cand in out:
                continue
            out.append(cand)
        return out

    def _conditioning_blocks(self, block_idx: int, time_idx: int) -> Tuple[str, int, List[Tuple[int, int]]]:
        local_prev = [
            int(v)
            for v in self.cluster_nns[block_idx]
            if int(v) >= 0 and int(v) < block_idx
        ]

        cond: List[Tuple[int, int]] = []
        self._append_unique_block_times(cond, time_idx, local_prev, self.n_neighbor_blocks_t)
        label = "A"
        max_cond_blocks = self.n_neighbor_blocks_t

        if time_idx > 0:
            label = "AB"
            prev_time = time_idx - 1
            if self.lag1_same_block:
                self._append_unique_block_times(cond, prev_time, [block_idx], 1)
            self._append_unique_block_times(cond, prev_time, local_prev, self.lag1_local_blocks)
            center = int(self.shift_lookup_lag1[block_idx])
            fresh = self._fresh_block_candidates(center, self.lag1_shifted_blocks)
            self._append_unique_block_times(cond, prev_time, fresh, self.lag1_shifted_blocks)
            max_cond_blocks += int(self.lag1_same_block) + self.lag1_local_blocks + self.lag1_shifted_blocks

        if time_idx >= self.daily_stride:
            label = "ABC"
            prev2_time = time_idx - self.daily_stride
            if self.lag2_same_block:
                self._append_unique_block_times(cond, prev2_time, [block_idx], 1)
            self._append_unique_block_times(cond, prev2_time, local_prev, self.lag2_local_blocks)
            center = int(self.shift_lookup_lag2[block_idx])
            fresh = self._fresh_block_candidates(center, self.lag2_shifted_blocks)
            self._append_unique_block_times(cond, prev2_time, fresh, self.lag2_shifted_blocks)
            max_cond_blocks += int(self.lag2_same_block) + self.lag2_local_blocks + self.lag2_shifted_blocks

        return label, max_cond_blocks * self.max_points_per_cluster, cond

    @staticmethod
    def _append_unique_block_times(
        out: List[Tuple[int, int]],
        time_idx: int,
        block_candidates: Sequence[int],
        cap: int,
    ):
        added = 0
        seen = set(out)
        for block_idx in block_candidates:
            if added >= cap:
                break
            key = (int(time_idx), int(block_idx))
            if key in seen:
                continue
            out.append(key)
            seen.add(key)
            added += 1

    # ------------------------------------------------------------------
    # Precomputation
    # ------------------------------------------------------------------

    def _precompute_message(self) -> str:
        return (
            "Pre-computing ClusterHybridVecchia "
            f"(smooth={self.smooth}, block={self.block_shape}, "
            f"A={self.n_neighbor_blocks_t}, "
            f"B=same{int(self.lag1_same_block)}+local{self.lag1_local_blocks}+fresh{self.lag1_shifted_blocks}, "
            f"C=same{int(self.lag2_same_block)}+local{self.lag2_local_blocks}+fresh{self.lag2_shifted_blocks}, "
            f"offsets={self.lag1_lon_offset:.4f}/{self.lag2_lon_offset:.4f})..."
        )

    def precompute_conditioning_sets(self):
        print(self._precompute_message(), end=" ")

        all_data_list = [
            torch.from_numpy(d) if isinstance(d, np.ndarray) else d
            for d in self.input_map.values()
        ]
        all_data_list = [d.to(self.device, dtype=torch.float32) for d in all_data_list]
        day_lengths = [int(d.shape[0]) for d in all_data_list]
        if len(set(day_lengths)) != 1:
            raise ValueError(f"ClusterHybridVecchiaFit requires equal grid length per time, got {day_lengths}")

        n_grid = day_lengths[0]
        self._build_clusters(n_grid)

        Real_Data = torch.cat(all_data_list, dim=0).contiguous()
        n_real, num_cols = Real_Data.shape
        is_nan_real = torch.isnan(Real_Data[:, 2])
        is_nan_np = is_nan_real.detach().cpu().numpy()

        valid_lats = Real_Data[~is_nan_real, 0]
        self.lat_mean_val = (
            valid_lats.mean().item() if valid_lats.numel() else Real_Data[:, 0].mean().item()
        )

        max_cond_blocks = (
            self.n_neighbor_blocks_t
            + int(self.lag1_same_block)
            + self.lag1_local_blocks
            + self.lag1_shifted_blocks
            + int(self.lag2_same_block)
            + self.lag2_local_blocks
            + self.lag2_shifted_blocks
        )
        n_dummies = max(1, max_cond_blocks * self.max_points_per_cluster)
        dummy_block = torch.zeros((n_dummies, num_cols), device=self.device, dtype=torch.float32)
        for k in range(n_dummies):
            dummy_block[k, 0] = (k + 1) * 1e8
            dummy_block[k, 1] = (k + 1) * 1e8
            dummy_block[k, 3] = (k + 1) * 1e8
        Full_Data = torch.cat([Real_Data, dummy_block], dim=0).contiguous()
        dummy_start = n_real

        cumulative_len = np.cumsum([0] + day_lengths)
        batch_rows: Dict[Tuple[str, int, int], List[List[int]]] = {}

        self.n_target_blocks = 0
        self.n_target_points = 0

        for time_idx in range(len(all_data_list)):
            offset = int(cumulative_len[time_idx])
            for block_idx, point_locals in enumerate(self.cluster_points):
                target = [
                    offset + int(p)
                    for p in point_locals
                    if not is_nan_np[offset + int(p)]
                ]
                if len(target) < self.min_target_points:
                    continue

                label, max_cond_points, cond_block_refs = self._conditioning_blocks(block_idx, time_idx)
                cond_points: List[int] = []
                seen_points = set()
                for cond_time, cond_block in cond_block_refs:
                    cond_offset = int(cumulative_len[cond_time])
                    for p in self.cluster_points[cond_block]:
                        g = cond_offset + int(p)
                        if g in seen_points or is_nan_np[g]:
                            continue
                        cond_points.append(g)
                        seen_points.add(g)

                if len(cond_points) < max_cond_points:
                    padded = [dummy_start + k for k in range(max_cond_points - len(cond_points))] + cond_points
                else:
                    padded = cond_points[-max_cond_points:]

                row = padded + target
                key = (label, int(max_cond_points), int(len(target)))
                batch_rows.setdefault(key, []).append(row)
                self.n_target_blocks += 1
                self.n_target_points += len(target)

        self._cluster_batches = []

        def build_batch(key: Tuple[str, int, int], rows: List[List[int]]) -> _ClusterBatch:
            label, max_cond_points, target_size = key
            T = torch.tensor(rows, device=self.device, dtype=torch.long)
            G = Full_Data[T]
            X = G[..., [0, 1, 3]].contiguous().to(torch.float64)
            Y = G[..., 2].unsqueeze(-1).contiguous().to(torch.float64)
            ones = torch.ones_like(G[..., 0]).unsqueeze(-1)
            lat = (G[..., 0] - self.lat_mean_val).unsqueeze(-1)
            dums = G[..., 4:11] if G.shape[-1] >= 11 else G.new_zeros((*G.shape[:-1], 7))
            Locs = torch.cat([ones, lat, dums], dim=-1).contiguous().to(torch.float64)
            is_dummy = (T >= dummy_start).unsqueeze(-1)
            Locs = Locs.masked_fill(is_dummy, 0.0)
            Y = Y.masked_fill(is_dummy, 0.0)
            return _ClusterBatch(
                label=label,
                max_cond_points=max_cond_points,
                target_size=target_size,
                X=X,
                Y=Y,
                Locs=Locs,
                T=T,
                is_dummy=is_dummy,
            )

        for key in sorted(batch_rows, key=lambda x: (x[0], x[1], x[2])):
            self._cluster_batches.append(build_batch(key, batch_rows[key]))

        self.Heads_data = torch.empty((0, num_cols), device=self.device, dtype=torch.float64)
        self.n_tails = self.n_target_points
        self._dummy_start_stored = dummy_start
        self._n_real_stored = n_real
        self._n_dummies_stored = n_dummies
        self.is_precomputed = True

        batch_summary = ", ".join(
            f"{b.label}:m{b.max_cond_points}:b{b.target_size}x{b.X.shape[0]}"
            for b in self._cluster_batches[:10]
        )
        more = "" if len(self._cluster_batches) <= 10 else f", ... ({len(self._cluster_batches)} batches)"
        print(
            f"Done. clusters={self.n_clusters}, max_points/block={self.max_points_per_cluster}, "
            f"target_blocks={self.n_target_blocks}, target_points={self.n_target_points}, "
            f"batches=[{batch_summary}{more}]"
        )
        return self

    # ------------------------------------------------------------------
    # Block conditional GLS accumulation
    # ------------------------------------------------------------------

    def _tail_batches(self):
        return self._cluster_batches

    def _accumulate_gls_stats(self, params, include_y_quad=True, catch_cholesky=False):
        self._check_precomputed()

        XT_Sinv_X = torch.zeros((self.n_features, self.n_features), device=self.device, dtype=torch.float64)
        XT_Sinv_y = torch.zeros((self.n_features, 1), device=self.device, dtype=torch.float64)
        yT_Sinv_y = torch.tensor(0.0, device=self.device, dtype=torch.float64)
        log_det = torch.tensor(0.0, device=self.device, dtype=torch.float64)

        chunk_size = max(1, int(self.target_chunk_size))
        for batch in self._cluster_batches:
            if batch.X.shape[0] == 0:
                continue
            target_slice = slice(batch.max_cond_points, batch.max_cond_points + batch.target_size)
            for start in range(0, batch.X.shape[0], chunk_size):
                end = min(start + chunk_size, batch.X.shape[0])
                cov_chunk = self.matern_cov_batched(params, batch.X[start:end])
                try:
                    L_chunk = torch.linalg.cholesky(cov_chunk)
                except torch.linalg.LinAlgError:
                    if catch_cholesky:
                        self._log_cholesky_failure(params, f"Cluster tails {batch.label}")
                        return None
                    raise

                Z_locs = torch.linalg.solve_triangular(
                    L_chunk, batch.Locs[start:end], upper=False
                )
                Z_y = torch.linalg.solve_triangular(
                    L_chunk, batch.Y[start:end], upper=False
                )

                u_X = Z_locs[:, target_slice, :].reshape(-1, self.n_features)
                u_y = Z_y[:, target_slice, :].reshape(-1, 1)
                diag_L = torch.diagonal(L_chunk, dim1=1, dim2=2)[:, target_slice]

                log_det += 2.0 * torch.sum(torch.log(diag_L))
                XT_Sinv_X += u_X.T @ u_X
                XT_Sinv_y += u_X.T @ u_y
                if include_y_quad:
                    yT_Sinv_y += (u_y.T @ u_y).squeeze()

        total_N = int(self.n_target_points)
        return XT_Sinv_X, XT_Sinv_y, yT_Sinv_y, log_det, total_N

    def cluster_summary(self) -> Dict[str, float]:
        if not self.is_precomputed:
            raise RuntimeError("Run precompute_conditioning_sets() first")
        return {
            "n_clusters": self.n_clusters,
            "block_shape_lat": self.block_shape[0],
            "block_shape_lon": self.block_shape[1],
            "max_points_per_cluster": self.max_points_per_cluster,
            "n_target_blocks": self.n_target_blocks,
            "n_target_points": self.n_target_points,
            "n_batches": len(self._cluster_batches),
            "target_chunk_size": self.target_chunk_size,
        }


__all__ = ["ClusterHybridVecchiaFit"]
