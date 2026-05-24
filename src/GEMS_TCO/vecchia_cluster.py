"""
vecchia_cluster.py

Created 2026-05-22.

Strategy-controlled cluster Vecchia models for testing temporal neighbor
geometry.  This module deliberately stays thin: it reuses the block likelihood,
GLS mean estimation, covariance, batching, and optimizer from
``matern_vecchia_cluster_hybrid.ClusterHybridVecchiaFit`` and only replaces the
cluster conditioning-set logic.

The five intended strategies are:

  center_full
      t, t-1, and t-2 all use clusters centered at the target block.  Each time
      layer uses the full requested block budget.

  center_tapered
      Same centers as center_full, but the lagged budgets are smaller.  By
      default t uses 6 clusters, t-1 keeps ceil(0.80 * 6)=5 clusters, and t-2
      keeps ceil(0.50 * 6)=3 clusters.

  offset_full
      t uses target-centered previous same-time clusters.  t-1 uses clusters
      around the lag-1 shifted center, and t-2 uses clusters around the lag-2
      shifted center.  No target-center lag cluster is forced.

  offset_tapered
      Same shifted centers as offset_full, but with the tapered lag budgets.

  offset_tapered_force_center
      Same as offset_tapered, except t-1 and t-2 always include the original
      target-center cluster.  If the target-center cluster is already among the
      offset-centered candidates, one additional offset-centered neighbor is
      taken so the forced-center model does not lose one conditioning block.

  offset_corridor_tapered
      t uses target-centered previous same-time clusters.  t-1 and t-2 do not
      use a single shifted center.  Instead, they choose lagged clusters whose
      block footprints cover a longitude corridor around the target block,
      e.g. 0.5x--1.5x of a reference one-step advection at t-1 and 0x--2x at
      t-2.  This is meant to test robust coverage when the true displacement
      varies day to day.  Two anchor modes are supported: "budget" places one
      anchor per requested lagged block, while "width" places only enough
      anchors to cover the corridor given the 4x4 block width and fills the
      remaining budget near the corridor midpoint.

Important geometry convention:
  Current-time conditioning must use only clusters that precede the target in
  the max-min Vecchia order.  Lagged times are already in the past, so their
  center/offset neighbor clusters are selected from all clusters, not only from
  previous-in-order clusters.  This is the main difference from the older
  cluster hybrid prototype, where lagged local blocks were inherited from the
  same-time previous-neighbor list.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

from GEMS_TCO import orderings
from GEMS_TCO.matern_vecchia_cluster_hybrid import ClusterHybridVecchiaFit


STRATEGIES = {
    "center_full",
    "center_tapered",
    "offset_full",
    "offset_tapered",
    "offset_tapered_force_center",
    "offset_corridor_tapered",
}


class StrategyClusterVecchiaFit(ClusterHybridVecchiaFit):
    """Cluster Vecchia with explicit temporal cluster-neighbor strategies."""

    def __init__(
        self,
        smooth: float,
        input_map,
        grid_coords: Optional[np.ndarray] = None,
        block_shape: Tuple[int, int] = (3, 3),
        strategy: str = "offset_tapered",
        lag0_block_count: int = 6,
        lag1_block_count: Optional[int] = None,
        lag2_block_count: Optional[int] = None,
        lag1_keep_fraction: float = 0.80,
        lag2_keep_fraction: float = 0.50,
        daily_stride: int = 2,
        lag1_lon_offset: float = 0.126,
        lag2_lon_offset: Optional[float] = 0.252,
        lag1_lon_interval: Optional[Tuple[float, float]] = None,
        lag2_lon_interval: Optional[Tuple[float, float]] = None,
        corridor_anchor_mode: str = "budget",
        target_chunk_size: int = 128,
        min_target_points: int = 1,
        max_neighbor_search: Optional[int] = None,
        block_row_offset: int = 0,
        block_col_offset: int = 0,
    ):
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {sorted(STRATEGIES)}, got {strategy!r}")

        self.strategy = str(strategy)
        self.temporal_basis = "offset" if self.strategy.startswith("offset") else "center"
        self.force_target_center = self.strategy == "offset_tapered_force_center"
        self.is_tapered = "tapered" in self.strategy
        self.use_lon_corridor = self.strategy == "offset_corridor_tapered"
        if self.use_lon_corridor:
            self.temporal_basis = "corridor"
        self.corridor_anchor_mode = str(corridor_anchor_mode)
        if self.corridor_anchor_mode not in {"budget", "width"}:
            raise ValueError("corridor_anchor_mode must be 'budget' or 'width'")
        self.lag0_block_count = int(lag0_block_count)
        if self.lag0_block_count <= 0:
            raise ValueError("lag0_block_count must be positive")

        if lag1_block_count is None:
            lag1_block_count = (
                math.ceil(self.lag0_block_count * float(lag1_keep_fraction))
                if self.is_tapered
                else self.lag0_block_count
            )
        if lag2_block_count is None:
            lag2_block_count = (
                math.ceil(self.lag0_block_count * float(lag2_keep_fraction))
                if self.is_tapered
                else self.lag0_block_count
            )
        self.lag1_block_count = max(1, int(lag1_block_count))
        self.lag2_block_count = max(1, int(lag2_block_count))
        self.lag1_keep_fraction = float(lag1_keep_fraction)
        self.lag2_keep_fraction = float(lag2_keep_fraction)
        self.block_row_offset = int(block_row_offset)
        self.block_col_offset = int(block_col_offset)

        lag1_offset_arg = float(abs(lag1_lon_offset))
        lag2_offset_arg = (
            float(abs(lag2_lon_offset))
            if lag2_lon_offset is not None
            else 2.0 * lag1_offset_arg
        )
        self.lag1_lon_interval = self._clean_lon_interval(
            lag1_lon_interval,
            default=(0.5 * lag1_offset_arg, 1.5 * lag1_offset_arg),
        )
        self.lag2_lon_interval = self._clean_lon_interval(
            lag2_lon_interval,
            default=(0.0, 2.0 * lag2_offset_arg),
        )

        # For the force-center strategy, the target-center block is added on
        # top of the offset budget.  With the default 6/5/3 budgets this yields
        # at most 6 clusters at t-1 and 4 clusters at t-2.
        self.lag1_max_blocks = self.lag1_block_count + int(self.force_target_center)
        self.lag2_max_blocks = self.lag2_block_count + int(self.force_target_center)

        parent_max_search = max(
            self.lag0_block_count,
            self.lag1_max_blocks,
            self.lag2_max_blocks,
            1,
        ) + 8
        if max_neighbor_search is None:
            max_neighbor_search = parent_max_search
        self.all_neighbor_search = int(max_neighbor_search)
        self.cluster_all_nns: Optional[np.ndarray] = None
        self.cluster_all_tree: Optional[cKDTree] = None
        self.cluster_lat_min: Optional[np.ndarray] = None
        self.cluster_lat_max: Optional[np.ndarray] = None
        self.cluster_lon_min: Optional[np.ndarray] = None
        self.cluster_lon_max: Optional[np.ndarray] = None
        self.grid_lon_step: float = float("nan")
        self.corridor_block_lon_width: float = float("nan")

        super().__init__(
            smooth=smooth,
            input_map=input_map,
            grid_coords=grid_coords,
            block_shape=block_shape,
            n_neighbor_blocks_t=self.lag0_block_count,
            lag1_same_block=False,
            lag1_local_blocks=self.lag1_max_blocks,
            lag1_shifted_blocks=0,
            lag2_same_block=False,
            lag2_local_blocks=self.lag2_max_blocks,
            lag2_shifted_blocks=0,
            daily_stride=daily_stride,
            lag1_lon_offset=lag1_lon_offset,
            lag2_lon_offset=lag2_lon_offset,
            target_chunk_size=target_chunk_size,
            min_target_points=min_target_points,
            max_neighbor_search=max_neighbor_search,
        )

    @staticmethod
    def _clean_lon_interval(
        interval: Optional[Tuple[float, float]],
        default: Tuple[float, float],
    ) -> Tuple[float, float]:
        if interval is None:
            lo, hi = default
        else:
            lo, hi = interval
        lo = float(abs(lo))
        hi = float(abs(hi))
        return (min(lo, hi), max(lo, hi))

    def _build_clusters(self, n_points: int):
        coords = self._grid_coords_np(n_points)
        _, row_idx = self._unique_inverse(coords[:, 0])
        unique_lons, col_idx = self._unique_inverse(coords[:, 1])
        if len(unique_lons) > 1:
            self.grid_lon_step = float(np.median(np.diff(unique_lons)))
            self.corridor_block_lon_width = float(self.block_shape[1] * self.grid_lon_step)
        else:
            self.grid_lon_step = float("nan")
            self.corridor_block_lon_width = float("nan")

        # The target clusters are non-overlapping grid blocks.  For odd shapes
        # like 3x3 this feels naturally centered.  For even shapes like 4x4
        # there is no unique center cell, so block_row_offset/block_col_offset
        # deliberately shift the partition origin.  A positive column offset
        # creates longitude-right-shifted 4x4 blocks after a small left-edge
        # remainder block, which lets us test whether the even-block anchoring
        # affects the cluster Vecchia fit.
        br = self.block_shape[0]
        bc = self.block_shape[1]
        block_rows = np.floor_divide(row_idx - self.block_row_offset, br)
        block_cols = np.floor_divide(col_idx - self.block_col_offset, bc)
        block_keys = list(zip(block_rows.tolist(), block_cols.tolist()))
        raw = {}
        for local_idx, key in enumerate(block_keys):
            raw.setdefault(key, []).append(local_idx)

        raw_keys = sorted(raw)
        raw_points = [np.array(raw[k], dtype=np.int64) for k in raw_keys]
        raw_centroids = np.vstack([coords[p].mean(axis=0) for p in raw_points])
        raw_lat_min = np.array([coords[p, 0].min() for p in raw_points], dtype=np.float64)
        raw_lat_max = np.array([coords[p, 0].max() for p in raw_points], dtype=np.float64)
        raw_lon_min = np.array([coords[p, 1].min() for p in raw_points], dtype=np.float64)
        raw_lon_max = np.array([coords[p, 1].max() for p in raw_points], dtype=np.float64)

        order = self._as_zero_based_order(orderings.maxmin_cpp(raw_centroids), len(raw_points))
        self.cluster_points = [raw_points[i] for i in order]
        self.cluster_centroids = raw_centroids[order]
        self.cluster_lat_min = raw_lat_min[order]
        self.cluster_lat_max = raw_lat_max[order]
        self.cluster_lon_min = raw_lon_min[order]
        self.cluster_lon_max = raw_lon_max[order]
        self.n_clusters = len(self.cluster_points)
        self.max_points_per_cluster = max(len(p) for p in self.cluster_points)

        max_blocks = self.max_neighbor_search
        if max_blocks is None:
            max_blocks = max(
                self.lag0_block_count,
                self.lag1_max_blocks,
                self.lag2_max_blocks,
                1,
            ) + 8
        self.cluster_nns = orderings.find_nns_l2(self.cluster_centroids, max_nn=int(max_blocks))
        self.shift_lookup_lag1 = self._build_shift_lookup(lon_offset=self.lag1_lon_offset)
        self.shift_lookup_lag2 = self._build_shift_lookup(lon_offset=self.lag2_lon_offset)

        if self.cluster_centroids is None:
            raise RuntimeError("Cluster centroids were not built")
        n = int(self.cluster_centroids.shape[0])
        k = min(n, max(1, self.all_neighbor_search + 1))
        tree = cKDTree(np.asarray(self.cluster_centroids, dtype=np.float64))
        self.cluster_all_tree = tree
        _, idx = tree.query(self.cluster_centroids, k=k)
        idx = np.asarray(idx, dtype=np.int64)
        if idx.ndim == 1:
            idx = idx[:, None]

        rows: List[np.ndarray] = []
        for i in range(n):
            row = [int(v) for v in idx[i] if int(v) != i and int(v) >= 0]
            rows.append(np.asarray(row, dtype=np.int64))
        max_len = max((len(r) for r in rows), default=0)
        arr = -np.ones((n, max_len), dtype=np.int64)
        for i, row in enumerate(rows):
            arr[i, : len(row)] = row
        self.cluster_all_nns = arr

    @staticmethod
    def _append_unique_int(out: List[int], values: Sequence[int]) -> None:
        seen = set(out)
        for value in values:
            value = int(value)
            if value < 0 or value in seen:
                continue
            out.append(value)
            seen.add(value)

    def _cluster_candidates_from_center(self, center_block: int, count: int) -> List[int]:
        """Return center block plus nearest all-order neighbors."""
        if count <= 0:
            return []
        center_block = int(center_block)
        out: List[int] = []
        self._append_unique_int(out, [center_block])
        if self.cluster_all_nns is not None and 0 <= center_block < len(self.cluster_all_nns):
            self._append_unique_int(out, self.cluster_all_nns[center_block])
        return out[: int(count)]

    def _cluster_candidates_from_lon_corridor(
        self,
        block_idx: int,
        interval: Tuple[float, float],
        count: int,
    ) -> List[int]:
        """Return lagged clusters that cover a longitude corridor.

        The current-time A set still obeys max-min ordering.  This routine is
        only for lagged times, where every cluster is already in the past.  It
        first snaps evenly spaced anchors in the requested corridor to cluster
        centroids, then fills any remaining budget with clusters whose regular
        grid block footprints intersect the corridor.
        """
        if count <= 0:
            return []
        if (
            self.cluster_centroids is None
            or self.cluster_all_tree is None
            or self.cluster_lat_min is None
            or self.cluster_lat_max is None
            or self.cluster_lon_min is None
            or self.cluster_lon_max is None
        ):
            return []

        block_idx = int(block_idx)
        target_lat = float(self.cluster_centroids[block_idx, 0])
        target_lon = float(self.cluster_centroids[block_idx, 1])
        lo, hi = self._clean_lon_interval(interval, default=interval)
        corridor_lo = target_lon + lo
        corridor_hi = target_lon + hi
        corridor_mid = 0.5 * (corridor_lo + corridor_hi)
        corridor_len = max(0.0, hi - lo)

        lon_min = float(np.nanmin(self.cluster_lon_min))
        lon_max = float(np.nanmax(self.cluster_lon_max))
        n = int(self.cluster_centroids.shape[0])
        k_anchor = min(n, max(1, self.all_neighbor_search + 1))
        anchor_count = int(count)
        if self.corridor_anchor_mode == "width":
            block_width = float(self.corridor_block_lon_width)
            if np.isfinite(block_width) and block_width > 0:
                anchor_count = max(1, min(int(count), int(math.ceil(corridor_len / block_width))))
        if anchor_count <= 1:
            anchors = np.asarray([corridor_mid], dtype=np.float64)
        else:
            anchors = np.linspace(corridor_lo, corridor_hi, num=anchor_count)
        out: List[int] = []
        anchor_rows: List[np.ndarray] = []

        for anchor_lon in anchors:
            query = np.array([target_lat, np.clip(anchor_lon, lon_min, lon_max)], dtype=np.float64)
            _, idx = self.cluster_all_tree.query(query, k=k_anchor)
            idx = np.asarray(idx, dtype=np.int64).reshape(-1)
            anchor_rows.append(idx)
            self._append_unique_int(out, idx[:1])
            if len(out) >= int(count):
                return out[: int(count)]

        if self.corridor_anchor_mode == "width":
            query = np.array([target_lat, np.clip(corridor_mid, lon_min, lon_max)], dtype=np.float64)
            _, idx = self.cluster_all_tree.query(query, k=k_anchor)
            idx = np.asarray(idx, dtype=np.int64).reshape(-1)
            self._append_unique_int(out, idx)
            if len(out) >= int(count):
                return out[: int(count)]

        # If several anchors snap to the same block, walk outward around each
        # anchor before falling back to the full footprint ranking.
        for depth in range(1, k_anchor):
            for idx in anchor_rows:
                self._append_unique_int(out, idx[depth : depth + 1])
                if len(out) >= int(count):
                    return out[: int(count)]

        lon_gap = np.maximum.reduce(
            [
                corridor_lo - self.cluster_lon_max,
                self.cluster_lon_min - corridor_hi,
                np.zeros_like(self.cluster_lon_min),
            ]
        )
        lat_gap = np.maximum.reduce(
            [
                self.cluster_lat_min - target_lat,
                target_lat - self.cluster_lat_max,
                np.zeros_like(self.cluster_lat_min),
            ]
        )
        mid_dist = np.abs(self.cluster_centroids[:, 1] - corridor_mid)
        centroid_dist = np.sqrt((self.cluster_centroids[:, 0] - target_lat) ** 2 + mid_dist**2)
        ranked = np.lexsort((centroid_dist, mid_dist, lat_gap, lon_gap))
        self._append_unique_int(out, ranked)
        return out[: int(count)]

    def _lag_center_block(self, block_idx: int, lag: int) -> int:
        if self.temporal_basis == "center":
            return int(block_idx)
        lookup = self.shift_lookup_lag1 if lag == 1 else self.shift_lookup_lag2
        if lookup is None:
            return int(block_idx)
        if block_idx >= len(lookup):
            return int(block_idx)
        return int(lookup[block_idx])

    def _lag_candidates(self, block_idx: int, lag: int) -> List[int]:
        base_count = self.lag1_block_count if lag == 1 else self.lag2_block_count
        if self.use_lon_corridor:
            interval = self.lag1_lon_interval if lag == 1 else self.lag2_lon_interval
            return self._cluster_candidates_from_lon_corridor(block_idx, interval, base_count)

        center = self._lag_center_block(block_idx, lag)
        if not self.force_target_center or self.temporal_basis == "center":
            return self._cluster_candidates_from_center(center, base_count)

        base = self._cluster_candidates_from_center(center, base_count)
        target = int(block_idx)
        if target in base:
            # Target-center is already represented; take one extra
            # offset-centered neighbor to keep the total conditioning budget.
            return self._cluster_candidates_from_center(center, base_count + 1)
        return [target] + base

    def _conditioning_blocks(self, block_idx: int, time_idx: int):
        if self.cluster_nns is None:
            raise RuntimeError("Cluster nearest-neighbor lists were not built")
        local_prev = [
            int(v)
            for v in self.cluster_nns[block_idx]
            if int(v) >= 0 and int(v) < block_idx
        ]

        cond: List[Tuple[int, int]] = []
        self._append_unique_block_times(cond, time_idx, local_prev, self.lag0_block_count)
        label = "A"
        max_cond_blocks = self.lag0_block_count

        if time_idx > 0:
            label = "AB"
            prev_time = time_idx - 1
            lag1 = self._lag_candidates(block_idx, lag=1)
            self._append_unique_block_times(cond, prev_time, lag1, self.lag1_max_blocks)
            max_cond_blocks += self.lag1_max_blocks

        if time_idx >= self.daily_stride:
            label = "ABC"
            prev2_time = time_idx - self.daily_stride
            lag2 = self._lag_candidates(block_idx, lag=2)
            self._append_unique_block_times(cond, prev2_time, lag2, self.lag2_max_blocks)
            max_cond_blocks += self.lag2_max_blocks

        return label, max_cond_blocks * self.max_points_per_cluster, cond

    def _precompute_message(self) -> str:
        return (
            "Pre-computing StrategyClusterVecchia "
            f"(smooth={self.smooth}, strategy={self.strategy}, block={self.block_shape}, "
            f"origin={self.block_row_offset}/{self.block_col_offset}, "
            f"lag_blocks={self.lag0_block_count}/{self.lag1_max_blocks}/{self.lag2_max_blocks}, "
            f"basis={self.temporal_basis}, force_center={int(self.force_target_center)}, "
            f"offsets={self.lag1_lon_offset:.4f}/{self.lag2_lon_offset:.4f}, "
            f"corridors={self.lag1_lon_interval}/{self.lag2_lon_interval}, "
            f"anchor_mode={self.corridor_anchor_mode})..."
        )

    def cluster_summary(self):
        out = super().cluster_summary()
        out.update(
            {
                "strategy": self.strategy,
                "temporal_basis": self.temporal_basis,
                "force_target_center": int(self.force_target_center),
                "lag0_block_count": self.lag0_block_count,
                "lag1_block_count": self.lag1_block_count,
                "lag2_block_count": self.lag2_block_count,
                "lag1_max_blocks": self.lag1_max_blocks,
                "lag2_max_blocks": self.lag2_max_blocks,
                "lag1_lon_offset": self.lag1_lon_offset,
                "lag2_lon_offset": self.lag2_lon_offset,
                "lag1_lon_interval_lo": self.lag1_lon_interval[0],
                "lag1_lon_interval_hi": self.lag1_lon_interval[1],
                "lag2_lon_interval_lo": self.lag2_lon_interval[0],
                "lag2_lon_interval_hi": self.lag2_lon_interval[1],
                "corridor_anchor_mode": self.corridor_anchor_mode,
                "grid_lon_step": self.grid_lon_step,
                "corridor_block_lon_width": self.corridor_block_lon_width,
                "block_row_offset": self.block_row_offset,
                "block_col_offset": self.block_col_offset,
            }
        )
        return out


__all__ = ["STRATEGIES", "StrategyClusterVecchiaFit"]
