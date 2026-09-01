"""
vecchia_realdata_corridor_width_4x4_lag432.py

Created 2026-05-24.

Lightweight real-data ST cluster Vecchia wrapper for quick CPU/GPU testing.
This uses the same corridor-width logic and delta=0.126 default as the
preferred 643 module, but reduces the conditioning budget to lag pattern 4/3/2:

  - t:   4 previous same-time clusters in max-min order.
  - t-1: 3 corridor-width lagged clusters.
  - t-2: 2 corridor-width lagged clusters.

Use this when a fast sanity check is more important than squeezing out the last
bit of accuracy.  For final real-data fits, the 643 module is the safer default.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

from GEMS_TCO.vecchia_realdata_corridor_width_4x4_lag643 import (
    BLOCK_SHAPE,
    CORRIDOR_ANCHOR_MODE,
    REFERENCE_ADVEC_LON_ABS,
    STRATEGY,
    corridor_intervals,
    move_input_map,
)
from GEMS_TCO.vecchia_cluster import StrategyClusterVecchiaFit


LAG_COUNTS = (4, 3, 2)
SPEC_NAME = "corridor_width_4x4_lag432_delta0p126"
DIRECTIONAL_SPEC_NAME = "directional_corridor_width_4x4_lag432"


def model_spec(reference_advec_lon_abs: float = REFERENCE_ADVEC_LON_ABS) -> dict[str, Any]:
    """Small metadata dict for logs and fit summary rows."""
    lag1_interval, lag2_interval = corridor_intervals(reference_advec_lon_abs)
    lag0, lag1, lag2 = LAG_COUNTS
    return {
        "spec_name": SPEC_NAME,
        "strategy": STRATEGY,
        "conditioning_mode": "corridor_width_light",
        "block_shape": BLOCK_SHAPE,
        "lag_counts": LAG_COUNTS,
        "lag_pattern": f"{lag0}/{lag1}/{lag2}",
        "reference_advec_lon_abs": float(abs(reference_advec_lon_abs)),
        "lag1_lon_offset": float(abs(reference_advec_lon_abs)),
        "lag2_lon_offset": 2.0 * float(abs(reference_advec_lon_abs)),
        "lag1_lon_interval": lag1_interval,
        "lag2_lon_interval": lag2_interval,
        "corridor_anchor_mode": CORRIDOR_ANCHOR_MODE,
    }


def directional_model_spec(
    reference_advec_lat: float,
    reference_advec_lon: float,
) -> dict[str, Any]:
    """Metadata for the two-dimensional, direction-adaptive corridor.

    The covariance uses ``h - v * tau``.  A current target therefore looks for
    its most correlated past conditioning points near ``-v * tau``.  The
    stored ``past_offset_*`` fields make that sign convention explicit.
    """
    advec = np.asarray([reference_advec_lat, reference_advec_lon], dtype=np.float64)
    past_offset = -advec
    lag0, lag1, lag2 = LAG_COUNTS
    return {
        "spec_name": DIRECTIONAL_SPEC_NAME,
        "strategy": STRATEGY,
        "conditioning_mode": "directional_corridor_width_light",
        "block_shape": BLOCK_SHAPE,
        "lag_counts": LAG_COUNTS,
        "lag_pattern": f"{lag0}/{lag1}/{lag2}",
        "reference_advec_lat": float(advec[0]),
        "reference_advec_lon": float(advec[1]),
        "reference_advec_norm": float(np.linalg.norm(advec)),
        "past_offset_lat": float(past_offset[0]),
        "past_offset_lon": float(past_offset[1]),
        "lag1_corridor_multipliers": (0.5, 1.5),
        "lag2_corridor_multipliers": (0.0, 2.0),
        "corridor_anchor_mode": CORRIDOR_ANCHOR_MODE,
    }


class RealDataCorridorWidth4x4Lag432VecchiaFit(StrategyClusterVecchiaFit):
    """Fixed lightweight corridor-width 4x4 lag-432 cluster Vecchia model."""

    spec_name = SPEC_NAME
    block_shape_fixed = BLOCK_SHAPE
    lag_counts_fixed = LAG_COUNTS
    reference_advec_lon_abs_default = REFERENCE_ADVEC_LON_ABS

    def __init__(
        self,
        smooth: float,
        input_map,
        grid_coords=None,
        reference_advec_lon_abs: float = REFERENCE_ADVEC_LON_ABS,
        lag1_lon_interval: Optional[Tuple[float, float]] = None,
        lag2_lon_interval: Optional[Tuple[float, float]] = None,
        daily_stride: int = 2,
        target_chunk_size: int = 128,
        min_target_points: int = 1,
        max_neighbor_search: Optional[int] = None,
        block_row_offset: int = 0,
        block_col_offset: int = 0,
    ):
        delta = float(abs(reference_advec_lon_abs))
        default_lag1_interval, default_lag2_interval = corridor_intervals(delta)
        super().__init__(
            smooth=smooth,
            input_map=input_map,
            grid_coords=grid_coords,
            block_shape=BLOCK_SHAPE,
            strategy=STRATEGY,
            lag0_block_count=LAG_COUNTS[0],
            lag1_block_count=LAG_COUNTS[1],
            lag2_block_count=LAG_COUNTS[2],
            daily_stride=daily_stride,
            lag1_lon_offset=delta,
            lag2_lon_offset=2.0 * delta,
            lag1_lon_interval=lag1_lon_interval or default_lag1_interval,
            lag2_lon_interval=lag2_lon_interval or default_lag2_interval,
            corridor_anchor_mode=CORRIDOR_ANCHOR_MODE,
            target_chunk_size=target_chunk_size,
            min_target_points=min_target_points,
            max_neighbor_search=max_neighbor_search,
            block_row_offset=block_row_offset,
            block_col_offset=block_col_offset,
        )
        self.realdata_spec_name = SPEC_NAME
        self.reference_advec_lon_abs = delta


class DirectionalRealDataCorridorWidth4x4Lag432VecchiaFit(
    RealDataCorridorWidth4x4Lag432VecchiaFit
):
    """Lag-432 corridor whose conditioning geometry follows a 2-D wind seed.

    ``reference_advec_lat/lon`` are model advection parameters, not past-point
    offsets.  For a current target, lagged conditioning clusters are placed on
    the opposite vector ``(-reference_advec_lat, -reference_advec_lon)``.  The
    final likelihood still estimates both advection parameters continuously;
    the seed fixes only the once-per-fit Vecchia conditioning geometry.
    """

    spec_name = DIRECTIONAL_SPEC_NAME
    lag1_corridor_multipliers = (0.5, 1.5)
    lag2_corridor_multipliers = (0.0, 2.0)

    def __init__(
        self,
        smooth: float,
        input_map,
        grid_coords=None,
        reference_advec_lat: float = 0.0,
        reference_advec_lon: float = -REFERENCE_ADVEC_LON_ABS,
        daily_stride: int = 2,
        target_chunk_size: int = 128,
        min_target_points: int = 1,
        max_neighbor_search: Optional[int] = None,
        block_row_offset: int = 0,
        block_col_offset: int = 0,
    ):
        self.reference_advec_lat = float(reference_advec_lat)
        self.reference_advec_lon = float(reference_advec_lon)
        self.reference_advec_norm = float(
            np.hypot(self.reference_advec_lat, self.reference_advec_lon)
        )
        self.past_offset_vector = np.asarray(
            [-self.reference_advec_lat, -self.reference_advec_lon],
            dtype=np.float64,
        )
        super().__init__(
            smooth=smooth,
            input_map=input_map,
            grid_coords=grid_coords,
            # The inherited scalar is retained for API/diagnostic compatibility;
            # _lag_candidates below uses the signed 2-D vector instead.
            reference_advec_lon_abs=self.reference_advec_norm,
            daily_stride=daily_stride,
            target_chunk_size=target_chunk_size,
            min_target_points=min_target_points,
            max_neighbor_search=max_neighbor_search,
            block_row_offset=block_row_offset,
            block_col_offset=block_col_offset,
        )
        self.realdata_spec_name = DIRECTIONAL_SPEC_NAME

    @staticmethod
    def _point_to_segment_distance_sq(
        points: np.ndarray,
        start: np.ndarray,
        end: np.ndarray,
    ) -> np.ndarray:
        segment = end - start
        denom = float(segment @ segment)
        if denom <= 1e-16:
            return np.sum((points - start) ** 2, axis=1)
        weights = np.clip(((points - start) @ segment) / denom, 0.0, 1.0)
        closest = start + weights[:, None] * segment
        return np.sum((points - closest) ** 2, axis=1)

    def _cluster_candidates_from_vector_corridor(
        self,
        block_idx: int,
        multiplier_interval: Tuple[float, float],
        count: int,
    ) -> List[int]:
        if count <= 0:
            return []
        if self.cluster_centroids is None or self.cluster_all_tree is None:
            return []

        block_idx = int(block_idx)
        target = np.asarray(self.cluster_centroids[block_idx], dtype=np.float64)
        lo, hi = sorted(float(x) for x in multiplier_interval)
        start = target + lo * self.past_offset_vector
        end = target + hi * self.past_offset_vector
        segment_len = float(np.linalg.norm(end - start))

        # A zero/near-zero seed has no identifiable direction.  Centered past
        # blocks are the stable fallback and make the behavior deterministic.
        if segment_len <= 1e-12:
            return self._cluster_candidates_from_center(block_idx, count)

        n_clusters = int(self.cluster_centroids.shape[0])
        k_anchor = min(n_clusters, max(1, self.all_neighbor_search + 1))
        anchor_count = int(count)
        if self.corridor_anchor_mode == "width":
            if (
                self.cluster_lat_min is not None
                and self.cluster_lat_max is not None
                and self.cluster_lon_min is not None
                and self.cluster_lon_max is not None
            ):
                lat_width = float(np.nanmedian(self.cluster_lat_max - self.cluster_lat_min))
                lon_width = float(np.nanmedian(self.cluster_lon_max - self.cluster_lon_min))
                block_span = max(float(np.hypot(lat_width, lon_width)), 1e-12)
                anchor_count = max(
                    1,
                    min(int(count), int(np.ceil(segment_len / block_span))),
                )

        fractions = np.asarray([0.5]) if anchor_count <= 1 else np.linspace(0.0, 1.0, anchor_count)
        anchors = start[None, :] + fractions[:, None] * (end - start)[None, :]
        out: List[int] = []
        anchor_rows: List[np.ndarray] = []
        for anchor in anchors:
            _, idx = self.cluster_all_tree.query(anchor, k=k_anchor)
            idx = np.asarray(idx, dtype=np.int64).reshape(-1)
            anchor_rows.append(idx)
            self._append_unique_int(out, idx[:1])
            if len(out) >= int(count):
                return out[: int(count)]

        # Fill close to the corridor midpoint first, matching the original
        # width-anchor behavior, then rank every cluster by distance to the
        # entire line segment so diagonal corridors remain well covered.
        midpoint = 0.5 * (start + end)
        _, midpoint_idx = self.cluster_all_tree.query(midpoint, k=k_anchor)
        self._append_unique_int(out, np.asarray(midpoint_idx).reshape(-1))
        if len(out) >= int(count):
            return out[: int(count)]

        for depth in range(1, k_anchor):
            for idx in anchor_rows:
                self._append_unique_int(out, idx[depth : depth + 1])
                if len(out) >= int(count):
                    return out[: int(count)]

        dist_sq = self._point_to_segment_distance_sq(
            np.asarray(self.cluster_centroids, dtype=np.float64),
            start,
            end,
        )
        midpoint_dist_sq = np.sum((self.cluster_centroids - midpoint) ** 2, axis=1)
        ranked = np.lexsort((midpoint_dist_sq, dist_sq))
        self._append_unique_int(out, ranked)
        return out[: int(count)]

    def _lag_candidates(self, block_idx: int, lag: int) -> List[int]:
        count = self.lag1_block_count if lag == 1 else self.lag2_block_count
        multipliers = (
            self.lag1_corridor_multipliers
            if lag == 1
            else self.lag2_corridor_multipliers
        )
        return self._cluster_candidates_from_vector_corridor(
            block_idx,
            multipliers,
            count,
        )

    def _precompute_message(self) -> str:
        return (
            super()._precompute_message().rstrip(".")
            + f", reference_advec=({self.reference_advec_lat:.4f},"
            f"{self.reference_advec_lon:.4f}), past_offset="
            f"({self.past_offset_vector[0]:.4f},{self.past_offset_vector[1]:.4f}))..."
        )

    def cluster_summary(self):
        out = super().cluster_summary()
        out.update(
            {
                "spec_name": DIRECTIONAL_SPEC_NAME,
                "reference_advec_lat": self.reference_advec_lat,
                "reference_advec_lon": self.reference_advec_lon,
                "reference_advec_norm": self.reference_advec_norm,
                "past_offset_lat": float(self.past_offset_vector[0]),
                "past_offset_lon": float(self.past_offset_vector[1]),
                "lag1_corridor_multiplier_lo": self.lag1_corridor_multipliers[0],
                "lag1_corridor_multiplier_hi": self.lag1_corridor_multipliers[1],
                "lag2_corridor_multiplier_lo": self.lag2_corridor_multipliers[0],
                "lag2_corridor_multiplier_hi": self.lag2_corridor_multipliers[1],
            }
        )
        return out


def build_model(
    smooth: float,
    input_map,
    grid_coords=None,
    device: Optional[str] = None,
    dtype=None,
    **kwargs,
) -> RealDataCorridorWidth4x4Lag432VecchiaFit:
    """Construct the fixed lightweight 432 model, optionally moving tensors."""
    mapped = move_input_map(input_map, device=device, dtype=dtype)
    return RealDataCorridorWidth4x4Lag432VecchiaFit(
        smooth=smooth,
        input_map=mapped,
        grid_coords=grid_coords,
        **kwargs,
    )


def build_directional_model(
    smooth: float,
    input_map,
    reference_advec_lat: float,
    reference_advec_lon: float,
    grid_coords=None,
    device: Optional[str] = None,
    dtype=None,
    **kwargs,
) -> DirectionalRealDataCorridorWidth4x4Lag432VecchiaFit:
    """Construct a 2-D direction-adaptive lag-432 corridor model."""
    mapped = move_input_map(input_map, device=device, dtype=dtype)
    return DirectionalRealDataCorridorWidth4x4Lag432VecchiaFit(
        smooth=smooth,
        input_map=mapped,
        grid_coords=grid_coords,
        reference_advec_lat=reference_advec_lat,
        reference_advec_lon=reference_advec_lon,
        **kwargs,
    )


__all__ = [
    "REFERENCE_ADVEC_LON_ABS",
    "BLOCK_SHAPE",
    "LAG_COUNTS",
    "STRATEGY",
    "CORRIDOR_ANCHOR_MODE",
    "SPEC_NAME",
    "DIRECTIONAL_SPEC_NAME",
    "corridor_intervals",
    "model_spec",
    "directional_model_spec",
    "move_input_map",
    "RealDataCorridorWidth4x4Lag432VecchiaFit",
    "DirectionalRealDataCorridorWidth4x4Lag432VecchiaFit",
    "build_model",
    "build_directional_model",
]
