"""Calibrated signed-2D corridor-width 4x4 lag-643 Vecchia model.

This is the direction-adapted counterpart of
``vecchia_realdata_corridor_width_4x4_lag643``.  It preserves the preferred
4x4 target blocks, 6/4/3 conditioning-block budgets, width-anchor rule, and
corridor multipliers, but replaces the hard-coded scalar longitude delta
``0.126`` with a calibrated two-dimensional advection vector.

Corridor geometry
-----------------
``reference_advec_lat/lon`` are covariance-model advection parameters.  With
the covariance convention ``d(h - v*tau)``, a current target looks for past
conditioning locations in the opposite direction ``-v``.  Therefore:

* t-1 uses the signed 2-D segment from ``target - 0.5*v`` to
  ``target - 1.5*v``;
* t-2 uses the signed 2-D segment from ``target`` to ``target - 2*v``.

This is intentionally not an exact shifted-center design.  It is the literal
two-dimensional calibrated version of the original longitude-only corridors
``[0.5*delta, 1.5*delta]`` and ``[0, 2*delta]``.

The conditioning graph is constructed once from the calibrated vector.  The
subsequent likelihood still estimates both advection parameters continuously
without rebuilding that graph inside optimizer iterations.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

from GEMS_TCO.vecchia_realdata_corridor_width_4x4_lag643 import (
    BLOCK_SHAPE,
    CORRIDOR_ANCHOR_MODE,
    LAG_COUNTS,
    REFERENCE_ADVEC_LON_ABS,
    RealDataCorridorWidth4x4Lag643VecchiaFit,
    move_input_map,
)


ADAPTED_SPEC_NAME = "adapted_directional_corridor_width_4x4_lag643"
LAG1_CORRIDOR_MULTIPLIERS = (0.5, 1.5)
LAG2_CORRIDOR_MULTIPLIERS = (0.0, 2.0)


def adapted_model_spec(
    reference_advec_lat: float,
    reference_advec_lon: float,
) -> dict[str, Any]:
    """Return explicit metadata for the calibrated signed-2D corridor."""
    advec = np.asarray(
        [float(reference_advec_lat), float(reference_advec_lon)],
        dtype=np.float64,
    )
    past_step = -advec
    lag0, lag1, lag2 = LAG_COUNTS
    return {
        "spec_name": ADAPTED_SPEC_NAME,
        "strategy": "offset_corridor_tapered",
        "conditioning_mode": "calibrated_signed_2d_corridor_width",
        "block_shape": BLOCK_SHAPE,
        "lag_counts": LAG_COUNTS,
        "lag_pattern": f"{lag0}/{lag1}/{lag2}",
        "reference_advec_lat": float(advec[0]),
        "reference_advec_lon": float(advec[1]),
        "reference_advec_norm": float(np.linalg.norm(advec)),
        "past_step_lat": float(past_step[0]),
        "past_step_lon": float(past_step[1]),
        "lag1_corridor_multipliers": LAG1_CORRIDOR_MULTIPLIERS,
        "lag2_corridor_multipliers": LAG2_CORRIDOR_MULTIPLIERS,
        "lag1_past_segment_start": (0.5 * past_step).tolist(),
        "lag1_past_segment_end": (1.5 * past_step).tolist(),
        "lag2_past_segment_start": (0.0 * past_step).tolist(),
        "lag2_past_segment_end": (2.0 * past_step).tolist(),
        "corridor_anchor_mode": CORRIDOR_ANCHOR_MODE,
    }


class AdaptedRealDataCorridorWidth4x4Lag643VecchiaFit(
    RealDataCorridorWidth4x4Lag643VecchiaFit
):
    """Preferred lag-643 corridor calibrated by a signed 2-D advection seed."""

    spec_name = ADAPTED_SPEC_NAME
    lag1_corridor_multipliers = LAG1_CORRIDOR_MULTIPLIERS
    lag2_corridor_multipliers = LAG2_CORRIDOR_MULTIPLIERS

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
        # The parent scalar is retained only for inherited metadata and lookup
        # construction.  _lag_candidates below uses the signed vector corridor.
        super().__init__(
            smooth=smooth,
            input_map=input_map,
            grid_coords=grid_coords,
            reference_advec_lon_abs=max(self.reference_advec_norm, 1e-12),
            daily_stride=daily_stride,
            target_chunk_size=target_chunk_size,
            min_target_points=min_target_points,
            max_neighbor_search=max_neighbor_search,
            block_row_offset=block_row_offset,
            block_col_offset=block_col_offset,
        )
        self.realdata_spec_name = ADAPTED_SPEC_NAME
        self.temporal_basis = "calibrated_signed_2d_corridor"

    @staticmethod
    def _point_to_segment_distance_sq(
        points: np.ndarray,
        start: np.ndarray,
        end: np.ndarray,
    ) -> np.ndarray:
        segment = end - start
        denominator = float(segment @ segment)
        if denominator <= 1e-16:
            return np.sum((points - start) ** 2, axis=1)
        weights = np.clip(((points - start) @ segment) / denominator, 0.0, 1.0)
        closest = start + weights[:, None] * segment
        return np.sum((points - closest) ** 2, axis=1)

    def _cluster_candidates_from_vector_corridor(
        self,
        block_idx: int,
        multiplier_interval: Tuple[float, float],
        count: int,
    ) -> List[int]:
        """Choose lagged blocks along a signed two-dimensional segment."""
        if count <= 0:
            return []
        if self.cluster_centroids is None or self.cluster_all_tree is None:
            return []

        block_idx = int(block_idx)
        target = np.asarray(self.cluster_centroids[block_idx], dtype=np.float64)
        lo, hi = sorted(float(value) for value in multiplier_interval)
        start = target + lo * self.past_offset_vector
        end = target + hi * self.past_offset_vector
        segment_length = float(np.linalg.norm(end - start))

        # A zero calibrated seed has no direction; the target-centered graph is
        # the deterministic and continuous limiting case.
        if segment_length <= 1e-12:
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
                lat_width = float(
                    np.nanmedian(self.cluster_lat_max - self.cluster_lat_min)
                )
                lon_width = float(
                    np.nanmedian(self.cluster_lon_max - self.cluster_lon_min)
                )
                # Generalize the original longitude block-width rule by using
                # the rectangular block footprint projected onto the corridor
                # direction.  For a longitude-only vector this is exactly the
                # parent's corridor_block_lon_width, so replacing delta=0.126
                # with a calibrated vector does not otherwise change the rule.
                direction = (end - start) / segment_length
                projected_block_width = (
                    abs(float(direction[0])) * lat_width
                    + abs(float(direction[1])) * lon_width
                )
                block_span = max(projected_block_width, 1e-12)
                anchor_count = max(
                    1,
                    min(int(count), int(np.ceil(segment_length / block_span))),
                )

        fractions = (
            np.asarray([0.5], dtype=np.float64)
            if anchor_count <= 1
            else np.linspace(0.0, 1.0, anchor_count)
        )
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

        # Match the original width-mode rule: after placing the minimum number
        # of width-covering anchors, fill the remaining budget near the segment
        # midpoint before using the full segment-distance ranking.
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

        distance_sq = self._point_to_segment_distance_sq(
            np.asarray(self.cluster_centroids, dtype=np.float64),
            start,
            end,
        )
        midpoint_distance_sq = np.sum((self.cluster_centroids - midpoint) ** 2, axis=1)
        ranked = np.lexsort((midpoint_distance_sq, distance_sq))
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
            "Pre-computing calibrated signed-2D corridor lag643 "
            f"(smooth={self.smooth}, block={self.block_shape}, "
            f"lag_blocks={LAG_COUNTS[0]}/{LAG_COUNTS[1]}/{LAG_COUNTS[2]}, "
            f"reference_advec=({self.reference_advec_lat:.6f},"
            f"{self.reference_advec_lon:.6f}), past_step="
            f"({self.past_offset_vector[0]:.6f},{self.past_offset_vector[1]:.6f}), "
            f"corridor_multipliers={self.lag1_corridor_multipliers}/"
            f"{self.lag2_corridor_multipliers})..."
        )

    def cluster_summary(self) -> dict[str, Any]:
        out = super().cluster_summary()
        out.update(
            {
                "spec_name": ADAPTED_SPEC_NAME,
                "geometry": "adapted",
                "geometry_definition": (
                    "t-1 calibrated corridor [0.5v,1.5v]; "
                    "t-2 calibrated corridor [0,2v], applied in the past -v direction"
                ),
                "conditioning_mode": "calibrated_signed_2d_corridor_width",
                "reference_advec_lat": self.reference_advec_lat,
                "reference_advec_lon": self.reference_advec_lon,
                "reference_advec_norm": self.reference_advec_norm,
                "past_step_lat": float(self.past_offset_vector[0]),
                "past_step_lon": float(self.past_offset_vector[1]),
                "lag1_corridor_multiplier_lo": self.lag1_corridor_multipliers[0],
                "lag1_corridor_multiplier_hi": self.lag1_corridor_multipliers[1],
                "lag2_corridor_multiplier_lo": self.lag2_corridor_multipliers[0],
                "lag2_corridor_multiplier_hi": self.lag2_corridor_multipliers[1],
            }
        )
        return out


def build_adapted_model(
    smooth: float,
    input_map,
    reference_advec_lat: float,
    reference_advec_lon: float,
    grid_coords=None,
    device: Optional[str] = None,
    dtype=None,
    **kwargs,
) -> AdaptedRealDataCorridorWidth4x4Lag643VecchiaFit:
    """Construct the calibrated signed-2D lag-643 corridor model."""
    mapped = move_input_map(input_map, device=device, dtype=dtype)
    return AdaptedRealDataCorridorWidth4x4Lag643VecchiaFit(
        smooth=smooth,
        input_map=mapped,
        grid_coords=grid_coords,
        reference_advec_lat=reference_advec_lat,
        reference_advec_lon=reference_advec_lon,
        **kwargs,
    )


__all__ = [
    "ADAPTED_SPEC_NAME",
    "BLOCK_SHAPE",
    "LAG_COUNTS",
    "LAG1_CORRIDOR_MULTIPLIERS",
    "LAG2_CORRIDOR_MULTIPLIERS",
    "adapted_model_spec",
    "AdaptedRealDataCorridorWidth4x4Lag643VecchiaFit",
    "build_adapted_model",
]
