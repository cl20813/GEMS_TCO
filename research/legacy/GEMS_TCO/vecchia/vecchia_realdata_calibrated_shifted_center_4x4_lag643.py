"""Calibrated signed-2D shifted-center 4x4 lag-643 Vecchia model.

This module is the point-center counterpart of
``vecchia_realdata_adapted_corridor_width_4x4_lag643``.  It uses the same 4x4
target blocks and 6/4/3 conditioning budgets, but replaces each temporal
corridor by one calibrated center:

* t-1: four nearest blocks around ``target - v``;
* t-2: three nearest blocks around ``target - 2*v``.

``v`` is the signed two-dimensional advection seed in the covariance
parameterization ``d(h - v*tau)``.  Consequently, conditioning locations in
the past lie in the ``-v`` direction.  The graph is fixed at construction;
the likelihood can still estimate both advection parameters continuously.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from GEMS_TCO.vecchia_cluster import StrategyClusterVecchiaFit
from GEMS_TCO.vecchia_realdata_corridor_width_4x4_lag643 import (
    BLOCK_SHAPE,
    LAG_COUNTS,
    REFERENCE_ADVEC_LON_ABS,
    move_input_map,
)


SHIFTED_SPEC_NAME = "calibrated_shifted_center_4x4_lag643"
LAG1_CENTER_MULTIPLIER = 1.0
LAG2_CENTER_MULTIPLIER = 2.0


def shifted_model_spec(
    reference_advec_lat: float,
    reference_advec_lon: float,
) -> dict[str, Any]:
    """Return explicit metadata for the calibrated shifted centers."""
    advec = np.asarray(
        [float(reference_advec_lat), float(reference_advec_lon)],
        dtype=np.float64,
    )
    past_step = -advec
    lag0, lag1, lag2 = LAG_COUNTS
    return {
        "spec_name": SHIFTED_SPEC_NAME,
        "strategy": "offset_tapered",
        "conditioning_mode": "calibrated_signed_2d_shifted_center",
        "block_shape": BLOCK_SHAPE,
        "lag_counts": LAG_COUNTS,
        "lag_pattern": f"{lag0}/{lag1}/{lag2}",
        "reference_advec_lat": float(advec[0]),
        "reference_advec_lon": float(advec[1]),
        "reference_advec_norm": float(np.linalg.norm(advec)),
        "past_step_lat": float(past_step[0]),
        "past_step_lon": float(past_step[1]),
        "lag1_center_multiplier": LAG1_CENTER_MULTIPLIER,
        "lag2_center_multiplier": LAG2_CENTER_MULTIPLIER,
        "lag1_past_center_offset": (LAG1_CENTER_MULTIPLIER * past_step).tolist(),
        "lag2_past_center_offset": (LAG2_CENTER_MULTIPLIER * past_step).tolist(),
    }


class CalibratedShiftedCenter4x4Lag643VecchiaFit(StrategyClusterVecchiaFit):
    """Lag-643 graph centered at calibrated one- and two-step shifts."""

    spec_name = SHIFTED_SPEC_NAME
    lag1_center_multiplier = LAG1_CENTER_MULTIPLIER
    lag2_center_multiplier = LAG2_CENTER_MULTIPLIER

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
        scalar_norm = max(self.reference_advec_norm, 1e-12)
        super().__init__(
            smooth=smooth,
            input_map=input_map,
            grid_coords=grid_coords,
            block_shape=BLOCK_SHAPE,
            strategy="offset_tapered",
            lag0_block_count=LAG_COUNTS[0],
            lag1_block_count=LAG_COUNTS[1],
            lag2_block_count=LAG_COUNTS[2],
            daily_stride=daily_stride,
            lag1_lon_offset=scalar_norm,
            lag2_lon_offset=2.0 * scalar_norm,
            target_chunk_size=target_chunk_size,
            min_target_points=min_target_points,
            max_neighbor_search=max_neighbor_search,
            block_row_offset=block_row_offset,
            block_col_offset=block_col_offset,
        )
        self.realdata_spec_name = SHIFTED_SPEC_NAME
        self.temporal_basis = "calibrated_signed_2d_shifted_center"

    def _cluster_candidates_from_shifted_center(
        self,
        block_idx: int,
        multiplier: float,
        count: int,
    ) -> List[int]:
        """Return the nearest blocks to one signed two-dimensional center."""
        if count <= 0:
            return []
        if self.cluster_centroids is None or self.cluster_all_tree is None:
            return []
        target = np.asarray(self.cluster_centroids[int(block_idx)], dtype=np.float64)
        center = target + float(multiplier) * self.past_offset_vector
        n_clusters = int(self.cluster_centroids.shape[0])
        k = min(n_clusters, max(1, int(count)))
        _, indices = self.cluster_all_tree.query(center, k=k)
        out: List[int] = []
        self._append_unique_int(out, np.asarray(indices, dtype=np.int64).reshape(-1))
        if len(out) < int(count) and k < n_clusters:
            _, indices = self.cluster_all_tree.query(center, k=n_clusters)
            self._append_unique_int(out, np.asarray(indices, dtype=np.int64).reshape(-1))
        return out[: int(count)]

    def _lag_candidates(self, block_idx: int, lag: int) -> List[int]:
        if int(lag) == 1:
            count = self.lag1_block_count
            multiplier = self.lag1_center_multiplier
        elif int(lag) == 2:
            count = self.lag2_block_count
            multiplier = self.lag2_center_multiplier
        else:
            raise ValueError(f"Only lag 1 and lag 2 are supported, got {lag}")
        return self._cluster_candidates_from_shifted_center(
            int(block_idx),
            multiplier,
            count,
        )

    def _precompute_message(self) -> str:
        return (
            "Pre-computing calibrated signed-2D shifted-center lag643 "
            f"(smooth={self.smooth}, block={self.block_shape}, "
            f"lag_blocks={LAG_COUNTS[0]}/{LAG_COUNTS[1]}/{LAG_COUNTS[2]}, "
            f"reference_advec=({self.reference_advec_lat:.6f},"
            f"{self.reference_advec_lon:.6f}), past_step="
            f"({self.past_offset_vector[0]:.6f},{self.past_offset_vector[1]:.6f}), "
            f"center_multipliers={self.lag1_center_multiplier}/"
            f"{self.lag2_center_multiplier})..."
        )

    def cluster_summary(self) -> dict[str, Any]:
        out = super().cluster_summary()
        lag1_offset = self.lag1_center_multiplier * self.past_offset_vector
        lag2_offset = self.lag2_center_multiplier * self.past_offset_vector
        out.update(
            {
                "spec_name": SHIFTED_SPEC_NAME,
                "geometry": "shifted",
                "geometry_definition": (
                    "t-1 four nearest blocks at calibrated v center; "
                    "t-2 three nearest blocks at calibrated 2v center, "
                    "applied in the past -v direction"
                ),
                "conditioning_mode": "calibrated_signed_2d_shifted_center",
                "reference_advec_lat": self.reference_advec_lat,
                "reference_advec_lon": self.reference_advec_lon,
                "reference_advec_norm": self.reference_advec_norm,
                "past_step_lat": float(self.past_offset_vector[0]),
                "past_step_lon": float(self.past_offset_vector[1]),
                "lag1_center_multiplier": self.lag1_center_multiplier,
                "lag2_center_multiplier": self.lag2_center_multiplier,
                "lag1_center_offset_lat": float(lag1_offset[0]),
                "lag1_center_offset_lon": float(lag1_offset[1]),
                "lag2_center_offset_lat": float(lag2_offset[0]),
                "lag2_center_offset_lon": float(lag2_offset[1]),
            }
        )
        return out


def build_shifted_model(
    smooth: float,
    input_map,
    reference_advec_lat: float,
    reference_advec_lon: float,
    grid_coords=None,
    device: Optional[str] = None,
    dtype=None,
    **kwargs,
) -> CalibratedShiftedCenter4x4Lag643VecchiaFit:
    """Construct the calibrated shifted-center lag-643 model."""
    mapped = move_input_map(input_map, device=device, dtype=dtype)
    return CalibratedShiftedCenter4x4Lag643VecchiaFit(
        smooth=smooth,
        input_map=mapped,
        grid_coords=grid_coords,
        reference_advec_lat=reference_advec_lat,
        reference_advec_lon=reference_advec_lon,
        **kwargs,
    )


__all__ = [
    "SHIFTED_SPEC_NAME",
    "BLOCK_SHAPE",
    "LAG_COUNTS",
    "LAG1_CENTER_MULTIPLIER",
    "LAG2_CENTER_MULTIPLIER",
    "shifted_model_spec",
    "CalibratedShiftedCenter4x4Lag643VecchiaFit",
    "build_shifted_model",
]
