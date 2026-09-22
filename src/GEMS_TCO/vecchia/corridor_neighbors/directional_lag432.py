"""Directional two-dimensional corridor Vecchia model with a 4/3/2 budget.

This module mirrors :mod:`directional_lag643` while preserving the smaller
4/3/2 conditioning budget and its conservative diagonal block-span rule.
The conditioning graph is built once from a reference advection vector; the
likelihood continues to estimate both advection parameters without rebuilding
that graph inside optimizer iterations.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from .corridor_lag432 import Lag432CorridorVecchia
from .corridor_lag643 import BLOCK_SHAPE, CORRIDOR_ANCHOR_MODE, REFERENCE_ADVEC_LON_ABS

LAG_COUNTS = (4, 3, 2)
DIRECTIONAL_SPEC_NAME = "directional_corridor_width_4x4_lag432"
LAG1_CORRIDOR_MULTIPLIERS = (0.5, 1.5)
LAG2_CORRIDOR_MULTIPLIERS = (0.0, 2.0)


def directional_model_spec(
    reference_advec_lat: float,
    reference_advec_lon: float,
) -> dict[str, Any]:
    """Return explicit metadata for the signed two-dimensional corridor.

    The covariance uses ``h - v * tau``. A current target therefore looks for
    its most correlated past conditioning points near ``-v * tau``; the
    ``past_step_*`` fields make that sign convention explicit.
    """

    advec = np.asarray([reference_advec_lat, reference_advec_lon], dtype=np.float64)
    if not np.isfinite(advec).all():
        raise ValueError("reference advection components must be finite")
    past_step = -advec
    lag0, lag1, lag2 = LAG_COUNTS
    return {
        "spec_name": DIRECTIONAL_SPEC_NAME,
        "conditioning_mode": "directional_corridor_width",
        "block_shape": BLOCK_SHAPE,
        "lag_counts": LAG_COUNTS,
        "lag_pattern": f"{lag0}/{lag1}/{lag2}",
        "reference_advec_lat": float(advec[0]),
        "reference_advec_lon": float(advec[1]),
        "reference_advec_norm": float(np.linalg.norm(advec)),
        "past_step_lat": float(past_step[0]),
        "past_step_lon": float(past_step[1]),
        "lag1_past_segment_start": (0.5 * past_step).tolist(),
        "lag1_past_segment_end": (1.5 * past_step).tolist(),
        "lag2_past_segment_start": (0.0 * past_step).tolist(),
        "lag2_past_segment_end": (2.0 * past_step).tolist(),
        "lag1_corridor_multipliers": LAG1_CORRIDOR_MULTIPLIERS,
        "lag2_corridor_multipliers": LAG2_CORRIDOR_MULTIPLIERS,
        "corridor_anchor_mode": CORRIDOR_ANCHOR_MODE,
    }


class DirectionalLag432CorridorVecchia(Lag432CorridorVecchia):
    """Lag-432 corridor whose conditioning geometry follows a 2-D wind seed.

    ``reference_advec_lat/lon`` are model advection parameters, not past-point
    offsets. For a current target, lagged conditioning clusters are placed on
    the opposite vector ``(-reference_advec_lat, -reference_advec_lon)``. The
    seed fixes only the once-per-fit Vecchia conditioning geometry.
    """

    spec_name = DIRECTIONAL_SPEC_NAME
    lag1_corridor_multipliers = LAG1_CORRIDOR_MULTIPLIERS
    lag2_corridor_multipliers = LAG2_CORRIDOR_MULTIPLIERS

    def __init__(
        self,
        smooth: float,
        input_map,
        grid_coords=None,
        reference_advec_lat: float = 0.0,
        reference_advec_lon: float = -REFERENCE_ADVEC_LON_ABS,
        second_lag_stride: int = 2,
        target_chunk_size: int = 128,
        min_target_points: int = 1,
        max_neighbor_search: Optional[int] = None,
        block_row_offset: int = 0,
        block_col_offset: int = 0,
    ):
        self.reference_advec_lat = float(reference_advec_lat)
        self.reference_advec_lon = float(reference_advec_lon)
        if not np.isfinite([self.reference_advec_lat, self.reference_advec_lon]).all():
            raise ValueError("reference advection components must be finite")
        self.reference_advec_norm = float(
            np.hypot(self.reference_advec_lat, self.reference_advec_lon)
        )
        self.past_step_vector = np.asarray(
            [-self.reference_advec_lat, -self.reference_advec_lon],
            dtype=np.float64,
        )
        super().__init__(
            smooth=smooth,
            input_map=input_map,
            grid_coords=grid_coords,
            # The inherited scalar initializes parent lookup metadata;
            # _lag_candidates below uses the signed 2-D vector instead.
            reference_advec_lon_abs=self.reference_advec_norm,
            second_lag_stride=second_lag_stride,
            target_chunk_size=target_chunk_size,
            min_target_points=min_target_points,
            max_neighbor_search=max_neighbor_search,
            block_row_offset=block_row_offset,
            block_col_offset=block_col_offset,
        )

    def _vector_corridor_block_span(self, direction: np.ndarray) -> float:
        """Use the conservative diagonal footprint retained by the 432 model."""

        del direction
        if (
            self.cluster_lat_min is None
            or self.cluster_lat_max is None
            or self.cluster_lon_min is None
            or self.cluster_lon_max is None
        ):
            return float("nan")
        lat_width = float(np.nanmedian(self.cluster_lat_max - self.cluster_lat_min))
        lon_width = float(np.nanmedian(self.cluster_lon_max - self.cluster_lon_min))
        return float(np.hypot(lat_width, lon_width))

    def _lag_candidates(self, block_idx: int, lag: int) -> List[int]:
        if lag not in (1, 2):
            raise ValueError(f"lag must be 1 or 2, got {lag}")
        count = self.lag1_block_count if lag == 1 else self.lag2_block_count
        multipliers = self.lag1_corridor_multipliers if lag == 1 else self.lag2_corridor_multipliers
        return self._cluster_candidates_from_vector_corridor(block_idx, multipliers, count)

    def _precompute_message(self) -> str:
        return (
            "Pre-computing directional reference-vector corridor lag432 "
            f"(smooth={self.smooth}, block={self.block_shape}, "
            f"lag_blocks={LAG_COUNTS[0]}/{LAG_COUNTS[1]}/{LAG_COUNTS[2]}, "
            f"reference_advec=({self.reference_advec_lat:.6f},"
            f"{self.reference_advec_lon:.6f}), past_step="
            f"({self.past_step_vector[0]:.6f},{self.past_step_vector[1]:.6f}), "
            f"corridor_multipliers={self.lag1_corridor_multipliers}/"
            f"{self.lag2_corridor_multipliers})..."
        )

    def cluster_summary(self) -> dict[str, Any]:
        """Return the grouped-batch summary with directional-corridor metadata."""

        out = super().cluster_summary()
        for key in (
            "lag1_lon_offset",
            "lag2_lon_offset",
            "lag1_lon_interval_lo",
            "lag1_lon_interval_hi",
            "lag2_lon_interval_lo",
            "lag2_lon_interval_hi",
            "grid_lon_step",
            "corridor_block_lon_width",
        ):
            out.pop(key, None)
        out.update(
            {
                "spec_name": DIRECTIONAL_SPEC_NAME,
                "geometry": "directional",
                "geometry_definition": (
                    "t-1 reference-vector corridor [0.5v,1.5v]; "
                    "t-2 reference-vector corridor [0,2v], applied in the past -v direction"
                ),
                "conditioning_mode": "directional_corridor_width",
                "reference_advec_lat": self.reference_advec_lat,
                "reference_advec_lon": self.reference_advec_lon,
                "reference_advec_norm": self.reference_advec_norm,
                "past_step_lat": float(self.past_step_vector[0]),
                "past_step_lon": float(self.past_step_vector[1]),
                "lag1_corridor_multiplier_lo": self.lag1_corridor_multipliers[0],
                "lag1_corridor_multiplier_hi": self.lag1_corridor_multipliers[1],
                "lag2_corridor_multiplier_lo": self.lag2_corridor_multipliers[0],
                "lag2_corridor_multiplier_hi": self.lag2_corridor_multipliers[1],
            }
        )
        return out


__all__ = [
    "DIRECTIONAL_SPEC_NAME",
    "BLOCK_SHAPE",
    "LAG_COUNTS",
    "LAG1_CORRIDOR_MULTIPLIERS",
    "LAG2_CORRIDOR_MULTIPLIERS",
    "directional_model_spec",
    "DirectionalLag432CorridorVecchia",
]
