"""Directional two-dimensional corridor Vecchia model with a 6/4/3 budget.

This is the direction-adapted counterpart of ``corridor_lag643``. It preserves
the 4x4 target blocks, 6/4/3 conditioning-block budgets, width-anchor rule, and
corridor multipliers, but replaces the scalar longitude displacement with a
two-dimensional reference advection vector.

Corridor geometry
-----------------
``reference_advec_lat/lon`` are covariance-model advection parameters.  With
the covariance convention ``d(h - v*tau)``, a current target looks for past
conditioning locations in the opposite direction ``-v``.  Therefore:

* t-1 uses the signed 2-D segment from ``target - 0.5*v`` to
  ``target - 1.5*v``;
* t-2 uses the signed 2-D segment from ``target`` to ``target - 2*v``.

This is intentionally not an exact shifted-center design. It is the
two-dimensional version of the fixed-longitude corridors
``[0.5*delta, 1.5*delta]`` and ``[0, 2*delta]``.

The conditioning graph is constructed once from the reference vector. The
subsequent likelihood still estimates both advection parameters continuously
without rebuilding that graph inside optimizer iterations.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from .corridor_lag643 import (
    BLOCK_SHAPE,
    CORRIDOR_ANCHOR_MODE,
    LAG_COUNTS,
    REFERENCE_ADVEC_LON_ABS,
    Lag643CorridorVecchia,
)

DIRECTIONAL_SPEC_NAME = "directional_corridor_width_4x4_lag643"
LAG1_CORRIDOR_MULTIPLIERS = (0.5, 1.5)
LAG2_CORRIDOR_MULTIPLIERS = (0.0, 2.0)


def directional_model_spec(
    reference_advec_lat: float,
    reference_advec_lon: float,
) -> dict[str, Any]:
    """Return explicit metadata for the signed two-dimensional corridor."""
    advec = np.asarray(
        [float(reference_advec_lat), float(reference_advec_lon)],
        dtype=np.float64,
    )
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
        "lag1_corridor_multipliers": LAG1_CORRIDOR_MULTIPLIERS,
        "lag2_corridor_multipliers": LAG2_CORRIDOR_MULTIPLIERS,
        "lag1_past_segment_start": (0.5 * past_step).tolist(),
        "lag1_past_segment_end": (1.5 * past_step).tolist(),
        "lag2_past_segment_start": (0.0 * past_step).tolist(),
        "lag2_past_segment_end": (2.0 * past_step).tolist(),
        "corridor_anchor_mode": CORRIDOR_ANCHOR_MODE,
    }


class DirectionalLag643CorridorVecchia(Lag643CorridorVecchia):
    """Directional 6/4/3 corridor defined by a reference advection vector."""

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
        # The parent scalar is retained only for inherited metadata and lookup
        # construction.  _lag_candidates below uses the signed vector corridor.
        super().__init__(
            smooth=smooth,
            input_map=input_map,
            grid_coords=grid_coords,
            reference_advec_lon_abs=max(self.reference_advec_norm, 1e-12),
            second_lag_stride=second_lag_stride,
            target_chunk_size=target_chunk_size,
            min_target_points=min_target_points,
            max_neighbor_search=max_neighbor_search,
            block_row_offset=block_row_offset,
            block_col_offset=block_col_offset,
        )

    def _lag_candidates(self, block_idx: int, lag: int) -> List[int]:
        if lag not in (1, 2):
            raise ValueError(f"lag must be 1 or 2, got {lag}")
        count = self.lag1_block_count if lag == 1 else self.lag2_block_count
        multipliers = self.lag1_corridor_multipliers if lag == 1 else self.lag2_corridor_multipliers
        return self._cluster_candidates_from_vector_corridor(
            block_idx,
            multipliers,
            count,
        )

    def _precompute_message(self) -> str:
        return (
            "Pre-computing directional reference-vector corridor lag643 "
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
    "DirectionalLag643CorridorVecchia",
]
