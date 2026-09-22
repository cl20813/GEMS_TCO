"""Fixed-longitude corridor Vecchia model with a 6/4/3 block budget.

This module defines one explicit conditioning design independent of the
machine or dataset used to run it.

Geometry:
  - 4x4 regular-grid target clusters.
  - lag pattern 6/4/3:
      t:   6 previous same-time clusters in max-min order.
      t-1: 4 lagged clusters covering the longitude corridor.
      t-2: 3 lagged clusters covering the longitude corridor.
  - reference one-step |advec_lon| delta defaults to 0.126.
  - t-1 corridor defaults to [0.5 delta, 1.5 delta].
  - t-2 corridor defaults to [0.0 delta, 2.0 delta].
  - corridor_anchor_mode="width", so anchors are placed to cover the corridor
    width, then the remaining budget is filled near the corridor midpoint.

Device convention:
  The base Vecchia engine infers CPU/GPU from the tensors in ``input_map``.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from ._geometry import _CorridorClusterVecchia

REFERENCE_ADVEC_LON_ABS = 0.126
BLOCK_SHAPE = (4, 4)
LAG_COUNTS = (6, 4, 3)
CORRIDOR_ANCHOR_MODE = "width"
SPEC_NAME = "corridor_width_4x4_lag643"


def _normalized_delta(reference_advec_lon_abs: float) -> float:
    delta = float(abs(reference_advec_lon_abs))
    if not np.isfinite(delta):
        raise ValueError("reference_advec_lon_abs must be finite")
    return delta


def corridor_intervals(reference_advec_lon_abs: float = REFERENCE_ADVEC_LON_ABS):
    """Return the default corridor intervals for a one-step displacement."""
    delta = _normalized_delta(reference_advec_lon_abs)
    return (0.5 * delta, 1.5 * delta), (0.0, 2.0 * delta)


def model_spec(reference_advec_lon_abs: float = REFERENCE_ADVEC_LON_ABS) -> dict[str, Any]:
    """Small metadata dict for logs and fit summary rows."""
    delta = _normalized_delta(reference_advec_lon_abs)
    lag1_interval, lag2_interval = corridor_intervals(delta)
    lag0, lag1, lag2 = LAG_COUNTS
    return {
        "spec_name": SPEC_NAME,
        "conditioning_mode": "fixed_longitude_corridor_width",
        "block_shape": BLOCK_SHAPE,
        "lag_counts": LAG_COUNTS,
        "lag_pattern": f"{lag0}/{lag1}/{lag2}",
        "reference_advec_lon_abs": delta,
        "lag1_lon_offset": delta,
        "lag2_lon_offset": 2.0 * delta,
        "lag1_lon_interval": lag1_interval,
        "lag2_lon_interval": lag2_interval,
        "corridor_anchor_mode": CORRIDOR_ANCHOR_MODE,
    }


class Lag643CorridorVecchia(_CorridorClusterVecchia):
    """Fixed-longitude corridor Vecchia model with 4x4 targets and 6/4/3 lags."""

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
        second_lag_stride: int = 2,
        target_chunk_size: int = 128,
        min_target_points: int = 1,
        max_neighbor_search: Optional[int] = None,
        block_row_offset: int = 0,
        block_col_offset: int = 0,
        covariance_backend: str = "auto",
    ):
        delta = _normalized_delta(reference_advec_lon_abs)
        default_lag1_interval, default_lag2_interval = corridor_intervals(delta)
        super().__init__(
            smooth=smooth,
            input_map=input_map,
            grid_coords=grid_coords,
            block_shape=BLOCK_SHAPE,
            lag0_block_count=LAG_COUNTS[0],
            lag1_block_count=LAG_COUNTS[1],
            lag2_block_count=LAG_COUNTS[2],
            second_lag_stride=second_lag_stride,
            lag1_lon_offset=delta,
            lag2_lon_offset=2.0 * delta,
            lag1_lon_interval=(
                default_lag1_interval if lag1_lon_interval is None else lag1_lon_interval
            ),
            lag2_lon_interval=(
                default_lag2_interval if lag2_lon_interval is None else lag2_lon_interval
            ),
            corridor_anchor_mode=CORRIDOR_ANCHOR_MODE,
            target_chunk_size=target_chunk_size,
            min_target_points=min_target_points,
            max_neighbor_search=max_neighbor_search,
            block_row_offset=block_row_offset,
            block_col_offset=block_col_offset,
            covariance_backend=covariance_backend,
        )
        self.reference_advec_lon_abs = delta


__all__ = [
    "REFERENCE_ADVEC_LON_ABS",
    "BLOCK_SHAPE",
    "LAG_COUNTS",
    "CORRIDOR_ANCHOR_MODE",
    "SPEC_NAME",
    "corridor_intervals",
    "model_spec",
    "Lag643CorridorVecchia",
]
