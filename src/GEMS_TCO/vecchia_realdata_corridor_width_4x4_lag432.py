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

from typing import Any, Optional, Tuple

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


__all__ = [
    "REFERENCE_ADVEC_LON_ABS",
    "BLOCK_SHAPE",
    "LAG_COUNTS",
    "STRATEGY",
    "CORRIDOR_ANCHOR_MODE",
    "SPEC_NAME",
    "corridor_intervals",
    "model_spec",
    "move_input_map",
    "RealDataCorridorWidth4x4Lag432VecchiaFit",
    "build_model",
]
