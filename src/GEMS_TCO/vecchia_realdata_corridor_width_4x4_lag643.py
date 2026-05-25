"""
vecchia_realdata_corridor_width_4x4_lag643.py

Created 2026-05-24.

Real-data ST cluster Vecchia wrapper for the preferred corridor-width geometry.
This file fixes the conditioning design so Amarel and local scripts can import
one obvious model name without rebuilding the simulation sweep logic.

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
  The base Vecchia engine infers CPU/GPU from the tensors in input_map.  For GPU
  on Amarel, move input_map tensors to cuda before constructing the model, or
  call build_model(..., device="cuda").  For CPU tests, leave tensors on CPU or
  call build_model(..., device="cpu").
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

from GEMS_TCO.vecchia_cluster import StrategyClusterVecchiaFit


REFERENCE_ADVEC_LON_ABS = 0.126
BLOCK_SHAPE = (4, 4)
LAG_COUNTS = (6, 4, 3)
STRATEGY = "offset_corridor_tapered"
CORRIDOR_ANCHOR_MODE = "width"
SPEC_NAME = "corridor_width_4x4_lag643_delta0p126"


def corridor_intervals(reference_advec_lon_abs: float = REFERENCE_ADVEC_LON_ABS):
    """Return the default real-data corridor intervals for a one-step delta."""
    delta = float(abs(reference_advec_lon_abs))
    return (0.5 * delta, 1.5 * delta), (0.0, 2.0 * delta)


def model_spec(reference_advec_lon_abs: float = REFERENCE_ADVEC_LON_ABS) -> dict[str, Any]:
    """Small metadata dict for logs and fit summary rows."""
    lag1_interval, lag2_interval = corridor_intervals(reference_advec_lon_abs)
    lag0, lag1, lag2 = LAG_COUNTS
    return {
        "spec_name": SPEC_NAME,
        "strategy": STRATEGY,
        "conditioning_mode": "corridor_width",
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


def move_input_map(input_map, device: Optional[str] = None, dtype=None):
    """Move an input_map to CPU/GPU without changing non-tensor values."""
    if device is None and dtype is None:
        return input_map
    out = {}
    for key, value in input_map.items():
        if hasattr(value, "to"):
            kwargs = {}
            if device is not None:
                kwargs["device"] = torch.device(device)
            if dtype is not None:
                kwargs["dtype"] = dtype
            out[key] = value.to(**kwargs)
        else:
            out[key] = value
    return out


class RealDataCorridorWidth4x4Lag643VecchiaFit(StrategyClusterVecchiaFit):
    """Fixed real-data corridor-width 4x4 lag-643 cluster Vecchia model."""

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
) -> RealDataCorridorWidth4x4Lag643VecchiaFit:
    """Construct the fixed 643 model, optionally moving tensors first."""
    mapped = move_input_map(input_map, device=device, dtype=dtype)
    return RealDataCorridorWidth4x4Lag643VecchiaFit(
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
    "RealDataCorridorWidth4x4Lag643VecchiaFit",
    "build_model",
]
