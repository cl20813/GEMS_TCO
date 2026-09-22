"""Amarel 6/4/3 corridor-neighbor configuration.

This module is the clear import path for the established Amarel configuration.
It deliberately re-exports the existing implementation without subclassing or
rewriting any computation.  The legacy module remains importable while callers
move to this package path.
"""

from GEMS_TCO.vecchia_realdata_corridor_width_4x4_lag643 import (
    BLOCK_SHAPE,
    CORRIDOR_ANCHOR_MODE,
    LAG_COUNTS,
    REFERENCE_ADVEC_LON_ABS,
    SPEC_NAME,
    STRATEGY,
    RealDataCorridorWidth4x4Lag643VecchiaFit,
    build_model,
    corridor_intervals,
    model_spec,
    move_input_map,
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
