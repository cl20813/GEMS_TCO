"""Local 4/3/2 corridor-neighbor configuration.

This module is the clear import path for the established local configuration.
It deliberately re-exports the existing implementation without subclassing or
rewriting any computation.  The legacy module remains importable while callers
move to this package path.
"""

from GEMS_TCO.vecchia_realdata_corridor_width_4x4_lag432 import (
    BLOCK_SHAPE,
    CORRIDOR_ANCHOR_MODE,
    DIRECTIONAL_SPEC_NAME,
    LAG_COUNTS,
    REFERENCE_ADVEC_LON_ABS,
    SPEC_NAME,
    STRATEGY,
    DirectionalRealDataCorridorWidth4x4Lag432VecchiaFit,
    RealDataCorridorWidth4x4Lag432VecchiaFit,
    build_directional_model,
    build_model,
    corridor_intervals,
    directional_model_spec,
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
