"""Publication API for fixed corridor-neighbor Vecchia models.

The 432 and 643 labels denote the numbers of conditioning blocks used at the
current, first-lag, and second-lag time layers.  Their modules document the
fixed 4x4 target-block geometry.
"""

from . import corridor_lag432, corridor_lag643, directional_lag432, directional_lag643
from .corridor_lag432 import Lag432CorridorVecchia
from .corridor_lag643 import Lag643CorridorVecchia
from .directional_lag432 import DirectionalLag432CorridorVecchia
from .directional_lag643 import DirectionalLag643CorridorVecchia
from .generalized_cauchy import (
    GeneralizedCauchyLag432CorridorVecchia,
    GeneralizedCauchyLag643CorridorVecchia,
    NoNuggetGeneralizedCauchyLag432CorridorVecchia,
    NoNuggetGeneralizedCauchyLag643CorridorVecchia,
)
from .spline import NoNuggetSplineMaternLag643CorridorVecchia, SplineMaternLag643CorridorVecchia

__all__ = [
    "corridor_lag432",
    "corridor_lag643",
    "directional_lag432",
    "directional_lag643",
    "Lag432CorridorVecchia",
    "DirectionalLag432CorridorVecchia",
    "Lag643CorridorVecchia",
    "DirectionalLag643CorridorVecchia",
    "GeneralizedCauchyLag432CorridorVecchia",
    "NoNuggetGeneralizedCauchyLag432CorridorVecchia",
    "GeneralizedCauchyLag643CorridorVecchia",
    "NoNuggetGeneralizedCauchyLag643CorridorVecchia",
    "SplineMaternLag643CorridorVecchia",
    "NoNuggetSplineMaternLag643CorridorVecchia",
]
