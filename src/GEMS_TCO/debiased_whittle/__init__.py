"""Debiased Whittle estimators and spatial-filter specifications.

The scalar estimators share one numerical implementation configured by an
immutable :class:`SpatialFilterSpec`.  The mixed-frequency and vector-gradient
estimators remain separate because they define different objectives.
"""

from ._core import DebiasedWhittleFitResult
from .engine import DebiasedWhittleEngine
from .filters import (
    CROSS_DIFFERENCE,
    FILTERS,
    IDENTITY,
    LATITUDE_DIFFERENCE,
    LONGITUDE_DIFFERENCE,
    SUMMED_FIRST_DIFFERENCES,
    SpatialFilterSpec,
    get_filter_spec,
)
from .mixed_frequency import (
    MixedFrequencyCrossDifferencePreprocessor,
    MixedFrequencyDebiasedWhittleLikelihood,
    MixedFrequencyIdentityPreprocessor,
)
from .vector_gradient import VectorGradientDebiasedWhittleLikelihood, VectorGradientPreprocessor

__all__ = [
    "DebiasedWhittleEngine",
    "DebiasedWhittleFitResult",
    "SpatialFilterSpec",
    "FILTERS",
    "IDENTITY",
    "LATITUDE_DIFFERENCE",
    "LONGITUDE_DIFFERENCE",
    "CROSS_DIFFERENCE",
    "SUMMED_FIRST_DIFFERENCES",
    "get_filter_spec",
    "MixedFrequencyIdentityPreprocessor",
    "MixedFrequencyCrossDifferencePreprocessor",
    "MixedFrequencyDebiasedWhittleLikelihood",
    "VectorGradientPreprocessor",
    "VectorGradientDebiasedWhittleLikelihood",
]
