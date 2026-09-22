"""Pure-spatial covariance models and Vecchia inference engines.

The package root exposes the fitted-model classes and high-level fitting
functions that form the supported public API.  Exports are loaded lazily so
the lightweight numerical submodules do not pull in the full inference stack.
Lower-level helpers remain available from their focused submodules.
"""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "NoNuggetFixedBetaAnisotropicCauchySpatialVecchia": (
        ".anisotropic_cauchy",
        "NoNuggetFixedBetaAnisotropicCauchySpatialVecchia",
    ),
    "FixedBetaAnisotropicCauchySpatialVecchia": (
        ".anisotropic_cauchy",
        "FixedBetaAnisotropicCauchySpatialVecchia",
    ),
    "NoNuggetAnisotropicCauchySpatialVecchia": (
        ".anisotropic_cauchy",
        "NoNuggetAnisotropicCauchySpatialVecchia",
    ),
    "NoNuggetAnisotropicMaternSpatialVecchia": (
        ".anisotropic_matern",
        "NoNuggetAnisotropicMaternSpatialVecchia",
    ),
    "AnisotropicMaternSpatialVecchia": (
        ".anisotropic_matern",
        "AnisotropicMaternSpatialVecchia",
    ),
    "NoNuggetIsotropicMaternSpatialVecchia": (
        ".isotropic",
        "NoNuggetIsotropicMaternSpatialVecchia",
    ),
    "IsotropicMaternSpatialVecchia": (
        ".isotropic",
        "IsotropicMaternSpatialVecchia",
    ),
    "MaternParameters": (".matern_bessel", "MaternParameters"),
    "SpatialLBFGSFitResult": (".base", "SpatialLBFGSFitResult"),
    "cauchy_phi_init_from_natural": (
        ".anisotropic_cauchy",
        "cauchy_phi_init_from_natural",
    ),
    "fit_full_matern": (".matern_bessel", "fit_full_matern"),
    "fit_vecchia_matern_from_batches": (
        ".matern_bessel",
        "fit_vecchia_matern_from_batches",
    ),
    "matern_corr_bessel": (".matern_bessel", "matern_corr_bessel"),
    "vecchia_batches_to_numpy": (".matern_bessel", "vecchia_batches_to_numpy"),
}

__all__ = [
    "AnisotropicMaternSpatialVecchia",
    "FixedBetaAnisotropicCauchySpatialVecchia",
    "IsotropicMaternSpatialVecchia",
    "NoNuggetAnisotropicCauchySpatialVecchia",
    "NoNuggetAnisotropicMaternSpatialVecchia",
    "NoNuggetFixedBetaAnisotropicCauchySpatialVecchia",
    "NoNuggetIsotropicMaternSpatialVecchia",
    "MaternParameters",
    "SpatialLBFGSFitResult",
    "cauchy_phi_init_from_natural",
    "fit_full_matern",
    "fit_vecchia_matern_from_batches",
    "matern_corr_bessel",
    "vecchia_batches_to_numpy",
]


def __getattr__(name: str) -> Any:
    """Load a supported public symbol on first access."""
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
