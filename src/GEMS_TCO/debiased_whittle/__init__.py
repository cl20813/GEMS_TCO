"""Unified Debiased Whittle interface.

The numerical implementation remains compatible with the historical modules
(``debiased_whittle_2110`` and friends).  New code should use the descriptive
filter names documented in :mod:`GEMS_TCO.debiased_whittle.filters` and obtain
configured components through :mod:`GEMS_TCO.debiased_whittle.engine`.

The engine import is intentionally lazy: the historical ``2110`` module uses
the filter specifications as its common numerical core, so eager importing
would create a circular dependency during compatibility imports.
"""

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

__all__ = [
    "DebiasedWhittleEngine",
    "EngineComponents",
    "components_for",
    "SpatialFilterSpec",
    "FILTERS",
    "IDENTITY",
    "LATITUDE_DIFFERENCE",
    "LONGITUDE_DIFFERENCE",
    "CROSS_DIFFERENCE",
    "SUMMED_FIRST_DIFFERENCES",
    "get_filter_spec",
]


def __getattr__(name):
    if name in {"DebiasedWhittleEngine", "EngineComponents", "components_for"}:
        from .engine import DebiasedWhittleEngine, EngineComponents, components_for

        exports = {
            "DebiasedWhittleEngine": DebiasedWhittleEngine,
            "EngineComponents": EngineComponents,
            "components_for": components_for,
        }
        return exports[name]
    raise AttributeError(name)
