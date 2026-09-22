"""Configured scalar Debiased Whittle engine.

The five historical modules duplicated the same taper, DFT, expected
periodogram, covariance kernel, parameterization, and optimizer.  Their real
differences are captured by :class:`~.filters.SpatialFilterSpec`:

* the spatial stencil and resulting grid size;
* which zero-frequency row/column is excluded from the objective; and
* for ``identity``, the historical per-time-slice spatial demeaning.

The common numerical implementation lives in the private ``_core`` module.
This module creates independent configured subclasses and never mutates a
process-wide active filter, so several filters can be compared safely in one
Python process.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Type

from .filters import SpatialFilterSpec, get_filter_spec


@dataclass(frozen=True)
class EngineComponents:
    """The numerical classes configured for one scalar spatial filter."""

    filter_spec: SpatialFilterSpec
    preprocess_class: Type
    likelihood_class: Type


@lru_cache(maxsize=None)
def _components_for_name(filter_name: str) -> EngineComponents:
    # Imported lazily so the filter package can expose its declarative layer
    # without eagerly importing PyTorch and the larger historical dependency
    # surface of the numerical core.
    from . import _core as common

    spec = get_filter_spec(filter_name)
    class_suffix = "".join(part.title() for part in spec.name.split("_"))

    likelihood_class = type(
        f"{class_suffix}DebiasedWhittleLikelihood",
        (common.BaseDebiasedWhittleLikelihood,),
        {
            "__module__": __name__,
            "__doc__": f"Debiased Whittle likelihood using {spec.name!r}.",
            "filter_spec": spec,
        },
    )
    preprocess_class = type(
        f"{class_suffix}DebiasedWhittlePreprocessor",
        (common.BaseDebiasedWhittlePreprocessor,),
        {
            "__module__": __name__,
            "__doc__": f"Grid preprocessing using {spec.name!r}.",
            "filter_spec": spec,
        },
    )
    # ``type`` does not register dynamically-created classes in their declared
    # module.  Registration makes these public classes importable by name and
    # therefore pickleable in multiprocessing and saved workflows.
    globals()[likelihood_class.__name__] = likelihood_class
    globals()[preprocess_class.__name__] = preprocess_class
    return EngineComponents(
        filter_spec=spec,
        preprocess_class=preprocess_class,
        likelihood_class=likelihood_class,
    )


def components_for(filter_name: str | SpatialFilterSpec) -> EngineComponents:
    """Return immutable numerical classes configured for ``filter_name``."""

    spec = get_filter_spec(filter_name)
    return _components_for_name(spec.name)


class DebiasedWhittleEngine:
    """Small public facade around a filter-configured numerical engine.

    Parameters
    ----------
    spatial_filter:
        One of ``identity``, ``latitude_difference``,
        ``longitude_difference``, ``cross_difference``, or
        ``summed_first_differences``.

    Notes
    -----
    This facade does not change parameter order, dtype/device behavior,
    numerical jitter, tapering, optimizer flow, or default spatial increments.
    It selects only the filter-specific behavior recorded in the immutable
    specification.

    ``cross_difference`` retains the historical ``1111`` stencil
    ``[[-1, 1], [1, -1]]``.  Under the forward-difference definitions used by
    ``latitude_difference`` and ``longitude_difference``, that stencil is
    ``-D_lat D_lon``.  The sign is documented rather than changed so existing
    numerical results remain reproducible.
    """

    def __init__(self, spatial_filter: str | SpatialFilterSpec):
        self._components = components_for(spatial_filter)

    @property
    def filter_spec(self) -> SpatialFilterSpec:
        """Return the immutable configuration selected for this engine."""

        return self._components.filter_spec

    @property
    def preprocess_class(self) -> Type:
        """Return the filter-configured preprocessing class."""

        return self._components.preprocess_class

    @property
    def likelihood_class(self) -> Type:
        """Return the filter-configured likelihood class."""

        return self._components.likelihood_class

    def make_preprocessor(self, *args, **kwargs):
        """Construct the preprocessor for this filter."""

        return self.preprocess_class(*args, **kwargs)

    def make_likelihood(self):
        """Construct the likelihood object for this filter."""

        return self.likelihood_class()


__all__ = ["DebiasedWhittleEngine", "EngineComponents", "components_for"]
