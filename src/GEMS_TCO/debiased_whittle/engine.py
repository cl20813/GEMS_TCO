"""Configured Debiased Whittle engine with stable legacy compatibility.

The five historical modules duplicated the same taper, DFT, expected
periodogram, covariance kernel, parameterization, and optimizer.  Their real
differences are captured by :class:`~.filters.SpatialFilterSpec`:

* the spatial stencil and resulting grid size;
* which zero-frequency row/column is excluded from the objective; and
* for ``identity``, the historical per-time-slice spatial demeaning.

The common numerical implementation lives in the private ``_core`` module.
This module creates independent configured subclasses; it never mutates a
process-wide "active filter".  Consequently several filters can be imported
and compared safely in the same Python process, while the old module paths are
kept as thin compatibility wrappers.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Type

from .filters import SpatialFilterSpec, get_filter_spec


@dataclass(frozen=True)
class EngineComponents:
    """The three legacy-compatible classes configured for one filter."""

    filter_spec: SpatialFilterSpec
    comparison_class: Type
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
        (common.debiased_whittle_likelihood,),
        {
            "__module__": __name__,
            "__doc__": f"Debiased Whittle likelihood using {spec.name!r}.",
            "filter_spec": spec,
        },
    )
    preprocess_class = type(
        f"{class_suffix}DebiasedWhittlePreprocess",
        (common.debiased_whittle_preprocess,),
        {
            "__module__": __name__,
            "__doc__": f"Grid preprocessing using {spec.name!r}.",
            "filter_spec": spec,
        },
    )
    comparison_class = type(
        f"{class_suffix}FullVecchiaDwLikelihoods",
        (common.full_vecc_dw_likelihoods,),
        {
            "__module__": __name__,
            "__doc__": f"Legacy full/Vecchia/DW comparison using {spec.name!r}.",
            "filter_spec": spec,
            "preprocess_class": preprocess_class,
            "likelihood_class": likelihood_class,
        },
    )
    return EngineComponents(
        filter_spec=spec,
        comparison_class=comparison_class,
        preprocess_class=preprocess_class,
        likelihood_class=likelihood_class,
    )


def components_for(filter_name: str | SpatialFilterSpec) -> EngineComponents:
    """Return immutable, legacy-compatible classes for ``filter_name``.

    Both descriptive names and old aliases are accepted.  For example,
    ``components_for("summed_first_differences")`` and
    ``components_for("2110")`` return the same cached classes.
    """

    spec = get_filter_spec(filter_name)
    return _components_for_name(spec.name)


class DebiasedWhittleEngine:
    """Small public facade around a filter-configured numerical engine.

    Parameters
    ----------
    spatial_filter:
        One of ``identity``, ``latitude_difference``,
        ``longitude_difference``, ``cross_difference``, or
        ``summed_first_differences``.  Historical aliases remain accepted.

    Notes
    -----
    This facade does not change parameter order, dtype/device behavior,
    numerical jitter, tapering, optimizer flow, or default spatial increments.
    It selects only the legacy filter-specific behavior recorded in the
    immutable specification.

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
        return self._components.filter_spec

    @property
    def preprocess_class(self) -> Type:
        return self._components.preprocess_class

    @property
    def likelihood_class(self) -> Type:
        return self._components.likelihood_class

    @property
    def comparison_class(self) -> Type:
        return self._components.comparison_class

    def make_preprocessor(self, *args, **kwargs):
        """Construct the legacy-compatible preprocessor for this filter."""

        return self.preprocess_class(*args, **kwargs)

    def make_likelihood(self):
        """Construct the legacy-compatible likelihood object."""

        return self.likelihood_class()


__all__ = ["DebiasedWhittleEngine", "EngineComponents", "components_for"]
