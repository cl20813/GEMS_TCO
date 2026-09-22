"""Spatial-filter definitions used by the Debiased Whittle engine.

The old module suffixes (``raw``, ``lat1``, ``lon1``, ``1111``, and
``2110``) describe coefficient patterns rather than intent.  The public names
below make that intent explicit while retaining the old names as aliases:

============================  ==============================
Legacy name                   Public name
============================  ==============================
``raw``                       ``identity``
``lat1``                      ``latitude_difference``
``lon1``                      ``longitude_difference``
``1111``                      ``cross_difference``
``2110``                      ``summed_first_differences``
============================  ==============================

This module is deliberately declarative.  It records the exact historical
stencil, output-grid reduction, and zero-frequency exclusion rule.  Keeping
those three properties together prevents a future refactor from changing one
without changing the others.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping, Tuple


Offset = Tuple[int, int]
WeightedOffset = Tuple[Offset, float]
FrequencyExclusion = Literal["dc", "latitude_axis", "longitude_axis", "both_axes"]


@dataclass(frozen=True)
class SpatialFilterSpec:
    """Complete, immutable specification of one historical spatial filter.

    ``weights`` are expressed on the latitude/longitude grid as
    ``((latitude_offset, longitude_offset), coefficient)``.  PyTorch's
    ``conv2d`` performs cross-correlation, so these are the coefficients used
    directly by the legacy implementations; they must not be flipped.

    ``excluded_frequencies`` preserves the exact frequency mask used by the
    corresponding legacy Whittle likelihood.  ``identity`` and
    ``summed_first_differences`` exclude only DC, the one-direction filters
    exclude their zero-frequency axis, and ``cross_difference`` excludes both
    axes.
    """

    name: str
    legacy_name: str
    description: str
    weights: Tuple[WeightedOffset, ...]
    grid_reduction: Tuple[int, int]
    excluded_frequencies: FrequencyExclusion
    demean_input: bool = False
    lbfgs_uses_sum_objective: bool = False

    @property
    def kernel_shape(self) -> Tuple[int, int]:
        """Return the exact valid-convolution kernel shape."""

        return self.grid_reduction[0] + 1, self.grid_reduction[1] + 1


IDENTITY = SpatialFilterSpec(
    name="identity",
    legacy_name="raw",
    description="No spatial filter; demean each time slice and retain the full grid.",
    weights=(((0, 0), 1.0),),
    grid_reduction=(0, 0),
    excluded_frequencies="dc",
    demean_input=True,
)

LATITUDE_DIFFERENCE = SpatialFilterSpec(
    name="latitude_difference",
    legacy_name="lat1",
    description="First difference in latitude: D_lat X(i,j) = X(i+1,j) - X(i,j).",
    weights=(((0, 0), -1.0), ((1, 0), 1.0)),
    grid_reduction=(1, 0),
    excluded_frequencies="latitude_axis",
)

LONGITUDE_DIFFERENCE = SpatialFilterSpec(
    name="longitude_difference",
    legacy_name="lon1",
    description="First difference in longitude: D_lon X(i,j) = X(i,j+1) - X(i,j).",
    weights=(((0, 0), -1.0), ((0, 1), 1.0)),
    grid_reduction=(0, 1),
    excluded_frequencies="longitude_axis",
)

CROSS_DIFFERENCE = SpatialFilterSpec(
    name="cross_difference",
    legacy_name="1111",
    description=(
        "Historical cross difference with stencil [[-1, 1], [1, -1]]. "
        "With the forward D_lat and D_lon definitions above, this is "
        "-D_lat D_lon; the sign is retained exactly from the 1111 implementation."
    ),
    weights=(
        ((0, 0), -1.0),
        ((1, 0), 1.0),
        ((0, 1), 1.0),
        ((1, 1), -1.0),
    ),
    grid_reduction=(1, 1),
    excluded_frequencies="both_axes",
    # The legacy 1111 optimizer obtains the summed objective first and then
    # divides by the number of retained frequencies inside its closure.
    lbfgs_uses_sum_objective=True,
)

SUMMED_FIRST_DIFFERENCES = SpatialFilterSpec(
    name="summed_first_differences",
    legacy_name="2110",
    description=(
        "Sum of latitude and longitude first differences: "
        "D_lat X + D_lon X, stencil [[-2, 1], [1, 0]]."
    ),
    weights=(((0, 0), -2.0), ((1, 0), 1.0), ((0, 1), 1.0)),
    grid_reduction=(1, 1),
    excluded_frequencies="dc",
)


_SPECS = (
    IDENTITY,
    LATITUDE_DIFFERENCE,
    LONGITUDE_DIFFERENCE,
    CROSS_DIFFERENCE,
    SUMMED_FIRST_DIFFERENCES,
)

FILTERS: Mapping[str, SpatialFilterSpec] = MappingProxyType(
    {name: spec for spec in _SPECS for name in (spec.name, spec.legacy_name)}
)


def get_filter_spec(name: str | SpatialFilterSpec) -> SpatialFilterSpec:
    """Resolve an intuitive public name or a historical alias.

    Passing an existing :class:`SpatialFilterSpec` is supported so callers can
    validate and forward configuration without resolving it twice.
    """

    if isinstance(name, SpatialFilterSpec):
        return name
    try:
        return FILTERS[name]
    except KeyError as exc:
        public_names = ", ".join(spec.name for spec in _SPECS)
        raise ValueError(
            f"Unknown Debiased Whittle spatial filter {name!r}. "
            f"Choose one of: {public_names}."
        ) from exc


__all__ = [
    "SpatialFilterSpec",
    "FILTERS",
    "IDENTITY",
    "LATITUDE_DIFFERENCE",
    "LONGITUDE_DIFFERENCE",
    "CROSS_DIFFERENCE",
    "SUMMED_FIRST_DIFFERENCES",
    "get_filter_spec",
]
