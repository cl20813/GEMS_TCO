"""Declarative spatial-filter definitions for Debiased Whittle estimation.

Each specification keeps the stencil, output-grid reduction, and frequency
exclusion rule together.  The public API intentionally accepts descriptive
names only; experiment-era numeric aliases do not belong in the package API.
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
    """Complete, immutable specification of a spatial filter.

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
    description: str
    weights: Tuple[WeightedOffset, ...]
    grid_reduction: Tuple[int, int]
    excluded_frequencies: FrequencyExclusion
    demean_input: bool = False

    @property
    def kernel_shape(self) -> Tuple[int, int]:
        """Return the exact valid-convolution kernel shape."""

        return self.grid_reduction[0] + 1, self.grid_reduction[1] + 1


IDENTITY = SpatialFilterSpec(
    name="identity",
    description="No spatial filter; demean each time slice and retain the full grid.",
    weights=(((0, 0), 1.0),),
    grid_reduction=(0, 0),
    excluded_frequencies="dc",
    demean_input=True,
)

LATITUDE_DIFFERENCE = SpatialFilterSpec(
    name="latitude_difference",
    description="First difference in latitude: D_lat X(i,j) = X(i+1,j) - X(i,j).",
    weights=(((0, 0), -1.0), ((1, 0), 1.0)),
    grid_reduction=(1, 0),
    excluded_frequencies="latitude_axis",
)

LONGITUDE_DIFFERENCE = SpatialFilterSpec(
    name="longitude_difference",
    description="First difference in longitude: D_lon X(i,j) = X(i,j+1) - X(i,j).",
    weights=(((0, 0), -1.0), ((0, 1), 1.0)),
    grid_reduction=(0, 1),
    excluded_frequencies="longitude_axis",
)

CROSS_DIFFERENCE = SpatialFilterSpec(
    name="cross_difference",
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
)

SUMMED_FIRST_DIFFERENCES = SpatialFilterSpec(
    name="summed_first_differences",
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

FILTERS: Mapping[str, SpatialFilterSpec] = MappingProxyType({spec.name: spec for spec in _SPECS})


def get_filter_spec(name: str | SpatialFilterSpec) -> SpatialFilterSpec:
    """Resolve a descriptive public filter name.

    Passing one of the package's canonical :class:`SpatialFilterSpec` objects
    is supported.  Ad-hoc specifications are deliberately rejected: the
    numerical engine is defined only for the five reviewed configurations.
    """

    if isinstance(name, SpatialFilterSpec):
        canonical = FILTERS.get(name.name)
        if canonical is None or name != canonical:
            raise ValueError(
                "Custom Debiased Whittle filter specifications are not supported; "
                "use one of the canonical objects exported by this module."
            )
        return canonical
    try:
        return FILTERS[name]
    except KeyError as exc:
        public_names = ", ".join(spec.name for spec in _SPECS)
        raise ValueError(
            f"Unknown Debiased Whittle spatial filter {name!r}. " f"Choose one of: {public_names}."
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
