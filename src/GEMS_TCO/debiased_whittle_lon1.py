"""Compatibility import for ``longitude_difference`` Debiased Whittle."""

from GEMS_TCO.debiased_whittle.engine import components_for


_components = components_for("longitude_difference")


class full_vecc_dw_likelihoods(_components.comparison_class):
    """Legacy comparison wrapper configured for the longitude difference."""


class debiased_whittle_preprocess(
    _components.preprocess_class, full_vecc_dw_likelihoods
):
    """Legacy class name for longitude first-difference preprocessing."""


class debiased_whittle_likelihood(_components.likelihood_class):
    """Legacy class name for the longitude first-difference likelihood."""


full_vecc_dw_likelihoods.preprocess_class = debiased_whittle_preprocess
full_vecc_dw_likelihoods.likelihood_class = debiased_whittle_likelihood


__all__ = [
    "full_vecc_dw_likelihoods",
    "debiased_whittle_preprocess",
    "debiased_whittle_likelihood",
]
