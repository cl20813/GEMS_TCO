"""Compatibility import for ``summed_first_differences`` Debiased Whittle."""

from GEMS_TCO.debiased_whittle.engine import components_for


_components = components_for("summed_first_differences")


class full_vecc_dw_likelihoods(_components.comparison_class):
    """Legacy comparison wrapper configured for summed first differences."""


class debiased_whittle_preprocess(
    _components.preprocess_class, full_vecc_dw_likelihoods
):
    """Legacy class name for summed-first-differences preprocessing."""


class debiased_whittle_likelihood(_components.likelihood_class):
    """Legacy class name for the summed-first-differences likelihood."""


full_vecc_dw_likelihoods.preprocess_class = debiased_whittle_preprocess
full_vecc_dw_likelihoods.likelihood_class = debiased_whittle_likelihood


__all__ = [
    "full_vecc_dw_likelihoods",
    "debiased_whittle_preprocess",
    "debiased_whittle_likelihood",
]
