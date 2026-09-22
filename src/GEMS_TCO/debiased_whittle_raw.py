"""Compatibility import for the ``identity`` Debiased Whittle filter.

``raw`` historically means no spatial convolution, but preprocessing still
removes the observed spatial mean within each time slice.  New code should use
``DebiasedWhittleEngine("identity")``.
"""

from GEMS_TCO.debiased_whittle.engine import components_for


_components = components_for("identity")


class full_vecc_dw_likelihoods(_components.comparison_class):
    """Legacy comparison wrapper configured for the identity filter."""


class debiased_whittle_preprocess(
    _components.preprocess_class, full_vecc_dw_likelihoods
):
    """Legacy class name for identity-filter preprocessing."""


class debiased_whittle_likelihood(_components.likelihood_class):
    """Legacy class name for the identity-filter likelihood."""


full_vecc_dw_likelihoods.preprocess_class = debiased_whittle_preprocess
full_vecc_dw_likelihoods.likelihood_class = debiased_whittle_likelihood


__all__ = [
    "full_vecc_dw_likelihoods",
    "debiased_whittle_preprocess",
    "debiased_whittle_likelihood",
]
