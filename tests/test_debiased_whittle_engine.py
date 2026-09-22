"""Characterization tests for the unified Debiased Whittle filter engine.

The numerical goldens were captured from the five pre-refactor modules.  They
protect filter stencils, grid shapes, frequency masks, covariance accumulation
order, expected-periodogram calculations, loss values, and gradients.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import unittest

import torch

from GEMS_TCO.debiased_whittle.engine import DebiasedWhittleEngine, components_for
from GEMS_TCO.debiased_whittle.filters import get_filter_spec


FILTERS = {
    "identity": "raw",
    "latitude_difference": "lat1",
    "longitude_difference": "lon1",
    "cross_difference": "1111",
    "summed_first_differences": "2110",
}


def _unsorted_grid() -> torch.Tensor:
    rows = []
    for i, latitude in enumerate((10.0, 11.0, 12.0)):
        for j, longitude in enumerate((100.0, 101.0, 102.0, 103.0)):
            rows.append((latitude, longitude, 10 * i + j, 5.0))
    grid = torch.tensor(rows, dtype=torch.float32)
    order = torch.tensor((11, 0, 5, 8, 3, 6, 1, 10, 7, 4, 9, 2))
    return grid[order]


class FilterSpecificationTests(unittest.TestCase):
    def test_descriptive_names_and_legacy_aliases_resolve_to_same_spec(self):
        for descriptive_name, legacy_name in FILTERS.items():
            with self.subTest(filter=descriptive_name):
                self.assertIs(
                    get_filter_spec(descriptive_name),
                    get_filter_spec(legacy_name),
                )

    def test_exact_stencil_order_grid_reduction_and_frequency_masks(self):
        expected = {
            "identity": (
                (((0, 0), 1.0),),
                (0, 0),
                "dc",
            ),
            "latitude_difference": (
                (((0, 0), -1.0), ((1, 0), 1.0)),
                (1, 0),
                "latitude_axis",
            ),
            "longitude_difference": (
                (((0, 0), -1.0), ((0, 1), 1.0)),
                (0, 1),
                "longitude_axis",
            ),
            "cross_difference": (
                (
                    ((0, 0), -1.0),
                    ((1, 0), 1.0),
                    ((0, 1), 1.0),
                    ((1, 1), -1.0),
                ),
                (1, 1),
                "both_axes",
            ),
            "summed_first_differences": (
                (((0, 0), -2.0), ((1, 0), 1.0), ((0, 1), 1.0)),
                (1, 1),
                "dc",
            ),
        }
        for name, (weights, reduction, exclusion) in expected.items():
            with self.subTest(filter=name):
                spec = get_filter_spec(name)
                self.assertEqual(spec.weights, weights)
                self.assertEqual(spec.grid_reduction, reduction)
                self.assertEqual(spec.excluded_frequencies, exclusion)


class PreprocessingCharacterizationTests(unittest.TestCase):
    def test_pre_refactor_stencils_shapes_dtype_and_raw_demeaning(self):
        grid = _unsorted_grid()
        expected = {
            "identity": (
                (12, 4),
                (11.5, -11.5, -0.5, 8.5, -8.5, 0.5,
                 -10.5, 10.5, 1.5, -1.5, 9.5, -9.5),
            ),
            "latitude_difference": ((8, 4), (10.0,) * 8),
            "longitude_difference": ((9, 4), (1.0,) * 9),
            "cross_difference": ((6, 4), (0.0,) * 6),
            "summed_first_differences": ((6, 4), (11.0,) * 6),
        }

        for name, (shape, values) in expected.items():
            with self.subTest(filter=name):
                preprocessor = DebiasedWhittleEngine(name).make_preprocessor(
                    [grid], [{0: grid}], 0, [0.0] * 7,
                    (10.0, 12.0), (100.0, 103.0),
                )
                output = preprocessor.generate_spatially_filtered_days(
                    10.0, 12.0, 100.0, 103.0
                )
                self.assertEqual(tuple(output.shape), shape)
                self.assertEqual(output.dtype, torch.float64)
                torch.testing.assert_close(
                    output[:, 2], torch.tensor(values, dtype=torch.float64),
                    rtol=0.0, atol=0.0,
                )

        # Identity/raw historically preserves input row order while demeaning.
        identity = DebiasedWhittleEngine("identity").make_preprocessor(
            [grid], [{0: grid}], 0, [0.0] * 7,
            (10.0, 12.0), (100.0, 103.0),
        ).generate_spatially_filtered_days(10.0, 12.0, 100.0, 103.0)
        torch.testing.assert_close(identity[:, :2], grid[:, :2].to(torch.float64))


class LikelihoodCharacterizationTests(unittest.TestCase):
    PARAMS = torch.tensor(
        (0.1, -0.2, 0.3, -0.4, 0.05, -0.07, -2.0), dtype=torch.float64
    )

    def test_covariance_matches_pre_refactor_values_exactly(self):
        u1 = torch.tensor(((0.0, 0.044), (-0.044, 0.088)), dtype=torch.float64)
        u2 = torch.tensor(0.063, dtype=torch.float64)
        time_lag = torch.tensor(1.0, dtype=torch.float64)
        expected = {
            "identity": (0.6833347595285333, 0.6844559861996307,
                         0.6804722928454556, 0.6838147292256055),
            "latitude_difference": (0.0017412400119802829, 0.0017624836451225478,
                                    0.0016878676303986317, 0.0017503104077534193),
            "longitude_difference": (0.0025503100944632706, 0.002560200975069815,
                                     0.0025252117407970065, 0.0025545399966567794),
            "cross_difference": (1.520747305971959e-05, 1.5551859019580228e-05,
                                 1.4351781347232695e-05, 1.535425270204893e-05),
            "summed_first_differences": (0.004379338875855843, 0.004340157915350584,
                                         0.0043680397806680205, 0.00425138360344679),
        }
        for name, golden in expected.items():
            with self.subTest(filter=name):
                actual = components_for(name).likelihood_class.cov_spatial_difference(
                    u1, u2, time_lag, self.PARAMS, 0.044, 0.063
                )
                torch.testing.assert_close(
                    actual,
                    torch.tensor(golden, dtype=torch.float64).reshape(2, 2),
                    rtol=0.0,
                    atol=0.0,
                )

    def test_frequency_exclusion_sum_and_count(self):
        terms = torch.arange(12, dtype=torch.float64).reshape(3, 4)
        expected = {
            "identity": (66.0, 11),
            "latitude_difference": (60.0, 8),
            "longitude_difference": (54.0, 9),
            "cross_difference": (48.0, 6),
            "summed_first_differences": (66.0, 11),
        }
        for name, (golden_sum, golden_count) in expected.items():
            with self.subTest(filter=name):
                actual_sum, actual_count = (
                    components_for(name).likelihood_class._retained_frequency_sum(
                        terms, 3, 4
                    )
                )
                self.assertEqual(actual_sum.item(), golden_sum)
                self.assertEqual(actual_count, golden_count)

    def test_loss_and_gradient_match_pre_refactor_goldens(self):
        n1, n2, p_time = 3, 4, 2
        taper = torch.arange(
            1, (2 * n1 - 1) * (2 * n2 - 1) + 1, dtype=torch.float64
        ).reshape(2 * n1 - 1, 2 * n2 - 1) / 35.0
        real = torch.arange(1, n1 * n2 * p_time + 1, dtype=torch.float64).reshape(
            n1, n2, p_time
        ) / 10.0
        imag = torch.flip(real, (0,)) / 7.0
        j_vector = torch.complex(real, imag)
        sample_periodogram = (
            j_vector.unsqueeze(-1) @ j_vector.unsqueeze(-2).conj()
        )

        goldens = {
            "identity": (
                1291.7828786055752,
                (-438.02454739224254, 54.017955733084165, -104.00935195287389,
                 25.050743147353934, 15.804948913498656, -119.69528694475721,
                 -861.6705873228259),
            ),
            "latitude_difference": (
                873.7278228954283,
                (-183.76002618123545, -5.439993529531193, -81.08572217201939,
                 -2.2832110274431727, -1.0992231742022653, 0.27031873325553235,
                 -697.9745505959103),
            ),
            "longitude_difference": (
                861.7809002650455,
                (-175.55293155109908, 0.24822107827536932, -96.23807357659113,
                 0.3323354511123835, 0.4124766960937336, -0.659514670285489,
                 -694.3921737889793),
            ),
            "cross_difference": (
                397.67081018709183,
                (-86.5000951626935, 0.04961523321458006, -32.30919046944056,
                 0.0008843963009810135, 0.0033293927024560244,
                 -0.002963306482552497, -317.3378106036605),
            ),
            "summed_first_differences": (
                336.46598648604663,
                (-51.853047565156885, -15.967077815374296, -42.41532638864612,
                 -6.697871036259073, -0.8136406981486743, 1.702983377326774,
                 -291.3603246779659),
            ),
        }

        for name, (golden_loss, golden_gradient) in goldens.items():
            with self.subTest(filter=name):
                params = self.PARAMS.clone().requires_grad_()
                loss = components_for(name).likelihood_class.whittle_likelihood_loss_tapered(
                    params, sample_periodogram, n1, n2, p_time,
                    taper, 0.044, 0.063,
                )
                loss.backward()
                torch.testing.assert_close(
                    loss,
                    torch.tensor(golden_loss, dtype=torch.float64),
                    rtol=1e-12,
                    atol=1e-12,
                )
                torch.testing.assert_close(
                    params.grad,
                    torch.tensor(golden_gradient, dtype=torch.float64),
                    rtol=1e-12,
                    atol=1e-12,
                )

    def test_one_step_optimizer_flow_matches_pre_refactor_goldens(self):
        class OneClosureOptimizer:
            def __init__(self, parameters):
                self.parameters = parameters

            def zero_grad(self):
                for parameter in self.parameters:
                    parameter.grad = None

            def step(self, closure):
                return closure()

        n1 = n2 = 2
        p_time = 1
        taper = torch.ones((3, 3), dtype=torch.float64)
        real = torch.tensor(
            (((1.0,), (2.0,)), ((3.0,), (4.0,))), dtype=torch.float64
        )
        j_vector = torch.complex(real, torch.zeros_like(real))
        sample_periodogram = (
            j_vector.unsqueeze(-1) @ j_vector.unsqueeze(-2).conj()
        )
        expected_final_loss = {
            "identity": 3542.621,
            "latitude_difference": 754.515,
            "longitude_difference": 597.287,
            "cross_difference": 240.551,
            "summed_first_differences": 342.873,
        }

        for name, golden_loss in expected_final_loss.items():
            with self.subTest(filter=name):
                parameters = [
                    torch.tensor([value], dtype=torch.float64, requires_grad=True)
                    for value in self.PARAMS.tolist()
                ]
                optimizer = OneClosureOptimizer(parameters)
                # The production method reports progress to stdout.  Suppress
                # that side effect while characterizing its returned state.
                with contextlib.redirect_stdout(io.StringIO()):
                    result = components_for(name).likelihood_class.run_lbfgs_tapered(
                        parameters,
                        optimizer,
                        sample_periodogram,
                        n1,
                        n2,
                        p_time,
                        taper,
                        max_steps=1,
                        device="cpu",
                    )
                self.assertEqual(result[3], golden_loss)
                self.assertEqual(result[4], 1)


class CompatibilityImportTests(unittest.TestCase):
    def test_legacy_modules_keep_public_classes_and_filter_behavior(self):
        for descriptive_name, legacy_suffix in FILTERS.items():
            with self.subTest(filter=descriptive_name):
                module = importlib.import_module(
                    f"GEMS_TCO.debiased_whittle_{legacy_suffix}"
                )
                self.assertEqual(
                    module.debiased_whittle_preprocess.filter_spec.name,
                    descriptive_name,
                )
                self.assertEqual(
                    module.debiased_whittle_likelihood.filter_spec.name,
                    descriptive_name,
                )
                self.assertTrue(hasattr(module, "full_vecc_dw_likelihoods"))
                self.assertTrue(
                    issubclass(
                        module.debiased_whittle_preprocess,
                        module.full_vecc_dw_likelihoods,
                    )
                )


if __name__ == "__main__":
    unittest.main()
