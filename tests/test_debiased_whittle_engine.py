"""Regression and edge-case tests for the Debiased Whittle estimators.

The numerical goldens use valid rectangular-taper autocorrelations and protect
filter stencils, grid shapes, frequency masks, covariance accumulation order,
expected-periodogram calculations, objective values, and gradients.
"""

from __future__ import annotations

import pickle
import unittest

import torch

import GEMS_TCO.debiased_whittle as debiased_whittle
from GEMS_TCO.debiased_whittle import (
    FILTERS,
    DebiasedWhittleEngine,
    MixedFrequencyDebiasedWhittleLikelihood,
    SpatialFilterSpec,
    VectorGradientDebiasedWhittleLikelihood,
    VectorGradientPreprocessor,
    get_filter_spec,
)
from GEMS_TCO.debiased_whittle._core import _optimize_parameters
from GEMS_TCO.debiased_whittle.engine import components_for

FILTER_NAMES = (
    "identity",
    "latitude_difference",
    "longitude_difference",
    "cross_difference",
    "summed_first_differences",
)


def _unsorted_grid() -> torch.Tensor:
    rows = []
    for i, latitude in enumerate((10.0, 11.0, 12.0)):
        for j, longitude in enumerate((100.0, 101.0, 102.0, 103.0)):
            rows.append((latitude, longitude, 10 * i + j, 5.0))
    grid = torch.tensor(rows, dtype=torch.float32)
    order = torch.tensor((11, 0, 5, 8, 3, 6, 1, 10, 7, 4, 9, 2))
    return grid[order]


def _rectangular_taper_autocorrelation(n1: int, n2: int) -> torch.Tensor:
    likelihood = components_for("identity").likelihood_class
    return likelihood.taper_autocorrelation(torch.ones((n1, n2), dtype=torch.float64), n1, n2)


class FilterSpecificationTests(unittest.TestCase):
    def test_public_registry_contains_descriptive_names_only(self):
        self.assertEqual(tuple(FILTERS), FILTER_NAMES)
        for name in FILTER_NAMES:
            with self.subTest(filter=name):
                self.assertEqual(get_filter_spec(name).name, name)

        for removed_alias in ("raw", "lat1", "lon1", "1111", "2110"):
            with self.subTest(removed_alias=removed_alias):
                with self.assertRaises(ValueError):
                    get_filter_spec(removed_alias)

    def test_ad_hoc_filter_specifications_are_rejected(self):
        forged_identity = SpatialFilterSpec(
            name="identity",
            description="not the reviewed identity filter",
            weights=(((0, 0), 2.0),),
            grid_reduction=(0, 0),
            excluded_frequencies="dc",
        )
        with self.assertRaisesRegex(ValueError, "Custom"):
            get_filter_spec(forged_identity)
        with self.assertRaisesRegex(ValueError, "Custom"):
            DebiasedWhittleEngine(forged_identity)

    def test_generated_engine_components_are_pickleable_but_not_reexported(self):
        redundant_exports = {
            "EngineComponents",
            "components_for",
            "IdentityDebiasedWhittlePreprocessor",
            "IdentityDebiasedWhittleLikelihood",
            "LatitudeDifferenceDebiasedWhittlePreprocessor",
            "LatitudeDifferenceDebiasedWhittleLikelihood",
            "LongitudeDifferenceDebiasedWhittlePreprocessor",
            "LongitudeDifferenceDebiasedWhittleLikelihood",
            "CrossDifferenceDebiasedWhittlePreprocessor",
            "CrossDifferenceDebiasedWhittleLikelihood",
            "SummedFirstDifferencesDebiasedWhittlePreprocessor",
            "SummedFirstDifferencesDebiasedWhittleLikelihood",
        }
        self.assertTrue(redundant_exports.isdisjoint(debiased_whittle.__all__))
        for name in redundant_exports:
            self.assertFalse(hasattr(debiased_whittle, name), name)

        for name in FILTER_NAMES:
            with self.subTest(filter=name):
                components = components_for(name)
                for generated_class in (
                    components.preprocess_class,
                    components.likelihood_class,
                ):
                    self.assertIs(pickle.loads(pickle.dumps(generated_class)), generated_class)
                restored = pickle.loads(pickle.dumps(DebiasedWhittleEngine(name)))
                self.assertEqual(restored.filter_spec, components.filter_spec)

    def test_optimizer_supports_explicitly_fixed_parameters(self):
        free = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
        fixed = torch.tensor(-2.0, dtype=torch.float64, requires_grad=False)
        optimizer = torch.optim.LBFGS([free], lr=1.0, max_iter=10)

        result = _optimize_parameters(
            (free, fixed),
            optimizer,
            lambda values: (values[0] - 1.0).square() + values[1].square(),
            max_steps=3,
            gradient_tolerance=1e-8,
        )

        self.assertAlmostEqual(float(fixed), -2.0)
        self.assertAlmostEqual(float(result.parameters[1]), -2.0)
        self.assertAlmostEqual(float(free), 1.0, places=7)

    def test_optimizer_fails_explicitly_when_every_state_is_invalid(self):
        parameter = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)

        class InvalidatingOptimizer:
            def zero_grad(self):
                parameter.grad = None

            def step(self, closure):
                closure()
                with torch.no_grad():
                    parameter.fill_(99.0)

        def invalid_objective(values):
            return values.sum() * 0.0 + float("inf")

        with self.assertRaisesRegex(RuntimeError, "no finite state"):
            _optimize_parameters(
                (parameter,),
                InvalidatingOptimizer(),
                invalid_objective,
                max_steps=1,
                gradient_tolerance=1e-8,
            )
        self.assertEqual(float(parameter), 2.0)

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
    def test_stencils_shapes_dtype_and_identity_demeaning(self):
        grid = _unsorted_grid()
        expected = {
            "identity": (
                (12, 4),
                (11.5, -11.5, -0.5, 8.5, -8.5, 0.5, -10.5, 10.5, 1.5, -1.5, 9.5, -9.5),
            ),
            "latitude_difference": ((8, 4), (10.0,) * 8),
            "longitude_difference": ((9, 4), (1.0,) * 9),
            "cross_difference": ((6, 4), (0.0,) * 6),
            "summed_first_differences": ((6, 4), (11.0,) * 6),
        }

        for name, (shape, values) in expected.items():
            with self.subTest(filter=name):
                preprocessor = DebiasedWhittleEngine(name).make_preprocessor(
                    [grid],
                )
                output = preprocessor.generate_filtered_data(10.0, 12.0, 100.0, 103.0)
                self.assertEqual(tuple(output.shape), shape)
                self.assertEqual(output.dtype, torch.float64)
                torch.testing.assert_close(
                    output[:, 2],
                    torch.tensor(values, dtype=torch.float64),
                    rtol=0.0,
                    atol=0.0,
                )

        # Identity/raw historically preserves input row order while demeaning.
        identity = (
            DebiasedWhittleEngine("identity")
            .make_preprocessor(
                [grid],
            )
            .generate_filtered_data(10.0, 12.0, 100.0, 103.0)
        )
        torch.testing.assert_close(identity[:, :2], grid[:, :2].to(torch.float64))


class PreprocessingValidationTests(unittest.TestCase):
    def test_time_slices_must_be_complete_and_unit_spaced(self):
        grid0 = _unsorted_grid().clone()
        grid0[:, 3] = 0.0
        grid1 = grid0.clone()
        grid1[:, 3] = 1.0
        preprocessor = DebiasedWhittleEngine("identity").make_preprocessor([grid0, grid1])
        self.assertEqual(len(preprocessor.time_slices), 2)

        empty = grid0[:0]
        with self.assertRaisesRegex(ValueError, "time-slice index 1 is empty"):
            DebiasedWhittleEngine("identity").make_preprocessor([grid0, empty])

        gapped = grid0.clone()
        gapped[:, 3] = 2.0
        with self.assertRaisesRegex(ValueError, "unit spacing"):
            DebiasedWhittleEngine("identity").make_preprocessor([grid0, gapped])

        repeated = grid0.clone()
        with self.assertRaisesRegex(ValueError, "unit spacing"):
            DebiasedWhittleEngine("identity").make_preprocessor([grid0, repeated])

    def test_duplicate_coordinate_cannot_hide_a_missing_grid_cell(self):
        invalid_grid = torch.tensor(
            (
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 1.0, 2.0, 0.0),
                (1.0, 0.0, 3.0, 0.0),
                (0.0, 0.0, 4.0, 0.0),
            ),
            dtype=torch.float32,
        )
        preprocessor = DebiasedWhittleEngine("latitude_difference").make_preprocessor(
            [invalid_grid]
        )

        with self.assertRaisesRegex(ValueError, "exactly once"):
            preprocessor.apply_filter(invalid_grid)
        with self.assertRaisesRegex(ValueError, "time-slice index 0"):
            preprocessor.generate_filtered_data(0.0, 1.0, 0.0, 1.0)

    def test_coordinate_rank_sorting_does_not_depend_on_magic_scale_factor(self):
        # The old ``latitude * 1e6 + longitude`` sort key collided for the
        # middle two coordinates in this grid and could swap latitude rows.
        grid = torch.tensor(
            (
                (1.0, 0.0, 5.0, 0.0),
                (0.0, 1_000_000.0, 2.0, 0.0),
                (1.0, 1_000_000.0, 11.0, 0.0),
                (0.0, 0.0, 0.0, 0.0),
            ),
            dtype=torch.float64,
        )
        output = (
            DebiasedWhittleEngine("latitude_difference")
            .make_preprocessor([grid])
            .apply_filter(grid)
        )
        torch.testing.assert_close(
            output[:, 2], torch.tensor((5.0, 9.0), dtype=torch.float64), rtol=0.0, atol=0.0
        )

    def test_one_time_slice_must_have_one_finite_time_value(self):
        grid = torch.tensor(
            (
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 1.0, 2.0, 1.0),
            ),
            dtype=torch.float64,
        )
        with self.assertRaisesRegex(ValueError, "one finite time"):
            DebiasedWhittleEngine("longitude_difference").make_preprocessor([grid])

    def test_spatial_coordinates_must_be_regularly_spaced(self):
        irregular = torch.tensor(
            (
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 1.0, 2.0, 0.0),
                (0.0, 3.0, 3.0, 0.0),
                (1.0, 0.0, 4.0, 0.0),
                (1.0, 1.0, 5.0, 0.0),
                (1.0, 3.0, 6.0, 0.0),
            ),
            dtype=torch.float64,
        )
        preprocessor = DebiasedWhittleEngine("longitude_difference").make_preprocessor([irregular])
        with self.assertRaisesRegex(ValueError, "regular grid"):
            preprocessor.apply_filter(irregular)

    def test_empty_filtered_outputs_preserve_device_and_use_float64(self):
        one_cell = torch.tensor(((0.0, 0.0, 1.0, 0.0),), dtype=torch.float32)
        scalar = DebiasedWhittleEngine("cross_difference").make_preprocessor([one_cell])
        scalar_output = scalar.apply_filter(one_cell)
        self.assertEqual(tuple(scalar_output.shape), (0, 4))
        self.assertEqual(scalar_output.dtype, torch.float64)
        self.assertEqual(scalar_output.device, one_cell.device)

        vector = VectorGradientPreprocessor([one_cell])
        latitude, longitude = vector.apply_gradient_filter(one_cell)
        for output in (latitude, longitude):
            self.assertEqual(tuple(output.shape), (0, 4))
            self.assertEqual(output.dtype, torch.float64)
            self.assertEqual(output.device, one_cell.device)


class FourierTransformValidationTests(unittest.TestCase):
    def test_scalar_transform_rejects_missing_cells(self):
        grid = torch.tensor(
            (
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 1.0, float("nan"), 0.0),
                (1.0, 0.0, 3.0, 0.0),
                (1.0, 1.0, 4.0, 0.0),
            ),
            dtype=torch.float64,
        )
        likelihood = components_for("identity").likelihood_class
        with self.assertRaisesRegex(ValueError, "one finite observation"):
            likelihood.tapered_fourier_transform(
                [grid], likelihood.hamming_taper, 0, 1, 2, torch.device("cpu")
            )

    def test_zero_norm_taper_is_rejected(self):
        likelihood = components_for("identity").likelihood_class
        with self.assertRaisesRegex(ValueError, "near-zero"):
            likelihood.taper_autocorrelation(torch.zeros((2, 2)), 2, 2)


class LikelihoodCharacterizationTests(unittest.TestCase):
    PARAMS = torch.tensor((0.1, -0.2, 0.3, -0.4, 0.05, -0.07, -2.0), dtype=torch.float64)

    def test_covariance_regression_goldens(self):
        u1 = torch.tensor(((0.0, 0.044), (-0.044, 0.088)), dtype=torch.float64)
        u2 = torch.tensor(0.063, dtype=torch.float64)
        time_lag = torch.tensor(1.0, dtype=torch.float64)
        expected = {
            "identity": (
                0.6833347595297052,
                0.6844559862008044,
                0.6804722928466228,
                0.6838147292267783,
            ),
            "latitude_difference": (
                0.0017412400119832805,
                0.0017624836451252124,
                0.0016878676304017404,
                0.001750310407756639,
            ),
            "longitude_difference": (
                0.002550310094469377,
                0.0025602009750756993,
                0.0025252117408033348,
                0.0025545399966631077,
            ),
            "cross_difference": (
                1.520747305971959e-05,
                1.5551859018914094e-05,
                1.4351781347898829e-05,
                1.5354252702715065e-05,
            ),
            "summed_first_differences": (
                0.0043793388758650575,
                0.004340157915358911,
                0.0043680397806784566,
                0.004251383603456338,
            ),
        }
        for name, golden in expected.items():
            with self.subTest(filter=name):
                actual = components_for(name).likelihood_class.filtered_covariance(
                    u1, u2, time_lag, self.PARAMS, 0.044, 0.063
                )
                torch.testing.assert_close(
                    actual,
                    torch.tensor(golden, dtype=torch.float64).reshape(2, 2),
                    rtol=0.0,
                    atol=0.0,
                )

    def test_zero_lag_covariance_has_no_distance_stabilizer_bias(self):
        likelihood = components_for("identity").likelihood_class
        actual = likelihood.spatiotemporal_covariance(0.0, 0.0, 0.0, self.PARAMS)
        expected = torch.exp(self.PARAMS[0]) / torch.exp(self.PARAMS[1]) + torch.exp(self.PARAMS[6])
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

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
                actual_sum, actual_count = components_for(
                    name
                ).likelihood_class._retained_frequency_sum(terms, 3, 4)
                self.assertEqual(actual_sum.item(), golden_sum)
                self.assertEqual(actual_count, golden_count)

    def test_nonfinite_excluded_frequencies_do_not_poison_the_sum(self):
        expected = {
            "identity": (66.0, 11, ((0, 0),)),
            "latitude_difference": (60.0, 8, tuple((0, j) for j in range(4))),
            "longitude_difference": (54.0, 9, tuple((i, 0) for i in range(3))),
            "cross_difference": (
                48.0,
                6,
                tuple((0, j) for j in range(4)) + tuple((i, 0) for i in range(1, 3)),
            ),
            "summed_first_differences": (66.0, 11, ((0, 0),)),
        }
        for name, (golden_sum, golden_count, excluded) in expected.items():
            with self.subTest(filter=name):
                terms = torch.arange(12, dtype=torch.float64).reshape(3, 4)
                for index in excluded:
                    terms[index] = float("nan")
                actual_sum, actual_count = components_for(
                    name
                ).likelihood_class._retained_frequency_sum(terms, 3, 4)
                self.assertEqual(actual_sum.item(), golden_sum)
                self.assertEqual(actual_count, golden_count)

    def test_out_of_support_taper_lags_are_zero(self):
        taper = _rectangular_taper_autocorrelation(3, 4)
        likelihood = components_for("identity").likelihood_class
        scalar_outside = likelihood._tapered_covariance(
            torch.tensor(-3.0),
            torch.tensor(0.0),
            torch.tensor(0.0),
            self.PARAMS,
            3,
            4,
            taper,
            0.044,
            0.063,
        )
        mixed_outside = MixedFrequencyDebiasedWhittleLikelihood._cn_bar(
            MixedFrequencyDebiasedWhittleLikelihood.identity_covariance,
            torch.tensor(-3.0),
            torch.tensor(0.0),
            torch.tensor(0.0),
            self.PARAMS,
            3,
            4,
            taper,
            0.044,
            0.063,
        )
        vector_outside = VectorGradientDebiasedWhittleLikelihood._tapered_gradient_covariance(
            torch.tensor(-3.0),
            torch.tensor(0.0),
            torch.tensor(0.0),
            self.PARAMS,
            3,
            4,
            taper,
            0.044,
            0.063,
            0,
            0,
            0,
            0,
        )
        for outside in (scalar_outside, mixed_outside, vector_outside):
            self.assertEqual(outside.item(), 0.0)

    def test_indefinite_expected_spectrum_returns_infinite_loss(self):
        base = components_for("identity").likelihood_class

        class IndefiniteSpectrumLikelihood(base):
            @classmethod
            def expected_periodogram(
                cls, params, n1, n2, p_time, taper_autocorr_grid, delta1, delta2
            ):
                matrix = -torch.eye(p_time, dtype=torch.complex128, device=params.device)
                return matrix.expand(n1, n2, -1, -1).clone()

        sample = torch.eye(2, dtype=torch.complex128).expand(2, 2, -1, -1).clone()
        loss = IndefiniteSpectrumLikelihood.negative_log_likelihood(
            self.PARAMS, sample, 2, 2, 2, _rectangular_taper_autocorrelation(2, 2), 0.044, 0.063
        )
        self.assertTrue(torch.isinf(loss))

    def test_nonfinite_sample_is_ignored_only_at_an_excluded_frequency(self):
        base = components_for("identity").likelihood_class

        class FixedSpectrumLikelihood(base):
            @classmethod
            def expected_periodogram(
                cls, params, n1, n2, p_time, taper_autocorr_grid, delta1, delta2
            ):
                matrix = torch.eye(p_time, dtype=torch.complex128, device=params.device)
                return matrix.expand(n1, n2, -1, -1).clone()

        sample = torch.eye(2, dtype=torch.complex128).expand(2, 2, -1, -1).clone()
        sample[0, 0] = torch.full((2, 2), complex(float("nan"), 0.0))
        excluded_loss = FixedSpectrumLikelihood.negative_log_likelihood(
            self.PARAMS, sample, 2, 2, 2, _rectangular_taper_autocorrelation(2, 2), 0.044, 0.063
        )
        self.assertTrue(torch.isfinite(excluded_loss))

        sample[0, 1] = torch.full((2, 2), complex(float("nan"), 0.0))
        retained_loss = FixedSpectrumLikelihood.negative_log_likelihood(
            self.PARAMS, sample, 2, 2, 2, _rectangular_taper_autocorrelation(2, 2), 0.044, 0.063
        )
        self.assertTrue(torch.isinf(retained_loss))

    def test_complex64_periodogram_is_promoted_consistently(self):
        n1, n2, p_time = 2, 2, 2
        taper = _rectangular_taper_autocorrelation(n1, n2)
        real = torch.arange(1, 9, dtype=torch.float64).reshape(n1, n2, p_time) / 10.0
        periodogram = (
            torch.complex(real, torch.flip(real, (0,)) / 7.0).unsqueeze(-1)
            @ torch.complex(real, torch.flip(real, (0,)) / 7.0).unsqueeze(-2).conj()
        )
        likelihood = components_for("identity").likelihood_class
        loss128 = likelihood.negative_log_likelihood(
            self.PARAMS, periodogram, n1, n2, p_time, taper, 0.044, 0.063
        )
        loss64 = likelihood.negative_log_likelihood(
            self.PARAMS, periodogram.to(torch.complex64), n1, n2, p_time, taper, 0.044, 0.063
        )
        torch.testing.assert_close(loss64, loss128, rtol=1e-6, atol=1e-6)

    def test_adaptive_jitter_is_included_in_the_autograd_objective(self):
        n1 = n2 = p_time = 2
        taper = _rectangular_taper_autocorrelation(n1, n2)
        real = torch.arange(1, 9, dtype=torch.float64).reshape(n1, n2, p_time) / 10.0
        fourier = torch.complex(real, torch.flip(real, (0,)) / 7.0)
        periodogram = fourier.unsqueeze(-1) @ fourier.unsqueeze(-2).conj()
        likelihood = components_for("identity").likelihood_class

        params = self.PARAMS.clone().requires_grad_()
        loss = likelihood.negative_log_likelihood(
            params, periodogram, n1, n2, p_time, taper, 0.044, 0.063
        )
        loss.backward()

        step = 1e-6
        plus = self.PARAMS.clone()
        minus = self.PARAMS.clone()
        plus[1] += step
        minus[1] -= step
        finite_difference = (
            likelihood.negative_log_likelihood(
                plus, periodogram, n1, n2, p_time, taper, 0.044, 0.063
            )
            - likelihood.negative_log_likelihood(
                minus, periodogram, n1, n2, p_time, taper, 0.044, 0.063
            )
        ) / (2 * step)
        torch.testing.assert_close(params.grad[1], finite_difference, rtol=5e-7, atol=5e-7)

    def test_fit_evaluates_and_restores_the_post_step_best_state(self):
        base = components_for("identity").likelihood_class

        class QuadraticLikelihood(base):
            @classmethod
            def negative_log_likelihood(cls, params, *args, **kwargs):
                return (params[0] - 3.0).square()

        parameter = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)

        class PostClosureUpdateOptimizer:
            def zero_grad(self):
                parameter.grad = None

            def step(self, closure):
                pre_step_loss = closure()
                with torch.no_grad():
                    parameter.fill_(3.0)
                return pre_step_loss

        result = QuadraticLikelihood.fit(
            [parameter],
            PostClosureUpdateOptimizer(),
            torch.ones((1, 1, 1, 1), dtype=torch.complex128),
            1,
            1,
            1,
            torch.ones((1, 1), dtype=torch.float64),
            max_steps=1,
        )
        self.assertEqual(result.loss, 0.0)
        self.assertEqual(result.parameters.item(), 3.0)
        self.assertEqual(parameter.item(), 3.0)

    def test_loss_and_gradient_regression_goldens(self):
        n1, n2, p_time = 3, 4, 2
        taper = _rectangular_taper_autocorrelation(n1, n2)
        real = (
            torch.arange(1, n1 * n2 * p_time + 1, dtype=torch.float64).reshape(n1, n2, p_time)
            / 10.0
        )
        imag = torch.flip(real, (0,)) / 7.0
        j_vector = torch.complex(real, imag)
        sample_periodogram = j_vector.unsqueeze(-1) @ j_vector.unsqueeze(-2).conj()

        goldens = {
            "identity": (
                881.4968495839253,
                (
                    -268.19192028872646,
                    6.704778689457271,
                    -94.73391463689815,
                    2.7671323698132255,
                    1.4068291407860554,
                    -0.9030652199897986,
                    -621.5232275381996,
                ),
            ),
            "latitude_difference": (
                489.0895369020227,
                (
                    -113.0299244902204,
                    0.1523128131640874,
                    -51.15592518597771,
                    0.03151397724852956,
                    0.03231254457044086,
                    -0.007006080857546948,
                    -382.90781166628796,
                ),
            ),
            "longitude_difference": (
                428.56759012612025,
                (
                    -106.60520054494503,
                    0.15542523016742393,
                    -22.0342911070449,
                    0.02596293319442111,
                    0.006685194637000624,
                    -0.02767522858644611,
                    -328.8495608336917,
                ),
            ),
            "cross_difference": (
                237.16909429971383,
                (
                    -49.67392474588618,
                    0.02976900630698509,
                    -16.698486465360595,
                    6.378865040801429e-05,
                    6.598964180831723e-05,
                    -6.878792081677076e-05,
                    -192.67427962481227,
                ),
            ),
            "summed_first_differences": (
                142.75325068088998,
                (
                    -36.76806127410998,
                    0.07977192311801673,
                    -13.843649794017356,
                    0.026618651341379973,
                    0.00786490664694528,
                    -0.008288508582518617,
                    -111.14609826723031,
                ),
            ),
        }

        for name, (golden_loss, golden_gradient) in goldens.items():
            with self.subTest(filter=name):
                params = self.PARAMS.clone().requires_grad_()
                loss = components_for(name).likelihood_class.negative_log_likelihood(
                    params,
                    sample_periodogram,
                    n1,
                    n2,
                    p_time,
                    taper,
                    0.044,
                    0.063,
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

    def test_one_step_optimizer_regression(self):
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
        taper = _rectangular_taper_autocorrelation(n1, n2)
        real = torch.tensor((((1.0,), (2.0,)), ((3.0,), (4.0,))), dtype=torch.float64)
        j_vector = torch.complex(real, torch.zeros_like(real))
        sample_periodogram = j_vector.unsqueeze(-1) @ j_vector.unsqueeze(-2).conj()
        expected_final_loss = {
            "identity": 2009.994,
            "latitude_difference": 956.377,
            "longitude_difference": 754.674,
            "cross_difference": 419.875,
            "summed_first_differences": 288.923,
        }

        for name, golden_loss in expected_final_loss.items():
            with self.subTest(filter=name):
                parameters = [
                    torch.tensor([value], dtype=torch.float64, requires_grad=True)
                    for value in self.PARAMS.tolist()
                ]
                optimizer = OneClosureOptimizer(parameters)
                result = components_for(name).likelihood_class.fit(
                    parameters,
                    optimizer,
                    sample_periodogram,
                    n1,
                    n2,
                    p_time,
                    taper,
                    max_steps=1,
                )
                self.assertEqual(round(result.loss, 3), golden_loss)
                self.assertEqual(result.steps, 1)


class DistinctEstimatorRegressionTests(unittest.TestCase):
    PARAMS = torch.tensor((0.1, -0.2, 0.3, -0.4, 0.05, -0.07, -2.0), dtype=torch.float64)

    @staticmethod
    def _periodogram(real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        j_vector = torch.complex(real, imag)
        return j_vector.unsqueeze(-1) @ j_vector.unsqueeze(-2).conj()

    def test_vector_gradient_preprocessing(self):
        grid = _unsorted_grid()
        preprocessor = VectorGradientPreprocessor([grid])
        latitude, longitude = preprocessor.apply_gradient_filter(grid)

        self.assertEqual(tuple(latitude.shape), (6, 4))
        self.assertEqual(tuple(longitude.shape), (6, 4))
        self.assertEqual(latitude.dtype, torch.float64)
        self.assertEqual(longitude.dtype, torch.float64)
        torch.testing.assert_close(
            latitude[:, 2],
            torch.full((6,), 10.0, dtype=torch.float64),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            longitude[:, 2],
            torch.ones(6, dtype=torch.float64),
            rtol=0.0,
            atol=0.0,
        )

    def test_mixed_frequency_loss_and_gradient(self):
        n1, n2, n1_diff, n2_diff, p_time = 3, 4, 2, 3, 2
        taper_raw = _rectangular_taper_autocorrelation(n1, n2)
        taper_diff = _rectangular_taper_autocorrelation(n1_diff, n2_diff)

        raw_real = (
            torch.arange(1, n1 * n2 * p_time + 1, dtype=torch.float64).reshape(n1, n2, p_time)
            / 10.0
        )
        raw_imag = torch.flip(raw_real, (0,)) / 7.0
        raw_periodogram = self._periodogram(raw_real, raw_imag)

        diff_real = (
            torch.arange(1, n1_diff * n2_diff * p_time + 1, dtype=torch.float64).reshape(
                n1_diff, n2_diff, p_time
            )
            / 8.0
        )
        diff_imag = torch.flip(diff_real, (1,)) / 9.0
        diff_periodogram = self._periodogram(diff_real, diff_imag)

        params = self.PARAMS.clone().requires_grad_()
        loss = MixedFrequencyDebiasedWhittleLikelihood.negative_log_likelihood(
            params,
            raw_periodogram,
            diff_periodogram,
            n1,
            n2,
            n1_diff,
            n2_diff,
            p_time,
            taper_raw,
            taper_diff,
            0,
            0,
            0.044,
            0.063,
        )
        loss.backward()

        torch.testing.assert_close(
            loss,
            torch.tensor(216.56601202179016, dtype=torch.float64),
            rtol=1e-12,
            atol=1e-12,
        )
        torch.testing.assert_close(
            params.grad,
            torch.tensor(
                (
                    -111.43737269240233,
                    9.732312954476981,
                    -54.70953912082037,
                    4.080267606843293,
                    2.0482318340941044,
                    -1.2521798456313267,
                    -111.91805542800046,
                ),
                dtype=torch.float64,
            ),
            rtol=1e-12,
            atol=1e-12,
        )
        promoted_loss = MixedFrequencyDebiasedWhittleLikelihood.negative_log_likelihood(
            self.PARAMS,
            raw_periodogram.to(torch.complex64),
            diff_periodogram.to(torch.complex64),
            n1,
            n2,
            n1_diff,
            n2_diff,
            p_time,
            taper_raw,
            taper_diff,
            0,
            0,
            0.044,
            0.063,
        )
        torch.testing.assert_close(promoted_loss, loss.detach(), rtol=1e-6, atol=1e-6)

    def test_vector_gradient_loss_and_gradient(self):
        n1, n2, matrix_size = 2, 3, 4
        taper = (
            _rectangular_taper_autocorrelation(n1, n2)
            .expand(matrix_size, matrix_size, -1, -1)
            .clone()
        )
        real = (
            torch.arange(1, n1 * n2 * matrix_size + 1, dtype=torch.float64).reshape(
                n1, n2, matrix_size
            )
            / 10.0
        )
        imag = torch.flip(real, (0,)) / 7.0
        periodogram = self._periodogram(real, imag)

        params = self.PARAMS.clone().requires_grad_()
        loss = VectorGradientDebiasedWhittleLikelihood.negative_log_likelihood(
            params,
            periodogram,
            n1,
            n2,
            matrix_size,
            taper,
            0.044,
            0.063,
        )
        loss.backward()

        torch.testing.assert_close(
            loss,
            torch.tensor(830.2673434489352, dtype=torch.float64),
            rtol=1e-12,
            atol=1e-12,
        )
        torch.testing.assert_close(
            params.grad,
            torch.tensor(
                (
                    -202.48489897140706,
                    0.39484631088157585,
                    -62.49969362638345,
                    0.11992664932429165,
                    0.053413812261931426,
                    -0.06584286969889419,
                    -643.425806619254,
                ),
                dtype=torch.float64,
            ),
            # Complex batched solves differ by a few ulps across BLAS builds.
            rtol=1e-11,
            atol=1e-9,
        )
        promoted_loss = VectorGradientDebiasedWhittleLikelihood.negative_log_likelihood(
            self.PARAMS,
            periodogram.to(torch.complex64),
            n1,
            n2,
            matrix_size,
            taper,
            0.044,
            0.063,
        )
        torch.testing.assert_close(promoted_loss, loss.detach(), rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
