"""Focused statistical and numerical regression tests for ``GEMS_TCO.spatial``."""

from __future__ import annotations

import math
import unittest
from unittest import mock

import numpy as np
import torch

import GEMS_TCO.spatial as spatial
from GEMS_TCO.spatial import matern_bessel
from GEMS_TCO.spatial.base import _PureSpaceVecchiaBase


def _four_column_replicate(values: tuple[float, float]) -> torch.Tensor:
    return torch.tensor(
        [
            [0.0, 0.0, values[0], 0.0],
            [1.0, 0.0, values[1], 0.0],
        ],
        dtype=torch.float64,
    )


class SpatialVecchiaCoreTests(unittest.TestCase):
    def test_latlon_mean_design_matches_direct_full_likelihood(self) -> None:
        coords = np.array([[1.0, 10.0], [2.0, 14.0], [4.0, 20.0]], dtype=np.float64)
        hour_indicators = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        rows = torch.tensor(
            np.column_stack([coords, [1.0, 2.0, 3.0], np.zeros(3), hour_indicators]),
            dtype=torch.float64,
        )

        latlon = spatial.IsotropicMaternSpatialVecchia(
            smooth=0.5,
            input_map={"replicate": rows[:, :4]},
            block_shape=(1, 1),
            mean_design="latlon",
        )
        latlon._make_full_data(0)
        cluster_design = latlon._design_from_rows(rows[:, :4]).numpy()
        direct_design = matern_bessel.make_mean_design(coords, "latlon")
        np.testing.assert_allclose(cluster_design, direct_design, rtol=0.0, atol=0.0)
        self.assertEqual(latlon.n_features, 3)

        latlon_hour = spatial.IsotropicMaternSpatialVecchia(
            smooth=0.5,
            input_map={"replicate": rows},
            block_shape=(1, 1),
            mean_design="latlon_hour",
        )
        latlon_hour._make_full_data(0)
        design_with_hours = latlon_hour._design_from_rows(rows).numpy()
        self.assertEqual(design_with_hours.shape, (3, 10))
        np.testing.assert_allclose(design_with_hours[:, :3], direct_design, rtol=0.0, atol=0.0)
        np.testing.assert_array_equal(design_with_hours[:, 3:], hour_indicators)

    def test_grouped_block_engine_precomputes_and_requires_a_shared_grid(self) -> None:
        first = _four_column_replicate((1.0, 2.0))
        model = spatial.IsotropicMaternSpatialVecchia(
            smooth=0.7,
            input_map={"replicate": first},
            block_shape=(1, 1),
            n_neighbor_blocks=1,
            mean_design="lat",
        )
        model.precompute_conditioning_sets()
        raw = torch.tensor([0.0, 0.0, -2.0], dtype=torch.float64, requires_grad=True)
        loss = model.profiled_negative_log_likelihood(raw)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(raw.grad).all())
        self.assertEqual(model.cluster_summary()["n_target_points"], 2)
        coefficients = model.estimate_gls_coefficients([value.detach() for value in raw.unbind()])
        self.assertEqual(tuple(coefficients.shape), (2, 1))
        self.assertTrue(torch.isfinite(coefficients).all())

        shifted = first.clone()
        shifted[1, 0] += 0.1
        mismatched = spatial.IsotropicMaternSpatialVecchia(
            smooth=0.7,
            input_map={"first": first, "shifted": shifted},
            block_shape=(1, 1),
            n_neighbor_blocks=1,
            mean_design="lat",
        )
        with self.assertRaisesRegex(ValueError, "same coordinates"):
            mismatched.precompute_conditioning_sets()

        with self.assertRaisesRegex(ValueError, "expected 3 covariance parameters"):
            model.profiled_negative_log_likelihood(torch.zeros(4, dtype=torch.float64))
        with self.assertRaisesRegex(ValueError, "torch.float64"):
            model.profiled_negative_log_likelihood(torch.zeros(3, dtype=torch.float32))

    def test_cluster_geometry_validation_is_explicit(self) -> None:
        first = _four_column_replicate((float("nan"), 2.0))
        first[0, :2] = float("nan")
        grid = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
        model = spatial.IsotropicMaternSpatialVecchia(
            smooth=0.7,
            input_map={"replicate": first},
            grid_coords=grid,
            block_shape=(1, 1),
            n_neighbor_blocks=1,
            mean_design="lat",
        )
        model.precompute_conditioning_sets()
        self.assertEqual(model.cluster_summary()["n_target_points"], 1)

        with self.assertRaisesRegex(ValueError, "exactly two"):
            spatial.IsotropicMaternSpatialVecchia(
                smooth=0.7,
                input_map={"replicate": first},
                grid_coords=grid,
                block_shape=(1,),
                mean_design="lat",
            )

        extra_grid = np.vstack([grid, [2.0, 0.0]])
        extra = spatial.IsotropicMaternSpatialVecchia(
            smooth=0.7,
            input_map={"replicate": first},
            grid_coords=extra_grid,
            block_shape=(1, 1),
            mean_design="lat",
        )
        with self.assertRaisesRegex(ValueError, r"shape \(2, 2\)"):
            extra.precompute_conditioning_sets()

    def test_zero_distance_covariances_are_exact_and_have_finite_gradients(self) -> None:
        data = {"replicate": _four_column_replicate((1.0, 2.0))}
        models_and_raw = (
            (
                spatial.IsotropicMaternSpatialVecchia(
                    smooth=0.7, input_map=data, block_shape=(1, 1), mean_design="lat"
                ),
                [0.0, 0.0, -2.0],
            ),
            (
                spatial.AnisotropicMaternSpatialVecchia(
                    smooth=0.7, input_map=data, block_shape=(1, 1), mean_design="lat"
                ),
                [0.0, 0.0, 0.0, -2.0],
            ),
            (
                spatial.NoNuggetAnisotropicCauchySpatialVecchia(
                    input_map=data, block_shape=(1, 1), mean_design="lat"
                ),
                [0.0, 0.0, 0.0, 0.0],
            ),
        )
        zero = torch.zeros(1, dtype=torch.float64)
        for model, raw_values in models_and_raw:
            with self.subTest(model=type(model).__name__):
                raw = torch.tensor(raw_values, dtype=torch.float64, requires_grad=True)
                covariance = model._cov_from_deltas(zero, zero, raw)
                self.assertEqual(float(covariance), 1.0)
                covariance.sum().backward()
                self.assertTrue(torch.isfinite(raw.grad).all())

    def test_cluster_matern_uses_the_standard_range_convention(self) -> None:
        distances = torch.tensor([0.0, 0.2, 1.0, 5.0], dtype=torch.float64)
        data = {"replicate": _four_column_replicate((1.0, 2.0))}
        for smooth in (0.5, 0.7, 1.5):
            with self.subTest(smooth=smooth):
                model = spatial.IsotropicMaternSpatialVecchia(
                    smooth=smooth,
                    input_map=data,
                    block_shape=(1, 1),
                    mean_design="lat",
                )
                actual = model._matern_corr(distances).numpy()
                expected = matern_bessel.matern_corr_bessel(distances.numpy(), smooth)
                np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-7)

        exponential = spatial.IsotropicMaternSpatialVecchia(
            smooth=0.5,
            input_map=data,
            block_shape=(1, 1),
            mean_design="lat",
        )
        np.testing.assert_allclose(
            exponential._matern_corr(distances).numpy(),
            np.exp(-distances.numpy()),
            rtol=0.0,
            atol=1e-15,
        )
        self.assertAlmostEqual(
            float(exponential._matern_corr(torch.tensor([21.0], dtype=torch.float64))),
            math.exp(-21.0),
            places=20,
        )

        spline = spatial.IsotropicMaternSpatialVecchia(
            smooth=0.7,
            input_map=data,
            block_shape=(1, 1),
            mean_design="lat",
        )
        self.assertEqual(
            float(spline._matern_corr(torch.tensor([21.0], dtype=torch.float64))),
            0.0,
        )

    def test_cluster_covariance_matches_direct_bessel_full_likelihood(self) -> None:
        coords_np = np.array([[0.0, 0.0], [0.4, 0.1], [1.0, 0.7]], dtype=np.float64)
        coords_torch = torch.tensor(coords_np, dtype=torch.float64).unsqueeze(0)
        data = {
            "replicate": torch.tensor(
                np.column_stack([coords_np, [1.0, 2.0, 3.0], np.zeros((3, 8))]),
                dtype=torch.float64,
            )
        }
        signal_variance = 1.3
        range_lat = 0.8
        range_lon = 1.1
        nugget = 0.2
        d_lat, d_lon = matern_bessel.pairwise_deltas(coords_np)

        for smooth in (0.5, 0.7, 1.5):
            with self.subTest(smooth=smooth):
                model = spatial.AnisotropicMaternSpatialVecchia(
                    smooth=smooth,
                    input_map=data,
                    block_shape=(1, 1),
                    mean_design="lat",
                )
                raw = torch.tensor(
                    [
                        math.log(signal_variance),
                        math.log(range_lat),
                        math.log(range_lon),
                        math.log(nugget),
                    ],
                    dtype=torch.float64,
                )
                cluster_covariance = model._cov_full(coords_torch, raw).squeeze(0).numpy()
                direct_parameters = matern_bessel.MaternParameters(
                    signal_variance=signal_variance,
                    range_lat=range_lat,
                    range_lon=range_lon,
                    smooth=smooth,
                    nugget=nugget,
                    phi1=signal_variance / range_lon,
                    phi2=1.0 / range_lon,
                    phi3=(range_lon / range_lat) ** 2,
                )
                direct_covariance = matern_bessel.covariance_from_deltas(
                    d_lat, d_lon, direct_parameters, jitter=1e-6
                )
                np.testing.assert_allclose(
                    cluster_covariance,
                    direct_covariance,
                    rtol=2e-5,
                    atol=2e-7,
                )

    def test_invalid_generalized_cauchy_shape_fails_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, r"\(0, 2\]"):
            spatial.NoNuggetAnisotropicCauchySpatialVecchia(
                input_map={"replicate": _four_column_replicate((1.0, 2.0))},
                gc_alpha=2.1,
            )
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            spatial.cauchy_phi_init_from_natural(0.0, 1.0, 1.0, 1.0)

    def test_transformed_covariance_parameters_must_remain_finite_and_positive(self) -> None:
        data = {"replicate": _four_column_replicate((1.0, 2.0))}
        isotropic = spatial.IsotropicMaternSpatialVecchia(
            smooth=0.5,
            input_map=data,
            block_shape=(1, 1),
            n_neighbor_blocks=1,
            mean_design="lat",
        )
        isotropic.precompute_conditioning_sets()
        underflow = torch.tensor([-1000.0, 0.0, -2.0], dtype=torch.float64, requires_grad=True)
        loss = isotropic.profiled_negative_log_likelihood(underflow)
        self.assertEqual(float(loss.detach()), 1.0e10)
        loss.backward()
        self.assertTrue(torch.isfinite(underflow.grad).all())
        with self.assertRaisesRegex(ValueError, "transform to finite natural"):
            isotropic._convert_params(underflow.detach().tolist())
        with self.assertRaisesRegex(ValueError, "transform to finite natural"):
            isotropic.estimate_gls_coefficients([value.detach() for value in underflow.unbind()])

        cauchy = spatial.NoNuggetAnisotropicCauchySpatialVecchia(
            input_map=data,
            block_shape=(1, 1),
            n_neighbor_blocks=1,
            mean_design="lat",
        )
        cauchy.precompute_conditioning_sets()
        zero_phi3 = torch.tensor(
            [0.0, 0.0, -1000.0, 0.0],
            dtype=torch.float64,
            requires_grad=True,
        )
        cauchy_loss = cauchy.profiled_negative_log_likelihood(zero_phi3)
        self.assertEqual(float(cauchy_loss.detach()), 1.0e10)
        with self.assertRaisesRegex(ValueError, "transform to finite natural"):
            cauchy._convert_params(zero_phi3.detach().tolist())

        with self.assertRaisesRegex(ValueError, "reparameterization"):
            spatial.cauchy_phi_init_from_natural(1.0e308, 1.0, 1.0e-308, 1.0)

    def test_lbfgs_returns_structured_best_valid_state(self) -> None:
        class QuadraticModel(_PureSpaceVecchiaBase):
            _n_covariance_parameters = 1

            def profiled_negative_log_likelihood(self, params: torch.Tensor) -> torch.Tensor:
                return (params[0] - 2.0).pow(2)

            def _convert_params(self, raw):
                return {"parameter": float(raw[0])}

        data = torch.zeros((1, 11), dtype=torch.float64)
        model = QuadraticModel(smooth=0.5, input_map={"replicate": data})
        model.is_precomputed = True
        parameter = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        optimizer = model.make_lbfgs_optimizer([parameter], max_iter=8, tolerance_grad=1e-12)
        result = model.fit_lbfgs([parameter], optimizer, max_steps=1, grad_tol=0.0)
        self.assertIsInstance(result, spatial.SpatialLBFGSFitResult)
        self.assertEqual(result.steps_completed, 1)
        self.assertTrue(result.valid)
        self.assertAlmostEqual(
            result.final_nll,
            float((parameter.detach() - 2.0).pow(2)),
            places=14,
        )
        self.assertEqual(result.raw_parameters, (float(parameter.detach()),))
        self.assertEqual(
            result.interpretable_parameters,
            {"parameter": float(parameter.detach())},
        )
        self.assertFalse(hasattr(model, "fit_vecc_lbfgs"))
        self.assertFalse(hasattr(model, "vecchia_batched_likelihood"))
        self.assertFalse(hasattr(model, "get_gls_beta"))
        self.assertFalse(hasattr(model, "set_optimizer"))

    def test_lbfgs_rejects_flat_invalid_start_and_restores_valid_best(self) -> None:
        class BoundedQuadraticModel(_PureSpaceVecchiaBase):
            _n_covariance_parameters = 1

            def profiled_negative_log_likelihood(self, params: torch.Tensor) -> torch.Tensor:
                if bool((params[0].abs() > 10.0).item()):
                    return params.sum() * 0.0 + 1.0e10
                return (params[0] - 2.0).pow(2)

            def _convert_params(self, raw):
                return {"parameter": float(raw[0])}

        class InvalidatingLBFGS(torch.optim.LBFGS):
            def step(self, closure):
                closure()
                with torch.no_grad():
                    self.param_groups[0]["params"][0].fill_(20.0)
                return closure()

        data = torch.zeros((1, 11), dtype=torch.float64)
        model = BoundedQuadraticModel(smooth=0.5, input_map={"replicate": data})
        model.is_precomputed = True

        invalid_start = torch.tensor(20.0, dtype=torch.float64, requires_grad=True)
        invalid_optimizer = model.make_lbfgs_optimizer([invalid_start], max_iter=1)
        with self.assertRaisesRegex(RuntimeError, "invalid likelihood"):
            model.fit_lbfgs([invalid_start], invalid_optimizer, max_steps=1)

        class NonFiniteGradientModel(BoundedQuadraticModel):
            def profiled_negative_log_likelihood(self, params: torch.Tensor) -> torch.Tensor:
                return torch.sqrt(params[0])

        gradient_model = NonFiniteGradientModel(smooth=0.5, input_map={"replicate": data})
        gradient_model.is_precomputed = True
        boundary = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        boundary_optimizer = gradient_model.make_lbfgs_optimizer([boundary], max_iter=1)
        with self.assertRaisesRegex(RuntimeError, "non-finite likelihood gradients"):
            gradient_model.fit_lbfgs([boundary], boundary_optimizer, max_steps=1)

        parameter = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        optimizer = InvalidatingLBFGS([parameter], max_iter=1)
        result = model.fit_lbfgs([parameter], optimizer, max_steps=1, grad_tol=1e-8)
        self.assertEqual(float(parameter.detach()), 0.0)
        self.assertEqual(result.best_step, 0)
        self.assertEqual(result.final_nll, 4.0)
        self.assertTrue(result.valid)
        self.assertFalse(result.converged)
        self.assertEqual(
            result.message,
            "invalid_post_step_best_valid_state_restored",
        )


class DirectBesselLikelihoodTests(unittest.TestCase):
    def setUp(self) -> None:
        self.coords = np.array([[0.0, 0.0], [0.5, 0.1], [1.0, 0.4], [1.5, 0.9]], dtype=np.float64)
        self.y = np.array([0.2, -0.1, 0.5, 0.7], dtype=np.float64)
        self.smooth_bounds = (0.05, 2.5)
        self.raw = matern_bessel.raw_from_natural(
            signal_variance=1.3,
            range_lat=0.8,
            range_lon=1.1,
            smooth=0.7,
            nugget=0.2,
            nugget_mode="free",
            smooth_bounds=self.smooth_bounds,
        )
        self.param_bounds = {
            "signal_variance": (1e-8, 100.0),
            "range_lat": (0.01, 10.0),
            "range_lon": (0.01, 10.0),
            "smooth": self.smooth_bounds,
            "nugget": (0.0, 10.0),
        }

    def test_parameter_roundtrip_uses_public_signal_variance_name(self) -> None:
        params = matern_bessel.natural_from_raw(
            self.raw, "free", fixed_nugget=0.0, smooth_bounds=self.smooth_bounds
        )
        self.assertAlmostEqual(params.signal_variance, 1.3)
        record = params.to_record()
        self.assertIn("signal_variance", record)
        self.assertIn("signal_standard_deviation", record)
        self.assertNotIn("sigmasq", record)
        with self.assertRaisesRegex(ValueError, "nugget_mode"):
            matern_bessel.natural_from_raw(
                self.raw, "typo", fixed_nugget=0.0, smooth_bounds=self.smooth_bounds
            )

    def test_duplicate_locations_have_signal_covariance_off_diagonal(self) -> None:
        params = matern_bessel.MaternParameters(
            signal_variance=2.0,
            range_lat=1.0,
            range_lon=1.0,
            smooth=0.7,
            nugget=0.3,
            phi1=2.0,
            phi2=1.0,
            phi3=1.0,
        )
        coords = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
        d_lat, d_lon = matern_bessel.pairwise_deltas(coords)
        covariance = matern_bessel.covariance_from_deltas(d_lat, d_lon, params, jitter=1e-6)
        self.assertEqual(covariance[0, 1], 2.0)
        self.assertAlmostEqual(covariance[0, 0], 2.300001)

    def test_full_likelihood_loss_and_total_nll_are_consistent(self) -> None:
        average = matern_bessel.profiled_full_nll(
            self.raw,
            self.y,
            self.coords,
            nugget_mode="free",
            fixed_nugget=0.0,
            smooth_bounds=self.smooth_bounds,
            param_bounds=self.param_bounds,
            mean_design="constant",
            scale_by_n=True,
        )
        total = matern_bessel.profiled_full_nll(
            self.raw,
            self.y,
            self.coords,
            nugget_mode="free",
            fixed_nugget=0.0,
            smooth_bounds=self.smooth_bounds,
            param_bounds=self.param_bounds,
            mean_design="constant",
            scale_by_n=False,
        )
        self.assertAlmostEqual(average * self.y.size, total, places=12)

    def test_vecchia_dummy_padding_is_likelihood_neutral(self) -> None:
        raw = matern_bessel.raw_from_natural(
            signal_variance=1.0,
            range_lat=1.0,
            range_lon=1.0,
            smooth=0.7,
            nugget=0.0,
            nugget_mode="fixed0",
            smooth_bounds=self.smooth_bounds,
        )
        coords = np.array([[[0.0, 0.0], [1.0, 0.0]]])
        base = {
            "max_cond_points": 0,
            "target_size": 2,
            "coords": coords,
            "X": np.ones((1, 2, 1)),
            "y": np.array([[[1.0], [2.0]]]),
        }
        padded = {
            "max_cond_points": 1,
            "target_size": 2,
            "coords": np.array([[[1e8, 1e8], [0.0, 0.0], [1.0, 0.0]]]),
            "X": np.array([[[0.0], [1.0], [1.0]]]),
            "y": np.array([[[0.0], [1.0], [2.0]]]),
            "is_dummy": np.array([[[True], [False], [False]]]),
        }
        kwargs = dict(
            raw=raw,
            n_features=1,
            nugget_mode="fixed0",
            fixed_nugget=0.0,
            smooth_bounds=self.smooth_bounds,
            param_bounds=self.param_bounds,
        )
        base_loss = matern_bessel.profiled_vecchia_cluster_nll(batches=[base], **kwargs)
        padded_loss = matern_bessel.profiled_vecchia_cluster_nll(batches=[padded], **kwargs)
        self.assertAlmostEqual(base_loss, padded_loss, places=12)

        total = matern_bessel.profiled_vecchia_cluster_nll(
            batches=[base], scale_by_n=False, **kwargs
        )
        self.assertAlmostEqual(base_loss * 2, total, places=12)

    def test_full_fit_selects_valid_state_even_when_optimizer_did_not_converge(
        self,
    ) -> None:
        invalid_result = mock.Mock(
            x=np.full_like(self.raw, np.nan),
            success=True,
            message="optimizer claimed convergence",
            status=0,
            nfev=3,
        )
        valid_result = mock.Mock(
            x=self.raw.copy(),
            success=False,
            message="iteration limit",
            status=1,
            nfev=5,
        )
        with mock.patch.object(
            matern_bessel, "minimize", side_effect=[invalid_result, valid_result]
        ):
            result = matern_bessel.fit_full_matern(
                self.y,
                self.coords,
                nugget_mode="free",
                mean_design="constant",
                smooth_bounds=self.smooth_bounds,
                n_restarts=2,
                maxiter=1,
            )

        self.assertTrue(result["valid"])
        self.assertFalse(result["converged"])
        self.assertFalse(result["success"])
        self.assertEqual(result["message"], "iteration limit")
        self.assertTrue(np.isfinite(result["loss"]))
        self.assertTrue(np.isfinite(result["nll"]))
        self.assertEqual(result["n_valid_restarts"], 1)
        self.assertEqual(result["n_converged_restarts"], 1)
        self.assertEqual(len(result["restart_records"]), 2)
        self.assertFalse(result["restart_records"][0]["valid"])
        self.assertTrue(result["restart_records"][0]["converged"])
        self.assertIn(
            "evaluation_failed",
            result["restart_records"][0]["evaluation_message"],
        )
        self.assertTrue(result["restart_records"][1]["valid"])
        self.assertFalse(result["restart_records"][1]["converged"])

    def test_vecchia_fit_returns_explicit_failure_when_all_restarts_are_invalid(
        self,
    ) -> None:
        batch = {
            "max_cond_points": 0,
            "target_size": 2,
            "coords": np.array([[[0.0, 0.0], [1.0, 0.0]]]),
            "X": np.ones((1, 2, 1)),
            "y": np.array([[[1.0], [2.0]]]),
        }
        invalid_result = mock.Mock(
            x=np.full(4, np.nan),
            success=True,
            message="optimizer claimed convergence",
            status=0,
            nfev=2,
        )
        with mock.patch.object(matern_bessel, "minimize", return_value=invalid_result):
            result = matern_bessel.fit_vecchia_matern_from_batches(
                batches=[batch],
                n_features=1,
                y_var=1.0,
                nugget_mode="fixed0",
                smooth_bounds=self.smooth_bounds,
                n_restarts=1,
                maxiter=1,
            )

        self.assertFalse(result["valid"])
        self.assertFalse(result["converged"])
        self.assertFalse(result["success"])
        self.assertEqual(result["message"], "all_restarts_invalid")
        self.assertEqual(result["evaluation_message"], "no_finite_evaluable_state")
        self.assertTrue(np.isinf(result["loss"]))
        self.assertTrue(np.isinf(result["nll"]))
        self.assertEqual(result["n_valid_restarts"], 0)
        self.assertEqual(result["n_converged_restarts"], 1)
        self.assertEqual(len(result["restart_records"]), 1)
        self.assertFalse(result["restart_records"][0]["valid"])
        self.assertTrue(result["restart_records"][0]["converged"])

    def test_optimizer_exception_preserves_valid_initial_state(self) -> None:
        with mock.patch.object(
            matern_bessel, "minimize", side_effect=RuntimeError("solver aborted")
        ):
            result = matern_bessel.fit_full_matern(
                self.y,
                self.coords,
                nugget_mode="free",
                mean_design="constant",
                smooth_bounds=self.smooth_bounds,
                n_restarts=1,
                maxiter=1,
            )

        self.assertTrue(result["valid"])
        self.assertFalse(result["converged"])
        self.assertFalse(result["success"])
        self.assertIn("optimizer_failed: RuntimeError", result["message"])
        self.assertEqual(result["n_valid_restarts"], 1)
        self.assertEqual(result["n_converged_restarts"], 0)


if __name__ == "__main__":
    unittest.main()
