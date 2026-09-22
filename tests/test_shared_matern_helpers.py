"""Regression checks for the canonical shared Matérn helper functions."""

from __future__ import annotations

import importlib.util
import inspect
import unittest

import numpy as np

import GEMS_TCO.spatial as spatial
from GEMS_TCO.spatial import matern_bessel as bessel
from GEMS_TCO.spatial import matern_spline


class SharedMaternHelperTests(unittest.TestCase):
    def test_spatial_package_exposes_curated_model_api(self) -> None:
        expected = {
            "IsotropicMaternSpatialVecchia",
            "AnisotropicMaternSpatialVecchia",
            "NoNuggetAnisotropicCauchySpatialVecchia",
            "SpatialLBFGSFitResult",
            "fit_full_matern",
            "fit_vecchia_matern_from_batches",
            "vecchia_batches_to_numpy",
        }
        self.assertTrue(expected.issubset(set(spatial.__all__)))
        for name in expected:
            self.assertIsNotNone(getattr(spatial, name))
        excluded = {
            "NearestNeighborSpatialVecchia",
            "ColumnSpatialVecchia",
            "fit_full_matern_torch",
            "TorchMaternParameters",
        }
        self.assertTrue(excluded.isdisjoint(spatial.__all__))

    def test_public_model_constructors_have_explicit_signatures(self) -> None:
        model_classes = (
            spatial.IsotropicMaternSpatialVecchia,
            spatial.NoNuggetIsotropicMaternSpatialVecchia,
            spatial.AnisotropicMaternSpatialVecchia,
            spatial.NoNuggetAnisotropicMaternSpatialVecchia,
            spatial.NoNuggetAnisotropicCauchySpatialVecchia,
            spatial.FixedBetaAnisotropicCauchySpatialVecchia,
            spatial.NoNuggetFixedBetaAnisotropicCauchySpatialVecchia,
        )
        for model_class in model_classes:
            with self.subTest(model=model_class.__name__):
                signature = inspect.signature(model_class)
                kinds = {parameter.kind for parameter in signature.parameters.values()}
                self.assertNotIn(inspect.Parameter.VAR_POSITIONAL, kinds)
                self.assertNotIn(inspect.Parameter.VAR_KEYWORD, kinds)
                self.assertIn("input_map", signature.parameters)
                self.assertIn("block_shape", signature.parameters)
                self.assertIn("mean_design", signature.parameters)
                self.assertEqual(
                    signature.parameters["mean_design"].default,
                    "latlon_hour",
                )
                for method_name in (
                    "profiled_negative_log_likelihood",
                    "estimate_gls_coefficients",
                    "make_lbfgs_optimizer",
                    "fit_lbfgs",
                ):
                    self.assertTrue(hasattr(model_class, method_name), method_name)
                for retired_name in (
                    "vecchia_batched_likelihood",
                    "get_gls_beta",
                    "set_optimizer",
                    "fit_vecc_lbfgs",
                ):
                    self.assertFalse(hasattr(model_class, retired_name), retired_name)

    def test_public_vecchia_batch_adapter_exports_precomputed_batches(self) -> None:
        rows = np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 2.0, 0.0],
            ],
            dtype=np.float64,
        )
        model = spatial.AnisotropicMaternSpatialVecchia(
            smooth=0.5,
            input_map={"replicate": rows},
            block_shape=(1, 1),
            n_neighbor_blocks=1,
            mean_design="lat",
        )
        with self.assertRaisesRegex(ValueError, "precomputed conditioning sets"):
            spatial.vecchia_batches_to_numpy(model)

        model.precompute_conditioning_sets()
        batches = spatial.vecchia_batches_to_numpy(model)
        self.assertGreater(len(batches), 0)
        self.assertEqual(batches[0]["coords"].shape[-1], 2)
        self.assertEqual(batches[0]["X"].shape[-1], model.n_features)
        self.assertEqual(batches[0]["y"].shape[-1], 1)

    def test_dated_spatial_modules_are_not_left_as_compatibility_shims(self) -> None:
        removed_modules = (
            "GEMS_TCO.kernels_space_base_engine_052126",
            "GEMS_TCO.kernels_space_iso_cluster_052426",
            "GEMS_TCO.kernels_space_aniso_cluster_060326",
            "GEMS_TCO.kernels_space_aniso_cauchy_cluster_060326",
            "GEMS_TCO.matern_bessel_anisotropic",
            "GEMS_TCO.matern_spline",
            "GEMS_TCO.torch_bessel_full_likelihood",
            "GEMS_TCO.spatial.torch_matern_bessel",
        )
        for module_name in removed_modules:
            with self.subTest(module=module_name):
                self.assertIsNone(importlib.util.find_spec(module_name))

    def test_spline_coefficients_match_pre_merge_golden(self) -> None:
        coeffs = matern_spline._build_matern_spline_coeffs(
            smooth=0.7,
            n_points=9,
            r_max=4.0,
        )

        np.testing.assert_allclose(
            coeffs["knots"],
            np.arange(0.0, 4.5, 0.5, dtype=np.float64),
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            coeffs["a"],
            np.array(
                [
                    1.0,
                    0.6720179816547904,
                    0.40618184037575733,
                    0.23868584059301937,
                    0.13828069713920702,
                    0.07941967122077294,
                    0.04534635178989962,
                    0.025781575046338653,
                ],
                dtype=np.float64,
            ),
            rtol=2e-15,
            atol=1e-15,
        )

    def test_bessel_values_match_pre_merge_golden(self) -> None:
        distances = np.array(
            [0.0, 0.1, 0.5, 1.0, 2.0, 10.0],
            dtype=np.float64,
        )
        expected = np.array(
            [
                1.0,
                0.948695629372086,
                0.6720179816547905,
                0.4061818403757575,
                0.13828069713920704,
                1.4298117667356191e-05,
            ],
            dtype=np.float64,
        )

        np.testing.assert_allclose(
            bessel.matern_corr_bessel(distances, smooth=0.7),
            expected,
            rtol=2e-15,
            atol=1e-15,
        )

    def test_bessel_rejects_negative_distance(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            bessel.matern_corr_bessel(np.array([-1.0]), smooth=0.7)


if __name__ == "__main__":
    unittest.main()
