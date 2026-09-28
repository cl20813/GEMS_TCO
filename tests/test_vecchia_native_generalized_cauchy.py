"""Parity tests for the optional fused generalized-Cauchy covariance.

The native implementation is an accelerator for the exact same fixed-shape
model.  These tests cover both the six-parameter no-nugget form used by the
interaction diagnostic and the seven-parameter fitted-nugget form.
"""

from __future__ import annotations

import importlib
import unittest

import torch


class NativeGeneralizedCauchyParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        module = importlib.import_module("GEMS_TCO.vecchia._native_covariance")
        if not module.native_generalized_cauchy_covariance_available("cpu"):
            raise unittest.SkipTest("updated CPU generalized-Cauchy extension is unavailable")
        cls.native = staticmethod(module.native_generalized_cauchy_covariance)
        cls.reference = staticmethod(module.torch_generalized_cauchy_covariance_reference)

    @staticmethod
    def _coordinates_and_dummy() -> tuple[torch.Tensor, torch.Tensor]:
        coordinates = torch.tensor(
            [
                [
                    [34.000, 126.000, 0.0],
                    [34.000, 126.000, 0.0],
                    [34.045, 126.030, 1.0],
                    [34.080, 126.075, 1.0],
                    [34.120, 126.135, 2.0],
                    [34.165, 126.210, 3.0],
                ],
                [
                    [35.000, 127.000, 0.0],
                    [35.025, 127.040, 0.0],
                    [35.070, 127.085, 1.0],
                    [35.070, 127.085, 1.0],
                    [35.110, 127.150, 2.0],
                    [35.180, 127.225, 3.0],
                ],
            ],
            dtype=torch.float64,
        )
        dummy = torch.tensor(
            [
                [[False], [False], [False], [True], [False], [False]],
                [[True], [False], [False], [False], [True], [False]],
            ],
            dtype=torch.bool,
        )
        return coordinates, dummy

    @staticmethod
    def _parameter_vectors() -> tuple[torch.Tensor, torch.Tensor]:
        no_nugget = torch.tensor(
            [0.25, -0.35, 0.45, -0.20, 0.018, -0.110],
            dtype=torch.float64,
        )
        with_nugget = torch.cat(
            (no_nugget, torch.tensor([-2.50], dtype=torch.float64))
        )
        return no_nugget, with_nugget

    def test_covariance_matches_reference_for_both_nugget_modes(self) -> None:
        coordinates, dummy = self._coordinates_and_dummy()
        for alpha, beta in ((0.75, 1.0), (1.4, 2.3)):
            for initial in self._parameter_vectors():
                with self.subTest(alpha=alpha, beta=beta, count=initial.numel()):
                    expected = self.reference(initial, coordinates, dummy, alpha, beta)
                    actual = self.native(
                        initial,
                        coordinates,
                        dummy,
                        alpha,
                        beta,
                        backend="native",
                    )
                    automatic = self.native(
                        initial,
                        coordinates,
                        dummy,
                        alpha,
                        beta,
                        backend="auto",
                    )
                    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
                    torch.testing.assert_close(automatic, actual, rtol=0.0, atol=0.0)
                    torch.testing.assert_close(actual, actual.transpose(-1, -2))

    def test_first_order_parameter_gradient_matches_reference(self) -> None:
        coordinates, dummy = self._coordinates_and_dummy()
        generator = torch.Generator().manual_seed(20260925)
        weight = torch.randn((2, 6, 6), generator=generator, dtype=torch.float64)

        for alpha, beta in ((0.75, 1.0), (1.4, 2.3)):
            for initial in self._parameter_vectors():
                with self.subTest(alpha=alpha, beta=beta, count=initial.numel()):
                    reference_params = initial.clone().requires_grad_(True)
                    native_params = initial.clone().requires_grad_(True)
                    reference_covariance = self.reference(
                        reference_params,
                        coordinates,
                        dummy,
                        alpha,
                        beta,
                    )
                    native_covariance = self.native(
                        native_params,
                        coordinates,
                        dummy,
                        alpha,
                        beta,
                        backend="native",
                    )
                    reference_gradient = torch.autograd.grad(
                        (reference_covariance * weight).sum(), reference_params
                    )[0]
                    native_gradient = torch.autograd.grad(
                        (native_covariance * weight).sum(), native_params
                    )[0]
                    torch.testing.assert_close(
                        native_gradient,
                        reference_gradient,
                        rtol=2e-9,
                        atol=2e-10,
                    )

    def test_native_request_rejects_an_old_extension_cleanly(self) -> None:
        # Availability is checked before dispatch, so a stale installed module
        # never fails later with an opaque missing pybind symbol.
        module = importlib.import_module("GEMS_TCO.vecchia._native_covariance")
        self.assertTrue(module.native_generalized_cauchy_covariance_available("cpu"))

    def test_corridor_model_auto_dispatches_to_cpu_native_kernel(self) -> None:
        from GEMS_TCO.vecchia.corridor_neighbors import (
            NoNuggetGeneralizedCauchyLag432CorridorVecchia,
        )

        rows = torch.zeros((4, 11), dtype=torch.float64)
        model = NoNuggetGeneralizedCauchyLag432CorridorVecchia(
            gc_alpha=0.75,
            gc_beta=1.0,
            input_map={"slot_0": rows},
            covariance_backend="auto",
        )
        self.assertEqual(model.resolved_covariance_backend(), "native")

        params = torch.tensor(
            [0.1, -0.2, 0.3, -0.1, 0.01, -0.126],
            dtype=torch.float64,
            requires_grad=True,
        )
        coordinates = torch.tensor(
            [[[0.0, 0.0, 0.0], [0.2, 0.3, 1.0], [0.4, 0.1, 2.0]]],
            dtype=torch.float64,
        )
        covariance = model._batched_covariance_with_dummy(
            params,
            coordinates,
            torch.zeros((1, 3), dtype=torch.bool),
        )
        (gradient,) = torch.autograd.grad(covariance.square().sum(), params)
        self.assertTrue(torch.isfinite(covariance).all())
        self.assertTrue(torch.isfinite(gradient).all())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
