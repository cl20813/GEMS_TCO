"""Parity checks for the optional CPU Vecchia covariance extension.

The native backend is an accelerator, not a second statistical model.  These
tests therefore compare it with the differentiable Torch reference at several
non-fitted parameter values.  They deliberately include duplicate coordinates
(exact zero distance) and padded dummy observations, because those are the two
places where a fused implementation can most easily change the likelihood.
"""

from __future__ import annotations

import importlib
import math
import unittest

import torch


class NativeVecchiaCovarianceParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            native_module = importlib.import_module("GEMS_TCO.vecchia._native_covariance")
        except ImportError as error:
            raise unittest.SkipTest(f"native covariance wrapper is unavailable: {error}")

        if not native_module.native_covariance_available():
            raise unittest.SkipTest("optional CPU Vecchia covariance extension is unavailable")

        cls.native_covariance = staticmethod(native_module.native_covariance)
        cls.torch_covariance_reference = staticmethod(native_module.torch_covariance_reference)

    @staticmethod
    def _coordinates_and_dummy_mask() -> tuple[torch.Tensor, torch.Tensor]:
        # Points 0 and 1 in the first batch are intentionally identical and
        # real.  Exact zero distance must produce correlation one without
        # introducing a square-root derivative singularity.
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
        is_dummy = torch.tensor(
            [
                [[False], [False], [False], [True], [False], [False]],
                [[True], [False], [False], [False], [True], [False]],
            ],
            dtype=torch.bool,
        )
        return coordinates, is_dummy

    @staticmethod
    def _off_optimum_parameter_vectors() -> tuple[torch.Tensor, ...]:
        # [log(phi1), log(phi2), log(phi3), log(phi4),
        #  advection_lat, advection_lon, log(nugget)]
        return tuple(
            torch.tensor(values, dtype=torch.float64)
            for values in (
                (0.25, -0.35, 0.45, -0.20, 0.018, -0.110, -2.50),
                (-0.40, 0.30, -0.60, 0.50, -0.025, 0.080, -4.00),
                (0.80, -0.70, 0.20, -0.50, 0.000, -0.126, -1.30),
            )
        )

    @staticmethod
    def _gaussian_nll(
        covariance: torch.Tensor,
        response: torch.Tensor,
        is_dummy: torch.Tensor,
    ) -> torch.Tensor:
        response = response.masked_fill(is_dummy, 0.0)
        factor = torch.linalg.cholesky(covariance)
        whitened = torch.linalg.solve_triangular(factor, response, upper=False)
        log_determinant = 2.0 * torch.log(torch.diagonal(factor, dim1=-2, dim2=-1)).sum()
        quadratic = whitened.square().sum()
        n_real = (~is_dummy.squeeze(-1)).sum()
        return 0.5 * (log_determinant + quadratic) / n_real

    def test_covariance_matches_torch_at_off_optimum_parameters(self) -> None:
        coordinates, is_dummy = self._coordinates_and_dummy_mask()

        for parameter_index, params in enumerate(self._off_optimum_parameter_vectors()):
            with self.subTest(parameter_index=parameter_index):
                expected = self.torch_covariance_reference(
                    params,
                    coordinates,
                    is_dummy,
                    smooth=0.5,
                )
                actual = self.native_covariance(
                    params,
                    coordinates,
                    is_dummy,
                    smooth=0.5,
                    backend="native",
                )
                automatic = self.native_covariance(
                    params,
                    coordinates,
                    is_dummy,
                    smooth=0.5,
                    backend="auto",
                )

                torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
                torch.testing.assert_close(automatic, actual, rtol=0.0, atol=0.0)
                torch.testing.assert_close(actual, actual.transpose(-1, -2))

                # Off-diagonal zero-distance covariance is signal variance;
                # the nugget and numerical jitter belong on the diagonal only.
                signal_variance = torch.exp(params[0] - params[1])
                torch.testing.assert_close(
                    actual[0, 0, 1],
                    signal_variance,
                    rtol=1e-12,
                    atol=1e-12,
                )

                dummy = is_dummy.squeeze(-1)
                for batch_index, point_index in dummy.nonzero(as_tuple=False).tolist():
                    expected_row = torch.zeros(6, dtype=torch.float64)
                    expected_row[point_index] = 1.0
                    torch.testing.assert_close(
                        actual[batch_index, point_index],
                        expected_row,
                        rtol=0.0,
                        atol=0.0,
                    )

    def test_nll_and_parameter_gradients_match_torch_reference(self) -> None:
        coordinates, is_dummy = self._coordinates_and_dummy_mask()
        response = torch.tensor(
            [
                [[0.30], [-0.20], [0.75], [8.00], [-0.45], [0.10]],
                [[-7.00], [0.25], [-0.55], [0.90], [6.00], [-0.15]],
            ],
            dtype=torch.float64,
        )

        for parameter_index, initial_params in enumerate(self._off_optimum_parameter_vectors()):
            with self.subTest(parameter_index=parameter_index):
                reference_params = initial_params.clone().requires_grad_(True)
                native_params = initial_params.clone().requires_grad_(True)

                reference_covariance = self.torch_covariance_reference(
                    reference_params,
                    coordinates,
                    is_dummy,
                    smooth=0.5,
                )
                native_covariance = self.native_covariance(
                    native_params,
                    coordinates,
                    is_dummy,
                    smooth=0.5,
                    backend="native",
                )

                reference_nll = self._gaussian_nll(
                    reference_covariance,
                    response,
                    is_dummy,
                )
                native_nll = self._gaussian_nll(
                    native_covariance,
                    response,
                    is_dummy,
                )
                reference_gradient = torch.autograd.grad(reference_nll, reference_params)[0]
                native_gradient = torch.autograd.grad(native_nll, native_params)[0]

                torch.testing.assert_close(native_nll, reference_nll, rtol=1e-10, atol=1e-11)
                self.assertTrue(torch.isfinite(native_gradient).all())
                torch.testing.assert_close(
                    native_gradient,
                    reference_gradient,
                    rtol=1e-5,
                    atol=1e-7,
                )
                self.assertLessEqual(
                    float((native_gradient - reference_gradient).abs().max()),
                    1e-5,
                )

                # Keep this assertion explicit: the parameter choices should
                # exercise a real gradient rather than accidentally comparing
                # two flat computations.
                self.assertGreater(
                    float(reference_gradient.abs().max()),
                    math.ulp(1.0),
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
