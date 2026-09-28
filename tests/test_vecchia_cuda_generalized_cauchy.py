"""CUDA parity checks for the fused generalized-Cauchy covariance kernel."""

from __future__ import annotations

import importlib
import unittest

import torch


class CudaGeneralizedCauchyParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is unavailable")
        module = importlib.import_module("GEMS_TCO.vecchia._native_covariance")
        if not module.native_generalized_cauchy_covariance_available("cuda"):
            raise unittest.SkipTest("updated CUDA generalized-Cauchy extension is unavailable")
        cls.native = staticmethod(module.native_generalized_cauchy_covariance)
        cls.reference = staticmethod(module.torch_generalized_cauchy_covariance_reference)
        cls.device = torch.device("cuda")

    @classmethod
    def _inputs(cls, with_nugget: bool):
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
            device=cls.device,
        )
        dummy = torch.tensor(
            [
                [[False], [False], [False], [True], [False], [False]],
                [[True], [False], [False], [False], [True], [False]],
            ],
            dtype=torch.bool,
            device=cls.device,
        )
        values = [0.25, -0.35, 0.45, -0.20, 0.018, -0.110]
        if with_nugget:
            values.append(-2.50)
        params = torch.tensor(values, dtype=torch.float64, device=cls.device)
        return params, coordinates, dummy

    def test_cuda_covariance_and_gradient_match_torch(self) -> None:
        generator = torch.Generator(device="cpu").manual_seed(20260925)
        weight = torch.randn((2, 6, 6), generator=generator, dtype=torch.float64).to(
            self.device
        )
        for alpha, beta in ((0.75, 1.0), (1.4, 2.3)):
            for with_nugget in (False, True):
                with self.subTest(alpha=alpha, beta=beta, nugget=with_nugget):
                    initial, coordinates, dummy = self._inputs(with_nugget)
                    reference_params = initial.clone().requires_grad_(True)
                    native_params = initial.clone().requires_grad_(True)
                    expected = self.reference(
                        reference_params, coordinates, dummy, alpha, beta
                    )
                    actual = self.native(
                        native_params,
                        coordinates,
                        dummy,
                        alpha,
                        beta,
                        backend="native",
                    )
                    expected_gradient = torch.autograd.grad(
                        (expected * weight).sum(), reference_params
                    )[0]
                    actual_gradient = torch.autograd.grad(
                        (actual * weight).sum(), native_params
                    )[0]
                    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
                    torch.testing.assert_close(
                        actual_gradient,
                        expected_gradient,
                        rtol=2e-8,
                        atol=2e-10,
                    )

    def test_cuda_no_nugget_nll_and_gradient_match_torch(self) -> None:
        initial, coordinates, dummy = self._inputs(False)
        # Keep this inverse-based check well conditioned while the linear
        # gradient test above continues to cover exact-zero distances.  A
        # duplicate in a nugget-free matrix is separated only by the 1e-6
        # numerical jitter and makes an NLL gradient comparison measure
        # Cholesky sensitivity to one-ulp forward differences instead of the
        # covariance kernel derivative.
        coordinates = coordinates.clone()
        coordinates[0, 1, 0] += 0.013
        coordinates[1, 3, 1] += 0.017
        response = torch.tensor(
            [
                [[0.30], [-0.20], [0.75], [0.0], [-0.45], [0.10]],
                [[0.0], [0.25], [-0.55], [0.90], [0.0], [-0.15]],
            ],
            dtype=torch.float64,
            device=self.device,
        )

        def nll(covariance: torch.Tensor) -> torch.Tensor:
            factor = torch.linalg.cholesky(covariance)
            whitened = torch.linalg.solve_triangular(factor, response, upper=False)
            log_determinant = 2.0 * torch.log(
                torch.diagonal(factor, dim1=-2, dim2=-1)
            ).sum()
            return 0.5 * (log_determinant + whitened.square().sum()) / (~dummy).sum()

        reference_params = initial.clone().requires_grad_(True)
        native_params = initial.clone().requires_grad_(True)
        reference_nll = nll(
            self.reference(reference_params, coordinates, dummy, 0.75, 1.0)
        )
        native_nll = nll(
            self.native(
                native_params,
                coordinates,
                dummy,
                0.75,
                1.0,
                backend="native",
            )
        )
        reference_gradient = torch.autograd.grad(reference_nll, reference_params)[0]
        native_gradient = torch.autograd.grad(native_nll, native_params)[0]

        torch.testing.assert_close(native_nll, reference_nll, rtol=2e-11, atol=2e-12)
        torch.testing.assert_close(
            native_gradient,
            reference_gradient,
            rtol=2e-8,
            atol=2e-10,
        )

    def test_cuda_gc_backward_uses_current_stream(self) -> None:
        initial, coordinates, dummy = self._inputs(False)
        weight = torch.arange(72, dtype=torch.float64).reshape(2, 6, 6).to(self.device)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            params = initial.clone().requires_grad_(True)
            covariance = self.native(
                params,
                coordinates,
                dummy,
                0.75,
                1.0,
                backend="native",
            )
            gradient = torch.autograd.grad((covariance * weight).sum(), params)[0]
        stream.synchronize()
        self.assertTrue(torch.isfinite(gradient).all())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
