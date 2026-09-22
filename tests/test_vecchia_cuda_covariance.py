"""CUDA parity checks for the optional fused Vecchia covariance extension.

These tests are intentionally skipped on ordinary CPU CI.  The Amarel smoke
job builds the extension for A100/L40S and runs this file on an allocated GPU.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import unittest

import numpy as np
import torch


class CudaVecchiaCovarianceParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is unavailable")
        try:
            native_module = importlib.import_module("GEMS_TCO.vecchia._native_covariance")
        except ImportError as error:
            raise unittest.SkipTest(f"native covariance wrapper is unavailable: {error}")
        if not native_module.native_covariance_available("cuda"):
            raise unittest.SkipTest("optional CUDA Vecchia covariance extension is unavailable")

        cls.native_covariance = staticmethod(native_module.native_covariance)
        cls.torch_covariance_reference = staticmethod(native_module.torch_covariance_reference)
        cls.device = torch.device("cuda")

    @classmethod
    def _inputs(cls) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        is_dummy = torch.tensor(
            [
                [[False], [False], [False], [True], [False], [False]],
                [[True], [False], [False], [False], [True], [False]],
            ],
            dtype=torch.bool,
            device=cls.device,
        )
        params = torch.tensor(
            [0.25, -0.35, 0.45, -0.20, 0.018, -0.110, -2.50],
            dtype=torch.float64,
            device=cls.device,
        )
        response = torch.tensor(
            [
                [[0.30], [-0.20], [0.75], [8.00], [-0.45], [0.10]],
                [[-7.00], [0.25], [-0.55], [0.90], [6.00], [-0.15]],
            ],
            dtype=torch.float64,
            device=cls.device,
        )
        return params, coordinates, is_dummy, response

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
        return 0.5 * (log_determinant + quadratic) / (~is_dummy.squeeze(-1)).sum()

    def test_cuda_covariance_and_dummy_decoupling_match_torch(self) -> None:
        params, coordinates, is_dummy, _ = self._inputs()
        expected = self.torch_covariance_reference(params, coordinates, is_dummy, smooth=0.5)
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

        torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
        torch.testing.assert_close(automatic, actual, rtol=0.0, atol=0.0)
        torch.testing.assert_close(actual, actual.transpose(-1, -2), rtol=0.0, atol=0.0)

        dummy = is_dummy.squeeze(-1)
        for batch_index, point_index in dummy.nonzero(as_tuple=False).cpu().tolist():
            expected_row = torch.zeros(6, dtype=torch.float64, device=self.device)
            expected_row[point_index] = 1.0
            torch.testing.assert_close(
                actual[batch_index, point_index],
                expected_row,
                rtol=0.0,
                atol=0.0,
            )

    def test_cuda_nll_and_first_order_gradient_match_torch(self) -> None:
        initial_params, coordinates, is_dummy, response = self._inputs()
        reference_params = initial_params.clone().requires_grad_(True)
        native_params = initial_params.clone().requires_grad_(True)

        reference_nll = self._gaussian_nll(
            self.torch_covariance_reference(
                reference_params,
                coordinates,
                is_dummy,
                smooth=0.5,
            ),
            response,
            is_dummy,
        )
        native_nll = self._gaussian_nll(
            self.native_covariance(
                native_params,
                coordinates,
                is_dummy,
                smooth=0.5,
                backend="native",
            ),
            response,
            is_dummy,
        )
        reference_gradient = torch.autograd.grad(reference_nll, reference_params)[0]
        native_gradient = torch.autograd.grad(native_nll, native_params)[0]

        torch.testing.assert_close(native_nll, reference_nll, rtol=2e-11, atol=2e-12)
        self.assertTrue(torch.isfinite(native_gradient).all())
        torch.testing.assert_close(
            native_gradient,
            reference_gradient,
            rtol=2e-8,
            atol=2e-10,
        )

    def test_cuda_backward_handles_nonsymmetric_upstream_on_current_stream(self) -> None:
        initial_params, coordinates, is_dummy, _ = self._inputs()
        generator = torch.Generator(device="cpu").manual_seed(20260922)
        weight = torch.randn(
            (2, 6, 6),
            generator=generator,
            dtype=torch.float64,
        ).to(self.device)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))

        with torch.cuda.stream(stream):
            reference_params = initial_params.clone().requires_grad_(True)
            native_params = initial_params.clone().requires_grad_(True)
            reference_covariance = self.torch_covariance_reference(
                reference_params,
                coordinates,
                is_dummy,
                smooth=0.5,
            )
            actual_covariance = self.native_covariance(
                native_params,
                coordinates,
                is_dummy,
                smooth=0.5,
                backend="native",
            )
            reference_gradient = torch.autograd.grad(
                (reference_covariance * weight).sum(),
                reference_params,
            )[0]
            native_gradient = torch.autograd.grad(
                (actual_covariance * weight).sum(),
                native_params,
            )[0]
            covariance_difference = (actual_covariance - reference_covariance).abs().max()
            gradient_difference = (native_gradient - reference_gradient).abs().max()

        stream.synchronize()
        self.assertLessEqual(float(covariance_difference), 2e-12)
        torch.testing.assert_close(
            native_gradient,
            reference_gradient,
            rtol=2e-8,
            atol=2e-10,
        )
        self.assertLessEqual(float(gradient_difference), 2e-8)

    def test_lag432_profiled_likelihood_dispatch_matches_torch(self) -> None:
        from GEMS_TCO.vecchia.corridor_neighbors import Lag432CorridorVecchia

        latitude = np.repeat(np.arange(4, dtype=float) * 0.05, 4)
        longitude = np.tile(np.arange(4, dtype=float) * 0.05, 4)
        grid_coordinates = np.column_stack((latitude, longitude))
        input_map = {}
        for time_index in range(3):
            values = np.zeros((16, 4), dtype=np.float64)
            values[:, :2] = grid_coordinates
            values[:, 2] = np.sin(np.arange(16) * 0.2) + time_index * 0.01
            values[:, 3] = float(time_index)
            input_map[str(time_index)] = torch.tensor(
                values,
                dtype=torch.float64,
                device=self.device,
            )

        model = Lag432CorridorVecchia(
            smooth=0.5,
            input_map=input_map,
            grid_coords=grid_coordinates,
            target_chunk_size=8,
            covariance_backend="torch",
        )
        with contextlib.redirect_stdout(io.StringIO()):
            model.precompute_conditioning_sets()
        initial_params = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.0, -0.126, -4.0],
            dtype=torch.float64,
            device=self.device,
        )

        reference_params = initial_params.clone().requires_grad_(True)
        reference_nll = model.profiled_negative_log_likelihood(reference_params)
        reference_gradient = torch.autograd.grad(reference_nll, reference_params)[0]

        model.covariance_backend = "native"
        self.assertEqual(model.resolved_covariance_backend(), "native")
        native_params = initial_params.clone().requires_grad_(True)
        native_nll = model.profiled_negative_log_likelihood(native_params)
        native_gradient = torch.autograd.grad(native_nll, native_params)[0]

        torch.testing.assert_close(native_nll, reference_nll, rtol=2e-10, atol=2e-11)
        torch.testing.assert_close(
            native_gradient,
            reference_gradient,
            rtol=2e-7,
            atol=2e-9,
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
