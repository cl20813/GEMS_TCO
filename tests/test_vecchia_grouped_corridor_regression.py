"""Regression checks for the active grouped-batch corridor Vecchia path."""

from __future__ import annotations

import contextlib
import io
import unittest
from unittest import mock

import numpy as np
import torch


class VecchiaGroupedCorridorRegressionTests(unittest.TestCase):
    @staticmethod
    def _deterministic_input():
        lat = np.repeat(np.arange(8, dtype=float) * 0.05, 8)
        lon = np.tile(np.arange(8, dtype=float) * 0.05, 8)
        grid_coords = np.column_stack([lat, lon])

        input_map = {}
        for time_index in range(3):
            values = np.zeros((64, 11), dtype=np.float64)
            values[:, :2] = grid_coords
            values[:, 2] = np.sin(np.arange(64) * 0.1) + time_index * 0.01
            values[:, 3] = float(time_index)
            if time_index > 0:
                values[:, 4 + (time_index - 1)] = 1.0
            input_map[str(time_index)] = torch.tensor(values, dtype=torch.float64)

        return input_map, grid_coords

    def test_corridor_engine_uses_grouped_only_base(self) -> None:
        from GEMS_TCO.vecchia._base import GroupedVecchiaBase
        from GEMS_TCO.vecchia.corridor_neighbors.corridor_lag643 import Lag643CorridorVecchia
        from GEMS_TCO.vecchia.grouped_batched import GroupedBatchedVecchia

        self.assertTrue(issubclass(GroupedBatchedVecchia, GroupedVecchiaBase))
        self.assertTrue(
            issubclass(
                Lag643CorridorVecchia,
                GroupedBatchedVecchia,
            )
        )
        self.assertFalse(hasattr(GroupedVecchiaBase, "refresh_y_from_input_map"))
        self.assertFalse(hasattr(GroupedVecchiaBase, "vecchia_per_unit_nll_terms"))

    def test_corridor_batch_geometry_and_likelihood_golden_values(self) -> None:
        from GEMS_TCO.vecchia.corridor_neighbors.corridor_lag432 import Lag432CorridorVecchia
        from GEMS_TCO.vecchia.corridor_neighbors.corridor_lag643 import Lag643CorridorVecchia

        cases = (
            (
                Lag432CorridorVecchia,
                [
                    ("A", 64, 16, (4, 80)),
                    ("AB", 112, 16, (4, 128)),
                    ("ABC", 144, 16, (4, 160)),
                ],
                -0.8765651812001503,
            ),
            (
                Lag643CorridorVecchia,
                [
                    ("A", 96, 16, (4, 112)),
                    ("AB", 160, 16, (4, 176)),
                    ("ABC", 208, 16, (4, 224)),
                ],
                -0.8763009287162223,
            ),
        )
        params = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.0, -0.126, -20.0],
            dtype=torch.float64,
        )

        for model_class, expected_shapes, expected_loss in cases:
            with self.subTest(model=model_class.__name__):
                input_map, grid_coords = self._deterministic_input()
                model = model_class(
                    smooth=0.5,
                    input_map=input_map,
                    grid_coords=grid_coords,
                    target_chunk_size=16,
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    model.precompute_conditioning_sets()

                batch_shapes = [
                    (
                        batch.label,
                        batch.max_cond_points,
                        batch.target_size,
                        tuple(batch.indices.shape),
                    )
                    for batch in model._cluster_batches
                ]
                self.assertEqual(batch_shapes, expected_shapes)

                loss = float(model.profiled_negative_log_likelihood(params))
                self.assertAlmostEqual(loss, expected_loss, places=10)

    def test_public_numerical_api_uses_descriptive_names_without_shims(self) -> None:
        from GEMS_TCO.vecchia.grouped_batched import GroupedBatchedVecchia

        canonical_names = {
            "pairwise_anisotropic_distance",
            "batched_anisotropic_distance",
            "batched_covariance",
            "point_covariance",
            "profiled_negative_log_likelihood",
            "estimate_gls_coefficients",
            "make_lbfgs_optimizer",
            "interpretable_parameters",
            "fit_lbfgs",
        }
        for name in canonical_names:
            self.assertTrue(hasattr(GroupedBatchedVecchia, name), name)

        retired_names = {
            "precompute_coords_aniso_STABLE",
            "batched_manual_dist",
            "matern_cov_batched",
            "matern_cov_aniso_STABLE_log_reparam",
            "vecchia_batched_likelihood",
            "get_gls_beta",
            "set_optimizer",
            "_convert_params",
            "fit_vecc_lbfgs",
        }
        for name in retired_names:
            self.assertFalse(hasattr(GroupedBatchedVecchia, name), name)

        nll_doc = GroupedBatchedVecchia.profiled_negative_log_likelihood.__doc__
        point_doc = GroupedBatchedVecchia.point_covariance.__doc__
        self.assertIn("profiled", nll_doc)
        self.assertIn("0.5 * log(2*pi)", nll_doc)
        self.assertIn("same ordered point set", point_doc)

    def test_zero_distance_is_exact_and_jitter_is_covariance_only(self) -> None:
        from GEMS_TCO.vecchia.corridor_neighbors.corridor_lag643 import Lag643CorridorVecchia

        values = torch.tensor(
            [[0.0, 0.0, 1.0, 0.0], [0.1, 0.2, 2.0, 1.0]],
            dtype=torch.float64,
        )
        model = Lag643CorridorVecchia(0.5, {"0": values})
        distance_parameters = torch.tensor([1.0, 1.0, 0.0, -0.126], dtype=torch.float64)
        pairwise = model.pairwise_anisotropic_distance(
            distance_parameters,
            values,
            values.clone(),
        )
        batched = model.batched_anisotropic_distance(
            distance_parameters,
            values[:, [0, 1, 3]].unsqueeze(0),
        )
        torch.testing.assert_close(
            torch.diagonal(pairwise),
            torch.zeros(2, dtype=torch.float64),
        )
        torch.testing.assert_close(
            torch.diagonal(batched[0]),
            torch.zeros(2, dtype=torch.float64),
        )

        params = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.0, -0.126, -20.0],
            dtype=torch.float64,
        )
        covariance = model.point_covariance(params, values, values.clone())
        expected_diagonal = torch.full(
            (2,),
            1.0 + float(torch.exp(params[6])) + 1e-8,
            dtype=torch.float64,
        )
        torch.testing.assert_close(torch.diagonal(covariance), expected_diagonal)

        with self.assertRaisesRegex(ValueError, "shape"):
            model.batched_covariance(params, values.unsqueeze(0))
        nonfinite_batch = values[:, [0, 1, 3]].unsqueeze(0)
        nonfinite_batch[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            model.batched_covariance(params, nonfinite_batch)

    def test_conditioning_graph_and_batch_invariants(self) -> None:
        from GEMS_TCO.vecchia.corridor_neighbors.corridor_lag432 import Lag432CorridorVecchia

        input_map, grid_coords = self._deterministic_input()
        model = Lag432CorridorVecchia(
            smooth=0.5,
            input_map=input_map,
            grid_coords=grid_coords,
            target_chunk_size=16,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            model.precompute_conditioning_sets()

        expected_labels = {0: "A", 1: "AB", 2: "ABC"}
        for time_index in range(3):
            for block_index in range(model.n_clusters):
                label, max_cond_points, conditioning = model._conditioning_blocks(
                    block_index,
                    time_index,
                )
                self.assertEqual(label, expected_labels[time_index])
                self.assertEqual(len(conditioning), len(set(conditioning)))
                self.assertLessEqual(
                    len(conditioning) * model.max_points_per_cluster,
                    max_cond_points,
                )
                for conditioning_time, conditioning_block in conditioning:
                    self.assertLessEqual(conditioning_time, time_index)
                    if conditioning_time == time_index:
                        self.assertLess(conditioning_block, block_index)

        target_indices = []
        target_count = 0
        for batch in model._cluster_batches:
            self.assertEqual(
                batch.indices.shape[1],
                batch.max_cond_points + batch.target_size,
            )
            self.assertEqual(batch.coordinates.shape[:2], batch.indices.shape)
            self.assertEqual(batch.response.shape[:2], batch.indices.shape)
            self.assertEqual(batch.design.shape[:2], batch.indices.shape)
            self.assertFalse(batch.is_dummy[:, batch.max_cond_points :].any())
            targets = batch.indices[:, batch.max_cond_points :]
            target_indices.append(targets.reshape(-1))
            target_count += int(targets.numel())

        targets = torch.cat(target_indices)
        self.assertEqual(target_count, model.n_target_points)
        self.assertEqual(torch.unique(targets).numel(), model.n_target_points)

    def test_directional_summaries_exclude_unused_fixed_longitude_metadata(self) -> None:
        from GEMS_TCO.vecchia.corridor_neighbors.directional_lag432 import (
            DirectionalLag432CorridorVecchia,
        )
        from GEMS_TCO.vecchia.corridor_neighbors.directional_lag643 import (
            DirectionalLag643CorridorVecchia,
        )

        input_map, grid_coords = self._deterministic_input()
        retired_keys = {
            "lag1_lon_offset",
            "lag2_lon_offset",
            "lag1_lon_interval_lo",
            "lag1_lon_interval_hi",
            "lag2_lon_interval_lo",
            "lag2_lon_interval_hi",
            "grid_lon_step",
            "corridor_block_lon_width",
        }
        for model_class in (
            DirectionalLag432CorridorVecchia,
            DirectionalLag643CorridorVecchia,
        ):
            with self.subTest(model=model_class.__name__):
                model = model_class(
                    smooth=0.5,
                    input_map=input_map,
                    grid_coords=grid_coords,
                    reference_advec_lat=0.05,
                    reference_advec_lon=-0.10,
                    target_chunk_size=16,
                )
                self.assertNotIn("offsets=", model._precompute_message())
                self.assertNotIn("corridors=", model._precompute_message())
                with contextlib.redirect_stdout(io.StringIO()):
                    model.precompute_conditioning_sets()
                summary = model.cluster_summary()
                self.assertEqual(summary["conditioning_mode"], "directional_corridor_width")
                self.assertTrue(retired_keys.isdisjoint(summary))

    def test_nugget_is_added_at_every_exact_observation_match(self) -> None:
        from GEMS_TCO.vecchia.corridor_neighbors.corridor_lag643 import Lag643CorridorVecchia
        from GEMS_TCO.vecchia.corridor_neighbors.generalized_cauchy import (
            GeneralizedCauchyLag643CorridorVecchia,
        )
        from GEMS_TCO.vecchia.corridor_neighbors.spline import SplineMaternLag643CorridorVecchia

        values = torch.tensor(
            [[0.0, 0.0, 1.0, 0.0], [0.1, 0.1, 2.0, 0.0]],
            dtype=torch.float64,
        )
        input_map = {"0": values}
        models = (
            Lag643CorridorVecchia(0.5, input_map),
            GeneralizedCauchyLag643CorridorVecchia(
                1.0,
                1.0,
                input_map,
            ),
            SplineMaternLag643CorridorVecchia(
                0.7,
                input_map,
                spline_n_points=20,
            ),
        )
        x = values.clone()
        y = values.clone()
        y[:, 0] += 0.25
        low_nugget = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.0, -0.126, -20.0],
            dtype=torch.float64,
        )
        high_nugget = low_nugget.clone()
        high_nugget[6] = 2.0

        for model in models:
            with self.subTest(model=type(model).__name__):
                cross_low = model.point_covariance(
                    low_nugget,
                    x,
                    y,
                )
                cross_high = model.point_covariance(
                    high_nugget,
                    x,
                    y,
                )
                torch.testing.assert_close(cross_low, cross_high)

                same_points = x.clone()
                same_points[:, 2] += 100.0  # responses do not define point identity
                self_low = model.point_covariance(
                    low_nugget,
                    x,
                    same_points,
                )
                self_high = model.point_covariance(
                    high_nugget,
                    x,
                    same_points,
                )
                expected = torch.full(
                    (x.shape[0],),
                    torch.exp(high_nugget[6]) - torch.exp(low_nugget[6]),
                    dtype=torch.float64,
                )
                torch.testing.assert_close(
                    torch.diagonal(self_high - self_low),
                    expected,
                )

                permuted = x[[1, 0]]
                permuted_difference = model.point_covariance(
                    high_nugget,
                    x,
                    permuted,
                ) - model.point_covariance(
                    low_nugget,
                    x,
                    permuted,
                )
                expected_permuted = torch.tensor(
                    [[0.0, expected[0]], [expected[0], 0.0]],
                    dtype=torch.float64,
                )
                torch.testing.assert_close(permuted_difference, expected_permuted)

                partial_overlap = torch.stack(
                    [
                        x[1],
                        torch.tensor([0.7, 0.8, 3.0, 1.0], dtype=torch.float64),
                    ]
                )
                partial_difference = model.point_covariance(
                    high_nugget,
                    x,
                    partial_overlap,
                ) - model.point_covariance(
                    low_nugget,
                    x,
                    partial_overlap,
                )
                expected_partial = torch.tensor(
                    [[0.0, 0.0], [expected[0], 0.0]],
                    dtype=torch.float64,
                )
                torch.testing.assert_close(partial_difference, expected_partial)

    def test_input_and_geometry_validation_fail_early(self) -> None:
        from GEMS_TCO.vecchia.corridor_neighbors._geometry import _CorridorClusterVecchia
        from GEMS_TCO.vecchia.corridor_neighbors.corridor_lag432 import Lag432CorridorVecchia
        from GEMS_TCO.vecchia.grouped_batched import GroupedBatchedVecchia

        with self.assertRaisesRegex(ValueError, "non-empty mapping"):
            GroupedBatchedVecchia(0.5, {})

        input_map, grid_coords = self._deterministic_input()
        with self.assertRaisesRegex(ValueError, "second_lag_stride"):
            GroupedBatchedVecchia(0.5, input_map, second_lag_stride=1)
        with self.assertRaisesRegex(ValueError, "block_shape"):
            GroupedBatchedVecchia(0.5, input_map, block_shape=(0, 3))
        with self.assertRaisesRegex(ValueError, "min_target_points"):
            GroupedBatchedVecchia(0.5, input_map, min_target_points=0)
        for option, value in (
            ("second_lag_stride", 2.5),
            ("n_neighbor_blocks_t", 2.5),
            ("target_chunk_size", True),
            ("max_neighbor_search", 4.5),
        ):
            with self.subTest(option=option):
                with self.assertRaisesRegex(TypeError, option):
                    GroupedBatchedVecchia(0.5, input_map, **{option: value})
        with self.assertRaisesRegex(TypeError, "block_shape"):
            GroupedBatchedVecchia(0.5, input_map, block_shape=(3.5, 3))
        with self.assertRaisesRegex(TypeError, "lag0_block_count"):
            _CorridorClusterVecchia(0.5, input_map, lag0_block_count=4.5)

        reversed_map = dict(reversed(tuple(input_map.items())))
        reversed_model = Lag432CorridorVecchia(
            0.5,
            reversed_map,
            grid_coords,
        )
        with self.assertRaisesRegex(ValueError, "strictly chronological"):
            with contextlib.redirect_stdout(io.StringIO()):
                reversed_model.precompute_conditioning_sets()

        extra_grid = np.vstack([grid_coords, grid_coords[:1]])
        extra_grid_model = Lag432CorridorVecchia(
            0.5,
            input_map,
            extra_grid,
        )
        with self.assertRaisesRegex(ValueError, "grid_coords"):
            with contextlib.redirect_stdout(io.StringIO()):
                extra_grid_model.precompute_conditioning_sets()

        partial_design = {key: value[:, :5].clone() for key, value in input_map.items()}
        partial_model = Lag432CorridorVecchia(
            0.5,
            partial_design,
            grid_coords,
        )
        with self.assertRaisesRegex(ValueError, "either 4 columns or at least 11"):
            with contextlib.redirect_stdout(io.StringIO()):
                partial_model.precompute_conditioning_sets()

        mixed_time, _ = self._deterministic_input()
        mixed_time["0"][0, 3] = 0.5
        mixed_time_model = Lag432CorridorVecchia(
            0.5,
            mixed_time,
            grid_coords,
        )
        with self.assertRaisesRegex(ValueError, "exactly one time value"):
            mixed_time_model.precompute_conditioning_sets()

        rank_deficient, _ = self._deterministic_input()
        for values in rank_deficient.values():
            values[:, 4:11] = 0.0
            values[:, 4] = 1.0
        rank_deficient_model = Lag432CorridorVecchia(
            0.5,
            rank_deficient,
            grid_coords,
        )
        with self.assertRaisesRegex(ValueError, "mean design is rank deficient"):
            rank_deficient_model.precompute_conditioning_sets()

    def test_precompute_preserves_float64_and_gls_has_no_hidden_ridge(self) -> None:
        from GEMS_TCO.vecchia.corridor_neighbors.corridor_lag432 import Lag432CorridorVecchia

        input_map, grid_coords = self._deterministic_input()
        input_map["0"][0, 0] = 0.1234567890123
        model = Lag432CorridorVecchia(0.5, input_map, grid_coords)
        prepared = model._prepared_time_slices()
        self.assertEqual(prepared[0].dtype, torch.float64)
        self.assertEqual(float(prepared[0][0, 0]), 0.1234567890123)
        self.assertFalse(hasattr(model, "_gls_jitter"))

        converted = model.interpretable_parameters([0.0] * 7)
        self.assertEqual(converted["signal_variance"], 1.0)
        self.assertNotIn("sigma_sq", converted)

    def test_lbfgs_returns_restored_post_step_result_without_stdout(self) -> None:
        from GEMS_TCO.vecchia import LBFGSFitResult
        from GEMS_TCO.vecchia.corridor_neighbors.corridor_lag432 import Lag432CorridorVecchia

        input_map, grid_coords = self._deterministic_input()
        model = Lag432CorridorVecchia(
            0.5,
            input_map,
            grid_coords,
            target_chunk_size=16,
        )
        model.precompute_conditioning_sets()
        parameters = [
            torch.tensor(value, dtype=torch.float64, requires_grad=True)
            for value in (0.0, 0.0, 0.0, 0.0, 0.0, -0.126, -20.0)
        ]
        optimizer = model.make_lbfgs_optimizer(parameters, max_iter=1, history_size=3)
        batch_snapshot = tuple(
            (
                id(batch),
                batch.coordinates.clone(),
                batch.response.clone(),
                batch.design.clone(),
                batch.indices.clone(),
                batch.is_dummy.clone(),
            )
            for batch in model._cluster_batches
        )
        geometry_message = "conditioning geometry must remain fixed during optimization"
        with (
            mock.patch.object(
                model,
                "precompute_conditioning_sets",
                side_effect=AssertionError(geometry_message),
            ),
            mock.patch.object(
                model,
                "_build_clusters",
                side_effect=AssertionError(geometry_message),
            ),
            mock.patch.object(
                model,
                "_conditioning_blocks",
                side_effect=AssertionError(geometry_message),
            ),
            mock.patch.object(
                model,
                "_lag_candidates",
                side_effect=AssertionError(geometry_message),
            ),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            result = model.fit_lbfgs(parameters, optimizer, max_steps=1, grad_tol=0.0)
        self.assertEqual(output.getvalue(), "")
        self.assertIsInstance(result, LBFGSFitResult)
        self.assertEqual(result.steps_completed, 1)
        self.assertGreaterEqual(result.cache_hits, 3)
        self.assertGreaterEqual(result.objective_evaluations, 1)
        restored = torch.stack([parameter.detach() for parameter in parameters])
        restored_nll = float(model.profiled_negative_log_likelihood(restored))
        self.assertAlmostEqual(result.final_nll, restored_nll, places=12)
        self.assertEqual(
            tuple(id(batch) for batch in model._cluster_batches),
            tuple(snapshot[0] for snapshot in batch_snapshot),
        )
        for batch, snapshot in zip(model._cluster_batches, batch_snapshot):
            for current, expected in zip(
                (
                    batch.coordinates,
                    batch.response,
                    batch.design,
                    batch.indices,
                    batch.is_dummy,
                ),
                snapshot[1:],
            ):
                self.assertTrue(torch.equal(current, expected))

    def test_lbfgs_exact_state_cache_avoids_duplicate_objective_passes(self) -> None:
        from GEMS_TCO.vecchia._base import GroupedVecchiaBase

        class QuadraticVecchia(GroupedVecchiaBase):
            covariance_parameter_count = 1

            def __init__(self):
                super().__init__(
                    smooth=0.5,
                    input_map={"time": torch.zeros((1, 4), dtype=torch.float64)},
                )
                self.precompute_calls = 0
                self.objective_calls = 0

            def precompute_conditioning_sets(self):
                self.precompute_calls += 1
                self.is_precomputed = True
                return self

            def _profiled_nll_with_validity(self, params):
                self.objective_calls += 1
                loss = (params[0] - 2.0).pow(2)
                return loss, True

            def interpretable_parameters(self, raw):
                return {"value": float(raw[0])}

        class ScriptedLBFGS(torch.optim.LBFGS):
            def __init__(self, parameter):
                super().__init__([parameter], lr=1.0, max_iter=1)
                self.parameter = parameter

            def step(self, closure):
                initial_loss = closure()
                with torch.no_grad():
                    self.parameter.fill_(1.0)
                closure()
                return initial_loss

        model = QuadraticVecchia()
        parameter = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        result = model.fit_lbfgs(
            [parameter],
            ScriptedLBFGS(parameter),
            max_steps=1,
            grad_tol=0.0,
        )

        self.assertEqual(model.precompute_calls, 1)
        self.assertEqual(model.objective_calls, 2)
        self.assertEqual(result.objective_evaluations, 2)
        self.assertEqual(result.cache_hits, 3)
        self.assertEqual(float(parameter.detach()), 1.0)
        self.assertEqual(result.final_nll, 1.0)
        self.assertEqual(float(parameter.grad), -2.0)

    def test_lbfgs_cache_reuses_an_earlier_trial_that_is_later_accepted(self) -> None:
        from GEMS_TCO.vecchia._base import GroupedVecchiaBase

        class QuadraticVecchia(GroupedVecchiaBase):
            covariance_parameter_count = 1

            def __init__(self):
                super().__init__(
                    smooth=0.5,
                    input_map={"time": torch.zeros((1, 4), dtype=torch.float64)},
                )
                self.objective_calls = 0
                self.is_precomputed = True

            def _profiled_nll_with_validity(self, params):
                self.objective_calls += 1
                return (params[0] - 2.0).pow(2), True

            def interpretable_parameters(self, raw):
                return {"value": float(raw[0])}

        class BacktrackingLBFGS(torch.optim.LBFGS):
            def __init__(self, parameter):
                super().__init__([parameter], lr=1.0, max_iter=1)
                self.parameter = parameter

            def step(self, closure):
                initial_loss = closure()
                with torch.no_grad():
                    self.parameter.fill_(2.0)
                closure()
                with torch.no_grad():
                    self.parameter.fill_(3.0)
                closure()
                with torch.no_grad():
                    self.parameter.fill_(2.0)
                return initial_loss

        model = QuadraticVecchia()
        parameter = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        result = model.fit_lbfgs(
            [parameter],
            BacktrackingLBFGS(parameter),
            max_steps=1,
            grad_tol=0.0,
        )

        self.assertEqual(model.objective_calls, 3)
        self.assertEqual(result.objective_evaluations, 3)
        self.assertEqual(result.cache_hits, 3)
        self.assertEqual(float(parameter.detach()), 2.0)
        self.assertEqual(float(parameter.grad), 0.0)
        self.assertEqual(result.final_nll, 0.0)

    def test_lbfgs_cache_does_not_merge_nearby_parameter_states(self) -> None:
        from GEMS_TCO.vecchia._base import GroupedVecchiaBase

        class QuadraticVecchia(GroupedVecchiaBase):
            covariance_parameter_count = 1

            def __init__(self):
                super().__init__(
                    smooth=0.5,
                    input_map={"time": torch.zeros((1, 4), dtype=torch.float64)},
                )
                self.evaluated_parameters = []
                self.is_precomputed = True

            def _profiled_nll_with_validity(self, params):
                self.evaluated_parameters.append(float(params[0].detach()))
                return (params[0] - 2.0).pow(2), True

            def interpretable_parameters(self, raw):
                return {"value": float(raw[0])}

        class TinyStepLBFGS(torch.optim.LBFGS):
            def __init__(self, parameter):
                super().__init__([parameter], lr=1.0, max_iter=1)
                self.parameter = parameter

            def step(self, closure):
                initial_loss = closure()
                with torch.no_grad():
                    self.parameter.fill_(1.0e-12)
                return initial_loss

        model = QuadraticVecchia()
        parameter = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        result = model.fit_lbfgs(
            [parameter],
            TinyStepLBFGS(parameter),
            max_steps=1,
            grad_tol=0.0,
        )

        self.assertEqual(model.evaluated_parameters, [0.0, 1.0e-12])
        self.assertEqual(result.objective_evaluations, 2)
        self.assertEqual(result.cache_hits, 2)
        self.assertEqual(float(parameter.detach()), 1.0e-12)
        self.assertAlmostEqual(float(parameter.grad), -3.999999999998, places=12)

    def test_lbfgs_cache_restores_the_best_valid_state_and_gradient(self) -> None:
        from GEMS_TCO.vecchia._base import GroupedVecchiaBase

        class BoundedQuadraticVecchia(GroupedVecchiaBase):
            covariance_parameter_count = 1

            def __init__(self):
                super().__init__(
                    smooth=0.5,
                    input_map={"time": torch.zeros((1, 4), dtype=torch.float64)},
                )
                self.objective_calls = 0
                self.is_precomputed = True

            def _profiled_nll_with_validity(self, params):
                self.objective_calls += 1
                if abs(float(params[0].detach())) > 10.0:
                    return params.sum() * 0.0 + 1.0e10, False
                return (params[0] - 2.0).pow(2), True

            def interpretable_parameters(self, raw):
                return {"value": float(raw[0])}

        class InvalidatingLBFGS(torch.optim.LBFGS):
            def __init__(self, parameter):
                super().__init__([parameter], lr=1.0, max_iter=1)
                self.parameter = parameter

            def step(self, closure):
                initial_loss = closure()
                with torch.no_grad():
                    self.parameter.fill_(20.0)
                closure()
                return initial_loss

        model = BoundedQuadraticVecchia()
        parameter = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        result = model.fit_lbfgs(
            [parameter],
            InvalidatingLBFGS(parameter),
            max_steps=1,
            grad_tol=1.0e-8,
        )

        self.assertEqual(model.objective_calls, 2)
        self.assertEqual(result.objective_evaluations, 2)
        self.assertEqual(result.cache_hits, 3)
        self.assertEqual(float(parameter.detach()), 0.0)
        self.assertEqual(float(parameter.grad), -4.0)
        self.assertEqual(result.final_nll, 4.0)
        self.assertFalse(result.converged)

    def test_lbfgs_rejects_an_invalid_initial_objective(self) -> None:
        from GEMS_TCO.vecchia.corridor_neighbors.corridor_lag432 import Lag432CorridorVecchia

        input_map, grid_coords = self._deterministic_input()
        model = Lag432CorridorVecchia(
            0.5,
            input_map,
            grid_coords,
            target_chunk_size=16,
        )
        model.precompute_conditioning_sets()
        parameters = [
            torch.tensor(value, dtype=torch.float64, requires_grad=True)
            for value in (0.0, 0.0, 0.0, 0.0, 0.0, -0.126, -20.0)
        ]
        optimizer = model.make_lbfgs_optimizer(parameters, max_iter=1, history_size=3)

        with mock.patch.object(model, "_accumulate_gls_stats", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "initial covariance parameters"):
                model.fit_lbfgs(parameters, optimizer, max_steps=1)

    def test_nonfinite_responses_are_masked_and_empty_targets_are_rejected(self) -> None:
        from GEMS_TCO.vecchia.corridor_neighbors.corridor_lag432 import Lag432CorridorVecchia

        input_map, grid_coords = self._deterministic_input()
        input_map["0"][0, 2] = float("nan")
        input_map["0"][0, :2] = float("nan")
        input_map["1"][1, 2] = float("inf")
        input_map["1"][1, :2] = float("nan")
        model = Lag432CorridorVecchia(
            0.5,
            input_map,
            grid_coords,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            model.precompute_conditioning_sets()
        self.assertEqual(model.n_target_points, 3 * 64 - 2)
        for batch in model._cluster_batches:
            target_response = batch.response[:, batch.max_cond_points :, :]
            self.assertTrue(torch.isfinite(target_response).all())
            self.assertTrue(torch.isfinite(batch.coordinates).all())

        invalid_coordinates, _ = self._deterministic_input()
        invalid_coordinates["0"][0, 0] = float("nan")
        observed_nan_model = Lag432CorridorVecchia(
            0.5,
            invalid_coordinates,
            grid_coords,
        )
        with self.assertRaisesRegex(ValueError, "observed responses require finite"):
            with contextlib.redirect_stdout(io.StringIO()):
                observed_nan_model.precompute_conditioning_sets()

        for values in input_map.values():
            values[:, 2] = float("nan")
        empty_model = Lag432CorridorVecchia(
            0.5,
            input_map,
            grid_coords,
        )
        with self.assertRaisesRegex(ValueError, "no valid target observations"):
            with contextlib.redirect_stdout(io.StringIO()):
                empty_model.precompute_conditioning_sets()

    def test_no_nugget_variants_use_six_parameters_and_report_zero(self) -> None:
        from GEMS_TCO.vecchia.corridor_neighbors.generalized_cauchy import (
            NoNuggetGeneralizedCauchyLag643CorridorVecchia,
        )
        from GEMS_TCO.vecchia.corridor_neighbors.spline import (
            NoNuggetSplineMaternLag643CorridorVecchia,
        )

        input_map, _ = self._deterministic_input()
        models = (
            NoNuggetGeneralizedCauchyLag643CorridorVecchia(
                1.0,
                1.0,
                input_map,
            ),
            NoNuggetSplineMaternLag643CorridorVecchia(
                0.7,
                input_map,
                spline_n_points=20,
            ),
        )
        for model in models:
            with self.subTest(model=type(model).__name__):
                self.assertEqual(model.covariance_parameter_count, 6)
                self.assertEqual(model.interpretable_parameters([0.0] * 6)["nugget"], 0.0)
                with self.assertLogs("GEMS_TCO.vecchia._base", level="WARNING") as output:
                    model._log_cholesky_failure(torch.zeros(6), "test")
                self.assertIn("nugget=0.0000e+00", output.output[0])

    def test_dummy_padding_is_exactly_decoupled_for_long_tail_kernels(self) -> None:
        from GEMS_TCO.vecchia.grouped_batched import GroupedBatchedVecchia

        covariance = torch.tensor(
            [
                [
                    [2.0, 0.4, 0.3, 0.2],
                    [0.4, 2.0, 0.5, 0.1],
                    [0.3, 0.5, 2.0, 0.6],
                    [0.2, 0.1, 0.6, 2.0],
                ]
            ],
            dtype=torch.float64,
            requires_grad=True,
        )
        is_dummy = torch.tensor(
            [[[True], [False], [True], [False]]],
            dtype=torch.bool,
        )
        decoupled = GroupedBatchedVecchia._decouple_dummy_covariance(
            covariance,
            is_dummy,
        )

        expected = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0, 0.1],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.1, 0.0, 2.0],
                ]
            ],
            dtype=torch.float64,
        )
        torch.testing.assert_close(decoupled, expected)
        decoupled.sum().backward()
        self.assertEqual(float(covariance.grad[0, 0, 1]), 0.0)
        self.assertEqual(float(covariance.grad[0, 1, 3]), 1.0)

    def test_family_specific_parameter_validation(self) -> None:
        from GEMS_TCO.vecchia.corridor_neighbors.generalized_cauchy import (
            GeneralizedCauchyLag643CorridorVecchia,
        )
        from GEMS_TCO.vecchia.corridor_neighbors.spline import SplineMaternLag643CorridorVecchia

        input_map, _ = self._deterministic_input()
        for alpha, beta in ((0.0, 1.0), (2.1, 1.0), (float("nan"), 1.0), (1.0, float("nan"))):
            with self.subTest(alpha=alpha, beta=beta):
                with self.assertRaises(ValueError):
                    GeneralizedCauchyLag643CorridorVecchia(
                        alpha,
                        beta,
                        input_map,
                    )

        invalid_spline_options = (
            {"smooth": float("nan")},
            {"smooth": 0.7, "spline_n_points": 1},
            {"smooth": 0.7, "spline_r_max": 0.0},
        )
        for options in invalid_spline_options:
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    SplineMaternLag643CorridorVecchia(
                        input_map=input_map,
                        **options,
                    )

    def test_spline_correlation_is_zero_beyond_its_table(self) -> None:
        from GEMS_TCO.vecchia.corridor_neighbors.spline import SplineMaternLag643CorridorVecchia

        input_map, _ = self._deterministic_input()
        model = SplineMaternLag643CorridorVecchia(
            smooth=0.7,
            input_map=input_map,
            spline_n_points=30,
            spline_r_max=4.0,
        )
        distances = torch.tensor([0.0, 4.0, 4.01, 100.0], dtype=torch.float64)
        correlations = model._correlation(distances)
        self.assertEqual(float(correlations[0]), 1.0)
        self.assertGreaterEqual(float(correlations[1]), 0.0)
        torch.testing.assert_close(correlations[2:], torch.zeros(2, dtype=torch.float64))

    def test_matern_range_uses_standard_sqrt_2nu_convention(self) -> None:
        from GEMS_TCO.vecchia.corridor_neighbors.spline import SplineMaternLag643CorridorVecchia
        from GEMS_TCO.vecchia.grouped_batched import GroupedBatchedVecchia

        input_map, _ = self._deterministic_input()
        closed_form = GroupedBatchedVecchia(
            smooth=1.5,
            input_map=input_map,
        )
        model = SplineMaternLag643CorridorVecchia(
            smooth=1.5 + 1e-4,
            input_map=input_map,
            spline_n_points=2000,
            spline_r_max=6.0,
        )
        distances = torch.tensor([0.0, 0.25, 0.5, 1.0, 2.0, 4.0], dtype=torch.float64)
        bessel_argument = np.sqrt(3.0) * distances
        expected = (1.0 + bessel_argument) * torch.exp(-bessel_argument)
        torch.testing.assert_close(closed_form._correlation(distances), expected)
        actual = model._correlation(distances)
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
