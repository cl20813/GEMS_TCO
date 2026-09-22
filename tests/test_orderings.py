from __future__ import annotations

import unittest

import numpy as np

from GEMS_TCO.orderings import maxmin_order, predecessor_neighbors


class MaxminOrderingTests(unittest.TestCase):
    def test_order_is_a_valid_greedy_maxmin_permutation(self):
        locations = np.array(
            [[0.0, 0.0], [0.2, 1.1], [1.5, 0.4], [2.0, 2.5], [4.0, 0.1]],
            dtype=np.float64,
        )
        order = maxmin_order(locations)

        np.testing.assert_array_equal(np.sort(order), np.arange(len(locations)))
        expected_first = np.argmin(np.linalg.norm(locations - locations.mean(axis=0), axis=1))
        self.assertEqual(order[0], expected_first)

        selected = [int(order[0])]
        for chosen in order[1:]:
            candidates = [index for index in range(len(locations)) if index not in selected]
            minimum_distances = {
                index: min(
                    np.linalg.norm(locations[index] - locations[previous]) for previous in selected
                )
                for index in candidates
            }
            self.assertAlmostEqual(
                minimum_distances[int(chosen)], max(minimum_distances.values()), places=12
            )
            selected.append(int(chosen))

    def test_noncontiguous_input_is_handled_safely(self):
        base = np.arange(30, dtype=np.float64).reshape(5, 6)
        view = base[:, ::2]
        self.assertFalse(view.flags.c_contiguous)
        np.testing.assert_array_equal(maxmin_order(view), maxmin_order(np.ascontiguousarray(view)))

    def test_large_coordinate_scale_does_not_change_greedy_definition(self):
        locations = np.array(
            [[0.0], [30_000.0], [70_000.0], [120_000.0], [200_000.0]],
            dtype=np.float64,
        )
        order = maxmin_order(locations)
        selected = [int(order[0])]
        for chosen in order[1:]:
            candidates = [index for index in range(len(locations)) if index not in selected]
            distances = {
                index: min(
                    np.linalg.norm(locations[index] - locations[previous]) for previous in selected
                )
                for index in candidates
            }
            self.assertAlmostEqual(distances[int(chosen)], max(distances.values()))
            selected.append(int(chosen))

    def test_large_common_offset_is_removed_before_native_calculation(self):
        baseline = np.array([[0.0], [1.0], [4.0], [9.0]], dtype=np.float64)
        offset = baseline + 1.0e12
        np.testing.assert_array_equal(maxmin_order(offset), maxmin_order(baseline))

    def test_invalid_locations_are_rejected(self):
        for locations in (np.array([]), np.empty((0, 2)), np.array([[0.0, np.nan]])):
            with self.subTest(shape=locations.shape):
                with self.assertRaises(ValueError):
                    maxmin_order(locations)


class PredecessorNeighborTests(unittest.TestCase):
    def test_exact_neighbors_are_sorted_by_distance_and_precede_target(self):
        locations = np.array([[0.0], [4.0], [1.0], [10.0], [3.0]], dtype=np.float64)
        expected = np.array([[-1, -1], [0, -1], [0, 1], [1, 2], [1, 2]], dtype=np.int64)
        actual = predecessor_neighbors(locations, max_neighbors=2)
        np.testing.assert_array_equal(actual, expected)
        for target, row in enumerate(actual):
            self.assertTrue(np.all(row[row >= 0] < target))

    def test_duplicate_locations_still_obey_predecessor_constraint(self):
        locations = np.array([[0.0], [0.0], [1.0], [0.0]], dtype=np.float64)
        neighbors = predecessor_neighbors(locations, max_neighbors=3)
        for target, row in enumerate(neighbors):
            self.assertTrue(np.all(row[row >= 0] < target))

    def test_equal_distance_neighbors_use_index_tie_break(self):
        locations = np.array([[-1.0], [1.0], [0.0]], dtype=np.float64)
        neighbors = predecessor_neighbors(locations, max_neighbors=1)
        self.assertEqual(neighbors[2, 0], 0)

    def test_empty_and_zero_neighbor_cases_have_stable_shapes(self):
        self.assertEqual(predecessor_neighbors(np.empty((0, 2)), 3).shape, (0, 3))
        self.assertEqual(predecessor_neighbors(np.ones((4, 2)), 0).shape, (4, 0))

    def test_invalid_neighbor_count_is_rejected(self):
        with self.assertRaises(TypeError):
            predecessor_neighbors(np.ones((2, 1)), max_neighbors=True)
        with self.assertRaises(ValueError):
            predecessor_neighbors(np.ones((2, 1)), max_neighbors=-1)


if __name__ == "__main__":
    unittest.main()
