from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
MODULE_DIR = HERE.parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from analyze_negative_two_rectangle import (  # noqa: E402
    pair_quadratic_attribution,
    quadratic_components,
)


def test_quadratic_components_sum_to_direct_form() -> None:
    matrix = np.asarray([[0.2, 1.7], [1.7, 0.4]])
    coefficients = np.asarray([0.8, -0.6])
    frame = quadratic_components(
        matrix,
        coefficients,
        matrix_name="example",
    )

    expected = float(coefficients @ matrix @ coefficients)
    component_sum = float(frame.loc[frame["component"] != "total", "value"].sum())
    reported_total = float(frame.loc[frame["component"] == "total", "value"].iloc[0])
    np.testing.assert_allclose(component_sum, expected, atol=1.0e-14)
    np.testing.assert_allclose(reported_total, expected, atol=1.0e-14)


def test_pair_attribution_sums_ordered_quadratic_form() -> None:
    weights = np.asarray([0.7, -0.4, 0.0, 0.2])
    difference = np.asarray(
        [
            [0.0, 0.5, 0.1, -0.2],
            [0.5, 0.0, -0.3, 0.4],
            [0.1, -0.3, 0.0, 0.2],
            [-0.2, 0.4, 0.2, 0.0],
        ]
    )
    points = pd.DataFrame(
        {
            "anchor_index": [0, 1, 2, 3],
            "time_index": [0, 1, 2, 3],
            "standardized_moving_latitude": [-1.0, -0.5, 0.5, 1.0],
            "standardized_moving_longitude": [-1.0, 0.0, 0.0, 1.0],
        }
    )

    frame, ordered_matrix, labels = pair_quadratic_attribution(
        weights,
        difference,
        points,
        contribution_column="quadratic_contribution",
    )
    expected = float(weights @ difference @ weights)

    assert labels == ["a00@t0", "a01@t1", "a03@t3"]
    np.testing.assert_allclose(frame["quadratic_contribution"].sum(), expected)
    np.testing.assert_allclose(ordered_matrix.sum(), expected)
    assert set(frame["multiplicity"]) == {1, 2}
