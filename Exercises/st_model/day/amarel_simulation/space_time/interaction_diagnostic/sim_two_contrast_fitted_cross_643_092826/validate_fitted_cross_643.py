#!/usr/bin/env python3
"""Local no-fit validation for the fitted cross-term Amarel study."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import fitted_cross_643_core as core


HERE = Path(__file__).resolve().parent


def main() -> None:
    config = json.loads((HERE / "fitted_cross_643_config.json").read_text())
    assert config["fit"]["lag_pattern"] == "6/4/3"
    assert config["fit"]["conditioning_geometry"].startswith("direction_adapted")
    assert config["fit"]["nugget"] == 0.0
    assert core.NoNuggetDirectionalMatern05Lag643.covariance_parameter_count == 6
    assert core.NoNuggetDirectionalGeneralizedCauchyLag643.covariance_parameter_count == 6
    ranges = core.effective_to_kernel_ranges("generalized_cauchy", config)
    scale = (math.exp(1.0 / 5.0) - 1.0)
    assert np.isclose(ranges["range_lon"], 0.3 / scale)
    raw = core.physical_to_raw(10.0, 0.2, 0.3, 2.0, 0.08, -0.2)
    assert raw.shape == (6,) and np.isfinite(raw).all()

    design_path = (HERE.parent / "sim_two_contrast_cross_center_092726" / "two_contrast_diagnostic_092726.json")
    design = json.loads(design_path.read_text())
    assert core.temporal_lag(design) == 1
    coordinates = np.asarray(
        [[
            [-0.4, -0.6, 0], [0.4, 0.6, 0],
            [-0.32, -0.8, 1], [0.48, 0.4, 1],
            [-0.2, -0.6, 0], [0.2, 0.6, 0],
            [-0.12, -0.8, 1], [0.28, 0.4, 1],
        ]],
        dtype=float,
    )
    physical = {
        "signal_variance": 10.0,
        "range_lat": 0.2,
        "range_lon": 0.3,
        "range_time": 2.0,
        "advec_lat": 0.08,
        "advec_lon": -0.2,
    }
    covariance = core.fitted_contrast_covariances(
        coordinates, physical, "matern", design, config
    )
    assert covariance.shape == (1, 2, 2)
    assert np.allclose(covariance, np.swapaxes(covariance, 1, 2))
    assert np.linalg.eigvalsh(covariance[0]).min() > 0
    samples = {"q_a": np.asarray([2.0, -1.0]), "q_b": np.asarray([3.0, 4.0])}
    covariance_pair = np.repeat(covariance, 2, axis=0)
    summary = core.summarize_fitted_cross(samples, covariance_pair, design)
    assert np.isclose(
        summary["empirical_minus_fitted_h_ab"], 1.0 - covariance[0, 0, 1]
    )
    assert np.isclose(
        summary["empirical_minus_fitted_l_variance"],
        summary["empirical_minus_fitted_l_diagonal_a"]
        + summary["empirical_minus_fitted_l_diagonal_b"]
        + summary["empirical_minus_fitted_l_cross_term"],
        rtol=0.0,
        atol=1e-14,
    )
    assert abs(summary["variance_residual_decomposition_error"]) < 1e-14
    strata = core.stratum_summaries(
        {
            **samples,
            "handedness": np.asarray(["clockwise", "clockwise"]),
            "time_t": np.asarray([0, 0]),
        },
        covariance_pair,
        design,
    )
    assert strata[0]["time_t_plus_1"] == 1
    print("PASS: adapted 6/4/3 fitting and exact Var(L) residual decomposition")


if __name__ == "__main__":
    main()
