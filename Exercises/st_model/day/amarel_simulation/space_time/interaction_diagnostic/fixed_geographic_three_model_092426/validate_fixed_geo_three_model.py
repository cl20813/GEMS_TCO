#!/usr/bin/env python3
"""Fast preflight checks for the frozen design, formulas, and date manifest."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[6]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from GEMS_TCO.vecchia.corridor_neighbors.separable_exponential import (
    NoNuggetAdvectedSeparableExponentialLag643CorridorVecchia,
)

from fixed_geo_three_model_core import (
    contrast_covariances,
    expected_gaussian_score,
    load_json,
    load_toml,
    physical_to_raw_parameters,
    prepare_fixed_contrasts,
    score_model,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, default=HERE / "fixed_geo_three_model.toml")
    result.add_argument("--design", type=Path, default=HERE / "frozen_design.json")
    result.add_argument("--dates", type=Path, default=HERE / "evaluation_dates.csv")
    result.add_argument("--check-data", action="store_true")
    result.add_argument("--data-root", type=Path)
    return result


def _synthetic_frames(source_latitude_drift: float = 0.0) -> dict[str, pd.DataFrame]:
    latitudes = np.arange(-3.0, 2.0001, 0.25)
    longitudes = np.arange(121.0, 131.0001, 0.25)
    latitude, longitude = np.meshgrid(latitudes, longitudes, indexing="ij")
    regular_latitude = latitude.ravel()
    regular_longitude = longitude.ravel()
    frames = {}
    for time_index in range(8):
        # A common shift at each time is removed by the spatial difference.
        # A small time-varying spatial warp instead exercises the nonzero-WX
        # branch that real source geolocation can induce.
        source_latitude = regular_latitude * (1.0 + source_latitude_drift * time_index)
        response = 300.0 + 0.7 * source_latitude + 0.4 * time_index + np.sin(regular_longitude)
        frames[f"slot_{time_index}"] = pd.DataFrame(
            {
                "Latitude": regular_latitude,
                "Longitude": regular_longitude,
                "Source_Latitude": source_latitude,
                "Source_Longitude": regular_longitude,
                "ColumnAmountO3": response,
                "Hours_elapsed": np.repeat(477700.0 + time_index, len(response)),
            }
        )
    return frames


def _check_manifest(dates: pd.DataFrame) -> None:
    if len(dates) != 60:
        raise AssertionError(f"expected 60 design-held-out days, found {len(dates)}")
    if dates["task_id"].tolist() != list(range(60)):
        raise AssertionError("task ids must be contiguous 0..59")
    date_values = set(dates["date"].astype(str))
    if "2024-07-01" in date_values:
        raise AssertionError("discovery date entered the design-held-out manifest")
    if "2025-07-24" in date_values:
        raise AssertionError("the seven-slot 2025-07-24 date entered the manifest")
    if not {"2024-07-31", "2025-07-31"}.issubset(date_values):
        raise AssertionError("both complete July 31 dates must be included")
    if not np.all(dates["expected_slots"].to_numpy() == 8):
        raise AssertionError("every design-held-out date must require eight time slots")


def _check_amarel_compute_settings(config: dict) -> None:
    models = config["models"]
    if models["conditioning_geometry"] != "corridor_width_4x4_lag643":
        raise AssertionError("the Amarel study must use the frozen 6/4/3 corridor geometry")
    if models["conditioning_lag_pattern"] != "6/4/3":
        raise AssertionError("the Amarel study must record conditioning_lag_pattern=6/4/3")
    if int(models["target_chunk_size"]) != 256:
        raise AssertionError("the Amarel lag-6/4/3 study requires target_chunk_size=256")


def _check_frozen_design(design: dict) -> None:
    if design["design_id"] != "fixed_geographic_ab_mirrors_lag1_v1_092426":
        raise AssertionError("unexpected frozen-design identifier")
    if design["coordinate_frame"] != "fixed_geographic":
        raise AssertionError("primary coordinate frame is not fixed geographic")
    if design["observation_geometry_moves_with_advection"]:
        raise AssertionError("the observation geometry must not follow advection")
    coefficient_a = np.asarray(design["contrast_coefficients_on_eight_points"]["Q_A"])
    coefficient_b = np.asarray(design["contrast_coefficients_on_eight_points"]["Q_B"])
    np.testing.assert_array_equal(coefficient_a, [1, -1, -1, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(coefficient_b, [0, 0, 0, 0, 1, -1, -1, 1])
    ranges = np.asarray([design["frozen_ranges"]["latitude"], design["frozen_ranges"]["longitude"]])
    for handedness, standardized in design["standardized_geometry"].items():
        physical = np.asarray(design["physical_offsets"][handedness])
        np.testing.assert_allclose(physical, np.asarray(standardized) * ranges, atol=0, rtol=1e-15)


def _check_mean_audit(design: dict, tolerance: float) -> None:
    synthetic_design = copy.deepcopy(design)
    synthetic_design.pop("expected_anchor_count_by_mirror", None)
    synthetic_design.pop("expected_anchor_mapping_sha256", None)
    synthetic_design.pop("expected_regular_grid_shape", None)
    synthetic_design.pop("expected_regular_grid_axes_sha256", None)
    synthetic_design.pop("expected_regular_grid_extent", None)
    frames = _synthetic_frames(source_latitude_drift=0.0)
    samples, coordinates, audit = prepare_fixed_contrasts(
        frames,
        monthly_mean=300.0,
        frozen_design=synthetic_design,
        wx_tolerance=tolerance,
    )
    if not audit["wx_numerically_zero"]:
        raise AssertionError("static source coordinates should give W X = 0")
    if audit["empirical_mean_rule"] != "monthly_centered_raw_y_because_wx_is_zero":
        raise AssertionError("raw-Y branch was not selected after a zero W X audit")
    if len(samples) == 0 or coordinates.shape != (len(samples), 8, 3):
        raise AssertionError("synthetic frozen contrasts have the wrong shape")

    drifting = _synthetic_frames(source_latitude_drift=1e-3)
    _, _, drift_audit = prepare_fixed_contrasts(
        drifting,
        monthly_mean=300.0,
        frozen_design=synthetic_design,
        wx_tolerance=tolerance,
    )
    if drift_audit["wx_numerically_zero"]:
        raise AssertionError("source-coordinate drift should be detected by W X")
    if drift_audit["empirical_mean_rule"] != "one_common_daywise_ols_mean_for_all_models":
        raise AssertionError("nonzero W X did not trigger the common mean rule")


def _check_covariance_formulas(design: dict) -> None:
    physical = {
        "signal_variance": 2.0,
        "range_lat": 0.8,
        "range_lon": 1.1,
        "range_time": 1.7,
        "advec_lat": 0.04,
        "advec_lon": -0.13,
        "nugget": 0.0,
    }
    rng = np.random.default_rng(92426)
    coordinates = rng.normal(size=(7, 8, 3))
    joint = contrast_covariances(coordinates, physical, "matern05", design)
    separable = contrast_covariances(coordinates, physical, "separable", design)
    if np.allclose(joint, separable):
        raise AssertionError("generic mixed lags must distinguish joint and separable models")
    self_score = expected_gaussian_score(joint, joint)
    misspecified_score = expected_gaussian_score(joint, separable)
    if np.any(self_score > misspecified_score + 1e-12):
        raise AssertionError("the population Gaussian score failed its propriety check")

    input_rows = torch.zeros((4, 11), dtype=torch.float64)
    source_model = NoNuggetAdvectedSeparableExponentialLag643CorridorVecchia(
        input_map={"slot": input_rows}
    )
    raw = torch.tensor(
        physical_to_raw_parameters(physical, include_nugget=False), dtype=torch.float64
    )
    torch_coordinates = torch.tensor(coordinates, dtype=torch.float64)
    full_covariance = source_model.batched_covariance(raw, torch_coordinates).numpy()
    full_covariance -= np.eye(8, dtype=np.float64)[None, :, :] * 1e-6
    coefficient = np.stack(
        [
            np.asarray(design["contrast_coefficients_on_eight_points"]["Q_A"]),
            np.asarray(design["contrast_coefficients_on_eight_points"]["Q_B"]),
        ]
    )
    source_contrast = np.einsum(
        "ai,nij,bj->nab", coefficient, full_covariance, coefficient, optimize=True
    )
    np.testing.assert_allclose(source_contrast, separable, atol=5e-14, rtol=5e-14)


def _check_score_summaries(design: dict) -> None:
    samples = pd.DataFrame(
        {
            "sample_id": [0, 1, 2],
            "q_a": [1.0, -2.0, 0.5],
            "q_b": [0.25, 1.5, -0.75],
        }
    )
    covariance = np.repeat(np.asarray([[[2.0, 0.4], [0.4, 1.5]]]), 3, axis=0)
    _, summary = score_model(samples, covariance, "gc", design)
    np.testing.assert_allclose(
        summary["empirical_second_moment_q_a"], np.mean(np.square(samples["q_a"]))
    )
    np.testing.assert_allclose(
        summary["empirical_cross_moment_q_a_q_b"],
        np.mean(samples["q_a"] * samples["q_b"]),
    )
    np.testing.assert_allclose(summary["model_rho_ab_pooled"], 0.4 / np.sqrt(2.0 * 1.5))
    if "model_rho_ab_mean" in summary or "empirical_v_a" in summary:
        raise AssertionError("ambiguous legacy moment names re-entered the score summary")


def _check_data_files(config: dict, dates: pd.DataFrame, data_root: Path) -> None:
    for year, group in dates.groupby("year"):
        path = data_root / f"pickle_{int(year)}" / f"tco_grid_{str(int(year))[2:]}_07.pkl"
        if not path.is_file():
            raise FileNotFoundError(f"expected monthly grid file is missing: {path}")
        print(f"Checking monthly grid: {path}", flush=True)
        loaded = pd.read_pickle(path)
        keys = list(loaded)
        for row in group.itertuples(index=False):
            token = f"day{int(row.day):02d}_"
            count = sum(token in str(key) for key in keys)
            if count != int(row.expected_slots):
                raise AssertionError(
                    f"{row.date}: found {count} slots, expected {row.expected_slots}"
                )


def main() -> None:
    args = parser().parse_args()
    config = load_toml(args.config.expanduser().resolve())
    design = load_json(args.design.expanduser().resolve())
    dates = pd.read_csv(args.dates.expanduser().resolve())
    if tuple(config["models"]["names"]) != ("gc", "matern05", "separable"):
        raise AssertionError("the frozen three-model order changed")
    if config["models"]["nugget_policy"] != "fixed_zero":
        raise AssertionError("all three primary models must use the same zero-nugget policy")
    if int(config["models"]["covariance_parameter_count"]) != 6:
        raise AssertionError("all three fixed-shape, zero-nugget models must have six parameters")
    if not bool(config["fit"]["require_convergence"]):
        raise AssertionError("nonconverged daily fits must not enter the primary aggregate")
    _check_manifest(dates)
    _check_amarel_compute_settings(config)
    _check_frozen_design(design)
    _check_mean_audit(design, float(config["diagnostic"]["wx_tolerance"]))
    _check_covariance_formulas(design)
    _check_score_summaries(design)
    if args.check_data:
        if args.data_root is not None:
            data_root = args.data_root
        else:
            amarel_root = Path(config["data"]["amarel_root"])
            data_root = (
                amarel_root
                if amarel_root.exists()
                else Path(config["data"]["local_root"])
            )
        data_root = data_root.expanduser().resolve()
        _check_data_files(config, dates, data_root)
    print("All fixed-geographic three-model preflight checks passed.")


if __name__ == "__main__":
    main()
