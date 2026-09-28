#!/usr/bin/env python3
"""Validate a generated controlled space-time interaction simulation suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE / "st_interaction_july2024_nugget0_092726"
DEFAULT_CONFIG = HERE / "st_interaction_scenarios_092726.json"
GENERATOR = HERE / "generate_st_interaction_suite_092726.py"
EXPECTED_SCENARIOS = (
    "matern05_lagrangian_separable_eta0",
    "matern05_lagrangian_mixture_eta0p5",
    "matern05_lagrangian_joint_eta1",
    "gencauchy_a1_b5_lagrangian_separable_eta0",
    "gencauchy_a1_b5_lagrangian_mixture_eta0p5",
    "gencauchy_a1_b5_lagrangian_joint_eta1",
)
SHARED_FIELDS = (
    "sigmasq",
    "range_lat",
    "range_lon",
    "range_time",
    "advec_lat",
    "advec_lon",
    "nugget",
    "mean_intercept",
    "mean_lat_slope",
    "mean_lat_center",
    "selected_dates",
    "n_independent_days",
    "n_hours",
    "lat_factor_hr",
    "lon_factor_hr",
    "hours_per_day",
    "common_random_numbers",
    "grid_axis_dates",
    "simulation_grid",
    "delta_latitude",
    "delta_longitude",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--expected-days", type=int, default=30)
    parser.add_argument("--deep", action="store_true", help="load both pickle maps and verify all keys")
    parser.add_argument(
        "--allow-nonproduction-grid",
        action="store_true",
        help="allow reduced-resolution local engineering tests",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def year_dir(root: Path, scenario: str) -> Path:
    return root / scenario / "2024_july_st_circulant"


def load_truth(root: Path, scenario: str) -> tuple[dict[str, Any], dict[str, Any], Path]:
    directory = year_dir(root, scenario)
    truth_path = directory / "sim_july2024_st_circulant_truth.json"
    complete_path = directory / "COMPLETE.json"
    if not truth_path.is_file() or not complete_path.is_file():
        raise FileNotFoundError(f"incomplete scenario: {directory}")
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if truth["run_request_sha256"] != complete["run_request_sha256"]:
        raise AssertionError(f"request hash mismatch: {scenario}")
    return truth, complete, directory


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    current_config_sha256 = sha256_file(config_path)
    current_generator_sha256 = sha256_file(GENERATOR)
    records: list[dict[str, Any]] = []
    truths: dict[str, dict[str, Any]] = {}
    reference_shared: dict[str, Any] | None = None
    reference_seeds: list[int] | None = None
    reference_embedding_shape: list[int] | None = None
    reference_input_sha256: str | None = None
    for scenario in EXPECTED_SCENARIOS:
        truth, complete, directory = load_truth(root, scenario)
        truths[scenario] = truth
        if truth["scenario_id"] != scenario:
            raise AssertionError(f"scenario ID mismatch under {directory}")
        request = truth["run_request"]
        if request["scenario_config_sha256"] != current_config_sha256:
            raise AssertionError(f"scenario config provenance mismatch: {scenario}")
        if request["generator_sha256"] != current_generator_sha256:
            raise AssertionError(f"generator provenance mismatch: {scenario}")
        input_sha256 = str(request["input_template_sha256"])
        if reference_input_sha256 is None:
            reference_input_sha256 = input_sha256
        elif input_sha256 != reference_input_sha256:
            raise AssertionError(f"input template provenance differs: {scenario}")
        if float(truth["nugget"]) != 0.0:
            raise AssertionError(f"nugget is not exactly zero: {scenario}")
        fitter = truth["fitter_truth_parameters"]
        if truth["family"] == "generalized_cauchy":
            a = float(truth["cauchy_a"])
            b = float(truth["cauchy_b"])
            radial_at_efold = (
                1.0
                + (float(truth["range_lat"]) / float(fitter["range_lat"])) ** a
            ) ** (-b / a)
        else:
            radial_at_efold = math.exp(
                -float(truth["range_lat"]) / float(fitter["range_lat"])
            )
        if not math.isclose(radial_at_efold, math.exp(-1.0), abs_tol=1e-14):
            raise AssertionError(f"fitter range adapter is inconsistent: {scenario}")
        if int(truth["n_independent_days"]) != args.expected_days:
            raise AssertionError(
                f"{scenario} has {truth['n_independent_days']} days, expected {args.expected_days}"
            )
        if int(truth["n_hours"]) != 8 * args.expected_days:
            raise AssertionError(f"wrong hour count: {scenario}")
        if int(complete["n_hours"]) != int(truth["n_hours"]):
            raise AssertionError(f"COMPLETE hour count differs from truth: {scenario}")
        if not args.allow_nonproduction_grid:
            expected_axis_dates = [
                value.strftime("%Y-%m-%d")
                for value in pd.date_range("2024-07-01", periods=30, freq="D")
            ]
            expected_selected_dates = expected_axis_dates[: args.expected_days]
            if int(truth["lat_factor_hr"]) != 100 or int(truth["lon_factor_hr"]) != 10:
                raise AssertionError(f"not a production x100/x10 grid: {scenario}")
            if int(request["axis_n_days"]) != 30:
                raise AssertionError(f"production axes do not use 30 dates: {scenario}")
            if list(truth["grid_axis_dates"]) != expected_axis_dates:
                raise AssertionError(f"production axis dates are wrong: {scenario}")
            if list(truth["selected_dates"]) != expected_selected_dates:
                raise AssertionError(f"generated dates are wrong: {scenario}")
        shared = {field: truth[field] for field in SHARED_FIELDS}
        if reference_shared is None:
            reference_shared = shared
        elif shared != reference_shared:
            raise AssertionError(f"shared design differs: {scenario}")
        seeds = [int(value["seed"]) for value in truth["day_summaries"]]
        if reference_seeds is None:
            reference_seeds = seeds
        elif seeds != reference_seeds:
            raise AssertionError(f"common-random-number seeds differ: {scenario}")
        embedding = truth["embedding_diagnostics"]
        embedding_shape = [int(value) for value in embedding["embedding_shape"]]
        if reference_embedding_shape is None:
            reference_embedding_shape = embedding_shape
        elif embedding_shape != reference_embedding_shape:
            raise AssertionError(f"embedding shape differs: {scenario}")
        if float(embedding["spectrum_max_abs_imaginary_over_max_abs_real"]) > 1e-10:
            raise AssertionError(f"imaginary spectrum leakage: {scenario}")
        maximum = float(truth["run_request"]["max_negative_spectral_mass"])
        if float(embedding["spectrum_negative_mass_fraction"]) > maximum:
            raise AssertionError(f"excess spectral clipping: {scenario}")
        maximum_distortion = float(
            truth["run_request"]["max_relative_covariance_distortion"]
        )
        if (
            float(embedding["relative_covariance_distortion_bound"])
            > maximum_distortion
        ):
            raise AssertionError(f"excess covariance distortion bound: {scenario}")
        if not math.isclose(
            float(embedding["variance_after_renormalization"]),
            float(truth["sigmasq"]),
            abs_tol=1e-10,
        ):
            raise AssertionError(f"variance normalization failed: {scenario}")
        manifest = pd.read_csv(directory / "sim_july2024_st_circulant_manifest.csv")
        if len(manifest) != 8 * args.expected_days:
            raise AssertionError(f"manifest row count failed: {scenario}")
        if manifest.groupby("date").size().tolist() != [8] * args.expected_days:
            raise AssertionError(f"manifest is not eight hours per date: {scenario}")
        max_lat_mapping_error = float(
            manifest["max_abs_latitude_mapping_error"].max()
        )
        max_lon_mapping_error = float(
            manifest["max_abs_longitude_mapping_error"].max()
        )
        if max_lat_mapping_error > float(truth["delta_latitude"]) / 2.0 + 1e-10:
            raise AssertionError(f"latitude lattice mapping failed: {scenario}")
        if max_lon_mapping_error > float(truth["delta_longitude"]) / 2.0 + 1e-10:
            raise AssertionError(f"longitude lattice mapping failed: {scenario}")
        if args.deep:
            real = pd.read_pickle(directory / "sim_july2024_st_circulant_real_locations.pkl")
            grid = pd.read_pickle(directory / "sim_july2024_st_circulant_gridded.pkl")
            if set(real) != set(grid) or len(real) != 8 * args.expected_days:
                raise AssertionError(f"pickle key contract failed: {scenario}")
            manifest_keys = set(manifest["hour_key"].astype(str))
            if set(real) != manifest_keys:
                raise AssertionError(f"manifest/pickle keys differ: {scenario}")
            expected_local = list(range(int(truth["hours_per_day"])))
            for date, group in manifest.groupby("date", sort=True):
                if sorted(group["local_time"].astype(int).tolist()) != expected_local:
                    raise AssertionError(
                        f"local times are not 0..7 for {scenario}, {date}"
                    )
                if group["block_index"].nunique() != 1:
                    raise AssertionError(f"block index varies within {scenario}, {date}")
            required = {
                "Simulation_Block",
                "Simulation_Time_Index",
                "Hours_elapsed",
                "Template_Hours_elapsed",
                "ColumnAmountO3",
                "Source_Latitude",
                "Source_Longitude",
            }
            indexed_manifest = manifest.set_index("hour_key")
            for map_name, frames in (("real", real), ("grid", grid)):
                for key, frame in frames.items():
                    if not required.issubset(frame.columns):
                        missing = sorted(required.difference(frame.columns))
                        raise AssertionError(
                            f"{map_name} columns missing for {scenario}, {key}: {missing}"
                        )
                    row = indexed_manifest.loc[str(key)]
                    if int(pd.to_numeric(frame["Simulation_Block"]).nunique()) != 1:
                        raise AssertionError(f"block metadata varies: {scenario}, {map_name}, {key}")
                    if int(pd.to_numeric(frame["Simulation_Time_Index"]).nunique()) != 1:
                        raise AssertionError(f"time-index metadata varies: {scenario}, {map_name}, {key}")
                    if int(frame["Simulation_Block"].iloc[0]) != int(row["block_index"]):
                        raise AssertionError(f"wrong block metadata: {scenario}, {map_name}, {key}")
                    if int(frame["Simulation_Time_Index"].iloc[0]) != int(row["local_time"]):
                        raise AssertionError(f"wrong time-index metadata: {scenario}, {map_name}, {key}")
                    analysis_time = pd.to_numeric(frame["Hours_elapsed"], errors="coerce")
                    if analysis_time.isna().any() or not np.allclose(
                        analysis_time.to_numpy(dtype=float),
                        float(row["analysis_hours_elapsed"]),
                        rtol=0.0,
                        atol=1e-12,
                    ):
                        raise AssertionError(f"wrong analysis time: {scenario}, {map_name}, {key}")
                    finite_y = pd.to_numeric(
                        frame["ColumnAmountO3"], errors="coerce"
                    ).notna()
                    needed = frame.loc[
                        finite_y,
                        [
                            "Source_Latitude",
                            "Source_Longitude",
                            "Hours_elapsed",
                            "Template_Hours_elapsed",
                        ],
                    ].apply(pd.to_numeric, errors="coerce")
                    if not np.isfinite(needed.to_numpy(dtype=float)).all():
                        raise AssertionError(
                            f"finite response lacks coordinates/time: {scenario}, {map_name}, {key}"
                        )
        records.append(
            {
                "scenario_id": scenario,
                "family": truth["family"],
                "interaction_eta": truth["interaction_eta"],
                "n_days": truth["n_independent_days"],
                "n_hours": truth["n_hours"],
                "R_space_1": truth["correlation_landmarks"]["R_space_1"],
                "R_flow_time_1": truth["correlation_landmarks"]["R_flow_time_1"],
                "R_corner_1_1": truth["correlation_landmarks"]["R_corner_1_1"],
                "negative_spectral_mass": embedding["spectrum_negative_mass_fraction"],
                "relative_covariance_distortion_bound": embedding[
                    "relative_covariance_distortion_bound"
                ],
                "imaginary_ratio": embedding["spectrum_max_abs_imaginary_over_max_abs_real"],
                "max_abs_latitude_mapping_error": max_lat_mapping_error,
                "max_abs_longitude_mapping_error": max_lon_mapping_error,
                "peak_rss_gib": truth["max_rss_gib"],
                "complete_hours": complete["n_hours"],
            }
        )

    for family, separable, intermediate, joint in (
        (
            "matern",
            "matern05_lagrangian_separable_eta0",
            "matern05_lagrangian_mixture_eta0p5",
            "matern05_lagrangian_joint_eta1",
        ),
        (
            "generalized_cauchy",
            "gencauchy_a1_b5_lagrangian_separable_eta0",
            "gencauchy_a1_b5_lagrangian_mixture_eta0p5",
            "gencauchy_a1_b5_lagrangian_joint_eta1",
        ),
    ):
        scenario_ids = (separable, intermediate, joint)
        expected_eta = (0.0, 0.5, 1.0)
        for scenario_id, eta in zip(scenario_ids, expected_eta):
            if not math.isclose(
                float(truths[scenario_id]["interaction_eta"]), eta, abs_tol=1e-14
            ):
                raise AssertionError(f"{family} eta level is wrong: {scenario_id}")
        landmarks = [truths[value]["correlation_landmarks"] for value in scenario_ids]
        for margin in ("R_space_1", "R_flow_time_1"):
            values = [float(value[margin]) for value in landmarks]
            if not all(
                math.isclose(values[0], value, abs_tol=1e-14)
                for value in values[1:]
            ):
                raise AssertionError(f"{family} controlled margin differs: {margin}")
        corners = [float(value["R_corner_1_1"]) for value in landmarks]
        if not corners[0] < corners[1] < corners[2]:
            raise AssertionError(f"{family} interaction dose response is not ordered")
        if not math.isclose(
            corners[1], 0.5 * (corners[0] + corners[2]), abs_tol=1e-14
        ):
            raise AssertionError(f"{family} eta=0.5 covariance is not the midpoint")

    summary = pd.DataFrame(records)
    summary_path = root / "VALIDATION_SUMMARY.csv"
    summary.to_csv(summary_path, index=False, float_format="%.17g")
    report = {
        "status": "PASS",
        "root": str(root),
        "expected_days": args.expected_days,
        "deep_validation": bool(args.deep),
        "scenarios": records,
        "controlled_margin_check": "PASS",
        "interaction_dose_response_check": "PASS",
        "common_random_number_seed_check": "PASS",
        "production_contract_check": (
            "SKIPPED_NONPRODUCTION_GRID"
            if args.allow_nonproduction_grid
            else "PASS"
        ),
        "provenance_hash_check": "PASS",
    }
    (root / "VALIDATION_REPORT.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)
    print(f"PASS: {root}", flush=True)


if __name__ == "__main__":
    main()
