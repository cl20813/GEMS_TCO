#!/usr/bin/env python3
"""Fit matched-family joint models and compare fitted versus empirical cross terms."""

from __future__ import annotations

import argparse
import fcntl
import json
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[6]
ORACLE_DIR = HERE.parent / "sim_two_contrast_cross_center_092726"
for path in (HERE, ORACLE_DIR, PROJECT_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import fitted_cross_643_core as fitted_core  # noqa: E402
import run_two_contrast_diagnostic as oracle_runner  # noqa: E402
from two_contrast_core import (  # noqa: E402
    atomic_csv,
    atomic_json,
    atomic_text,
    clean_json,
    load_json,
    sha256_file,
    signature_for_files,
)


SCHEMA_VERSION = 1
DEFAULT_CONFIG = HERE / "fitted_cross_643_config.json"
SUMMARY_NAME = "daily_fitted_cross_centers.csv"
STRATA_NAME = "daily_fitted_cross_strata.csv"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    group = result.add_mutually_exclusive_group(required=True)
    group.add_argument("--scenario-index", type=int)
    group.add_argument("--scenario-id")
    result.add_argument("--data-root", type=Path)
    result.add_argument("--output-root", type=Path)
    result.add_argument("--dates", nargs="+")
    result.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    result.add_argument("--preflight-only", action="store_true")
    result.add_argument("--overwrite", action="store_true")
    return result


def resolve_project_path(path_text: str, relative_to: Path) -> Path:
    requested = Path(path_text)
    candidates = (
        requested,
        PROJECT_ROOT / requested,
        relative_to / requested,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(candidates[1])


def metadata_prefix(
    scenario_index: int,
    scenario: Mapping[str, Any],
    truth: Mapping[str, Any],
    date: str,
    day_summary: Mapping[str, Any],
    signature: str,
    truth_sha: str,
    run_request_sha: str,
) -> dict[str, Any]:
    family = str(truth["family"])
    return {
        "schema_version": SCHEMA_VERSION,
        "diagnostic_signature": signature,
        "scenario_index": int(scenario_index),
        "scenario_id": str(scenario["scenario_id"]),
        "family": family,
        "interaction_eta": float(truth["interaction_eta"]),
        "fitted_model": (
            "joint_matern05"
            if family == "matern"
            else "joint_generalized_cauchy_a1_b5"
        ),
        "date": date,
        "block_index": int(day_summary["block_index"]),
        "seed": int(day_summary["seed"]),
        "truth_sha256": truth_sha,
        "source_run_request_sha256": run_request_sha,
        "geometry": "truth_flow_following_two_contrast",
        "vecchia_geometry": "direction_adapted_corridor_width_4x4_lag643",
        "vecchia_lag_pattern": "6/4/3",
        "primary_quantity": "empirical_var_l_minus_fitted_var_l_with_component_attribution",
    }


def compatible(record: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return record.get("status") == "complete" and all(
        record.get(key) == value for key, value in expected.items()
    )


def rebuild_scenario(
    scenario_dir: Path,
    selected_dates: Sequence[str],
    expected_common: Mapping[str, Any],
) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    strata: list[dict[str, Any]] = []
    incompatible: list[str] = []
    for date in selected_dates:
        path = scenario_dir / "day_json" / f"{date}.json"
        if not path.is_file():
            continue
        record = load_json(path)
        expected = {**expected_common, "date": date}
        if not compatible(record, expected):
            incompatible.append(date)
            continue
        summaries.append(dict(record["summary"]))
        strata.extend(dict(row) for row in record.get("strata", []))
    summary_frame = pd.DataFrame(summaries)
    if not summary_frame.empty:
        summary_frame = summary_frame.sort_values(["block_index", "date"]).reset_index(drop=True)
    stratum_frame = pd.DataFrame(strata)
    if not stratum_frame.empty:
        stratum_frame = stratum_frame.sort_values(
            ["block_index", "date", "handedness", "time_t"]
        ).reset_index(drop=True)
    summary_path = scenario_dir / SUMMARY_NAME
    strata_path = scenario_dir / STRATA_NAME
    if summary_frame.empty:
        summary_path.unlink(missing_ok=True)
    else:
        atomic_csv(summary_path, summary_frame)
    if stratum_frame.empty:
        strata_path.unlink(missing_ok=True)
    else:
        atomic_csv(strata_path, stratum_frame)
    progress = {
        **expected_common,
        "configured_days": len(selected_dates),
        "completed_days": len(summary_frame),
        "remaining_days": len(selected_dates) - len(summary_frame),
        "completed_dates": summary_frame["date"].astype(str).tolist()
        if not summary_frame.empty else [],
        "incompatible_checkpoint_dates": incompatible,
    }
    atomic_json(scenario_dir / "progress.json", progress)
    complete_path = scenario_dir / "COMPLETE.json"
    if progress["remaining_days"] == 0 and not incompatible:
        atomic_json(complete_path, {**progress, "complete": True})
    else:
        complete_path.unlink(missing_ok=True)
    return progress


def rebuild_master(output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".master_csv.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        summaries: list[pd.DataFrame] = []
        strata: list[pd.DataFrame] = []
        for path in sorted((output_root / "scenarios").glob(f"*/{SUMMARY_NAME}")):
            if path.stat().st_size <= 1:
                continue
            try:
                frame = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                continue
            if not frame.empty:
                summaries.append(frame)
        for path in sorted((output_root / "scenarios").glob(f"*/{STRATA_NAME}")):
            if path.stat().st_size <= 1:
                continue
            try:
                frame = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                continue
            if not frame.empty:
                strata.append(frame)
        master = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
        if not master.empty:
            master = master.sort_values(["scenario_index", "block_index", "date"]).reset_index(drop=True)
        master_strata = pd.concat(strata, ignore_index=True) if strata else pd.DataFrame()
        if not master_strata.empty:
            master_strata = master_strata.sort_values(
                ["scenario_index", "block_index", "date", "handedness", "time_t"]
            ).reset_index(drop=True)
        master_path = output_root / "all_daily_fitted_cross_centers.csv"
        master_strata_path = output_root / "all_daily_fitted_cross_strata.csv"
        if master.empty:
            master_path.unlink(missing_ok=True)
        else:
            atomic_csv(master_path, master)
        if master_strata.empty:
            master_strata_path.unlink(missing_ok=True)
        else:
            atomic_csv(master_strata_path, master_strata)
        atomic_json(
            output_root / "master_progress.json",
            {
                "schema_version": SCHEMA_VERSION,
                "completed_scenario_days": int(len(master)),
                "expected_scenario_days": 180,
                "scenario_count_present": int(master["scenario_id"].nunique())
                if not master.empty else 0,
                "complete": bool(
                    len(master) == 180
                    and master["scenario_id"].nunique() == 6
                    and not master.duplicated(["scenario_id", "date"]).any()
                ) if not master.empty else False,
            },
        )


def main() -> None:
    args = parser().parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_json(config_path)
    scenario_config_path = resolve_project_path(
        config["input"]["scenario_config"], config_path.parent
    )
    suite = load_json(scenario_config_path)
    scenario_index, scenario = oracle_runner._select_scenario(
        suite, args.scenario_index, args.scenario_id
    )
    oracle_design_path = resolve_project_path(
        config["diagnostic"]["oracle_design"], config_path.parent
    )
    design = load_json(oracle_design_path)
    data_root = (
        args.data_root.expanduser()
        if args.data_root is not None
        else Path(config["input"]["amarel_data_root"])
    )
    output_root = (
        args.output_root.expanduser()
        if args.output_root is not None
        else Path(config["output"]["amarel_root"])
    )
    year = int(config["input"]["year"])
    prefix = f"sim_july{year}_st_circulant"
    scenario_id = str(scenario["scenario_id"])
    year_dir = data_root / scenario_id / f"{year}_july_st_circulant"
    truth_path = year_dir / f"{prefix}_truth.json"
    complete_path = year_dir / "COMPLETE.json"
    truth = load_json(truth_path)
    source_complete = load_json(complete_path)
    oracle_runner._validate_truth(
        truth, source_complete, scenario, suite, scenario_config_path, design
    )
    source = oracle_runner.DaySource(year_dir, truth)
    configured_dates = [str(value) for value in truth["selected_dates"]]
    if args.dates:
        unknown = sorted(set(args.dates).difference(configured_dates))
        if unknown:
            raise ValueError(f"unknown requested dates: {unknown}")
        selected_dates = [date for date in configured_dates if date in set(args.dates)]
    else:
        selected_dates = configured_dates
    signature = signature_for_files(
        [
            config_path,
            HERE / "fitted_cross_643_core.py",
            Path(__file__).resolve(),
            oracle_design_path,
            ORACLE_DIR / "two_contrast_core.py",
            ORACLE_DIR / "run_two_contrast_diagnostic.py",
            scenario_config_path,
        ]
    )
    truth_sha = sha256_file(truth_path)
    run_request_sha = str(truth["run_request_sha256"])
    day_by_date = {str(row["date"]): dict(row) for row in truth["day_summaries"]}
    scenario_dir = output_root / "scenarios" / scenario_id
    scenario_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "schema_version": SCHEMA_VERSION,
        "diagnostic_signature": signature,
        "scenario_id": scenario_id,
        "truth_sha256": truth_sha,
        "source_run_request_sha256": run_request_sha,
    }

    if args.preflight_only:
        for date in selected_dates:
            source.validate_metadata(date)
        print(f"PASS preflight: {scenario_id}, {len(selected_dates)} days")
        return

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    failures: list[str] = []
    for date in selected_dates:
        provenance = source.provenance(date)
        prefix_metadata = metadata_prefix(
            scenario_index,
            scenario,
            truth,
            date,
            day_by_date[date],
            signature,
            truth_sha,
            run_request_sha,
        )
        expected = {
            **common,
            "date": date,
            **{
                key: provenance[key]
                for key in (
                    "source_success_sha256",
                    "source_manifest_sha256",
                    "source_gridded_sha256",
                    "source_day_request_sha256",
                )
            },
        }
        day_json = scenario_dir / "day_json" / f"{date}.json"
        fit_dir = scenario_dir / "fits" / date
        fit_json = fit_dir / "fit.json"
        fit_complete = fit_dir / "FIT_COMPLETE.json"
        if day_json.is_file() and not args.overwrite:
            saved = load_json(day_json)
            if compatible(saved, expected):
                print(f"REUSE {scenario_id} {date}", flush=True)
                rebuild_scenario(scenario_dir, configured_dates, common)
                rebuild_master(output_root)
                continue
        try:
            frames, manifest, loaded_provenance = source.load(date)
            for key, value in provenance.items():
                if loaded_provenance.get(key) != value:
                    raise RuntimeError(f"source provenance changed while loading: {key}")
            cube = fitted_core.oracle.build_day_cube(frames, manifest, truth)
            samples, audit = fitted_core.oracle.prepare_flow_following_samples(
                cube, truth, design
            )
            minimum = int(design["diagnostic"]["minimum_complete_samples_per_day"])
            if len(samples["q_a"]) < minimum:
                raise RuntimeError(f"only {len(samples['q_a'])} complete contrast samples")

            input_cpu, grid_coords, response_center = fitted_core.build_input_map(
                frames, manifest, torch.device("cpu")
            )
            initializer = fitted_core.make_initializer(
                input_cpu, grid_coords, date, truth, config
            )
            fit_expected = {
                **expected,
                "family": str(truth["family"]),
                "vecchia_lag_pattern": "6/4/3",
                "vecchia_geometry": "direction_adapted_corridor_width_4x4_lag643",
            }
            if (
                fit_json.is_file()
                and fit_complete.is_file()
                and not args.overwrite
                and compatible(load_json(fit_complete), fit_expected)
            ):
                fit_record = load_json(fit_json)
                print(f"REUSE FIT {scenario_id} {date}", flush=True)
            else:
                input_device = {
                    key: value.to(device=device, dtype=torch.float64).contiguous()
                    for key, value in input_cpu.items()
                }
                fit_record, model = fitted_core.fit_model(
                    str(truth["family"]), input_device, grid_coords, initializer, config
                )
                fit_record.update(
                    {
                        **fit_expected,
                        "response_center": response_center,
                        "device": str(device),
                        "gpu_name": torch.cuda.get_device_name(device)
                        if device.type == "cuda" else None,
                    }
                )
                atomic_json(fit_json, clean_json(fit_record))
                atomic_json(fit_complete, {**fit_expected, "status": "complete"})
                del model, input_device
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            model_coordinates = fitted_core.reconstruct_source_coordinates(
                samples, cube, truth, design
            )
            fitted_covariance = fitted_core.fitted_contrast_covariances(
                model_coordinates,
                fit_record["interpretable_parameters"],
                str(truth["family"]),
                design,
                config,
            )
            fitted_summary = fitted_core.summarize_fitted_cross(
                samples, fitted_covariance, design
            )
            oracle_separable, oracle_joint = fitted_core.oracle.pointwise_contrast_covariances(
                samples["coordinates"],
                truth,
                design,
                int(design["diagnostic"]["oracle_covariance_chunk_size"]),
            )
            d1 = float(design["geometry"]["secondary_l_coefficients"]["d1"])
            d2 = float(design["geometry"]["secondary_l_coefficients"]["d2"])
            oracle_summary = fitted_core.oracle.summarize_center(
                samples["q_a"],
                samples["q_b"],
                oracle_separable,
                oracle_joint,
                float(truth["interaction_eta"]),
                d1,
                d2,
            )
            summary = {
                **prefix_metadata,
                **{
                    key: provenance[key]
                    for key in (
                        "source_success_sha256",
                        "source_manifest_sha256",
                        "source_gridded_sha256",
                        "source_day_request_sha256",
                    )
                },
                **fitted_summary,
                "truth_h_ab": oracle_summary["truth_h_ab"],
                "truth_l_cross_term": oracle_summary["truth_l_cross_term"],
                "truth_l_variance": oracle_summary["truth_l_second_moment"],
                "empirical_minus_truth_h_ab": oracle_summary["empirical_minus_truth_h_ab"],
                "empirical_minus_truth_l_cross_term": oracle_summary[
                    "empirical_minus_truth_l_cross_term"
                ],
                "empirical_minus_truth_l_variance": fitted_summary[
                    "empirical_l_variance"
                ] - oracle_summary["truth_l_second_moment"],
                "fitted_minus_truth_h_ab": fitted_summary["fitted_h_ab"]
                - oracle_summary["truth_h_ab"],
                "fitted_minus_truth_l_cross_term": fitted_summary["fitted_l_cross_term"]
                - oracle_summary["truth_l_cross_term"],
                "fitted_minus_truth_l_variance": fitted_summary["fitted_l_variance"]
                - oracle_summary["truth_l_second_moment"],
                "fit_nll_per_target": fit_record["profiled_vecchia_nll_per_target"],
                "fit_converged": fit_record["converged_at_outer_grad_tol"],
                "fit_max_abs_gradient": fit_record["maximum_absolute_gradient"],
                "fit_seconds": fit_record["fit_seconds"],
                "fit_target_chunk_size": fit_record["target_chunk_size"],
                "fit_advec_lat": fit_record["interpretable_parameters"]["advec_lat"],
                "fit_advec_lon": fit_record["interpretable_parameters"]["advec_lon"],
                "fit_signal_variance": fit_record["interpretable_parameters"]["signal_variance"],
                "fit_range_lat": fit_record["interpretable_parameters"]["range_lat"],
                "fit_range_lon": fit_record["interpretable_parameters"]["range_lon"],
                "fit_range_time": fit_record["interpretable_parameters"]["range_time"],
                "initializer_advec_lat": initializer["seed_lat"],
                "initializer_advec_lon": initializer["seed_lon"],
                "complete_fraction_of_geometry_candidates": audit[
                    "complete_fraction_of_geometry_candidates"
                ],
            }
            fitted_strata = fitted_core.stratum_summaries(
                samples, fitted_covariance, design
            )
            strata: list[dict[str, Any]] = []
            for fitted_row in fitted_strata:
                selected = (samples["handedness"] == fitted_row["handedness"]) & (
                    samples["time_t"].astype(int) == int(fitted_row["time_t"])
                )
                oracle_row = fitted_core.oracle.summarize_center(
                    samples["q_a"][selected],
                    samples["q_b"][selected],
                    oracle_separable[selected],
                    oracle_joint[selected],
                    float(truth["interaction_eta"]),
                    d1,
                    d2,
                )
                fitted_row.update(
                    {
                        **prefix_metadata,
                        "truth_h_ab": oracle_row["truth_h_ab"],
                        "truth_l_cross_term": oracle_row["truth_l_cross_term"],
                        "truth_l_variance": oracle_row["truth_l_second_moment"],
                        "empirical_minus_truth_h_ab": oracle_row[
                            "empirical_minus_truth_h_ab"
                        ],
                        "fitted_minus_truth_h_ab": fitted_row["fitted_h_ab"]
                        - oracle_row["truth_h_ab"],
                        "fitted_minus_truth_l_variance": fitted_row[
                            "fitted_l_variance"
                        ] - oracle_row["truth_l_second_moment"],
                    }
                )
                strata.append(fitted_row)
            record = {
                **expected,
                "status": "complete",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "host": platform.node(),
                "python": sys.version,
                "torch": torch.__version__,
                "summary": clean_json(summary),
                "strata": clean_json(strata),
                "fit_json": str(fit_json),
            }
            atomic_json(day_json, record)
            (scenario_dir / "failures" / f"{date}.json").unlink(missing_ok=True)
            print(
                f"DONE {scenario_id} {date}: Var(L) data={summary['empirical_l_variance']:.6g} "
                f"fit={summary['fitted_l_variance']:.6g} "
                f"delta={summary['empirical_minus_fitted_l_variance']:.6g}; "
                f"cross delta={summary['empirical_minus_fitted_l_cross_term']:.6g}",
                flush=True,
            )
        except Exception as error:
            failures.append(date)
            atomic_json(
                scenario_dir / "failures" / f"{date}.json",
                {
                    **prefix_metadata,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            print(f"FAILED {scenario_id} {date}: {error}", file=sys.stderr, flush=True)
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()
        rebuild_scenario(scenario_dir, configured_dates, common)
        rebuild_master(output_root)

    progress = rebuild_scenario(scenario_dir, configured_dates, common)
    rebuild_master(output_root)
    print(json.dumps(clean_json(progress), indent=2), flush=True)
    if failures:
        raise SystemExit(f"failed dates for {scenario_id}: {failures}")


if __name__ == "__main__":
    main()
