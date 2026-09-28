#!/usr/bin/env python3
"""Run the restartable Q_A/Q_B cross-center diagnostic for one scenario."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[6]
DEFAULT_CONFIG = HERE / "two_contrast_diagnostic_092726.json"

from two_contrast_core import (  # noqa: E402
    SCHEMA_VERSION,
    atomic_json,
    build_day_cube,
    checkpoint_compatible,
    evaluate_day,
    load_json,
    rebuild_master_outputs,
    rebuild_scenario_outputs,
    sha256_file,
    signature_for_files,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    group = result.add_mutually_exclusive_group(required=True)
    group.add_argument("--scenario-index", type=int)
    group.add_argument("--scenario-id")
    result.add_argument("--data-root", type=Path)
    result.add_argument("--output-root", type=Path)
    result.add_argument(
        "--dates",
        nargs="+",
        help="optional YYYY-MM-DD subset for a smoke run; defaults to all 30 dates",
    )
    result.add_argument("--preflight-only", action="store_true")
    result.add_argument("--rebuild-only", action="store_true")
    result.add_argument("--overwrite", action="store_true")
    return result


def _resolve_scenario_config(
    diagnostic_config: dict[str, Any], config_path: Path
) -> Path:
    requested = Path(diagnostic_config["input"]["scenario_config"])
    if requested.is_absolute():
        return requested
    project_candidate = PROJECT_ROOT / requested
    if project_candidate.is_file():
        return project_candidate
    relative_candidate = config_path.parent / requested
    if relative_candidate.is_file():
        return relative_candidate
    raise FileNotFoundError(project_candidate)


def _select_scenario(
    suite_config: dict[str, Any], index: int | None, scenario_id: str | None
) -> tuple[int, dict[str, Any]]:
    scenarios = list(suite_config["scenarios"])
    if scenario_id is not None:
        matches = [
            (position, value)
            for position, value in enumerate(scenarios)
            if str(value["scenario_id"]) == scenario_id
        ]
        if len(matches) != 1:
            raise ValueError(f"unknown or duplicated scenario_id {scenario_id!r}")
        return matches[0]
    assert index is not None
    if index < 0 or index >= len(scenarios):
        raise ValueError(f"scenario index {index} is outside 0..{len(scenarios)-1}")
    return int(index), dict(scenarios[index])


def _validate_truth(
    truth: dict[str, Any],
    complete: dict[str, Any],
    scenario: dict[str, Any],
    suite_config: dict[str, Any],
    scenario_config_path: Path,
    diagnostic_config: dict[str, Any],
) -> None:
    scenario_id = str(scenario["scenario_id"])
    if truth.get("scenario_id") != scenario_id:
        raise ValueError("truth scenario_id does not match the selected scenario")
    if complete.get("scenario_id") != scenario_id:
        raise ValueError("COMPLETE scenario_id does not match the selected scenario")
    if truth.get("run_request_sha256") != complete.get("run_request_sha256"):
        raise ValueError("truth and COMPLETE run_request_sha256 differ")
    if truth.get("run_request", {}).get("scenario_config_sha256") != sha256_file(
        scenario_config_path
    ):
        raise ValueError("truth was generated from a different scenario configuration")
    expected_days = int(diagnostic_config["input"]["expected_days"])
    expected_hours_per_day = int(diagnostic_config["input"]["hours_per_day"])
    if int(truth["n_independent_days"]) != expected_days:
        raise ValueError("truth does not contain the expected 30 independent days")
    if int(truth["hours_per_day"]) != expected_hours_per_day:
        raise ValueError("truth does not contain eight hours per day")
    if int(truth["n_hours"]) != expected_days * expected_hours_per_day:
        raise ValueError("truth n_hours is inconsistent")
    if int(complete["n_days"]) != expected_days or int(complete["n_hours"]) != int(
        truth["n_hours"]
    ):
        raise ValueError("COMPLETE counts are inconsistent")
    if not math_isclose(float(truth["nugget"]), 0.0):
        raise ValueError("the six-scenario diagnostic requires nugget exactly zero")
    if str(truth["family"]) != str(scenario["family"]):
        raise ValueError("truth covariance family differs from the suite configuration")
    if not math_isclose(
        float(truth["interaction_eta"]), float(scenario["interaction_eta"])
    ):
        raise ValueError("truth eta differs from the suite configuration")
    for name, expected in suite_config["shared_parameters"].items():
        if not math_isclose(float(truth[name]), float(expected)):
            raise ValueError(f"truth shared parameter {name} differs from configuration")
    dates = [str(value) for value in truth["selected_dates"]]
    if len(dates) != expected_days or len(set(dates)) != expected_days:
        raise ValueError("truth selected_dates is incomplete or duplicated")
    if any(pd.Timestamp(value).month != int(diagnostic_config["input"]["month"]) for value in dates):
        raise ValueError("truth selected_dates includes a date outside July")


def math_isclose(left: float, right: float) -> bool:
    return bool(np.isclose(left, right, rtol=0.0, atol=1e-12))


class DaySource:
    """Validated low-memory access to generator day checkpoints."""

    def __init__(self, year_dir: Path, truth: dict[str, Any]) -> None:
        self.year_dir = year_dir
        self.truth = truth
        self.by_date = {
            str(value["date"]): dict(value) for value in truth["day_summaries"]
        }
        if set(self.by_date) != {str(value) for value in truth["selected_dates"]}:
            raise ValueError("truth day_summaries and selected_dates differ")

    def paths(self, date: str) -> dict[str, Path]:
        root = self.year_dir / "day_checkpoints" / date
        return {
            "root": root,
            "success": root / "SUCCESS.json",
            "manifest": root / "manifest.csv",
            "gridded": root / "gridded.pkl",
        }

    def validate_metadata(self, date: str) -> dict[str, Any]:
        if date not in self.by_date:
            raise ValueError(f"date {date} is absent from truth day_summaries")
        paths = self.paths(date)
        for name in ("success", "manifest", "gridded"):
            if not paths[name].is_file():
                raise FileNotFoundError(paths[name])
        success = load_json(paths["success"])
        expected = self.by_date[date]
        for name in ("date", "block_index", "seed", "day_request_sha256"):
            if success.get(name) != expected.get(name):
                raise ValueError(f"{date} SUCCESS {name} differs from truth day summary")
        if int(success["n_hours"]) != int(self.truth["hours_per_day"]):
            raise ValueError(f"{date} SUCCESS has the wrong hour count")
        manifest = pd.read_csv(paths["manifest"])
        if len(manifest) != int(self.truth["hours_per_day"]):
            raise ValueError(f"{date} manifest has the wrong row count")
        required_manifest = {
            "date",
            "block_index",
            "seed",
            "local_time",
            "simulation_time_index",
            "hour_key",
        }
        missing_manifest = required_manifest.difference(manifest.columns)
        if missing_manifest:
            raise ValueError(
                f"{date} manifest is missing {sorted(missing_manifest)}"
            )
        if manifest["date"].astype(str).nunique() != 1 or str(
            manifest["date"].iloc[0]
        ) != date:
            raise ValueError(f"{date} manifest date is inconsistent")
        for name in ("block_index", "seed"):
            values = pd.to_numeric(manifest[name], errors="raise")
            if values.nunique() != 1 or int(values.iloc[0]) != int(expected[name]):
                raise ValueError(f"{date} manifest {name} differs from truth")
        expected_slots = np.arange(int(self.truth["hours_per_day"]), dtype=np.int64)
        ordered = manifest.sort_values("local_time").reset_index(drop=True)
        if not np.array_equal(
            pd.to_numeric(ordered["local_time"], errors="raise").to_numpy(np.int64),
            expected_slots,
        ):
            raise ValueError(f"{date} manifest local_time is not exactly 0,...,7")
        if not np.array_equal(
            pd.to_numeric(
                ordered["simulation_time_index"], errors="raise"
            ).to_numpy(np.int64),
            expected_slots,
        ):
            raise ValueError(
                f"{date} manifest simulation_time_index is not exactly 0,...,7"
            )
        if manifest["hour_key"].astype(str).nunique() != int(
            self.truth["hours_per_day"]
        ):
            raise ValueError(f"{date} manifest hour keys are not unique")
        return {"paths": paths, "success": success, "manifest": manifest}

    def load(self, date: str) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, Any]]:
        metadata = self.validate_metadata(date)
        paths = metadata["paths"]
        frames = pd.read_pickle(paths["gridded"])
        if not isinstance(frames, dict):
            raise TypeError(f"{paths['gridded']} is not a dictionary of frames")
        expected_keys = set(metadata["manifest"]["hour_key"].astype(str))
        if set(map(str, frames)) != expected_keys:
            raise ValueError(f"{date} manifest and gridded pickle hour keys differ")
        provenance = self.provenance(date, metadata)
        return frames, metadata["manifest"], provenance

    def provenance(
        self, date: str, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        metadata = self.validate_metadata(date) if metadata is None else metadata
        paths = metadata["paths"]
        return {
            "source_success_sha256": sha256_file(paths["success"]),
            "source_manifest_sha256": sha256_file(paths["manifest"]),
            "source_gridded_sha256": sha256_file(paths["gridded"]),
            "source_day_request_sha256": metadata["success"][
                "day_request_sha256"
            ],
            "source_checkpoint_directory": str(paths["root"]),
            "hour_keys": metadata["manifest"]
            .sort_values("local_time")["hour_key"]
            .astype(str)
            .tolist(),
        }


def _metadata_prefix(
    scenario_index: int,
    scenario: dict[str, Any],
    truth: dict[str, Any],
    date: str,
    day_summary: dict[str, Any],
    diagnostic_signature: str,
    truth_sha256: str,
    source_run_request_sha256: str,
) -> dict[str, Any]:
    return {
        "scenario_index": int(scenario_index),
        "scenario_id": str(scenario["scenario_id"]),
        "family": str(truth["family"]),
        "interaction_eta": float(truth["interaction_eta"]),
        "date": date,
        "block_index": int(day_summary["block_index"]),
        "seed": int(day_summary["seed"]),
        "diagnostic_signature": diagnostic_signature,
        "truth_sha256": truth_sha256,
        "source_run_request_sha256": source_run_request_sha256,
        "coordinate_frame": "truth_lagrangian_flow_following",
        "primary_quantity": "weighted_cross_term_and_signed_excess_over_matched_separable_center",
    }


def _collect_input_identities(
    data_root: Path, scenario_ids: list[str], year: int
) -> dict[str, dict[str, str]]:
    prefix = f"sim_july{year}_st_circulant"
    identities: dict[str, dict[str, str]] = {}
    for scenario_id in scenario_ids:
        truth_path = (
            data_root
            / scenario_id
            / f"{year}_july_st_circulant"
            / f"{prefix}_truth.json"
        )
        if not truth_path.is_file():
            raise FileNotFoundError(truth_path)
        scenario_truth = load_json(truth_path)
        if scenario_truth.get("scenario_id") != scenario_id:
            raise ValueError(f"truth scenario mismatch at {truth_path}")
        identities[scenario_id] = {
            "truth_sha256": sha256_file(truth_path),
            "source_run_request_sha256": str(
                scenario_truth["run_request_sha256"]
            ),
        }
    return identities


def _checkpoint_expectation(
    diagnostic_signature: str,
    truth_sha256: str,
    scenario_id: str,
    date: str,
    day_summary: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    return {
        "diagnostic_signature": diagnostic_signature,
        "truth_sha256": truth_sha256,
        "scenario_id": scenario_id,
        "date": date,
        "block_index": int(day_summary["block_index"]),
        "source_day_request_sha256": str(day_summary["day_request_sha256"]),
        "source_success_sha256": provenance["source_success_sha256"],
        "source_manifest_sha256": provenance["source_manifest_sha256"],
        "source_gridded_sha256": provenance["source_gridded_sha256"],
    }


def _update_scenario_complete(
    scenario_dir: Path,
    scenario_id: str,
    selected_dates: list[str],
    diagnostic_signature: str,
    truth_sha256: str,
    source_run_request_sha256: str,
    progress: dict[str, Any],
) -> None:
    marker_path = scenario_dir / "COMPLETE.json"
    if int(progress["completed_days"]) != len(selected_dates):
        marker_path.unlink(missing_ok=True)
        return
    scenario_csv = scenario_dir / "daily_two_contrast_centers.csv"
    stratum_csv = scenario_dir / "daily_two_contrast_strata.csv"
    atomic_json(
        marker_path,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "diagnostic_signature": diagnostic_signature,
            "truth_sha256": truth_sha256,
            "source_run_request_sha256": source_run_request_sha256,
            "scenario_id": scenario_id,
            "completed_days": int(progress["completed_days"]),
            "daily_csv_sha256": sha256_file(scenario_csv),
            "stratum_csv_sha256": sha256_file(stratum_csv),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def main() -> None:
    args = parser().parse_args()
    config_path = args.config.expanduser().resolve()
    diagnostic_config = load_json(config_path)
    if int(diagnostic_config.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported diagnostic configuration schema")
    scenario_config_path = _resolve_scenario_config(diagnostic_config, config_path)
    suite_config = load_json(scenario_config_path)
    scenario_index, scenario = _select_scenario(
        suite_config, args.scenario_index, args.scenario_id
    )
    scenario_ids = [str(value["scenario_id"]) for value in suite_config["scenarios"]]
    if len(scenario_ids) != 6 or len(set(scenario_ids)) != 6:
        raise ValueError("the production suite must contain exactly six scenarios")
    data_root = (
        args.data_root.expanduser()
        if args.data_root is not None
        else Path(diagnostic_config["input"]["amarel_data_root"])
    )
    output_root = (
        args.output_root.expanduser()
        if args.output_root is not None
        else Path(diagnostic_config["output"]["amarel_output_root"])
    )
    year = int(diagnostic_config["input"]["year"])
    year_dir = data_root / str(scenario["scenario_id"]) / f"{year}_july_st_circulant"
    prefix = f"sim_july{year}_st_circulant"
    truth_path = year_dir / f"{prefix}_truth.json"
    complete_path = year_dir / "COMPLETE.json"
    if not truth_path.is_file() or not complete_path.is_file():
        raise FileNotFoundError(f"scenario is incomplete under {year_dir}")
    truth = load_json(truth_path)
    complete = load_json(complete_path)
    _validate_truth(
        truth,
        complete,
        scenario,
        suite_config,
        scenario_config_path,
        diagnostic_config,
    )
    source_files = (
        config_path,
        scenario_config_path,
        Path(__file__).resolve(),
        HERE / "two_contrast_core.py",
    )
    diagnostic_signature = signature_for_files(source_files)
    truth_sha256 = sha256_file(truth_path)
    source_run_request_sha256 = str(truth["run_request_sha256"])
    input_identities = _collect_input_identities(data_root, scenario_ids, year)
    selected_identity = input_identities[str(scenario["scenario_id"])]
    if selected_identity != {
        "truth_sha256": truth_sha256,
        "source_run_request_sha256": source_run_request_sha256,
    }:
        raise RuntimeError("selected scenario input identity is inconsistent")
    selected_dates = [str(value) for value in truth["selected_dates"]]
    if args.dates:
        requested = list(dict.fromkeys(str(value) for value in args.dates))
        unknown = sorted(set(requested).difference(selected_dates))
        if unknown:
            raise ValueError(f"requested dates are absent from truth: {unknown}")
        run_dates = [value for value in selected_dates if value in set(requested)]
    else:
        run_dates = selected_dates
    scenario_dir = output_root / "scenarios" / str(scenario["scenario_id"])
    scenario_dir.mkdir(parents=True, exist_ok=True)
    run_manifest_path = scenario_dir / "run_manifest.json"
    manifest_record = {
        "schema_version": SCHEMA_VERSION,
        "study_id": diagnostic_config["study_id"],
        "diagnostic_signature": diagnostic_signature,
        "scenario_index": scenario_index,
        "scenario": scenario,
        "data_root": str(data_root),
        "output_root": str(output_root),
        "year_directory": str(year_dir),
        "truth_path": str(truth_path),
        "truth_sha256": truth_sha256,
        "source_run_request_sha256": source_run_request_sha256,
        "diagnostic_config": str(config_path),
        "diagnostic_config_sha256": sha256_file(config_path),
        "scenario_config": str(scenario_config_path),
        "scenario_config_sha256": sha256_file(scenario_config_path),
        "selected_dates": selected_dates,
        "requested_dates_this_invocation": run_dates,
        "host": {
            "hostname": platform.node(),
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    previous_signature: str | None = None
    if run_manifest_path.is_file():
        old = load_json(run_manifest_path)
        previous_signature = old.get("diagnostic_signature")
        if previous_signature != diagnostic_signature and not args.overwrite:
            raise RuntimeError(
                f"existing output has a different diagnostic signature: {scenario_dir}; "
                "use --overwrite only after reviewing the changed design"
            )
    if args.overwrite:
        dates_to_replace = (
            selected_dates
            if previous_signature not in (None, diagnostic_signature)
            else run_dates
        )
        for date in dates_to_replace:
            (scenario_dir / "day_json" / f"{date}.json").unlink(missing_ok=True)
            (scenario_dir / "failures" / f"{date}.json").unlink(missing_ok=True)
        (scenario_dir / "COMPLETE.json").unlink(missing_ok=True)
    atomic_json(run_manifest_path, manifest_record)

    source = DaySource(year_dir, truth)
    by_date = {str(value["date"]): dict(value) for value in truth["day_summaries"]}
    for date in run_dates:
        source.validate_metadata(date)
    # Invalidate final-facing markers and progress views before checking cached
    # source provenance. They are recreated only after every current JSON and
    # source hash passes, including in preflight/rebuild-only modes.
    for path in (
        scenario_dir / "COMPLETE.json",
        scenario_dir / "progress.json",
        output_root / "FINAL_COMPLETE.json",
        output_root / "master_progress.json",
        output_root / "RESULTS.md",
        output_root / "PROVISIONAL_RESULTS.md",
    ):
        path.unlink(missing_ok=True)
    provenance_cache: dict[str, dict[str, Any]] = {}
    for date in selected_dates:
        checkpoint_path = scenario_dir / "day_json" / f"{date}.json"
        if not checkpoint_path.is_file():
            continue
        provenance = source.provenance(date)
        provenance_cache[date] = provenance
        expected_checkpoint = _checkpoint_expectation(
            diagnostic_signature,
            truth_sha256,
            str(scenario["scenario_id"]),
            date,
            by_date[date],
            provenance,
        )
        saved = load_json(checkpoint_path)
        if not checkpoint_compatible(saved, expected_checkpoint):
            raise RuntimeError(
                f"incompatible day checkpoint exists at {checkpoint_path}; "
                "use --overwrite only after reviewing the mismatch"
            )
        (scenario_dir / "failures" / f"{date}.json").unlink(missing_ok=True)
    print(
        f"preflight PASS scenario={scenario['scenario_id']} dates={len(run_dates)} "
        f"signature={diagnostic_signature[:12]}",
        flush=True,
    )
    progress = rebuild_scenario_outputs(
        scenario_dir,
        str(scenario["scenario_id"]),
        selected_dates,
        diagnostic_signature,
        truth_sha256,
        source_run_request_sha256,
    )
    _update_scenario_complete(
        scenario_dir,
        str(scenario["scenario_id"]),
        selected_dates,
        diagnostic_signature,
        truth_sha256,
        source_run_request_sha256,
        progress,
    )
    master = rebuild_master_outputs(
        output_root,
        scenario_ids,
        selected_dates,
        diagnostic_signature,
        input_identities,
    )
    if args.preflight_only or args.rebuild_only:
        print(json.dumps({"scenario": progress, "master": master}, indent=2), flush=True)
        return

    for date in run_dates:
        checkpoint_path = scenario_dir / "day_json" / f"{date}.json"
        day_summary = by_date[date]
        if checkpoint_path.is_file() and not args.overwrite:
            saved = load_json(checkpoint_path)
            current_provenance = provenance_cache.get(date) or source.provenance(date)
            expected_checkpoint = _checkpoint_expectation(
                diagnostic_signature,
                truth_sha256,
                str(scenario["scenario_id"]),
                date,
                day_summary,
                current_provenance,
            )
            if checkpoint_compatible(
                saved,
                expected_checkpoint,
            ):
                print(f"reuse {scenario['scenario_id']} {date}", flush=True)
                (scenario_dir / "failures" / f"{date}.json").unlink(
                    missing_ok=True
                )
                continue
            raise RuntimeError(
                f"incompatible day checkpoint exists at {checkpoint_path}; "
                "use --overwrite only after reviewing the mismatch"
            )

        started = time.perf_counter()
        try:
            frames, day_manifest, provenance = source.load(date)
            cube = build_day_cube(frames, day_manifest, truth)
            pooled, strata, geometry_audit = evaluate_day(
                cube, truth, diagnostic_config
            )
            prefix_metadata = _metadata_prefix(
                scenario_index,
                scenario,
                truth,
                date,
                day_summary,
                diagnostic_signature,
                truth_sha256,
                source_run_request_sha256,
            )
            summary = {**prefix_metadata, **pooled}
            stratum_rows = [
                {**prefix_metadata, **row}
                for row in strata
            ]
            record = {
                "schema_version": SCHEMA_VERSION,
                "status": "complete",
                "study_id": diagnostic_config["study_id"],
                "diagnostic_signature": diagnostic_signature,
                "diagnostic_config_sha256": sha256_file(config_path),
                "truth_sha256": truth_sha256,
                "source_run_request_sha256": source_run_request_sha256,
                **provenance,
                "scenario_id": str(scenario["scenario_id"]),
                "family": str(truth["family"]),
                "interaction_eta": float(truth["interaction_eta"]),
                "date": date,
                "block_index": int(day_summary["block_index"]),
                "seed": int(day_summary["seed"]),
                "summary": summary,
                "strata": stratum_rows,
                "geometry_audit": geometry_audit,
                "elapsed_seconds": time.perf_counter() - started,
                "completed_utc": datetime.now(timezone.utc).isoformat(),
            }
            atomic_json(checkpoint_path, record)
            failure_path = scenario_dir / "failures" / f"{date}.json"
            failure_path.unlink(missing_ok=True)
            # Transaction order is intentional: JSON is the source of truth;
            # CSVs are regenerated views and therefore self-heal on restart.
            progress = rebuild_scenario_outputs(
                scenario_dir,
                str(scenario["scenario_id"]),
                selected_dates,
                diagnostic_signature,
                truth_sha256,
                source_run_request_sha256,
            )
            rebuild_master_outputs(
                output_root,
                scenario_ids,
                selected_dates,
                diagnostic_signature,
                input_identities,
            )
            print(
                f"completed {scenario['scenario_id']} {date}: "
                f"n={summary['sample_count']} "
                f"empirical_cross={summary['empirical_l_cross_term']:.6g} "
                f"truth_cross={summary['truth_l_cross_term']:.6g} "
                f"excess_vs_sep={summary['empirical_l_cross_excess_vs_separable']:.6g} "
                f"eta_hat={summary['eta_hat_from_cross_center']:.6g} "
                f"elapsed={record['elapsed_seconds']:.1f}s",
                flush=True,
            )
        except Exception as error:
            atomic_json(
                scenario_dir / "failures" / f"{date}.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "diagnostic_signature": diagnostic_signature,
                    "scenario_id": str(scenario["scenario_id"]),
                    "date": date,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "failed_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            raise

    progress = rebuild_scenario_outputs(
        scenario_dir,
        str(scenario["scenario_id"]),
        selected_dates,
        diagnostic_signature,
        truth_sha256,
        source_run_request_sha256,
    )
    _update_scenario_complete(
        scenario_dir,
        str(scenario["scenario_id"]),
        selected_dates,
        diagnostic_signature,
        truth_sha256,
        source_run_request_sha256,
        progress,
    )
    master = rebuild_master_outputs(
        output_root,
        scenario_ids,
        selected_dates,
        diagnostic_signature,
        input_identities,
    )
    print(json.dumps({"scenario": progress, "master": master}, indent=2), flush=True)


if __name__ == "__main__":
    main()
