#!/usr/bin/env python3
"""Rebuild and summarize all available two-contrast daily JSON results."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[6]

from two_contrast_core import (  # noqa: E402
    SCHEMA_VERSION,
    atomic_csv,
    atomic_json,
    atomic_text,
    load_json,
    rebuild_master_outputs,
    rebuild_scenario_outputs,
    sha256_file,
    signature_for_files,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--config", type=Path, default=HERE / "two_contrast_diagnostic_092726.json"
    )
    result.add_argument("--output-root", type=Path)
    result.add_argument("--data-root", type=Path)
    result.add_argument("--require-complete", action="store_true")
    return result


def _scenario_config_path(config: dict[str, Any]) -> Path:
    value = Path(config["input"]["scenario_config"])
    return value if value.is_absolute() else PROJECT_ROOT / value


def _format(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):.6g}"


def _validate_saved_day_sources(
    scenario_dir: Path,
    year_dir: Path,
    truth: dict[str, Any],
    selected_dates: list[str],
) -> None:
    day_summaries = {
        str(value["date"]): dict(value) for value in truth["day_summaries"]
    }
    for date in selected_dates:
        checkpoint_path = scenario_dir / "day_json" / f"{date}.json"
        if not checkpoint_path.is_file():
            continue
        record = load_json(checkpoint_path)
        source_root = year_dir / "day_checkpoints" / date
        source_paths = {
            "source_success_sha256": source_root / "SUCCESS.json",
            "source_manifest_sha256": source_root / "manifest.csv",
            "source_gridded_sha256": source_root / "gridded.pkl",
        }
        for name, path in source_paths.items():
            if not path.is_file():
                raise FileNotFoundError(path)
            current = sha256_file(path)
            if record.get(name) != current:
                raise RuntimeError(
                    f"stale diagnostic checkpoint {checkpoint_path}: {name} changed"
                )
        expected_request = str(day_summaries[date]["day_request_sha256"])
        if record.get("source_day_request_sha256") != expected_request:
            raise RuntimeError(
                f"stale diagnostic checkpoint {checkpoint_path}: day request changed"
            )


def _report(frame: pd.DataFrame, progress: dict[str, Any]) -> str:
    lines = [
        "# Two-contrast simulation diagnostic",
        "",
        f"- Completed scenario-days: `{progress['completed_scenario_days']}` / "
        f"`{progress['expected_scenario_days']}`.",
        f"- Complete: `{progress['complete']}`.",
        "- Primary daily quantities: the weighted cross term "
        "`2*d1*d2*mean(Q_A*Q_B)` and its signed excess over the matched "
        "separable center; raw `mean(Q_A*Q_B)` is also retained.",
        "- Geometry: fixed selected Q_A/Q_B pair in truth-Lagrangian flow coordinates.",
        "- The inference unit is one independently generated day, not an overlapping anchor.",
        "",
    ]
    if frame.empty:
        lines.extend(["No completed daily results are available.", ""])
        return "\n".join(lines)
    lines.extend(
        [
            "## Available daily summaries",
            "",
            "| scenario | days | eta | empirical cross term | truth cross term | "
            "empirical excess vs sep | truth excess vs sep | mean eta-hat |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for (scenario, eta), group in frame.groupby(
        ["scenario_id", "interaction_eta"], sort=False
    ):
        eta_hat = pd.to_numeric(
            group["eta_hat_from_cross_center"], errors="coerce"
        ).dropna()
        lines.append(
            f"| {scenario} | {len(group)} | {_format(float(eta))} | "
            f"{_format(group['empirical_l_cross_term'].mean())} | "
            f"{_format(group['truth_l_cross_term'].mean())} | "
            f"{_format(group['empirical_l_cross_excess_vs_separable'].mean())} | "
            f"{_format(group['truth_l_cross_excess_vs_separable'].mean())} | "
            f"{_format(eta_hat.mean() if len(eta_hat) else None)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation guardrail",
            "",
            "The separable endpoint has a positive H_AB; this is not a zero-cross-term "
            "test. The interaction signal is the movement of H_AB away from its matched "
            "separable center. The saved `empirical_l_cross_excess_vs_separable` retains "
            "the signed negative contribution selected by the original two-rectangle "
            "search.",
            "The centered covariance columns are descriptive sensitivity checks only; "
            "overlapping translated contrasts make their sample-mean correction "
            "ineligible as an unbiased eta estimator.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parser().parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_json(config_path)
    if int(config.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported diagnostic configuration schema")
    scenario_config_path = _scenario_config_path(config).resolve()
    suite = load_json(scenario_config_path)
    scenario_ids = [str(value["scenario_id"]) for value in suite["scenarios"]]
    if len(scenario_ids) != 6 or len(set(scenario_ids)) != 6:
        raise ValueError("the production suite must contain exactly six scenarios")
    year = int(config["input"]["year"])
    data_root = (
        args.data_root.expanduser()
        if args.data_root is not None
        else Path(config["input"]["amarel_data_root"])
    )
    output_root = (
        args.output_root.expanduser()
        if args.output_root is not None
        else Path(config["output"]["amarel_output_root"])
    )
    signature = signature_for_files(
        (
            config_path,
            scenario_config_path,
            HERE / "run_two_contrast_diagnostic.py",
            HERE / "two_contrast_core.py",
        )
    )
    # Remove final-facing markers before validating current simulation inputs.
    # They are recreated only if every current source hash and checkpoint passes.
    for path in (
        output_root / "FINAL_COMPLETE.json",
        output_root / "master_progress.json",
        output_root / "completeness.csv",
        output_root / "RESULTS.md",
        output_root / "PROVISIONAL_RESULTS.md",
    ):
        path.unlink(missing_ok=True)
    for scenario_id in scenario_ids:
        scenario_dir = output_root / "scenarios" / scenario_id
        (scenario_dir / "COMPLETE.json").unlink(missing_ok=True)
        (scenario_dir / "progress.json").unlink(missing_ok=True)
    input_identities: dict[str, dict[str, str]] = {}
    truths: dict[str, dict[str, Any]] = {}
    year_dirs: dict[str, Path] = {}
    selected_dates: list[str] | None = None
    prefix = f"sim_july{year}_st_circulant"
    for scenario_id in scenario_ids:
        year_dir = data_root / scenario_id / f"{year}_july_st_circulant"
        truth_path = year_dir / f"{prefix}_truth.json"
        complete_path = year_dir / "COMPLETE.json"
        if not truth_path.is_file() or not complete_path.is_file():
            raise FileNotFoundError(f"simulation scenario is incomplete under {year_dir}")
        truth = load_json(truth_path)
        complete = load_json(complete_path)
        if truth.get("scenario_id") != scenario_id or complete.get(
            "scenario_id"
        ) != scenario_id:
            raise ValueError(f"simulation identity mismatch under {year_dir}")
        if truth.get("run_request_sha256") != complete.get("run_request_sha256"):
            raise ValueError(f"truth and COMPLETE request hashes differ under {year_dir}")
        dates = [str(value) for value in truth["selected_dates"]]
        if selected_dates is None:
            selected_dates = dates
        elif dates != selected_dates:
            raise ValueError("simulation scenarios do not share the same dates")
        truths[scenario_id] = truth
        year_dirs[scenario_id] = year_dir
        input_identities[scenario_id] = {
            "truth_sha256": sha256_file(truth_path),
            "source_run_request_sha256": str(truth["run_request_sha256"]),
        }
    assert selected_dates is not None
    if len(selected_dates) != int(config["input"]["expected_days"]):
        raise ValueError("simulation truth does not contain the configured day count")
    rows = []
    for scenario_id in scenario_ids:
        scenario_dir = output_root / "scenarios" / scenario_id
        identity = input_identities[scenario_id]
        _validate_saved_day_sources(
            scenario_dir,
            year_dirs[scenario_id],
            truths[scenario_id],
            selected_dates,
        )
        progress = rebuild_scenario_outputs(
            scenario_dir,
            scenario_id,
            selected_dates,
            signature,
            identity["truth_sha256"],
            identity["source_run_request_sha256"],
        )
        marker_path = scenario_dir / "COMPLETE.json"
        if int(progress["completed_days"]) == len(selected_dates):
            summary_path = scenario_dir / "daily_two_contrast_centers.csv"
            stratum_path = scenario_dir / "daily_two_contrast_strata.csv"
            atomic_json(
                marker_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "complete",
                    "diagnostic_signature": signature,
                    "truth_sha256": identity["truth_sha256"],
                    "source_run_request_sha256": identity[
                        "source_run_request_sha256"
                    ],
                    "scenario_id": scenario_id,
                    "completed_days": int(progress["completed_days"]),
                    "daily_csv_sha256": sha256_file(summary_path),
                    "stratum_csv_sha256": sha256_file(stratum_path),
                },
            )
        else:
            marker_path.unlink(missing_ok=True)
        rows.append({"scenario_id": scenario_id, **progress})
    atomic_csv(output_root / "completeness.csv", pd.DataFrame(rows))
    progress = rebuild_master_outputs(
        output_root, scenario_ids, selected_dates, signature, input_identities
    )
    master_path = output_root / "all_daily_two_contrast_centers.csv"
    try:
        master = pd.read_csv(master_path)
    except pd.errors.EmptyDataError:
        master = pd.DataFrame()
    report_name = "RESULTS.md" if progress["complete"] else "PROVISIONAL_RESULTS.md"
    atomic_text(output_root / report_name, _report(master, progress))
    if progress["complete"]:
        (output_root / "PROVISIONAL_RESULTS.md").unlink(missing_ok=True)
    else:
        (output_root / "RESULTS.md").unlink(missing_ok=True)
    print(progress)
    if args.require_complete and not progress["complete"]:
        raise SystemExit(
            f"incomplete diagnostic: {progress['completed_scenario_days']} of "
            f"{progress['expected_scenario_days']} scenario-days"
        )


if __name__ == "__main__":
    main()
