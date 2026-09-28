#!/usr/bin/env python3
"""Validate the completed 59-day adapted/fixed native-NLL rerun."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


METHODS = ("adapted", "fixed")
EXPECTED_DATES = tuple(
    f"{year}-07-{day:02d}"
    for year in (2024, 2025)
    for day in range(1, 31)
    if not (year == 2025 and day == 24)
)
EXPECTED_PAIRS = {(date, method) for date in EXPECTED_DATES for method in METHODS}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"{path.name} has no header")
        return list(reader)


def validate_pairs(rows: list[dict[str, Any]], label: str) -> None:
    pairs = [(str(row.get("date", "")), str(row.get("method", ""))) for row in rows]
    counts = Counter(pairs)
    duplicates = sorted(pair for pair, count in counts.items() if count != 1)
    missing = sorted(EXPECTED_PAIRS - set(pairs))
    unexpected = sorted(set(pairs) - EXPECTED_PAIRS)
    if len(rows) != 118 or duplicates or missing or unexpected:
        raise ValueError(
            f"{label}: expected 118 unique date-method rows; rows={len(rows)}, "
            f"duplicates={duplicates}, missing={missing}, unexpected={unexpected}"
        )


def derive_winners(rows: list[dict[str, str]]) -> dict[str, Any]:
    by_date: dict[str, dict[str, float]] = {}
    target_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        date = str(row["date"])
        method = str(row["method"])
        total = float(row["native_nll_total"])
        if not math.isfinite(total):
            raise ValueError(f"non-finite native_nll_total for {date} {method}")
        by_date.setdefault(date, {})[method] = total
        target_counts.setdefault(date, {})[method] = int(row["n_target_points"])

    unequal_targets = sorted(
        date
        for date, counts in target_counts.items()
        if counts["adapted"] != counts["fixed"]
    )
    if unequal_targets:
        raise ValueError(
            "adapted and fixed target counts differ on dates: " f"{unequal_targets}"
        )

    differences = {
        date: values["fixed"] - values["adapted"] for date, values in by_date.items()
    }
    adapted_dates = sorted(date for date, value in differences.items() if value > 0.0)
    fixed_dates = sorted(date for date, value in differences.items() if value < 0.0)
    tie_dates = sorted(date for date, value in differences.items() if value == 0.0)
    return {
        "n_paired_dates": len(differences),
        "adapted_wins": len(adapted_dates),
        "fixed_wins": len(fixed_dates),
        "exact_ties": len(tie_dates),
        "adapted_win_dates": adapted_dates,
        "fixed_win_dates": fixed_dates,
        "exact_tie_dates": tie_dates,
    }


def validate(root: Path) -> dict[str, Any]:
    required = (
        "fit_checkpoint_native_nll.json",
        "daily_fit_results.csv",
        "daily_native_nll.csv",
        "native_nll_summary.csv",
        "daily_winner_summary.json",
        "paper_daily_comparison.csv",
        "paper_nll_summary.csv",
        "paper_nll_summary.tex",
        "daily_native_nll.png",
        "run_config.json",
        "RUN_COMPLETE.json",
    )
    missing = [
        name for name in required if not (root / name).is_file() or (root / name).stat().st_size == 0
    ]
    if missing:
        raise ValueError(f"missing or empty required outputs: {missing}")

    fit_rows = read_csv(root / "daily_fit_results.csv")
    nll_rows = read_csv(root / "daily_native_nll.csv")
    validate_pairs(fit_rows, "daily_fit_results.csv")
    validate_pairs(nll_rows, "daily_native_nll.csv")
    lag_counts = {
        (
            int(row["lag0_block_count"]),
            int(row["lag1_block_count"]),
            int(row["lag2_block_count"]),
        )
        for row in fit_rows
    }
    if lag_counts != {(6, 4, 3)}:
        raise ValueError(f"daily_fit_results.csv has unexpected lag counts: {lag_counts}")

    paper_daily = read_csv(root / "paper_daily_comparison.csv")
    if len(paper_daily) != 59 or {row["date"] for row in paper_daily} != set(EXPECTED_DATES):
        raise ValueError("paper_daily_comparison.csv must have one row per expected date")
    paper_summary = read_csv(root / "paper_nll_summary.csv")
    if [row["period"] for row in paper_summary] != ["July 2024", "July 2025", "Overall"]:
        raise ValueError("paper_nll_summary.csv has unexpected period rows")

    nll_by_date: dict[str, dict[str, dict[str, str]]] = {}
    for row in nll_rows:
        nll_by_date.setdefault(row["date"], {})[row["method"]] = row
    for row in paper_daily:
        date = row["date"]
        adapted = nll_by_date[date]["adapted"]
        fixed = nll_by_date[date]["fixed"]
        n_target = int(adapted["n_target_points"])
        if int(row["n_target_points"]) != n_target:
            raise ValueError(f"paper_daily_comparison.csv target count mismatch on {date}")
        adapted_per_target = float(adapted["native_nll_per_target"])
        fixed_per_target = float(fixed["native_nll_per_target"])
        expected_delta = fixed_per_target - adapted_per_target
        if not math.isclose(
            float(row["fixed_minus_adapted_nll_per_target"]),
            expected_delta,
            rel_tol=1e-9,
            # Both source NLL columns were written independently with %.10g.
            # Subtracting the rounded values can accumulate about 1e-9 of
            # absolute error even when the full-precision checkpoint agrees.
            abs_tol=2e-9,
        ):
            raise ValueError(f"paper_daily_comparison.csv NLL delta mismatch on {date}")
        expected_winner = (
            "adapted" if expected_delta > 0.0 else "fixed" if expected_delta < 0.0 else "tie"
        )
        if row["winner"] != expected_winner:
            raise ValueError(f"paper_daily_comparison.csv winner mismatch on {date}")
        zero_initializer = (
            float(row["init_advec_lat"]) == 0.0
            and float(row["init_advec_lon"]) == 0.0
        )
        recorded_zero = row.get("zero_initializer_case", "").strip().lower()
        if recorded_zero and recorded_zero not in {"true", "false"}:
            raise ValueError(
                f"paper_daily_comparison.csv has invalid zero-initializer provenance on {date}"
            )
        if recorded_zero and (recorded_zero == "true") != zero_initializer:
            raise ValueError(
                f"paper_daily_comparison.csv zero-initializer mismatch on {date}"
            )

    summary_by_period = {row["period"]: row for row in paper_summary}
    expected_period_days = {"July 2024": 30, "July 2025": 29, "Overall": 59}
    for period, n_dates in expected_period_days.items():
        row = summary_by_period[period]
        if int(row["n_dates"]) != n_dates:
            raise ValueError(f"paper_nll_summary.csv has wrong date count for {period}")
        if (
            int(row["adapted_wins"])
            + int(row["fixed_wins"])
            + int(row["exact_ties"])
            != n_dates
        ):
            raise ValueError(f"paper_nll_summary.csv wins do not sum for {period}")
        year = 2024 if period == "July 2024" else 2025 if period == "July 2025" else None
        if "comparison_dates" in row:
            period_daily = [
                daily_row
                for daily_row in paper_daily
                if year is None or int(daily_row["year"]) == year
            ]
            zero_count = sum(
                daily_row["zero_initializer_case"].strip().lower() == "true"
                for daily_row in period_daily
            )
            comparison_count = n_dates - zero_count
            if int(row["zero_initializer_dates"]) != zero_count:
                raise ValueError(
                    f"paper_nll_summary.csv zero-initializer count mismatch for {period}"
                )
            if int(row["comparison_dates"]) != comparison_count:
                raise ValueError(
                    f"paper_nll_summary.csv comparison-date count mismatch for {period}"
                )
            adapted_lower = int(row["adapted_lower_comparison_dates"])
            fixed_lower = int(row["fixed_lower_comparison_dates"])
            if adapted_lower + fixed_lower != comparison_count:
                raise ValueError(
                    f"paper_nll_summary.csv lower-NLL counts do not sum for {period}"
                )
            expected_fraction = adapted_lower / comparison_count
            if not math.isclose(
                float(row["adapted_lower_fraction_comparison_dates"]),
                expected_fraction,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"paper_nll_summary.csv lower-NLL fraction mismatch for {period}"
                )

    checkpoint = read_json(root / "fit_checkpoint_native_nll.json")
    checkpoint_rows = checkpoint.get("records") if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(checkpoint_rows, list):
        raise ValueError("fit_checkpoint_native_nll.json has no records list")
    validate_pairs(checkpoint_rows, "fit_checkpoint_native_nll.json")

    derived = derive_winners(nll_rows)
    recorded = read_json(root / "daily_winner_summary.json")
    complete = read_json(root / "RUN_COMPLETE.json")
    config = read_json(root / "run_config.json")
    if config.get("lag_block_counts") != [6, 4, 3]:
        raise ValueError("run_config.json does not report lag counts 6/4/3")
    if complete.get("lag_block_counts") != [6, 4, 3]:
        raise ValueError("RUN_COMPLETE.json does not report lag counts 6/4/3")
    for key in ("n_paired_dates", "adapted_wins", "fixed_wins", "exact_ties"):
        if int(recorded.get(key, -1)) != int(derived[key]):
            raise ValueError(f"daily_winner_summary.json disagrees on {key}")
        if int(complete.get(key, -1)) != int(derived[key]):
            raise ValueError(f"RUN_COMPLETE.json disagrees on {key}")
    if complete.get("n_dates") != 59 or complete.get("n_fit_rows") != 118:
        raise ValueError("RUN_COMPLETE.json does not report 59 dates and 118 fits")
    if complete.get("device") != "cuda":
        raise ValueError(f"RUN_COMPLETE.json device is not cuda: {complete.get('device')!r}")
    if not complete.get("cuda_device_name"):
        raise ValueError("RUN_COMPLETE.json does not record the CUDA device name")

    return {
        "result_directory": str(root),
        "validated_dates": 59,
        "validated_fit_rows": 118,
        "device": "cuda",
        "cuda_device_name": complete["cuda_device_name"],
        **derived,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_directory", type=Path)
    args = parser.parse_args()
    root = args.result_directory.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"result directory does not exist: {root}")
    print(json.dumps(validate(root), indent=2))


if __name__ == "__main__":
    main()
