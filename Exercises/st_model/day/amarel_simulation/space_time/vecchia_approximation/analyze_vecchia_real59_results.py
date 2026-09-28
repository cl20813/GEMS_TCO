#!/usr/bin/env python3
"""Audit and summarize the completed 59-day adapted/fixed Vecchia run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from verify_vecchia_real59_nll_results import validate as validate_download


METHODS = ("adapted", "fixed")
THRESHOLDS = (0.25, 0.50, 1.00)
PRIMARY_NEAR_ZERO_THRESHOLD = 0.50


def linear_quantile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("quantile requires at least one value")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + weight * (ordered[upper] - ordered[lower])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def pct(count: int, total: int) -> float:
    return 100.0 * count / total if total else math.nan


def load_checkpoint(root: Path) -> list[dict[str, Any]]:
    payload = json.loads((root / "fit_checkpoint_native_nll.json").read_text())
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError("checkpoint does not contain a records list")
    return records


def build_daily_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_date: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        date = str(record["date"])
        method = str(record["method"])
        if method not in METHODS or method in by_date[date]:
            raise ValueError(f"duplicate or unexpected pair: {date} {method}")
        by_date[date][method] = record

    rows: list[dict[str, Any]] = []
    for date in sorted(by_date):
        if set(by_date[date]) != set(METHODS):
            raise ValueError(f"incomplete method pair for {date}")
        adapted = by_date[date]["adapted"]
        fixed = by_date[date]["fixed"]
        n_target = int(adapted["n_target_points"])
        if n_target != int(fixed["n_target_points"]):
            raise ValueError(f"target counts differ on {date}")

        for record in (adapted, fixed):
            for field in (
                "native_nll_per_target",
                "native_nll_total",
                "advec_lat",
                "advec_lon",
                "grid_step_lat",
                "grid_step_lon",
                "init_advec_lat",
                "init_advec_lon",
                "max_abs_gradient",
            ):
                if not math.isfinite(float(record[field])):
                    raise ValueError(f"non-finite {field} for {date} {record['method']}")
            expected_total = float(record["native_nll_per_target"]) * n_target
            if not math.isclose(
                float(record["native_nll_total"]),
                expected_total,
                rel_tol=2e-15,
                abs_tol=2e-8,
            ):
                raise ValueError(f"NLL total does not reconcile for {date} {record['method']}")
            lag_counts = tuple(int(record[f"lag{lag}_block_count"]) for lag in range(3))
            if lag_counts != (6, 4, 3):
                raise ValueError(f"unexpected lag counts for {date} {record['method']}: {lag_counts}")

        for field in ("grid_step_lat", "grid_step_lon", "init_advec_lat", "init_advec_lon"):
            if not math.isclose(
                float(adapted[field]), float(fixed[field]), rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(f"paired {field} differs on {date}")

        step_lat = abs(float(adapted["grid_step_lat"]))
        step_lon = abs(float(adapted["grid_step_lon"]))
        if step_lat == 0.0 or step_lon == 0.0:
            raise ValueError(f"zero grid step on {date}")

        adapted_nll = float(adapted["native_nll_per_target"])
        fixed_nll = float(fixed["native_nll_per_target"])
        delta = fixed_nll - adapted_nll
        delta_total = float(fixed["native_nll_total"]) - float(adapted["native_nll_total"])
        if not math.isclose(delta_total, delta * n_target, rel_tol=1e-10, abs_tol=2e-8):
            raise ValueError(f"paired NLL delta does not reconcile on {date}")
        # Compare unrounded total NLLs. Per-target NLL remains a diagnostic,
        # whereas the paper reports the contribution after multiplying by the
        # number of target observations.
        winner = (
            "adapted" if delta_total > 0.0 else "fixed" if delta_total < 0.0 else "tie"
        )
        zero_initializer_case = (
            float(adapted["init_advec_lat"]) == 0.0
            and float(adapted["init_advec_lon"]) == 0.0
        )

        adapted_lat_cells = float(adapted["advec_lat"]) / step_lat
        adapted_lon_cells = float(adapted["advec_lon"]) / step_lon
        fixed_lat_cells = float(fixed["advec_lat"]) / step_lat
        fixed_lon_cells = float(fixed["advec_lon"]) / step_lon
        init_lat_cells = float(adapted["init_advec_lat"]) / step_lat
        init_lon_cells = float(adapted["init_advec_lon"]) / step_lon
        adapted_magnitude = math.hypot(adapted_lat_cells, adapted_lon_cells)
        fixed_magnitude = math.hypot(fixed_lat_cells, fixed_lon_cells)
        init_magnitude = math.hypot(init_lat_cells, init_lon_cells)

        rows.append(
            {
                "date": date,
                "year": int(adapted["year"]),
                "n_target_points": n_target,
                "adapted_nll_per_target": adapted_nll,
                "fixed_nll_per_target": fixed_nll,
                "fixed_minus_adapted_nll_per_target": delta,
                "adapted_nll_total": float(adapted["native_nll_total"]),
                "fixed_nll_total": float(fixed["native_nll_total"]),
                "fixed_minus_adapted_total_nll": delta_total,
                "winner": winner,
                "zero_initializer_case": zero_initializer_case,
                "grid_step_lat_deg": step_lat,
                "grid_step_lon_deg": step_lon,
                "adapted_fit_v_lat_deg_per_hour": float(adapted["advec_lat"]),
                "adapted_fit_v_lon_deg_per_hour": float(adapted["advec_lon"]),
                "adapted_fit_v_lat_cells_per_hour": adapted_lat_cells,
                "adapted_fit_v_lon_cells_per_hour": adapted_lon_cells,
                "adapted_fit_magnitude_cells_per_hour": adapted_magnitude,
                "fixed_fit_v_lat_deg_per_hour": float(fixed["advec_lat"]),
                "fixed_fit_v_lon_deg_per_hour": float(fixed["advec_lon"]),
                "fixed_fit_magnitude_cells_per_hour": fixed_magnitude,
                "initializer_v_lat_deg_per_hour": float(adapted["init_advec_lat"]),
                "initializer_v_lon_deg_per_hour": float(adapted["init_advec_lon"]),
                "initializer_magnitude_cells_per_hour": init_magnitude,
                "adapted_max_abs_gradient": float(adapted["max_abs_gradient"]),
                "fixed_max_abs_gradient": float(fixed["max_abs_gradient"]),
                "adapted_outer_steps": int(adapted["outer_steps"]),
                "fixed_outer_steps": int(fixed["outer_steps"]),
                "adapted_precompute_seconds": float(adapted["precompute_seconds"]),
                "fixed_precompute_seconds": float(fixed["precompute_seconds"]),
                "adapted_fit_seconds": float(adapted["fit_time_seconds"]),
                "fixed_fit_seconds": float(fixed["fit_time_seconds"]),
            }
        )
    if len(rows) != 59:
        raise ValueError(f"expected 59 paired dates, found {len(rows)}")
    return rows


def summarize_period(period: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    targets = sum(int(row["n_target_points"]) for row in rows)
    adapted_total = sum(float(row["adapted_nll_total"]) for row in rows)
    fixed_total = sum(float(row["fixed_nll_total"]) for row in rows)
    deltas_per_target = [
        float(row["fixed_minus_adapted_nll_per_target"]) for row in rows
    ]
    deltas_total = [float(row["fixed_minus_adapted_total_nll"]) for row in rows]
    adapted_wins = sum(row["winner"] == "adapted" for row in rows)
    fixed_wins = sum(row["winner"] == "fixed" for row in rows)
    ties = sum(row["winner"] == "tie" for row in rows)
    non_ties = adapted_wins + fixed_wins
    zero_initializer_dates = sum(bool(row["zero_initializer_case"]) for row in rows)
    comparison_dates = len(rows) - zero_initializer_dates
    adapted_lower_comparison = sum(
        not bool(row["zero_initializer_case"])
        and float(row["fixed_minus_adapted_total_nll"]) > 0.0
        for row in rows
    )
    fixed_lower_comparison = sum(
        not bool(row["zero_initializer_case"])
        and float(row["fixed_minus_adapted_total_nll"]) < 0.0
        for row in rows
    )
    return {
        "period": period,
        "n_dates": len(rows),
        "total_target_points": targets,
        "adapted_wins": adapted_wins,
        "fixed_wins": fixed_wins,
        "exact_ties": ties,
        "zero_initializer_dates": zero_initializer_dates,
        "comparison_dates": comparison_dates,
        "adapted_lower_comparison_dates": adapted_lower_comparison,
        "fixed_lower_comparison_dates": fixed_lower_comparison,
        "adapted_lower_percent_comparison_dates": pct(
            adapted_lower_comparison, comparison_dates
        ),
        "adapted_win_percent_all_days": pct(adapted_wins, len(rows)),
        "adapted_win_percent_non_tied_days": pct(adapted_wins, non_ties),
        "adapted_total_nll": adapted_total,
        "fixed_total_nll": fixed_total,
        "fixed_minus_adapted_total_nll": fixed_total - adapted_total,
        "adapted_mean_daily_total_nll": adapted_total / len(rows),
        "fixed_mean_daily_total_nll": fixed_total / len(rows),
        "fixed_minus_adapted_mean_daily_total_nll": (
            fixed_total - adapted_total
        )
        / len(rows),
        "adapted_pooled_nll_per_target": adapted_total / targets,
        "fixed_pooled_nll_per_target": fixed_total / targets,
        "fixed_minus_adapted_pooled_nll_per_target": (fixed_total - adapted_total)
        / targets,
        "mean_daily_fixed_minus_adapted_total_nll": statistics.mean(deltas_total),
        "median_daily_fixed_minus_adapted_total_nll": statistics.median(
            deltas_total
        ),
        "q25_daily_fixed_minus_adapted_total_nll": linear_quantile(
            deltas_total, 0.25
        ),
        "q75_daily_fixed_minus_adapted_total_nll": linear_quantile(
            deltas_total, 0.75
        ),
        "mean_daily_fixed_minus_adapted_nll_per_target": statistics.mean(
            deltas_per_target
        ),
        "median_daily_fixed_minus_adapted_nll_per_target": statistics.median(
            deltas_per_target
        ),
        "q25_daily_fixed_minus_adapted_nll_per_target": linear_quantile(
            deltas_per_target, 0.25
        ),
        "q75_daily_fixed_minus_adapted_nll_per_target": linear_quantile(
            deltas_per_target, 0.75
        ),
    }


def paper_summary_records(
    summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return the runner-compatible paper-summary schema with explicit denominators."""
    records: list[dict[str, Any]] = []
    for row in summaries:
        records.append(
            {
                "period": row["period"],
                "n_dates": row["n_dates"],
                "total_target_points": row["total_target_points"],
                "adapted_wins": row["adapted_wins"],
                "fixed_wins": row["fixed_wins"],
                "exact_ties": row["exact_ties"],
                "zero_initializer_dates": row["zero_initializer_dates"],
                "comparison_dates": row["comparison_dates"],
                "adapted_lower_comparison_dates": row[
                    "adapted_lower_comparison_dates"
                ],
                "fixed_lower_comparison_dates": row[
                    "fixed_lower_comparison_dates"
                ],
                "adapted_lower_fraction_comparison_dates": row[
                    "adapted_lower_percent_comparison_dates"
                ]
                / 100.0,
                "adapted_total_nll": row["adapted_total_nll"],
                "fixed_total_nll": row["fixed_total_nll"],
                "fixed_minus_adapted_total_nll": row[
                    "fixed_minus_adapted_total_nll"
                ],
                "adapted_mean_daily_total_nll": row[
                    "adapted_mean_daily_total_nll"
                ],
                "fixed_mean_daily_total_nll": row["fixed_mean_daily_total_nll"],
                "fixed_minus_adapted_mean_daily_total_nll": row[
                    "fixed_minus_adapted_mean_daily_total_nll"
                ],
                "adapted_pooled_nll_per_target": row[
                    "adapted_pooled_nll_per_target"
                ],
                "fixed_pooled_nll_per_target": row[
                    "fixed_pooled_nll_per_target"
                ],
                "fixed_minus_adapted_pooled_nll_per_target": row[
                    "fixed_minus_adapted_pooled_nll_per_target"
                ],
                "mean_daily_fixed_minus_adapted_nll_per_target": row[
                    "mean_daily_fixed_minus_adapted_nll_per_target"
                ],
                "median_daily_fixed_minus_adapted_nll_per_target": row[
                    "median_daily_fixed_minus_adapted_nll_per_target"
                ],
                "q25_daily_fixed_minus_adapted_nll_per_target": row[
                    "q25_daily_fixed_minus_adapted_nll_per_target"
                ],
                "q75_daily_fixed_minus_adapted_nll_per_target": row[
                    "q75_daily_fixed_minus_adapted_nll_per_target"
                ],
            }
        )
    return records


def paper_daily_records(daily: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the runner-compatible daily comparison, including zero-init provenance."""
    records: list[dict[str, Any]] = []
    for row in daily:
        init_lat = float(row["initializer_v_lat_deg_per_hour"])
        init_lon = float(row["initializer_v_lon_deg_per_hour"])
        records.append(
            {
                "date": row["date"],
                "year": row["year"],
                "n_target_points": row["n_target_points"],
                "adapted_native_nll_per_target": row["adapted_nll_per_target"],
                "fixed_native_nll_per_target": row["fixed_nll_per_target"],
                "fixed_minus_adapted_nll_per_target": row[
                    "fixed_minus_adapted_nll_per_target"
                ],
                "adapted_native_nll_total": row["adapted_nll_total"],
                "fixed_native_nll_total": row["fixed_nll_total"],
                "fixed_minus_adapted_total_nll": row[
                    "fixed_minus_adapted_total_nll"
                ],
                "winner": row["winner"],
                "zero_initializer_case": row["zero_initializer_case"],
                "init_advec_lat": init_lat,
                "init_advec_lon": init_lon,
                "init_advec_magnitude_coordinate_units": math.hypot(
                    init_lat, init_lon
                ),
                "init_advec_magnitude_grid_cells": row[
                    "initializer_magnitude_cells_per_hour"
                ],
                "adapted_fit_advec_lat": row["adapted_fit_v_lat_deg_per_hour"],
                "adapted_fit_advec_lon": row["adapted_fit_v_lon_deg_per_hour"],
                "adapted_fit_advec_magnitude_grid_cells": row[
                    "adapted_fit_magnitude_cells_per_hour"
                ],
                "fixed_fit_advec_lat": row["fixed_fit_v_lat_deg_per_hour"],
                "fixed_fit_advec_lon": row["fixed_fit_v_lon_deg_per_hour"],
                "fixed_fit_advec_magnitude_grid_cells": row[
                    "fixed_fit_magnitude_cells_per_hour"
                ],
                "adapted_precompute_seconds": row["adapted_precompute_seconds"],
                "fixed_precompute_seconds": row["fixed_precompute_seconds"],
                "adapted_fit_seconds": row["adapted_fit_seconds"],
                "fixed_fit_seconds": row["fixed_fit_seconds"],
            }
        )
    return records


def summarize_advection(label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    magnitudes = [float(row["adapted_fit_magnitude_cells_per_hour"]) for row in rows]
    result: dict[str, Any] = {
        "outcome_group": label,
        "n_dates": len(rows),
        "median_cells_per_hour": statistics.median(magnitudes),
        "q25_cells_per_hour": linear_quantile(magnitudes, 0.25),
        "q75_cells_per_hour": linear_quantile(magnitudes, 0.75),
        "minimum_cells_per_hour": min(magnitudes),
        "maximum_cells_per_hour": max(magnitudes),
    }
    for threshold in THRESHOLDS:
        tag = str(threshold).replace(".", "_")
        count = sum(value <= threshold for value in magnitudes)
        result[f"n_le_{tag}_cells_per_hour"] = count
        result[f"percent_le_{tag}_cells_per_hour"] = pct(count, len(rows))
    return result


def write_nll_latex(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Mean daily constant-free profiled Vecchia negative log-likelihood (NLL) for the two conditioning constructions.}",
        r"\label{tab:real-data-native-nll}",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\hline",
        r" & & \multicolumn{2}{c}{Mean daily total NLL} & Adapted lower NLL \\",
        r"Period & Days & Adapted & Fixed & $n/N$ (\%) \\",
        r"\hline",
    ]
    for row in rows:
        comparison_dates = row["comparison_dates"]
        adapted_lower = (
            f"{row['adapted_lower_comparison_dates']}/{comparison_dates} "
            f"({row['adapted_lower_percent_comparison_dates']:.1f})"
        )
        period = row["period"]
        lines.append(
            f"{period} & {row['n_dates']} & "
            f"\\textbf{{{row['adapted_mean_daily_total_nll']:,.0f}}} & "
            f"{row['fixed_mean_daily_total_nll']:,.0f} & "
            f"{adapted_lower} \\\\"
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\par\smallskip",
            r"\footnotesize\textit{Note:} Daily total NLL is NLL per target multiplied by the number "
            r"of target observations. Reported means include all dates, and text "
            r"differences use unrounded values. In the last column, $N$ excludes the "
            r"two zero-initializer dates (one per year), on which the two "
            r"constructions coincide.",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_markdown(
    path: Path,
    summaries: list[dict[str, Any]],
    advection_groups: list[dict[str, Any]],
    daily: list[dict[str, Any]],
    audit: dict[str, Any],
) -> None:
    overall = next(row for row in summaries if row["period"] == "Overall")
    fixed_group = next(row for row in advection_groups if row["outcome_group"] == "Fixed wins")
    adapted_group = next(row for row in advection_groups if row["outcome_group"] == "Adapted wins")
    gradient_over = audit["fits_with_gradient_above_1e_5"]
    comparison_dates = overall["comparison_dates"]

    lines = [
        "# Adapted versus fixed Vecchia result summary",
        "",
        "## Main result",
        "",
        (
            f"Across all {overall['n_dates']} dates, mean daily total NLL was "
            f"{overall['adapted_mean_daily_total_nll']:,.0f} for the adapted graph and "
            f"{overall['fixed_mean_daily_total_nll']:,.0f} for the fixed graph. "
            f"The adapted graph reduced mean daily NLL by "
            f"{overall['fixed_minus_adapted_mean_daily_total_nll']:,.2f}; the "
            f"cumulative reduction was {overall['fixed_minus_adapted_total_nll']:,.2f}."
        ),
        "",
        (
            f"After excluding the two zero-initializer dates, the adapted graph had "
            f"lower NLL on {overall['adapted_lower_comparison_dates']} of "
            f"{comparison_dates} dates "
            f"({overall['adapted_lower_percent_comparison_dates']:.1f}%). The excluded "
            "dates are retained in the mean NLL."
        ),
        "",
        "## NLL comparison",
        "",
        "| Period | Days | Adapted mean daily total NLL | Fixed mean daily total NLL | Adapted lower NLL, n/N (%) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        period = row["period"]
        row_comparison_dates = row["comparison_dates"]
        lines.append(
            f"| {period} | {row['n_dates']} | "
            f"{row['adapted_mean_daily_total_nll']:,.0f} | "
            f"{row['fixed_mean_daily_total_nll']:,.0f} | "
            f"{row['adapted_lower_comparison_dates']}/{row_comparison_dates} "
            f"({row['adapted_lower_percent_comparison_dates']:.1f}%) |"
        )
    lines.extend(
        [
            "",
            "Daily total NLL is NLL per target multiplied by the number of target observations. Means include all dates.",
            "",
            (
                f"On {fixed_group['n_le_0_5_cells_per_hour']}/{fixed_group['n_dates']} "
                f"({fixed_group['percent_le_0_5_cells_per_hour']:.1f}%) fixed-lower "
                "dates, the fitted advection magnitude under the adapted construction "
                "was no greater than 0.5 grid cells/hour. The corresponding "
                f"count was {adapted_group['n_le_0_5_cells_per_hour']}/"
                f"{adapted_group['n_dates']} "
                f"({adapted_group['percent_le_0_5_cells_per_hour']:.1f}%) among dates "
                "favoring the adapted graph. This association is descriptive."
            ),
            "",
        ]
    )
    lines.extend(
        [
            "",
            "## Interpretation and checks",
            "",
            "- NLL is the constant-free profiled Vecchia objective evaluated on each method's own conditioning graph. Lower is better.",
            "- Reported NLL differences are computed from unrounded values.",
            "- The half-grid-cell threshold is a resolution-based descriptive definition of near zero, not a hypothesis test.",
            (
                f"- All 118 fits were finite and reconciled to the checkpoint. "
                f"{gradient_over}/118 fits ended with a recorded maximum absolute gradient "
                "above the nominal 1e-5 tolerance; the run capped the outer optimization at five steps. "
                "The largest recorded value was "
                f"{audit['maximum_recorded_gradient']:.3g}."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_paper_paragraph(
    path: Path,
    summaries: list[dict[str, Any]],
    advection_groups: list[dict[str, Any]],
    daily: list[dict[str, Any]],
) -> None:
    overall = next(row for row in summaries if row["period"] == "Overall")
    fixed_group = next(row for row in advection_groups if row["outcome_group"] == "Fixed wins")
    adapted_group = next(row for row in advection_groups if row["outcome_group"] == "Adapted wins")
    comparison_dates = overall["comparison_dates"]
    paragraph = (
        f"Across all {overall['n_dates']} dates, the mean daily total constant-free "
        f"profiled Vecchia negative log-likelihood (NLL) was "
        f"{overall['adapted_mean_daily_total_nll']:,.0f} "
        f"for the advection-adapted graph and "
        f"{overall['fixed_mean_daily_total_nll']:,.0f} for the fixed graph. "
        f"The adapted graph reduced mean daily NLL by "
        f"{overall['fixed_minus_adapted_mean_daily_total_nll']:,.2f}, corresponding "
        f"to a cumulative reduction of {overall['fixed_minus_adapted_total_nll']:,.2f}. "
        f"After excluding the two zero-initializer dates, the adapted graph had lower "
        f"NLL on {overall['adapted_lower_comparison_dates']} of "
        f"{comparison_dates} dates "
        f"({overall['adapted_lower_percent_comparison_dates']:.1f}\\%). On "
        f"{fixed_group['n_le_0_5_cells_per_hour']}/{fixed_group['n_dates']} "
        f"({fixed_group['percent_le_0_5_cells_per_hour']:.1f}\\%) fixed-lower dates, "
        f"the fitted advection magnitude under the adapted construction was no "
        f"greater than 0.5 grid cells per hour, compared with "
        f"{adapted_group['n_le_0_5_cells_per_hour']}/{adapted_group['n_dates']} "
        f"({adapted_group['percent_le_0_5_cells_per_hour']:.1f}\\%) among dates "
        f"favoring the adapted graph. This association is descriptive."
    )
    path.write_text(paragraph + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_directory", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument(
        "--sync-paper-artifacts",
        action="store_true",
        help="also refresh paper_daily_comparison.csv and paper_nll_summary.csv/.tex in the result directory",
    )
    args = parser.parse_args()
    root = args.result_directory.expanduser().resolve()
    output = (
        args.output_directory.expanduser().resolve()
        if args.output_directory
        else root / "analysis"
    )
    output.mkdir(parents=True, exist_ok=True)

    transfer_validation = validate_download(root)
    records = load_checkpoint(root)
    daily = build_daily_rows(records)
    summaries = [
        summarize_period("July 2024", [row for row in daily if row["year"] == 2024]),
        summarize_period("July 2025", [row for row in daily if row["year"] == 2025]),
        summarize_period("Overall", daily),
    ]
    advection_groups = [
        summarize_advection("Adapted wins", [row for row in daily if row["winner"] == "adapted"]),
        summarize_advection("Fixed wins", [row for row in daily if row["winner"] == "fixed"]),
        summarize_advection(
            "Identical results", [row for row in daily if row["winner"] == "tie"]
        ),
    ]
    fixed_days = [row for row in daily if row["winner"] == "fixed"]
    ties = [row for row in daily if row["winner"] == "tie"]
    gradients = [
        float(record["max_abs_gradient"])
        for record in records
        if math.isfinite(float(record["max_abs_gradient"]))
    ]
    audit = {
        "transfer_validation": transfer_validation,
        "n_checkpoint_records": len(records),
        "n_paired_dates": len(daily),
        "n_fits_with_finite_gradient": len(gradients),
        "fits_with_gradient_above_1e_5": sum(value > 1e-5 for value in gradients),
        "maximum_recorded_gradient": max(gradients),
        "minimum_absolute_nonzero_daily_nll_delta": min(
            abs(float(row["fixed_minus_adapted_nll_per_target"]))
            for row in daily
            if float(row["fixed_minus_adapted_nll_per_target"]) != 0.0
        ),
        "tie_dates": [row["date"] for row in ties],
        "primary_near_zero_threshold_cells_per_hour": PRIMARY_NEAR_ZERO_THRESHOLD,
    }

    write_csv(output / "daily_paired_full_precision.csv", daily)
    paper_daily = paper_daily_records(daily)
    write_csv(output / "paper_daily_comparison.csv", paper_daily)
    write_csv(output / "nll_summary.csv", summaries)
    paper_summaries = paper_summary_records(summaries)
    write_csv(output / "paper_nll_summary.csv", paper_summaries)
    write_csv(output / "advection_by_outcome.csv", advection_groups)
    write_csv(output / "fixed_win_days.csv", fixed_days)
    write_nll_latex(output / "nll_summary_table.tex", summaries)
    write_nll_latex(output / "paper_nll_summary.tex", summaries)
    if args.sync_paper_artifacts:
        write_csv(root / "paper_daily_comparison.csv", paper_daily)
        write_csv(root / "paper_nll_summary.csv", paper_summaries)
        write_nll_latex(root / "paper_nll_summary.tex", summaries)
    # Keep the detailed advection diagnostics in CSV, but do not generate a
    # second main-text table. The compact paper summary reports the key counts
    # in prose instead.
    (output / "advection_by_outcome_table.tex").unlink(missing_ok=True)
    write_markdown(output / "analysis_summary.md", summaries, advection_groups, daily, audit)
    write_paper_paragraph(
        output / "paper_results_paragraph.tex", summaries, advection_groups, daily
    )
    (output / "analysis_summary.json").write_text(
        json.dumps(
            {
                "summary": summaries,
                "advection_by_outcome": advection_groups,
                "daily_paired": daily,
                "fixed_win_days": fixed_days,
                "audit": audit,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_directory": str(output), **audit}, indent=2))


if __name__ == "__main__":
    main()
