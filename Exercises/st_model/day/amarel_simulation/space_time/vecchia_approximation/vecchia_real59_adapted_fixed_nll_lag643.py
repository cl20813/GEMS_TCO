#!/usr/bin/env python3
"""Fit and compare adapted/fixed lag-6/4/3 Vecchia likelihoods.

The production window is July 1--30 in 2024 and 2025, excluding the
incomplete 2025-07-24 record, for 59 usable dates.  Each date is fitted twice:

* ``adapted`` uses the estimated advection corridor;
* ``fixed`` keeps past conditioning corridors centered on the target block.

This runner intentionally computes no conditional-eigen, dense-eigen, SLQ,
Lanczos, or Ritz diagnostic.  Its comparison target is the native Vecchia
negative log likelihood: each method's fitted objective on its own
conditioning graph, evaluated over all valid target observations.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
LOCAL_SRC = Path("/Users/joonwonlee/Documents/GEMS_TCO-1/src")
SOURCE_ROOT = Path(os.environ.get("GEMS_TCO_SRC", str(LOCAL_SRC))).resolve()
required_source = SOURCE_ROOT / "GEMS_TCO" / "data" / "loading.py"
if not required_source.is_file():
    raise RuntimeError(
        f"GEMS_TCO_SRC does not contain the required data loader: {required_source}"
    )
preferred_paths = [str(SOURCE_ROOT), str(HERE)]
sys.path[:] = preferred_paths + [
    entry for entry in sys.path if entry not in preferred_paths
]

import vecchia_adapted_fixed_lag643_core as core  # noqa: E402


METHODS = ("adapted", "fixed")
METHOD_ORDER = {method: index for index, method in enumerate(METHODS)}
LAG_COUNTS = (6, 4, 3)
if tuple(core.LAG_COUNTS) != LAG_COUNTS:
    raise RuntimeError(
        f"Runner requires lag counts {LAG_COUNTS}, found {tuple(core.LAG_COUNTS)}"
    )
COLORS = {"adapted": "#1f77b4", "fixed": "#d62728"}
LABELS = {"adapted": "adapted corridor", "fixed": "fixed center"}
PARAMETERS = (
    "sigmasq",
    "range_lat",
    "range_lon",
    "range_time",
    "advec_lat",
    "advec_lon",
    "nugget",
)
RESULT_COLUMNS = (
    "dataset_id",
    "year",
    "date",
    "method",
    *PARAMETERS,
    "precompute_seconds",
    "fit_time_seconds",
    "outer_steps",
    "max_abs_gradient",
    "init_advec_lat",
    "init_advec_lon",
    "grid_step_lat",
    "grid_step_lon",
    "initializer_seconds",
    "lag0_block_count",
    "lag1_block_count",
    "lag2_block_count",
    "n_target_points",
    "native_nll_per_target",
    "native_nll_total",
    "gls_beta",
)
CHECKPOINT_NAME = "fit_checkpoint_native_nll.json"
FIT_CSV_NAME = "daily_fit_results.csv"
NLL_CSV_NAME = "daily_native_nll.csv"
SUMMARY_CSV_NAME = "native_nll_summary.csv"
WINNER_SUMMARY_NAME = "daily_winner_summary.json"
PAPER_DAILY_CSV_NAME = "paper_daily_comparison.csv"
PAPER_SUMMARY_CSV_NAME = "paper_nll_summary.csv"
PAPER_SUMMARY_TEX_NAME = "paper_nll_summary.tex"
PLOT_NAME = "daily_native_nll.png"
CONFIG_NAME = "run_config.json"
COMPLETE_NAME = "RUN_COMPLETE.json"

# fit_one_geometry normally offers an optional conditional-eigen diagnostic.
# This runner is likelihood-only, so disable it before any fit is constructed.
core.EIGEN_GEOMETRIES = ()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_ready(value), indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def date_specs() -> list[dict[str, Any]]:
    specs = [
        {
            "dataset_id": f"real_{year}07{day:02d}",
            "data_kind": "real",
            "data_source": f"real_july_{year}",
            "year": year,
            "month": 7,
            "day": day,
            "date": f"{year}-07-{day:02d}",
        }
        for year in (2024, 2025)
        for day in range(1, 31)
        if not (year == 2025 and day == 24)
    ]
    if len(specs) != 59:
        raise RuntimeError(f"Internal date selection error: {len(specs)} dates")
    return specs


def normalize_record(record: dict[str, Any], source: Path) -> dict[str, Any]:
    """Normalize this runner's records and the earlier full-eigen checkpoint."""
    n_target_points = record.get("n_target_points", record.get("n_observations"))
    nll_per_target = record.get(
        "native_nll_per_target", record.get("native_nll_per_observation")
    )
    if n_target_points is None or nll_per_target is None:
        raise ValueError(
            f"{source}: record lacks target count or native Vecchia NLL"
        )
    n_target_points = int(n_target_points)
    nll_per_target = float(nll_per_target)
    total = record.get("native_nll_total", nll_per_target * n_target_points)
    normalized = {
        "dataset_id": str(record["dataset_id"]),
        "year": int(record["year"]),
        "date": str(record["date"]),
        "method": str(record["method"]),
        **{name: float(record[name]) for name in PARAMETERS},
        "precompute_seconds": float(record.get("precompute_seconds", 0.0)),
        "fit_time_seconds": float(record["fit_time_seconds"]),
        "outer_steps": int(record.get("outer_steps", 0)),
        "max_abs_gradient": record.get("max_abs_gradient"),
        "init_advec_lat": float(record["init_advec_lat"]),
        "init_advec_lon": float(record["init_advec_lon"]),
        "grid_step_lat": float(record.get("grid_step_lat", np.nan)),
        "grid_step_lon": float(record.get("grid_step_lon", np.nan)),
        "initializer_seconds": float(record.get("initializer_seconds", np.nan)),
        "lag0_block_count": int(record.get("lag0_block_count", LAG_COUNTS[0])),
        "lag1_block_count": int(record.get("lag1_block_count", LAG_COUNTS[1])),
        "lag2_block_count": int(record.get("lag2_block_count", LAG_COUNTS[2])),
        "n_target_points": n_target_points,
        "native_nll_per_target": nll_per_target,
        "native_nll_total": float(total),
        "gls_beta": [float(value) for value in record.get("gls_beta", [])],
    }
    if normalized["method"] not in METHODS:
        raise ValueError(f"{source}: unexpected method {normalized['method']!r}")
    if n_target_points <= 0 or not math.isfinite(nll_per_target):
        raise ValueError(f"{source}: invalid likelihood record {normalized}")
    return normalized


def read_checkpoint(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(f"{path} must contain a JSON list or a records object")
    normalized = []
    seen: set[tuple[str, str]] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{path}: record {index} is not an object")
        item = normalize_record(record, path)
        key = (item["dataset_id"], item["method"])
        if key in seen:
            raise ValueError(f"{path}: duplicate fit record {key}")
        seen.add(key)
        normalized.append(item)
    return normalized


def load_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    checkpoint = args.output_root / CHECKPOINT_NAME
    if checkpoint.is_file():
        if args.import_checkpoint is not None:
            print(
                f"Ignoring --import-checkpoint because {checkpoint} already exists",
                flush=True,
            )
        return read_checkpoint(checkpoint)
    if args.import_checkpoint is None:
        return []
    if not args.import_checkpoint.is_file():
        raise FileNotFoundError(args.import_checkpoint)
    records = read_checkpoint(args.import_checkpoint)
    print(
        f"Imported {len(records)} fit records from {args.import_checkpoint}",
        flush=True,
    )
    return records


def ordered_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda row: (str(row["date"]), METHOD_ORDER[str(row["method"])]),
    )


def results_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(ordered_records(records), columns=RESULT_COLUMNS)


def write_fit_csv(frame: pd.DataFrame, output_root: Path) -> None:
    columns = [
        "date",
        "method",
        *PARAMETERS,
        "precompute_seconds",
        "fit_time_seconds",
        "outer_steps",
        "max_abs_gradient",
        "init_advec_lat",
        "init_advec_lon",
        "grid_step_lat",
        "grid_step_lon",
        "initializer_seconds",
        "lag0_block_count",
        "lag1_block_count",
        "lag2_block_count",
    ]
    output = frame.loc[:, columns].copy()
    numeric = [column for column in columns if column not in {"date", "method"}]
    output.loc[:, numeric] = output.loc[:, numeric].apply(
        pd.to_numeric, errors="coerce"
    )
    output.to_csv(output_root / FIT_CSV_NAME, index=False, float_format="%.8g")


def add_paired_difference(frame: pd.DataFrame) -> pd.DataFrame:
    totals = frame.pivot(index="date", columns="method", values="native_nll_total")
    if set(METHODS).issubset(totals.columns):
        difference = totals["fixed"] - totals["adapted"]
    else:
        difference = pd.Series(dtype=float)
    output = frame.loc[
        :,
        [
            "date",
            "year",
            "method",
            "n_target_points",
            "native_nll_per_target",
            "native_nll_total",
        ],
    ].copy()
    output["fixed_minus_adapted_total_nll"] = output["date"].map(difference)
    return output


def build_winner_summary(frame: pd.DataFrame) -> dict[str, Any]:
    totals = frame.pivot(index="date", columns="method", values="native_nll_total")
    if not set(METHODS).issubset(totals.columns):
        difference = pd.Series(dtype=float)
    else:
        target_counts = frame.pivot(
            index="date", columns="method", values="n_target_points"
        ).dropna()
        unequal_targets = target_counts[
            target_counts["adapted"] != target_counts["fixed"]
        ]
        if not unequal_targets.empty:
            raise RuntimeError(
                "Adapted and fixed fits must use the same target observations; "
                f"mismatched dates: {list(unequal_targets.index)}"
            )
        difference = (totals["fixed"] - totals["adapted"]).dropna().sort_index()
    adapted_dates = [str(date) for date in difference.index[difference > 0.0]]
    fixed_dates = [str(date) for date in difference.index[difference < 0.0]]
    tie_dates = [str(date) for date in difference.index[difference == 0.0]]
    return {
        "criterion": "fixed total native Vecchia NLL minus adapted total native Vecchia NLL",
        "positive_means": "adapted has the lower native Vecchia NLL",
        "n_paired_dates": int(len(difference)),
        "adapted_wins": len(adapted_dates),
        "fixed_wins": len(fixed_dates),
        "exact_ties": len(tie_dates),
        "adapted_win_dates": adapted_dates,
        "fixed_win_dates": fixed_dates,
        "exact_tie_dates": tie_dates,
    }


def build_paper_daily_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one reproducible paired-comparison row per completed date."""
    columns = [
        "date",
        "year",
        "n_target_points",
        "adapted_native_nll_per_target",
        "fixed_native_nll_per_target",
        "fixed_minus_adapted_nll_per_target",
        "adapted_native_nll_total",
        "fixed_native_nll_total",
        "fixed_minus_adapted_total_nll",
        "winner",
        "zero_initializer_case",
        "init_advec_lat",
        "init_advec_lon",
        "init_advec_magnitude_coordinate_units",
        "init_advec_magnitude_grid_cells",
        "adapted_fit_advec_lat",
        "adapted_fit_advec_lon",
        "adapted_fit_advec_magnitude_grid_cells",
        "fixed_fit_advec_lat",
        "fixed_fit_advec_lon",
        "fixed_fit_advec_magnitude_grid_cells",
        "adapted_precompute_seconds",
        "fixed_precompute_seconds",
        "adapted_fit_seconds",
        "fixed_fit_seconds",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    duplicate = frame.duplicated(["date", "method"], keep=False)
    if duplicate.any():
        values = frame.loc[duplicate, ["date", "method"]].to_dict("records")
        raise RuntimeError(f"Duplicate date-method fit rows: {values}")

    rows: list[dict[str, Any]] = []
    for date, date_rows in frame.groupby("date", sort=True):
        indexed = date_rows.set_index("method")
        if not set(METHODS).issubset(indexed.index):
            continue
        adapted = indexed.loc["adapted"]
        fixed = indexed.loc["fixed"]
        if int(adapted["n_target_points"]) != int(fixed["n_target_points"]):
            raise RuntimeError(
                f"Adapted and fixed target counts differ on {date}: "
                f"{adapted['n_target_points']} versus {fixed['n_target_points']}"
            )
        for name in ("init_advec_lat", "init_advec_lon"):
            if not np.isclose(
                float(adapted[name]), float(fixed[name]), rtol=0.0, atol=1e-12
            ):
                raise RuntimeError(f"Adapted and fixed initializers differ on {date}")

        n_target = int(adapted["n_target_points"])
        adapted_total = float(adapted["native_nll_total"])
        fixed_total = float(fixed["native_nll_total"])
        adapted_per_target = float(adapted["native_nll_per_target"])
        fixed_per_target = float(fixed["native_nll_per_target"])
        delta_total = fixed_total - adapted_total
        delta_per_target = fixed_per_target - adapted_per_target
        if delta_total > 0.0:
            winner = "adapted"
        elif delta_total < 0.0:
            winner = "fixed"
        else:
            winner = "tie"

        lat_step = abs(float(adapted["grid_step_lat"]))
        lon_step = abs(float(adapted["grid_step_lon"]))

        def grid_magnitude(lat_value: float, lon_value: float) -> float:
            if not (
                math.isfinite(lat_step)
                and math.isfinite(lon_step)
                and lat_step > 0.0
                and lon_step > 0.0
            ):
                return np.nan
            return float(np.hypot(lat_value / lat_step, lon_value / lon_step))

        init_lat = float(adapted["init_advec_lat"])
        init_lon = float(adapted["init_advec_lon"])
        zero_initializer_case = init_lat == 0.0 and init_lon == 0.0
        adapted_lat = float(adapted["advec_lat"])
        adapted_lon = float(adapted["advec_lon"])
        fixed_lat = float(fixed["advec_lat"])
        fixed_lon = float(fixed["advec_lon"])
        rows.append(
            {
                "date": str(date),
                "year": int(adapted["year"]),
                "n_target_points": n_target,
                "adapted_native_nll_per_target": adapted_per_target,
                "fixed_native_nll_per_target": fixed_per_target,
                "fixed_minus_adapted_nll_per_target": delta_per_target,
                "adapted_native_nll_total": adapted_total,
                "fixed_native_nll_total": fixed_total,
                "fixed_minus_adapted_total_nll": delta_total,
                "winner": winner,
                "zero_initializer_case": zero_initializer_case,
                "init_advec_lat": init_lat,
                "init_advec_lon": init_lon,
                "init_advec_magnitude_coordinate_units": float(
                    np.hypot(init_lat, init_lon)
                ),
                "init_advec_magnitude_grid_cells": grid_magnitude(init_lat, init_lon),
                "adapted_fit_advec_lat": adapted_lat,
                "adapted_fit_advec_lon": adapted_lon,
                "adapted_fit_advec_magnitude_grid_cells": grid_magnitude(
                    adapted_lat, adapted_lon
                ),
                "fixed_fit_advec_lat": fixed_lat,
                "fixed_fit_advec_lon": fixed_lon,
                "fixed_fit_advec_magnitude_grid_cells": grid_magnitude(
                    fixed_lat, fixed_lon
                ),
                "adapted_precompute_seconds": float(adapted["precompute_seconds"]),
                "fixed_precompute_seconds": float(fixed["precompute_seconds"]),
                "adapted_fit_seconds": float(adapted["fit_time_seconds"]),
                "fixed_fit_seconds": float(fixed["fit_time_seconds"]),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_paper_summary(daily: pd.DataFrame) -> pd.DataFrame:
    """Build year-specific and pooled rows for the paper's main table."""
    columns = [
        "period",
        "n_dates",
        "total_target_points",
        "adapted_wins",
        "fixed_wins",
        "exact_ties",
        "zero_initializer_dates",
        "comparison_dates",
        "adapted_lower_comparison_dates",
        "fixed_lower_comparison_dates",
        "adapted_lower_fraction_comparison_dates",
        "adapted_total_nll",
        "fixed_total_nll",
        "fixed_minus_adapted_total_nll",
        "adapted_mean_daily_total_nll",
        "fixed_mean_daily_total_nll",
        "fixed_minus_adapted_mean_daily_total_nll",
        "adapted_pooled_nll_per_target",
        "fixed_pooled_nll_per_target",
        "fixed_minus_adapted_pooled_nll_per_target",
        "mean_daily_fixed_minus_adapted_nll_per_target",
        "median_daily_fixed_minus_adapted_nll_per_target",
        "q25_daily_fixed_minus_adapted_nll_per_target",
        "q75_daily_fixed_minus_adapted_nll_per_target",
    ]
    if daily.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    subsets = [
        ("July 2024", daily[daily["year"] == 2024]),
        ("July 2025", daily[daily["year"] == 2025]),
        ("Overall", daily),
    ]
    for period, subset in subsets:
        if subset.empty:
            continue
        targets = int(subset["n_target_points"].sum())
        adapted_total = float(subset["adapted_native_nll_total"].sum())
        fixed_total = float(subset["fixed_native_nll_total"].sum())
        delta = subset["fixed_minus_adapted_nll_per_target"].astype(float)
        adapted_wins = int((subset["winner"] == "adapted").sum())
        zero_initializer = subset["zero_initializer_case"].astype(bool)
        comparison_dates = int((~zero_initializer).sum())
        adapted_lower_comparison = int(
            ((~zero_initializer) & (subset["fixed_minus_adapted_total_nll"] > 0.0)).sum()
        )
        fixed_lower_comparison = int(
            ((~zero_initializer) & (subset["fixed_minus_adapted_total_nll"] < 0.0)).sum()
        )
        rows.append(
            {
                "period": period,
                "n_dates": int(len(subset)),
                "total_target_points": targets,
                "adapted_wins": adapted_wins,
                "fixed_wins": int((subset["winner"] == "fixed").sum()),
                "exact_ties": int((subset["winner"] == "tie").sum()),
                "zero_initializer_dates": int(zero_initializer.sum()),
                "comparison_dates": comparison_dates,
                "adapted_lower_comparison_dates": adapted_lower_comparison,
                "fixed_lower_comparison_dates": fixed_lower_comparison,
                "adapted_lower_fraction_comparison_dates": adapted_lower_comparison
                / float(comparison_dates),
                "adapted_total_nll": adapted_total,
                "fixed_total_nll": fixed_total,
                "fixed_minus_adapted_total_nll": fixed_total - adapted_total,
                "adapted_mean_daily_total_nll": adapted_total / float(len(subset)),
                "fixed_mean_daily_total_nll": fixed_total / float(len(subset)),
                "fixed_minus_adapted_mean_daily_total_nll": (
                    fixed_total - adapted_total
                )
                / float(len(subset)),
                "adapted_pooled_nll_per_target": adapted_total / targets,
                "fixed_pooled_nll_per_target": fixed_total / targets,
                "fixed_minus_adapted_pooled_nll_per_target": (
                    fixed_total - adapted_total
                )
                / targets,
                "mean_daily_fixed_minus_adapted_nll_per_target": float(delta.mean()),
                "median_daily_fixed_minus_adapted_nll_per_target": float(
                    delta.median()
                ),
                "q25_daily_fixed_minus_adapted_nll_per_target": float(
                    delta.quantile(0.25)
                ),
                "q75_daily_fixed_minus_adapted_nll_per_target": float(
                    delta.quantile(0.75)
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def write_paper_latex(summary: pd.DataFrame, path: Path) -> None:
    """Write a compact table fragment that needs no nonstandard table package."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Mean daily constant-free profiled Vecchia negative log-likelihood (NLL) for the "
        r"two conditioning constructions.}",
        r"\label{tab:real-data-native-nll}",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\hline",
        r" & & \multicolumn{2}{c}{Mean daily total NLL} & Adapted lower NLL \\",
        r"Period & Days & Adapted & Fixed & $n/N$ (\%) \\",
        r"\hline",
    ]
    for row in summary.itertuples(index=False):
        comparison_dates = int(row.comparison_dates)
        adapted_lower = int(row.adapted_lower_comparison_dates)
        adapted_percent = 100.0 * adapted_lower / comparison_dates
        period = row.period
        lines.append(
            f"{period} & {int(row.n_dates)} & "
            f"\\textbf{{{row.adapted_mean_daily_total_nll:,.0f}}} & "
            f"{row.fixed_mean_daily_total_nll:,.0f} & "
            f"{adapted_lower}/{comparison_dates} ({adapted_percent:.1f}) \\\\"
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


def write_nll_outputs(frame: pd.DataFrame, output_root: Path) -> None:
    nll = add_paired_difference(frame)
    nll.to_csv(
        output_root / NLL_CSV_NAME,
        index=False,
        float_format="%.10g",
    )
    if nll.empty:
        pd.DataFrame().to_csv(output_root / SUMMARY_CSV_NAME, index=False)
        build_paper_daily_comparison(frame).to_csv(
            output_root / PAPER_DAILY_CSV_NAME, index=False
        )
        empty_summary = build_paper_summary(pd.DataFrame())
        empty_summary.to_csv(output_root / PAPER_SUMMARY_CSV_NAME, index=False)
        write_paper_latex(empty_summary, output_root / PAPER_SUMMARY_TEX_NAME)
        return
    summary = (
        nll.groupby(["year", "method"], as_index=False)
        .agg(
            n_dates=("date", "nunique"),
            total_target_points=("n_target_points", "sum"),
            total_native_nll=("native_nll_total", "sum"),
            mean_daily_nll_per_target=("native_nll_per_target", "mean"),
        )
        .sort_values(["year", "method"])
    )
    summary["pooled_native_nll_per_target"] = (
        summary["total_native_nll"] / summary["total_target_points"]
    )
    summary.to_csv(
        output_root / SUMMARY_CSV_NAME,
        index=False,
        float_format="%.10g",
    )
    paper_daily = build_paper_daily_comparison(frame)
    paper_daily.to_csv(
        output_root / PAPER_DAILY_CSV_NAME,
        index=False,
        float_format="%.10g",
    )
    paper_summary = build_paper_summary(paper_daily)
    paper_summary.to_csv(
        output_root / PAPER_SUMMARY_CSV_NAME,
        index=False,
        float_format="%.10g",
    )
    write_paper_latex(paper_summary, output_root / PAPER_SUMMARY_TEX_NAME)
    write_json(output_root / WINNER_SUMMARY_NAME, build_winner_summary(frame))


def plot_native_nll(frame: pd.DataFrame, output_root: Path) -> None:
    if frame.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharey=True)
    for ax, year in zip(axes, (2024, 2025)):
        year_rows = frame[frame["year"] == year]
        for method in METHODS:
            rows = year_rows[year_rows["method"] == method].sort_values("date")
            ax.plot(
                rows["date"],
                rows["native_nll_per_target"],
                marker="o",
                markersize=3,
                linewidth=1.5,
                color=COLORS[method],
                label=LABELS[method],
            )
        ax.set_title(f"Real {year} July: native Vecchia NLL per target")
        ax.set_ylabel("NLL / target")
        ax.tick_params(axis="x", rotation=75, labelsize=7)
        ax.grid(alpha=0.22)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_root / PLOT_NAME, dpi=190, bbox_inches="tight")
    plt.close(fig)


def persist(records: list[dict[str, Any]], output_root: Path) -> None:
    compact = [
        {column: json_ready(record.get(column)) for column in RESULT_COLUMNS}
        for record in ordered_records(records)
    ]
    write_json(
        output_root / CHECKPOINT_NAME,
        {"schema_version": 1, "records": compact},
    )
    frame = results_frame(compact)
    write_fit_csv(frame, output_root)
    write_nll_outputs(frame, output_root)
    plot_native_nll(frame, output_root)


def build_fit_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        smooth=args.smooth,
        daily_stride=2,
        target_chunk_size=args.target_chunk_size,
        union_target_chunk_size=0,
        min_target_points=1,
        fixed_nugget=None,
        zero_nugget_fit_init=core.DEFAULT_REAL_INIT["nugget"],
        lbfgs_lr=args.lbfgs_lr,
        lbfgs_steps=args.lbfgs_steps,
        lbfgs_eval=args.lbfgs_eval,
        lbfgs_history=args.lbfgs_history,
        grad_tol=args.grad_tol,
        suppress_fit_prints=args.suppress_fit_prints,
        diag_chunk_size=1,
        union_diag_chunk_size=1,
        resample_grid=1,
        empirical_max_lat_offset=20,
        empirical_max_lon_offset=20,
        empirical_min_pair_count=1000,
        empirical_smooth_bandwidth_deg=0.063,
        subgrid_max_condition_number=100.0,
        lat_range=args.lat_range,
        lon_range=args.lon_range,
        real_data_root=args.real_data_root,
        synthetic_data_root=Path("/unused"),
        hours_per_day=8,
        keep_exact_loc=True,
        truth_nugget=None,
        device=args.device,
        require_cuda=args.device == "cuda",
    )


def clear_device(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def fit_day(
    spec: dict[str, Any],
    records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    completed = {
        str(record["method"])
        for record in records
        if str(record["dataset_id"]) == spec["dataset_id"]
    }
    if completed == set(METHODS):
        print("  Both likelihood fits already complete", flush=True)
        return

    fit_args = build_fit_args(args)
    device = core.resolve_device(fit_args)
    asset = core.load_real_asset(spec, fit_args)
    seed = core.m3_q3_seed(asset, fit_args)
    init = dict(core.DEFAULT_REAL_INIT)

    for method in METHODS:
        if method in completed:
            print(f"  Fit already complete: {method}", flush=True)
            continue
        clear_device(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        print(f"  Fitting {method}", flush=True)
        row, raw, _, _ = core.fit_one_geometry(
            method, asset, seed, init, device, fit_args
        )
        del raw
        actual_lag_counts = tuple(
            int(row[f"lag{lag}_block_count"]) for lag in range(3)
        )
        if actual_lag_counts != LAG_COUNTS:
            raise RuntimeError(
                f"{spec['date']} {method}: expected lag counts {LAG_COUNTS}, "
                f"found {actual_lag_counts}"
            )
        n_target_points = int(row["n_target_points"])
        nll_per_target = float(row["final_native_nll"])
        record = {
            "dataset_id": spec["dataset_id"],
            "year": spec["year"],
            "date": spec["date"],
            "method": method,
            **{name: float(row[f"est_{name}"]) for name in PARAMETERS},
            "precompute_seconds": float(row["precompute_s"]),
            "fit_time_seconds": float(row["fit_s"]),
            "outer_steps": int(row["outer_steps"]),
            "max_abs_gradient": float(row["max_abs_gradient"]),
            "init_advec_lat": float(seed["seed_lat"]),
            "init_advec_lon": float(seed["seed_lon"]),
            "grid_step_lat": float(seed["lat_step"]),
            "grid_step_lon": float(seed["lon_step"]),
            "initializer_seconds": float(seed["initializer_s"]),
            "lag0_block_count": actual_lag_counts[0],
            "lag1_block_count": actual_lag_counts[1],
            "lag2_block_count": actual_lag_counts[2],
            "n_target_points": n_target_points,
            "native_nll_per_target": nll_per_target,
            "native_nll_total": nll_per_target * n_target_points,
            "gls_beta": [float(value) for value in row["gls_beta"]],
        }
        records.append(record)
        persist(records, args.output_root)
        print(
            f"    NLL/target={nll_per_target:.8f}; "
            f"total NLL={record['native_nll_total']:.4f}; "
            f"fit={record['fit_time_seconds']:.2f}s",
            flush=True,
        )
    del asset
    clear_device(device)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real-data-root", type=Path, default=Path("/home/jl2815/tco/data")
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/home/jl2815/tco/exercise_output/fall_26/"
            "vecchia_real59_adapted_fixed_nll_lag643"
        ),
    )
    parser.add_argument(
        "--import-checkpoint",
        type=Path,
        default=None,
        help=(
            "Seed an empty output root from this runner's checkpoint or the "
            "older fit_checkpoint_full_precision.json."
        ),
    )
    parser.add_argument("--lat-range", default="-3,2")
    parser.add_argument("--lon-range", default="121,131")
    parser.add_argument("--smooth", type=float, default=0.5, choices=(0.5,))
    parser.add_argument("--target-chunk-size", type=int, default=256)
    parser.add_argument("--lbfgs-lr", type=float, default=1.0)
    parser.add_argument("--lbfgs-steps", type=int, default=5)
    parser.add_argument("--lbfgs-eval", type=int, default=20)
    parser.add_argument("--lbfgs-history", type=int, default=40)
    parser.add_argument("--grad-tol", type=float, default=1e-5)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--suppress-fit-prints", action="store_true")
    return parser


def validate_scope(records: list[dict[str, Any]], specs: list[dict[str, Any]]) -> None:
    selected = {str(spec["dataset_id"]) for spec in specs}
    unexpected = [
        (record["dataset_id"], record["method"])
        for record in records
        if str(record["dataset_id"]) not in selected
        or str(record["method"]) not in METHODS
    ]
    if unexpected:
        raise ValueError(f"Checkpoint contains records outside this run: {unexpected}")


def write_run_config(args: argparse.Namespace, n_imported: int) -> None:
    cuda_available = bool(torch.cuda.is_available())
    write_json(
        args.output_root / CONFIG_NAME,
        {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "host": socket.gethostname(),
            "methods": METHODS,
            "target_block_shape": [4, 4],
            "lag_block_counts": list(LAG_COUNTS),
            "n_dates": 59,
            "excluded_date": "2025-07-24",
            "diagnostics": [],
            "comparison": "native Vecchia NLL on each method's own graph",
            "n_imported_records": n_imported,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cuda_available": cuda_available,
            "cuda_device_name": torch.cuda.get_device_name(0) if cuda_available else None,
            "gems_tco_source_root": str(core.SRC),
            "gems_tco_package_root": str(core.ACTUAL_PACKAGE_ROOT),
            "arguments": vars(args),
        },
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    print(f"Runner GEMS_TCO: {core.ACTUAL_PACKAGE_ROOT}", flush=True)
    data_loader_module = sys.modules[core.ProcessedDataLoader.__module__]
    print(
        f"Runner data loader: {Path(data_loader_module.__file__).resolve()}",
        flush=True,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    specs = date_specs()
    records = load_records(args)
    validate_scope(records, specs)
    write_run_config(args, len(records))
    if records:
        persist(records, args.output_root)

    for position, spec in enumerate(specs, start=1):
        print(f"\n[{position}/{len(specs)}] {spec['date']}", flush=True)
        fit_day(spec, records, args)

    persist(records, args.output_root)
    expected_rows = len(specs) * len(METHODS)
    if len(records) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} fit rows, found {len(records)}")
    winner_summary = build_winner_summary(results_frame(records))
    if winner_summary["n_paired_dates"] != len(specs):
        raise RuntimeError(
            "Expected one adapted/fixed NLL comparison for every date, found "
            f"{winner_summary['n_paired_dates']}"
        )
    write_json(
        args.output_root / COMPLETE_NAME,
        {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "n_dates": len(specs),
            "n_methods": len(METHODS),
            "n_fit_rows": len(records),
            "target_block_shape": [4, 4],
            "lag_block_counts": list(LAG_COUNTS),
            "device": args.device,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cuda_device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            **winner_summary,
        },
    )
    print(
        f"Complete: {len(specs)} dates, {len(records)} likelihood fits in "
        f"{args.output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
