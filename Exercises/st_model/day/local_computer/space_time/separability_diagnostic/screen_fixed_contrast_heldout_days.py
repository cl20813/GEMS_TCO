#!/usr/bin/env python3
"""Fit and descriptively screen three preselected held-out July 2024 days.

The July-1 contrast geometry, coefficients, lag, and mapping rule are frozen.
Each held-out day receives an independent joint-GC 4/3/2 corridor-Vecchia and
GLS fit.  No bootstrap, contrast search, or coefficient tuning is performed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from apply_fixed_canonical_contrast_real_gems import (
    DEFAULT_DATA_FILE,
    DEFAULT_FIT_CSV,
    DEFAULT_ORACLE_DIR,
    _atomic_csv,
    _atomic_text,
    _canonical_samples,
    _load_filtered_frames,
    _summaries,
)
from run_fixed_contrast_full_pipeline_bootstrap import (
    _device,
    _fit_replicate,
    _fit_series,
    _raw_parameters,
    _residual_cube,
    _source_map,
)


DEFAULT_OUTPUT_DIR = DEFAULT_ORACLE_DIR / "heldout_three_day_screen_202407"
DEFAULT_INITIAL_PHYSICAL = {
    "est_sigmasq": 13.059,
    "est_range_lat": 0.20,
    "est_range_lon": 0.25,
    "est_range_time": 1.50,
    "est_advec_lat": 0.0218,
    "est_advec_lon": -0.1689,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--fit-csv", type=Path, default=DEFAULT_FIT_CSV)
    parser.add_argument("--oracle-dir", type=Path, default=DEFAULT_ORACLE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--strategy", default="standard_432")
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--days", default="", help="Optional 1-based days, e.g. 3,22,31")
    parser.add_argument("--sample-size", type=int, default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--target-chunk-size", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--max-eval", type=int, default=20)
    parser.add_argument("--history-size", type=int, default=10)
    parser.add_argument("--grad-tol", type=float, default=1.0e-4)
    parser.add_argument("--tolerance-grad", type=float, default=1.0e-5)
    parser.add_argument("--latitude-min", type=float, default=-3.0)
    parser.add_argument("--latitude-max", type=float, default=2.0)
    parser.add_argument("--longitude-min", type=float, default=121.0)
    parser.add_argument("--longitude-max", type=float, default=131.0)
    return parser


def _select_days(args: argparse.Namespace, available_days: set[int]) -> list[int]:
    candidates = sorted(day for day in available_days if day != 1 and not 23 <= day <= 30)
    if args.days.strip():
        selected = sorted({int(value.strip()) for value in args.days.split(",") if value.strip()})
    else:
        if args.sample_size > len(candidates):
            raise ValueError("sample size exceeds the eligible day count")
        selected = sorted(
            np.random.default_rng(args.seed).choice(
                np.asarray(candidates), size=args.sample_size, replace=False
            ).tolist()
        )
    invalid = [day for day in selected if day not in candidates]
    if invalid:
        raise ValueError(
            f"days {invalid} are unavailable or excluded; eligible days are {candidates}"
        )
    if len(selected) != args.sample_size:
        raise ValueError("selected day count differs from --sample-size")
    return selected


def _initial_raw() -> np.ndarray:
    row = pd.Series(DEFAULT_INITIAL_PHYSICAL)
    return _raw_parameters(row)


def _saved_fit_series(row: pd.Series) -> pd.Series:
    values = {
        key: float(row[key])
        for key in (
            "est_sigmasq",
            "est_range_lat",
            "est_range_lon",
            "est_range_time",
            "est_advec_lat",
            "est_advec_lon",
            "est_nugget",
            "gc_alpha",
            "gc_beta",
            "gls_lat_mean",
        )
    }
    for index in range(9):
        values[f"beta_{index}"] = float(row[f"beta_{index}"])
    return pd.Series(values)


def _write_figure(screen: pd.DataFrame, output_dir: Path) -> None:
    dates = screen["day"].tolist()
    x = np.arange(len(screen))
    width = 0.25
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 4.7), constrained_layout=True)
    axes[0].bar(x - width, screen["empirical_h_ab"], width, label="empirical", color="0.25")
    axes[0].bar(x, screen["joint_h_ab"], width, label="joint GC", color="tab:blue")
    axes[0].bar(x + width, screen["separable_h_ab"], width, label="matched separable", color="tab:orange")
    axes[0].set_title("Frozen Cov(QA,QB)")
    axes[0].axhline(0.0, color="0.3", linewidth=0.8)
    axes[1].bar(x - width, screen["empirical_mean_l_squared"], width, label="empirical", color="0.25")
    axes[1].bar(x, screen["joint_mean_variance_l"], width, label="joint GC", color="tab:blue")
    axes[1].bar(x + width, screen["separable_mean_variance_l"], width, label="matched separable", color="tab:orange")
    axes[1].set_title("Frozen Var(L)")
    for axis in axes:
        axis.set_xticks(x, dates, rotation=20)
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
    figure.savefig(output_dir / "heldout_three_day_fixed_contrast.png", dpi=220)
    figure.savefig(output_dir / "heldout_three_day_fixed_contrast.pdf")
    plt.close(figure)


def main() -> None:
    args = build_parser().parse_args()
    device = _device(args.device)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fit_table = pd.read_csv(args.fit_csv.expanduser())
    frozen_rows = fit_table.loc[
        (fit_table["strategy"] == args.strategy)
        & (fit_table["day_idx"].astype(int) == 0)
    ]
    if len(frozen_rows) != 1:
        raise ValueError("July-1 frozen fit selection must yield exactly one row")
    frozen_fit = frozen_rows.iloc[0]
    coefficient_path = args.oracle_dir.expanduser() / "global_two_rectangle_search/global_pair_ties.csv"
    coefficient_row = pd.read_csv(coefficient_path, float_precision="round_trip").iloc[0]
    d1 = float(coefficient_row["raw_coefficient_first"])
    d2 = float(coefficient_row["raw_coefficient_second"])

    frames, monthly_mean = _load_filtered_frames(
        args.data_file.expanduser(),
        (args.latitude_min, args.latitude_max),
        (args.longitude_min, args.longitude_max),
    )
    keys = sorted(frames)
    if len(keys) % 8 != 0:
        raise ValueError("July frame count is not divisible into eight-hour days")
    available_days = set(range(1, len(keys) // 8 + 1))
    selected_days = _select_days(args, available_days)
    selection = pd.DataFrame(
        {
            "selection_order": np.arange(1, len(selected_days) + 1),
            "day_of_month": selected_days,
            "day_index": np.asarray(selected_days) - 1,
            "date": [f"2024-07-{day:02d}" for day in selected_days],
            "random_seed": args.seed,
            "candidate_rule": "July days 2-22 and 31; exclude discovery day 1 and days 23-30",
        }
    )
    _atomic_csv(output_dir / "selected_days.csv", selection)
    _atomic_text(
        output_dir / "selection_manifest.json",
        json.dumps(
            {
                "seed": args.seed,
                "eligible_days": sorted(day for day in available_days if day != 1 and not 23 <= day <= 30),
                "selected_days": selected_days,
                "selection_completed_before_fitting": True,
            },
            indent=2,
        )
        + "\n",
    )

    fit_path = output_dir / "daily_joint_gc_fits.csv"
    saved_fits = pd.read_csv(fit_path) if fit_path.is_file() else pd.DataFrame()
    fit_records = saved_fits.to_dict("records") if not saved_fits.empty else []
    initial_raw = _initial_raw()
    screen_rows: list[dict[str, Any]] = []

    for day in selected_days:
        day_index = day - 1
        day_frames = [frames[key] for key in keys[day_index * 8 : (day_index + 1) * 8]]
        base_map, grid_coordinates, latitudes, longitudes, latitude_step, longitude_step = (
            _source_map(day_frames, monthly_mean, device)
        )
        cached = saved_fits.loc[saved_fits["day_of_month"] == day] if not saved_fits.empty else pd.DataFrame()
        if len(cached) == 1:
            fit_record = cached.iloc[0]
            daily_fit = _saved_fit_series(fit_record)
            beta = torch.as_tensor(
                [daily_fit[f"beta_{index}"] for index in range(9)],
                dtype=torch.float64,
                device=device,
            )
        else:
            fit_result, beta, fitted_model, fit_seconds = _fit_replicate(
                base_map,
                grid_coordinates,
                frozen_fit,
                initial_raw,
                args,
            )
            daily_fit = _fit_series(
                fit_result, frozen_fit, beta, fitted_model.lat_mean_val
            )
            interpreted = fit_result.interpretable_parameters
            fit_record = {
                "day_of_month": day,
                "day_index": day_index,
                "day": f"2024-07-{day:02d}",
                "n_valid_o3": int(sum(torch.isfinite(rows[:, 2]).sum().item() for rows in base_map.values())),
                "est_sigmasq": float(interpreted["signal_variance"]),
                "est_range_lat": float(interpreted["range_lat"]),
                "est_range_lon": float(interpreted["range_lon"]),
                "est_range_time": float(interpreted["range_time"]),
                "est_advec_lat": float(interpreted["advec_lat"]),
                "est_advec_lon": float(interpreted["advec_lon"]),
                "est_nugget": float(interpreted["nugget"]),
                "gc_alpha": float(frozen_fit["gc_alpha"]),
                "gc_beta": float(frozen_fit["gc_beta"]),
                "gls_lat_mean": float(fitted_model.lat_mean_val),
                "fit_nll": float(fit_result.final_nll),
                "fit_steps": int(fit_result.steps_completed),
                "fit_converged": bool(fit_result.converged),
                "fit_max_abs_gradient": float(fit_result.max_abs_gradient),
                "fit_objective_evaluations": int(fit_result.objective_evaluations),
                "fit_seconds": fit_seconds,
            }
            for index, value in enumerate(beta.detach().cpu().numpy().reshape(-1)):
                fit_record[f"beta_{index}"] = float(value)
            fit_records.append(fit_record)
            _atomic_csv(
                fit_path,
                pd.DataFrame(fit_records).sort_values("day_of_month").reset_index(drop=True),
            )
            del fitted_model

        cube = _residual_cube(
            base_map,
            beta,
            float(daily_fit["gls_lat_mean"]),
            grid_coordinates,
            latitudes,
            longitudes,
            latitude_step,
            longitude_step,
        )
        samples = _canonical_samples(
            cube,
            frozen_fit,
            d1,
            d2,
            covariance_fit=daily_fit,
        )
        _atomic_csv(
            output_dir / f"translated_contrast_samples_2024-07-{day:02d}.csv",
            samples,
        )
        total = _summaries(samples).loc[lambda frame: frame["group"] == "all"].iloc[0]
        cross_joint_error = abs(float(total["empirical_h_ab"] - total["joint_h_ab"]))
        cross_separable_error = abs(
            float(total["empirical_h_ab"] - total["separable_h_ab"])
        )
        var_joint_error = abs(
            np.log(float(total["empirical_mean_l_squared"] / total["joint_mean_variance_l"]))
        )
        var_separable_error = abs(
            np.log(float(total["empirical_mean_l_squared"] / total["separable_mean_variance_l"]))
        )
        row = {
            "day_of_month": day,
            "day_index": day_index,
            "day": f"2024-07-{day:02d}",
            **total.to_dict(),
            "cross_abs_error_joint": cross_joint_error,
            "cross_abs_error_separable": cross_separable_error,
            "cross_closer_model": "joint" if cross_joint_error < cross_separable_error else "separable",
            "var_l_abs_log_error_joint": var_joint_error,
            "var_l_abs_log_error_separable": var_separable_error,
            "var_l_closer_model": "joint" if var_joint_error < var_separable_error else "separable",
        }
        screen_rows.append(row)
        _atomic_csv(
            output_dir / "heldout_fixed_contrast_screen.csv",
            pd.DataFrame(screen_rows).sort_values("day_of_month").reset_index(drop=True),
        )
        print(
            f"day={day:02d} n={int(total['sample_count'])} "
            f"Hab(emp/joint/sep)={total['empirical_h_ab']:.4f}/"
            f"{total['joint_h_ab']:.4f}/{total['separable_h_ab']:.4f} "
            f"VarL(emp/joint/sep)={total['empirical_mean_l_squared']:.4f}/"
            f"{total['joint_mean_variance_l']:.4f}/{total['separable_mean_variance_l']:.4f}",
            flush=True,
        )

    screen = pd.DataFrame(screen_rows).sort_values("day_of_month").reset_index(drop=True)
    _write_figure(screen, output_dir)
    joint_cross_count = int((screen["cross_closer_model"] == "joint").sum())
    joint_var_count = int((screen["var_l_closer_model"] == "joint").sum())
    lines = [
        "# Frozen diagnostic on three randomly selected held-out days",
        "",
        f"Selection seed: `{args.seed}`. Eligible dates were July 2-22 and July 31; July 1 was the discovery day and July 23-30 were excluded before selection. Selected dates: {', '.join(screen['day'])}.",
        "",
        "The July-1 A/B geometry, d1/d2 coefficients, lag one, range-unit transfer, and nearest-grid rule were frozen. Each selected day was independently fitted with the joint generalized-Cauchy 4/3/2 corridor-Vecchia model and GLS mean. This is a descriptive screen, not a calibrated test.",
        "",
        "| date | n | empirical Hab | joint Hab | separable Hab | cross closer | empirical Var(L) | joint Var(L) | separable Var(L) | Var(L) closer |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in screen.itertuples(index=False):
        lines.append(
            f"| {row.day} | {row.sample_count} | {row.empirical_h_ab:.6g} | "
            f"{row.joint_h_ab:.6g} | {row.separable_h_ab:.6g} | `{row.cross_closer_model}` | "
            f"{row.empirical_mean_l_squared:.6g} | {row.joint_mean_variance_l:.6g} | "
            f"{row.separable_mean_variance_l:.6g} | `{row.var_l_closer_model}` |"
        )
    lines.extend(
        [
            "",
            f"Joint GC was closer for the cross covariance on `{joint_cross_count}/3` days and for Var(L) on `{joint_var_count}/3` days.",
            "",
            "No held-out result was used to alter the contrast. Overlapping translated contrasts mean these rows are descriptive model checks; bootstrap calibration remains separate.",
            "",
        ]
    )
    _atomic_text(output_dir / "REPORT.md", "\n".join(lines))
    manifest = {
        "seed": args.seed,
        "selected_days": selected_days,
        "frozen_discovery_day": "2024-07-01",
        "frozen_fit_csv": str(args.fit_csv.expanduser().resolve()),
        "coefficient_source": str(coefficient_path.resolve()),
        "initial_physical_parameters_shared_across_days": DEFAULT_INITIAL_PHYSICAL,
        "optimizer": {
            "max_steps": args.max_steps,
            "max_eval": args.max_eval,
            "grad_tol": args.grad_tol,
            "tolerance_grad": args.tolerance_grad,
        },
        "interpretation": "independent-day descriptive screen; not bootstrap-calibrated",
    }
    _atomic_text(output_dir / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(screen.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
