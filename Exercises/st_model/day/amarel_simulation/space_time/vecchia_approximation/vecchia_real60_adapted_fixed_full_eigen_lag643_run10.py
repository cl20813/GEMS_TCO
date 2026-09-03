#!/usr/bin/env python3
"""Sequential 60-day real-data adapted-vs-fixed lag-643 study.

The run fits July 1--30 of 2024 and 2025 (60 dates total), then validates each
fit with a common 3,200-point full-covariance eigen-whitening diagnostic.  Each
hour is fully max-min ordered and the first 400 valid points are retained.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import socket
import sys
import time
import traceback
from datetime import datetime
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
import scipy.linalg
import torch


HERE = Path(__file__).resolve().parent
SUBMIT_DIR = Path(os.environ.get("SLURM_SUBMIT_DIR", str(HERE))).resolve()
AMAREL_ROOT = Path("/home/jl2815/tco")
LOCAL_SRC = Path("/Users/joonwonlee/Documents/GEMS_TCO-1/src")
for candidate in (AMAREL_ROOT, SUBMIT_DIR, HERE, LOCAL_SRC):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from GEMS_TCO import orderings  # noqa: E402
import vecchia_adapted_vs_fixed_lag643_090126 as core  # noqa: E402


METHODS = ("adapted", "fixed")
COLORS = {"adapted": "#1f77b4", "fixed": "#d62728"}
LINESTYLES = {"adapted": "-", "fixed": "-."}
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
BROWN_BRIDGE_Q95 = 1.3581015157406195
EPS = 1e-12

# The requested validation is the separate 3,200-point full-eigen diagnostic.
# Disable the older conditional-eigen calculation inside fit_one_geometry.
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
    path.write_text(
        json.dumps(json_ready(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def date_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for year in (2024, 2025):
        for day in range(1, 31):
            date = f"{year}-07-{day:02d}"
            specs.append(
                {
                    "dataset_id": f"real_{year}07{day:02d}",
                    "data_kind": "real",
                    "data_source": f"real_july_{year}",
                    "year": year,
                    "month": 7,
                    "day": day,
                    "date": date,
                }
            )
    if len(specs) != 60:
        raise RuntimeError(f"Internal date selection error: {len(specs)} dates")
    return specs


def gpu_snapshot() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "gpu_name": "",
            "gpu_peak_allocated_gib": np.nan,
            "gpu_peak_reserved_gib": np.nan,
        }
    return {
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }


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
        device="cuda",
        require_cuda=True,
    )


def load_or_fit_day(
    spec: dict[str, Any], args: argparse.Namespace
) -> tuple[core.DayAsset, pd.DataFrame]:
    task_dir = args.output_root / f"task_{spec['dataset_id']}"
    task_dir.mkdir(parents=True, exist_ok=True)
    fits_path = task_dir / "fits.csv"
    raw_path = task_dir / "fitted_raw_parameters.json"
    fit_rows = pd.read_csv(fits_path).to_dict("records") if fits_path.is_file() else []
    fitted_raw = json.loads(raw_path.read_text()) if raw_path.is_file() else {}

    fit_args = build_fit_args(args)
    device = core.resolve_device(fit_args)
    asset = core.load_real_asset(spec, fit_args)
    seed = core.m3_q3_seed(asset, fit_args)
    pd.DataFrame([{**spec, **seed}]).to_csv(task_dir / "initializer.csv", index=False)
    init = dict(core.DEFAULT_REAL_INIT)

    completed = {
        str(row["geometry"])
        for row in fit_rows
        if str(row.get("status", "")) == "ok"
    }
    for method in METHODS:
        if method in completed and method in fitted_raw:
            print(f"  Fit already complete: {method}", flush=True)
            continue
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f"  Fitting {method}", flush=True)
        row, raw, _, _ = core.fit_one_geometry(
            method, asset, seed, init, device, fit_args
        )
        row.update(gpu_snapshot())
        fit_rows = [old for old in fit_rows if str(old.get("geometry")) != method]
        fit_rows.append(row)
        fitted_raw[method] = raw
        pd.DataFrame(fit_rows).to_csv(fits_path, index=False)
        write_json(raw_path, fitted_raw)
        print(
            f"    NLL/target={row['final_native_nll']:.10f}; "
            f"fit={row['fit_s']:.2f}s; peak={row['gpu_peak_allocated_gib']:.2f} GiB",
            flush=True,
        )
        gc.collect()
        torch.cuda.empty_cache()

    fits = pd.DataFrame(fit_rows)
    successful = fits[fits["status"].astype(str) == "ok"]
    if set(successful["geometry"].astype(str)) != set(METHODS):
        raise RuntimeError(f"Incomplete fits for {spec['dataset_id']}")
    write_json(
        task_dir / "fit_config.json",
        {
            "date": spec["date"],
            "methods": METHODS,
            "block_shape": "4x4",
            "lag_pattern": "6/4/3",
            "target_chunk_size": args.target_chunk_size,
            "nugget_mode": "estimated",
            "optimizer": {
                "lr": args.lbfgs_lr,
                "line_search_fn": "strong_wolfe",
                "max_iter": args.lbfgs_eval,
                "max_eval": args.lbfgs_eval,
                "history_size": args.lbfgs_history,
                "tolerance_grad": 1e-5,
                "tolerance_change": 1e-9,
                "outer_max_steps": args.lbfgs_steps,
                "outer_grad_tol": args.grad_tol,
            },
        },
    )
    (task_dir / "FIT_COMPLETE").write_text("complete\n", encoding="utf-8")
    return asset, successful.copy()


def point_maxmin_sample(
    asset: core.DayAsset, points_per_hour: int
) -> tuple[np.ndarray, pd.DataFrame]:
    if len(asset.keys) != 8:
        raise ValueError(f"Expected 8 hours for {asset.dataset_id}, got {len(asset.keys)}")
    selected_chunks: list[np.ndarray] = []
    manifests: list[dict[str, Any]] = []
    for hour_index, key in enumerate(asset.keys):
        rows = asset.source_map[key].detach().cpu().numpy().astype(np.float64, copy=False)
        valid = np.isfinite(rows[:, 0]) & np.isfinite(rows[:, 1]) & np.isfinite(rows[:, 2])
        valid_rows = np.ascontiguousarray(rows[valid])
        if len(valid_rows) < points_per_hour:
            raise ValueError(
                f"{asset.dataset_id} {key}: {len(valid_rows)} valid points, "
                f"need {points_per_hour}"
            )
        started = time.perf_counter()
        full_order = np.asarray(
            orderings.maxmin_cpp(np.ascontiguousarray(valid_rows[:, :2])),
            dtype=np.int64,
        )
        elapsed = time.perf_counter() - started
        if full_order.size != len(valid_rows) or np.unique(full_order).size != len(valid_rows):
            raise RuntimeError(f"Invalid max-min permutation for {asset.dataset_id} {key}")
        selected_chunks.append(np.ascontiguousarray(valid_rows[full_order[:points_per_hour]]))
        manifests.append(
            {
                "dataset_id": asset.dataset_id,
                "date": asset.date,
                "hour_index": hour_index,
                "hour_key": key,
                "n_valid": len(valid_rows),
                "n_selected": points_per_hour,
                "maxmin_seconds": elapsed,
            }
        )
    selected = np.ascontiguousarray(np.concatenate(selected_chunks, axis=0))
    if len(selected) != 8 * points_per_hour:
        raise RuntimeError(f"Selected {len(selected)}, expected {8 * points_per_hour}")
    return selected, pd.DataFrame(manifests)


def mean_design(selected: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    if selected.shape[1] < 11:
        raise ValueError(f"Expected at least 11 columns, got {selected.shape}")
    lat = selected[:, 0]
    design = np.column_stack(
        [np.ones(len(selected)), lat - float(np.mean(lat)), selected[:, 4:11]]
    ).astype(np.float64, copy=False)
    rank = int(np.linalg.matrix_rank(design))
    if rank != design.shape[1]:
        raise RuntimeError(f"Mean design is rank deficient: {rank}/{design.shape[1]}")
    return (
        np.ascontiguousarray(design),
        np.ascontiguousarray(selected[:, 2], dtype=np.float64),
        rank,
    )


def fitted_parameters(row: pd.Series) -> dict[str, float]:
    values = {name: float(row[f"est_{name}"]) for name in PARAMETERS}
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError(f"Non-finite fitted parameters: {values}")
    return values


def fitted_covariance(
    selected: np.ndarray, est: dict[str, float], smooth: float, jitter: float
) -> np.ndarray:
    lat = selected[:, 0].astype(np.float64, copy=False)
    lon = selected[:, 1].astype(np.float64, copy=False)
    time_coordinate = selected[:, 3].astype(np.float64, copy=False)
    delta_t = time_coordinate[:, None] - time_coordinate[None, :]

    scaled_sq = lat[:, None] - lat[None, :]
    scaled_sq -= est["advec_lat"] * delta_t
    np.square(scaled_sq, out=scaled_sq)
    scaled_sq /= est["range_lat"] ** 2

    work = lon[:, None] - lon[None, :]
    work -= est["advec_lon"] * delta_t
    np.square(work, out=work)
    work /= est["range_lon"] ** 2
    scaled_sq += work
    del work

    np.square(delta_t, out=delta_t)
    delta_t /= est["range_time"] ** 2
    scaled_sq += delta_t
    del delta_t

    np.maximum(scaled_sq, 0.0, out=scaled_sq)
    np.sqrt(scaled_sq, out=scaled_sq)
    if not math.isclose(smooth, 0.5, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"This run supports smooth=0.5, got {smooth}")
    np.negative(scaled_sq, out=scaled_sq)
    np.exp(scaled_sq, out=scaled_sq)
    scaled_sq *= est["sigmasq"]
    scaled_sq.flat[:: len(selected) + 1] += est["nugget"] + jitter
    return np.ascontiguousarray(scaled_sq)


def full_eigen_diagnostic(
    selected: np.ndarray,
    design: np.ndarray,
    values: np.ndarray,
    est: dict[str, float],
    fit_row: pd.Series,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    started = time.perf_counter()
    covariance = fitted_covariance(selected, est, args.smooth, args.cov_jitter)
    covariance_seconds = time.perf_counter() - started

    started = time.perf_counter()
    covariance = np.asarray(0.5 * (covariance + covariance.T), order="F")
    eigenvalues, eigenvectors = scipy.linalg.eigh(
        covariance,
        lower=True,
        overwrite_a=True,
        check_finite=False,
        driver="evd",
    )
    eigen_seconds = time.perf_counter() - started
    del covariance
    if not np.all(np.isfinite(eigenvalues)) or float(eigenvalues.min()) <= 0.0:
        raise RuntimeError(f"Covariance is not positive definite: min={eigenvalues.min()}")
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.ascontiguousarray(eigenvalues[order])
    eigenvectors = np.ascontiguousarray(eigenvectors[:, order])
    root = np.sqrt(eigenvalues)
    whitened_y = (eigenvectors.T @ values) / root
    whitened_x = (eigenvectors.T @ design) / root[:, None]
    xtx = whitened_x.T @ whitened_x
    xty = whitened_x.T @ whitened_y
    try:
        xtx_inv = scipy.linalg.inv(xtx, check_finite=False)
        beta = scipy.linalg.solve(xtx, xty, assume_a="sym", check_finite=False)
    except (ValueError, np.linalg.LinAlgError):
        xtx_inv = np.linalg.pinv(xtx)
        beta = xtx_inv @ xty
    scores = whitened_y - whitened_x @ beta
    squared = np.square(scores)
    leverage = np.einsum("ij,ij->i", whitened_x @ xtx_inv, whitened_x)
    expected_increment = np.clip(1.0 - leverage, 0.0, None)
    cumulative = np.cumsum(squared)
    cumulative_expected = np.cumsum(expected_increment)
    residual_df = float(expected_increment.sum())
    scaled_expected = cumulative_expected / residual_df
    scaled_cumulative = cumulative / residual_df
    d_stat = float(
        np.max(np.abs(cumulative - cumulative_expected))
        / math.sqrt(2.0 * residual_df)
    )
    band_width = args.brown_bridge_q * math.sqrt(2.0 / residual_df)
    curve = pd.DataFrame(
        {
            "index": np.arange(1, len(eigenvalues) + 1),
            "scaled_expected": scaled_expected,
            "scaled_cumulative": scaled_cumulative,
            "eigenvalue": eigenvalues,
            "squared_score": squared,
            "leverage": leverage,
            "band_lower": scaled_expected - band_width,
            "band_upper": scaled_expected + band_width,
        }
    )
    summary = {
        "n_selected": len(selected),
        "residual_df": residual_df,
        "mean_y2": float(cumulative[-1] / residual_df),
        "D": d_stat,
        "min_eigenvalue": float(eigenvalues[-1]),
        "max_eigenvalue": float(eigenvalues[0]),
        "native_nll": float(fit_row["final_native_nll"]),
        "fit_seconds": float(fit_row["fit_s"]),
        "covariance_seconds": covariance_seconds,
        "eigen_seconds": eigen_seconds,
    }
    del eigenvalues, eigenvectors, whitened_y, whitened_x, scores, squared
    gc.collect()
    return curve, summary


def plot_daily_curves(
    curves: dict[str, pd.DataFrame],
    summaries: dict[str, dict[str, Any]],
    date: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 7.0))
    ymax = 1.05
    for method in METHODS:
        curve = curves[method]
        summary = summaries[method]
        ymax = max(ymax, float(curve["scaled_cumulative"].max()) * 1.03)
        ax.plot(
            curve["scaled_expected"],
            curve["scaled_cumulative"],
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            linewidth=2.0,
            label=(
                f"{LABELS[method]}: NLL={summary['native_nll']:.4f}, "
                f"mean Y²={summary['mean_y2']:.4f}, D={summary['D']:.4f}"
            ),
        )
    reference = curves[METHODS[0]]
    ax.plot([0, 1], [0, 1], color="0.25", linewidth=1.4, label="ideal y=x")
    ax.plot(
        reference["scaled_expected"], reference["band_lower"],
        color="0.65", linestyle="--", linewidth=0.9,
    )
    ax.plot(
        reference["scaled_expected"], reference["band_upper"],
        color="0.65", linestyle="--", linewidth=0.9, label="approx. 95% band",
    )
    ax.set(xlim=(0, 1), ylim=(0, ymax))
    ax.set_xlabel("cumulative expected residual df fraction (largest eigenvalue first)")
    ax.set_ylabel("cumulative whitened squared score / residual df")
    ax.set_title(f"Real {date}: 400 x 8 point-maxmin full-eigen diagnostic")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def diagnose_day(
    spec: dict[str, Any], asset: core.DayAsset, fits: pd.DataFrame, args: argparse.Namespace
) -> None:
    task_dir = args.output_root / f"task_{spec['dataset_id']}"
    if (task_dir / "DIAGNOSTIC_COMPLETE").is_file():
        print("  Full-eigen diagnostic already complete", flush=True)
        return
    selected, hourly = point_maxmin_sample(asset, args.points_per_hour)
    hourly.to_csv(task_dir / "hourly_maxmin_manifest.csv", index=False)
    design, values, mean_rank = mean_design(selected)
    curves: dict[str, pd.DataFrame] = {}
    summaries: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        matches = fits[(fits["geometry"] == method) & (fits["status"] == "ok")]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one {method} fit for {spec['dataset_id']}")
        fit_row = matches.iloc[0]
        print(f"  Full eigen: {method}", flush=True)
        curve, summary = full_eigen_diagnostic(
            selected, design, values, fitted_parameters(fit_row), fit_row, args
        )
        curve.insert(0, "method", method)
        curve.insert(0, "date", spec["date"])
        curve.insert(0, "year", spec["year"])
        curve.insert(0, "dataset_id", spec["dataset_id"])
        curve.to_csv(task_dir / f"full_eigen_curve_{method}.csv", index=False)
        summary.update(
            {
                "dataset_id": spec["dataset_id"],
                "date": spec["date"],
                "year": spec["year"],
                "method": method,
                "mean_rank": mean_rank,
                "points_per_hour": args.points_per_hour,
                "n_hours": 8,
                "maxmin_seconds": float(hourly["maxmin_seconds"].sum()),
            }
        )
        curves[method] = curve
        summaries[method] = summary
        rows.append(summary)
        print(
            f"    meanY2={summary['mean_y2']:.4f}; D={summary['D']:.4f}; "
            f"eigen={summary['eigen_seconds']:.2f}s",
            flush=True,
        )
    pd.DataFrame(rows).to_csv(task_dir / "daily_metrics.csv", index=False)
    plot_daily_curves(
        curves, summaries, spec["date"], task_dir / "full_eigen_adapted_fixed.png"
    )
    (task_dir / "DIAGNOSTIC_COMPLETE").write_text("complete\n", encoding="utf-8")


def resample_curve(curve: pd.DataFrame, n_grid: int) -> pd.DataFrame:
    grid = np.linspace(1.0 / n_grid, 1.0, n_grid)
    ordered = curve.sort_values("scaled_expected")
    return pd.DataFrame(
        {
            "fraction": grid,
            "scaled_cumulative": np.interp(
                grid,
                ordered["scaled_expected"].to_numpy(float),
                ordered["scaled_cumulative"].to_numpy(float),
            ),
        }
    )


def compact_parameter_table(fits: pd.DataFrame) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "date": fits["date"],
            "method": fits["geometry"],
            "smooth": fits["smooth"],
            "block_shape": fits["block_shape"],
            "lag_pattern": fits["lag_pattern"],
            "sigmasq": fits["est_sigmasq"],
            "range_lat": fits["est_range_lat"],
            "range_lon": fits["est_range_lon"],
            "range_time": fits["est_range_time"],
            "advec_lat": fits["est_advec_lat"],
            "advec_lon": fits["est_advec_lon"],
            "nugget": fits["est_nugget"],
            "native_nll": fits["final_native_nll"],
            "fit_seconds": fits["fit_s"],
            "rmsre": np.zeros(len(fits), dtype=np.float64),
            "init_advec_lat": fits["init_advec_lat"],
            "init_advec_lon": fits["init_advec_lon"],
        }
    )
    return table.sort_values(["date", "method"]).reset_index(drop=True)


def plot_daily_metrics(metrics: pd.DataFrame, output_root: Path) -> None:
    definitions = (
        ("native_nll", "Native Vecchia NLL / target"),
        ("mean_y2", "Mean Y² (ideal 1)"),
        ("D", "Eigen cumulative-curve D (smaller better)"),
    )
    fig, axes = plt.subplots(3, 2, figsize=(15, 11), sharex="col")
    for column, year in enumerate((2024, 2025)):
        subset = metrics[metrics["year"] == year].copy()
        for row_index, (metric, ylabel) in enumerate(definitions):
            ax = axes[row_index, column]
            for method in METHODS:
                part = subset[subset["method"] == method].sort_values("date")
                ax.plot(
                    part["date"], part[metric], marker="o", markersize=3,
                    linewidth=1.5, color=COLORS[method], label=LABELS[method],
                )
            if metric == "mean_y2":
                ax.axhline(1.0, color="0.35", linewidth=1.0, linestyle="--")
            ax.set_title(f"{year} July: {ylabel}")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.22)
            if row_index == 0:
                ax.legend(fontsize=8)
            if row_index == 2:
                ax.tick_params(axis="x", rotation=75, labelsize=7)
    fig.tight_layout()
    fig.savefig(output_root / "daily_metrics_2024_2025.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_yearly_mean_curves(curves: pd.DataFrame, metrics: pd.DataFrame, output_root: Path) -> None:
    mean_rows: list[pd.DataFrame] = []
    for (year, method), group in curves.groupby(["year", "method"]):
        sampled: list[pd.DataFrame] = []
        for _, day_curve in group.groupby("dataset_id"):
            sampled.append(resample_curve(day_curve, 1000))
        stacked = pd.concat(sampled, ignore_index=True)
        mean_curve = stacked.groupby("fraction", as_index=False).agg(
            mean_scaled_cumulative=("scaled_cumulative", "mean"),
            sd_scaled_cumulative=("scaled_cumulative", "std"),
        )
        mean_curve["year"] = int(year)
        mean_curve["method"] = method
        mean_rows.append(mean_curve)
    means = pd.concat(mean_rows, ignore_index=True)
    means.to_csv(output_root / "yearly_july_mean_full_eigen_curves.csv", index=False)
    for year in (2024, 2025):
        fig, ax = plt.subplots(figsize=(9.2, 7.0))
        ymax = 1.05
        for method in METHODS:
            curve = means[(means["year"] == year) & (means["method"] == method)]
            summary = metrics[(metrics["year"] == year) & (metrics["method"] == method)]
            y = curve["mean_scaled_cumulative"].to_numpy(float)
            ymax = max(ymax, float(np.max(y)) * 1.03)
            ax.plot(
                curve["fraction"], y, color=COLORS[method],
                linestyle=LINESTYLES[method], linewidth=2.2,
                label=(
                    f"{LABELS[method]}: mean NLL={summary['native_nll'].mean():.4f}, "
                    f"mean Y²={summary['mean_y2'].mean():.4f}, "
                    f"mean D={summary['D'].mean():.4f}"
                ),
            )
        ax.plot([0, 1], [0, 1], color="0.25", linewidth=1.4, label="ideal y=x")
        ax.set(xlim=(0, 1), ylim=(0, ymax))
        ax.set_xlabel("cumulative expected residual df fraction")
        ax.set_ylabel("mean cumulative whitened squared score / residual df")
        ax.set_title(f"Real {year} July: 30-day mean full-eigen diagnostic")
        ax.grid(alpha=0.22)
        ax.legend(fontsize=8.5)
        fig.tight_layout()
        fig.savefig(
            output_root / f"yearly_july_mean_full_eigen_{year}.png",
            dpi=190,
            bbox_inches="tight",
        )
        plt.close(fig)


def aggregate(specs: list[dict[str, Any]], args: argparse.Namespace) -> None:
    fit_frames: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []
    curve_frames: list[pd.DataFrame] = []
    for spec in specs:
        task_dir = args.output_root / f"task_{spec['dataset_id']}"
        if not (task_dir / "FIT_COMPLETE").is_file():
            raise RuntimeError(f"Missing completed fit: {task_dir}")
        if not (task_dir / "DIAGNOSTIC_COMPLETE").is_file():
            raise RuntimeError(f"Missing completed diagnostic: {task_dir}")
        fit_frames.append(pd.read_csv(task_dir / "fits.csv"))
        metric_frames.append(pd.read_csv(task_dir / "daily_metrics.csv"))
        for method in METHODS:
            curve_frames.append(pd.read_csv(task_dir / f"full_eigen_curve_{method}.csv"))
    fits = pd.concat(fit_frames, ignore_index=True)
    metrics = pd.concat(metric_frames, ignore_index=True)
    curves = pd.concat(curve_frames, ignore_index=True)
    fits.to_csv(args.output_root / "all_fits_full.csv", index=False)
    compact_parameter_table(fits).to_csv(
        args.output_root / "fitted_parameters_real60.csv", index=False
    )
    metrics.sort_values(["date", "method"]).to_csv(
        args.output_root / "daily_metrics.csv", index=False
    )
    curves.to_csv(args.output_root / "full_eigen_curves_all_dates.csv", index=False)
    summary = (
        metrics.groupby(["year", "method"], as_index=False)
        .agg(
            n_dates=("dataset_id", "nunique"),
            mean_native_nll=("native_nll", "mean"),
            sd_native_nll=("native_nll", "std"),
            mean_y2=("mean_y2", "mean"),
            sd_y2=("mean_y2", "std"),
            mean_D=("D", "mean"),
            sd_D=("D", "std"),
            mean_fit_seconds=("fit_seconds", "mean"),
            mean_eigen_seconds=("eigen_seconds", "mean"),
        )
    )
    summary.to_csv(args.output_root / "yearly_july_method_summary.csv", index=False)
    pd.DataFrame(specs).to_csv(args.output_root / "selected_60_dates.csv", index=False)
    plot_daily_metrics(metrics, args.output_root)
    plot_yearly_mean_curves(curves, metrics, args.output_root)
    print(f"Aggregated all 60 dates: {args.output_root}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real-data-root", type=Path, default=Path("/home/jl2815/tco/data")
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/home/jl2815/tco/exercise_output/summer/"
            "vecchia_real60_adapted_fixed_full_eigen_lag643_run10"
        ),
    )
    parser.add_argument("--lat-range", default="-3,2")
    parser.add_argument("--lon-range", default="121,131")
    parser.add_argument("--smooth", type=float, default=0.5, choices=(0.5,))
    parser.add_argument("--target-chunk-size", type=int, default=256)
    parser.add_argument("--points-per-hour", type=int, default=400)
    parser.add_argument("--lbfgs-lr", type=float, default=1.0)
    parser.add_argument("--lbfgs-steps", type=int, default=5)
    parser.add_argument("--lbfgs-eval", type=int, default=20)
    parser.add_argument("--lbfgs-history", type=int, default=40)
    parser.add_argument("--grad-tol", type=float, default=1e-5)
    parser.add_argument("--cov-jitter", type=float, default=1e-8)
    parser.add_argument("--brown-bridge-q", type=float, default=BROWN_BRIDGE_Q95)
    parser.add_argument("--suppress-fit-prints", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.points_per_hour != 400:
        raise ValueError("This production run requires exactly 400 points per hour")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args.output_root.mkdir(parents=True, exist_ok=True)
    specs = date_specs()
    write_json(
        args.output_root / "run_config.json",
        {
            "created": datetime.now().isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "gpu": torch.cuda.get_device_name(0),
            "dates": [spec["date"] for spec in specs],
            "n_dates": len(specs),
            "methods": METHODS,
            "block_shape": "4x4",
            "lag_pattern": "6/4/3",
            "target_chunk_size": args.target_chunk_size,
            "points_per_hour": args.points_per_hour,
            "full_eigen_points": 8 * args.points_per_hour,
            "nugget_mode": "estimated",
            "optimizer": {
                "lr": args.lbfgs_lr,
                "line_search_fn": "strong_wolfe",
                "max_iter": args.lbfgs_eval,
                "max_eval": args.lbfgs_eval,
                "history_size": args.lbfgs_history,
                "tolerance_grad": 1e-5,
                "tolerance_change": 1e-9,
                "outer_max_steps": args.lbfgs_steps,
                "outer_grad_tol": args.grad_tol,
            },
        },
    )
    for position, spec in enumerate(specs, start=1):
        print(f"\n[{position}/60] {spec['date']}", flush=True)
        try:
            asset, fits = load_or_fit_day(spec, args)
            diagnose_day(spec, asset, fits, args)
            del asset, fits
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as exc:
            write_json(
                args.output_root / f"FAILED_{spec['dataset_id']}.json",
                {
                    "date": spec["date"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    **gpu_snapshot(),
                },
            )
            raise
    aggregate(specs, args)


if __name__ == "__main__":
    main()
