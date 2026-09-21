#!/usr/bin/env python3
"""Clean July 2024/2025 adapted-vs-fixed lag-643 study.

The run fits July 1--30 of 2024 and 2025, excluding only 2025-07-24 because
that date has seven rather than eight real-data slots (59 dates total).  Every
fit is validated with a common 3,200-point full-covariance eigen-whitening
diagnostic: each hour is fully max-min ordered and its first 400 valid points
are retained.

The output intentionally has one compact fitted-parameter CSV, one native-NLL
CSV, one checkpoint JSON, one ``daily_plots`` directory, and two monthly mean
plots. No per-date directories or diagnostic curve CSVs are created.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
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
import vecchia_adapted_fixed_lag643_core as core  # noqa: E402


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
RESULT_COLUMNS = (
    "dataset_id",
    "year",
    "date",
    "method",
    *PARAMETERS,
    "fit_time_seconds",
    "rmsre",
    "init_advec_lat",
    "init_advec_lon",
    "n_observations",
    "native_nll_per_observation",
    "native_nll_total",
    "gls_beta",
)
FIT_CSV_COLUMNS = (
    "date",
    "method",
    *PARAMETERS,
    "fit_time_seconds",
    "rmsre",
    "init_advec_lat",
    "init_advec_lon",
)
NLL_CSV_COLUMNS = (
    "date",
    "method",
    "n_observations",
    "native_nll_per_observation",
    "native_nll_total",
    "fixed_minus_adapted_total_nll",
)
RESULT_JSON_NAME = "fit_checkpoint_full_precision.json"
RESULT_CSV_NAME = "daily_fit_results.csv"
NLL_CSV_NAME = "daily_native_nll.csv"
DAILY_PLOT_DIR_NAME = "daily_plots"

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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_ready(value), indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_fit_results(output_root: Path) -> list[dict[str, Any]]:
    path = output_root / RESULT_JSON_NAME
    if not path.is_file():
        return []
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"{path} must contain a JSON list")
    required = set(RESULT_COLUMNS)
    seen: set[tuple[str, str]] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{path}: record {index} is not an object")
        missing = required.difference(record)
        if missing:
            raise ValueError(
                f"{path}: record {index} is missing {sorted(missing)}. "
                "Use a new --output-root for this compact-output run."
            )
        key = (str(record["dataset_id"]), str(record["method"]))
        if key in seen:
            raise ValueError(f"{path}: duplicate fit record {key}")
        seen.add(key)
    return records


def persist_fit_results(records: list[dict[str, Any]], output_root: Path) -> None:
    method_order = {method: index for index, method in enumerate(METHODS)}
    ordered = sorted(
        records,
        key=lambda row: (str(row["date"]), method_order[str(row["method"])]),
    )
    compact = [
        {column: json_ready(record[column]) for column in RESULT_COLUMNS}
        for record in ordered
    ]
    write_json(output_root / RESULT_JSON_NAME, compact)

    frame = pd.DataFrame(compact, columns=RESULT_COLUMNS)
    csv_path = output_root / RESULT_CSV_NAME
    temporary = csv_path.with_suffix(csv_path.suffix + ".tmp")
    fitted = frame.loc[:, FIT_CSV_COLUMNS].copy()
    fitted_numeric = [
        *PARAMETERS,
        "fit_time_seconds",
        "rmsre",
        "init_advec_lat",
        "init_advec_lon",
    ]
    fitted.loc[:, fitted_numeric] = fitted.loc[:, fitted_numeric].round(4)
    fitted.to_csv(temporary, index=False, float_format="%.4f")
    temporary.replace(csv_path)

    totals = frame.pivot(index="date", columns="method", values="native_nll_total")
    daily_difference = (
        totals["fixed"] - totals["adapted"]
        if set(METHODS).issubset(totals.columns)
        else pd.Series(dtype=float)
    )
    nll = frame.loc[
        :,
        [
            "date",
            "method",
            "n_observations",
            "native_nll_per_observation",
            "native_nll_total",
        ],
    ].copy()
    nll["fixed_minus_adapted_total_nll"] = nll["date"].map(daily_difference)
    nll["native_nll_per_observation"] = nll["native_nll_per_observation"].round(6)
    nll["native_nll_total"] = nll["native_nll_total"].round(4)
    nll["fixed_minus_adapted_total_nll"] = nll[
        "fixed_minus_adapted_total_nll"
    ].round(4)
    nll["native_nll_per_observation"] = nll["native_nll_per_observation"].map(
        lambda value: f"{float(value):.6f}"
    )
    nll["native_nll_total"] = nll["native_nll_total"].map(
        lambda value: f"{float(value):.4f}"
    )
    nll["fixed_minus_adapted_total_nll"] = nll[
        "fixed_minus_adapted_total_nll"
    ].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
    nll_path = output_root / NLL_CSV_NAME
    nll_temporary = nll_path.with_suffix(nll_path.suffix + ".tmp")
    nll.loc[:, NLL_CSV_COLUMNS].to_csv(nll_temporary, index=False)
    nll_temporary.replace(nll_path)


def date_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for year in (2024, 2025):
        for day in range(1, 31):
            date = f"{year}-07-{day:02d}"
            if date == "2025-07-24":
                continue
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
    if len(specs) != 59:
        raise RuntimeError(f"Internal date selection error: {len(specs)} dates")
    selected_dates = {spec["date"] for spec in specs}
    if "2024-07-24" not in selected_dates or "2025-07-24" in selected_dates:
        raise RuntimeError("The July 24 exclusion was applied to the wrong year")
    return specs


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
    spec: dict[str, Any],
    records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[core.DayAsset, pd.DataFrame]:
    fit_args = build_fit_args(args)
    device = core.resolve_device(fit_args)
    asset = core.load_real_asset(spec, fit_args)
    seed = core.m3_q3_seed(asset, fit_args)
    init = dict(core.DEFAULT_REAL_INIT)

    completed = {
        str(record["method"])
        for record in records
        if str(record["dataset_id"]) == spec["dataset_id"]
    }
    for method in METHODS:
        if method in completed:
            print(f"  Fit already complete: {method}", flush=True)
            continue
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f"  Fitting {method}", flush=True)
        row, raw, _, _ = core.fit_one_geometry(
            method, asset, seed, init, device, fit_args
        )
        del raw
        record = {
            "dataset_id": spec["dataset_id"],
            "year": spec["year"],
            "date": spec["date"],
            "method": method,
            **{name: float(row[f"est_{name}"]) for name in PARAMETERS},
            "fit_time_seconds": float(row["fit_s"]),
            "rmsre": 0.0,
            "init_advec_lat": float(seed["seed_lat"]),
            "init_advec_lon": float(seed["seed_lon"]),
            "n_observations": int(row["n_target_points"]),
            "native_nll_per_observation": float(row["final_native_nll"]),
            "native_nll_total": (
                float(row["final_native_nll"]) * int(row["n_target_points"])
            ),
            "gls_beta": [float(value) for value in row["gls_beta"]],
        }
        records.append(record)
        persist_fit_results(records, args.output_root)
        print(
            f"    fit={record['fit_time_seconds']:.4f}s; "
            f"NLL/obs={record['native_nll_per_observation']:.6f}; "
            f"total NLL={record['native_nll_total']:.4f}",
            flush=True,
        )
        gc.collect()
        torch.cuda.empty_cache()

    fits = pd.DataFrame(
        [
            record
            for record in records
            if str(record["dataset_id"]) == spec["dataset_id"]
        ],
        columns=RESULT_COLUMNS,
    )
    if set(fits["method"].astype(str)) != set(METHODS) or len(fits) != len(METHODS):
        raise RuntimeError(f"Incomplete fits for {spec['dataset_id']}")
    return asset, fits.copy()


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
    values = {name: float(row[name]) for name in PARAMETERS}
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
    gls_beta: np.ndarray,
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
    beta = np.asarray(gls_beta, dtype=np.float64).reshape(-1)
    if beta.size != design.shape[1] or not np.all(np.isfinite(beta)):
        raise ValueError(
            f"Invalid fitted GLS beta shape/value: {beta.shape}; "
            f"expected {design.shape[1]} coefficients"
        )
    residual = values - design @ beta
    # Sigma = Q Lambda Q'. For a correctly specified fitted covariance,
    # z = Lambda^(-1/2) Q' residual is approximately iid N(0, 1).
    scores = (eigenvectors.T @ residual) / root
    squared = np.square(scores)
    cumulative = np.cumsum(squared)
    n_scores = int(len(scores))
    cumulative_expected = np.arange(1, n_scores + 1, dtype=np.float64)
    scaled_expected = cumulative_expected / n_scores
    scaled_cumulative = cumulative / n_scores
    d_stat = float(
        np.max(np.abs(cumulative - cumulative_expected))
        / math.sqrt(2.0 * n_scores)
    )
    curve = pd.DataFrame(
        {
            "index": cumulative_expected.astype(np.int64),
            "scaled_expected": scaled_expected,
            "scaled_cumulative": scaled_cumulative,
            "eigenvalue": eigenvalues,
            "squared_score": squared,
        }
    )
    summary = {
        "n_selected": len(selected),
        "mean_y2": float(cumulative[-1] / n_scores),
        "D": d_stat,
        "min_eigenvalue": float(eigenvalues[-1]),
        "max_eigenvalue": float(eigenvalues[0]),
        "covariance_seconds": covariance_seconds,
        "eigen_seconds": eigen_seconds,
    }
    del eigenvalues, eigenvectors, residual, scores, squared
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
                f"{LABELS[method]}: NLL/obs="
                f"{summary['native_nll_per_observation']:.4f}, "
                f"mean Y²={summary['mean_y2']:.4f}, "
                f"D={summary['D']:.4f}"
            ),
        )
    ax.plot([0, 1], [0, 1], color="0.25", linewidth=1.4, label="ideal y=x")
    ax.set(xlim=(0, 1), ylim=(0, ymax))
    ax.set_xlabel("eigen-component fraction k / n (largest eigenvalue first)")
    ax.set_ylabel("cumulative squared whitened residual / n")
    ax.set_title(f"Real {date}: 400 x 8 point-maxmin full-eigen diagnostic")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def diagnose_day(
    spec: dict[str, Any], asset: core.DayAsset, fits: pd.DataFrame, args: argparse.Namespace
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected, hourly = point_maxmin_sample(asset, args.points_per_hour)
    design, values, mean_rank = mean_design(selected)
    curves: dict[str, pd.DataFrame] = {}
    summaries: dict[str, dict[str, Any]] = {}
    sampled_curves: list[pd.DataFrame] = []
    for method in METHODS:
        matches = fits[fits["method"] == method]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one {method} fit for {spec['dataset_id']}")
        fit_row = matches.iloc[0]
        print(f"  Full eigen: {method}", flush=True)
        curve, summary = full_eigen_diagnostic(
            selected,
            design,
            values,
            fitted_parameters(fit_row),
            np.asarray(fit_row["gls_beta"], dtype=np.float64),
            args,
        )
        sampled = resample_curve(curve, 1000)
        sampled.insert(0, "method", method)
        sampled.insert(0, "date", spec["date"])
        sampled.insert(0, "year", spec["year"])
        sampled.insert(0, "dataset_id", spec["dataset_id"])
        sampled_curves.append(sampled)
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
                "native_nll_per_observation": float(
                    fit_row["native_nll_per_observation"]
                ),
                "native_nll_total": float(fit_row["native_nll_total"]),
            }
        )
        curves[method] = curve
        summaries[method] = summary
        print(
            f"    meanY2={summary['mean_y2']:.4f}; D={summary['D']:.4f}; "
            f"eigen={summary['eigen_seconds']:.4f}s",
            flush=True,
        )
    daily_plot_dir = args.output_root / DAILY_PLOT_DIR_NAME
    daily_plot_dir.mkdir(parents=True, exist_ok=True)
    plot_daily_curves(
        curves,
        summaries,
        spec["date"],
        daily_plot_dir / f"{spec['date']}_daily_maxmin_full_eigen.png",
    )
    del selected, design, values, curves
    gc.collect()
    return pd.concat(sampled_curves, ignore_index=True), pd.DataFrame(
        list(summaries.values())
    )


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


def plot_monthly_mean_curves(
    curves: pd.DataFrame, metrics: pd.DataFrame, output_root: Path
) -> None:
    expected_days = {2024: 30, 2025: 29}
    for year in (2024, 2025):
        year_curves = curves[curves["year"] == year]
        for method in METHODS:
            n_dates = year_curves.loc[
                year_curves["method"] == method, "dataset_id"
            ].nunique()
            if n_dates != expected_days[year]:
                raise RuntimeError(
                    f"Expected {expected_days[year]} {method} curves for {year}, "
                    f"got {n_dates}"
                )
        fig, ax = plt.subplots(figsize=(9.2, 7.0))
        ymax = 1.05
        for method in METHODS:
            method_curves = year_curves[year_curves["method"] == method]
            curve = method_curves.groupby("fraction", as_index=False).agg(
                mean_scaled_cumulative=("scaled_cumulative", "mean")
            )
            summary = metrics[(metrics["year"] == year) & (metrics["method"] == method)]
            y = curve["mean_scaled_cumulative"].to_numpy(float)
            ymax = max(ymax, float(np.max(y)) * 1.03)
            ax.plot(
                curve["fraction"], y, color=COLORS[method],
                linestyle=LINESTYLES[method], linewidth=2.2,
                label=(
                    f"{LABELS[method]}: mean NLL/obs="
                    f"{summary['native_nll_per_observation'].mean():.4f}, "
                    f"mean Y²={summary['mean_y2'].mean():.4f}, "
                    f"mean D={summary['D'].mean():.4f}"
                ),
            )
        ax.plot([0, 1], [0, 1], color="0.25", linewidth=1.4, label="ideal y=x")
        ax.set(xlim=(0, 1), ylim=(0, ymax))
        ax.set_xlabel("eigen-component fraction k / n")
        ax.set_ylabel("mean cumulative squared whitened residual / n")
        exclusion = "; 2025-07-24 excluded" if year == 2025 else ""
        ax.set_title(
            f"Real {year} July: {expected_days[year]}-day mean "
            f"400 x 8 max-min full-eigen diagnostic{exclusion}"
        )
        ax.grid(alpha=0.22)
        ax.legend(fontsize=8.5)
        fig.tight_layout()
        fig.savefig(
            output_root / f"{year}_07_monthly_plot.png",
            dpi=190,
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_native_nll(records: list[dict[str, Any]], output_root: Path) -> None:
    frame = pd.DataFrame(records)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharey=True)
    for ax, year in zip(axes, (2024, 2025)):
        year_rows = frame[frame["year"] == year]
        for method in METHODS:
            rows = year_rows[year_rows["method"] == method].sort_values("date")
            ax.plot(
                rows["date"],
                rows["native_nll_per_observation"],
                marker="o",
                markersize=3,
                linewidth=1.5,
                color=COLORS[method],
                label=LABELS[method],
            )
        ax.set_title(f"Real {year} July: Native Vecchia NLL per observation")
        ax.set_ylabel("NLL / observation")
        ax.tick_params(axis="x", rotation=75, labelsize=7)
        ax.grid(alpha=0.22)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_root / "daily_native_nll.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


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
            "vecchia_real59_adapted_fixed_full_eigen_lag643_clean_v2"
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
    parser.add_argument("--suppress-fit-prints", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.points_per_hour != 400:
        raise ValueError("This production run requires exactly 400 points per hour")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / DAILY_PLOT_DIR_NAME).mkdir(parents=True, exist_ok=True)
    specs = date_specs()
    records = load_fit_results(args.output_root)
    selected_ids = {spec["dataset_id"] for spec in specs}
    unexpected = [
        record
        for record in records
        if str(record["dataset_id"]) not in selected_ids
        or str(record["method"]) not in METHODS
    ]
    if unexpected:
        raise ValueError(
            f"{args.output_root / RESULT_JSON_NAME} contains records outside this run"
        )
    all_curves: list[pd.DataFrame] = []
    all_metrics: list[pd.DataFrame] = []
    for position, spec in enumerate(specs, start=1):
        print(f"\n[{position}/{len(specs)}] {spec['date']}", flush=True)
        asset, fits = load_or_fit_day(spec, records, args)
        curves, metrics = diagnose_day(spec, asset, fits, args)
        all_curves.append(curves)
        all_metrics.append(metrics)
        del asset, fits, curves, metrics
        gc.collect()
        torch.cuda.empty_cache()
    persist_fit_results(records, args.output_root)
    plot_native_nll(records, args.output_root)
    plot_monthly_mean_curves(
        pd.concat(all_curves, ignore_index=True),
        pd.concat(all_metrics, ignore_index=True),
        args.output_root,
    )
    print(
        f"Complete: {len(specs)} dates, {len(records)} date-method fit rows in "
        f"{args.output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
