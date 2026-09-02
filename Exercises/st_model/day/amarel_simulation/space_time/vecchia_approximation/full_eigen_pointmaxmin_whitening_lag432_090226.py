#!/usr/bin/env python3
"""Point-maxmin full-covariance eigen-whitening diagnostic for lag-432 fits.

For each selected day, this script independently max-min orders every valid
hourly point cloud, keeps the first ``points_per_hour`` points from each of the
eight hours, and stacks the resulting rows.  With the default of 400 this gives
3,200 observations.  The *same* selected observations are used for the fitted
adapted, shifted, fixed, and union parameter vectors.

For the fitted covariance ``Sigma`` and fitted mean design ``X``, the diagnostic
follows the legacy space-time *full eigen* diagnostic:

    Sigma = S Lambda S'
    y_star = Lambda^(-1/2) S' y
    X_star = Lambda^(-1/2) S' X
    beta_hat = (X_star' X_star)^(-1) X_star' y_star
    e = y_star - X_star beta_hat

All 3,200 covariance eigenpairs are retained and ordered by decreasing
eigenvalue.  The observed cumulative curve is ``cumsum(e_j^2)`` and its GLS
mean-estimation-adjusted expectation is ``cumsum(1 - h_j)``, where ``h_j`` is
the eigen-coordinate leverage.  The reference bands are descriptive because
the covariance parameters were fitted from the same day.
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
from typing import Any, Sequence

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
LOCAL_REPO = Path("/Users/joonwonlee/Documents/GEMS_TCO-1")
LOCAL_SRC = LOCAL_REPO / "src"
AMAREL_SRC = Path("/home/jl2815/tco")
SRC = AMAREL_SRC if (AMAREL_SRC / "GEMS_TCO").exists() else LOCAL_SRC
for candidate in (SRC, HERE):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from GEMS_TCO import orderings  # noqa: E402
import vecchia_adapted_vs_fixed_lag643_090126 as fit_core  # noqa: E402


METHODS = ("adapted", "shifted", "fixed", "union")
COLORS = {
    "adapted": "#1f77b4",
    "shifted": "#ff7f0e",
    "fixed": "#d62728",
    "union": "#2ca02c",
}
LINESTYLES = {
    "adapted": "-",
    "shifted": "--",
    "fixed": "-.",
    "union": ":",
}
LABELS = {
    "adapted": "adapted corridor",
    "shifted": "shifted center",
    "fixed": "fixed center",
    "union": "three-way union",
}
PARAMETERS = (
    "sigmasq",
    "range_lat",
    "range_lon",
    "range_time",
    "advec_lat",
    "advec_lon",
    "nugget",
)
EPS = 1e-12
BROWN_BRIDGE_Q95 = 1.3581015157406195


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection-file",
        type=Path,
        default=HERE / "vecchia_local_lag432_selection_090226.json",
    )
    parser.add_argument(
        "--fit-root",
        type=Path,
        default=LOCAL_REPO
        / "outputs/summer_26/vecchia_four_geometry_lag432_local_real5_synthetic5_run01",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--real-data-root", type=Path, default=Path("/Users/joonwonlee/Documents/GEMS_DATA")
    )
    parser.add_argument(
        "--synthetic-data-root",
        type=Path,
        default=Path(
            "/Users/joonwonlee/Documents/GEMS_DATA/simulation/"
            "july_st_circulant_realpattern_smooth0p5"
        ),
    )
    parser.add_argument("--dataset-indices", default="all")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--points-per-hour", type=int, default=400)
    parser.add_argument("--hours-per-day", type=int, default=8)
    parser.add_argument("--lat-range", default="-3,2")
    parser.add_argument("--lon-range", default="121,131")
    parser.add_argument("--smooth", type=float, default=0.5, choices=[0.5])
    parser.add_argument("--truth-nugget", type=float, default=1.0)
    parser.add_argument(
        "--keep-exact-loc", dest="keep_exact_loc", action="store_true", default=True
    )
    parser.add_argument("--no-keep-exact-loc", dest="keep_exact_loc", action="store_false")
    parser.add_argument("--cov-jitter", type=float, default=1e-8)
    parser.add_argument("--eigenvalue-rtol", type=float, default=1e-10)
    parser.add_argument("--eigenvalue-atol", type=float, default=1e-12)
    parser.add_argument("--brown-bridge-q", type=float, default=BROWN_BRIDGE_Q95)
    parser.add_argument("--resample-grid", type=int, default=1000)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    if args.output_root is None:
        args.output_root = (
            args.fit_root
            / "comparison_report"
            / f"full_pointmaxmin_full_eigen_whitening_{int(args.points_per_hour)}_run02"
        )
    return args


def clean_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(clean_json(value), indent=2, sort_keys=True), encoding="utf-8"
    )


def parse_names(text: str) -> list[str]:
    names = [part.strip() for part in str(text).split(",") if part.strip()]
    if not names:
        raise ValueError("At least one method is required")
    unknown = sorted(set(names).difference(METHODS))
    if unknown:
        raise ValueError(f"Unknown methods {unknown}; expected entries from {METHODS}")
    if len(set(names)) != len(names):
        raise ValueError("--methods contains duplicates")
    return names


def parse_indices(text: str, n: int) -> list[int]:
    value = str(text).strip().lower()
    if value == "all":
        return list(range(n))
    out = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not out:
        raise ValueError("--dataset-indices must be 'all' or a comma-separated list")
    if len(set(out)) != len(out) or any(index < 0 or index >= n for index in out):
        raise ValueError(f"Invalid --dataset-indices={text!r} for {n} data sets")
    return out


def load_asset(spec: dict[str, Any], args: argparse.Namespace) -> fit_core.DayAsset:
    if str(spec["data_kind"]) == "real":
        return fit_core.load_real_asset(spec, args)
    if str(spec["data_kind"]) == "synthetic":
        return fit_core.load_synthetic_asset(spec, args)
    raise ValueError(f"Unknown data_kind={spec['data_kind']!r}")


def point_maxmin_sample(
    asset: fit_core.DayAsset, points_per_hour: int
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Fully order every valid hour, then retain the requested prefix."""
    if len(asset.keys) != 8:
        raise ValueError(f"Expected eight hours for {asset.dataset_id}, got {len(asset.keys)}")
    selected_chunks: list[np.ndarray] = []
    selected_rows: list[dict[str, Any]] = []
    hour_rows: list[dict[str, Any]] = []
    for hour_index, key in enumerate(asset.keys):
        rows = asset.source_map[key].detach().cpu().numpy().astype(np.float64, copy=False)
        valid = np.isfinite(rows[:, 0]) & np.isfinite(rows[:, 1]) & np.isfinite(rows[:, 2])
        source_indices = np.flatnonzero(valid).astype(np.int64)
        valid_rows = np.ascontiguousarray(rows[source_indices])
        n_valid = int(valid_rows.shape[0])
        if n_valid < int(points_per_hour):
            raise ValueError(
                f"{asset.dataset_id} {key} has {n_valid} valid points, fewer than "
                f"points_per_hour={int(points_per_hour)}"
            )
        coords = np.ascontiguousarray(valid_rows[:, 0:2], dtype=np.float64)
        started = time.perf_counter()
        full_order = np.asarray(orderings.maxmin_cpp(coords), dtype=np.int64)
        order_seconds = time.perf_counter() - started
        if full_order.shape != (n_valid,):
            raise RuntimeError(
                f"maxmin order length {full_order.size} != valid point count {n_valid}"
            )
        if np.unique(full_order).size != n_valid or full_order.min() < 0 or full_order.max() >= n_valid:
            raise RuntimeError(f"maxmin output is not a permutation for {asset.dataset_id} {key}")
        prefix = full_order[: int(points_per_hour)]
        picked = np.ascontiguousarray(valid_rows[prefix])
        selected_chunks.append(picked)
        hour_rows.append(
            {
                "dataset_id": asset.dataset_id,
                "data_kind": asset.data_kind,
                "date": asset.date,
                "hour_index": hour_index,
                "hour_key": key,
                "n_valid_before_ordering": n_valid,
                "full_maxmin_order_size": int(full_order.size),
                "n_selected": int(points_per_hour),
                "maxmin_order_seconds": order_seconds,
            }
        )
        for rank, (ordered_index, source_index) in enumerate(
            zip(prefix.tolist(), source_indices[prefix].tolist()), start=1
        ):
            row = picked[rank - 1]
            selected_rows.append(
                {
                    "dataset_id": asset.dataset_id,
                    "data_kind": asset.data_kind,
                    "date": asset.date,
                    "hour_index": hour_index,
                    "hour_key": key,
                    "maxmin_rank_within_hour": rank,
                    "valid_array_index": ordered_index,
                    "source_row_index": source_index,
                    "latitude": row[0],
                    "longitude": row[1],
                    "time_coordinate": row[3],
                    "centered_value": row[2],
                }
            )
    selected = np.ascontiguousarray(np.concatenate(selected_chunks, axis=0))
    expected = 8 * int(points_per_hour)
    if selected.shape[0] != expected:
        raise RuntimeError(f"Selected {selected.shape[0]} points, expected {expected}")
    return selected, pd.DataFrame(selected_rows), pd.DataFrame(hour_rows)


def mean_design(selected: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Return the fitted model's [intercept, latitude, seven hour dummies] design."""
    if selected.shape[1] < 11:
        raise ValueError(f"Expected at least 11 columns, got shape={selected.shape}")
    lat = selected[:, 0]
    design = np.column_stack(
        [np.ones(len(selected)), lat - float(np.mean(lat)), selected[:, 4:11]]
    ).astype(np.float64, copy=False)
    rank = int(np.linalg.matrix_rank(design))
    if rank != design.shape[1]:
        raise RuntimeError(
            f"Mean design is rank deficient: rank={rank}, columns={design.shape[1]}"
        )
    values = selected[:, 2].astype(np.float64, copy=False)
    return np.ascontiguousarray(design), np.ascontiguousarray(values), rank


def fit_parameters(fit_row: pd.Series) -> dict[str, float]:
    out = {name: float(fit_row[f"est_{name}"]) for name in PARAMETERS}
    if not all(np.isfinite(value) for value in out.values()):
        raise ValueError(f"Non-finite fitted parameters: {out}")
    for name in ("sigmasq", "range_lat", "range_lon", "range_time"):
        if out[name] <= 0.0:
            raise ValueError(f"Fitted {name} must be positive, got {out[name]}")
    if out["nugget"] < 0.0:
        raise ValueError(f"Fitted nugget must be nonnegative, got {out['nugget']}")
    return out


def fitted_st_covariance(
    selected: np.ndarray, est: dict[str, float], smooth: float, jitter: float
) -> np.ndarray:
    """Build the same natural-parameter ST Matérn covariance as the fitted model."""
    lat = selected[:, 0].astype(np.float64, copy=False)
    lon = selected[:, 1].astype(np.float64, copy=False)
    time_coord = selected[:, 3].astype(np.float64, copy=False)

    delta_t = time_coord[:, None] - time_coord[None, :]
    scaled_sq = lat[:, None] - lat[None, :]
    scaled_sq -= float(est["advec_lat"]) * delta_t
    np.square(scaled_sq, out=scaled_sq)
    scaled_sq /= float(est["range_lat"]) ** 2

    work = lon[:, None] - lon[None, :]
    work -= float(est["advec_lon"]) * delta_t
    np.square(work, out=work)
    work /= float(est["range_lon"]) ** 2
    scaled_sq += work
    del work

    np.square(delta_t, out=delta_t)
    delta_t /= float(est["range_time"]) ** 2
    scaled_sq += delta_t
    del delta_t

    np.maximum(scaled_sq, 0.0, out=scaled_sq)
    np.sqrt(scaled_sq, out=scaled_sq)
    if math.isclose(float(smooth), 0.5, rel_tol=0.0, abs_tol=1e-12):
        np.negative(scaled_sq, out=scaled_sq)
        np.exp(scaled_sq, out=scaled_sq)
    elif math.isclose(float(smooth), 1.5, rel_tol=0.0, abs_tol=1e-12):
        distance = scaled_sq.copy()
        np.negative(scaled_sq, out=scaled_sq)
        np.exp(scaled_sq, out=scaled_sq)
        scaled_sq *= 1.0 + distance
        del distance
    else:
        raise ValueError(f"Unsupported smooth={smooth}")
    scaled_sq *= float(est["sigmasq"])
    scaled_sq.flat[:: len(selected) + 1] += float(est["nugget"]) + float(jitter)
    return np.ascontiguousarray(scaled_sq)


def full_eigen_whitening(
    selected: np.ndarray,
    design: np.ndarray,
    values: np.ndarray,
    est: dict[str, float],
    fit_row: pd.Series,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    started_cov = time.perf_counter()
    covariance = fitted_st_covariance(selected, est, args.smooth, args.cov_jitter)
    covariance_seconds = time.perf_counter() - started_cov

    started_eigen = time.perf_counter()
    covariance = np.asarray(0.5 * (covariance + covariance.T), order="F")
    evals, evecs = scipy.linalg.eigh(
        covariance,
        lower=True,
        overwrite_a=True,
        check_finite=False,
        driver="evd",
    )
    eigen_seconds = time.perf_counter() - started_eigen
    del covariance

    n = int(len(selected))
    max_eval = max(float(np.max(evals)), EPS)
    threshold = max(
        float(args.eigenvalue_atol), float(args.eigenvalue_rtol) * max_eval
    )
    if evals.size != n:
        raise RuntimeError(f"Full covariance eigh returned {evals.size} eigenpairs, expected {n}")
    if not np.all(np.isfinite(evals)) or float(np.min(evals)) <= 0.0:
        raise RuntimeError(
            "Full covariance is not numerically positive definite: "
            f"min eigenvalue={float(np.nanmin(evals)):.6g}"
        )
    order = np.argsort(evals)[::-1]
    evals = np.ascontiguousarray(evals[order])
    evecs = np.ascontiguousarray(evecs[:, order])
    sqrt_evals = np.sqrt(evals)

    # This is the same eigenbasis GLS whitening used by the ST reference file.
    whitened_values = (evecs.T @ values) / sqrt_evals
    whitened_design = (evecs.T @ design) / sqrt_evals[:, None]
    xtx = whitened_design.T @ whitened_design
    xty = whitened_design.T @ whitened_values
    try:
        beta = scipy.linalg.solve(
            xtx, xty, assume_a="sym", check_finite=False
        )
        xtx_inv = scipy.linalg.inv(xtx, check_finite=False)
    except (ValueError, np.linalg.LinAlgError):
        xtx_inv = np.linalg.pinv(xtx)
        beta = xtx_inv @ xty

    scores = whitened_values - whitened_design @ beta
    y2 = np.square(scores)
    leverage = np.einsum(
        "ij,ij->i", whitened_design @ xtx_inv, whitened_design
    )
    expected_increment = np.clip(1.0 - leverage, 0.0, None)
    cumulative = np.cumsum(y2)
    cumulative_expected = np.cumsum(expected_increment)
    residual_df = float(np.sum(expected_increment))
    mean_rank = int(np.linalg.matrix_rank(design))
    theoretical_df = float(n - mean_rank)
    if not math.isclose(residual_df, theoretical_df, rel_tol=1e-8, abs_tol=1e-6):
        raise RuntimeError(
            f"GLS residual df={residual_df:.10g} does not match n-rank(X)={theoretical_df:.10g}"
        )
    m = int(evals.size)
    index = np.arange(1, m + 1, dtype=np.int64)
    frac_expected = cumulative_expected / residual_df
    scaled_cumulative = cumulative / residual_df
    band_width = float(args.brown_bridge_q) * math.sqrt(2.0 / residual_df)
    bridge = scaled_cumulative - frac_expected
    bridge_d = float(
        np.max(np.abs(cumulative - cumulative_expected))
        / math.sqrt(2.0 * residual_df)
    )
    full_gaussian_nll = 0.5 * (
        float(np.sum(np.log(evals)))
        + float(np.sum(y2))
        + n * math.log(2.0 * math.pi)
    )
    curve = pd.DataFrame(
        {
            "index": index,
            "frac_index": frac_expected,
            "eigenvalue": evals,
            "whitened_score": scores,
            "squared_score": y2,
            "leverage": leverage,
            "expected_increment": expected_increment,
            "cumulative_squared_score": cumulative,
            "expected": cumulative_expected,
            "scaled_cumulative": scaled_cumulative,
            "scaled_expected": frac_expected,
            "scaled_band_lower": frac_expected - band_width,
            "scaled_band_upper": frac_expected + band_width,
        }
    )
    summary: dict[str, Any] = {
        "n_selected": int(len(selected)),
        "mean_rank": mean_rank,
        "residual_df": residual_df,
        "n_eigen_kept": m,
        "n_eigen_dropped": 0,
        "eigen_threshold": threshold,
        "min_kept_eigenvalue": float(evals[-1]),
        "max_kept_eigenvalue": float(evals[0]),
        "sum_squared_score": float(cumulative[-1]),
        "mean_squared_score": float(cumulative[-1] / residual_df),
        "score2_per_residual_df": float(cumulative[-1] / residual_df),
        "max_abs_bridge_scaled": bridge_d,
        "brown_bridge_width": band_width,
        "full_gaussian_nll": full_gaussian_nll,
        "full_gaussian_nll_per_obs": full_gaussian_nll / n,
        "covariance_seconds": covariance_seconds,
        "eigendecomposition_seconds": eigen_seconds,
        "diagnostic_seconds": covariance_seconds + eigen_seconds,
        "native_nll_per_target": float(fit_row["final_native_nll"]),
        "n_fit_target_points": int(fit_row["n_target_points"]),
        "fit_seconds": float(fit_row["fit_s"]),
    }
    summary.update({f"fit_{name}": float(est[name]) for name in PARAMETERS})
    summary.update(
        {
            "gls_beta_intercept": float(beta[0]),
            "gls_beta_centered_latitude": float(beta[1]),
            **{
                f"gls_beta_hour_dummy_{hour}": float(beta[hour + 1])
                for hour in range(1, 8)
            },
        }
    )
    del (
        evals,
        evecs,
        scores,
        y2,
        cumulative,
        cumulative_expected,
        whitened_values,
        whitened_design,
        xtx,
        xtx_inv,
    )
    gc.collect()
    return curve, summary


def plot_four_methods(
    curves: dict[str, pd.DataFrame],
    summaries: dict[str, dict[str, Any]],
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 7.4))
    all_y = [1.0]
    for method in METHODS:
        if method not in curves:
            continue
        curve = curves[method]
        summary = summaries[method]
        x = curve["scaled_expected"].to_numpy(dtype=float)
        y = curve["scaled_cumulative"].to_numpy(dtype=float)
        all_y.append(float(np.nanmax(y)))
        label = (
            f"{LABELS[method]}  D={summary['max_abs_bridge_scaled']:.4f}, "
            f"mean $Y^2$={summary['mean_squared_score']:.4f}, "
            f"NLL/target={summary['native_nll_per_target']:.4f}"
        )
        ax.plot(
            x,
            y,
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            linewidth=2.0,
            label=label,
        )
    any_curve = next(iter(curves.values()))
    x = any_curve["scaled_expected"].to_numpy(dtype=float)
    lower = any_curve["scaled_band_lower"].to_numpy(dtype=float)
    upper = any_curve["scaled_band_upper"].to_numpy(dtype=float)
    ax.plot([0.0, 1.0], [0.0, 1.0], color="0.25", linewidth=1.5, label="ideal y=x")
    ax.plot(x, lower, color="0.60", linestyle=(0, (4, 4)), linewidth=1.0)
    ax.plot(x, upper, color="0.60", linestyle=(0, (4, 4)), linewidth=1.0, label="approx. 95% band")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, max(1.05, 1.03 * max(all_y)))
    ax.set_xlabel("cumulative expected residual df fraction (largest eigenvalue first)")
    ax.set_ylabel("cumulative whitened squared score / residual df")
    ax.set_title(title)
    ax.grid(alpha=0.22)
    ax.legend(loc="best", fontsize=8.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def resample_curve(curve: pd.DataFrame, n_grid: int) -> pd.DataFrame:
    grid = np.linspace(1.0 / int(n_grid), 1.0, int(n_grid))
    x = curve["scaled_expected"].to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "frac_index": grid,
            "scaled_cumulative": np.interp(
                grid, x, curve["scaled_cumulative"].to_numpy(dtype=float)
            ),
            "scaled_expected": grid,
        }
    )


def aggregate_outputs(
    output_root: Path,
    selected_specs: Sequence[dict[str, Any]],
    methods: Sequence[str],
    n_grid: int,
) -> None:
    summary_frames: list[pd.DataFrame] = []
    curve_frames: list[pd.DataFrame] = []
    hour_frames: list[pd.DataFrame] = []
    point_frames: list[pd.DataFrame] = []
    for spec in selected_specs:
        task_dir = output_root / str(spec["dataset_id"])
        summary_path = task_dir / "full_eigen_whitening_summary.csv"
        if summary_path.is_file():
            summary_frames.append(pd.read_csv(summary_path))
        hour_path = task_dir / "hourly_maxmin_manifest.csv"
        if hour_path.is_file():
            hour_frames.append(pd.read_csv(hour_path))
        points_path = task_dir / "selected_points.csv"
        if points_path.is_file():
            point_frames.append(pd.read_csv(points_path))
        for method in methods:
            curve_path = task_dir / f"full_eigen_whitening_curve_{method}.csv"
            if curve_path.is_file():
                curve_frames.append(pd.read_csv(curve_path))
    if not summary_frames or not curve_frames:
        raise RuntimeError("No completed full-eigen outputs were found for aggregation")
    summary = pd.concat(summary_frames, ignore_index=True)
    curves = pd.concat(curve_frames, ignore_index=True)
    summary.to_csv(
        output_root / "full_eigen_whitening_summary.csv",
        index=False,
        float_format="%.4f",
    )
    curves.to_csv(
        output_root / "full_eigen_whitening_curves.csv",
        index=False,
        float_format="%.10g",
    )
    if hour_frames:
        pd.concat(hour_frames, ignore_index=True).to_csv(
            output_root / "hourly_maxmin_manifest.csv", index=False, float_format="%.4f"
        )
    if point_frames:
        pd.concat(point_frames, ignore_index=True).to_csv(
            output_root / "selected_points.csv", index=False, float_format="%.10g"
        )

    resampled_rows: list[pd.DataFrame] = []
    for (data_kind, method), group in curves.groupby(["data_kind", "method"], sort=False):
        day_rows: list[pd.DataFrame] = []
        for dataset_id, day_curve in group.groupby("dataset_id", sort=False):
            sampled = resample_curve(day_curve.sort_values("frac_index"), n_grid)
            sampled["dataset_id"] = dataset_id
            day_rows.append(sampled)
        stacked = pd.concat(day_rows, ignore_index=True)
        mean_curve = (
            stacked.groupby("frac_index", as_index=False)
            .agg(
                scaled_cumulative_mean=("scaled_cumulative", "mean"),
                scaled_cumulative_sd=("scaled_cumulative", "std"),
                n_dates=("dataset_id", "nunique"),
            )
            .sort_values("frac_index")
        )
        mean_curve["data_kind"] = data_kind
        mean_curve["method"] = method
        resampled_rows.append(mean_curve)
    means = pd.concat(resampled_rows, ignore_index=True)
    means.to_csv(
        output_root / "mean_full_eigen_whitening_curves.csv",
        index=False,
        float_format="%.10g",
    )

    for data_kind in sorted(means["data_kind"].unique()):
        fig, ax = plt.subplots(figsize=(9.4, 7.4))
        subset_summary = summary[summary["data_kind"] == data_kind]
        all_y = [1.0]
        for method in METHODS:
            sub = means[(means["data_kind"] == data_kind) & (means["method"] == method)]
            if sub.empty:
                continue
            sub = sub.sort_values("frac_index")
            x = sub["frac_index"].to_numpy(dtype=float)
            y = sub["scaled_cumulative_mean"].to_numpy(dtype=float)
            all_y.append(float(np.nanmax(y)))
            method_summary = subset_summary[subset_summary["method"] == method]
            d_mean = float(method_summary["max_abs_bridge_scaled"].mean())
            y2_mean = float(method_summary["mean_squared_score"].mean())
            nll_mean = float(method_summary["native_nll_per_target"].mean())
            label = (
                f"{LABELS[method]}  mean D={d_mean:.4f}, "
                f"mean $Y^2$={y2_mean:.4f}, mean NLL={nll_mean:.4f}"
            )
            ax.plot(
                x,
                y,
                color=COLORS[method],
                linestyle=LINESTYLES[method],
                linewidth=2.2,
                label=label,
            )
        ax.plot([0.0, 1.0], [0.0, 1.0], color="0.25", linewidth=1.5, label="ideal y=x")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, max(1.05, 1.03 * max(all_y)))
        ax.set_xlabel(
            "cumulative expected residual df fraction (largest eigenvalue first)"
        )
        ax.set_ylabel("mean cumulative whitened squared score / residual df")
        ax.set_title(
            f"{str(data_kind).capitalize()}: mean point-maxmin full-covariance "
            f"eigen-whitening diagnostic"
        )
        ax.grid(alpha=0.22)
        ax.legend(loc="best", fontsize=8.5)
        fig.tight_layout()
        fig.savefig(
            output_root / f"{data_kind}_mean_full_eigen_whitening_four_methods.png",
            dpi=190,
            bbox_inches="tight",
        )
        plt.close(fig)

    mean_summary = (
        summary.groupby(["data_kind", "method"], as_index=False)
        .agg(
            n_dates=("dataset_id", "nunique"),
            mean_squared_score=("mean_squared_score", "mean"),
            max_abs_bridge_scaled=("max_abs_bridge_scaled", "mean"),
            native_nll_per_target=("native_nll_per_target", "mean"),
            diagnostic_seconds=("diagnostic_seconds", "mean"),
        )
    )
    mean_summary.to_csv(
        output_root / "mean_full_eigen_whitening_summary.csv",
        index=False,
        float_format="%.4f",
    )


def write_notes(output_root: Path, args: argparse.Namespace) -> None:
    text = f"""# Point-maxmin full-covariance eigen-whitening diagnostic

For each of eight hours, all valid points are max-min ordered at point level and
the first {int(args.points_per_hour)} are retained. The combined diagnostic has
{8 * int(args.points_per_hour)} observations, shared by all four fitted methods.

The fitted mean design is the same one used by the lag-432 fits: intercept,
centered latitude, and seven hourly dummy variables. For every method the
diagnostic directly decomposes the full 3,200 by 3,200 fitted covariance:

`Sigma_hat = S Lambda S'`


`y_star = Lambda^(-1/2) S' y`,

`X_star = Lambda^(-1/2) S' X`,

`beta_hat = (X_star' X_star)^(-1) X_star' y_star`, and

`e = y_star - X_star beta_hat`.

Thus the denominator is the square root of the eigenvalue, not the eigenvalue.
All 3,200 covariance eigenpairs are retained and ordered from largest fitted
eigenvalue to smallest. The plotted observed curve is cumulative `e_j^2`; its
mean-estimation-adjusted expected curve is cumulative `1-h_j`, where `h_j` is
the GLS leverage of eigen-coordinate `j`. Both are divided by residual df, so a
well-calibrated fitted covariance should follow the identity line.

The fitted parameters were estimated from the same day, so the plotted 95%
bands are descriptive diagnostics rather than exact hypothesis tests. All four
curves use the same selected points; differences arise only from their fitted
covariance parameters. The statistical nugget is taken from each fitted row
(zero in this experiment), with numerical covariance jitter
`{float(args.cov_jitter):.2e}` added to the diagonal.
"""
    (output_root / "MATH_AND_INTERPRETATION.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.points_per_hour <= 0 or args.hours_per_day != 8:
        raise ValueError("This diagnostic requires positive points-per-hour and exactly 8 hours")
    methods = parse_names(args.methods)
    selection, all_specs = fit_core.load_selection(args.selection_file)
    indices = parse_indices(args.dataset_indices, len(all_specs))
    selected_specs = [all_specs[index] for index in indices]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    write_notes(output_root, args)
    write_json(
        output_root / "run_config.json",
        {
            "selection_file": args.selection_file,
            "fit_root": args.fit_root,
            "output_root": output_root,
            "dataset_indices": indices,
            "methods": methods,
            "points_per_hour": int(args.points_per_hour),
            "hours_per_day": int(args.hours_per_day),
            "smooth": float(args.smooth),
            "cov_jitter": float(args.cov_jitter),
            "eigenvalue_rtol": float(args.eigenvalue_rtol),
            "eigenvalue_atol": float(args.eigenvalue_atol),
            "brown_bridge_q": float(args.brown_bridge_q),
            "selection": selection,
        },
    )
    fits = pd.read_csv(Path(args.fit_root) / "all_fits.csv")

    for position, spec in zip(indices, selected_specs):
        dataset_id = str(spec["dataset_id"])
        task_dir = output_root / dataset_id
        complete_path = task_dir / "COMPLETE"
        if args.skip_existing and complete_path.is_file():
            print(f"Skipping completed {position}: {dataset_id}", flush=True)
            continue
        task_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[{position + 1}/{len(all_specs)}] Loading {dataset_id} ({spec['date']})",
            flush=True,
        )
        asset = load_asset(spec, args)
        selected, selected_frame, hourly_frame = point_maxmin_sample(
            asset, args.points_per_hour
        )
        selected_frame.to_csv(
            task_dir / "selected_points.csv", index=False, float_format="%.10g"
        )
        hourly_frame.to_csv(
            task_dir / "hourly_maxmin_manifest.csv", index=False, float_format="%.4f"
        )
        design, values, mean_rank = mean_design(selected)
        print(
            f"  Selected {len(selected)} rows; mean rank={mean_rank}; "
            f"point max-min seconds={hourly_frame['maxmin_order_seconds'].sum():.2f}",
            flush=True,
        )
        dataset_fits = fits[fits["dataset_id"] == dataset_id]
        curves: dict[str, pd.DataFrame] = {}
        summaries: dict[str, dict[str, Any]] = {}
        summary_rows: list[dict[str, Any]] = []
        for method in methods:
            fit_matches = dataset_fits[
                (dataset_fits["geometry"] == method) & (dataset_fits["status"] == "ok")
            ]
            if len(fit_matches) != 1:
                raise RuntimeError(
                    f"Expected one successful fit for {dataset_id} {method}, got {len(fit_matches)}"
                )
            fit_row = fit_matches.iloc[0]
            est = fit_parameters(fit_row)
            print(f"  Full eigen-whitening: {method}", flush=True)
            curve, summary = full_eigen_whitening(
                selected, design, values, est, fit_row, args
            )
            curve.insert(0, "method", method)
            curve.insert(0, "date", str(spec["date"]))
            curve.insert(0, "data_kind", str(spec["data_kind"]))
            curve.insert(0, "dataset_id", dataset_id)
            curve.to_csv(
                task_dir / f"full_eigen_whitening_curve_{method}.csv",
                index=False,
                float_format="%.10g",
            )
            summary.update(
                {
                    "dataset_id": dataset_id,
                    "data_kind": str(spec["data_kind"]),
                    "date": str(spec["date"]),
                    "method": method,
                    "method_label": LABELS[method],
                    "points_per_hour": int(args.points_per_hour),
                    "n_hours": 8,
                    "point_maxmin_seconds": float(
                        hourly_frame["maxmin_order_seconds"].sum()
                    ),
                }
            )
            curves[method] = curve
            summaries[method] = summary
            summary_rows.append(summary)
            print(
                f"    D={summary['max_abs_bridge_scaled']:.4f}, "
                f"meanY2={summary['mean_squared_score']:.4f}, "
                f"eig seconds={summary['eigendecomposition_seconds']:.2f}",
                flush=True,
            )
        summary_frame = pd.DataFrame(summary_rows)
        leading = [
            "date",
            "data_kind",
            "dataset_id",
            "method",
            "method_label",
            "points_per_hour",
            "n_hours",
        ]
        summary_frame = summary_frame[
            leading + [column for column in summary_frame.columns if column not in leading]
        ]
        summary_frame.to_csv(
            task_dir / "full_eigen_whitening_summary.csv",
            index=False,
            float_format="%.4f",
        )
        plot_four_methods(
            curves,
            summaries,
            (
                f"{str(spec['data_kind']).capitalize()} {spec['date']}: point-maxmin "
                f"full-covariance eigen-whitening ({int(args.points_per_hour)} per hour)"
            ),
            task_dir / "full_eigen_whitening_four_methods.png",
        )
        complete_path.write_text("complete\n", encoding="utf-8")
        del asset, selected, design, values, curves, summaries
        gc.collect()

    aggregate_outputs(output_root, selected_specs, methods, args.resample_grid)
    print(f"Completed and aggregated: {output_root}", flush=True)


if __name__ == "__main__":
    main()
