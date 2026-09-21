#!/usr/bin/env python3
"""Ten-day full-data Vecchia comparison with cumulative Ritz diagnostics.

July 3, 5, 7, 12, and 15 are analyzed in both 2024 and 2025, giving ten test
dates spaced across the first half of each month.  This run deliberately omits
the 400x8 dense covariance eigendecomposition.  It compares only:

1. native full-data Vecchia negative log likelihood per observation; and
2. a full-data sparse-precision SLQ/Lanczos diagnostic.

SLQ estimates full-spectrum mode-count quantiles using identical probes for
adapted and fixed.  Four paired random-start Lanczos replicates each supply 512
implicit Ritz modes: 170/170/172 representatives from the low/middle/high
precision thirds.  If e_j is standardized residual energy for one Ritz mode
and w_j is the full-spectrum fraction it represents, plots show the replicate
mean of

    x_k = sum_{j<=k} w_j,    y_k = sum_{j<=k} w_j e_j,

against y=x.  Each spectral third has total weight 1/3.  The endpoint is not
renormalized, so overall residual variance miscalibration remains visible.

The low-to-high precision ordering is a spectral frequency proxy, not an exact
Fourier frequency on the irregular space-time Vecchia graph.  Ritz tail
residual quality is saved and displayed with every diagnostic.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import socket
import time
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
import torch


HERE = Path(__file__).resolve().parent
SUBMIT_DIR = Path(os.environ.get("SLURM_SUBMIT_DIR", str(HERE))).resolve()
AMAREL_ROOT = Path("/home/jl2815/tco")
LOCAL_SRC = Path("/Users/joonwonlee/Documents/GEMS_TCO-1/src")
INTERACTION_DIR = HERE.parent / "interaction_diagnostic"
for candidate in (AMAREL_ROOT, SUBMIT_DIR, HERE, LOCAL_SRC, INTERACTION_DIR):
    if candidate.exists() and str(candidate) not in os.sys.path:
        os.sys.path.insert(0, str(candidate))

import vecchia_real59_adapted_fixed_threeway_slq512_lag643 as engine  # noqa: E402


core = engine.core
reference = engine.reference
METHODS = engine.METHODS
COLORS = engine.COLORS
LINESTYLES = engine.LINESTYLES
LABELS = engine.LABELS
BANDS = engine.BAND_NAMES
BAND_COLORS = engine.BAND_COLORS
DAILY_DIR = "daily_subplots"
DAILY_BAND_DIR = "daily_band_subplots"
CACHE_DIR = ".diagnostic_cache"
TEST_DAYS = (3, 5, 7, 12, 15)


def selected_date_specs() -> list[dict[str, Any]]:
    """Return the same real-data metadata as the reference for ten test dates."""
    available = {spec["date"]: spec for spec in reference.date_specs()}
    requested = [
        f"{year}-07-{day:02d}" for year in (2024, 2025) for day in TEST_DAYS
    ]
    missing = [date for date in requested if date not in available]
    if missing:
        raise RuntimeError(f"Reference date metadata is missing {missing}")
    return [available[date] for date in requested]


def common_rng(base_seed: int, date: str, stream: int) -> np.random.Generator:
    """Common random numbers for paired adapted/fixed comparisons."""
    sequence = np.random.SeedSequence(
        [int(base_seed), int(date.replace("-", "")), int(stream)]
    )
    return np.random.default_rng(sequence)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def add_metadata(
    frame: pd.DataFrame, spec: dict[str, Any], method: str
) -> pd.DataFrame:
    out = frame.copy()
    values = {
        "dataset_id": spec["dataset_id"],
        "year": spec["year"],
        "date": spec["date"],
        "method": method,
    }
    for column, value in values.items():
        out[column] = value
    leading = list(values)
    return out[leading + [column for column in out if column not in leading]]


def attach_cumulative_columns(
    modes: pd.DataFrame,
    confidence_z: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    out = modes.sort_values("selection_index").reset_index(drop=True).copy()
    if len(out) != 512 or out["selection_index"].nunique() != 512:
        raise RuntimeError(f"Expected 512 distinct selected modes, found {len(out)}")
    expected_indices = np.arange(1, 513)
    if not np.array_equal(out["selection_index"].to_numpy(int), expected_indices):
        raise RuntimeError("selection_index must run consecutively from 1 to 512")
    precision_values = out["precision_ritz_value"].to_numpy(float)
    if np.any(np.diff(precision_values) < 0.0):
        raise RuntimeError("Selected Ritz modes are not ordered by precision eigenvalue")
    energy = out["standardized_residual_energy"].to_numpy(float)
    if not np.all(np.isfinite(energy)) or np.any(energy < 0.0):
        raise RuntimeError("Ritz energies must be finite and nonnegative")

    counts = out.groupby("band").size().to_dict()
    observed = tuple(int(counts.get(band, 0)) for band in BANDS)
    if observed != (170, 170, 172):
        raise RuntimeError(f"Expected band counts 170/170/172, found {observed}")
    weights = np.asarray(
        [1.0 / (3.0 * float(counts[band])) for band in out["band"]],
        dtype=np.float64,
    )
    global_expected = np.cumsum(weights)
    global_expected[-1] = 1.0
    global_cumulative = np.cumsum(weights * energy)
    global_null_sd = np.sqrt(2.0 * np.cumsum(np.square(weights)))
    out["spectral_weight"] = weights
    out["scaled_expected"] = global_expected
    out["scaled_cumulative"] = global_cumulative
    out["diagonal_departure"] = global_cumulative - global_expected
    out["pointwise_null_lower"] = np.maximum(
        0.0, global_expected - confidence_z * global_null_sd
    )
    out["pointwise_null_upper"] = global_expected + confidence_z * global_null_sd

    out["within_band_expected"] = np.nan
    out["within_band_cumulative"] = np.nan
    out["within_band_departure"] = np.nan
    out["within_band_null_lower"] = np.nan
    out["within_band_null_upper"] = np.nan
    metrics: dict[str, float] = {
        "weighted_mean_energy": float(global_cumulative[-1]),
        "max_abs_diagonal_departure": float(
            np.max(np.abs(global_cumulative - global_expected))
        ),
        "signed_endpoint_departure": float(global_cumulative[-1] - 1.0),
    }
    for band in BANDS:
        indices = out.index[out["band"].eq(band)].to_numpy(int)
        band_energy = energy[indices]
        count = len(indices)
        band_expected = np.arange(1, count + 1, dtype=np.float64) / count
        band_cumulative = np.cumsum(band_energy) / count
        band_null_sd = np.sqrt(2.0 * np.arange(1, count + 1)) / count
        out.loc[indices, "within_band_expected"] = band_expected
        out.loc[indices, "within_band_cumulative"] = band_cumulative
        out.loc[indices, "within_band_departure"] = band_cumulative - band_expected
        out.loc[indices, "within_band_null_lower"] = np.maximum(
            0.0, band_expected - confidence_z * band_null_sd
        )
        out.loc[indices, "within_band_null_upper"] = (
            band_expected + confidence_z * band_null_sd
        )
        metrics[f"{band}_mean_energy"] = float(band_cumulative[-1])
        metrics[f"{band}_max_abs_departure"] = float(
            np.max(np.abs(band_cumulative - band_expected))
        )
    return out, metrics


def cache_paths(output_root: Path, date: str, method: str) -> dict[str, Path]:
    root = output_root / CACHE_DIR / date / method
    return {
        "root": root,
        "slq_curve": root / "slq_spectrum_curve.csv",
        "bands": root / "slq_three_bands.csv",
        "modes": root / "selected_512x4_ritz_modes_with_cumulative.csv",
        "summary": root / "nll_ritz512x4_summary.json",
        "complete": root / "COMPLETE",
    }


def load_cached_method(
    paths: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    required = ("slq_curve", "bands", "modes", "summary")
    if not paths["complete"].is_file() or any(
        not paths[name].is_file() for name in required
    ):
        raise FileNotFoundError("incomplete diagnostic cache")
    return (
        pd.read_csv(paths["slq_curve"]),
        pd.read_csv(paths["bands"]),
        pd.read_csv(paths["modes"]),
        json.loads(paths["summary"].read_text(encoding="utf-8")),
    )


def run_method(
    spec: dict[str, Any],
    method: str,
    asset: Any,
    fit_row: pd.Series,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    paths = cache_paths(args.output_root, spec["date"], method)
    try:
        cached = load_cached_method(paths)
        print(f"    {method}: complete full-data cache reused", flush=True)
        return cached
    except FileNotFoundError:
        pass

    print(f"    {method}: build full sparse Vecchia precision", flush=True)
    precision, precision_summary = engine.rebuild_precision(method, asset, fit_row, args)
    slq_rng = common_rng(int(args.random_seed), spec["date"], stream=0)
    print(
        f"    {method}: SLQ full-spectrum CDF and thirds (paired probes)",
        flush=True,
    )
    slq_curve, bands, slq_summary, slq_atoms, slq_cumulative = engine.slq_spectrum(
        precision, args, slq_rng
    )
    replicate_modes: list[pd.DataFrame] = []
    replicate_rows: list[dict[str, float]] = []
    ritz_rows: list[dict[str, Any]] = []
    for replicate in range(1, int(args.ritz_replicates) + 1):
        print(
            f"    {method}: paired Ritz replicate {replicate}/{args.ritz_replicates}",
            flush=True,
        )
        ritz_rng = common_rng(
            int(args.random_seed), spec["date"], stream=1000 + replicate
        )
        one_modes, one_ritz = engine.select_ritz_modes(
            precision, args, ritz_rng, bands, slq_atoms, slq_cumulative
        )
        one_modes, one_cumulative = attach_cumulative_columns(
            one_modes, args.confidence_z
        )
        one_modes.insert(0, "ritz_replicate", replicate)
        replicate_modes.append(one_modes)
        replicate_rows.append(one_cumulative)
        ritz_rows.append(one_ritz)
    modes = pd.concat(replicate_modes, ignore_index=True)

    mean_global = modes.groupby("selection_index", as_index=False).agg(
        expected=("scaled_expected", "mean"),
        cumulative=("scaled_cumulative", "mean"),
    )
    endpoints = np.asarray(
        [row["weighted_mean_energy"] for row in replicate_rows], dtype=float
    )
    quality = np.asarray(
        [row["selected_fraction_within_ritz_tolerance"] for row in ritz_rows],
        dtype=float,
    )
    cumulative_summary: dict[str, float | int] = {
        "ritz_replicates": int(args.ritz_replicates),
        "selected_modes_per_replicate": 512,
        "selected_modes_total": int(512 * args.ritz_replicates),
        "weighted_mean_energy": float(endpoints.mean()),
        "weighted_mean_energy_replicate_sd": float(endpoints.std(ddof=1)),
        "weighted_mean_energy_replicate_se": float(
            endpoints.std(ddof=1) / math.sqrt(len(endpoints))
        ),
        "max_abs_diagonal_departure": float(
            np.max(
                np.abs(
                    mean_global["cumulative"].to_numpy(float)
                    - mean_global["expected"].to_numpy(float)
                )
            )
        ),
        "replicate_mean_max_abs_diagonal_departure": float(
            np.mean([row["max_abs_diagonal_departure"] for row in replicate_rows])
        ),
        "signed_endpoint_departure": float(endpoints.mean() - 1.0),
        "selected_fraction_within_ritz_tolerance": float(quality.mean()),
        "selected_fraction_within_ritz_tolerance_sd": float(quality.std(ddof=1)),
        "ritz_candidate_seconds_total": float(
            sum(float(row["ritz_candidate_seconds"]) for row in ritz_rows)
        ),
        "ritz_candidate_matvec_seconds_total": float(
            sum(float(row["ritz_candidate_matvec_seconds"]) for row in ritz_rows)
        ),
        "selected_p95_relative_ritz_residual_mean": float(
            np.mean([row["selected_p95_relative_ritz_residual"] for row in ritz_rows])
        ),
        "selected_p95_spectral_quantile_error_mean": float(
            np.mean([row["selected_p95_spectral_quantile_error"] for row in ritz_rows])
        ),
    }
    for band in BANDS:
        mean_band = (
            modes[modes["band"].eq(band)]
            .groupby("band_rank", as_index=False)
            .agg(
                expected=("within_band_expected", "mean"),
                cumulative=("within_band_cumulative", "mean"),
            )
        )
        band_endpoints = np.asarray(
            [row[f"{band}_mean_energy"] for row in replicate_rows], dtype=float
        )
        cumulative_summary[f"{band}_mean_energy"] = float(band_endpoints.mean())
        cumulative_summary[f"{band}_mean_energy_replicate_sd"] = float(
            band_endpoints.std(ddof=1)
        )
        cumulative_summary[f"{band}_max_abs_departure"] = float(
            np.max(
                np.abs(
                    mean_band["cumulative"].to_numpy(float)
                    - mean_band["expected"].to_numpy(float)
                )
            )
        )
    summary: dict[str, Any] = {
        "dataset_id": spec["dataset_id"],
        "year": spec["year"],
        "date": spec["date"],
        "method": method,
        "native_nll_per_observation": float(fit_row["native_nll_per_observation"]),
        **precision_summary,
        **slq_summary,
        **cumulative_summary,
    }
    if summary["selected_fraction_within_ritz_tolerance"] < args.quality_warning_fraction:
        print(
            "      WARNING: selected Ritz tolerance pass is only "
            f"{summary['selected_fraction_within_ritz_tolerance']:.1%}",
            flush=True,
        )
    print(
        "      cumulative: "
        f"end={summary['weighted_mean_energy']:.4f}; "
        f"max|y-x|={summary['max_abs_diagonal_departure']:.4f}; "
        f"quality pass={summary['selected_fraction_within_ritz_tolerance']:.1%}",
        flush=True,
    )

    slq_curve = add_metadata(slq_curve, spec, method)
    bands = add_metadata(bands, spec, method)
    modes = add_metadata(modes, spec, method)
    paths["root"].mkdir(parents=True, exist_ok=True)
    atomic_csv(paths["slq_curve"], slq_curve)
    atomic_csv(paths["bands"], bands)
    atomic_csv(paths["modes"], modes)
    write_json(paths["summary"], summary)
    paths["complete"].write_text("complete\n", encoding="utf-8")
    del precision
    gc.collect()
    return slq_curve, bands, modes, summary


def shade_bands(ax: plt.Axes, show_labels: bool = True) -> None:
    starts = (0.0, 1.0 / 3.0, 2.0 / 3.0)
    ends = (1.0 / 3.0, 2.0 / 3.0, 1.0)
    for band, start, end in zip(BANDS, starts, ends):
        ax.axvspan(start, end, color=BAND_COLORS[band], alpha=0.48, zorder=0)
        if show_labels:
            ax.text(
                0.5 * (start + end),
                0.985,
                band,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=8.5,
                color="0.35",
            )
    for edge in (1.0 / 3.0, 2.0 / 3.0):
        ax.axvline(edge, color="0.70", linewidth=0.8, zorder=1)


def replicate_mean_global(modes: pd.DataFrame) -> pd.DataFrame:
    return modes.groupby("selection_index", as_index=False).agg(
        x=("scaled_expected", "mean"),
        weight=("spectral_weight", "mean"),
        mean_y=("scaled_cumulative", "mean"),
        se_y=("scaled_cumulative", "sem"),
    )


def replicate_mean_band(modes: pd.DataFrame, band: str) -> pd.DataFrame:
    subset = modes[modes["band"].eq(band)]
    return subset.groupby("band_rank", as_index=False).agg(
        x=("within_band_expected", "mean"),
        mean_y=("within_band_cumulative", "mean"),
        se_y=("within_band_cumulative", "sem"),
    )


def cumulative_axis(
    ax: plt.Axes,
    mode_frames: dict[str, pd.DataFrame],
    summaries: dict[str, dict[str, Any]],
    confidence_z: float,
    monthly: bool = False,
) -> None:
    shade_bands(ax)
    ax.plot([0.0, 1.0], [0.0, 1.0], color="0.25", linestyle="--", linewidth=1.1)
    reference_mode = replicate_mean_global(mode_frames[METHODS[0]])
    x_reference = reference_mode["x"].to_numpy(float)
    weights = reference_mode["weight"].to_numpy(float)
    n_replicates = int(mode_frames[METHODS[0]]["ritz_replicate"].nunique())
    null_sd = np.sqrt(2.0 * np.cumsum(np.square(weights)) / n_replicates)
    ax.fill_between(
        x_reference,
        np.maximum(0.0, x_reference - confidence_z * null_sd),
        x_reference + confidence_z * null_sd,
        color="0.45",
        alpha=0.10,
        linewidth=0,
        label=f"heuristic 95% envelope for {n_replicates}-start mean"
        if not monthly
        else None,
    )
    ymax_values = [1.0]
    for method in METHODS:
        modes = replicate_mean_global(mode_frames[method])
        summary = summaries[method]
        x = modes["x"].to_numpy(float)
        y = modes["mean_y"].to_numpy(float)
        se = modes["se_y"].to_numpy(float)
        ymax_values.append(float(y[-1]))
        label = (
            f"{LABELS[method]}: end={summary['weighted_mean_energy']:.3f}, "
            f"max|dev|={summary['max_abs_diagonal_departure']:.3f}, "
            f"quality={summary['selected_fraction_within_ritz_tolerance']:.1%}"
        )
        ax.plot(
            np.r_[0.0, x],
            np.r_[0.0, y],
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            linewidth=2.0,
            label=label,
        )
        ax.fill_between(
            x,
            np.maximum(0.0, y - se),
            y + se,
            color=COLORS[method],
            alpha=0.12,
            linewidth=0,
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, max(1.05, max(ymax_values) * 1.06))
    ax.set_xlabel("full-spectrum mode fraction (low → high precision)")
    ax.set_ylabel("cumulative standardized residual energy")
    ax.grid(alpha=0.18)
    ax.legend(fontsize=7.5, loc="upper left")


def plot_daily(
    date: str,
    fits: pd.DataFrame,
    mode_frames: dict[str, pd.DataFrame],
    summaries: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.4), constrained_layout=True)
    nll = [
        float(fits.loc[fits["method"].eq(method), "native_nll_per_observation"].iloc[0])
        for method in METHODS
    ]
    axes[0].bar(
        np.arange(len(METHODS)),
        nll,
        color=[COLORS[method] for method in METHODS],
        alpha=0.82,
        width=0.62,
    )
    axes[0].set_xticks(np.arange(len(METHODS)), [LABELS[m] for m in METHODS])
    axes[0].tick_params(axis="x", rotation=10)
    axes[0].set(ylabel="NLL / observation", title="(1) Native full-data Vecchia likelihood")
    for index, value in enumerate(nll):
        axes[0].text(index, value, f" {value:.6f}", ha="center", va="bottom", fontsize=8)
    axes[0].grid(alpha=0.18)

    cumulative_axis(axes[1], mode_frames, summaries, args.confidence_z)
    axes[1].set_title(
        f"(2) Full-data 512-mode cumulative diagnostic; "
        f"mean of {args.ritz_replicates} paired starts"
    )
    fig.suptitle(f"Real {date}: adapted vs fixed Vecchia", fontsize=14)
    output = args.output_root / DAILY_DIR / f"{date}_nll_ritz512x4_cumulative.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_daily_bands(
    date: str,
    mode_frames: dict[str, pd.DataFrame],
    summaries: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.0), constrained_layout=True)
    for axis, band in zip(axes, BANDS):
        axis.set_facecolor(BAND_COLORS[band])
        axis.plot([0.0, 1.0], [0.0, 1.0], color="0.25", linestyle="--", linewidth=1.0)
        reference_band = replicate_mean_band(mode_frames[METHODS[0]], band)
        n_replicates = int(mode_frames[METHODS[0]]["ritz_replicate"].nunique())
        count = len(reference_band)
        null_sd = np.sqrt(2.0 * np.arange(1, count + 1)) / (
            count * math.sqrt(n_replicates)
        )
        axis.fill_between(
            reference_band["x"].to_numpy(float),
            np.maximum(
                0.0,
                reference_band["x"].to_numpy(float) - args.confidence_z * null_sd,
            ),
            reference_band["x"].to_numpy(float) + args.confidence_z * null_sd,
            color="0.45",
            alpha=0.10,
            linewidth=0,
        )
        endpoints = [1.0]
        for method in METHODS:
            part = replicate_mean_band(mode_frames[method], band)
            x = part["x"].to_numpy(float)
            y = part["mean_y"].to_numpy(float)
            se = part["se_y"].to_numpy(float)
            endpoints.append(float(y[-1]))
            axis.plot(
                np.r_[0.0, x],
                np.r_[0.0, y],
                color=COLORS[method],
                linestyle=LINESTYLES[method],
                linewidth=1.9,
                label=(
                    f"{LABELS[method]}: end={summaries[method][f'{band}_mean_energy']:.3f}, "
                    f"max|dev|={summaries[method][f'{band}_max_abs_departure']:.3f}"
                ),
            )
            axis.fill_between(
                x,
                np.maximum(0.0, y - se),
                y + se,
                color=COLORS[method],
                alpha=0.12,
                linewidth=0,
            )
        axis.set(xlim=(0.0, 1.0), ylim=(0.0, max(1.05, max(endpoints) * 1.06)))
        axis.set_title(f"{band}-frequency proxy")
        axis.set_xlabel("within-band mode fraction")
        axis.grid(alpha=0.18)
        axis.legend(fontsize=7.5, loc="upper left")
    axes[0].set_ylabel("within-band cumulative standardized energy")
    fig.suptitle(
        f"Real {date}: low/middle/high band-reset diagnostics; "
        f"mean of {args.ritz_replicates} paired starts",
        fontsize=14,
    )
    output = args.output_root / DAILY_BAND_DIR / f"{date}_ritz512x4_band_reset.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def aggregate(
    specs: list[dict[str, Any]],
    records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    slq_frames: list[pd.DataFrame] = []
    band_frames: list[pd.DataFrame] = []
    mode_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for spec in specs:
        for method in METHODS:
            slq, bands, modes, summary = load_cached_method(
                cache_paths(args.output_root, spec["date"], method)
            )
            slq_frames.append(add_metadata(slq, spec, method))
            band_frames.append(add_metadata(bands, spec, method))
            mode_frames.append(add_metadata(modes, spec, method))
            summaries.append(summary)
    atomic_csv(
        args.output_root / "daily_slq_spectrum_curves.csv",
        pd.concat(slq_frames, ignore_index=True),
    )
    atomic_csv(
        args.output_root / "daily_slq_three_band_boundaries.csv",
        pd.concat(band_frames, ignore_index=True),
    )
    all_modes = pd.concat(mode_frames, ignore_index=True)
    all_summaries = pd.DataFrame(summaries)
    atomic_csv(args.output_root / "daily_ritz512x4_cumulative_curves.csv", all_modes)
    atomic_csv(args.output_root / "daily_nll_ritz512x4_metrics.csv", all_summaries)
    reference.persist_fit_results(records, args.output_root)
    return all_modes, all_summaries


def plot_monthly(
    year: int,
    records: list[dict[str, Any]],
    modes: pd.DataFrame,
    summaries: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    expected_days = len(TEST_DAYS)
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.4), constrained_layout=True)
    fits = pd.DataFrame(records)
    fits = fits[fits["year"].eq(year)]
    nll_means: list[float] = []
    nll_sds: list[float] = []
    for method in METHODS:
        values = fits.loc[
            fits["method"].eq(method), "native_nll_per_observation"
        ].astype(float)
        if len(values) != expected_days:
            raise RuntimeError(f"Incomplete {year} {method} NLL rows")
        nll_means.append(float(values.mean()))
        nll_sds.append(float(values.std(ddof=1)))
    axes[0].bar(
        np.arange(len(METHODS)),
        nll_means,
        yerr=nll_sds,
        capsize=4,
        color=[COLORS[m] for m in METHODS],
        alpha=0.82,
        width=0.62,
    )
    axes[0].set_xticks(np.arange(len(METHODS)), [LABELS[m] for m in METHODS])
    axes[0].tick_params(axis="x", rotation=10)
    axes[0].set(
        ylabel="mean NLL / observation (error: daily SD)",
        title="(1) Native full-data likelihood",
    )
    axes[0].grid(alpha=0.18)

    shade_bands(axes[1])
    axes[1].plot([0.0, 1.0], [0.0, 1.0], color="0.25", linestyle="--", linewidth=1.1)
    ymax = [1.0]
    for method in METHODS:
        subset = modes[modes["year"].eq(year) & modes["method"].eq(method)]
        if subset["dataset_id"].nunique() != expected_days:
            raise RuntimeError(f"Incomplete {year} {method} Ritz curves")
        daily = subset.groupby(
            ["dataset_id", "selection_index"], as_index=False
        ).agg(
            x=("scaled_expected", "mean"),
            daily_y=("scaled_cumulative", "mean"),
        )
        grouped = daily.groupby("selection_index", as_index=False).agg(
            x=("x", "mean"),
            mean_y=("daily_y", "mean"),
            se_y=("daily_y", "sem"),
        )
        metric = summaries[
            summaries["year"].eq(year) & summaries["method"].eq(method)
        ]
        mean_end = float(metric["weighted_mean_energy"].mean())
        mean_maxdev = float(metric["max_abs_diagonal_departure"].mean())
        mean_quality = float(metric["selected_fraction_within_ritz_tolerance"].mean())
        ymax.append(float(grouped["mean_y"].iloc[-1]))
        axes[1].plot(
            np.r_[0.0, grouped["x"].to_numpy(float)],
            np.r_[0.0, grouped["mean_y"].to_numpy(float)],
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            linewidth=2.1,
            label=(
                f"{LABELS[method]}: mean end={mean_end:.3f}, "
                f"mean max|dev|={mean_maxdev:.3f}, quality={mean_quality:.1%}"
            ),
        )
        axes[1].fill_between(
            grouped["x"].to_numpy(float),
            np.maximum(0.0, grouped["mean_y"].to_numpy(float) - grouped["se_y"].to_numpy(float)),
            grouped["mean_y"].to_numpy(float) + grouped["se_y"].to_numpy(float),
            color=COLORS[method],
            alpha=0.12,
            linewidth=0,
        )
    axes[1].set(
        xlim=(0.0, 1.0),
        ylim=(0.0, max(1.05, max(ymax) * 1.06)),
        xlabel="full-spectrum mode fraction (low → high precision)",
        ylabel="five-date mean cumulative standardized energy",
        title=(
            f"(2) Full-data 512-mode selected-date mean; "
            f"{args.ritz_replicates} paired starts/date"
        ),
    )
    axes[1].grid(alpha=0.18)
    axes[1].legend(fontsize=7.5, loc="upper left")
    fig.suptitle(
        f"Real {year} July: selected days 3, 5, 7, 12, 15", fontsize=14
    )
    fig.savefig(
        args.output_root / f"{year}_07_selected5_average_nll_ritz512x4_cumulative.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_monthly_bands(
    year: int,
    modes: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    expected_days = len(TEST_DAYS)
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.0), constrained_layout=True)
    for axis, band in zip(axes, BANDS):
        axis.set_facecolor(BAND_COLORS[band])
        axis.plot([0.0, 1.0], [0.0, 1.0], color="0.25", linestyle="--", linewidth=1.0)
        ymax = [1.0]
        for method in METHODS:
            subset = modes[
                modes["year"].eq(year)
                & modes["method"].eq(method)
                & modes["band"].eq(band)
            ]
            if subset["dataset_id"].nunique() != expected_days:
                raise RuntimeError(f"Incomplete {year} {method} {band} Ritz curves")
            daily = subset.groupby(
                ["dataset_id", "band_rank"], as_index=False
            ).agg(
                x=("within_band_expected", "mean"),
                daily_y=("within_band_cumulative", "mean"),
            )
            grouped = daily.groupby("band_rank", as_index=False).agg(
                x=("x", "mean"),
                mean_y=("daily_y", "mean"),
                se_y=("daily_y", "sem"),
            )
            ymax.append(float(grouped["mean_y"].iloc[-1]))
            axis.plot(
                np.r_[0.0, grouped["x"].to_numpy(float)],
                np.r_[0.0, grouped["mean_y"].to_numpy(float)],
                color=COLORS[method],
                linestyle=LINESTYLES[method],
                linewidth=2.0,
                label=f"{LABELS[method]}: end={grouped['mean_y'].iloc[-1]:.3f}",
            )
            axis.fill_between(
                grouped["x"].to_numpy(float),
                np.maximum(
                    0.0,
                    grouped["mean_y"].to_numpy(float)
                    - grouped["se_y"].to_numpy(float),
                ),
                grouped["mean_y"].to_numpy(float)
                + grouped["se_y"].to_numpy(float),
                color=COLORS[method],
                alpha=0.12,
                linewidth=0,
            )
        axis.set(
            xlim=(0.0, 1.0),
            ylim=(0.0, max(1.05, max(ymax) * 1.06)),
            xlabel="within-band mode fraction",
            title=f"{band}-frequency proxy",
        )
        axis.grid(alpha=0.18)
        axis.legend(fontsize=7.5, loc="upper left")
    axes[0].set_ylabel("five-date mean within-band cumulative energy")
    fig.suptitle(
        f"Real {year} July: selected-date low/middle/high diagnostics; "
        f"{args.ritz_replicates} paired starts/date",
        fontsize=14,
    )
    fig.savefig(
        args.output_root / f"{year}_07_selected5_average_ritz512x4_bands.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def write_run_config(args: argparse.Namespace, specs: list[dict[str, Any]]) -> None:
    path = args.output_root / "run_config_nll_ritz512x4_cumulative.json"
    signature = {
        "dates": [spec["date"] for spec in specs],
        "methods": list(METHODS),
        "smooth": args.smooth,
        "target_chunk_size": args.target_chunk_size,
        "lbfgs_lr": args.lbfgs_lr,
        "lbfgs_steps": args.lbfgs_steps,
        "lbfgs_eval": args.lbfgs_eval,
        "lbfgs_history": args.lbfgs_history,
        "grad_tol": args.grad_tol,
        "slq_probes": args.slq_probes,
        "slq_steps": args.slq_steps,
        "ritz_candidate_steps": args.ritz_candidate_steps,
        "ritz_replicates": args.ritz_replicates,
        "band_mode_counts": list(args.band_mode_counts),
        "spectrum_grid": args.spectrum_grid,
        "random_seed": args.random_seed,
        "ritz_relative_residual_tolerance": args.ritz_relative_residual_tolerance,
        "precision_identity_tolerance": args.precision_identity_tolerance,
        "coefficient_drop_tolerance": args.coefficient_drop_tolerance,
        "confidence_z": args.confidence_z,
    }
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("configuration_signature") != signature:
            raise ValueError(f"Configuration mismatch in {path}; use a new output root")
        created = previous.get("created", datetime.now().isoformat(timespec="seconds"))
    else:
        created = datetime.now().isoformat(timespec="seconds")
    write_json(
        path,
        {
            "created": created,
            "last_started": datetime.now().isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "gpu": torch.cuda.get_device_name(0),
            "configuration_signature": signature,
            "n_test_dates": len(specs),
            "selected_days_each_year": list(TEST_DAYS),
            "comparison_axes": [
                "native full-data Vecchia NLL per observation",
                "full-data SLQ plus four-start mean cumulative 512-mode Ritz diagnostic",
            ],
            "omitted": "400x8 dense covariance full eigendecomposition",
            "frequency_proxy_convention": (
                "Ascending precision eigenvalue corresponds to descending implied "
                "covariance eigenvalue and is interpreted as low-to-high spectral "
                "frequency proxy; it is not exact physical Fourier frequency."
            ),
            "cumulative_weighting": (
                "Each SLQ mode-count third has total weight 1/3; its selected Ritz "
                "modes share that weight equally. The endpoint is not renormalized."
            ),
            "paired_randomness": (
                "Adapted and fixed receive identical Rademacher SLQ probes and "
                "identical Lanczos starting vectors for each date/replicate."
            ),
            "important_limitation": (
                "The 512 objects are implicit random-start Rayleigh-Ritz vectors, "
                "not 512 claimed exact eigenvectors. Tail residual quality is saved."
            ),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-data-root", type=Path, default=Path("/home/jl2815/tco/data"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/home/jl2815/tco/exercise_output/summer/"
            "vecchia_real10_selected5x2_adapted_fixed_nll_ritz512x4_cumulative_lag643"
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
    parser.add_argument("--suppress-fit-prints", action="store_true")
    parser.add_argument("--slq-probes", type=int, default=8)
    parser.add_argument("--slq-steps", type=int, default=256)
    parser.add_argument("--ritz-candidate-steps", type=int, default=1536)
    parser.add_argument("--ritz-replicates", type=int, default=4)
    parser.add_argument(
        "--band-mode-counts",
        type=engine.parse_band_counts,
        default=engine.parse_band_counts("170,170,172"),
    )
    parser.add_argument("--spectrum-grid", type=int, default=400)
    parser.add_argument("--random-seed", type=int, default=20260907)
    parser.add_argument("--ritz-relative-residual-tolerance", type=float, default=0.05)
    parser.add_argument("--quality-warning-fraction", type=float, default=0.85)
    parser.add_argument("--precision-identity-tolerance", type=float, default=1e-8)
    parser.add_argument("--coefficient-drop-tolerance", type=float, default=0.0)
    parser.add_argument("--confidence-z", type=float, default=1.96)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.slq_probes < 2 or args.slq_steps < 16:
        raise ValueError("Use at least 2 SLQ probes and 16 SLQ steps")
    if args.ritz_candidate_steps < 512:
        raise ValueError("--ritz-candidate-steps must be at least 512")
    if args.ritz_replicates < 2:
        raise ValueError("--ritz-replicates must be at least 2")
    if args.spectrum_grid < 32:
        raise ValueError("--spectrum-grid must be at least 32")
    if not 0.0 < args.quality_warning_fraction <= 1.0:
        raise ValueError("--quality-warning-fraction must be in (0,1]")
    if args.ritz_relative_residual_tolerance <= 0.0:
        raise ValueError("--ritz-relative-residual-tolerance must be positive")
    if args.precision_identity_tolerance <= 0.0:
        raise ValueError("--precision-identity-tolerance must be positive")
    if args.coefficient_drop_tolerance < 0.0:
        raise ValueError("--coefficient-drop-tolerance must be nonnegative")
    if args.confidence_z <= 0.0:
        raise ValueError("--confidence-z must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for fitting and full precision construction")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / DAILY_DIR).mkdir(parents=True, exist_ok=True)
    (args.output_root / DAILY_BAND_DIR).mkdir(parents=True, exist_ok=True)
    specs = selected_date_specs()
    write_run_config(args, specs)
    records = reference.load_fit_results(args.output_root)
    selected_ids = {spec["dataset_id"] for spec in specs}
    unexpected = [
        row
        for row in records
        if str(row["dataset_id"]) not in selected_ids or str(row["method"]) not in METHODS
    ]
    if unexpected:
        raise ValueError("Fit checkpoint contains rows outside this run")

    started = time.perf_counter()
    for position, spec in enumerate(specs, start=1):
        print(f"\n[{position}/{len(specs)}] {spec['date']}", flush=True)
        asset, fits = reference.load_or_fit_day(spec, records, args)
        mode_frames: dict[str, pd.DataFrame] = {}
        summaries: dict[str, dict[str, Any]] = {}
        for method in METHODS:
            fit_rows = fits[fits["method"].eq(method)]
            if len(fit_rows) != 1:
                raise RuntimeError(f"Expected one {method} fit for {spec['date']}")
            _, _, modes, summary = run_method(
                spec, method, asset, fit_rows.iloc[0], args
            )
            mode_frames[method] = modes
            summaries[method] = summary
        plot_daily(spec["date"], fits, mode_frames, summaries, args)
        plot_daily_bands(spec["date"], mode_frames, summaries, args)
        reference.persist_fit_results(records, args.output_root)
        del asset, fits, mode_frames, summaries
        gc.collect()
        torch.cuda.empty_cache()

    modes, summaries = aggregate(specs, records, args)
    reference.plot_native_nll(records, args.output_root)
    for year in (2024, 2025):
        plot_monthly(year, records, modes, summaries, args)
        plot_monthly_bands(year, modes, args)
    write_json(
        args.output_root / "RUN_COMPLETE.json",
        {
            "completed": datetime.now().isoformat(timespec="seconds"),
            "elapsed_seconds": time.perf_counter() - started,
            "n_dates": len(specs),
            "n_methods": len(METHODS),
            "selected_modes_per_replicate": 512,
            "ritz_replicates_per_date_method": int(args.ritz_replicates),
            "dense_full_eigen_omitted": True,
        },
    )
    print(f"Complete: outputs in {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
