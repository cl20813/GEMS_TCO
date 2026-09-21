#!/usr/bin/env python3
"""Replot cached 512-mode Ritz diagnostics as cumulative y=x curves.

This is a post-processing-only script.  It does not refit a Vecchia model and
does not rerun SLQ or Lanczos.  It reads either the aggregate
``daily_selected_512_ritz_modes.csv`` or the per-day cache files written by
``vecchia_real59_adapted_fixed_threeway_slq512_lag643.py``.

Each of the low/middle/high spectral bands represents one third of the full
SLQ-estimated precision spectrum.  Consequently, the 170/170/172 selected
Ritz modes receive weights 1/(3*170), 1/(3*170), and 1/(3*172), respectively.
For standardized Ritz energy e_j, the plotted curve is

    x_k = sum_{j<=k} w_j,    y_k = sum_{j<=k} w_j e_j.

Under spectral calibration the expectation is y_k=x_k.  The curve is never
renormalized by its endpoint, because doing that would hide overall variance
miscalibration.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = ("adapted", "fixed")
BANDS = ("low", "middle", "high")
COLORS = {"adapted": "#1f77b4", "fixed": "#e41a1c"}
LINESTYLES = {"adapted": "-", "fixed": "-."}
LABELS = {"adapted": "adapted corridor", "fixed": "fixed center"}
BAND_COLORS = {"low": "#DCEEFF", "middle": "#E9E5FF", "high": "#FFE4DC"}


def parse_dates(text: str) -> tuple[str, ...]:
    dates = tuple(part.strip() for part in text.split(",") if part.strip())
    if not dates:
        raise argparse.ArgumentTypeError("at least one date is required")
    return dates


def load_modes(input_root: Path, dates: tuple[str, ...]) -> pd.DataFrame:
    aggregate = input_root / "daily_selected_512_ritz_modes.csv"
    if aggregate.is_file():
        frame = pd.read_csv(aggregate)
        required_metadata = {"date", "method"}
        if not required_metadata.issubset(frame.columns):
            raise ValueError(f"{aggregate} lacks {sorted(required_metadata)}")
        frame["date"] = frame["date"].astype(str)
        return frame[frame["date"].isin(dates)].copy()

    pieces: list[pd.DataFrame] = []
    for date in dates:
        for method in METHODS:
            path = (
                input_root
                / ".diagnostic_cache"
                / date
                / method
                / "selected_512_ritz_modes.csv"
            )
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing {path}. Wait until that date/method cache is complete, "
                    "or point --input-root to a downloaded result directory."
                )
            part = pd.read_csv(path)
            part["date"] = date
            part["method"] = method
            pieces.append(part)
    return pd.concat(pieces, ignore_index=True)


def cumulative_diagnostic(modes: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    required = {
        "selection_index",
        "band",
        "precision_ritz_value",
        "standardized_residual_energy",
    }
    missing = required.difference(modes.columns)
    if missing:
        raise ValueError(f"Ritz table lacks columns: {sorted(missing)}")
    out = modes.sort_values("selection_index").reset_index(drop=True).copy()
    if len(out) != 512 or out["selection_index"].nunique() != 512:
        raise ValueError(f"Expected 512 distinct selected modes, found {len(out)}")
    if not np.array_equal(out["selection_index"].to_numpy(int), np.arange(1, 513)):
        raise ValueError("selection_index must run consecutively from 1 through 512")
    energy = out["standardized_residual_energy"].to_numpy(float)
    if not np.all(np.isfinite(energy)) or np.any(energy < 0.0):
        raise ValueError("standardized Ritz energies must be finite and nonnegative")

    band_sizes = out.groupby("band").size().to_dict()
    if tuple(int(band_sizes.get(band, 0)) for band in BANDS) != (170, 170, 172):
        raise ValueError(f"Expected low/middle/high counts 170/170/172; got {band_sizes}")
    weights = np.asarray(
        [1.0 / (3.0 * float(band_sizes[band])) for band in out["band"]],
        dtype=np.float64,
    )
    expected = np.cumsum(weights)
    cumulative = np.cumsum(weights * energy)
    expected[-1] = 1.0
    out["spectral_weight"] = weights
    out["scaled_expected"] = expected
    out["scaled_cumulative"] = cumulative
    out["diagonal_departure"] = cumulative - expected

    metrics: dict[str, float] = {
        "weighted_mean_energy": float(cumulative[-1]),
        "max_abs_diagonal_departure": float(np.max(np.abs(cumulative - expected))),
        "signed_endpoint_departure": float(cumulative[-1] - 1.0),
    }
    for band in BANDS:
        part = out[out["band"].eq(band)]
        metrics[f"{band}_mean_energy"] = float(
            part["standardized_residual_energy"].mean()
        )
    if "ritz_tail_residual_relative" in out.columns:
        quality = out["ritz_tail_residual_relative"].to_numpy(float)
        metrics["ritz_relative_residual_median"] = float(np.nanmedian(quality))
        metrics["ritz_relative_residual_p95"] = float(np.nanquantile(quality, 0.95))
    return out, metrics


def shade_bands(ax: plt.Axes, labels: bool = True) -> None:
    starts = (0.0, 1.0 / 3.0, 2.0 / 3.0)
    ends = (1.0 / 3.0, 2.0 / 3.0, 1.0)
    for band, start, end in zip(BANDS, starts, ends):
        ax.axvspan(start, end, color=BAND_COLORS[band], alpha=0.55, zorder=0)
        if labels:
            ax.text(
                0.5 * (start + end),
                0.985,
                band,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=9,
                color="0.35",
            )
    for edge in (1.0 / 3.0, 2.0 / 3.0):
        ax.axvline(edge, color="0.72", linewidth=0.8, zorder=1)


def plot_combined(
    curves: dict[tuple[str, str], pd.DataFrame],
    metrics: pd.DataFrame,
    dates: tuple[str, ...],
    output: Path,
) -> None:
    fig, axes = plt.subplots(
        1,
        len(dates),
        figsize=(5.4 * len(dates), 5.2),
        squeeze=False,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    ymax = max(
        1.05,
        max(float(frame["scaled_cumulative"].iloc[-1]) for frame in curves.values())
        * 1.06,
    )
    for column, date in enumerate(dates):
        ax = axes[0, column]
        shade_bands(ax)
        ax.plot([0.0, 1.0], [0.0, 1.0], color="0.25", linestyle="--", linewidth=1.1)
        for method in METHODS:
            curve = curves[(date, method)]
            row = metrics[
                metrics["date"].eq(date) & metrics["method"].eq(method)
            ].iloc[0]
            ax.plot(
                np.r_[0.0, curve["scaled_expected"].to_numpy(float)],
                np.r_[0.0, curve["scaled_cumulative"].to_numpy(float)],
                color=COLORS[method],
                linestyle=LINESTYLES[method],
                linewidth=2.0,
                label=(
                    f"{LABELS[method]}: end={row['weighted_mean_energy']:.3f}, "
                    f"max|dev|={row['max_abs_diagonal_departure']:.3f}"
                ),
            )
        ax.set(xlim=(0.0, 1.0), ylim=(0.0, ymax), title=date)
        ax.set_xlabel("SLQ spectral-mode fraction (low → high precision)")
        if column == 0:
            ax.set_ylabel("cumulative standardized residual energy")
        ax.grid(alpha=0.18)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(
        "Full-data 512 Ritz-mode cumulative diagnostic (expected: y=x)", fontsize=14
    )
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_band_reset(
    modes: pd.DataFrame,
    dates: tuple[str, ...],
    output: Path,
) -> None:
    fig, axes = plt.subplots(
        len(dates),
        3,
        figsize=(14.5, 4.3 * len(dates)),
        squeeze=False,
        constrained_layout=True,
    )
    all_endpoints = []
    prepared: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}
    for date in dates:
        for method in METHODS:
            selected = modes[modes["date"].eq(date) & modes["method"].eq(method)]
            for band in BANDS:
                part = selected[selected["band"].eq(band)].sort_values("band_rank")
                energy = part["standardized_residual_energy"].to_numpy(float)
                x = np.arange(1, len(energy) + 1, dtype=float) / len(energy)
                y = np.cumsum(energy) / len(energy)
                prepared[(date, method, band)] = (x, y)
                all_endpoints.append(float(y[-1]))
    ymax = max(1.05, max(all_endpoints) * 1.06)
    for row, date in enumerate(dates):
        for column, band in enumerate(BANDS):
            ax = axes[row, column]
            ax.set_facecolor(BAND_COLORS[band])
            ax.plot([0.0, 1.0], [0.0, 1.0], color="0.25", linestyle="--", linewidth=1.0)
            for method in METHODS:
                x, y = prepared[(date, method, band)]
                ax.plot(
                    np.r_[0.0, x],
                    np.r_[0.0, y],
                    color=COLORS[method],
                    linestyle=LINESTYLES[method],
                    linewidth=1.9,
                    label=f"{LABELS[method]}: end={y[-1]:.3f}",
                )
            ax.set(xlim=(0.0, 1.0), ylim=(0.0, ymax), title=f"{date} — {band}")
            if row == len(dates) - 1:
                ax.set_xlabel("within-band mode fraction")
            if column == 0:
                ax.set_ylabel("within-band cumulative energy")
            ax.grid(alpha=0.18)
            ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Band-reset Ritz diagnostics (each panel expected: y=x)", fontsize=14)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument(
        "--dates",
        type=parse_dates,
        default=parse_dates("2024-07-01,2024-07-02,2024-07-03"),
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    dates = tuple(args.dates)
    output_dir = args.output_dir or (args.input_root / "ritz512_cumulative_replots")
    output_dir.mkdir(parents=True, exist_ok=True)

    modes = load_modes(args.input_root, dates)
    modes["date"] = modes["date"].astype(str)
    curves: dict[tuple[str, str], pd.DataFrame] = {}
    metric_rows: list[dict[str, float | str]] = []
    curve_rows: list[pd.DataFrame] = []
    for date in dates:
        for method in METHODS:
            subset = modes[modes["date"].eq(date) & modes["method"].eq(method)]
            curve, metric = cumulative_diagnostic(subset)
            curve["date"] = date
            curve["method"] = method
            leading = ["date", "method"]
            curve = curve[leading + [column for column in curve if column not in leading]]
            curves[(date, method)] = curve
            curve_rows.append(curve)
            metric_rows.append({"date": date, "method": method, **metric})
    metrics = pd.DataFrame(metric_rows)
    pd.concat(curve_rows, ignore_index=True).to_csv(
        output_dir / "ritz512_cumulative_curves_2024-07-01_to_03.csv", index=False
    )
    metrics.to_csv(
        output_dir / "ritz512_cumulative_metrics_2024-07-01_to_03.csv", index=False
    )
    plot_combined(
        curves,
        metrics,
        dates,
        output_dir / "ritz512_cumulative_2024-07-01_to_03.png",
    )
    plot_band_reset(
        modes,
        dates,
        output_dir / "ritz512_band_reset_2024-07-01_to_03.png",
    )
    print(f"Wrote cumulative plots and CSV files to {output_dir}")


if __name__ == "__main__":
    main()
