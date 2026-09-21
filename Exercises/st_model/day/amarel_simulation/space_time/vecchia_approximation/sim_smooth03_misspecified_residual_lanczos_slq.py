#!/usr/bin/env python3
"""Diagnose an assumed Matern smoothness 0.3 on smoothness-0.5 ST data.

The response, covariance parameters, mean, and lag-6/4/3 Vecchia graph are the
same as the fixed-parameter simulation validation.  Only the assumed Matern
smoothness changes from 0.5 to 0.3.  The arbitrary-smoothness covariance is
evaluated by ``GEMS_TCO.vecchia_st_spline`` through the existing spline model.

The already-computed smoothness-0.5 null result is reused, while the 0.3
precision, residual-started Lanczos curve, and random-probe SLQ spectrum are
computed here.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sim_true_vs_misspecified_residual_lanczos_slq as base
import vecchia_local_20240703_adapted_fixed_residual_lanczos_slq as diagnostic


METHOD = "wrong_smooth_03"
LABEL = "assumed smoothness 0.3"
REFERENCE_ROOT = base.REPO / (
    "outputs/summer_26/"
    "sim_20240713_true_vs_misspecified_residual_lanczos512_slq8x192"
)
DEFAULT_OUTPUT_ROOT = base.REPO / (
    "outputs/summer_26/"
    "sim_20240713_smooth05_data_smooth03_vecchia_lanczos512_slq8x192"
)
BANDS = (
    ("low", 0.0, 1.0 / 3.0, "#DCEEFF"),
    ("middle", 1.0 / 3.0, 2.0 / 3.0, "#E8E4FF"),
    ("high", 2.0 / 3.0, 1.0, "#FFE8E0"),
)


def parser() -> argparse.ArgumentParser:
    out = base.parser()
    out.description = __doc__
    out.set_defaults(output_root=DEFAULT_OUTPUT_ROOT)
    return out


def assumed_smooth03(truth: dict[str, float]) -> dict[str, float]:
    keys = (
        "smooth", "sigmasq", "range_lat", "range_lon", "range_time",
        "advec_lat", "advec_lon", "nugget",
    )
    assumed = {key: float(truth[key]) for key in keys}
    assumed["smooth"] = 0.3
    return assumed


def load_reference(name: str, steps: int) -> pd.DataFrame:
    path = REFERENCE_ROOT / name
    if not path.is_file():
        raise FileNotFoundError(f"Required smoothness-0.5 reference is missing: {path}")
    frame = pd.read_csv(path)
    if "residual_lanczos_steps" in frame.columns:
        frame = frame[frame["residual_lanczos_steps"].le(int(steps))]
    frame = frame[frame["method"].eq("true")].copy()
    if frame.empty:
        raise RuntimeError(f"No true-model rows found in {path}")
    return frame


def final_curve(frame: pd.DataFrame, method: str, steps: int) -> tuple[np.ndarray, np.ndarray]:
    part = frame[
        frame["method"].eq(method)
        & frame["residual_lanczos_steps"].eq(int(steps))
    ].sort_values("estimated_mode_fraction")
    x = part["estimated_mode_fraction"].to_numpy(float)
    y = part["cumulative_energy_per_n"].to_numpy(float)
    if x.size == 0:
        raise ValueError(f"No final curve for method={method}, steps={steps}")
    unique_x, indices = np.unique(x, return_index=True)
    return unique_x, y[indices]


def reset_segment(
    x: np.ndarray,
    y: np.ndarray,
    lower: float,
    upper: float,
) -> tuple[np.ndarray, np.ndarray]:
    inside = (x > lower) & (x < upper)
    segment_x = np.concatenate(([lower], x[inside], [upper]))
    segment_y = np.concatenate((
        [np.interp(lower, x, y)], y[inside], [np.interp(upper, x, y)]
    ))
    width = upper - lower
    return (segment_x - lower) / width, (segment_y - segment_y[0]) / width


def ordered_thirds(frame: pd.DataFrame, method: str) -> np.ndarray:
    order = pd.CategoricalDtype(["low", "middle", "high"], ordered=True)
    part = frame[frame["method"].eq(method)].copy()
    part["frequency_third"] = part["frequency_third"].astype(order)
    values = part.sort_values("frequency_third")["energy_per_mode"].to_numpy(float)
    if values.size != 3:
        raise ValueError(f"Expected three frequency thirds for {method}, got {values.size}")
    return values


def plot_localization(
    curves: pd.DataFrame,
    thirds: pd.DataFrame,
    args: argparse.Namespace,
) -> Path:
    steps = int(args.residual_lanczos_steps)
    colors = {"true": "#169873", METHOD: "#D94841"}
    labels = {"true": "true smoothness 0.5", METHOD: LABEL}
    line_data = {
        method: final_curve(curves, method, steps) for method in ("true", METHOD)
    }
    ratio_data = {
        method: ordered_thirds(thirds, method) for method in ("true", METHOD)
    }
    ceiling = max(1.1, 1.08 * float(np.max(np.r_[ratio_data["true"], ratio_data[METHOD]])))
    fig, axes = plt.subplots(1, 5, figsize=(23.0, 5.2), constrained_layout=True)

    ax = axes[0]
    for _, lower, upper, background in BANDS:
        ax.axvspan(lower, upper, color=background, alpha=0.55, zorder=0)
    ax.plot([0, 1], [0, 1], "--", color="0.25", linewidth=1.1, label="null: y=x")
    for method in ("true", METHOD):
        ax.plot(*line_data[method], color=colors[method], linewidth=2.2, label=labels[method])
    ax.set(
        xlim=(0, 1), xlabel="full-spectrum mode fraction",
        ylabel="cumulative standardized residual energy / n",
        title="Global cumulative curve",
    )
    ax.grid(alpha=0.18)
    ax.legend(fontsize=8)

    for column, (band, lower, upper, background) in enumerate(BANDS, start=1):
        ax = axes[column]
        ax.set_facecolor(background)
        ax.plot([0, 1], [0, 1], "--", color="0.25", linewidth=1.1)
        for method in ("true", METHOD):
            xb, yb = reset_segment(*line_data[method], lower, upper)
            ax.plot(xb, yb, color=colors[method], linewidth=2.2, label=labels[method])
        ax.set(
            xlim=(0, 1), ylim=(0, ceiling), xlabel="within-band mode fraction",
            title=f"{band} band; smooth 0.3 R={ratio_data[METHOD][column - 1]:.3f}",
        )
        if column == 1:
            ax.set_ylabel("within-band cumulative energy / mode")
        ax.grid(alpha=0.18)

    ax = axes[4]
    positions = np.arange(3)
    width = 0.34
    ax.axhline(1.0, color="0.25", linestyle="--", linewidth=1.1)
    ax.bar(
        positions - width / 2, ratio_data["true"], width,
        color=colors["true"], alpha=0.88, label=labels["true"],
    )
    ax.bar(
        positions + width / 2, ratio_data[METHOD], width,
        color=colors[METHOD], alpha=0.88, label=labels[METHOD],
    )
    for index, value in enumerate(ratio_data[METHOD]):
        ax.text(index + width / 2, value + 0.02 * ceiling, f"{value:.3f}", ha="center", fontsize=9)
    ax.set(
        xticks=positions, xticklabels=["low", "middle", "high"],
        ylim=(0, ceiling), xlabel="precision-spectrum third",
        ylabel="band energy ratio R", title="Direct localization",
    )
    ax.grid(alpha=0.18, axis="y")
    ax.legend(fontsize=8)

    fig.suptitle(
        "Smoothness misspecification: data Matern nu=0.5, assumed Vecchia Matern nu=0.3\n"
        "Fixed true covariance parameters; spline Matern; residual-Lanczos 512 + SLQ 8x192",
        fontsize=15,
    )
    output = args.output_root / f"{base.SIM_DATE}_smooth05_vs_smooth03_band_localization.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    args = parser().parse_args()
    if int(args.residual_lanczos_steps) != 512:
        raise ValueError("This comparison reuses a Lanczos-512 true reference; use 512 steps")
    if int(args.slq_probes) != 8 or int(args.slq_steps) != 192:
        raise ValueError("This comparison reuses an SLQ 8x192 true reference")
    args.output_root.mkdir(parents=True, exist_ok=True)
    diagnostic.DATE = base.SIM_DATE
    started = time.perf_counter()

    truth = base.load_truth(args.data_root)
    asset = base.load_asset(args)
    assumed = assumed_smooth03(truth)
    print(f"{METHOD}: building spline-Matern smoothness-0.3 precision", flush=True)
    precision, build_summary = base.build_precision(METHOD, assumed, truth, asset, args)
    slq, curves, bands, result = diagnostic.evaluate_method(METHOD, precision, args)

    true_curves = load_reference("residual_lanczos_cumulative_curves.csv", 512)
    true_bands = load_reference("frequency_band_energy_ratios.csv", 512)
    true_thirds = load_reference("frequency_third_energy_ratios.csv", 512)
    true_slq = load_reference("slq_spectrum.csv", 512)
    curve_frame = pd.concat([true_curves, curves], ignore_index=True)
    band_frame = pd.concat([true_bands, bands], ignore_index=True)
    third_frame = pd.concat([true_thirds, result["thirds"]], ignore_index=True)
    slq_frame = pd.concat([true_slq, slq.assign(date=base.SIM_DATE, method=METHOD)], ignore_index=True)

    base.atomic_csv(args.output_root / "residual_lanczos_cumulative_curves.csv", curve_frame)
    base.atomic_csv(args.output_root / "frequency_band_energy_ratios.csv", band_frame)
    base.atomic_csv(args.output_root / "frequency_third_energy_ratios.csv", third_frame)
    base.atomic_csv(args.output_root / "slq_spectrum.csv", slq_frame)
    plot = plot_localization(curve_frame, third_frame, args)
    elapsed = time.perf_counter() - started
    ratios = ordered_thirds(third_frame, METHOD)
    base.write_json(
        args.output_root / "RUN_COMPLETE.json",
        {
            "completed": datetime.now().isoformat(timespec="seconds"),
            "date": base.SIM_DATE,
            "data_generating_smoothness": float(truth["smooth"]),
            "assumed_vecchia_smoothness": 0.3,
            "other_assumed_parameters": assumed,
            "mean_handling": "known DGP mean removed; no parameter fitting",
            "spline_engine": "GEMS_TCO.vecchia_st_spline",
            "residual_lanczos_steps": 512,
            "slq_probes": 8,
            "slq_steps": 192,
            "true_reference_reused_from": str(REFERENCE_ROOT),
            "smooth03_low_middle_high": ratios,
            "smooth03_summary": {**build_summary, **result["summary"]},
            "plot": str(plot),
            "total_wall_seconds": elapsed,
            "individual_eigenvectors_computed": False,
        },
    )
    print(
        "smoothness 0.3 thirds: "
        + ", ".join(f"{name}={value:.3f}" for (name, *_), value in zip(BANDS, ratios)),
        flush=True,
    )
    print(f"Complete in {elapsed:.2f}s: {args.output_root}", flush=True)
    del precision
    gc.collect()


if __name__ == "__main__":
    main()
