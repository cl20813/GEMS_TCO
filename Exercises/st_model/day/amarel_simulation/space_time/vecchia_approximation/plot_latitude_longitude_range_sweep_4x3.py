#!/usr/bin/env python3
"""Plot four latitude/longitude range checks as band-reset diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path("/Users/joonwonlee/Documents/GEMS_TCO-1")
LAT2_ROOT = REPO / (
    "outputs/summer_26/"
    "sim_20240713_latitude_short_long_smooth10_lanczos512_slq8x192"
)
SWEEP_ROOT = REPO / (
    "outputs/summer_26/"
    "sim_20240713_latitude_longitude_range_sweep_lanczos512_slq8x192"
)
OUTPUT_ROOT = REPO / "outputs/summer_26/sim_20240713_spectral_localization_validation"
OUTPUT = OUTPUT_ROOT / "2024-07-13_latitude_longitude_range_sweep_4x3.png"
STEPS = 512
BANDS = (
    ("Low precision-spectrum third", "low", 0.0, 1.0 / 3.0, "#DCEEFF"),
    ("Middle precision-spectrum third", "middle", 1.0 / 3.0, 2.0 / 3.0, "#E8E4FF"),
    ("High precision-spectrum third", "high", 2.0 / 3.0, 1.0, "#FFE8E0"),
)
CASES = (
    (LAT2_ROOT, "lat_range_2x", "latitude range 2x", "Latitude range 0.20 -> 0.40; longitude 0.30 fixed"),
    (SWEEP_ROOT, "lat_range_3x", "latitude range 3x", "Latitude range 0.20 -> 0.60; longitude 0.30 fixed"),
    (SWEEP_ROOT, "lon_range_half", "longitude range 0.5x", "Longitude range 0.30 -> 0.15; latitude 0.20 fixed"),
    (SWEEP_ROOT, "lon_range_2x", "longitude range 2x", "Longitude range 0.30 -> 0.60; latitude 0.20 fixed"),
)


def final_curve(frame: pd.DataFrame, method: str) -> tuple[np.ndarray, np.ndarray]:
    part = frame[
        frame["method"].eq(method)
        & frame["residual_lanczos_steps"].eq(STEPS)
    ].sort_values("estimated_mode_fraction")
    if part.empty:
        raise ValueError(f"No Lanczos-{STEPS} curve for {method}")
    x = part["estimated_mode_fraction"].to_numpy(float)
    y = part["cumulative_energy_per_n"].to_numpy(float)
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


def ratios(frame: pd.DataFrame, method: str) -> dict[str, float]:
    part = frame[frame["method"].eq(method)].set_index("frequency_third")
    return {
        band: float(part.loc[band, "energy_per_mode"])
        for band in ("low", "middle", "high")
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    reference_curves = pd.read_csv(LAT2_ROOT / "residual_lanczos_cumulative_curves.csv")
    reference_thirds = pd.read_csv(LAT2_ROOT / "frequency_third_energy_ratios.csv")
    true_curve = final_curve(reference_curves, "true")
    true_ratios = ratios(reference_thirds, "true")
    figure = plt.figure(figsize=(18.0, 19.5), constrained_layout=True)
    subfigures = figure.subfigures(nrows=len(CASES), ncols=1)
    true_color = "#169873"
    wrong_color = "#D94841"

    for row, (subfigure, (root, method, label, row_title)) in enumerate(zip(subfigures, CASES)):
        curves = pd.read_csv(root / "residual_lanczos_cumulative_curves.csv")
        thirds = pd.read_csv(root / "frequency_third_energy_ratios.csv")
        wrong_curve = final_curve(curves, method)
        wrong_ratios = ratios(thirds, method)
        segments = {}
        maximum = 1.0
        for _, band, lower, upper, _ in BANDS:
            segments[(band, "true")] = reset_segment(*true_curve, lower, upper)
            segments[(band, "wrong")] = reset_segment(*wrong_curve, lower, upper)
            maximum = max(
                maximum,
                float(np.max(segments[(band, "true")][1])),
                float(np.max(segments[(band, "wrong")][1])),
            )
        ceiling = max(1.08, 1.05 * maximum)

        axes = subfigure.subplots(1, 3)
        subfigure.suptitle(row_title, fontsize=14, fontweight="bold")
        for column, (ax, (title, band, _, _, background)) in enumerate(zip(axes, BANDS)):
            ax.set_facecolor(background)
            ax.plot([0, 1], [0, 1], "--", color="0.25", linewidth=1.15, label="null: y=x")
            ax.plot(
                *segments[(band, "true")], color=true_color, linewidth=2.3,
                label=f"true covariance: R={true_ratios[band]:.3f}",
            )
            ax.plot(
                *segments[(band, "wrong")], color=wrong_color, linewidth=2.3,
                label=f"{label}: R={wrong_ratios[band]:.3f}",
            )
            ax.scatter(
                [1, 1], [true_ratios[band], wrong_ratios[band]],
                color=[true_color, wrong_color], s=26, zorder=4,
            )
            ax.text(
                0.97, 0.05,
                f"R = {wrong_ratios[band]:.3f}\nR-1 = {wrong_ratios[band] - 1:+.3f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.78, "edgecolor": "0.75"},
            )
            ax.set(
                xlim=(0, 1), ylim=(0, ceiling),
                xlabel="within-band mode fraction",
                ylabel="within-band cumulative standardized residual energy / mode",
            )
            if row == 0:
                ax.set_title(title, fontsize=12)
            ax.grid(alpha=0.18)
            if column == 0:
                ax.legend(loc="upper left", fontsize=8.2)

    figure.suptitle(
        "Simulated 2024-07-13: latitude/longitude range parameterization check\n"
        "Data ranges: latitude 0.20, longitude 0.30, time 2.0; known DGP mean; Lanczos 512 + paired SLQ 8x192",
        fontsize=17,
    )
    figure.savefig(OUTPUT, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(OUTPUT)


if __name__ == "__main__":
    main()
