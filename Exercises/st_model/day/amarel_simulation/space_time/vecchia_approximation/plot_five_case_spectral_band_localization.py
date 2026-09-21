#!/usr/bin/env python3
"""Create the five-case by three-band Lanczos--SLQ localization figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path("/Users/joonwonlee/Documents/GEMS_TCO-1")
NEW_ROOT = REPO / (
    "outputs/summer_26/"
    "sim_20240713_latitude_short_long_smooth10_lanczos512_slq8x192"
)
SMOOTH03_ROOT = REPO / (
    "outputs/summer_26/"
    "sim_20240713_smooth05_data_smooth03_vecchia_lanczos512_slq8x192"
)
NUGGET_ROOT = REPO / (
    "outputs/summer_26/"
    "sim_20240713_true_vs_misspecified_residual_lanczos512_slq8x192"
)
OUTPUT_ROOT = REPO / "outputs/summer_26/sim_20240713_spectral_localization_validation"
OUTPUT = OUTPUT_ROOT / "2024-07-13_five_case_5x3_band_localization.png"
STEPS = 512

BANDS = (
    ("Low precision-spectrum third", "low", 0.0, 1.0 / 3.0, "#DCEEFF"),
    ("Middle precision-spectrum third", "middle", 1.0 / 3.0, 2.0 / 3.0, "#E8E4FF"),
    ("High precision-spectrum third", "high", 2.0 / 3.0, 1.0, "#FFE8E0"),
)

CASES = (
    {
        "root": NEW_ROOT,
        "method": "lat_range_quarter_512",
        "label": "latitude range 0.25x",
        "title": "Latitude range too short: assumed 0.05 (0.25x truth 0.20)",
    },
    {
        "root": NEW_ROOT,
        "method": "lat_range_2x",
        "label": "latitude range 2x",
        "title": "Latitude range too long: assumed 0.40 (2x truth 0.20)",
    },
    {
        "root": SMOOTH03_ROOT,
        "method": "wrong_smooth_03",
        "label": "smoothness 0.3",
        "title": "Rougher assumed covariance: Matern smoothness 0.3 (truth 0.5)",
    },
    {
        "root": NEW_ROOT,
        "method": "wrong_smooth_10",
        "label": "smoothness 1.0",
        "title": "Smoother assumed covariance: Matern smoothness 1.0 (truth 0.5)",
    },
    {
        "root": NUGGET_ROOT,
        "method": "wrong_nugget",
        "label": "nugget 0",
        "title": "Nugget omitted: assumed 0 (truth 1)",
    },
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
        name: float(part.loc[name, "energy_per_mode"])
        for name in ("low", "middle", "high")
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(18.0, 24.5), constrained_layout=True)
    subfigures = figure.subfigures(nrows=len(CASES), ncols=1)
    true_color = "#169873"
    wrong_color = "#D94841"

    for row, (subfigure, case) in enumerate(zip(subfigures, CASES)):
        curves = pd.read_csv(case["root"] / "residual_lanczos_cumulative_curves.csv")
        thirds = pd.read_csv(case["root"] / "frequency_third_energy_ratios.csv")
        line_data = {
            "true": final_curve(curves, "true"),
            "wrong": final_curve(curves, case["method"]),
        }
        ratio_data = {
            "true": ratios(thirds, "true"),
            "wrong": ratios(thirds, case["method"]),
        }
        reset_data: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
        maximum = 1.0
        for _, band, lower, upper, _ in BANDS:
            for method in ("true", "wrong"):
                values = reset_segment(*line_data[method], lower, upper)
                reset_data[(band, method)] = values
                maximum = max(maximum, float(np.max(values[1])))
        ceiling = max(1.08, 1.05 * maximum)

        axes = subfigure.subplots(1, 3)
        subfigure.suptitle(case["title"], fontsize=14, fontweight="bold")
        for column, (ax, (column_title, band, _, _, background)) in enumerate(zip(axes, BANDS)):
            ax.set_facecolor(background)
            ax.plot([0, 1], [0, 1], "--", color="0.25", linewidth=1.15, label="null: y=x")
            ax.plot(
                *reset_data[(band, "true")], color=true_color, linewidth=2.3,
                label=f"true covariance: R={ratio_data['true'][band]:.3f}",
            )
            ax.plot(
                *reset_data[(band, "wrong")], color=wrong_color, linewidth=2.3,
                label=f"{case['label']}: R={ratio_data['wrong'][band]:.3f}",
            )
            ax.scatter(
                [1, 1], [ratio_data["true"][band], ratio_data["wrong"][band]],
                color=[true_color, wrong_color], s=26, zorder=4,
            )
            ax.text(
                0.97, 0.05,
                f"R = {ratio_data['wrong'][band]:.3f}\nR-1 = {ratio_data['wrong'][band] - 1:+.3f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.78, "edgecolor": "0.75"},
            )
            ax.set(
                xlim=(0, 1), ylim=(0, ceiling),
                xlabel="within-band mode fraction",
                ylabel="within-band cumulative standardized residual energy / mode",
            )
            if row == 0:
                ax.set_title(column_title, fontsize=12)
            ax.grid(alpha=0.18)
            if column == 0:
                ax.legend(loc="upper left", fontsize=8.2)

    figure.suptitle(
        "Simulated 2024-07-13: five fixed covariance misspecifications localized by Lanczos--SLQ\n"
        "Data Matern smoothness 0.5; known DGP mean; all panels Lanczos 512 + paired SLQ 8x192",
        fontsize=17,
    )
    figure.savefig(OUTPUT, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(OUTPUT)


if __name__ == "__main__":
    main()
