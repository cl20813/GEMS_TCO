#!/usr/bin/env python3
"""Replot stored Lanczos--SLQ results as band-localized diagnostics.

No precision matrices or Lanczos recurrences are recomputed.  Each band-reset
curve is literally a segment of the corresponding full cumulative curve,
translated to the origin and divided by the band's mode fraction.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path("/Users/joonwonlee/Documents/GEMS_TCO-1")
LOW_ROOT = REPO / (
    "outputs/summer_26/"
    "sim_20240713_low_frequency_misspecification_lanczos256_slq4x128"
)
HIGH_ROOT = REPO / (
    "outputs/summer_26/"
    "sim_20240713_true_vs_misspecified_residual_lanczos512_slq8x192"
)
SMOOTH_ROOT = REPO / (
    "outputs/summer_26/"
    "sim_20240713_smooth05_data_smooth03_vecchia_lanczos512_slq8x192"
)
OUTPUT_ROOT = REPO / "outputs/summer_26/sim_20240713_spectral_localization_validation"
OUTPUT = OUTPUT_ROOT / "2024-07-13_low_smooth_high_misspecification_band_localization.png"

BANDS = (
    ("low", 0.0, 1.0 / 3.0, "#DCEEFF"),
    ("middle", 1.0 / 3.0, 2.0 / 3.0, "#E8E4FF"),
    ("high", 2.0 / 3.0, 1.0, "#FFE8E0"),
)

CASES = (
    {
        "root": LOW_ROOT,
        "steps": 256,
        "wrong": "lat_range_quarter",
        "wrong_label": "latitude range 0.25x",
        "row_title": "Low-frequency-targeted misspecification: latitude range 0.25x",
    },
    {
        "root": SMOOTH_ROOT,
        "steps": 512,
        "wrong": "wrong_smooth_03",
        "wrong_label": "smoothness 0.5 -> 0.3",
        "row_title": "Smoothness misspecification: data 0.5, assumed 0.3",
    },
    {
        "root": HIGH_ROOT,
        "steps": 512,
        "wrong": "wrong_nugget",
        "wrong_label": "nugget 1 -> 0",
        "row_title": "High-frequency-dominant misspecification: nugget 1 -> 0",
    },
)


def final_curve(frame: pd.DataFrame, method: str, steps: int) -> tuple[np.ndarray, np.ndarray]:
    part = frame[
        frame["method"].eq(method)
        & frame["residual_lanczos_steps"].eq(int(steps))
    ].sort_values("estimated_mode_fraction")
    if part.empty:
        raise ValueError(f"No cumulative curve for method={method!r}, steps={steps}")
    x = part["estimated_mode_fraction"].to_numpy(float)
    y = part["cumulative_energy_per_n"].to_numpy(float)
    unique_x, unique_index = np.unique(x, return_index=True)
    return unique_x, y[unique_index]


def reset_segment(
    x: np.ndarray,
    y: np.ndarray,
    lower: float,
    upper: float,
) -> tuple[np.ndarray, np.ndarray]:
    inside = (x > lower) & (x < upper)
    segment_x = np.concatenate(([lower], x[inside], [upper]))
    segment_y = np.concatenate((
        [np.interp(lower, x, y)],
        y[inside],
        [np.interp(upper, x, y)],
    ))
    width = upper - lower
    return (segment_x - lower) / width, (segment_y - segment_y[0]) / width


def third_ratios(frame: pd.DataFrame, method: str) -> np.ndarray:
    order = pd.CategoricalDtype(["low", "middle", "high"], ordered=True)
    part = frame[frame["method"].eq(method)].copy()
    if part.empty:
        raise ValueError(f"No frequency thirds for method={method!r}")
    part["frequency_third"] = part["frequency_third"].astype(order)
    return part.sort_values("frequency_third")["energy_per_mode"].to_numpy(float)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(CASES), 4, figsize=(20.0, 4.55 * len(CASES)), constrained_layout=True)
    true_color = "#169873"
    wrong_color = "#D94841"

    for row, case in enumerate(CASES):
        curves = pd.read_csv(case["root"] / "residual_lanczos_cumulative_curves.csv")
        thirds = pd.read_csv(case["root"] / "frequency_third_energy_ratios.csv")
        row_curves = {
            "true": final_curve(curves, "true", case["steps"]),
            "wrong": final_curve(curves, case["wrong"], case["steps"]),
        }
        true_ratio = third_ratios(thirds, "true")
        wrong_ratio = third_ratios(thirds, case["wrong"])
        row_ceiling = max(1.1, 1.08 * float(np.max(np.r_[true_ratio, wrong_ratio])))

        for column, (band, lower, upper, background) in enumerate(BANDS):
            ax = axes[row, column]
            ax.set_facecolor(background)
            ax.plot([0.0, 1.0], [0.0, 1.0], "--", color="0.25", linewidth=1.1, label="null: y=x")
            for key, label, color, width in (
                ("true", "true covariance", true_color, 2.3),
                ("wrong", case["wrong_label"], wrong_color, 2.1),
            ):
                xb, yb = reset_segment(*row_curves[key], lower, upper)
                ax.plot(xb, yb, color=color, linewidth=width, label=label)
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, row_ceiling)
            ax.set_title(f"{band} band; endpoint R={wrong_ratio[column]:.3f}")
            ax.set_xlabel("within-band mode fraction")
            if column == 0:
                ax.set_ylabel("within-band cumulative energy / mode")
                ax.text(
                    0.0, 1.12, f"{case['row_title']}  (Lanczos {case['steps']})",
                    transform=ax.transAxes, fontsize=12.5, fontweight="bold", va="bottom",
                )
            ax.grid(alpha=0.18)
            if row == 0 and column == 0:
                ax.legend(loc="upper left", fontsize=8)

        ax = axes[row, 3]
        centers = np.arange(3)
        width = 0.34
        ax.axhline(1.0, color="0.25", linestyle="--", linewidth=1.1)
        ax.bar(centers - width / 2, true_ratio, width, color=true_color, alpha=0.88, label="true covariance")
        ax.bar(centers + width / 2, wrong_ratio, width, color=wrong_color, alpha=0.88, label=case["wrong_label"])
        for index, value in enumerate(wrong_ratio):
            ax.text(index + width / 2, value + 0.025 * row_ceiling, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(centers, ["low", "middle", "high"])
        ax.set_ylim(0.0, row_ceiling)
        ax.set_xlabel("precision-spectrum third")
        ax.set_ylabel("band energy ratio R")
        ax.set_title("Direct band-localization summary")
        ax.grid(alpha=0.18, axis="y")
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        "Can Lanczos--SLQ distinguish low-frequency, smoothness, and high-frequency misspecification?\n"
        "Band-reset curves are translated/rescaled segments of the same global cumulative curve",
        fontsize=16,
    )
    fig.savefig(OUTPUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
