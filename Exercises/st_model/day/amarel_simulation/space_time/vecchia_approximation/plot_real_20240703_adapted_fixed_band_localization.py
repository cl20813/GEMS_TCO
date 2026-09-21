#!/usr/bin/env python3
"""Replot real 2024-07-03 adapted/fixed results as three band-reset curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path("/Users/joonwonlee/Documents/GEMS_TCO-1")
RESULT_ROOT = REPO / (
    "outputs/summer_26/"
    "vecchia_local_20240703_adapted_fixed_residual_lanczos512_slq8x192"
)
OUTPUT = RESULT_ROOT / "2024-07-03_adapted_fixed_low_middle_high_band_localization.png"
STEPS = 512
METHODS = ("adapted", "fixed")
LABELS = {"adapted": "adapted corridor", "fixed": "fixed center"}
COLORS = {"adapted": "#1f77b4", "fixed": "#d62728"}
BANDS = (
    ("low-frequency proxy", 0.0, 1.0 / 3.0, "#DCEEFF"),
    ("middle-frequency proxy", 1.0 / 3.0, 2.0 / 3.0, "#E8E4FF"),
    ("high-frequency proxy", 2.0 / 3.0, 1.0, "#FFE8E0"),
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


def third_ratios(frame: pd.DataFrame, method: str) -> dict[str, float]:
    part = frame[frame["method"].eq(method)].set_index("frequency_third")
    return {
        name: float(part.loc[name, "energy_per_mode"])
        for name in ("low", "middle", "high")
    }


def main() -> None:
    curves = pd.read_csv(RESULT_ROOT / "residual_lanczos_cumulative_curves.csv")
    thirds = pd.read_csv(RESULT_ROOT / "frequency_third_energy_ratios.csv")
    line_data = {method: final_curve(curves, method) for method in METHODS}
    ratio_data = {method: third_ratios(thirds, method) for method in METHODS}

    reset_data: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    maximum = 1.0
    for band, lower, upper, _ in BANDS:
        for method in METHODS:
            values = reset_segment(*line_data[method], lower, upper)
            reset_data[(band, method)] = values
            maximum = max(maximum, float(np.max(values[1])))
    ceiling = 1.06 * maximum

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.4), constrained_layout=True)
    for index, (ax, (title, _, _, background)) in enumerate(zip(axes, BANDS)):
        band_name = ("low", "middle", "high")[index]
        ax.set_facecolor(background)
        ax.plot([0, 1], [0, 1], "--", color="0.25", linewidth=1.2, label="null: y=x")
        for method in METHODS:
            x, y = reset_data[(title, method)]
            ratio = ratio_data[method][band_name]
            ax.plot(
                x, y, color=COLORS[method], linewidth=2.35,
                label=f"{LABELS[method]}: R={ratio:.3f}",
            )
            ax.scatter([1.0], [ratio], s=30, color=COLORS[method], zorder=4)
        ax.set(
            xlim=(0, 1), ylim=(0, ceiling),
            xlabel="within-band mode fraction",
            ylabel="within-band cumulative standardized residual energy / mode",
            title=title,
        )
        ax.grid(alpha=0.18)
        ax.legend(loc="upper left", fontsize=8.5)

    fig.suptitle(
        "Real 2024-07-03: adapted versus fixed Vecchia, localized spectral residual diagnostics\n"
        "Each panel is the corresponding third of the same global curve, translated and rescaled; Lanczos 512 + SLQ 8x192",
        fontsize=15,
    )
    fig.savefig(OUTPUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
