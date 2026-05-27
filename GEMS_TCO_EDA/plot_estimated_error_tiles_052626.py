#!/usr/bin/env python3
"""Plot GEMS O3 EstimatedError on the same 4x8 geographic tile grid as nugget maps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/Users/joonwonlee/Documents/GEMS_TCO-1")
OUT_DIR = ROOT / "outputs" / "uncertainty_exploration"
CSV_PATH = OUT_DIR / "estimated_error_tile_summary_july_2025.csv"
PLOT_LABEL = "July 2025"

LAT_RANGE = (-3.0, 7.0)
LON_RANGE = (111.0, 131.0)
NY, NX = 4, 8
NADIR_LON, NADIR_LAT = 128.0, 0.0


def fmt_geo_tick(x: float) -> str:
    return f"{x:.2f}".rstrip("0").rstrip(".")


def summarize_tiles(df: pd.DataFrame) -> pd.DataFrame:
    agg_spec = dict(
        n_files=("file", "nunique"),
        total_n=("n", "sum"),
        median_error=("median_error", "median"),
        mean_error=("mean_error", "mean"),
        median_vza=("median_vza", "median"),
        median_sza=("median_sza", "median"),
    )
    if "median_o3" in df.columns:
        agg_spec["median_o3"] = ("median_o3", "median")
    if "mean_o3" in df.columns:
        agg_spec["mean_o3"] = ("mean_o3", "mean")

    grouped = (
        df.groupby(["tile_y", "tile_x"])
        .agg(**agg_spec)
        .reset_index()
    )

    if "median_o3" in grouped.columns:
        grouped["median_error_pct"] = 100.0 * grouped["median_error"] / grouped["median_o3"]

    lat_edges = np.linspace(LAT_RANGE[0], LAT_RANGE[1], NY + 1)
    lon_edges = np.linspace(LON_RANGE[0], LON_RANGE[1], NX + 1)
    grouped["tile_lat_min"] = grouped["tile_y"].map(lambda y: lat_edges[int(y)])
    grouped["tile_lat_max"] = grouped["tile_y"].map(lambda y: lat_edges[int(y) + 1])
    grouped["tile_lon_min"] = grouped["tile_x"].map(lambda x: lon_edges[int(x)])
    grouped["tile_lon_max"] = grouped["tile_x"].map(lambda x: lon_edges[int(x) + 1])
    grouped["tile_lat_center"] = 0.5 * (grouped["tile_lat_min"] + grouped["tile_lat_max"])
    grouped["tile_lon_center"] = 0.5 * (grouped["tile_lon_min"] + grouped["tile_lon_max"])
    return grouped


def tile_matrix(summary: pd.DataFrame, value_col: str) -> np.ndarray:
    mat = np.full((NY, NX), np.nan, dtype=float)
    for row in summary.itertuples(index=False):
        mat[int(row.tile_y), int(row.tile_x)] = float(getattr(row, value_col))
    return mat


def draw_tile_heatmap(
    ax: plt.Axes,
    mat: np.ndarray,
    title: str,
    cbar_label: str,
    cmap: str = "viridis",
) -> None:
    lat_edges = np.linspace(LAT_RANGE[0], LAT_RANGE[1], NY + 1)
    lon_edges = np.linspace(LON_RANGE[0], LON_RANGE[1], NX + 1)
    extent = [lon_edges[0], lon_edges[-1], lat_edges[0], lat_edges[-1]]

    im = ax.imshow(mat, origin="lower", cmap=cmap, extent=extent, aspect="auto")

    for lon in lon_edges:
        ax.axvline(lon, color="white", lw=0.6, alpha=0.45)
    for lat in lat_edges:
        ax.axhline(lat, color="white", lw=0.6, alpha=0.45)

    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    finite_vals = mat[np.isfinite(mat)]
    vmin = float(np.nanmin(finite_vals)) if finite_vals.size else 0.0
    vmax = float(np.nanmax(finite_vals)) if finite_vals.size else 1.0
    threshold = vmin + 0.55 * (vmax - vmin)

    for y_idx, lat in enumerate(lat_centers):
        for x_idx, lon in enumerate(lon_centers):
            val = mat[y_idx, x_idx]
            if not np.isfinite(val):
                continue
            color = "white" if val > threshold else "black"
            ax.text(lon, lat, f"{val:.3f}", ha="center", va="center", fontsize=8, color=color)

    if extent[0] <= NADIR_LON <= extent[1] and extent[2] <= NADIR_LAT <= extent[3]:
        ax.scatter(
            [NADIR_LON],
            [NADIR_LAT],
            marker="*",
            s=230,
            c="white",
            edgecolors="black",
            linewidths=1.1,
            label=f"nadir ({NADIR_LON:g}E, {NADIR_LAT:g}N)",
            zorder=5,
        )
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xticks(lon_edges)
    ax.set_yticks(lat_edges)
    ax.set_xticklabels([fmt_geo_tick(x) for x in lon_edges])
    ax.set_yticklabels([fmt_geo_tick(y) for y in lat_edges])
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.86)
    cbar.set_label(cbar_label)


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing tile summary CSV: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    summary = summarize_tiles(df)
    summary_path = OUT_DIR / "estimated_error_tile_summary_july_2025_4x8_geo.csv"
    summary.round(6).to_csv(summary_path, index=False)

    error_mat = tile_matrix(summary, "median_error")
    vza_mat = tile_matrix(summary, "median_vza")
    has_error_pct = "median_error_pct" in summary.columns
    pct_mat = tile_matrix(summary, "median_error_pct") if has_error_pct else None

    ncols = 3 if has_error_pct else 2
    fig, axes = plt.subplots(1, ncols, figsize=(8.0 * ncols, 6.2))
    axes = np.atleast_1d(axes)
    fig.suptitle(
        f"{PLOT_LABEL} GEMS O3 EstimatedError and viewing geometry, 4x8 geographic tiles\n"
        "FinalAlgorithmFlags in [0, 2], same tile grid as nugget heatmaps",
        fontsize=13,
        fontweight="bold",
    )
    draw_tile_heatmap(
        axes[0],
        error_mat,
        "Median retrieved O3 estimated error",
        "median EstimatedError",
        cmap="magma",
    )
    if has_error_pct and pct_mat is not None:
        draw_tile_heatmap(
            axes[1],
            pct_mat,
            "Median retrieved O3 estimated error rate",
            "median EstimatedError / O3 (%)",
            cmap="magma",
        )
        vza_ax = axes[2]
    else:
        vza_ax = axes[1]
    draw_tile_heatmap(
        vza_ax,
        vza_mat,
        "Median viewing zenith angle",
        "median ViewingZenithAngle",
        cmap="plasma",
    )
    fig.tight_layout()
    out_combined = OUT_DIR / "estimated_error_vza_tile_july_2025_4x8_geo.png"
    fig.savefig(out_combined, dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    draw_tile_heatmap(
        ax,
        error_mat,
        f"{PLOT_LABEL} median retrieved O3 estimated error, 4x8",
        "median EstimatedError",
        cmap="magma",
    )
    fig.tight_layout()
    out_error = OUT_DIR / "estimated_error_tile_july_2025_4x8_geo.png"
    fig.savefig(out_error, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", summary_path)
    print("Saved:", out_combined)
    print("Saved:", out_error)


if __name__ == "__main__":
    main()
