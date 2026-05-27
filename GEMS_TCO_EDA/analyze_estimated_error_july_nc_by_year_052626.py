from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon
from netCDF4 import Dataset


RAW_ROOT = Path("/Volumes/Backup Plus/GEMS_UNZIPPED")
OUT_DIR = Path("/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/uncertainty_exploration")
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = [2022, 2023, 2024, 2025]
MONTH = 7
LAT_RANGE = (-3.0, 7.0)
LON_RANGE = (111.0, 131.0)
NADIR_LON, NADIR_LAT = 128.0, 0.0
GOOD_FLAGS = {0.0, 2.0}
NY, NX = 4, 8


def month_dir(year: int) -> Path:
    return RAW_ROOT / f"{year}{MONTH:02d}0131"


def file_list(year: int) -> list[Path]:
    d = month_dir(year)
    if not d.exists():
        raise FileNotFoundError(d)
    return sorted(p for p in d.glob("*.nc") if not p.name.startswith("._"))


def _as_float_array(var) -> np.ndarray:
    arr = var[:]
    if np.ma.isMaskedArray(arr):
        arr = arr.filled(np.nan)
    return np.asarray(arr, dtype=np.float64)


def _as_flag_array(var) -> np.ndarray:
    arr = var[:]
    if np.ma.isMaskedArray(arr):
        arr = arr.filled(np.nan)
    return np.asarray(arr, dtype=np.float64)


def add_tiles(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat_edges = np.linspace(LAT_RANGE[0], LAT_RANGE[1], NY + 1)
    lon_edges = np.linspace(LON_RANGE[0], LON_RANGE[1], NX + 1)
    in_region = (
        np.isfinite(lat)
        & np.isfinite(lon)
        & (lat >= LAT_RANGE[0])
        & (lat <= LAT_RANGE[1])
        & (lon >= LON_RANGE[0])
        & (lon <= LON_RANGE[1])
    )
    tile_y = np.digitize(lat, lat_edges, right=False) - 1
    tile_x = np.digitize(lon, lon_edges, right=False) - 1
    tile_y = np.clip(tile_y, 0, NY - 1)
    tile_x = np.clip(tile_x, 0, NX - 1)
    return in_region, tile_y.astype(np.int16), tile_x.astype(np.int16)


def summarize_file(path: Path, year: int) -> tuple[pd.DataFrame, dict[str, object]]:
    with Dataset(path) as ds:
        data = ds.groups["Data Fields"]
        geo = ds.groups["Geolocation Fields"]
        lat = _as_float_array(geo.variables["Latitude"]).ravel()
        lon = _as_float_array(geo.variables["Longitude"]).ravel()
        o3 = _as_float_array(data.variables["ColumnAmountO3"]).ravel()
        err = _as_float_array(data.variables["EstimatedError"]).ravel()
        flags = _as_flag_array(data.variables["FinalAlgorithmFlags"]).ravel()
        vza = _as_float_array(geo.variables["ViewingZenithAngle"]).ravel()
        sza = _as_float_array(geo.variables["SolarZenithAngle"]).ravel()

    in_region, tile_y, tile_x = add_tiles(lat, lon)
    finite_o3 = np.isfinite(o3)
    finite_err = np.isfinite(err)
    good_flag = np.isin(flags, list(GOOD_FLAGS))
    keep = in_region & finite_o3 & finite_err & good_flag & (o3 != 0)

    meta = {
        "year": year,
        "file": path.name,
        "rows": int(lat.size),
        "rows_in_region": int(in_region.sum()),
        "good_error_rows": int(keep.sum()),
    }
    if not keep.any():
        return pd.DataFrame(), meta

    ratio_pct = 100.0 * err[keep] / o3[keep]
    df = pd.DataFrame(
        {
            "tile_y": tile_y[keep],
            "tile_x": tile_x[keep],
            "ColumnAmountO3": o3[keep],
            "EstimatedError": err[keep],
            "EstimatedErrorPct": ratio_pct,
            "ViewingZenithAngle": vza[keep],
            "SolarZenithAngle": sza[keep],
        }
    )
    summary = (
        df.groupby(["tile_y", "tile_x"])
        .agg(
            n=("EstimatedError", "size"),
            median_error=("EstimatedError", "median"),
            mean_error=("EstimatedError", "mean"),
            median_error_pct=("EstimatedErrorPct", "median"),
            mean_error_pct=("EstimatedErrorPct", "mean"),
            median_o3=("ColumnAmountO3", "median"),
            mean_o3=("ColumnAmountO3", "mean"),
            median_vza=("ViewingZenithAngle", "median"),
            median_sza=("SolarZenithAngle", "median"),
        )
        .reset_index()
    )
    summary["year"] = year
    summary["file"] = path.name
    return summary, meta


def _geo_edges() -> tuple[np.ndarray, np.ndarray]:
    return np.linspace(LAT_RANGE[0], LAT_RANGE[1], NY + 1), np.linspace(LON_RANGE[0], LON_RANGE[1], NX + 1)


def summarize_year(file_summaries: pd.DataFrame, year: int) -> pd.DataFrame:
    agg = (
        file_summaries.groupby(["tile_y", "tile_x"])
        .agg(
            n_files=("file", "nunique"),
            total_n=("n", "sum"),
            median_error=("median_error", "median"),
            mean_error=("mean_error", "mean"),
            median_error_pct=("median_error_pct", "median"),
            mean_error_pct=("mean_error_pct", "mean"),
            median_o3=("median_o3", "median"),
            mean_o3=("mean_o3", "mean"),
            median_vza=("median_vza", "median"),
            median_sza=("median_sza", "median"),
        )
        .reset_index()
    )
    lat_edges, lon_edges = _geo_edges()
    rows = []
    for y in range(NY):
        for x in range(NX):
            row = agg[(agg["tile_y"] == y) & (agg["tile_x"] == x)]
            if row.empty:
                item = {"year": year, "tile_y": y, "tile_x": x}
                for col in agg.columns:
                    if col not in item:
                        item[col] = np.nan
            else:
                item = row.iloc[0].to_dict()
            item["year"] = year
            item["tile_lat_min"] = lat_edges[y]
            item["tile_lat_max"] = lat_edges[y + 1]
            item["tile_lon_min"] = lon_edges[x]
            item["tile_lon_max"] = lon_edges[x + 1]
            item["tile_lat_center"] = 0.5 * (lat_edges[y] + lat_edges[y + 1])
            item["tile_lon_center"] = 0.5 * (lon_edges[x] + lon_edges[x + 1])
            rows.append(item)
    return pd.DataFrame(rows)


def matrix(tile_summary: pd.DataFrame, value_col: str) -> np.ndarray:
    mat = np.full((NY, NX), np.nan, dtype=float)
    for row in tile_summary.itertuples(index=False):
        mat[int(row.tile_y), int(row.tile_x)] = float(getattr(row, value_col))
    return mat


def draw_schematic_background(ax) -> None:
    ax.set_facecolor("#77b7de")
    land_specs = [
        [(111.0, 7.0), (118.0, 7.0), (118.6, 4.1), (118.6, 1.6), (119.0, 0.1),
         (118.2, -1.4), (116.3, -1.9), (114.1, -1.4), (112.4, -0.4), (111.0, -0.5)],
        [(119.5, -3.0), (121.0, -3.0), (121.0, -1.0), (124.7, -1.0), (125.0, 1.0),
         (122.0, 1.0), (121.0, 2.0), (120.0, 2.0), (119.5, 0.0)],
        [(126.0, 6.0), (128.7, 6.0), (129.0, 7.0), (126.0, 7.0)],
        [(127.0, -1.0), (129.0, -1.0), (129.0, 2.0), (127.0, 2.0)],
        [(112.0, -3.0), (119.8, -3.0), (119.6, -2.55), (115.5, -2.35), (112.0, -2.55)],
        [(120.7, 7.0), (122.7, 7.0), (122.4, 5.9), (121.4, 5.7), (120.9, 6.3)],
        [(129.0, -3.0), (131.0, -3.0), (131.0, -1.0), (129.0, -1.2)],
    ]
    for coords in land_specs:
        ax.add_patch(Polygon(coords, closed=True, facecolor="#2fbf38", edgecolor="#07351f", lw=0.8, alpha=0.82, zorder=0))
    ax.text(115.0, 5.0, "SOUTH CHINA SEA", color="white", fontsize=7, alpha=0.65, ha="center", zorder=0.2)
    ax.text(122.4, 4.7, "CELEBES SEA", color="white", fontsize=7, alpha=0.65, ha="center", zorder=0.2)
    ax.text(124.2, -2.0, "MOLUCCA SEA", color="white", fontsize=7, alpha=0.65, ha="center", zorder=0.2)


def fmt_tick(x: float) -> str:
    return f"{x:.2f}".rstrip("0").rstrip(".")


def draw_tile_heatmap(ax, mat: np.ndarray, title: str, cbar_label: str, cmap: str = "magma") -> None:
    lat_edges, lon_edges = _geo_edges()
    extent = [lon_edges[0], lon_edges[-1], lat_edges[0], lat_edges[-1]]
    draw_schematic_background(ax)
    im = ax.imshow(mat, origin="lower", cmap=cmap, extent=extent, aspect="auto", alpha=0.58, zorder=1)
    for lon in lon_edges:
        ax.axvline(lon, color="white", lw=0.7, alpha=0.55, zorder=2)
    for lat in lat_edges:
        ax.axhline(lat, color="white", lw=0.7, alpha=0.55, zorder=2)
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    for y, lat in enumerate(lat_centers):
        for x, lon in enumerate(lon_centers):
            val = mat[y, x]
            text = "NA" if not np.isfinite(val) else f"{val:.3f}"
            ax.text(lon, lat, text, ha="center", va="center", fontsize=8, color="white", zorder=3)
    ax.scatter([NADIR_LON], [NADIR_LAT], marker="*", s=230, c="gold", edgecolors="black", linewidths=1.1, label="nadir point (128E, 0N)", zorder=5)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(LON_RANGE)
    ax.set_ylim(LAT_RANGE)
    ax.set_xticks(lon_edges)
    ax.set_yticks(lat_edges)
    ax.set_xticklabels([fmt_tick(x) for x in lon_edges])
    ax.set_yticklabels([fmt_tick(y) for y in lat_edges])
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.86)
    cbar.set_label(cbar_label)


def plot_year(tile_summary: pd.DataFrame, year: int) -> tuple[Path, Path]:
    mats = {
        "median_error": matrix(tile_summary, "median_error"),
        "median_error_pct": matrix(tile_summary, "median_error_pct"),
        "median_vza": matrix(tile_summary, "median_vza"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(24, 6.2))
    fig.suptitle(
        f"GEMS O3 retrieval uncertainty diagnostics, July {year}, 4x8 geographic tiles\n"
        f"region lat {LAT_RANGE[0]:g} to {LAT_RANGE[1]:g}, lon {LON_RANGE[0]:g} to {LON_RANGE[1]:g}; FinalAlgorithmFlags in [0, 2]",
        fontsize=13,
        fontweight="bold",
    )
    draw_tile_heatmap(axes[0], mats["median_error"], "Median retrieved O3 estimated error", "median EstimatedError", "magma")
    draw_tile_heatmap(axes[1], mats["median_error_pct"], "Median retrieved O3 estimated error rate", "median EstimatedError / O3 (%)", "magma")
    draw_tile_heatmap(axes[2], mats["median_vza"], "Median viewing zenith angle", "median ViewingZenithAngle", "plasma")
    plt.tight_layout()
    combined = OUT_DIR / f"estimated_error_rate_tile_july_{year}_4x8_geo.png"
    fig.savefig(combined, dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    draw_tile_heatmap(ax, mats["median_error_pct"], f"July {year} median retrieved O3 estimated error rate, 4x8", "median EstimatedError / O3 (%)", "magma")
    plt.tight_layout()
    single = OUT_DIR / f"estimated_error_rate_only_tile_july_{year}_4x8_geo.png"
    fig.savefig(single, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return combined, single


def main() -> None:
    all_year = []
    all_files = []
    all_meta = []
    for year in YEARS:
        paths = file_list(year)
        print(f"{year}: {len(paths)} NetCDF files in {month_dir(year)}")
        summaries = []
        metas = []
        for idx, path in enumerate(paths, start=1):
            try:
                summary, meta = summarize_file(path, year)
            except Exception as exc:
                print(f"WARNING: failed {path}: {exc}")
                continue
            metas.append(meta)
            if len(summary):
                summaries.append(summary)
            if idx % 25 == 0 or idx == len(paths):
                print(f"{year}: processed {idx}/{len(paths)} files")
        if not summaries:
            print(f"WARNING: no usable summaries for {year}")
            continue
        file_summary = pd.concat(summaries, ignore_index=True)
        meta_df = pd.DataFrame(metas)
        year_summary = summarize_year(file_summary, year)

        file_path = OUT_DIR / f"estimated_error_file_tile_summary_july_{year}_4x8.csv"
        tile_path = OUT_DIR / f"estimated_error_tile_summary_july_{year}_4x8_geo.csv"
        meta_path = OUT_DIR / f"estimated_error_file_meta_july_{year}.csv"
        file_summary.round(6).to_csv(file_path, index=False)
        year_summary.round(6).to_csv(tile_path, index=False)
        meta_df.to_csv(meta_path, index=False)
        combined_png, single_png = plot_year(year_summary, year)
        print(f"Saved {file_path}")
        print(f"Saved {tile_path}")
        print(f"Saved {meta_path}")
        print(f"Saved {combined_png}")
        print(f"Saved {single_png}")
        all_files.append(file_summary)
        all_year.append(year_summary)
        all_meta.append(meta_df)

    if all_year:
        pd.concat(all_year, ignore_index=True).round(6).to_csv(
            OUT_DIR / "estimated_error_tile_summary_july_2022_2025_4x8_geo.csv",
            index=False,
        )
        pd.concat(all_files, ignore_index=True).round(6).to_csv(
            OUT_DIR / "estimated_error_file_tile_summary_july_2022_2025_4x8.csv",
            index=False,
        )
        pd.concat(all_meta, ignore_index=True).to_csv(
            OUT_DIR / "estimated_error_file_meta_july_2022_2025.csv",
            index=False,
        )


if __name__ == "__main__":
    main()
