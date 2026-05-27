from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DATA_ROOT = Path("/Users/joonwonlee/Documents/GEMS_DATA")
OUT_DIR = Path("/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/uncertainty_exploration")
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = [2022, 2023, 2024, 2025]
MONTH = 7
LAT_RANGE = (-3.0, 7.0)
LON_RANGE = (111.0, 131.0)
NY, NX = 4, 8
GOOD_FLAGS = {0.0, 2.0}
CHUNKSIZE = 1_000_000


def july_csv_path(year: int) -> Path:
    yy = str(year)[2:]
    return DATA_ROOT / f"data_{year}" / f"data_{yy}_{MONTH:02d}_0131_N-37_E111131.csv"


def add_tile_columns(df: pd.DataFrame) -> pd.DataFrame:
    lat_edges = np.linspace(LAT_RANGE[0], LAT_RANGE[1], NY + 1)
    lon_edges = np.linspace(LON_RANGE[0], LON_RANGE[1], NX + 1)
    out = df.copy()
    out["tile_y"] = pd.cut(out["Latitude"], lat_edges, labels=False, include_lowest=True)
    out["tile_x"] = pd.cut(out["Longitude"], lon_edges, labels=False, include_lowest=True)
    out = out.dropna(subset=["tile_y", "tile_x"])
    out["tile_y"] = out["tile_y"].astype(np.int16)
    out["tile_x"] = out["tile_x"].astype(np.int16)
    return out


def summarize_year(year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = july_csv_path(year)
    if not path.exists():
        raise FileNotFoundError(path)

    usecols = ["Latitude", "Longitude", "ColumnAmountO3", "FinalAlgorithmFlags"]
    tile_stats: dict[tuple[int, int], dict[str, float]] = {}
    tile_values: dict[tuple[int, int], list[np.ndarray]] = {}
    overall = {
        "year": year,
        "source_csv": str(path),
        "rows": 0,
        "rows_in_region": 0,
        "finite_o3_rows": 0,
        "good_flag_rows": 0,
        "good_finite_o3_rows": 0,
        "lat_min": np.inf,
        "lat_max": -np.inf,
        "lon_min": np.inf,
        "lon_max": -np.inf,
        "finite_lat_min": np.inf,
        "finite_lat_max": -np.inf,
        "finite_lon_min": np.inf,
        "finite_lon_max": -np.inf,
    }

    for chunk_idx, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=CHUNKSIZE), start=1):
        overall["rows"] += len(chunk)
        chunk = chunk[
            chunk["Latitude"].between(*LAT_RANGE)
            & chunk["Longitude"].between(*LON_RANGE)
        ]
        if chunk.empty:
            continue

        overall["rows_in_region"] += len(chunk)
        overall["lat_min"] = min(overall["lat_min"], float(chunk["Latitude"].min()))
        overall["lat_max"] = max(overall["lat_max"], float(chunk["Latitude"].max()))
        overall["lon_min"] = min(overall["lon_min"], float(chunk["Longitude"].min()))
        overall["lon_max"] = max(overall["lon_max"], float(chunk["Longitude"].max()))

        chunk = add_tile_columns(chunk)
        finite = np.isfinite(chunk["ColumnAmountO3"].to_numpy(dtype=float))
        good = chunk["FinalAlgorithmFlags"].isin(GOOD_FLAGS).to_numpy()
        good_finite = finite & good

        overall["finite_o3_rows"] += int(finite.sum())
        overall["good_flag_rows"] += int(good.sum())
        overall["good_finite_o3_rows"] += int(good_finite.sum())

        if finite.any():
            finite_chunk = chunk.loc[finite]
            overall["finite_lat_min"] = min(overall["finite_lat_min"], float(finite_chunk["Latitude"].min()))
            overall["finite_lat_max"] = max(overall["finite_lat_max"], float(finite_chunk["Latitude"].max()))
            overall["finite_lon_min"] = min(overall["finite_lon_min"], float(finite_chunk["Longitude"].min()))
            overall["finite_lon_max"] = max(overall["finite_lon_max"], float(finite_chunk["Longitude"].max()))

        for (tile_y, tile_x), group in chunk.groupby(["tile_y", "tile_x"], sort=False):
            key = (int(tile_y), int(tile_x))
            stats = tile_stats.setdefault(
                key,
                {
                    "year": year,
                    "tile_y": key[0],
                    "tile_x": key[1],
                    "rows": 0,
                    "finite_o3_rows": 0,
                    "good_flag_rows": 0,
                    "good_finite_o3_rows": 0,
                    "sum_o3_good": 0.0,
                    "sumsq_o3_good": 0.0,
                    "min_o3_good": np.inf,
                    "max_o3_good": -np.inf,
                },
            )
            vals = group["ColumnAmountO3"].to_numpy(dtype=float)
            flags = group["FinalAlgorithmFlags"].to_numpy(dtype=float)
            finite_vals = np.isfinite(vals)
            good_vals = np.isin(flags, list(GOOD_FLAGS))
            keep = finite_vals & good_vals

            stats["rows"] += len(group)
            stats["finite_o3_rows"] += int(finite_vals.sum())
            stats["good_flag_rows"] += int(good_vals.sum())
            stats["good_finite_o3_rows"] += int(keep.sum())

            if keep.any():
                kept_vals = vals[keep].astype(np.float32, copy=False)
                stats["sum_o3_good"] += float(kept_vals.sum(dtype=np.float64))
                stats["sumsq_o3_good"] += float(np.square(kept_vals, dtype=np.float64).sum())
                stats["min_o3_good"] = min(stats["min_o3_good"], float(np.min(kept_vals)))
                stats["max_o3_good"] = max(stats["max_o3_good"], float(np.max(kept_vals)))
                tile_values.setdefault(key, []).append(kept_vals.copy())

        if chunk_idx % 10 == 0:
            print(f"{year}: processed {chunk_idx:,} chunks / {overall['rows']:,} rows")

    rows = []
    lat_edges = np.linspace(LAT_RANGE[0], LAT_RANGE[1], NY + 1)
    lon_edges = np.linspace(LON_RANGE[0], LON_RANGE[1], NX + 1)
    for tile_y in range(NY):
        for tile_x in range(NX):
            key = (tile_y, tile_x)
            stats = tile_stats.get(
                key,
                {
                    "year": year,
                    "tile_y": tile_y,
                    "tile_x": tile_x,
                    "rows": 0,
                    "finite_o3_rows": 0,
                    "good_flag_rows": 0,
                    "good_finite_o3_rows": 0,
                    "sum_o3_good": 0.0,
                    "sumsq_o3_good": 0.0,
                    "min_o3_good": np.nan,
                    "max_o3_good": np.nan,
                },
            ).copy()
            n = int(stats["good_finite_o3_rows"])
            if n:
                vals = np.concatenate(tile_values[key])
                stats["mean_o3_good"] = stats["sum_o3_good"] / n
                stats["median_o3_good"] = float(np.median(vals))
                stats["std_o3_good"] = float(np.std(vals, ddof=1)) if n > 1 else 0.0
                stats["q05_o3_good"] = float(np.quantile(vals, 0.05))
                stats["q95_o3_good"] = float(np.quantile(vals, 0.95))
            else:
                stats["mean_o3_good"] = np.nan
                stats["median_o3_good"] = np.nan
                stats["std_o3_good"] = np.nan
                stats["q05_o3_good"] = np.nan
                stats["q95_o3_good"] = np.nan
            stats["good_finite_fraction"] = n / stats["rows"] if stats["rows"] else np.nan
            stats["tile_lat_min"] = lat_edges[tile_y]
            stats["tile_lat_max"] = lat_edges[tile_y + 1]
            stats["tile_lon_min"] = lon_edges[tile_x]
            stats["tile_lon_max"] = lon_edges[tile_x + 1]
            stats["tile_lat_center"] = 0.5 * (lat_edges[tile_y] + lat_edges[tile_y + 1])
            stats["tile_lon_center"] = 0.5 * (lon_edges[tile_x] + lon_edges[tile_x + 1])
            rows.append(stats)

    for key, val in list(overall.items()):
        if isinstance(val, float) and not np.isfinite(val):
            overall[key] = np.nan
    overall["good_finite_fraction"] = (
        overall["good_finite_o3_rows"] / overall["rows_in_region"]
        if overall["rows_in_region"]
        else np.nan
    )
    return pd.DataFrame(rows), pd.DataFrame([overall])


def main() -> None:
    all_tile = []
    all_overall = []
    for year in YEARS:
        print(f"Processing {year}: {july_csv_path(year)}")
        tile_df, overall_df = summarize_year(year)
        tile_path = OUT_DIR / f"gems_o3_expanded_july_{year}_tile_summary_4x8.csv"
        overall_path = OUT_DIR / f"gems_o3_expanded_july_{year}_overall_summary.csv"
        tile_df.round(6).to_csv(tile_path, index=False)
        overall_df.round(6).to_csv(overall_path, index=False)
        print(f"Saved {tile_path}")
        print(f"Saved {overall_path}")
        all_tile.append(tile_df)
        all_overall.append(overall_df)

    combined_tile = pd.concat(all_tile, ignore_index=True)
    combined_overall = pd.concat(all_overall, ignore_index=True)
    combined_tile_path = OUT_DIR / "gems_o3_expanded_july_2022_2025_tile_summary_4x8.csv"
    combined_overall_path = OUT_DIR / "gems_o3_expanded_july_2022_2025_overall_summary.csv"
    combined_tile.round(6).to_csv(combined_tile_path, index=False)
    combined_overall.round(6).to_csv(combined_overall_path, index=False)
    print(f"Saved {combined_tile_path}")
    print(f"Saved {combined_overall_path}")


if __name__ == "__main__":
    main()
