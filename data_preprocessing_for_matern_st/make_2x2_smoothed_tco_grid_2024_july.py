#!/usr/bin/env python3
"""Create a 2x2 block-averaged tco_grid pickle for July 2024.

Default input:
    /Users/joonwonlee/Documents/GEMS_DATA/pickle_2024/
        tco_grid_24_07.pkl

Default output:
    /Users/joonwonlee/Documents/GEMS_DATA/pickle_2024/
        2x2_tco_grid_24_07.pkl

The regular grid coordinates are averaged over every 2x2 grid block. Ozone is
averaged with missing values excluded. Source_Latitude/Source_Longitude are
averaged only over the non-missing ozone locations that contributed to the
block-level ozone mean.
"""

from __future__ import annotations

import argparse
import importlib.util
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GEMS_TCO_SRC = PROJECT_ROOT / "src" / "GEMS_TCO"
DEFAULT_YEAR = 2024
DEFAULT_MONTH = 7
DEFAULT_INPUT_FILENAME = "tco_grid_24_07.pkl"
CORE_COLUMNS = {
    "Latitude",
    "Longitude",
    "ColumnAmountO3",
    "Hours_elapsed",
    "Source_Latitude",
    "Source_Longitude",
}


def load_default_data_root() -> Path:
    config_path = GEMS_TCO_SRC / "configuration.py"
    if not config_path.exists():
        return Path("/Users/joonwonlee/Documents/GEMS_DATA")

    spec = importlib.util.spec_from_file_location("gems_tco_configuration", config_path)
    if spec is None or spec.loader is None:
        return Path("/Users/joonwonlee/Documents/GEMS_DATA")

    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return Path(config.mac_data_load_path)


def yy(year: int) -> str:
    return str(year)[2:]


def fallback_input_filename(year: int, month: int) -> str:
    if year == DEFAULT_YEAR and month == DEFAULT_MONTH:
        return DEFAULT_INPUT_FILENAME
    return f"tco_grid_{yy(year)}_{month:02d}.pkl"


def block_nanmean(blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(blocks)
    counts = valid.sum(axis=(1, 3))
    sums = np.where(valid, blocks, 0.0).sum(axis=(1, 3))
    out = np.full(counts.shape, np.nan, dtype=float)
    np.divide(sums, counts, out=out, where=counts > 0)
    return out, counts


def block_coord_mean(coord_blocks: np.ndarray, valid_o3_blocks: np.ndarray) -> np.ndarray:
    valid = valid_o3_blocks & np.isfinite(coord_blocks)
    counts = valid.sum(axis=(1, 3))
    sums = np.where(valid, coord_blocks, 0.0).sum(axis=(1, 3))
    out = np.full(counts.shape, np.nan, dtype=float)
    np.divide(sums, counts, out=out, where=counts > 0)
    return out


def block_first_nonmissing(values: np.ndarray) -> np.ndarray:
    n_lat2, _, n_lon2, _ = values.shape
    flat = values.transpose(0, 2, 1, 3).reshape(n_lat2 * n_lon2, 4)
    out: list[Any] = []
    for row in flat:
        first = np.nan
        for val in row:
            if pd.notna(val):
                first = val
                break
        out.append(first)
    return np.asarray(out, dtype=object).reshape(n_lat2, n_lon2)


def regular_grid_sort(df: pd.DataFrame, key: str) -> tuple[pd.DataFrame, int, int]:
    required = {"Latitude", "Longitude", "ColumnAmountO3", "Source_Latitude", "Source_Longitude"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{key}: missing required columns: {missing}")

    sorted_df = df.sort_values(
        ["Latitude", "Longitude"],
        ascending=[False, False],
        kind="mergesort",
    ).reset_index(drop=True)

    lats = np.sort(pd.to_numeric(sorted_df["Latitude"], errors="coerce").dropna().unique())[::-1]
    lons = np.sort(pd.to_numeric(sorted_df["Longitude"], errors="coerce").dropna().unique())[::-1]
    n_lat = len(lats)
    n_lon = len(lons)
    expected_rows = n_lat * n_lon
    if expected_rows != len(sorted_df):
        raise ValueError(
            f"{key}: rows={len(sorted_df)} but unique Latitude x Longitude "
            f"gives {n_lat} x {n_lon} = {expected_rows}"
        )

    lat_expected = np.repeat(lats, n_lon)
    lon_expected = np.tile(lons, n_lat)
    lat_actual = pd.to_numeric(sorted_df["Latitude"], errors="coerce").to_numpy(dtype=float)
    lon_actual = pd.to_numeric(sorted_df["Longitude"], errors="coerce").to_numpy(dtype=float)
    if not (np.allclose(lat_actual, lat_expected) and np.allclose(lon_actual, lon_expected)):
        raise ValueError(f"{key}: grid coordinates are not a complete regular row-major grid")

    return sorted_df, n_lat, n_lon


def numeric_blocks(df: pd.DataFrame, col: str, n_lat: int, n_lon: int) -> np.ndarray:
    arr = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float).reshape(n_lat, n_lon)
    even_lat = n_lat - (n_lat % 2)
    even_lon = n_lon - (n_lon % 2)
    return arr[:even_lat, :even_lon].reshape(even_lat // 2, 2, even_lon // 2, 2)


def object_blocks(df: pd.DataFrame, col: str, n_lat: int, n_lon: int) -> np.ndarray:
    arr = df[col].to_numpy(dtype=object).reshape(n_lat, n_lon)
    even_lat = n_lat - (n_lat % 2)
    even_lon = n_lon - (n_lon % 2)
    return arr[:even_lat, :even_lon].reshape(even_lat // 2, 2, even_lon // 2, 2)


def smooth_one_hour(df: pd.DataFrame, key: str) -> tuple[pd.DataFrame, dict[str, int]]:
    sorted_df, n_lat, n_lon = regular_grid_sort(df, key)
    even_lat = n_lat - (n_lat % 2)
    even_lon = n_lon - (n_lon % 2)
    n_lat2 = even_lat // 2
    n_lon2 = even_lon // 2

    lat_blocks = numeric_blocks(sorted_df, "Latitude", n_lat, n_lon)
    lon_blocks = numeric_blocks(sorted_df, "Longitude", n_lat, n_lon)
    o3_blocks = numeric_blocks(sorted_df, "ColumnAmountO3", n_lat, n_lon)
    src_lat_blocks = numeric_blocks(sorted_df, "Source_Latitude", n_lat, n_lon)
    src_lon_blocks = numeric_blocks(sorted_df, "Source_Longitude", n_lat, n_lon)

    grid_lat = lat_blocks.mean(axis=(1, 3))
    grid_lon = lon_blocks.mean(axis=(1, 3))
    o3_mean, o3_counts = block_nanmean(o3_blocks)
    valid_o3 = np.isfinite(o3_blocks)
    src_lat = block_coord_mean(src_lat_blocks, valid_o3)
    src_lon = block_coord_mean(src_lon_blocks, valid_o3)

    out_cols: dict[str, Any] = {}
    for col in sorted_df.columns:
        if col == "Latitude":
            out_cols[col] = grid_lat.ravel()
        elif col == "Longitude":
            out_cols[col] = grid_lon.ravel()
        elif col == "ColumnAmountO3":
            out_cols[col] = o3_mean.ravel()
        elif col == "Source_Latitude":
            out_cols[col] = src_lat.ravel()
        elif col == "Source_Longitude":
            out_cols[col] = src_lon.ravel()
        elif col == "Hours_elapsed":
            mean, _ = block_nanmean(numeric_blocks(sorted_df, col, n_lat, n_lon))
            out_cols[col] = mean.ravel()
        elif pd.api.types.is_numeric_dtype(sorted_df[col]):
            mean, _ = block_nanmean(numeric_blocks(sorted_df, col, n_lat, n_lon))
            out_cols[col] = mean.ravel()
        else:
            first = block_first_nonmissing(object_blocks(sorted_df, col, n_lat, n_lon))
            out_cols[col] = first.ravel()

    out = pd.DataFrame(out_cols, columns=sorted_df.columns)
    stats = {
        "input_rows": len(sorted_df),
        "output_rows": len(out),
        "n_lat": n_lat,
        "n_lon": n_lon,
        "dropped_lat_rows": n_lat - even_lat,
        "dropped_lon_cols": n_lon - even_lon,
        "all_missing_blocks": int((o3_counts == 0).sum()),
    }
    return out, stats


def smooth_pickle(
    input_path: Path,
    output_path: Path,
    overwrite: bool = False,
    max_hours: int | None = None,
) -> Path:
    if not input_path.exists():
        raise FileNotFoundError(f"input pickle not found: {input_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output exists, pass --overwrite to replace: {output_path}")

    print(f"[read] {input_path}")
    with input_path.open("rb") as f:
        hourly_map = pickle.load(f)
    if not isinstance(hourly_map, dict):
        raise TypeError(f"expected dict[str, DataFrame], got {type(hourly_map)!r}")

    smoothed: dict[str, pd.DataFrame] = {}
    first_stats: dict[str, int] | None = None
    keys = sorted(hourly_map)
    if max_hours is not None:
        keys = keys[:max_hours]

    for idx, key in enumerate(keys, start=1):
        out_df, stats = smooth_one_hour(hourly_map[key], key)
        smoothed[key] = out_df
        if first_stats is None:
            first_stats = stats
        if idx == 1 or idx == len(keys) or idx % 25 == 0:
            print(
                f"[{idx:04d}/{len(keys):04d}] {key}: "
                f"{stats['input_rows']} -> {stats['output_rows']} rows, "
                f"all-missing blocks={stats['all_missing_blocks']}"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[write] {output_path}")
    with output_path.open("wb") as f:
        pickle.dump(smoothed, f, protocol=pickle.HIGHEST_PROTOCOL)

    if first_stats is not None:
        print(
            "[summary] "
            f"hours={len(smoothed)} "
            f"grid={first_stats['n_lat']}x{first_stats['n_lon']} "
            f"new_rows_per_hour={first_stats['output_rows']} "
            f"dropped_lat_rows={first_stats['dropped_lat_rows']} "
            f"dropped_lon_cols={first_stats['dropped_lon_cols']}"
        )
    print("[done]")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=load_default_data_root())
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR)
    parser.add_argument("--month", type=int, default=DEFAULT_MONTH)
    parser.add_argument(
        "--input-filename",
        default=None,
        help="Input file under pickle_YEAR. Defaults to July 2024 -3to2,121to131.",
    )
    parser.add_argument(
        "--output-filename",
        default=None,
        help="Output filename. Defaults to 2x2_<input-filename>.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to the input pickle_YEAR directory.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-hours", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved paths and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pickle_dir = args.data_root / f"pickle_{args.year}"
    input_filename = args.input_filename or fallback_input_filename(args.year, args.month)
    output_filename = args.output_filename or f"2x2_{input_filename}"
    input_path = pickle_dir / input_filename
    output_dir = args.output_dir or pickle_dir
    output_path = output_dir / output_filename

    print(f"data_root={args.data_root}")
    print(f"input={input_path}")
    print(f"output={output_path}")
    if args.dry_run:
        return

    smooth_pickle(
        input_path=input_path,
        output_path=output_path,
        overwrite=args.overwrite,
        max_hours=args.max_hours,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise
