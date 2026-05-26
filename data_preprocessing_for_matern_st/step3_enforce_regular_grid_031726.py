#!/usr/bin/env python3
"""Build regular-grid tco_grid pickle files from orbit_map pickle files.

Default/narrow domain:
    --bounds=-3,2,121,131
    reads orbit_mapYY_MM.pkl and writes tco_grid_YY_MM.pkl

Expanded domain:
    --bounds=-3,7,111,131
    reads orbit_map_lat-3to7_lon111to131_YY_MM.pkl and writes
    tco_grid_lat-3to7_lon111to131_YY_MM.pkl
"""

from __future__ import annotations

import argparse
import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import KDTree


ROOT = Path("/Users/joonwonlee/Documents/GEMS_TCO-1")
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

GEMS_TCO_SRC = SRC / "GEMS_TCO"
spec = importlib.util.spec_from_file_location("gems_tco_configuration", GEMS_TCO_SRC / "configuration.py")
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

from GEMS_TCO import data_preprocess as dmbh  # noqa: E402


DEFAULT_YEARS = [2022, 2023, 2024, 2025]
DEFAULT_MONTHS = [7]
DEFAULT_BOUNDS = (-3.0, 2.0, 121.0, 131.0)
DEFAULT_STEP_SIZES = (0.044, 0.063)


def format_bound_token(value: float) -> str:
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def bounds_tag(bounds: tuple[float, float, float, float]) -> str:
    lat_start, lat_end, lon_start, lon_end = bounds
    return (
        f"lat{format_bound_token(lat_start)}to{format_bound_token(lat_end)}_"
        f"lon{format_bound_token(lon_start)}to{format_bound_token(lon_end)}"
    )


def orbit_map_filename(year: int, month: int, bounds: tuple[float, float, float, float]) -> str:
    yy = str(year)[2:]
    if tuple(bounds) == DEFAULT_BOUNDS:
        return f"orbit_map{yy}_{month:02d}.pkl"
    return f"orbit_map_{bounds_tag(bounds)}_{yy}_{month:02d}.pkl"


def tco_grid_filename(year: int, month: int, bounds: tuple[float, float, float, float]) -> str:
    yy = str(year)[2:]
    if tuple(bounds) == DEFAULT_BOUNDS:
        return f"tco_grid_{yy}_{month:02d}.pkl"
    return f"tco_grid_{bounds_tag(bounds)}_{yy}_{month:02d}.pkl"


def forward_bin_for_whittle(
    loaded_map: dict[str, pd.DataFrame],
    base_center_points: np.ndarray,
    step_lat: float,
    step_lon: float,
    v_drift_lon: float = -0.0048,
) -> dict[str, pd.DataFrame]:
    coarse_cen_map: dict[str, pd.DataFrame] = {}
    sorted_keys = sorted(loaded_map.keys())
    lat_thresh = step_lat / 2.0
    lon_thresh = step_lon / 2.0

    if isinstance(base_center_points, pd.DataFrame):
        base_center_points = base_center_points.iloc[:, 0:2].to_numpy(dtype=float)
    else:
        base_center_points = np.asarray(base_center_points, dtype=float)
    n_grid = len(base_center_points)

    for i, key in enumerate(sorted_keys):
        t_idx = i % 8
        df_raw = loaded_map[key]
        if len(df_raw) == 0:
            continue

        shifted_grid = base_center_points + np.array([0.0, t_idx * v_drift_lon])
        raw_coords = df_raw[["Latitude", "Longitude"]].to_numpy(dtype=float)

        value_cols = [
            c for c in df_raw.columns
            if c not in ("Latitude", "Longitude", "FinalAlgorithmFlags", "Orbit")
            and pd.api.types.is_numeric_dtype(df_raw[c])
        ]

        grid_tree = KDTree(shifted_grid)
        _, grid_indices = grid_tree.query(raw_coords)
        lat_diffs = np.abs(raw_coords[:, 0] - shifted_grid[grid_indices, 0])
        lon_diffs = np.abs(raw_coords[:, 1] - shifted_grid[grid_indices, 1])
        valid = (lat_diffs <= lat_thresh) & (lon_diffs <= lon_thresh)

        df_valid = df_raw.loc[valid].copy()
        df_valid["_grid_idx"] = grid_indices[valid]
        df_valid["_src_lat"] = raw_coords[valid, 0]
        df_valid["_src_lon"] = raw_coords[valid, 1]
        df_valid["_dist"] = np.hypot(
            (raw_coords[valid, 0] - shifted_grid[grid_indices[valid], 0]) / lat_thresh,
            (raw_coords[valid, 1] - shifted_grid[grid_indices[valid], 1]) / lon_thresh,
        )

        idx_nearest = df_valid.groupby("_grid_idx")["_dist"].idxmin()
        df_nearest = df_valid.loc[idx_nearest].set_index("_grid_idx")

        df_result = pd.DataFrame(
            np.nan,
            index=range(n_grid),
            columns=value_cols + ["Source_Latitude", "Source_Longitude"],
        )
        df_result.loc[df_nearest.index, value_cols] = df_nearest[value_cols].to_numpy()
        df_result.loc[df_nearest.index, "Source_Latitude"] = df_nearest["_src_lat"].to_numpy()
        df_result.loc[df_nearest.index, "Source_Longitude"] = df_nearest["_src_lon"].to_numpy()
        df_result.insert(0, "Latitude", base_center_points[:, 0])
        df_result.insert(1, "Longitude", base_center_points[:, 1])
        coarse_cen_map[key] = df_result

    return coarse_cen_map


def parse_bounds(text: str) -> tuple[float, float, float, float]:
    vals = [float(x.strip()) for x in text.split(",")]
    if len(vals) != 4:
        raise argparse.ArgumentTypeError("bounds must be lat_min,lat_max,lon_min,lon_max")
    return tuple(vals)  # type: ignore[return-value]


def parse_steps(text: str) -> tuple[float, float]:
    vals = [float(x.strip()) for x in text.split(",")]
    if len(vals) != 2:
        raise argparse.ArgumentTypeError("steps must be step_lat,step_lon")
    return tuple(vals)  # type: ignore[return-value]


def build_one(
    base_path: Path,
    year: int,
    month: int,
    bounds: tuple[float, float, float, float],
    step_sizes: tuple[float, float],
    overwrite: bool,
) -> Path:
    pickle_dir = base_path / f"pickle_{year}"
    in_path = pickle_dir / orbit_map_filename(year, month, bounds)
    out_path = pickle_dir / tco_grid_filename(year, month, bounds)
    if not in_path.exists():
        raise FileNotFoundError(f"orbit_map not found: {in_path}")
    if out_path.exists() and not overwrite:
        print(f"[skip] exists: {out_path}")
        return out_path

    lat_start, lat_end, lon_start, lon_end = bounds
    step_lat, step_lon = step_sizes
    instance = dmbh.center_matching_hour(None, lat_start, lat_end, lon_start, lon_end)
    center_points = instance.make_center_points_wo_calibration(step_lat=step_lat, step_lon=step_lon)

    print(f"[read] {in_path}")
    with open(in_path, "rb") as f:
        loaded_map = pickle.load(f)

    coarse_cen_map = forward_bin_for_whittle(loaded_map, center_points, step_lat, step_lon)
    with open(out_path, "wb") as f:
        pickle.dump(coarse_cen_map, f, protocol=pickle.HIGHEST_PROTOCOL)

    first_key = sorted(coarse_cen_map)[0]
    first = coarse_cen_map[first_key]
    finite = first.dropna(subset=["ColumnAmountO3"])
    print(
        f"[saved] {out_path} | hours={len(coarse_cen_map)} grid_rows={len(first)} "
        f"first_hour_finite={len(finite)} "
        f"finite_lat=[{finite['Latitude'].min():.3f},{finite['Latitude'].max():.3f}] "
        f"finite_lon=[{finite['Longitude'].min():.3f},{finite['Longitude'].max():.3f}]"
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-path", type=Path, default=Path(config.mac_data_load_path))
    parser.add_argument("--years", type=int, nargs="+", default=DEFAULT_YEARS)
    parser.add_argument("--months", type=int, nargs="+", default=DEFAULT_MONTHS)
    parser.add_argument("--bounds", type=parse_bounds, default=DEFAULT_BOUNDS)
    parser.add_argument("--steps", type=parse_steps, default=DEFAULT_STEP_SIZES)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    print(f"base_path={args.base_path}")
    print(f"years={args.years} months={args.months} bounds={args.bounds} steps={args.steps}")
    for year in args.years:
        for month in args.months:
            build_one(args.base_path, year, month, args.bounds, args.steps, args.overwrite)


if __name__ == "__main__":
    main()
