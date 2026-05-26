#!/usr/bin/env python3
"""Build monthly GEMS CSV and orbit_map pickle files for arbitrary bounds.

This script replaces the old step2 notebook as the reproducible entry point.

Default/narrow domain:
    --bounds=-3,2,121,131
    writes data_YY_MM_..._N-32_E121131.csv and orbit_mapYY_MM.pkl

Expanded domain:
    --bounds=-3,7,111,131
    writes data_YY_MM_..._N-37_E111131.csv and
    orbit_map_lat-3to7_lon111to131_YY_MM.pkl
"""

from __future__ import annotations

import argparse
import importlib.util
import pickle
import sys
from pathlib import Path

import pandas as pd


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


def month_day_token(year: int, month: int) -> str:
    if month == 2 and year == 2024:
        return "0129"
    if month == 2:
        return "0128"
    return "0131" if month in {1, 3, 5, 7, 8, 10, 12} else "0130"


def csv_filename(year: int, month: int, bounds: tuple[float, float, float, float]) -> str:
    yy = str(year)[2:]
    day_str = month_day_token(year, month)
    lat_start, lat_end, lon_start, lon_end = bounds
    return (
        f"data_{yy}_{month:02d}_{day_str}_"
        f"N{format_bound_token(lat_start)}{format_bound_token(lat_end)}_"
        f"E{format_bound_token(lon_start)}{format_bound_token(lon_end)}.csv"
    )


def orbit_map_filename(year: int, month: int, bounds: tuple[float, float, float, float]) -> str:
    yy = str(year)[2:]
    if tuple(bounds) == DEFAULT_BOUNDS:
        return f"orbit_map{yy}_{month:02d}.pkl"
    return f"orbit_map_{bounds_tag(bounds)}_{yy}_{month:02d}.pkl"


def parse_bounds(text: str) -> tuple[float, float, float, float]:
    vals = [float(x.strip()) for x in text.split(",")]
    if len(vals) != 4:
        raise argparse.ArgumentTypeError("bounds must be lat_min,lat_max,lon_min,lon_max")
    return tuple(vals)  # type: ignore[return-value]


def build_csv(
    base_path: Path,
    portable_disk: str,
    year: int,
    month: int,
    bounds: tuple[float, float, float, float],
    overwrite: bool,
) -> Path:
    out_dir = base_path / f"data_{year}"
    out_path = out_dir / csv_filename(year, month, bounds)
    if out_path.exists() and not overwrite:
        print(f"[skip csv] exists: {out_path}")
        return out_path

    lat_start, lat_end, lon_start, lon_end = bounds
    filelist_instance = dmbh.file_path_list(year, month, portable_disk)
    file_paths_list = filelist_instance.file_names_july24()
    instance = dmbh.MonthAggregatedCSV(lat_start, lat_end, lon_start, lon_end)
    good_quality = instance.aggregate_july24tocsv(file_paths_list)
    out_dir.mkdir(parents=True, exist_ok=True)
    good_quality.to_csv(out_path, index=False)
    print(f"[saved csv] {out_path} rows={len(good_quality)}")
    return out_path


def group_data_by_orbits(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = df.copy()
    out["Orbit"] = out["Time"].astype(str).str[0:16]
    orbit_map: dict[str, pd.DataFrame] = {}
    for orbit in out["Orbit"].dropna().unique():
        key = f"y{orbit[2:4]}m{int(orbit[5:7]):02d}day{int(orbit[8:10]):02d}_hm{orbit[11:16]}"
        orbit_map[key] = out.loc[out["Orbit"] == orbit].reset_index(drop=True)
    return orbit_map


def build_orbit_map(
    base_path: Path,
    year: int,
    month: int,
    bounds: tuple[float, float, float, float],
    overwrite: bool,
) -> Path:
    in_path = base_path / f"data_{year}" / csv_filename(year, month, bounds)
    out_dir = base_path / f"pickle_{year}"
    out_path = out_dir / orbit_map_filename(year, month, bounds)
    if not in_path.exists():
        raise FileNotFoundError(f"CSV not found: {in_path}")
    if out_path.exists() and not overwrite:
        print(f"[skip orbit] exists: {out_path}")
        return out_path

    print(f"[read csv] {in_path}")
    df = pd.read_csv(in_path)
    required = {"Latitude", "Longitude", "Time", "ColumnAmountO3"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{in_path} missing required columns: {sorted(missing)}")
    orbit_map = group_data_by_orbits(df)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(orbit_map, f, protocol=pickle.HIGHEST_PROTOCOL)
    n_rows = sum(len(v) for v in orbit_map.values())
    print(f"[saved orbit] {out_path} hours={len(orbit_map)} rows={n_rows}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-path", type=Path, default=Path(config.mac_data_load_path))
    parser.add_argument("--portable-disk", default=getattr(config, "portable_disk_path", "/Volumes/Backup Plus/GEMS_UNZIPPED/"))
    parser.add_argument("--years", type=int, nargs="+", default=DEFAULT_YEARS)
    parser.add_argument("--months", type=int, nargs="+", default=DEFAULT_MONTHS)
    parser.add_argument("--bounds", type=parse_bounds, default=DEFAULT_BOUNDS)
    parser.add_argument("--make-csv", action="store_true", help="Rebuild CSV from netCDF files.")
    parser.add_argument("--make-orbit-map", action="store_true", help="Build orbit_map from existing CSV.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.make_csv and not args.make_orbit_map:
        args.make_orbit_map = True

    print(f"base_path={args.base_path}")
    print(f"years={args.years} months={args.months} bounds={args.bounds}")
    print(f"make_csv={args.make_csv} make_orbit_map={args.make_orbit_map}")
    for year in args.years:
        for month in args.months:
            if args.make_csv:
                build_csv(args.base_path, args.portable_disk, year, month, args.bounds, args.overwrite)
            if args.make_orbit_map:
                build_orbit_map(args.base_path, year, month, args.bounds, args.overwrite)


if __name__ == "__main__":
    main()
