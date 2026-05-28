#!/usr/bin/env python3
"""Missing/usable comparison for low-latitude West Pacific candidate boxes."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset


RAW_ROOT = Path("/Volumes/Backup Plus/GEMS_UNZIPPED/2022070131")
OUT_DIR = Path("/Users/joonwonlee/Documents/GEMS_TCO-1/GEMS_TCO_EDA")
SUMMARY_CSV = OUT_DIR / "candidate_missing_2022_july_micronesia_052826_summary.csv"
DAILY_CSV = OUT_DIR / "candidate_missing_2022_july_micronesia_052826_daily.csv"
HOURLY_CSV = OUT_DIR / "candidate_missing_2022_july_micronesia_052826_hourly.csv"

CANDIDATES = [
    ("lat1to6_lon135to145", 1.0, 6.0, 135.0, 145.0, 722_626),
    ("lat1to6_lon129to139", 1.0, 6.0, 129.0, 139.0, 3_614_698),
]


def as_float_array(var) -> np.ndarray:
    return np.ma.asarray(var[:]).filled(np.nan).astype(float, copy=False).ravel()


def parse_slot(path: Path) -> tuple[int, int, int]:
    match = re.match(r"202207(\d{2})_(\d{2})(\d{2})\.nc", path.name)
    if not match:
        raise ValueError(f"Unexpected filename: {path.name}")
    return tuple(int(x) for x in match.groups())


def main() -> None:
    rows: list[dict[str, object]] = []
    for path in sorted(RAW_ROOT.glob("202207*_*.nc")):
        day, hour, minute = parse_slot(path)
        with Dataset(path) as ds:
            geo = ds.groups["Geolocation Fields"]
            data = ds.groups["Data Fields"]
            lat = as_float_array(geo.variables["Latitude"])
            lon = as_float_array(geo.variables["Longitude"])
            time = as_float_array(geo.variables["Time"])
            o3 = as_float_array(data.variables["ColumnAmountO3"])
            flag = as_float_array(data.variables["FinalAlgorithmFlags"])

        finite_location = np.isfinite(lat) & np.isfinite(lon)
        if not np.isfinite(time).any():
            finite_location = np.zeros_like(finite_location, dtype=bool)
        finite_o3_flag = np.isfinite(o3) & np.isfinite(flag)

        for name, lat_min, lat_max, lon_min, lon_max, _ in CANDIDATES:
            bbox = (
                finite_location
                & (lat >= lat_min)
                & (lat <= lat_max)
                & (lon >= lon_min)
                & (lon <= lon_max)
            )
            raw_n = int(bbox.sum())
            nanfill_n = int((bbox & (~finite_o3_flag | (o3 >= 1000))).sum())
            good_n = int((bbox & finite_o3_flag & (o3 < 1000) & np.isin(flag, [0, 2])).sum())
            rows.append(
                {
                    "candidate": name,
                    "date": f"2022-07-{day:02d}",
                    "hour": hour,
                    "minute": minute,
                    "raw_bbox_pixels": raw_n,
                    "nan_or_fill_pixels": nanfill_n,
                    "step2_good_pixels": good_n,
                    "unusable_for_step2_pixels": raw_n - good_n,
                }
            )

    slots = pd.DataFrame(rows)
    summary = slots.groupby("candidate", as_index=False).agg(
        raw_source_slots=("hour", "size"),
        effective_slots_raw_positive=("raw_bbox_pixels", lambda s: int((s > 0).sum())),
        raw_bbox_pixels=("raw_bbox_pixels", "sum"),
        nan_or_fill_pixels=("nan_or_fill_pixels", "sum"),
        step2_good_pixels=("step2_good_pixels", "sum"),
        unusable_for_step2_pixels=("unusable_for_step2_pixels", "sum"),
    )
    summary["nan_or_fill_pct"] = 100 * summary.nan_or_fill_pixels / summary.raw_bbox_pixels
    summary["step2_good_pct"] = 100 * summary.step2_good_pixels / summary.raw_bbox_pixels
    summary["unusable_for_step2_pct"] = (
        100 * summary.unusable_for_step2_pixels / summary.raw_bbox_pixels
    )
    step2_lookup = {name: step2_rows for name, *_rest, step2_rows in CANDIDATES}
    summary["step2_rows_from_log"] = summary["candidate"].map(step2_lookup)
    summary["step2_count_delta"] = summary["step2_good_pixels"] - summary["step2_rows_from_log"]

    daily = slots.groupby(["candidate", "date"], as_index=False).agg(
        raw_bbox_pixels=("raw_bbox_pixels", "sum"),
        step2_good_pixels=("step2_good_pixels", "sum"),
        unusable_for_step2_pixels=("unusable_for_step2_pixels", "sum"),
    )
    daily["unusable_for_step2_pct"] = (
        100 * daily.unusable_for_step2_pixels / daily.raw_bbox_pixels
    )
    daily["step2_good_pct"] = 100 * daily.step2_good_pixels / daily.raw_bbox_pixels

    hourly = slots.groupby(["candidate", "hour"], as_index=False).agg(
        n_source_slots=("hour", "size"),
        n_effective_slots=("raw_bbox_pixels", lambda s: int((s > 0).sum())),
        raw_bbox_pixels=("raw_bbox_pixels", "sum"),
        step2_good_pixels=("step2_good_pixels", "sum"),
        unusable_for_step2_pixels=("unusable_for_step2_pixels", "sum"),
    )
    hourly["unusable_for_step2_pct"] = (
        100 * hourly.unusable_for_step2_pixels / hourly.raw_bbox_pixels
    )
    hourly["step2_good_pct"] = 100 * hourly.step2_good_pixels / hourly.raw_bbox_pixels

    summary.to_csv(SUMMARY_CSV, index=False)
    daily.to_csv(DAILY_CSV, index=False)
    hourly.to_csv(HOURLY_CSV, index=False)

    print("SUMMARY")
    print(summary.to_string(index=False))
    print("\nHOURLY")
    print(hourly.to_string(index=False))
    print("\nWORST DAILY")
    print(
        daily.sort_values(["candidate", "unusable_for_step2_pct"], ascending=[True, False])
        .groupby("candidate")
        .head(5)
        .to_string(index=False)
    )
    print(f"\nWrote {SUMMARY_CSV}")
    print(f"Wrote {DAILY_CSV}")
    print(f"Wrote {HOURLY_CSV}")


if __name__ == "__main__":
    main()
