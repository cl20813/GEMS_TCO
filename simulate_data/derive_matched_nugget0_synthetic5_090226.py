#!/usr/bin/env python3
"""Derive matched nugget=0 days from the existing nugget=1 simulation asset.

The original generator records the NumPy RNG seed used for every hourly
nugget realization.  This script regenerates that exact noise vector and
subtracts it from the stored real-location simulation before applying the
original one-to-one griddification.  The latent Matérn field, observation
locations, dates, and all non-nugget truth parameters therefore remain exactly
the same as in the source asset.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from generate_july_st_circulant_real_locations_2022_2025 import (  # noqa: E402
    griddify_one_to_one,
    parse_gems_hour_key,
)


SELECTED_DATES = (
    "2023-07-04",
    "2023-07-29",
    "2024-07-13",
    "2024-07-19",
    "2025-07-06",
)


def parse_args() -> argparse.Namespace:
    repo = Path("/Users/joonwonlee/Documents/GEMS_TCO-1")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(
            "/Users/joonwonlee/Documents/GEMS_DATA/simulation/"
            "july_st_circulant_realpattern_smooth0p5"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo
        / "outputs/sim_data/"
        "july_st_circulant_realpattern_smooth0p5_nugget0_matched5_090226",
    )
    parser.add_argument("--dates", default=",".join(SELECTED_DATES))
    return parser.parse_args()


def date_from_key(key: str) -> str:
    stamp = parse_gems_hour_key(str(key))
    if stamp is None:
        raise ValueError(f"Unrecognized GEMS hour key: {key}")
    return stamp.strftime("%Y-%m-%d")


def reconstruct_year(
    year: int,
    selected_dates: set[str],
    source_root: Path,
    output_root: Path,
) -> dict[str, object]:
    source_dir = source_root / f"{year}_july_st_circulant"
    prefix = f"sim_july{year}_st_circulant"
    real_path = source_dir / f"{prefix}_real_locations.pkl"
    manifest_path = source_dir / f"{prefix}_manifest.csv"
    truth_path = source_dir / f"{prefix}_truth.json"
    if not real_path.is_file() or not manifest_path.is_file() or not truth_path.is_file():
        raise FileNotFoundError(f"Incomplete source asset under {source_dir}")

    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    source_nugget = float(truth["nugget"])
    if source_nugget <= 0.0:
        raise ValueError(f"Expected positive source nugget, got {source_nugget}")
    real_map = pd.read_pickle(real_path)
    manifest = pd.read_csv(manifest_path)
    manifest_by_key = manifest.set_index("hour_key", verify_integrity=True)
    chosen_keys = sorted(key for key in real_map if date_from_key(key) in selected_dates)
    expected_dates = sorted(date for date in selected_dates if date.startswith(f"{year}-"))
    observed_dates = sorted({date_from_key(key) for key in chosen_keys})
    if observed_dates != expected_dates:
        raise RuntimeError(
            f"Selected dates mismatch for {year}: observed={observed_dates}, expected={expected_dates}"
        )
    if len(chosen_keys) != 8 * len(expected_dates):
        raise RuntimeError(f"Expected {8 * len(expected_dates)} keys for {year}, got {len(chosen_keys)}")

    zero_real_map: dict[str, pd.DataFrame] = {}
    zero_grid_map: dict[str, pd.DataFrame] = {}
    validation_rows: list[dict[str, object]] = []
    for key in chosen_keys:
        frame = real_map[key]
        y_with_nugget = pd.to_numeric(
            frame["ColumnAmountO3"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        source_lat = pd.to_numeric(
            frame["Source_Latitude"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        source_lon = pd.to_numeric(
            frame["Source_Longitude"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        valid = np.isfinite(source_lat) & np.isfinite(source_lon)
        seed = int(manifest_by_key.loc[key, "seed_nugget"])
        rng = np.random.default_rng(seed)
        epsilon = rng.normal(0.0, math.sqrt(source_nugget), size=int(valid.sum()))
        y_without_nugget = y_with_nugget.copy()
        y_without_nugget[valid] -= epsilon
        max_reconstruction_error = float(
            np.max(
                np.abs(
                    (y_without_nugget[valid] + epsilon)
                    - y_with_nugget[valid]
                )
            )
        )
        if max_reconstruction_error > 1e-12:
            raise RuntimeError(
                f"Nugget reconstruction failed for {key}: {max_reconstruction_error}"
            )

        zero_real = frame.copy()
        zero_real["ColumnAmountO3"] = y_without_nugget
        zero_grid, grid_diag = griddify_one_to_one(
            frame, y_without_nugget, source_lat, source_lon, valid
        )
        hour_index = int(manifest_by_key.loc[key, "hour_index"])
        if "Hours_elapsed" in zero_grid.columns:
            zero_grid["Hours_elapsed"] = float(hour_index)
        zero_real_map[key] = zero_real
        zero_grid_map[key] = zero_grid
        validation_rows.append(
            {
                "year": year,
                "date": date_from_key(key),
                "hour_key": key,
                "hour_index": hour_index,
                "seed_nugget": seed,
                "source_nugget_removed": source_nugget,
                "n_source_valid": int(valid.sum()),
                "n_grid_finite": int(
                    pd.to_numeric(
                        zero_grid["ColumnAmountO3"], errors="coerce"
                    ).notna().sum()
                ),
                "max_reconstruction_error": max_reconstruction_error,
                **{f"grid_{name}": value for name, value in grid_diag.items()},
            }
        )

    output_dir = output_root / f"{year}_july_st_circulant"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(zero_real_map, output_dir / f"{prefix}_real_locations.pkl")
    pd.to_pickle(zero_grid_map, output_dir / f"{prefix}_gridded.pkl")
    pd.DataFrame(validation_rows).to_csv(
        output_dir / f"{prefix}_matched_nugget0_validation.csv",
        index=False,
        float_format="%.10g",
    )
    derived_truth = dict(truth)
    derived_truth.update(
        {
            "nugget": 0.0,
            "n_hours": len(chosen_keys),
            "selected_dates": expected_dates,
            "matched_latent_field_source": str(real_path),
            "derivation": (
                "Exact removal of the stored N(0,1) nugget using each "
                "hour's recorded seed_nugget, followed by the original "
                "one-to-one griddification."
            ),
            "source_truth_nugget": source_nugget,
            "maximum_reconstruction_error": max(
                float(row["max_reconstruction_error"])
                for row in validation_rows
            ),
        }
    )
    (output_dir / f"{prefix}_truth.json").write_text(
        json.dumps(derived_truth, indent=2), encoding="utf-8"
    )
    return {
        "year": year,
        "dates": expected_dates,
        "n_hours": len(chosen_keys),
        "max_reconstruction_error": derived_truth["maximum_reconstruction_error"],
    }


def main() -> None:
    args = parse_args()
    selected_dates = {
        str(pd.Timestamp(value.strip()).date())
        for value in str(args.dates).split(",")
        if value.strip()
    }
    if not selected_dates:
        raise ValueError("At least one selected date is required")
    years = sorted({int(date[:4]) for date in selected_dates})
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = [
        reconstruct_year(
            year, selected_dates, Path(args.source_root), Path(args.output_root)
        )
        for year in years
    ]
    (Path(args.output_root) / "DERIVATION_MANIFEST.json").write_text(
        json.dumps(
            {
                "source_root": str(args.source_root),
                "output_root": str(args.output_root),
                "selected_dates": sorted(selected_dates),
                "truth_nugget": 0.0,
                "years": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(rows, indent=2), flush=True)
    print(f"Saved matched nugget=0 asset: {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
