#!/usr/bin/env python3
"""Time FFT empirical and reverse-L stride-2 advection seeds on real GEMS data.

This benchmark uses the native 0.044 x 0.063 degree regular grid over
latitude [-3, 2] and longitude [121, 131]: 114 x 159 = 18,126 cells per
hour, eight hours per day.  No sub-grid interpolation is used.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REPO = Path("/Users/joonwonlee/Documents/GEMS_TCO-1")
SRC = REPO / "src"
HERE = Path(__file__).resolve().parent
for path in (REPO, SRC, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from GEMS_TCO.data_loader import load_data_dynamic_processed
from synthetic_advection_initializer_benchmark_corridor432_081526 import DayAsset
from synthetic_m3_reverse_l_initialization_benchmark_corridor432_083126 import (
    reverse_l_refine,
    run_fft_seed,
)
from compare_advection_seed_methods_corridor432_081626 import fit_final_corridor


DEFAULT_NUISANCE = {
    "sigmasq": 13.059,
    "range_lat": 0.20,
    "range_lon": 0.25,
    "range_time": 1.50,
    "advec_lat": 0.0,
    "advec_lon": 0.0,
    "nugget": 0.247,
}


def parse_ints(text: str) -> list[int]:
    return [int(token.strip()) for token in str(text).split(",") if token.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/Users/joonwonlee/Documents/GEMS_DATA"))
    parser.add_argument("--output-root", type=Path, default=REPO / "outputs/day/estimates/realdata_fft_reverseL_timing_083126")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--month", type=int, default=7)
    parser.add_argument("--days", default="0,1,2,3,4", help="Zero-based day indices.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fft-repeats", type=int, default=5)
    parser.add_argument("--smooth", type=float, default=0.5)
    parser.add_argument("--empirical-max-lat-offset", type=int, default=20)
    parser.add_argument("--empirical-max-lon-offset", type=int, default=20)
    parser.add_argument("--empirical-min-pair-count", type=int, default=1000)
    parser.add_argument("--empirical-smooth-bandwidth-deg", type=float, default=0.063)
    parser.add_argument("--subgrid-max-condition-number", type=float, default=100.0)
    parser.add_argument("--reverse-l-head-right-cols", type=int, default=0)
    parser.add_argument("--reverse-l-above-count", type=int, default=2)
    parser.add_argument("--reverse-l-right-col-count", type=int, default=3)
    parser.add_argument("--reverse-l-per-lag-count", type=int, default=14)
    parser.add_argument("--reverse-l-lag-count", type=int, default=2)
    parser.add_argument("--reverse-l-target-chunk-size", type=int, default=1024)
    parser.add_argument("--reverse-l-max-eval", type=int, default=5)
    parser.add_argument("--lbfgs-lr", type=float, default=1.0)
    parser.add_argument("--lbfgs-history", type=int, default=5)
    parser.add_argument(
        "--run-full-fit",
        action="store_true",
        help="Also time one identical full seven-parameter corridor fit per seed.",
    )
    parser.add_argument("--daily-stride", type=int, default=2)
    parser.add_argument("--target-chunk-size", type=int, default=128)
    parser.add_argument("--min-target-points", type=int, default=1)
    parser.add_argument("--final-lbfgs-steps", type=int, default=1)
    parser.add_argument("--final-lbfgs-eval", type=int, default=20)
    parser.add_argument("--final-lbfgs-history", type=int, default=10)
    parser.add_argument("--grad-tol", type=float, default=1e-5)
    parser.add_argument("--suppress-fit-prints", action="store_true", default=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    days = parse_ints(args.days)
    device = torch.device(args.device)
    args.output_root.mkdir(parents=True, exist_ok=True)
    loader = load_data_dynamic_processed(str(args.data_root))

    load_started = time.perf_counter()
    df_map, _, _, monthly_mean = loader.load_maxmin_ordered_data_bymonthyear(
        lat_lon_resolution=[1, 1],
        mm_cond_number=1,
        years_=[str(args.year)],
        months_=[int(args.month)],
        lat_range=[-3.0, 2.0],
        lon_range=[121.0, 131.0],
        is_whittle=True,
    )
    monthly_load_s = time.perf_counter() - load_started
    keys = sorted(df_map)
    grid_coords = df_map[keys[0]][["Latitude", "Longitude"]].to_numpy(dtype=np.float64)
    unique_lat = np.sort(np.unique(np.round(grid_coords[:, 0], 6)))
    unique_lon = np.sort(np.unique(np.round(grid_coords[:, 1], 6)))
    lat_step = float(np.median(np.diff(unique_lat)))
    lon_step = float(np.median(np.diff(unique_lon)))

    rows: list[dict] = []
    fft_repeat_rows: list[dict] = []
    for day_idx in days:
        day_label = f"{args.year}-{args.month:02d}-{day_idx + 1:02d}"
        day_started = time.perf_counter()
        day_map, _ = loader.load_working_data(
            df_map,
            monthly_mean,
            [day_idx * 8, (day_idx + 1) * 8],
            ord_mm=None,
            dtype=torch.double,
            keep_ori=True,
        )
        day_build_s = time.perf_counter() - day_started
        if len(day_map) != 8:
            raise RuntimeError(f"Expected eight hours for {day_label}, got {len(day_map)}")
        total = sum(int(value.shape[0]) for value in day_map.values())
        valid = sum(int(torch.isfinite(value[:, 2]).sum().item()) for value in day_map.values())
        asset = DayAsset(
            year=int(args.year),
            month=int(args.month),
            day_idx=int(day_idx),
            day=day_label,
            source_map=day_map,
            grid_coords=grid_coords,
            n_valid=valid,
            n_total=total,
        )

        fft_results = []
        for repeat in range(int(args.fft_repeats)):
            result, _ = run_fft_seed(asset, args, robust=False, subgrid=False)
            fft_results.append(result)
            fft_repeat_rows.append(
                {
                    "day": day_label,
                    "day_idx": day_idx,
                    "repeat": repeat,
                    "fft_s": float(result["search_s"]),
                    "seed_lat": float(result["seed_lat"]),
                    "seed_lon": float(result["seed_lon"]),
                }
            )
        fft_times = np.asarray([value["search_s"] for value in fft_results], dtype=float)
        fft_seed = fft_results[-1]

        nuisance = dict(DEFAULT_NUISANCE)
        nuisance["advec_lat"] = float(fft_seed["seed_lat"])
        nuisance["advec_lon"] = float(fft_seed["seed_lon"])
        reverse_seed, reverse_diag = reverse_l_refine(
            asset,
            fft_seed,
            nuisance,
            args,
            device,
            spatial_stride=2,
        )
        incremental_reverse_s = float(
            reverse_diag["reverse_l_precompute_s"] + reverse_diag["reverse_l_refine_s"]
        )
        reverse_total_s = float(np.median(fft_times) + incremental_reverse_s)
        movement = float(
            np.hypot(
                reverse_seed["seed_lat"] - fft_seed["seed_lat"],
                reverse_seed["seed_lon"] - fft_seed["seed_lon"],
            )
        )
        row = {
            "day": day_label,
            "day_idx": day_idx,
            "grid_n_lat": len(unique_lat),
            "grid_n_lon": len(unique_lon),
            "points_per_hour": len(grid_coords),
            "n_hours": len(day_map),
            "rows_total": total,
            "rows_valid": valid,
            "valid_rate": valid / total,
            "lat_step": lat_step,
            "lon_step": lon_step,
            "day_build_s": day_build_s,
            "fft_first_s": float(fft_times[0]),
            "fft_median_s": float(np.median(fft_times)),
            "fft_min_s": float(np.min(fft_times)),
            "fft_max_s": float(np.max(fft_times)),
            "fft_seed_lat": float(fft_seed["seed_lat"]),
            "fft_seed_lon": float(fft_seed["seed_lon"]),
            "reverse_precompute_s": float(reverse_diag["reverse_l_precompute_s"]),
            "reverse_refine_s": float(reverse_diag["reverse_l_refine_s"]),
            "reverse_incremental_s": incremental_reverse_s,
            "reverse_total_seed_s": reverse_total_s,
            "reverse_closure_calls": int(reverse_diag["reverse_l_closure_calls"]),
            "reverse_points_per_hour": int(reverse_diag["reverse_l_points_per_hour"]),
            "reverse_n_tails": int(reverse_diag["reverse_l_n_tails"]),
            "reverse_seed_lat": float(reverse_seed["seed_lat"]),
            "reverse_seed_lon": float(reverse_seed["seed_lon"]),
            "reverse_seed_movement": movement,
            "reverse_over_fft_ratio": reverse_total_s / float(np.median(fft_times)),
        }
        if args.run_full_fit:
            for label, seed, seed_s in (
                ("fft", fft_seed, float(np.median(fft_times))),
                ("reverse_l", reverse_seed, reverse_total_s),
            ):
                final = fit_final_corridor(
                    day_map=day_map,
                    regular_grid_coords=grid_coords,
                    seed_lat=float(seed["seed_lat"]),
                    seed_lon=float(seed["seed_lon"]),
                    args=args,
                    device=device,
                )
                row[f"{label}_full_precompute_s"] = float(final["precompute_s"])
                row[f"{label}_full_fit_s"] = float(final["fit_s"])
                row[f"{label}_full_total_s"] = float(final["total_s"])
                row[f"{label}_end_to_end_s"] = float(seed_s + final["total_s"])
                row[f"{label}_final_nll"] = float(final["nll"])
                row[f"{label}_final_steps"] = int(final["steps"])
                for parameter, value in final["est"].items():
                    row[f"{label}_est_{parameter}"] = float(value)
        rows.append(row)
        pd.DataFrame(rows).to_csv(args.output_root / "day_timings.csv", index=False, float_format="%.10f")
        pd.DataFrame(fft_repeat_rows).to_csv(
            args.output_root / "fft_repeat_timings.csv", index=False, float_format="%.10f"
        )
        print(
            f"{day_label} points/hour={len(grid_coords):,} valid={valid:,}/{total:,} "
            f"FFT median={row['fft_median_s']:.4f}s reverse add={incremental_reverse_s:.4f}s "
            f"ratio={row['reverse_over_fft_ratio']:.1f}x",
            flush=True,
        )
        del asset, day_map
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    table = pd.DataFrame(rows)
    numeric = [
        "points_per_hour", "rows_total", "rows_valid", "valid_rate", "day_build_s",
        "fft_first_s", "fft_median_s", "reverse_precompute_s", "reverse_refine_s",
        "reverse_incremental_s", "reverse_total_seed_s", "reverse_over_fft_ratio",
        "reverse_seed_movement",
    ]
    if args.run_full_fit:
        numeric.extend(
            [
                "fft_full_precompute_s", "fft_full_fit_s", "fft_full_total_s",
                "fft_end_to_end_s", "reverse_l_full_precompute_s",
                "reverse_l_full_fit_s", "reverse_l_full_total_s",
                "reverse_l_end_to_end_s", "fft_final_nll", "reverse_l_final_nll",
            ]
        )
    summary = table[numeric].agg(["mean", "median", "min", "max"])
    summary.to_csv(args.output_root / "timing_summary.csv", float_format="%.10f")
    config = {
        "year": args.year,
        "month": args.month,
        "days": days,
        "device": str(device),
        "torch_num_threads": torch.get_num_threads(),
        "monthly_load_s": monthly_load_s,
        "grid": {
            "n_lat": len(unique_lat),
            "n_lon": len(unique_lon),
            "points_per_hour": len(grid_coords),
            "lat_step": lat_step,
            "lon_step": lon_step,
        },
        "fft_repeats": args.fft_repeats,
        "reverse_l_max_eval": args.reverse_l_max_eval,
        "reverse_l_spatial_stride": 2,
        "run_full_fit": args.run_full_fit,
        "full_fit": {
            "daily_stride": args.daily_stride,
            "target_chunk_size": args.target_chunk_size,
            "outer_steps": args.final_lbfgs_steps,
            "max_eval_per_step": args.final_lbfgs_eval,
            "history": args.final_lbfgs_history,
        },
        "fixed_nuisance": DEFAULT_NUISANCE,
    }
    (args.output_root / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("\nSummary\n", summary.to_string(), flush=True)


if __name__ == "__main__":
    main()
