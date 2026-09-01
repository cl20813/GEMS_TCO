#!/usr/bin/env python3
"""Time M0/M3/S1--S4 advection seeds on one full real GEMS day."""

from __future__ import annotations

import argparse
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
from synthetic_short_model_initializer_competition_corridor432_083126 import (
    build_parser as competition_parser,
    fft_gated_seed,
    fixed_stencil_seed,
    iterative_corridor_seed,
    pairwise_polar_seed,
    pairwise_sufficient_statistics,
    zero_seed,
)
from synthetic_m3_reverse_l_initialization_benchmark_corridor432_083126 import run_fft_seed


DEFAULT_NUISANCE = {
    "sigmasq": 13.059,
    "range_lat": 0.20,
    "range_lon": 0.25,
    "range_time": 1.50,
    "advec_lat": 0.0,
    "advec_lon": 0.0,
    "nugget": 0.247,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/Users/joonwonlee/Documents/GEMS_DATA"))
    parser.add_argument("--output-root", type=Path, default=REPO / "outputs/day/estimates/realdata_short_model_initializer_timing_090126")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--month", type=int, default=7)
    parser.add_argument("--day", type=int, default=0, help="Zero-based day index.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fft-repeats", type=int, default=5)
    return parser


def main() -> None:
    cli = build_parser().parse_args()
    args = competition_parser().parse_args([])
    args.device = cli.device
    device = torch.device(cli.device)
    cli.output_root.mkdir(parents=True, exist_ok=True)

    loader = load_data_dynamic_processed(str(cli.data_root))
    load_started = time.perf_counter()
    df_map, _, _, monthly_mean = loader.load_maxmin_ordered_data_bymonthyear(
        lat_lon_resolution=[1, 1],
        mm_cond_number=1,
        years_=[str(cli.year)],
        months_=[int(cli.month)],
        lat_range=[-3.0, 2.0],
        lon_range=[121.0, 131.0],
        is_whittle=True,
    )
    monthly_load_s = time.perf_counter() - load_started
    keys = sorted(df_map)
    grid_coords = df_map[keys[0]][["Latitude", "Longitude"]].to_numpy(dtype=np.float64)
    day_started = time.perf_counter()
    day_map, _ = loader.load_working_data(
        df_map,
        monthly_mean,
        [int(cli.day) * 8, (int(cli.day) + 1) * 8],
        ord_mm=None,
        dtype=torch.double,
        keep_ori=True,
    )
    day_build_s = time.perf_counter() - day_started
    total = sum(int(value.shape[0]) for value in day_map.values())
    valid = sum(int(torch.isfinite(value[:, 2]).sum()) for value in day_map.values())
    day_label = f"{cli.year}-{cli.month:02d}-{cli.day + 1:02d}"
    asset = DayAsset(
        year=cli.year,
        month=cli.month,
        day_idx=cli.day,
        day=day_label,
        source_map=day_map,
        grid_coords=grid_coords,
        n_valid=valid,
        n_total=total,
    )

    fft_runs = []
    fft_diag = None
    for _ in range(int(cli.fft_repeats)):
        m3, fft_diag = run_fft_seed(asset, args, robust=False, subgrid=False)
        fft_runs.append(m3)
    fft_times = np.asarray([item["search_s"] for item in fft_runs], dtype=float)
    m3 = dict(fft_runs[-1])
    m3["seed_total_s"] = float(np.median(fft_times))

    stats, stats_s = pairwise_sufficient_statistics(asset, args)
    seeds = {
        "M0_zero": zero_seed(),
        "M3_fft": m3,
        "S1_fixed_stencil": fixed_stencil_seed(asset, DEFAULT_NUISANCE, args, device),
        "S2_pairwise_polar": pairwise_polar_seed(DEFAULT_NUISANCE, stats, stats_s, args),
        "S3_fft_gated": fft_gated_seed(m3, fft_diag, DEFAULT_NUISANCE, stats, stats_s, args),
        "S4_iterative_corridor": iterative_corridor_seed(
            asset, m3, DEFAULT_NUISANCE, args, device
        ),
    }
    rows = []
    for method, seed in seeds.items():
        rows.append(
            {
                "day": day_label,
                "method": method,
                "seed_lat": float(seed["seed_lat"]),
                "seed_lon": float(seed["seed_lon"]),
                "seed_total_s": float(seed.get("seed_total_s", 0.0)),
                "closure_calls": int(seed.get("closure_calls", 0)),
                "selected_candidate": seed.get("selected_candidate", ""),
                "fft_strong": seed.get("fft_strong", np.nan),
                "fft_near_count": seed.get("fft_near_count", np.nan),
                "fft_basin_gap": seed.get("fft_basin_gap", np.nan),
            }
        )
    result = pd.DataFrame(rows)
    m3_time = float(result.loc[result.method.eq("M3_fft"), "seed_total_s"].iloc[0])
    result["time_ratio_vs_M3"] = result["seed_total_s"] / m3_time
    result.to_csv(cli.output_root / "initializer_timings.csv", index=False, float_format="%.10f")
    config = {
        "day": day_label,
        "device": str(device),
        "torch_num_threads": torch.get_num_threads(),
        "monthly_load_s": monthly_load_s,
        "day_build_s": day_build_s,
        "points_per_hour": len(grid_coords),
        "rows_total": total,
        "rows_valid": valid,
        "lat_step": 0.044,
        "lon_step": 0.063,
        "fft_repeats": cli.fft_repeats,
        "fixed_nuisance": DEFAULT_NUISANCE,
    }
    (cli.output_root / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(result.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
