#!/usr/bin/env python3
"""Current-design 360-dataset initializer stress test for M3, Q3, and Q5."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO = Path("/Users/joonwonlee/Documents/GEMS_TCO-1")
SRC = REPO / "src"
HERE = Path(__file__).resolve().parent
for path in (REPO, SRC, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from m3_q3_q5_quick_benchmark_090126 import METHODS, add_seed_truth_metrics, run_initializers
from m3_surface_diagnostic_subgrid_benchmark_090126 import SIGNAL_REGIMES, truth_conditions
from synthetic_advection_initializer_benchmark_corridor432_081526 import (
    P_LABELS,
    clean_json,
    load_assets,
    parse_pair,
)
from synthetic_initializer_factorial_robustness_corridor432_083126 import (
    coordinate_indices,
    crop_regular_asset,
    simulate_asset,
)


def parse_float_list(text: str) -> list[float]:
    return [float(token.strip()) for token in str(text).split(",") if token.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synthetic-data-root",
        type=Path,
        default=Path(
            "/Users/joonwonlee/Documents/GEMS_DATA/simulation/"
            "july_st_circulant_realpattern_smooth0p5"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO / "outputs/summer_26/m3_q3_q5_initializer_stress_090126",
    )
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--speeds-cells", default="0.75,2.0,4.0")
    parser.add_argument("--angle-offset-deg", type=float, default=22.5)
    parser.add_argument("--signal-regimes", default="strong,reference,weak")
    parser.add_argument("--lat-range", default="-3,-1")
    parser.add_argument("--lon-range", default="121,125")
    parser.add_argument("--n-lat", type=int, default=32)
    parser.add_argument("--n-lon", type=int, default=48)
    parser.add_argument("--seed", type=int, default=901264)
    parser.add_argument("--smooth", type=float, default=0.5)
    parser.add_argument("--empirical-max-lat-offset", type=int, default=20)
    parser.add_argument("--empirical-max-lon-offset", type=int, default=20)
    parser.add_argument("--empirical-min-pair-count", type=int, default=1000)
    parser.add_argument("--empirical-smooth-bandwidth-deg", type=float, default=0.063)
    parser.add_argument("--subgrid-max-condition-number", type=float, default=100.0)
    parser.add_argument("--q5-gaussian-sigma-cells", type=float, default=1.25)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    truth_path = (
        Path(args.synthetic_data_root)
        / f"{args.year}_july_st_circulant"
        / f"sim_july{args.year}_st_circulant_truth.json"
    )
    input_path = (
        Path(args.synthetic_data_root)
        / f"{args.year}_july_st_circulant"
        / f"sim_july{args.year}_st_circulant_gridded.pkl"
    )
    base_json = json.loads(truth_path.read_text(encoding="utf-8"))
    base_truth = {key: float(base_json[key]) for key in P_LABELS}
    templates = load_assets(
        input_path,
        base_json,
        list(range(int(args.replicates))),
        parse_pair(args.lat_range, float),
        parse_pair(args.lon_range, float),
        False,
    )
    templates = [crop_regular_asset(asset, int(args.n_lat), int(args.n_lon)) for asset in templates]
    _, _, lat_step, lon_step, _, _ = coordinate_indices(templates[0].grid_coords)
    conditions = truth_conditions(
        lat_step,
        lon_step,
        parse_float_list(args.speeds_cells),
        float(args.angle_offset_deg),
    )
    regimes = [token.strip() for token in str(args.signal_regimes).split(",") if token.strip()]
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    total = len(conditions) * len(regimes) * int(args.replicates)
    counter = 0
    for condition_index, condition in enumerate(conditions):
        for regime_index, regime in enumerate(regimes):
            truth = dict(base_truth)
            truth["advec_lat"] = float(condition["advec_lat"])
            truth["advec_lon"] = float(condition["advec_lon"])
            truth["range_time"] *= float(SIGNAL_REGIMES[regime]["range_time_factor"])
            truth["nugget"] *= float(SIGNAL_REGIMES[regime]["nugget_factor"])
            for replicate in range(int(args.replicates)):
                counter += 1
                simulation_seed = int(
                    args.seed + condition_index * 100_000 + regime_index * 10_000 + replicate * 101
                )
                dataset_id = f"{condition['truth_id']}_{regime}_r{replicate:02d}"
                asset, _ = simulate_asset(
                    templates[replicate], truth, simulation_seed, args, dataset_id, replicate
                )
                method_rows, _ = run_initializers(asset, args)
                for method_row in method_rows:
                    row = {
                        **method_row,
                        "dataset_id": dataset_id,
                        "direction_deg_grid": condition["direction_deg_grid"],
                        "speed_cells": condition["speed_cells"],
                        "signal_regime": regime,
                        "replicate": replicate,
                        "true_advec_lat": truth["advec_lat"],
                        "true_advec_lon": truth["advec_lon"],
                    }
                    add_seed_truth_metrics(row, truth, lat_step, lon_step)
                    rows.append(clean_json(row))
                if counter % 60 == 0 or counter == total:
                    print(f"[{counter}/{total}]", flush=True)

    detail = pd.DataFrame(rows)
    detail.to_csv(output_root / "initializer_results.csv", index=False, float_format="%.10f")
    summary = (
        detail.groupby("method", as_index=False)
        .agg(
            n=("dataset_id", "size"),
            mean_seed_error=("seed_error_euclid", "mean"),
            median_seed_error=("seed_error_euclid", "median"),
            mean_error_cells=("seed_error_grid_cells", "mean"),
            median_error_cells=("seed_error_grid_cells", "median"),
            mean_angle_error_deg=("seed_angle_error_deg", "mean"),
            correct_quadrant_rate=("correct_quadrant", "mean"),
            mean_seed_s=("seed_total_s", "mean"),
            subgrid_use_rate=("subgrid_used", "mean"),
        )
    )
    summary.to_csv(output_root / "initializer_summary.csv", index=False, float_format="%.10f")
    config = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "methods": METHODS,
        "base_truth": base_truth,
        "conditions": conditions,
        "signal_regimes": {key: SIGNAL_REGIMES[key] for key in regimes},
        "arguments": vars(args),
    }
    (output_root / "run_config.json").write_text(
        json.dumps(clean_json(config), indent=2, sort_keys=True), encoding="utf-8"
    )
    elapsed = float(time.perf_counter() - started)
    (output_root / "total_runtime.json").write_text(
        json.dumps({"wall_s": elapsed, "wall_minutes": elapsed / 60.0}, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False), flush=True)
    print(f"DONE in {elapsed:.2f}s", flush=True)


if __name__ == "__main__":
    main()
