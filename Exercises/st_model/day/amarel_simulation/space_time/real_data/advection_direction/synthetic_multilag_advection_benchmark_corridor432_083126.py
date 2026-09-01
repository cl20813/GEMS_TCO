#!/usr/bin/env python3
"""Compare zero, tau-1 empirical, and robust multi-lag advection seeds.

The two fixed baselines are:

* M0_zero: no-advection seed.
* M3_empirical_tau1: the existing smoothed tau=1 empirical
  space-time semivariogram minimum.

The new competitor, M6_robust_multilag, computes regular-grid empirical
semivariograms for tau=1,2,3, maps each displacement surface h to a common
velocity surface through h=tau*v, and combines relative semivariograms using
lag weights based on the number of temporal pairs and empirical ridge
contrast.  No spatial-distance binning is used because the data are already
on a regular grid.

For downstream comparison, each seed defines one fixed directional lag-432
corridor.  All seven covariance parameters are then fitted with the same
nuisance starts and optimizer budget; corridor geometry is not rebuilt during
optimization.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import RegularGridInterpolator

from synthetic_advection_initializer_benchmark_corridor432_081526 import (
    P_LABELS,
    clean_json,
    empirical_seed,
    fit_full_corridor,
    initializer_row,
    load_assets,
    make_hourly_grids,
    nan_gaussian_filter,
    parse_days,
    parse_pair,
    shifted_sum_half_sq,
)


LOCAL_REPO = Path("/Users/joonwonlee/Documents/GEMS_TCO-1")
AMAREL_REPO = Path("/home/jl2815/tco")
REPO = AMAREL_REPO if AMAREL_REPO.exists() else LOCAL_REPO
METHODS = ("M0_zero", "M3_empirical_tau1", "M6_robust_multilag")


def parse_int_list(text: str) -> list[int]:
    values = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("Temporal lags must be positive integers")
    return sorted(set(values))


def empirical_surface(
    grids: Sequence[np.ndarray],
    tau: int,
    lat_step: float,
    lon_step: float,
    max_velocity_lat: float,
    max_velocity_lon: float,
    min_pair_count: int,
    smooth_bandwidth_deg: float,
) -> dict[str, Any]:
    """Compute one pair-count-weighted regular-grid semivariogram surface."""
    tau = int(tau)
    if tau <= 0 or tau >= len(grids):
        raise ValueError(f"tau={tau} is invalid for {len(grids)} time slices")

    # One extra cell protects linear interpolation at the requested velocity
    # boundary.  These are exact integer grid shifts, not distance bins.
    max_lat_offset = int(math.ceil(tau * max_velocity_lat / abs(lat_step))) + 1
    max_lon_offset = int(math.ceil(tau * max_velocity_lon / abs(lon_step))) + 1
    max_lat_offset = min(max_lat_offset, grids[0].shape[0] - 1)
    max_lon_offset = min(max_lon_offset, grids[0].shape[1] - 1)
    offsets_lat = np.arange(-max_lat_offset, max_lat_offset + 1, dtype=np.int64)
    offsets_lon = np.arange(-max_lon_offset, max_lon_offset + 1, dtype=np.int64)
    sums = np.zeros((len(offsets_lat), len(offsets_lon)), dtype=np.float64)
    counts = np.zeros_like(sums, dtype=np.int64)

    for row, di in enumerate(offsets_lat):
        for col, dj in enumerate(offsets_lon):
            for hour in range(len(grids) - tau):
                value, count = shifted_sum_half_sq(
                    grids[hour], grids[hour + tau], int(di), int(dj)
                )
                sums[row, col] += value
                counts[row, col] += count

    gamma = np.full_like(sums, np.nan)
    valid = counts >= int(min_pair_count)
    gamma[valid] = sums[valid] / counts[valid]
    sigma = (
        float(smooth_bandwidth_deg) / abs(lat_step),
        float(smooth_bandwidth_deg) / abs(lon_step),
    )
    smoothed = nan_gaussian_filter(gamma, sigma)
    if not np.isfinite(smoothed).any():
        raise RuntimeError(f"tau={tau}: no finite semivariogram cells")

    finite = smoothed[np.isfinite(smoothed)]
    surface_median = float(np.median(finite))
    surface_min = float(np.min(finite))
    if not np.isfinite(surface_median) or surface_median <= 0:
        raise RuntimeError(f"tau={tau}: invalid semivariogram median {surface_median}")
    contrast = max(0.0, (surface_median - surface_min) / surface_median)
    min_row, min_col = np.unravel_index(int(np.nanargmin(smoothed)), smoothed.shape)
    boundary = bool(
        min_row in {0, smoothed.shape[0] - 1}
        or min_col in {0, smoothed.shape[1] - 1}
    )
    return {
        "tau": tau,
        "n_temporal_pairs": len(grids) - tau,
        "offsets_lat": offsets_lat,
        "offsets_lon": offsets_lon,
        "h_lat": offsets_lat.astype(np.float64) * float(lat_step),
        "h_lon": offsets_lon.astype(np.float64) * float(lon_step),
        "gamma": gamma,
        "smoothed": smoothed,
        "counts": counts,
        "median": surface_median,
        "minimum": surface_min,
        "contrast": contrast,
        "minimum_boundary": boundary,
        "grid_seed_lat": float(offsets_lat[min_row] * lat_step / tau),
        "grid_seed_lon": float(offsets_lon[min_col] * lon_step / tau),
        "grid_min_pair_count": int(counts[min_row, min_col]),
    }


def evaluate_surface_on_velocity_grid(
    surface: dict[str, Any],
    velocity_lat: np.ndarray,
    velocity_lon: np.ndarray,
) -> np.ndarray:
    """Evaluate gamma_tau(tau*v)/median(gamma_tau) by linear interpolation."""
    tau = int(surface["tau"])
    interpolator = RegularGridInterpolator(
        (surface["h_lat"], surface["h_lon"]),
        surface["smoothed"] / float(surface["median"]),
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    mesh_lat, mesh_lon = np.meshgrid(velocity_lat, velocity_lon, indexing="ij")
    query = np.column_stack([(tau * mesh_lat).ravel(), (tau * mesh_lon).ravel()])
    return interpolator(query).reshape(mesh_lat.shape)


def robust_multilag_seed(asset, args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return the contrast-weighted tau-consistent empirical velocity seed."""
    started = time.perf_counter()
    grids, lat_step, lon_step = make_hourly_grids(asset.source_map, asset.grid_coords)
    surfaces = [
        empirical_surface(
            grids=grids,
            tau=tau,
            lat_step=lat_step,
            lon_step=lon_step,
            max_velocity_lat=float(args.multilag_max_velocity_lat),
            max_velocity_lon=float(args.multilag_max_velocity_lon),
            min_pair_count=int(args.empirical_min_pair_count),
            smooth_bandwidth_deg=float(args.empirical_smooth_bandwidth_deg),
        )
        for tau in args.multilag_taus
    ]

    refine = int(args.multilag_velocity_refine)
    # Anchor the refined velocity grid at exactly zero.  Starting arange at an
    # arbitrary negative bound would shift the candidate lattice and introduce
    # a purely numerical velocity bias.
    velocity_lat_step = abs(lat_step) / refine
    velocity_lon_step = abs(lon_step) / refine
    n_velocity_lat = int(math.floor(float(args.multilag_max_velocity_lat) / velocity_lat_step))
    n_velocity_lon = int(math.floor(float(args.multilag_max_velocity_lon) / velocity_lon_step))
    velocity_lat = np.arange(-n_velocity_lat, n_velocity_lat + 1) * velocity_lat_step
    velocity_lon = np.arange(-n_velocity_lon, n_velocity_lon + 1) * velocity_lon_step

    numerator = np.zeros((len(velocity_lat), len(velocity_lon)), dtype=np.float64)
    denominator = np.zeros_like(numerator)
    diagnostics: list[dict[str, Any]] = []
    for surface in surfaces:
        relative = evaluate_surface_on_velocity_grid(surface, velocity_lat, velocity_lon)
        # A flat lag has contrast near zero and therefore receives negligible
        # influence.  sqrt(n_pairs) rewards replicated temporal information
        # without letting tau=1 dominate linearly by pair count.
        usable = bool(
            not surface["minimum_boundary"]
            and float(surface["contrast"]) >= float(args.multilag_min_contrast)
        )
        weight = (
            math.sqrt(float(surface["n_temporal_pairs"])) * float(surface["contrast"])
            if usable
            else 0.0
        )
        finite = np.isfinite(relative)
        numerator[finite] += weight * relative[finite]
        denominator[finite] += weight
        diagnostics.append(
            {
                "tau": int(surface["tau"]),
                "n_temporal_pairs": int(surface["n_temporal_pairs"]),
                "grid_seed_lat": float(surface["grid_seed_lat"]),
                "grid_seed_lon": float(surface["grid_seed_lon"]),
                "surface_median": float(surface["median"]),
                "surface_minimum": float(surface["minimum"]),
                "relative_contrast": float(surface["contrast"]),
                "lag_weight": float(weight),
                "used_in_combination": usable,
                "minimum_boundary": bool(surface["minimum_boundary"]),
                "minimum_pair_count": int(surface["grid_min_pair_count"]),
                "lat_offset_min": int(surface["offsets_lat"][0]),
                "lat_offset_max": int(surface["offsets_lat"][-1]),
                "lon_offset_min": int(surface["offsets_lon"][0]),
                "lon_offset_max": int(surface["offsets_lon"][-1]),
            }
        )

    combined = np.full_like(numerator, np.nan)
    valid = denominator > 0
    combined[valid] = numerator[valid] / denominator[valid]
    if not np.isfinite(combined).any():
        raise RuntimeError("No common finite multi-lag velocity candidates")
    row, col = np.unravel_index(int(np.nanargmin(combined)), combined.shape)
    seed_lat = float(velocity_lat[row])
    seed_lon = float(velocity_lon[col])
    usable_diagnostics = [item for item in diagnostics if item["used_in_combination"]]
    lag_seeds = np.asarray(
        [[item["grid_seed_lat"], item["grid_seed_lon"]] for item in usable_diagnostics],
        dtype=np.float64,
    )
    agreement_rms = float(np.sqrt(np.mean(np.sum((lag_seeds - [seed_lat, seed_lon]) ** 2, axis=1))))
    elapsed = time.perf_counter() - started
    result = {
        "seed_lat": seed_lat,
        "seed_lon": seed_lon,
        "selection_nll": np.nan,
        "search_s": float(elapsed),
        "closure_calls": 0,
        "profile_evals": int(combined.size),
        "selected_candidate": "contrast_weighted_tau_" + "_".join(str(x) for x in args.multilag_taus),
        "ridge_near_min_count": np.nan,
        "ridge_is_ambiguous": bool(
            len(usable_diagnostics) < 2
            or agreement_rms > float(args.multilag_max_agreement_rms)
            or any(item["minimum_boundary"] for item in diagnostics)
        ),
        "multilag_agreement_rms": agreement_rms,
        "multilag_combined_score": float(combined[row, col]),
        "multilag_weight_sum": float(sum(item["lag_weight"] for item in diagnostics)),
        "est_range_time_seed_stage": np.nan,
    }
    return result, diagnostics


def zero_seed() -> dict[str, Any]:
    return {
        "seed_lat": 0.0,
        "seed_lon": 0.0,
        "selection_nll": np.nan,
        "search_s": 0.0,
        "closure_calls": 0,
        "profile_evals": 0,
        "selected_candidate": "zero",
        "ridge_near_min_count": np.nan,
        "ridge_is_ambiguous": np.nan,
        "est_range_time_seed_stage": np.nan,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/Users/joonwonlee/Documents/GEMS_DATA/simulation/july_st_circulant_realpattern_smooth0p5"),
    )
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--days", default="0,3")
    parser.add_argument("--full-fit-days", default="0,3")
    parser.add_argument("--lat-range", default="-3,-1")
    parser.add_argument("--lon-range", default="121,125")
    parser.add_argument("--keep-exact-loc", action="store_true", default=True)
    parser.add_argument("--no-keep-exact-loc", dest="keep_exact_loc", action="store_false")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO / "outputs/summer_26/synthetic_multilag_advection_benchmark_corridor432_083126",
    )
    parser.add_argument("--smooth", type=float, default=0.5)
    parser.add_argument("--daily-stride", type=int, default=2)
    parser.add_argument("--target-chunk-size", type=int, default=128)
    parser.add_argument("--empirical-max-lat-offset", type=int, default=20)
    parser.add_argument("--empirical-max-lon-offset", type=int, default=20)
    parser.add_argument("--empirical-min-pair-count", type=int, default=1000)
    parser.add_argument("--empirical-smooth-bandwidth-deg", type=float, default=0.063)
    parser.add_argument("--empirical-near-min-rel-tol", type=float, default=0.01)
    parser.add_argument("--empirical-near-min-abs-tol", type=float, default=0.05)
    parser.add_argument("--empirical-ambiguous-min-near-cells", type=int, default=5)
    parser.add_argument("--multilag-taus", type=parse_int_list, default=[1, 2, 3])
    parser.add_argument("--multilag-max-velocity-lat", type=float, default=0.30)
    parser.add_argument("--multilag-max-velocity-lon", type=float, default=0.40)
    parser.add_argument("--multilag-velocity-refine", type=int, default=8)
    parser.add_argument("--multilag-min-contrast", type=float, default=0.05)
    parser.add_argument("--multilag-max-agreement-rms", type=float, default=0.08)
    parser.add_argument("--final-outer-steps", type=int, default=1)
    parser.add_argument("--final-max-eval", type=int, default=10)
    parser.add_argument("--lbfgs-lr", type=float, default=1.0)
    parser.add_argument("--lbfgs-history", type=int, default=10)
    parser.add_argument("--initializer-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if int(args.multilag_velocity_refine) < 1:
        raise ValueError("--multilag-velocity-refine must be >= 1")
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    input_path = (
        Path(args.data_root)
        / f"{args.year}_july_st_circulant"
        / f"sim_july{args.year}_st_circulant_gridded.pkl"
    )
    truth_path = (
        Path(args.data_root)
        / f"{args.year}_july_st_circulant"
        / f"sim_july{args.year}_st_circulant_truth.json"
    )
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    truth_physical = {key: float(truth[key]) for key in P_LABELS}
    day_indices = parse_days(args.days)
    full_fit_days = set() if args.initializer_only else set(parse_days(args.full_fit_days))
    lat_range = parse_pair(args.lat_range, float)
    lon_range = parse_pair(args.lon_range, float)
    config = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "input": input_path,
        "truth": truth_physical,
        "days": day_indices,
        "full_fit_days": sorted(full_fit_days),
        "lat_range": lat_range,
        "lon_range": lon_range,
        "device": str(device),
        "methods": METHODS,
        "arguments": vars(args),
    }
    (output_root / "run_config.json").write_text(
        json.dumps(clean_json(config), indent=2, sort_keys=True), encoding="utf-8"
    )
    assets = load_assets(
        input_path, truth, day_indices, lat_range, lon_range, bool(args.keep_exact_loc)
    )
    initializer_rows: list[dict[str, Any]] = []
    lag_rows: list[dict[str, Any]] = []
    full_rows: list[dict[str, Any]] = []
    benchmark_started = time.perf_counter()

    print("input:", input_path)
    print("truth:", truth_physical)
    print("methods:", METHODS)
    print("device:", device)
    print("output:", output_root)
    for asset in assets:
        print(f"\n=== {asset.day}; rows/hour={len(asset.grid_coords)}, valid={asset.n_valid}/{asset.n_total} ===")
        empirical = empirical_seed(asset, args)
        multilag, diagnostics = robust_multilag_seed(asset, args)
        results = {
            "M0_zero": zero_seed(),
            "M3_empirical_tau1": empirical,
            "M6_robust_multilag": multilag,
        }
        day_seed_rows: dict[str, dict[str, Any]] = {}
        for diagnostic in diagnostics:
            lag_rows.append(
                {
                    "year": asset.year,
                    "day_idx": asset.day_idx,
                    "day": asset.day,
                    **diagnostic,
                }
            )
        for method in METHODS:
            row = initializer_row(
                asset=asset,
                scenario="oracle_nuisance",
                method=method,
                result=results[method],
                truth=truth_physical,
                pilot_points=0,
                precompute_s=0.0,
                common_eval_nll=np.nan,
            )
            if method == "M6_robust_multilag":
                row.update(
                    {
                        "multilag_agreement_rms": multilag["multilag_agreement_rms"],
                        "multilag_combined_score": multilag["multilag_combined_score"],
                        "multilag_weight_sum": multilag["multilag_weight_sum"],
                    }
                )
            initializer_rows.append(clean_json(row))
            day_seed_rows[method] = row
            print(
                f"{method}: seed=({row['seed_lat']:+.5f},{row['seed_lon']:+.5f}) "
                f"error={row['seed_error_euclid']:.5f} angle={row['seed_angle_error_deg']!s} "
                f"time={row['seed_total_s']:.3f}s"
            )

        pd.DataFrame(initializer_rows).to_csv(
            output_root / "initializer_results.csv", index=False, float_format="%.8f"
        )
        pd.DataFrame(lag_rows).to_csv(
            output_root / "multilag_diagnostics.csv", index=False, float_format="%.8f"
        )

        if asset.day_idx in full_fit_days:
            print("-- full 7-parameter fixed-corridor fits --")
            for method in METHODS:
                attempt_started = time.perf_counter()
                try:
                    row = fit_full_corridor(
                        asset, day_seed_rows[method], truth_physical, args, device
                    )
                    print(
                        f"{method}: nll={row['final_nll']:.6f} "
                        f"final_adv_error={row['final_seed_error_euclid']:.5f} "
                        f"fit={row['final_total_s']:.2f}s end_to_end={row['end_to_end_s']:.2f}s"
                    )
                except Exception as exc:
                    row = {
                        "year": asset.year,
                        "day_idx": asset.day_idx,
                        "day": asset.day,
                        "method": method,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(limit=8),
                        "attempt_s": time.perf_counter() - attempt_started,
                    }
                    print("ERROR", method, row["error"])
                full_rows.append(clean_json(row))
                pd.DataFrame(full_rows).to_csv(
                    output_root / "full_fit_results.csv", index=False, float_format="%.8f"
                )

    init_df = pd.DataFrame(initializer_rows)
    full_df = pd.DataFrame(full_rows)
    initializer_summary = init_df.groupby("method", as_index=False).agg(
        n_days=("day_idx", "size"),
        mean_seed_s=("seed_total_s", "mean"),
        median_seed_s=("seed_total_s", "median"),
        mean_seed_error=("seed_error_euclid", "mean"),
        median_seed_error=("seed_error_euclid", "median"),
        max_seed_error=("seed_error_euclid", "max"),
        mean_angle_error_deg=("seed_angle_error_deg", "mean"),
        correct_quadrant_rate=("correct_quadrant", "mean"),
    )
    initializer_summary.to_csv(
        output_root / "initializer_summary.csv", index=False, float_format="%.8f"
    )
    if not full_df.empty and "final_nll" in full_df:
        ok = full_df[full_df["status"].eq("ok")]
        full_summary = ok.groupby("method", as_index=False).agg(
            n_days=("day_idx", "size"),
            mean_final_nll=("final_nll", "mean"),
            mean_final_advection_error=("final_seed_error_euclid", "mean"),
            mean_final_rmsre_7param=("final_rmsre_7param", "mean"),
            mean_final_fit_s=("final_total_s", "mean"),
            mean_end_to_end_s=("end_to_end_s", "mean"),
        )
        full_summary.to_csv(
            output_root / "full_fit_summary.csv", index=False, float_format="%.8f"
        )
    elapsed = time.perf_counter() - benchmark_started
    (output_root / "total_runtime.json").write_text(
        json.dumps(
            {"benchmark_wall_s": elapsed, "benchmark_wall_minutes": elapsed / 60.0},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nDONE in {elapsed:.2f}s ({elapsed / 60.0:.2f} min)")
    print(initializer_summary.to_string(index=False))


if __name__ == "__main__":
    main()
