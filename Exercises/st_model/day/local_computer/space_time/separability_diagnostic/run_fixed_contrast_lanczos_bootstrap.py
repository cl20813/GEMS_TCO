#!/usr/bin/env python3
"""Primary joint-GC pilot: FFT-accelerated Lanczos generation, Vecchia refit.

Generation targets the finite BTTB joint-GC covariance.  Zero-padded FFT is
used only for exact covariance matrix-vector products, and Lanczos evaluates
the covariance square-root action without clipping a circulant spectrum.
The original mask and mean model are restored before the unchanged 4/3/2
corridor-Vecchia and GLS refit.  The diagnostic contrast remains frozen.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch

from apply_fixed_canonical_contrast_real_gems import (
    DEFAULT_DATA_FILE,
    DEFAULT_FIT_CSV,
    DEFAULT_ORACLE_DIR,
    _atomic_csv,
    _atomic_text,
    _canonical_samples,
    _load_filtered_frames,
    _summaries,
)
from audit_fft_generator_frozen_contrast import _axes
from matrix_free_lanczos import build_bttb_operator, lanczos_decomposition, sqrt_action
from run_fixed_contrast_fft_bootstrap import (
    BASE_LAT_STEP,
    BASE_LON_STEP,
    _analytic_covariance,
)
from run_fixed_contrast_full_pipeline_bootstrap import (
    _device,
    _fit_replicate,
    _fit_series,
    _observed_summary,
    _raw_parameters,
    _replicate_source_map,
    _residual_cube,
    _saved_mean_vector,
    _source_map,
)


DEFAULT_OUTPUT_DIR = DEFAULT_ORACLE_DIR / "lanczos_full_pipeline_bootstrap_20240701"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--fit-csv", type=Path, default=DEFAULT_FIT_CSV)
    parser.add_argument("--oracle-dir", type=Path, default=DEFAULT_ORACLE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--strategy", default="standard_432")
    parser.add_argument("--day-index", type=int, default=0)
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--target-chunk-size", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--max-eval", type=int, default=20)
    parser.add_argument("--history-size", type=int, default=10)
    parser.add_argument("--grad-tol", type=float, default=1.0e-4)
    parser.add_argument("--tolerance-grad", type=float, default=1.0e-5)
    parser.add_argument("--lat-factor-hr", type=int, default=2)
    parser.add_argument("--lon-factor-hr", type=int, default=2)
    parser.add_argument("--lanczos-dimension", type=int, default=280)
    parser.add_argument("--lanczos-check-dimension", type=int, default=240)
    parser.add_argument("--hr-pad", type=float, default=0.1)
    parser.add_argument("--latitude-min", type=float, default=-3.0)
    parser.add_argument("--latitude-max", type=float, default=2.0)
    parser.add_argument("--longitude-min", type=float, default=121.0)
    parser.add_argument("--longitude-max", type=float, default=131.0)
    return parser


def _sample_sources(base_map, field, lats, lons):
    dlat = float(lats[1] - lats[0])
    dlon = float(lons[1] - lons[0])
    blocks = []
    for time_index, rows in enumerate(base_map.values()):
        array = rows.detach().cpu().numpy()
        values = np.zeros(len(array), dtype=np.float64)
        valid = (
            np.isfinite(array[:, 0])
            & np.isfinite(array[:, 1])
            & np.isfinite(array[:, 2])
        )
        i = np.rint((array[valid, 0] - lats[0]) / dlat).astype(np.int64)
        j = np.rint((array[valid, 1] - lons[0]) / dlon).astype(np.int64)
        if np.any(i < 0) or np.any(i >= len(lats)) or np.any(j < 0) or np.any(j >= len(lons)):
            raise ValueError("observed source lies outside the Lanczos grid")
        values[valid] = field[i, j, time_index]
        blocks.append(values)
    return np.concatenate(blocks)


def main() -> None:
    args = build_parser().parse_args()
    if args.replicates < 1:
        raise ValueError("replicates must be positive")
    if not 1 <= args.lanczos_check_dimension < args.lanczos_dimension:
        raise ValueError("check dimension must be positive and below final dimension")
    device = _device(args.device)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fit_table = pd.read_csv(args.fit_csv.expanduser())
    selected = fit_table.loc[
        (fit_table["strategy"] == args.strategy)
        & (fit_table["day_idx"].astype(int) == int(args.day_index))
    ]
    if len(selected) != 1:
        raise ValueError("saved fit selection must yield exactly one row")
    original_fit = selected.iloc[0]
    coefficient_path = args.oracle_dir.expanduser() / "global_two_rectangle_search/global_pair_ties.csv"
    coefficients = pd.read_csv(coefficient_path, float_precision="round_trip").iloc[0]
    d1 = float(coefficients["raw_coefficient_first"])
    d2 = float(coefficients["raw_coefficient_second"])
    observed = _observed_summary(output_dir)

    frames, monthly_mean = _load_filtered_frames(
        args.data_file.expanduser(),
        (args.latitude_min, args.latitude_max),
        (args.longitude_min, args.longitude_max),
    )
    keys = sorted(frames)
    start = int(args.day_index) * 8
    day_frames = [frames[key] for key in keys[start : start + 8]]
    base_map, grid_coordinates, latitudes, longitudes, latitude_step, longitude_step = (
        _source_map(day_frames, monthly_mean, device)
    )
    saved_mean = _saved_mean_vector(base_map, original_fit)
    raw = _raw_parameters(original_fit)
    axis_args = SimpleNamespace(
        latitude_min=args.latitude_min,
        latitude_max=args.latitude_max,
        longitude_min=args.longitude_min,
        longitude_max=args.longitude_max,
        hr_pad=args.hr_pad,
    )
    lats, lons = _axes(axis_args, args.lat_factor_hr, args.lon_factor_hr)
    grid_shape = (len(lats), len(lons), 8)
    spacings = (
        BASE_LAT_STEP / args.lat_factor_hr,
        BASE_LON_STEP / args.lon_factor_hr,
        1.0,
    )
    operator_started = time.perf_counter()
    operator = build_bttb_operator(
        grid_shape,
        spacings,
        lambda hlat, hlon, htime: _analytic_covariance(
            hlat, hlon, htime, original_fit, "joint"
        ),
    )
    operator_seconds = time.perf_counter() - operator_started

    result_path = output_dir / "bootstrap_replicates.csv"
    existing = pd.read_csv(result_path) if result_path.is_file() else pd.DataFrame()
    result_rows = existing.to_dict("records") if not existing.empty else []
    completed = set(existing["replicate"].astype(int)) if not existing.empty else set()
    for replicate in range(args.replicates):
        if replicate in completed:
            continue
        seed = int(args.seed + replicate)
        rng = np.random.default_rng(seed)
        started = time.perf_counter()
        normal = rng.standard_normal(operator.size)
        decomposition = lanczos_decomposition(
            operator.matvec,
            normal,
            max_dimension=args.lanczos_dimension,
            full_reorthogonalization=True,
        )
        check_field, _ = sqrt_action(decomposition, args.lanczos_check_dimension)
        field_vector, lanczos_diag = sqrt_action(decomposition, args.lanczos_dimension)
        relative_lanczos_change = float(
            np.linalg.norm(field_vector - check_field) / np.linalg.norm(field_vector)
        )
        generation_seconds = time.perf_counter() - started
        field = field_vector.reshape(grid_shape)
        simulated = _sample_sources(base_map, field, lats, lons)
        replicate_map = _replicate_source_map(base_map, simulated, saved_mean)
        fit_result, beta, fitted_model, fit_seconds = _fit_replicate(
            replicate_map, grid_coordinates, original_fit, raw, args
        )
        fitted_series = _fit_series(fit_result, original_fit, beta, fitted_model.lat_mean_val)
        cube = _residual_cube(
            replicate_map,
            beta,
            fitted_model.lat_mean_val,
            grid_coordinates,
            latitudes,
            longitudes,
            latitude_step,
            longitude_step,
        )
        samples = _canonical_samples(
            cube, original_fit, d1, d2, covariance_fit=fitted_series
        )
        total = _summaries(samples).loc[lambda frame: frame["group"] == "all"].iloc[0]
        interpreted = fit_result.interpretable_parameters
        row: dict[str, Any] = {
            "generator": "fft_bttb_matvec_lanczos_sqrt",
            "dgp": "joint",
            "replicate": replicate,
            "seed": seed,
            "sample_count": int(total["sample_count"]),
            "statistic_empirical_over_joint": float(total["empirical_over_joint_pooled"]),
            "statistic_empirical_over_separable": float(total["empirical_over_separable_pooled"]),
            "empirical_mean_l_squared": float(total["empirical_mean_l_squared"]),
            "joint_mean_variance_l": float(total["joint_mean_variance_l"]),
            "separable_mean_variance_l": float(total["separable_mean_variance_l"]),
            "empirical_h_ab": float(total["empirical_h_ab"]),
            "joint_h_ab": float(total["joint_h_ab"]),
            "separable_h_ab": float(total["separable_h_ab"]),
            "fit_signal_variance": float(interpreted["signal_variance"]),
            "fit_range_lat": float(interpreted["range_lat"]),
            "fit_range_lon": float(interpreted["range_lon"]),
            "fit_range_time": float(interpreted["range_time"]),
            "fit_advec_lat": float(interpreted["advec_lat"]),
            "fit_advec_lon": float(interpreted["advec_lon"]),
            "fit_nll": float(fit_result.final_nll),
            "fit_steps": int(fit_result.steps_completed),
            "fit_converged": bool(fit_result.converged),
            "fit_max_abs_gradient": float(fit_result.max_abs_gradient),
            "fit_objective_evaluations": int(fit_result.objective_evaluations),
            "lanczos_dimension": args.lanczos_dimension,
            "lanczos_check_dimension": args.lanczos_check_dimension,
            "relative_lanczos_field_change": relative_lanczos_change,
            "ritz_min": lanczos_diag["ritz_min"],
            "ritz_max": lanczos_diag["ritz_max"],
            "generation_seconds": generation_seconds,
            "fit_seconds": fit_seconds,
            "replicate_seconds": time.perf_counter() - started,
        }
        for index, value in enumerate(beta.detach().cpu().numpy().reshape(-1)):
            row[f"beta_{index}"] = float(value)
        result_rows.append(row)
        _atomic_csv(
            result_path,
            pd.DataFrame(result_rows).sort_values("replicate").reset_index(drop=True),
        )
        print(
            f"replicate={replicate} Tjoint={row['statistic_empirical_over_joint']:.6f} "
            f"lanczos_change={relative_lanczos_change:.3e} "
            f"gen_s={generation_seconds:.1f} fit_s={fit_seconds:.1f}",
            flush=True,
        )

    results = pd.DataFrame(result_rows).sort_values("replicate").reset_index(drop=True)
    observed_statistic = float(observed["empirical_over_joint_pooled"])
    values = results["statistic_empirical_over_joint"].to_numpy(dtype=np.float64)
    summary = pd.DataFrame(
        [
            {
                "dgp": "joint",
                "replicate_count": len(values),
                "observed_statistic": observed_statistic,
                "bootstrap_mean": float(values.mean()),
                "bootstrap_sd": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                "bootstrap_q025": float(np.quantile(values, 0.025)),
                "bootstrap_q50": float(np.quantile(values, 0.5)),
                "bootstrap_q975": float(np.quantile(values, 0.975)),
                "lower_tail_p_value": float((1 + np.sum(values <= observed_statistic)) / (len(values) + 1)),
                "upper_tail_p_value": float((1 + np.sum(values >= observed_statistic)) / (len(values) + 1)),
            }
        ]
    )
    _atomic_csv(output_dir / "bootstrap_summary.csv", summary)
    manifest = {
        "data_file": str(args.data_file.expanduser().resolve()),
        "fit_csv": str(args.fit_csv.expanduser().resolve()),
        "coefficient_source": str(coefficient_path.resolve()),
        "generator": "finite BTTB covariance; exact zero-padded FFT matvec; fully reorthogonalized Lanczos square-root action",
        "circulant_spectrum_clipping": False,
        "grid_shape": list(grid_shape),
        "grid_factors": [args.lat_factor_hr, args.lon_factor_hr],
        "lanczos_dimension": args.lanczos_dimension,
        "lanczos_check_dimension": args.lanczos_check_dimension,
        "operator_build_seconds": operator_seconds,
        "analysis": "joint GC 4/3/2 corridor Vecchia refit plus GLS; frozen contrast",
        "requested_replicates": args.replicates,
    }
    _atomic_text(output_dir / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    row = summary.iloc[0]
    report = "\n".join(
        [
            "# FFT-accelerated matrix-free Lanczos full-pipeline pilot",
            "",
            "The generator targets the finite joint-GC BTTB covariance and uses FFT only for exact zero-padded matrix-vector products. No circulant eigenvalue is clipped. Each field is restored to the saved GLS mean and original missingness mask, then re-fitted with the unchanged 4/3/2 corridor-Vecchia pipeline.",
            "",
            f"Replicates: `{int(row['replicate_count'])}`. Observed empirical/joint statistic: `{row['observed_statistic']:.8f}`. Pilot bootstrap mean: `{row['bootstrap_mean']:.8f}`.",
            "",
            "This remains a computational pilot. Do not interpret its tail probability until at least 99 replicates have been run and the separable-null refit path has been implemented separately.",
            "",
            "See `bootstrap_replicates.csv` for Lanczos convergence, optimizer diagnostics, fitted parameters, and timing.",
            "",
        ]
    )
    _atomic_text(output_dir / "REPORT.md", report)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
