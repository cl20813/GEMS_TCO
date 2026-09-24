#!/usr/bin/env python3
"""Small matrix-free Lanczos generator sensitivity for the frozen contrast.

The finite regular-lattice target covariance is used directly.  FFT supplies
only exact zero-padded BTTB matrix-vector products; no circulant eigenvalue is
square-rooted or clipped.  Lanczos dimensions are compared within each fixed
Gaussian starting vector to assess numerical convergence of ``Sigma^(1/2) z``.
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

from apply_fixed_canonical_contrast_real_gems import (
    DEFAULT_DATA_FILE,
    DEFAULT_FIT_CSV,
    DEFAULT_ORACLE_DIR,
    _atomic_csv,
    _atomic_text,
    _grid_cube,
    _load_filtered_frames,
)
from audit_fft_generator_frozen_contrast import (
    _axes,
    _contrast_coordinates,
)
from matrix_free_lanczos import (
    build_bttb_operator,
    lanczos_decomposition,
    sqrt_action,
)
from run_fixed_contrast_fft_bootstrap import (
    BASE_LAT_STEP,
    BASE_LON_STEP,
    _analytic_covariance,
)


DEFAULT_OUTPUT_DIR = DEFAULT_ORACLE_DIR / "lanczos_generator_sensitivity_20240701"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--fit-csv", type=Path, default=DEFAULT_FIT_CSV)
    parser.add_argument("--oracle-dir", type=Path, default=DEFAULT_ORACLE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--strategy", default="standard_432")
    parser.add_argument("--day-index", type=int, default=0)
    parser.add_argument("--dgp", choices=("joint", "separable", "both"), default="joint")
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--dimensions", default="20,40,80")
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--lat-factor-hr", type=int, default=1)
    parser.add_argument("--lon-factor-hr", type=int, default=1)
    parser.add_argument("--hr-pad", type=float, default=0.1)
    parser.add_argument("--latitude-min", type=float, default=-3.0)
    parser.add_argument("--latitude-max", type=float, default=2.0)
    parser.add_argument("--longitude-min", type=float, default=121.0)
    parser.add_argument("--longitude-max", type=float, default=131.0)
    return parser


def _dimensions(text: str) -> tuple[int, ...]:
    values = tuple(sorted({int(item.strip()) for item in text.split(",") if item.strip()}))
    if not values or values[0] < 1:
        raise ValueError("Lanczos dimensions must be positive integers")
    return values


def _field_moments(values: np.ndarray, coefficient_a, coefficient_b, d1, d2):
    q_a = values @ coefficient_a
    q_b = values @ coefficient_b
    contrast = d1 * q_a + d2 * q_b
    return {
        "empirical_h_aa": float(np.mean(np.square(q_a))),
        "empirical_h_bb": float(np.mean(np.square(q_b))),
        "empirical_h_ab": float(np.mean(q_a * q_b)),
        "empirical_mean_l_squared": float(np.mean(np.square(contrast))),
    }


def main() -> None:
    args = build_parser().parse_args()
    dimensions = _dimensions(args.dimensions)
    if args.replicates < 1:
        raise ValueError("replicates must be positive")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fit_table = pd.read_csv(args.fit_csv.expanduser())
    selected = fit_table.loc[
        (fit_table["strategy"] == args.strategy)
        & (fit_table["day_idx"].astype(int) == int(args.day_index))
    ]
    if len(selected) != 1:
        raise ValueError("saved fit selection must yield exactly one row")
    fit = selected.iloc[0]
    coefficient_path = args.oracle_dir.expanduser() / "global_two_rectangle_search/global_pair_ties.csv"
    coefficients = pd.read_csv(coefficient_path, float_precision="round_trip").iloc[0]
    d1 = float(coefficients["raw_coefficient_first"])
    d2 = float(coefficients["raw_coefficient_second"])
    coefficient_a = np.asarray([1.0, -1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    coefficient_b = np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0, 1.0])

    frames, monthly_mean = _load_filtered_frames(
        args.data_file.expanduser(),
        (args.latitude_min, args.latitude_max),
        (args.longitude_min, args.longitude_max),
    )
    keys = sorted(frames)
    start = int(args.day_index) * 8
    cube = _grid_cube([frames[key] for key in keys[start : start + 8]], monthly_mean, fit)
    samples_path = args.oracle_dir.expanduser() / "real_gems_pilot_20240701/translated_contrast_samples.csv"
    samples = pd.read_csv(samples_path)
    axis_args = SimpleNamespace(
        latitude_min=args.latitude_min,
        latitude_max=args.latitude_max,
        longitude_min=args.longitude_min,
        longitude_max=args.longitude_max,
        hr_pad=args.hr_pad,
    )
    lats, lons = _axes(axis_args, args.lat_factor_hr, args.lon_factor_hr)
    _, _, fft_indices, _ = _contrast_coordinates(samples, cube, fit, lats, lons)
    grid_shape = (len(lats), len(lons), 8)
    spacings = (
        BASE_LAT_STEP / args.lat_factor_hr,
        BASE_LON_STEP / args.lon_factor_hr,
        1.0,
    )

    requested_dgps = ("joint", "separable") if args.dgp == "both" else (args.dgp,)
    rows: list[dict[str, Any]] = []
    operator_rows: list[dict[str, Any]] = []
    for dgp_index, dgp in enumerate(requested_dgps):
        build_started = time.perf_counter()
        operator = build_bttb_operator(
            grid_shape,
            spacings,
            lambda hlat, hlon, htime, selected_dgp=dgp: _analytic_covariance(
                hlat, hlon, htime, fit, selected_dgp
            ),
        )
        build_seconds = time.perf_counter() - build_started
        diagnostic_rng = np.random.default_rng(args.seed + 90_000_000 + dgp_index)
        left = diagnostic_rng.standard_normal(operator.size)
        right = diagnostic_rng.standard_normal(operator.size)
        a_left = operator.matvec(left)
        a_right = operator.matvec(right)
        symmetry_scale = max(abs(float(left @ a_right)), abs(float(right @ a_left)), 1.0)
        contrast_coefficient = d1 * coefficient_a + d2 * coefficient_b
        validation_errors = []
        for contrast_index in np.linspace(
            0, len(fft_indices) - 1, num=min(9, len(fft_indices)), dtype=np.int64
        ):
            indices = fft_indices[contrast_index]
            flat_indices = np.ravel_multi_index(
                (indices[:, 0], indices[:, 1], indices[:, 2]), grid_shape
            )
            sparse = np.zeros(operator.size, dtype=np.float64)
            np.add.at(sparse, flat_indices, contrast_coefficient)
            operator_variance = float(sparse @ operator.matvec(sparse))
            coordinates = np.column_stack(
                [lats[indices[:, 0]], lons[indices[:, 1]], indices[:, 2]]
            )
            delta = coordinates[:, None, :] - coordinates[None, :, :]
            dense_covariance = _analytic_covariance(
                delta[..., 0], delta[..., 1], delta[..., 2], fit, dgp
            )
            direct_variance = float(
                contrast_coefficient @ dense_covariance @ contrast_coefficient
            )
            validation_errors.append(abs(operator_variance - direct_variance))
        operator_rows.append(
            {
                "dgp": dgp,
                "grid_lat": grid_shape[0],
                "grid_lon": grid_shape[1],
                "grid_time": grid_shape[2],
                "vector_size": operator.size,
                "convolution_lat": operator.convolution_shape[0],
                "convolution_lon": operator.convolution_shape[1],
                "convolution_time": operator.convolution_shape[2],
                "operator_build_seconds": build_seconds,
                "symmetry_relative_error": abs(float(left @ a_right - right @ a_left)) / symmetry_scale,
                "left_quadratic_form": float(left @ a_left),
                "right_quadratic_form": float(right @ a_right),
                "max_abs_fixed_contrast_variance_error": float(max(validation_errors)),
            }
        )
        for replicate in range(args.replicates):
            seed = int(args.seed + dgp_index * 1_000_000 + replicate)
            rng = np.random.default_rng(seed)
            normal = rng.standard_normal(operator.size)
            decomposition_started = time.perf_counter()
            decomposition = lanczos_decomposition(
                operator.matvec,
                normal,
                max_dimension=max(dimensions),
                full_reorthogonalization=True,
            )
            decomposition_seconds = time.perf_counter() - decomposition_started
            previous = None
            actions: dict[int, np.ndarray] = {}
            for dimension in dimensions:
                if dimension > len(decomposition.diagonal):
                    continue
                field_vector, lanczos_diag = sqrt_action(decomposition, dimension)
                actions[dimension] = field_vector
                relative_previous = (
                    np.linalg.norm(field_vector - previous) / np.linalg.norm(field_vector)
                    if previous is not None
                    else np.nan
                )
                field = field_vector.reshape(grid_shape)
                point_values = field[
                    fft_indices[..., 0],
                    fft_indices[..., 1],
                    fft_indices[..., 2],
                ]
                row = {
                    "dgp": dgp,
                    "replicate": replicate,
                    "seed": seed,
                    "lanczos_dimension": dimension,
                    "available_lanczos_dimension": len(decomposition.diagonal),
                    "lanczos_breakdown": decomposition.breakdown,
                    "relative_field_change_from_previous": relative_previous,
                    "decomposition_seconds": decomposition_seconds,
                    **lanczos_diag,
                    **_field_moments(point_values, coefficient_a, coefficient_b, d1, d2),
                }
                rows.append(row)
                previous = field_vector
            maximum_action = actions[max(actions)]
            maximum_norm = np.linalg.norm(maximum_action)
            for row in rows:
                if row["dgp"] == dgp and row["replicate"] == replicate:
                    action = actions[int(row["lanczos_dimension"])]
                    row["relative_field_error_vs_max_dimension"] = float(
                        np.linalg.norm(action - maximum_action) / maximum_norm
                    )
            print(
                f"dgp={dgp} replicate={replicate} m={max(actions)} "
                f"seconds={decomposition_seconds:.2f} "
                f"Traw={rows[-1]['empirical_mean_l_squared']:.6f}",
                flush=True,
            )

    results = pd.DataFrame(rows)
    operators = pd.DataFrame(operator_rows)
    _atomic_csv(output_dir / "lanczos_replicates.csv", results)
    _atomic_csv(output_dir / "operator_diagnostics.csv", operators)
    convergence = results.groupby(["dgp", "lanczos_dimension"], as_index=False).agg(
        replicate_count=("replicate", "count"),
        max_relative_field_error_vs_max=("relative_field_error_vs_max_dimension", "max"),
        mean_relative_field_error_vs_max=("relative_field_error_vs_max_dimension", "mean"),
        mean_l_squared=("empirical_mean_l_squared", "mean"),
        sd_l_squared=("empirical_mean_l_squared", "std"),
    )
    audit_path = (
        args.oracle_dir.expanduser()
        / "fft_generator_frozen_contrast_audit_20240701/frozen_contrast_generator_bias.csv"
    )
    analytic_reference = pd.read_csv(audit_path)
    clipped_reference = analytic_reference.loc[
        (analytic_reference["component"] == "variance_l")
        & (analytic_reference["embedding_factor"] == 8)
    ][["dgp", "fft_generated"]].rename(columns={"fft_generated": "clipped_8x_var_l"})
    mapping_path = (
        args.oracle_dir.expanduser()
        / "fft_generator_frozen_contrast_audit_20240701/frozen_contrast_mapping_bias.csv"
    )
    mapping_reference = pd.read_csv(mapping_path)
    reference_rows = mapping_reference.loc[
        (mapping_reference["component"] == "variance_l")
        & (mapping_reference["lat_factor_hr"] == args.lat_factor_hr)
        & (mapping_reference["lon_factor_hr"] == args.lon_factor_hr)
    ][["dgp", "target_mapped_source"]].rename(
        columns={"target_mapped_source": "analytic_target_mapped_var_l"}
    )
    if args.lat_factor_hr == 1 and args.lon_factor_hr == 1:
        reference_rows = reference_rows.merge(
            clipped_reference, on="dgp", how="left", validate="one_to_one"
        )
    else:
        reference_rows["clipped_8x_var_l"] = np.nan
    convergence = convergence.merge(reference_rows, on="dgp", how="left", validate="many_to_one")
    convergence["mean_minus_analytic_target"] = (
        convergence["mean_l_squared"] - convergence["analytic_target_mapped_var_l"]
    )
    _atomic_csv(output_dir / "lanczos_convergence.csv", convergence)
    lines = [
        "# Matrix-free Lanczos generator sensitivity",
        "",
        "The finite regular-lattice target covariance was not replaced by a clipped circulant covariance. Zero-padded FFT was used only for exact BTTB matrix-vector products, and Lanczos approximated the square-root action on a fixed Gaussian vector.",
        "",
        "| DGP | m | replicates | max relative field error vs largest m | mean L squared | analytic target | clipped 8x expectation |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in convergence.itertuples(index=False):
        lines.append(
            f"| `{row.dgp}` | {row.lanczos_dimension} | {row.replicate_count} | "
            f"{row.max_relative_field_error_vs_max:.6g} | {row.mean_l_squared:.9g} | "
            f"{row.analytic_target_mapped_var_l:.9g} | {row.clipped_8x_var_l:.9g} |"
        )
    lines.extend(
        [
            "",
            "These few fields assess numerical generator convergence only; they are not a bootstrap calibration and their empirical contrast moments have Monte Carlo variation. The analytic target moments remain the reference for generator bias.",
            "",
        ]
    )
    _atomic_text(output_dir / "REPORT.md", "\n".join(lines))
    manifest = {
        "data_file": str(args.data_file.expanduser().resolve()),
        "fit_csv": str(args.fit_csv.expanduser().resolve()),
        "coefficient_source": str(coefficient_path.resolve()),
        "grid_shape": list(grid_shape),
        "spacings": list(spacings),
        "dgp": args.dgp,
        "replicates": args.replicates,
        "dimensions": list(dimensions),
        "seed": args.seed,
        "method": "zero-padded BTTB FFT matvec plus fully reorthogonalized symmetric Lanczos square-root action",
        "circulant_spectrum_clipping": False,
    }
    _atomic_text(output_dir / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(operators.to_string(index=False), flush=True)
    print(convergence.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
