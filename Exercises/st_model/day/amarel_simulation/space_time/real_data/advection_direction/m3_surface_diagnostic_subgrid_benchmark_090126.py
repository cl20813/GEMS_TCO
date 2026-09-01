#!/usr/bin/env python3
"""Benchmark a parsimonious diagnostic-gated M3 sub-grid initializer.

The three paired methods deliberately share exactly the same tau=1 masked-FFT
semivariogram surface:

M3_fft
    Discrete regular-grid minimum.
M3_fft_Q
    Safeguarded 3x3 quadratic interpolation, limited to half of one grid cell.
M3_fft_JKQ
    The same interpolation only when a single scientific diagnostic passes.

The diagnostic is leave-one-transition-out (jackknife) sub-grid stability. For
each of the seven consecutive hourly transitions, recompute the pooled surface
without that transition and record the change in the safeguarded quadratic
seed in grid-cell units.  The scalar diagnostic JQ90 is the 90th percentile
Chebyshev displacement.  The primary gate is JQ90 <= 0.5 cell, with at least
six of seven leave-one-out quadratic fits passing the numerical safeguards.

Positive curvature, Hessian conditioning, an interior 3x3 patch, and a
half-cell displacement bound are numerical safeguards for interpolation; they
are not counted as additional scientific ambiguity criteria.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch


REPO = Path("/Users/joonwonlee/Documents/GEMS_TCO-1")
SRC = REPO / "src"
HERE = Path(__file__).resolve().parent
for path in (REPO, SRC, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from synthetic_advection_initializer_benchmark_corridor432_081526 import (
    DayAsset,
    P_LABELS,
    clean_json,
    load_assets,
    make_hourly_grids,
    parse_pair,
    vector_metrics,
)
from synthetic_initializer_factorial_robustness_corridor432_083126 import (
    coordinate_indices,
    crop_regular_asset,
    simulate_asset,
)
from synthetic_m3_reverse_l_initialization_benchmark_corridor432_083126 import (
    fft_pair_squared_difference,
    smooth_semivariogram,
    surface_minimum,
)


METHODS = ("M3_fft", "M3_fft_Q", "M3_fft_JKQ")
SIGNAL_REGIMES = {
    "strong": {"range_time_factor": 1.50, "nugget_factor": 0.50},
    "reference": {"range_time_factor": 1.00, "nugget_factor": 1.00},
    "weak": {"range_time_factor": 0.50, "nugget_factor": 3.00},
}


def parse_int_list(text: str) -> list[int]:
    return [int(token.strip()) for token in str(text).split(",") if token.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(token.strip()) for token in str(text).split(",") if token.strip()]


def transition_components(
    grids: Sequence[np.ndarray],
    offsets_lat: np.ndarray,
    offsets_lon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sumsq = []
    counts = []
    for hour in range(len(grids) - 1):
        pair_sumsq, pair_counts = fft_pair_squared_difference(
            grids[hour], grids[hour + 1], offsets_lat, offsets_lon
        )
        sumsq.append(pair_sumsq)
        counts.append(pair_counts)
    return np.stack(sumsq, axis=0), np.stack(counts, axis=0)


def pooled_surface(
    sumsq: np.ndarray,
    counts: np.ndarray,
    selected: np.ndarray,
    lat_step: float,
    lon_step: float,
    min_pair_count: int,
    bandwidth_deg: float,
) -> dict[str, np.ndarray]:
    total_sumsq = np.sum(sumsq[selected], axis=0)
    total_counts = np.sum(counts[selected], axis=0)
    # Scale the full-data threshold by the retained transition fraction.
    required = max(
        1,
        int(math.ceil(float(min_pair_count) * len(selected) / max(1, sumsq.shape[0]))),
    )
    gamma = np.full_like(total_sumsq, np.nan, dtype=np.float64)
    valid = total_counts >= required
    gamma[valid] = 0.5 * total_sumsq[valid] / total_counts[valid]
    smoothed = smooth_semivariogram(gamma, lat_step, lon_step, bandwidth_deg)
    return {"gamma": gamma, "smoothed": smoothed, "counts": total_counts}


def half_cell_quadratic_minimum(
    surface: np.ndarray,
    offsets_lat: np.ndarray,
    offsets_lon: np.ndarray,
    lat_step: float,
    lon_step: float,
    max_condition_number: float,
) -> dict[str, Any]:
    row, col, grid_lat, grid_lon = surface_minimum(
        surface, offsets_lat, offsets_lon, lat_step, lon_step
    )
    fallback: dict[str, Any] = {
        "seed_lat": grid_lat,
        "seed_lon": grid_lon,
        "grid_seed_lat": grid_lat,
        "grid_seed_lon": grid_lon,
        "subgrid_accepted": False,
        "subgrid_reason": "fallback",
        "subgrid_delta_lat": 0.0,
        "subgrid_delta_lon": 0.0,
        "subgrid_hessian_condition": np.nan,
        "subgrid_min_eigenvalue": np.nan,
    }
    if row < 1 or col < 1 or row >= surface.shape[0] - 1 or col >= surface.shape[1] - 1:
        fallback["subgrid_reason"] = "boundary"
        return fallback
    patch = np.asarray(surface[row - 1 : row + 2, col - 1 : col + 2], dtype=np.float64)
    if not np.isfinite(patch).all():
        fallback["subgrid_reason"] = "nonfinite_patch"
        return fallback

    design = []
    values = []
    for local_row, di in enumerate((-1, 0, 1)):
        for local_col, dj in enumerate((-1, 0, 1)):
            x = float(di * lat_step)
            y = float(dj * lon_step)
            design.append([1.0, x, y, 0.5 * x * x, x * y, 0.5 * y * y])
            values.append(float(patch[local_row, local_col]))
    coefficients, *_ = np.linalg.lstsq(
        np.asarray(design, dtype=np.float64), np.asarray(values, dtype=np.float64), rcond=None
    )
    gradient = coefficients[1:3]
    hessian = np.asarray(
        [[coefficients[3], coefficients[4]], [coefficients[4], coefficients[5]]],
        dtype=np.float64,
    )
    eigenvalues = np.linalg.eigvalsh(hessian)
    fallback["subgrid_min_eigenvalue"] = float(eigenvalues[0])
    if not np.all(eigenvalues > 0.0):
        fallback["subgrid_reason"] = "non_positive_hessian"
        return fallback
    condition = float(np.linalg.cond(hessian))
    fallback["subgrid_hessian_condition"] = condition
    if not np.isfinite(condition) or condition > float(max_condition_number):
        fallback["subgrid_reason"] = "ill_conditioned_hessian"
        return fallback
    delta = -np.linalg.solve(hessian, gradient)
    if (
        abs(float(delta[0])) > 0.5 * abs(float(lat_step))
        or abs(float(delta[1])) > 0.5 * abs(float(lon_step))
    ):
        fallback["subgrid_reason"] = "delta_outside_half_cell"
        return fallback
    return {
        **fallback,
        "seed_lat": float(grid_lat + delta[0]),
        "seed_lon": float(grid_lon + delta[1]),
        "subgrid_accepted": True,
        "subgrid_reason": "accepted",
        "subgrid_delta_lat": float(delta[0]),
        "subgrid_delta_lon": float(delta[1]),
        "subgrid_hessian_condition": condition,
        "subgrid_min_eigenvalue": float(eigenvalues[0]),
    }


def jackknife_basin_diagnostic(
    sumsq: np.ndarray,
    counts: np.ndarray,
    full_grid_seed: tuple[float, float],
    offsets_lat: np.ndarray,
    offsets_lon: np.ndarray,
    lat_step: float,
    lon_step: float,
    min_pair_count: int,
    bandwidth_deg: float,
    threshold_cells: float,
    full_quadratic_seed: tuple[float, float],
    max_condition_number: float,
) -> dict[str, Any]:
    n_transition = int(sumsq.shape[0])
    full_lat, full_lon = map(float, full_grid_seed)
    distances_euclid = []
    distances_chebyshev = []
    loo_lat = []
    loo_lon = []
    quadratic_distances_chebyshev = []
    quadratic_valid = []
    loo_q_lat = []
    loo_q_lon = []
    full_q_lat, full_q_lon = map(float, full_quadratic_seed)
    for omitted in range(n_transition):
        selected = np.asarray([idx for idx in range(n_transition) if idx != omitted], dtype=int)
        surface = pooled_surface(
            sumsq,
            counts,
            selected,
            lat_step,
            lon_step,
            min_pair_count,
            bandwidth_deg,
        )["smoothed"]
        _, _, seed_lat, seed_lon = surface_minimum(
            surface, offsets_lat, offsets_lon, lat_step, lon_step
        )
        dlat_cells = (float(seed_lat) - full_lat) / abs(float(lat_step))
        dlon_cells = (float(seed_lon) - full_lon) / abs(float(lon_step))
        loo_lat.append(float(seed_lat))
        loo_lon.append(float(seed_lon))
        distances_euclid.append(float(np.hypot(dlat_cells, dlon_cells)))
        distances_chebyshev.append(float(max(abs(dlat_cells), abs(dlon_cells))))
        loo_q = half_cell_quadratic_minimum(
            surface,
            offsets_lat,
            offsets_lon,
            lat_step,
            lon_step,
            max_condition_number,
        )
        loo_q_lat.append(float(loo_q["seed_lat"]))
        loo_q_lon.append(float(loo_q["seed_lon"]))
        quadratic_valid.append(bool(loo_q["subgrid_accepted"]))
        q_dlat_cells = (float(loo_q["seed_lat"]) - full_q_lat) / abs(float(lat_step))
        q_dlon_cells = (float(loo_q["seed_lon"]) - full_q_lon) / abs(float(lon_step))
        quadratic_distances_chebyshev.append(
            float(max(abs(q_dlat_cells), abs(q_dlon_cells)))
        )
    cheb = np.asarray(distances_chebyshev, dtype=np.float64)
    euclid = np.asarray(distances_euclid, dtype=np.float64)
    q_cheb = np.asarray(quadratic_distances_chebyshev, dtype=np.float64)
    j90 = float(np.quantile(cheb, 0.90))
    q_j90 = float(np.quantile(q_cheb, 0.90))
    q_valid_fraction = float(np.mean(quadratic_valid))
    return {
        "jk_j90_cells": j90,
        "jk_max_cells": float(np.max(cheb)),
        "jk_median_cells": float(np.median(cheb)),
        "jk_mean_euclid_cells": float(np.mean(euclid)),
        "jk_stable_fraction_1cell": float(np.mean(cheb <= 1.0 + 1e-12)),
        "surface_clear": bool(j90 <= float(threshold_cells)),
        "jkq_j90_cells": q_j90,
        "jkq_max_cells": float(np.max(q_cheb)),
        "jkq_stable_fraction_halfcell": float(np.mean(q_cheb <= 0.5 + 1e-12)),
        "jkq_valid_fraction": q_valid_fraction,
        "subgrid_stable": bool(
            q_j90 <= float(threshold_cells) and q_valid_fraction >= 6.0 / 7.0
        ),
        "jk_threshold_cells": float(threshold_cells),
        "jk_loo_seed_lat": json.dumps(loo_lat),
        "jk_loo_seed_lon": json.dumps(loo_lon),
        "jk_loo_q_seed_lat": json.dumps(loo_q_lat),
        "jk_loo_q_seed_lon": json.dumps(loo_q_lon),
    }


def run_three_initializers(asset: DayAsset, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    total_started = time.perf_counter()
    grids, lat_step, lon_step = make_hourly_grids(asset.source_map, asset.grid_coords)
    offsets_lat = np.arange(-int(args.empirical_max_lat_offset), int(args.empirical_max_lat_offset) + 1)
    offsets_lon = np.arange(-int(args.empirical_max_lon_offset), int(args.empirical_max_lon_offset) + 1)
    components_started = time.perf_counter()
    sumsq, counts = transition_components(grids, offsets_lat, offsets_lon)
    components_s = time.perf_counter() - components_started
    aggregate_started = time.perf_counter()
    full = pooled_surface(
        sumsq,
        counts,
        np.arange(sumsq.shape[0]),
        lat_step,
        lon_step,
        int(args.empirical_min_pair_count),
        float(args.empirical_smooth_bandwidth_deg),
    )
    aggregate_s = time.perf_counter() - aggregate_started
    _, _, grid_lat, grid_lon = surface_minimum(
        full["smoothed"], offsets_lat, offsets_lon, lat_step, lon_step
    )
    m3_s = float(components_s + aggregate_s)

    quadratic_started = time.perf_counter()
    quadratic = half_cell_quadratic_minimum(
        full["smoothed"],
        offsets_lat,
        offsets_lon,
        lat_step,
        lon_step,
        float(args.subgrid_max_condition_number),
    )
    quadratic_s = time.perf_counter() - quadratic_started

    diagnostic_started = time.perf_counter()
    diagnostic = jackknife_basin_diagnostic(
        sumsq,
        counts,
        (grid_lat, grid_lon),
        offsets_lat,
        offsets_lon,
        lat_step,
        lon_step,
        int(args.empirical_min_pair_count),
        float(args.empirical_smooth_bandwidth_deg),
        float(args.jkq_threshold_cells),
        (float(quadratic["seed_lat"]), float(quadratic["seed_lon"])),
        float(args.subgrid_max_condition_number),
    )
    diagnostic_s = time.perf_counter() - diagnostic_started

    if bool(diagnostic["subgrid_stable"]) and bool(quadratic["subgrid_accepted"]):
        gated_lat = float(quadratic["seed_lat"])
        gated_lon = float(quadratic["seed_lon"])
        gated_reason = "clear_and_quadratic_accepted"
        gated_accepted = True
    elif not bool(diagnostic["subgrid_stable"]):
        gated_lat, gated_lon = float(grid_lat), float(grid_lon)
        gated_reason = "jackknife_subgrid_unstable"
        gated_accepted = False
    else:
        gated_lat, gated_lon = float(grid_lat), float(grid_lon)
        gated_reason = str(quadratic["subgrid_reason"])
        gated_accepted = False

    common = {
        **diagnostic,
        "lat_step": float(lat_step),
        "lon_step": float(lon_step),
        "fft_components_s": float(components_s),
        "fft_aggregate_s": float(aggregate_s),
        "quadratic_s": float(quadratic_s),
        "diagnostic_s": float(diagnostic_s),
        "subgrid_numerically_accepted": bool(quadratic["subgrid_accepted"]),
        "subgrid_numerical_reason": str(quadratic["subgrid_reason"]),
        "subgrid_hessian_condition": float(quadratic["subgrid_hessian_condition"]),
        "subgrid_min_eigenvalue": float(quadratic["subgrid_min_eigenvalue"]),
        "subgrid_delta_lat": float(quadratic["subgrid_delta_lat"]),
        "subgrid_delta_lon": float(quadratic["subgrid_delta_lon"]),
    }
    rows = [
        {
            "method": "M3_fft",
            "seed_lat": float(grid_lat),
            "seed_lon": float(grid_lon),
            "seed_total_s": m3_s,
            "subgrid_used": False,
            "selection_reason": "discrete_fft_argmin",
            **common,
        },
        {
            "method": "M3_fft_Q",
            "seed_lat": float(quadratic["seed_lat"]),
            "seed_lon": float(quadratic["seed_lon"]),
            "seed_total_s": float(m3_s + quadratic_s),
            "subgrid_used": bool(quadratic["subgrid_accepted"]),
            "selection_reason": str(quadratic["subgrid_reason"]),
            **common,
        },
        {
            "method": "M3_fft_JKQ",
            "seed_lat": gated_lat,
            "seed_lon": gated_lon,
            "seed_total_s": float(m3_s + quadratic_s + diagnostic_s),
            "subgrid_used": gated_accepted,
            "selection_reason": gated_reason,
            **common,
        },
    ]
    work = {
        "surface": full,
        "offsets_lat": offsets_lat,
        "offsets_lon": offsets_lon,
        "lat_step": lat_step,
        "lon_step": lon_step,
        "wall_s": float(time.perf_counter() - total_started),
    }
    return rows, work


def truth_conditions(
    lat_step: float,
    lon_step: float,
    speeds: Sequence[float],
    angle_offset_deg: float,
) -> list[dict[str, Any]]:
    conditions = []
    for angle_deg in np.arange(float(angle_offset_deg), 360.0, 45.0):
        theta = math.radians(float(angle_deg))
        for speed_cells in speeds:
            advec_lat = float(speed_cells * lat_step * math.cos(theta))
            advec_lon = float(speed_cells * lon_step * math.sin(theta))
            conditions.append(
                {
                    "direction_deg_grid": float(angle_deg),
                    "speed_cells": float(speed_cells),
                    "advec_lat": advec_lat,
                    "advec_lon": advec_lon,
                    "truth_id": f"a{int(angle_deg):03d}_s{str(speed_cells).replace('.', 'p')}",
                }
            )
    return conditions


def add_truth_metrics(row: dict[str, Any], truth: dict[str, float], lat_step: float, lon_step: float) -> None:
    row.update(vector_metrics(row["seed_lat"], row["seed_lon"], truth))
    row["seed_error_grid_cells"] = float(
        np.hypot(
            (float(row["seed_lat"]) - float(truth["advec_lat"])) / abs(lat_step),
            (float(row["seed_lon"]) - float(truth["advec_lon"])) / abs(lon_step),
        )
    )


def summarize_synthetic(detail: pd.DataFrame, output_root: Path) -> None:
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
    summary.to_csv(output_root / "synthetic_summary.csv", index=False, float_format="%.10f")

    wide = detail.pivot(index="dataset_id", columns="method", values="seed_error_grid_cells")
    paired = pd.DataFrame(index=wide.index)
    paired["Q_minus_M3_cells"] = wide["M3_fft_Q"] - wide["M3_fft"]
    paired["JKQ_minus_M3_cells"] = wide["M3_fft_JKQ"] - wide["M3_fft"]
    diag = detail[detail.method.eq("M3_fft")].set_index("dataset_id")
    paired = paired.join(
        diag[
            [
                "signal_regime",
                "speed_cells",
                "jk_j90_cells",
                "surface_clear",
                "jkq_j90_cells",
                "jkq_valid_fraction",
                "subgrid_stable",
            ]
        ]
    ).reset_index()
    paired.to_csv(output_root / "synthetic_paired_effects.csv", index=False, float_format="%.10f")

    threshold_rows = []
    q_meta = detail[detail.method.eq("M3_fft_Q")].set_index("dataset_id")
    for threshold in (0.10, 0.25, 0.50, 0.75, 1.00):
        use_q = (
            (diag["jkq_j90_cells"] <= threshold)
            & (diag["jkq_valid_fraction"] >= 6.0 / 7.0)
            & q_meta["subgrid_numerically_accepted"]
        )
        gated_error = wide["M3_fft"].where(~use_q, wide["M3_fft_Q"])
        delta = gated_error - wide["M3_fft"]
        threshold_rows.append(
            {
                "threshold_cells": threshold,
                "acceptance_rate": float(use_q.mean()),
                "mean_error_cells": float(gated_error.mean()),
                "median_error_cells": float(gated_error.median()),
                "harm_rate": float((delta > 1e-12).mean()),
                "material_harm_rate_0p25cell": float((delta > 0.25).mean()),
                "improvement_rate": float((delta < -1e-12).mean()),
            }
        )
    pd.DataFrame(threshold_rows).to_csv(
        output_root / "diagnostic_threshold_sensitivity.csv", index=False, float_format="%.10f"
    )


def run_synthetic(args: argparse.Namespace) -> None:
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
    lat_range = parse_pair(args.lat_range, float)
    lon_range = parse_pair(args.lon_range, float)
    templates = load_assets(
        input_path,
        base_json,
        list(range(int(args.replicates))),
        lat_range,
        lon_range,
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
    unknown = sorted(set(regimes) - set(SIGNAL_REGIMES))
    if unknown:
        raise ValueError(f"Unknown signal regimes: {unknown}")

    config = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "mode": "synthetic",
        "script": str(Path(__file__).resolve()),
        "input_template": str(input_path),
        "base_truth": base_truth,
        "conditions": conditions,
        "signal_regimes": {key: SIGNAL_REGIMES[key] for key in regimes},
        "methods": METHODS,
        "diagnostic_definition": (
            "JQ90 of LOTO safeguarded-quadratic seed Chebyshev displacement in grid cells"
        ),
        "arguments": vars(args),
    }
    (output_root / "run_config.json").write_text(
        json.dumps(clean_json(config), indent=2, sort_keys=True), encoding="utf-8"
    )

    rows: list[dict[str, Any]] = []
    simulation_rows: list[dict[str, Any]] = []
    total = len(conditions) * len(regimes) * int(args.replicates)
    counter = 0
    started = time.perf_counter()
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
                asset, sim_diag = simulate_asset(
                    templates[replicate],
                    truth,
                    simulation_seed,
                    args,
                    f"{condition['truth_id']}_{regime}",
                    replicate,
                )
                dataset_id = f"{condition['truth_id']}_{regime}_r{replicate:02d}"
                sim_diag.update(
                    {
                        "dataset_id": dataset_id,
                        "signal_regime": regime,
                        "speed_cells": condition["speed_cells"],
                        "direction_deg_grid": condition["direction_deg_grid"],
                        "true_range_time": truth["range_time"],
                        "true_nugget": truth["nugget"],
                    }
                )
                simulation_rows.append(sim_diag)
                method_rows, _ = run_three_initializers(asset, args)
                for row in method_rows:
                    row.update(
                        {
                            "dataset_id": dataset_id,
                            "replicate": replicate,
                            "signal_regime": regime,
                            "speed_cells": condition["speed_cells"],
                            "direction_deg_grid": condition["direction_deg_grid"],
                            "true_advec_lat": truth["advec_lat"],
                            "true_advec_lon": truth["advec_lon"],
                            "true_range_time": truth["range_time"],
                            "true_nugget": truth["nugget"],
                        }
                    )
                    add_truth_metrics(row, truth, lat_step, lon_step)
                    rows.append(clean_json(row))
                if counter % 10 == 0 or counter == total:
                    print(f"[{counter}/{total}] {dataset_id}", flush=True)
                    pd.DataFrame(rows).to_csv(
                        output_root / "synthetic_initializer_results.csv",
                        index=False,
                        float_format="%.10f",
                    )
                    pd.DataFrame(simulation_rows).to_csv(
                        output_root / "synthetic_simulation_diagnostics.csv",
                        index=False,
                        float_format="%.10f",
                    )
    detail = pd.DataFrame(rows)
    summarize_synthetic(detail, output_root)
    elapsed = float(time.perf_counter() - started)
    (output_root / "total_runtime.json").write_text(
        json.dumps({"wall_s": elapsed, "wall_minutes": elapsed / 60.0}, indent=2),
        encoding="utf-8",
    )
    print(pd.read_csv(output_root / "synthetic_summary.csv").to_string(index=False), flush=True)


def load_real_assets(args: argparse.Namespace) -> tuple[list[DayAsset], dict[str, Any]]:
    from GEMS_TCO.data_loader import load_data_dynamic_processed

    loader = load_data_dynamic_processed(str(args.real_data_root))
    started = time.perf_counter()
    df_map, _, _, monthly_mean = loader.load_maxmin_ordered_data_bymonthyear(
        lat_lon_resolution=[1, 1],
        mm_cond_number=1,
        years_=[str(args.year)],
        months_=[int(args.month)],
        lat_range=[-3.0, 2.0],
        lon_range=[121.0, 131.0],
        is_whittle=True,
    )
    monthly_load_s = float(time.perf_counter() - started)
    keys = sorted(df_map)
    grid_coords = df_map[keys[0]][["Latitude", "Longitude"]].to_numpy(dtype=np.float64)
    assets = []
    day_build_s = {}
    for day_idx in parse_int_list(args.days):
        day_started = time.perf_counter()
        day_map, _ = loader.load_working_data(
            df_map,
            monthly_mean,
            [day_idx * 8, (day_idx + 1) * 8],
            ord_mm=None,
            dtype=torch.double,
            keep_ori=True,
        )
        if len(day_map) != 8:
            continue
        label = f"{args.year}-{args.month:02d}-{day_idx + 1:02d}"
        day_build_s[label] = float(time.perf_counter() - day_started)
        total = sum(int(value.shape[0]) for value in day_map.values())
        valid = sum(int(torch.isfinite(value[:, 2]).sum().item()) for value in day_map.values())
        assets.append(
            DayAsset(
                year=int(args.year),
                month=int(args.month),
                day_idx=int(day_idx),
                day=label,
                source_map=day_map,
                grid_coords=grid_coords,
                n_valid=valid,
                n_total=total,
            )
        )
    return assets, {"monthly_load_s": monthly_load_s, "day_build_s": day_build_s}


def run_real(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    assets, loading = load_real_assets(args)
    rows = []
    started = time.perf_counter()
    for asset in assets:
        method_rows, _ = run_three_initializers(asset, args)
        for row in method_rows:
            row.update(
                {
                    "day": asset.day,
                    "day_idx": asset.day_idx,
                    "points_per_hour": len(asset.grid_coords),
                    "rows_total": asset.n_total,
                    "rows_valid": asset.n_valid,
                    "valid_rate": asset.n_valid / asset.n_total,
                }
            )
            rows.append(clean_json(row))
        print(
            f"{asset.day} M3=({method_rows[0]['seed_lat']:+.4f},{method_rows[0]['seed_lon']:+.4f}) "
            f"JQ90={method_rows[0]['jkq_j90_cells']:.2f} stable={method_rows[0]['subgrid_stable']} "
            f"JKQ_used={method_rows[2]['subgrid_used']}",
            flush=True,
        )
    detail = pd.DataFrame(rows)
    detail.to_csv(output_root / "real_initializer_results.csv", index=False, float_format="%.10f")
    base = detail[detail.method.eq("M3_fft")].set_index("day")
    quadratic = detail[detail.method.eq("M3_fft_Q")].set_index("day")
    gated = detail[detail.method.eq("M3_fft_JKQ")].set_index("day")
    daily = pd.DataFrame(index=base.index)
    daily["m3_seed_lat"] = base["seed_lat"]
    daily["m3_seed_lon"] = base["seed_lon"]
    daily["q_seed_lat"] = quadratic["seed_lat"]
    daily["q_seed_lon"] = quadratic["seed_lon"]
    daily["jkq_seed_lat"] = gated["seed_lat"]
    daily["jkq_seed_lon"] = gated["seed_lon"]
    daily["q_movement_cells"] = np.hypot(
        (quadratic["seed_lat"] - base["seed_lat"]) / base["lat_step"],
        (quadratic["seed_lon"] - base["seed_lon"]) / base["lon_step"],
    )
    for column in (
        "jk_j90_cells",
        "jkq_j90_cells",
        "jkq_valid_fraction",
        "surface_clear",
        "subgrid_stable",
        "subgrid_numerically_accepted",
        "subgrid_numerical_reason",
        "diagnostic_s",
    ):
        daily[column] = base[column]
    daily.reset_index().to_csv(
        output_root / "real_day_diagnostics.csv", index=False, float_format="%.10f"
    )
    summary = (
        detail.groupby("method", as_index=False)
        .agg(
            n_days=("day", "size"),
            mean_seed_s=("seed_total_s", "mean"),
            median_seed_s=("seed_total_s", "median"),
            subgrid_use_rate=("subgrid_used", "mean"),
            mean_jkq_j90_cells=("jkq_j90_cells", "mean"),
        )
    )
    summary.to_csv(output_root / "real_summary.csv", index=False, float_format="%.10f")
    config = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "mode": "real",
        "script": str(Path(__file__).resolve()),
        "methods": METHODS,
        "diagnostic_definition": (
            "JQ90 of LOTO safeguarded-quadratic seed Chebyshev displacement in grid cells"
        ),
        "loading": loading,
        "arguments": vars(args),
        "wall_s_excluding_month_load": float(time.perf_counter() - started),
    }
    (output_root / "run_config.json").write_text(
        json.dumps(clean_json(config), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("synthetic", "real"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--synthetic-data-root",
        type=Path,
        default=Path(
            "/Users/joonwonlee/Documents/GEMS_DATA/simulation/"
            "july_st_circulant_realpattern_smooth0p5"
        ),
    )
    parser.add_argument(
        "--real-data-root", type=Path, default=Path("/Users/joonwonlee/Documents/GEMS_DATA")
    )
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--month", type=int, default=7)
    parser.add_argument("--days", default="0,1,2,3,4")
    parser.add_argument("--lat-range", default="-3,-1")
    parser.add_argument("--lon-range", default="121,125")
    parser.add_argument("--n-lat", type=int, default=32)
    parser.add_argument("--n-lon", type=int, default=48)
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--speeds-cells", default="0.75,2.0,4.0")
    parser.add_argument(
        "--angle-offset-deg",
        type=float,
        default=22.5,
        help="Offset eight 45-degree-spaced directions away from grid axes/diagonals.",
    )
    parser.add_argument("--signal-regimes", default="strong,reference,weak")
    parser.add_argument("--seed", type=int, default=901260)
    parser.add_argument("--smooth", type=float, default=0.5)
    parser.add_argument("--empirical-max-lat-offset", type=int, default=20)
    parser.add_argument("--empirical-max-lon-offset", type=int, default=20)
    parser.add_argument("--empirical-min-pair-count", type=int, default=1000)
    parser.add_argument("--empirical-smooth-bandwidth-deg", type=float, default=0.063)
    parser.add_argument("--subgrid-max-condition-number", type=float, default=100.0)
    parser.add_argument("--jkq-threshold-cells", type=float, default=0.5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "synthetic":
        run_synthetic(args)
    else:
        run_real(args)


if __name__ == "__main__":
    main()
