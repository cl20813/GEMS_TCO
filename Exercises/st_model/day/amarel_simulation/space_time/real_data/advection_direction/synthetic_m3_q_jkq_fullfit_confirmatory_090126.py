#!/usr/bin/env python3
"""Paired downstream confirmation of M3, M3+Q, and diagnostic-gated M3+Q.

For every simulated eight-hour dataset, the three initializers share the same
masked-FFT tau=1 surface.  Each selected seed constructs one directional
lag-432 corridor, which remains fixed while all seven covariance parameters
are optimized.  Within a dataset and nuisance-start scenario, all methods use
the same non-advection starts and optimizer budget.

Since own-corridor Vecchia objectives use different factorizations, final raw
parameters are additionally evaluated on one common oracle-direction corridor.
The experiment records numerical failures, gradient convergence diagnostics,
budget hits, common-objective values, parameter errors, optimizer work, and
timing separately; an operational regret flag is never called an optimizer
failure.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
import traceback
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

from m3_surface_diagnostic_subgrid_benchmark_090126 import (
    METHODS,
    SIGNAL_REGIMES,
    run_three_initializers,
    truth_conditions,
)
from synthetic_advection_initializer_benchmark_corridor432_081526 import (
    P_LABELS,
    clean_json,
    load_assets,
    parse_pair,
)
from synthetic_initializer_factorial_robustness_corridor432_083126 import (
    NUISANCE_SCENARIOS,
    POSITIVE_PARAMS,
    build_corridor_model,
    crop_regular_asset,
    evaluate_raw,
    fit_method_corridor,
    nuisance_start,
    optimize_model,
    simulate_asset,
)


def parse_float_list(text: str) -> list[float]:
    return [float(token.strip()) for token in str(text).split(",") if token.strip()]


def parse_angle_list(text: str) -> list[float]:
    return [float(token.strip()) for token in str(text).split(",") if token.strip()]


def condition_grid(
    lat_step: float,
    lon_step: float,
    speeds: Sequence[float],
    angles: Sequence[float],
) -> list[dict[str, Any]]:
    all_conditions = truth_conditions(lat_step, lon_step, speeds, 22.5)
    selected = []
    for condition in all_conditions:
        if any(
            math.isclose(float(condition["direction_deg_grid"]), float(angle), abs_tol=1e-8)
            for angle in angles
        ):
            selected.append(condition)
    expected = len(speeds) * len(angles)
    if len(selected) != expected:
        raise RuntimeError(f"Selected {len(selected)} direction-speed conditions; expected {expected}")
    return selected


def initializer_record(
    dataset_id: str,
    condition: dict[str, Any],
    signal_regime: str,
    replicate: int,
    method_row: dict[str, Any],
    truth: dict[str, float],
) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "truth_id": condition["truth_id"],
        "direction_deg_grid": float(condition["direction_deg_grid"]),
        "speed_cells": float(condition["speed_cells"]),
        "signal_regime": signal_regime,
        "replicate": int(replicate),
        "method": method_row["method"],
        "seed_lat": float(method_row["seed_lat"]),
        "seed_lon": float(method_row["seed_lon"]),
        "seed_total_s": float(method_row["seed_total_s"]),
        "seed_error_euclid": float(
            np.hypot(
                float(method_row["seed_lat"]) - float(truth["advec_lat"]),
                float(method_row["seed_lon"]) - float(truth["advec_lon"]),
            )
        ),
        "subgrid_used": bool(method_row["subgrid_used"]),
        "selection_reason": method_row["selection_reason"],
        "jkq_j90_cells": float(method_row["jkq_j90_cells"]),
        "jkq_valid_fraction": float(method_row["jkq_valid_fraction"]),
        "subgrid_stable": bool(method_row["subgrid_stable"]),
        "subgrid_numerically_accepted": bool(method_row["subgrid_numerically_accepted"]),
        "subgrid_numerical_reason": method_row["subgrid_numerical_reason"],
    }


def add_case_regrets(
    full_rows: list[dict[str, Any]],
    row_indices: Sequence[int],
    reference_nll: float,
    regret_threshold: float,
    parameter_threshold: float,
) -> None:
    finite_indices = [
        idx
        for idx in row_indices
        if full_rows[idx].get("status") == "ok"
        and np.isfinite(full_rows[idx].get("common_eval_nll", np.nan))
    ]
    candidates = [float(reference_nll)] + [
        float(full_rows[idx]["common_eval_nll"]) for idx in finite_indices
    ]
    case_best = float(np.min(candidates))
    by_start: dict[str, float] = {}
    for idx in finite_indices:
        scenario = str(full_rows[idx]["nuisance_scenario"])
        by_start[scenario] = min(
            by_start.get(scenario, np.inf), float(full_rows[idx]["common_eval_nll"])
        )
    for idx in row_indices:
        row = full_rows[idx]
        numerical_failure = bool(row.get("status") != "ok" or not row.get("finite_fit", False))
        if numerical_failure:
            row.update(
                {
                    "common_regret_vs_case_best": np.nan,
                    "common_regret_vs_reference": np.nan,
                    "common_regret_within_start": np.nan,
                    "numerical_failure": True,
                    "gradient_nonconvergence": True,
                    "large_common_regret_flag": True,
                    "large_parameter_error_flag": True,
                    "any_operational_flag": True,
                }
            )
            continue
        value = float(row["common_eval_nll"])
        case_regret = value - case_best
        start_regret = value - by_start[str(row["nuisance_scenario"])]
        gradient_nonconvergence = not bool(row.get("converged_grad", False))
        large_regret = bool(case_regret > float(regret_threshold))
        large_parameter = bool(
            float(row["final_standardized_rmse_7param"]) > float(parameter_threshold)
        )
        row.update(
            {
                "common_regret_vs_case_best": float(case_regret),
                "common_regret_vs_reference": float(value - reference_nll),
                "common_regret_within_start": float(start_regret),
                "numerical_failure": False,
                "gradient_nonconvergence": gradient_nonconvergence,
                "large_common_regret_flag": large_regret,
                "large_parameter_error_flag": large_parameter,
                "any_operational_flag": bool(
                    gradient_nonconvergence or large_regret or large_parameter
                ),
            }
        )


def paired_bootstrap(
    full: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    comparisons = (
        ("M3_fft", "M3_fft_Q"),
        ("M3_fft", "M3_fft_JKQ"),
        ("M3_fft_JKQ", "M3_fft_Q"),
    )
    metrics = (
        "common_eval_nll",
        "common_regret_vs_case_best",
        "final_standardized_rmse_7param",
        "final_seed_error_euclid",
        "optimizer_n_iter",
        "optimizer_func_evals",
        "final_fit_s",
        "end_to_end_s",
    )
    ok = full[full.status.eq("ok")].copy()
    key = ["dataset_id", "nuisance_scenario"]
    rng = np.random.default_rng(int(seed))
    rows = []
    for baseline, competitor in comparisons:
        paired = ok[ok.method.eq(competitor)].merge(
            ok[ok.method.eq(baseline)], on=key, suffixes=("_new", "_base")
        )
        dataset_ids = np.asarray(sorted(paired.dataset_id.unique()))
        for metric in metrics:
            difference = paired[f"{metric}_new"] - paired[f"{metric}_base"]
            boot = []
            for _ in range(int(n_boot)):
                sampled_ids = rng.choice(dataset_ids, size=len(dataset_ids), replace=True)
                sampled_parts = [paired[paired.dataset_id.eq(item)] for item in sampled_ids]
                sampled = pd.concat(sampled_parts, ignore_index=True)
                boot.append(
                    float((sampled[f"{metric}_new"] - sampled[f"{metric}_base"]).mean())
                )
            rows.append(
                {
                    "baseline": baseline,
                    "competitor": competitor,
                    "metric": metric,
                    "n_paired_rows": int(len(paired)),
                    "n_dataset_clusters": int(len(dataset_ids)),
                    "mean_difference_competitor_minus_baseline": float(difference.mean()),
                    "median_difference_competitor_minus_baseline": float(difference.median()),
                    "paired_win_rate_lower_is_better": float((difference < 0).mean()),
                    "paired_tie_rate": float(np.isclose(difference, 0.0, atol=1e-12).mean()),
                    "cluster_bootstrap_ci_low": float(np.quantile(boot, 0.025)),
                    "cluster_bootstrap_ci_high": float(np.quantile(boot, 0.975)),
                }
            )
    return pd.DataFrame(rows)


def start_sensitivity(full: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ok = full[full.status.eq("ok")].copy()
    rows = []
    for (dataset_id, method), group in ok.groupby(["dataset_id", "method"], sort=False):
        truth_speed = float(np.hypot(group.true_advec_lat.iloc[0], group.true_advec_lon.iloc[0]))
        columns = []
        for parameter in P_LABELS:
            values = group[f"est_{parameter}"].to_numpy(dtype=float)
            truth_values = group[f"true_{parameter}"].to_numpy(dtype=float)
            if parameter in POSITIVE_PARAMS:
                columns.append(np.log(values / truth_values))
            else:
                columns.append((values - truth_values) / max(truth_speed, 1e-12))
        matrix = np.column_stack(columns)
        variance = np.var(matrix, axis=0, ddof=1) if len(group) > 1 else np.full(7, np.nan)
        rows.append(
            {
                "dataset_id": dataset_id,
                "method": method,
                "n_nuisance_starts": int(len(group)),
                "trace_standardized_start_variance": float(np.sum(variance)),
                "mean_standardized_start_variance": float(np.mean(variance)),
                **{
                    f"start_variance_{parameter}": float(value)
                    for parameter, value in zip(P_LABELS, variance)
                },
            }
        )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby("method", as_index=False)
        .agg(
            n_datasets=("dataset_id", "size"),
            mean_trace_start_variance=("trace_standardized_start_variance", "mean"),
            median_trace_start_variance=("trace_standardized_start_variance", "median"),
            p90_trace_start_variance=(
                "trace_standardized_start_variance",
                lambda x: float(np.quantile(x, 0.90)),
            ),
        )
    )
    return detail, summary


def write_summaries(
    output_root: Path,
    full_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    full = pd.DataFrame(full_rows)
    full.to_csv(output_root / "full_fit_results.csv", index=False, float_format="%.10f")
    summary = (
        full.groupby("method", as_index=False)
        .agg(
            n_cases=("dataset_id", "size"),
            numerical_success_rate=("numerical_failure", lambda x: float(1.0 - x.mean())),
            gradient_convergence_rate=("gradient_nonconvergence", lambda x: float(1.0 - x.mean())),
            iteration_budget_hit_rate=("hit_iteration_budget", "mean"),
            evaluation_budget_hit_rate=("hit_evaluation_budget", "mean"),
            large_common_regret_rate=("large_common_regret_flag", "mean"),
            large_parameter_error_rate=("large_parameter_error_flag", "mean"),
            mean_common_nll=("common_eval_nll", "mean"),
            mean_common_regret=("common_regret_vs_case_best", "mean"),
            median_common_regret=("common_regret_vs_case_best", "median"),
            mean_parameter_error=("final_standardized_rmse_7param", "mean"),
            median_parameter_error=("final_standardized_rmse_7param", "median"),
            mean_final_advection_error=("final_seed_error_euclid", "mean"),
            mean_optimizer_iterations=("optimizer_n_iter", "mean"),
            mean_function_evals=("optimizer_func_evals", "mean"),
            mean_fit_s=("final_fit_s", "mean"),
            mean_end_to_end_s=("end_to_end_s", "mean"),
        )
    )
    summary.to_csv(output_root / "overall_summary.csv", index=False, float_format="%.10f")
    by_scenario = (
        full.groupby(["nuisance_scenario", "method"], as_index=False)
        .agg(
            n_cases=("dataset_id", "size"),
            numerical_success_rate=("numerical_failure", lambda x: float(1.0 - x.mean())),
            gradient_convergence_rate=("gradient_nonconvergence", lambda x: float(1.0 - x.mean())),
            large_common_regret_rate=("large_common_regret_flag", "mean"),
            mean_common_regret=("common_regret_vs_case_best", "mean"),
            mean_parameter_error=("final_standardized_rmse_7param", "mean"),
            mean_final_advection_error=("final_seed_error_euclid", "mean"),
            mean_optimizer_iterations=("optimizer_n_iter", "mean"),
            mean_fit_s=("final_fit_s", "mean"),
        )
    )
    by_scenario.to_csv(output_root / "nuisance_summary.csv", index=False, float_format="%.10f")
    by_signal = (
        full.groupby(["signal_regime", "method"], as_index=False)
        .agg(
            n_cases=("dataset_id", "size"),
            large_common_regret_rate=("large_common_regret_flag", "mean"),
            mean_common_regret=("common_regret_vs_case_best", "mean"),
            mean_parameter_error=("final_standardized_rmse_7param", "mean"),
            mean_final_advection_error=("final_seed_error_euclid", "mean"),
        )
    )
    by_signal.to_csv(output_root / "signal_summary.csv", index=False, float_format="%.10f")
    paired_bootstrap(full, int(args.bootstrap_replicates), int(args.seed)).to_csv(
        output_root / "paired_comparisons.csv", index=False, float_format="%.10f"
    )
    start_detail, start_summary = start_sensitivity(full)
    start_detail.to_csv(
        output_root / "nuisance_start_sensitivity_by_dataset.csv",
        index=False,
        float_format="%.10f",
    )
    start_summary.to_csv(
        output_root / "nuisance_start_sensitivity_summary.csv",
        index=False,
        float_format="%.10f",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            "/Users/joonwonlee/Documents/GEMS_DATA/simulation/"
            "july_st_circulant_realpattern_smooth0p5"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO / "outputs/summer_26/synthetic_m3_q_jkq_fullfit_confirmatory_090126",
    )
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--angles-deg", default="22.5,112.5,202.5,292.5")
    parser.add_argument("--speeds-cells", default="0.75,2.0,4.0")
    parser.add_argument("--signal-regimes", default="strong,reference,weak")
    parser.add_argument("--nuisance-scenarios", default=",".join(NUISANCE_SCENARIOS))
    parser.add_argument("--lat-range", default="-3,-1")
    parser.add_argument("--lon-range", default="121,125")
    parser.add_argument("--n-lat", type=int, default=32)
    parser.add_argument("--n-lon", type=int, default=48)
    parser.add_argument("--seed", type=int, default=901261)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smooth", type=float, default=0.5)
    parser.add_argument("--daily-stride", type=int, default=2)
    parser.add_argument("--target-chunk-size", type=int, default=128)
    parser.add_argument("--empirical-max-lat-offset", type=int, default=20)
    parser.add_argument("--empirical-max-lon-offset", type=int, default=20)
    parser.add_argument("--empirical-min-pair-count", type=int, default=1000)
    parser.add_argument("--empirical-smooth-bandwidth-deg", type=float, default=0.063)
    parser.add_argument("--subgrid-max-condition-number", type=float, default=100.0)
    parser.add_argument("--jkq-threshold-cells", type=float, default=0.5)
    parser.add_argument("--lbfgs-lr", type=float, default=1.0)
    parser.add_argument("--lbfgs-history", type=int, default=10)
    parser.add_argument("--final-max-eval", type=int, default=20)
    parser.add_argument("--reference-max-eval", type=int, default=25)
    parser.add_argument("--grad-tol", type=float, default=1e-5)
    parser.add_argument("--convergence-grad-threshold", type=float, default=1e-3)
    parser.add_argument("--large-regret-threshold", type=float, default=0.002)
    parser.add_argument("--large-parameter-error-threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip dataset/nuisance/method rows already present in full_fit_results.csv.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    truth_path = (
        Path(args.data_root)
        / f"{args.year}_july_st_circulant"
        / f"sim_july{args.year}_st_circulant_truth.json"
    )
    input_path = (
        Path(args.data_root)
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
    coords = np.asarray(templates[0].grid_coords)
    lat_step = float(np.median(np.diff(np.sort(np.unique(np.round(coords[:, 0], 6))))))
    lon_step = float(np.median(np.diff(np.sort(np.unique(np.round(coords[:, 1], 6))))))
    speeds = parse_float_list(args.speeds_cells)
    angles = parse_angle_list(args.angles_deg)
    conditions = condition_grid(lat_step, lon_step, speeds, angles)
    signal_regimes = [token.strip() for token in str(args.signal_regimes).split(",") if token.strip()]
    nuisance_scenarios = [
        token.strip() for token in str(args.nuisance_scenarios).split(",") if token.strip()
    ]
    if unknown := sorted(set(signal_regimes) - set(SIGNAL_REGIMES)):
        raise ValueError(f"Unknown signal regimes: {unknown}")
    if unknown := sorted(set(nuisance_scenarios) - set(NUISANCE_SCENARIOS)):
        raise ValueError(f"Unknown nuisance scenarios: {unknown}")

    config = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "input_template": str(input_path),
        "base_truth": base_truth,
        "methods": METHODS,
        "conditions": conditions,
        "signal_regimes": {key: SIGNAL_REGIMES[key] for key in signal_regimes},
        "nuisance_scenarios": nuisance_scenarios,
        "common_objective": "oracle-direction fixed corridor",
        "flag_definitions": {
            "numerical_failure": "exception, nonfinite objective/gradient/parameters",
            "gradient_nonconvergence": (
                f"finite fit with final raw-gradient infinity norm > "
                f"{args.convergence_grad_threshold}"
            ),
            "large_common_regret_flag": (
                f"common objective regret > {args.large_regret_threshold}; operational threshold, "
                "not an optimizer failure or statistical test"
            ),
            "large_parameter_error_flag": (
                f"standardized seven-parameter RMSE > {args.large_parameter_error_threshold}; "
                "operational threshold"
            ),
        },
        "arguments": vars(args),
        "device": str(device),
    }
    (output_root / "run_config.json").write_text(
        json.dumps(clean_json(config), indent=2, sort_keys=True), encoding="utf-8"
    )

    initializer_rows: list[dict[str, Any]] = []
    simulation_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    full_rows: list[dict[str, Any]] = []
    completed: set[tuple[str, str, str]] = set()
    full_path = output_root / "full_fit_results.csv"
    if bool(args.resume) and full_path.exists():
        existing = pd.read_csv(full_path)
        full_rows = existing.to_dict("records")
        completed = set(
            zip(existing.dataset_id.astype(str), existing.nuisance_scenario.astype(str), existing.method.astype(str))
        )

    total_datasets = len(conditions) * len(signal_regimes) * int(args.replicates)
    dataset_counter = 0
    benchmark_started = time.perf_counter()
    print("methods:", METHODS, flush=True)
    print("conditions:", len(conditions), "signals:", signal_regimes, flush=True)
    print("nuisance starts:", nuisance_scenarios, "replicates:", args.replicates, flush=True)
    print("total datasets:", total_datasets, "planned fits:", total_datasets * len(nuisance_scenarios) * len(METHODS), flush=True)

    for condition_index, condition in enumerate(conditions):
        for signal_index, signal_regime in enumerate(signal_regimes):
            truth = dict(base_truth)
            truth["advec_lat"] = float(condition["advec_lat"])
            truth["advec_lon"] = float(condition["advec_lon"])
            truth["range_time"] *= float(SIGNAL_REGIMES[signal_regime]["range_time_factor"])
            truth["nugget"] *= float(SIGNAL_REGIMES[signal_regime]["nugget_factor"])
            for replicate in range(int(args.replicates)):
                dataset_counter += 1
                dataset_id = f"{condition['truth_id']}_{signal_regime}_r{replicate:02d}"
                sim_seed = int(
                    args.seed + condition_index * 100_000 + signal_index * 10_000 + replicate * 101
                )
                asset, sim_diag = simulate_asset(
                    templates[replicate], truth, sim_seed, args, dataset_id, replicate
                )
                sim_diag.update(
                    {
                        "dataset_id": dataset_id,
                        "signal_regime": signal_regime,
                        "speed_cells": condition["speed_cells"],
                        "direction_deg_grid": condition["direction_deg_grid"],
                    }
                )
                simulation_rows.append(sim_diag)
                method_rows, _ = run_three_initializers(asset, args)
                seeds = {row["method"]: row for row in method_rows}
                for row in method_rows:
                    initializer_rows.append(
                        initializer_record(
                            dataset_id, condition, signal_regime, replicate, row, truth
                        )
                    )
                pd.DataFrame(initializer_rows).to_csv(
                    output_root / "initializer_results.csv", index=False, float_format="%.10f"
                )
                pd.DataFrame(simulation_rows).to_csv(
                    output_root / "simulation_diagnostics.csv", index=False, float_format="%.10f"
                )

                print(
                    f"\n[{dataset_counter}/{total_datasets}] {dataset_id} "
                    f"truth=({truth['advec_lat']:+.4f},{truth['advec_lon']:+.4f})",
                    flush=True,
                )
                common_model = None
                common_mapped = None
                dataset_indices = []
                try:
                    common_model, common_mapped, common_precompute_s = build_corridor_model(
                        asset, truth["advec_lat"], truth["advec_lon"], args, device
                    )
                    reference = optimize_model(
                        common_model,
                        truth,
                        truth,
                        int(args.reference_max_eval),
                        args,
                        device,
                    )
                    reference_nll = float(reference["final_nll"])
                    reference_rows.append(
                        {
                            "dataset_id": dataset_id,
                            "truth_id": condition["truth_id"],
                            "direction_deg_grid": condition["direction_deg_grid"],
                            "speed_cells": condition["speed_cells"],
                            "signal_regime": signal_regime,
                            "replicate": replicate,
                            "status": "ok" if reference["finite_fit"] else "nonfinite",
                            "common_precompute_s": common_precompute_s,
                            "reference_nll": reference_nll,
                            "reference_fit_s": reference["fit_s"],
                            "reference_n_iter": reference["optimizer_n_iter"],
                            "reference_func_evals": reference["optimizer_func_evals"],
                            "reference_grad_inf": reference["final_grad_inf"],
                        }
                    )
                    pd.DataFrame(reference_rows).to_csv(
                        output_root / "reference_results.csv", index=False, float_format="%.10f"
                    )

                    for nuisance_index, nuisance_scenario in enumerate(nuisance_scenarios):
                        nuisance = nuisance_start(truth, nuisance_scenario)
                        rotation = (condition_index + signal_index + replicate + nuisance_index) % len(METHODS)
                        method_order = METHODS[rotation:] + METHODS[:rotation]
                        for method in method_order:
                            key = (dataset_id, nuisance_scenario, method)
                            if key in completed:
                                matching = [
                                    idx
                                    for idx, row in enumerate(full_rows)
                                    if str(row.get("dataset_id")) == dataset_id
                                    and str(row.get("nuisance_scenario")) == nuisance_scenario
                                    and str(row.get("method")) == method
                                ]
                                dataset_indices.extend(matching)
                                continue
                            base = {
                                "dataset_id": dataset_id,
                                "truth_id": condition["truth_id"],
                                "direction_deg_grid": condition["direction_deg_grid"],
                                "speed_cells": condition["speed_cells"],
                                "signal_regime": signal_regime,
                                "replicate": replicate,
                                "simulation_seed": sim_seed,
                                "nuisance_scenario": nuisance_scenario,
                                "method": method,
                                **{f"true_{parameter}": truth[parameter] for parameter in P_LABELS},
                            }
                            attempt_started = time.perf_counter()
                            try:
                                result = fit_method_corridor(
                                    asset,
                                    method,
                                    seeds[method],
                                    nuisance,
                                    truth,
                                    common_model,
                                    args,
                                    device,
                                )
                                row = {**base, **result}
                                print(
                                    f"  {nuisance_scenario:24s} {method:11s} "
                                    f"common={row['common_eval_nll']:.6f} "
                                    f"param={row['final_standardized_rmse_7param']:.3f} "
                                    f"iter={row['optimizer_n_iter']:2d} "
                                    f"time={row['end_to_end_s']:.2f}s",
                                    flush=True,
                                )
                            except Exception as exc:
                                row = {
                                    **base,
                                    "status": "error",
                                    "error": f"{type(exc).__name__}: {exc}",
                                    "traceback": traceback.format_exc(limit=8),
                                    "attempt_s": float(time.perf_counter() - attempt_started),
                                    "finite_fit": False,
                                    "converged_grad": False,
                                }
                                print("  ERROR", nuisance_scenario, method, row["error"], flush=True)
                            full_rows.append(clean_json(row))
                            dataset_indices.append(len(full_rows) - 1)
                            pd.DataFrame(full_rows).to_csv(
                                full_path, index=False, float_format="%.10f"
                            )

                    add_case_regrets(
                        full_rows,
                        dataset_indices,
                        reference_nll,
                        float(args.large_regret_threshold),
                        float(args.large_parameter_error_threshold),
                    )
                    pd.DataFrame(full_rows).to_csv(
                        full_path, index=False, float_format="%.10f"
                    )
                finally:
                    del common_model, common_mapped, asset
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                elapsed = time.perf_counter() - benchmark_started
                print(f"  cumulative={elapsed/60.0:.2f} min", flush=True)

    write_summaries(output_root, full_rows, args)
    elapsed = float(time.perf_counter() - benchmark_started)
    (output_root / "total_runtime.json").write_text(
        json.dumps({"wall_s": elapsed, "wall_minutes": elapsed / 60.0}, indent=2),
        encoding="utf-8",
    )
    print(f"\nDONE in {elapsed/60.0:.2f} min", flush=True)
    print(pd.read_csv(output_root / "overall_summary.csv").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
