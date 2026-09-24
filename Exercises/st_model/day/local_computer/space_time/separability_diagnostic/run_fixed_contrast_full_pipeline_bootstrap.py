#!/usr/bin/env python3
"""Full-pipeline parametric bootstrap for the frozen GEMS contrast.

Each replicate is generated from a block-Vecchia factorization on the exact
4/3/2 graph and observed missingness pattern, then the joint generalized-
Cauchy model and its GLS mean are re-fitted.  The contrast geometry, lag,
coefficient ratio, advection path, spatial scale, and nearest-grid rule remain
fixed at their pre-bootstrap 2024-07-01 values.

Two generating models are supported:

* ``joint``: the fitted joint space-time generalized-Cauchy covariance;
* ``separable``: the product of its matched spatial and temporal margins.

The fitted analysis model is the joint GC in both cases, matching the actual
pipeline that produces a joint fit and its matched separable comparator.
"""

from __future__ import annotations

import argparse
import heapq
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from GEMS_TCO.vecchia.corridor_neighbors.generalized_cauchy import (
    NoNuggetGeneralizedCauchyLag432CorridorVecchia,
)

from apply_fixed_canonical_contrast_real_gems import (
    DEFAULT_DATA_FILE,
    DEFAULT_FIT_CSV,
    DEFAULT_ORACLE_DIR,
    _atomic_csv,
    _atomic_text,
    _canonical_samples,
    _load_filtered_frames,
    _summaries,
    _uniform_axis,
)


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = DEFAULT_ORACLE_DIR / "full_pipeline_bootstrap_20240701"


@dataclass(frozen=True)
class ConditionalNode:
    conditioning_indices: np.ndarray
    target_indices: np.ndarray
    regression: np.ndarray
    conditional_cholesky: np.ndarray


@dataclass(frozen=True)
class VecchiaSimulationPlan:
    nodes: tuple[ConditionalNode, ...]
    topological_order: np.ndarray
    vector_length: int
    real_length: int
    valid_target_indices: np.ndarray
    dgp: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--fit-csv", type=Path, default=DEFAULT_FIT_CSV)
    parser.add_argument("--oracle-dir", type=Path, default=DEFAULT_ORACLE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--strategy", default="standard_432")
    parser.add_argument("--day-index", type=int, default=0)
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--dgp", choices=("joint", "separable", "both"), default="both")
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--target-chunk-size", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--max-eval", type=int, default=20)
    parser.add_argument("--history-size", type=int, default=10)
    parser.add_argument("--grad-tol", type=float, default=1.0e-4)
    parser.add_argument("--tolerance-grad", type=float, default=1.0e-5)
    parser.add_argument("--latitude-min", type=float, default=-3.0)
    parser.add_argument("--latitude-max", type=float, default=2.0)
    parser.add_argument("--longitude-min", type=float, default=121.0)
    parser.add_argument("--longitude-max", type=float, default=131.0)
    return parser


def _device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is unavailable")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _raw_parameters(fit: pd.Series) -> np.ndarray:
    range_lon = float(fit["est_range_lon"])
    phi2 = 1.0 / range_lon
    return np.asarray(
        [
            np.log(float(fit["est_sigmasq"]) * phi2),
            np.log(phi2),
            np.log((range_lon / float(fit["est_range_lat"])) ** 2),
            np.log((range_lon / float(fit["est_range_time"])) ** 2),
            float(fit["est_advec_lat"]),
            float(fit["est_advec_lon"]),
        ],
        dtype=np.float64,
    )


def _source_map(
    frames: list[pd.DataFrame], monthly_mean: float, device: torch.device
) -> tuple[dict[str, torch.Tensor], np.ndarray, np.ndarray, np.ndarray, float, float]:
    reference = frames[0]
    latitudes, latitude_step = _uniform_axis(reference["Latitude"], "latitude")
    longitudes, longitude_step = _uniform_axis(reference["Longitude"], "longitude")
    grid_coordinates = reference[["Latitude", "Longitude"]].to_numpy(dtype=np.float64)
    result: dict[str, torch.Tensor] = {}
    for time_index, frame in enumerate(frames):
        coordinates = frame[["Latitude", "Longitude"]].to_numpy(dtype=np.float64)
        if coordinates.shape != grid_coordinates.shape or not np.allclose(
            coordinates, grid_coordinates, rtol=0.0, atol=1.0e-10
        ):
            raise ValueError("regular-grid coordinates or order changed within the day")
        rows = np.zeros((len(frame), 11), dtype=np.float64)
        rows[:, 0] = pd.to_numeric(
            frame["Source_Latitude"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        rows[:, 1] = pd.to_numeric(
            frame["Source_Longitude"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        rows[:, 2] = (
            pd.to_numeric(frame["ColumnAmountO3"], errors="coerce").to_numpy(
                dtype=np.float64
            )
            - float(monthly_mean)
        )
        rows[:, 3] = float(time_index)
        if time_index > 0:
            rows[:, 3 + time_index] = 1.0
        result[f"t{time_index}"] = torch.as_tensor(
            rows, dtype=torch.float64, device=device
        ).contiguous()
    return (
        result,
        grid_coordinates,
        latitudes,
        longitudes,
        latitude_step,
        longitude_step,
    )


def _new_model(
    source_map: dict[str, torch.Tensor],
    grid_coordinates: np.ndarray,
    fit: pd.Series,
    target_chunk_size: int,
) -> NoNuggetGeneralizedCauchyLag432CorridorVecchia:
    return NoNuggetGeneralizedCauchyLag432CorridorVecchia(
        gc_alpha=float(fit["gc_alpha"]),
        gc_beta=float(fit["gc_beta"]),
        input_map=source_map,
        grid_coords=grid_coordinates,
        reference_advec_lon_abs=float(fit["lag1_lon_offset"]),
        target_chunk_size=int(target_chunk_size),
        min_target_points=1,
    )


def _separable_covariance(
    model: NoNuggetGeneralizedCauchyLag432CorridorVecchia,
    parameters: torch.Tensor,
    coordinates: torch.Tensor,
    is_dummy: torch.Tensor,
) -> torch.Tensor:
    phi1, phi2, phi3, phi4 = torch.exp(parameters[:4]).unbind()
    transformed_latitude = coordinates[..., 0] - parameters[4] * coordinates[..., 2]
    transformed_longitude = coordinates[..., 1] - parameters[5] * coordinates[..., 2]
    delta_latitude = transformed_latitude.unsqueeze(2) - transformed_latitude.unsqueeze(1)
    delta_longitude = transformed_longitude.unsqueeze(2) - transformed_longitude.unsqueeze(1)
    delta_time = coordinates[..., 2].unsqueeze(2) - coordinates[..., 2].unsqueeze(1)
    spatial_scaled = phi2 * torch.sqrt(
        (phi3 * delta_latitude.square() + delta_longitude.square()).clamp_min(0.0)
    )
    temporal_scaled = phi2 * torch.sqrt(phi4) * torch.abs(delta_time)
    covariance = (phi1 / phi2) * model._correlation(spatial_scaled) * model._correlation(
        temporal_scaled
    )
    covariance.diagonal(dim1=-2, dim2=-1).add_(1.0e-6)
    return model._decouple_dummy_covariance(covariance, is_dummy)


def _conditional_factors(covariance: torch.Tensor, conditioning_count: int):
    target_count = covariance.shape[-1] - conditioning_count
    target_target = covariance[
        :, conditioning_count : conditioning_count + target_count,
        conditioning_count : conditioning_count + target_count,
    ]
    if conditioning_count == 0:
        regression = covariance.new_zeros((covariance.shape[0], target_count, 0))
        conditional = target_target
    else:
        conditioning = covariance[:, :conditioning_count, :conditioning_count]
        conditioning_target = covariance[
            :, :conditioning_count, conditioning_count : conditioning_count + target_count
        ]
        factor = torch.linalg.cholesky(conditioning)
        solved = torch.cholesky_solve(conditioning_target, factor)
        regression = solved.transpose(1, 2)
        conditional = target_target - conditioning_target.transpose(1, 2) @ solved
    conditional = 0.5 * (conditional + conditional.transpose(1, 2))
    try:
        conditional_factor = torch.linalg.cholesky(conditional)
    except torch.linalg.LinAlgError:
        scale = conditional.diagonal(dim1=-2, dim2=-1).mean(dim=1)
        eye = torch.eye(
            target_count, dtype=conditional.dtype, device=conditional.device
        ).unsqueeze(0)
        conditional_factor = torch.linalg.cholesky(
            conditional + eye * (1.0e-10 * scale.clamp_min(1.0))[:, None, None]
        )
    return regression, conditional_factor


def _topological_order(nodes: list[ConditionalNode], real_length: int) -> np.ndarray:
    owner: dict[int, int] = {}
    for node_index, node in enumerate(nodes):
        for target in node.target_indices:
            target_integer = int(target)
            if target_integer in owner:
                raise ValueError("a valid observation is targeted by more than one block")
            owner[target_integer] = node_index
    adjacency: list[list[int]] = [[] for _ in nodes]
    indegree = np.zeros(len(nodes), dtype=np.int64)
    for node_index, node in enumerate(nodes):
        dependencies = {
            owner[int(index)]
            for index in node.conditioning_indices
            if int(index) < real_length and int(index) in owner
        }
        dependencies.discard(node_index)
        indegree[node_index] = len(dependencies)
        for dependency in dependencies:
            adjacency[dependency].append(node_index)
    queue = [int(index) for index in np.flatnonzero(indegree == 0)]
    heapq.heapify(queue)
    order: list[int] = []
    while queue:
        current = heapq.heappop(queue)
        order.append(current)
        for child in adjacency[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(queue, child)
    if len(order) != len(nodes):
        raise ValueError("Vecchia conditioning graph is not acyclic")
    if len(owner) != sum(len(node.target_indices) for node in nodes):
        raise AssertionError("target ownership count changed")
    return np.asarray(order, dtype=np.int64)


def _simulation_plan(
    model: NoNuggetGeneralizedCauchyLag432CorridorVecchia,
    parameters: torch.Tensor,
    dgp: str,
    real_length: int,
    chunk_size: int,
) -> VecchiaSimulationPlan:
    if not model.is_precomputed:
        raise RuntimeError("precompute the model before constructing a simulation plan")
    nodes: list[ConditionalNode] = []
    maximum_index = real_length - 1
    with torch.no_grad():
        for batch in model._cluster_batches:
            for start in range(0, batch.coordinates.shape[0], chunk_size):
                stop = min(start + chunk_size, batch.coordinates.shape[0])
                coordinates = batch.coordinates[start:stop]
                is_dummy = batch.is_dummy[start:stop]
                if dgp == "joint":
                    covariance = model._batched_covariance_with_dummy(
                        parameters, coordinates, is_dummy
                    )
                elif dgp == "separable":
                    covariance = _separable_covariance(
                        model, parameters, coordinates, is_dummy
                    )
                else:  # pragma: no cover - parser and caller constrain this.
                    raise ValueError(dgp)
                regression, conditional_factor = _conditional_factors(
                    covariance, batch.max_cond_points
                )
                indices = batch.indices[start:stop].detach().cpu().numpy()
                regression_np = regression.detach().cpu().numpy()
                factor_np = conditional_factor.detach().cpu().numpy()
                for row_index in range(indices.shape[0]):
                    conditioning_indices = indices[
                        row_index, : batch.max_cond_points
                    ].astype(np.int64, copy=True)
                    target_indices = indices[
                        row_index,
                        batch.max_cond_points : batch.max_cond_points + batch.target_size,
                    ].astype(np.int64, copy=True)
                    maximum_index = max(
                        maximum_index,
                        int(indices[row_index].max(initial=maximum_index)),
                    )
                    nodes.append(
                        ConditionalNode(
                            conditioning_indices=conditioning_indices,
                            target_indices=target_indices,
                            regression=regression_np[row_index].copy(),
                            conditional_cholesky=factor_np[row_index].copy(),
                        )
                    )
    target_indices = np.concatenate([node.target_indices for node in nodes])
    if np.unique(target_indices).size != target_indices.size:
        raise ValueError("simulation plan has duplicate target observations")
    order = _topological_order(nodes, real_length)
    return VecchiaSimulationPlan(
        nodes=tuple(nodes),
        topological_order=order,
        vector_length=maximum_index + 1,
        real_length=real_length,
        valid_target_indices=np.sort(target_indices),
        dgp=dgp,
    )


def _simulate(plan: VecchiaSimulationPlan, rng: np.random.Generator) -> np.ndarray:
    values = np.zeros(plan.vector_length, dtype=np.float64)
    assigned = np.zeros(plan.real_length, dtype=bool)
    for node_index in plan.topological_order:
        node = plan.nodes[int(node_index)]
        real_conditioning = node.conditioning_indices[node.conditioning_indices < plan.real_length]
        if real_conditioning.size and not assigned[real_conditioning].all():
            raise RuntimeError("topological simulation reached an unassigned conditioning value")
        conditional_mean = node.regression @ values[node.conditioning_indices]
        innovation = node.conditional_cholesky @ rng.standard_normal(
            len(node.target_indices)
        )
        values[node.target_indices] = conditional_mean + innovation
        assigned[node.target_indices] = True
    if not assigned[plan.valid_target_indices].all():
        raise RuntimeError("not every valid target was simulated")
    return values[: plan.real_length]


def _saved_mean_vector(
    source_map: dict[str, torch.Tensor], fit: pd.Series
) -> np.ndarray:
    beta = np.asarray([float(fit[f"beta_{index}"]) for index in range(9)])
    blocks = []
    for rows in source_map.values():
        array = rows.detach().cpu().numpy()
        design = np.column_stack(
            [
                np.ones(len(array)),
                array[:, 0] - float(fit["gls_lat_mean"]),
                array[:, 4:11],
            ]
        )
        blocks.append(design @ beta)
    return np.concatenate(blocks)


def _replicate_source_map(
    base_map: dict[str, torch.Tensor], simulated: np.ndarray, mean: np.ndarray
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    offset = 0
    for key, base in base_map.items():
        rows = base.clone()
        count = len(rows)
        original_valid = torch.isfinite(rows[:, 2])
        generated = torch.as_tensor(
            simulated[offset : offset + count] + mean[offset : offset + count],
            dtype=rows.dtype,
            device=rows.device,
        )
        rows[original_valid, 2] = generated[original_valid]
        rows[~original_valid, 2] = torch.nan
        result[key] = rows.contiguous()
        offset += count
    if offset != len(simulated):
        raise AssertionError("replicate response length differs from the base map")
    return result


def _fit_replicate(
    source_map: dict[str, torch.Tensor],
    grid_coordinates: np.ndarray,
    original_fit: pd.Series,
    initial_raw: np.ndarray,
    args: argparse.Namespace,
) -> tuple[Any, torch.Tensor, NoNuggetGeneralizedCauchyLag432CorridorVecchia, float]:
    model = _new_model(
        source_map, grid_coordinates, original_fit, args.target_chunk_size
    )
    started = time.perf_counter()
    model.precompute_conditioning_sets()
    parameters = [
        torch.tensor(
            value,
            dtype=torch.float64,
            device=model.device,
            requires_grad=True,
        )
        for value in initial_raw
    ]
    optimizer = model.make_lbfgs_optimizer(
        parameters,
        max_iter=int(args.max_eval),
        max_eval=int(args.max_eval),
        tolerance_grad=float(args.tolerance_grad),
        history_size=int(args.history_size),
    )
    fit_result = model.fit_lbfgs(
        parameters,
        optimizer,
        max_steps=int(args.max_steps),
        grad_tol=float(args.grad_tol),
    )
    parameter_tensor = torch.stack([parameter.reshape(()) for parameter in parameters]).detach()
    beta = model.estimate_gls_coefficients(parameter_tensor).detach()
    return fit_result, beta, model, time.perf_counter() - started


def _residual_cube(
    source_map: dict[str, torch.Tensor],
    beta: torch.Tensor,
    latitude_mean: float,
    grid_coordinates: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    latitude_step: float,
    longitude_step: float,
) -> dict[str, np.ndarray]:
    shape = (len(source_map), len(latitudes), len(longitudes))
    residual = np.full(shape, np.nan, dtype=np.float64)
    source_latitude = np.full(shape, np.nan, dtype=np.float64)
    source_longitude = np.full(shape, np.nan, dtype=np.float64)
    beta_np = beta.detach().cpu().numpy().reshape(-1)
    lat_index = np.rint((grid_coordinates[:, 0] - latitudes[0]) / latitude_step).astype(
        np.int64
    )
    lon_index = np.rint(
        (grid_coordinates[:, 1] - longitudes[0]) / longitude_step
    ).astype(np.int64)
    for time_index, rows in enumerate(source_map.values()):
        array = rows.detach().cpu().numpy()
        design = np.column_stack(
            [
                np.ones(len(array)),
                array[:, 0] - float(latitude_mean),
                array[:, 4:11],
            ]
        )
        values = array[:, 2] - design @ beta_np
        valid = np.isfinite(values) & np.isfinite(array[:, 0]) & np.isfinite(array[:, 1])
        residual[time_index, lat_index[valid], lon_index[valid]] = values[valid]
        source_latitude[time_index, lat_index[valid], lon_index[valid]] = array[valid, 0]
        source_longitude[time_index, lat_index[valid], lon_index[valid]] = array[valid, 1]
    return {
        "latitudes": latitudes,
        "longitudes": longitudes,
        "latitude_step": np.asarray(latitude_step),
        "longitude_step": np.asarray(longitude_step),
        "residual": residual,
        "source_latitude": source_latitude,
        "source_longitude": source_longitude,
    }


def _fit_series(
    fit_result: Any,
    original_fit: pd.Series,
    beta: torch.Tensor,
    latitude_mean: float,
) -> pd.Series:
    interpreted = fit_result.interpretable_parameters
    values: dict[str, Any] = {
        "est_sigmasq": interpreted["signal_variance"],
        "est_range_lat": interpreted["range_lat"],
        "est_range_lon": interpreted["range_lon"],
        "est_range_time": interpreted["range_time"],
        "est_advec_lat": interpreted["advec_lat"],
        "est_advec_lon": interpreted["advec_lon"],
        "est_nugget": interpreted["nugget"],
        "gc_alpha": float(original_fit["gc_alpha"]),
        "gc_beta": float(original_fit["gc_beta"]),
        "gls_lat_mean": float(latitude_mean),
    }
    for index, value in enumerate(beta.detach().cpu().numpy().reshape(-1)):
        values[f"beta_{index}"] = float(value)
    return pd.Series(values)


def _observed_summary(output_dir: Path) -> pd.Series:
    path = output_dir.parent / "real_gems_pilot_20240701/real_gems_fixed_canonical_summary.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"run apply_fixed_canonical_contrast_real_gems.py first; missing {path}"
        )
    table = pd.read_csv(path)
    selected = table.loc[table["group"] == "all"]
    if len(selected) != 1:
        raise ValueError("real-data pilot summary must have exactly one all row")
    return selected.iloc[0]


def _bootstrap_summary(results: pd.DataFrame, observed: pd.Series) -> pd.DataFrame:
    rows = []
    observed_statistic = float(observed["empirical_over_separable_pooled"])
    for dgp, group in results.groupby("dgp", sort=True):
        values = group["statistic_empirical_over_separable"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "dgp": dgp,
                "replicate_count": len(values),
                "observed_statistic": observed_statistic,
                "bootstrap_mean": float(values.mean()),
                "bootstrap_sd": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                "bootstrap_q025": float(np.quantile(values, 0.025)),
                "bootstrap_q50": float(np.quantile(values, 0.5)),
                "bootstrap_q975": float(np.quantile(values, 0.975)),
                "lower_tail_p_value": float(
                    (1 + np.sum(values <= observed_statistic)) / (len(values) + 1)
                ),
                "upper_tail_p_value": float(
                    (1 + np.sum(values >= observed_statistic)) / (len(values) + 1)
                ),
            }
        )
    return pd.DataFrame(rows)


def _write_figure(results: pd.DataFrame, observed: pd.Series, output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    dgps = sorted(results["dgp"].unique())
    data = [
        results.loc[results["dgp"] == dgp, "statistic_empirical_over_separable"]
        for dgp in dgps
    ]
    axis.boxplot(data, tick_labels=dgps, showmeans=True)
    axis.axhline(
        float(observed["empirical_over_separable_pooled"]),
        color="tab:red",
        linewidth=1.5,
        label="observed 2024-07-01",
    )
    axis.axhline(1.0, color="0.3", linestyle="--", linewidth=1.0, label="unit reference")
    axis.set_ylabel("empirical mean L squared / refitted matched-separable Var(L)")
    axis.set_title("Frozen-contrast full-pipeline parametric bootstrap")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "full_pipeline_bootstrap.png", dpi=220)
    figure.savefig(output_dir / "full_pipeline_bootstrap.pdf")
    plt.close(figure)


def _write_report(
    output_dir: Path,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    observed: pd.Series,
    manifest: dict[str, Any],
) -> None:
    rows = []
    for row in summary.itertuples(index=False):
        rows.append(
            f"| `{row.dgp}` | {row.replicate_count} | {row.bootstrap_mean:.6f} | "
            f"{row.bootstrap_q025:.6f} | {row.bootstrap_q975:.6f} | "
            f"{row.lower_tail_p_value:.6f} |"
        )
    minimum_recommended = 99
    provisional = len(results) < minimum_recommended or results.groupby("dgp").size().min() < 40
    lines = [
        "# Frozen-contrast full-pipeline parametric bootstrap",
        "",
        "The canonical A/B geometry, lag one, raw coefficient ratio, fitted spatial scale, fitted advection path, and nearest-grid mapping were frozen before this bootstrap. Every replicate uses the original GEMS missingness pattern, is simulated jointly through the fixed 4/3/2 block-Vecchia graph, and is analyzed after re-fitting the joint generalized-Cauchy covariance and GLS mean.",
        "",
        f"Observed statistic (empirical mean L^2 / refitted matched-separable Var(L)): `{float(observed['empirical_over_separable_pooled']):.8f}`.",
        "",
        "| generating model | replicates | mean | 2.5% | 97.5% | lower-tail p-value |",
        "|---|---:|---:|---:|---:|---:|",
        *rows,
        "",
    ]
    if provisional:
        lines.extend(
            [
                "**Status: computational pilot only.** The current replicate count is too small for confirmatory tail calibration. The driver is resumable; run it to at least 99, preferably 199 or more, replicates per generating model on CUDA before interpreting p-values.",
                "",
            ]
        )
    lines.extend(
        [
            "The `joint` generating distribution checks compatibility with the fitted joint GC. The `separable` distribution checks whether the observed lower standardized energy is unusual under the matched-separable construction. Both distributions repeat joint-GC refitting because that is the frozen analysis pipeline used to construct the comparator.",
            "",
            "This remains a parametric, model-based calibration. It does not replace external validation on held-out days.",
            "",
            "## Run configuration",
            "",
            "```json",
            json.dumps(manifest, indent=2),
            "```",
            "",
        ]
    )
    _atomic_text(output_dir / "REPORT.md", "\n".join(lines))


def main() -> None:
    args = build_parser().parse_args()
    if args.replicates < 1:
        raise ValueError("replicates must be positive")
    device = _device(args.device)
    data_file = args.data_file.expanduser().resolve()
    fit_csv = args.fit_csv.expanduser().resolve()
    oracle_dir = args.oracle_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    coefficient_source = oracle_dir / "global_two_rectangle_search/global_pair_ties.csv"

    fit_table = pd.read_csv(fit_csv)
    selected = fit_table.loc[
        (fit_table["strategy"] == args.strategy)
        & (fit_table["day_idx"].astype(int) == int(args.day_index))
    ]
    if len(selected) != 1:
        raise ValueError("saved fit selection must yield exactly one row")
    original_fit = selected.iloc[0]
    ties = pd.read_csv(coefficient_source, float_precision="round_trip")
    coefficient_first = float(ties.iloc[0]["raw_coefficient_first"])
    coefficient_second = float(ties.iloc[0]["raw_coefficient_second"])
    observed = _observed_summary(output_dir)

    frames, monthly_mean = _load_filtered_frames(
        data_file,
        (float(args.latitude_min), float(args.latitude_max)),
        (float(args.longitude_min), float(args.longitude_max)),
    )
    keys = sorted(frames)
    start = int(args.day_index) * 8
    day_keys = keys[start : start + 8]
    if len(day_keys) != 8:
        raise ValueError("the requested day does not contain eight frames")
    base_map, grid_coordinates, latitudes, longitudes, latitude_step, longitude_step = (
        _source_map([frames[key] for key in day_keys], monthly_mean, device)
    )
    real_length = sum(len(rows) for rows in base_map.values())
    raw = _raw_parameters(original_fit)
    raw_tensor = torch.as_tensor(raw, dtype=torch.float64, device=device)
    base_model = _new_model(
        base_map, grid_coordinates, original_fit, args.target_chunk_size
    )
    base_model.precompute_conditioning_sets()
    if not np.isclose(
        float(base_model.lat_mean_val),
        float(original_fit["gls_lat_mean"]),
        rtol=0.0,
        atol=1.0e-7,
    ):
        raise ValueError(
            "reconstructed source-map latitude mean differs from the saved fit: "
            f"{base_model.lat_mean_val:.17g} versus "
            f"{float(original_fit['gls_lat_mean']):.17g}"
        )
    saved_mean = _saved_mean_vector(base_map, original_fit)

    requested_dgps = ("joint", "separable") if args.dgp == "both" else (args.dgp,)
    existing_path = output_dir / "bootstrap_replicates.csv"
    existing = pd.read_csv(existing_path) if existing_path.is_file() else pd.DataFrame()
    result_rows = existing.to_dict("records") if not existing.empty else []

    for dgp_index, dgp in enumerate(requested_dgps):
        completed = (
            set(existing.loc[existing["dgp"] == dgp, "replicate"].astype(int))
            if not existing.empty and "dgp" in existing
            else set()
        )
        plan_started = time.perf_counter()
        plan = _simulation_plan(
            base_model,
            raw_tensor,
            dgp,
            real_length,
            int(args.target_chunk_size),
        )
        plan_seconds = time.perf_counter() - plan_started
        if len(plan.valid_target_indices) != int(original_fit["n_valid_o3"]):
            raise ValueError("simulation graph target count differs from the saved fit")
        for replicate in range(int(args.replicates)):
            if replicate in completed:
                continue
            seed = int(args.seed) + 1_000_000 * dgp_index + replicate
            rng = np.random.default_rng(seed)
            replicate_started = time.perf_counter()
            simulated = _simulate(plan, rng)
            replicate_map = _replicate_source_map(base_map, simulated, saved_mean)
            fit_result, beta, fitted_model, fit_seconds = _fit_replicate(
                replicate_map,
                grid_coordinates,
                original_fit,
                raw,
                args,
            )
            fitted_series = _fit_series(
                fit_result, original_fit, beta, fitted_model.lat_mean_val
            )
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
                cube,
                original_fit,
                coefficient_first,
                coefficient_second,
                covariance_fit=fitted_series,
            )
            total = _summaries(samples).loc[lambda frame: frame["group"] == "all"].iloc[0]
            interpreted = fit_result.interpretable_parameters
            row: dict[str, Any] = {
                "dgp": dgp,
                "replicate": replicate,
                "seed": seed,
                "sample_count": int(total["sample_count"]),
                "statistic_empirical_over_separable": float(
                    total["empirical_over_separable_pooled"]
                ),
                "statistic_empirical_over_joint": float(
                    total["empirical_over_joint_pooled"]
                ),
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
                "plan_seconds": plan_seconds,
                "fit_seconds": fit_seconds,
                "replicate_seconds": time.perf_counter() - replicate_started,
            }
            for index, value in enumerate(beta.detach().cpu().numpy().reshape(-1)):
                row[f"beta_{index}"] = float(value)
            result_rows.append(row)
            current = pd.DataFrame(result_rows).sort_values(
                ["dgp", "replicate"]
            ).reset_index(drop=True)
            _atomic_csv(existing_path, current)
            print(
                f"dgp={dgp} replicate={replicate} "
                f"Tsep={row['statistic_empirical_over_separable']:.6f} "
                f"fit_s={fit_seconds:.1f} total_s={row['replicate_seconds']:.1f}",
                flush=True,
            )
            del fitted_model, replicate_map

    results = pd.DataFrame(result_rows).sort_values(["dgp", "replicate"]).reset_index(
        drop=True
    )
    relevant = results.loc[results["dgp"].isin(requested_dgps)].copy()
    summary = _bootstrap_summary(relevant, observed)
    _atomic_csv(output_dir / "bootstrap_summary.csv", summary)
    _write_figure(relevant, observed, output_dir)
    manifest = {
        "data_file": str(data_file),
        "fit_csv": str(fit_csv),
        "coefficient_source": str(coefficient_source),
        "day": str(original_fit["day"]),
        "strategy": str(original_fit["strategy"]),
        "dgp": args.dgp,
        "requested_replicates_per_dgp": int(args.replicates),
        "seed": int(args.seed),
        "device": str(device),
        "max_steps": int(args.max_steps),
        "max_eval": int(args.max_eval),
        "grad_tol": float(args.grad_tol),
        "tolerance_grad": float(args.tolerance_grad),
        "frozen_geometry": {
            "range_lat": float(original_fit["est_range_lat"]),
            "range_lon": float(original_fit["est_range_lon"]),
            "advec_lat": float(original_fit["est_advec_lat"]),
            "advec_lon": float(original_fit["est_advec_lon"]),
            "temporal_lag": 1,
            "coefficient_first": coefficient_first,
            "coefficient_second": coefficient_second,
            "nearest_grid_rule": "axis-wise nearest regular cell within half a grid step",
        },
        "simulation": "exact block-conditional simulation from the fixed Vecchia 4/3/2 graph",
        "analysis_refit": "joint GC plus GLS mean for both generating models",
    }
    _atomic_text(output_dir / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    _write_report(output_dir, relevant, summary, observed, manifest)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
