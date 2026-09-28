#!/usr/bin/env python3
"""Fitting and cross-term calculations for the adapted lag-6/4/3 study."""

from __future__ import annotations

import argparse
import gc
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import Parameter


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[6]
ORACLE_DIR = HERE.parent / "sim_two_contrast_cross_center_092726"
ADAPTED_DIR = HERE.parent.parent / "vecchia_approximation"
for path in (PROJECT_ROOT / "src", ORACLE_DIR, ADAPTED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import two_contrast_core as oracle  # noqa: E402
import vecchia_adapted_fixed_lag643_core as adapted  # noqa: E402
from GEMS_TCO.vecchia.corridor_neighbors.directional_lag643 import (  # noqa: E402
    DirectionalLag643CorridorVecchia,
)
from GEMS_TCO.vecchia.corridor_neighbors.corridor_lag643 import (  # noqa: E402
    Lag643CorridorVecchia,
)
from GEMS_TCO.vecchia.corridor_neighbors.generalized_cauchy import (  # noqa: E402
    _STNoNuggetGeneralizedCauchyMixin,
)


DTYPE = torch.float64
POINT_SPECS = (
    (0, 0), (0, 1), (1, 0), (1, 1),
    (0, 2), (0, 3), (1, 2), (1, 3),
)


def temporal_lag(design: Mapping[str, Any]) -> int:
    """Return the validated temporal lag from the frozen contrast design."""

    lag = int(design["geometry"]["temporal_lag"])
    if lag <= 0:
        raise ValueError(f"temporal_lag must be positive, got {lag}")
    return lag


class NoNuggetDirectionalMatern05Lag643(DirectionalLag643CorridorVecchia):
    """Direction-adapted lag-643 joint Matérn-0.5 with nugget fixed at zero."""

    covariance_parameter_count = 6

    def _nugget_from_params(self, params: torch.Tensor) -> torch.Tensor:
        return params.new_tensor(0.0)


class NoNuggetDirectionalGeneralizedCauchyLag643(
    _STNoNuggetGeneralizedCauchyMixin,
    DirectionalLag643CorridorVecchia,
):
    """Direction-adapted lag-643 joint generalized Cauchy, nugget fixed zero."""

    def __init__(
        self,
        *,
        gc_alpha: float,
        gc_beta: float,
        input_map: dict[str, torch.Tensor],
        grid_coords: np.ndarray,
        reference_advec_lat: float,
        reference_advec_lon: float,
        target_chunk_size: int,
        min_target_points: int,
        covariance_backend: str = "native",
    ) -> None:
        # DirectionalLag643CorridorVecchia predates the public
        # covariance_backend argument.  Initialize its directional state here,
        # then call the maintained lag-643 parent directly so GC can require
        # the exact fused native backend without changing package-wide APIs.
        self.reference_advec_lat = float(reference_advec_lat)
        self.reference_advec_lon = float(reference_advec_lon)
        if not np.isfinite([self.reference_advec_lat, self.reference_advec_lon]).all():
            raise ValueError("reference advection components must be finite")
        self.reference_advec_norm = float(
            np.hypot(self.reference_advec_lat, self.reference_advec_lon)
        )
        self.past_step_vector = np.asarray(
            [-self.reference_advec_lat, -self.reference_advec_lon], dtype=np.float64
        )
        Lag643CorridorVecchia.__init__(
            self,
            smooth=0.5,
            input_map=input_map,
            grid_coords=grid_coords,
            reference_advec_lon_abs=max(self.reference_advec_norm, 1e-12),
            second_lag_stride=2,
            target_chunk_size=target_chunk_size,
            min_target_points=min_target_points,
            covariance_backend=covariance_backend,
        )
        self._init_st_cauchy(gc_alpha=gc_alpha, gc_beta=gc_beta)


def build_input_map(
    frames: Mapping[str, pd.DataFrame],
    manifest: pd.DataFrame,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], np.ndarray, float]:
    """Build the maintained Vecchia tensor layout without using truth parameters."""

    ordered = manifest.sort_values("local_time").reset_index(drop=True)
    keys = ordered["hour_key"].astype(str).tolist()
    values = [
        pd.to_numeric(frames[key]["ColumnAmountO3"], errors="coerce").to_numpy(float)
        for key in keys
    ]
    center = float(np.nanmean(np.concatenate(values)))
    first = frames[keys[0]]
    grid_coords = first[["Latitude", "Longitude"]].to_numpy(dtype=np.float64)
    input_map: dict[str, torch.Tensor] = {}
    for local_time, key in enumerate(keys):
        frame = frames[key]
        current_grid = frame[["Latitude", "Longitude"]].to_numpy(dtype=np.float64)
        if current_grid.shape != grid_coords.shape or not np.allclose(
            current_grid, grid_coords, rtol=0.0, atol=1e-10, equal_nan=True
        ):
            raise RuntimeError(f"regular-grid order differs at {key}")
        grid_lat = pd.to_numeric(frame["Latitude"], errors="coerce").to_numpy(float)
        grid_lon = pd.to_numeric(frame["Longitude"], errors="coerce").to_numpy(float)
        source_lat = pd.to_numeric(frame["Source_Latitude"], errors="coerce").to_numpy(float)
        source_lon = pd.to_numeric(frame["Source_Longitude"], errors="coerce").to_numpy(float)
        source_lat = np.where(np.isfinite(source_lat), source_lat, grid_lat)
        source_lon = np.where(np.isfinite(source_lon), source_lon, grid_lon)
        response = pd.to_numeric(frame["ColumnAmountO3"], errors="coerce").to_numpy(float) - center
        base = torch.from_numpy(
            np.column_stack(
                [source_lat, source_lon, response, np.full(len(frame), float(local_time))]
            )
        ).to(dtype=DTYPE)
        dummies = F.one_hot(torch.tensor([local_time]), num_classes=8).repeat(len(frame), 1)[:, 1:]
        input_map[key] = torch.cat([base, dummies.to(dtype=DTYPE)], dim=1).to(
            device=device, dtype=DTYPE
        ).contiguous()
    return input_map, grid_coords, center


def make_initializer(
    input_map_cpu: dict[str, torch.Tensor],
    grid_coords: np.ndarray,
    date: str,
    truth: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the data-only M3/Q3 advection initializer used by adapted corridors."""

    asset = adapted.DayAsset(
        dataset_id=str(date),
        data_kind="synthetic",
        year=int(str(date)[:4]),
        month=int(str(date)[5:7]),
        day=int(str(date)[8:10]),
        date=str(date),
        keys=sorted(input_map_cpu),
        source_map=input_map_cpu,
        grid_coords=grid_coords,
        center_value=0.0,
        n_valid=sum(int(torch.isfinite(value[:, 2]).sum()) for value in input_map_cpu.values()),
        n_total=sum(int(value.shape[0]) for value in input_map_cpu.values()),
        truth={
            "advec_lat": float(truth["advec_lat"]),
            "advec_lon": float(truth["advec_lon"]),
        },
        source_path="day_checkpoint",
    )
    settings = config["initializer"]
    args = argparse.Namespace(
        empirical_max_lat_offset=int(settings["maximum_latitude_offset_cells"]),
        empirical_max_lon_offset=int(settings["maximum_longitude_offset_cells"]),
        empirical_min_pair_count=int(settings["minimum_pair_count"]),
        empirical_smooth_bandwidth_deg=float(settings["smooth_bandwidth_degrees"]),
        subgrid_max_condition_number=float(settings["maximum_subgrid_condition_number"]),
    )
    return adapted.m3_q3_seed(asset, args)


def effective_to_kernel_ranges(
    family: str, config: Mapping[str, Any]
) -> dict[str, float]:
    fit = config["fit"]
    ranges = {
        "range_lat": float(fit["initial_effective_range_lat"]),
        "range_lon": float(fit["initial_effective_range_lon"]),
        "range_time": float(fit["initial_effective_range_time"]),
    }
    if family == "generalized_cauchy":
        alpha = float(fit["gc_alpha"])
        beta = float(fit["gc_beta"])
        scale = (math.exp(alpha / beta) - 1.0) ** (1.0 / alpha)
        ranges = {name: value / scale for name, value in ranges.items()}
    return ranges


def physical_to_raw(
    signal_variance: float,
    range_lat: float,
    range_lon: float,
    range_time: float,
    advec_lat: float,
    advec_lon: float,
) -> np.ndarray:
    phi2 = 1.0 / float(range_lon)
    return np.asarray(
        [
            math.log(float(signal_variance) * phi2),
            math.log(phi2),
            math.log((float(range_lon) / float(range_lat)) ** 2),
            math.log((float(range_lon) / float(range_time)) ** 2),
            float(advec_lat),
            float(advec_lon),
        ],
        dtype=np.float64,
    )


def build_model(
    family: str,
    input_map: dict[str, torch.Tensor],
    grid_coords: np.ndarray,
    initializer: Mapping[str, Any],
    config: Mapping[str, Any],
):
    fit = config["fit"]
    common = {
        "input_map": input_map,
        "grid_coords": grid_coords,
        "reference_advec_lat": float(initializer["seed_lat"]),
        "reference_advec_lon": float(initializer["seed_lon"]),
        "target_chunk_size": int(fit["target_chunk_size"][family]),
        "min_target_points": int(fit["min_target_points"]),
    }
    if family == "matern":
        return NoNuggetDirectionalMatern05Lag643(smooth=0.5, **common)
    if family == "generalized_cauchy":
        return NoNuggetDirectionalGeneralizedCauchyLag643(
            gc_alpha=float(fit["gc_alpha"]),
            gc_beta=float(fit["gc_beta"]),
            covariance_backend="native",
            **common,
        )
    raise ValueError(f"unsupported family {family!r}")


def fit_model(
    family: str,
    input_map: dict[str, torch.Tensor],
    grid_coords: np.ndarray,
    initializer: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], Any]:
    """Fit one matched-family joint model and return a JSON-safe record."""

    fit = config["fit"]
    model = build_model(family, input_map, grid_coords, initializer, config)
    started = time.perf_counter()
    model.precompute_conditioning_sets()
    precompute_seconds = time.perf_counter() - started
    ranges = effective_to_kernel_ranges(family, config)
    initial_raw = physical_to_raw(
        float(fit["initial_signal_variance"]),
        ranges["range_lat"],
        ranges["range_lon"],
        ranges["range_time"],
        float(initializer["seed_lat"]),
        float(initializer["seed_lon"]),
    )
    parameters = [
        Parameter(torch.tensor(value, dtype=DTYPE, device=model.device))
        for value in initial_raw
    ]
    optimizer = model.make_lbfgs_optimizer(
        parameters,
        lr=float(fit["lbfgs_lr"]),
        max_iter=int(fit["lbfgs_inner_max_iter"]),
        max_eval=int(fit["lbfgs_inner_max_iter"]),
        tolerance_grad=float(fit["lbfgs_tolerance_grad"]),
        tolerance_change=float(fit["lbfgs_tolerance_change"]),
        history_size=int(fit["lbfgs_history_size"]),
    )
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
        torch.cuda.reset_peak_memory_stats(model.device)
    fit_started = time.perf_counter()
    result = model.fit_lbfgs(
        parameters,
        optimizer,
        max_steps=int(fit["lbfgs_outer_max_steps"]),
        grad_tol=float(fit["outer_grad_tol"]),
    )
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    fit_seconds = time.perf_counter() - fit_started
    raw = torch.as_tensor(result.raw_parameters, dtype=DTYPE, device=model.device)
    with torch.no_grad():
        beta = model.estimate_gls_coefficients(raw).detach().cpu().reshape(-1).tolist()
    physical = model.interpretable_parameters(list(result.raw_parameters))
    record = {
        "status": "complete",
        "family": family,
        "fitted_model": "joint_matern05" if family == "matern" else "joint_generalized_cauchy_a1_b5",
        "geometry": "direction_adapted_corridor_width_4x4_lag643",
        "lag_pattern": "6/4/3",
        "target_chunk_size": int(model.target_chunk_size),
        "nugget_policy": "fixed_zero",
        "initializer": dict(initializer),
        "initial_raw_parameters": initial_raw.tolist(),
        "raw_parameters": list(result.raw_parameters),
        "interpretable_parameters": physical,
        "gls_coefficients": beta,
        "gls_latitude_center": float(model.lat_mean_val),
        "profiled_vecchia_nll_per_target": float(result.final_nll),
        "steps_completed": int(result.steps_completed),
        "converged_at_outer_grad_tol": bool(result.converged),
        "maximum_absolute_gradient": float(result.max_abs_gradient),
        "objective_evaluations": int(result.objective_evaluations),
        "cache_hits": int(result.cache_hits),
        "precompute_seconds": float(precompute_seconds),
        "fit_seconds": float(fit_seconds),
        "cluster_summary": model.cluster_summary(),
        "peak_cuda_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(model.device))
            if model.device.type == "cuda" else None
        ),
        "peak_cuda_memory_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(model.device))
            if model.device.type == "cuda" else None
        ),
    }
    if bool(fit["require_convergence"]) and not bool(result.converged):
        raise RuntimeError(
            f"fit did not reach outer grad tolerance; max gradient={result.max_abs_gradient:.6g}"
        )
    del parameters, optimizer
    gc.collect()
    return record, model


def reconstruct_source_coordinates(
    samples: Mapping[str, np.ndarray],
    cube: Mapping[str, Any],
    truth: Mapping[str, Any],
    design: Mapping[str, Any],
) -> np.ndarray:
    """Recover the exact observed source coordinates used by the fitted model."""

    result = np.full((len(samples["q_a"]), 8, 3), np.nan, dtype=np.float64)
    latitudes = np.asarray(cube["latitudes"], dtype=np.float64)
    longitudes = np.asarray(cube["longitudes"], dtype=np.float64)
    source_lat = np.asarray(cube["source_latitude"], dtype=np.float64)
    source_lon = np.asarray(cube["source_longitude"], dtype=np.float64)
    lag = temporal_lag(design)
    for handedness, standardized in design["geometry"]["standardized_geometry"].items():
        offsets = oracle._physical_offsets(standardized, truth)
        for time_t in range(int(truth["hours_per_day"]) - lag):
            mask = (samples["handedness"] == handedness) & (
                samples["time_t"].astype(np.int64) == time_t
            )
            index = np.flatnonzero(mask)
            if index.size == 0:
                continue
            center_lat = latitudes[samples["anchor_row"][index].astype(np.int64)]
            center_lon = longitudes[samples["anchor_column"][index].astype(np.int64)]
            for point_index, (time_side, endpoint) in enumerate(POINT_SPECS):
                slot = time_t if time_side == 0 else time_t + lag
                desired_lat = center_lat + offsets[endpoint, 0] + float(truth["advec_lat"]) * slot
                desired_lon = center_lon + offsets[endpoint, 1] + float(truth["advec_lon"]) * slot
                rr, _, lat_valid = oracle._nearest_axis_indices(
                    desired_lat, latitudes, float(cube["latitude_step"])
                )
                cc, _, lon_valid = oracle._nearest_axis_indices(
                    desired_lon, longitudes, float(cube["longitude_step"])
                )
                if not np.all(lat_valid & lon_valid):
                    raise RuntimeError("a retained contrast endpoint no longer maps inside the grid")
                result[index, point_index, 0] = source_lat[slot, rr, cc]
                result[index, point_index, 1] = source_lon[slot, rr, cc]
                result[index, point_index, 2] = float(slot)
    if not np.isfinite(result).all():
        raise RuntimeError("fitted-model contrast coordinates contain nonfinite values")
    return result


def fitted_contrast_covariances(
    coordinates: np.ndarray,
    physical: Mapping[str, Any],
    family: str,
    design: Mapping[str, Any],
    config: Mapping[str, Any],
) -> np.ndarray:
    coefficient = oracle.contrast_coefficients(design)
    output = np.empty((coordinates.shape[0], 2, 2), dtype=np.float64)
    chunk = int(config["diagnostic"]["model_covariance_chunk_size"])
    for start in range(0, len(coordinates), chunk):
        stop = min(start + chunk, len(coordinates))
        points = coordinates[start:stop]
        delta = points[:, :, None, :] - points[:, None, :, :]
        tau = delta[..., 2]
        shifted_lat = delta[..., 0] - float(physical["advec_lat"]) * tau
        shifted_lon = delta[..., 1] - float(physical["advec_lon"]) * tau
        distance = np.sqrt(
            np.square(shifted_lat / float(physical["range_lat"]))
            + np.square(shifted_lon / float(physical["range_lon"]))
            + np.square(tau / float(physical["range_time"]))
        )
        if family == "matern":
            correlation = np.exp(-distance)
        elif family == "generalized_cauchy":
            alpha = float(config["fit"]["gc_alpha"])
            beta = float(config["fit"]["gc_beta"])
            correlation = np.power(1.0 + np.power(distance, alpha), -beta / alpha)
        else:
            raise ValueError(family)
        covariance = float(physical["signal_variance"]) * correlation
        output[start:stop] = np.einsum(
            "ai,nij,bj->nab", coefficient, covariance, coefficient, optimize=True
        )
    return output


def summarize_fitted_cross(
    samples: Mapping[str, np.ndarray],
    fitted_covariance: np.ndarray,
    design: Mapping[str, Any],
) -> dict[str, Any]:
    q_a = np.asarray(samples["q_a"], dtype=np.float64)
    q_b = np.asarray(samples["q_b"], dtype=np.float64)
    fitted_mean = np.mean(fitted_covariance, axis=0)
    empirical_aa = float(np.mean(np.square(q_a)))
    empirical_bb = float(np.mean(np.square(q_b)))
    empirical_ab = float(np.mean(q_a * q_b))
    centered_a = q_a - np.mean(q_a)
    centered_b = q_b - np.mean(q_b)
    centered_aa = float(np.mean(np.square(centered_a)))
    centered_bb = float(np.mean(np.square(centered_b)))
    centered_ab = float(np.mean(centered_a * centered_b))
    d1 = float(design["geometry"]["secondary_l_coefficients"]["d1"])
    d2 = float(design["geometry"]["secondary_l_coefficients"]["d2"])
    fitted_aa = float(fitted_mean[0, 0])
    fitted_ab = float(fitted_mean[0, 1])
    fitted_bb = float(fitted_mean[1, 1])

    empirical_a = d1 * d1 * empirical_aa
    empirical_b = d2 * d2 * empirical_bb
    empirical_cross = 2.0 * d1 * d2 * empirical_ab
    empirical_total = empirical_a + empirical_b + empirical_cross
    fitted_a = d1 * d1 * fitted_aa
    fitted_b = d2 * d2 * fitted_bb
    fitted_cross = 2.0 * d1 * d2 * fitted_ab
    fitted_total = fitted_a + fitted_b + fitted_cross

    centered_a_component = d1 * d1 * centered_aa
    centered_b_component = d2 * d2 * centered_bb
    centered_cross = 2.0 * d1 * d2 * centered_ab
    centered_total = centered_a_component + centered_b_component + centered_cross
    total_residual = empirical_total - fitted_total
    a_residual = empirical_a - fitted_a
    b_residual = empirical_b - fitted_b
    cross_residual = empirical_cross - fitted_cross
    return {
        "sample_count": int(len(q_a)),
        "empirical_mean_q_a": float(np.mean(q_a)),
        "empirical_mean_q_b": float(np.mean(q_b)),
        "empirical_second_moment_q_a": empirical_aa,
        "empirical_second_moment_q_b": empirical_bb,
        "empirical_cross_moment_q_a_q_b": empirical_ab,
        "empirical_centered_variance_q_a": centered_aa,
        "empirical_centered_variance_q_b": centered_bb,
        "empirical_centered_covariance_q_a_q_b": centered_ab,
        "fitted_h_aa": fitted_aa,
        "fitted_h_ab": fitted_ab,
        "fitted_h_bb": fitted_bb,
        "empirical_minus_fitted_h_ab": empirical_ab - fitted_ab,
        "absolute_empirical_minus_fitted_h_ab": abs(empirical_ab - fitted_ab),
        # L is itself a zero-sum contrast, so E(L)=0 under the fitted mean
        # model.  These known-zero second moments are therefore the primary
        # empirical variance decomposition.  Centered versions are retained
        # as a finite-sample sensitivity analysis.
        "empirical_l_diagonal_a": empirical_a,
        "empirical_l_diagonal_b": empirical_b,
        "empirical_l_cross_term": empirical_cross,
        "empirical_l_variance": empirical_total,
        "fitted_l_diagonal_a": fitted_a,
        "fitted_l_diagonal_b": fitted_b,
        "fitted_l_cross_term": fitted_cross,
        "fitted_l_variance": fitted_total,
        "empirical_minus_fitted_l_diagonal_a": a_residual,
        "empirical_minus_fitted_l_diagonal_b": b_residual,
        "empirical_minus_fitted_l_cross_term": cross_residual,
        "empirical_minus_fitted_l_variance": total_residual,
        "variance_residual_decomposition_error": total_residual
        - (a_residual + b_residual + cross_residual),
        "empirical_centered_l_diagonal_a": centered_a_component,
        "empirical_centered_l_diagonal_b": centered_b_component,
        "empirical_centered_l_cross_term": centered_cross,
        "empirical_centered_l_variance": centered_total,
        "empirical_centered_minus_fitted_l_variance": centered_total - fitted_total,
        "empirical_centered_minus_fitted_l_cross_term": centered_cross - fitted_cross,
    }


def stratum_summaries(
    samples: Mapping[str, np.ndarray],
    fitted_covariance: np.ndarray,
    design: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for handedness in design["geometry"]["standardized_geometry"]:
        times = np.unique(samples["time_t"][samples["handedness"] == handedness]).astype(int)
        for time_t in times:
            selected = (samples["handedness"] == handedness) & (
                samples["time_t"].astype(int) == int(time_t)
            )
            subset = {name: np.asarray(value)[selected] for name, value in samples.items()}
            row = summarize_fitted_cross(subset, fitted_covariance[selected], design)
            row.update(
                handedness=str(handedness),
                time_t=int(time_t),
                time_t_plus_1=int(time_t + temporal_lag(design)),
            )
            rows.append(row)
    return rows


__all__ = [
    "NoNuggetDirectionalMatern05Lag643",
    "NoNuggetDirectionalGeneralizedCauchyLag643",
    "build_input_map",
    "make_initializer",
    "fit_model",
    "reconstruct_source_coordinates",
    "fitted_contrast_covariances",
    "summarize_fitted_cross",
    "stratum_summaries",
    "temporal_lag",
    "oracle",
]
