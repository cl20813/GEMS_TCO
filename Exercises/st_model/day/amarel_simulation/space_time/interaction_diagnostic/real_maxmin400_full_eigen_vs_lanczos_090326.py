#!/usr/bin/env python3
"""Validate full-eigen and Lanczos/SLQ residual diagnostics on 8 x 400 real data.

The default run reuses the fitted adapted lag-4/3/2, batch-64 parameters from
the local benchmark for 2024-07-03.  It does not refit the covariance model.
GLS beta is recomputed once at the stored parameter vector so that the same
raw residual is used by both spectral methods.

Exactly 400 spatial grid positions that are valid in all eight hourly fields
are selected by max-min order, giving n=3,200 observations.  The reference is
the exact dense fitted covariance K and its full eigendecomposition.  Lanczos
only receives a matvec callback.  Its residual-start quadrature approximates

    r' K^{-1} 1{K >= lambda_threshold} r,

while stochastic Lanczos quadrature (SLQ) with Rademacher probes approximates

    tr 1{K >= lambda_threshold}.

The default Lanczos backend wraps dense BLAS because the dense K must already
exist for the full-eigen reference.  A separate streamed-kernel matvec check
verifies that the same callback can be evaluated without storing K.  This
experiment isolates Lanczos/SLQ numerical error; a later large-n experiment
should replace the callback with the adapted-Vecchia sparse precision matvec.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.linalg
import torch


HERE = Path(__file__).resolve().parent
REPO = next(parent for parent in HERE.parents if (parent / "src/GEMS_TCO").is_dir())
SRC = REPO / "src"
VECCHIA_APPROX = HERE.parent / "vecchia_approximation"
for path in (SRC, VECCHIA_APPROX):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import vecchia_adapted_fixed_lag643_core as core  # noqa: E402
from GEMS_TCO import orderings  # noqa: E402
from GEMS_TCO.vecchia_realdata_corridor_width_4x4_lag432 import (  # noqa: E402
    DirectionalRealDataCorridorWidth4x4Lag432VecchiaFit,
)


DTYPE = torch.float64
PARAMETER_NAMES = (
    "sigmasq",
    "range_lat",
    "range_lon",
    "range_time",
    "advec_lat",
    "advec_lon",
    "nugget",
)


@dataclass
class LanczosRun:
    alpha: np.ndarray
    beta: np.ndarray
    start_norm_sq: float
    steps_completed: int
    matvec_s: float
    total_s: float
    breakdown: bool


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_ready(value), indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def parse_steps(values: list[int]) -> list[int]:
    out = sorted(set(int(value) for value in values))
    if not out or any(value < 2 for value in out):
        raise ValueError("Lanczos steps must be integers >= 2")
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="2024-07-03")
    parser.add_argument(
        "--real-data-root",
        type=Path,
        default=Path("/Users/joonwonlee/Documents/GEMS_DATA"),
    )
    parser.add_argument(
        "--fit-results",
        type=Path,
        default=HERE / "batch_benchmark_results_090326" / "batch_benchmark_results.json",
    )
    parser.add_argument(
        "--fit-initializer",
        type=Path,
        default=HERE / "batch_benchmark_results_090326" / "initializer.json",
    )
    parser.add_argument("--n-spatial", type=int, default=400)
    parser.add_argument("--hours", type=int, default=8, choices=[8])
    parser.add_argument("--smooth", type=float, default=0.5, choices=[0.5])
    parser.add_argument("--target-chunk-size", type=int, default=64, choices=[64])
    parser.add_argument(
        "--lanczos-steps", nargs="+", type=int, default=[32, 64, 128, 256, 512]
    )
    parser.add_argument("--slq-probes", type=int, default=32)
    parser.add_argument("--random-seed", type=int, default=20260903)
    parser.add_argument("--bands", type=int, default=20)
    parser.add_argument("--kernel-block-rows", type=int, default=256)
    parser.add_argument(
        "--reorthogonalization",
        choices=["none", "full"],
        default="full",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "real_maxmin400_full_eigen_vs_lanczos_20240703_090326",
    )
    return parser


def parse_date(value: str) -> tuple[int, int, int]:
    parsed = datetime.strptime(value, "%Y-%m-%d")
    return parsed.year, parsed.month, parsed.day


def loader_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        real_data_root=Path(args.real_data_root),
        lat_range="-3,2",
        lon_range="121,131",
        hours_per_day=int(args.hours),
        keep_exact_loc=True,
        empirical_max_lat_offset=20,
        empirical_max_lon_offset=20,
        empirical_min_pair_count=1000,
        empirical_smooth_bandwidth_deg=0.063,
        subgrid_max_condition_number=100.0,
    )


def load_fitted_record(path: Path) -> dict[str, Any]:
    records = json.loads(path.read_text(encoding="utf-8"))
    matches = [
        record
        for record in records
        if record.get("status") == "ok"
        and str(record.get("lag_pattern", "")).replace("/", "") == "432"
        and int(record.get("target_chunk_size", -1)) == 64
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one adapted 432/batch64 result in {path}, got {len(matches)}")
    record = dict(matches[0])
    if str(record.get("geometry")) != "adapted":
        raise RuntimeError(f"Stored fit is not adapted geometry: {record.get('geometry')}")
    return record


def fitted_physical(record: dict[str, Any]) -> dict[str, float]:
    return {name: float(record[f"est_{name}"]) for name in PARAMETER_NAMES}


def recompute_gls_beta(
    asset: core.DayAsset,
    fitted: dict[str, float],
    seed: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    mapped = {
        key: tensor.to(device="cpu", dtype=DTYPE).contiguous()
        for key, tensor in asset.source_map.items()
    }
    model = DirectionalRealDataCorridorWidth4x4Lag432VecchiaFit(
        smooth=float(args.smooth),
        input_map=mapped,
        grid_coords=asset.grid_coords,
        reference_advec_lat=float(seed["seed_lat"]),
        reference_advec_lon=float(seed["seed_lon"]),
        daily_stride=2,
        target_chunk_size=int(args.target_chunk_size),
        min_target_points=1,
    )
    precompute_started = time.perf_counter()
    model.precompute_conditioning_sets()
    precompute_s = time.perf_counter() - precompute_started
    raw = torch.as_tensor(core.physical_to_raw(fitted), dtype=DTYPE)
    beta_started = time.perf_counter()
    with torch.no_grad():
        beta = model.get_gls_beta(raw).detach().cpu().numpy().reshape(-1)
    beta_s = time.perf_counter() - beta_started
    lat_mean = float(model.lat_mean_val)
    summary = model.cluster_summary()
    del model, mapped
    gc.collect()
    return beta, lat_mean, {
        "vecchia_precompute_s": precompute_s,
        "gls_beta_s": beta_s,
        "vecchia_summary": summary,
    }


def zero_based_order(values: np.ndarray, n: int) -> np.ndarray:
    order = np.asarray(values, dtype=np.int64).reshape(-1)
    if order.size == n and order.min() == 1 and order.max() == n:
        order = order - 1
    if order.size != n or np.unique(order).size != n or order.min() < 0 or order.max() >= n:
        raise RuntimeError("Invalid max-min ordering returned by orderings.maxmin_cpp")
    return order


def select_common_maxmin(
    asset: core.DayAsset,
    n_spatial: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    arrays = [tensor.detach().cpu().numpy() for tensor in asset.source_map.values()]
    common = np.ones(arrays[0].shape[0], dtype=bool)
    for array in arrays:
        common &= np.all(np.isfinite(array), axis=1)
    eligible = np.flatnonzero(common)
    if eligible.size < int(n_spatial):
        raise RuntimeError(
            f"Only {eligible.size} spatial rows are valid in all hours; requested {n_spatial}"
        )
    grid = np.asarray(asset.grid_coords, dtype=np.float64)
    lon_lat = np.column_stack([grid[eligible, 1], grid[eligible, 0]])
    started = time.perf_counter()
    order = zero_based_order(orderings.maxmin_cpp(lon_lat), len(eligible))
    elapsed = time.perf_counter() - started
    selected = eligible[order[: int(n_spatial)]]
    return selected, order, {
        "n_grid": int(grid.shape[0]),
        "n_common_valid": int(eligible.size),
        "n_spatial_selected": int(selected.size),
        "maxmin_s": elapsed,
        "coordinate_order": "longitude,latitude",
    }


def subset_residual_data(
    asset: core.DayAsset,
    selected: np.ndarray,
    beta: np.ndarray,
    lat_mean: float,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    coords_parts: list[np.ndarray] = []
    residual_parts: list[np.ndarray] = []
    rows: list[pd.DataFrame] = []
    for time_index, (key, tensor) in enumerate(asset.source_map.items()):
        array = tensor.detach().cpu().numpy()[selected]
        design = np.column_stack(
            [
                np.ones(len(array), dtype=np.float64),
                array[:, 0] - float(lat_mean),
                array[:, 4:11],
            ]
        )
        residual = array[:, 2] - design @ beta
        coords = array[:, [0, 1, 3]].astype(np.float64)
        coords_parts.append(coords)
        residual_parts.append(residual)
        rows.append(
            pd.DataFrame(
                {
                    "row_index": np.arange(time_index * len(selected), (time_index + 1) * len(selected)),
                    "time_index": int(time_index),
                    "time_key": str(key),
                    "maxmin_rank": np.arange(1, len(selected) + 1),
                    "original_grid_index": selected,
                    "latitude": coords[:, 0],
                    "longitude": coords[:, 1],
                    "time_coordinate": coords[:, 2],
                    "centered_response": array[:, 2],
                    "fitted_mean": design @ beta,
                    "raw_residual": residual,
                }
            )
        )
    all_coords = np.vstack(coords_parts)
    all_residual = np.concatenate(residual_parts)
    if not np.all(np.isfinite(all_coords)) or not np.all(np.isfinite(all_residual)):
        raise RuntimeError("Non-finite values survived common-valid subset selection")
    return all_coords, all_residual, pd.concat(rows, ignore_index=True)


def covariance_block(
    left: np.ndarray,
    right: np.ndarray,
    params: dict[str, float],
) -> np.ndarray:
    dlat = left[:, None, 0] - right[None, :, 0]
    dlon = left[:, None, 1] - right[None, :, 1]
    dt = left[:, None, 2] - right[None, :, 2]
    shifted_lat = dlat - float(params["advec_lat"]) * dt
    shifted_lon = dlon - float(params["advec_lon"]) * dt
    distance = np.sqrt(
        (shifted_lat / float(params["range_lat"])) ** 2
        + (shifted_lon / float(params["range_lon"])) ** 2
        + (dt / float(params["range_time"])) ** 2
    )
    return float(params["sigmasq"]) * np.exp(-distance)


def build_dense_covariance(
    coords: np.ndarray,
    params: dict[str, float],
    block_rows: int,
) -> np.ndarray:
    n = len(coords)
    out = np.empty((n, n), dtype=np.float64)
    for start in range(0, n, int(block_rows)):
        end = min(start + int(block_rows), n)
        out[start:end] = covariance_block(coords[start:end], coords, params)
    out = 0.5 * (out + out.T)
    out.flat[:: n + 1] += float(params["nugget"])
    return out


def streamed_kernel_matmat(
    coords: np.ndarray,
    vectors: np.ndarray,
    params: dict[str, float],
    block_rows: int,
) -> np.ndarray:
    vectors_2d = np.asarray(vectors, dtype=np.float64)
    was_vector = vectors_2d.ndim == 1
    if was_vector:
        vectors_2d = vectors_2d[:, None]
    out = np.empty((len(coords), vectors_2d.shape[1]), dtype=np.float64)
    for start in range(0, len(coords), int(block_rows)):
        end = min(start + int(block_rows), len(coords))
        out[start:end] = covariance_block(coords[start:end], coords, params) @ vectors_2d
        out[start:end] += float(params["nugget"]) * vectors_2d[start:end]
    return out[:, 0] if was_vector else out


def lanczos_tridiagonal(
    matvec: Callable[[np.ndarray], np.ndarray],
    start: np.ndarray,
    max_steps: int,
    reorthogonalization: str,
    breakdown_tolerance: float = 1e-13,
) -> LanczosRun:
    vector = np.asarray(start, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("Lanczos start must have positive finite norm")
    n = len(vector)
    requested = min(int(max_steps), n)
    basis = np.empty((n, requested), dtype=np.float64)
    alpha = np.empty(requested, dtype=np.float64)
    beta = np.empty(max(requested - 1, 0), dtype=np.float64)
    q = vector / norm
    q_previous = np.zeros_like(q)
    beta_previous = 0.0
    matvec_s = 0.0
    breakdown = False
    total_started = time.perf_counter()
    completed = 0
    for step in range(requested):
        basis[:, step] = q
        mv_started = time.perf_counter()
        z = np.asarray(matvec(q), dtype=np.float64).reshape(-1)
        matvec_s += time.perf_counter() - mv_started
        if step:
            z -= beta_previous * q_previous
        a = float(q @ z)
        z -= a * q
        if reorthogonalization == "full":
            # Two passes make the Ritz spectrum reliable enough to assess hard
            # spectral projectors, whose discontinuity is numerically demanding.
            active = basis[:, : step + 1]
            for _ in range(2):
                z -= active @ (active.T @ z)
        b = float(np.linalg.norm(z))
        alpha[step] = a
        completed = step + 1
        if step == requested - 1:
            break
        beta[step] = b
        if not math.isfinite(b) or b <= breakdown_tolerance:
            breakdown = True
            break
        q_previous, q = q, z / b
        beta_previous = b
    total_s = time.perf_counter() - total_started
    return LanczosRun(
        alpha=alpha[:completed].copy(),
        beta=beta[: max(completed - 1, 0)].copy(),
        start_norm_sq=norm * norm,
        steps_completed=completed,
        matvec_s=matvec_s,
        total_s=total_s,
        breakdown=breakdown,
    )


def quadrature_atoms(run: LanczosRun, steps: int) -> tuple[np.ndarray, np.ndarray]:
    use = min(int(steps), int(run.steps_completed))
    theta, vectors = scipy.linalg.eigh_tridiagonal(
        run.alpha[:use],
        run.beta[: max(use - 1, 0)],
        check_finite=False,
    )
    weights = float(run.start_norm_sq) * vectors[0, :] ** 2
    positive = theta > max(np.finfo(np.float64).eps * float(np.max(np.abs(theta))), 0.0)
    return theta[positive], weights[positive]


def threshold_sum(
    atoms: np.ndarray,
    weights: np.ndarray,
    thresholds_descending: np.ndarray,
    inverse_atom: bool,
) -> np.ndarray:
    order = np.argsort(atoms)[::-1]
    atoms_desc = np.asarray(atoms[order], dtype=np.float64)
    terms = np.asarray(weights[order], dtype=np.float64)
    if inverse_atom:
        terms = terms / atoms_desc
    cumulative = np.cumsum(terms)
    count = np.searchsorted(-atoms_desc, -thresholds_descending, side="right")
    out = np.zeros_like(thresholds_descending, dtype=np.float64)
    positive = count > 0
    out[positive] = cumulative[count[positive] - 1]
    return out


def exact_rank_thresholds(eigenvalues_descending: np.ndarray) -> np.ndarray:
    values = np.asarray(eigenvalues_descending, dtype=np.float64)
    thresholds = np.empty_like(values)
    thresholds[:-1] = np.sqrt(values[:-1] * values[1:])
    # Zero is below the spectrum of an SPD covariance and guarantees that the
    # final hard projector is exactly I for both the reference and every SLQ
    # quadrature rule, despite tiny Ritz-bound roundoff at the lower edge.
    thresholds[-1] = 0.0
    return thresholds


def matrix_free_curves(
    residual_run: LanczosRun,
    probe_runs: list[LanczosRun],
    levels: list[int],
    thresholds: np.ndarray,
    n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    curve_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    exact_count = np.arange(1, n + 1, dtype=np.float64)
    for steps in levels:
        residual_atoms, residual_weights = quadrature_atoms(residual_run, steps)
        energy = threshold_sum(
            residual_atoms,
            residual_weights,
            thresholds,
            inverse_atom=True,
        )
        probe_counts = []
        for probe_run in probe_runs:
            atoms, weights = quadrature_atoms(probe_run, steps)
            probe_counts.append(
                threshold_sum(atoms, weights, thresholds, inverse_atom=False)
            )
        mode_count = np.mean(np.vstack(probe_counts), axis=0)
        mode_count_se = (
            np.std(np.vstack(probe_counts), axis=0, ddof=1) / math.sqrt(len(probe_counts))
            if len(probe_counts) > 1
            else np.full(n, np.nan)
        )
        frame = pd.DataFrame(
            {
                "lanczos_steps": int(steps),
                "rank_reference": np.arange(1, n + 1),
                "threshold": thresholds,
                "mode_count_exact": exact_count,
                "mode_count_slq": mode_count,
                "mode_count_slq_se": mode_count_se,
                "mode_fraction_exact": exact_count / n,
                "mode_fraction_slq": mode_count / n,
                "energy_lanczos": energy,
                "energy_lanczos_per_n": energy / n,
            }
        )
        curve_frames.append(frame)
        summaries.append(
            {
                "lanczos_steps": int(steps),
                "n_slq_probes": int(len(probe_runs)),
                "mode_fraction_rmse": float(
                    np.sqrt(np.mean(((mode_count - exact_count) / n) ** 2))
                ),
                "mode_fraction_max_abs_error": float(
                    np.max(np.abs(mode_count - exact_count)) / n
                ),
                "slq_endpoint_mode_count": float(mode_count[-1]),
                "lanczos_endpoint_energy": float(energy[-1]),
            }
        )
    return pd.concat(curve_frames, ignore_index=True), pd.DataFrame(summaries)


def add_exact_comparisons(
    curves: pd.DataFrame,
    summaries: pd.DataFrame,
    exact_energy: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(exact_energy)
    curves = curves.copy()
    repeated = np.tile(exact_energy, curves["lanczos_steps"].nunique())
    curves["energy_exact"] = repeated
    curves["energy_exact_per_n"] = repeated / n
    curves["energy_error"] = curves["energy_lanczos"] - curves["energy_exact"]
    curves["energy_error_per_n"] = curves["energy_error"] / n
    curves["mode_fraction_error"] = (
        curves["mode_fraction_slq"] - curves["mode_fraction_exact"]
    )
    exact_total = float(exact_energy[-1])
    exact_count = np.arange(1, n + 1, dtype=np.float64)
    exact_bridge = (exact_energy - exact_count / n * exact_total) / math.sqrt(2.0 * n)
    curves["shape_bridge_exact"] = np.tile(
        exact_bridge, curves["lanczos_steps"].nunique()
    )
    curves["shape_bridge_lanczos"] = (
        curves["energy_lanczos"]
        - curves["mode_count_slq"] / n
        * curves.groupby("lanczos_steps")["energy_lanczos"].transform("last")
    ) / math.sqrt(2.0 * n)
    summary_rows = []
    for row in summaries.to_dict(orient="records"):
        steps = int(row["lanczos_steps"])
        group = curves[curves["lanczos_steps"].eq(steps)]
        endpoint = float(group["energy_lanczos"].iloc[-1])
        row.update(
            {
                "energy_per_n_rmse": float(
                    np.sqrt(np.mean(group["energy_error_per_n"].to_numpy() ** 2))
                ),
                "energy_per_n_max_abs_error": float(
                    np.max(np.abs(group["energy_error_per_n"].to_numpy()))
                ),
                "endpoint_energy_abs_error": float(abs(endpoint - exact_total)),
                "endpoint_energy_relative_error": float(abs(endpoint / exact_total - 1.0)),
                "scale_stat_exact": float((exact_total - n) / math.sqrt(2.0 * n)),
                "scale_stat_lanczos": float((endpoint - n) / math.sqrt(2.0 * n)),
                "shape_D_exact": float(np.max(np.abs(exact_bridge))),
                "shape_D_lanczos_slq": float(
                    np.max(np.abs(group["shape_bridge_lanczos"].to_numpy()))
                ),
            }
        )
        summary_rows.append(row)
    return curves, pd.DataFrame(summary_rows)


def make_band_table(
    curves: pd.DataFrame,
    exact_energy: np.ndarray,
    n_bands: int,
) -> pd.DataFrame:
    n = len(exact_energy)
    edges = np.linspace(0, n, int(n_bands) + 1).round().astype(int)
    edges[0], edges[-1] = 0, n
    exact_padded = np.concatenate([[0.0], exact_energy])
    rows: list[dict[str, Any]] = []
    for steps, group in curves.groupby("lanczos_steps", sort=True):
        group = group.sort_values("rank_reference")
        approx_energy = np.concatenate([[0.0], group["energy_lanczos"].to_numpy()])
        approx_count = np.concatenate([[0.0], group["mode_count_slq"].to_numpy()])
        for band in range(len(edges) - 1):
            lo, hi = int(edges[band]), int(edges[band + 1])
            exact_modes = hi - lo
            exact_band_energy = float(exact_padded[hi] - exact_padded[lo])
            approx_modes = float(approx_count[hi] - approx_count[lo])
            approx_band_energy = float(approx_energy[hi] - approx_energy[lo])
            rows.append(
                {
                    "lanczos_steps": int(steps),
                    "band": int(band + 1),
                    "rank_start": int(lo + 1),
                    "rank_end": int(hi),
                    "eigen_fraction_mid": float((lo + hi) / (2.0 * n)),
                    "exact_mode_count": int(exact_modes),
                    "slq_mode_count": approx_modes,
                    "exact_energy": exact_band_energy,
                    "lanczos_energy": approx_band_energy,
                    "exact_energy_per_mode": exact_band_energy / exact_modes,
                    "matrix_free_energy_per_mode": (
                        approx_band_energy / approx_modes if approx_modes > 1e-12 else np.nan
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    frame["energy_per_mode_error"] = (
        frame["matrix_free_energy_per_mode"] - frame["exact_energy_per_mode"]
    )
    return frame


def plot_comparison(
    curves: pd.DataFrame,
    bands: pd.DataFrame,
    eigenvalues: np.ndarray,
    output_dir: Path,
) -> Path:
    levels = sorted(curves["lanczos_steps"].unique())
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(levels)))
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.5), constrained_layout=True)

    exact = curves[curves["lanczos_steps"].eq(levels[-1])]
    axes[0, 0].plot(
        exact["mode_fraction_exact"],
        exact["energy_exact_per_n"],
        color="black",
        lw=2.2,
        label="full eigen",
    )
    for color, steps in zip(colors, levels):
        group = curves[curves["lanczos_steps"].eq(steps)]
        axes[0, 0].plot(
            group["mode_fraction_slq"],
            group["energy_lanczos_per_n"],
            color=color,
            lw=1.25,
            alpha=0.9,
            label=f"Lanczos/SLQ m={steps}",
        )
    axes[0, 0].set_xlabel("mode fraction")
    axes[0, 0].set_ylabel("cumulative standardized energy / n")
    axes[0, 0].set_title("Combined matrix-free curve")
    axes[0, 0].legend(fontsize=8)

    for color, steps in zip(colors, levels):
        group = curves[curves["lanczos_steps"].eq(steps)]
        axes[0, 1].plot(
            group["mode_fraction_exact"],
            group["energy_error_per_n"],
            color=color,
            lw=1.2,
            label=f"m={steps}",
        )
    axes[0, 1].axhline(0.0, color="black", lw=0.8)
    axes[0, 1].set_xlabel("exact mode fraction")
    axes[0, 1].set_ylabel("Lanczos - full energy, divided by n")
    axes[0, 1].set_title("Residual quadrature error")

    for color, steps in zip(colors, levels):
        group = curves[curves["lanczos_steps"].eq(steps)]
        axes[1, 0].plot(
            group["mode_fraction_exact"],
            group["mode_fraction_error"],
            color=color,
            lw=1.2,
            label=f"m={steps}",
        )
    axes[1, 0].axhline(0.0, color="black", lw=0.8)
    axes[1, 0].set_xlabel("exact mode fraction")
    axes[1, 0].set_ylabel("SLQ - exact mode fraction")
    axes[1, 0].set_title("SLQ spectral-CDF error")

    max_steps = levels[-1]
    selected_bands = bands[bands["lanczos_steps"].eq(max_steps)]
    axes[1, 1].plot(
        selected_bands["band"],
        selected_bands["exact_energy_per_mode"],
        color="black",
        marker="o",
        lw=2.0,
        label="full eigen",
    )
    axes[1, 1].plot(
        selected_bands["band"],
        selected_bands["matrix_free_energy_per_mode"],
        color=colors[-1],
        marker="s",
        lw=1.5,
        label=f"Lanczos/SLQ m={max_steps}",
    )
    axes[1, 1].axhline(1.0, color="0.5", ls="--", lw=0.9)
    axes[1, 1].set_xlabel("equal-rank eigenvalue band (large to small eigenvalue)")
    axes[1, 1].set_ylabel("standardized energy per estimated mode")
    axes[1, 1].set_title("20-band local energy")
    axes[1, 1].legend(fontsize=8)

    for axis in axes.flat:
        axis.grid(alpha=0.2)
    fig.suptitle(
        f"Full eigen vs Lanczos/SLQ: n={len(eigenvalues):,}, "
        f"condition number={eigenvalues[0] / eigenvalues[-1]:.2e}",
        fontsize=14,
    )
    path = output_dir / "full_eigen_vs_lanczos_slq.png"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_spectrum_and_bridge(
    curves: pd.DataFrame,
    eigenvalues: np.ndarray,
    output_dir: Path,
) -> Path:
    levels = sorted(curves["lanczos_steps"].unique())
    colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(levels)))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), constrained_layout=True)
    fraction = np.arange(1, len(eigenvalues) + 1) / len(eigenvalues)
    axes[0].plot(fraction, eigenvalues, color="black", lw=1.5)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("eigenvalue rank fraction")
    axes[0].set_ylabel("fitted covariance eigenvalue")
    axes[0].set_title("Exact covariance spectrum")
    exact = curves[curves["lanczos_steps"].eq(levels[-1])]
    axes[1].plot(
        exact["mode_fraction_exact"],
        exact["shape_bridge_exact"],
        color="black",
        lw=2.0,
        label="full eigen",
    )
    for color, steps in zip(colors, levels):
        group = curves[curves["lanczos_steps"].eq(steps)]
        axes[1].plot(
            group["mode_fraction_slq"],
            group["shape_bridge_lanczos"],
            color=color,
            lw=1.1,
            label=f"m={steps}",
        )
    axes[1].axhline(0.0, color="0.4", lw=0.8)
    axes[1].set_xlabel("mode fraction")
    axes[1].set_ylabel("scale-removed bridge")
    axes[1].set_title("Spectral-shape statistic")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
    path = output_dir / "exact_spectrum_and_shape_bridge.png"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def write_results_note(
    output_dir: Path,
    summary: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    best = summary.sort_values("lanczos_steps").iloc[-1]
    text = f"""# Full eigen versus Lanczos/SLQ on a real 8 x 400 subset

## Scope

This run uses 400 max-min ordered spatial locations that are valid in all
eight hours of 2024-07-03 (`n=3,200`).  The covariance parameters are the
stored adapted lag-4/3/2, batch-64 fit.  No covariance refit was performed.
GLS beta was recomputed at the stored parameters, and both numerical methods
use exactly the same raw residual and exact fitted covariance.

The full eigendecomposition is the reference.  Lanczos approximates the
residual-weighted inverse-covariance spectral measure, while SLQ estimates the
spectral mode count.  The largest run uses `{int(best['lanczos_steps'])}`
Lanczos steps and `{int(best['n_slq_probes'])}` fixed-seed Rademacher probes.

## Largest-run agreement

- Spectral-CDF mode-fraction RMSE: `{best['mode_fraction_rmse']:.6g}`
- Spectral-CDF maximum absolute error: `{best['mode_fraction_max_abs_error']:.6g}`
- Cumulative energy/n RMSE: `{best['energy_per_n_rmse']:.6g}`
- Cumulative energy/n maximum absolute error: `{best['energy_per_n_max_abs_error']:.6g}`
- Endpoint energy relative error: `{best['endpoint_energy_relative_error']:.6g}`
- Exact versus matrix-free shape D: `{best['shape_D_exact']:.6g}` versus
  `{best['shape_D_lanczos_slq']:.6g}`

## Interpretation boundary

The Lanczos code accesses the covariance only through a matvec callback.  For
this validation run that callback uses the already materialized dense matrix,
because the dense matrix is required for the full-eigen reference.  A separate
streamed-kernel callback, which never stores the full covariance, agreed with
the dense callback to relative error
`{metadata['streamed_matvec_relative_error']:.3e}`.

Thus this run validates the numerical Lanczos/SLQ formulation on the exact
fitted covariance.  It is not yet the scalable 144,000-point implementation:
the next operator should be the sparse adapted-Vecchia precision
`v -> A.T @ D^-1 @ (A @ v)`.  That next comparison will additionally contain
Vecchia approximation error, whereas this one intentionally does not.

Hard spectral thresholds are discontinuous, so convergence is slowest near
eigenvalue clusters and at very small-rank tails.  Smooth overlapping filters
should be added after this exact-reference test is accepted.
"""
    (output_dir / "RESULTS.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    levels = parse_steps(args.lanczos_steps)
    if int(args.n_spatial) < 2 or int(args.slq_probes) < 1 or int(args.bands) < 2:
        raise ValueError("n-spatial>=2, slq-probes>=1, and bands>=2 are required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    workflow_started = time.perf_counter()

    year, month, day = parse_date(args.date)
    record = load_fitted_record(Path(args.fit_results))
    fitted = fitted_physical(record)
    seed = json.loads(Path(args.fit_initializer).read_text(encoding="utf-8"))
    spec = {
        "dataset_id": f"real_{year}{month:02d}{day:02d}",
        "year": year,
        "month": month,
        "day": day,
        "date": args.date,
    }
    print(f"Loading real data {args.date}", flush=True)
    load_started = time.perf_counter()
    asset = core.load_real_asset(spec, loader_args(args))
    load_s = time.perf_counter() - load_started
    if asset.date != str(record["date"]):
        raise RuntimeError(f"Fit date {record['date']} does not match data date {asset.date}")

    print("Recomputing GLS beta at stored adapted-432/batch64 parameters", flush=True)
    beta, lat_mean, gls_meta = recompute_gls_beta(asset, fitted, seed, args)
    print(
        f"Selecting {int(args.n_spatial)} common-valid spatial locations by max-min order",
        flush=True,
    )
    selected, _, selection_meta = select_common_maxmin(asset, int(args.n_spatial))
    coords, residual, subset = subset_residual_data(asset, selected, beta, lat_mean)
    n = len(residual)
    expected_n = int(args.hours) * int(args.n_spatial)
    if n != expected_n:
        raise RuntimeError(f"Expected n={expected_n}, got n={n}")
    atomic_csv(args.output_dir / "selected_subset_points.csv", subset)

    print(f"Building exact fitted covariance K ({n:,} x {n:,})", flush=True)
    covariance_started = time.perf_counter()
    covariance = build_dense_covariance(
        coords, fitted, block_rows=int(args.kernel_block_rows)
    )
    covariance_s = time.perf_counter() - covariance_started
    symmetry_error = float(np.max(np.abs(covariance - covariance.T)))

    rng = np.random.default_rng(int(args.random_seed))
    validation_vectors = rng.standard_normal((n, 3))
    dense_mv_started = time.perf_counter()
    dense_mv = covariance @ validation_vectors
    dense_mv_s = time.perf_counter() - dense_mv_started
    stream_mv_started = time.perf_counter()
    streamed_mv = streamed_kernel_matmat(
        coords,
        validation_vectors,
        fitted,
        block_rows=int(args.kernel_block_rows),
    )
    streamed_mv_s = time.perf_counter() - stream_mv_started
    streamed_relative_error = float(
        np.linalg.norm(streamed_mv - dense_mv) / np.linalg.norm(dense_mv)
    )
    if streamed_relative_error > 1e-11:
        raise RuntimeError(
            f"Streamed kernel matvec disagrees with dense K: {streamed_relative_error:.3e}"
        )

    print("Computing full eigendecomposition reference", flush=True)
    eigen_started = time.perf_counter()
    eigenvalues, eigenvectors = scipy.linalg.eigh(
        covariance,
        lower=True,
        check_finite=False,
        overwrite_a=False,
        driver="evd",
    )
    full_eigen_s = time.perf_counter() - eigen_started
    if eigenvalues[0] <= 0.0:
        raise RuntimeError(f"Fitted covariance is not positive definite: min={eigenvalues[0]}")
    eigenvalues = eigenvalues[::-1].copy()
    eigenvectors = eigenvectors[:, ::-1].copy()
    coefficients = eigenvectors.T @ residual
    exact_y2 = coefficients**2 / eigenvalues
    exact_energy = np.cumsum(exact_y2)
    thresholds = exact_rank_thresholds(eigenvalues)
    exact_solve_energy = float(residual @ scipy.linalg.solve(
        covariance, residual, assume_a="pos", check_finite=False
    ))
    endpoint_identity_relative_error = float(
        abs(exact_energy[-1] - exact_solve_energy) / max(abs(exact_solve_energy), 1e-15)
    )
    eigen_frame = pd.DataFrame(
        {
            "rank": np.arange(1, n + 1),
            "rank_fraction": np.arange(1, n + 1) / n,
            "eigenvalue": eigenvalues,
            "raw_projection": coefficients,
            "standardized_score": coefficients / np.sqrt(eigenvalues),
            "standardized_energy": exact_y2,
            "cumulative_standardized_energy": exact_energy,
            "cumulative_energy_per_n": exact_energy / n,
            "threshold_below_rank": thresholds,
        }
    )
    atomic_csv(args.output_dir / "full_eigen_reference.csv", eigen_frame)
    del eigenvectors, coefficients
    gc.collect()

    def dense_matvec(vector: np.ndarray) -> np.ndarray:
        return covariance @ vector

    max_steps = max(levels)
    print(f"Lanczos residual measure: m={max_steps}", flush=True)
    residual_run = lanczos_tridiagonal(
        dense_matvec,
        residual,
        max_steps=max_steps,
        reorthogonalization=str(args.reorthogonalization),
    )
    print(
        f"SLQ trace measure: probes={args.slq_probes}, m={max_steps}", flush=True
    )
    probe_runs: list[LanczosRun] = []
    slq_started = time.perf_counter()
    for probe_index in range(int(args.slq_probes)):
        probe = rng.choice(np.asarray([-1.0, 1.0]), size=n, replace=True)
        run = lanczos_tridiagonal(
            dense_matvec,
            probe,
            max_steps=max_steps,
            reorthogonalization=str(args.reorthogonalization),
        )
        probe_runs.append(run)
        print(
            f"  probe {probe_index + 1}/{args.slq_probes}: "
            f"steps={run.steps_completed}, time={run.total_s:.2f}s",
            flush=True,
        )
    slq_s = time.perf_counter() - slq_started

    curves, summaries = matrix_free_curves(
        residual_run, probe_runs, levels, thresholds, n
    )
    curves, summaries = add_exact_comparisons(curves, summaries, exact_energy)
    bands = make_band_table(curves, exact_energy, int(args.bands))
    atomic_csv(args.output_dir / "lanczos_slq_cumulative_comparison.csv", curves)
    atomic_csv(args.output_dir / "lanczos_slq_accuracy_summary.csv", summaries)
    atomic_csv(args.output_dir / "lanczos_slq_band_comparison.csv", bands)

    figure_paths = [
        plot_comparison(curves, bands, eigenvalues, args.output_dir),
        plot_spectrum_and_bridge(curves, eigenvalues, args.output_dir),
    ]
    metadata = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "date": args.date,
        "data_source": asset.source_path,
        "fit_results": str(Path(args.fit_results).resolve()),
        "fit_initializer": str(Path(args.fit_initializer).resolve()),
        "fit_reused_without_refit": True,
        "fit": {
            "geometry": "adapted",
            "lag_pattern": "4/3/2",
            "target_chunk_size": 64,
            "fitted_parameters": fitted,
            "fitted_native_nll": float(record["final_native_nll"]),
            "gls_beta": beta,
            "gls_lat_mean": lat_mean,
        },
        "subset": {
            **selection_meta,
            "hours": int(args.hours),
            "n_observations": n,
            "ordering": "time-major, then max-min spatial rank",
            "common_valid_required": True,
        },
        "full_covariance": {
            "shape": list(covariance.shape),
            "storage_bytes": int(covariance.nbytes),
            "symmetry_max_abs_error": symmetry_error,
            "min_eigenvalue": float(eigenvalues[-1]),
            "max_eigenvalue": float(eigenvalues[0]),
            "condition_number": float(eigenvalues[0] / eigenvalues[-1]),
            "exact_endpoint_energy": float(exact_energy[-1]),
            "solve_endpoint_energy": exact_solve_energy,
            "endpoint_identity_relative_error": endpoint_identity_relative_error,
        },
        "lanczos": {
            "operator": "fitted exact covariance K",
            "algorithm_interface": "matvec-only",
            "benchmark_backend": "dense BLAS reference",
            "streamed_kernel_matvec_validated": True,
            "streamed_matvec_relative_error": streamed_relative_error,
            "steps_reported": levels,
            "max_steps_computed": max_steps,
            "slq_probes": int(args.slq_probes),
            "probe_distribution": "Rademacher",
            "random_seed": int(args.random_seed),
            "reorthogonalization": str(args.reorthogonalization),
            "hard_projector": True,
        },
        "streamed_matvec_relative_error": streamed_relative_error,
        "timings_s": {
            "data_load": load_s,
            "vecchia_precompute": gls_meta["vecchia_precompute_s"],
            "gls_beta": gls_meta["gls_beta_s"],
            "maxmin": selection_meta["maxmin_s"],
            "dense_covariance_build": covariance_s,
            "dense_matmat_three_vectors": dense_mv_s,
            "streamed_matmat_three_vectors": streamed_mv_s,
            "full_eigendecomposition": full_eigen_s,
            "residual_lanczos": residual_run.total_s,
            "residual_lanczos_matvec_only": residual_run.matvec_s,
            "slq_all_probes": slq_s,
            "slq_all_probe_matvec_only": float(sum(run.matvec_s for run in probe_runs)),
            "workflow_total": time.perf_counter() - workflow_started,
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
        "figures": figure_paths,
    }
    metadata["gls_recomputation"] = gls_meta
    atomic_json(args.output_dir / "run_metadata.json", metadata)
    write_results_note(args.output_dir, summaries, metadata)
    print("\nAccuracy summary", flush=True)
    print(summaries.to_string(index=False), flush=True)
    print(f"Saved results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
