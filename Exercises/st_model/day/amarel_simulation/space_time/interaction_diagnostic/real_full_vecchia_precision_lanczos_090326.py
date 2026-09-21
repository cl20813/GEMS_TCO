#!/usr/bin/env python3
"""Matrix-free eigen diagnostic for the full 8-hour real-data Vecchia fit.

This is the scalable continuation of
``real_maxmin400_full_eigen_vs_lanczos_090326.py``.  It converts the fitted
adapted lag-4/3/2 block-Vecchia conditionals into a sparse whitening matrix B
and exposes only the precision matvec

    Omega v = B.T @ (B @ v).

Two checks precede the full diagnostic:

1. ||B r||^2 must match the native fitted Vecchia GLS quadratic exactly.
2. On the same 8 x 400 max-min variables used by the dense experiment, full
   eigendecomposition of the 3,200 x 3,200 principal precision is compared to
   sparse-matvec Lanczos/SLQ.

The full-data curve orders covariance modes from largest to smallest, which is
equivalent to ordering precision eigenvalues from smallest to largest.  For a
precision threshold eta it estimates

    N(eta) = tr 1{Omega <= eta}
    C(eta) = r.T Omega 1{Omega <= eta} r.

Under a known, correct Gaussian model E[C(eta)] = N(eta).  The present output
is exploratory because beta and covariance parameters were estimated from the
same day; calibration is a separate bootstrap step.
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
import scipy.sparse
import torch


HERE = Path(__file__).resolve().parent
REPO = next(parent for parent in HERE.parents if (parent / "src/GEMS_TCO").is_dir())
SRC = REPO / "src"
VECCHIA_APPROX = HERE.parent / "vecchia_approximation"
for path in (HERE, SRC, VECCHIA_APPROX):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import vecchia_adapted_fixed_lag643_core as core  # noqa: E402
from GEMS_TCO.vecchia_realdata_corridor_width_4x4_lag432 import (  # noqa: E402
    DirectionalRealDataCorridorWidth4x4Lag432VecchiaFit,
)
from real_maxmin400_full_eigen_vs_lanczos_090326 import (  # noqa: E402
    LanczosRun,
    atomic_csv,
    atomic_json,
    fitted_physical,
    lanczos_tridiagonal,
    load_fitted_record,
    parse_date,
    parse_steps,
    quadrature_atoms,
    select_common_maxmin,
)
from vecchia_sparse_precision_operator_090326 import (  # noqa: E402
    SparseVecchiaPrecision,
    build_sparse_vecchia_precision,
    native_gls_quadratic,
    verify_precision_identity,
)


DTYPE = torch.float64


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
    parser.add_argument("--smooth", type=float, default=0.5, choices=[0.5])
    parser.add_argument("--target-chunk-size", type=int, default=64, choices=[64])
    parser.add_argument("--subset-spatial", type=int, default=400)
    parser.add_argument(
        "--subset-lanczos-steps", nargs="+", type=int, default=[64, 128, 256, 512]
    )
    parser.add_argument("--subset-slq-probes", type=int, default=16)
    parser.add_argument(
        "--full-lanczos-steps", nargs="+", type=int, default=[64, 128, 256, 512]
    )
    parser.add_argument("--full-slq-probes", type=int, default=12)
    parser.add_argument("--full-thresholds", type=int, default=1200)
    parser.add_argument("--bands", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=20260903)
    parser.add_argument("--coefficient-drop-tolerance", type=float, default=0.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "real_full_vecchia_precision_lanczos_20240703_090326",
    )
    return parser


def loader_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        real_data_root=Path(args.real_data_root),
        lat_range="-3,2",
        lon_range="121,131",
        hours_per_day=8,
        keep_exact_loc=True,
        empirical_max_lat_offset=20,
        empirical_max_lon_offset=20,
        empirical_min_pair_count=1000,
        empirical_smooth_bandwidth_deg=0.063,
        subgrid_max_condition_number=100.0,
    )


def setup_model(
    asset: core.DayAsset,
    fitted: dict[str, float],
    seed: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[DirectionalRealDataCorridorWidth4x4Lag432VecchiaFit, torch.Tensor, np.ndarray, dict[str, float]]:
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
    started = time.perf_counter()
    model.precompute_conditioning_sets()
    precompute_s = time.perf_counter() - started
    params = torch.as_tensor(core.physical_to_raw(fitted), dtype=DTYPE)
    started = time.perf_counter()
    with torch.no_grad():
        beta = model.get_gls_beta(params).detach().cpu().numpy().reshape(-1)
    beta_s = time.perf_counter() - started
    return model, params, beta, {
        "vecchia_precompute": precompute_s,
        "gls_beta": beta_s,
    }


def threshold_sum_leq(
    atoms: np.ndarray,
    weights: np.ndarray,
    thresholds_ascending: np.ndarray,
    multiply_atom: bool,
) -> np.ndarray:
    order = np.argsort(atoms)
    atom_values = np.asarray(atoms[order], dtype=np.float64)
    terms = np.asarray(weights[order], dtype=np.float64)
    if multiply_atom:
        terms = terms * atom_values
    cumulative = np.cumsum(terms)
    counts = np.searchsorted(atom_values, thresholds_ascending, side="right")
    out = np.zeros_like(thresholds_ascending, dtype=np.float64)
    positive = counts > 0
    out[positive] = cumulative[counts[positive] - 1]
    return out


def ascending_rank_thresholds(eigenvalues_ascending: np.ndarray) -> np.ndarray:
    values = np.asarray(eigenvalues_ascending, dtype=np.float64)
    thresholds = np.empty_like(values)
    thresholds[:-1] = np.sqrt(values[:-1] * values[1:])
    thresholds[-1] = np.inf
    return thresholds


def run_lanczos_family(
    matvec: Callable[[np.ndarray], np.ndarray],
    residual: np.ndarray,
    levels: list[int],
    n_probes: int,
    rng: np.random.Generator,
    reorthogonalization: str,
    label: str,
) -> tuple[LanczosRun, list[LanczosRun], dict[str, float]]:
    max_steps = max(levels)
    print(f"{label}: residual Lanczos m={max_steps}", flush=True)
    residual_run = lanczos_tridiagonal(
        matvec,
        residual,
        max_steps=max_steps,
        reorthogonalization=reorthogonalization,
    )
    probe_runs: list[LanczosRun] = []
    started = time.perf_counter()
    for probe_index in range(int(n_probes)):
        probe = rng.choice(np.asarray([-1.0, 1.0]), size=len(residual), replace=True)
        run = lanczos_tridiagonal(
            matvec,
            probe,
            max_steps=max_steps,
            reorthogonalization=reorthogonalization,
        )
        probe_runs.append(run)
        print(
            f"  {label} probe {probe_index + 1}/{n_probes}: "
            f"steps={run.steps_completed}, time={run.total_s:.2f}s",
            flush=True,
        )
    return residual_run, probe_runs, {
        "residual_lanczos": residual_run.total_s,
        "residual_matvec": residual_run.matvec_s,
        "slq": time.perf_counter() - started,
        "slq_matvec": float(sum(run.matvec_s for run in probe_runs)),
    }


def evaluate_precision_curves(
    residual_run: LanczosRun,
    probe_runs: list[LanczosRun],
    levels: list[int],
    thresholds: np.ndarray,
    n: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for steps in levels:
        atoms, weights = quadrature_atoms(residual_run, steps)
        energy = threshold_sum_leq(atoms, weights, thresholds, multiply_atom=True)
        probe_counts = []
        for run in probe_runs:
            probe_atoms, probe_weights = quadrature_atoms(run, steps)
            probe_counts.append(
                threshold_sum_leq(
                    probe_atoms, probe_weights, thresholds, multiply_atom=False
                )
            )
        count_array = np.vstack(probe_counts)
        mode_count = count_array.mean(axis=0)
        mode_se = (
            count_array.std(axis=0, ddof=1) / math.sqrt(len(probe_runs))
            if len(probe_runs) > 1
            else np.full(len(thresholds), np.nan)
        )
        total_energy = float(energy[-1])
        bridge = (
            energy - mode_count / n * total_energy
        ) / math.sqrt(2.0 * n)
        frames.append(
            pd.DataFrame(
                {
                    "lanczos_steps": int(steps),
                    "threshold_index": np.arange(len(thresholds)),
                    "precision_threshold": thresholds,
                    "mode_count_slq": mode_count,
                    "mode_count_slq_se": mode_se,
                    "mode_fraction_slq": mode_count / n,
                    "cumulative_energy_lanczos": energy,
                    "cumulative_energy_per_n": energy / n,
                    "shape_bridge": bridge,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def save_lanczos_runs(
    path: Path,
    residual_run: LanczosRun,
    probe_runs: list[LanczosRun],
) -> None:
    np.savez(
        path,
        residual_alpha=residual_run.alpha,
        residual_beta=residual_run.beta,
        residual_start_norm_sq=residual_run.start_norm_sq,
        probe_alpha=np.vstack([run.alpha for run in probe_runs]),
        probe_beta=np.vstack([run.beta for run in probe_runs]),
        probe_start_norm_sq=np.asarray([run.start_norm_sq for run in probe_runs]),
    )


def subset_compact_indices(
    asset: core.DayAsset,
    precision: SparseVecchiaPrecision,
    selected_spatial: np.ndarray,
) -> np.ndarray:
    n_grid = int(asset.grid_coords.shape[0])
    globals_time_major = np.concatenate(
        [time_index * n_grid + selected_spatial for time_index in range(len(asset.keys))]
    )
    compact = precision.global_to_compact[globals_time_major]
    if np.any(compact < 0):
        raise RuntimeError("The common-valid 8x400 subset contains an invalid compact row")
    return compact.astype(np.int64)


def validate_subset_precision(
    asset: core.DayAsset,
    precision: SparseVecchiaPrecision,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], Path]:
    levels = parse_steps(args.subset_lanczos_steps)
    selected, _, selection_meta = select_common_maxmin(
        asset, int(args.subset_spatial)
    )
    compact = subset_compact_indices(asset, precision, selected)
    b_columns = precision.whitener[:, compact].tocsr()
    residual = precision.residual[compact]
    print(
        f"Subset sparse principal precision: n={len(compact):,}, "
        f"B-columns nnz={b_columns.nnz:,}",
        flush=True,
    )
    build_started = time.perf_counter()
    omega_sparse = (b_columns.T @ b_columns).tocsr()
    omega_dense = omega_sparse.toarray()
    build_s = time.perf_counter() - build_started
    test = rng.standard_normal((len(compact), 3))
    sparse_product = np.asarray(b_columns.T @ (b_columns @ test))
    dense_product = omega_dense @ test
    matmat_relative_error = float(
        np.linalg.norm(sparse_product - dense_product) / np.linalg.norm(dense_product)
    )

    eigen_started = time.perf_counter()
    eigenvalues, eigenvectors = scipy.linalg.eigh(
        omega_dense,
        lower=True,
        check_finite=False,
        overwrite_a=False,
        driver="evd",
    )
    eigen_s = time.perf_counter() - eigen_started
    if eigenvalues[0] <= 0.0:
        raise RuntimeError(f"Subset principal precision is not SPD: {eigenvalues[0]}")
    coefficients = eigenvectors.T @ residual
    exact_y2 = eigenvalues * coefficients**2
    exact_energy = np.cumsum(exact_y2)
    thresholds = ascending_rank_thresholds(eigenvalues)
    exact = pd.DataFrame(
        {
            "rank": np.arange(1, len(compact) + 1),
            "rank_fraction": np.arange(1, len(compact) + 1) / len(compact),
            "precision_eigenvalue": eigenvalues,
            "implied_covariance_eigenvalue": 1.0 / eigenvalues,
            "standardized_energy": exact_y2,
            "cumulative_energy": exact_energy,
            "cumulative_energy_per_n": exact_energy / len(compact),
            "threshold_above_rank": thresholds,
        }
    )

    def subset_matvec(vector: np.ndarray) -> np.ndarray:
        return np.asarray(b_columns.T @ (b_columns @ vector)).reshape(-1)

    residual_run, probe_runs, lanczos_timings = run_lanczos_family(
        subset_matvec,
        residual,
        levels,
        int(args.subset_slq_probes),
        rng,
        reorthogonalization="full",
        label="subset",
    )
    curves = evaluate_precision_curves(
        residual_run, probe_runs, levels, thresholds, len(compact)
    )
    repeated_energy = np.tile(exact_energy, len(levels))
    repeated_count = np.tile(np.arange(1, len(compact) + 1), len(levels))
    curves["mode_count_exact"] = repeated_count
    curves["mode_fraction_exact"] = repeated_count / len(compact)
    curves["cumulative_energy_exact"] = repeated_energy
    curves["cumulative_energy_exact_per_n"] = repeated_energy / len(compact)
    curves["energy_error_per_n"] = (
        curves["cumulative_energy_lanczos"] - repeated_energy
    ) / len(compact)
    curves["mode_fraction_error"] = (
        curves["mode_count_slq"] - repeated_count
    ) / len(compact)
    exact_total = float(exact_energy[-1])
    exact_bridge = (
        exact_energy
        - np.arange(1, len(compact) + 1) / len(compact) * exact_total
    ) / math.sqrt(2.0 * len(compact))
    curves["shape_bridge_exact"] = np.tile(exact_bridge, len(levels))

    summary_rows = []
    for steps, group in curves.groupby("lanczos_steps", sort=True):
        summary_rows.append(
            {
                "lanczos_steps": int(steps),
                "n_slq_probes": int(args.subset_slq_probes),
                "mode_fraction_rmse": float(
                    np.sqrt(np.mean(group["mode_fraction_error"] ** 2))
                ),
                "mode_fraction_max_abs_error": float(
                    np.max(np.abs(group["mode_fraction_error"]))
                ),
                "energy_per_n_rmse": float(
                    np.sqrt(np.mean(group["energy_error_per_n"] ** 2))
                ),
                "energy_per_n_max_abs_error": float(
                    np.max(np.abs(group["energy_error_per_n"]))
                ),
                "endpoint_energy_relative_error": float(
                    abs(group["cumulative_energy_lanczos"].iloc[-1] / exact_total - 1.0)
                ),
                "shape_D_exact": float(np.max(np.abs(exact_bridge))),
                "shape_D_lanczos_slq": float(np.max(np.abs(group["shape_bridge"]))),
            }
        )
    summary = pd.DataFrame(summary_rows)
    save_lanczos_runs(
        args.output_dir / "subset_lanczos_tridiagonals.npz",
        residual_run,
        probe_runs,
    )
    atomic_csv(args.output_dir / "subset_principal_precision_full_eigen.csv", exact)
    atomic_csv(args.output_dir / "subset_precision_lanczos_comparison.csv", curves)
    atomic_csv(args.output_dir / "subset_precision_accuracy_summary.csv", summary)

    figure = plot_subset_validation(curves, summary, args.output_dir)
    meta = {
        **selection_meta,
        "n_observations": int(len(compact)),
        "interpretation": "principal precision; conditional on variables outside subset",
        "b_columns_nnz": int(b_columns.nnz),
        "omega_principal_nnz": int(omega_sparse.nnz),
        "sparse_dense_matmat_relative_error": matmat_relative_error,
        "precision_min_eigenvalue": float(eigenvalues[0]),
        "precision_max_eigenvalue": float(eigenvalues[-1]),
        "condition_number": float(eigenvalues[-1] / eigenvalues[0]),
        "principal_precision_build_s": build_s,
        "full_eigendecomposition_s": eigen_s,
        **lanczos_timings,
    }
    del omega_sparse, omega_dense, eigenvectors, b_columns
    gc.collect()
    return curves, summary, meta, figure


def thresholds_from_runs(
    residual_run: LanczosRun,
    probe_runs: list[LanczosRun],
    steps: int,
    n_thresholds: int,
) -> np.ndarray:
    atoms = [quadrature_atoms(residual_run, steps)[0]]
    atoms.extend(quadrature_atoms(run, steps)[0] for run in probe_runs)
    positive = np.concatenate(atoms)
    positive = positive[np.isfinite(positive) & (positive > 0.0)]
    if positive.size == 0:
        raise RuntimeError("No positive full-data precision Ritz values")
    lo = float(np.min(positive)) * (1.0 - 1e-10)
    hi = float(np.max(positive)) * (1.0 + 1e-10)
    interior = np.geomspace(lo, hi, max(2, int(n_thresholds) - 2))
    return np.concatenate([[0.0], interior, [np.inf]])


def full_convergence_summary(curves: pd.DataFrame, n: int) -> pd.DataFrame:
    levels = sorted(curves["lanczos_steps"].unique())
    reference = curves[curves["lanczos_steps"].eq(levels[-1])]
    rows = []
    for steps in levels:
        group = curves[curves["lanczos_steps"].eq(steps)]
        rows.append(
            {
                "lanczos_steps": int(steps),
                "mode_fraction_rmse_vs_max_m": float(
                    np.sqrt(
                        np.mean(
                            (
                                group["mode_fraction_slq"].to_numpy()
                                - reference["mode_fraction_slq"].to_numpy()
                            )
                            ** 2
                        )
                    )
                ),
                "energy_per_n_rmse_vs_max_m": float(
                    np.sqrt(
                        np.mean(
                            (
                                group["cumulative_energy_per_n"].to_numpy()
                                - reference["cumulative_energy_per_n"].to_numpy()
                            )
                            ** 2
                        )
                    )
                ),
                "endpoint_mode_count": float(group["mode_count_slq"].iloc[-1]),
                "endpoint_energy": float(group["cumulative_energy_lanczos"].iloc[-1]),
                "mean_standardized_energy": float(
                    group["cumulative_energy_lanczos"].iloc[-1] / n
                ),
                "scale_statistic": float(
                    (group["cumulative_energy_lanczos"].iloc[-1] - n)
                    / math.sqrt(2.0 * n)
                ),
                "shape_D": float(np.max(np.abs(group["shape_bridge"]))),
                "median_mode_count_se": float(np.nanmedian(group["mode_count_slq_se"])),
                "max_mode_count_se": float(np.nanmax(group["mode_count_slq_se"])),
            }
        )
    return pd.DataFrame(rows)


def full_equal_mode_bands(curve: pd.DataFrame, n_bands: int) -> pd.DataFrame:
    ordered = curve.sort_values("mode_fraction_slq")
    x = ordered["mode_fraction_slq"].to_numpy(dtype=np.float64)
    y = ordered["cumulative_energy_per_n"].to_numpy(dtype=np.float64)
    unique_x, unique_index = np.unique(x, return_index=True)
    unique_y = y[unique_index]
    edges = np.linspace(0.0, 1.0, int(n_bands) + 1)
    energy_edges = np.interp(edges, unique_x, unique_y, left=0.0, right=y[-1])
    rows = []
    for index in range(int(n_bands)):
        lo, hi = float(edges[index]), float(edges[index + 1])
        energy_increment = float(energy_edges[index + 1] - energy_edges[index])
        rows.append(
            {
                "band": index + 1,
                "covariance_mode_fraction_lo": lo,
                "covariance_mode_fraction_hi": hi,
                "covariance_mode_fraction_mid": 0.5 * (lo + hi),
                "estimated_mode_count": (hi - lo) * float(curve["mode_count_slq"].iloc[-1]),
                "standardized_energy": energy_increment * float(curve["mode_count_slq"].iloc[-1]),
                "standardized_energy_per_mode": energy_increment / (hi - lo),
            }
        )
    return pd.DataFrame(rows)


def plot_subset_validation(
    curves: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
) -> Path:
    levels = sorted(curves["lanczos_steps"].unique())
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(levels)))
    reference = curves[curves["lanczos_steps"].eq(levels[-1])]
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 4.8), constrained_layout=True)
    axes[0].plot(
        reference["mode_fraction_exact"],
        reference["cumulative_energy_exact_per_n"],
        color="black",
        lw=2.2,
        label="full eigen of principal precision",
    )
    for color, steps in zip(colors, levels):
        group = curves[curves["lanczos_steps"].eq(steps)]
        axes[0].plot(
            group["mode_fraction_slq"],
            group["cumulative_energy_per_n"],
            color=color,
            lw=1.1,
            label=f"Lanczos/SLQ m={steps}",
        )
        axes[1].plot(
            group["mode_fraction_exact"],
            group["energy_error_per_n"],
            color=color,
            lw=1.1,
        )
        axes[2].plot(
            group["mode_fraction_exact"],
            group["mode_fraction_error"],
            color=color,
            lw=1.1,
        )
    axes[0].set_title("Principal-precision eigen curve")
    axes[0].set_xlabel("covariance-mode fraction")
    axes[0].set_ylabel("cumulative standardized energy / n")
    axes[0].legend(fontsize=7)
    axes[1].axhline(0.0, color="black", lw=0.8)
    axes[1].set_title("Residual quadrature error")
    axes[1].set_xlabel("exact mode fraction")
    axes[1].set_ylabel("Lanczos - exact energy, divided by n")
    axes[2].axhline(0.0, color="black", lw=0.8)
    axes[2].set_title("SLQ mode-fraction error")
    axes[2].set_xlabel("exact mode fraction")
    axes[2].set_ylabel("SLQ - exact fraction")
    for axis in axes:
        axis.grid(alpha=0.2)
    best = summary.sort_values("lanczos_steps").iloc[-1]
    fig.suptitle(
        f"Sparse Vecchia precision validation on 8x400 subset: "
        f"energy RMSE={best['energy_per_n_rmse']:.3g}, "
        f"CDF RMSE={best['mode_fraction_rmse']:.3g}",
        fontsize=13,
    )
    path = output_dir / "subset_sparse_precision_full_eigen_vs_lanczos.png"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_full_diagnostic(
    curves: pd.DataFrame,
    bands: pd.DataFrame,
    output_dir: Path,
) -> Path:
    levels = sorted(curves["lanczos_steps"].unique())
    colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(levels)))
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.0), constrained_layout=True)
    for color, steps in zip(colors, levels):
        group = curves[curves["lanczos_steps"].eq(steps)]
        axes[0].plot(
            group["mode_fraction_slq"],
            group["cumulative_energy_per_n"],
            color=color,
            lw=1.5,
            label=f"m={steps}",
        )
        axes[1].plot(
            group["mode_fraction_slq"],
            group["shape_bridge"],
            color=color,
            lw=1.2,
            label=f"m={steps}",
        )
    axes[0].plot([0, 1], [0, 1], color="0.35", ls="--", lw=1.0)
    axes[0].set_xlabel("estimated covariance-mode fraction")
    axes[0].set_ylabel("cumulative standardized energy / n")
    axes[0].set_title("Full-data eigen diagnostic")
    axes[0].legend(fontsize=8)
    axes[1].axhline(0.0, color="0.35", lw=0.8)
    axes[1].set_xlabel("estimated covariance-mode fraction")
    axes[1].set_ylabel("scale-removed bridge")
    axes[1].set_title("Spectral shape")
    axes[1].legend(fontsize=8)
    axes[2].plot(
        bands["band"],
        bands["standardized_energy_per_mode"],
        color="#1f77b4",
        marker="o",
        lw=1.5,
    )
    axes[2].axhline(1.0, color="0.35", ls="--", lw=1.0)
    axes[2].set_xlabel("equal-mode band: large to small implied covariance eigenvalue")
    axes[2].set_ylabel("standardized energy per estimated mode")
    axes[2].set_title("Full-data 20-band energy")
    for axis in axes:
        axis.grid(alpha=0.2)
    path = output_dir / "full_vecchia_precision_eigen_diagnostic.png"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def write_results_note(
    output_dir: Path,
    identity: dict[str, float],
    subset_summary: pd.DataFrame,
    full_summary: pd.DataFrame,
    bands: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    subset_best = subset_summary.sort_values("lanczos_steps").iloc[-1]
    ordered_full = full_summary.sort_values("lanczos_steps")
    full_best = ordered_full.iloc[-1]
    full_previous = ordered_full.iloc[-2] if len(ordered_full) > 1 else None
    largest_band = bands.loc[bands["standardized_energy_per_mode"].idxmax()]
    first_band = bands.iloc[0]
    second_band = bands.iloc[1] if len(bands) > 1 else bands.iloc[0]
    last_three = bands.tail(min(3, len(bands)))["standardized_energy_per_mode"]
    convergence_text = (
        f"- Relative to m={int(full_best['lanczos_steps'])}, m={int(full_previous['lanczos_steps'])} "
        f"has cumulative energy/n RMSE `{full_previous['energy_per_n_rmse_vs_max_m']:.6g}` "
        f"and spectral-CDF RMSE `{full_previous['mode_fraction_rmse_vs_max_m']:.6g}`\n"
        if full_previous is not None
        else ""
    )
    text = f"""# Full real-data matrix-free Vecchia eigen diagnostic

## Construction check

The fitted adapted lag-4/3/2 block conditionals were stacked into a sparse
whitening matrix `B`, with precision applied only as `B.T @ (B @ v)`.

- Valid observations: `{metadata['operator']['n_valid']:,}`
- B nonzeros: `{metadata['operator']['nnz']:,}`
- Mean nonzeros per row: `{metadata['operator']['nnz_per_row_mean']:.2f}`
- CSR storage: `{metadata['operator']['csr_storage_bytes'] / 2**20:.2f} MiB`
- Native-vs-sparse quadratic relative error: `{identity['relative_error']:.3e}`

## 8 x 400 sparse-operator validation

Full eigendecomposition of the 3,200-variable principal precision was compared
with Lanczos/SLQ using the sparse `B` operator.  At
`m={int(subset_best['lanczos_steps'])}`:

- cumulative energy/n RMSE: `{subset_best['energy_per_n_rmse']:.6g}`
- energy/n maximum error: `{subset_best['energy_per_n_max_abs_error']:.6g}`
- spectral-CDF RMSE: `{subset_best['mode_fraction_rmse']:.6g}`
- spectral-CDF maximum error: `{subset_best['mode_fraction_max_abs_error']:.6g}`
- endpoint energy relative error: `{subset_best['endpoint_energy_relative_error']:.3e}`

This validates Lanczos on the actual fitted sparse Vecchia precision, not only
on a dense exact covariance callback.  The principal precision is a
conditional-subset object; it is not the marginal precision of the 3,200-point
subset and is used only as a numerical reference.

## Full-data diagnostic

The full curve used `m={int(full_best['lanczos_steps'])}` and
`{metadata['settings']['full_slq_probes']}` Rademacher SLQ probes.

- Mean standardized energy: `{full_best['mean_standardized_energy']:.6g}`
- Scale statistic: `{full_best['scale_statistic']:.6g}`
- Shape D: `{full_best['shape_D']:.6g}`
- Highest-energy 5% band: band `{int(largest_band['band'])}`, energy/mode
  `{largest_band['standardized_energy_per_mode']:.6g}`
- First two large-covariance bands: `{first_band['standardized_energy_per_mode']:.6g}`,
  `{second_band['standardized_energy_per_mode']:.6g}`
- Final three small-covariance bands: `{', '.join(f'{value:.6g}' for value in last_three)}`
{convergence_text}

The fitted scale makes the endpoint mean close to one by construction; the
non-uniform distribution across eigenvalue bands is the informative feature.
The broad low-high-low pattern is visible at both m=256 and m=512, but the
hard-projector maximum `D` is more sensitive to Lanczos order.  These values
remain exploratory: beta and covariance parameters were fitted on the same
day, so neither the reference line nor `D` is calibrated for estimation
leverage.  The next statistical step is a parametric bootstrap that repeats
fitting and the same matrix-free diagnostic.
"""
    (output_dir / "RESULTS.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    subset_levels = parse_steps(args.subset_lanczos_steps)
    full_levels = parse_steps(args.full_lanczos_steps)
    if min(
        int(args.subset_slq_probes),
        int(args.full_slq_probes),
        int(args.full_thresholds),
        int(args.bands),
    ) < 1:
        raise ValueError("Probe, threshold, and band counts must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    workflow_started = time.perf_counter()
    rng = np.random.default_rng(int(args.random_seed))

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
    load_started = time.perf_counter()
    print(f"Loading real data {args.date}", flush=True)
    asset = core.load_real_asset(spec, loader_args(args))
    load_s = time.perf_counter() - load_started
    if str(record["date"]) != asset.date:
        raise RuntimeError("Stored fit and loaded asset dates differ")

    print("Precomputing fitted adapted-432 model and GLS beta", flush=True)
    model, params, beta, setup_timings = setup_model(asset, fitted, seed, args)
    native_started = time.perf_counter()
    native_quad = native_gls_quadratic(model, params, beta)
    native_quad_s = time.perf_counter() - native_started

    print("Building sparse block-Vecchia whitening matrix B", flush=True)
    precision = build_sparse_vecchia_precision(
        model,
        params,
        beta,
        chunk_size=int(args.target_chunk_size),
        coefficient_drop_tolerance=float(args.coefficient_drop_tolerance),
        progress=lambda message: print(f"  {message}", flush=True),
    )
    identity = verify_precision_identity(precision, native_quad)
    if identity["relative_error"] > 1e-10:
        raise RuntimeError(f"Sparse precision identity failed: {identity}")
    reconstructed_nll = (
        float(precision.metadata["conditional_logdet_half"])
        + 0.5 * float(identity["sparse_quadratic"])
    ) / precision.n
    fitted_nll_error = float(reconstructed_nll - float(record["final_native_nll"]))
    print(
        f"B shape={precision.whitener.shape}, nnz={precision.whitener.nnz:,}, "
        f"native quadratic relative error={identity['relative_error']:.3e}",
        flush=True,
    )

    scipy.sparse.save_npz(
        args.output_dir / "vecchia_whitener_B.npz",
        precision.whitener,
        compressed=False,
    )
    np.savez(
        args.output_dir / "vecchia_compact_vectors.npz",
        valid_global_indices=precision.valid_global_indices,
        global_to_compact=precision.global_to_compact,
        response=precision.response,
        residual=precision.residual,
        beta=beta,
        lat_mean=np.asarray([float(model.lat_mean_val)]),
    )

    print("Validating sparse precision Lanczos on the 8x400 subset", flush=True)
    subset_curves, subset_summary, subset_meta, subset_figure = validate_subset_precision(
        asset, precision, args, rng
    )
    print("Subset accuracy summary", flush=True)
    print(subset_summary.to_string(index=False), flush=True)

    # The sparse operator and vectors are now self-contained; release the much
    # larger precomputed batch tensors before the full Lanczos run.
    model_summary = model.cluster_summary()
    lat_mean = float(model.lat_mean_val)
    del model, params
    gc.collect()

    print(f"Running full precision Lanczos on n={precision.n:,}", flush=True)
    residual_run, probe_runs, full_lanczos_timings = run_lanczos_family(
        precision.matvec,
        precision.residual,
        full_levels,
        int(args.full_slq_probes),
        rng,
        reorthogonalization="none",
        label="full",
    )
    thresholds = thresholds_from_runs(
        residual_run,
        probe_runs,
        max(full_levels),
        int(args.full_thresholds),
    )
    full_curves = evaluate_precision_curves(
        residual_run, probe_runs, full_levels, thresholds, precision.n
    )
    full_summary = full_convergence_summary(full_curves, precision.n)
    max_curve = full_curves[
        full_curves["lanczos_steps"].eq(max(full_levels))
    ].copy()
    full_bands = full_equal_mode_bands(max_curve, int(args.bands))
    save_lanczos_runs(
        args.output_dir / "full_lanczos_tridiagonals.npz",
        residual_run,
        probe_runs,
    )
    atomic_csv(args.output_dir / "full_precision_lanczos_curves.csv", full_curves)
    atomic_csv(args.output_dir / "full_precision_convergence_summary.csv", full_summary)
    atomic_csv(args.output_dir / "full_precision_equal_mode_bands.csv", full_bands)
    full_figure = plot_full_diagnostic(full_curves, full_bands, args.output_dir)

    metadata = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "date": asset.date,
        "data_source": asset.source_path,
        "fit_reused_without_refit": True,
        "fit": {
            "geometry": "adapted",
            "lag_pattern": "4/3/2",
            "target_chunk_size": 64,
            "fitted_parameters": fitted,
            "stored_native_nll": float(record["final_native_nll"]),
            "reconstructed_native_nll": reconstructed_nll,
            "reconstructed_minus_stored_nll": fitted_nll_error,
            "gls_beta": beta,
            "gls_lat_mean": lat_mean,
            "model_summary": model_summary,
        },
        "operator": precision.metadata,
        "quadratic_identity": identity,
        "subset_validation": subset_meta,
        "settings": {
            "subset_lanczos_steps": subset_levels,
            "subset_slq_probes": int(args.subset_slq_probes),
            "full_lanczos_steps": full_levels,
            "full_slq_probes": int(args.full_slq_probes),
            "full_thresholds": int(args.full_thresholds),
            "bands": int(args.bands),
            "random_seed": int(args.random_seed),
            "full_reorthogonalization": "none",
            "hard_projector": True,
        },
        "timings_s": {
            "data_load": load_s,
            **setup_timings,
            "native_quadratic": native_quad_s,
            "sparse_operator_build": precision.metadata["build_s"],
            **{f"full_{key}": value for key, value in full_lanczos_timings.items()},
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
        "figures": [subset_figure, full_figure],
    }
    atomic_json(args.output_dir / "run_metadata.json", metadata)
    write_results_note(
        args.output_dir,
        identity,
        subset_summary,
        full_summary,
        full_bands,
        metadata,
    )
    print("Full-data convergence summary", flush=True)
    print(full_summary.to_string(index=False), flush=True)
    print("Full-data bands", flush=True)
    print(full_bands.to_string(index=False), flush=True)
    print(f"Saved outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
