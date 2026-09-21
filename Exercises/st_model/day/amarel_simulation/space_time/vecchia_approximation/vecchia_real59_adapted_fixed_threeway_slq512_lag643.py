#!/usr/bin/env python3
"""Three-way adapted/fixed Vecchia comparison for the usable July dates.

The experiment repeats the adapted and fixed lag-6/4/3 fits on July 1--30 of
2024 and 2025.  The incomplete 2025-07-24 day is excluded, so the nominal
60-day window contains 59 usable dates.  Each fitted method is compared by:

1. native Vecchia negative log likelihood per observation;
2. the existing dense full-eigen diagnostic on 400 max-min points per hour;
3. a full-data, matrix-free sparse-precision Lanczos diagnostic.

For (3), fitted block conditionals are stacked into a sparse whitener B and
the precision is applied only as ``Omega v = B.T @ (B @ v)``.  Random-probe
SLQ estimates the empirical spectral CDF of Omega.  Its 1/3 and 2/3 mode-count
quantiles define low-, middle-, and high-frequency proxy bands (small to large
precision eigenvalues).  An independent random-start Lanczos run constructs
implicit Rayleigh--Ritz vectors; exactly 170, 170, and 172 representatives are
selected across the three bands.  These are approximate Ritz vectors, not 512
exact eigenvectors, so the tail Ritz-residual estimate is always reported.

Daily three-panel plots are written below ``daily_subplots``.  The two yearly
monthly-average three-panel plots are written directly below ``output_root``.
Per-method checkpoints make the long Amarel job restartable.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import socket
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
import scipy.linalg
import torch


HERE = Path(__file__).resolve().parent
SUBMIT_DIR = Path(os.environ.get("SLURM_SUBMIT_DIR", str(HERE))).resolve()
AMAREL_ROOT = Path("/home/jl2815/tco")
LOCAL_SRC = Path("/Users/joonwonlee/Documents/GEMS_TCO-1/src")
INTERACTION_DIR = HERE.parent / "interaction_diagnostic"
for candidate in (AMAREL_ROOT, SUBMIT_DIR, HERE, LOCAL_SRC, INTERACTION_DIR):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import vecchia_adapted_fixed_lag643_core as core  # noqa: E402
import vecchia_real60_adapted_fixed_full_eigen_lag643_run10 as reference  # noqa: E402
from vecchia_sparse_precision_operator_090326 import (  # noqa: E402
    SparseVecchiaPrecision,
    build_sparse_vecchia_precision,
    native_gls_quadratic,
    verify_precision_identity,
)


METHODS = reference.METHODS
COLORS = reference.COLORS
LINESTYLES = reference.LINESTYLES
LABELS = reference.LABELS
BAND_NAMES = ("low", "middle", "high")
BAND_LABELS = {
    "low": "low-frequency proxy",
    "middle": "middle-frequency proxy",
    "high": "high-frequency proxy",
}
BAND_COLORS = {
    "low": "#DCEEFF",
    "middle": "#E9E5FF",
    "high": "#FFE4DC",
}
DAILY_DIR_NAME = "daily_subplots"
CACHE_DIR_NAME = ".diagnostic_cache"


@dataclass
class KrylovRun:
    alpha: np.ndarray
    beta: np.ndarray
    tail_beta: float
    start_norm_sq: float
    overlap: np.ndarray | None
    matvec_seconds: float
    total_seconds: float
    breakdown: bool

    @property
    def steps(self) -> int:
        return int(len(self.alpha))


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def parse_band_counts(text: str) -> tuple[int, int, int]:
    values = tuple(int(part.strip()) for part in str(text).split(","))
    if len(values) != 3 or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("band counts must be three positive integers")
    if sum(values) != 512:
        raise argparse.ArgumentTypeError("band counts must sum to exactly 512")
    return values


def stable_rng(seed: int, date: str, method: str) -> np.random.Generator:
    date_number = int(date.replace("-", ""))
    method_number = METHODS.index(method) + 1
    sequence = np.random.SeedSequence([int(seed), date_number, method_number])
    return np.random.default_rng(sequence)


def lanczos_tridiagonal(
    matvec: Callable[[np.ndarray], np.ndarray],
    start: np.ndarray,
    max_steps: int,
    overlap_vector: np.ndarray | None = None,
    breakdown_tolerance: float = 1e-13,
) -> KrylovRun:
    """Run a memory-light symmetric Lanczos recurrence without reorthogonalization.

    If ``overlap_vector`` is supplied, q_j' overlap_vector is retained.  This
    permits residual projections onto every implicit Ritz vector without
    storing the n-by-m Krylov basis.
    """
    vector = np.asarray(start, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("Lanczos start must have positive finite norm")
    n = len(vector)
    requested = min(int(max_steps), n)
    if requested < 2:
        raise ValueError("Lanczos requires at least two steps")
    overlap_source = None
    if overlap_vector is not None:
        overlap_source = np.asarray(overlap_vector, dtype=np.float64).reshape(-1)
        if overlap_source.size != n:
            raise ValueError("overlap vector and Lanczos start have different sizes")

    alpha = np.empty(requested, dtype=np.float64)
    beta = np.empty(requested - 1, dtype=np.float64)
    overlaps = np.empty(requested, dtype=np.float64) if overlap_source is not None else None
    q = vector / norm
    q_previous = np.zeros_like(q)
    beta_previous = 0.0
    matvec_seconds = 0.0
    tail_beta = np.nan
    breakdown = False
    completed = 0
    started = time.perf_counter()
    for step in range(requested):
        if overlaps is not None and overlap_source is not None:
            overlaps[step] = float(q @ overlap_source)
        matvec_started = time.perf_counter()
        z = np.asarray(matvec(q), dtype=np.float64).reshape(-1)
        matvec_seconds += time.perf_counter() - matvec_started
        if z.size != n or not np.all(np.isfinite(z)):
            raise RuntimeError("precision matvec returned an invalid vector")
        if step:
            z -= beta_previous * q_previous
        diagonal = float(q @ z)
        z -= diagonal * q
        next_beta = float(np.linalg.norm(z))
        alpha[step] = diagonal
        completed = step + 1
        tail_beta = next_beta
        if step == requested - 1:
            break
        beta[step] = next_beta
        if not math.isfinite(next_beta) or next_beta <= breakdown_tolerance:
            breakdown = True
            break
        q_previous, q = q, z / next_beta
        beta_previous = next_beta

    return KrylovRun(
        alpha=alpha[:completed].copy(),
        beta=beta[: max(completed - 1, 0)].copy(),
        tail_beta=float(tail_beta),
        start_norm_sq=norm * norm,
        overlap=None if overlaps is None else overlaps[:completed].copy(),
        matvec_seconds=matvec_seconds,
        total_seconds=time.perf_counter() - started,
        breakdown=breakdown,
    )


def tridiagonal_eigenpairs(run: KrylovRun) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = scipy.linalg.eigh_tridiagonal(
        run.alpha,
        run.beta,
        check_finite=False,
        lapack_driver="auto",
    )
    scale = max(float(np.max(np.abs(values))), 1.0)
    keep = np.isfinite(values) & (values > np.finfo(np.float64).eps * scale)
    if not np.any(keep):
        raise RuntimeError("Lanczos tridiagonal has no positive Ritz values")
    return values[keep], vectors[:, keep]


def slq_spectrum(
    precision: SparseVecchiaPrecision,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], np.ndarray, np.ndarray]:
    """Estimate the full empirical spectral CDF and its one-third boundaries."""
    probe_atoms: list[np.ndarray] = []
    probe_weights: list[np.ndarray] = []
    probe_seconds: list[float] = []
    for probe_index in range(int(args.slq_probes)):
        start = rng.choice(np.asarray([-1.0, 1.0]), size=precision.n)
        run = lanczos_tridiagonal(
            precision.matvec,
            start,
            max_steps=int(args.slq_steps),
        )
        atoms, vectors = tridiagonal_eigenpairs(run)
        weights = float(run.start_norm_sq) * np.square(vectors[0, :])
        probe_atoms.append(atoms)
        probe_weights.append(weights)
        probe_seconds.append(run.total_seconds)
        print(
            f"      SLQ probe {probe_index + 1}/{args.slq_probes}: "
            f"m={run.steps}, {run.total_seconds:.2f}s",
            flush=True,
        )

    atoms = np.concatenate(probe_atoms)
    weights = np.concatenate(
        [values / int(args.slq_probes) for values in probe_weights]
    )
    order = np.argsort(atoms)
    atoms = atoms[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    estimated_total = float(cumulative[-1])
    if abs(estimated_total - precision.n) / precision.n > 1e-8:
        raise RuntimeError(
            f"SLQ endpoint count {estimated_total} does not equal n={precision.n}"
        )

    edge_values = []
    for fraction in (1.0 / 3.0, 2.0 / 3.0):
        position = int(np.searchsorted(cumulative, fraction * precision.n, side="left"))
        edge_values.append(float(atoms[min(position, len(atoms) - 1)]))
    edge1, edge2 = edge_values
    if not 0.0 < edge1 < edge2:
        raise RuntimeError(f"Invalid SLQ band boundaries: {edge_values}")

    minimum = float(atoms[0])
    maximum = float(atoms[-1])
    thresholds = np.geomspace(minimum, maximum, int(args.spectrum_grid))
    indices = np.searchsorted(atoms, thresholds, side="right") - 1
    counts = np.zeros_like(thresholds)
    positive = indices >= 0
    counts[positive] = cumulative[indices[positive]]
    per_probe_counts = []
    for one_atoms, one_weights in zip(probe_atoms, probe_weights):
        one_order = np.argsort(one_atoms)
        sorted_atoms = one_atoms[one_order]
        sorted_cumulative = np.cumsum(one_weights[one_order])
        one_indices = np.searchsorted(sorted_atoms, thresholds, side="right") - 1
        one_counts = np.zeros_like(thresholds)
        one_positive = one_indices >= 0
        one_counts[one_positive] = sorted_cumulative[one_indices[one_positive]]
        per_probe_counts.append(one_counts)
    probe_count_array = np.vstack(per_probe_counts)
    count_se = (
        probe_count_array.std(axis=0, ddof=1) / math.sqrt(int(args.slq_probes))
        if int(args.slq_probes) > 1
        else np.full_like(thresholds, np.nan)
    )
    curve = pd.DataFrame(
        {
            "precision_threshold": thresholds,
            "implied_covariance_eigenvalue": 1.0 / thresholds,
            "mode_count_slq": counts,
            "mode_count_slq_se": count_se,
            "mode_fraction_slq": counts / precision.n,
        }
    )

    band_counts = args.band_mode_counts
    bounds = ((-np.inf, edge1), (edge1, edge2), (edge2, np.inf))
    rows: list[dict[str, Any]] = []
    for band_index, (band, requested, (lower, upper)) in enumerate(
        zip(BAND_NAMES, band_counts, bounds), start=1
    ):
        probe_estimates = []
        for one_atoms, one_weights in zip(probe_atoms, probe_weights):
            mask = (one_atoms > lower) & (one_atoms <= upper)
            probe_estimates.append(float(one_weights[mask].sum()))
        rows.append(
            {
                "band_index": band_index,
                "band": band,
                "frequency_label": BAND_LABELS[band],
                "precision_lower": 0.0 if not np.isfinite(lower) else lower,
                "precision_upper": np.inf if not np.isfinite(upper) else upper,
                "implied_covariance_upper": (
                    np.inf if not np.isfinite(lower) or lower <= 0.0 else 1.0 / lower
                ),
                "implied_covariance_lower": 0.0 if not np.isfinite(upper) else 1.0 / upper,
                "estimated_mode_count": float(np.mean(probe_estimates)),
                "estimated_mode_count_se": (
                    float(np.std(probe_estimates, ddof=1) / math.sqrt(len(probe_estimates)))
                    if len(probe_estimates) > 1
                    else np.nan
                ),
                "estimated_mode_fraction": float(np.mean(probe_estimates) / precision.n),
                "requested_ritz_modes": int(requested),
            }
        )
    bands = pd.DataFrame(rows)
    summary = {
        "slq_probes": int(args.slq_probes),
        "slq_steps": int(args.slq_steps),
        "slq_total_seconds": float(sum(probe_seconds)),
        "slq_endpoint_mode_count": estimated_total,
        "slq_precision_min": minimum,
        "slq_precision_max": maximum,
        "slq_low_middle_boundary": edge1,
        "slq_middle_high_boundary": edge2,
    }
    return curve, bands, summary, atoms, cumulative


def cdf_at(values: np.ndarray, atoms: np.ndarray, cumulative: np.ndarray, n: int) -> np.ndarray:
    indices = np.searchsorted(atoms, values, side="right") - 1
    out = np.zeros(len(values), dtype=np.float64)
    positive = indices >= 0
    out[positive] = cumulative[indices[positive]] / n
    return np.clip(out, 0.0, 1.0)


def select_nearest_distinct(
    pool_indices: np.ndarray,
    quantiles: np.ndarray,
    lower_fraction: float,
    upper_fraction: float,
    count: int,
) -> np.ndarray:
    if len(pool_indices) < count:
        raise RuntimeError(
            f"Only {len(pool_indices)} candidate Ritz values in a band; need {count}. "
            "Increase --ritz-candidate-steps."
        )
    targets = lower_fraction + (np.arange(count) + 0.5) / count * (
        upper_fraction - lower_fraction
    )
    available = list(int(value) for value in pool_indices)
    chosen: list[int] = []
    for target in targets:
        position = min(
            range(len(available)),
            key=lambda index: abs(float(quantiles[available[index]]) - float(target)),
        )
        chosen.append(available.pop(position))
    return np.asarray(chosen, dtype=np.int64)


def select_ritz_modes(
    precision: SparseVecchiaPrecision,
    args: argparse.Namespace,
    rng: np.random.Generator,
    bands: pd.DataFrame,
    slq_atoms: np.ndarray,
    slq_cumulative: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Construct and select 512 implicit random-start Rayleigh--Ritz vectors."""
    start = rng.choice(np.asarray([-1.0, 1.0]), size=precision.n)
    print(
        f"      random-start Ritz candidates: m={args.ritz_candidate_steps}",
        flush=True,
    )
    run = lanczos_tridiagonal(
        precision.matvec,
        start,
        max_steps=int(args.ritz_candidate_steps),
        overlap_vector=precision.residual,
    )
    values, tridiagonal_vectors = tridiagonal_eigenpairs(run)
    if run.overlap is None:
        raise RuntimeError("internal error: residual overlaps were not retained")
    residual_projection = tridiagonal_vectors.T @ run.overlap
    standardized_energy = values * np.square(residual_projection)
    quantiles = cdf_at(values, slq_atoms, slq_cumulative, precision.n)
    absolute_tail_residual = np.abs(run.tail_beta * tridiagonal_vectors[-1, :])
    relative_tail_residual = absolute_tail_residual / np.maximum(
        np.abs(values), np.finfo(np.float64).eps * float(np.max(values))
    )

    first_edge = float(bands.iloc[0]["precision_upper"])
    second_edge = float(bands.iloc[1]["precision_upper"])
    masks = (
        values <= first_edge,
        (values > first_edge) & (values <= second_edge),
        values > second_edge,
    )
    fraction_bounds = ((0.0, 1.0 / 3.0), (1.0 / 3.0, 2.0 / 3.0), (2.0 / 3.0, 1.0))
    selected_rows: list[pd.DataFrame] = []
    pool_counts: dict[str, int] = {}
    selection_offset = 0
    for band_index, (band, requested, mask, fraction_bound) in enumerate(
        zip(BAND_NAMES, args.band_mode_counts, masks, fraction_bounds), start=1
    ):
        pool = np.flatnonzero(mask)
        pool_counts[band] = int(len(pool))
        selected = select_nearest_distinct(
            pool,
            quantiles,
            fraction_bound[0],
            fraction_bound[1],
            int(requested),
        )
        selected = selected[np.argsort(values[selected])]
        target_quantiles = fraction_bound[0] + (
            np.arange(len(selected)) + 0.5
        ) / len(selected) * (fraction_bound[1] - fraction_bound[0])
        frame = pd.DataFrame(
            {
                "selection_index": selection_offset + np.arange(1, len(selected) + 1),
                "band_index": band_index,
                "band": band,
                "frequency_label": BAND_LABELS[band],
                "band_rank": np.arange(1, len(selected) + 1),
                "candidate_ritz_index": selected,
                "precision_ritz_value": values[selected],
                "implied_covariance_eigenvalue": 1.0 / values[selected],
                "target_spectral_quantile": target_quantiles,
                "spectral_quantile_slq": quantiles[selected],
                "spectral_quantile_absolute_error": np.abs(
                    quantiles[selected] - target_quantiles
                ),
                "residual_projection": residual_projection[selected],
                "standardized_residual_energy": standardized_energy[selected],
                "ritz_tail_residual_absolute": absolute_tail_residual[selected],
                "ritz_tail_residual_relative": relative_tail_residual[selected],
            }
        )
        selected_rows.append(frame)
        selection_offset += len(selected)
    modes = pd.concat(selected_rows, ignore_index=True)
    if len(modes) != 512 or modes["selection_index"].nunique() != 512:
        raise RuntimeError(f"Ritz selection produced {len(modes)} rows, expected 512")
    summary = {
        "ritz_candidate_steps_requested": int(args.ritz_candidate_steps),
        "ritz_candidate_steps_completed": run.steps,
        "ritz_candidate_seconds": run.total_seconds,
        "ritz_candidate_matvec_seconds": run.matvec_seconds,
        "ritz_breakdown": bool(run.breakdown),
        "ritz_tail_beta": run.tail_beta,
        "ritz_candidate_pool_low": pool_counts["low"],
        "ritz_candidate_pool_middle": pool_counts["middle"],
        "ritz_candidate_pool_high": pool_counts["high"],
        "selected_modes": int(len(modes)),
        "selected_mean_energy": float(modes["standardized_residual_energy"].mean()),
        "selected_median_relative_ritz_residual": float(
            modes["ritz_tail_residual_relative"].median()
        ),
        "selected_p95_relative_ritz_residual": float(
            modes["ritz_tail_residual_relative"].quantile(0.95)
        ),
        "selected_p95_spectral_quantile_error": float(
            modes["spectral_quantile_absolute_error"].quantile(0.95)
        ),
        "selected_fraction_within_ritz_tolerance": float(
            np.mean(
                modes["ritz_tail_residual_relative"].to_numpy()
                <= float(args.ritz_relative_residual_tolerance)
            )
        ),
    }
    for band in BAND_NAMES:
        part = modes[modes["band"].eq(band)]
        summary[f"{band}_selected_modes"] = int(len(part))
        summary[f"{band}_mean_energy"] = float(
            part["standardized_residual_energy"].mean()
        )
        summary[f"{band}_median_relative_ritz_residual"] = float(
            part["ritz_tail_residual_relative"].median()
        )
    return modes, summary


def rebuild_precision(
    method: str,
    asset: core.DayAsset,
    fit_row: pd.Series,
    args: argparse.Namespace,
) -> tuple[SparseVecchiaPrecision, dict[str, Any]]:
    fit_args = reference.build_fit_args(args)
    device = core.resolve_device(fit_args)
    # These are the exact M3/Q3 advection seeds used when the stored fit was
    # created.  Reusing them avoids repeating the empirical initializer and
    # guarantees that the adapted conditioning geometry is reconstructed.
    seed = {
        "seed_lat": float(fit_row["init_advec_lat"]),
        "seed_lon": float(fit_row["init_advec_lon"]),
    }
    model = core.build_geometry_model(method, asset, seed, device, fit_args)
    setup_started = time.perf_counter()
    model.precompute_conditioning_sets()
    params = torch.as_tensor(
        core.physical_to_raw(reference.fitted_parameters(fit_row)),
        dtype=core.DTYPE,
        device=device,
    )
    beta = np.asarray(fit_row["gls_beta"], dtype=np.float64)
    native_quadratic = native_gls_quadratic(model, params, beta)
    precision = build_sparse_vecchia_precision(
        model,
        params,
        beta,
        chunk_size=int(args.target_chunk_size),
        coefficient_drop_tolerance=float(args.coefficient_drop_tolerance),
    )
    identity = verify_precision_identity(precision, native_quadratic)
    if identity["relative_error"] > float(args.precision_identity_tolerance):
        raise RuntimeError(f"Sparse precision identity failed: {identity}")
    reconstructed_nll = (
        float(precision.metadata["conditional_logdet_half"])
        + 0.5 * float(identity["sparse_quadratic"])
    ) / precision.n
    stored_nll = float(fit_row["native_nll_per_observation"])
    model_summary = model.cluster_summary()
    summary = {
        "n_full_observations": precision.n,
        "precision_nnz": int(precision.whitener.nnz),
        "precision_csr_mebibytes": float(
            precision.metadata["csr_storage_bytes"] / 2**20
        ),
        "precision_build_seconds": float(precision.metadata["build_s"]),
        "precision_setup_total_seconds": time.perf_counter() - setup_started,
        "native_quadratic": native_quadratic,
        "sparse_quadratic": float(identity["sparse_quadratic"]),
        "precision_identity_relative_error": float(identity["relative_error"]),
        "reconstructed_native_nll_per_observation": reconstructed_nll,
        "reconstructed_minus_stored_nll": reconstructed_nll - stored_nll,
        "model_n_target_blocks": int(model_summary["n_target_blocks"]),
    }
    del model, params
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return precision, summary


def cache_paths(output_root: Path, date: str, method: str) -> dict[str, Path]:
    root = output_root / CACHE_DIR_NAME / date / method
    return {
        "root": root,
        "full_curve": root / "full_eigen_curve.csv",
        "full_summary": root / "full_eigen_summary.json",
        "slq_curve": root / "slq_spectrum_curve.csv",
        "bands": root / "slq_three_bands.csv",
        "modes": root / "selected_512_ritz_modes.csv",
        "summary": root / "lanczos_summary.json",
        "complete": root / "COMPLETE",
    }


def load_cached_method(paths: dict[str, Path]) -> tuple[
    pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]
]:
    required = ("full_curve", "full_summary", "slq_curve", "bands", "modes", "summary")
    if not paths["complete"].is_file() or any(not paths[name].is_file() for name in required):
        raise FileNotFoundError("incomplete diagnostic cache")
    return (
        pd.read_csv(paths["full_curve"]),
        json.loads(paths["full_summary"].read_text(encoding="utf-8")),
        pd.read_csv(paths["slq_curve"]),
        pd.read_csv(paths["bands"]),
        pd.read_csv(paths["modes"]),
        json.loads(paths["summary"].read_text(encoding="utf-8")),
    )


def run_method_diagnostics(
    spec: dict[str, Any],
    method: str,
    asset: core.DayAsset,
    fit_row: pd.Series,
    selected: np.ndarray,
    design: np.ndarray,
    values: np.ndarray,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    paths = cache_paths(args.output_root, spec["date"], method)
    try:
        cached = load_cached_method(paths)
        print(f"    {method}: complete diagnostic cache reused", flush=True)
        return cached
    except FileNotFoundError:
        pass

    print(f"    {method}: 3,200-point full eigendecomposition", flush=True)
    full_curve, full_summary = reference.full_eigen_diagnostic(
        selected,
        design,
        values,
        reference.fitted_parameters(fit_row),
        np.asarray(fit_row["gls_beta"], dtype=np.float64),
        args,
    )
    print(f"    {method}: build full sparse Vecchia precision", flush=True)
    precision, precision_summary = rebuild_precision(method, asset, fit_row, args)
    rng = stable_rng(int(args.random_seed), spec["date"], method)
    print(f"    {method}: estimate spectral CDF and three bands", flush=True)
    slq_curve, bands, slq_summary, slq_atoms, slq_cumulative = slq_spectrum(
        precision, args, rng
    )
    print(f"    {method}: select 170/170/172 implicit Ritz modes", flush=True)
    modes, ritz_summary = select_ritz_modes(
        precision,
        args,
        rng,
        bands,
        slq_atoms,
        slq_cumulative,
    )
    print(
        "      candidate pools low/middle/high="
        f"{ritz_summary['ritz_candidate_pool_low']}/"
        f"{ritz_summary['ritz_candidate_pool_middle']}/"
        f"{ritz_summary['ritz_candidate_pool_high']}; "
        "selected quality pass="
        f"{ritz_summary['selected_fraction_within_ritz_tolerance']:.1%}",
        flush=True,
    )
    full_summary.update(
        {
            "dataset_id": spec["dataset_id"],
            "date": spec["date"],
            "year": spec["year"],
            "method": method,
            "native_nll_per_observation": float(
                fit_row["native_nll_per_observation"]
            ),
        }
    )
    lanczos_summary = {
        "dataset_id": spec["dataset_id"],
        "date": spec["date"],
        "year": spec["year"],
        "method": method,
        "native_nll_per_observation": float(fit_row["native_nll_per_observation"]),
        **precision_summary,
        **slq_summary,
        **ritz_summary,
    }
    for frame in (full_curve, slq_curve, bands, modes):
        frame.insert(0, "method", method)
        frame.insert(0, "date", spec["date"])
        frame.insert(0, "year", spec["year"])
        frame.insert(0, "dataset_id", spec["dataset_id"])

    paths["root"].mkdir(parents=True, exist_ok=True)
    atomic_csv(paths["full_curve"], full_curve)
    reference.write_json(paths["full_summary"], full_summary)
    atomic_csv(paths["slq_curve"], slq_curve)
    atomic_csv(paths["bands"], bands)
    atomic_csv(paths["modes"], modes)
    reference.write_json(paths["summary"], lanczos_summary)
    paths["complete"].write_text("complete\n", encoding="utf-8")
    del precision
    gc.collect()
    return full_curve, full_summary, slq_curve, bands, modes, lanczos_summary


def rolling_mode_energy(frame: pd.DataFrame, window: int = 17) -> np.ndarray:
    return (
        frame.sort_values("selection_index")["standardized_residual_energy"]
        .rolling(window=window, min_periods=max(3, window // 3), center=True)
        .mean()
        .to_numpy(dtype=np.float64)
    )


def shade_frequency_bands(ax: plt.Axes, counts: tuple[int, int, int]) -> None:
    start = 0.5
    for band, count in zip(BAND_NAMES, counts):
        end = start + count
        ax.axvspan(start, end, color=BAND_COLORS[band], alpha=0.42, zorder=0)
        ax.text(
            (start + end) / 2.0,
            0.98,
            band,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            color="0.35",
        )
        start = end


def plot_daily_threeway(
    date: str,
    fits: pd.DataFrame,
    full_curves: dict[str, pd.DataFrame],
    full_summaries: dict[str, dict[str, Any]],
    mode_frames: dict[str, pd.DataFrame],
    lanczos_summaries: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(19.5, 5.5), constrained_layout=True)
    for method in METHODS:
        curve = full_curves[method]
        summary = full_summaries[method]
        axes[0].plot(
            curve["scaled_expected"],
            curve["scaled_cumulative"],
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            linewidth=1.8,
            label=f"{LABELS[method]}: D={summary['D']:.3f}",
        )
    axes[0].plot([0, 1], [0, 1], color="0.3", linestyle="--", linewidth=1.0)
    axes[0].set(
        xlabel="3,200 covariance-mode fraction",
        ylabel="cumulative whitened residual energy / n",
        title="(1) 400 x 8 dense full eigen",
    )
    axes[0].legend(fontsize=8)

    nll = [
        float(fits.loc[fits["method"].eq(method), "native_nll_per_observation"].iloc[0])
        for method in METHODS
    ]
    axes[1].bar(
        np.arange(len(METHODS)),
        nll,
        color=[COLORS[method] for method in METHODS],
        alpha=0.82,
        width=0.62,
    )
    axes[1].set_xticks(np.arange(len(METHODS)), [LABELS[method] for method in METHODS])
    axes[1].tick_params(axis="x", rotation=12)
    axes[1].set(ylabel="NLL / observation", title="(2) Native Vecchia likelihood")
    for index, value in enumerate(nll):
        axes[1].text(index, value, f" {value:.6f}", ha="center", va="bottom", fontsize=8)

    shade_frequency_bands(axes[2], args.band_mode_counts)
    energy_for_limit = []
    for method in METHODS:
        modes = mode_frames[method].sort_values("selection_index")
        smooth = rolling_mode_energy(modes)
        energy_for_limit.extend(smooth[np.isfinite(smooth)].tolist())
        axes[2].scatter(
            modes["selection_index"],
            modes["standardized_residual_energy"],
            s=5,
            alpha=0.10,
            color=COLORS[method],
        )
        summary = lanczos_summaries[method]
        axes[2].plot(
            modes["selection_index"],
            smooth,
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            linewidth=1.8,
            label=(
                f"{LABELS[method]}: mean={summary['selected_mean_energy']:.3f}, "
                f"Ritz<=tol={summary['selected_fraction_within_ritz_tolerance']:.1%}"
            ),
        )
    axes[2].axhline(1.0, color="0.3", linestyle="--", linewidth=1.0)
    if energy_for_limit:
        axes[2].set_ylim(0.0, max(1.5, float(np.nanquantile(energy_for_limit, 0.98)) * 1.25))
    axes[2].set(
        xlim=(0.5, 512.5),
        xlabel="512 selected Ritz modes (low -> high precision; frequency proxy)",
        ylabel="standardized residual energy; 17-mode mean",
        title="(3) Full-data SLQ bands + random-start Ritz",
    )
    axes[2].legend(fontsize=7.5, loc="upper right")
    for axis in axes:
        axis.grid(alpha=0.20)
    fig.suptitle(f"Real {date}: adapted vs fixed Vecchia, three-way comparison", fontsize=14)
    output = args.output_root / DAILY_DIR_NAME / f"{date}_threeway.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def annotate_metadata(
    frame: pd.DataFrame, spec: dict[str, Any], method: str
) -> pd.DataFrame:
    out = frame.copy()
    for column, value in reversed(
        (
            ("dataset_id", spec["dataset_id"]),
            ("year", spec["year"]),
            ("date", spec["date"]),
            ("method", method),
        )
    ):
        if column not in out.columns:
            out.insert(0, column, value)
    return out


def aggregate_outputs(
    specs: list[dict[str, Any]],
    records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full_curves: list[pd.DataFrame] = []
    full_summaries: list[dict[str, Any]] = []
    slq_curves: list[pd.DataFrame] = []
    bands: list[pd.DataFrame] = []
    modes: list[pd.DataFrame] = []
    lanczos_summaries: list[dict[str, Any]] = []
    for spec in specs:
        for method in METHODS:
            paths = cache_paths(args.output_root, spec["date"], method)
            full_curve, full_summary, slq_curve, band, mode, lanczos_summary = (
                load_cached_method(paths)
            )
            full_curves.append(annotate_metadata(full_curve, spec, method))
            full_summaries.append(full_summary)
            slq_curves.append(annotate_metadata(slq_curve, spec, method))
            bands.append(annotate_metadata(band, spec, method))
            modes.append(annotate_metadata(mode, spec, method))
            lanczos_summaries.append(lanczos_summary)
    full_curve_frame = pd.concat(full_curves, ignore_index=True)
    full_summary_frame = pd.DataFrame(full_summaries)
    slq_curve_frame = pd.concat(slq_curves, ignore_index=True)
    band_frame = pd.concat(bands, ignore_index=True)
    mode_frame = pd.concat(modes, ignore_index=True)
    lanczos_summary_frame = pd.DataFrame(lanczos_summaries)
    atomic_csv(args.output_root / "daily_full_eigen_curves.csv", full_curve_frame)
    atomic_csv(args.output_root / "daily_full_eigen_metrics.csv", full_summary_frame)
    atomic_csv(args.output_root / "daily_slq_spectrum_curves.csv", slq_curve_frame)
    atomic_csv(args.output_root / "daily_slq_three_band_boundaries.csv", band_frame)
    atomic_csv(args.output_root / "daily_selected_512_ritz_modes.csv", mode_frame)
    atomic_csv(args.output_root / "daily_lanczos512_metrics.csv", lanczos_summary_frame)
    reference.persist_fit_results(records, args.output_root)
    return (
        full_curve_frame,
        full_summary_frame,
        band_frame,
        mode_frame,
        lanczos_summary_frame,
    )


def plot_monthly_average_threeway(
    year: int,
    records: list[dict[str, Any]],
    full_curves: pd.DataFrame,
    full_summaries: pd.DataFrame,
    modes: pd.DataFrame,
    lanczos_summaries: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    expected_days = 30 if year == 2024 else 29
    fig, axes = plt.subplots(1, 3, figsize=(19.5, 5.5), constrained_layout=True)
    for method in METHODS:
        subset = full_curves[
            full_curves["year"].eq(year) & full_curves["method"].eq(method)
        ]
        if subset["dataset_id"].nunique() != expected_days:
            raise RuntimeError(f"Incomplete {year} {method} full-eigen curves")
        sampled = []
        for _, daily in subset.groupby("dataset_id"):
            sampled.append(reference.resample_curve(daily, 1000))
        mean_curve = pd.concat(sampled).groupby("fraction", as_index=False).agg(
            mean_energy=("scaled_cumulative", "mean")
        )
        metric = full_summaries[
            full_summaries["year"].eq(year) & full_summaries["method"].eq(method)
        ]
        axes[0].plot(
            mean_curve["fraction"],
            mean_curve["mean_energy"],
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            linewidth=2.0,
            label=f"{LABELS[method]}: mean D={metric['D'].mean():.3f}",
        )
    axes[0].plot([0, 1], [0, 1], color="0.3", linestyle="--", linewidth=1.0)
    axes[0].set(
        xlabel="3,200 covariance-mode fraction",
        ylabel="monthly mean cumulative energy / n",
        title="(1) Dense full-eigen monthly mean",
    )
    axes[0].legend(fontsize=8)

    fit_frame = pd.DataFrame(records)
    fit_frame = fit_frame[fit_frame["year"].eq(year)]
    nll_means = []
    nll_sds = []
    for method in METHODS:
        values = fit_frame.loc[
            fit_frame["method"].eq(method), "native_nll_per_observation"
        ].astype(float)
        if len(values) != expected_days:
            raise RuntimeError(f"Incomplete {year} {method} likelihood rows")
        nll_means.append(float(values.mean()))
        nll_sds.append(float(values.std(ddof=1)))
    axes[1].bar(
        np.arange(len(METHODS)),
        nll_means,
        yerr=nll_sds,
        capsize=4,
        color=[COLORS[method] for method in METHODS],
        alpha=0.82,
        width=0.62,
    )
    axes[1].set_xticks(np.arange(len(METHODS)), [LABELS[method] for method in METHODS])
    axes[1].tick_params(axis="x", rotation=12)
    axes[1].set(
        ylabel="mean NLL / observation (error bar: daily SD)",
        title="(2) Native likelihood monthly mean",
    )

    shade_frequency_bands(axes[2], args.band_mode_counts)
    monthly_limit_values = []
    for method in METHODS:
        subset = modes[modes["year"].eq(year) & modes["method"].eq(method)]
        if subset["dataset_id"].nunique() != expected_days:
            raise RuntimeError(f"Incomplete {year} {method} Ritz modes")
        grouped = subset.groupby("selection_index", as_index=False).agg(
            mean_energy=("standardized_residual_energy", "mean"),
            se_energy=("standardized_residual_energy", "sem"),
        )
        smooth_mean = (
            grouped["mean_energy"].rolling(17, min_periods=5, center=True).mean()
        )
        smooth_se = grouped["se_energy"].rolling(17, min_periods=5, center=True).mean()
        monthly_limit_values.extend(smooth_mean.dropna().tolist())
        summary = lanczos_summaries[
            lanczos_summaries["year"].eq(year)
            & lanczos_summaries["method"].eq(method)
        ]
        axes[2].plot(
            grouped["selection_index"],
            smooth_mean,
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            linewidth=2.0,
            label=(
                f"{LABELS[method]}: mean quality pass="
                f"{summary['selected_fraction_within_ritz_tolerance'].mean():.1%}"
            ),
        )
        axes[2].fill_between(
            grouped["selection_index"].to_numpy(dtype=float),
            np.maximum(0.0, (smooth_mean - smooth_se).to_numpy(dtype=float)),
            (smooth_mean + smooth_se).to_numpy(dtype=float),
            color=COLORS[method],
            alpha=0.12,
            linewidth=0,
        )
    axes[2].axhline(1.0, color="0.3", linestyle="--", linewidth=1.0)
    if monthly_limit_values:
        axes[2].set_ylim(
            0.0,
            max(1.5, float(np.nanquantile(monthly_limit_values, 0.98)) * 1.25),
        )
    axes[2].set(
        xlim=(0.5, 512.5),
        xlabel="512 selected Ritz modes (low -> high precision; frequency proxy)",
        ylabel="monthly mean standardized energy; 17-mode mean",
        title="(3) Full-data SLQ/Ritz monthly mean",
    )
    axes[2].legend(fontsize=7.5)
    for axis in axes:
        axis.grid(alpha=0.20)
    exclusion = "; 2025-07-24 excluded" if year == 2025 else ""
    fig.suptitle(
        f"Real {year} July: {expected_days}-day adapted vs fixed mean{exclusion}",
        fontsize=14,
    )
    fig.savefig(
        args.output_root / f"{year}_07_monthly_average_threeway.png",
        dpi=190,
        bbox_inches="tight",
    )
    plt.close(fig)


def write_run_config(args: argparse.Namespace, specs: list[dict[str, Any]]) -> None:
    path = args.output_root / "run_config_threeway_slq512.json"
    signature = {
        "dates": [spec["date"] for spec in specs],
        "methods": list(METHODS),
        "smooth": args.smooth,
        "target_chunk_size": args.target_chunk_size,
        "points_per_hour": args.points_per_hour,
        "lbfgs_lr": args.lbfgs_lr,
        "lbfgs_steps": args.lbfgs_steps,
        "lbfgs_eval": args.lbfgs_eval,
        "lbfgs_history": args.lbfgs_history,
        "grad_tol": args.grad_tol,
        "cov_jitter": args.cov_jitter,
        "slq_probes": args.slq_probes,
        "slq_steps": args.slq_steps,
        "ritz_candidate_steps": args.ritz_candidate_steps,
        "band_mode_counts": list(args.band_mode_counts),
        "spectrum_grid": args.spectrum_grid,
        "random_seed": args.random_seed,
        "ritz_relative_residual_tolerance": args.ritz_relative_residual_tolerance,
        "precision_identity_tolerance": args.precision_identity_tolerance,
        "coefficient_drop_tolerance": args.coefficient_drop_tolerance,
    }
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("configuration_signature") != signature:
            raise ValueError(
                f"{path} was created with different fit/SLQ settings. "
                "Use a new --output-root rather than mixing diagnostic caches."
            )
        created = previous.get("created", datetime.now().isoformat(timespec="seconds"))
    else:
        created = datetime.now().isoformat(timespec="seconds")
    reference.write_json(
        path,
        {
            "created": created,
            "last_started": datetime.now().isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "gpu": torch.cuda.get_device_name(0),
            "configuration_signature": signature,
            "dates": [spec["date"] for spec in specs],
            "n_usable_dates": len(specs),
            "nominal_window_dates": 60,
            "excluded_incomplete_date": "2025-07-24",
            "methods": METHODS,
            "comparison_axes": [
                "400x8 dense full eigendecomposition",
                "native Vecchia NLL per observation",
                "full-data sparse-precision SLQ plus 512 implicit Ritz modes",
            ],
            "frequency_proxy_convention": (
                "Ascending precision eigenvalue runs from large implied covariance "
                "variance to small implied covariance variance. For smooth covariance "
                "models this is interpreted as low-to-high frequency, but on an "
                "irregular space-time Vecchia graph it is a spectral proxy rather "
                "than an exact Fourier frequency."
            ),
            "slq_probes": args.slq_probes,
            "slq_steps": args.slq_steps,
            "ritz_candidate_steps": args.ritz_candidate_steps,
            "selected_modes_by_band": dict(zip(BAND_NAMES, args.band_mode_counts)),
            "ritz_relative_residual_tolerance": args.ritz_relative_residual_tolerance,
            "important_limitation": (
                "The 512 objects are selected implicit random-start Rayleigh--Ritz "
                "vectors. They are not claimed to be 512 converged exact eigenvectors; "
                "tail Ritz residual estimates quantify approximation quality."
            ),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real-data-root", type=Path, default=Path("/home/jl2815/tco/data")
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/home/jl2815/tco/exercise_output/summer/"
            "vecchia_real59_adapted_fixed_threeway_slq512_lag643"
        ),
    )
    parser.add_argument("--lat-range", default="-3,2")
    parser.add_argument("--lon-range", default="121,131")
    parser.add_argument("--smooth", type=float, default=0.5, choices=(0.5,))
    parser.add_argument("--target-chunk-size", type=int, default=256)
    parser.add_argument("--points-per-hour", type=int, default=400)
    parser.add_argument("--lbfgs-lr", type=float, default=1.0)
    parser.add_argument("--lbfgs-steps", type=int, default=5)
    parser.add_argument("--lbfgs-eval", type=int, default=20)
    parser.add_argument("--lbfgs-history", type=int, default=40)
    parser.add_argument("--grad-tol", type=float, default=1e-5)
    parser.add_argument("--cov-jitter", type=float, default=1e-8)
    parser.add_argument("--suppress-fit-prints", action="store_true")
    parser.add_argument("--slq-probes", type=int, default=8)
    parser.add_argument("--slq-steps", type=int, default=256)
    parser.add_argument("--ritz-candidate-steps", type=int, default=1536)
    parser.add_argument(
        "--band-mode-counts", type=parse_band_counts, default=parse_band_counts("170,170,172")
    )
    parser.add_argument("--spectrum-grid", type=int, default=400)
    parser.add_argument("--random-seed", type=int, default=20260907)
    parser.add_argument("--ritz-relative-residual-tolerance", type=float, default=0.05)
    parser.add_argument("--precision-identity-tolerance", type=float, default=1e-8)
    parser.add_argument("--coefficient-drop-tolerance", type=float, default=0.0)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.points_per_hour != 400:
        raise ValueError("This comparison requires exactly 400 points per hour")
    if args.slq_probes < 2 or args.slq_steps < 16:
        raise ValueError("Use at least 2 SLQ probes and 16 SLQ steps")
    if args.ritz_candidate_steps < 512:
        raise ValueError("--ritz-candidate-steps must be at least 512")
    if args.spectrum_grid < 32:
        raise ValueError("--spectrum-grid must be at least 32")
    if args.coefficient_drop_tolerance < 0.0:
        raise ValueError("--coefficient-drop-tolerance must be nonnegative")
    if args.ritz_relative_residual_tolerance <= 0.0:
        raise ValueError("--ritz-relative-residual-tolerance must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the 59-day fit and precision build")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / DAILY_DIR_NAME).mkdir(parents=True, exist_ok=True)
    specs = reference.date_specs()
    write_run_config(args, specs)
    records = reference.load_fit_results(args.output_root)
    selected_ids = {spec["dataset_id"] for spec in specs}
    unexpected = [
        row
        for row in records
        if str(row["dataset_id"]) not in selected_ids
        or str(row["method"]) not in METHODS
    ]
    if unexpected:
        raise ValueError("Fit checkpoint contains rows outside this three-way run")

    workflow_started = time.perf_counter()
    for position, spec in enumerate(specs, start=1):
        print(f"\n[{position}/{len(specs)}] {spec['date']}", flush=True)
        asset, fits = reference.load_or_fit_day(spec, records, args)
        selected, _ = reference.point_maxmin_sample(asset, args.points_per_hour)
        design, values, _ = reference.mean_design(selected)
        full_curves: dict[str, pd.DataFrame] = {}
        full_summaries: dict[str, dict[str, Any]] = {}
        modes: dict[str, pd.DataFrame] = {}
        lanczos_summaries: dict[str, dict[str, Any]] = {}
        for method in METHODS:
            fit_rows = fits[fits["method"].eq(method)]
            if len(fit_rows) != 1:
                raise RuntimeError(f"Expected one {method} fit for {spec['date']}")
            full_curve, full_summary, _, _, mode_frame, lanczos_summary = (
                run_method_diagnostics(
                    spec,
                    method,
                    asset,
                    fit_rows.iloc[0],
                    selected,
                    design,
                    values,
                    args,
                )
            )
            full_curves[method] = full_curve
            full_summaries[method] = full_summary
            modes[method] = mode_frame
            lanczos_summaries[method] = lanczos_summary
        plot_daily_threeway(
            spec["date"],
            fits,
            full_curves,
            full_summaries,
            modes,
            lanczos_summaries,
            args,
        )
        reference.persist_fit_results(records, args.output_root)
        del asset, fits, selected, design, values, full_curves, modes
        gc.collect()
        torch.cuda.empty_cache()

    full_curves, full_summaries, _, modes, lanczos_summaries = aggregate_outputs(
        specs, records, args
    )
    reference.plot_native_nll(records, args.output_root)
    for year in (2024, 2025):
        plot_monthly_average_threeway(
            year,
            records,
            full_curves,
            full_summaries,
            modes,
            lanczos_summaries,
            args,
        )
    reference.write_json(
        args.output_root / "RUN_COMPLETE.json",
        {
            "completed": datetime.now().isoformat(timespec="seconds"),
            "elapsed_seconds": time.perf_counter() - workflow_started,
            "n_dates": len(specs),
            "n_methods": len(METHODS),
            "selected_modes_per_date_method": 512,
        },
    )
    print(f"Complete: outputs in {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
