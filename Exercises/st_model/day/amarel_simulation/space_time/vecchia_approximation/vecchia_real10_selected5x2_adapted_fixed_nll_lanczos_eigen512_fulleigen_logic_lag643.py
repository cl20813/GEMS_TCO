#!/usr/bin/env python3
"""Full-eigen-logic residual diagnostic with 512 approximate Lanczos eigenpairs.

The experiment analyzes July 3, 5, 7, 12, and 15 in 2024 and 2025.  For each
date and each fitted Vecchia method it reports native full-data NLL and applies
the *same statistic* as the existing dense full eigendecomposition diagnostic:

    Omega u_j = omega_j u_j,
    z_j = sqrt(omega_j) u_j' r,
    e_j = z_j^2,
    x_k = k / K,
    y_k = sum_{j<=k} e_j / K,             K = 512.

For an exact fitted covariance with known parameters, e_j is chi-square with
one degree of freedom and E[y_k] = x_k.  The only computational substitution
is that implicitly restarted Lanczos (scipy.sparse.linalg.eigsh) obtains 512
explicit approximate eigenvectors of the full sparse Vecchia precision:

* 170 smallest precision eigenpairs (low-frequency proxy), using the largest
  eigenpairs of Omega^{-1};
* 170 eigenpairs nearest the SLQ-estimated spectral median (middle proxy),
  using a folded operator -(Omega-sigma I)^2 followed by Rayleigh--Ritz
  diagonalization in the returned subspace; and
* 172 largest precision eigenpairs (high-frequency proxy).

SLQ is used only to locate spectral thirds and the median.  It never computes
residual energy and it never replaces eigenvectors.  Solver starts are merely
numerical initial values: there is one eigenpair set and no averaging across
random starts.  Eigenpair residuals are reported as approximation-quality
metrics rather than used as an all-512 hard gate.  Nonfinite eigenpairs,
insufficient in-band candidates, or a serious combined orthogonality failure
still stop the run because those conditions invalidate the diagnostic.

The 512 modes preserve the full-eigen diagnostic logic, but they are a selected
512-direction diagnostic rather than the unavailable all-n eigenbasis.  As in
the dense fitted-data diagnostic, parameter estimation makes the chi-square
reference approximate rather than exact.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import socket
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
import scipy.linalg
import scipy.sparse.linalg
import scipy.stats
import torch


HERE = Path(__file__).resolve().parent
SUBMIT_DIR = Path(os.environ.get("SLURM_SUBMIT_DIR", str(HERE))).resolve()
AMAREL_ROOT = Path("/home/jl2815/tco")
LOCAL_SRC = Path("/Users/joonwonlee/Documents/GEMS_TCO-1/src")
INTERACTION_DIR = HERE.parent / "interaction_diagnostic"
for candidate in (AMAREL_ROOT, SUBMIT_DIR, HERE, LOCAL_SRC, INTERACTION_DIR):
    if candidate.exists() and str(candidate) not in os.sys.path:
        os.sys.path.insert(0, str(candidate))

import vecchia_real59_adapted_fixed_threeway_slq512_lag643 as engine  # noqa: E402


core = engine.core
reference = engine.reference
METHODS = engine.METHODS
COLORS = engine.COLORS
LINESTYLES = engine.LINESTYLES
LABELS = engine.LABELS
BANDS = engine.BAND_NAMES
BAND_LABELS = engine.BAND_LABELS
BAND_COLORS = engine.BAND_COLORS
TEST_DAYS = (3, 5, 7, 12, 15)
N_MODES = 512
DAILY_DIR = "daily_subplots"
DAILY_SEGMENT_DIR = "daily_band_segments"
CACHE_DIR = ".eigen512_cache"


def selected_date_specs() -> list[dict[str, Any]]:
    available = {spec["date"]: spec for spec in reference.date_specs()}
    requested = [
        f"{year}-07-{day:02d}" for year in (2024, 2025) for day in TEST_DAYS
    ]
    missing = [date for date in requested if date not in available]
    if missing:
        raise RuntimeError(f"Reference date metadata is missing {missing}")
    return [available[date] for date in requested]


def common_rng(seed: int, date: str, stream: int) -> np.random.Generator:
    sequence = np.random.SeedSequence(
        [int(seed), int(date.replace("-", "")), int(stream)]
    )
    return np.random.default_rng(sequence)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def annotate(
    frame: pd.DataFrame, spec: dict[str, Any], method: str
) -> pd.DataFrame:
    out = frame.copy()
    metadata = {
        "dataset_id": spec["dataset_id"],
        "year": spec["year"],
        "date": spec["date"],
        "method": method,
    }
    for column, value in metadata.items():
        out[column] = value
    leading = list(metadata)
    return out[leading + [column for column in out if column not in leading]]


def cache_paths(output_root: Path, date: str, method: str) -> dict[str, Path]:
    root = output_root / CACHE_DIR / date / method
    return {
        "root": root,
        "slq": root / "slq_spectrum_curve.csv",
        "bands": root / "slq_spectral_boundaries.csv",
        "modes": root / "approximate_eigen512_modes.csv",
        "summary": root / "eigen512_summary.json",
        "vectors": root / "approximate_eigenvectors_float32.npy",
        "complete": root / "COMPLETE",
    }


def load_cached_method(
    paths: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    required = ("slq", "bands", "modes", "summary")
    if not paths["complete"].is_file() or any(
        not paths[name].is_file() for name in required
    ):
        raise FileNotFoundError("incomplete eigen512 cache")
    return (
        pd.read_csv(paths["slq"]),
        pd.read_csv(paths["bands"]),
        pd.read_csv(paths["modes"]),
        json.loads(paths["summary"].read_text(encoding="utf-8")),
    )


def slq_quantile(
    fraction: float, atoms: np.ndarray, cumulative: np.ndarray, n: int
) -> float:
    position = int(np.searchsorted(cumulative, fraction * n, side="left"))
    return float(atoms[min(max(position, 0), len(atoms) - 1)])


def linear_operator(
    n: int,
    matvec: Callable[[np.ndarray], np.ndarray],
    matmat: Callable[[np.ndarray], np.ndarray] | None = None,
) -> scipy.sparse.linalg.LinearOperator:
    return scipy.sparse.linalg.LinearOperator(
        shape=(n, n),
        matvec=matvec,
        matmat=matmat,
        rmatvec=matvec,
        dtype=np.dtype(np.float64),
    )


def run_eigsh(
    operator: scipy.sparse.linalg.LinearOperator,
    k: int,
    which: str,
    v0: np.ndarray,
    args: argparse.Namespace,
    label: str,
    minimum_returned: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    ncv = min(
        operator.shape[0] - 1,
        max(k + 2, int(args.eigsh_ncv_factor * k) + int(args.eigsh_ncv_extra)),
    )
    if ncv <= k:
        raise ValueError(f"{label}: ncv={ncv} must exceed k={k}")
    started = time.perf_counter()
    converged_exception = False
    try:
        values, vectors = scipy.sparse.linalg.eigsh(
            operator,
            k=k,
            which=which,
            v0=np.asarray(v0, dtype=np.float64),
            ncv=ncv,
            tol=float(args.eigsh_tolerance),
            maxiter=int(args.eigsh_maxiter),
            return_eigenvectors=True,
        )
    except scipy.sparse.linalg.ArpackNoConvergence as error:
        converged_exception = True
        values = error.eigenvalues
        vectors = error.eigenvectors
        found = 0 if values is None else len(values)
        if values is None or vectors is None or found < minimum_returned:
            raise RuntimeError(
                f"{label}: ARPACK returned only {found} candidates, fewer than "
                f"the minimum {minimum_returned}; increase --eigsh-maxiter or "
                "reduce the requested mode count"
            ) from error
    elapsed = time.perf_counter() - started
    if vectors.ndim != 2 or vectors.shape[0] != operator.shape[0]:
        raise RuntimeError(f"{label}: unexpected eigsh vector shape {vectors.shape}")
    return values, vectors, {
        "label": label,
        "requested_candidates": int(k),
        "returned_candidates": int(len(values)),
        "ncv": int(ncv),
        "seconds": float(elapsed),
        "arpack_no_convergence_recovered": bool(converged_exception),
    }


def orthonormalize(vectors: np.ndarray) -> tuple[np.ndarray, float]:
    gram = np.asarray(vectors.T @ vectors, dtype=np.float64)
    error_before = float(np.max(np.abs(gram - np.eye(gram.shape[0]))))
    if error_before <= 5e-10:
        return np.asarray(vectors, dtype=np.float64, order="F"), error_before
    eigenvalues, eigenvectors = scipy.linalg.eigh(
        0.5 * (gram + gram.T), check_finite=False
    )
    if float(eigenvalues.min()) <= 1e-10:
        raise RuntimeError(
            f"Candidate eigenvector subspace is rank deficient: min Gram={eigenvalues.min()}"
        )
    transform = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
    return np.asarray(vectors @ transform, order="F"), error_before


def refine_precision_subspace(
    precision: Any,
    candidates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    basis, orthogonality_before = orthonormalize(candidates)
    applied = precision.matmat(basis)
    projected = basis.T @ applied
    projected = 0.5 * (projected + projected.T)
    values, rotation = scipy.linalg.eigh(projected, check_finite=False)
    vectors = np.asarray(basis @ rotation, order="F")
    applied_vectors = np.asarray(applied @ rotation, order="F")
    residual = applied_vectors - vectors * values.reshape(1, -1)
    residual_norm = np.linalg.norm(residual, axis=0)
    scale = np.maximum(np.abs(values), np.finfo(np.float64).tiny)
    relative_residual = residual_norm / scale
    gram = vectors.T @ vectors
    orthogonality_after = float(np.max(np.abs(gram - np.eye(len(values)))))
    return values, vectors, relative_residual, {
        "candidate_orthogonality_error": orthogonality_before,
        "refined_orthogonality_error": orthogonality_after,
    }


def choose_approximate(
    region: str,
    values: np.ndarray,
    vectors: np.ndarray,
    relative_residual: np.ndarray,
    requested: int,
    edge1: float,
    edge2: float,
    median: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite = (
        np.isfinite(values)
        & (values > 0.0)
        & np.isfinite(relative_residual)
    )
    if region == "low":
        eligible = np.flatnonzero(finite & (values <= edge1))
        order = eligible[np.argsort(values[eligible])]
    elif region == "middle":
        eligible = np.flatnonzero(finite & (values > edge1) & (values <= edge2))
        order = eligible[np.argsort(np.abs(values[eligible] - median))]
    elif region == "high":
        eligible = np.flatnonzero(finite & (values > edge2))
        order = eligible[np.argsort(values[eligible])[::-1]]
    else:
        raise ValueError(region)
    if len(order) < requested:
        raise RuntimeError(
            f"{region}: only {len(order)} finite positive in-band eigenpairs for "
            f"{requested} requested. Increase --eigen-oversample or "
            "--eigsh-maxiter."
        )
    chosen = order[:requested]
    chosen = chosen[np.argsort(values[chosen])]
    return values[chosen], np.asarray(vectors[:, chosen], order="F"), relative_residual[chosen]


def solve_selected_eigenpairs(
    precision: Any,
    args: argparse.Namespace,
    rng: np.random.Generator,
    edge1: float,
    edge2: float,
    median: float,
    spectral_min: float,
    spectral_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    counts = dict(zip(BANDS, args.band_mode_counts))
    expected_modes = int(sum(counts.values()))
    n = precision.n
    base_start = rng.choice(np.asarray([-1.0, 1.0]), size=n).astype(np.float64)
    base_start /= np.linalg.norm(base_start)
    omega = linear_operator(n, precision.matvec, precision.matmat)
    summaries: list[dict[str, Any]] = []
    selected: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    low_k = int(counts["low"] + args.eigen_oversample)
    if getattr(args, "low_solver", "inverse") == "direct":
        # Laptop fallback: avoids the very costly triangular solve at every
        # ARPACK iteration.  It uses only Omega matvecs, at the cost of slower
        # convergence of the smallest eigenvalues.
        print("      low: direct smallest-precision Lanczos (no sparse LU)", flush=True)
        _, raw_low, run_summary = run_eigsh(
            omega, low_k, "SA", base_start, args, "low_direct_lanczos",
            minimum_returned=int(counts["low"]),
        )
        run_summary["sparse_whitener_lu_seconds"] = 0.0
    else:
        # Low precision modes are largest covariance modes.  One sparse LU of B
        # applies Omega^{-1}=B^{-1}B^{-T>; Omega itself is never materialized.
        print(
            "      low: factor sparse whitener and solve largest Omega^{-1} modes",
            flush=True,
        )
        factor_started = time.perf_counter()
        whitener_lu = scipy.sparse.linalg.splu(precision.whitener.tocsc())
        factor_seconds = time.perf_counter() - factor_started

        def covariance_matvec(vector: np.ndarray) -> np.ndarray:
            intermediate = whitener_lu.solve(
                np.asarray(vector, dtype=np.float64), trans="T"
            )
            return np.asarray(whitener_lu.solve(intermediate), dtype=np.float64)

        def covariance_matmat(vectors: np.ndarray) -> np.ndarray:
            intermediate = whitener_lu.solve(
                np.asarray(vectors, dtype=np.float64), trans="T"
            )
            return np.asarray(whitener_lu.solve(intermediate), dtype=np.float64)

        covariance = linear_operator(n, covariance_matvec, covariance_matmat)
        _, raw_low, run_summary = run_eigsh(
            covariance, low_k, "LA", base_start, args, "low_inverse_lanczos",
            minimum_returned=int(counts["low"]),
        )
        run_summary["sparse_whitener_lu_seconds"] = float(factor_seconds)
        del covariance, whitener_lu
    low_values, low_vectors, low_residual, refinement = refine_precision_subspace(
        precision, raw_low
    )
    selected["low"] = choose_approximate(
        "low", low_values, low_vectors, low_residual, int(counts["low"]),
        edge1, edge2, median
    )
    run_summary.update(refinement)
    summaries.append(run_summary)
    del raw_low, low_values, low_vectors, low_residual
    gc.collect()

    # A folded spectrum maps eigenvalues nearest sigma to the largest (closest
    # to zero) eigenvalues of -(Omega-sigma I)^2.  Diagonalizing Omega in that
    # subspace separates the two sides of sigma and yields Omega Ritz vectors.
    print("      middle: folded-spectrum Lanczos around SLQ median", flush=True)
    fold_scale = max(
        abs(float(spectral_min) - median),
        abs(float(spectral_max) - median),
        abs(median),
        np.finfo(np.float64).tiny,
    )

    def folded_matvec(vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float64)
        shifted = precision.matvec(vector) - median * vector
        twice = precision.matvec(shifted) - median * shifted
        return -twice / (fold_scale * fold_scale)

    def folded_matmat(vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=np.float64)
        shifted = precision.matmat(vectors) - median * vectors
        twice = precision.matmat(shifted) - median * shifted
        return -twice / (fold_scale * fold_scale)

    folded = linear_operator(n, folded_matvec, folded_matmat)
    middle_k = int(counts["middle"] + args.eigen_oversample)
    _, raw_middle, run_summary = run_eigsh(
        folded, middle_k, "LA", base_start, args, "middle_folded_lanczos",
        minimum_returned=int(counts["middle"]),
    )
    middle_values, middle_vectors, middle_residual, refinement = (
        refine_precision_subspace(precision, raw_middle)
    )
    selected["middle"] = choose_approximate(
        "middle", middle_values, middle_vectors, middle_residual,
        int(counts["middle"]), edge1, edge2, median
    )
    run_summary.update(refinement)
    run_summary["fold_center_precision"] = float(median)
    run_summary["fold_scale"] = float(fold_scale)
    summaries.append(run_summary)
    del folded, raw_middle, middle_values, middle_vectors, middle_residual
    gc.collect()

    print("      high: direct largest-precision Lanczos", flush=True)
    high_k = int(counts["high"] + args.eigen_oversample)
    _, raw_high, run_summary = run_eigsh(
        omega, high_k, "LA", base_start, args, "high_direct_lanczos",
        minimum_returned=int(counts["high"]),
    )
    high_values, high_vectors, high_residual, refinement = refine_precision_subspace(
        precision, raw_high
    )
    selected["high"] = choose_approximate(
        "high", high_values, high_vectors, high_residual, int(counts["high"]),
        edge1, edge2, median
    )
    run_summary.update(refinement)
    summaries.append(run_summary)
    del raw_high, high_values, high_vectors, high_residual
    gc.collect()

    values = np.concatenate([selected[band][0] for band in BANDS])
    vectors = np.asfortranarray(np.column_stack([selected[band][1] for band in BANDS]))
    residuals = np.concatenate([selected[band][2] for band in BANDS])
    order = np.argsort(values)
    values = values[order]
    vectors = np.asfortranarray(vectors[:, order])
    residuals = residuals[order]
    if len(values) != expected_modes:
        raise RuntimeError(
            f"Selected {len(values)} eigenpairs, expected {expected_modes}"
        )
    gram = vectors.T @ vectors
    global_orthogonality = float(
        np.max(np.abs(gram - np.eye(expected_modes)))
    )
    if global_orthogonality > float(args.eigenvector_orthogonality_tolerance):
        raise RuntimeError(
            f"Combined eigenvectors fail orthogonality: max|U'U-I|="
            f"{global_orthogonality:.3e} > "
            f"{args.eigenvector_orthogonality_tolerance:.3e}"
        )
    summaries.append(
        {
            "label": "combined_selected_eigenvectors",
            "selected_modes": expected_modes,
            "max_orthogonality_error": global_orthogonality,
            "max_relative_eigenpair_residual": float(np.max(residuals)),
            "p95_relative_eigenpair_residual": float(np.quantile(residuals, 0.95)),
            "fraction_within_quality_tolerance": float(
                np.mean(residuals <= args.eigenpair_relative_residual_tolerance)
            ),
        }
    )
    return values, vectors, residuals, summaries


def full_eigen_logic_curve(
    precision: Any,
    values: np.ndarray,
    vectors: np.ndarray,
    relative_residual: np.ndarray,
    bands: pd.DataFrame,
    slq_atoms: np.ndarray,
    slq_cumulative: np.ndarray,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    projection = vectors.T @ np.asarray(precision.residual, dtype=np.float64)
    scores = np.sqrt(values) * projection
    squared = np.square(scores)
    cumulative = np.cumsum(squared)
    n_modes = int(len(values))
    index = np.arange(1, n_modes + 1, dtype=np.int64)
    x = index / float(n_modes)
    y = cumulative / float(n_modes)
    lower = scipy.stats.chi2.ppf(0.025, index) / float(n_modes)
    upper = scipy.stats.chi2.ppf(0.975, index) / float(n_modes)
    counts = tuple(int(value) for value in args.band_mode_counts)
    labels = np.repeat(np.asarray(BANDS, dtype=object), counts)
    band_rank = np.concatenate(
        [np.arange(1, count + 1, dtype=np.int64) for count in counts]
    )
    spectral_quantiles = engine.cdf_at(
        values, slq_atoms, slq_cumulative, precision.n
    )
    curve = pd.DataFrame(
        {
            "index": index,
            "scaled_expected": x,
            "scaled_cumulative": y,
            "precision_eigenvalue": values,
            "implied_covariance_eigenvalue": 1.0 / values,
            "residual_projection": projection,
            "standardized_score": scores,
            "squared_score": squared,
            "chi_square_pointwise_lower": lower,
            "chi_square_pointwise_upper": upper,
            "relative_eigenpair_residual": relative_residual,
            "slq_spectral_quantile": spectral_quantiles,
            "band": labels,
            "band_rank": band_rank,
        }
    )
    d_stat = float(
        np.max(np.abs(cumulative - index)) / math.sqrt(2.0 * n_modes)
    )
    summary: dict[str, Any] = {
        "selected_eigenpairs": n_modes,
        "mean_y2": float(y[-1]),
        "D": d_stat,
        "max_abs_scaled_departure": float(np.max(np.abs(y - x))),
        "min_precision_eigenvalue": float(values[0]),
        "max_precision_eigenvalue": float(values[-1]),
        "max_relative_eigenpair_residual": float(np.max(relative_residual)),
        "median_relative_eigenpair_residual": float(
            np.median(relative_residual)
        ),
        "p95_relative_eigenpair_residual": float(
            np.quantile(relative_residual, 0.95)
        ),
        "eigenpair_quality_tolerance": float(
            args.eigenpair_relative_residual_tolerance
        ),
        "fraction_within_eigenpair_quality_tolerance": float(
            np.mean(relative_residual <= args.eigenpair_relative_residual_tolerance)
        ),
    }
    offset = 0
    for band, count in zip(BANDS, counts):
        part = squared[offset : offset + count]
        summary[f"{band}_selected_modes"] = int(count)
        summary[f"{band}_mean_squared_score"] = float(np.mean(part))
        summary[f"{band}_sum_squared_score"] = float(np.sum(part))
        summary[f"{band}_max_relative_eigenpair_residual"] = float(
            np.max(relative_residual[offset : offset + count])
        )
        summary[f"{band}_fraction_within_quality_tolerance"] = float(
            np.mean(
                relative_residual[offset : offset + count]
                <= args.eigenpair_relative_residual_tolerance
            )
        )
        offset += count
    return curve, summary


def run_method(
    spec: dict[str, Any],
    method: str,
    asset: Any,
    fit_row: pd.Series,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    paths = cache_paths(args.output_root, spec["date"], method)
    try:
        cached = load_cached_method(paths)
        print(f"    {method}: approximate eigen512 cache reused", flush=True)
        return cached
    except FileNotFoundError:
        pass

    print(f"    {method}: build full sparse Vecchia precision", flush=True)
    precision, precision_summary = engine.rebuild_precision(method, asset, fit_row, args)
    slq_rng = common_rng(int(args.random_seed), spec["date"], stream=0)
    print(f"    {method}: SLQ for boundaries only", flush=True)
    slq, bands, slq_summary, atoms, cumulative = engine.slq_spectrum(
        precision, args, slq_rng
    )
    edge1 = float(bands.iloc[0]["precision_upper"])
    edge2 = float(bands.iloc[1]["precision_upper"])
    median = slq_quantile(0.5, atoms, cumulative, precision.n)
    spectral_min = float(slq_summary["slq_precision_min"])
    spectral_max = float(slq_summary["slq_precision_max"])
    bands = bands.copy()
    bands["slq_spectral_median"] = median

    eigen_rng = common_rng(int(args.random_seed), spec["date"], stream=1001)
    print(f"    {method}: solve one approximate 170/170/172 eigenvector set", flush=True)
    started = time.perf_counter()
    values, vectors, residuals, solver_summaries = solve_selected_eigenpairs(
        precision,
        args,
        eigen_rng,
        edge1,
        edge2,
        median,
        spectral_min,
        spectral_max,
    )
    eigen_seconds = time.perf_counter() - started
    curve, diagnostic = full_eigen_logic_curve(
        precision, values, vectors, residuals, bands, atoms, cumulative, args
    )
    if args.retain_eigenvectors:
        paths["root"].mkdir(parents=True, exist_ok=True)
        temporary = paths["vectors"].with_suffix(".npy.tmp")
        with temporary.open("wb") as stream:
            np.save(stream, vectors.astype(np.float32))
        temporary.replace(paths["vectors"])

    summary: dict[str, Any] = {
        "dataset_id": spec["dataset_id"],
        "year": spec["year"],
        "date": spec["date"],
        "method": method,
        "native_nll_per_observation": float(fit_row["native_nll_per_observation"]),
        **precision_summary,
        **slq_summary,
        "slq_spectral_median": median,
        "eigenpair_solver_seconds": float(eigen_seconds),
        "eigenpair_solver_regions": solver_summaries,
        **diagnostic,
    }
    print(
        f"      full-eigen logic: end={summary['mean_y2']:.4f}; "
        f"D={summary['D']:.3f}; p95 eigen residual="
        f"{summary['p95_relative_eigenpair_residual']:.3e}; quality pass="
        f"{summary['fraction_within_eigenpair_quality_tolerance']:.1%}",
        flush=True,
    )
    slq = annotate(slq, spec, method)
    bands = annotate(bands, spec, method)
    curve = annotate(curve, spec, method)
    paths["root"].mkdir(parents=True, exist_ok=True)
    atomic_csv(paths["slq"], slq)
    atomic_csv(paths["bands"], bands)
    atomic_csv(paths["modes"], curve)
    write_json(paths["summary"], summary)
    paths["complete"].write_text("complete\n", encoding="utf-8")
    del precision, values, vectors, residuals
    gc.collect()
    return slq, bands, curve, summary


def shade_selected_regions(ax: plt.Axes, counts: tuple[int, int, int]) -> None:
    edges = np.cumsum((0,) + counts) / float(sum(counts))
    for band, left, right in zip(BANDS, edges[:-1], edges[1:]):
        ax.axvspan(left, right, color=BAND_COLORS[band], alpha=0.46, zorder=0)
        ax.text(
            0.5 * (left + right), 0.985, band,
            transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=8.5, color="0.35"
        )
    for edge in edges[1:-1]:
        ax.axvline(edge, color="0.70", linewidth=0.8)


def cumulative_axis(
    ax: plt.Axes,
    curves: dict[str, pd.DataFrame],
    summaries: dict[str, dict[str, Any]],
    counts: tuple[int, int, int],
) -> None:
    n_modes = int(sum(counts))
    shade_selected_regions(ax, counts)
    reference_curve = curves[METHODS[0]].sort_values("index")
    ax.fill_between(
        reference_curve["scaled_expected"].to_numpy(float),
        reference_curve["chi_square_pointwise_lower"].to_numpy(float),
        reference_curve["chi_square_pointwise_upper"].to_numpy(float),
        color="0.5", alpha=0.12, linewidth=0,
        label=rf"pointwise 95% $\chi^2_k/{n_modes}$ reference",
    )
    ax.plot([0, 1], [0, 1], color="0.25", linestyle="--", linewidth=1.1)
    for method in METHODS:
        curve = curves[method].sort_values("index")
        summary = summaries[method]
        ax.plot(
            curve["scaled_expected"], curve["scaled_cumulative"],
            color=COLORS[method], linestyle=LINESTYLES[method], linewidth=1.8,
            label=(
                f"{LABELS[method]}: end={summary['mean_y2']:.3f}, "
                f"D={summary['D']:.2f}, p95 eigres="
                f"{summary['p95_relative_eigenpair_residual']:.1e}, pass="
                f"{summary['fraction_within_eigenpair_quality_tolerance']:.0%}"
            ),
        )
    ax.set(
        xlim=(0, 1), xlabel="selected eigenmode fraction (low → high precision)",
        ylabel=f"cumulative squared standardized score / {n_modes}",
    )
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(alpha=0.20)


def plot_daily(
    date: str,
    fits: pd.DataFrame,
    curves: dict[str, pd.DataFrame],
    summaries: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.3), constrained_layout=True)
    nll = [
        float(fits.loc[fits["method"].eq(method), "native_nll_per_observation"].iloc[0])
        for method in METHODS
    ]
    axes[0].bar(
        np.arange(len(METHODS)), nll,
        color=[COLORS[method] for method in METHODS], alpha=0.82, width=0.62
    )
    axes[0].set_xticks(np.arange(len(METHODS)), [LABELS[m] for m in METHODS])
    axes[0].tick_params(axis="x", rotation=12)
    axes[0].set(ylabel="NLL / observation", title="(1) Native full-data Vecchia likelihood")
    axes[0].grid(alpha=0.18, axis="y")
    for index, value in enumerate(nll):
        axes[0].text(index, value, f"{value:.6f}", ha="center", va="bottom", fontsize=9)
    cumulative_axis(axes[1], curves, summaries, tuple(args.band_mode_counts))
    n_modes = int(sum(args.band_mode_counts))
    axes[1].set_title(
        f"(2) Full-eigen-logic diagnostic: {n_modes} approximate Lanczos eigenpairs"
    )
    fig.suptitle(f"Real {date}: adapted vs fixed Vecchia", fontsize=15)
    output = (
        args.output_root / DAILY_DIR
        / f"{date}_nll_lanczos_eigen{n_modes}_fulleigen_logic.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def segment_with_boundary(curve: pd.DataFrame, start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
    ordered = curve.sort_values("index")
    part = ordered[(ordered["index"] > start) & (ordered["index"] <= stop)]
    if start == 0:
        start_x, start_y = 0.0, 0.0
    else:
        previous = ordered[ordered["index"].eq(start)].iloc[0]
        start_x = float(previous["scaled_expected"])
        start_y = float(previous["scaled_cumulative"])
    return (
        np.concatenate(([start_x], part["scaled_expected"].to_numpy(float))),
        np.concatenate(([start_y], part["scaled_cumulative"].to_numpy(float))),
    )


def plot_daily_segments(
    date: str,
    curves: dict[str, pd.DataFrame],
    summaries: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    counts = tuple(int(value) for value in args.band_mode_counts)
    n_modes = int(sum(counts))
    stops = np.cumsum(counts)
    starts = np.concatenate(([0], stops[:-1]))
    fig, axes = plt.subplots(
        1, 3, figsize=(18.5, 5.8), sharey=True, constrained_layout=True
    )
    global_ymax = max(
        1.0,
        max(
            float(curves[method]["scaled_cumulative"].max())
            for method in METHODS
        ),
    )
    for ax, band, start, stop in zip(axes, BANDS, starts, stops):
        left, right = start / n_modes, stop / n_modes
        ax.axvspan(left, right, color=BAND_COLORS[band], alpha=0.55)
        ax.plot([left, right], [left, right], color="0.25", linestyle="--", linewidth=1.0)
        for method in METHODS:
            x, y = segment_with_boundary(curves[method], int(start), int(stop))
            ax.plot(
                x, y, color=COLORS[method], linestyle=LINESTYLES[method],
                linewidth=1.8,
                label=(f"{LABELS[method]}: mean $z^2$="
                       f"{summaries[method][f'{band}_mean_squared_score']:.3f}")
            )
        ax.set(
            xlim=(left, right), ylim=(0.0, 1.05 * global_ymax),
            xlabel="global selected-mode fraction",
            title=f"{BAND_LABELS[band]}: exact global-curve segment"
        )
        ax.grid(alpha=0.20)
        ax.legend(fontsize=8)
    axes[0].set_ylabel(f"global cumulative squared score / {n_modes}")
    fig.suptitle(
        f"Real {date}: exact slices of the same {n_modes}-mode curve "
        "(no reset, no start averaging)",
        fontsize=14,
    )
    output = (
        args.output_root / DAILY_SEGMENT_DIR
        / f"{date}_eigen{n_modes}_exact_band_segments.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def aggregate(
    specs: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    slq_frames: list[pd.DataFrame] = []
    band_frames: list[pd.DataFrame] = []
    curve_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for spec in specs:
        for method in METHODS:
            slq, bands, curve, summary = load_cached_method(
                cache_paths(args.output_root, spec["date"], method)
            )
            slq_frames.append(slq)
            band_frames.append(bands)
            curve_frames.append(curve)
            summaries.append(summary)
    slq = pd.concat(slq_frames, ignore_index=True)
    bands = pd.concat(band_frames, ignore_index=True)
    curves = pd.concat(curve_frames, ignore_index=True)
    summary_frame = pd.DataFrame(summaries)
    atomic_csv(args.output_root / "daily_slq_boundaries_only.csv", slq)
    atomic_csv(args.output_root / "daily_slq_spectral_bands.csv", bands)
    atomic_csv(args.output_root / "daily_approximate_eigen512_curves.csv", curves)
    atomic_csv(args.output_root / "daily_nll_eigen512_metrics.csv", summary_frame)
    return slq, bands, curves, summary_frame


def plot_year_average(
    year: int,
    records: list[dict[str, Any]],
    curves: pd.DataFrame,
    summaries: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.3), constrained_layout=True)
    fit_frame = pd.DataFrame(records)
    nll = [
        float(fit_frame[(fit_frame["year"].eq(year)) & (fit_frame["method"].eq(method))]
              ["native_nll_per_observation"].mean())
        for method in METHODS
    ]
    axes[0].bar(
        np.arange(len(METHODS)), nll,
        color=[COLORS[m] for m in METHODS], alpha=0.82, width=0.62
    )
    axes[0].set_xticks(np.arange(len(METHODS)), [LABELS[m] for m in METHODS])
    axes[0].tick_params(axis="x", rotation=12)
    axes[0].set(ylabel="five-date mean NLL / observation", title="(1) Native likelihood")
    for index, value in enumerate(nll):
        axes[0].text(index, value, f"{value:.6f}", ha="center", va="bottom", fontsize=9)
    axes[0].grid(alpha=0.18, axis="y")

    counts = tuple(int(value) for value in args.band_mode_counts)
    shade_selected_regions(axes[1], counts)
    axes[1].plot([0, 1], [0, 1], color="0.25", linestyle="--", linewidth=1.1)
    for method in METHODS:
        part = curves[(curves["year"].eq(year)) & (curves["method"].eq(method))]
        averaged = part.groupby("index", as_index=False).agg(
            x=("scaled_expected", "mean"),
            mean_y=("scaled_cumulative", "mean"),
            se_y=("scaled_cumulative", "sem"),
        )
        metric = summaries[(summaries["year"].eq(year)) & (summaries["method"].eq(method))]
        mean_end = float(metric["mean_y2"].mean())
        axes[1].plot(
            averaged["x"], averaged["mean_y"], color=COLORS[method],
            linestyle=LINESTYLES[method], linewidth=1.8,
            label=f"{LABELS[method]}: five-date mean end={mean_end:.3f}"
        )
        axes[1].fill_between(
            averaged["x"].to_numpy(float),
            (averaged["mean_y"] - 1.96 * averaged["se_y"]).to_numpy(float),
            (averaged["mean_y"] + 1.96 * averaged["se_y"]).to_numpy(float),
            color=COLORS[method], alpha=0.10, linewidth=0,
        )
    axes[1].set(
        xlim=(0, 1), xlabel="selected eigenmode fraction (low → high precision)",
        ylabel="five-date mean cumulative squared score / 512",
        title="(2) Mean of five daily full-eigen-logic curves",
    )
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.20)
    fig.suptitle(
        f"Real {year}-07 selected days {TEST_DAYS}: adapted vs fixed", fontsize=15
    )
    fig.savefig(
        args.output_root / f"{year}_07_selected5_average_nll_lanczos_eigen512_fulleigen_logic.png",
        dpi=200, bbox_inches="tight"
    )
    plt.close(fig)


def write_run_config(args: argparse.Namespace, specs: list[dict[str, Any]]) -> None:
    path = args.output_root / "run_config_lanczos_eigen512_fulleigen_logic.json"
    signature = {
        "dates": [spec["date"] for spec in specs],
        "methods": list(METHODS),
        "real_data_root": str(args.real_data_root),
        "lat_range": args.lat_range,
        "lon_range": args.lon_range,
        "smooth": args.smooth,
        "target_chunk_size": args.target_chunk_size,
        "lbfgs_lr": args.lbfgs_lr,
        "lbfgs_steps": args.lbfgs_steps,
        "lbfgs_eval": args.lbfgs_eval,
        "lbfgs_history": args.lbfgs_history,
        "grad_tol": args.grad_tol,
        "band_mode_counts": list(args.band_mode_counts),
        "slq_probes": args.slq_probes,
        "slq_steps": args.slq_steps,
        "eigen_oversample": args.eigen_oversample,
        "eigsh_tolerance": args.eigsh_tolerance,
        "eigsh_maxiter": args.eigsh_maxiter,
        "eigenpair_relative_residual_tolerance": args.eigenpair_relative_residual_tolerance,
        "eigenvector_orthogonality_tolerance": args.eigenvector_orthogonality_tolerance,
        "random_seed": args.random_seed,
        "precision_identity_tolerance": args.precision_identity_tolerance,
        "coefficient_drop_tolerance": args.coefficient_drop_tolerance,
        "retain_eigenvectors": bool(args.retain_eigenvectors),
    }
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("configuration_signature") != signature:
            raise ValueError(f"Configuration mismatch in {path}; use a fresh output root")
        created = previous.get("created", datetime.now().isoformat(timespec="seconds"))
    else:
        created = datetime.now().isoformat(timespec="seconds")
    write_json(
        path,
        {
            "created": created,
            "last_started": datetime.now().isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "gpu": torch.cuda.get_device_name(0),
            "configuration_signature": signature,
            "diagnostic_equations": {
                "precision_eigenpair": "Omega u_j = omega_j u_j",
                "standardized_score": "z_j = sqrt(omega_j) * (u_j^T r)",
                "squared_score": "e_j = z_j^2 approximately chi-square_1",
                "curve": "x_k=k/512; y_k=sum_{j<=k} e_j/512",
                "dense_reference_D": "max_k |sum_{j<=k}e_j-k| / sqrt(2*512)",
            },
            "slq_role": "spectral boundaries and median only; no residual-energy quadrature",
            "random_start_role": "numerical initialization only; no averaging across starts",
            "selection": (
                "170 smallest precision eigenpairs, 170 nearest the SLQ median, "
                "and 172 largest precision eigenpairs"
            ),
            "important_scope": (
                "Same statistic as dense full eigenanalysis on one approximate set of "
                "512 full-data precision eigenvectors; not an all-n eigendecomposition."
            ),
            "frequency_caveat": (
                "Ascending precision is used as a low-to-high frequency proxy on the "
                "irregular Vecchia graph, not as exact Fourier frequency."
            ),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-data-root", type=Path, default=Path("/home/jl2815/tco/data"))
    parser.add_argument(
        "--output-root", type=Path,
        default=Path(
            "/home/jl2815/tco/exercise_output/summer/"
            "vecchia_real10_selected5x2_adapted_fixed_nll_lanczos_eigen512_"
            "fulleigen_logic_light_tol1e4_lag643"
        )
    )
    parser.add_argument("--lat-range", default="-3,2")
    parser.add_argument("--lon-range", default="121,131")
    parser.add_argument("--smooth", type=float, default=0.5, choices=(0.5,))
    parser.add_argument("--target-chunk-size", type=int, default=256)
    parser.add_argument("--lbfgs-lr", type=float, default=1.0)
    parser.add_argument("--lbfgs-steps", type=int, default=5)
    parser.add_argument("--lbfgs-eval", type=int, default=20)
    parser.add_argument("--lbfgs-history", type=int, default=40)
    parser.add_argument("--grad-tol", type=float, default=1e-5)
    parser.add_argument("--suppress-fit-prints", action="store_true")
    parser.add_argument("--slq-probes", type=int, default=4)
    parser.add_argument("--slq-steps", type=int, default=128)
    parser.add_argument(
        "--band-mode-counts", type=engine.parse_band_counts,
        default=engine.parse_band_counts("170,170,172")
    )
    parser.add_argument("--spectrum-grid", type=int, default=400)
    parser.add_argument("--eigen-oversample", type=int, default=32)
    parser.add_argument("--eigsh-tolerance", type=float, default=1e-4)
    parser.add_argument("--eigsh-maxiter", type=int, default=2000)
    parser.add_argument("--eigsh-ncv-factor", type=float, default=2.0)
    parser.add_argument("--eigsh-ncv-extra", type=int, default=24)
    parser.add_argument(
        "--eigenpair-relative-residual-tolerance", type=float, default=1e-3
    )
    parser.add_argument(
        "--eigenvector-orthogonality-tolerance", type=float, default=1e-5
    )
    parser.add_argument("--random-seed", type=int, default=20260907)
    parser.add_argument("--precision-identity-tolerance", type=float, default=1e-8)
    parser.add_argument("--coefficient-drop-tolerance", type=float, default=0.0)
    parser.add_argument("--retain-eigenvectors", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if sum(args.band_mode_counts) != N_MODES:
        raise ValueError(f"Band counts must sum to {N_MODES}")
    if args.slq_probes < 2 or args.slq_steps < 16:
        raise ValueError("Use at least 2 SLQ probes and 16 SLQ steps")
    if args.spectrum_grid < 32:
        raise ValueError("--spectrum-grid must be at least 32")
    if args.eigen_oversample < 8:
        raise ValueError("--eigen-oversample must be at least 8")
    if not 0.0 < args.eigsh_tolerance < 1.0:
        raise ValueError("--eigsh-tolerance must be in (0,1)")
    if args.eigsh_maxiter < 100:
        raise ValueError("--eigsh-maxiter must be at least 100")
    if args.eigsh_ncv_factor <= 1.0:
        raise ValueError("--eigsh-ncv-factor must exceed 1")
    if args.eigenpair_relative_residual_tolerance <= 0.0:
        raise ValueError("Eigenpair residual tolerance must be positive")
    if args.eigenvector_orthogonality_tolerance <= 0.0:
        raise ValueError("Orthogonality tolerance must be positive")
    if args.precision_identity_tolerance <= 0.0:
        raise ValueError("Precision identity tolerance must be positive")
    if args.coefficient_drop_tolerance < 0.0:
        raise ValueError("Coefficient drop tolerance must be nonnegative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for fitting and precision construction")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / DAILY_DIR).mkdir(parents=True, exist_ok=True)
    (args.output_root / DAILY_SEGMENT_DIR).mkdir(parents=True, exist_ok=True)
    specs = selected_date_specs()
    write_run_config(args, specs)
    records = reference.load_fit_results(args.output_root)
    selected_ids = {spec["dataset_id"] for spec in specs}
    unexpected = [
        row for row in records
        if str(row["dataset_id"]) not in selected_ids or str(row["method"]) not in METHODS
    ]
    if unexpected:
        raise ValueError("Fit checkpoint contains rows outside this run")

    started = time.perf_counter()
    for position, spec in enumerate(specs, start=1):
        print(f"\n[{position}/{len(specs)}] {spec['date']}", flush=True)
        asset, fits = reference.load_or_fit_day(spec, records, args)
        curves: dict[str, pd.DataFrame] = {}
        summaries: dict[str, dict[str, Any]] = {}
        for method in METHODS:
            rows = fits[fits["method"].eq(method)]
            if len(rows) != 1:
                raise RuntimeError(f"Expected one {method} fit for {spec['date']}")
            _, _, curve, summary = run_method(spec, method, asset, rows.iloc[0], args)
            curves[method] = curve
            summaries[method] = summary
        plot_daily(spec["date"], fits, curves, summaries, args)
        plot_daily_segments(spec["date"], curves, summaries, args)
        reference.persist_fit_results(records, args.output_root)
        del asset, fits, curves, summaries
        gc.collect()
        torch.cuda.empty_cache()

    _, _, curves, summaries = aggregate(specs, args)
    reference.plot_native_nll(records, args.output_root)
    for year in (2024, 2025):
        plot_year_average(year, records, curves, summaries, args)
    write_json(
        args.output_root / "RUN_COMPLETE.json",
        {
            "completed": datetime.now().isoformat(timespec="seconds"),
            "elapsed_seconds": time.perf_counter() - started,
            "n_dates": len(specs),
            "n_methods": len(METHODS),
            "selected_eigenpairs_per_date_method": N_MODES,
            "random_start_averaging": False,
            "full_eigen_statistic_preserved": True,
            "all_n_eigendecomposition": False,
        },
    )
    print(f"Complete: outputs in {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
