#!/usr/bin/env python3
"""Light one-day CPU check of adapted versus fixed Vecchia on 2024-07-03.

This intentionally preserves the dense full-eigendecomposition diagnostic:

    Omega u_j = omega_j u_j,
    z_j = sqrt(omega_j) (u_j^T r),
    e_j = z_j^2,
    y_k = sum_{j <= k} e_j / K.

Only the numerical eigenpair construction is approximate.  SLQ is used solely
to estimate the 1/3 and 2/3 spectral boundaries.  To keep a laptop CPU run
practical, K=48 modes are used (16 low, 16 middle, and 16 high precision).
Previously fitted lag-6/4/3 parameters and native NLL values are reused.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import socket
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib")
)

import numpy as np
import pandas as pd
import scipy.linalg
import scipy.sparse
import torch


HERE = Path(__file__).resolve().parent
REPO = next(parent for parent in HERE.parents if (parent / "src/GEMS_TCO").is_dir())
SRC = REPO / "src"
INTERACTION = HERE.parent / "interaction_diagnostic"
for candidate in (HERE, SRC, INTERACTION):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import vecchia_real10_selected5x2_adapted_fixed_nll_lanczos_eigen512_fulleigen_logic_lag643 as diagnostic  # noqa: E402
import vecchia_real59_adapted_fixed_threeway_slq512_lag643 as engine  # noqa: E402


core = engine.core
reference = engine.reference
METHODS = engine.METHODS
DATE = "2024-07-03"
N_MODES = 48


@dataclass
class CachedPrecision:
    whitener: scipy.sparse.csr_matrix
    residual: np.ndarray
    metadata: dict[str, Any]

    @property
    def n(self) -> int:
        return int(self.whitener.shape[0])

    def matvec(self, vector: np.ndarray) -> np.ndarray:
        x = np.asarray(vector, dtype=np.float64).reshape(-1)
        return np.asarray(self.whitener.T @ (self.whitener @ x)).reshape(-1)

    def matmat(self, vectors: np.ndarray) -> np.ndarray:
        x = np.asarray(vectors, dtype=np.float64)
        return np.asarray(self.whitener.T @ (self.whitener @ x))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real-data-root",
        type=Path,
        default=Path("/Users/joonwonlee/Documents/GEMS_DATA"),
    )
    parser.add_argument(
        "--fit-csv",
        type=Path,
        default=(
            REPO
            / "outputs/summer_26/vecchia_local_real1_adapted_fixed_lag643_20240703_run01"
            / "task_00_real_20240703/fits.csv"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            REPO
            / "outputs/summer_26/vecchia_local_20240703_adapted_fixed_"
            "eigen48_fulleigen_logic_krylov512"
        ),
    )
    parser.add_argument(
        "--operator-cache-root",
        type=Path,
        default=(
            REPO
            / "outputs/summer_26/vecchia_local_20240703_lag643_operator_cache"
        ),
    )
    parser.add_argument("--band-mode-counts", default="16,16,16")
    parser.add_argument("--slq-probes", type=int, default=4)
    parser.add_argument("--slq-steps", type=int, default=128)
    parser.add_argument("--spectrum-grid", type=int, default=240)
    parser.add_argument("--eigenpair-relative-residual-tolerance", type=float, default=1e-3)
    parser.add_argument("--eigenvector-orthogonality-tolerance", type=float, default=1e-5)
    parser.add_argument("--krylov-steps", type=int, default=512)
    parser.add_argument("--precision-identity-tolerance", type=float, default=1e-8)
    parser.add_argument("--coefficient-drop-tolerance", type=float, default=0.0)
    parser.add_argument("--target-chunk-size", type=int, default=64)
    parser.add_argument("--random-seed", type=int, default=20260908)
    parser.add_argument("--no-operator-cache", action="store_true")
    return parser


def parse_band_counts(value: str) -> tuple[int, int, int]:
    try:
        counts = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("band counts must be three integers") from error
    if len(counts) != 3 or any(count <= 0 for count in counts):
        raise argparse.ArgumentTypeError("band counts must be three positive integers")
    return counts


def write_json(path: Path, value: Any) -> None:
    def clean(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(key): clean(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(val) for val in item]
        if isinstance(item, (np.integer,)):
            return int(item)
        if isinstance(item, (np.floating,)):
            return None if not np.isfinite(item) else float(item)
        if isinstance(item, float):
            return None if not np.isfinite(item) else item
        return item

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def loader_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        real_data_root=args.real_data_root,
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


def model_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        smooth=0.5,
        daily_stride=2,
        target_chunk_size=int(args.target_chunk_size),
        union_target_chunk_size=0,
        min_target_points=1,
        fixed_nugget=None,
    )


def fitted_parameters(row: pd.Series) -> dict[str, float]:
    return {
        "sigmasq": float(row["est_sigmasq"]),
        "range_lat": float(row["est_range_lat"]),
        "range_lon": float(row["est_range_lon"]),
        "range_time": float(row["est_range_time"]),
        "advec_lat": float(row["est_advec_lat"]),
        "advec_lon": float(row["est_advec_lon"]),
        "nugget": float(row["est_nugget"]),
    }


def operator_paths(args: argparse.Namespace, method: str) -> dict[str, Path]:
    root = args.operator_cache_root / method
    return {
        "root": root,
        "whitener": root / "whitener_B.npz",
        "residual": root / "residual.npy",
        "metadata": root / "metadata.json",
    }


def load_operator(args: argparse.Namespace, method: str) -> CachedPrecision | None:
    paths = operator_paths(args, method)
    if args.no_operator_cache or not all(
        paths[name].is_file() for name in ("whitener", "residual", "metadata")
    ):
        return None
    print(f"  {method}: reuse cached sparse precision", flush=True)
    return CachedPrecision(
        scipy.sparse.load_npz(paths["whitener"]).tocsr(),
        np.load(paths["residual"]),
        json.loads(paths["metadata"].read_text(encoding="utf-8")),
    )


def save_operator(args: argparse.Namespace, method: str, precision: Any) -> None:
    if args.no_operator_cache:
        return
    paths = operator_paths(args, method)
    paths["root"].mkdir(parents=True, exist_ok=True)
    scipy.sparse.save_npz(paths["whitener"], precision.whitener, compressed=False)
    np.save(paths["residual"], np.asarray(precision.residual, dtype=np.float64))
    write_json(paths["metadata"], precision.metadata)


def build_operator(
    args: argparse.Namespace,
    method: str,
    asset: Any,
    fit_row: pd.Series,
) -> tuple[Any, dict[str, Any]]:
    cached = load_operator(args, method)
    if cached is not None:
        return cached, {"operator_cache_reused": True, **cached.metadata}

    print(f"  {method}: reconstruct CPU lag-6/4/3 model", flush=True)
    seed = {
        "seed_lat": float(fit_row["init_advec_lat"]),
        "seed_lon": float(fit_row["init_advec_lon"]),
    }
    model = core.build_geometry_model(
        method, asset, seed, torch.device("cpu"), model_args(args)
    )
    started = time.perf_counter()
    model.precompute_conditioning_sets()
    precompute_seconds = time.perf_counter() - started
    params = torch.as_tensor(
        core.physical_to_raw(fitted_parameters(fit_row)), dtype=core.DTYPE
    )
    with torch.no_grad():
        beta = model.get_gls_beta(params).detach().cpu().numpy().reshape(-1)
    native_quadratic = engine.native_gls_quadratic(model, params, beta)
    precision = engine.build_sparse_vecchia_precision(
        model,
        params,
        beta,
        chunk_size=int(args.target_chunk_size),
        coefficient_drop_tolerance=float(args.coefficient_drop_tolerance),
        progress=lambda message: print(f"    B: {message}", flush=True),
    )
    identity = engine.verify_precision_identity(precision, native_quadratic)
    if identity["relative_error"] > float(args.precision_identity_tolerance):
        raise RuntimeError(f"Sparse precision identity failed: {identity}")
    reconstructed_nll = (
        float(precision.metadata["conditional_logdet_half"])
        + 0.5 * float(identity["sparse_quadratic"])
    ) / precision.n
    metadata = {
        **precision.metadata,
        "operator_cache_reused": False,
        "precompute_seconds": precompute_seconds,
        "native_quadratic": native_quadratic,
        "precision_identity_relative_error": identity["relative_error"],
        "reconstructed_native_nll_per_observation": reconstructed_nll,
        "stored_native_nll_per_observation": float(fit_row["final_native_nll"]),
        "reconstructed_minus_stored_nll": (
            reconstructed_nll - float(fit_row["final_native_nll"])
        ),
        "gls_beta": beta.tolist(),
    }
    precision.metadata.update(metadata)
    save_operator(args, method, precision)
    del model, params
    gc.collect()
    return precision, metadata


def method_result_paths(args: argparse.Namespace, method: str) -> dict[str, Path]:
    root = args.output_root / "method_results" / method
    return {
        "root": root,
        "curve": root / f"eigen{N_MODES}_curve.csv",
        "slq": root / "slq_spectrum.csv",
        "bands": root / "slq_bands.csv",
        "summary": root / "summary.json",
        "complete": root / "COMPLETE",
    }


def load_method_result(
    args: argparse.Namespace, method: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]] | None:
    paths = method_result_paths(args, method)
    if not paths["complete"].is_file():
        return None
    print(f"  {method}: reuse completed diagnostic", flush=True)
    return (
        pd.read_csv(paths["slq"]),
        pd.read_csv(paths["bands"]),
        pd.read_csv(paths["curve"]),
        json.loads(paths["summary"].read_text(encoding="utf-8")),
    )


def run_method(
    args: argparse.Namespace,
    method: str,
    asset: Any,
    fit_row: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    completed = load_method_result(args, method)
    if completed is not None:
        return completed

    precision, precision_summary = build_operator(args, method, asset, fit_row)
    slq_rng = diagnostic.common_rng(args.random_seed, DATE, stream=0)
    print(f"  {method}: SLQ boundaries ({args.slq_probes} x {args.slq_steps})", flush=True)
    slq, bands, slq_summary, atoms, cumulative = engine.slq_spectrum(
        precision, args, slq_rng
    )
    edge1 = float(bands.iloc[0]["precision_upper"])
    edge2 = float(bands.iloc[1]["precision_upper"])
    median = diagnostic.slq_quantile(0.5, atoms, cumulative, precision.n)
    bands = bands.copy()
    bands["slq_spectral_median"] = median

    eigen_rng = diagnostic.common_rng(args.random_seed, DATE, stream=1001)
    print(
        f"  {method}: solve {sum(args.band_mode_counts)} approximate eigenpairs "
        f"({args.band_mode_counts})",
        flush=True,
    )
    started = time.perf_counter()
    values, vectors, residuals, solver_summaries = explicit_random_start_ritz(
        precision, args, eigen_rng, edge1, edge2, atoms, cumulative
    )
    eigen_seconds = time.perf_counter() - started
    curve, curve_summary = diagnostic.full_eigen_logic_curve(
        precision, values, vectors, residuals, bands, atoms, cumulative, args
    )
    summary = {
        "date": DATE,
        "method": method,
        "native_nll_per_observation": float(fit_row["final_native_nll"]),
        **precision_summary,
        **slq_summary,
        "slq_spectral_median": median,
        "eigenpair_solver_seconds": eigen_seconds,
        "eigenpair_solver_regions": solver_summaries,
        **curve_summary,
    }
    paths = method_result_paths(args, method)
    paths["root"].mkdir(parents=True, exist_ok=True)
    diagnostic.atomic_csv(paths["slq"], slq)
    diagnostic.atomic_csv(paths["bands"], bands)
    diagnostic.atomic_csv(paths["curve"], curve)
    write_json(paths["summary"], summary)
    paths["complete"].write_text("complete\n", encoding="utf-8")
    print(
        f"    {method}: end={summary['mean_y2']:.4f}, D={summary['D']:.3f}, "
        f"p95 eigres={summary['p95_relative_eigenpair_residual']:.2e}",
        flush=True,
    )
    del precision, values, vectors, residuals
    gc.collect()
    return slq, bands, curve, summary


def explicit_random_start_ritz(
    precision: Any,
    args: argparse.Namespace,
    rng: np.random.Generator,
    edge1: float,
    edge2: float,
    slq_atoms: np.ndarray,
    slq_cumulative: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Build explicit full-data Ritz vectors from one stored Lanczos basis.

    Unlike residual-started quadrature, the start is independent Rademacher
    noise and the n-dimensional vectors U=Q S are explicitly reconstructed.
    The full-eigen statistic is subsequently evaluated from U.T @ residual.
    """
    n = int(precision.n)
    requested_steps = min(int(args.krylov_steps), n - 1)
    if requested_steps <= sum(args.band_mode_counts):
        raise ValueError("--krylov-steps must exceed the selected mode count")
    basis = np.empty((n, requested_steps), dtype=np.float64, order="F")
    alpha = np.empty(requested_steps, dtype=np.float64)
    beta = np.empty(max(requested_steps - 1, 0), dtype=np.float64)
    q_previous = np.zeros(n, dtype=np.float64)
    q = rng.choice(np.asarray([-1.0, 1.0]), size=n).astype(np.float64)
    q /= np.linalg.norm(q)
    beta_previous = 0.0
    completed = 0
    started = time.perf_counter()
    for step in range(requested_steps):
        basis[:, step] = q
        z = np.asarray(precision.matvec(q), dtype=np.float64)
        diagonal = float(q @ z)
        z -= diagonal * q
        if step:
            z -= beta_previous * q_previous
        active = basis[:, : step + 1]
        # Two-pass classical reorthogonalization keeps the reconstructed Ritz
        # vectors suitable for the same orthogonality check as dense eigh.
        for _ in range(2):
            correction = active.T @ z
            z -= active @ correction
        next_beta = float(np.linalg.norm(z))
        alpha[step] = diagonal
        completed = step + 1
        if step == requested_steps - 1:
            break
        if not np.isfinite(next_beta) or next_beta <= 1e-12:
            break
        beta[step] = next_beta
        q_previous, q = q, z / next_beta
        beta_previous = next_beta
        if completed % 32 == 0:
            print(
                f"      explicit Lanczos basis {completed}/{requested_steps} steps",
                flush=True,
            )

    basis = np.asfortranarray(basis[:, :completed])
    ritz_values, small_vectors = scipy.linalg.eigh_tridiagonal(
        alpha[:completed], beta[: max(completed - 1, 0)], check_finite=False
    )
    positive = np.isfinite(ritz_values) & (ritz_values > 0.0)
    ritz_values = ritz_values[positive]
    small_vectors = small_vectors[:, positive]
    quantiles = engine.cdf_at(
        ritz_values, slq_atoms, slq_cumulative, precision.n
    )
    bounds = (
        ("low", -np.inf, edge1, 0.0, 1.0 / 3.0),
        ("middle", edge1, edge2, 1.0 / 3.0, 2.0 / 3.0),
        ("high", edge2, np.inf, 2.0 / 3.0, 1.0),
    )
    selected_indices: list[np.ndarray] = []
    pool_counts: dict[str, int] = {}
    for (band, lower, upper, q_lower, q_upper), count in zip(
        bounds, args.band_mode_counts
    ):
        pool = np.flatnonzero((ritz_values > lower) & (ritz_values <= upper))
        pool_counts[band] = int(len(pool))
        if len(pool) < int(count):
            raise RuntimeError(
                f"{band}: {len(pool)} Ritz candidates for {count} requested; "
                "increase --krylov-steps"
            )
        targets = q_lower + (np.arange(int(count)) + 0.5) / int(count) * (
            q_upper - q_lower
        )
        available = list(int(index) for index in pool)
        chosen: list[int] = []
        for target in targets:
            position = min(
                range(len(available)),
                key=lambda idx: abs(float(quantiles[available[idx]]) - float(target)),
            )
            chosen.append(available.pop(position))
        selected_indices.append(np.asarray(chosen, dtype=np.int64))

    chosen = np.concatenate(selected_indices)
    approximate_vectors = np.asfortranarray(basis @ small_vectors[:, chosen])
    values, vectors, relative_residual, refinement = diagnostic.refine_precision_subspace(
        precision, approximate_vectors
    )
    order = np.argsort(values)
    values = values[order]
    vectors = np.asfortranarray(vectors[:, order])
    relative_residual = relative_residual[order]
    gram_error = float(
        np.max(np.abs(vectors.T @ vectors - np.eye(len(values))))
    )
    if gram_error > float(args.eigenvector_orthogonality_tolerance):
        raise RuntimeError(
            f"Explicit Ritz vectors fail orthogonality: {gram_error:.3e}"
        )
    summaries = [
        {
            "label": "single_random_start_explicit_lanczos",
            "requested_krylov_steps": requested_steps,
            "completed_krylov_steps": completed,
            "seconds": time.perf_counter() - started,
            "candidate_pool_counts": pool_counts,
            "selected_modes": int(len(values)),
            "max_orthogonality_error": gram_error,
            "max_relative_eigenpair_residual": float(np.max(relative_residual)),
            "p95_relative_eigenpair_residual": float(
                np.quantile(relative_residual, 0.95)
            ),
            **refinement,
        }
    ]
    return values, vectors, relative_residual, summaries


def main() -> None:
    args = build_parser().parse_args()
    args.band_mode_counts = parse_band_counts(args.band_mode_counts)
    if sum(args.band_mode_counts) != N_MODES:
        raise ValueError(f"Band mode counts must sum to {N_MODES}")
    if args.slq_probes < 2 or args.slq_steps < 16:
        raise ValueError("Use at least 2 SLQ probes and 16 SLQ steps")
    if not args.fit_csv.is_file():
        raise FileNotFoundError(args.fit_csv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    fits = pd.read_csv(args.fit_csv)
    fits = fits[(fits["date"].eq(DATE)) & (fits["geometry"].isin(METHODS))].copy()
    if set(fits["geometry"]) != set(METHODS):
        raise RuntimeError(f"Fit CSV lacks adapted/fixed rows for {DATE}")
    fits["method"] = fits["geometry"]
    fits["native_nll_per_observation"] = fits["final_native_nll"].astype(float)

    spec = next(spec for spec in reference.date_specs() if spec["date"] == DATE)
    print(f"Load full 8-hour real data for {DATE}", flush=True)
    asset = core.load_real_asset(spec, loader_args(args))
    print(f"  valid full-data observations: {asset.n_valid:,}", flush=True)

    run_started = time.perf_counter()
    curves: dict[str, pd.DataFrame] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        fit_row = fits.loc[fits["method"].eq(method)].iloc[0]
        _, _, curve, summary = run_method(args, method, asset, fit_row)
        curves[method] = curve
        summaries[method] = summary

    diagnostic.plot_daily(DATE, fits, curves, summaries, args)
    diagnostic.plot_daily_segments(DATE, curves, summaries, args)
    summary_rows = []
    for method in METHODS:
        row = {key: value for key, value in summaries[method].items() if not isinstance(value, (list, dict))}
        summary_rows.append(row)
    diagnostic.atomic_csv(args.output_root / "comparison_summary.csv", pd.DataFrame(summary_rows))
    write_json(
        args.output_root / "RUN_COMPLETE.json",
        {
            "completed": datetime.now().isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "device": "cpu",
            "date": DATE,
            "methods": list(METHODS),
            "elapsed_seconds": time.perf_counter() - run_started,
            "selected_eigenpairs_per_method": N_MODES,
            "band_mode_counts": list(args.band_mode_counts),
            "reported_eigenpair_quality_threshold": (
                args.eigenpair_relative_residual_tolerance
            ),
            "krylov_steps": args.krylov_steps,
            "slq_role": "spectral band boundaries only",
            "diagnostic": "z_j^2 = omega_j (u_j^T r)^2; cumulative sum divided by K",
        },
    )
    print(f"Complete: {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
