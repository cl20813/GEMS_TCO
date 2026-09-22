#!/usr/bin/env python3
"""Evaluate a predeclared held-out generalized-eigenmode count path.

This is a post-processing analysis for a completed five-day pilot.  It reads
only the saved pilot manifest and selected point table, reconstructs the true
and fitted-null covariance matrices, and recomputes the covariance-only
training generalized eigenvectors.  Held-out responses enter only after the
directions and the complete K grid have been fixed.

For a projected covariance pair with generalized eigenvalues ``lambda_i``,
the Gaussian log likelihood ratio has the following exact representations::

    H0:  0.5 * sum(-log(lambda_i) + (1 - 1/lambda_i) * Z_i**2)
    H1:  0.5 * sum(-log(lambda_i) + (lambda_i - 1) * Z_i**2)

where the ``Z_i`` are independent standard normal variables.  The Monte Carlo
calibration therefore uses chunked chi-square contributions rather than
allocating correlated Gaussian draws.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import scipy.linalg

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diagnostic_core import (
    CovarianceParameters,
    advected_separable_covariance,
    covariance_log_likelihood_ratio,
    joint_matern_half_covariance,
    pairwise_lags,
    solve_generalized_eigenproblem,
    standardize_directions_for_design,
)


HERE = Path(__file__).resolve().parent
DEFAULT_PILOT = HERE / "outputs/nugget0_five_day_092226"
DEFAULT_K_GRID = (1, 2, 4, 8, 12, 20, 40, 80, 100, 200, 350, 800)


@dataclass(frozen=True)
class DayData:
    date: str
    coordinates: np.ndarray
    residual: np.ndarray
    ordering: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class LLRMonteCarloResult:
    null_critical: float
    p_value: float
    power: float
    null_exceedances: int
    alternative_rejections: int
    p_value_mc_se: float
    power_mc_se: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--k-grid", type=int, nargs="+", default=DEFAULT_K_GRID)
    parser.add_argument("--replicates", type=int, default=50_000)
    parser.add_argument("--random-seed", type=int, default=20260922)
    parser.add_argument("--chunk-size", type=int, default=2_048)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
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
    temporary.write_text(
        json.dumps(json_ready(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.12g")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def covariance_parameters(value: dict[str, Any]) -> CovarianceParameters:
    required = (
        "variance",
        "range_lat",
        "range_lon",
        "range_time",
        "advec_lat",
        "advec_lon",
        "nugget",
    )
    missing = [name for name in required if name not in value]
    if missing:
        raise ValueError(f"covariance parameters are missing: {missing}")
    return CovarianceParameters(**{name: float(value[name]) for name in required})


def load_day_data(point_path: Path, dates: Iterable[str]) -> list[DayData]:
    frame = pd.read_csv(point_path)
    required = {
        "date",
        "time_index",
        "anchor_rank",
        "source_latitude",
        "source_longitude",
        "true_residual",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"selected point table is missing columns: {missing}")
    frame["date"] = frame["date"].astype(str)
    requested_dates = tuple(str(date) for date in dates)
    unexpected = sorted(set(frame["date"]).difference(requested_dates))
    if unexpected:
        raise ValueError(f"selected point table contains unexpected dates: {unexpected}")

    day_data: list[DayData] = []
    reference_ordering: tuple[tuple[int, int], ...] | None = None
    reference_dimension: int | None = None
    for date in requested_dates:
        day = frame.loc[frame["date"] == date].copy()
        if day.empty:
            raise ValueError(f"selected point table has no rows for {date}")
        if day.duplicated(["time_index", "anchor_rank"]).any():
            raise ValueError(f"duplicate time/anchor positions found for {date}")
        day.sort_values(["time_index", "anchor_rank"], inplace=True)
        values = day[
            ["source_latitude", "source_longitude", "time_index", "true_residual"]
        ].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite coordinate or response found for {date}")
        ordering = tuple(
            (int(time_index), int(anchor_rank))
            for time_index, anchor_rank in day[["time_index", "anchor_rank"]].itertuples(
                index=False, name=None
            )
        )
        if reference_ordering is None:
            reference_ordering = ordering
            reference_dimension = len(day)
        elif ordering != reference_ordering or len(day) != reference_dimension:
            raise ValueError("time/anchor ordering differs across independent days")
        day_data.append(
            DayData(
                date=date,
                coordinates=np.ascontiguousarray(values[:, :3]),
                residual=np.ascontiguousarray(values[:, 3]),
                ordering=ordering,
            )
        )
    return day_data


def clipped_k_grid(values: Iterable[int], dimension: int) -> list[int]:
    if dimension < 1:
        raise ValueError("dimension must be positive")
    result: list[int] = []
    for raw in values:
        value = int(raw)
        if value < 1:
            raise ValueError("all K values must be positive")
        clipped = min(value, dimension)
        if clipped not in result:
            result.append(clipped)
    return result


def llr_from_eigenvalues(
    standard_normal_squared: np.ndarray,
    eigenvalues: np.ndarray,
    *,
    hypothesis: str,
) -> np.ndarray:
    """Evaluate exact Gaussian covariance LLRs from squared normal draws."""

    eigenvalues = np.asarray(eigenvalues, dtype=np.float64).reshape(-1)
    squared = np.asarray(standard_normal_squared, dtype=np.float64)
    if squared.ndim != 2 or squared.shape[1] != len(eigenvalues):
        raise ValueError("draws and generalized eigenvalues do not align")
    if not np.isfinite(eigenvalues).all() or np.any(eigenvalues <= 0.0):
        raise ValueError("generalized eigenvalues must be finite and positive")
    constant = -0.5 * np.log(eigenvalues).sum()
    if hypothesis == "null":
        weights = 0.5 * (1.0 - 1.0 / eigenvalues)
    elif hypothesis == "alternative":
        weights = 0.5 * (eigenvalues - 1.0)
    else:
        raise ValueError("hypothesis must be 'null' or 'alternative'")
    return constant + squared @ weights


def monte_carlo_llr(
    eigenvalues: np.ndarray,
    observed_llr: float,
    *,
    replicates: int,
    alpha: float,
    random_seed: int,
    chunk_size: int,
) -> LLRMonteCarloResult:
    """Calibrate one LLR using independent, chunked chi-square contributions."""

    eigenvalues = np.asarray(eigenvalues, dtype=np.float64).reshape(-1)
    if replicates < 100:
        raise ValueError("at least 100 Monte Carlo replicates are required")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    if not np.isfinite(eigenvalues).all() or np.any(eigenvalues <= 0.0):
        raise ValueError("generalized eigenvalues must be finite and positive")

    seed_sequence = np.random.SeedSequence(int(random_seed))
    null_seed, alternative_seed = seed_sequence.spawn(2)
    null_rng = np.random.default_rng(null_seed)
    alternative_rng = np.random.default_rng(alternative_seed)
    null_llr = np.empty(replicates, dtype=np.float64)
    alternative_llr = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, chunk_size):
        stop = min(replicates, start + chunk_size)
        shape = (stop - start, len(eigenvalues))
        null_squared = np.square(null_rng.standard_normal(shape))
        null_llr[start:stop] = llr_from_eigenvalues(null_squared, eigenvalues, hypothesis="null")
        del null_squared
        alternative_squared = np.square(alternative_rng.standard_normal(shape))
        alternative_llr[start:stop] = llr_from_eigenvalues(
            alternative_squared,
            eigenvalues,
            hypothesis="alternative",
        )
        del alternative_squared

    null_critical = float(np.quantile(null_llr, 1.0 - alpha, method="higher"))
    null_exceedances = int(np.sum(null_llr >= float(observed_llr)))
    alternative_rejections = int(np.sum(alternative_llr > null_critical))
    p_value = float((1 + null_exceedances) / (replicates + 1))
    power = float(alternative_rejections / replicates)
    return LLRMonteCarloResult(
        null_critical=null_critical,
        p_value=p_value,
        power=power,
        null_exceedances=null_exceedances,
        alternative_rejections=alternative_rejections,
        p_value_mc_se=float(np.sqrt(p_value * (1.0 - p_value) / (replicates + 1))),
        power_mc_se=float(np.sqrt(power * (1.0 - power) / replicates)),
    )


def projected_generalized_eigenvalues(
    true_covariance: np.ndarray,
    null_covariance: np.ndarray,
) -> np.ndarray:
    true_covariance = (np.asarray(true_covariance) + np.asarray(true_covariance).T) * 0.5
    null_covariance = (np.asarray(null_covariance) + np.asarray(null_covariance).T) * 0.5
    values = scipy.linalg.eigvalsh(
        true_covariance,
        null_covariance,
        check_finite=False,
        driver="gv",
    )
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise scipy.linalg.LinAlgError("projected covariance pair is not positive definite")
    return values


def plot_path(frame: pd.DataFrame, path: Path, alpha: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = frame["k"].to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    axes[0, 0].plot(x, frame["training_cumulative_kl_fraction"], marker="o")
    axes[0, 0].set_ylabel("Training cumulative KL fraction")
    axes[0, 0].set_ylim(0.0, 1.03)

    axes[0, 1].plot(x, frame["oracle_power"], marker="o", color="#26734d")
    axes[0, 1].axhline(alpha, color="0.45", linestyle="--", linewidth=1.0)
    axes[0, 1].set_ylabel(f"Oracle power (alpha={alpha:g})")
    axes[0, 1].set_ylim(0.0, 1.03)

    p_floor = 0.5 / (int(frame["replicates"].iloc[0]) + 1)
    axes[1, 0].plot(
        x,
        np.maximum(frame["bootstrap_p_value"], p_floor),
        marker="o",
        color="#7a3e9d",
    )
    axes[1, 0].axhline(alpha, color="0.45", linestyle="--", linewidth=1.0)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylabel("Held-out Monte Carlo p-value")

    axes[1, 1].plot(x, frame["observed_llr"], marker="o", label="observed")
    axes[1, 1].plot(
        x,
        frame["null_critical"],
        marker="s",
        linestyle="--",
        label=f"null {100 * (1 - alpha):g}% critical",
    )
    axes[1, 1].axhline(0.0, color="0.75", linewidth=0.8)
    axes[1, 1].set_ylabel("Held-out log likelihood ratio")
    axes[1, 1].legend(frameon=False)

    for axis in axes.ravel():
        axis.set_xscale("log", base=2)
        axis.set_xlabel("Predeclared number of training modes K")
        axis.set_xticks(x)
        axis.set_xticklabels([str(value) for value in x], rotation=45)
        axis.grid(alpha=0.2)
    fig.suptitle("Exact held-out generalized-eigenmode path")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    pilot_dir = args.pilot_dir.expanduser().resolve()
    manifest_path = pilot_dir / "run_manifest.json"
    point_path = pilot_dir / "selected_flow_tube_points.csv"
    if not manifest_path.is_file() or not point_path.is_file():
        raise FileNotFoundError("pilot-dir must contain run_manifest.json and selected points")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split = manifest["split"]
    if bool(split.get("responses_used_for_null_fit_or_direction_selection", True)):
        raise ValueError("pilot manifest does not certify covariance-only direction selection")
    design_dates = tuple(str(item) for item in split["design_dates"])
    heldout_dates = tuple(str(item) for item in split["heldout_dates"])
    all_dates = design_dates + heldout_dates
    if not design_dates or not heldout_dates or len(set(all_dates)) != len(all_dates):
        raise ValueError("manifest must define disjoint, nonempty design and held-out dates")

    days = load_day_data(point_path, all_dates)
    day_by_date = {day.date: day for day in days}
    dimension = len(days[0].residual)
    manifest_dimension = int(manifest["subset"]["dimension_per_day"])
    if dimension != manifest_dimension:
        raise ValueError(
            f"point dimension {dimension} differs from manifest dimension {manifest_dimension}"
        )
    truth = covariance_parameters(manifest["truth"])
    fitted_null = covariance_parameters(manifest["null"]["fit"]["parameters"])
    if truth.nugget != 0.0 or fitted_null.nugget != 0.0:
        raise ValueError("this exact pilot path expects statistical nugget zero")
    jitter_ratio = float(manifest["numerics"]["numerical_jitter_ratio"])

    true_covariances: dict[str, np.ndarray] = {}
    null_covariances: dict[str, np.ndarray] = {}
    for day in days:
        geometry = pairwise_lags(day.coordinates)
        true_covariances[day.date] = joint_matern_half_covariance(
            geometry,
            truth,
            numerical_jitter_ratio=jitter_ratio,
        )
        null_covariances[day.date] = advected_separable_covariance(
            geometry,
            fitted_null,
            numerical_jitter_ratio=jitter_ratio,
        )

    reference_true = np.mean(np.stack([true_covariances[date] for date in design_dates]), axis=0)
    reference_null = np.mean(np.stack([null_covariances[date] for date in design_dates]), axis=0)
    training_eigen = solve_generalized_eigenproblem(reference_true, reference_null)
    directions = training_eigen.eigenvectors

    projected: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for date in heldout_dates:
        standardized, projected_null, projected_true = standardize_directions_for_design(
            directions,
            null_covariances[date],
            true_covariances[date],
        )
        projected[date] = (
            projected_null,
            projected_true,
            standardized.T @ day_by_date[date].residual,
        )

    k_grid = clipped_k_grid(args.k_grid, dimension)
    rows: list[dict[str, Any]] = []
    total_training_kl = float(training_eigen.scores.sum())
    for k in k_grid:
        observed_llr = 0.0
        heldout_eigenvalues: list[np.ndarray] = []
        day_minimum: list[float] = []
        day_maximum: list[float] = []
        for date in heldout_dates:
            projected_null, projected_true, observed = projected[date]
            null_k = projected_null[:k, :k]
            true_k = projected_true[:k, :k]
            observed_k = observed[:k]
            observed_llr += covariance_log_likelihood_ratio(observed_k, null_k, true_k)
            values = projected_generalized_eigenvalues(true_k, null_k)
            heldout_eigenvalues.append(values)
            day_minimum.append(float(values.min()))
            day_maximum.append(float(values.max()))
        eigenvalues = np.concatenate(heldout_eigenvalues)
        # Keying the stream by K makes each row reproducible even if another
        # predeclared K value is added later, while retaining one master seed.
        k_seed = int(np.random.SeedSequence([int(args.random_seed), int(k)]).generate_state(1)[0])
        simulation = monte_carlo_llr(
            eigenvalues,
            observed_llr,
            replicates=int(args.replicates),
            alpha=float(args.alpha),
            random_seed=k_seed,
            chunk_size=int(args.chunk_size),
        )
        training_kl = float(training_eigen.scores[:k].sum())
        heldout_kl = float(0.5 * np.sum(eigenvalues - 1.0 - np.log(eigenvalues)))
        heldout_null_expected_llr = float(
            0.5 * np.sum(1.0 - 1.0 / eigenvalues - np.log(eigenvalues))
        )
        rows.append(
            {
                "k": int(k),
                "heldout_day_count": len(heldout_dates),
                "projected_dimension": len(eigenvalues),
                "training_cumulative_kl": training_kl,
                "training_cumulative_kl_fraction": training_kl / total_training_kl,
                "heldout_oracle_kl": heldout_kl,
                "heldout_null_expected_llr": heldout_null_expected_llr,
                "observed_llr": float(observed_llr),
                "null_critical": simulation.null_critical,
                "bootstrap_p_value": simulation.p_value,
                "oracle_power": simulation.power,
                "p_value_mc_se": simulation.p_value_mc_se,
                "power_mc_se": simulation.power_mc_se,
                "null_exceedances": simulation.null_exceedances,
                "alternative_rejections": simulation.alternative_rejections,
                "minimum_heldout_eigenvalue": min(day_minimum),
                "maximum_heldout_eigenvalue": max(day_maximum),
                "replicates": int(args.replicates),
                "alpha": float(args.alpha),
                "master_random_seed": int(args.random_seed),
                "derived_k_seed": k_seed,
            }
        )
        print(
            f"K={k:4d}: observed={observed_llr: .5f}, "
            f"critical={simulation.null_critical: .5f}, "
            f"p={simulation.p_value:.5f}, power={simulation.power:.5f}",
            flush=True,
        )

    result = pd.DataFrame(rows)
    output_dir = pilot_dir / "mode_count_path"
    csv_path = output_dir / "mode_count_path.csv"
    json_path = output_dir / "mode_count_path_summary.json"
    figure_path = output_dir / "mode_count_path.png"
    atomic_csv(csv_path, result)
    plot_path(result, figure_path, float(args.alpha))
    summary = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "pilot_dir": str(pilot_dir),
        "inputs": {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "selected_points": str(point_path),
            "selected_points_sha256": sha256(point_path),
        },
        "split": {
            "design_dates": design_dates,
            "heldout_dates": heldout_dates,
            "responses_used_for_direction_selection": False,
            "heldout_responses_used_only_for_observed_statistics": True,
        },
        "predeclared_k_grid_requested": [int(value) for value in args.k_grid],
        "k_grid_after_dimension_clipping": k_grid,
        "dimension_per_day": dimension,
        "truth": truth.to_dict(),
        "fitted_null": fitted_null.to_dict(),
        "numerical_jitter_ratio": jitter_ratio,
        "training_generalized_eigenproblem": {
            "matrix_kl": training_eigen.matrix_kl,
            "spectral_kl": training_eigen.spectral_kl,
            "max_relative_residual": training_eigen.max_relative_residual,
            "max_null_orthonormality_error": (training_eigen.max_null_orthonormality_error),
        },
        "monte_carlo": {
            "replicates": int(args.replicates),
            "alpha": float(args.alpha),
            "master_random_seed": int(args.random_seed),
            "stream_derivation": "SeedSequence([master_random_seed, K])",
            "chunk_size": int(args.chunk_size),
            "draw_representation": "independent chi-square(1) contributions from projected generalized eigenvalues",
        },
        "results": result.to_dict(orient="records"),
        "outputs": {
            "csv": str(csv_path.resolve()),
            "figure": str(figure_path.resolve()),
        },
        "interpretation_boundary": (
            "Oracle alternative-specific power for fixed covariance-selected directions; "
            "the two realized held-out days supply observed statistics only."
        ),
    }
    atomic_json(json_path, summary)
    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "figure": str(figure_path)}))


if __name__ == "__main__":
    main()
