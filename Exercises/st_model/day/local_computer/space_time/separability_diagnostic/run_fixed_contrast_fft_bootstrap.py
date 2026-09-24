#!/usr/bin/env python3
"""FFT/circulant joint-GC pilot for the frozen GEMS contrast.

The latent residual field is generated on a complete regular lattice from the
fitted covariance, sampled at the observed GEMS source coordinates, and then
masked with the original 2024-07-01 missingness pattern.  Every replicate is
re-fitted with the unchanged 4/3/2 corridor-Vecchia procedure and GLS mean.

This driver never silently calls a clipped embedding exact.  It records the
negative spectral mass, clips negative circulant eigenvalues, renormalizes the
zero-lag variance, and audits the resulting covariance against the analytic
target at all requested lags.  If no eigenvalue is negative (up to numerical
tolerance), the same code path is an exact circulant-embedding simulation.
"""

from __future__ import annotations

import argparse
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

from apply_fixed_canonical_contrast_real_gems import (
    DEFAULT_DATA_FILE,
    DEFAULT_FIT_CSV,
    DEFAULT_ORACLE_DIR,
    _atomic_csv,
    _atomic_text,
    _canonical_samples,
    _load_filtered_frames,
    _summaries,
)
from run_fixed_contrast_full_pipeline_bootstrap import (
    _device,
    _fit_replicate,
    _fit_series,
    _observed_summary,
    _raw_parameters,
    _replicate_source_map,
    _residual_cube,
    _saved_mean_vector,
    _source_map,
)


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = DEFAULT_ORACLE_DIR / "fft_full_pipeline_bootstrap_20240701"
BASE_LAT_STEP = 0.044
BASE_LON_STEP = 0.063


@dataclass(frozen=True)
class CirculantPlan:
    dgp: str
    lats: np.ndarray
    lons: np.ndarray
    time_steps: int
    embedding_shape: tuple[int, int, int]
    sqrt_spectrum: np.ndarray
    corrected_covariance: np.ndarray
    diagnostics: dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--fit-csv", type=Path, default=DEFAULT_FIT_CSV)
    parser.add_argument("--oracle-dir", type=Path, default=DEFAULT_ORACLE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--strategy", default="standard_432")
    parser.add_argument("--day-index", type=int, default=0)
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--dgp", choices=("joint", "separable", "both"), default="joint")
    parser.add_argument("--seed", type=int, default=20260925)
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
    parser.add_argument("--lat-factor-hr", type=int, default=1)
    parser.add_argument("--lon-factor-hr", type=int, default=1)
    parser.add_argument("--hr-pad", type=float, default=0.1)
    parser.add_argument("--embedding-spatial-factor", type=int, default=2)
    parser.add_argument("--embedding-temporal-factor", type=int, default=2)
    parser.add_argument("--spectral-zero-tolerance", type=float, default=1.0e-10)
    parser.add_argument(
        "--max-negative-spectral-mass",
        type=float,
        default=0.10,
        help="Abort if clipping removes more than this fraction of |spectrum|.",
    )
    return parser


def _high_resolution_axes(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    if args.lat_factor_hr < 1 or args.lon_factor_hr < 1:
        raise ValueError("high-resolution factors must be positive integers")
    dlat = BASE_LAT_STEP / int(args.lat_factor_hr)
    dlon = BASE_LON_STEP / int(args.lon_factor_hr)
    lats = np.arange(
        float(args.latitude_min) - float(args.hr_pad),
        float(args.latitude_max) + float(args.hr_pad) + 0.5 * dlat,
        dlat,
        dtype=np.float64,
    )
    lons = np.arange(
        float(args.longitude_min) - float(args.hr_pad),
        float(args.longitude_max) + float(args.hr_pad) + 0.5 * dlon,
        dlon,
        dtype=np.float64,
    )
    return lats, lons


def _gc_correlation(distance: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    if not 0.0 < alpha <= 2.0:
        raise ValueError("GC alpha must be in (0, 2]")
    if beta <= 0.0:
        raise ValueError("GC beta must be positive")
    return np.power(1.0 + np.power(np.maximum(distance, 0.0), alpha), -beta / alpha)


def _analytic_covariance(
    dlat: np.ndarray | float,
    dlon: np.ndarray | float,
    dt: np.ndarray | float,
    fit: pd.Series,
    dgp: str,
) -> np.ndarray:
    dlat_array = np.asarray(dlat, dtype=np.float64)
    dlon_array = np.asarray(dlon, dtype=np.float64)
    dt_array = np.asarray(dt, dtype=np.float64)
    spatial = np.sqrt(
        np.square((dlat_array - float(fit["est_advec_lat"]) * dt_array) / float(fit["est_range_lat"]))
        + np.square((dlon_array - float(fit["est_advec_lon"]) * dt_array) / float(fit["est_range_lon"]))
    )
    alpha = float(fit["gc_alpha"])
    beta = float(fit["gc_beta"])
    if dgp == "joint":
        distance = np.sqrt(
            np.square(spatial) + np.square(dt_array / float(fit["est_range_time"]))
        )
        correlation = _gc_correlation(distance, alpha, beta)
    elif dgp == "separable":
        temporal = np.abs(dt_array) / float(fit["est_range_time"])
        correlation = _gc_correlation(spatial, alpha, beta) * _gc_correlation(
            temporal, alpha, beta
        )
    else:
        raise ValueError(f"unsupported DGP: {dgp}")
    return float(fit["est_sigmasq"]) * correlation


def _wrapped_lags(length: int, spacing: float) -> np.ndarray:
    indices = np.arange(length, dtype=np.float64)
    indices[indices >= (length + 1) // 2] -= length
    return indices * float(spacing)


def _inverse_indices(length: int) -> np.ndarray:
    return (-np.arange(length, dtype=np.int64)) % length


def _circulant_plan(
    fit: pd.Series,
    dgp: str,
    lats: np.ndarray,
    lons: np.ndarray,
    time_steps: int,
    args: argparse.Namespace,
) -> CirculantPlan:
    spatial_factor = int(args.embedding_spatial_factor)
    temporal_factor = int(args.embedding_temporal_factor)
    if spatial_factor < 2 or temporal_factor < 2:
        raise ValueError("embedding factors must be at least 2")
    shape = (
        spatial_factor * len(lats),
        spatial_factor * len(lons),
        temporal_factor * int(time_steps),
    )
    dlat = float(BASE_LAT_STEP / int(args.lat_factor_hr))
    dlon = float(BASE_LON_STEP / int(args.lon_factor_hr))
    hlat = _wrapped_lags(shape[0], dlat)[:, None, None]
    hlon = _wrapped_lags(shape[1], dlon)[None, :, None]
    htime = _wrapped_lags(shape[2], 1.0)[None, None, :]
    covariance = _analytic_covariance(hlat, hlon, htime, fit, dgp)

    # At Nyquist faces a real BCCB first column must be centrally symmetric.
    # Away from those faces C(h,u)=C(-h,-u), so this changes only the ambiguous
    # wrapped boundary and removes the imaginary FFT component explicitly.
    inverse = np.ix_(*[_inverse_indices(size) for size in shape])
    symmetry_error = float(np.max(np.abs(covariance - covariance[inverse])))
    covariance = 0.5 * (covariance + covariance[inverse])
    spectrum_complex = np.fft.rfftn(covariance)
    max_imaginary = float(np.max(np.abs(spectrum_complex.imag)))
    spectrum = spectrum_complex.real
    negative = spectrum < -float(args.spectral_zero_tolerance)
    frequency_weights = np.full(spectrum.shape[-1], 2.0, dtype=np.float64)
    frequency_weights[0] = 1.0
    if shape[-1] % 2 == 0:
        frequency_weights[-1] = 1.0
    weighted = frequency_weights.reshape((1, 1, -1))
    negative_sum = float((np.abs(spectrum) * negative * weighted).sum())
    absolute_sum = float((np.abs(spectrum) * weighted).sum())
    negative_mass = negative_sum / absolute_sum if absolute_sum > 0.0 else 0.0
    if negative_mass > float(args.max_negative_spectral_mass):
        raise RuntimeError(
            f"{dgp} embedding negative spectral mass {negative_mass:.6g} exceeds "
            f"limit {float(args.max_negative_spectral_mass):.6g}"
        )
    clipped = np.maximum(spectrum, 0.0)
    corrected_before_scale = np.fft.irfftn(clipped, s=shape).real
    variance_before_scale = float(corrected_before_scale[0, 0, 0])
    target_variance = float(fit["est_sigmasq"])
    variance_scale = target_variance / variance_before_scale
    clipped *= variance_scale
    corrected = np.fft.irfftn(clipped, s=shape).real
    exact = bool(not np.any(negative))
    diagnostics: dict[str, Any] = {
        "dgp": dgp,
        "n_lat": len(lats),
        "n_lon": len(lons),
        "time_steps": int(time_steps),
        "embedding_lat": shape[0],
        "embedding_lon": shape[1],
        "embedding_time": shape[2],
        "embedding_size": int(np.prod(shape)),
        "spectrum_min_before_clip": float(spectrum.min()),
        "spectrum_max_before_clip": float(spectrum.max()),
        "spectrum_negative_count": int((negative * weighted).sum()),
        "spectrum_negative_fraction": float(
            (negative * weighted).sum() / np.prod(shape)
        ),
        "spectrum_negative_mass_fraction": negative_mass,
        "variance_before_renormalization": variance_before_scale,
        "variance_target": target_variance,
        "variance_renormalization_scale": variance_scale,
        "variance_after_renormalization": float(corrected[0, 0, 0]),
        "central_symmetry_max_adjustment_twice": symmetry_error,
        "fft_max_abs_imaginary_after_symmetrization": max_imaginary,
        "embedding_is_exact_nonnegative": exact,
        "generator_label": (
            "exact_circulant_embedding"
            if exact
            else "spectrally_corrected_circulant_embedding"
        ),
    }
    return CirculantPlan(
        dgp=dgp,
        lats=lats.copy(),
        lons=lons.copy(),
        time_steps=int(time_steps),
        embedding_shape=shape,
        sqrt_spectrum=np.sqrt(clipped),
        corrected_covariance=corrected,
        diagnostics=diagnostics,
    )


def _lag_audit(
    plan: CirculantPlan, fit: pd.Series, args: argparse.Namespace
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    dlat_step = BASE_LAT_STEP / int(args.lat_factor_hr)
    dlon_step = BASE_LON_STEP / int(args.lon_factor_hr)
    max_lat = min(40 * int(args.lat_factor_hr), len(plan.lats) - 1)
    max_lon = min(60 * int(args.lon_factor_hr), len(plan.lons) - 1)
    for time_lag in (0, 1, 2):
        if time_lag >= plan.corrected_covariance.shape[2]:
            continue
        for lat_index in range(max_lat + 1):
            for lon_index in range(max_lon + 1):
                hlat = lat_index * dlat_step
                hlon = lon_index * dlon_step
                target = float(
                    _analytic_covariance(hlat, hlon, time_lag, fit, plan.dgp)
                )
                actual = float(
                    plan.corrected_covariance[lat_index, lon_index, time_lag]
                )
                rows.append(
                    {
                        "dgp": plan.dgp,
                        "lat_index_hr": lat_index,
                        "lon_index_hr": lon_index,
                        "time_lag": time_lag,
                        "lag_lat": hlat,
                        "lag_lon": hlon,
                        "analytic_covariance": target,
                        "generated_covariance": actual,
                        "signed_error": actual - target,
                        "absolute_error": abs(actual - target),
                        "error_over_signal_variance": (
                            actual - target
                        )
                        / float(fit["est_sigmasq"]),
                        "relative_error": (
                            (actual - target) / target if target != 0.0 else np.nan
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _simulate_field(plan: CirculantPlan, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(plan.embedding_shape)
    field = np.fft.irfftn(
        plan.sqrt_spectrum * np.fft.rfftn(white), s=plan.embedding_shape
    ).real
    return field[: len(plan.lats), : len(plan.lons), : plan.time_steps]


def _sample_at_observed_sources(
    base_map: dict[str, torch.Tensor], plan: CirculantPlan, field: np.ndarray
) -> np.ndarray:
    dlat = float(plan.lats[1] - plan.lats[0])
    dlon = float(plan.lons[1] - plan.lons[0])
    blocks: list[np.ndarray] = []
    for time_index, rows in enumerate(base_map.values()):
        array = rows.detach().cpu().numpy()
        values = np.zeros(len(array), dtype=np.float64)
        valid = (
            np.isfinite(array[:, 0])
            & np.isfinite(array[:, 1])
            & np.isfinite(array[:, 2])
        )
        lat_index = np.rint((array[valid, 0] - plan.lats[0]) / dlat).astype(
            np.int64
        )
        lon_index = np.rint((array[valid, 1] - plan.lons[0]) / dlon).astype(
            np.int64
        )
        if (
            np.any(lat_index < 0)
            or np.any(lat_index >= len(plan.lats))
            or np.any(lon_index < 0)
            or np.any(lon_index >= len(plan.lons))
        ):
            raise ValueError("an observed source location lies outside the FFT grid")
        values[valid] = field[lat_index, lon_index, time_index]
        blocks.append(values)
    return np.concatenate(blocks)


def _bootstrap_summary(results: pd.DataFrame, observed: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    observed_statistic = float(observed["empirical_over_separable_pooled"])
    for dgp, group in results.groupby("dgp", sort=True):
        values = group["statistic_empirical_over_separable"].to_numpy(dtype=float)
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
    axis.axhline(1.0, color="0.3", linestyle="--", linewidth=1.0)
    axis.set_ylabel("empirical mean L squared / refitted matched-separable Var(L)")
    axis.set_title("Frozen-contrast FFT/circulant full-pipeline bootstrap")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    figure.savefig(output_dir / "fft_full_pipeline_bootstrap.png", dpi=220)
    figure.savefig(output_dir / "fft_full_pipeline_bootstrap.pdf")
    plt.close(figure)


def _write_report(
    output_dir: Path,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    observed: pd.Series,
    embedding: pd.DataFrame,
    lag_audit: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    table_rows = []
    for row in summary.itertuples(index=False):
        table_rows.append(
            f"| `{row.dgp}` | {row.replicate_count} | {row.bootstrap_mean:.6f} | "
            f"{row.bootstrap_q025:.6f} | {row.bootstrap_q975:.6f} | "
            f"{row.lower_tail_p_value:.6f} |"
        )
    embedding_rows = []
    for row in embedding.itertuples(index=False):
        subset = lag_audit.loc[lag_audit["dgp"] == row.dgp]
        embedding_rows.append(
            f"| `{row.dgp}` | `{row.generator_label}` | "
            f"{row.spectrum_negative_mass_fraction:.6g} | "
            f"{subset['error_over_signal_variance'].abs().max():.6g} | "
            f"{np.sqrt(np.mean(np.square(subset['error_over_signal_variance']))):.6g} |"
        )
    minimum_count = int(results.groupby("dgp").size().min())
    lines = [
        "# Frozen-contrast FFT/circulant full-pipeline bootstrap",
        "",
        "The latent residual field was generated on a complete lattice, sampled at the original source coordinates, masked with the original O3 missingness pattern, and then re-fitted with the unchanged 4/3/2 corridor-Vecchia plus GLS pipeline. Geometry, temporal lag, coefficients, advection path, and grid rule were frozen.",
        "",
        "## Embedding validity",
        "",
        "| DGP | generator classification | negative spectral mass | max audited error / variance | RMSE / variance |",
        "|---|---|---:|---:|---:|",
        *embedding_rows,
        "",
        "A generator is called exact only when the unmodified embedding spectrum is nonnegative. Otherwise the report deliberately uses `spectrally_corrected_circulant_embedding`: negative eigenvalues were clipped and the spectrum was rescaled to restore the target marginal variance. See `embedding_diagnostics.csv` and `lag_covariance_audit.csv`; this approximation must be disclosed in any inferential use.",
        "",
        f"Observed statistic: `{float(observed['empirical_over_separable_pooled']):.8f}`.",
        "",
        "| generating model | replicates | mean | 2.5% | 97.5% | lower-tail p-value |",
        "|---|---:|---:|---:|---:|---:|",
        *table_rows,
        "",
    ]
    if minimum_count < 99:
        lines.extend(
            [
                "**Status: computational pilot only.** Fewer than 99 replicates per requested DGP are available, so the tail probabilities are not confirmatory.",
                "",
            ]
        )
    lines.extend(
        [
            "The earlier block-Vecchia-generated replicate is not pooled here; it remains a generator-sensitivity result. This directory contains only FFT/circulant-generated replicates.",
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
    if args.dgp != "joint":
        raise NotImplementedError(
            "The separable DGP must be re-fitted with a separable covariance model. "
            "The earlier pilot incorrectly re-fitted it with the joint-GC model and "
            "is retained only as a failed pipeline audit. Run the joint primary pilot "
            "here; do not use --dgp separable/both until that refit path is implemented."
        )
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
    raw = _raw_parameters(original_fit)
    saved_mean = _saved_mean_vector(base_map, original_fit)
    lats_hr, lons_hr = _high_resolution_axes(args)
    requested_dgps = ("joint", "separable") if args.dgp == "both" else (args.dgp,)

    plans: dict[str, CirculantPlan] = {}
    embedding_rows: list[dict[str, Any]] = []
    lag_tables: list[pd.DataFrame] = []
    for dgp in requested_dgps:
        plan = _circulant_plan(original_fit, dgp, lats_hr, lons_hr, 8, args)
        plans[dgp] = plan
        embedding_rows.append(plan.diagnostics)
        lag_tables.append(_lag_audit(plan, original_fit, args))
    embedding_table = pd.DataFrame(embedding_rows)
    lag_audit = pd.concat(lag_tables, ignore_index=True)
    _atomic_csv(output_dir / "embedding_diagnostics.csv", embedding_table)
    _atomic_csv(output_dir / "lag_covariance_audit.csv", lag_audit)

    existing_path = output_dir / "bootstrap_replicates.csv"
    existing = pd.read_csv(existing_path) if existing_path.is_file() else pd.DataFrame()
    result_rows = existing.to_dict("records") if not existing.empty else []
    for dgp_index, dgp in enumerate(requested_dgps):
        completed = (
            set(existing.loc[existing["dgp"] == dgp, "replicate"].astype(int))
            if not existing.empty and "dgp" in existing
            else set()
        )
        plan = plans[dgp]
        for replicate in range(int(args.replicates)):
            if replicate in completed:
                continue
            seed = int(args.seed) + 1_000_000 * dgp_index + replicate
            rng = np.random.default_rng(seed)
            started = time.perf_counter()
            field = _simulate_field(plan, rng)
            simulation_seconds = time.perf_counter() - started
            simulated = _sample_at_observed_sources(base_map, plan, field)
            replicate_map = _replicate_source_map(base_map, simulated, saved_mean)
            fit_result, beta, fitted_model, fit_seconds = _fit_replicate(
                replicate_map, grid_coordinates, original_fit, raw, args
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
                "generator": plan.diagnostics["generator_label"],
                "dgp": dgp,
                "replicate": replicate,
                "seed": seed,
                "sample_count": int(total["sample_count"]),
                "statistic_empirical_over_separable": float(total["empirical_over_separable_pooled"]),
                "statistic_empirical_over_joint": float(total["empirical_over_joint_pooled"]),
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
                "negative_spectral_mass_fraction": float(plan.diagnostics["spectrum_negative_mass_fraction"]),
                "simulation_seconds": simulation_seconds,
                "fit_seconds": fit_seconds,
                "replicate_seconds": time.perf_counter() - started,
            }
            for index, value in enumerate(beta.detach().cpu().numpy().reshape(-1)):
                row[f"beta_{index}"] = float(value)
            result_rows.append(row)
            current = pd.DataFrame(result_rows).sort_values(["dgp", "replicate"]).reset_index(drop=True)
            _atomic_csv(existing_path, current)
            print(
                f"generator={row['generator']} dgp={dgp} replicate={replicate} "
                f"Tsep={row['statistic_empirical_over_separable']:.6f} "
                f"sim_s={simulation_seconds:.2f} fit_s={fit_seconds:.1f}",
                flush=True,
            )
            del field, fitted_model, replicate_map

    results = pd.DataFrame(result_rows).sort_values(["dgp", "replicate"]).reset_index(drop=True)
    relevant = results.loc[results["dgp"].isin(requested_dgps)].copy()
    summary = _bootstrap_summary(relevant, observed)
    _atomic_csv(output_dir / "bootstrap_summary.csv", summary)
    _write_figure(relevant, observed, output_dir)
    manifest: dict[str, Any] = {
        "data_file": str(data_file),
        "fit_csv": str(fit_csv),
        "coefficient_source": str(coefficient_source),
        "day": str(original_fit["day"]),
        "strategy": str(original_fit["strategy"]),
        "dgp": args.dgp,
        "requested_replicates_per_dgp": int(args.replicates),
        "seed": int(args.seed),
        "device_for_refit": str(device),
        "optimizer": {
            "max_steps": int(args.max_steps),
            "max_eval": int(args.max_eval),
            "grad_tol": float(args.grad_tol),
            "tolerance_grad": float(args.tolerance_grad),
        },
        "fft_grid": {
            "lat_factor_hr": int(args.lat_factor_hr),
            "lon_factor_hr": int(args.lon_factor_hr),
            "base_lat_step": BASE_LAT_STEP,
            "base_lon_step": BASE_LON_STEP,
            "pad": float(args.hr_pad),
            "n_lat": len(lats_hr),
            "n_lon": len(lons_hr),
            "embedding_spatial_factor": int(args.embedding_spatial_factor),
            "embedding_temporal_factor": int(args.embedding_temporal_factor),
            "spectral_correction": "clip negative eigenvalues then rescale to target variance",
        },
        "mask_and_sampling": "nearest high-resolution cell at original source coordinates; preserve original O3 missingness exactly",
        "analysis_refit": "joint GC 4/3/2 corridor Vecchia plus GLS mean for both generating models",
        "frozen_geometry": {
            "range_lat": float(original_fit["est_range_lat"]),
            "range_lon": float(original_fit["est_range_lon"]),
            "advec_lat": float(original_fit["est_advec_lat"]),
            "advec_lon": float(original_fit["est_advec_lon"]),
            "temporal_lag": 1,
            "coefficient_first": coefficient_first,
            "coefficient_second": coefficient_second,
        },
    }
    _atomic_text(output_dir / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    _write_report(output_dir, relevant, summary, observed, embedding_table, lag_audit, manifest)
    print(embedding_table.to_string(index=False), flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
