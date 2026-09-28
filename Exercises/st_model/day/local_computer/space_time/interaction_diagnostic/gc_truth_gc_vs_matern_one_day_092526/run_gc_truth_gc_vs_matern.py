#!/usr/bin/env python3
"""Simulate one GC day by FFT, fit GC/Matérn, and compare cross terms."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[6]
SRC = PROJECT_ROOT / "src"
SIMULATE_DATA = PROJECT_ROOT / "simulate_data"
AMAREL_STUDY = (
    PROJECT_ROOT
    / "Exercises/st_model/day/amarel_simulation/space_time/interaction_diagnostic"
    / "fixed_geographic_three_model_092426"
)
for path in (SRC, SIMULATE_DATA, AMAREL_STUDY):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from GEMS_TCO.data import ProcessedDataLoader  # noqa: E402
from fixed_geo_three_model_core import (  # noqa: E402
    atomic_csv,
    atomic_json,
    atomic_text,
    clean_json,
    contrast_covariances,
    expected_gaussian_score,
    load_json,
    load_toml,
    prepare_fixed_contrasts,
    score_model,
    select_day_frames,
    sha256_file,
    task_source_files,
)
from generate_one_day_gc_fft_local import (  # noqa: E402
    embedding_contrast_covariances,
    generate_day,
    truth_physical,
)
from run_fixed_geo_three_model_day import (  # noqa: E402
    _build_model,
    _fit_one,
    _git_revision,
)


LOGGER = logging.getLogger("gc_truth_one_day")
CONFIG_PATH = HERE / "gc_truth_gc_vs_matern.toml"
DESIGN_PATH = AMAREL_STUDY / "frozen_design.json"
GENERATOR_PATH = SIMULATE_DATA / "generate_one_day_gc_fft_local.py"
MODEL_NAMES = ("gc", "matern05")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-root", type=Path, default=HERE / "outputs")
    result.add_argument(
        "--regenerate",
        action="store_true",
        help="replace the deterministic generated day before fitting",
    )
    result.add_argument(
        "--prepare-only",
        action="store_true",
        help="generate data, audit truth/contrasts, and instantiate models without fitting",
    )
    result.add_argument(
        "--summary-only",
        action="store_true",
        help="rebuild the comparison/report from existing model_summary.csv",
    )
    result.add_argument("--log-level", default="INFO")
    return result


def _physical_from_config(config: Mapping[str, Any]) -> dict[str, float]:
    values = config["simulation"]
    return {
        "signal_variance": float(values["signal_variance"]),
        "range_lat": float(values["range_lat"]),
        "range_lon": float(values["range_lon"]),
        "range_time": float(values["range_time"]),
        "advec_lat": float(values["advec_lat"]),
        "advec_lon": float(values["advec_lon"]),
        "nugget": float(values["nugget"]),
    }


def _validate_config(config: Mapping[str, Any], design: Mapping[str, Any]) -> None:
    if tuple(config["models"]["names"]) != MODEL_NAMES:
        raise ValueError(f"models.names must be exactly {list(MODEL_NAMES)}")
    if config["models"]["nugget_policy"] != "fixed_zero":
        raise ValueError("this comparison requires the same fixed-zero nugget policy")
    if float(config["simulation"]["nugget"]) != 0.0:
        raise ValueError("the GC truth must have zero nugget")
    if not np.isclose(
        float(config["simulation"]["gc_alpha"]),
        float(config["models"]["gc_alpha"]),
    ) or not np.isclose(
        float(config["simulation"]["gc_beta"]),
        float(config["models"]["gc_beta"]),
    ):
        raise ValueError("simulation and fitted GC shape parameters must match")
    if design["coordinate_frame"] != "fixed_geographic" or design[
        "observation_geometry_moves_with_advection"
    ]:
        raise ValueError("the frozen diagnostic must keep geographic endpoints fixed")
    if str(config["models"]["conditioning_lag_pattern"]) != "4/3/2":
        raise ValueError("this local experiment is frozen to corridor 4/3/2")


def _simulation_paths(output_root: Path, date: str) -> tuple[Path, Path]:
    simulation_dir = output_root / "simulation"
    prefix = f"gc_fft_{date}"
    return (
        simulation_dir / f"{prefix}_gridded.pkl",
        simulation_dir / f"{prefix}_truth.json",
    )


def _ensure_simulation(
    config: Mapping[str, Any], output_root: Path, regenerate: bool
) -> tuple[Path, Path]:
    values = config["simulation"]
    date = str(config["study"]["date"])
    generated = generate_day(
        date=date,
        input_root=Path(config["data"]["local_root"]).expanduser().resolve(),
        output_dir=(output_root / "simulation").resolve(),
        seed=int(values["seed"]),
        physical=_physical_from_config(config),
        gc_alpha=float(values["gc_alpha"]),
        gc_beta=float(values["gc_beta"]),
        mean_intercept=float(values["mean_intercept"]),
        mean_lat_slope=float(values["mean_lat_slope"]),
        mean_lat_center=float(values["mean_lat_center"]),
        lat_factor_hr=int(values["lat_factor_hr"]),
        lon_factor_hr=int(values["lon_factor_hr"]),
        pad=float(values["pad"]),
        embedding_spatial_factor=int(values["embedding_spatial_factor"]),
        embedding_temporal_factor=int(values["embedding_temporal_factor"]),
        max_negative_spectral_mass=float(values["max_negative_spectral_mass"]),
        overwrite=regenerate,
    )
    return generated.gridded, generated.truth


def _monthly_mean(frames: Mapping[str, pd.DataFrame]) -> float:
    values = np.concatenate(
        [
            pd.to_numeric(frame["ColumnAmountO3"], errors="coerce").to_numpy(
                dtype=np.float64
            )
            for frame in frames.values()
        ]
    )
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("generated day has no finite responses")
    return float(np.mean(finite))


def _unique_paths(paths: list[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return result


def _signature(data_path: Path, truth_path: Path) -> tuple[str, dict[str, str]]:
    native_binaries = sorted((SRC / "GEMS_TCO").glob("_vecchia_covariance_*.so"))
    sources = _unique_paths(
        [
            CONFIG_PATH,
            DESIGN_PATH,
            GENERATOR_PATH,
            Path(__file__),
            data_path,
            truth_path,
            *task_source_files(),
            *native_binaries,
        ]
    )
    digest = hashlib.sha256()
    source_hashes: dict[str, str] = {}
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            label = path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            label = str(path)
        value = sha256_file(path)
        source_hashes[label] = value
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest(), source_hashes


def _validate_truth(
    truth: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, float]:
    physical = truth_physical(truth)
    expected = _physical_from_config(config)
    for name, expected_value in expected.items():
        if not np.isclose(physical[name], expected_value, rtol=0.0, atol=1.0e-12):
            raise ValueError(
                f"truth {name}={physical[name]} does not match config {expected_value}"
            )
    if not np.isclose(
        float(truth["cauchy_a"]),
        float(config["models"]["gc_alpha"]),
        rtol=0.0,
        atol=1.0e-12,
    ) or not np.isclose(
        float(truth["cauchy_b"]),
        float(config["models"]["gc_beta"]),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("truth and fitted GC shape parameters differ")
    hours = np.asarray(truth["hours_elapsed"], dtype=np.float64)
    if hours.size != int(config["study"]["required_time_slots"]) or not np.array_equal(
        np.diff(np.rint(hours)).astype(np.int64),
        np.ones(hours.size - 1, dtype=np.int64),
    ):
        raise ValueError("truth Hours_elapsed do not follow the frozen eight-hour contract")
    return physical


def _covariance_error_fields(
    candidate: np.ndarray, truth: np.ndarray, prefix: str
) -> dict[str, float]:
    candidate = np.asarray(candidate, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    difference = candidate - truth
    pooled = np.mean(candidate, axis=0) - np.mean(truth, axis=0)
    cross = difference[:, 0, 1]
    frobenius_squared = (
        np.square(difference[:, 0, 0])
        + np.square(difference[:, 1, 1])
        + 2.0 * np.square(cross)
    )
    return {
        f"{prefix}_pooled_c_ab_abs_error": abs(float(pooled[0, 1])),
        f"{prefix}_pooled_v_a_abs_error": abs(float(pooled[0, 0])),
        f"{prefix}_pooled_v_b_abs_error": abs(float(pooled[1, 1])),
        f"{prefix}_pooled_covariance_frobenius_error": float(
            np.sqrt(pooled[0, 0] ** 2 + pooled[1, 1] ** 2 + 2.0 * pooled[0, 1] ** 2)
        ),
        f"{prefix}_pointwise_c_ab_rmse": float(np.sqrt(np.mean(np.square(cross)))),
        f"{prefix}_pointwise_covariance_frobenius_rmse": float(
            np.sqrt(np.mean(frobenius_squared))
        ),
        f"{prefix}_pointwise_c_ab_mean_error": float(np.mean(cross)),
    }


def _population_fields(
    candidate: np.ndarray, analytic_truth: np.ndarray, fft_truth: np.ndarray
) -> dict[str, float]:
    analytic_oracle = float(
        np.mean(expected_gaussian_score(analytic_truth, analytic_truth))
    )
    fft_oracle = float(np.mean(expected_gaussian_score(fft_truth, fft_truth)))
    analytic_score = float(np.mean(expected_gaussian_score(analytic_truth, candidate)))
    fft_score = float(np.mean(expected_gaussian_score(fft_truth, candidate)))
    return {
        "analytic_population_score": analytic_score,
        "analytic_population_regret": analytic_score - analytic_oracle,
        "fft_population_score": fft_score,
        "fft_population_regret": fft_score - fft_oracle,
    }


def _candidate_row(
    *,
    date: str,
    label: str,
    samples: pd.DataFrame,
    covariance: np.ndarray,
    analytic_truth: np.ndarray,
    fft_truth: np.ndarray,
    design: Mapping[str, Any],
    fit_record: Mapping[str, Any] | None = None,
    fit_physical: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    predicted, score_summary = score_model(samples, covariance, label, design)
    row: dict[str, Any] = {
        "date": date,
        **score_summary,
        **_covariance_error_fields(covariance, analytic_truth, "analytic_truth"),
        **_covariance_error_fields(covariance, fft_truth, "fft_truth"),
        **_population_fields(covariance, analytic_truth, fft_truth),
        "vecchia_nll": None,
        "fit_seconds": None,
        "fit_max_abs_gradient": None,
        "fit_converged": None,
        "resolved_covariance_backend": None,
    }
    if fit_record is not None:
        row.update(
            {
                "vecchia_nll": fit_record["profiled_vecchia_nll_per_target"],
                "fit_seconds": fit_record["fit_seconds"],
                "fit_max_abs_gradient": fit_record["maximum_absolute_gradient"],
                "fit_converged": fit_record["converged_at_outer_grad_tol"],
                "resolved_covariance_backend": fit_record[
                    "resolved_covariance_backend"
                ],
            }
        )
    if fit_physical is not None:
        row.update(
            {
                f"fit_{name}": value
                for name, value in fit_physical.items()
                if isinstance(value, (int, float))
            }
        )
    return predicted, row


def _comparison(summary: pd.DataFrame) -> pd.DataFrame:
    if "model" not in summary or summary["model"].isna().any():
        raise ValueError("model summary has missing model labels")
    expected = {"truth_gc_analytic", "truth_gc_fft", *MODEL_NAMES}
    if len(summary) != len(expected) or set(summary["model"]) != expected:
        raise ValueError(
            "model summary must contain exactly one row for each of "
            f"{sorted(expected)}"
        )
    if not summary["model"].is_unique:
        raise ValueError("model summary contains duplicate model rows")
    if summary["date"].astype(str).nunique() != 1:
        raise ValueError("model summary must contain exactly one common date")
    if summary["sample_count"].astype(int).nunique() != 1:
        raise ValueError("model summary rows disagree on sample_count")
    indexed = summary.set_index("model")
    analytic = indexed.loc["truth_gc_analytic"]
    fft = indexed.loc["truth_gc_fft"]
    gc_row = indexed.loc["gc"]
    matern = indexed.loc["matern05"]
    row = {
        "date": str(fft["date"]),
        "sample_count": int(fft["sample_count"]),
        "empirical_c_ab": float(fft["empirical_cross_moment_q_a_q_b"]),
        "analytic_truth_c_ab": float(analytic["model_c_ab"]),
        "fft_truth_c_ab": float(fft["model_c_ab"]),
        "fft_minus_analytic_truth_c_ab": float(fft["model_c_ab"])
        - float(analytic["model_c_ab"]),
        "gc_c_ab": float(gc_row["model_c_ab"]),
        "matern_c_ab": float(matern["model_c_ab"]),
        "gc_abs_c_ab_error_to_fft_truth": float(
            gc_row["fft_truth_pooled_c_ab_abs_error"]
        ),
        "matern_abs_c_ab_error_to_fft_truth": float(
            matern["fft_truth_pooled_c_ab_abs_error"]
        ),
        "gc_abs_c_ab_error_to_analytic_truth": float(
            gc_row["analytic_truth_pooled_c_ab_abs_error"]
        ),
        "matern_abs_c_ab_error_to_analytic_truth": float(
            matern["analytic_truth_pooled_c_ab_abs_error"]
        ),
        "analytic_truth_var_l_cross": float(analytic["model_var_l_cross"]),
        "fft_truth_var_l_cross": float(fft["model_var_l_cross"]),
        "gc_var_l_cross": float(gc_row["model_var_l_cross"]),
        "matern_var_l_cross": float(matern["model_var_l_cross"]),
        "gc_fft_population_regret": float(gc_row["fft_population_regret"]),
        "matern_fft_population_regret": float(matern["fft_population_regret"]),
        "gc_analytic_population_regret": float(gc_row["analytic_population_regret"]),
        "matern_analytic_population_regret": float(
            matern["analytic_population_regret"]
        ),
        "gc_empirical_score": float(gc_row["mean_contrast_score"]),
        "matern_empirical_score": float(matern["mean_contrast_score"]),
        "gc_vecchia_nll": float(gc_row["vecchia_nll"]),
        "matern_vecchia_nll": float(matern["vecchia_nll"]),
        "gc_fit_seconds": float(gc_row["fit_seconds"]),
        "matern_fit_seconds": float(matern["fit_seconds"]),
    }
    return pd.DataFrame([row])


def _report(comparison: pd.DataFrame, truth: Mapping[str, Any], output_root: Path) -> None:
    row = comparison.iloc[0]
    gc_fft_error = float(row["gc_abs_c_ab_error_to_fft_truth"])
    matern_fft_error = float(row["matern_abs_c_ab_error_to_fft_truth"])
    gc_analytic_error = float(row["gc_abs_c_ab_error_to_analytic_truth"])
    matern_analytic_error = float(row["matern_abs_c_ab_error_to_analytic_truth"])
    fft_winner = "GC" if gc_fft_error < matern_fft_error else "Matérn"
    analytic_winner = "GC" if gc_analytic_error < matern_analytic_error else "Matérn"
    diagnostic = truth["embedding_diagnostics"]
    lines = [
        "# One-day GC-truth FFT audit: GC versus Matérn",
        "",
        "A generalized-Cauchy field was generated for one eight-hour GEMS day. ",
        "The FFT is applied to a zero-advection comoving field and observations ",
        "are sampled at `s - v(t-t0)`. This is covariance-equivalent to constant ",
        "advection while avoiding the non-centrosymmetric Nyquist artifact in the ",
        "historical direct-advected embedding.",
        "",
        "The primary population target is the effective, post-clipping and variance-",
        "renormalized FFT covariance—the covariance that actually generated the field. ",
        "The intended analytic GC covariance is reported as a sensitivity target.",
        "",
        "| quantity | value |",
        "|---|---:|",
        f"| analytic truth C_AB | {float(row['analytic_truth_c_ab']):.9g} |",
        f"| effective FFT truth C_AB | {float(row['fft_truth_c_ab']):.9g} |",
        f"| empirical one-field C_AB | {float(row['empirical_c_ab']):.9g} |",
        f"| fitted GC C_AB | {float(row['gc_c_ab']):.9g} |",
        f"| fitted Matérn C_AB | {float(row['matern_c_ab']):.9g} |",
        f"| fitted GC abs(C_AB - FFT truth) | {gc_fft_error:.9g} |",
        f"| fitted Matérn abs(C_AB - FFT truth) | {matern_fft_error:.9g} |",
        f"| fitted GC abs(C_AB - analytic truth) | {gc_analytic_error:.9g} |",
        f"| fitted Matérn abs(C_AB - analytic truth) | {matern_analytic_error:.9g} |",
        f"| analytic truth 2*d1*d2*C_AB | {float(row['analytic_truth_var_l_cross']):.9g} |",
        f"| effective FFT truth 2*d1*d2*C_AB | {float(row['fft_truth_var_l_cross']):.9g} |",
        f"| fitted GC 2*d1*d2*C_AB | {float(row['gc_var_l_cross']):.9g} |",
        f"| fitted Matérn 2*d1*d2*C_AB | {float(row['matern_var_l_cross']):.9g} |",
        "",
        f"- Smaller C_AB error to effective FFT truth: **{fft_winner}**.",
        f"- Smaller C_AB error to intended analytic truth: **{analytic_winner}**.",
        "- Effective-FFT population score regret, GC versus Matérn: "
        f"`{float(row['gc_fft_population_regret']):.9g}` versus "
        f"`{float(row['matern_fft_population_regret']):.9g}`.",
        "- Analytic-GC population score regret, GC versus Matérn: "
        f"`{float(row['gc_analytic_population_regret']):.9g}` versus "
        f"`{float(row['matern_analytic_population_regret']):.9g}`.",
        "- Observed one-field contrast score, GC versus Matérn: "
        f"`{float(row['gc_empirical_score']):.9g}` versus "
        f"`{float(row['matern_empirical_score']):.9g}`.",
        "- Vecchia NLL per target, GC versus Matérn: "
        f"`{float(row['gc_vecchia_nll']):.9g}` versus "
        f"`{float(row['matern_vecchia_nll']):.9g}`.",
        "",
        "## FFT integrity",
        "",
        f"- Simulation grid: `{diagnostic['simulation_grid_shape']}`; embedding: "
        f"`{diagnostic['embedding_shape']}`.",
        "- Negative spectral mass fraction before correction: "
        f"`{float(diagnostic['spectrum_negative_mass_fraction']):.9g}`.",
        "- Maximum imaginary/real spectrum ratio: "
        f"`{float(diagnostic['spectrum_max_abs_imaginary_over_max_abs_real']):.9g}`.",
        "- Variance before renormalization and scale: "
        f"`{float(diagnostic['variance_before_renormalization']):.9g}`, "
        f"`{float(diagnostic['variance_renormalization_scale']):.9g}`.",
        "- FFT minus analytic truth C_AB: "
        f"`{float(row['fft_minus_analytic_truth_c_ab']):.9g}`.",
        "",
        "## Interpretation boundary",
        "",
        "This is a one-realization mechanism audit, not a power study or a general ",
        "claim that GC always estimates cross covariance better. The empirical C_AB ",
        "is noisy because overlapping within-day filters are not independent. Model ",
        "recovery should therefore be read primarily against the two population-truth ",
        "columns and their population score regrets.",
        "",
    ]
    atomic_text(output_root / "RESULTS.md", "\n".join(lines))


def _read_existing(output_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    summary_path = output_root / "model_summary.csv"
    manifest_path = output_root / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = load_json(manifest_path)
    date = str(manifest["date"])
    _, truth_path = _simulation_paths(output_root, date)
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    if not truth_path.is_file():
        raise FileNotFoundError(truth_path)
    summary = pd.read_csv(summary_path, float_precision="round_trip")
    truth = load_json(truth_path)
    if summary["date"].astype(str).nunique() != 1 or str(summary["date"].iloc[0]) != date:
        raise RuntimeError("model summary date does not match run manifest")
    if str(truth["date"]) != date:
        raise RuntimeError("simulation truth date does not match run manifest")
    return summary, truth


def main() -> None:
    args = parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.summary_only and (args.prepare_only or args.regenerate):
        raise ValueError("--summary-only cannot be combined with generation/prepare flags")
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.summary_only:
        summary, truth = _read_existing(output_root)
        comparison = _comparison(summary)
        atomic_csv(output_root / "cross_term_comparison.csv", comparison)
        _report(comparison, truth, output_root)
        print(comparison.to_string(index=False), flush=True)
        return

    config = load_toml(CONFIG_PATH)
    design = load_json(DESIGN_PATH)
    _validate_config(config, design)
    date = str(config["study"]["date"])
    manifest_path = output_root / "run_manifest.json"
    if args.regenerate and manifest_path.is_file():
        raise RuntimeError(
            "--regenerate cannot overwrite an output root that already contains a run "
            "manifest; choose a fresh --output-root"
        )
    data_path, truth_path = _ensure_simulation(config, output_root, args.regenerate)
    truth = load_json(truth_path)
    physical_truth = _validate_truth(truth, config)
    signature, source_hashes = _signature(data_path, truth_path)

    if manifest_path.is_file():
        previous = load_json(manifest_path)
        if previous.get("study_signature") != signature:
            raise RuntimeError(
                f"refusing to mix source/configuration signatures in {output_root}; "
                "choose a fresh --output-root"
            )
    (output_root / "COMPLETE").unlink(missing_ok=True)

    frames = pd.read_pickle(data_path)
    if not isinstance(frames, dict):
        raise TypeError(f"{data_path} is not a dict pickle")
    day_frames = select_day_frames(
        frames,
        day=int(pd.Timestamp(date).day),
        expected_slots=int(config["study"]["required_time_slots"]),
    )
    monthly_mean = _monthly_mean(day_frames)
    samples, sample_coordinates, mean_audit = prepare_fixed_contrasts(
        day_frames,
        monthly_mean,
        design,
        wx_tolerance=float(config["diagnostic"]["wx_tolerance"]),
    )
    atomic_csv(output_root / "empirical_contrasts.csv.gz", samples)
    atomic_json(output_root / "mean_and_geometry_audit.json", clean_json(mean_audit))

    analytic_truth = contrast_covariances(
        sample_coordinates,
        physical_truth,
        "gc",
        design,
        gc_alpha=float(config["models"]["gc_alpha"]),
        gc_beta=float(config["models"]["gc_beta"]),
    )
    fft_truth, fft_mapping_audit = embedding_contrast_covariances(
        sample_coordinates, truth, design
    )
    analytic_predicted, analytic_row = _candidate_row(
        date=date,
        label="truth_gc_analytic",
        samples=samples,
        covariance=analytic_truth,
        analytic_truth=analytic_truth,
        fft_truth=fft_truth,
        design=design,
        fit_physical=physical_truth,
    )
    fft_predicted, fft_row = _candidate_row(
        date=date,
        label="truth_gc_fft",
        samples=samples,
        covariance=fft_truth,
        analytic_truth=analytic_truth,
        fft_truth=fft_truth,
        design=design,
        fit_physical=physical_truth,
    )
    atomic_csv(
        output_root / "truth_gc_analytic_predicted_contrast_covariances.csv.gz",
        analytic_predicted,
    )
    atomic_csv(
        output_root / "truth_gc_fft_predicted_contrast_covariances.csv.gz",
        fft_predicted,
    )

    manifest = {
        "study_signature": signature,
        "study": str(config["study"]["name"]),
        "date": date,
        "git_revision": _git_revision(),
        "config": str(CONFIG_PATH),
        "frozen_design": str(DESIGN_PATH),
        "simulation_data": str(data_path),
        "simulation_truth": str(truth_path),
        "frozen_design_identifier": str(design["design_id"]),
        "models": list(MODEL_NAMES),
        "vecchia_conditioning_geometry": str(
            config["models"]["conditioning_geometry"]
        ),
        "vecchia_lag_pattern": str(config["models"]["conditioning_lag_pattern"]),
        "target_chunk_sizes": clean_json(config["models"]["target_chunk_sizes"]),
        "source_sha256": source_hashes,
        "fft_mapping_audit": fft_mapping_audit,
        "embedding_diagnostics": truth["embedding_diagnostics"],
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }
    atomic_json(manifest_path, clean_json(manifest))

    loader = ProcessedDataLoader(output_root / "simulation")
    model_input_cpu, _ = loader.build_model_tensors(
        day_frames,
        ozone_mean=monthly_mean,
        time_slice=(0, int(config["study"]["required_time_slots"])),
        dtype=torch.float64,
        use_source_coordinates=bool(
            config["data"]["use_source_coordinates_for_covariance"]
        ),
        time_origin_hours=float(design["model_time_origin_hours"]),
    )
    first_frame = next(iter(day_frames.values()))
    grid_coordinates = first_frame[["Latitude", "Longitude"]].to_numpy(dtype=np.float64)

    if args.prepare_only:
        preflight: dict[str, Any] = {}
        for model_name in MODEL_NAMES:
            model = _build_model(model_name, model_input_cpu, grid_coordinates, config)
            preflight[model_name] = {
                "covariance_parameter_count": int(model.covariance_parameter_count),
                "target_chunk_size": int(model.target_chunk_size),
                "resolved_covariance_backend": model.resolved_covariance_backend(),
            }
            if model.covariance_parameter_count != int(
                config["models"]["covariance_parameter_count"]
            ):
                raise RuntimeError(f"unexpected {model_name} covariance parameter count")
            if model_name == "gc" and model.resolved_covariance_backend() != "native":
                raise RuntimeError("configured GC native CPU covariance backend is unavailable")
            del model
        atomic_json(output_root / "model_preflight.json", preflight)
        atomic_text(output_root / "PREPARE_COMPLETE", "complete\n")
        print(json.dumps(clean_json(preflight), indent=2), flush=True)
        return

    rows: list[dict[str, Any]] = [analytic_row, fft_row]
    for model_name in MODEL_NAMES:
        model_dir = output_root / f"model_{model_name}"
        (model_dir / "COMPLETE").unlink(missing_ok=True)
        model = _build_model(model_name, model_input_cpu, grid_coordinates, config)
        if model.covariance_parameter_count != int(
            config["models"]["covariance_parameter_count"]
        ):
            raise RuntimeError(f"unexpected {model_name} covariance parameter count")
        if model_name == "gc" and model.resolved_covariance_backend() != "native":
            raise RuntimeError("configured GC native CPU covariance backend is unavailable")
        metadata = {
            "date": date,
            "study_signature": signature,
            "model": model_name,
            "frozen_design_identifier": str(design["design_id"]),
            "vecchia_conditioning_geometry": str(
                config["models"]["conditioning_geometry"]
            ),
            "vecchia_lag_pattern": str(config["models"]["conditioning_lag_pattern"]),
            "vecchia_target_chunk_size": int(model.target_chunk_size),
            "git_revision": _git_revision(),
        }
        fit_record = _fit_one(
            model_name,
            model,
            config,
            model_dir,
            signature,
            metadata,
        )
        fit_physical = fit_record["interpretable_parameters"]
        fitted_covariance = contrast_covariances(
            sample_coordinates,
            fit_physical,
            model_name,
            design,
            gc_alpha=float(config["models"]["gc_alpha"]),
            gc_beta=float(config["models"]["gc_beta"]),
        )
        predicted, row = _candidate_row(
            date=date,
            label=model_name,
            samples=samples,
            covariance=fitted_covariance,
            analytic_truth=analytic_truth,
            fft_truth=fft_truth,
            design=design,
            fit_record=fit_record,
            fit_physical=fit_physical,
        )
        atomic_csv(model_dir / "predicted_contrast_covariances.csv.gz", predicted)
        atomic_json(model_dir / "truth_error_summary.json", clean_json(row))
        atomic_text(model_dir / "COMPLETE", "complete\n")
        rows.append(row)
        del model, fitted_covariance
        gc.collect()

    summary = pd.DataFrame(rows)
    comparison = _comparison(summary)
    atomic_csv(output_root / "model_summary.csv", summary)
    atomic_csv(output_root / "cross_term_comparison.csv", comparison)
    _report(comparison, truth, output_root)
    atomic_text(output_root / "COMPLETE", "complete\n")
    print(comparison.to_string(index=False), flush=True)
    print(f"Wrote {output_root / 'RESULTS.md'}", flush=True)


if __name__ == "__main__":
    main()
