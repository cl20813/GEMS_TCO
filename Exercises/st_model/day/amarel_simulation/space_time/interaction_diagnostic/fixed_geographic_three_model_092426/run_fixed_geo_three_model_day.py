#!/usr/bin/env python3
"""Fit and score three covariance specifications on one design-held-out day."""

from __future__ import annotations

import argparse
import gc
import logging
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn import Parameter


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[6]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from GEMS_TCO.data import ProcessedDataLoader
from GEMS_TCO.vecchia.corridor_neighbors.generalized_cauchy import (
    NoNuggetGeneralizedCauchyLag643CorridorVecchia,
)
from GEMS_TCO.vecchia.corridor_neighbors.separable_exponential import (
    NoNuggetAdvectedSeparableExponentialLag643CorridorVecchia,
)
from GEMS_TCO.vecchia.corridor_neighbors.spline import (
    NoNuggetSplineMaternLag643CorridorVecchia,
)

from fixed_geo_three_model_core import (
    MODEL_ORDER,
    atomic_csv,
    atomic_json,
    atomic_text,
    clean_json,
    contrast_covariances,
    load_json,
    load_toml,
    physical_to_raw_parameters,
    prepare_fixed_contrasts,
    score_model,
    select_day_frames,
    sha256_file,
    study_signature,
    task_source_files,
)


LOGGER = logging.getLogger("fixed_geo_three_model")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--task-id", type=int, required=True)
    result.add_argument("--config", type=Path, default=HERE / "fixed_geo_three_model.toml")
    result.add_argument("--design", type=Path, default=HERE / "frozen_design.json")
    result.add_argument("--dates", type=Path, default=HERE / "evaluation_dates.csv")
    result.add_argument("--data-root", type=Path)
    result.add_argument("--output-root", type=Path)
    result.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    result.add_argument("--models", nargs="+", choices=MODEL_ORDER, default=list(MODEL_ORDER))
    result.add_argument(
        "--prepare-only",
        action="store_true",
        help="audit data/geometry/mean construction without fitting any covariance model",
    )
    result.add_argument("--log-level", default="INFO")
    return result


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _resolve_roots(config: dict[str, Any], args: argparse.Namespace) -> tuple[Path, Path]:
    if args.data_root is not None:
        data_root = args.data_root
    else:
        amarel = Path(config["data"]["amarel_root"])
        data_root = amarel if amarel.exists() else Path(config["data"]["local_root"])
    output_root = (
        args.output_root
        if args.output_root is not None
        else Path(config["paths"]["amarel_output_root"])
    )
    return data_root.expanduser().resolve(), output_root.expanduser()


def _build_model(
    model_name: str,
    input_map: dict[str, torch.Tensor],
    grid_coordinates: np.ndarray,
    config: dict[str, Any],
):
    options = {
        "input_map": input_map,
        "grid_coords": grid_coordinates,
        "reference_advec_lon_abs": float(config["models"]["reference_advec_lon_abs"]),
        "target_chunk_size": int(config["models"]["target_chunk_size"]),
        "min_target_points": int(config["models"]["min_target_points"]),
    }
    if model_name == "gc":
        return NoNuggetGeneralizedCauchyLag643CorridorVecchia(
            gc_alpha=float(config["models"]["gc_alpha"]),
            gc_beta=float(config["models"]["gc_beta"]),
            **options,
        )
    if model_name == "matern05":
        return NoNuggetSplineMaternLag643CorridorVecchia(
            smooth=float(config["models"]["matern_smooth"]),
            **options,
        )
    if model_name == "separable":
        return NoNuggetAdvectedSeparableExponentialLag643CorridorVecchia(**options)
    raise ValueError(model_name)


def _initial_raw(config: dict[str, Any]) -> np.ndarray:
    values = config["fit"]
    physical = {
        "signal_variance": values["signal_variance"],
        "range_lat": values["range_lat"],
        "range_lon": values["range_lon"],
        "range_time": values["range_time"],
        "advec_lat": values["advec_lat"],
        "advec_lon": values["advec_lon"],
    }
    return physical_to_raw_parameters(physical, include_nugget=False)


def _fit_one(
    model_name: str,
    model,
    config: dict[str, Any],
    model_dir: Path,
    signature: str,
    reproducibility_metadata: dict[str, Any],
) -> dict[str, Any]:
    fit_path = model_dir / "fit.json"
    fit_complete_path = model_dir / "FIT_COMPLETE"
    if fit_complete_path.is_file() and fit_path.is_file():
        saved = load_json(fit_path)
        if saved.get("study_signature") != signature or saved.get("model") != model_name:
            raise RuntimeError(f"refusing incompatible cached fit in {model_dir}")
        LOGGER.info("Reusing completed %s fit", model_name)
        return saved

    model_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    model.precompute_conditioning_sets()
    precompute_seconds = time.perf_counter() - started
    parameters = [
        Parameter(torch.tensor(value, dtype=torch.float64, device=model.device))
        for value in _initial_raw(config)
    ]
    optimizer = model.make_lbfgs_optimizer(
        parameters,
        lr=float(config["fit"]["lbfgs_lr"]),
        max_iter=int(config["fit"]["lbfgs_inner_max_iter"]),
        tolerance_grad=float(config["fit"]["lbfgs_tolerance_grad"]),
        tolerance_change=float(config["fit"]["lbfgs_tolerance_change"]),
        history_size=int(config["fit"]["lbfgs_history_size"]),
    )
    fit_started = time.perf_counter()
    result = model.fit_lbfgs(
        parameters,
        optimizer,
        max_steps=int(config["fit"]["lbfgs_outer_max_steps"]),
        grad_tol=float(config["fit"]["outer_grad_tol"]),
    )
    fit_seconds = time.perf_counter() - fit_started
    raw = torch.tensor(result.raw_parameters, dtype=torch.float64, device=model.device)
    beta = model.estimate_gls_coefficients(raw).detach().cpu().reshape(-1).tolist()
    fit_record = {
        **reproducibility_metadata,
        "study_signature": signature,
        "model": model_name,
        "raw_parameters": list(result.raw_parameters),
        "interpretable_parameters": clean_json(result.interpretable_parameters),
        "profiled_vecchia_nll_per_target": result.final_nll,
        "steps_completed": result.steps_completed,
        "converged_at_outer_grad_tol": result.converged,
        "maximum_absolute_gradient": result.max_abs_gradient,
        "objective_evaluations": result.objective_evaluations,
        "cache_hits": result.cache_hits,
        "gls_coefficients": beta,
        "gls_latitude_center": float(model.lat_mean_val),
        "precompute_seconds": precompute_seconds,
        "fit_seconds": fit_seconds,
        "resolved_covariance_backend": model.resolved_covariance_backend(),
    }
    atomic_json(fit_path, clean_json(fit_record))
    if bool(config["fit"]["require_convergence"]) and not result.converged:
        raise RuntimeError(
            f"{model_name} did not meet outer_grad_tol="
            f"{float(config['fit']['outer_grad_tol']):.3g}; "
            f"max_abs_gradient={result.max_abs_gradient:.6g}"
        )
    atomic_text(fit_complete_path, "complete\n")
    return fit_record


def main() -> None:
    args = parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config_path = args.config.expanduser().resolve()
    design_path = args.design.expanduser().resolve()
    dates_path = args.dates.expanduser().resolve()
    config = load_toml(config_path)
    design = load_json(design_path)
    if tuple(config["models"]["names"]) != MODEL_ORDER:
        raise ValueError("configuration must preserve the frozen GC/JM/separable model order")
    if config["models"]["nugget_policy"] != "fixed_zero":
        raise ValueError("this frozen study requires the same zero-nugget policy for all models")
    if (
        design["coordinate_frame"] != "fixed_geographic"
        or design["observation_geometry_moves_with_advection"]
    ):
        raise ValueError("the frozen primary design must use fixed geographic coordinates")
    dates = pd.read_csv(dates_path)
    selected = dates.loc[dates["task_id"].astype(int) == int(args.task_id)]
    if len(selected) != 1:
        raise ValueError(f"task id {args.task_id} occurs {len(selected)} times")
    task = selected.iloc[0]
    if str(task["date"]) == str(config["study"]["discovery_day"]):
        raise ValueError("the discovery day cannot enter design-held-out evaluation")
    signature = study_signature(config_path, design_path, dates_path)
    data_root, output_root = _resolve_roots(config, args)
    task_dir = output_root / f"task_{int(task['task_id']):03d}_{task['date']}"
    task_manifest_path = task_dir / "task_manifest.json"
    if task_manifest_path.is_file():
        previous = load_json(task_manifest_path)
        if previous.get("study_signature") != signature:
            raise RuntimeError(f"refusing to mix configurations in {task_dir}")
        fully_complete = (
            (task_dir / "COMPLETE").is_file()
            and (task_dir / "daily_scores.csv").is_file()
            and all((task_dir / f"model_{name}" / "COMPLETE").is_file() for name in MODEL_ORDER)
        )
        if fully_complete:
            LOGGER.info("Task %s is already complete; nothing to do", args.task_id)
            return
    task_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    frozen_design_sha256 = sha256_file(design_path)
    reproducibility_metadata = {
        "task_id": int(task["task_id"]),
        "date": str(task["date"]),
        "frozen_design_identifier": str(design["design_id"]),
        "frozen_design_sha256": frozen_design_sha256,
        "vecchia_conditioning_geometry": str(config["models"]["conditioning_geometry"]),
        "vecchia_lag_pattern": str(config["models"]["conditioning_lag_pattern"]),
        "vecchia_target_chunk_size": int(config["models"]["target_chunk_size"]),
        "git_revision": _git_revision(),
    }
    manifest = {
        **reproducibility_metadata,
        "study_signature": signature,
        "year": int(task["year"]),
        "month": int(task["month"]),
        "day": int(task["day"]),
        "models": list(args.models),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "data_root": str(data_root),
        "output_root": str(output_root),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "frozen_design": str(design_path),
        "evaluation_dates": str(dates_path),
        "evaluation_dates_sha256": sha256_file(dates_path),
        "task_source_sha256": {
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in task_source_files()
        },
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }
    atomic_json(task_manifest_path, manifest)

    loader = ProcessedDataLoader(data_root)
    monthly_frames, _, _, monthly_mean = loader.load_monthly_grids(
        years=[int(task["year"])],
        months=[int(task["month"])],
        latitude_range=(
            float(config["data"]["latitude_min"]),
            float(config["data"]["latitude_max"]),
        ),
        longitude_range=(
            float(config["data"]["longitude_min"]),
            float(config["data"]["longitude_max"]),
        ),
        compute_ordering=False,
    )
    day_frames = select_day_frames(
        monthly_frames,
        day=int(task["day"]),
        expected_slots=int(task["expected_slots"]),
    )
    samples, sample_coordinates, mean_audit = prepare_fixed_contrasts(
        day_frames,
        monthly_mean,
        design,
        wx_tolerance=float(config["diagnostic"]["wx_tolerance"]),
    )
    atomic_csv(task_dir / "empirical_contrasts.csv.gz", samples)
    atomic_json(task_dir / "mean_and_geometry_audit.json", clean_json(mean_audit))
    if args.prepare_only:
        atomic_text(task_dir / "PREPARE_COMPLETE", "complete\n")
        LOGGER.info("Prepared task %s (%s) without fitting", args.task_id, task["date"])
        return

    model_input_cpu, _ = loader.build_model_tensors(
        day_frames,
        ozone_mean=monthly_mean,
        time_slice=(0, int(task["expected_slots"])),
        dtype=torch.float64,
        use_source_coordinates=bool(config["data"]["use_source_coordinates_for_covariance"]),
        time_origin_hours=float(design["model_time_origin_hours"]),
    )
    model_input = {key: value.to(device) for key, value in model_input_cpu.items()}
    first_frame = next(iter(day_frames.values()))
    grid_coordinates = first_frame[["Latitude", "Longitude"]].to_numpy(dtype=np.float64)

    daily_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for model_name in args.models:
        model_dir = task_dir / f"model_{model_name}"
        try:
            model = _build_model(model_name, model_input, grid_coordinates, config)
            expected_parameter_count = int(config["models"]["covariance_parameter_count"])
            if model.covariance_parameter_count != expected_parameter_count:
                raise RuntimeError(
                    f"{model_name} has {model.covariance_parameter_count} covariance "
                    f"parameters; expected {expected_parameter_count}"
                )
            fit_record = _fit_one(
                model_name,
                model,
                config,
                model_dir,
                signature,
                reproducibility_metadata,
            )
            physical = fit_record["interpretable_parameters"]
            covariance = contrast_covariances(
                sample_coordinates,
                physical,
                model_name,
                design,
                gc_alpha=float(config["models"]["gc_alpha"]),
                gc_beta=float(config["models"]["gc_beta"]),
            )
            predicted, score_summary = score_model(samples, covariance, model_name, design)
            score_summary = {
                **reproducibility_metadata,
                "year": int(task["year"]),
                **score_summary,
            }
            atomic_csv(model_dir / "predicted_contrast_covariances.csv.gz", predicted)
            atomic_json(model_dir / "score_summary.json", clean_json(score_summary))
            (model_dir / "FAILED.json").unlink(missing_ok=True)
            atomic_text(model_dir / "COMPLETE", "complete\n")
            daily_rows.append(
                {
                    **score_summary,
                    "vecchia_nll": fit_record["profiled_vecchia_nll_per_target"],
                    "fit_converged": fit_record["converged_at_outer_grad_tol"],
                    "fit_max_abs_gradient": fit_record["maximum_absolute_gradient"],
                    "fit_seconds": fit_record["fit_seconds"],
                    "wx_max_abs": mean_audit["wx_max_abs"],
                    "mean_rule": mean_audit["empirical_mean_rule"],
                    "potential_sample_count": mean_audit["potential_sample_count"],
                    "complete_contrast_fraction": mean_audit["complete_contrast_fraction"],
                    **{
                        f"fit_{name}": value
                        for name, value in physical.items()
                        if isinstance(value, (int, float))
                    },
                }
            )
        except Exception as error:  # keep independent model failures inspectable
            failures.append(model_name)
            atomic_json(
                model_dir / "FAILED.json",
                {
                    "study_signature": signature,
                    "model": model_name,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            LOGGER.exception("Model %s failed", model_name)
        finally:
            if "model" in locals():
                del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if daily_rows:
        atomic_csv(task_dir / "daily_scores.csv", pd.DataFrame(daily_rows))
    if failures:
        atomic_json(task_dir / "FAILED.json", {"failed_models": failures})
        raise RuntimeError(f"task {args.task_id} failed models: {failures}")
    if set(args.models) != set(MODEL_ORDER):
        raise RuntimeError("task COMPLETE requires all three pre-specified models")
    (task_dir / "FAILED.json").unlink(missing_ok=True)
    atomic_text(task_dir / "COMPLETE", "complete\n")
    LOGGER.info("Completed task %s (%s)", args.task_id, task["date"])


if __name__ == "__main__":
    main()
