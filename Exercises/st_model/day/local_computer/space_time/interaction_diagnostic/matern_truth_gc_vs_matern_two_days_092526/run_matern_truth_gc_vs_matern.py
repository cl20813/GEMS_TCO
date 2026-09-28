#!/usr/bin/env python3
"""Compare fitted GC and Matérn contrast covariances on two Matérn simulations."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[6]
SRC = PROJECT_ROOT / "src"
AMAREL_STUDY = (
    PROJECT_ROOT
    / "Exercises/st_model/day/amarel_simulation/space_time/interaction_diagnostic"
    / "fixed_geographic_three_model_092426"
)
for path in (SRC, AMAREL_STUDY):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from GEMS_TCO.data import ProcessedDataLoader  # noqa: E402
from fixed_geo_three_model_core import (  # noqa: E402
    atomic_csv,
    atomic_json,
    atomic_text,
    clean_json,
    contrast_covariances,
    load_json,
    load_toml,
    prepare_fixed_contrasts,
    score_model,
    select_day_frames,
    sha256_file,
)
from run_fixed_geo_three_model_day import (  # noqa: E402
    _build_model,
    _fit_one,
    _git_revision,
)


LOGGER = logging.getLogger("matern_truth_two_day_audit")
DESIGN_PATH = AMAREL_STUDY / "frozen_design.json"
CONFIG_PATH = HERE / "simulation_gc_vs_matern.toml"
SELECTED = ((2024, "2024-07-13"), (2024, "2024-07-19"))
MODEL_NAMES = ("gc", "matern05")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-root", type=Path, default=HERE / "outputs")
    result.add_argument("--summary-only", action="store_true")
    result.add_argument("--log-level", default="INFO")
    return result


def _simulation_paths(root: Path, year: int) -> tuple[Path, Path]:
    directory = root / f"{year}_july_st_circulant"
    prefix = f"sim_july{year}_st_circulant"
    # Use the asset that reproduces the real-data nearest-cell/half-cell
    # threshold.  Its retained observations still carry their original
    # irregular Source_Latitude/Source_Longitude, which are used in covariance
    # fitting below; Latitude/Longitude only identify translated filter cells.
    return (
        directory / f"{prefix}_gridded.pkl",
        directory / f"{prefix}_truth.json",
    )


def _monthly_mean(frames: dict[str, pd.DataFrame]) -> float:
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
        raise ValueError("simulation asset has no finite ozone values")
    return float(np.mean(finite))


def _truth_physical(truth: dict[str, Any]) -> dict[str, float]:
    if float(truth["smooth"]) != 0.5:
        raise ValueError(f"expected Matérn smoothness 0.5, got {truth['smooth']}")
    if float(truth["nugget"]) != 0.0:
        raise ValueError(f"expected zero nugget, got {truth['nugget']}")
    return {
        "signal_variance": float(truth["sigmasq"]),
        "range_lat": float(truth["range_lat"]),
        "range_lon": float(truth["range_lon"]),
        "range_time": float(truth["range_time"]),
        "advec_lat": float(truth["advec_lat"]),
        "advec_lon": float(truth["advec_lon"]),
        "nugget": float(truth["nugget"]),
    }


def _signature(paths: list[Path], labels: list[str]) -> str:
    digest = hashlib.sha256()
    for label, path in zip(labels, paths, strict=True):
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _covariance_error(
    fitted: np.ndarray, truth: np.ndarray
) -> dict[str, float]:
    delta = np.asarray(fitted, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    cross = delta[:, 0, 1]
    frobenius_squared = (
        np.square(delta[:, 0, 0])
        + np.square(delta[:, 1, 1])
        + 2.0 * np.square(cross)
    )
    return {
        "truth_pointwise_c_ab_rmse": float(np.sqrt(np.mean(np.square(cross)))),
        "truth_pointwise_covariance_frobenius_rmse": float(
            np.sqrt(np.mean(frobenius_squared))
        ),
        "truth_pointwise_c_ab_mean_error": float(np.mean(cross)),
    }


def _fit_day(
    year: int,
    date: str,
    config: dict[str, Any],
    design: dict[str, Any],
    output_root: Path,
) -> pd.DataFrame:
    simulation_root = PROJECT_ROOT / str(config["data"]["simulation_root"])
    data_path, truth_path = _simulation_paths(simulation_root, year)
    for path in (data_path, truth_path, CONFIG_PATH, DESIGN_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    truth_physical = _truth_physical(truth)
    frames = pd.read_pickle(data_path)
    day = int(date[-2:])
    day_frames = select_day_frames(frames, day=day, expected_slots=8)
    monthly_mean = _monthly_mean(frames)
    samples, sample_coordinates, mean_audit = prepare_fixed_contrasts(
        day_frames,
        monthly_mean,
        design,
        wx_tolerance=float(config["diagnostic"]["wx_tolerance"]),
    )

    task_dir = output_root / date
    task_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(task_dir / "empirical_contrasts.csv.gz", samples)
    atomic_json(task_dir / "mean_and_geometry_audit.json", clean_json(mean_audit))

    signature = _signature(
        [CONFIG_PATH, DESIGN_PATH, data_path, truth_path, Path(__file__).resolve()],
        ["config", "design", "simulation", "truth", "runner"],
    )
    metadata = {
        "dataset": date,
        "year": year,
        "data_kind": "synthetic",
        "simulation_coordinate_asset": "gridded_with_irregular_source_coordinates",
        "truth_family": "joint_matern",
        "truth_smooth": 0.5,
        "truth_nugget": 0.0,
        "frozen_design_identifier": str(design["design_id"]),
        "frozen_design_sha256": sha256_file(DESIGN_PATH),
        "vecchia_conditioning_geometry": str(config["models"]["conditioning_geometry"]),
        "vecchia_lag_pattern": str(config["models"]["conditioning_lag_pattern"]),
        "vecchia_target_chunk_sizes_by_model": {
            name: int(config["models"]["target_chunk_sizes"][name])
            for name in MODEL_NAMES
        },
        "git_revision": _git_revision(),
        "simulation_path": str(data_path),
        "simulation_sha256": sha256_file(data_path),
        "truth_path": str(truth_path),
        "truth_sha256": sha256_file(truth_path),
        "study_signature": signature,
    }
    atomic_json(task_dir / "run_manifest.json", clean_json({**metadata, "truth": truth}))

    loader = ProcessedDataLoader(simulation_root)
    model_input_cpu, _ = loader.build_model_tensors(
        day_frames,
        ozone_mean=monthly_mean,
        time_slice=(0, 8),
        dtype=torch.float64,
        use_source_coordinates=bool(config["data"]["use_source_coordinates_for_covariance"]),
        time_origin_hours=float(design["model_time_origin_hours"]),
    )
    first_frame = next(iter(day_frames.values()))
    grid_coordinates = first_frame[["Latitude", "Longitude"]].to_numpy(dtype=np.float64)

    truth_covariance = contrast_covariances(
        sample_coordinates,
        truth_physical,
        "matern05",
        design,
        gc_alpha=float(config["models"]["gc_alpha"]),
        gc_beta=float(config["models"]["gc_beta"]),
    )
    truth_predicted, truth_summary = score_model(
        samples, truth_covariance, "truth_matern05", design
    )
    atomic_csv(task_dir / "truth_predicted_contrast_covariances.csv.gz", truth_predicted)

    rows: list[dict[str, Any]] = [
        {
            "date": date,
            "year": year,
            "model": "truth_matern05",
            **truth_summary,
            "truth_pooled_c_ab_error": 0.0,
            "truth_pooled_v_a_error": 0.0,
            "truth_pooled_v_b_error": 0.0,
            "truth_pooled_covariance_frobenius_error": 0.0,
            "truth_pointwise_c_ab_rmse": 0.0,
            "truth_pointwise_covariance_frobenius_rmse": 0.0,
            "truth_pointwise_c_ab_mean_error": 0.0,
            "vecchia_nll": None,
            "fit_seconds": None,
            "fit_max_abs_gradient": None,
            **{f"fit_{name}": value for name, value in truth_physical.items()},
        }
    ]

    for model_name in MODEL_NAMES:
        model_dir = task_dir / f"model_{model_name}"
        model = _build_model(model_name, model_input_cpu, grid_coordinates, config)
        fit_metadata = {
            **metadata,
            "model": model_name,
            "vecchia_target_chunk_size": int(model.target_chunk_size),
        }
        fit_record = _fit_one(
            model_name,
            model,
            config,
            model_dir,
            signature,
            fit_metadata,
        )
        fitted_covariance = contrast_covariances(
            sample_coordinates,
            fit_record["interpretable_parameters"],
            model_name,
            design,
            gc_alpha=float(config["models"]["gc_alpha"]),
            gc_beta=float(config["models"]["gc_beta"]),
        )
        predicted, summary = score_model(samples, fitted_covariance, model_name, design)
        atomic_csv(model_dir / "predicted_contrast_covariances.csv.gz", predicted)
        truth_error = _covariance_error(fitted_covariance, truth_covariance)
        pooled_delta_aa = float(summary["model_v_a"] - truth_summary["model_v_a"])
        pooled_delta_bb = float(summary["model_v_b"] - truth_summary["model_v_b"])
        pooled_delta_ab = float(summary["model_c_ab"] - truth_summary["model_c_ab"])
        row = {
            "date": date,
            "year": year,
            **summary,
            "truth_pooled_c_ab_error": abs(pooled_delta_ab),
            "truth_pooled_v_a_error": abs(pooled_delta_aa),
            "truth_pooled_v_b_error": abs(pooled_delta_bb),
            "truth_pooled_covariance_frobenius_error": float(
                np.sqrt(pooled_delta_aa**2 + pooled_delta_bb**2 + 2.0 * pooled_delta_ab**2)
            ),
            **truth_error,
            "vecchia_nll": fit_record["profiled_vecchia_nll_per_target"],
            "fit_seconds": fit_record["fit_seconds"],
            "fit_max_abs_gradient": fit_record["maximum_absolute_gradient"],
            **{
                f"fit_{name}": value
                for name, value in fit_record["interpretable_parameters"].items()
                if isinstance(value, (int, float))
            },
        }
        rows.append(row)
        atomic_json(model_dir / "truth_error_summary.json", clean_json(row))
        del model
        gc.collect()

    result = pd.DataFrame(rows)
    atomic_csv(task_dir / "model_summary.csv", result)
    return result


def _read_existing(output_root: Path) -> pd.DataFrame:
    frames = []
    for _, date in SELECTED:
        path = output_root / date / "model_summary.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        frames.append(pd.read_csv(path, float_precision="round_trip"))
    return pd.concat(frames, ignore_index=True)


def _comparison(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date, frame in summary.groupby("date", sort=True):
        model = frame.set_index("model")
        truth = model.loc["truth_matern05"]
        gc_row = model.loc["gc"]
        matern = model.loc["matern05"]
        rows.append(
            {
                "date": date,
                "sample_count": int(truth["sample_count"]),
                "empirical_c_ab": float(truth["empirical_cross_moment_q_a_q_b"]),
                "truth_c_ab": float(truth["model_c_ab"]),
                "gc_c_ab": float(gc_row["model_c_ab"]),
                "matern_c_ab": float(matern["model_c_ab"]),
                "gc_abs_c_ab_error_to_truth": float(gc_row["truth_pooled_c_ab_error"]),
                "matern_abs_c_ab_error_to_truth": float(
                    matern["truth_pooled_c_ab_error"]
                ),
                "gc_pointwise_c_ab_rmse_to_truth": float(
                    gc_row["truth_pointwise_c_ab_rmse"]
                ),
                "matern_pointwise_c_ab_rmse_to_truth": float(
                    matern["truth_pointwise_c_ab_rmse"]
                ),
                "truth_score": float(truth["mean_contrast_score"]),
                "gc_score": float(gc_row["mean_contrast_score"]),
                "matern_score": float(matern["mean_contrast_score"]),
                "matern_minus_gc_score": float(matern["mean_contrast_score"])
                - float(gc_row["mean_contrast_score"]),
                "gc_covariance_frobenius_error_to_truth": float(
                    gc_row["truth_pooled_covariance_frobenius_error"]
                ),
                "matern_covariance_frobenius_error_to_truth": float(
                    matern["truth_pooled_covariance_frobenius_error"]
                ),
                "gc_pointwise_covariance_frobenius_rmse_to_truth": float(
                    gc_row["truth_pointwise_covariance_frobenius_rmse"]
                ),
                "matern_pointwise_covariance_frobenius_rmse_to_truth": float(
                    matern["truth_pointwise_covariance_frobenius_rmse"]
                ),
                "gc_vecchia_nll": float(gc_row["vecchia_nll"]),
                "matern_vecchia_nll": float(matern["vecchia_nll"]),
                "gc_fit_seconds": float(gc_row["fit_seconds"]),
                "matern_fit_seconds": float(matern["fit_seconds"]),
            }
        )
    return pd.DataFrame(rows)


def _report(comparison: pd.DataFrame, output_root: Path) -> None:
    gc_cross_wins = int(
        (
            comparison["gc_abs_c_ab_error_to_truth"]
            < comparison["matern_abs_c_ab_error_to_truth"]
        ).sum()
    )
    gc_score_wins = int((comparison["matern_minus_gc_score"] > 0).sum())
    lines = [
        "# Matérn-truth two-day GC versus Matérn audit",
        "",
        "The frozen fixed-geographic A/B filter and lag 1 were applied at every valid ",
        "anchor and adjacent time pair. Both models used corridor 4/3/2, 4x4 target ",
        "blocks, CPU chunk 64, zero nugget, and the same truth-based initial values.",
        "",
        "The target truth is the analytic joint Matérn covariance with nu=0.5. The ",
        "stored fields were generated by the historical clipped circulant-embedding ",
        "generator, so the analytic truth comparison does not include any small covariance ",
        "perturbation introduced by FFT eigenvalue clipping.",
        "",
        "| date | empirical C_AB | analytic truth C_AB | fitted GC C_AB | fitted Matérn C_AB | |GC-truth| | |Matérn-truth| | GC score | Matérn score |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.date} | {row.empirical_c_ab:.8g} | {row.truth_c_ab:.8g} | "
            f"{row.gc_c_ab:.8g} | {row.matern_c_ab:.8g} | "
            f"{row.gc_abs_c_ab_error_to_truth:.8g} | "
            f"{row.matern_abs_c_ab_error_to_truth:.8g} | "
            f"{row.gc_score:.8g} | {row.matern_score:.8g} |"
        )
    lines.extend(
        [
            "",
            f"- GC has the smaller pooled C_AB error to analytic truth on {gc_cross_wins}/2 days.",
            f"- GC has the lower observed bivariate contrast score on {gc_score_wins}/2 days.",
            "- The two rows are a mechanism audit, not a power or model-selection study.",
            "- Overlapping within-day contrasts are not treated as independent replicates.",
            "",
        ]
    )
    atomic_text(output_root / "RESULTS.md", "\n".join(lines))


def main() -> None:
    args = parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config = load_toml(CONFIG_PATH)
    design = load_json(DESIGN_PATH)
    if args.summary_only:
        summary = _read_existing(output_root)
    else:
        summary = pd.concat(
            [
                _fit_day(year, date, config, design, output_root)
                for year, date in SELECTED
            ],
            ignore_index=True,
        )
    comparison = _comparison(summary)
    atomic_csv(output_root / "all_model_summaries.csv", summary)
    atomic_csv(output_root / "cross_term_comparison.csv", comparison)
    _report(comparison, output_root)
    print(comparison.to_string(index=False), flush=True)
    print(f"Wrote {output_root / 'RESULTS.md'}", flush=True)


if __name__ == "__main__":
    main()
