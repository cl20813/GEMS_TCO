#!/usr/bin/env python3
"""Compute additional fixed-parameter misspecifications for the five-row plot.

Data are the 2024-07-13 Matern smoothness-0.5 simulation.  The known DGP mean
is removed, no parameters are fitted, and the lag-6/4/3 Vecchia graph is shared
across three assumed models:

* latitude range 0.25 times truth;
* latitude range 2 times truth;
* Matern smoothness 1.0 instead of 0.5.

The arbitrary smoothness case uses the spline Matern implementation.  All
three models use residual-Lanczos 512 and paired random-probe SLQ 8x192.
"""

from __future__ import annotations

import argparse
import gc
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import sim_true_vs_misspecified_residual_lanczos_slq as base
import vecchia_local_20240703_adapted_fixed_residual_lanczos_slq as diagnostic
from vecchia_sparse_precision_operator_090326 import (
    build_sparse_vecchia_precision,
    native_gls_quadratic,
    verify_precision_identity,
)


SCENARIOS = (
    "lat_range_quarter_512",
    "lat_range_2x",
    "wrong_smooth_10",
)
DEFAULT_OUTPUT_ROOT = base.REPO / (
    "outputs/summer_26/"
    "sim_20240713_latitude_short_long_smooth10_lanczos512_slq8x192"
)
REFERENCE_ROOT = base.REPO / (
    "outputs/summer_26/"
    "sim_20240713_true_vs_misspecified_residual_lanczos512_slq8x192"
)


def parser() -> argparse.ArgumentParser:
    out = base.parser()
    out.description = __doc__
    out.set_defaults(output_root=DEFAULT_OUTPUT_ROOT)
    return out


def scenario_parameters(truth: dict[str, float]) -> dict[str, dict[str, float]]:
    keys = (
        "smooth", "sigmasq", "range_lat", "range_lon", "range_time",
        "advec_lat", "advec_lon", "nugget",
    )
    original = {key: float(truth[key]) for key in keys}
    short = dict(original)
    short["range_lat"] *= 0.25
    long = dict(original)
    long["range_lat"] *= 2.0
    smooth = dict(original)
    smooth["smooth"] = 1.0
    return {
        "lat_range_quarter_512": short,
        "lat_range_2x": long,
        "wrong_smooth_10": smooth,
    }


def prepare_shared_model(
    asset: Any,
    truth: dict[str, float],
    args: argparse.Namespace,
) -> tuple[Any, np.ndarray, float]:
    source_map = {
        key: tensor.to(device="cpu", dtype=torch.float64).contiguous()
        for key, tensor in asset.source_map.items()
    }
    model = base.sim_models.RealDataCorridorWidth4x4Lag643FixedNuggetSplineFit(
        smooth=float(truth["smooth"]),
        fixed_nugget=float(truth["nugget"]),
        input_map=source_map,
        grid_coords=asset.grid_coords_np,
        lag1_lon_offset=float(args.reference_advec_lon_abs),
        daily_stride=int(args.daily_stride),
        target_chunk_size=int(args.target_chunk_size),
        min_target_points=1,
        spline_n_points=int(args.spline_n_points),
        spline_r_max=float(args.spline_r_max),
    )
    started = time.perf_counter()
    model.precompute_conditioning_sets()
    precompute_seconds = time.perf_counter() - started
    beta = base.true_mean_beta(model, asset, truth)
    return model, beta, precompute_seconds


def build_precision(
    model: Any,
    beta: np.ndarray,
    scenario: str,
    assumed: dict[str, float],
    args: argparse.Namespace,
) -> tuple[Any, dict[str, Any]]:
    model.smooth = float(assumed["smooth"])
    params = torch.as_tensor(
        base.sim_data.physical_to_log_phi(assumed), dtype=torch.float64, device="cpu"
    )
    native = native_gls_quadratic(model, params, beta)
    precision = build_sparse_vecchia_precision(
        model,
        params,
        beta,
        chunk_size=int(args.target_chunk_size),
        coefficient_drop_tolerance=float(args.coefficient_drop_tolerance),
        progress=lambda message: print(f"    {scenario} B: {message}", flush=True),
    )
    identity = verify_precision_identity(precision, native)
    if float(identity["relative_error"]) > 1e-8:
        raise RuntimeError(f"Sparse precision identity failed for {scenario}: {identity}")
    summary = {
        "scenario": scenario,
        "assumed_parameters": assumed,
        "precision_build_seconds": float(precision.metadata["build_s"]),
        "precision_nnz": int(precision.whitener.nnz),
        "n_observations": int(precision.n),
        "direct_energy_per_observation": float(identity["sparse_quadratic"] / precision.n),
        "precision_identity_relative_error": float(identity["relative_error"]),
    }
    del params
    return precision, summary


def load_true_reference(filename: str) -> pd.DataFrame:
    path = REFERENCE_ROOT / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required true-model reference is missing: {path}")
    frame = pd.read_csv(path)
    return frame[frame["method"].eq("true")].copy()


def main() -> None:
    args = parser().parse_args()
    if int(args.residual_lanczos_steps) != 512:
        raise ValueError("Use residual-Lanczos 512 for the common five-case comparison")
    if int(args.slq_probes) != 8 or int(args.slq_steps) != 192:
        raise ValueError("Use paired SLQ 8x192 for the common five-case comparison")
    args.output_root.mkdir(parents=True, exist_ok=True)
    diagnostic.DATE = base.SIM_DATE
    workflow_started = time.perf_counter()
    truth = base.load_truth(args.data_root)
    asset = base.load_asset(args)
    assumed_by_scenario = scenario_parameters(truth)
    model, beta, graph_seconds = prepare_shared_model(asset, truth, args)
    print(f"Shared lag-6/4/3 graph prepared in {graph_seconds:.2f}s", flush=True)

    curve_frames = [load_true_reference("residual_lanczos_cumulative_curves.csv")]
    band_frames = [load_true_reference("frequency_band_energy_ratios.csv")]
    third_frames = [load_true_reference("frequency_third_energy_ratios.csv")]
    slq_frames = [load_true_reference("slq_spectrum.csv")]
    summaries: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []

    for scenario in SCENARIOS:
        scenario_started = time.perf_counter()
        assumed = assumed_by_scenario[scenario]
        print(f"\n{scenario}: {assumed}", flush=True)
        precision, build_summary = build_precision(
            model, beta, scenario, assumed, args
        )
        slq, curves, bands, result = diagnostic.evaluate_method(
            scenario, precision, args
        )
        elapsed = time.perf_counter() - scenario_started
        summary = {**build_summary, **result["summary"], "scenario_total_seconds": elapsed}
        summaries.append(summary)
        curve_frames.append(curves)
        band_frames.append(bands)
        third_frames.append(result["thirds"])
        slq_frames.append(slq.assign(date=base.SIM_DATE, method=scenario))
        timings.append({
            "scenario": scenario,
            "precision_build_seconds": build_summary["precision_build_seconds"],
            "slq_seconds": result["summary"]["slq_seconds"],
            "residual_lanczos_seconds": result["summary"]["residual_lanczos_seconds"],
            "scenario_total_seconds": elapsed,
        })
        print(
            "  thirds: "
            + ", ".join(
                f"{row.frequency_third}={row.energy_per_mode:.3f}"
                for row in result["thirds"].itertuples(index=False)
            ),
            flush=True,
        )
        del precision
        gc.collect()

    base.atomic_csv(
        args.output_root / "residual_lanczos_cumulative_curves.csv",
        pd.concat(curve_frames, ignore_index=True),
    )
    base.atomic_csv(
        args.output_root / "frequency_band_energy_ratios.csv",
        pd.concat(band_frames, ignore_index=True),
    )
    base.atomic_csv(
        args.output_root / "frequency_third_energy_ratios.csv",
        pd.concat(third_frames, ignore_index=True),
    )
    base.atomic_csv(
        args.output_root / "slq_spectrum.csv",
        pd.concat(slq_frames, ignore_index=True),
    )
    base.atomic_csv(args.output_root / "timings.csv", pd.DataFrame(timings))
    total_seconds = time.perf_counter() - workflow_started
    base.write_json(
        args.output_root / "RUN_COMPLETE.json",
        {
            "completed": datetime.now().isoformat(timespec="seconds"),
            "date": base.SIM_DATE,
            "truth": truth,
            "assumed_parameters": assumed_by_scenario,
            "scenarios": list(SCENARIOS),
            "shared_graph_precompute_seconds": graph_seconds,
            "total_wall_seconds": total_seconds,
            "residual_lanczos_steps": 512,
            "slq_probes": 8,
            "slq_steps": 192,
            "true_reference_reused_from": str(REFERENCE_ROOT),
            "summaries": summaries,
            "mean_handling": "known DGP mean removed; no parameter fitting",
            "spline_engine": "GEMS_TCO.vecchia_st_spline",
            "individual_eigenvectors_computed": False,
        },
    )
    print(f"\nComplete in {total_seconds:.2f}s: {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
