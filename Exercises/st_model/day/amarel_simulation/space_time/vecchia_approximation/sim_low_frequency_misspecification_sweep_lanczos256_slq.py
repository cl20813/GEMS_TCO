#!/usr/bin/env python3
"""One-day low-frequency misspecification sweep at Lanczos resolution 256.

This extends ``sim_true_vs_misspecified_residual_lanczos_slq.py``.  Parameters
are fixed rather than fitted, the DGP mean is known, and every case uses the
same precomputed lag-6/4/3 Vecchia graph.  The sweep contains:

* a common-scale change preserving the Matern nu=0.5 microergodic proxy
  kappa = sigma^2 / rho;
* sigma-only and all-range-only changes that break kappa by factors 0.5 or 2;
* separate latitude, longitude, and temporal range-ratio failures;
* zero, reversed, and three-times-stronger advection.

Residual-started Lanczos estimates cumulative spectral residual energy and
paired random-probe SLQ estimates the spectral mode-count CDF.  Individual
eigenvectors are neither requested nor stored.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch


HERE = Path(__file__).resolve().parent
REPO = next(parent for parent in HERE.parents if (parent / "src/GEMS_TCO").is_dir())
SRC = REPO / "src"
EIGEN_DIR = HERE.parent / "eigen_analysis"
INTERACTION_DIR = HERE.parent / "interaction_diagnostic"
for candidate in (HERE, SRC, EIGEN_DIR, INTERACTION_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import vecchia_conditional_eigen_sort_common_engine_061926 as sim_data  # noqa: E402
import vecchia_conditional_eigen_sort_sim_smooth0p5_parameter_mismatch_071126 as sim_models  # noqa: E402
import vecchia_local_20240703_adapted_fixed_residual_lanczos_slq as diagnostic  # noqa: E402
import sim_true_vs_misspecified_residual_lanczos_slq as base_experiment  # noqa: E402
from vecchia_sparse_precision_operator_090326 import (  # noqa: E402
    build_sparse_vecchia_precision,
    native_gls_quadratic,
    verify_precision_identity,
)


SIM_DATE = base_experiment.SIM_DATE
TRUE = "true"
SCENARIOS = (
    TRUE,
    "microergodic_preserved_2x",
    "sigma2_2x",
    "sigma2_half",
    "all_ranges_2x",
    "all_ranges_half",
    "lat_range_quarter",
    "lon_range_4x",
    "time_range_quarter",
    "time_range_4x",
    "advection_zero",
    "advection_reversed",
    "advection_3x",
)
LABELS = {
    TRUE: "true",
    "microergodic_preserved_2x": "σ² & all ranges 2× (κ kept)",
    "sigma2_2x": "σ² 2× (κ 2×)",
    "sigma2_half": "σ² 0.5× (κ 0.5×)",
    "all_ranges_2x": "all ranges 2× (κ 0.5×)",
    "all_ranges_half": "all ranges 0.5× (κ 2×)",
    "lat_range_quarter": "latitude range 0.25×",
    "lon_range_4x": "longitude range 4×",
    "time_range_quarter": "time range 0.25×",
    "time_range_4x": "time range 4×",
    "advection_zero": "advection = 0",
    "advection_reversed": "advection reversed",
    "advection_3x": "advection 3×",
}
GROUPS = {
    "common scale / microergodic": SCENARIOS[:6],
    "anisotropic range ratios": (TRUE, *SCENARIOS[6:10]),
    "advection": (TRUE, *SCENARIOS[10:]),
}


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser(description=__doc__)
    out.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            "/Users/joonwonlee/Documents/GEMS_DATA/simulation/"
            "july_st_circulant_realpattern_smooth0p5"
        ),
    )
    out.add_argument(
        "--output-root",
        type=Path,
        default=REPO
        / "outputs/summer_26/sim_20240713_low_frequency_misspecification_lanczos256_slq4x128",
    )
    out.add_argument("--residual-lanczos-steps", type=int, default=256)
    out.add_argument("--slq-probes", type=int, default=4)
    out.add_argument("--slq-steps", type=int, default=128)
    out.add_argument("--curve-points", type=int, default=121)
    out.add_argument("--frequency-bins", type=int, default=12)
    out.add_argument("--random-seed", type=int, default=20260908)
    out.add_argument("--target-chunk-size", type=int, default=32)
    out.add_argument("--coefficient-drop-tolerance", type=float, default=0.0)
    out.add_argument("--daily-stride", type=int, default=2)
    out.add_argument("--spline-n-points", type=int, default=4000)
    out.add_argument("--spline-r-max", type=float, default=30.0)
    out.add_argument("--reference-advec-lon-abs", type=float, default=0.2)
    return out


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_ready(value), indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def make_scenarios(truth: dict[str, float]) -> dict[str, dict[str, float]]:
    keys = (
        "smooth", "sigmasq", "range_lat", "range_lon", "range_time",
        "advec_lat", "advec_lon", "nugget",
    )
    base = {key: float(truth[key]) for key in keys}
    out = {name: dict(base) for name in SCENARIOS}
    for key in ("sigmasq", "range_lat", "range_lon", "range_time"):
        out["microergodic_preserved_2x"][key] *= 2.0
    out["sigma2_2x"]["sigmasq"] *= 2.0
    out["sigma2_half"]["sigmasq"] *= 0.5
    for key in ("range_lat", "range_lon", "range_time"):
        out["all_ranges_2x"][key] *= 2.0
        out["all_ranges_half"][key] *= 0.5
    out["lat_range_quarter"]["range_lat"] *= 0.25
    out["lon_range_4x"]["range_lon"] *= 4.0
    out["time_range_quarter"]["range_time"] *= 0.25
    out["time_range_4x"]["range_time"] *= 4.0
    out["advection_zero"]["advec_lat"] = 0.0
    out["advection_zero"]["advec_lon"] = 0.0
    out["advection_reversed"]["advec_lat"] *= -1.0
    out["advection_reversed"]["advec_lon"] *= -1.0
    out["advection_3x"]["advec_lat"] *= 3.0
    out["advection_3x"]["advec_lon"] *= 3.0
    return out


def scenario_metadata(name: str) -> dict[str, Any]:
    common_scale = {
        TRUE: (1.0, 1.0),
        "microergodic_preserved_2x": (2.0, 2.0),
        "sigma2_2x": (2.0, 1.0),
        "sigma2_half": (0.5, 1.0),
        "all_ranges_2x": (1.0, 2.0),
        "all_ranges_half": (1.0, 0.5),
    }
    if name not in common_scale:
        return {"microergodic_proxy_multiplier": np.nan}
    sigma_multiplier, range_multiplier = common_scale[name]
    # nu=0.5: (sigma^2/rho^(2 nu)) ratio = sigma_multiplier/range_multiplier.
    return {
        "sigma2_multiplier": sigma_multiplier,
        "common_range_multiplier": range_multiplier,
        "microergodic_proxy_multiplier": sigma_multiplier / range_multiplier,
    }


def prepare_common_model(
    asset: sim_data.DayAsset,
    truth: dict[str, float],
    args: argparse.Namespace,
) -> tuple[Any, np.ndarray, float]:
    source_map = {
        key: tensor.to(device="cpu", dtype=torch.float64).contiguous()
        for key, tensor in asset.source_map.items()
    }
    model = sim_models.RealDataCorridorWidth4x4Lag643FixedNuggetSplineFit(
        smooth=float(truth["smooth"]), fixed_nugget=float(truth["nugget"]),
        input_map=source_map, grid_coords=asset.grid_coords_np,
        lag1_lon_offset=float(args.reference_advec_lon_abs),
        daily_stride=int(args.daily_stride),
        target_chunk_size=int(args.target_chunk_size), min_target_points=1,
        spline_n_points=int(args.spline_n_points), spline_r_max=float(args.spline_r_max),
    )
    started = time.perf_counter()
    model.precompute_conditioning_sets()
    precompute_seconds = time.perf_counter() - started
    beta = base_experiment.true_mean_beta(model, asset, truth)
    return model, beta, precompute_seconds


def build_precision(
    model: Any,
    beta: np.ndarray,
    assumed: dict[str, float],
    args: argparse.Namespace,
) -> tuple[Any, dict[str, Any]]:
    params = torch.as_tensor(sim_data.physical_to_log_phi(assumed), dtype=torch.float64)
    native = native_gls_quadratic(model, params, beta)
    precision = build_sparse_vecchia_precision(
        model, params, beta, chunk_size=int(args.target_chunk_size),
        coefficient_drop_tolerance=float(args.coefficient_drop_tolerance),
    )
    identity = verify_precision_identity(precision, native)
    if float(identity["relative_error"]) > 1e-8:
        raise RuntimeError(f"Sparse precision identity failed: {identity}")
    summary = {
        "precision_build_seconds": float(precision.metadata["build_s"]),
        "precision_nnz": int(precision.whitener.nnz),
        "n_observations": int(precision.n),
        "direct_energy_per_observation": float(identity["sparse_quadratic"] / precision.n),
        "precision_identity_relative_error": float(identity["relative_error"]),
    }
    del params
    return precision, summary


def curve_metrics(curves: pd.DataFrame, bands: pd.DataFrame, steps: int) -> dict[str, float]:
    curve = curves[curves["residual_lanczos_steps"].eq(int(steps))].sort_values(
        "estimated_mode_fraction"
    )
    x = curve["estimated_mode_fraction"].to_numpy(float)
    deviation = curve["cumulative_energy_per_n"].to_numpy(float) - x
    low_mask = x <= (1.0 / 3.0 + 1e-8)
    final_bands = bands[bands["residual_lanczos_steps"].eq(int(steps))].sort_values(
        "frequency_bin"
    )
    ratios = final_bands["energy_per_mode"].to_numpy(float)
    low_bins = ratios[: len(ratios) // 3]
    return {
        "overall_max_abs_cumulative_deviation": float(np.max(np.abs(deviation))),
        "low_max_abs_cumulative_deviation": float(np.max(np.abs(deviation[low_mask]))),
        "low_endpoint_cumulative_deviation": float(deviation[low_mask][-1]),
        "low_four_bin_rmse_from_one": float(np.sqrt(np.mean(np.square(low_bins - 1.0)))),
        "all_bin_rmse_from_one": float(np.sqrt(np.mean(np.square(ratios - 1.0)))),
    }


def plot_heatmap(summary: pd.DataFrame, output_root: Path) -> Path:
    ordered_labels = [LABELS[name] for name in SCENARIOS]
    matrix = summary.set_index("label").loc[ordered_labels, ["low", "middle", "high"]]
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 8.2), gridspec_kw={"width_ratios": [1.25, 1.0]}, constrained_layout=True)
    vmax = max(0.35, float(np.nanmax(np.abs(matrix.to_numpy() - 1.0))))
    sns.heatmap(
        matrix - 1.0, annot=matrix, fmt=".3f", center=0.0,
        cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        cbar_kws={"label": "energy/count minus 1"}, ax=axes[0],
    )
    axes[0].set(xlabel="precision-frequency proxy", ylabel="assumed covariance", title="A. Low/middle/high residual-energy ratios")

    ranked = summary.copy()
    y = np.arange(len(ranked))
    true_low = float(ranked.loc[ranked["scenario"].eq(TRUE), "low"].iloc[0])
    axes[1].axvline(1.0, color="0.2", linestyle="--", linewidth=1.0)
    axes[1].axvline(true_low, color="#169873", linestyle=":", linewidth=1.5, label=f"true={true_low:.3f}")
    colors = ["#169873" if name == TRUE else "#4c78a8" for name in ranked["scenario"]]
    axes[1].barh(y, ranked["low"], color=colors, alpha=0.85)
    axes[1].set_yticks(y, ranked["label"])
    axes[1].invert_yaxis()
    axes[1].set(xlabel="low-third energy/count", title="B. Low-frequency-proxy sensitivity")
    axes[1].grid(alpha=0.18, axis="x")
    axes[1].legend()
    fig.suptitle(f"Simulated {SIM_DATE}: fixed-parameter misspecification sweep (Lanczos 256)", fontsize=15)
    output = output_root / f"{SIM_DATE}_lanczos256_misspecification_heatmap.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_group_curves(curves: pd.DataFrame, args: argparse.Namespace) -> Path:
    final = curves[curves["residual_lanczos_steps"].eq(int(args.residual_lanczos_steps))]
    palette = sns.color_palette("tab10", 10)
    fig, axes = plt.subplots(1, 3, figsize=(19.5, 6.0), constrained_layout=True)
    for ax, (group, names) in zip(axes, GROUPS.items()):
        ax.axvspan(0.0, 1.0 / 3.0, color="#DCEEFF", alpha=0.45, zorder=0)
        ax.axvspan(1.0 / 3.0, 2.0 / 3.0, color="#E8E4FF", alpha=0.45, zorder=0)
        ax.axvspan(2.0 / 3.0, 1.0, color="#FFE8E0", alpha=0.45, zorder=0)
        ax.plot([0.0, 1.0], [0.0, 1.0], color="0.2", linestyle="--", linewidth=1.1, label=r"null: $y=x$")
        for index, name in enumerate(names):
            part = final[final["method"].eq(name)].sort_values("estimated_mode_fraction")
            x = part["estimated_mode_fraction"].to_numpy(float)
            y = part["cumulative_energy_per_n"].to_numpy(float)
            color = "#169873" if name == TRUE else palette[(index - 1) % len(palette)]
            ax.plot(x, y, linewidth=2.2 if name == TRUE else 1.65, color=color, label=LABELS[name])
        ax.axvline(1.0 / 3.0, color="0.6", linewidth=0.9)
        ax.axvline(2.0 / 3.0, color="0.6", linewidth=0.9)
        ax.set(
            xlim=(0, 1), xlabel=r"model-implied cumulative mode fraction $N(\eta)/n$",
            ylabel=r"observed cumulative standardized residual energy $C(\eta)/n$", title=group,
        )
        ax.grid(alpha=0.18)
        ax.legend(fontsize=7.5)
    fig.suptitle(
        "Cumulative spectral residual-energy calibration "
        "(low / middle / high precision-spectrum thirds)",
        fontsize=15,
    )
    output = args.output_root / f"{SIM_DATE}_lanczos256_grouped_cumulative_y_equals_x.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    args = parser().parse_args()
    if int(args.residual_lanczos_steps) != 256:
        raise ValueError("This focused sweep is intentionally fixed at Lanczos resolution 256")
    if args.frequency_bins % 3:
        raise ValueError("--frequency-bins must be divisible by three")
    args.output_root.mkdir(parents=True, exist_ok=True)
    diagnostic.DATE = SIM_DATE
    workflow_started = time.perf_counter()
    truth = base_experiment.load_truth(args.data_root)
    asset = base_experiment.load_asset(args)
    assumed_by_scenario = make_scenarios(truth)
    common_model, beta, precompute_seconds = prepare_common_model(asset, truth, args)
    print(f"Common Vecchia graph precomputed in {precompute_seconds:.2f}s", flush=True)

    all_slq: list[pd.DataFrame] = []
    all_curves: list[pd.DataFrame] = []
    all_bands: list[pd.DataFrame] = []
    all_thirds: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []

    for index, scenario in enumerate(SCENARIOS, start=1):
        started = time.perf_counter()
        assumed = assumed_by_scenario[scenario]
        print(f"\n[{index}/{len(SCENARIOS)}] {scenario}: {LABELS[scenario]}", flush=True)
        print(f"  assumed={assumed}", flush=True)
        precision, build = build_precision(common_model, beta, assumed, args)
        slq, curves, bands, result = diagnostic.evaluate_method(scenario, precision, args)
        metrics = curve_metrics(curves, bands, int(args.residual_lanczos_steps))
        elapsed = time.perf_counter() - started
        third_map = {
            str(row.frequency_third): float(row.energy_per_mode)
            for row in result["thirds"].itertuples(index=False)
        }
        row = {
            "scenario": scenario,
            "label": LABELS[scenario],
            **scenario_metadata(scenario),
            **{f"assumed_{key}": value for key, value in assumed.items()},
            **build,
            **metrics,
            "low": third_map["low"],
            "middle": third_map["middle"],
            "high": third_map["high"],
            "slq_seconds": float(result["summary"]["slq_seconds"]),
            "residual_lanczos_seconds": float(result["summary"]["residual_lanczos_seconds"]),
            "max_curve_change_128_to_256": float(result["summary"]["max_curve_change_previous_level"]),
            "scenario_total_seconds": elapsed,
        }
        summaries.append(row)
        timing_rows.append({key: row[key] for key in (
            "scenario", "precision_build_seconds", "slq_seconds",
            "residual_lanczos_seconds", "scenario_total_seconds",
        )})
        all_slq.append(slq.assign(date=SIM_DATE, method=scenario))
        all_curves.append(curves)
        all_bands.append(bands)
        all_thirds.append(result["thirds"])
        print(
            f"  low/middle/high={third_map['low']:.3f}/{third_map['middle']:.3f}/{third_map['high']:.3f}; "
            f"low max|cum dev|={metrics['low_max_abs_cumulative_deviation']:.3f}; {elapsed:.1f}s",
            flush=True,
        )
        del precision
        gc.collect()

    summary_frame = pd.DataFrame(summaries)
    true_row = summary_frame[summary_frame["scenario"].eq(TRUE)].iloc[0]
    for band in ("low", "middle", "high"):
        summary_frame[f"{band}_minus_true"] = summary_frame[band] - float(true_row[band])
    curve_frame = pd.concat(all_curves, ignore_index=True)
    band_frame = pd.concat(all_bands, ignore_index=True)
    third_frame = pd.concat(all_thirds, ignore_index=True)
    atomic_csv(args.output_root / "scenario_summary.csv", summary_frame)
    atomic_csv(args.output_root / "frequency_third_energy_ratios.csv", third_frame)
    atomic_csv(args.output_root / "frequency_band_energy_ratios.csv", band_frame)
    atomic_csv(args.output_root / "residual_lanczos_cumulative_curves.csv", curve_frame)
    atomic_csv(args.output_root / "slq_spectrum.csv", pd.concat(all_slq, ignore_index=True))
    atomic_csv(args.output_root / "timings.csv", pd.DataFrame(timing_rows))
    heatmap = plot_heatmap(summary_frame, args.output_root)
    grouped_curves = plot_group_curves(curve_frame, args)
    total_seconds = time.perf_counter() - workflow_started
    write_json(
        args.output_root / "RUN_COMPLETE.json",
        {
            "completed": datetime.now().isoformat(timespec="seconds"),
            "date": SIM_DATE,
            "truth": truth,
            "assumed_parameters": assumed_by_scenario,
            "scenario_order": list(SCENARIOS),
            "common_graph_precompute_seconds": precompute_seconds,
            "total_wall_seconds": total_seconds,
            "residual_lanczos_steps": args.residual_lanczos_steps,
            "slq_probes": args.slq_probes,
            "slq_steps": args.slq_steps,
            "main_heatmap": heatmap,
            "grouped_curves": grouped_curves,
            "individual_eigenvectors_computed": False,
            "parameter_fitting_performed": False,
            "microergodic_note": (
                "For the common all-range scale rho and Matern nu=0.5, the proxy "
                "kappa=sigma^2/rho^(2nu)=sigma^2/rho.  It is only a scalar label "
                "for common-scale cases, not for anisotropic one-coordinate changes."
            ),
        },
    )
    del common_model
    print(f"\nComplete in {total_seconds:.2f}s: {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
