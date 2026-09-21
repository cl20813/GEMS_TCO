#!/usr/bin/env python3
"""Validate the residual-Lanczos + SLQ diagnostic on one simulated ST day.

The same simulated realization and the same lag-6/4/3 Vecchia graph are used
for three *fixed*, non-fitted covariance specifications:

1. the data-generating covariance parameters;
2. the true parameters except that nugget=0 instead of nugget=1;
3. sigma^2 and every spatial/temporal range set to twice their true values.

For each assumed precision Omega, residual-started Lanczos estimates

    C(eta) = r.T Omega 1{Omega <= eta} r,

while paired random-probe SLQ estimates

    N(eta) = tr 1{Omega <= eta}.

Under the assumed Gaussian model, increments satisfy E[dC] = dN.  Therefore
C(eta)/n should follow N(eta)/n and each equal-mode-count band should have
energy/count near one.  No individual eigenvectors are computed.
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
import scipy.sparse
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
from vecchia_sparse_precision_operator_090326 import (  # noqa: E402
    build_sparse_vecchia_precision,
    native_gls_quadratic,
    verify_precision_identity,
)


SIM_DATE = "2024-07-13"
SIM_YEAR = 2024
SIM_DAY_INDEX = 12
SCENARIOS = ("true", "wrong_nugget", "wrong_scale_range")
LABELS = {
    "true": "true parameters",
    "wrong_nugget": "wrong nugget: 1 → 0",
    "wrong_scale_range": "wrong σ²/ranges: 2×",
}
COLORS = {
    "true": "#169873",
    "wrong_nugget": "#d62728",
    "wrong_scale_range": "#7b4ab0",
}
LINESTYLES = {"true": "-", "wrong_nugget": "-.", "wrong_scale_range": ":"}
TRUE_DEFAULTS = {
    "smooth": 0.5,
    "sigmasq": 10.0,
    "range_lat": 0.2,
    "range_lon": 0.3,
    "range_time": 2.0,
    "advec_lat": 0.08,
    "advec_lon": -0.2,
    "nugget": 1.0,
    "mean_intercept": 260.0,
    "mean_lat_slope": 1.0,
    "mean_lat_center": -0.5,
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
        / "outputs/summer_26/sim_20240713_true_vs_misspecified_residual_lanczos512_slq8x192",
    )
    out.add_argument("--residual-lanczos-steps", type=int, default=512)
    out.add_argument("--slq-probes", type=int, default=8)
    out.add_argument("--slq-steps", type=int, default=192)
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


def load_truth(data_root: Path) -> dict[str, float]:
    path = (
        data_root
        / f"{SIM_YEAR}_july_st_circulant"
        / f"sim_july{SIM_YEAR}_st_circulant_truth.json"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Simulation truth file is missing: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    truth = dict(TRUE_DEFAULTS)
    truth.update(raw)
    expected = {
        "smooth": 0.5,
        "sigmasq": 10.0,
        "range_lat": 0.2,
        "range_lon": 0.3,
        "range_time": 2.0,
        "advec_lat": 0.08,
        "advec_lon": -0.2,
        "nugget": 1.0,
    }
    for key, value in expected.items():
        if not np.isclose(float(truth[key]), value, rtol=0.0, atol=1e-10):
            raise RuntimeError(f"Unexpected simulation truth {key}={truth[key]}, expected {value}")
    return {key: float(value) if isinstance(value, (int, float)) else value for key, value in truth.items()}


def load_asset(args: argparse.Namespace) -> sim_data.DayAsset:
    load_args = argparse.Namespace(
        data_root=args.data_root,
        years=[str(SIM_YEAR)],
        month=7,
        sim_kind="gridded",
        days=str(SIM_DAY_INDEX),
        hours_per_day=8,
        lat_range="-3,2",
        lon_range="121,131",
        keep_exact_loc=True,
    )
    assets = sim_data.load_day_assets(load_args)
    if len(assets) != 1:
        raise RuntimeError(f"Expected one simulated day, got {len(assets)}")
    asset = assets[0]
    if str(asset.day_label) != SIM_DATE:
        raise RuntimeError(f"Selected {asset.day_label}; expected {SIM_DATE}")
    return asset


def scenario_parameters(truth: dict[str, float]) -> dict[str, dict[str, float]]:
    covariance_keys = (
        "smooth",
        "sigmasq",
        "range_lat",
        "range_lon",
        "range_time",
        "advec_lat",
        "advec_lon",
        "nugget",
    )
    base = {key: float(truth[key]) for key in covariance_keys}
    wrong_nugget = dict(base)
    wrong_nugget["nugget"] = 0.0
    wrong_scale_range = dict(base)
    for key in ("sigmasq", "range_lat", "range_lon", "range_time"):
        wrong_scale_range[key] = 2.0 * base[key]
    return {
        "true": base,
        "wrong_nugget": wrong_nugget,
        "wrong_scale_range": wrong_scale_range,
    }


def true_mean_beta(model: Any, asset: sim_data.DayAsset, truth: dict[str, float]) -> np.ndarray:
    # The stored response was centered by asset.monthly_mean.  The model design
    # is [1, latitude-model.lat_mean_val, seven hour indicators].
    intercept = (
        float(truth["mean_intercept"])
        + float(truth["mean_lat_slope"])
        * (float(model.lat_mean_val) - float(truth["mean_lat_center"]))
        - float(asset.monthly_mean)
    )
    beta = np.zeros(9, dtype=np.float64)
    beta[0] = intercept
    beta[1] = float(truth["mean_lat_slope"])
    return beta


def build_precision(
    scenario: str,
    assumed: dict[str, float],
    truth: dict[str, float],
    asset: sim_data.DayAsset,
    args: argparse.Namespace,
) -> tuple[Any, dict[str, Any]]:
    source_map = {
        key: tensor.to(device="cpu", dtype=torch.float64).contiguous()
        for key, tensor in asset.source_map.items()
    }
    model = sim_models.RealDataCorridorWidth4x4Lag643FixedNuggetSplineFit(
        smooth=float(assumed["smooth"]),
        fixed_nugget=float(assumed["nugget"]),
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
    beta = true_mean_beta(model, asset, truth)
    params = torch.as_tensor(
        sim_data.physical_to_log_phi(assumed), dtype=torch.float64, device="cpu"
    )
    native_quadratic_value = native_gls_quadratic(model, params, beta)
    precision = build_sparse_vecchia_precision(
        model,
        params,
        beta,
        chunk_size=int(args.target_chunk_size),
        coefficient_drop_tolerance=float(args.coefficient_drop_tolerance),
        progress=lambda message: print(f"    {scenario} B: {message}", flush=True),
    )
    identity = verify_precision_identity(precision, native_quadratic_value)
    if float(identity["relative_error"]) > 1e-8:
        raise RuntimeError(f"Sparse precision identity failed for {scenario}: {identity}")
    summary = {
        "scenario": scenario,
        "assumed_parameters": assumed,
        "true_mean_beta": beta,
        "precompute_seconds": precompute_seconds,
        "precision_build_seconds": float(precision.metadata["build_s"]),
        "precision_nnz": int(precision.whitener.nnz),
        "n_observations": int(precision.n),
        "native_quadratic": native_quadratic_value,
        "direct_energy_per_observation": float(identity["sparse_quadratic"] / precision.n),
        "precision_identity_relative_error": float(identity["relative_error"]),
    }
    precision.metadata.update(summary)
    del model, params, source_map
    gc.collect()
    return precision, summary


def shade_thirds(ax: plt.Axes) -> None:
    colors = ("#DCEEFF", "#E9E5FF", "#FFE4DC")
    for index, (name, color) in enumerate(zip(("low", "middle", "high"), colors)):
        left, right = index / 3.0, (index + 1) / 3.0
        ax.axvspan(left, right, color=color, alpha=0.42, zorder=0)
        ax.text(
            0.5 * (left + right), 0.985, name,
            transform=ax.get_xaxis_transform(), ha="center", va="top",
            color="0.4", fontsize=9,
        )


def plot_diagnostic(
    curves: pd.DataFrame,
    bands: pd.DataFrame,
    thirds: pd.DataFrame,
    args: argparse.Namespace,
) -> Path:
    final_steps = int(args.residual_lanczos_steps)
    final_curves = curves[curves["residual_lanczos_steps"].eq(final_steps)]
    final_bands = bands[bands["residual_lanczos_steps"].eq(final_steps)]
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 11.0), constrained_layout=True)

    shade_thirds(axes[0, 0])
    axes[0, 0].plot([0, 1], [0, 1], color="0.2", linestyle="--", linewidth=1.1)
    for scenario in SCENARIOS:
        part = final_curves[final_curves["method"].eq(scenario)].sort_values(
            "estimated_mode_fraction"
        )
        axes[0, 0].plot(
            part["estimated_mode_fraction"], part["cumulative_energy_per_n"],
            color=COLORS[scenario], linestyle=LINESTYLES[scenario], linewidth=2.1,
            label=LABELS[scenario],
        )
    axes[0, 0].set(
        xlim=(0, 1), xlabel=r"SLQ mode fraction $N(\eta)/n$ (low → high precision)",
        ylabel=r"residual energy $C(\eta)/n$",
        title="A. Cumulative residual spectral energy",
    )
    axes[0, 0].grid(alpha=0.2)
    axes[0, 0].legend(fontsize=9)

    shade_thirds(axes[0, 1])
    axes[0, 1].axhline(0.0, color="0.2", linestyle="--", linewidth=1.1)
    for scenario in SCENARIOS:
        part = final_curves[final_curves["method"].eq(scenario)].sort_values(
            "estimated_mode_fraction"
        )
        x = part["estimated_mode_fraction"].to_numpy(float)
        y = part["cumulative_energy_per_n"].to_numpy(float) - x
        axes[0, 1].plot(
            x, y, color=COLORS[scenario], linestyle=LINESTYLES[scenario],
            linewidth=2.1, label=LABELS[scenario],
        )
    axes[0, 1].set(
        xlim=(0, 1), xlabel="SLQ mode fraction (low → high precision)",
        ylabel=r"deviation $C(\eta)/n-N(\eta)/n$",
        title="B. Deviation from the null line",
    )
    axes[0, 1].grid(alpha=0.2)
    axes[0, 1].legend(fontsize=9)

    x = np.arange(1, int(args.frequency_bins) + 1)
    width = 0.24
    for index, scenario in enumerate(SCENARIOS):
        part = final_bands[final_bands["method"].eq(scenario)].sort_values("frequency_bin")
        axes[1, 0].bar(
            x + (index - 1) * width, part["energy_per_mode"], width=width,
            color=COLORS[scenario], alpha=0.82, label=LABELS[scenario],
        )
    axes[1, 0].axhline(1.0, color="0.2", linestyle="--", linewidth=1.1)
    for edge in (args.frequency_bins / 3 + 0.5, 2 * args.frequency_bins / 3 + 0.5):
        axes[1, 0].axvline(edge, color="0.65", linewidth=0.9)
    axes[1, 0].set(
        xticks=x, xlabel="equal-mode-count bin (low → high precision)",
        ylabel=r"energy/count $E_b/N_b$", title="C. Twelve-bin localization",
    )
    axes[1, 0].grid(alpha=0.18, axis="y")
    axes[1, 0].legend(fontsize=8)

    third_order = ("low", "middle", "high")
    x3 = np.arange(3)
    width3 = 0.24
    for index, scenario in enumerate(SCENARIOS):
        part = thirds[thirds["method"].eq(scenario)].set_index("frequency_third").loc[list(third_order)]
        axes[1, 1].bar(
            x3 + (index - 1) * width3,
            part["energy_per_mode"], width=width3,
            color=COLORS[scenario], alpha=0.82, label=LABELS[scenario],
        )
    axes[1, 1].axhline(1.0, color="0.2", linestyle="--", linewidth=1.1)
    axes[1, 1].set(
        xticks=x3, xticklabels=third_order,
        xlabel="broad precision-frequency proxy", ylabel=r"energy/count $E_b/N_b$",
        title="D. Primary low/middle/high summary",
    )
    axes[1, 1].grid(alpha=0.18, axis="y")
    axes[1, 1].legend(fontsize=8)

    fig.suptitle(
        f"Simulated {SIM_DATE}: true vs fixed parameter misspecification\n"
        "one common lag-6/4/3 Vecchia graph; residual-Lanczos + paired SLQ",
        fontsize=15,
    )
    output = args.output_root / f"{SIM_DATE}_true_vs_misspecified_residual_lanczos_slq.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_convergence(curves: pd.DataFrame, args: argparse.Namespace) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.2), constrained_layout=True)
    for ax, scenario in zip(axes, SCENARIOS):
        shade_thirds(ax)
        ax.plot([0, 1], [0, 1], color="0.2", linestyle="--", linewidth=1.0)
        subset = curves[curves["method"].eq(scenario)]
        for steps, part in subset.groupby("residual_lanczos_steps"):
            part = part.sort_values("estimated_mode_fraction")
            ax.plot(
                part["estimated_mode_fraction"], part["cumulative_energy_per_n"],
                linewidth=1.5, label=f"m={int(steps)}",
            )
        ax.set(
            xlim=(0, 1), xlabel="SLQ mode fraction", ylabel="cumulative energy / n",
            title=LABELS[scenario],
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Residual-Lanczos convergence")
    output = args.output_root / f"{SIM_DATE}_residual_lanczos_convergence.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    args = parser().parse_args()
    if args.residual_lanczos_steps < 128:
        raise ValueError("Use at least 128 residual Lanczos steps")
    if args.slq_probes < 2 or args.slq_steps < 32:
        raise ValueError("Use at least 2 SLQ probes and 32 SLQ steps")
    if args.frequency_bins < 3 or args.frequency_bins % 3:
        raise ValueError("--frequency-bins must be divisible by three")
    args.output_root.mkdir(parents=True, exist_ok=True)
    diagnostic.DATE = SIM_DATE
    workflow_started = time.perf_counter()
    truth = load_truth(args.data_root)
    asset = load_asset(args)
    assumed_by_scenario = scenario_parameters(truth)
    all_slq: list[pd.DataFrame] = []
    all_curves: list[pd.DataFrame] = []
    all_bands: list[pd.DataFrame] = []
    all_thirds: list[pd.DataFrame] = []
    all_summaries: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []

    for scenario in SCENARIOS:
        scenario_started = time.perf_counter()
        assumed = assumed_by_scenario[scenario]
        print(f"\n{scenario}: build fixed-parameter sparse precision", flush=True)
        print(f"  assumed={assumed}", flush=True)
        precision, build_summary = build_precision(
            scenario, assumed, truth, asset, args
        )
        print(
            f"  n={precision.n:,}, nnz={precision.whitener.nnz:,}, "
            f"direct energy/n={build_summary['direct_energy_per_observation']:.4f}",
            flush=True,
        )
        slq, curves, bands, result = diagnostic.evaluate_method(
            scenario, precision, args
        )
        summary = {**build_summary, **result["summary"]}
        summary["scenario_total_seconds"] = time.perf_counter() - scenario_started
        final_curve = curves[
            curves["residual_lanczos_steps"].eq(int(args.residual_lanczos_steps))
        ]
        summary["max_absolute_cumulative_deviation"] = float(
            np.max(
                np.abs(
                    final_curve["cumulative_energy_per_n"].to_numpy(float)
                    - final_curve["estimated_mode_fraction"].to_numpy(float)
                )
            )
        )
        final_bands = bands[
            bands["residual_lanczos_steps"].eq(int(args.residual_lanczos_steps))
        ]
        summary["twelve_bin_rmse_from_one"] = float(
            np.sqrt(np.mean(np.square(final_bands["energy_per_mode"].to_numpy(float) - 1.0)))
        )
        all_summaries.append(summary)
        all_slq.append(slq.assign(date=SIM_DATE, method=scenario))
        all_curves.append(curves)
        all_bands.append(bands)
        all_thirds.append(result["thirds"])
        timing_rows.append(
            {
                "scenario": scenario,
                "precompute_seconds": build_summary["precompute_seconds"],
                "precision_build_seconds": build_summary["precision_build_seconds"],
                "slq_seconds": summary["slq_seconds"],
                "residual_lanczos_seconds": summary["residual_lanczos_seconds"],
                "scenario_total_seconds": summary["scenario_total_seconds"],
            }
        )
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

    slq_frame = pd.concat(all_slq, ignore_index=True)
    curve_frame = pd.concat(all_curves, ignore_index=True)
    band_frame = pd.concat(all_bands, ignore_index=True)
    third_frame = pd.concat(all_thirds, ignore_index=True)
    timing_frame = pd.DataFrame(timing_rows)
    atomic_csv(args.output_root / "slq_spectrum.csv", slq_frame)
    atomic_csv(args.output_root / "residual_lanczos_cumulative_curves.csv", curve_frame)
    atomic_csv(args.output_root / "frequency_band_energy_ratios.csv", band_frame)
    atomic_csv(args.output_root / "frequency_third_energy_ratios.csv", third_frame)
    atomic_csv(args.output_root / "timings.csv", timing_frame)
    write_json(args.output_root / "scenario_summaries.json", all_summaries)
    main_plot = plot_diagnostic(curve_frame, band_frame, third_frame, args)
    convergence_plot = plot_convergence(curve_frame, args)
    total_seconds = time.perf_counter() - workflow_started
    write_json(
        args.output_root / "RUN_COMPLETE.json",
        {
            "completed": datetime.now().isoformat(timespec="seconds"),
            "date": SIM_DATE,
            "truth": truth,
            "assumed_parameters": assumed_by_scenario,
            "scenarios": list(SCENARIOS),
            "total_wall_seconds": total_seconds,
            "residual_lanczos_steps": args.residual_lanczos_steps,
            "slq_probes": args.slq_probes,
            "slq_steps": args.slq_steps,
            "main_plot": main_plot,
            "convergence_plot": convergence_plot,
            "individual_eigenvectors_computed": False,
            "parameter_fitting_performed": False,
            "mean_handling": "known DGP mean removed; no GLS or covariance fitting",
            "slq_probe_pairing": "identical Rademacher probes for all scenarios",
            "important_scope": (
                "The DGP is the full circulant-embedding Gaussian process. "
                "All three diagnostics use one common lag-6/4/3 Vecchia graph, so "
                "the true-parameter curve also includes Vecchia approximation error."
            ),
        },
    )
    print(f"\nComplete in {total_seconds:.2f}s: {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
