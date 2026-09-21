#!/usr/bin/env python3
"""Compare global max-min and contiguous 20x20 four-way eigen diagnostics."""

from __future__ import annotations

import argparse
import json
import os
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


HERE = Path(__file__).resolve().parent


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser(description=__doc__)
    out.add_argument(
        "--maxmin-dir",
        type=Path,
        default=HERE / "real_maxmin400_exact_vs_subset_vecchia_4way_20240703_090326",
    )
    out.add_argument(
        "--contiguous-dir",
        type=Path,
        default=HERE / "real_contiguous20x20_exact_vs_subset_vecchia_4way_20240703_090326",
    )
    out.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "maxmin_vs_contiguous_4way_robustness_20240703_090326",
    )
    return out


def load_run(path: Path, short_name: str) -> dict[str, Any]:
    metadata = json.loads((path / "run_metadata.json").read_text(encoding="utf-8"))
    eigen = pd.read_csv(path / "E1_V1_full_eigen_curves.csv")
    expected = pd.read_csv(path / "vecchia_expected_bands_under_exact_K.csv")
    subspaces = pd.read_csv(path / "exact_vecchia_band_subspace_comparison.csv")
    simulation = pd.read_csv(path / "exact_K_simulated_vecchia_distortion.csv")
    numerical = pd.read_csv(path / "E2_V2_lanczos_accuracy.csv")
    hard_bands = pd.read_csv(path / "E2_V2_hard_band_accuracy.csv")
    return {
        "name": short_name,
        "path": path,
        "metadata": metadata,
        "eigen": eigen,
        "expected": expected,
        "subspaces": subspaces,
        "simulation": simulation,
        "numerical": numerical,
        "hard_bands": hard_bands,
    }


def summary_row(run: dict[str, Any]) -> dict[str, Any]:
    metadata = run["metadata"]
    approximation = metadata["approximation_metrics"]
    operator = metadata["subset_vecchia"]["operator"]
    numerical = run["numerical"]
    hard_bands = run["hard_bands"]
    selected_num = numerical[
        numerical["lanczos_steps"].eq(numerical["lanczos_steps"].max())
        & numerical["reorthogonalization"].eq("full")
    ].set_index("operator")
    selected_band = hard_bands[
        hard_bands["lanczos_steps"].eq(hard_bands["lanczos_steps"].max())
        & hard_bands["reorthogonalization"].eq("full")
    ].set_index("operator")
    simulation = run["simulation"]
    return {
        "selection": run["name"],
        "selection_description": metadata["subset"].get(
            "selection",
            "global max-min among common-valid locations",
        ),
        "n_observations": metadata["subset"]["n_observations"],
        "n_nonempty_blocks": metadata["subset_vecchia"]["n_nonempty_subset_blocks"],
        "whitener_nnz": operator["nnz"],
        "whitener_nnz_per_row_mean": operator["nnz_per_row_mean"],
        "whitener_nnz_per_row_max": operator["nnz_per_row_max"],
        "exact_lanczos_energy_per_n_rmse": selected_num.loc[
            "exact_precision", "energy_per_n_rmse"
        ],
        "vecchia_lanczos_energy_per_n_rmse": selected_num.loc[
            "subset_vecchia_precision", "energy_per_n_rmse"
        ],
        "exact_lanczos_hard_band_rmse": selected_band.loc[
            "exact_precision", "band_energy_per_mode_rmse"
        ],
        "vecchia_lanczos_hard_band_rmse": selected_band.loc[
            "subset_vecchia_precision", "band_energy_per_mode_rmse"
        ],
        **approximation,
        "simulation_curve_rmse_median": simulation["curve_rmse_per_n"].median(),
        "simulation_curve_rmse_q95": simulation["curve_rmse_per_n"].quantile(0.95),
        "simulation_band_rmse_median": simulation[
            "band_energy_per_mode_rmse"
        ].median(),
        "simulation_band_rmse_q95": simulation[
            "band_energy_per_mode_rmse"
        ].quantile(0.95),
    }


def plot_comparison(runs: list[dict[str, Any]], output_dir: Path) -> Path:
    colors = {"global_maxmin": "#d95f02", "contiguous_20x20": "#1b9e77"}
    labels = {
        "global_maxmin": "global max-min",
        "contiguous_20x20": "contiguous 20x20",
    }
    fig, axes = plt.subplots(2, 3, figsize=(18.0, 10.0), constrained_layout=True)
    for run in runs:
        name = run["name"]
        color = colors[name]
        label = labels[name]
        eigen = run["eigen"]
        exact = eigen[eigen["operator"].eq("exact_covariance")].sort_values("rank")
        vecchia = eigen[eigen["operator"].eq("subset_vecchia_covariance")].sort_values("rank")
        x = exact["rank_fraction"].to_numpy()
        difference = (
            vecchia["cumulative_energy_per_n"].to_numpy()
            - exact["cumulative_energy_per_n"].to_numpy()
        )
        axes[0, 0].plot(x, difference, color=color, lw=1.5, label=label)
        axes[0, 1].plot(
            run["expected"]["band"],
            run["expected"]["expected_vecchia_energy_per_mode_under_exact_K"],
            color=color,
            marker="o",
            lw=1.4,
            label=label,
        )
        axes[0, 2].plot(
            run["subspaces"]["band"],
            run["subspaces"]["normalized_projector_frobenius_distance"],
            color=color,
            marker="o",
            lw=1.4,
            label=label,
        )
    axes[0, 0].axhline(0.0, color="0.4", lw=0.8)
    axes[0, 0].set_title("V1 - E1 observed cumulative curve")
    axes[0, 0].set_xlabel("covariance-mode rank fraction")
    axes[0, 0].set_ylabel("difference / n")
    axes[0, 1].axhline(1.0, color="0.4", ls="--", lw=0.8)
    axes[0, 1].set_title("Expected Vecchia energy under exact K")
    axes[0, 1].set_xlabel("Vecchia equal-rank band")
    axes[0, 1].set_ylabel("expected energy per mode")
    axes[0, 2].set_ylim(0.0, 1.0)
    axes[0, 2].set_title("Exact vs Vecchia band subspaces")
    axes[0, 2].set_xlabel("corresponding equal-rank band")
    axes[0, 2].set_ylabel("normalized projector distance")

    curve_data = [run["simulation"]["curve_rmse_per_n"] for run in runs]
    band_data = [run["simulation"]["band_energy_per_mode_rmse"] for run in runs]
    box_labels = [labels[run["name"]] for run in runs]
    axes[1, 0].boxplot(curve_data, tick_labels=box_labels, showfliers=False)
    axes[1, 0].set_title("Exact-K simulations: curve distortion")
    axes[1, 0].set_ylabel("V1-E1 curve RMSE / n")
    axes[1, 1].boxplot(band_data, tick_labels=box_labels, showfliers=False)
    axes[1, 1].set_title("Exact-K simulations: band distortion")
    axes[1, 1].set_ylabel("V1-E1 band energy RMSE")

    summary = pd.DataFrame([summary_row(run) for run in runs]).set_index("selection")
    metric_names = [
        "real_residual_curve_rmse_per_n",
        "covariance_relative_frobenius_error",
        "max_abs_expected_vecchia_band_energy_minus_one",
    ]
    display_names = ["real curve", "covariance Frobenius", "expected band bias"]
    xloc = np.arange(len(metric_names))
    width = 0.34
    for index, run in enumerate(runs):
        axes[1, 2].bar(
            xloc + (index - 0.5) * width,
            summary.loc[run["name"], metric_names].to_numpy(dtype=float),
            width=width,
            color=colors[run["name"]],
            label=labels[run["name"]],
        )
    axes[1, 2].set_xticks(xloc, display_names, rotation=12)
    axes[1, 2].set_yscale("log")
    axes[1, 2].set_title("Vecchia approximation metrics")
    axes[1, 2].set_ylabel("error (log scale)")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    for axis in axes[0, :]:
        axis.legend(fontsize=8)
    axes[1, 2].legend(fontsize=8)
    path = output_dir / "maxmin_vs_contiguous_robustness.png"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    args = parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = [
        load_run(args.maxmin_dir, "global_maxmin"),
        load_run(args.contiguous_dir, "contiguous_20x20"),
    ]
    summary = pd.DataFrame([summary_row(run) for run in runs])
    summary.to_csv(args.output_dir / "selection_robustness_summary.csv", index=False)
    figure = plot_comparison(runs, args.output_dir)
    indexed = summary.set_index("selection")
    maxmin = indexed.loc["global_maxmin"]
    contiguous = indexed.loc["contiguous_20x20"]
    curve_ratio = float(
        maxmin["real_residual_curve_rmse_per_n"]
        / contiguous["real_residual_curve_rmse_per_n"]
    )
    expected_ratio = float(
        maxmin["max_abs_expected_vecchia_band_energy_minus_one"]
        / contiguous["max_abs_expected_vecchia_band_energy_minus_one"]
    )
    text = f"""# Global max-min versus contiguous 20x20 robustness

Both experiments use 400 locations x 8 times, the same stored adapted 4/3/2
parameters, and the same four-way E1--E2--V1--V2 construction.

| metric | global max-min | contiguous 20x20 |
|---|---:|---:|
| mean B nonzeros/row | {maxmin['whitener_nnz_per_row_mean']:.2f} | {contiguous['whitener_nnz_per_row_mean']:.2f} |
| E1--V1 real curve RMSE/n | {maxmin['real_residual_curve_rmse_per_n']:.6g} | {contiguous['real_residual_curve_rmse_per_n']:.6g} |
| real 20-band RMSE | {maxmin['real_residual_band_energy_per_mode_rmse']:.6g} | {contiguous['real_residual_band_energy_per_mode_rmse']:.6g} |
| covariance relative Frobenius error | {maxmin['covariance_relative_frobenius_error']:.6g} | {contiguous['covariance_relative_frobenius_error']:.6g} |
| KL exact-to-Vecchia / observation | {maxmin['kl_exact_to_vecchia_per_observation']:.6g} | {contiguous['kl_exact_to_vecchia_per_observation']:.6g} |
| max expected 20-band bias | {maxmin['max_abs_expected_vecchia_band_energy_minus_one']:.6g} | {contiguous['max_abs_expected_vecchia_band_energy_minus_one']:.6g} |
| simulated curve RMSE median | {maxmin['simulation_curve_rmse_median']:.6g} | {contiguous['simulation_curve_rmse_median']:.6g} |
| simulated band RMSE median | {maxmin['simulation_band_rmse_median']:.6g} | {contiguous['simulation_band_rmse_median']:.6g} |

The observed E1--V1 curve discrepancy is {curve_ratio:.2f} times larger under
global max-min thinning, and the maximum expected band bias is
{expected_ratio:.2f} times larger.  Conditioning density is therefore a major
driver of the apparent Vecchia spectral error.  The contiguous result is much
closer to the full-data operator, whose B has about 131 nonzeros per row, but it
still does not directly prove the full-data Vecchia approximation error.

Hard-band Lanczos errors remain around 0.02 at m=512 and 32 probes, so smooth
spectral filters and probe convergence are still required before statistical
calibration.
"""
    (args.output_dir / "RESULTS.md").write_text(text, encoding="utf-8")
    metadata = {
        "maxmin_dir": str(args.maxmin_dir.resolve()),
        "contiguous_dir": str(args.contiguous_dir.resolve()),
        "figure": str(figure.resolve()),
        "curve_error_ratio_maxmin_over_contiguous": curve_ratio,
        "expected_band_bias_ratio_maxmin_over_contiguous": expected_ratio,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(f"Saved comparison to {args.output_dir}")


if __name__ == "__main__":
    main()
