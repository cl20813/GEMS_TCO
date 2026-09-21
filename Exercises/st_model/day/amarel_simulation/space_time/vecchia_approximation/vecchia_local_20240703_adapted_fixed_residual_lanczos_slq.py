#!/usr/bin/env python3
"""One-day full-data frequency-failure diagnostic on a local CPU.

For adapted and fixed lag-6/4/3 Vecchia precisions on 2024-07-03, estimate

    C(eta) = r.T Omega 1{Omega <= eta} r

with residual-started Lanczos and

    N(eta) = tr 1{Omega <= eta}

with paired random-probe SLQ.  No individual Ritz eigenvectors are selected.
The diagnostic compares C(eta)/n against N(eta)/n and reports incremental
energy/count ratios in equal-mode-count spectral bands.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
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
import scipy.linalg
import scipy.sparse


HERE = Path(__file__).resolve().parent
REPO = next(parent for parent in HERE.parents if (parent / "src/GEMS_TCO").is_dir())
SRC = REPO / "src"
INTERACTION = HERE.parent / "interaction_diagnostic"
for candidate in (HERE, SRC, INTERACTION):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import vecchia_real59_adapted_fixed_threeway_slq512_lag643 as engine  # noqa: E402


DATE = "2024-07-03"
METHODS = ("adapted", "fixed")
LABELS = {"adapted": "adapted corridor", "fixed": "fixed center"}
COLORS = {"adapted": "#1f77b4", "fixed": "#e41a1c"}
LINESTYLES = {"adapted": "-", "fixed": "-."}
BAND_NAMES = ("low", "middle", "high")
BAND_COLORS = {"low": "#DCEEFF", "middle": "#E9E5FF", "high": "#FFE4DC"}


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--operator-cache-root",
        type=Path,
        default=(
            REPO / "outputs/summer_26/vecchia_local_20240703_lag643_operator_cache"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            REPO
            / "outputs/summer_26/vecchia_local_20240703_adapted_fixed_"
            "residual_lanczos512_slq8x192"
        ),
    )
    parser.add_argument("--residual-lanczos-steps", type=int, default=512)
    parser.add_argument("--slq-probes", type=int, default=8)
    parser.add_argument("--slq-steps", type=int, default=192)
    parser.add_argument("--curve-points", type=int, default=121)
    parser.add_argument("--frequency-bins", type=int, default=12)
    parser.add_argument("--random-seed", type=int, default=20260908)
    return parser


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_ready(value), indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def load_precision(root: Path, method: str) -> tuple[CachedPrecision, float]:
    method_root = root / method
    required = {
        "whitener": method_root / "whitener_B.npz",
        "residual": method_root / "residual.npy",
        "metadata": method_root / "metadata.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing cached precision files: {missing}")
    started = time.perf_counter()
    precision = CachedPrecision(
        whitener=scipy.sparse.load_npz(required["whitener"]).tocsr(),
        residual=np.load(required["residual"]),
        metadata=json.loads(required["metadata"].read_text(encoding="utf-8")),
    )
    return precision, time.perf_counter() - started


def quadrature_atoms(run: Any, steps: int) -> tuple[np.ndarray, np.ndarray]:
    completed = min(int(steps), int(run.steps))
    if completed < 2:
        raise RuntimeError(f"Lanczos run has only {completed} usable steps")
    values, vectors = scipy.linalg.eigh_tridiagonal(
        np.asarray(run.alpha[:completed], dtype=np.float64),
        np.asarray(run.beta[: completed - 1], dtype=np.float64),
        check_finite=False,
    )
    scale = max(float(np.max(np.abs(values))), 1.0)
    keep = np.isfinite(values) & (values > np.finfo(np.float64).eps * scale)
    values = values[keep]
    weights = float(run.start_norm_sq) * np.square(vectors[0, keep])
    order = np.argsort(values)
    return values[order], weights[order]


def weights_from_cumulative(cumulative: np.ndarray) -> np.ndarray:
    return np.diff(np.concatenate(([0.0], np.asarray(cumulative, dtype=np.float64))))


def quantile_thresholds(
    atoms: np.ndarray,
    cumulative: np.ndarray,
    n: int,
    fractions: np.ndarray,
) -> np.ndarray:
    thresholds = np.empty(len(fractions), dtype=np.float64)
    thresholds[0] = -np.inf
    thresholds[-1] = np.inf
    for index, fraction in enumerate(fractions[1:-1], start=1):
        position = int(np.searchsorted(cumulative, float(fraction) * n, side="left"))
        thresholds[index] = float(atoms[min(max(position, 0), len(atoms) - 1)])
    return thresholds


def cumulative_at_thresholds(
    atoms: np.ndarray,
    terms: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    order = np.argsort(atoms)
    sorted_atoms = np.asarray(atoms[order], dtype=np.float64)
    cumulative = np.cumsum(np.asarray(terms[order], dtype=np.float64))
    positions = np.searchsorted(sorted_atoms, thresholds, side="right") - 1
    result = np.zeros(len(thresholds), dtype=np.float64)
    valid = positions >= 0
    result[valid] = cumulative[positions[valid]]
    return result


def evaluate_method(
    method: str,
    precision: CachedPrecision,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    # Recreate the same Rademacher probes for adapted and fixed: this makes the
    # SLQ trace-estimation noise paired across the two model comparisons.
    slq_rng = np.random.default_rng(int(args.random_seed))
    slq_args = argparse.Namespace(
        slq_probes=int(args.slq_probes),
        slq_steps=int(args.slq_steps),
        spectrum_grid=400,
        band_mode_counts=(170, 170, 172),
    )
    print(f"  {method}: paired SLQ {args.slq_probes} x {args.slq_steps}", flush=True)
    slq_started = time.perf_counter()
    slq_curve, _, slq_summary, slq_atoms, slq_cumulative = engine.slq_spectrum(
        precision, slq_args, slq_rng
    )
    slq_wall = time.perf_counter() - slq_started
    slq_weights = weights_from_cumulative(slq_cumulative)

    print(
        f"  {method}: residual-started Lanczos m={args.residual_lanczos_steps}",
        flush=True,
    )
    residual_started = time.perf_counter()
    residual_run = engine.lanczos_tridiagonal(
        precision.matvec,
        np.asarray(precision.residual, dtype=np.float64),
        max_steps=int(args.residual_lanczos_steps),
    )
    residual_wall = time.perf_counter() - residual_started
    print(
        f"    residual Lanczos completed m={residual_run.steps} in "
        f"{residual_wall:.2f}s",
        flush=True,
    )

    curve_fractions = np.linspace(0.0, 1.0, int(args.curve_points))
    curve_thresholds = quantile_thresholds(
        slq_atoms, slq_cumulative, precision.n, curve_fractions
    )
    estimated_counts = cumulative_at_thresholds(
        slq_atoms, slq_weights, curve_thresholds
    )
    convergence_frames: list[pd.DataFrame] = []
    requested_levels = sorted(
        set(
            level
            for level in (128, 256, int(args.residual_lanczos_steps))
            if level <= int(residual_run.steps)
        )
    )
    for steps in requested_levels:
        residual_atoms, residual_weights = quadrature_atoms(residual_run, steps)
        energy_terms = residual_atoms * residual_weights
        cumulative_energy = cumulative_at_thresholds(
            residual_atoms, energy_terms, curve_thresholds
        )
        convergence_frames.append(
            pd.DataFrame(
                {
                    "date": DATE,
                    "method": method,
                    "residual_lanczos_steps": steps,
                    "target_mode_fraction": curve_fractions,
                    "estimated_mode_fraction": estimated_counts / precision.n,
                    "cumulative_energy_per_n": cumulative_energy / precision.n,
                    "precision_threshold": curve_thresholds,
                }
            )
        )
    curves = pd.concat(convergence_frames, ignore_index=True)

    bin_fractions = np.linspace(0.0, 1.0, int(args.frequency_bins) + 1)
    bin_thresholds = quantile_thresholds(
        slq_atoms, slq_cumulative, precision.n, bin_fractions
    )
    cumulative_counts = cumulative_at_thresholds(
        slq_atoms, slq_weights, bin_thresholds
    )
    band_frames: list[pd.DataFrame] = []
    for steps in requested_levels:
        residual_atoms, residual_weights = quadrature_atoms(residual_run, steps)
        cumulative_energy = cumulative_at_thresholds(
            residual_atoms, residual_atoms * residual_weights, bin_thresholds
        )
        counts = np.diff(cumulative_counts)
        energy = np.diff(cumulative_energy)
        ratio = np.divide(
            energy,
            counts,
            out=np.full_like(energy, np.nan),
            where=counts > 0.0,
        )
        standard_error = np.sqrt(2.0 / counts)
        band_frames.append(
            pd.DataFrame(
                {
                    "date": DATE,
                    "method": method,
                    "residual_lanczos_steps": steps,
                    "frequency_bin": np.arange(1, len(counts) + 1),
                    "mode_fraction_lower": bin_fractions[:-1],
                    "mode_fraction_upper": bin_fractions[1:],
                    "precision_lower": bin_thresholds[:-1],
                    "precision_upper": bin_thresholds[1:],
                    "estimated_mode_count": counts,
                    "estimated_residual_energy": energy,
                    "energy_per_mode": ratio,
                    "known_parameter_chi2_standard_error": standard_error,
                }
            )
        )
    bands = pd.concat(band_frames, ignore_index=True)
    final_bands = bands[
        bands["residual_lanczos_steps"].eq(int(residual_run.steps))
    ].copy()
    if len(final_bands) != int(args.frequency_bins):
        raise RuntimeError("Final frequency-band table has an unexpected size")

    third_rows: list[dict[str, Any]] = []
    bins_per_third = int(args.frequency_bins) // 3
    if bins_per_third * 3 != int(args.frequency_bins):
        raise ValueError("--frequency-bins must be divisible by three")
    for third_index, band_name in enumerate(BAND_NAMES):
        subset = final_bands.iloc[
            third_index * bins_per_third : (third_index + 1) * bins_per_third
        ]
        count = float(subset["estimated_mode_count"].sum())
        energy = float(subset["estimated_residual_energy"].sum())
        third_rows.append(
            {
                "date": DATE,
                "method": method,
                "frequency_third": band_name,
                "estimated_mode_count": count,
                "estimated_residual_energy": energy,
                "energy_per_mode": energy / count,
                "known_parameter_chi2_standard_error": math.sqrt(2.0 / count),
            }
        )
    thirds = pd.DataFrame(third_rows)

    final_curve = curves[
        curves["residual_lanczos_steps"].eq(int(residual_run.steps))
    ].sort_values("target_mode_fraction")
    previous_level = requested_levels[-2] if len(requested_levels) > 1 else requested_levels[-1]
    previous_curve = curves[
        curves["residual_lanczos_steps"].eq(previous_level)
    ].sort_values("target_mode_fraction")
    convergence_max = float(
        np.max(
            np.abs(
                final_curve["cumulative_energy_per_n"].to_numpy(float)
                - previous_curve["cumulative_energy_per_n"].to_numpy(float)
            )
        )
    )
    endpoint = float(final_curve["cumulative_energy_per_n"].iloc[-1])
    direct_endpoint = float(
        np.asarray(precision.residual) @ precision.matvec(precision.residual)
        / precision.n
    )
    summary = {
        "date": DATE,
        "method": method,
        "n_observations": precision.n,
        "slq_probes": int(args.slq_probes),
        "slq_steps": int(args.slq_steps),
        "slq_seconds": float(slq_wall),
        "residual_lanczos_requested_steps": int(args.residual_lanczos_steps),
        "residual_lanczos_completed_steps": int(residual_run.steps),
        "residual_lanczos_seconds": float(residual_wall),
        "residual_lanczos_matvec_seconds": float(residual_run.matvec_seconds),
        "endpoint_energy_per_n": endpoint,
        "direct_quadratic_energy_per_n": direct_endpoint,
        "endpoint_absolute_error": abs(endpoint - direct_endpoint),
        "max_curve_change_previous_level": convergence_max,
        "previous_convergence_level": int(previous_level),
        "slq_precision_min": float(slq_summary["slq_precision_min"]),
        "slq_precision_max": float(slq_summary["slq_precision_max"]),
        "operator_build_seconds_from_cache_metadata": float(
            precision.metadata.get("build_s", np.nan)
        ),
        "operator_precompute_seconds_from_cache_metadata": float(
            precision.metadata.get("precompute_seconds", np.nan)
        ),
    }
    for row in thirds.itertuples(index=False):
        summary[f"{row.frequency_third}_energy_per_mode"] = float(row.energy_per_mode)
    return slq_curve, curves, bands, {"summary": summary, "thirds": thirds}


def shade_thirds(ax: plt.Axes) -> None:
    for index, band in enumerate(BAND_NAMES):
        left, right = index / 3.0, (index + 1) / 3.0
        ax.axvspan(left, right, color=BAND_COLORS[band], alpha=0.48, zorder=0)
        ax.text(
            0.5 * (left + right), 0.985, band,
            transform=ax.get_xaxis_transform(), ha="center", va="top",
            color="0.35", fontsize=9,
        )


def plot_results(
    curves: pd.DataFrame,
    bands: pd.DataFrame,
    timings: pd.DataFrame,
    args: argparse.Namespace,
) -> Path:
    final_steps = int(args.residual_lanczos_steps)
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8), constrained_layout=True)
    shade_thirds(axes[0])
    axes[0].plot([0, 1], [0, 1], color="0.25", linestyle="--", linewidth=1.1)
    for method in METHODS:
        part = curves[
            curves["method"].eq(method)
            & curves["residual_lanczos_steps"].eq(final_steps)
        ].sort_values("target_mode_fraction")
        axes[0].plot(
            part["estimated_mode_fraction"], part["cumulative_energy_per_n"],
            color=COLORS[method], linestyle=LINESTYLES[method], linewidth=2.0,
            label=LABELS[method],
        )
    axes[0].set(
        xlim=(0, 1),
        xlabel=r"SLQ mode fraction $N(\eta)/n$ (low $\to$ high precision)",
        ylabel=r"residual spectral energy $C(\eta)/n$",
        title="Residual-started cumulative spectral energy",
    )
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    final_bands = bands[bands["residual_lanczos_steps"].eq(final_steps)]
    x = np.arange(1, int(args.frequency_bins) + 1)
    width = 0.37
    for offset, method in zip((-width / 2, width / 2), METHODS):
        part = final_bands[final_bands["method"].eq(method)].sort_values(
            "frequency_bin"
        )
        axes[1].bar(
            x + offset, part["energy_per_mode"], width=width,
            color=COLORS[method], alpha=0.78, label=LABELS[method],
        )
    axes[1].axhline(1.0, color="0.25", linestyle="--", linewidth=1.1)
    for edge in (args.frequency_bins / 3 + 0.5, 2 * args.frequency_bins / 3 + 0.5):
        axes[1].axvline(edge, color="0.65", linewidth=0.9)
    axes[1].set(
        xticks=x,
        xlabel="equal-mode-count frequency bin (low → high precision)",
        ylabel=r"band residual energy / estimated mode count $E_b/N_b$",
        title=f"{args.frequency_bins}-band frequency-failure diagnostic",
    )
    axes[1].grid(alpha=0.18, axis="y")
    axes[1].legend()

    indices = np.arange(len(METHODS))
    bottom = np.zeros(len(METHODS))
    for column, label, color in (
        ("precision_cache_load_seconds", "load cached precision", "#999999"),
        ("slq_seconds", "paired random-probe SLQ", "#66c2a5"),
        ("residual_lanczos_seconds", "residual-started Lanczos", "#fc8d62"),
    ):
        values = np.asarray(
            [float(timings.loc[timings["method"].eq(m), column].iloc[0]) for m in METHODS]
        )
        axes[2].bar(indices, values, bottom=bottom, color=color, label=label)
        bottom += values
    axes[2].set_xticks(indices, [LABELS[m] for m in METHODS], rotation=10)
    axes[2].set(
        ylabel="wall time (seconds)",
        title="Local CPU diagnostic timing",
    )
    axes[2].grid(alpha=0.18, axis="y")
    axes[2].legend(fontsize=8)
    for index, total in enumerate(bottom):
        axes[2].text(index, total, f"{total:.1f}s", ha="center", va="bottom")

    fig.suptitle(
        f"Real {DATE}: adapted vs fixed, residual-Lanczos + SLQ", fontsize=15
    )
    output = args.output_root / "2024-07-03_residual_lanczos_slq_frequency_failure.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_convergence(curves: pd.DataFrame, args: argparse.Namespace) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.3), constrained_layout=True)
    for ax, method in zip(axes, METHODS):
        shade_thirds(ax)
        ax.plot([0, 1], [0, 1], color="0.25", linestyle="--", linewidth=1.0)
        subset = curves[curves["method"].eq(method)]
        for steps, part in subset.groupby("residual_lanczos_steps"):
            part = part.sort_values("target_mode_fraction")
            ax.plot(
                part["estimated_mode_fraction"], part["cumulative_energy_per_n"],
                linewidth=1.6, label=f"m={int(steps)}",
            )
        ax.set(
            xlim=(0, 1), xlabel="SLQ mode fraction", ylabel="cumulative energy / n",
            title=LABELS[method],
        )
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle("Residual-Lanczos step convergence")
    output = args.output_root / "2024-07-03_residual_lanczos_step_convergence.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    args = build_parser().parse_args()
    if args.residual_lanczos_steps < 128:
        raise ValueError("Use at least 128 residual Lanczos steps")
    if args.slq_probes < 2 or args.slq_steps < 32:
        raise ValueError("Use at least 2 SLQ probes and 32 SLQ steps")
    if args.frequency_bins < 3 or args.frequency_bins % 3:
        raise ValueError("--frequency-bins must be a positive multiple of three")
    args.output_root.mkdir(parents=True, exist_ok=True)
    workflow_started = time.perf_counter()
    all_slq: list[pd.DataFrame] = []
    all_curves: list[pd.DataFrame] = []
    all_bands: list[pd.DataFrame] = []
    all_thirds: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []

    for method in METHODS:
        method_started = time.perf_counter()
        print(f"\n{method}: load cached 2024-07-03 precision", flush=True)
        precision, load_seconds = load_precision(args.operator_cache_root, method)
        print(
            f"  n={precision.n:,}, nnz={precision.whitener.nnz:,}, "
            f"cache load={load_seconds:.2f}s",
            flush=True,
        )
        slq, curves, bands, results = evaluate_method(method, precision, args)
        summary = results["summary"]
        summary["precision_cache_load_seconds"] = load_seconds
        summary["method_total_seconds"] = time.perf_counter() - method_started
        summaries.append(summary)
        all_slq.append(slq.assign(date=DATE, method=method))
        all_curves.append(curves)
        all_bands.append(bands)
        all_thirds.append(results["thirds"])
        timing_rows.append(
            {
                "method": method,
                "precision_cache_load_seconds": load_seconds,
                "slq_seconds": summary["slq_seconds"],
                "residual_lanczos_seconds": summary["residual_lanczos_seconds"],
                "method_total_seconds": summary["method_total_seconds"],
            }
        )
        print(
            f"  {method} thirds: "
            + ", ".join(
                f"{row.frequency_third}={row.energy_per_mode:.3f}"
                for row in results["thirds"].itertuples(index=False)
            ),
            flush=True,
        )

    slq_frame = pd.concat(all_slq, ignore_index=True)
    curve_frame = pd.concat(all_curves, ignore_index=True)
    band_frame = pd.concat(all_bands, ignore_index=True)
    third_frame = pd.concat(all_thirds, ignore_index=True)
    timing_frame = pd.DataFrame(timing_rows)
    total_seconds = time.perf_counter() - workflow_started
    atomic_csv(args.output_root / "slq_spectrum.csv", slq_frame)
    atomic_csv(args.output_root / "residual_lanczos_cumulative_curves.csv", curve_frame)
    atomic_csv(args.output_root / "frequency_band_energy_ratios.csv", band_frame)
    atomic_csv(args.output_root / "frequency_third_energy_ratios.csv", third_frame)
    atomic_csv(args.output_root / "timings.csv", timing_frame)
    write_json(args.output_root / "method_summaries.json", summaries)
    main_plot = plot_results(curve_frame, band_frame, timing_frame, args)
    convergence_plot = plot_convergence(curve_frame, args)
    write_json(
        args.output_root / "RUN_COMPLETE.json",
        {
            "completed": datetime.now().isoformat(timespec="seconds"),
            "date": DATE,
            "device": "local CPU",
            "methods": list(METHODS),
            "total_wall_seconds": total_seconds,
            "residual_lanczos_steps": args.residual_lanczos_steps,
            "slq_probes": args.slq_probes,
            "slq_steps": args.slq_steps,
            "frequency_bins": args.frequency_bins,
            "main_plot": str(main_plot),
            "convergence_plot": str(convergence_plot),
            "individual_eigenvectors_computed": False,
            "slq_probe_pairing": "identical Rademacher probes for adapted and fixed",
        },
    )
    print(f"\nComplete in {total_seconds:.2f}s: {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
