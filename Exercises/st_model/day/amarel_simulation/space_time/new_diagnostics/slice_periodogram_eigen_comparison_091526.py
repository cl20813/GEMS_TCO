#!/usr/bin/env python3
"""Compare directional empirical periodograms with the matched eigenanalysis.

This diagnostic uses exactly the same residual profiles as
``slice_band_eigen_anisotropy_091526.py``: 5-degree east-west and north-south
profiles are linearly resampled to 80 equally spaced positions.  For each
day, band, and direction it then

1. removes the across-profile mean at every resampled position;
2. averages the ordinary profile periodograms;
3. normalizes spectral power to sum to one;
4. compares the frequency-ordered periodogram with the ranked eigenvalues of
   the lag-pooled Toeplitz correlation matrix.

The biased pooled autocovariance used by the eigenanalysis and the averaged
periodogram are a Fourier-transform pair.  The eigenvalues are nevertheless
only approximately Fourier powers because the fitted finite matrix is
Toeplitz rather than circulant, and the plotted eigenvalues are sorted by
magnitude rather than retained in frequency order.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import slice_band_eigen_anisotropy_091526 as eigen


PROFILE_EXTENT_DEGREES = 5.0
DEFAULT_OUTPUT = eigen.DEFAULT_OUTPUT


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--simulation", type=Path, default=eigen.DEFAULT_SIM_PATH)
    p.add_argument("--truth", type=Path, default=eigen.DEFAULT_TRUTH_PATH)
    p.add_argument("--real", type=Path, default=eigen.DEFAULT_REAL_PATH)
    p.add_argument("--days", nargs="+", type=int, default=[13, 19, 25])
    p.add_argument("--positions", type=int, default=80)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return p


def averaged_periodogram(
    profiles: np.ndarray,
    extent_degrees: float = PROFILE_EXTENT_DEGREES,
) -> dict[str, np.ndarray | float]:
    """Return full and one-sided normalized averaged periodograms.

    Centering matches ``eigen.eigenspectrum`` exactly: the ensemble mean at
    each resampled position is removed, rather than separately demeaning every
    individual profile.
    """

    centered = profiles - np.mean(profiles, axis=0, keepdims=True)
    n_positions = centered.shape[1]
    spacing_degrees = extent_degrees / (n_positions - 1)

    transforms = np.fft.fft(centered, axis=1)
    full_power = np.mean(np.abs(transforms) ** 2 / n_positions, axis=0)
    full_power = np.maximum(np.asarray(full_power, dtype=np.float64), 0.0)
    full_fraction = full_power / np.sum(full_power)

    one_indices = np.arange(n_positions // 2 + 1)
    one_frequency = one_indices / (n_positions * spacing_degrees)
    one_fraction = full_fraction[one_indices].copy()
    upper = n_positions // 2 if n_positions % 2 == 0 else n_positions // 2 + 1
    if upper > 1:
        one_fraction[1:upper] += full_fraction[-1 : -upper : -1]
    one_fraction /= np.sum(one_fraction)

    cumulative = np.cumsum(one_fraction)
    median_index = int(np.searchsorted(cumulative, 0.5, side="left"))
    return {
        "full_fraction": full_fraction,
        "one_frequency_cpd": one_frequency,
        "one_fraction": one_fraction,
        "spectral_centroid_cpd": float(np.sum(one_frequency * one_fraction)),
        "median_frequency_cpd": float(one_frequency[median_index]),
        "low_frequency_fraction_le_1_cpd": float(
            np.sum(one_fraction[one_frequency <= 1.0 + 1e-12])
        ),
    }


def analyze_dataset(
    dataset: str,
    residuals: dict[int, np.ndarray],
    lats: np.ndarray,
    lons: np.ndarray,
    days: list[int],
    positions: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    period_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []

    directions = [
        ("east_west", "latitude_band", eigen.LAT_BANDS),
        ("north_south", "longitude_band", eigen.LON_BANDS_PRIMARY),
    ]
    for day in days:
        cube = residuals[day]
        for direction, band_type, bands in directions:
            for band_index, band in enumerate(bands, start=1):
                if direction == "east_west":
                    profiles = eigen.ew_profiles_for_band(
                        cube, lats, lons, band, positions
                    )
                else:
                    profiles = eigen.ns_profiles_for_band(
                        cube, lats, lons, band, positions
                    )

                eigenvalues, eigen_metrics = eigen.eigenspectrum(profiles)
                eigen_fraction = eigenvalues / np.sum(eigenvalues)
                period = averaged_periodogram(profiles)
                full_fraction = np.asarray(period["full_fraction"], dtype=float)
                ranked_period = np.sort(full_fraction)[::-1]
                common = {
                    "dataset": dataset,
                    "day": int(day),
                    "direction": direction,
                    "band_type": band_type,
                    "band_index": int(band_index),
                    "band_low": float(band[0]),
                    "band_high": float(band[1]),
                    "n_profiles": int(len(profiles)),
                    "n_positions": int(positions),
                }

                frequencies = np.asarray(period["one_frequency_cpd"], dtype=float)
                one_fraction = np.asarray(period["one_fraction"], dtype=float)
                for frequency_index, (frequency, fraction) in enumerate(
                    zip(frequencies, one_fraction)
                ):
                    period_rows.append(
                        {
                            **common,
                            "frequency_index": int(frequency_index),
                            "frequency_cycles_per_degree": float(frequency),
                            "wavelength_degrees": (
                                float("inf") if frequency == 0.0 else float(1.0 / frequency)
                            ),
                            "one_sided_power_fraction": float(fraction),
                            "cumulative_low_frequency_fraction": float(
                                np.sum(one_fraction[: frequency_index + 1])
                            ),
                        }
                    )

                for rank, (eigen_value, period_value) in enumerate(
                    zip(eigen_fraction, ranked_period), start=1
                ):
                    rank_rows.append(
                        {
                            **common,
                            "rank": int(rank),
                            "eigenvalue_fraction": float(eigen_value),
                            "ranked_periodogram_fraction": float(period_value),
                        }
                    )

                metric_rows.append(
                    {
                        **common,
                        "spectral_centroid_cycles_per_degree": float(
                            period["spectral_centroid_cpd"]
                        ),
                        "median_frequency_cycles_per_degree": float(
                            period["median_frequency_cpd"]
                        ),
                        "low_frequency_fraction_le_1_cpd": float(
                            period["low_frequency_fraction_le_1_cpd"]
                        ),
                        "ranked_total_variation_distance": float(
                            0.5 * np.sum(np.abs(eigen_fraction - ranked_period))
                        ),
                        "eigen_effective_rank": float(
                            eigen_metrics["effective_rank_entropy"]
                        ),
                        "eigen_lambda1_fraction": float(
                            eigen_metrics["lambda1_fraction"]
                        ),
                    }
                )

    return (
        pd.DataFrame(period_rows),
        pd.DataFrame(rank_rows),
        pd.DataFrame(metric_rows),
    )


def aggregate_periodograms(periodograms: pd.DataFrame) -> pd.DataFrame:
    return (
        periodograms.groupby(
            ["dataset", "direction", "frequency_index", "frequency_cycles_per_degree"],
            as_index=False,
        )
        .agg(
            power_fraction_mean=("one_sided_power_fraction", "mean"),
            power_fraction_min=("one_sided_power_fraction", "min"),
            power_fraction_max=("one_sided_power_fraction", "max"),
        )
        .sort_values(["dataset", "direction", "frequency_index"])
    )


def aggregate_ranks(ranks: pd.DataFrame) -> pd.DataFrame:
    return (
        ranks.groupby(["dataset", "direction", "rank"], as_index=False)
        .agg(
            eigenvalue_fraction_mean=("eigenvalue_fraction", "mean"),
            eigenvalue_fraction_min=("eigenvalue_fraction", "min"),
            eigenvalue_fraction_max=("eigenvalue_fraction", "max"),
            periodogram_fraction_mean=("ranked_periodogram_fraction", "mean"),
            periodogram_fraction_min=("ranked_periodogram_fraction", "min"),
            periodogram_fraction_max=("ranked_periodogram_fraction", "max"),
        )
        .sort_values(["dataset", "direction", "rank"])
    )


def plot_comparison(
    periodograms: pd.DataFrame,
    ranks: pd.DataFrame,
    output: Path,
) -> None:
    colors = eigen.COLORS
    labels = {
        "east_west": "E–W: longitude frequency",
        "north_south": "N–S: latitude frequency",
    }
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), constrained_layout=True)

    for row, dataset in enumerate(("simulation", "real")):
        frequency_axis = axes[row, 0]
        rank_axis = axes[row, 1]
        for direction in ("east_west", "north_south"):
            period_part = periodograms[
                periodograms["dataset"].eq(dataset)
                & periodograms["direction"].eq(direction)
            ].sort_values("frequency_index")
            frequency = period_part["frequency_cycles_per_degree"].to_numpy(float)
            color = colors[direction]
            frequency_axis.fill_between(
                frequency,
                period_part["power_fraction_min"].to_numpy(float),
                period_part["power_fraction_max"].to_numpy(float),
                color=color,
                alpha=0.12,
            )
            frequency_axis.plot(
                frequency,
                period_part["power_fraction_mean"].to_numpy(float),
                color=color,
                linewidth=2.1,
                label=labels[direction],
            )

            rank_part = ranks[
                ranks["dataset"].eq(dataset) & ranks["direction"].eq(direction)
            ].sort_values("rank")
            rank = rank_part["rank"].to_numpy(float)
            rank_axis.plot(
                rank,
                rank_part["eigenvalue_fraction_mean"].to_numpy(float),
                color=color,
                linewidth=2.2,
                linestyle="-",
                label=f"{labels[direction]} · eigen",
            )
            rank_axis.plot(
                rank,
                rank_part["periodogram_fraction_mean"].to_numpy(float),
                color=color,
                linewidth=1.8,
                linestyle="--",
                label=f"{labels[direction]} · periodogram",
            )

        frequency_axis.set_yscale("log")
        frequency_axis.set(
            xlabel="spatial frequency (cycles per degree)",
            ylabel="one-sided normalized power per bin",
            title=f"{dataset.capitalize()}: frequency-ordered periodogram",
        )
        frequency_axis.grid(alpha=0.2)
        frequency_axis.legend(fontsize=8.5)

        rank_axis.set_yscale("log")
        rank_axis.set(
            xlabel="power rank (1 = largest)",
            ylabel="fraction of total variance",
            title=f"{dataset.capitalize()}: ranked periodogram vs eigenvalues",
        )
        rank_axis.grid(alpha=0.2)
        rank_axis.legend(fontsize=7.8, ncol=2)

    fig.suptitle(
        "Directional periodograms and Toeplitz eigenanalysis\n"
        "same 5° profiles, same 80 interpolated positions; mean across 3 days × 5 bands",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_report(output: Path, metrics: pd.DataFrame) -> None:
    summary = (
        metrics.groupby(["dataset", "direction"], as_index=False)
        .agg(
            low_frequency_fraction=("low_frequency_fraction_le_1_cpd", "mean"),
            low_frequency_min=("low_frequency_fraction_le_1_cpd", "min"),
            low_frequency_max=("low_frequency_fraction_le_1_cpd", "max"),
            spectral_centroid=("spectral_centroid_cycles_per_degree", "mean"),
            median_frequency=("median_frequency_cycles_per_degree", "mean"),
            ranked_tv_distance=("ranked_total_variation_distance", "mean"),
            eigen_effective_rank=("eigen_effective_rank", "mean"),
        )
        .sort_values(["dataset", "direction"])
    )

    lines = [
        "# 방향별 periodogram과 eigenanalysis 비교",
        "",
        "- 두 진단 모두 같은 5도 profile을 80개 등간격 위치로 선형보간한 자료를 사용한다.",
        "- Eigenanalysis와 동일하게 각 위치에서 profile 평균을 제거한 뒤 periodogram을 평균했다.",
        "- 각 periodogram과 eigenspectrum은 합이 1이 되도록 정규화했다.",
        "- Low-frequency fraction은 1 cycle/degree 이하를 뜻한다. 이는 적도 부근에서 파장 1도, 약 111 km 이상에 해당한다.",
        "- Ranked periodogram은 eigenvalue concentration과 비교하는 용도일 뿐이고, frequency 해석은 frequency-ordered plot을 사용해야 한다.",
        "",
        "| 자료 | 방향 | low-frequency fraction | spectral centroid (cycles/degree) | median frequency (cycles/degree) | ranked TV distance | eigen effective rank |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.direction} | {row.low_frequency_fraction:.4f} "
            f"({row.low_frequency_min:.4f}–{row.low_frequency_max:.4f}) | "
            f"{row.spectral_centroid:.4f} | {row.median_frequency:.4f} | "
            f"{row.ranked_tv_distance:.4f} | {row.eigen_effective_rank:.3f} |"
        )

    lines += ["", "## 방향 차이", ""]
    for dataset, group in summary.groupby("dataset", sort=False):
        indexed = group.set_index("direction")
        ew = indexed.loc["east_west"]
        ns = indexed.loc["north_south"]
        lines.append(
            f"- {dataset}: E-W/N-S low-frequency fraction ratio="
            f"{ew['low_frequency_fraction']/ns['low_frequency_fraction']:.3f}; "
            f"E-W/N-S spectral-centroid ratio="
            f"{ew['spectral_centroid']/ns['spectral_centroid']:.3f}."
        )

    lines += [
        "",
        "## 해석",
        "",
        "- Lag-pooled autocovariance와 averaged periodogram은 Fourier transform 관계이므로 ranked periodogram과 eigenvalue 곡선이 비슷한 것이 정상이다.",
        "- 유한 Toeplitz 행렬의 eigenvalue가 Fourier power와 정확히 같을 필요는 없다. 정확한 일치는 circulant covariance matrix에서 성립한다.",
        "- Low-frequency fraction이 크고 spectral centroid가 작을수록 해당 방향의 구조가 더 부드럽고 장거리 scale에 집중되어 있다는 뜻이다.",
        "- 이 same-time power 진단은 축별 anisotropy를 보여주지만 동쪽 또는 서쪽 이동의 부호는 식별하지 못한다.",
    ]
    (output / "PERIODOGRAM_COMPARISON_KO.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    eigen.atomic_csv(output / "directional_periodogram_summary.csv", summary)


def main() -> None:
    args = parser().parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    truth = json.loads(args.truth.read_text(encoding="utf-8"))

    all_periodograms: list[pd.DataFrame] = []
    all_ranks: list[pd.DataFrame] = []
    all_metrics: list[pd.DataFrame] = []
    for dataset, path, kind in (
        ("simulation", args.simulation, "simulation"),
        ("real", args.real, "real"),
    ):
        residuals, lats, lons, _ = eigen.reconstruct_residuals(
            path,
            list(args.days),
            kind,
            truth if kind == "simulation" else None,
        )
        periodograms, ranks, metrics = analyze_dataset(
            dataset,
            residuals,
            lats,
            lons,
            list(args.days),
            int(args.positions),
        )
        all_periodograms.append(periodograms)
        all_ranks.append(ranks)
        all_metrics.append(metrics)
        del residuals

    periodograms = pd.concat(all_periodograms, ignore_index=True)
    ranks = pd.concat(all_ranks, ignore_index=True)
    metrics = pd.concat(all_metrics, ignore_index=True)
    aggregate_period = aggregate_periodograms(periodograms)
    aggregate_rank = aggregate_ranks(ranks)

    eigen.atomic_csv(args.output / "slice_directional_periodograms.csv", periodograms)
    eigen.atomic_csv(args.output / "ranked_periodogram_eigen_comparison.csv", ranks)
    eigen.atomic_csv(args.output / "directional_periodogram_metrics.csv", metrics)
    eigen.atomic_csv(
        args.output / "aggregate_directional_periodograms.csv", aggregate_period
    )
    eigen.atomic_csv(
        args.output / "aggregate_ranked_periodogram_eigen.csv", aggregate_rank
    )
    plot_comparison(
        aggregate_period,
        aggregate_rank,
        args.output / "directional_periodogram_eigen_comparison.png",
    )
    write_report(args.output, metrics)
    print((args.output / "directional_periodogram_summary.csv").read_text())
    print(f"Wrote periodogram comparison to {args.output}")


if __name__ == "__main__":
    main()
