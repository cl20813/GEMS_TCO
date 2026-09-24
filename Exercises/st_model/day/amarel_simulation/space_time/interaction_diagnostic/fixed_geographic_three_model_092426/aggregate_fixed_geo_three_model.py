#!/usr/bin/env python3
"""Aggregate completed date-level tasks without treating contrasts as iid."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fixed_geo_three_model_core import (
    MODEL_ORDER,
    atomic_csv,
    atomic_json,
    atomic_text,
    load_json,
    load_toml,
    study_signature,
)


HERE = Path(__file__).resolve().parent
REQUIRED_COMPLETE_DAYS = 60
FINAL_ONLY_ARTIFACTS = (
    "day_block_bootstrap_intervals.csv",
    "daily_paired_contrast_scores.png",
    "daily_paired_contrast_scores.pdf",
    "RESULTS.md",
    "FINAL_COMPLETE",
)
DESCRIPTIVE_ARTIFACTS = (
    "all_daily_model_scores.csv",
    "daily_pairwise_score_differences.csv",
    "daily_moment_diagnostics.csv",
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, default=HERE / "fixed_geo_three_model.toml")
    result.add_argument("--design", type=Path, default=HERE / "frozen_design.json")
    result.add_argument("--dates", type=Path, default=HERE / "evaluation_dates.csv")
    result.add_argument("--output-root", type=Path)
    return result


def _circular_blocks(values: np.ndarray, block_length: int, rng: np.random.Generator) -> np.ndarray:
    n = len(values)
    if n == 0:
        return values
    block_count = math.ceil(n / block_length)
    starts = rng.integers(0, n, size=block_count)
    sampled = [values[(start + np.arange(block_length)) % n] for start in starts]
    return np.concatenate(sampled)[:n]


def _block_bootstrap(
    paired: pd.DataFrame,
    columns: list[str],
    block_length: int,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    statistics: dict[tuple[str, str], np.ndarray] = {}
    for column in columns:
        statistics[(column, "mean")] = np.empty(replicates)
        statistics[(column, "median")] = np.empty(replicates)
        statistics[(column, "positive_fraction")] = np.empty(replicates)
    grouped: list[pd.DataFrame] = []
    for _, year_group in paired.groupby("year", sort=True):
        year_group = year_group.sort_values("date").reset_index(drop=True)
        dates = pd.to_datetime(year_group["date"])
        segment = dates.diff().dt.days.ne(1).cumsum()
        grouped.extend(
            contiguous.reset_index(drop=True)
            for _, contiguous in year_group.groupby(segment, sort=True)
        )
    for replicate in range(replicates):
        resampled_parts = []
        for group in grouped:
            indices = _circular_blocks(
                np.arange(len(group)), block_length=block_length, rng=rng
            ).astype(np.int64)
            resampled_parts.append(group.iloc[indices])
        resampled = pd.concat(resampled_parts, ignore_index=True)
        for column in columns:
            values = resampled[column].to_numpy(dtype=np.float64)
            statistics[(column, "mean")][replicate] = np.mean(values)
            statistics[(column, "median")][replicate] = np.median(values)
            statistics[(column, "positive_fraction")][replicate] = np.mean(values > 0)
    rows: list[dict[str, Any]] = []
    for (column, statistic), values in statistics.items():
        observed_values = paired[column].to_numpy(dtype=np.float64)
        if statistic == "mean":
            observed = np.mean(observed_values)
        elif statistic == "median":
            observed = np.median(observed_values)
        else:
            observed = np.mean(observed_values > 0)
        rows.append(
            {
                "contrast": column,
                "statistic": statistic,
                "observed": observed,
                "bootstrap_ci_lower_2p5": np.quantile(values, 0.025),
                "bootstrap_ci_upper_97p5": np.quantile(values, 0.975),
                "block_length_days_within_contiguous_year_segment": block_length,
                "bootstrap_replicates": replicates,
                "seed": seed,
            }
        )
    return pd.DataFrame(rows)


def _paired_scores(all_scores: pd.DataFrame) -> pd.DataFrame:
    score = all_scores.pivot(
        index=["task_id", "date", "year"], columns="model", values="mean_contrast_score"
    )
    missing = set(MODEL_ORDER).difference(score.columns)
    if missing:
        raise ValueError(f"aggregate is missing model score columns {sorted(missing)}")
    score = score.reset_index()
    score["delta_score_jm_minus_gc"] = score["matern05"] - score["gc"]
    score["delta_score_sep_minus_jm"] = score["separable"] - score["matern05"]
    score["delta_score_sep_minus_gc"] = score["separable"] - score["gc"]
    return score


def _daily_moment_diagnostics(all_scores: pd.DataFrame) -> pd.DataFrame:
    identifiers = ["task_id", "date", "year"]
    shared_columns = [
        "sample_count",
        "potential_sample_count",
        "complete_contrast_fraction",
        "wx_max_abs",
        "mean_rule",
        "empirical_mean_q_a",
        "empirical_mean_q_b",
        "empirical_second_moment_q_a",
        "empirical_second_moment_q_b",
        "empirical_cross_moment_q_a_q_b",
        "empirical_second_moment_rho_ab",
        "empirical_centered_variance_q_a",
        "empirical_centered_variance_q_b",
        "empirical_centered_covariance_q_a_q_b",
        "empirical_centered_rho_ab",
        "empirical_second_moment_l_diagonal_a",
        "empirical_second_moment_l_diagonal_b",
        "empirical_second_moment_l_cross",
        "empirical_second_moment_l",
    ]
    model_columns = [
        "model_v_a",
        "model_v_b",
        "model_c_ab",
        "model_rho_ab_pooled",
        "model_mean_pointwise_rho_ab",
        "model_var_l_diagonal_a",
        "model_var_l_diagonal_b",
        "model_var_l_cross",
        "model_var_l",
    ]
    missing = set(identifiers + shared_columns + model_columns).difference(all_scores.columns)
    if missing:
        raise ValueError(f"daily score summaries are missing columns {sorted(missing)}")

    shared = all_scores.loc[all_scores["model"] == MODEL_ORDER[0], identifiers + shared_columns]
    if len(shared) * len(MODEL_ORDER) != len(all_scores):
        raise ValueError("every complete day must contribute exactly one row per model")
    wide = all_scores.pivot(index=identifiers, columns="model", values=model_columns)
    wide.columns = [f"{model}_{metric}" for metric, model in wide.columns]
    result = shared.merge(wide.reset_index(), on=identifiers, how="left", validate="one_to_one")

    for model in MODEL_ORDER:
        result[f"{model}_minus_empirical_q_a_second_moment"] = (
            result[f"{model}_model_v_a"] - result["empirical_second_moment_q_a"]
        )
        result[f"{model}_minus_empirical_q_b_second_moment"] = (
            result[f"{model}_model_v_b"] - result["empirical_second_moment_q_b"]
        )
        result[f"{model}_minus_empirical_q_a_q_b_cross_moment"] = (
            result[f"{model}_model_c_ab"] - result["empirical_cross_moment_q_a_q_b"]
        )
        result[f"{model}_minus_empirical_l_second_moment"] = (
            result[f"{model}_model_var_l"] - result["empirical_second_moment_l"]
        )
    return result.sort_values("date").reset_index(drop=True)


def _plot(paired: pd.DataFrame, output_root: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(11.5, 7.5), sharex=True, constrained_layout=True)
    x = np.arange(len(paired))
    for axis, column, title in (
        (
            axes[0],
            "delta_score_jm_minus_gc",
            "Score(Joint Matérn 0.5) - Score(GC 0.75, 1); positive favors GC",
        ),
        (
            axes[1],
            "delta_score_sep_minus_jm",
            "Score(advected separable) - Score(Joint Matérn 0.5); positive favors Joint Matérn",
        ),
    ):
        values = paired[column].to_numpy(dtype=np.float64)
        axis.axhline(0.0, color="0.2", linewidth=0.8)
        axis.plot(x, values, marker="o", markersize=3.2, linewidth=0.9)
        axis.fill_between(x, 0.0, values, where=values >= 0, alpha=0.18, color="tab:blue")
        axis.fill_between(x, 0.0, values, where=values < 0, alpha=0.18, color="tab:orange")
        axis.set_ylabel("paired daily score difference")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    axes[1].set_xticks(x[::5], paired["date"].astype(str).iloc[::5], rotation=45, ha="right")
    figure.savefig(output_root / "daily_paired_contrast_scores.png", dpi=220)
    figure.savefig(output_root / "daily_paired_contrast_scores.pdf")
    plt.close(figure)


def _remove_artifacts(output_root: Path, names: tuple[str, ...]) -> None:
    for name in names:
        (output_root / name).unlink(missing_ok=True)


def _descriptive_line(paired: pd.DataFrame, column: str, label: str) -> str:
    values = paired[column].to_numpy(dtype=np.float64)
    return (
        f"- {label}: median `{np.median(values):.8g}`, IQR "
        f"`[{np.quantile(values, 0.25):.8g}, {np.quantile(values, 0.75):.8g}]`, "
        f"positive-day fraction `{np.mean(values > 0):.3f}`."
    )


def _manifest(
    signature: str,
    configured_days: int,
    complete_days: int,
    complete: bool,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "study_signature": signature,
        "expected_days": configured_days,
        "required_complete_days": REQUIRED_COMPLETE_DAYS,
        "complete_days": complete_days,
        "complete": complete,
        "complete_all": complete,
        "models": list(MODEL_ORDER),
        "bootstrap": config["bootstrap"],
    }


def main() -> None:
    args = parser().parse_args()
    config_path = args.config.expanduser().resolve()
    design_path = args.design.expanduser().resolve()
    dates_path = args.dates.expanduser().resolve()
    config = load_toml(config_path)
    dates = pd.read_csv(dates_path)
    output_root = (
        args.output_root.expanduser()
        if args.output_root is not None
        else Path(config["paths"]["amarel_output_root"])
    )
    output_root.mkdir(parents=True, exist_ok=True)
    signature = study_signature(config_path, design_path, dates_path)

    completeness_rows: list[dict[str, Any]] = []
    score_frames: list[pd.DataFrame] = []
    for row in dates.itertuples(index=False):
        task_dir = output_root / f"task_{int(row.task_id):03d}_{row.date}"
        task_complete = (task_dir / "COMPLETE").is_file()
        daily_scores = task_dir / "daily_scores.csv"
        signature_ok = False
        manifest_path = task_dir / "task_manifest.json"
        if manifest_path.is_file():
            signature_ok = load_json(manifest_path).get("study_signature") == signature
        model_complete = {
            model: (task_dir / f"model_{model}" / "COMPLETE").is_file() for model in MODEL_ORDER
        }
        complete = (
            task_complete
            and signature_ok
            and all(model_complete.values())
            and daily_scores.is_file()
        )
        completeness_rows.append(
            {
                "task_id": int(row.task_id),
                "date": str(row.date),
                "task_directory_exists": task_dir.is_dir(),
                "signature_ok": signature_ok,
                **{f"complete_{model}": status for model, status in model_complete.items()},
                "task_complete": task_complete,
                "included_in_aggregate": complete,
            }
        )
        if complete:
            frame = pd.read_csv(daily_scores, float_precision="round_trip")
            if set(frame["model"]) != set(MODEL_ORDER):
                raise ValueError(f"{daily_scores} does not contain exactly the three models")
            score_frames.append(frame)

    completeness = pd.DataFrame(completeness_rows)
    atomic_csv(output_root / "completeness.csv", completeness)
    missing = completeness.loc[~completeness["included_in_aggregate"], "task_id"].astype(int)
    atomic_text(
        output_root / "missing_task_ids.txt",
        "\n".join(str(value) for value in missing) + ("\n" if len(missing) else ""),
    )

    # A final marker from an earlier invocation must never survive while this
    # invocation is validating or regenerating the aggregate.
    _remove_artifacts(output_root, FINAL_ONLY_ARTIFACTS)

    paired: pd.DataFrame | None = None
    if score_frames:
        all_scores = pd.concat(score_frames, ignore_index=True).sort_values(["date", "model"])
        paired = _paired_scores(all_scores)
        moment_diagnostics = _daily_moment_diagnostics(all_scores)
        atomic_csv(output_root / "all_daily_model_scores.csv", all_scores)
        atomic_csv(output_root / "daily_pairwise_score_differences.csv", paired)
        atomic_csv(output_root / "daily_moment_diagnostics.csv", moment_diagnostics)
    else:
        _remove_artifacts(output_root, DESCRIPTIVE_ARTIFACTS)

    complete_days = 0 if paired is None else len(paired)
    complete_all = (
        len(dates) == REQUIRED_COMPLETE_DAYS
        and dates["task_id"].nunique() == REQUIRED_COMPLETE_DAYS
        and dates["date"].astype(str).nunique() == REQUIRED_COMPLETE_DAYS
        and len(missing) == 0
        and complete_days == REQUIRED_COMPLETE_DAYS
    )
    manifest = _manifest(
        signature=signature,
        configured_days=len(dates),
        complete_days=complete_days,
        complete=False,
        config=config,
    )

    if not complete_all:
        atomic_json(output_root / "aggregate_manifest.json", manifest)
        provisional_lines = [
            "# INCOMPLETE RUN — PROVISIONAL RESULTS ONLY",
            "",
            "> **WARNING: This aggregate is incomplete and must not be treated as final. "
            "Bootstrap intervals, final plots, and final conclusions are intentionally withheld.**",
            "",
            f"- Complete design-held-out evaluation days: `{complete_days}` of the required "
            f"`{REQUIRED_COMPLETE_DAYS}`.",
            f"- Rows in the supplied evaluation-date file: `{len(dates)}`.",
            "- See `completeness.csv` and `missing_task_ids.txt` for task status.",
            "- Every available descriptive row is one day; overlapping within-day "
            "contrasts are not treated as iid replicates.",
        ]
        if paired is None:
            provisional_lines.extend(
                [
                    "",
                    "No complete design-held-out days are currently available, so no descriptive "
                    "score CSVs were written.",
                ]
            )
        else:
            provisional_lines.extend(
                [
                    "",
                    "## Available paired descriptive summaries",
                    "",
                    _descriptive_line(
                        paired,
                        "delta_score_jm_minus_gc",
                        "Score(JM)-Score(GC); positive favors GC",
                    ),
                    _descriptive_line(
                        paired,
                        "delta_score_sep_minus_jm",
                        "Score(Sep)-Score(JM); positive favors JM",
                    ),
                    _descriptive_line(
                        paired,
                        "delta_score_sep_minus_gc",
                        "Score(Sep)-Score(GC); positive favors GC",
                    ),
                ]
            )
        provisional_lines.extend(
            [
                "",
                "## Interpretation boundary",
                "",
                "These provisional descriptions compare only the three pre-specified fitted "
                "covariance specifications. They are not a general family ranking or a "
                "separability hypothesis test. Each model is fit and scored on the same day, "
                "so this is targeted in-sample covariance adequacy, not predictive holdout.",
                "",
            ]
        )
        atomic_text(output_root / "PROVISIONAL_RESULTS.md", "\n".join(provisional_lines))
        return

    assert paired is not None
    intervals = _block_bootstrap(
        paired,
        columns=[
            "delta_score_jm_minus_gc",
            "delta_score_sep_minus_jm",
            "delta_score_sep_minus_gc",
        ],
        block_length=int(config["bootstrap"]["day_block_length"]),
        replicates=int(config["bootstrap"]["replicates"]),
        seed=int(config["bootstrap"]["seed"]),
    )
    atomic_csv(output_root / "day_block_bootstrap_intervals.csv", intervals)
    _plot(paired, output_root)

    report = "\n".join(
        [
            "# Fixed-geographic three-model interaction diagnostic",
            "",
            f"- Complete design-held-out evaluation days: `{len(paired)}` of `{len(dates)}`.",
            f"- Configuration complete: `{complete_all}`.",
            "- Lower mean bivariate Gaussian contrast score is better.",
            "- Every row is one day; overlapping within-day contrasts are not treated as iid replicates.",
            "",
            "## Paired descriptive summaries",
            "",
            _descriptive_line(
                paired,
                "delta_score_jm_minus_gc",
                "Score(JM)-Score(GC); positive favors GC",
            ),
            _descriptive_line(
                paired,
                "delta_score_sep_minus_jm",
                "Score(Sep)-Score(JM); positive favors JM",
            ),
            _descriptive_line(
                paired,
                "delta_score_sep_minus_gc",
                "Score(Sep)-Score(GC); positive favors GC",
            ),
            "",
            "## Uncertainty",
            "",
            "The accompanying interval table uses a circular moving-block bootstrap within contiguous calendar-day segments of each year; it does not bridge the excluded 2025-07-24 date. It resamples days, never individual overlapping contrasts.",
            "",
            "## Interpretation boundary",
            "",
            "The comparison is among the three pre-specified, equal-dimensional fitted covariance specifications. Each model is fit and scored on the same day, so this is a targeted in-sample covariance-adequacy comparison, not an out-of-sample predictive score. It is not a general ranking of all generalized-Cauchy, Matérn, or separable families, and it is not a separability hypothesis test.",
            "",
        ]
    )
    atomic_text(output_root / "RESULTS.md", report)
    manifest["complete"] = True
    manifest["complete_all"] = True
    atomic_json(
        output_root / "aggregate_manifest.json",
        manifest,
    )
    (output_root / "PROVISIONAL_RESULTS.md").unlink(missing_ok=True)
    atomic_text(output_root / "FINAL_COMPLETE", "complete\n")


if __name__ == "__main__":
    main()
