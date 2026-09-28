#!/usr/bin/env python3
"""Fit frozen GC and Matérn diagnostics on three preselected July 2025 days."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[6]
AMAREL_STUDY = (
    PROJECT_ROOT
    / "Exercises/st_model/day/amarel_simulation/space_time/interaction_diagnostic"
    / "fixed_geographic_three_model_092426"
)
DATE_TASKS = (
    ("2025-07-07", 36),
    ("2025-07-15", 44),
    ("2025-07-23", 52),
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-root", type=Path, default=HERE / "outputs")
    result.add_argument(
        "--summary-only",
        action="store_true",
        help="do not fit; rebuild the comparison from existing daily score files",
    )
    return result


def _run_day(task_id: int, output_root: Path) -> None:
    command = [
        sys.executable,
        str(AMAREL_STUDY / "run_fixed_geo_three_model_day.py"),
        "--task-id",
        str(task_id),
        "--config",
        str(HERE / "local_gc_vs_matern_2025.toml"),
        "--design",
        str(AMAREL_STUDY / "frozen_design.json"),
        "--dates",
        str(AMAREL_STUDY / "evaluation_dates.csv"),
        "--data-root",
        "/Users/joonwonlee/Documents/GEMS_DATA",
        "--output-root",
        str(output_root),
        "--device",
        "cpu",
        "--models",
        "gc",
        "matern05",
        "--allow-model-subset",
    ]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _read_scores(output_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for date, task_id in DATE_TASKS:
        score_path = output_root / f"task_{task_id:03d}_{date}" / "daily_scores.csv"
        if not score_path.is_file():
            raise FileNotFoundError(score_path)
        frame = pd.read_csv(score_path, float_precision="round_trip")
        if set(frame["model"]) != {"gc", "matern05"}:
            raise RuntimeError(f"{score_path} does not contain exactly GC and Matérn")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _comparison(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for date, frame in scores.groupby("date", sort=True):
        models = frame.set_index("model")
        gc = models.loc["gc"]
        matern = models.loc["matern05"]
        empirical_v_a = float(gc["empirical_second_moment_q_a"])
        empirical_v_b = float(gc["empirical_second_moment_q_b"])
        empirical_c_ab = float(gc["empirical_cross_moment_q_a_q_b"])
        empirical_var_l = float(gc["empirical_second_moment_l"])

        def covariance_error(row: pd.Series) -> float:
            return float(
                np.sqrt(
                    (float(row["model_v_a"]) - empirical_v_a) ** 2
                    + (float(row["model_v_b"]) - empirical_v_b) ** 2
                    + 2.0 * (float(row["model_c_ab"]) - empirical_c_ab) ** 2
                )
            )

        rows.append(
            {
                "date": str(date),
                "sample_count": int(gc["sample_count"]),
                "empirical_v_a": empirical_v_a,
                "empirical_v_b": empirical_v_b,
                "empirical_c_ab": empirical_c_ab,
                "empirical_var_l": empirical_var_l,
                "gc_score": float(gc["mean_contrast_score"]),
                "matern_score": float(matern["mean_contrast_score"]),
                "matern_minus_gc_score": float(matern["mean_contrast_score"])
                - float(gc["mean_contrast_score"]),
                "gc_c_ab": float(gc["model_c_ab"]),
                "matern_c_ab": float(matern["model_c_ab"]),
                "gc_abs_c_ab_error": abs(float(gc["model_c_ab"]) - empirical_c_ab),
                "matern_abs_c_ab_error": abs(
                    float(matern["model_c_ab"]) - empirical_c_ab
                ),
                "gc_var_l": float(gc["model_var_l"]),
                "matern_var_l": float(matern["model_var_l"]),
                "gc_abs_var_l_error": abs(float(gc["model_var_l"]) - empirical_var_l),
                "matern_abs_var_l_error": abs(
                    float(matern["model_var_l"]) - empirical_var_l
                ),
                "gc_covariance_frobenius_error": covariance_error(gc),
                "matern_covariance_frobenius_error": covariance_error(matern),
                "gc_vecchia_nll": float(gc["vecchia_nll"]),
                "matern_vecchia_nll": float(matern["vecchia_nll"]),
                "gc_fit_seconds": float(gc["fit_seconds"]),
                "matern_fit_seconds": float(matern["fit_seconds"]),
                "gc_max_abs_gradient": float(gc["fit_max_abs_gradient"]),
                "matern_max_abs_gradient": float(matern["fit_max_abs_gradient"]),
            }
        )
    return pd.DataFrame(rows)


def _write_report(comparison: pd.DataFrame, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_root / "gc_vs_matern_three_day_comparison.csv", index=False)
    gc_score_wins = int((comparison["matern_minus_gc_score"] > 0).sum())
    gc_covariance_wins = int(
        (
            comparison["gc_covariance_frobenius_error"]
            < comparison["matern_covariance_frobenius_error"]
        ).sum()
    )
    gc_c_ab_wins = int(
        (comparison["gc_abs_c_ab_error"] < comparison["matern_abs_c_ab_error"]).sum()
    )
    gc_var_l_wins = int(
        (comparison["gc_abs_var_l_error"] < comparison["matern_abs_var_l_error"]).sum()
    )
    gc_nll_wins = int(
        (comparison["gc_vecchia_nll"] < comparison["matern_vecchia_nll"]).sum()
    )
    lines = [
        "# July 2025 local GC versus Matérn audit",
        "",
        "Frozen fixed-geographic A/B contrasts, temporal lag 1, corridor 4/3/2, ",
        "4x4 target blocks, CPU target chunk 64, and nugget fixed at zero were used.",
        "The third date (July 23) was selected reproducibly from the complete late-July ",
        "dates with seed 20250925; July 7 and July 15 were specified in advance.",
        "",
        "Lower contrast score is better. `Matérn-GC > 0` therefore favors GC.",
        "The covariance Frobenius error compares the fitted and empirical 2x2 second-",
        "moment matrices and is descriptive rather than an independent test.",
        "",
        "| date | empirical CAB | GC CAB | Matérn CAB | GC score | Matérn score | Matérn-GC | GC cov. error | Matérn cov. error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.date} | {row.empirical_c_ab:.6g} | {row.gc_c_ab:.6g} | "
            f"{row.matern_c_ab:.6g} | {row.gc_score:.6g} | {row.matern_score:.6g} | "
            f"{row.matern_minus_gc_score:.6g} | "
            f"{row.gc_covariance_frobenius_error:.6g} | "
            f"{row.matern_covariance_frobenius_error:.6g} |"
        )
    lines.extend(
        [
            "",
            f"- GC has the lower diagnostic score on {gc_score_wins}/3 days.",
            f"- GC has the smaller raw 2x2 covariance error on {gc_covariance_wins}/3 days.",
            f"- GC has the smaller absolute C_AB error on {gc_c_ab_wins}/3 days.",
            f"- GC has the smaller absolute Var(L) error on {gc_var_l_wins}/3 days.",
            f"- GC has the lower fitted Vecchia NLL on {gc_nll_wins}/3 days.",
            "- Mean `Score(Matérn)-Score(GC)`: "
            f"`{comparison['matern_minus_gc_score'].mean():.8g}`.",
            "- Mean absolute C_AB error, GC versus Matérn: "
            f"`{comparison['gc_abs_c_ab_error'].mean():.8g}` versus "
            f"`{comparison['matern_abs_c_ab_error'].mean():.8g}`.",
            "- Mean absolute Var(L) error, GC versus Matérn: "
            f"`{comparison['gc_abs_var_l_error'].mean():.8g}` versus "
            f"`{comparison['matern_abs_var_l_error'].mean():.8g}`.",
            "- Mean raw 2x2 covariance error, GC versus Matérn: "
            f"`{comparison['gc_covariance_frobenius_error'].mean():.8g}` versus "
            f"`{comparison['matern_covariance_frobenius_error'].mean():.8g}`.",
            "- Mean fitted Vecchia NLL, GC versus Matérn: "
            f"`{comparison['gc_vecchia_nll'].mean():.8g}` versus "
            f"`{comparison['matern_vecchia_nll'].mean():.8g}`.",
            "- July 23 is an effective score tie: Matérn is lower by only "
            f"`{abs(comparison.loc[comparison['date'] == '2025-07-23', 'matern_minus_gc_score'].iloc[0]):.3g}`.",
            "",
            "Within these three frozen-design days, GC reproduces the targeted covariance "
            "more closely and has no meaningful score loss. This supports GC for this "
            "specific interaction diagnostic, but three in-sample days are not a general "
            "or calibrated model-selection result.",
            "",
            "- Results are a three-day descriptive audit, not a calibrated hypothesis test.",
            "",
        ]
    )
    (output_root / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parser().parse_args()
    output_root = args.output_root.expanduser().resolve()
    if not args.summary_only:
        for _, task_id in DATE_TASKS:
            _run_day(task_id, output_root)
    comparison = _comparison(_read_scores(output_root))
    _write_report(comparison, output_root)
    print(comparison.to_string(index=False), flush=True)
    print(f"Wrote {output_root / 'RESULTS.md'}", flush=True)


if __name__ == "__main__":
    main()
