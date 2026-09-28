#!/usr/bin/env python3
"""Aggregate day-level fitted-versus-empirical cross-term diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tcross_fit643_matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "fitted_cross_643_config.json"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    result.add_argument("--output-root", type=Path)
    result.add_argument("--require-complete", action="store_true")
    return result


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False, float_format="%.17g")
    temporary.replace(path)


def mean_summary(values: pd.Series) -> dict[str, float]:
    x = values.to_numpy(dtype=float)
    mean = float(np.mean(x))
    if len(x) == 1:
        return {"mean": mean, "sd": float("nan"), "se": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan")}
    sd = float(np.std(x, ddof=1))
    se = sd / math.sqrt(len(x))
    half = float(stats.t.ppf(0.975, len(x) - 1) * se)
    return {"mean": mean, "sd": sd, "se": se, "ci_low": mean - half, "ci_high": mean + half}


def markdown_table(frame: pd.DataFrame, digits: int = 5) -> str:
    values = frame.copy()
    for column in values.select_dtypes(include=[np.number]).columns:
        values[column] = values[column].map(lambda value: f"{float(value):.{digits}f}")
    header = "| " + " | ".join(map(str, values.columns)) + " |"
    separator = "|" + "|".join("---" for _ in values.columns) + "|"
    rows = [
        "| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |"
        for row in values.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def main() -> None:
    args = parser().parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = args.output_root or Path(config["output"]["amarel_root"])
    master_path = output / "all_daily_fitted_cross_centers.csv"
    if not master_path.is_file():
        raise FileNotFoundError(master_path)
    daily = pd.read_csv(master_path)
    duplicates = int(daily.duplicated(["scenario_id", "date"]).sum())
    expected = 180
    complete = bool(
        len(daily) == expected
        and daily["scenario_id"].nunique() == 6
        and duplicates == 0
        and daily["fit_converged"].astype(bool).all()
    )
    if args.require_complete and not complete:
        raise RuntimeError(
            f"incomplete fitted study: rows={len(daily)}, scenarios={daily.scenario_id.nunique()}, "
            f"duplicates={duplicates}, converged={daily.fit_converged.astype(bool).sum()}"
        )

    total_metric = "empirical_minus_fitted_l_variance"
    cross_metric = "empirical_minus_fitted_l_cross_term"
    rows: list[dict[str, Any]] = []
    for (scenario_id, family, eta, fitted_model), group in daily.groupby(
        ["scenario_id", "family", "interaction_eta", "fitted_model"], sort=False
    ):
        total_residual = mean_summary(group[total_metric])
        cross_residual = mean_summary(group[cross_metric])
        rows.append(
            {
                "scenario_id": scenario_id,
                "family": family,
                "eta": eta,
                "fitted_model": fitted_model,
                "days": len(group),
                "empirical_l_variance_mean": group["empirical_l_variance"].mean(),
                "fitted_l_variance_mean": group["fitted_l_variance"].mean(),
                "truth_l_variance_mean": group["truth_l_variance"].mean(),
                "var_data_minus_fit_mean": total_residual["mean"],
                "var_data_minus_fit_day_sd": total_residual["sd"],
                "var_data_minus_fit_se": total_residual["se"],
                "var_data_minus_fit_ci_low": total_residual["ci_low"],
                "var_data_minus_fit_ci_high": total_residual["ci_high"],
                "diagonal_a_data_minus_fit_mean": group[
                    "empirical_minus_fitted_l_diagonal_a"
                ].mean(),
                "diagonal_b_data_minus_fit_mean": group[
                    "empirical_minus_fitted_l_diagonal_b"
                ].mean(),
                "cross_data_minus_fit_mean": cross_residual["mean"],
                "cross_data_minus_fit_day_sd": cross_residual["sd"],
                "cross_data_minus_fit_se": cross_residual["se"],
                "cross_data_minus_fit_ci_low": cross_residual["ci_low"],
                "cross_data_minus_fit_ci_high": cross_residual["ci_high"],
                "empirical_l_cross_mean": group["empirical_l_cross_term"].mean(),
                "fitted_l_cross_mean": group["fitted_l_cross_term"].mean(),
                "truth_l_cross_mean": group["truth_l_cross_term"].mean(),
                "centered_var_data_minus_fit_mean": group[
                    "empirical_centered_minus_fitted_l_variance"
                ].mean(),
                "fitted_minus_truth_l_variance_mean": group[
                    "fitted_minus_truth_l_variance"
                ].mean(),
                "fitted_minus_truth_l_cross_mean": group[
                    "fitted_minus_truth_l_cross_term"
                ].mean(),
                "fit_nll_mean": group["fit_nll_per_target"].mean(),
                "fit_seconds_mean": group["fit_seconds"].mean(),
                "fit_advec_lat_mean": group["fit_advec_lat"].mean(),
                "fit_advec_lon_mean": group["fit_advec_lon"].mean(),
            }
        )
    scenario = pd.DataFrame(rows).sort_values(["family", "eta"]).reset_index(drop=True)
    atomic_csv(output / "scenario_fitted_cross_summary.csv", scenario)

    paired_rows: list[dict[str, Any]] = []
    for family, group in daily.groupby("family"):
        for component, metric in (("total_variance", total_metric), ("cross_term", cross_metric)):
            wide = group.pivot(index="date", columns="interaction_eta", values=metric)
            for label, values in (
                ("residual_eta0p5_minus_eta0", wide[0.5] - wide[0.0]),
                ("residual_eta1_minus_eta0", wide[1.0] - wide[0.0]),
                ("residual_linearity_curvature", wide[0.0] + wide[1.0] - 2.0 * wide[0.5]),
            ):
                summary = mean_summary(values)
                paired_rows.append(
                    {"family": family, "component": component, "effect": label,
                     "days": len(values), **summary}
                )
    paired = pd.DataFrame(paired_rows)
    atomic_csv(output / "paired_eta_fitted_residual_effects.csv", paired)

    # Directly show total Var(L), then attribute data-minus-fit to its exact
    # A-diagonal, B-diagonal, and cross components.
    family_order = ["matern", "generalized_cauchy"]
    labels = {"matern": "Matérn", "generalized_cauchy": "Generalized Cauchy"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for row, family in enumerate(family_order):
        means = scenario[scenario.family == family].set_index("eta").loc[[0.0, 0.5, 1.0]]
        x = np.arange(3)
        ax = axes[row, 0]
        ax.plot(x, means["empirical_l_variance_mean"], marker="o", label="data Var(L)")
        ax.plot(x, means["fitted_l_variance_mean"], marker="s", label="fitted Var(L)")
        ax.plot(x, means["truth_l_variance_mean"], marker="^", linestyle="--", label="truth Var(L)")
        ax.set_xticks(x, ["0", "0.5", "1"])
        ax.set_xlabel("Simulation interaction eta")
        ax.set_ylabel("Variance")
        ax.set_title(f"{labels[family]}: total Var(L)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

        ax = axes[row, 1]
        width = 0.22
        ax.bar(x - width, means["diagonal_a_data_minus_fit_mean"], width,
               label="A diagonal", color="#4c78a8")
        ax.bar(x, means["diagonal_b_data_minus_fit_mean"], width,
               label="B diagonal", color="#f58518")
        ax.bar(x + width, means["cross_data_minus_fit_mean"], width,
               label="cross", color="#54a24b")
        ax.plot(x, means["var_data_minus_fit_mean"], color="black", marker="D",
                linestyle="--", label="total data-fit")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(x, ["0", "0.5", "1"])
        ax.set_xlabel("Simulation interaction eta")
        ax.set_ylabel("Data contribution - fitted contribution")
        ax.set_title(f"{labels[family]}: exact residual decomposition")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    figure_path = output / "fitted_vs_empirical_variance_decomposition.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    displayed = scenario[[
        "family", "eta", "days", "empirical_l_variance_mean", "fitted_l_variance_mean",
        "truth_l_variance_mean", "var_data_minus_fit_mean", "var_data_minus_fit_ci_low",
        "var_data_minus_fit_ci_high", "diagonal_a_data_minus_fit_mean",
        "diagonal_b_data_minus_fit_mean", "cross_data_minus_fit_mean",
    ]]
    report = f"""# Fitted-model two-contrast variance diagnostic

- Completed scenario-days: `{len(daily)}` / `{expected}`.
- Complete: `{complete}`.
- Fitted model rule: matched-family joint covariance.
- Vecchia: direction-adapted 4x4 corridor, lag `6/4/3`; nugget fixed at zero.
- Primary daily diagnostic: empirical Var(L) minus fitted-model Var(L), where L=d1*Q_A+d2*Q_B.
- Exact attribution: A diagonal difference + B diagonal difference + cross-term difference equals the total Var(L) difference.
- Cross contribution: 2*d1*d2*Cov(Q_A,Q_B); it is not Var(Q_A*Q_B).
- Because L, Q_A, and Q_B are zero-sum contrasts, the primary empirical variance uses their known-zero second moments. Centered empirical versions are also saved as sensitivity columns.
- Inference unit: one independently simulated day. Overlapping anchors are not treated as independent replicates.

## Scenario summaries

{markdown_table(displayed)}

## Interpretation boundary

The eta=1 scenarios are matched-family correctly specified joint endpoints. The eta=0 and eta=0.5 scenarios are intentionally outside the fitted joint family. First inspect total Var(L) misfit; then use the exact three-component decomposition to determine whether that misfit is specifically driven by the cross contribution rather than either marginal contrast variance. Simulation truth is retained only as an evaluation reference; it is not the fitted-model diagnostic.
"""
    atomic_text(output / ("RESULTS.md" if complete else "PROVISIONAL_RESULTS.md"), report)
    final_path = output / "FINAL_COMPLETE.json"
    if complete:
        digest = hashlib.sha256(master_path.read_bytes()).hexdigest()
        atomic_json(
            final_path,
            {
                "schema_version": 1,
                "complete": True,
                "completed_scenario_days": len(daily),
                "expected_scenario_days": expected,
                "scenario_count": int(daily.scenario_id.nunique()),
                "all_fits_converged": bool(daily.fit_converged.astype(bool).all()),
                "master_csv_sha256": digest,
            },
        )
        (output / "PROVISIONAL_RESULTS.md").unlink(missing_ok=True)
    else:
        final_path.unlink(missing_ok=True)
    print(report)


if __name__ == "__main__":
    main()
