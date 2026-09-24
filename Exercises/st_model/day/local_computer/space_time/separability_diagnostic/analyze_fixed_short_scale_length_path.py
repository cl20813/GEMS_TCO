#!/usr/bin/env python3
"""Analyze the long-contrast path with d_minus fixed at its optimal scale.

The saved exact-comoving covariance parameters are used analytically.  This is
not a simulation, covariance refit, dictionary search, or coefficient search.
The path is

    phi = 0,
    r_a = s + r_star / 2,
    r_b = s - r_star / 2,

so d_minus = r_star and d_plus = 2s.  The numerator mechanisms and the
generalized minimum eigenvalue under the axis-symmetrized fitted null are
evaluated as one-dimensional functions of s.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq


HERE = Path(__file__).resolve().parent
DEFAULT_ORACLE_DIR = HERE / "outputs/exact_comoving_rectangle_dictionary_092226"
DEFAULT_OUTPUT_DIR = DEFAULT_ORACLE_DIR / "fixed_short_scale_length_path"


@dataclass(frozen=True)
class PathKernels:
    truth_variance: float
    truth_temporal_scale: float
    null_variance: float
    null_spatial_decay: float
    null_temporal_decay: float
    null_jitter_ratio: float

    def k_delta(self, radius: np.ndarray | float) -> np.ndarray:
        """Truth-minus-matched covariance of the one-step difference field."""

        radius = np.asarray(radius, dtype=np.float64)
        tau = self.truth_temporal_scale
        return 2.0 * self.truth_variance * (
            np.exp(-(radius + tau)) - np.exp(-np.hypot(radius, tau))
        )

    def k_null(self, radius: np.ndarray | float) -> np.ndarray:
        """Axis-symmetrized fitted-null covariance of the difference field."""

        radius = np.asarray(radius, dtype=np.float64)
        base = (
            2.0
            * self.null_variance
            * np.exp(-self.null_spatial_decay * radius)
            * (1.0 - np.exp(-self.null_temporal_decay))
        )
        diagonal_increment = 2.0 * self.null_variance * self.null_jitter_ratio
        return np.where(radius == 0.0, base + diagonal_increment, base)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-dir", type=Path, default=DEFAULT_ORACLE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--maximum-s", type=float, default=8.0)
    parser.add_argument("--grid-size", type=int, default=2401)
    return parser


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.17g")
    temporary.replace(path)


def _atomic_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(contents, encoding="utf-8")
    temporary.replace(path)


def _parameter(values: dict[str, Any], name: str) -> float:
    value = float(values[name])
    if not np.isfinite(value):
        raise ValueError(f"parameter {name} is not finite")
    return value


def _optimal_short_scale(tau: float) -> float:
    """Return the positive maximizer of |k_delta(r)| for fixed tau."""

    def derivative_of_positive_gap(radius: float) -> float:
        joint_radius = math.hypot(radius, tau)
        return math.exp(-(radius + tau)) - (
            radius / joint_radius
        ) * math.exp(-joint_radius)

    return float(brentq(derivative_of_positive_gap, 0.0, 20.0))


def _generalized_roots(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    denominator_determinant = p * q - r * r
    numerator_determinant = a * b - c * c
    linear_coefficient = a * q + b * p - 2.0 * c * r
    discriminant = (
        linear_coefficient * linear_coefficient
        - 4.0 * denominator_determinant * numerator_determinant
    )
    tolerance = (
        128.0
        * np.finfo(np.float64).eps
        * np.maximum(linear_coefficient * linear_coefficient, 1.0)
    )
    if np.any(discriminant < -tolerance):
        raise ArithmeticError("materially negative generalized-eigen discriminant")
    if np.any(denominator_determinant <= 0.0):
        raise ArithmeticError("axis-symmetrized H0 is not positive definite")
    square_root = np.sqrt(np.maximum(discriminant, 0.0))
    minimum = (linear_coefficient - square_root) / (2.0 * denominator_determinant)
    maximum = (linear_coefficient + square_root) / (2.0 * denominator_determinant)
    return (
        minimum,
        maximum,
        denominator_determinant,
        numerator_determinant,
        linear_coefficient,
    )


def _geometry_frame(
    s: np.ndarray,
    r_star: float,
    kernels: PathKernels,
) -> pd.DataFrame:
    s = np.asarray(s, dtype=np.float64)
    r_a = s + 0.5 * r_star
    r_b = s - 0.5 * r_star
    if np.any(r_b <= 0.0):
        raise ValueError("the path requires s > r_star / 2")
    d_minus = np.full_like(s, r_star)
    d_plus = 2.0 * s
    two_r_a = 2.0 * r_a
    two_r_b = 2.0 * r_b

    a = -2.0 * kernels.k_delta(two_r_a)
    b = -2.0 * kernels.k_delta(two_r_b)
    c_short = 2.0 * kernels.k_delta(d_minus)
    c_far = -2.0 * kernels.k_delta(d_plus)
    c = c_short + c_far

    p = 2.0 * (kernels.k_null(0.0) - kernels.k_null(two_r_a))
    q = 2.0 * (kernels.k_null(0.0) - kernels.k_null(two_r_b))
    r = 2.0 * (kernels.k_null(d_minus) - kernels.k_null(d_plus))
    lambda_minimum, lambda_maximum, d0, d_delta, linear = _generalized_roots(
        a, b, c, p, q, r
    )
    return pd.DataFrame(
        {
            "s": s,
            "r_a": r_a,
            "r_b": r_b,
            "phi_degrees": np.zeros_like(s),
            "d_minus": d_minus,
            "d_plus": d_plus,
            "two_r_a": two_r_a,
            "two_r_b": two_r_b,
            "A_diagonal_penalty": a,
            "B_diagonal_penalty": b,
            "C_short": c_short,
            "C_far": c_far,
            "C_total": c,
            "absolute_C_total": np.abs(c),
            "sqrt_A_times_B": np.sqrt(np.maximum(a * b, 0.0)),
            "negative_inertia_margin_C2_minus_AB": c * c - a * b,
            "D_delta_AB_minus_C2": d_delta,
            "P_null": p,
            "Q_null": q,
            "R_null": r,
            "D0_PQ_minus_R2": d0,
            "S_linear_coefficient": linear,
            "lambda_min": lambda_minimum,
            "lambda_max": lambda_maximum,
        }
    )


def _arbitrary_geometry(
    r_a: float,
    r_b: float,
    phi: float,
    kernels: PathKernels,
) -> dict[str, float]:
    cosine = math.cos(phi)
    d_minus = math.sqrt(max(0.0, r_a * r_a + r_b * r_b - 2.0 * r_a * r_b * cosine))
    d_plus = math.sqrt(max(0.0, r_a * r_a + r_b * r_b + 2.0 * r_a * r_b * cosine))
    a = float(-2.0 * kernels.k_delta(2.0 * r_a))
    b = float(-2.0 * kernels.k_delta(2.0 * r_b))
    c_short = float(2.0 * kernels.k_delta(d_minus))
    c_far = float(-2.0 * kernels.k_delta(d_plus))
    c = c_short + c_far
    p = float(2.0 * (kernels.k_null(0.0) - kernels.k_null(2.0 * r_a)))
    q = float(2.0 * (kernels.k_null(0.0) - kernels.k_null(2.0 * r_b)))
    r = float(2.0 * (kernels.k_null(d_minus) - kernels.k_null(d_plus)))
    roots = _generalized_roots(
        np.asarray([a]),
        np.asarray([b]),
        np.asarray([c]),
        np.asarray([p]),
        np.asarray([q]),
        np.asarray([r]),
    )
    return {
        "r_a": r_a,
        "r_b": r_b,
        "phi_degrees": math.degrees(phi),
        "d_minus": d_minus,
        "d_plus": d_plus,
        "A_diagonal_penalty": a,
        "B_diagonal_penalty": b,
        "C_short": c_short,
        "C_far": c_far,
        "C_total": c,
        "P_null": p,
        "Q_null": q,
        "R_null": r,
        "lambda_min": float(roots[0][0]),
        "lambda_max": float(roots[1][0]),
    }


def _asymptotic_values(r_star: float, kernels: PathKernels) -> dict[str, float]:
    a = 0.0
    b = 0.0
    c_short = float(2.0 * kernels.k_delta(r_star))
    c_far = 0.0
    p = float(2.0 * kernels.k_null(0.0))
    q = p
    r = float(2.0 * kernels.k_null(r_star))
    roots = _generalized_roots(
        np.asarray([a]),
        np.asarray([b]),
        np.asarray([c_short]),
        np.asarray([p]),
        np.asarray([q]),
        np.asarray([r]),
    )
    return {
        "s": math.inf,
        "r_a": math.inf,
        "r_b": math.inf,
        "phi_degrees": 0.0,
        "d_minus": r_star,
        "d_plus": math.inf,
        "two_r_a": math.inf,
        "two_r_b": math.inf,
        "A_diagonal_penalty": a,
        "B_diagonal_penalty": b,
        "C_short": c_short,
        "C_far": c_far,
        "C_total": c_short,
        "absolute_C_total": abs(c_short),
        "sqrt_A_times_B": 0.0,
        "negative_inertia_margin_C2_minus_AB": c_short * c_short,
        "D_delta_AB_minus_C2": -(c_short * c_short),
        "P_null": p,
        "Q_null": q,
        "R_null": r,
        "D0_PQ_minus_R2": p * q - r * r,
        "S_linear_coefficient": -2.0 * c_short * r,
        "lambda_min": float(roots[0][0]),
        "lambda_max": float(roots[1][0]),
    }


def _interpolate_row(frame: pd.DataFrame, s: float) -> pd.Series:
    values = {
        column: float(np.interp(s, frame["s"], frame[column]))
        for column in frame.columns
    }
    return pd.Series(values)


def _write_figure(
    frame: pd.DataFrame,
    r_star: float,
    threshold_s: float,
    selected_s: float,
    discrete_lambda: float,
    asymptotic_lambda: float,
    output_dir: Path,
) -> None:
    visible = frame.loc[frame["s"] <= 6.0]
    selected_path = _interpolate_row(frame, selected_s)
    figure, axes = plt.subplots(1, 3, figsize=(16.2, 5.2), constrained_layout=True)

    axes[0].plot(visible["s"], visible["A_diagonal_penalty"], label=r"$A$", linewidth=1.8)
    axes[0].plot(visible["s"], visible["B_diagonal_penalty"], label=r"$B$", linewidth=1.8)
    axes[0].plot(
        visible["s"],
        -visible["C_short"],
        label=r"$-C_{\rm short}$ (gain)",
        linewidth=2.0,
        linestyle="--",
    )
    axes[0].plot(
        visible["s"],
        visible["C_far"],
        label=r"$C_{\rm far}$ (cancellation)",
        linewidth=1.8,
        linestyle="-.",
    )
    axes[0].plot(
        visible["s"],
        visible["absolute_C_total"],
        label=r"$|C|$ (net cross)",
        linewidth=2.2,
    )
    axes[0].set_title("Numerator mechanisms")
    axes[0].set_ylabel("covariance discrepancy")
    axes[0].legend(frameon=False, fontsize=8, loc="center right")

    axes[1].plot(
        visible["s"], visible["absolute_C_total"], label=r"$|C|$", linewidth=2.2
    )
    axes[1].plot(
        visible["s"],
        visible["sqrt_A_times_B"],
        label=r"$\sqrt{AB}$",
        linewidth=2.0,
        linestyle="--",
    )
    axes[1].fill_between(
        visible["s"],
        visible["sqrt_A_times_B"],
        visible["absolute_C_total"],
        where=visible["s"] >= threshold_s,
        alpha=0.14,
        label=r"negative direction: $C^2>AB$",
    )
    axes[1].axvline(threshold_s, color="0.25", linewidth=1.0, linestyle=":")
    axes[1].annotate(
        rf"threshold $s={threshold_s:.3f}$",
        xy=(threshold_s, float(np.interp(threshold_s, visible["s"], visible["absolute_C_total"]))),
        xytext=(threshold_s + 0.35, 1.0),
        arrowprops={"arrowstyle": "->", "linewidth": 0.8},
        fontsize=8,
    )
    axes[1].set_title("Exact numerator sign condition")
    axes[1].set_ylabel("magnitude")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")

    axes[2].plot(visible["s"], visible["lambda_min"], linewidth=2.2, label="fixed-short-scale path")
    axes[2].axhline(0.0, color="0.35", linewidth=0.9)
    axes[2].axhline(
        asymptotic_lambda,
        color="0.35",
        linewidth=1.0,
        linestyle="--",
        label=rf"limit {asymptotic_lambda:.4f}",
    )
    axes[2].axvline(selected_s, color="0.55", linewidth=1.0, linestyle=":")
    axes[2].scatter(
        [selected_s],
        [selected_path["lambda_min"]],
        s=42,
        marker="o",
        zorder=4,
        label=rf"path at $s={selected_s:.3f}$",
    )
    axes[2].scatter(
        [selected_s],
        [discrete_lambda],
        s=48,
        marker="x",
        linewidth=1.8,
        zorder=4,
        label="discrete selected geometry",
    )
    axes[2].set_title(r"Standardized signal under symmetric $H_0$")
    axes[2].set_ylabel(r"$\lambda_{\min}(\Delta H,H_0)$")
    axes[2].legend(frameon=False, fontsize=8, loc="upper right")

    for axis in axes:
        axis.set_xlabel(r"mean half-length $s$")
        axis.set_xlim(float(visible["s"].min()), float(visible["s"].max()))
        axis.grid(alpha=0.2, linewidth=0.6)
        axis.axvline(r_star, color="0.65", linewidth=0.8, linestyle=(0, (2, 3)))

    figure.suptitle(
        rf"Fixed short-scale path: $d_-=r_\star={r_star:.6f}$, $\phi=0$",
        fontsize=13,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "fixed_short_scale_length_path.png", dpi=220)
    figure.savefig(output_dir / "fixed_short_scale_length_path.pdf")
    plt.close(figure)


def _report(
    oracle_dir: Path,
    r_star: float,
    k_delta_at_r_star: float,
    threshold_s: float,
    selected_s: float,
    selected_path: pd.Series,
    discrete: dict[str, float],
    saved_discrete_lambda: float,
    asymptote: dict[str, float],
    kernels: PathKernels,
    monotone_lambda: bool,
) -> str:
    path_fraction = abs(float(selected_path["lambda_min"])) / abs(asymptote["lambda_min"])
    continuous_improvement = (
        abs(float(selected_path["lambda_min"])) / abs(discrete["lambda_min"]) - 1.0
    )
    return "\n".join(
        [
            "# Fixed-short-scale long-contrast path",
            "",
            "This is an analytic population-covariance mechanism audit. It performs no simulation, covariance refit, dictionary search, or coefficient search.",
            f"Input oracle: `{oracle_dir}`.",
            "",
            "## Path",
            "",
            "The path fixes",
            "",
            "`phi=0`, `r_a=s+r_star/2`, `r_b=s-r_star/2`,",
            "",
            "so `d_minus=r_star`, `d_plus=2s`, `2r_a=2s+r_star`, and `2r_b=2s-r_star`.",
            "",
            f"For the saved one-step temporal scale, `r_star={r_star:.15g}` maximizes `|k_delta(r)|`, with `k_delta(r_star)={k_delta_at_r_star:+.15g}` and `C_short={2.0 * k_delta_at_r_star:+.15g}`.",
            "",
            "The denominator uses the fitted null after restoring the exact spatial range ratio while preserving the fitted spatial-range product. In truth-standardized moving coordinates its spatial decay is "
            f"`{kernels.null_spatial_decay:.15g}` and its one-step temporal decay is `{kernels.null_temporal_decay:.15g}`.",
            "",
            "## What the path shows",
            "",
            f"- The exact numerator threshold is `s={threshold_s:.15g}`. Below it, `C^2<AB`; above it, `C^2>AB` and one negative generalized eigenvalue exists.",
            "- `A` and the far-lag cancellation decrease along the full admissible path. `B` first rises because the shorter contrast grows from zero length, reaches its maximum at `s=r_star`, and then decreases. Thus all three unwanted terms decrease together on the long-contrast branch `s>r_star`.",
            f"- On the evaluated path, `lambda_min` is {'monotonically decreasing' if monotone_lambda else 'not monotone'} and approaches `{asymptote['lambda_min']:+.15g}` as `s` tends to infinity.",
            "",
            "At the mean half-length of the selected discrete pair,",
            "",
            f"- `s_selected=(sqrt(8)+sqrt(5))/2={selected_s:.15g}`;",
            f"- path geometry: `(r_a,r_b,phi,d_minus,d_plus)=({selected_path['r_a']:.12g}, {selected_path['r_b']:.12g}, 0, {selected_path['d_minus']:.12g}, {selected_path['d_plus']:.12g})`;",
            f"- numerator components: `(A,B,C_short,C_far,C)=({selected_path['A_diagonal_penalty']:.12g}, {selected_path['B_diagonal_penalty']:.12g}, {selected_path['C_short']:.12g}, {selected_path['C_far']:.12g}, {selected_path['C_total']:.12g})`;",
            f"- path `lambda_min={selected_path['lambda_min']:+.15g}`, which is `{path_fraction:.3%}` of the limiting magnitude;",
            f"- original discrete geometry has `(phi,d_minus,d_plus)=({discrete['phi_degrees']:.10g} degrees, {discrete['d_minus']:.12g}, {discrete['d_plus']:.12g})` and symmetric-null `lambda_min={discrete['lambda_min']:+.15g}`.",
            "",
            f"At the same mean half-length, tuning the collinear separation to `r_star` makes the objective magnitude `{continuous_improvement:.3%}` larger than for the selected lattice geometry. This is a comparison on the specified path, not a claim of a global continuous optimum.",
            "",
            f"The analytic discrete value agrees with the saved axis-symmetrized matrix audit within `{abs(discrete['lambda_min'] - saved_discrete_lambda):.3e}`.",
            "",
            "## Limit",
            "",
            "As `s` tends to infinity, `A`, `B`, and `C_far` vanish, while `C` approaches the fixed short-lag contribution. The limiting values are",
            "",
            f"- `C={asymptote['C_total']:+.15g}`;",
            f"- `(P,Q,R)=({asymptote['P_null']:.15g}, {asymptote['Q_null']:.15g}, {asymptote['R_null']:.15g})`;",
            f"- `lambda_min={asymptote['lambda_min']:+.15g}`.",
            "",
            "Therefore the path confirms the proposed mechanism: once `d_minus` is held at the optimal nonzero scale, longer near-parallel contrasts progressively remove diagonal penalties and far-lag cancellation. The improvement has a finite asymptote rather than growing without bound.",
            "",
            "## Reproduction",
            "",
            "From the diagnostic directory:",
            "",
            "```bash",
            "python analyze_fixed_short_scale_length_path.py",
            "```",
            "",
        ]
    )


def main() -> None:
    args = build_parser().parse_args()
    oracle_dir = args.oracle_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    manifest_path = oracle_dir / "experiment_manifest.json"
    saved_sensitivity_path = (
        oracle_dir / "global_two_rectangle_search/axis_symmetry_sensitivity.csv"
    )
    for path in (manifest_path, saved_sensitivity_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.maximum_s <= 1.0 or args.grid_size < 101:
        raise ValueError("maximum-s must exceed one and grid-size must be at least 101")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not bool(manifest.get("oracle_not_real_data_test", False)):
        raise ValueError("the supplied directory is not the exact-comoving oracle")
    truth = manifest["truth"]
    fitted = manifest["fitted_null_fixed_advection"]
    truth_range_lat = _parameter(truth, "range_lat")
    truth_range_lon = _parameter(truth, "range_lon")
    fitted_range_lat = _parameter(fitted, "range_lat")
    fitted_range_lon = _parameter(fitted, "range_lon")
    target_ratio = truth_range_lon / truth_range_lat
    symmetric_range_lat = math.sqrt(
        fitted_range_lat * fitted_range_lon / target_ratio
    )
    symmetric_range_lon = target_ratio * symmetric_range_lat
    spatial_decay_lat = truth_range_lat / symmetric_range_lat
    spatial_decay_lon = truth_range_lon / symmetric_range_lon
    if not math.isclose(spatial_decay_lat, spatial_decay_lon, rel_tol=0.0, abs_tol=1e-14):
        raise ArithmeticError("axis symmetrization did not produce a radial null kernel")

    kernels = PathKernels(
        truth_variance=_parameter(truth, "variance"),
        truth_temporal_scale=1.0 / _parameter(truth, "range_time"),
        null_variance=_parameter(fitted, "variance"),
        null_spatial_decay=0.5 * (spatial_decay_lat + spatial_decay_lon),
        null_temporal_decay=1.0 / _parameter(fitted, "range_time"),
        null_jitter_ratio=float(manifest["numerics"]["numerical_jitter_ratio"]),
    )
    r_star = _optimal_short_scale(kernels.truth_temporal_scale)
    minimum_s = 0.5 * r_star + 1.0e-4
    if args.maximum_s <= minimum_s:
        raise ValueError("maximum-s does not extend beyond the admissible lower bound")
    regular_grid = np.linspace(minimum_s, args.maximum_s, args.grid_size)

    def inertia_margin(s_value: float) -> float:
        row = _geometry_frame(np.asarray([s_value]), r_star, kernels).iloc[0]
        return float(row["negative_inertia_margin_C2_minus_AB"])

    threshold_s = float(brentq(inertia_margin, r_star, args.maximum_s))
    selected_s = 0.5 * (math.sqrt(8.0) + math.sqrt(5.0))
    special_s = np.asarray(
        [r_star, threshold_s, 1.0, 2.0, selected_s, 3.0, 4.0, 6.0],
        dtype=np.float64,
    )
    s_values = np.unique(
        np.concatenate([regular_grid, special_s[special_s <= args.maximum_s]])
    )
    frame = _geometry_frame(s_values, r_star, kernels)
    selected_path = _geometry_frame(np.asarray([selected_s]), r_star, kernels).iloc[0]

    selected_r_a = math.sqrt(8.0)
    selected_r_b = math.sqrt(5.0)
    selected_phi = math.acos(6.0 / (selected_r_a * selected_r_b))
    discrete = _arbitrary_geometry(
        selected_r_a, selected_r_b, selected_phi, kernels
    )
    saved_sensitivity = pd.read_csv(
        saved_sensitivity_path, float_precision="round_trip"
    )
    saved_discrete_lambda = float(
        saved_sensitivity["axis_symmetrized_scipy_eigenvalue"].mean()
    )
    if abs(discrete["lambda_min"] - saved_discrete_lambda) > 1.0e-12:
        raise ArithmeticError("analytic selected geometry does not match the saved audit")

    asymptote = _asymptotic_values(r_star, kernels)
    monotone_tolerance = 64.0 * np.finfo(np.float64).eps
    monotone_lambda = bool(
        np.all(np.diff(frame["lambda_min"].to_numpy()) <= monotone_tolerance)
    )
    if not monotone_lambda:
        raise ArithmeticError("lambda_min was expected to decrease along this path")

    key_labels = {
        r_star: "B penalty peak; start of all-penalties-decrease branch",
        threshold_s: "negative-eigenvalue threshold",
        1.0: "s=1",
        2.0: "s=2",
        selected_s: "mean half-length of selected discrete pair",
        3.0: "s=3",
        4.0: "s=4",
        6.0: "s=6",
    }
    key_rows = []
    for s_value, label in sorted(key_labels.items()):
        row = _geometry_frame(np.asarray([s_value]), r_star, kernels).iloc[0].to_dict()
        row = {"label": label, **row}
        key_rows.append(row)
    key_rows.append({"label": "infinite-length limit", **asymptote})
    key_frame = pd.DataFrame(key_rows)

    summary = pd.DataFrame(
        [
            {"quantity": "r_star", "value": r_star},
            {"quantity": "k_delta_at_r_star", "value": float(kernels.k_delta(r_star))},
            {"quantity": "negative_eigenvalue_threshold_s", "value": threshold_s},
            {"quantity": "selected_mean_half_length_s", "value": selected_s},
            {"quantity": "path_lambda_at_selected_s", "value": float(selected_path["lambda_min"])},
            {"quantity": "discrete_selected_symmetric_H0_lambda", "value": discrete["lambda_min"]},
            {"quantity": "asymptotic_lambda", "value": asymptote["lambda_min"]},
            {
                "quantity": "path_fraction_of_asymptotic_magnitude_at_selected_s",
                "value": abs(float(selected_path["lambda_min"])) / abs(asymptote["lambda_min"]),
            },
            {
                "quantity": "path_magnitude_improvement_over_discrete_at_selected_s",
                "value": abs(float(selected_path["lambda_min"])) / abs(discrete["lambda_min"]) - 1.0,
            },
            {
                "quantity": "analytic_vs_saved_discrete_absolute_error",
                "value": abs(discrete["lambda_min"] - saved_discrete_lambda),
            },
        ]
    )
    settings = pd.DataFrame(
        [
            {
                "truth_variance": kernels.truth_variance,
                "truth_temporal_scale_tau": kernels.truth_temporal_scale,
                "r_star": r_star,
                "fitted_null_variance": kernels.null_variance,
                "axis_symmetrized_range_lat": symmetric_range_lat,
                "axis_symmetrized_range_lon": symmetric_range_lon,
                "null_spatial_decay_in_truth_standardized_coordinates": kernels.null_spatial_decay,
                "null_one_step_temporal_decay": kernels.null_temporal_decay,
                "numerical_jitter_ratio": kernels.null_jitter_ratio,
                "coordinate_interpretation": "truth-range-standardized moving coordinates",
            }
        ]
    )

    _atomic_csv(output_dir / "length_path_values.csv", frame)
    _atomic_csv(output_dir / "length_path_key_points.csv", key_frame)
    _atomic_csv(output_dir / "analysis_summary.csv", summary)
    _atomic_csv(output_dir / "model_settings.csv", settings)
    _write_figure(
        frame,
        r_star,
        threshold_s,
        selected_s,
        discrete["lambda_min"],
        asymptote["lambda_min"],
        output_dir,
    )
    _atomic_text(
        output_dir / "REPORT.md",
        _report(
            oracle_dir,
            r_star,
            float(kernels.k_delta(r_star)),
            threshold_s,
            selected_s,
            selected_path,
            discrete,
            saved_discrete_lambda,
            asymptote,
            kernels,
            monotone_lambda,
        ),
    )
    print(f"Wrote fixed-short-scale path analysis to {output_dir}")
    print(
        f"r_star={r_star:.15g}; threshold_s={threshold_s:.15g}; "
        f"lambda_limit={asymptote['lambda_min']:.15g}"
    )


if __name__ == "__main__":
    main()
