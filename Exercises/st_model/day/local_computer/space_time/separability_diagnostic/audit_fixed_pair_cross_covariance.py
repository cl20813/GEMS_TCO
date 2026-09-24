#!/usr/bin/env python3
"""Audit the fixed global two-rectangle result by spatial displacement.

This script does not search for contrasts, refit covariance parameters, or
simulate responses.  It freezes the representative pair selected by the
existing exact-comoving oracle, reconstructs its two raw rectangle atoms, and
checks the cross-covariance interpretation in two independent ways:

1. direct quadratic forms against the saved 200 by 200 covariance design;
2. four-term covariances of the time-difference field at fixed spatial lags.

Only files in the dedicated audit output directory are written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from diagnostic_core import (
    CovarianceParameters,
    advected_separable_covariance,
    joint_matern_half_covariance,
    pairwise_lags,
)
from rectangle_dictionary_core import rectangle_matrix


HERE = Path(__file__).resolve().parent
DEFAULT_ORACLE_DIR = HERE / "outputs/exact_comoving_rectangle_dictionary_092226"
DEFAULT_OUTPUT_DIR = DEFAULT_ORACLE_DIR / "fixed_pair_cross_covariance_audit"

FIRST_RECTANGLE_ID = "s000_024_t00_01"
SECOND_RECTANGLE_ID = "s005_019_t00_01"
EXPECTED_ENDPOINTS = np.asarray([[0, 24, 0, 1], [5, 19, 0, 1]], dtype=np.int64)
REQUESTED_LAGS = ((0, 0), (4, 4), (2, 4), (1, 0), (3, 4))
MODEL_LABELS = ("1", "M", "0")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-dir", type=Path, default=DEFAULT_ORACLE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.17g")
    temporary.replace(path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _parameters(values: dict[str, Any]) -> CovarianceParameters:
    names = (
        "variance",
        "range_lat",
        "range_lon",
        "range_time",
        "advec_lat",
        "advec_lon",
        "nugget",
    )
    return CovarianceParameters(**{name: float(values[name]) for name in names})


def _ordered_points(points: pd.DataFrame, anchor_count: int, time_count: int) -> pd.DataFrame:
    ordered = points.sort_values(["time_index", "anchor_index"]).reset_index(drop=True)
    expected = pd.MultiIndex.from_product(
        [range(time_count), range(anchor_count)], names=["time_index", "anchor_index"]
    ).to_frame(index=False)
    actual = ordered[["time_index", "anchor_index"]].astype(np.int64).reset_index(drop=True)
    if not actual.equals(expected):
        raise ValueError("saved point table does not have the declared time-major Cartesian order")
    return ordered


def _fixed_pair(
    metadata: pd.DataFrame,
    ties: pd.DataFrame,
    anchor_count: int,
    time_count: int,
) -> tuple[np.ndarray, pd.Series]:
    selected = ties.loc[
        (ties["first_rectangle_id"] == FIRST_RECTANGLE_ID)
        & (ties["second_rectangle_id"] == SECOND_RECTANGLE_ID)
    ]
    if len(selected) != 1:
        raise ValueError("the saved strict ties do not contain exactly one representative fixed pair")
    tie = selected.iloc[0]

    indexed = metadata.set_index("rectangle_id")
    endpoints = np.asarray(
        [
            indexed.loc[FIRST_RECTANGLE_ID][
                ["spatial_endpoint_p", "spatial_endpoint_q", "time_endpoint_k", "time_endpoint_l"]
            ].to_numpy(dtype=np.int64),
            indexed.loc[SECOND_RECTANGLE_ID][
                ["spatial_endpoint_p", "spatial_endpoint_q", "time_endpoint_k", "time_endpoint_l"]
            ].to_numpy(dtype=np.int64),
        ]
    )
    if not np.array_equal(endpoints, EXPECTED_ENDPOINTS):
        raise ValueError(f"saved representative endpoints differ from the audit target: {endpoints}")
    atoms, rebuilt = rectangle_matrix(anchor_count, time_count, endpoints)
    if not np.array_equal(rebuilt, endpoints):
        raise AssertionError("rectangle endpoint reconstruction changed the endpoint order")
    return atoms, tie


def _atom_table(atoms: np.ndarray, points: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column, atom in enumerate(("Q_A", "Q_B")):
        nonzero = np.flatnonzero(atoms[:, column])
        for observation_index in nonzero:
            point = points.iloc[int(observation_index)]
            rows.append(
                {
                    "atom": atom,
                    "observation_index": int(observation_index),
                    "time_index": int(point["time_index"]),
                    "anchor_index": int(point["anchor_index"]),
                    "standardized_moving_latitude": float(point["standardized_moving_latitude"]),
                    "standardized_moving_longitude": float(point["standardized_moving_longitude"]),
                    "coefficient": float(atoms[observation_index, column]),
                }
            )
    return pd.DataFrame(rows)


def _moving_anchor_map(points: pd.DataFrame) -> tuple[dict[int, tuple[float, float]], dict[tuple[float, float], int]]:
    anchors = points.loc[points["time_index"] == 0].sort_values("anchor_index")
    forward = {
        int(row.anchor_index): (
            float(row.standardized_moving_latitude),
            float(row.standardized_moving_longitude),
        )
        for row in anchors.itertuples(index=False)
    }
    reverse = {tuple(np.round(value, 12)): key for key, value in forward.items()}
    if len(reverse) != len(forward):
        raise ValueError("moving-anchor coordinates are not unique")
    return forward, reverse


def _k_delta_placements(
    covariance: np.ndarray,
    h: tuple[int, int],
    anchor_count: int,
    forward: dict[int, tuple[float, float]],
    reverse: dict[tuple[float, float], int],
) -> np.ndarray:
    """Compute Cov(Y(s,0)-Y(s,1), Y(s+h,0)-Y(s+h,1)) from four entries."""

    values = []
    for first_anchor, (latitude, longitude) in forward.items():
        target = tuple(np.round((latitude + h[0], longitude + h[1]), 12))
        second_anchor = reverse.get(target)
        if second_anchor is None:
            continue
        first_time0 = first_anchor
        first_time1 = anchor_count + first_anchor
        second_time0 = second_anchor
        second_time1 = anchor_count + second_anchor
        values.append(
            covariance[first_time0, second_time0]
            - covariance[first_time0, second_time1]
            - covariance[first_time1, second_time0]
            + covariance[first_time1, second_time1]
        )
    if not values:
        raise ValueError(f"requested lag {h} has no placements in the moving grid")
    return np.asarray(values, dtype=np.float64)


def _h_decomposition(k_delta: dict[tuple[int, int], float]) -> np.ndarray:
    return np.asarray(
        [
            [
                2.0 * (k_delta[(0, 0)] - k_delta[(4, 4)]),
                2.0 * (k_delta[(1, 0)] - k_delta[(3, 4)]),
            ],
            [
                2.0 * (k_delta[(1, 0)] - k_delta[(3, 4)]),
                2.0 * (k_delta[(0, 0)] - k_delta[(2, 4)]),
            ],
        ],
        dtype=np.float64,
    )


def _matrix_rows(
    direct: dict[str, np.ndarray],
    decomposed: dict[str, np.ndarray],
) -> pd.DataFrame:
    entries = (("AA", 0, 0), ("AB", 0, 1), ("BB", 1, 1))
    rows = []
    for model in MODEL_LABELS:
        for entry, row, column in entries:
            direct_value = float(direct[model][row, column])
            reconstructed = float(decomposed[model][row, column])
            rows.append(
                {
                    "model": model,
                    "entry": entry,
                    "direct_full_matrix": direct_value,
                    "independent_k_delta_decomposition": reconstructed,
                    "signed_error": reconstructed - direct_value,
                    "absolute_error": abs(reconstructed - direct_value),
                }
            )
    return pd.DataFrame(rows)


def _format_matrix(matrix: np.ndarray) -> str:
    return (
        "[["
        f"{matrix[0, 0]:.12g}, {matrix[0, 1]:.12g}], "
        f"[{matrix[1, 0]:.12g}, {matrix[1, 1]:.12g}]]"
    )


def _report(
    oracle_dir: Path,
    truth: CovarianceParameters,
    matched: CovarianceParameters,
    fitted_null: CovarianceParameters,
    numerical_jitter_ratio: float,
    coefficients: np.ndarray,
    standardized_coefficients: np.ndarray,
    direct_h: dict[str, np.ndarray],
    cross_short: float,
    cross_far: float,
    contribution_frame: pd.DataFrame,
    numerator: float,
    null_variance: float,
    objective: float,
    saved_objective: float,
    delta_k_zero: float,
    maximum_stationarity_span: float,
    maximum_h_error: float,
    maximum_normalization_error: float,
) -> str:
    contribution = contribution_frame.set_index("term")
    null_scales = np.sqrt(np.diag(direct_h["0"]))
    standardized_delta_h = (direct_h["1"] - direct_h["M"]) / np.outer(
        null_scales, null_scales
    )
    standardized_h0 = direct_h["0"] / np.outer(null_scales, null_scales)
    weighted_short = float(contribution.loc["cross_short", "raw_value"])
    weighted_far = float(contribution.loc["cross_far", "raw_value"])
    diagonal_a = float(contribution.loc["diagonal_A", "raw_value"])
    diagonal_b = float(contribution.loc["diagonal_B", "raw_value"])
    return "\n".join(
        [
            "# Fixed-pair cross-covariance audit",
            "",
            "This audit freezes the representative pair and time endpoints `(0,1)` already selected by the earlier exact-comoving global search. It performs no contrast search, coefficient optimization, parameter fit, simulation, bootstrap, or power calculation.",
            "",
            "## Fixed inputs and normalization",
            "",
            f"- Oracle directory: `{oracle_dir}`",
            "- Parameter source: `experiment_manifest.json`; coordinate source: `exact_comoving_points.csv`; atom source: `rectangle_dictionary_metadata.csv`; coefficient source: `global_two_rectangle_search/global_pair_ties.csv`.",
            f"- Rectangles: `{FIRST_RECTANGLE_ID}` and `{SECOND_RECTANGLE_ID}`.",
            f"- `Sigma_1`: joint Matern-half with `(variance, range_lat, range_lon, range_time, advec_lat, advec_lon, nugget)=({truth.variance:.17g}, {truth.range_lat:.17g}, {truth.range_lon:.17g}, {truth.range_time:.17g}, {truth.advec_lat:.17g}, {truth.advec_lon:.17g}, {truth.nugget:.17g})`.",
            f"- `Sigma_M`: advected-separable with the saved matched-margin tuple `({matched.variance:.17g}, {matched.range_lat:.17g}, {matched.range_lon:.17g}, {matched.range_time:.17g}, {matched.advec_lat:.17g}, {matched.advec_lon:.17g}, {matched.nugget:.17g})`.",
            f"- `Sigma_0`: advected-separable fitted null with tuple `({fitted_null.variance:.17g}, {fitted_null.range_lat:.17g}, {fitted_null.range_lon:.17g}, {fitted_null.range_time:.17g}, {fitted_null.advec_lat:.17g}, {fitted_null.advec_lon:.17g}, {fitted_null.nugget:.17g})`.",
            f"- Saved numerical jitter ratio `{numerical_jitter_ratio:.17g}` is retained solely to reproduce the original matrices; it is distinct from the zero statistical nugget.",
            "- Every reported displacement `h=(h_lat,h_lon)` is in the standardized moving-coordinate grid.",
            "- Raw atoms use `+p,k -q,k -p,l +q,l`, so they are exactly `Q_A=A_0-A_1` and `Q_B=B_0-B_1`.",
            f"- Full-precision raw coefficients from `global_pair_ties.csv`: `d1={coefficients[0]:.17g}`, `d2={coefficients[1]:.17g}`.",
            f"- Reconstructed unit-Sigma0 coefficients: `{standardized_coefficients[0]:.17g}`, `{standardized_coefficients[1]:.17g}`.",
            "- Each dictionary atom was divided by its fitted-null standard deviation before the 2-by-2 eigenproblem. The saved raw coefficient is therefore the standardized eigenvector coefficient divided by that standard deviation.",
            f"- Maximum saved-versus-reconstructed normalization error: `{maximum_normalization_error:.3e}`.",
            "",
            "## Direct full-matrix calculation",
            "",
            f"- `H_1 = {_format_matrix(direct_h['1'])}`",
            f"- `H_M = {_format_matrix(direct_h['M'])}`",
            f"- `H_0 = {_format_matrix(direct_h['0'])}`",
            f"- After the original unit-Sigma0 atom normalization, `D = {_format_matrix(standardized_delta_h)}` and `G = {_format_matrix(standardized_h0)}`; these reproduce the saved reduced matrices within `{maximum_normalization_error:.3e}`.",
            f"- Maximum error against the independent K_delta decomposition: `{maximum_h_error:.3e}`.",
            f"- Maximum stationarity spread over all available placements: `{maximum_stationarity_span:.3e}`.",
            "",
            "## A. Structural claim",
            "",
            "Confirmed. With `I(s)=Y(s,0)-Y(s,1)`, the two atoms are `Q_A=I((-2,-2))-I((2,2))` and `Q_B=I((-1,-2))-I((1,2))`. Spatial stationarity and covariance symmetry, without an isotropy assumption, give",
            "",
            "`Var(Q_A)=2[K_delta(0,0)-K_delta(4,4)]`,",
            "",
            "`Var(Q_B)=2[K_delta(0,0)-K_delta(2,4)]`,",
            "",
            "`Cov(Q_A,Q_B)=2[K_delta(1,0)-K_delta(3,4)]`.",
            "",
            "Thus the short vector lag `(1,0)` is absent from both individual variance formulas and is present in the cross-covariance formula. The implementation and the direct 200-dimensional calculation agree to the error reported above.",
            "",
            "## B. Numerical claim",
            "",
            f"The matched-margin check was not forced: `deltaK(0,0)={delta_k_zero:+.17g}`, numerically zero up to roundoff.",
            "",
            "For the raw `DeltaH_AB=H_1,AB-H_M,AB` discrepancy:",
            "",
            f"- short contribution `2 deltaK(1,0) = {cross_short:+.17g}`;",
            f"- far contribution `-2 deltaK(3,4) = {cross_far:+.17g}`;",
            f"- sum `{cross_short + cross_far:+.17g}`, versus direct `DeltaH_AB={direct_h['1'][0,1] - direct_h['M'][0,1]:+.17g}`.",
            "",
            "The far term has the opposite sign and partially offsets the short-lag term; it does not reinforce it.",
            "",
            "For the fixed final `L=d1 Q_A+d2 Q_B` numerator:",
            "",
            "| contribution | raw value | divided by V0 |",
            "|---|---:|---:|",
            f"| diagonal A | {diagonal_a:+.12g} | {float(contribution.loc['diagonal_A', 'divided_by_V0']):+.12g} |",
            f"| diagonal B | {diagonal_b:+.12g} | {float(contribution.loc['diagonal_B', 'divided_by_V0']):+.12g} |",
            f"| cross short | {weighted_short:+.12g} | {float(contribution.loc['cross_short', 'divided_by_V0']):+.12g} |",
            f"| cross far | {weighted_far:+.12g} | {float(contribution.loc['cross_far', 'divided_by_V0']):+.12g} |",
            f"| total N | {numerator:+.12g} | {objective:+.12g} |",
            "",
            f"`V0={null_variance:.17g}` and `N/V0={objective:.17g}`; the saved global-search objective is `{saved_objective:.17g}`.",
            "",
            "## Answers to the four audit questions",
            "",
            "1. **Yes.** The formulas, atom signs, indices, and direct implementation agree: `(1,0)` enters only the cross-covariance, not either individual variance.",
            f"2. The unweighted cross-covariance contributions are short `{cross_short:+.12g}` and far `{cross_far:+.12g}`. At the final-L numerator scale they are respectively `{weighted_short:+.12g}` and `{weighted_far:+.12g}`.",
            "3. The dominant discrepancy is the negative short-lag cross contribution. The far cross term and both diagonal terms are positive offsets.",
            "4. The proposed interpretation is supported for this fixed selected exact-comoving pair, but should be stated more precisely: **the standardized moving-coordinate lag `(1,0)` supplies the dominant negative truth-minus-matched contribution; the `(3,4)` cross lag and both diagonal terms partially offset it.** This is a result-specific decomposition, not a universal causal claim about all contrasts.",
            "",
            "## Reproduction",
            "",
            "From the diagnostic directory:",
            "",
            "```bash",
            "python audit_fixed_pair_cross_covariance.py",
            "```",
            "",
            "Inputs are read from the saved experiment manifest, exact-comoving point table, rectangle metadata, and global strict-tie table. Outputs are written only inside this audit directory.",
            "",
        ]
    )


def main() -> None:
    args = build_parser().parse_args()
    oracle_dir = args.oracle_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = oracle_dir / "experiment_manifest.json"
    points_path = oracle_dir / "exact_comoving_points.csv"
    metadata_path = oracle_dir / "rectangle_dictionary_metadata.csv"
    ties_path = oracle_dir / "global_two_rectangle_search/global_pair_ties.csv"
    for path in (manifest_path, points_path, metadata_path, ties_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not bool(manifest.get("oracle_not_real_data_test", False)):
        raise ValueError("the supplied directory is not marked as the exact-comoving oracle")
    anchor_count = int(manifest["geometry"]["anchor_count"])
    time_count = int(manifest["geometry"]["time_count"])
    points = _ordered_points(pd.read_csv(points_path), anchor_count, time_count)
    metadata = pd.read_csv(metadata_path)
    # The tie table was written with 17 significant digits.  Round-trip parsing
    # preserves the saved eigenvector coefficients rather than perturbing their
    # last binary digit through pandas' faster default parser.
    ties = pd.read_csv(ties_path, float_precision="round_trip")
    atoms, tie = _fixed_pair(metadata, ties, anchor_count, time_count)

    truth = _parameters(manifest["truth"])
    matched = _parameters(manifest["matched_margin"])
    fitted_null = _parameters(manifest["fitted_null_fixed_advection"])
    jitter = float(manifest["numerics"]["numerical_jitter_ratio"])
    coordinates = points[["source_latitude", "source_longitude", "time_index"]].to_numpy(
        dtype=np.float64
    )
    geometry = pairwise_lags(coordinates)
    covariances = {
        "1": joint_matern_half_covariance(
            geometry, truth, numerical_jitter_ratio=jitter
        ),
        "M": advected_separable_covariance(
            geometry, matched, numerical_jitter_ratio=jitter
        ),
        "0": advected_separable_covariance(
            geometry, fitted_null, numerical_jitter_ratio=jitter
        ),
    }

    direct_h = {model: atoms.T @ covariance @ atoms for model, covariance in covariances.items()}
    forward, reverse = _moving_anchor_map(points)
    k_delta: dict[str, dict[tuple[int, int], float]] = {model: {} for model in MODEL_LABELS}
    stationarity_rows = []
    for h in REQUESTED_LAGS:
        for model in MODEL_LABELS:
            values = _k_delta_placements(
                covariances[model], h, anchor_count, forward, reverse
            )
            negative_values = _k_delta_placements(
                covariances[model], (-h[0], -h[1]), anchor_count, forward, reverse
            )
            k_delta[model][h] = float(np.mean(values))
            stationarity_rows.append(
                {
                    "model": model,
                    "h_lat": h[0],
                    "h_lon": h[1],
                    "placement_count": len(values),
                    "mean": float(np.mean(values)),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                    "stationarity_span": float(np.ptp(values)),
                    "negative_lag_mean": float(np.mean(negative_values)),
                    "even_symmetry_absolute_error": abs(
                        float(np.mean(values)) - float(np.mean(negative_values))
                    ),
                }
            )

    decomposed_h = {model: _h_decomposition(k_delta[model]) for model in MODEL_LABELS}
    h_frame = _matrix_rows(direct_h, decomposed_h)
    stationarity_frame = pd.DataFrame(stationarity_rows)
    maximum_h_error = float(h_frame["absolute_error"].max())
    maximum_stationarity_span = float(stationarity_frame["stationarity_span"].max())
    maximum_even_error = float(stationarity_frame["even_symmetry_absolute_error"].max())
    if max(maximum_h_error, maximum_stationarity_span, maximum_even_error) > 1.0e-12:
        raise ArithmeticError("stationarity or K_delta decomposition check exceeded tolerance")

    delta_k = {h: k_delta["1"][h] - k_delta["M"][h] for h in REQUESTED_LAGS}
    membership = {
        (0, 0): "Var(Q_A), Var(Q_B)",
        (4, 4): "Var(Q_A)",
        (2, 4): "Var(Q_B)",
        (1, 0): "Cov(Q_A,Q_B): short, coefficient +2",
        (3, 4): "Cov(Q_A,Q_B): far, coefficient -2",
    }
    displacement_frame = pd.DataFrame(
        [
            {
                "h": f"({h[0]},{h[1]})",
                "h_lat": h[0],
                "h_lon": h[1],
                "K_delta_1": k_delta["1"][h],
                "K_delta_M": k_delta["M"][h],
                "deltaK_1_minus_M": delta_k[h],
                "enters": membership[h],
                "K_delta_0": k_delta["0"][h],
            }
            for h in REQUESTED_LAGS
        ]
    )

    delta_h = direct_h["1"] - direct_h["M"]
    coefficients = np.asarray(
        [float(tie["raw_coefficient_first"]), float(tie["raw_coefficient_second"])],
        dtype=np.float64,
    )
    saved_standardized = np.asarray(
        [float(tie["coefficient_first"]), float(tie["coefficient_second"])],
        dtype=np.float64,
    )
    null_standard_deviations = np.sqrt(np.diag(direct_h["0"]))
    reconstructed_standardized = coefficients * null_standard_deviations
    maximum_normalization_error = float(
        np.max(np.abs(reconstructed_standardized - saved_standardized))
    )

    standardized_delta_h = delta_h / np.outer(
        null_standard_deviations, null_standard_deviations
    )
    standardized_h0 = direct_h["0"] / np.outer(
        null_standard_deviations, null_standard_deviations
    )
    normalization_rows = [
        ("coefficient_A_standardized", reconstructed_standardized[0], saved_standardized[0]),
        ("coefficient_B_standardized", reconstructed_standardized[1], saved_standardized[1]),
        ("D_AA", standardized_delta_h[0, 0], float(tie["d_ii"])),
        ("D_AB", standardized_delta_h[0, 1], float(tie["d_ij"])),
        ("D_BB", standardized_delta_h[1, 1], float(tie["d_jj"])),
        ("G_AA", standardized_h0[0, 0], float(tie["g_ii"])),
        ("G_AB", standardized_h0[0, 1], float(tie["g_ij"])),
        ("G_BB", standardized_h0[1, 1], float(tie["g_jj"])),
    ]
    normalization_frame = pd.DataFrame(
        [
            {
                "quantity": name,
                "recomputed": float(recomputed),
                "saved": float(saved),
                "signed_error": float(recomputed - saved),
                "absolute_error": abs(float(recomputed - saved)),
            }
            for name, recomputed, saved in normalization_rows
        ]
    )
    maximum_normalization_error = max(
        maximum_normalization_error,
        float(normalization_frame["absolute_error"].max()),
    )
    if maximum_normalization_error > 1.0e-12:
        raise ArithmeticError("saved normalization does not match the reconstructed raw atoms")

    cross_short = 2.0 * delta_k[(1, 0)]
    cross_far = -2.0 * delta_k[(3, 4)]
    direct_cross = float(delta_h[0, 1])
    if abs(cross_short + cross_far - direct_cross) > 1.0e-12:
        raise ArithmeticError("short/far cross-covariance contributions do not reconstruct DeltaH_AB")

    d1, d2 = coefficients
    null_variance = float(coefficients @ direct_h["0"] @ coefficients)
    numerator = float(coefficients @ delta_h @ coefficients)
    contribution_values = {
        "diagonal_A": d1 * d1 * delta_h[0, 0],
        "diagonal_B": d2 * d2 * delta_h[1, 1],
        "cross_short": 4.0 * d1 * d2 * delta_k[(1, 0)],
        "cross_far": -4.0 * d1 * d2 * delta_k[(3, 4)],
    }
    contribution_frame = pd.DataFrame(
        [
            {
                "term": term,
                "raw_value": float(value),
                "divided_by_V0": float(value / null_variance),
            }
            for term, value in contribution_values.items()
        ]
        + [
            {
                "term": "total_N",
                "raw_value": numerator,
                "divided_by_V0": numerator / null_variance,
            }
        ]
    )
    if abs(sum(contribution_values.values()) - numerator) > 1.0e-12:
        raise ArithmeticError("four fixed-L contributions do not reconstruct the numerator")

    objective = numerator / null_variance
    saved_objective = float(tie["analytic_eigenvalue"])
    summary_frame = pd.DataFrame(
        [
            {"quantity": "deltaH_AB_short", "value": cross_short},
            {"quantity": "deltaH_AB_far", "value": cross_far},
            {"quantity": "deltaH_AB_sum", "value": cross_short + cross_far},
            {"quantity": "deltaH_AB_direct", "value": direct_cross},
            {"quantity": "N", "value": numerator},
            {"quantity": "V0", "value": null_variance},
            {"quantity": "N_over_V0", "value": objective},
            {"quantity": "saved_objective", "value": saved_objective},
            {"quantity": "deltaK_0_0", "value": delta_k[(0, 0)]},
            {"quantity": "max_H_reconstruction_error", "value": maximum_h_error},
            {"quantity": "max_stationarity_span", "value": maximum_stationarity_span},
            {"quantity": "max_even_symmetry_error", "value": maximum_even_error},
            {"quantity": "max_normalization_error", "value": maximum_normalization_error},
        ]
    )
    if abs(objective - saved_objective) > 1.0e-12:
        raise ArithmeticError("fixed-coefficient objective does not reproduce the saved objective")

    settings_frame = pd.DataFrame(
        [
            {
                "model": model,
                "covariance_family": family,
                **parameters.to_dict(),
                "numerical_jitter_ratio": jitter,
                "source": source,
            }
            for model, family, parameters, source in (
                ("1", "joint_matern_half", truth, "experiment_manifest.json: truth"),
                ("M", "advected_separable", matched, "experiment_manifest.json: matched_margin"),
                ("0", "advected_separable", fitted_null, "experiment_manifest.json: fitted_null_fixed_advection"),
            )
        ]
    )

    _atomic_csv(output_dir / "atom_coefficients.csv", _atom_table(atoms, points))
    _atomic_csv(output_dir / "model_settings.csv", settings_frame)
    _atomic_csv(output_dir / "H_matrix_audit.csv", h_frame)
    _atomic_csv(output_dir / "k_delta_displacement_audit.csv", displacement_frame)
    _atomic_csv(output_dir / "k_delta_stationarity_audit.csv", stationarity_frame)
    _atomic_csv(output_dir / "normalization_audit.csv", normalization_frame)
    _atomic_csv(output_dir / "L_contribution_audit.csv", contribution_frame)
    _atomic_csv(output_dir / "audit_summary.csv", summary_frame)
    _atomic_text(
        output_dir / "REPORT.md",
        _report(
            oracle_dir=oracle_dir,
            truth=truth,
            matched=matched,
            fitted_null=fitted_null,
            numerical_jitter_ratio=jitter,
            coefficients=coefficients,
            standardized_coefficients=reconstructed_standardized,
            direct_h=direct_h,
            cross_short=cross_short,
            cross_far=cross_far,
            contribution_frame=contribution_frame,
            numerator=numerator,
            null_variance=null_variance,
            objective=objective,
            saved_objective=saved_objective,
            delta_k_zero=delta_k[(0, 0)],
            maximum_stationarity_span=maximum_stationarity_span,
            maximum_h_error=maximum_h_error,
            maximum_normalization_error=maximum_normalization_error,
        ),
    )
    print(f"Wrote fixed-pair audit to {output_dir}")
    print(f"N/V0 = {objective:.17g}; saved = {saved_objective:.17g}")


if __name__ == "__main__":
    main()
