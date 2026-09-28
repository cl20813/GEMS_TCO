#!/usr/bin/env python3
"""Deterministic analytic validation for the selected Q_A/Q_B diagnostic."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent

from two_contrast_core import (  # noqa: E402
    contrast_coefficients,
    load_json,
    pointwise_contrast_covariances,
    radial_correlation,
)


EXPECTED = {
    "matern": {
        "separable": np.asarray(
            [
                [15.683790374655344, 5.683924218495644],
                [5.683924218495644, 15.558991316159705],
            ]
        ),
        "joint": np.asarray(
            [
                [15.735725516570034, 1.631663760460713],
                [1.631663760460713, 15.726217833285678],
            ]
        ),
    },
    "generalized_cauchy": {
        "separable": np.asarray(
            [
                [16.055030076441441, 5.616577430124419],
                [5.616577430124419, 15.813508559223635],
            ]
        ),
        "joint": np.asarray(
            [
                [16.329372593659782, 1.465493851561316],
                [1.465493851561316, 16.317126734429044],
            ]
        ),
    },
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--config", type=Path, default=HERE / "two_contrast_diagnostic_092726.json"
    )
    return result


def truth_for(family: str, advection: tuple[float, float]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "family": family,
        "sigmasq": 10.0,
        "range_lat": 0.2,
        "range_lon": 0.3,
        "range_time": 2.0,
        "advec_lat": float(advection[0]),
        "advec_lon": float(advection[1]),
        "nugget": 0.0,
    }
    if family == "matern":
        result["matern_nu"] = 0.5
    else:
        result.update(
            {
                "cauchy_a": 1.0,
                "cauchy_b": 5.0,
                "cauchy_efold_scale": math.exp(0.2) - 1.0,
            }
        )
    return result


def ideal_coordinates(
    standardized: np.ndarray,
    truth: dict[str, Any],
    time_t: int,
    translation: tuple[float, float],
) -> np.ndarray:
    offsets = np.asarray(standardized, dtype=np.float64).copy()
    offsets[:, 0] *= float(truth["range_lat"])
    offsets[:, 1] *= float(truth["range_lon"])
    point_specs = (
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
    )
    coordinates = np.empty((8, 3), dtype=np.float64)
    for point_index, (side, endpoint) in enumerate(point_specs):
        slot = time_t + side
        coordinates[point_index, 0] = (
            translation[0]
            + offsets[endpoint, 0]
            + float(truth["advec_lat"]) * slot
        )
        coordinates[point_index, 1] = (
            translation[1]
            + offsets[endpoint, 1]
            + float(truth["advec_lon"]) * slot
        )
        coordinates[point_index, 2] = float(slot)
    return coordinates


def validate(config: dict[str, Any]) -> None:
    coefficients = contrast_coefficients(config)
    np.testing.assert_allclose(coefficients.sum(axis=1), 0.0, atol=0.0)
    mirrors = config["geometry"]["standardized_geometry"]
    d1 = float(config["geometry"]["secondary_l_coefficients"]["d1"])
    d2 = float(config["geometry"]["secondary_l_coefficients"]["d2"])
    expected_signed = {
        "matern": -0.1955163967779494,
        "generalized_cauchy": -0.2002844862539626,
    }
    for family in ("matern", "generalized_cauchy"):
        reference: tuple[np.ndarray, np.ndarray] | None = None
        for advection in ((0.08, -0.2), (-0.17, 0.11)):
            truth = truth_for(family, advection)
            for time_t in (0, 6):
                for translation in ((0.0, 0.0), (4.25, 126.75)):
                    for standardized in mirrors.values():
                        coordinates = ideal_coordinates(
                            np.asarray(standardized), truth, time_t, translation
                        )[None, :, :]
                        separable, joint = pointwise_contrast_covariances(
                            coordinates, truth, config, chunk_size=1
                        )
                        np.testing.assert_allclose(
                            separable[0], EXPECTED[family]["separable"], atol=1e-12
                        )
                        np.testing.assert_allclose(
                            joint[0], EXPECTED[family]["joint"], atol=1e-12
                        )
                        if reference is None:
                            reference = (separable[0].copy(), joint[0].copy())
                        else:
                            np.testing.assert_allclose(separable[0], reference[0], atol=1e-12)
                            np.testing.assert_allclose(joint[0], reference[1], atol=1e-12)
        midpoint = 0.5 * (
            EXPECTED[family]["separable"] + EXPECTED[family]["joint"]
        )
        np.testing.assert_allclose(
            midpoint,
            (1.0 - 0.5) * EXPECTED[family]["separable"]
            + 0.5 * EXPECTED[family]["joint"],
            atol=0.0,
        )
        signed = (
            2.0
            * d1
            * d2
            * (
                EXPECTED[family]["joint"][0, 1]
                - EXPECTED[family]["separable"][0, 1]
            )
        )
        np.testing.assert_allclose(signed, expected_signed[family], atol=1e-14)

    gc_truth = truth_for("generalized_cauchy", (0.08, -0.2))
    np.testing.assert_allclose(
        radial_correlation(np.asarray(1.0), gc_truth), math.exp(-1.0), atol=1e-15
    )
    print("PASS: two-contrast analytic targets, mirrors, flow invariance, and GC scaling")


def main() -> None:
    args = parser().parse_args()
    validate(load_json(args.config.expanduser().resolve()))


if __name__ == "__main__":
    main()
