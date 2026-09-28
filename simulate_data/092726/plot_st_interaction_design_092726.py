#!/usr/bin/env python3
"""Plot the theoretical correlation surfaces in the frozen interaction suite."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent


def radial(distance: np.ndarray, scenario: dict) -> np.ndarray:
    if scenario["family"] == "matern":
        return np.exp(-distance)
    a = float(scenario["cauchy_a"])
    b = float(scenario["cauchy_b"])
    scale = (math.exp(a / b) - 1.0) ** (1.0 / a)
    return np.power(1.0 + np.power(scale * distance, a), -b / a)


def surface(ds: np.ndarray, dt: np.ndarray, scenario: dict) -> np.ndarray:
    eta = float(scenario["interaction_eta"])
    separable = radial(ds, scenario) * radial(dt, scenario)
    joint = radial(np.hypot(ds, dt), scenario)
    return (1.0 - eta) * separable + eta * joint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=HERE / "st_interaction_scenarios_092726.json"
    )
    parser.add_argument(
        "--output", type=Path, default=HERE / "st_interaction_design_092726.png"
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    by_id = {value["scenario_id"]: value for value in config["scenarios"]}
    pairs = [
        (
            "Matérn ν=0.5",
            by_id["matern05_lagrangian_separable_eta0"],
            by_id["matern05_lagrangian_mixture_eta0p5"],
            by_id["matern05_lagrangian_joint_eta1"],
        ),
        (
            "Generalized Cauchy a=1, b=5",
            by_id["gencauchy_a1_b5_lagrangian_separable_eta0"],
            by_id["gencauchy_a1_b5_lagrangian_mixture_eta0p5"],
            by_id["gencauchy_a1_b5_lagrangian_joint_eta1"],
        ),
    ]
    axis = np.linspace(0.0, 3.0, 241)
    ds, dt = np.meshgrid(axis, axis)
    fig, axes = plt.subplots(2, 4, figsize=(17.0, 8.0), constrained_layout=True)
    correlation_images = []
    difference_images = []
    for row, (family_label, eta0, eta05, eta1) in enumerate(pairs):
        left = surface(ds, dt, eta0)
        middle = surface(ds, dt, eta05)
        right = surface(ds, dt, eta1)
        difference = right - left
        for col, (value, title) in enumerate(
            [
                (left, "η=0: Lagrangian separable"),
                (middle, "η=0.5: intermediate mixture"),
                (right, "η=1: joint radial"),
            ]
        ):
            image = axes[row, col].imshow(
                value,
                origin="lower",
                extent=(0, 3, 0, 3),
                vmin=0,
                vmax=1,
                cmap="viridis",
                aspect="auto",
            )
            correlation_images.append(image)
            axes[row, col].set_title(f"{family_label}\n{title}")
        maximum = max(float(difference.max()), 1e-12)
        image = axes[row, 3].imshow(
            difference,
            origin="lower",
            extent=(0, 3, 0, 3),
            vmin=0,
            vmax=maximum,
            cmap="magma",
            aspect="auto",
        )
        difference_images.append(image)
        axes[row, 3].set_title(f"{family_label}\ninteraction increment: C₁−C₀")
        axes[row, 3].text(
            0.07,
            2.78,
            "exactly 0 on both axes",
            color="white",
            fontsize=9,
            ha="left",
            va="top",
        )
        for col in range(4):
            axes[row, col].set_xlabel("standardized spatial lag  dₛ")
            axes[row, col].set_ylabel("standardized time lag  dₜ")
            axes[row, col].plot([1], [1], marker="o", ms=4, color="white", mec="black", mew=0.5)
    fig.colorbar(correlation_images[0], ax=axes[:, :3], label="correlation", shrink=0.85)
    for row, image in enumerate(difference_images):
        fig.colorbar(image, ax=axes[row, 3], label="C(η=1) − C(η=0)", shrink=0.8)
    fig.suptitle(
        "Controlled interaction design: margins fixed, interior coupling changed",
        fontsize=15,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
