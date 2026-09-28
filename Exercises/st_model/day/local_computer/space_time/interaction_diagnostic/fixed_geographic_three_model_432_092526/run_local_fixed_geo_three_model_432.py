#!/usr/bin/env python3
"""Run the frozen fixed-geographic diagnostic locally with corridor 4/3/2."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[6]
AMAREL_STUDY = (
    PROJECT_ROOT
    / "Exercises/st_model/day/amarel_simulation/space_time/interaction_diagnostic"
    / "fixed_geographic_three_model_092426"
)
MODEL_ORDER = ("gc", "matern05", "separable")
DEFAULT_OUTPUT = HERE / "outputs_current" / "2024-07-02"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--task-id",
        type=int,
        default=0,
        help="Row in the frozen evaluation manifest; task 0 is 2024-07-02.",
    )
    result.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_ORDER,
        default=MODEL_ORDER,
    )
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return result


def main() -> None:
    args = parser().parse_args()
    runner = AMAREL_STUDY / "run_fixed_geo_three_model_day.py"
    design = AMAREL_STUDY / "frozen_design.json"
    dates = AMAREL_STUDY / "evaluation_dates.csv"
    config = HERE / "local_fixed_geo_three_model_432.toml"
    for path in (runner, design, dates, config):
        if not path.is_file():
            raise FileNotFoundError(path)

    command = [
        sys.executable,
        str(runner),
        "--task-id",
        str(args.task_id),
        "--config",
        str(config),
        "--design",
        str(design),
        "--dates",
        str(dates),
        "--data-root",
        "/Users/joonwonlee/Documents/GEMS_DATA",
        "--output-root",
        str(args.output_root.expanduser().resolve()),
        "--device",
        "cpu",
        "--models",
        *args.models,
    ]
    if tuple(args.models) != MODEL_ORDER:
        command.append("--allow-model-subset")
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
